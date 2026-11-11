#!/usr/bin/env python3
"""Earlier-script (legacy) results that are correct enough to show, as one normalized table.

Reads legacy/data/<cpu|gpu>/<DEVICE>/sweep_*.csv and keeps only the benches and rows that
configs/legacy.json marks as usable. Every legacy bench writes its own column layout; this maps them to:

    device, cls, bench, workload_id, precision, units (GPUs or CPU threads), size, size_label,
    density, metric, unit, gflops, gbps, power_w, h2d_ms, compute_ms, d2h_ms, iters

bench -> workload: gemm -> C1, saxpy -> B1, gemv sparse rows -> B3 (SpMV), spgemm -> B4,
gemv dense rows -> GEMV (extra page, not one of the 16 workloads).
GPU throughput is the kernel-only figure (agg_*), the same convention as our harness; the old
pct_peak columns (computed against A100 peaks on every GPU) are ignored.
"""
from __future__ import annotations

import csv
import glob
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "legacy.json"
WORKLOAD = {"gemm": "C1", "saxpy": "B1", "spmv": "B3", "spgemm": "B4", "gemv": "GEMV"}
BENCH_NAME = {"gemm": "Matrix multiply (GEMM)", "saxpy": "SAXPY", "spmv": "Sparse matrix x vector (SpMV)",
              "spgemm": "Sparse matrix x matrix (SpGEMM)", "gemv": "Matrix x vector (GEMV)"}


def config() -> dict:
    return json.loads(CFG.read_text())


def _f(r, k, default=float("nan")):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else default
    except ValueError:
        return default


def _bench_of(path: str, row: dict) -> str | None:
    name = Path(path).name
    if "_gemv_" in name:
        return "spmv" if row.get("engine") == "sparse" else "gemv"
    for b in ("saxpy", "spgemm", "gemm"):
        if f"_{b}_" in name:
            return b
    if name.startswith("sweep_2"):                      # GPU GEMM files: sweep_<date>.csv
        return "gemm"
    return None


def _row(dev: str, cls: str, bench: str, r: dict) -> dict | None:
    units = int(_f(r, "gpus" if cls == "gpu" else "threads", 1))
    prec = r.get("precision", "")
    rec = {"device": dev, "cls": cls, "bench": bench, "workload_id": WORKLOAD[bench], "precision": prec,
           "units": units, "mode": r.get("mode", ""), "iters": _f(r, "iters"),
           "power_w": _f(r, "power_w_avg"), "h2d_ms": _f(r, "h2d_ms"), "compute_ms": _f(r, "compute_ms", _f(r, "comp_ms")),
           "d2h_ms": _f(r, "d2h_ms"), "density": _f(r, "density", _f(r, "density_a")), "gflops": float("nan"),
           "gbps": float("nan")}
    if bench == "gemm":
        m, n, k = int(_f(r, "M")), int(_f(r, "N")), int(_f(r, "K"))
        rec.update(size=m, size_label=f"{m}" if m == n == k else f"{m}x{n}x{k}",
                   gflops=1e3 * _f(r, "per_gpu_tflops" if cls == "gpu" else "agg_tflops"))
        rec.update(metric=rec["gflops"], unit="GFLOP/s")
    elif bench == "saxpy":
        n = int(_f(r, "N"))
        rec.update(size=n, size_label=_pow2(n), gbps=_f(r, "per_gpu_gbps" if cls == "gpu" else "agg_gbps"))
        rec.update(metric=rec["gbps"], unit="GB/s")
    elif bench in ("gemv", "spmv"):
        s = int(_f(r, "S"))
        rec.update(size=s, size_label=f"{s}" + ("" if bench == "gemv" else f", density {rec['density']:g}"),
                   gbps=_f(r, "per_gpu_gbps" if cls == "gpu" else "agg_gbps"))
        rec.update(metric=rec["gbps"], unit="GB/s")
    elif bench == "spgemm":
        s = int(_f(r, "S"))
        rec.update(size=s, size_label=f"{s}, density {rec['density']:g}", gbps=_f(r, "agg_gbps"))
        rec.update(metric=rec["gbps"], unit="GB/s")
    if not rec["metric"] == rec["metric"] or rec["metric"] <= 0:
        return None
    return rec


@lru_cache(maxsize=None)
def load() -> pd.DataFrame:
    """All usable legacy rows (status ok, usable bench, device filters applied)."""
    cfg = config()["devices"]
    out = []
    for dev, d in cfg.items():
        cls = d["class"]
        use, drop = d.get("use", {}), d.get("drop", {})
        for path in sorted(glob.glob(str(ROOT / "legacy" / "data" / cls / dev / "sweep_*.csv"))):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("status") != "ok":
                        continue
                    bench = _bench_of(path, r)
                    if not bench or not use.get(bench):
                        continue
                    if bench == "gemm" and ("sparse" in (r.get("tag") or "") or r.get("engine") == "sparse"):
                        continue                      # 2:4 sparse GEMM: density ignored, TF32 tensor cores
                    p = r.get("precision", "")
                    if drop.get("bf16") and p == "bf16" or drop.get("fp64t") and p == "fp64t":
                        continue
                    if bench == "gemv" and drop.get("gemv_dense_max_size") and _f(r, "S") <= drop["gemv_dense_max_size"]:
                        continue
                    rec = _row(dev, cls, bench, r)
                    if rec is None:
                        continue
                    if drop.get("power") or not rec["power_w"] > 0:
                        rec["power_w"] = float("nan")
                    out.append(rec)
    df = pd.DataFrame(out)
    if len(df):
        df["dram"] = [_ws(r) >= 4 * _llc(r["device"]) for r in df.to_dict("records")]
        df["precision"] = df["precision"].replace({"fp64t": "fp64 (tensor cores)"})
    return df


ELEM = {"fp64": 8, "fp64t": 8, "fp32": 4, "bf16": 2}


def _ws(r: dict) -> float:
    """Approximate working set in bytes, to tell memory-sized problems from cache-sized ones."""
    e = ELEM.get(r["precision"], 4)
    if r["bench"] == "saxpy":
        return 2 * r["size"] * e
    if r["bench"] == "gemv":
        return r["size"] ** 2 * e
    if r["bench"] == "spmv":
        return r["size"] ** 2 * r["density"] * (e + 4)
    return float("inf")                                   # gemm / spgemm: not judged against memory speed


@lru_cache(maxsize=None)
def _llc(dev: str) -> float:
    try:
        return json.loads((ROOT / "configs" / "devices" / f"{dev}.json").read_text()).get("llc_mb", 32) * 2 ** 20
    except OSError:
        return 32 * 2 ** 20


def main_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (device, bench, precision, size[, density]): 1 GPU, or the largest CPU thread count;
    'split' and 'replicas' are the same work on one GPU, so the faster is kept."""
    if df.empty:
        return df
    keep = []
    for dev, g in df.groupby("device"):
        if g["cls"].iloc[0] == "gpu":
            g = g[g["units"] == 1]
        else:
            g = g[g["units"] == g.groupby("bench")["units"].transform("max")]
        keep.append(g)
    m = pd.concat(keep)
    m = (m.sort_values("metric", ascending=False)
          .drop_duplicates(["device", "bench", "precision", "size", "density"]).sort_values(["device", "bench", "precision", "size"]))
    return m


def cell_states() -> dict:
    """{device: {workload_id: ('earlier', why) | ('bad', why)}} for the progress page."""
    cfg = config()
    df = load()
    out = {}
    for dev, d in cfg["devices"].items():
        cells = {}
        for bench, ok in d.get("use", {}).items():
            wid = WORKLOAD[bench]
            if wid == "GEMV":
                continue
            if ok and not df[(df.device == dev) & (df.bench == bench)].empty:
                cells[wid] = ("earlier", cfg["caveats"].get(bench, ""))
            elif not ok:
                cells[wid] = ("bad", d.get("why_not", {}).get(bench, "The earlier script measured this incorrectly."))
        for wid, why in d.get("study_invalid", {}).items():
            cells.setdefault(wid, ("bad", why))
        out[dev] = cells
    return out


def _pow2(n: int) -> str:
    return f"2^{n.bit_length() - 1}" if n > 0 and n & (n - 1) == 0 else f"{n:,}"


if __name__ == "__main__":
    d = load()
    print(d.groupby(["device", "bench"]).size().to_string())
    print(json.dumps(cell_states(), indent=1)[:1500])
