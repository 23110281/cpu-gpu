#!/usr/bin/env python3
"""M2: per-device ceilings for pct_measured_peak (docs/methodology.md §10, plan A12).

  measured.bandwidth_gbps   BabelStream (UoB-HPC, well-known) Triad, OpenMP backend on CPU
                            (CUDA backend on GPU), arrays >= 32x LLC, pinned like the harness,
                            median of `--reps` independent runs (each = BabelStream's own 100 iters).
  measured.peak_gflops[p]   best C1 GEMM median throughput over the two largest sizes,
                            measured by the v2 harness itself (publication policy).
  datasheet.*               copied from configs/devices/<DEVICE>.json "datasheet" with derivation.
Writes results/calibration/<DEVICE>.json (raw BabelStream output kept for provenance).

  python3 scripts/calibrate.py --device-config configs/devices/AMD_C6525_100G.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "torch"))
from bench import devices  # noqa: E402

BABEL_URL = "https://github.com/UoB-HPC/BabelStream.git"
BABEL_TAG = "v5.0"
BABEL_DIR = REPO / "third_party" / "BabelStream"


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def build_babelstream(model: str) -> Path:
    exe = BABEL_DIR / f"build-{model}" / f"{model}-stream"
    if exe.exists():
        return exe
    if not BABEL_DIR.exists():
        sh(["git", "clone", "--depth", "1", "--branch", BABEL_TAG, BABEL_URL, str(BABEL_DIR)])
    bdir = BABEL_DIR / f"build-{model}"
    args = ["cmake", "-S", str(BABEL_DIR), "-B", str(bdir), f"-DMODEL={model}", "-DCMAKE_BUILD_TYPE=Release"]
    if model == "omp":
        args += ["-DCMAKE_CXX_COMPILER=g++", "-DCXX_EXTRA_FLAGS=-march=native"]
    if model == "cuda":
        args += ["-DCMAKE_CUDA_COMPILER=" + os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
                 "-DCUDA_ARCH=" + os.environ.get("CUDA_ARCH", "sm_80")]
    sh(args)
    sh(["cmake", "--build", str(bdir), "-j"])
    return exe


def run_babelstream(exe: Path, n_elems: int, env: dict, prefix: list[str]) -> tuple[float, str]:
    out = sh(prefix + [str(exe), "-s", str(n_elems), "-n", "100"], env=env).stdout
    m = re.search(r"^Triad\s+([\d.]+)", out, re.M)
    if not m:
        raise RuntimeError("could not parse BabelStream output:\n" + out)
    unit_mb = "MBytes/sec" in out
    v = float(m.group(1))
    return (v / 1e3 if unit_mb else v), out    # -> GB/s


def run_gemm(dev: dict, n: int, precision: str, placement: str) -> float:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src" / "torch"))
    prefix = []
    args = ["--workload", "C1", "--params", json.dumps({"precision": precision, "variant": "base", "n": n}),
            "--device-id", dev["device_id"], "--out", str(REPO / "results" / "calibration" / "_runs")]
    if dev["device_class"] == "cpu":
        p = devices.resolve_placement(placement)
        env.update(devices.placement_env(p))
        prefix = devices.numactl_prefix(p)
        if prefix:
            env["BENCH_MEMBIND"] = ",".join(map(str, p.nodes))
        args += ["--device", "cpu", "--placement", placement]
    else:
        args += ["--device", "cuda", "--gpu", str(dev.get("gpu_index", 0))]
    r = sh(prefix + [sys.executable, "-m", "bench", "run"] + args, env=env)
    row = json.loads(r.stdout.strip().splitlines()[-1])
    if row["status"] != "ok":
        raise RuntimeError(f"GEMM {precision} n={n}: {row}")
    return float(row["metric_value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-config", required=True)
    ap.add_argument("--placement", default="socket0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-gemm", action="store_true")
    a = ap.parse_args()
    dev = json.loads(Path(a.device_config).read_text())
    cls = dev["device_class"]
    rec = {"device_id": dev["device_id"], "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "placement": a.placement if cls == "cpu" else f"gpu{dev.get('gpu_index', 0)}",
           "datasheet": dev.get("datasheet", {}), "measured": {}, "provenance": {}}

    # ── bandwidth: BabelStream Triad ────────────────────────────────────────
    model = "omp" if cls == "cpu" else "cuda"
    exe = build_babelstream(model)
    env, prefix = dict(os.environ), []
    if cls == "cpu":
        p = devices.resolve_placement(a.placement)
        env.update(devices.placement_env(p))
        prefix = devices.numactl_prefix(p)
        llc = devices.llc_bytes_for(p)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(dev.get("gpu_index", 0))
        llc = 50 << 20
    n = 1 << 25
    while 3 * 8 * n < 32 * llc:     # 3 fp64 arrays >= 32 x LLC
        n <<= 1
    triads, raw = [], []
    for _ in range(a.reps):
        v, out = run_babelstream(exe, n, env, prefix)
        triads.append(v)
        raw.append(out)
    bw = statistics.median(triads)
    rec["measured"]["bandwidth_gbps"] = bw
    rec["provenance"]["babelstream"] = {
        "url": BABEL_URL, "tag": BABEL_TAG, "model": model, "array_elems": n,
        "array_bytes_total": 3 * 8 * n, "ws_over_llc": 3 * 8 * n / llc,
        "triad_gbps_runs": triads, "command": " ".join(prefix + [str(exe), "-s", str(n), "-n", "100"]),
        "raw_output_last": raw[-1]}
    print(f"BabelStream Triad: median {bw:.1f} GB/s over {a.reps} runs {['%.1f' % t for t in triads]}")

    # ── compute: best GEMM from the harness ─────────────────────────────────
    if not a.skip_gemm:
        peaks = {}
        for prec in ("fp32", "fp64"):
            vals = {sz: run_gemm(dev, sz, prec, a.placement) for sz in (8192, 16384)}
            peaks[prec] = max(vals.values())
            rec["provenance"][f"gemm_{prec}"] = vals
            print(f"GEMM {prec}: {vals} -> peak {peaks[prec]:.0f} GFLOP/s")
        rec["measured"]["peak_gflops"] = peaks

    ds = rec["datasheet"]
    if ds.get("bandwidth_gbps"):
        rec["measured_over_datasheet"] = {"bandwidth": bw / ds["bandwidth_gbps"]}
        for prec, v in rec["measured"].get("peak_gflops", {}).items():
            if ds.get("peak_gflops", {}).get(prec):
                rec["measured_over_datasheet"][f"gemm_{prec}"] = v / ds["peak_gflops"][prec]
    dst = REPO / "results" / "calibration" / f"{dev['device_id']}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rec, indent=2))
    print(f"-> {dst}\n{json.dumps(rec.get('measured_over_datasheet', {}), indent=2)}")


if __name__ == "__main__":
    main()
