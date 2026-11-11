#!/usr/bin/env python3
"""M2: per-device ceilings for pct_measured_peak (docs/methodology.md §10, plan A12).

  measured.bandwidth_gbps   BabelStream (UoB-HPC, well-known), OpenMP backend on CPU (CUDA on GPU),
                            arrays >= 32x LLC, pinned like the harness, median of `--reps` runs.
                            Expressed as ACTUAL DRAM traffic, the convention the workloads use
                            (compulsory bytes). BabelStream/STREAM count Triad as 3 arrays, but on
                            CPUs a normal store first reads the line (write-allocate), so Triad
                            moves 4 arrays and Copy 3: CPU ceiling = max(Dot, Triad*4/3, Copy*3/2)
                            -- e.g. EPYC 7402P: Triad 91.2 -> 121.6, Copy 83.4 -> 125.0, Dot 123.8.
                            A corrected value > 1.05 x Dot is rejected: that CPU did not read the
                            written lines (Xeon 5512U: Copy 181 -> 272 vs Dot 214; ceiling = Dot).
                            Re-derive stored files: calibrate.py --recompute-bandwidth <file.json>...
                            GPU ceiling = reported Triad (no write-allocate for full-line stores).
                            Raw per-kernel rates are stored in provenance.
  measured.peak_gflops[p]   best C1 GEMM median throughput at N=8192 over every BLAS library
                            on the node (torch/OpenBLAS/BLIS on CPU, cuBLAS on GPU), measured by
                            the benchmark harness itself (publication policy).
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
sys.path.insert(0, str(REPO))
from bench import devices  # noqa: E402

BABEL_URL = "https://github.com/UoB-HPC/BabelStream.git"
BABEL_TAG = "v5.0"
BABEL_DIR = REPO / "third_party" / "BabelStream"


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def gpu_arch(gpu_index: int = 0) -> str:
    """sm_XY of the GPU actually present (CUDA_ARCH env overrides). A fixed sm_80 build fails on
    V100 (sm_70) with 'no kernel image is available' -- found on CloudLab c4130."""
    if os.environ.get("CUDA_ARCH"):
        return os.environ["CUDA_ARCH"]
    out = sh(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", str(gpu_index)]).stdout
    major, minor = out.strip().splitlines()[0].split(".")
    return f"sm_{major}{minor}"


def build_babelstream(model: str, gpu_index: int = 0) -> Path:
    tag = model if model != "cuda" else f"cuda-{gpu_arch(gpu_index)}"      # one build dir per GPU arch
    exe = BABEL_DIR / f"build-{tag}" / f"{model}-stream"
    if exe.exists():
        return exe
    if not BABEL_DIR.exists():
        sh(["git", "clone", "--depth", "1", "--branch", BABEL_TAG, BABEL_URL, str(BABEL_DIR)])
    bdir = BABEL_DIR / f"build-{tag}"
    args = ["cmake", "-S", str(BABEL_DIR), "-B", str(bdir), f"-DMODEL={model}", "-DCMAKE_BUILD_TYPE=Release"]
    if model == "omp":
        args += ["-DCMAKE_CXX_COMPILER=g++", "-DCXX_EXTRA_FLAGS=-march=native"]
    if model == "cuda":
        args += ["-DCMAKE_CUDA_COMPILER=" + os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
                 "-DCUDA_ARCH=" + gpu_arch(gpu_index)]
    sh(args)
    sh(["cmake", "--build", str(bdir), "-j"])
    return exe


def run_babelstream(exe: Path, n_elems: int, env: dict, prefix: list[str]) -> tuple[dict, str]:
    """Returns {kernel: GB/s as reported by BabelStream} and the raw output."""
    out = sh(prefix + [str(exe), "-s", str(n_elems), "-n", "100"], env=env).stdout
    scale = 1e-3 if "MBytes/sec" in out else 1.0
    rates = {k: float(v) * scale for k, v in re.findall(r"^(Copy|Mul|Add|Triad|Dot)\s+([\d.]+)", out, re.M)}
    if "Triad" not in rates:
        raise RuntimeError("could not parse BabelStream output:\n" + out)
    return rates, out


# extra write-allocate traffic factor per kernel on CPUs (reads of the written line)
WRITE_ALLOCATE = {"Copy": 3 / 2, "Mul": 3 / 2, "Add": 4 / 3, "Triad": 4 / 3, "Dot": 1.0}


# A read+write stream cannot move more DRAM traffic than a pure-read stream (Dot) on the same memory
# system. A write-allocate-corrected rate more than this far above Dot means the CPU did not read the
# written lines (e.g. Intel SpecI2M on Sapphire/Emerald Rapids), so that correction is not applied.
WA_MAX_OVER_DOT = 1.05


def dram_traffic(rates: dict, device_class: str) -> tuple[float, str]:
    """(ceiling GB/s, how it was derived)."""
    if device_class == "gpu":
        return rates["Triad"], "BabelStream Triad as reported (GPU)"
    cands = {f"{k} (reported)": v for k, v in rates.items()}
    dot = rates.get("Dot")
    for k, v in rates.items():
        if WRITE_ALLOCATE[k] != 1.0:
            c = v * WRITE_ALLOCATE[k]
            if dot is None or c <= WA_MAX_OVER_DOT * dot:
                cands[f"{k} x{WRITE_ALLOCATE[k]:.3g} write-allocate"] = c
    best = max(cands, key=cands.get)
    rejected = [k for k, v in rates.items() if WRITE_ALLOCATE[k] != 1.0 and dot and v * WRITE_ALLOCATE[k] > WA_MAX_OVER_DOT * dot]
    note = f"actual DRAM traffic (CPU): {best}"
    if rejected:
        note += (f"; write-allocate correction not applied ({', '.join(rejected)} corrected would exceed "
                 f"pure-read Dot by >{100 * (WA_MAX_OVER_DOT - 1):.0f}%: this CPU does not read written lines)")
    return cands[best], note


def dram_traffic_gbps(rates: dict, device_class: str) -> float:
    return dram_traffic(rates, device_class)[0]


def recompute_bandwidth(path: Path) -> None:
    """Re-derive the bandwidth ceiling from the BabelStream rates already stored in a calibration file."""
    rec = json.loads(path.read_text())
    med = rec["provenance"]["babelstream"]["reported_gbps_median"]
    cls = "gpu" if "Triad as reported" in rec["measured"].get("bandwidth_convention", "") else "cpu"
    old = rec["measured"]["bandwidth_gbps"]
    bw, note = dram_traffic(med, cls)
    rec["measured"]["bandwidth_gbps"], rec["measured"]["bandwidth_convention"] = bw, note
    if abs(bw - old) > 1e-9:
        rec.setdefault("recomputed", []).append(
            {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "bandwidth_gbps_before": old,
             "reason": "write-allocate correction checked against pure-read Dot (WA_MAX_OVER_DOT)"})
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{path.name}: {old:.1f} -> {bw:.1f} GB/s  [{note}]")


def run_gemm(dev: dict, n: int, precision: str, placement: str, lib: str = "torch") -> float:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    prefix = []
    args = ["--workload", "C1", "--params", json.dumps({"precision": precision, "variant": "base", "n": n, "lib": lib}),
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
    if sys.argv[1:2] == ["--recompute-bandwidth"]:       # offline: re-derive from stored BabelStream rates
        for f in sys.argv[2:]:
            recompute_bandwidth(Path(f))
        return
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
    exe = build_babelstream(model, dev.get("gpu_index", 0))
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
    runs, raw = [], []
    for _ in range(a.reps):
        rates, out = run_babelstream(exe, n, env, prefix)
        runs.append(rates)
        raw.append(out)
    med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
    bw, note = dram_traffic(med, cls)
    rec["measured"]["bandwidth_gbps"] = bw
    rec["measured"]["bandwidth_convention"] = note
    rec["provenance"]["babelstream"] = {
        "url": BABEL_URL, "tag": BABEL_TAG, "model": model, "array_elems": n,
        "array_bytes_total": 3 * 8 * n, "ws_over_llc": 3 * 8 * n / llc,
        "reported_gbps_median": med, "reported_gbps_runs": runs,
        "write_allocate_factors": WRITE_ALLOCATE if cls == "cpu" else None,
        "command": " ".join(prefix + [str(exe), "-s", str(n), "-n", "100"]), "raw_output_last": raw[-1]}
    print("BabelStream (reported, GB/s): " + ", ".join(f"{k} {v:.1f}" for k, v in med.items()))
    print(f"-> bandwidth ceiling {bw:.1f} GB/s ({rec['measured']['bandwidth_convention']})")

    # ── compute: best GEMM from the harness ─────────────────────────────────
    if not a.skip_gemm:
        peaks = {}
        libs = ("torch", "openblas", "blis") if cls == "cpu" else ("torch",)
        for prec in ("fp32", "fp64"):
            vals = {}
            for lib in libs:                       # peak = best well-known library at N=8192
                try:
                    vals[lib] = run_gemm(dev, 8192, prec, a.placement, lib)
                except Exception as e:  # library absent on this node
                    vals[lib] = f"unavailable: {str(e)[:80]}"
            nums = [v for v in vals.values() if isinstance(v, float)]
            peaks[prec] = max(nums)
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
