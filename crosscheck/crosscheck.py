#!/usr/bin/env python3
"""Cross-check: the benchmark harness vs two independent native C benches (docs/methodology.md).

Run from the repo root:  python3 crosscheck/crosscheck.py --device-id <DEVICE>
The native binaries are built on demand (make -C crosscheck, BLAS=openblas by default).

For each configuration, run the harness (publication policy) and the native binary under
identical pinning (OMP_PLACES + numactl) and compare per-call times. Two questions:
  1. Agreement: do two independent implementations measure the same thing? (|diff| <= 3%
     when both use the same library class; otherwise the gap is a *library* difference.)
  2. Fairness: is torch's backend the fastest well-known library on this CPU? If the native
     library is faster by > 3%, the harness must switch backend for that workload.
Output: results/crosscheck/<DEVICE>.json  (+ a printed table)

  python3 scripts/crosscheck_native.py --device-id AMD_C6525_100G --placement socket0
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bench import devices  # noqa: E402

CASES = [
    # (workload, harness params, native binary, native args, native time column)
    # Like-for-like: the harness calls the SAME library as the native binary (OpenBLAS via
    # CBLAS for GEMM; SAXPY is a single streaming pass in both). Library-vs-library speed is
    # a separate question answered by the sweep's `lib` variants.
    ("C1", {"precision": "fp32", "variant": "base", "n": 4096, "lib": "openblas"}, "cpu_gemm_bench", ["--sizes", "4096", "--precisions", "fp32"], "comp_ms"),
    ("C1", {"precision": "fp32", "variant": "base", "n": 8192, "lib": "openblas"}, "cpu_gemm_bench", ["--sizes", "8192", "--precisions", "fp32"], "comp_ms"),
    ("C1", {"precision": "fp64", "variant": "base", "n": 4096, "lib": "openblas"}, "cpu_gemm_bench", ["--sizes", "4096", "--precisions", "fp64"], "comp_ms"),
    ("C1", {"precision": "fp64", "variant": "base", "n": 8192, "lib": "openblas"}, "cpu_gemm_bench", ["--sizes", "8192", "--precisions", "fp64"], "comp_ms"),
    ("B1", {"precision": "fp32", "variant": "base", "n": 1 << 28}, "cpu_saxpy_bench", ["--sizes", str(1 << 28), "--precisions", "fp32"], "comp_ms"),
    ("B1", {"precision": "fp64", "variant": "base", "n": 1 << 27}, "cpu_saxpy_bench", ["--sizes", str(1 << 27), "--precisions", "fp64"], "comp_ms"),
]
TOL = 0.03


def run_harness(wid, params, placement, device_id, out):
    p = devices.resolve_placement(placement)
    env = dict(os.environ, PYTHONPATH=str(REPO), **devices.placement_env(p))
    pre = devices.numactl_prefix(p)
    if pre:
        env["BENCH_MEMBIND"] = ",".join(map(str, p.nodes))
    cmd = pre + [sys.executable, "-m", "bench", "run", "--workload", wid, "--placement", placement,
                 "--params", json.dumps(params), "--device-id", device_id, "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_native(binary, args, col, placement, iters):
    p = devices.resolve_placement(placement)
    env = dict(os.environ, **devices.placement_env(p))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    os.unlink(path)
    exe = REPO / "crosscheck" / "bin" / binary
    if not exe.exists():                       # cpu_gemm_bench -> make target cpu_gemm
        subprocess.run(["make", "-C", str(REPO / "crosscheck"), binary.replace("_bench", "")], check=True)
    cmd = devices.numactl_prefix(p) + [str(exe), *args, "--threads", str(p.threads),
                                       "--iters", str(iters), "--warmup", "5", "--csv", path]
    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    rows = list(csv.DictReader(open(path)))
    os.unlink(path)
    return float(rows[-1][col]), rows[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--placement", default="socket0")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    out = tempfile.mkdtemp(prefix="xcheck_")
    results = []
    print(f"{'case':34} {'harness ms':>11} {'native ms':>10} {'diff':>7}  verdict")
    for wid, params, binary, nargs, col in CASES:
        h = run_harness(wid, params, a.placement, a.device_id, out)
        if h["status"] != "ok":
            print(f"{wid} {params}: harness status {h['status']}")
            continue
        n_ms, nrow = run_native(binary, nargs, col, a.placement, a.iters)
        diff = (h["median_ms"] - n_ms) / n_ms          # + = harness slower
        lib_native = nrow.get("mkl_version") or nrow.get("engine") or "?"
        if abs(diff) <= TOL:
            verdict = "agree"
        elif diff > 0:
            verdict = f"NATIVE FASTER ({lib_native}) -> library choice must be revisited"
        else:
            verdict = "harness faster (native library slower)"
        label = f"{wid} {params['precision']} n={params['n']} {params.get('lib', '')}"
        print(f"{label:34} {h['median_ms']:11.4f} {n_ms:10.4f} {diff:+7.1%}  {verdict}")
        results.append({"workload": wid, "params": params, "harness_median_ms": h["median_ms"],
                        "harness_cv": h["cv"], "harness_flag": h["flag"], "native_mean_ms": n_ms,
                        "native_binary": binary, "native_row": nrow, "diff": diff, "verdict": verdict})
    dst = REPO / "results" / "crosscheck" / f"{a.device_id}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"device_id": a.device_id, "placement": a.placement,
                               "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "tolerance": TOL, "cases": results}, indent=2, default=str))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
