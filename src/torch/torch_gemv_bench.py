#!/usr/bin/env python3
"""src/torch/torch_gemv_bench.py

PyTorch GEMV (Dense Matrix-Vector) & SpMV (Sparse CSR Matrix-Vector) Benchmark.
Computes: y = alpha * A * x + beta * y
Mirrors src/cpu/cpu_gemv_bench.c & src/gpu/gpu_gemv_bench.cu.

Workloads:
  1. Dense GEMV:
     - fp32: torch.mv (float32)
     - fp64: torch.mv (float64)
  2. Sparse SpMV (CSR format):
     - fp32: torch.sparse.mm / torch.mv on torch.sparse_csr_tensor
     - fp64: torch.sparse.mm / torch.mv on torch.sparse_csr_tensor

Arithmetic Intensity: ~2 FLOPs / element. Strictly memory-bandwidth bound!
Headline metrics: Achieved Memory Bandwidth (GB/s) and Compute (TFLOPS).

Emits identical CSV schema for direct ingestion by results/plotting/cpu/generate_cpu_gemv_plots.py.
"""

import os
import sys
import time
import argparse
import platform
import numpy as np
import torch

from torch_telemetry import HardwareTelemetry, get_system_host_info

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp64": torch.float64,
}

ELEM_BYTES = {
    "fp32": 4,
    "fp64": 8,
}


def build_random_csr_matrix(S, density, dtype, device):
    """Build a valid, deterministically distributed CSR sparse tensor in PyTorch."""
    nnz_target = max(1, int(S * S * density))
    entries_per_row = max(1, nnz_target // S)
    if entries_per_row > S:
        entries_per_row = S

    crow_indices = [0]
    col_indices_list = []
    current_nnz = 0

    rng = np.random.default_rng(seed=42)
    for _ in range(S):
        cols = rng.choice(S, size=entries_per_row, replace=False)
        cols.sort()
        col_indices_list.extend(cols)
        current_nnz += entries_per_row
        crow_indices.append(current_nnz)

    actual_nnz = current_nnz
    values = torch.randn(actual_nnz, dtype=dtype, device=device)
    crow = torch.tensor(crow_indices, dtype=torch.int32, device=device)
    col = torch.tensor(col_indices_list, dtype=torch.int32, device=device)

    # PyTorch sparse_csr_tensor
    A_csr = torch.sparse_csr_tensor(crow, col, values, size=(S, S), dtype=dtype, device=device)
    return A_csr, actual_nnz


def run_dense_gemv(args, S, prec_str, threads_or_gpu, device, telem, csv_file, host, engine_info):
    dtype = DTYPE_MAP[prec_str]
    elem_bytes = ELEM_BYTES[prec_str]

    if device.type == "cpu":
        torch.set_num_threads(threads_or_gpu)
        torch.set_num_interop_threads(1)

    torch.manual_seed(42)
    A = torch.randn((S, S), dtype=dtype, device=device)
    x = torch.randn(S, dtype=dtype, device=device)
    y = torch.empty(S, dtype=dtype, device=device)

    # Warmup
    for _ in range(args.warmup):
        torch.mv(A, x, out=y)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed Loop
    telem.start()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.iters):
        torch.mv(A, x, out=y)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    telem.stop()

    wall_sec = t1 - t0
    comp_ms = (wall_sec * 1000.0) / float(args.iters)
    total_flops = 2.0 * float(S) * float(S) * float(args.iters)
    agg_tflops = (total_flops / wall_sec / 1e12) if wall_sec > 0.0 else 0.0

    total_bytes = float(S * S + 2 * S) * elem_bytes * float(args.iters)
    agg_gbps = (total_bytes / wall_sec / 1e9) if wall_sec > 0.0 else 0.0
    mem_gb = (float(S * S + 2 * S) * elem_bytes) / 1e9

    # Validation
    validated = 0
    max_rel = -1.0
    if args.validate and S <= args.validate_max:
        check_samples = min(32, S)
        A_cpu = A.cpu().to(torch.float64)
        x_cpu = x.cpu().to(torch.float64)
        y_comp = y.cpu()
        max_rel = 0.0
        for sample in range(check_samples):
            idx = (sample * 1337 + 7) % S
            ref = torch.dot(A_cpu[idx, :], x_cpu).item()
            comp = float(y_comp[idx].item())
            denom = abs(ref) if abs(ref) > 1e-12 else 1.0
            rel = abs(comp - ref) / denom
            if rel > max_rel:
                max_rel = rel
        validated = 1

    status = "HIERR" if (validated and max_rel > 1e-2) else "ok"

    print(f"{'dense':<6} {prec_str:<5} {threads_or_gpu:<3} {S:>6} {'-':>6} "
          f"{agg_gbps:>8.2f} {agg_tflops:>7.3f} {telem.avg_power_watts:>6.1f}W {telem.avg_freq_mhz:>7.1f} {status}",
          flush=True)

    if csv_file:
        ts = int(time.time())
        row = (
            f"{args.tag},{ts},{host},{engine_info},dense,split,{prec_str},{threads_or_gpu},"
            f"{S},1.000000,{S * S},{args.iters},{args.warmup},"
            f"{agg_gbps:.2f},{agg_tflops:.3f},{comp_ms:.3f},{wall_sec * 1000.0:.3f},"
            f"{telem.avg_freq_mhz:.1f},{telem.avg_power_watts:.1f},{telem.max_temp_c:.1f},"
            f"{mem_gb:.3f},{validated},{max_rel:.3e},{status}\n"
        )
        csv_file.write(row)
        csv_file.flush()

    del A, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_sparse_spmv(args, S, density, prec_str, threads_or_gpu, device, telem, csv_file, host, engine_info):
    dtype = DTYPE_MAP[prec_str]
    elem_bytes = ELEM_BYTES[prec_str]

    if device.type == "cpu":
        torch.set_num_threads(threads_or_gpu)
        torch.set_num_interop_threads(1)

    A_csr, nnz = build_random_csr_matrix(S, density, dtype, device)
    x = torch.randn((S, 1), dtype=dtype, device=device)
    y = torch.empty((S, 1), dtype=dtype, device=device)

    # Warmup
    for _ in range(args.warmup):
        torch.sparse.mm(A_csr, x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed Loop
    telem.start()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.iters):
        y = torch.sparse.mm(A_csr, x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    telem.stop()

    wall_sec = t1 - t0
    comp_ms = (wall_sec * 1000.0) / float(args.iters)
    total_flops = 2.0 * float(nnz) * float(args.iters)
    agg_tflops = (total_flops / wall_sec / 1e12) if wall_sec > 0.0 else 0.0

    # Data volume in CSR format: values (nnz*elem) + col_indices (nnz*4) + crow ( (S+1)*4 ) + x (S*elem) + y (S*elem)
    single_iter_bytes = float(nnz * elem_bytes + nnz * 4 + (S + 1) * 4 + 2 * S * elem_bytes)
    total_bytes = single_iter_bytes * float(args.iters)
    agg_gbps = (total_bytes / wall_sec / 1e9) if wall_sec > 0.0 else 0.0
    mem_gb = single_iter_bytes / 1e9

    # Validation
    validated = 0
    max_rel = -1.0
    if args.validate and S <= args.validate_max:
        check_samples = min(32, S)
        # Convert A_csr to dense for reference verification
        A_dense_cpu = A_csr.to_dense().cpu().to(torch.float64)
        x_cpu = x.squeeze(1).cpu().to(torch.float64)
        y_comp = y.squeeze(1).cpu()
        max_rel = 0.0
        for sample in range(check_samples):
            idx = (sample * 1337 + 7) % S
            ref = torch.dot(A_dense_cpu[idx, :], x_cpu).item()
            comp = float(y_comp[idx].item())
            denom = abs(ref) if abs(ref) > 1e-12 else 1.0
            rel = abs(comp - ref) / denom
            if rel > max_rel:
                max_rel = rel
        validated = 1

    status = "HIERR" if (validated and max_rel > 1e-2) else "ok"

    print(f"{'sparse':<6} {prec_str:<5} {threads_or_gpu:<3} {S:>6} {density:>6.2f} "
          f"{agg_gbps:>8.2f} {agg_tflops:>7.3f} {telem.avg_power_watts:>6.1f}W {telem.avg_freq_mhz:>7.1f} {status}",
          flush=True)

    if csv_file:
        ts = int(time.time())
        row = (
            f"{args.tag},{ts},{host},{engine_info},sparse,split,{prec_str},{threads_or_gpu},"
            f"{S},{density:.6f},{nnz},{args.iters},{args.warmup},"
            f"{agg_gbps:.2f},{agg_tflops:.3f},{comp_ms:.3f},{wall_sec * 1000.0:.3f},"
            f"{telem.avg_freq_mhz:.1f},{telem.avg_power_watts:.1f},{telem.max_temp_c:.1f},"
            f"{mem_gb:.3f},{validated},{max_rel:.3e},{status}\n"
        )
        csv_file.write(row)
        csv_file.flush()

    del A_csr, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="PyTorch GEMV & SpMV Benchmark")
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,8192,16384",
                        help="Matrix dimensions S,S,... [default: 1024,2048,4096,8192,16384]")
    parser.add_argument("--engine", type=str, default="all", choices=["dense", "sparse", "all"],
                        help="Engine workload: dense, sparse, or all [default: all]")
    parser.add_argument("--density", type=str, default="0.01,0.05,0.10",
                        help="CSR densities for SpMV [default: 0.01,0.05,0.10]")
    parser.add_argument("--precisions", type=str, default="fp32,fp64",
                        help="Precisions: fp32,fp64 [default: fp32,fp64]")
    parser.add_argument("--threads", type=str, default=None,
                        help="Thread counts T,T,... [default: powers of 2 up to max cores]")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Alias for --threads or GPU IDs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Compute device: cpu or cuda [default: cpu]")
    parser.add_argument("--iters", type=int, default=15,
                        help="Timed loop iterations [default: 15]")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed warmup runs [default: 3]")
    parser.add_argument("--validate", action="store_true",
                        help="Validate numerical accuracy")
    parser.add_argument("--validate-max", type=int, default=2048,
                        help="Max matrix dimension to validate [default: 2048]")
    parser.add_argument("--csv", type=str, default=None,
                        help="Append results to CSV file")
    parser.add_argument("--tag", type=str, default="torch_gemv",
                        help="Label tag in CSV [default: torch_gemv]")

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    densities = [float(d.strip()) for d in args.density.split(",") if d.strip()]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")

    threads_arg = args.threads or args.gpus
    if threads_arg:
        thread_counts = [int(t.strip()) for t in threads_arg.split(",") if t.strip()]
    else:
        if device.type == "cpu":
            max_cores = os.cpu_count() or 4
            thread_counts = []
            t = 1
            while t <= max_cores:
                thread_counts.append(t)
                t *= 2
            if thread_counts[-1] != max_cores:
                thread_counts.append(max_cores)
        else:
            thread_counts = [1]

    host, _ = get_system_host_info()
    engine_info = f"PyTorch_{torch.__version__}"

    csv_file = None
    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_file = open(args.csv, "a", encoding="utf-8")
        if new_file:
            header = (
                "tag,ts,host,mkl_version,engine,mode,precision,threads,S,density,nnz,iters,warmup,"
                "agg_gbps,agg_tflops,comp_ms,wall_ms,"
                "cpu_mhz_avg,power_w_avg,temp_c_max,"
                "mem_gb,validated,max_rel_err,status\n"
            )
            csv_file.write(header)
            csv_file.flush()

    telem = HardwareTelemetry(device_type=device.type)

    print(f"# PyTorch GEMV/SpMV Benchmark ({engine_info}) host={host} device={device}")
    print(f"{'eng':<6} {'prec':<5} {'T':<3} {'S':>6} {'dens':>6} {'GB/s':>8} {'TFLOPS':>7} {'power':>7} {'MHz':>7} {'status'}")

    run_dense = args.engine in ["dense", "all"]
    run_sparse = args.engine in ["sparse", "all"]

    try:
        for S in sizes:
            # Dense Run
            if run_dense:
                for prec in precisions:
                    if prec not in DTYPE_MAP:
                        continue
                    for T in thread_counts:
                        run_dense_gemv(args, S, prec, T, device, telem, csv_file, host, engine_info)

            # Sparse Run
            if run_sparse:
                for dens in densities:
                    for prec in precisions:
                        if prec not in DTYPE_MAP:
                            continue
                        for T in thread_counts:
                            run_sparse_spmv(args, S, dens, prec, T, device, telem, csv_file, host, engine_info)
    finally:
        telem.close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
