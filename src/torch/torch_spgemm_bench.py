#!/usr/bin/env python3
"""src/torch/torch_spgemm_bench.py

PyTorch Sparse-Sparse GEMM (SpGEMM) Benchmark:
  C = alpha * A * B   where A, B, and C are ALL sparse CSR matrices.
Mirrors src/cpu/cpu_spgemm_bench.c & src/gpu/gpu_spgemm_bench.cu.

Workload:
  Matrix-Matrix multiplication of two sparse matrices in CSR format.
  Uses torch.sparse.mm on torch.sparse_csr_tensor operands.

Precisions:
  - fp32: torch.float32
  - fp64: torch.float64

FLOP Counting:
  Theoretical FLOPs = 2 * density_A * density_B * S^3 * iters

Emits identical CSV schema for direct ingestion by results/plotting/cpu/generate_cpu_spgemm_plots.py.
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


def build_random_csr_matrix(S, density, dtype, device, seed=42):
    """Build a deterministically distributed CSR sparse tensor in PyTorch."""
    nnz_target = max(1, int(S * S * density))
    entries_per_row = max(1, nnz_target // S)
    if entries_per_row > S:
        entries_per_row = S

    crow_indices = [0]
    col_indices_list = []
    current_nnz = 0

    rng = np.random.default_rng(seed=seed)
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

    csr_mat = torch.sparse_csr_tensor(crow, col, values, size=(S, S), dtype=dtype, device=device)
    return csr_mat, actual_nnz


def run_spgemm(args, S, density, prec_str, threads_or_gpu, device, telem, csv_file, host, engine_info):
    dtype = DTYPE_MAP[prec_str]
    elem_bytes = ELEM_BYTES[prec_str]

    if device.type == "cpu":
        torch.set_num_threads(threads_or_gpu)
        torch.set_num_interop_threads(1)

    A_csr, nnz_A = build_random_csr_matrix(S, density, dtype, device, seed=42)
    B_csr, nnz_B = build_random_csr_matrix(S, density, dtype, device, seed=1337)

    # Warmup
    C = None
    for _ in range(args.warmup):
        C = torch.sparse.mm(A_csr, B_csr)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed Loop
    telem.start()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.iters):
        C = torch.sparse.mm(A_csr, B_csr)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    telem.stop()

    wall_sec = t1 - t0
    comp_ms = (wall_sec * 1000.0) / float(args.iters)

    # FLOP counting matches the benchmark suite formula
    total_flops = 2.0 * density * density * float(S) * float(S) * float(S) * float(args.iters)
    agg_tflops = (total_flops / wall_sec / 1e12) if wall_sec > 0.0 else 0.0

    nnz_C = C._nnz() if hasattr(C, "_nnz") else (C._values().numel() if hasattr(C, "_values") else 0)

    # Memory bytes for CSR storage
    bytes_A = nnz_A * elem_bytes + nnz_A * 4 + (S + 1) * 4
    bytes_B = nnz_B * elem_bytes + nnz_B * 4 + (S + 1) * 4
    bytes_C = nnz_C * elem_bytes + nnz_C * 4 + (S + 1) * 4
    total_bytes = float(bytes_A + bytes_B + bytes_C) * float(args.iters)
    agg_gbps = (total_bytes / wall_sec / 1e9) if wall_sec > 0.0 else 0.0
    mem_gb = float(bytes_A + bytes_B + bytes_C) / 1e9

    # Validation
    validated = 0
    max_rel = -1.0
    if args.validate and S <= args.validate_max:
        check_samples = min(16, S)
        A_dense = A_csr.to_dense().cpu().to(torch.float64)
        B_dense = B_csr.to_dense().cpu().to(torch.float64)
        C_dense = C.to_dense().cpu().to(torch.float64)
        max_rel = 0.0
        for sample in range(check_samples):
            r = (sample * 1337 + 7) % S
            c = (sample * 31337 + 13) % S
            ref = torch.dot(A_dense[r, :], B_dense[:, c]).item()
            comp = C_dense[r, c].item()
            denom = abs(ref) if abs(ref) > 1e-12 else 1.0
            rel = abs(comp - ref) / denom
            if rel > max_rel:
                max_rel = rel
        validated = 1

    status = "HIERR" if (validated and max_rel > 1e-2) else "ok"

    print(f"{'spgemm':<6} {prec_str:<5} {threads_or_gpu:<3} {S:>6} {density:>6.2f} "
          f"{agg_tflops:>7.3f} {agg_gbps:>8.2f} {comp_ms:>8.2f} "
          f"{telem.avg_power_watts:>6.1f}W {telem.avg_freq_mhz:>7.1f} {status}",
          flush=True)

    if csv_file:
        ts = int(time.time())
        row = (
            f"{args.tag},{ts},{host},{engine_info},split,{prec_str},{threads_or_gpu},"
            f"{S},{density:.6f},{nnz_A},{nnz_B},{nnz_C},{args.iters},{args.warmup},"
            f"{agg_tflops:.3f},{agg_gbps:.2f},{comp_ms:.3f},{wall_sec * 1000.0:.3f},"
            f"{telem.avg_freq_mhz:.1f},{telem.avg_power_watts:.1f},{telem.max_temp_c:.1f},"
            f"{mem_gb:.3f},{validated},{max_rel:.3e},{status}\n"
        )
        csv_file.write(row)
        csv_file.flush()

    del A_csr, B_csr, C
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Sparse-Sparse GEMM (SpGEMM) Benchmark")
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,8192",
                        help="Matrix dimensions S,S,... [default: 1024,2048,4096,8192]")
    parser.add_argument("--density", type=str, default="0.01,0.05,0.10",
                        help="CSR densities [default: 0.01,0.05,0.10]")
    parser.add_argument("--precisions", type=str, default="fp32,fp64",
                        help="Precisions: fp32,fp64 [default: fp32,fp64]")
    parser.add_argument("--threads", type=str, default=None,
                        help="Thread counts T,T,... [default: powers of 2 up to max cores]")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Alias for --threads or GPU IDs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Compute device: cpu or cuda [default: cpu]")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed loop iterations [default: 10]")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Untimed warmup runs [default: 2]")
    parser.add_argument("--validate", action="store_true",
                        help="Validate numerical accuracy")
    parser.add_argument("--validate-max", type=int, default=2048,
                        help="Max matrix dimension to validate [default: 2048]")
    parser.add_argument("--csv", type=str, default=None,
                        help="Append results to CSV file")
    parser.add_argument("--tag", type=str, default="torch_spgemm",
                        help="Label tag in CSV [default: torch_spgemm]")

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
                "tag,ts,host,mkl_version,mode,precision,threads,S,density,nnz_A,nnz_B,nnz_C,iters,warmup,"
                "agg_tflops,agg_gbps,comp_ms,wall_ms,"
                "cpu_mhz_avg,power_w_avg,temp_c_max,"
                "mem_gb,validated,max_rel_err,status\n"
            )
            csv_file.write(header)
            csv_file.flush()

    telem = HardwareTelemetry(device_type=device.type)

    print(f"# PyTorch SpGEMM Benchmark ({engine_info}) host={host} device={device}")
    print(f"{'work':<6} {'prec':<5} {'T':<3} {'S':>6} {'dens':>6} {'TFLOPS':>7} {'GB/s':>8} {'comp_ms':>8} {'power':>7} {'MHz':>7} {'status'}")

    try:
        for S in sizes:
            for dens in densities:
                for prec in precisions:
                    if prec not in DTYPE_MAP:
                        continue
                    for T in thread_counts:
                        run_spgemm(args, S, dens, prec, T, device, telem, csv_file, host, engine_info)
    finally:
        telem.close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
