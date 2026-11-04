#!/usr/bin/env python3
"""src/torch/torch_gemm_bench.py

PyTorch Dense Matrix-Matrix Multiplication (GEMM) Benchmark (C = alpha * A * B + beta * C).
Mirrors src/cpu/cpu_gemm_bench.c & src/gpu/gpu_gemm_bench.cu.

Supported precisions:
  - fp32: torch.float32
  - fp64: torch.float64
  - bf16: torch.bfloat16

Devices:
  - cpu (Intel Xeon with AMX/AVX-512, AMD EPYC with Zen3/Zen4)
  - cuda (NVIDIA Tensor Core GPUs)

Scaling & Features:
  - Sweeps problem sizes (S) x precisions x thread counts (T) or GPUs.
  - Multi-threaded CPU scaling via torch.set_num_threads(T).
  - Telemetry integration (Watts, MHz, peak °C).
  - Emits identical CSV schema for direct ingestion by plotting scripts.
"""

import os
import sys
import time
import argparse
import platform
import torch

from torch_telemetry import HardwareTelemetry, get_system_host_info

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "bf16": torch.bfloat16,
}

ELEM_BYTES = {
    "fp32": 4,
    "fp64": 8,
    "bf16": 2,
}


def validate_gemm(S, A, B, C, precision):
    """Numerical accuracy validation against double-precision dot product reference."""
    check_samples = 32
    max_rel = 0.0

    A_ref = A.to(torch.float64) if A.device.type == "cpu" else A.cpu().to(torch.float64)
    B_ref = B.to(torch.float64) if B.device.type == "cpu" else B.cpu().to(torch.float64)
    C_comp = C.cpu()

    for sample in range(check_samples):
        r = (sample * 1337 + 7) % S
        c = (sample * 31337 + 13) % S

        # Reference dot product
        ref = torch.dot(A_ref[r, :], B_ref[:, c]).item()
        computed = float(C_comp[r, c].item())

        denom = abs(ref) if abs(ref) > 1e-12 else 1.0
        rel = abs(computed - ref) / denom
        if rel > max_rel:
            max_rel = rel

    return max_rel


def run_gemm(args, S, prec_str, threads_or_gpu, device, telem, csv_file, host, engine_info):
    dtype = DTYPE_MAP[prec_str]
    elem_bytes = ELEM_BYTES[prec_str]
    mem_gb = (3.0 * S * S * elem_bytes) / 1e9

    if device.type == "cpu":
        torch.set_num_threads(threads_or_gpu)
        torch.set_num_interop_threads(1)

    # 1. Deterministic Tensor Allocation (Outside timed region)
    # Use deterministic seed for reproducibility
    torch.manual_seed(42)
    A = torch.randn((S, S), dtype=dtype, device=device)
    B = torch.randn((S, S), dtype=dtype, device=device)
    C = torch.empty((S, S), dtype=dtype, device=device)

    # 2. Untimed Warmup Iterations
    for _ in range(args.warmup):
        torch.mm(A, B, out=C)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 3. Timed Region with Telemetry
    telem.start()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.iters):
        torch.mm(A, B, out=C)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    telem.stop()

    wall_sec = t1 - t0
    comp_ms = (wall_sec * 1000.0) / float(args.iters)
    total_flops = 2.0 * float(S) * float(S) * float(S) * float(args.iters)
    agg_tflops = (total_flops / wall_sec / 1e12) if wall_sec > 0.0 else 0.0

    # 4. Numerical Accuracy Validation
    validated = 0
    max_rel = -1.0
    if args.validate and S <= args.validate_max:
        max_rel = validate_gemm(S, A, B, C, prec_str)
        validated = 1

    # Tolerate higher relative error in BF16 due to 8-bit mantissa
    err_tol = 1e-1 if prec_str == "bf16" else 1e-2
    status = "HIERR" if (validated and max_rel > err_tol) else "ok"

    # 5. Console Output
    print(f"{'torch':<6} {prec_str:<5} {threads_or_gpu:<3} {S:>6} {agg_tflops:>9.2f} "
          f"{comp_ms:>8.2f} {telem.avg_power_watts:>6.1f}W {telem.avg_freq_mhz:>7.1f} {status}",
          flush=True)

    # 6. Append to CSV
    if csv_file:
        ts = int(time.time())
        # Identical CSV Schema
        row = (
            f"{args.tag},{ts},{host},{engine_info},{args.mode},{prec_str},{threads_or_gpu},"
            f"{S},{S},{S},{args.iters},{args.warmup},"
            f"{agg_tflops:.2f},{comp_ms:.3f},{wall_sec * 1000.0:.3f},"
            f"{telem.avg_freq_mhz:.1f},{telem.avg_power_watts:.1f},{telem.max_temp_c:.1f},"
            f"{mem_gb:.3f},{validated},{max_rel:.3e},{status}\n"
        )
        csv_file.write(row)
        csv_file.flush()

    del A, B, C
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="PyTorch GEMM Benchmark")
    parser.add_argument("--sizes", type=str, default="512,1024,2048,4096,8192",
                        help="Matrix dimensions S,S,... [default: 512,1024,2048,4096,8192]")
    parser.add_argument("--precisions", type=str, default="fp32,fp64,bf16",
                        help="Precisions: fp32,fp64,bf16 [default: fp32,fp64,bf16]")
    parser.add_argument("--threads", type=str, default=None,
                        help="Thread counts T,T,... [default: powers of 2 up to max cores]")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Alias for --threads or GPU IDs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Compute device: cpu or cuda [default: cpu]")
    parser.add_argument("--mode", type=str, default="split", choices=["split", "replicas"],
                        help="Execution mode [default: split]")
    parser.add_argument("--iters", type=int, default=15,
                        help="Timed loop iterations [default: 15]")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed warmup runs [default: 3]")
    parser.add_argument("--validate", action="store_true",
                        help="Validate numerical accuracy against double reference")
    parser.add_argument("--validate-max", type=int, default=2048,
                        help="Max matrix dimension to validate [default: 2048]")
    parser.add_argument("--csv", type=str, default=None,
                        help="Append results to CSV file")
    parser.add_argument("--tag", type=str, default="torch_gemm",
                        help="Label tag in CSV [default: torch_gemm]")

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]

    # Hardware & Threads resolution
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

    host, cpu_model = get_system_host_info()
    engine_info = f"PyTorch_{torch.__version__}"
    if device.type == "cpu":
        # Check backend BLAS if available
        engine_info += "_CPU"
    else:
        engine_info += f"_{torch.cuda.get_device_name(0).replace(' ', '_')}"

    # Setup CSV
    csv_file = None
    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_file = open(args.csv, "a", encoding="utf-8")
        if new_file:
            header = (
                "tag,ts,host,mkl_version,mode,precision,threads,M,N,K,iters,warmup,"
                "agg_tflops,comp_ms,wall_ms,"
                "cpu_mhz_avg,power_w_avg,temp_c_max,"
                "mem_gb,validated,max_rel_err,status\n"
            )
            csv_file.write(header)
            csv_file.flush()

    telem = HardwareTelemetry(device_type=device.type)

    print(f"# PyTorch GEMM Benchmark ({engine_info}) host={host} device={device}")
    print(f"# Precisions swept: {', '.join(precisions)}")
    print(f"{'eng':<6} {'prec':<5} {'T':<3} {'N':>6} {'aggTF':>9} {'comp_ms':>8} {'power':>7} {'MHz':>7} {'status'}")

    try:
        for S in sizes:
            for prec in precisions:
                if prec not in DTYPE_MAP:
                    continue
                for T in thread_counts:
                    run_gemm(args, S, prec, T, device, telem, csv_file, host, engine_info)
    finally:
        telem.close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
