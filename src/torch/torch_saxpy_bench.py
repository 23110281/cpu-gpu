#!/usr/bin/env python3
"""src/torch/torch_saxpy_bench.py

PyTorch SAXPY / DAXPY (Vector AXPY) Benchmark.
Computes: y = alpha * x + y
Mirrors src/cpu/cpu_saxpy_bench.c & src/gpu/gpu_saxpy_bench.cu.

Precisions:
  - fp32: torch.float32
  - fp64: torch.float64

Arithmetic Intensity:
  - fp32: 2 FLOPs / (3 * 4 bytes) = 0.167 FLOPs/byte
  - fp64: 2 FLOPs / (3 * 8 bytes) = 0.083 FLOPs/byte
Strictly memory-bandwidth bound at large vector lengths (N).
Headline metric: achieved memory bandwidth (GB/s).

Emits identical CSV schema for direct ingestion by results/plotting/cpu/generate_cpu_saxpy_plots.py.
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
}

ELEM_BYTES = {
    "fp32": 4,
    "fp64": 8,
}


def run_saxpy(args, N, prec_str, threads_or_gpu, device, telem, csv_file, host, engine_info):
    dtype = DTYPE_MAP[prec_str]
    elem_bytes = ELEM_BYTES[prec_str]
    alpha = 2.5

    if device.type == "cpu":
        torch.set_num_threads(threads_or_gpu)
        torch.set_num_interop_threads(1)

    torch.manual_seed(42)
    x = torch.randn(N, dtype=dtype, device=device)
    y = torch.randn(N, dtype=dtype, device=device)

    # Save copy of y for validation if requested
    y_init = None
    if args.validate and N <= args.validate_max:
        y_init = y.clone()

    # Warmup
    for _ in range(args.warmup):
        y.add_(x, alpha=alpha)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Reset y for benchmark
    if y_init is not None:
        y.copy_(y_init)
    else:
        y.fill_(1.0)

    # Timed Loop
    telem.start()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.iters):
        y.add_(x, alpha=alpha)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    telem.stop()

    wall_sec = t1 - t0
    comp_ms = (wall_sec * 1000.0) / float(args.iters)
    total_flops = 2.0 * float(N) * float(args.iters)
    agg_tflops = (total_flops / wall_sec / 1e12) if wall_sec > 0.0 else 0.0

    # 3 memory transactions per element per iteration: 2 reads (x, y) + 1 write (y)
    total_bytes = 3.0 * float(N) * elem_bytes * float(args.iters)
    agg_gbps = (total_bytes / wall_sec / 1e9) if wall_sec > 0.0 else 0.0
    mem_gb = (2.0 * float(N) * elem_bytes) / 1e9

    # Validation
    validated = 0
    max_rel = -1.0
    if y_init is not None:
        check_samples = min(64, N)
        x_cpu = x.cpu().to(torch.float64)
        y0_cpu = y_init.cpu().to(torch.float64)
        y_comp = y.cpu()
        max_rel = 0.0
        for sample in range(check_samples):
            idx = (sample * 1337 + 7) % N
            ref = y0_cpu[idx].item() + (float(args.iters) * alpha * x_cpu[idx].item())
            comp = float(y_comp[idx].item())
            denom = abs(ref) if abs(ref) > 1e-12 else 1.0
            rel = abs(comp - ref) / denom
            if rel > max_rel:
                max_rel = rel
        validated = 1

    status = "HIERR" if (validated and max_rel > 1e-2) else "ok"

    print(f"{'saxpy':<6} {prec_str:<5} {threads_or_gpu:<3} {N:>10} "
          f"{agg_gbps:>8.2f} {agg_tflops:>7.3f} {comp_ms:>7.2f} "
          f"{telem.avg_power_watts:>6.1f}W {telem.avg_freq_mhz:>7.1f} {status}",
          flush=True)

    if csv_file:
        ts = int(time.time())
        row = (
            f"{args.tag},{ts},{host},{engine_info},split,{prec_str},{threads_or_gpu},"
            f"{N},{args.iters},{args.warmup},"
            f"{agg_gbps:.2f},{agg_tflops:.3f},{comp_ms:.3f},{wall_sec * 1000.0:.3f},"
            f"{telem.avg_freq_mhz:.1f},{telem.avg_power_watts:.1f},{telem.max_temp_c:.1f},"
            f"{mem_gb:.3f},{validated},{max_rel:.3e},{status}\n"
        )
        csv_file.write(row)
        csv_file.flush()

    del x, y, y_init
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Vector SAXPY Benchmark")
    parser.add_argument("--sizes", type=str,
                        default="1048576,4194304,16777216,33554432,67108864,134217728",
                        help="Vector lengths N,N,... [default: 1M..134M]")
    parser.add_argument("--precisions", type=str, default="fp32,fp64",
                        help="Precisions: fp32,fp64 [default: fp32,fp64]")
    parser.add_argument("--threads", type=str, default=None,
                        help="Thread counts T,T,... [default: powers of 2 up to max cores]")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Alias for --threads or GPU IDs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Compute device: cpu or cuda [default: cpu]")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timed loop iterations [default: 20]")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed warmup runs [default: 3]")
    parser.add_argument("--validate", action="store_true",
                        help="Validate numerical accuracy")
    parser.add_argument("--validate-max", type=int, default=16777216,
                        help="Max vector length to validate [default: 16777216]")
    parser.add_argument("--csv", type=str, default=None,
                        help="Append results to CSV file")
    parser.add_argument("--tag", type=str, default="torch_saxpy",
                        help="Label tag in CSV [default: torch_saxpy]")

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]

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
                "tag,ts,host,mkl_version,mode,precision,threads,N,iters,warmup,"
                "agg_gbps,agg_tflops,comp_ms,wall_ms,"
                "cpu_mhz_avg,power_w_avg,temp_c_max,"
                "mem_gb,validated,max_rel_err,status\n"
            )
            csv_file.write(header)
            csv_file.flush()

    telem = HardwareTelemetry(device_type=device.type)

    print(f"# PyTorch SAXPY Benchmark ({engine_info}) host={host} device={device}")
    print(f"{'work':<6} {'prec':<5} {'T':<3} {'N':>10} {'GB/s':>8} {'TFLOPS':>7} {'comp_ms':>7} {'power':>7} {'MHz':>7} {'status'}")

    try:
        for N in sizes:
            for prec in precisions:
                if prec not in DTYPE_MAP:
                    continue
                for T in thread_counts:
                    run_saxpy(args, N, prec, T, device, telem, csv_file, host, engine_info)
    finally:
        telem.close()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
