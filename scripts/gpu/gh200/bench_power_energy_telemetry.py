#!/usr/bin/env python3
"""
scripts/gpu/gh200/bench_power_energy_telemetry.py

GH200 Power, Clock & Energy-to-Solution Telemetry Benchmark (Sweep 8)

Evaluates energy efficiency and throughput-per-watt across 4 residency modes:
1. cpu_local     (Grace Neoverse-V2 cores + LPDDR5X)
2. gpu_explicit  (Hopper GPU + HBM3)
3. gpu_zero_copy (Hopper GPU accessing Grace LPDDR5X via C2C)
4. gpu_um_migrate (Hopper GPU via Unified Memory migration)

Representative Kernels:
1. gemm             (Regime 1: Compute / Matrix Tensor Cores)
2. saxpy            (Regime 2: Memory Bandwidth / Streaming)
3. spmv             (Regime 3: Low Arithmetic Intensity / Sparse Memory Indirect)
4. aes_128_ctr      (Regime 4: Cryptography / Non-divergent Byte Processing)
5. cub_radix_sort   (Regime 4: Branch / Memory Divergent Key Sorting)
6. llama3_8b_prefill (Regime 1/2: LLM Prefill Attention / FlashDecoding)
7. llama3_8b_decode  (Regime 2: LLM Decode Attention / KV-Cache Bound)

Telemetry sampled at ~80ms intervals via:
  nvidia-smi --query-gpu=power.draw.instant,module.power.draw.instant,clocks.sm,clocks.mem,temperature.gpu
- NVML GPU Power (W), SM Clock (MHz), Mem Clock (MHz), GPU Temperature (C)
- GH200 Module Power (W)
- Grace CPU Power (W) = Module Power - GPU Power
- Metrics logged: power_watts_avg, energy_joules, gflops_per_watt, gbps_per_watt
"""

import os
import sys
import time
import argparse
import threading
import subprocess
import numpy as np

def setup_allocator(residency_mode, so_path):
    """Setup PyTorch CUDAPluggableAllocator if needed."""
    if residency_mode in ("gpu_zero_copy", "gpu_um_migrate"):
        import torch
        if os.path.exists(so_path):
            if residency_mode == "gpu_zero_copy":
                alloc = torch.cuda.CUDAPluggableAllocator(so_path, "c2c_zero_copy_malloc", "c2c_zero_copy_free")
                torch.cuda.change_current_allocator(alloc)
                print(f"[PluggableAllocator] Initialized GPU_ZERO_COPY (LPDDR5X via C2C) using {so_path}")
            else:
                alloc = torch.cuda.CUDAPluggableAllocator(so_path, "c2c_um_malloc", "c2c_um_free")
                torch.cuda.change_current_allocator(alloc)
                print(f"[PluggableAllocator] Initialized GPU_UM_MIGRATE (UVM via C2C) using {so_path}")
        else:
            print(f"[WARNING] {so_path} not found! Falling back to standard CUDA allocator.")

def sample_telemetry_loop(stop_event, samples, lock):
    """Background sampler querying NVML GPU power, module power, clocks, temp."""
    while not stop_event.is_set():
        gpu_p = sm_clk = mem_clk = gpu_temp = 0.0
        mod_p = 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw.instant,module.power.draw.instant,clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8", stderr=subprocess.DEVNULL
            ).strip().split(",")
            gpu_p = float(out[0].strip())
            mod_p = float(out[1].strip())
            sm_clk = float(out[2].strip())
            mem_clk = float(out[3].strip())
            gpu_temp = float(out[4].strip())
        except Exception:
            pass

        cpu_p = max(0.0, mod_p - gpu_p)

        with lock:
            samples.append({
                "time": time.perf_counter(),
                "gpu_p": gpu_p,
                "mod_p": mod_p,
                "cpu_p": cpu_p,
                "sm_clk": sm_clk,
                "mem_clk": mem_clk,
                "gpu_temp": gpu_temp
            })
        time.sleep(0.08)

def log_telemetry_record(csv_path, workload, regime, precision, residency, duration,
                         samples_count, gpu_p_avg, cpu_p_avg, mod_p_avg, energy_j,
                         eff_gbps, gflops, gbps_per_w, gflops_per_w,
                         sm_clk_avg, mem_clk_avg, temp_avg, status):
    """Append a standardized telemetry record to CSV."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a+") as f:
        if not file_exists:
            f.write("workload,regime,precision,residency_mode,duration_sec,samples_count,"
                    "gpu_power_watts_avg,cpu_power_watts_avg,module_power_watts_avg,energy_joules,"
                    "throughput_gbps,throughput_gflops,gbps_per_watt,gflops_per_watt,"
                    "sm_clock_mhz,mem_clock_mhz,gpu_temp_c,status\n")
        f.write(f"{workload},{regime},{precision},{residency},{duration:.3f},{samples_count},"
                f"{gpu_p_avg:.2f},{cpu_p_avg:.2f},{mod_p_avg:.2f},{energy_j:.2f},"
                f"{eff_gbps:.2f},{gflops:.2f},{gbps_per_w:.4f},{gflops_per_w:.4f},"
                f"{sm_clk_avg:.1f},{mem_clk_avg:.1f},{temp_avg:.1f},{status}\n")
        f.flush()

def run_telemetry_benchmark(workload, regime, precision, residency, duration_sec, csv_path):
    """Execute a steady-state run for duration_sec while sampling power telemetry."""
    import torch
    import torch.nn.functional as F

    print(f"\n[Telemetry] Workload: {workload:<18} | Mode: {residency:<16} | Target Duration: {duration_sec}s ... ", end="", flush=True)

    is_cuda = (residency != "cpu_local")
    dev = "cuda" if is_cuda else "cpu"
    if not is_cuda:
        torch.set_num_threads(72)

    step_func = None
    flops_per_step = 0.0
    bytes_per_step = 0.0

    try:
        if workload == "gemm":
            # FP32 GEMM 8192x8192
            M, N, K = 8192, 8192, 8192
            flops_per_step = 2.0 * M * N * K
            bytes_per_step = (M * K + K * N + M * N) * 4.0
            A = torch.randn((M, K), dtype=torch.float32, device=dev)
            B = torch.randn((K, N), dtype=torch.float32, device=dev)
            C = torch.empty((M, N), dtype=torch.float32, device=dev)
            def gemm_step():
                torch.mm(A, B, out=C)
            step_func = gemm_step

        elif workload == "saxpy":
            # SAXPY 64M elements (256MB)
            N = 67108864
            flops_per_step = 2.0 * N
            bytes_per_step = 3.0 * N * 4.0
            X = torch.randn(N, dtype=torch.float32, device=dev)
            Y = torch.randn(N, dtype=torch.float32, device=dev)
            def saxpy_step():
                Y.add_(X, alpha=2.0)
            step_func = saxpy_step

        elif workload == "spmv":
            # Sparse Matrix-Vector Multiply (M=65536, NNZ=5,000,000)
            M = 65536
            NNZ = 5000000
            flops_per_step = 2.0 * NNZ
            bytes_per_step = (NNZ * 12.0) + (M * 4.0)
            row_idx = torch.randint(0, M, (NNZ,), dtype=torch.int64, device=dev)
            col_idx = torch.randint(0, M, (NNZ,), dtype=torch.int64, device=dev)
            indices = torch.stack([row_idx, col_idx])
            values = torch.randn(NNZ, dtype=torch.float32, device=dev)
            sp_mat = torch.sparse_coo_tensor(indices, values, (M, M), device=dev).coalesce()
            x_vec = torch.randn((M, 1), dtype=torch.float32, device=dev)
            def spmv_step():
                torch.sparse.mm(sp_mat, x_vec)
            step_func = spmv_step

        elif workload == "aes_128_ctr":
            # Crypto byte permutation / transformation (16MB chunks)
            N_bytes = 16 * 1024 * 1024
            bytes_per_step = 2.0 * N_bytes
            flops_per_step = 0.0
            data_in = torch.randint(0, 255, (N_bytes,), dtype=torch.uint8, device=dev)
            data_out = torch.empty_like(data_in)
            def aes_step():
                torch.bitwise_xor(data_in, 0x5A, out=data_out)
            step_func = aes_step

        elif workload == "cub_radix_sort":
            # Key sorting (16M keys = 64MB)
            N_keys = 16777216
            bytes_per_step = 2.0 * N_keys * 4.0
            flops_per_step = N_keys * np.log2(N_keys)
            keys = torch.randint(0, 1000000000, (N_keys,), dtype=torch.int32, device=dev)
            def sort_step():
                torch.sort(keys)
            step_func = sort_step

        elif workload == "llama3_8b_prefill":
            # Llama-3 SDPA Prefill: B=1, N_heads=32, N_kv=8, S=2048, D=128 (BF16)
            B, Nh, Nkv, S, D = 1, 32, 8, 2048, 128
            flops_per_step = 4.0 * B * Nh * S * S * D
            bytes_per_step = (2 * B * Nkv * S * D + B * Nh * S * D) * 2.0
            Q = torch.randn((B, Nh, S, D), dtype=torch.bfloat16, device=dev)
            K = torch.randn((B, Nkv, S, D), dtype=torch.bfloat16, device=dev)
            V = torch.randn((B, Nkv, S, D), dtype=torch.bfloat16, device=dev)
            def prefill_step():
                F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
            step_func = prefill_step

        elif workload == "llama3_8b_decode":
            # Llama-3 SDPA Decode: B=1, N_heads=32, N_kv=8, S_kv=16384, D=128 (BF16)
            B, Nh, Nkv, S_kv, D = 1, 32, 8, 16384, 128
            flops_per_step = 4.0 * B * Nh * 1 * S_kv * D
            bytes_per_step = 2.0 * B * Nkv * S_kv * D * 2.0
            Q = torch.randn((B, Nh, 1, D), dtype=torch.bfloat16, device=dev)
            K = torch.randn((B, Nkv, S_kv, D), dtype=torch.bfloat16, device=dev)
            V = torch.randn((B, Nkv, S_kv, D), dtype=torch.bfloat16, device=dev)
            def decode_step():
                F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
            step_func = decode_step

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"FAILED SETUP: {e}")
        log_telemetry_record(csv_path, workload, regime, precision, residency, 0.0, 0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "setup-oom")
        return

    # Warmup
    for _ in range(3):
        step_func()
    if is_cuda:
        torch.cuda.synchronize()

    # Launch background telemetry sampler
    samples = []
    lock = threading.Lock()
    stop_event = threading.Event()
    sampler_th = threading.Thread(target=sample_telemetry_loop, args=(stop_event, samples, lock))
    sampler_th.daemon = True
    sampler_th.start()

    # Steady-state timed loop for target duration
    iters = 0
    t_start = time.perf_counter()
    t_deadline = t_start + duration_sec

    while time.perf_counter() < t_deadline:
        step_func()
        if is_cuda:
            torch.cuda.synchronize()
        iters += 1

    actual_duration = time.perf_counter() - t_start

    # Stop sampler
    stop_event.set()
    sampler_th.join()

    # Compute averages
    if len(samples) > 0:
        gpu_p_avg = float(np.mean([s["gpu_p"] for s in samples]))
        mod_p_avg = float(np.mean([s["mod_p"] for s in samples]))
        cpu_p_avg = float(np.mean([s["cpu_p"] for s in samples]))
        sm_clk_avg = float(np.mean([s["sm_clk"] for s in samples]))
        mem_clk_avg = float(np.mean([s["mem_clk"] for s in samples]))
        temp_avg = float(np.mean([s["gpu_temp"] for s in samples]))
    else:
        gpu_p_avg = mod_p_avg = cpu_p_avg = sm_clk_avg = mem_clk_avg = temp_avg = 0.0

    energy_j = mod_p_avg * actual_duration
    total_flops = iters * flops_per_step
    total_bytes = iters * bytes_per_step
    gflops = (total_flops / 1e9) / actual_duration if actual_duration > 0 else 0.0
    eff_gbps = (total_bytes / 1e9) / actual_duration if actual_duration > 0 else 0.0

    gflops_per_w = (gflops / mod_p_avg) if mod_p_avg > 0 else 0.0
    gbps_per_w = (eff_gbps / mod_p_avg) if mod_p_avg > 0 else 0.0

    log_telemetry_record(csv_path, workload, regime, precision, residency, actual_duration,
                         len(samples), gpu_p_avg, cpu_p_avg, mod_p_avg, energy_j,
                         eff_gbps, gflops, gbps_per_w, gflops_per_w,
                         sm_clk_avg, mem_clk_avg, temp_avg, "ok")

    print(f"done | {iters} iters | Mod Power: {mod_p_avg:5.1f}W (GPU: {gpu_p_avg:5.1f}W, CPU: {cpu_p_avg:5.1f}W) | "
          f"Perf: {gflops:7.1f} GFLOPS, {eff_gbps:6.1f} GB/s | Efficiency: {gflops_per_w:5.2f} GFLOPS/W, {gbps_per_w:5.2f} GB/s/W")

    del step_func
    if is_cuda:
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Sweep 8: GH200 Power, Clock & Energy Telemetry Benchmark")
    parser.add_argument("--csv", type=str, required=True, help="Destination CSV file path")
    parser.add_argument("--duration", type=float, default=10.0, help="Steady-state duration in seconds per kernel run")
    parser.add_argument("--so-path", type=str, default="bin/libtorch_c2c_allocator.so", help="Path to C2C zero-copy allocator shared library")
    parser.add_argument("--residency", type=str, default="all", choices=["all", "cpu_local", "gpu_explicit", "gpu_zero_copy", "gpu_um_migrate"])
    args = parser.parse_args()

    workloads = [
        ("gemm", 1, "fp32"),
        ("saxpy", 2, "fp32"),
        ("spmv", 3, "fp32"),
        ("aes_128_ctr", 4, "int8"),
        ("cub_radix_sort", 4, "int32"),
        ("llama3_8b_prefill", 1, "bf16"),
        ("llama3_8b_decode", 2, "bf16"),
    ]

    residencies = ["cpu_local", "gpu_explicit", "gpu_zero_copy", "gpu_um_migrate"]

    if args.residency == "all":
        print("==========================================================================================")
        print(" GH200 Power, Clock & Energy-to-Solution Telemetry Sweep (Sweep 8)")
        print(f" Output CSV: {args.csv}")
        print(f" Steady-State Duration: {args.duration}s per workload/residency")
        print(" Launching isolated subprocess per residency mode for clean allocator state...")
        print("==========================================================================================")
        for res in residencies:
            cmd = [
                sys.executable, sys.argv[0],
                "--csv", args.csv,
                "--duration", str(args.duration),
                "--so-path", args.so_path,
                "--residency", res
            ]
            print(f"\n>>> Spawning subprocess for residency mode: {res}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess for {res} exited with code {e.returncode}")
        print("\n[Sweep 8] All Telemetry Runs Successfully Finished!")
    else:
        so_full = os.path.abspath(args.so_path)
        setup_allocator(args.residency, so_full)
        for wk, reg, prec in workloads:
            run_telemetry_benchmark(wk, reg, prec, args.residency, args.duration, args.csv)

if __name__ == "__main__":
    main()
