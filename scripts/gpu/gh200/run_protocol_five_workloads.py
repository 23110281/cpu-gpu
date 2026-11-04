#!/usr/bin/env python3
"""
run_protocol_five_workloads.py

Automates the GH200 Mandatory Protocol execution for the 5 key workloads:
  1. Grace CPU STREAM (72 Cores, Triad & Copy)
  2. 65k-Bin Histogram (Uniform & Skewed distributions)
  3. 64-bit Binary Search (Standard vs Eytzinger BFS)
  4. Zero-Copy GEMM (Pure Zero-Copy vs HBM-Staged End-to-End)
  5. Llama-3.2 1B Decode Attention (FlashDecoding Split-KV)

Strictly enforces:
  - 5 independent process launches per workload
  - 30 timed iterations per launch
  - 10 Hz NVML clock and power telemetry
  - Hardware state snapshot (clocks, power, ECC, CPU governor, NUMA, THP)
  - Raw log logging under results/raw/protocol_rerun_<ts>/
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"results/raw/protocol_rerun_{TIMESTAMP}"
os.makedirs(RUN_DIR, exist_ok=True)

print(f"=== GH200 Protocol Benchmark Suite Runner ===")
print(f"Target Directory: {RUN_DIR}")

# 1. Capture Hardware State Snapshot (Protocol Section 9)
print("[1/4] Capturing Hardware Environment Snapshot...")
hw_log = os.path.join(RUN_DIR, "hardware_environment.log")
with open(hw_log, "w") as f:
    f.write("=== GH200 HARDWARE ENVIRONMENT SNAPSHOT ===\n")
    f.write(f"Timestamp: {TIMESTAMP}\n\n")
    
    # GPU state
    f.write("--- NVIDIA-SMI SUMMARY ---\n")
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)
    f.write("\n--- GPU CLOCKS AND POWER SPECS ---\n")
    subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE,ECC"], stdout=f, stderr=subprocess.STDOUT)
    
    # CPU governor
    f.write("\n--- CPU GOVERNOR ---\n")
    subprocess.run(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], stdout=f, stderr=subprocess.STDOUT)
    
    # NUMA state
    f.write("\n--- CPU AND NUMA TOPOLOGY (LSCPU) ---\n")
    subprocess.run(["lscpu"], stdout=f, stderr=subprocess.STDOUT)
    f.write("\n--- NUMA NODES IN SYSFS ---\n")
    subprocess.run(["ls -ld /sys/devices/system/node/node*"], shell=True, stdout=f, stderr=subprocess.STDOUT)
    if subprocess.run(["which", "numactl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        subprocess.run(["numactl", "--hardware"], stdout=f, stderr=subprocess.STDOUT)
    
    # THP
    f.write("\n--- TRANSPARENT HUGE PAGES ---\n")
    subprocess.run(["cat", "/sys/kernel/mm/transparent_hugepage/enabled"], stdout=f, stderr=subprocess.STDOUT)

# 2. Start 10 Hz NVML Telemetry Logger (Protocol Section 9)
print("[2/4] Starting 10 Hz NVML Telemetry Background Logger...")
telemetry_csv = os.path.join(RUN_DIR, "telemetry_nvml_10hz.csv")
telemetry_proc = subprocess.Popen([
    "nvidia-smi",
    "--query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
    "--format=csv,nounits",
    "-lms", "100"
], stdout=open(telemetry_csv, "w"), stderr=subprocess.DEVNULL)

workloads = [
    ("cpu_stream", "workload1_grace_stream", "OMP_NUM_THREADS=72 OMP_PROC_BIND=spread numactl --cpunodebind=0 --membind=0 ./bin/protocol_benchmark_suite cpu_stream 30 5"),
    ("histogram", "workload2_histogram", "./bin/protocol_benchmark_suite histogram 30 5"),
    ("binary_search", "workload3_binary_search", "./bin/protocol_benchmark_suite binary_search 30 5"),
    ("zerocopy_gemm", "workload4_zerocopy_gemm", "./bin/protocol_benchmark_suite zerocopy_gemm 30 5"),
    ("llama_decode", "workload5_llama_decode", "./bin/protocol_benchmark_suite llama_decode 30 5"),
]

# 3. Execute 5 Independent Process Launches per Workload (Protocol Section 8)
print("[3/5] Running 5 Independent Process Launches for Each Workload...")
summary_log = os.path.join(RUN_DIR, "protocol_execution_summary.log")

with open(summary_log, "w") as s_file:
    s_file.write(f"Protocol Run ID: protocol_rerun_{TIMESTAMP}\n\n")
    for name, prefix, cmd in workloads:
        print(f"\n--> Workload: {name}")
        s_file.write(f"================================================================================\n")
        s_file.write(f"Workload: {name}\n")
        s_file.write(f"Command:  {cmd}\n")
        s_file.write(f"================================================================================\n")
        
        for launch_idx in range(1, 6):
            log_path = os.path.join(RUN_DIR, f"{prefix}_launch{launch_idx}.log")
            print(f"    Launch {launch_idx}/5 -> {log_path}")
            
            t0 = time.time()
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - t0
            
            with open(log_path, "w") as lf:
                lf.write(f"Run: {name} (Launch {launch_idx}/5)\n")
                lf.write(f"Command: {cmd}\n")
                lf.write(f"Duration: {elapsed:.2f} s\n")
                lf.write(f"Exit Code: {res.returncode}\n\n")
                lf.write(res.stdout)
                
            s_file.write(f"\n[Launch {launch_idx}/5] (Duration: {elapsed:.2f} s, Exit Code: {res.returncode})\n")
            s_file.write(res.stdout)
            s_file.flush()
            time.sleep(0.5)

# 4. Hardware Counter Profiling via Nsight Compute (Protocol Section 10)
print("\n[4/5] Profiling Hardware Counters via Nsight Compute (ncu)...")
ncu_bin = "/usr/local/cuda-12.8/bin/ncu"
ncu_targets = [
    ("histogram", f"sudo {ncu_bin} --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum ./bin/protocol_benchmark_suite histogram 1 1"),
    ("binary_search", f"sudo {ncu_bin} --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum ./bin/protocol_benchmark_suite binary_search_c2c 1 1"),
    ("zerocopy_gemm", f"sudo {ncu_bin} --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum /tmp/test_zc_gemm_16k"),
    ("llama_decode", f"sudo {ncu_bin} --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum ./bin/protocol_benchmark_suite llama_decode 1 1"),
]
for target_name, ncu_cmd in ncu_targets:
    ncu_out_path = os.path.join(RUN_DIR, f"ncu_profile_{target_name}.log")
    print(f"    Profiling {target_name} -> {ncu_out_path}")
    res = subprocess.run(ncu_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(ncu_out_path, "w") as f:
        f.write(f"Command: {ncu_cmd}\n\n")
        f.write(res.stdout)

# 5. Stop Telemetry
print("\n[5/5] Finalizing Telemetry and Logs...")
telemetry_proc.terminate()
telemetry_proc.wait()

print(f"\nProtocol execution completed successfully!")
print(f"All raw logs saved to: {RUN_DIR}")
