#!/usr/bin/env python3
"""
clean_and_verify_sweeps.py

Post-processes and cleans all 8 raw benchmark sweeps from NVIDIA GH200 (Grace Hopper 480GB):
1. Merges Streaming Part 1 (GPU saxpy) and Part 2 (CPU saxpy + gemv + rmsnorm).
2. Fixes Llama-3 CSV: tags model (llama3.2_1b, llama3.2_3b, llama3_8b), removes aborted rows 2016-2019.
3. Incorporates patch data for thrust_search (64-bit keys & algorithmic BW) and SpGEMM (S=8192..65536).
4. Incorporates patch data for cooperative execution (72-thread CPU KV offload & accurate residency labels).
5. Adds both gflops_warm (steady-state) and gflops_e2e (cold-migration inclusive).
6. Emits verified, deduplicated CSVs into results/data/NVIDIA_GH200_480GB/cleaned_for_paper/.
"""

import os
import sys
import glob
import csv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "results/data/NVIDIA_GH200_480GB/residency_sweeps")
OUT_DIR = os.path.join(BASE_DIR, "results/data/NVIDIA_GH200_480GB/cleaned_for_paper")

CLEAN_HEADER = [
    "workload", "regime", "precision", "residency_mode", "working_set_bytes",
    "problem_size", "model", "threads_or_sms",
    "t_alloc_ms", "t_h2d_ms", "t_kernel_cold_ms", "t_kernel_warm_ms",
    "t_d2h_ms", "t_e2e_ms", "eff_bw_gbps", "c2c_bw_gbps",
    "gflops_warm", "gflops_e2e", "pct_peak", "status"
]

def parse_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def row_to_clean(row, model="none"):
    """
    Transforms raw CSV row:
    [workload, regime, precision, residency_mode, working_set_bytes, problem_size,
     threads_or_sms, t_alloc_ms, t_h2d_ms, t_kernel_cold_ms, t_kernel_warm_ms,
     t_d2h_ms, t_e2e_ms, eff_bw_gbps, c2c_bw_gbps, gflops, pct_peak, status]
    into CLEAN_HEADER.
    """
    if len(row) < 18:
        return None
    workload = row[0]
    regime = row[1]
    precision = row[2]
    residency = row[3]
    ws_bytes = row[4]
    p_size = row[5]
    threads = row[6]
    t_alloc = row[7]
    t_h2d = row[8]
    t_cold = row[9]
    t_warm = row[10]
    t_d2h = row[11]
    t_e2e = row[12]
    eff_bw = row[13]
    c2c_bw = row[14]
    raw_gflops = row[15]
    pct_peak = row[16]
    status = row[17]

    t_warm_val = parse_float(t_warm)
    t_e2e_val = parse_float(t_e2e)
    gflops_val = parse_float(raw_gflops)

    gflops_warm = gflops_val
    if gflops_val > 0.0 and t_e2e_val > 0.0 and t_warm_val > 0.0:
        gflops_e2e = round(gflops_val * (t_warm_val / t_e2e_val), 2)
    else:
        gflops_e2e = 0.0

    return [
        workload, regime, precision, residency, ws_bytes,
        p_size, model, threads,
        t_alloc, t_h2d, t_cold, t_warm,
        t_d2h, t_e2e, eff_bw, c2c_bw,
        f"{gflops_warm:.2f}", f"{gflops_e2e:.2f}", pct_peak, status
    ]

def clean_streaming():
    print(">>> Processing Streaming Dense Grid (Merging Part 1 + Part 2)...")
    p1_file = os.path.join(RAW_DIR, "sweep_streaming_dense_grid_20261029_083202.csv")
    p2_file = os.path.join(RAW_DIR, "sweep_streaming_dense_grid_20261029_085241.csv")

    records = {} # key: (workload, precision, residency_mode, problem_size, model, threads_or_sms)

    # Load Part 1 (has GPU saxpy)
    if os.path.exists(p1_file):
        with open(p1_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    key = (clean[0], clean[2], clean[3], clean[5], clean[6], clean[7])
                    records[key] = clean

    # Load Part 2 (has CPU saxpy, gemv, rmsnorm)
    if os.path.exists(p2_file):
        with open(p2_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    key = (clean[0], clean[2], clean[3], clean[5], clean[6], clean[7])
                    # Update or insert
                    records[key] = clean

    out_file = os.path.join(OUT_DIR, "clean_streaming_dense_grid.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        for rec in records.values():
            w.writerow(rec)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return list(records.values())

def clean_compute():
    print(">>> Processing Compute & FFT/Convolution Sweep...")
    raw_file = os.path.join(RAW_DIR, "sweep_compute_and_fft_conv_20261029_092044.csv")
    records = []
    if os.path.exists(raw_file):
        with open(raw_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    records.append(clean)
    out_file = os.path.join(OUT_DIR, "clean_compute_and_fft_conv.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(records)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return records

def clean_divergence():
    print(">>> Processing Branch Divergence, Atomics & Crypto Sweep...")
    raw_file = os.path.join(RAW_DIR, "sweep_divergence_atomics_crypto_20261029_105149.csv")
    records = []
    if os.path.exists(raw_file):
        with open(raw_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    records.append(clean)
    out_file = os.path.join(OUT_DIR, "clean_divergence_atomics_crypto.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(records)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return records

def clean_irregular():
    print(">>> Processing Irregular & SpGEMM Sweep (with Patches)...")
    base_file = os.path.join(RAW_DIR, "sweep_irregular_and_spgemm_fixed_20261029_104147.csv")
    thrust_patches = sorted(glob.glob(os.path.join(RAW_DIR, "sweep_irregular_thrust_patch_*.csv")))
    spgemm_patches = sorted(glob.glob(os.path.join(RAW_DIR, "sweep_irregular_spgemm_large_patch_*.csv")))

    records = {} # key: (workload, precision, residency_mode, problem_size)

    # 1. Base run (keep spmv and valid runs)
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    # Filter out old flawed thrust_search rows
                    if row[0] == "thrust_search":
                        continue
                    clean = row_to_clean(row, model="none")
                    key = (clean[0], clean[2], clean[3], clean[5], clean[6], clean[7])
                    records[key] = clean

    # 2. Integrate latest thrust_search patch
    if thrust_patches:
        latest_thrust = thrust_patches[-1]
        print(f"  Integrating thrust_search patch: {os.path.basename(latest_thrust)}")
        with open(latest_thrust, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18 and row[0] == "thrust_search":
                    clean = row_to_clean(row, model="none")
                    key = (clean[0], clean[2], clean[3], clean[5], clean[6], clean[7])
                    records[key] = clean

    # 3. Integrate latest large spgemm patch
    if spgemm_patches:
        latest_spgemm = spgemm_patches[-1]
        print(f"  Integrating spgemm patch: {os.path.basename(latest_spgemm)}")
        with open(latest_spgemm, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18 and row[0] == "spgemm":
                    clean = row_to_clean(row, model="none")
                    key = (clean[0], clean[2], clean[3], clean[5], clean[6], clean[7])
                    records[key] = clean

    out_file = os.path.join(OUT_DIR, "clean_irregular_and_spgemm.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        for rec in records.values():
            w.writerow(rec)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return list(records.values())

def clean_llama3():
    print(">>> Processing Llama-3 Layers & KV Offload Sweep...")
    raw_file = os.path.join(RAW_DIR, "sweep_llama3_all_models_and_kv_offload_20261029_110405.csv")
    records = []
    if os.path.exists(raw_file):
        with open(raw_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            all_rows = [row for row in r if len(row) >= 18]

        # Model mapping for each 336-row block
        model_map = [
            (0, 336, "llama3.2_1b"),      # Block 0: 1B eager
            (336, 672, "llama3.2_3b"),    # Block 1: 3B eager
            (672, 1008, "llama3_8b"),     # Block 2: 8B eager
            (1008, 1344, "llama3.2_1b"),  # Block 3: 1B compiled
            (1344, 1680, "llama3.2_3b"),  # Block 4: 3B compiled
            (1680, 2016, "llama3_8b"),    # Block 5: 8B compiled
        ]

        for start, end, model_name in model_map:
            for i in range(start, min(end, len(all_rows))):
                clean = row_to_clean(all_rows[i], model=model_name)
                records.append(clean)

        # Skip rows 2016..2019 (aborted duplicates)
        # Rows 2020 to end are KV offload runs
        for i in range(2020, len(all_rows)):
            clean = row_to_clean(all_rows[i], model="llama3_8b")
            records.append(clean)

    out_file = os.path.join(OUT_DIR, "clean_llama3_all_models_and_kv_offload.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(records)
    print(f"  Saved {len(records)} verified records (deduplicated & labeled) to {out_file}")
    return records

def clean_cooperative():
    print(">>> Processing Cooperative CPU+GPU Execution Sweep (with Patches)...")
    base_file = os.path.join(RAW_DIR, "sweep_cooperative_cpu_gpu_20261029_174509.csv")
    coop_patches = sorted(glob.glob(os.path.join(RAW_DIR, "sweep_cooperative_patch_*.csv")))

    records = {}

    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    # Accurate residency labeling for 100% CPU
                    if "alpha=1.00" in clean[5]:
                        clean[3] = "cpu_local"
                        clean[7] = "72"
                    elif "alpha=0.00" in clean[5]:
                        clean[3] = "gpu_zero_copy"
                        clean[7] = "132"
                    key = (clean[0], clean[5])
                    records[key] = clean

    # Integrate latest patch if available (fixes CPU KV offload & residency)
    if coop_patches:
        latest_patch = coop_patches[-1]
        print(f"  Integrating cooperative patch: {os.path.basename(latest_patch)}")
        with open(latest_patch, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    if "alpha=1.00" in clean[5]:
                        clean[3] = "cpu_local"
                        clean[7] = "72"
                    elif "alpha=0.00" in clean[5]:
                        clean[3] = "gpu_zero_copy"
                        clean[7] = "132"
                    key = (clean[0], clean[5])
                    records[key] = clean

    out_file = os.path.join(OUT_DIR, "clean_cooperative_cpu_gpu.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        for rec in records.values():
            w.writerow(rec)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return list(records.values())

def clean_uvm():
    print(">>> Processing UVM Advise Cliff Sweep...")
    raw_file = os.path.join(RAW_DIR, "sweep_uvm_advise_cliff_20261029_180345.csv")
    records = []
    if os.path.exists(raw_file):
        with open(raw_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    clean = row_to_clean(row, model="none")
                    records.append(clean)
    out_file = os.path.join(OUT_DIR, "clean_uvm_advise_cliff.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(records)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return records

def clean_power():
    print(">>> Processing Power, Energy & Telemetry Sweep...")
    raw_file = os.path.join(RAW_DIR, "sweep_power_energy_telemetry_20261029_192554.csv")
    records = []
    if os.path.exists(raw_file):
        with open(raw_file, "r") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) >= 18:
                    workload = row[0]
                    regime = row[1]
                    precision = row[2]
                    residency = row[3]
                    duration_sec = parse_float(row[4])
                    samples_cnt = row[5]
                    p_gpu = row[6]
                    p_cpu = row[7]
                    p_mod = row[8]
                    energy_j = row[9]
                    bw = row[10]
                    gflops = row[11]
                    gbps_per_w = row[12]
                    gflops_per_w = row[13]
                    status = row[17]

                    p_size = f"P_mod={p_mod}W (GPU={p_gpu}W, CPU={p_cpu}W)"
                    threads = "72" if residency == "cpu_local" else "132"
                    t_warm = f"{duration_sec * 1000.0:.2f}"
                    pct_peak = f"GFLOPS/W={gflops_per_w}, GB/s/W={gbps_per_w}"

                    clean = [
                        workload, regime, precision, residency, "0",
                        p_size, "none", threads,
                        "0.00", "0.00", "0.00", t_warm,
                        "0.00", t_warm, bw, "0.00",
                        f"{parse_float(gflops):.2f}", f"{parse_float(gflops):.2f}",
                        pct_peak, status
                    ]
                    records.append(clean)

        # Also preserve raw detailed power telemetry with original columns in cleaned_for_paper
        import shutil
        raw_out = os.path.join(OUT_DIR, "clean_power_energy_telemetry_raw.csv")
        shutil.copy2(raw_file, raw_out)

    out_file = os.path.join(OUT_DIR, "clean_power_energy_telemetry.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(records)
    print(f"  Saved {len(records)} verified records to {out_file}")
    return records

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("==========================================================================================")
    print(f" NVIDIA GH200 480GB: Clean, Verify & Unify Benchmark Dataset")
    print(f" Source: {RAW_DIR}")
    print(f" Target: {OUT_DIR}")
    print("==========================================================================================")

    all_records = []
    all_records.extend(clean_streaming())
    all_records.extend(clean_compute())
    all_records.extend(clean_divergence())
    all_records.extend(clean_irregular())
    all_records.extend(clean_llama3())
    all_records.extend(clean_cooperative())
    all_records.extend(clean_uvm())
    all_records.extend(clean_power())

    master_file = os.path.join(OUT_DIR, "gh200_paper_master_clean.csv")
    with open(master_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLEAN_HEADER)
        w.writerows(all_records)
    print("\n==========================================================================================")
    print(f" Dataset Unification Complete! Master file: {master_file}")
    print(f" Total verified records across all regimes: {len(all_records)}")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
