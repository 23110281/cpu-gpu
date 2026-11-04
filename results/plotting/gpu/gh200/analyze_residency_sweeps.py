#!/usr/bin/env python3
"""
analyze_residency_sweeps.py

Synthesizes empirical benchmarking results across all 5 Cycles and 4 Coherent NUMA Modes:
  - Cycle 1: Streaming (SAXPY & GEMV, 16MB up to 206GB)
  - Cycle 2: Arithmetic Compute (Monte Carlo & GEMM FP32/FP64/BF16)
  - Cycle 3: Llama-3.2 Layer Suite (Embedding, RMSNorm, RoPE, QKV, SDPA, SwiGLU, LM Head)
  - Cycle 4: Irregular & Sparse (SpMV, SpGEMM, Thrust Binary Search)
  - Cycle 5: Divergence, Sorting & Cryptography (CUB Radix Sort, Histogram, AES-128-CTR)
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "results/data/gpu/NVIDIA_GH200_480GB/residency_sweeps"

def analyze():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "sweep_cycle*.csv")))
    print(f"Found {len(csv_files)} sweep CSVs in {DATA_DIR}:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}: {os.path.getsize(f)} bytes")
        
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal benchmark records collected: {len(all_df)}")
    print("Workloads covered:", all_df["workload"].unique())
    print("Residency modes:", all_df["residency_mode"].unique())
    
    # Save combined dataset
    combined_csv = os.path.join(DATA_DIR, "gh200_combined_taxonomy_benchmark.csv")
    all_df.to_csv(combined_csv, index=False)
    print(f"\nSaved master combined dataset to: {combined_csv}")

if __name__ == "__main__":
    analyze()
