#!/usr/bin/env python3
"""
results/plotting/gpu/gh200/generate_protocol_and_sweep_plots.py

Generates publication-quality figures (PNG at 300 DPI + PDF vector format)
for the newly collected NVIDIA GH200 480GB Superchip sweeps:
  - Figure 5: GH200 Protocol Verification & Microarchitectural Characterization (6 Panels)
  - Figure 6: Real-Time 10 Hz NVML Power Draw & SM Clock Telemetry Profiles
  - Figure 7: Energy Efficiency & Cooperative CPU+GPU Workload Partitioning (2 Panels)
"""

import os
import sys
import csv
import json
import re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Set publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 15,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
    "lines.linewidth": 2.2,
    "lines.markersize": 7.0,
    "figure.autolayout": False
})

# Harmonious publication palette
PALETTE = {
    "cpu_local": "#1f77b4",          # Steel Blue
    "gpu_explicit": "#2ca02c",       # Forest Green
    "gpu_zero_copy": "#d62728",      # Crimson
    "gpu_um_migrate": "#ff7f0e",     # Amber / Orange
    "cluster_dsm": "#9467bd",        # Purple
    "cub_baseline": "#8c564b",       # Slate / Brown
    "hugepages": "#17becf",          # Cyan / Teal
    "ceiling": "#4a5568",            # Slate Gray
    "accent_yellow": "#bcbd22",      # Gold
    "accent_pink": "#e377c2"         # Pink
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CLEAN_DIR = os.path.join(BASE_DIR, "results/data/gpu/NVIDIA_GH200_480GB/cleaned_for_paper")
RAW_RUN_DIR = os.path.join(BASE_DIR, "results/raw/protocol_rerun_20261030_224002")
OUT_DIR = os.path.join(BASE_DIR, "results/plots/NVIDIA_GH200_480GB")
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def parse_val(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ==============================================================================
# Figure 5: GH200 Mandatory Protocol Benchmark Suite Characterization (6 Panels)
# ==============================================================================
def plot_figure_5_protocol_characterization():
    print(">>> Generating Figure 5: GH200 Protocol Verification Characterization (6 Panels)...")
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel (a): Grace CPU STREAM Thread Scaling (1 to 72 Cores)
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    streaming_csv = os.path.join(CLEAN_DIR, "clean_streaming_dense_grid.csv")
    s_rows = load_csv(streaming_csv)
    
    threads = [1, 4, 8, 16, 36, 54, 72]
    saxpy_bw = []
    gemv_bw = []
    
    for t in threads:
        # SAXPY CPU local
        sub_s = [r for r in s_rows if r["workload"] == "saxpy" and r["residency_mode"] == "cpu_local" 
                 and r["precision"] == "fp32" and r["threads_or_sms"] == str(t)]
        if sub_s:
            max_bw = max(parse_val(r["eff_bw_gbps"]) for r in sub_s)
            saxpy_bw.append(max_bw)
        else:
            saxpy_bw.append(0.0)
            
        # GEMV CPU local
        sub_g = [r for r in s_rows if r["workload"] == "gemv" and r["residency_mode"] == "cpu_local" 
                 and r["precision"] == "fp32" and r["threads_or_sms"] == str(t)]
        if sub_g:
            max_bw = max(parse_val(r["eff_bw_gbps"]) for r in sub_g)
            gemv_bw.append(max_bw)
        else:
            gemv_bw.append(0.0)

    # Also STREAM Triad / Copy at 72c
    ax_a.plot(threads, saxpy_bw, marker="o", color=PALETTE["cpu_local"], label="Grace SAXPY FP32 (LPDDR5X)")
    ax_a.plot(threads, gemv_bw, marker="s", color=PALETTE["gpu_um_migrate"], label="Grace GEMV FP32 (LPDDR5X)")
    
    # 72c STREAM points
    ax_a.scatter([72], [307.4], color="#b2182b", marker="*", s=160, zorder=5, label="STREAM Triad 72c (307.4 GB/s)")
    ax_a.scatter([72], [317.7], color="#17becf", marker="D", s=110, zorder=5, label="STREAM Copy 72c (317.7 GB/s)")

    ax_a.axhline(y=384.0, color="#b2182b", linestyle="--", linewidth=1.8, label="LPDDR5X Peak (384.0 GB/s)")
    ax_a.axhline(y=328.0, color="#7f7f7f", linestyle=":", linewidth=1.6, label="NVIDIA 328G Spec Target")
    
    ax_a.set_xlabel("Grace Neoverse-V2 Core Count")
    ax_a.set_ylabel("Sustained Bandwidth (GB/s)")
    ax_a.set_title("(a) Grace CPU Core Scaling: Hits 82.7% DRAM Peak", fontweight="bold", pad=8)
    ax_a.set_xticks(threads)
    ax_a.set_ylim(0, 420)
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    ax_a.annotate("82.7% of Theoretical Peak\n(Triad: 307.4, Copy: 317.7 GB/s)", xy=(72, 317.7), xytext=(32, 230),
                  arrowprops=dict(facecolor="#b2182b", shrink=0.08, width=1.4, headwidth=5),
                  fontsize=9.5, fontweight="bold", color="#b2182b")

    # --------------------------------------------------------------------------
    # Panel (b): 65k-Bin Histogram Architecture (Cluster DSM vs CUB vs Global)
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    dists = ["Uniform Distribution", "Skewed (Clustered Zipf)"]
    x_pos = np.arange(len(dists))
    width = 0.25

    bw_global = [365.5, 119.6]
    bw_cub = [98.2, 99.5]
    bw_dsm = [575.0, 561.9]

    b1 = ax_b.bar(x_pos - width, bw_global, width, label="Standard Global Atomics", color=PALETTE["ceiling"], alpha=0.85)
    b2 = ax_b.bar(x_pos, bw_cub, width, label="NVIDIA CUB DeviceHistogram", color=PALETTE["cub_baseline"], alpha=0.9)
    b3 = ax_b.bar(x_pos + width, bw_dsm, width, label="Hopper Cluster DSM (8-SM)", color=PALETTE["cluster_dsm"], alpha=0.95)

    ax_b.set_ylabel("Effective DRAM Bandwidth (GB/s)")
    ax_b.set_title("(b) 65k-Bin Histogram: Cluster DSM Achieves 5.85x Speedup", fontweight="bold", pad=8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(dists)
    ax_b.set_ylim(0, 820)
    ax_b.legend(loc="upper center", ncol=1, frameon=True, framealpha=0.9, fontsize=8.8)

    # Add speedup annotations cleanly
    ax_b.text(x_pos[0] + width, 595, "5.85x vs CUB\n(1.57x vs Global)", ha="center", fontsize=8.8, fontweight="bold", color=PALETTE["cluster_dsm"])
    ax_b.text(x_pos[1] + width, 580, "5.64x vs CUB\n(4.70x vs Global)", ha="center", fontsize=8.8, fontweight="bold", color=PALETTE["cluster_dsm"])

    for rects in [b1, b2, b3]:
        for r in rects:
            h = r.get_height()
            ax_b.text(r.get_x() + r.get_width()/2., h + 10, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel (c): 64-bit Binary Search: The 144x Translation Cliff & Hugepages
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    categories = [
        "HBM3 Local\n(268 MB)",
        "C2C Zero-Copy\n(268 MB L2-fit)",
        "C2C Zero-Copy\n(34.3 GB, 64KB)",
        "C2C Hugepages\n(34.3 GB, 512MB)"
    ]
    mops = [6714.5, 222.3, 1.47, 211.0]
    colors = [PALETTE["gpu_explicit"], PALETTE["cpu_local"], PALETTE["gpu_zero_copy"], PALETTE["hugepages"]]

    x_c = np.arange(len(categories))
    b_c = ax_c.bar(x_c, mops, width=0.55, color=colors, alpha=0.9, log=True)

    ax_c.set_ylabel("Query Throughput (M-Ops/s, log scale)")
    ax_c.set_title("(c) 64-bit Binary Search: 144.7x SMMU Translation Cliff", fontweight="bold", pad=8)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(categories, fontsize=9.0)
    ax_c.set_ylim(0.2, 25000)

    # Speedup arrows
    ax_c.annotate("144.7x Speedup\nvia 512MB Hugepages!", xy=(3, 211.0), xytext=(1.8, 30.0),
                  arrowprops=dict(facecolor=PALETTE["hugepages"], shrink=0.08, width=1.4, headwidth=5),
                  fontsize=9.5, fontweight="bold", color="#0e7490")
    ax_c.annotate("151x TLB Eviction Collapse\n(524,288 pages @ 64KB)", xy=(2, 1.47), xytext=(0.8, 0.4),
                  arrowprops=dict(facecolor=PALETTE["gpu_zero_copy"], shrink=0.08, width=1.4, headwidth=5),
                  fontsize=8.5, fontweight="bold", color=PALETTE["gpu_zero_copy"])

    for r in b_c:
        h = r.get_height()
        ax_c.text(r.get_x() + r.get_width()/2., h * 1.35, f"{h:.1f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel (d): Llama-3.2 1B Decode Attention: Split-KV FlashDecoding Sweep
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    splits = [32, 64, 128, 256, 512, 1024]
    time_us = [231.25, 165.14, 171.26, 185.97, 230.82, 318.53]
    alg_bw = [580.4, 812.8, 783.7, 721.8, 581.5, 421.4]

    color_line = PALETTE["gpu_explicit"]
    ax_d.plot(splits, alg_bw, marker="o", color=color_line, linewidth=2.4, label="Algorithmic Bandwidth (GB/s)")
    ax_d.set_xscale("log", base=2)
    ax_d.set_xlabel(r"Split Count Across KV Cache ($S_{kv}=65,536$)")
    ax_d.set_ylabel("Algorithmic Bandwidth (GB/s)", color=color_line)
    ax_d.tick_params(axis="y", labelcolor=color_line)
    ax_d.set_ylim(0, 1000)

    # Twin axis for latency in microseconds
    ax_d_twin = ax_d.twinx()
    color_twin = PALETTE["gpu_zero_copy"]
    ax_d_twin.plot(splits, time_us, marker="^", linestyle="--", color=color_twin, linewidth=2.0, label=r"Latency ($\mu$s)")
    ax_d_twin.set_ylabel(r"Execution Latency ($\mu$s)", color=color_twin)
    ax_d_twin.tick_params(axis="y", labelcolor=color_twin)
    ax_d_twin.set_ylim(0, 380)

    # 35.8 us Theoretical Target Line
    ax_d_twin.axhline(y=35.8, color="#b2182b", linestyle=":", linewidth=2.0, label="Ideal 35.8 us Target (3.75 TB/s)")
    ax_d.set_title("(d) Llama Decode: Peak at 64 Splits (165.1 us, 812.8 GB/s)", fontweight="bold", pad=8)
    ax_d.set_xticks(splits)
    ax_d.set_xticklabels([str(s) for s in splits])

    ax_d.annotate("Peak: 165.1 us\n812.8 GB/s (64 splits)", xy=(64, 812.8), xytext=(110, 890),
                  arrowprops=dict(facecolor=color_line, shrink=0.08, width=1.4, headwidth=5),
                  fontsize=9.5, fontweight="bold", color=color_line)

    # --------------------------------------------------------------------------
    # Panel (e): Zero-Copy GEMM C2C Wire Traffic vs HBM-Staged
    # --------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    gemm_cases = ["Zero-Copy\n(M=8192)", "Zero-Copy\n(M=16384)", "HBM-Staged\n(M=16384 E2E)"]
    tflops_vals = [21.79, 28.77, 611.70]
    bar_cols = [PALETTE["gpu_zero_copy"], PALETTE["gpu_zero_copy"], PALETTE["gpu_explicit"]]

    b_e = ax_e.bar(np.arange(len(gemm_cases)), tflops_vals, width=0.5, color=bar_cols, alpha=0.9, log=True)
    ax_e.set_ylabel("Sustained Compute (TFLOPS, log scale)")
    ax_e.set_title("(e) Zero-Copy vs Staged GEMM: 21.3x Staged Math Speedup", fontweight="bold", pad=8)
    ax_e.set_xticks(np.arange(len(gemm_cases)))
    ax_e.set_xticklabels(gemm_cases, fontsize=9.5)
    ax_e.set_ylim(5, 1800)

    ax_e.axhline(y=989.36, color="#b2182b", linestyle="--", linewidth=1.6, label="Hopper BF16 Peak (989.4 TF)")
    ax_e.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.0)

    ax_e.annotate("21.3x Staging Speedup\n(Math: 9.4ms, C2C: 5.0ms)", xy=(2, 611.7), xytext=(0.8, 120),
                  arrowprops=dict(facecolor=PALETTE["gpu_explicit"], shrink=0.08, width=1.4, headwidth=5),
                  fontsize=9.5, fontweight="bold", color=PALETTE["gpu_explicit"])
    ax_e.annotate("229.8 GB/s Wire Traffic\n(69.8 GB in 304 ms)", xy=(1, 28.77), xytext=(0.3, 10.0),
                  arrowprops=dict(facecolor=PALETTE["gpu_zero_copy"], shrink=0.08, width=1.4, headwidth=5),
                  fontsize=8.5, fontweight="bold", color=PALETTE["gpu_zero_copy"],
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=PALETTE["gpu_zero_copy"], alpha=0.9))

    for r in b_e:
        h = r.get_height()
        ax_e.text(r.get_x() + r.get_width()/2., h * 1.25, f"{h:.1f} TF", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel (f): NVLink-C2C Bandwidth Landscape & Measured Limits
    # --------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    benchmarks = [
        "C2C BiDir Peak",
        "C2C UniDir Peak",
        "Copy Engine (512MB)",
        "LPDDR5X DRAM Peak",
        "Copy Engine (64KB)",
        "SM Direct Read",
        "SAXPY 103GB (Verified)",
        "Grace STREAM Triad",
        "ZC GEMM Wire BW"
    ]
    bws = [900.0, 450.0, 407.86, 384.0, 381.07, 355.09, 323.82, 307.4, 229.82]
    bar_c = [
        "#4b5563", "#6b7280", PALETTE["hugepages"], "#b2182b",
        PALETTE["cpu_local"], PALETTE["cluster_dsm"], PALETTE["gpu_zero_copy"],
        PALETTE["cpu_local"], PALETTE["gpu_zero_copy"]
    ]

    y_pos = np.arange(len(benchmarks))
    ax_f.barh(y_pos, bws, color=bar_c, alpha=0.9, height=0.6)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(benchmarks, fontsize=9.0)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Bandwidth (GB/s)")
    ax_f.set_title("(f) NVLink-C2C Bandwidth Landscape vs DRAM Limits", fontweight="bold", pad=8)
    ax_f.set_xlim(0, 1100)

    for i, v in enumerate(bws):
        ax_f.text(v + 15, i, f"{v:.1f} GB/s", va="center", fontsize=8.5, fontweight="bold")

    ax_f.annotate("98.0% of NVIDIA 416G Spec\n(407.9 GB/s via Hugepages)", xy=(407.86, 2), xytext=(590, 2.7),
                  arrowprops=dict(facecolor=PALETTE["hugepages"], shrink=0.08, width=1.4, headwidth=5),
                  fontsize=9.0, fontweight="bold", color="#0e7490")

    # Save outputs
    pdf_path = os.path.join(OUT_DIR, "fig5_protocol_verification_suite_characterization.pdf")
    png_path = os.path.join(OUT_DIR, "fig5_protocol_verification_suite_characterization.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Fig 5 to {pdf_path} and {png_path}")


# ==============================================================================
# Figure 6: Real-Time 10 Hz NVML Power Draw & SM Clock Telemetry Profiles
# ==============================================================================
def plot_figure_6_telemetry_profiles():
    print(">>> Generating Figure 6: Real-Time 10 Hz NVML Power & Clock Profiles...")
    tel_csv = os.path.join(RAW_RUN_DIR, "telemetry_nvml_10hz.csv")
    if not os.path.exists(tel_csv):
        print(f"[WARNING] Telemetry CSV not found: {tel_csv}")
        return

    timestamps, sm_clocks, mem_clocks, powers = [], [], [], []
    with open(tel_csv, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4 and not parts[1].strip().startswith("clocks"):
                try:
                    t = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                    sm = float(parts[1].strip())
                    mem = float(parts[2].strip())
                    pwr = float(parts[3].strip())
                    timestamps.append(t)
                    sm_clocks.append(sm)
                    mem_clocks.append(mem)
                    powers.append(pwr)
                except ValueError:
                    pass

    if not timestamps:
        print("[WARNING] No valid telemetry records found.")
        return

    t0 = timestamps[0]
    time_sec = np.array([(t - t0).total_seconds() for t in timestamps])
    sm_clocks = np.array(sm_clocks)
    powers = np.array(powers)

    fig, ax1 = plt.subplots(figsize=(18, 6.5))

    color_pwr = "#d95f02" # Deep Orange
    line1 = ax1.plot(time_sec, powers, color=color_pwr, linewidth=1.8, label="Hopper GPU Power Draw (W)", alpha=0.9)
    ax1.set_xlabel("Elapsed Time (Seconds, 10 Hz NVML Polling)")
    ax1.set_ylabel("GPU Power Draw (Watts)", color=color_pwr)
    ax1.tick_params(axis="y", labelcolor=color_pwr)
    ax1.set_ylim(40, 250)

    ax2 = ax1.twinx()
    color_sm = "#1f77b4" # Steel Blue
    line2 = ax2.plot(time_sec, sm_clocks, color=color_sm, linewidth=1.6, linestyle="--", label="Hopper SM Clock (MHz)", alpha=0.85)
    ax2.set_ylabel("Hopper SM Clock Frequency (MHz)", color=color_sm)
    ax2.tick_params(axis="y", labelcolor=color_sm)
    ax2.set_ylim(200, 2300)

    windows = [
        ("Grace CPU STREAM\n(GPU Idle: 77W, 345MHz)", 0.0, 6.0, "#f0fdf4"),
        ("65k Histogram\n(DSM Clusters: 147W, 1874MHz)", 6.0, 15.3, "#faf5ff"),
        ("64-bit Binary Search\n(Latency Bound: 125W, 1980MHz)", 15.3, 518.1, "#eff6ff"),
        ("Zero-Copy GEMM\n(C2C Tiles: 174W, 1980MHz)", 518.1, 588.7, "#fef2f2"),
        ("Llama Decode\n(124W, 1980MHz)", 588.7, 600.2, "#fffbeb")
    ]

    for label, s, e, col in windows:
        if s < time_sec[-1]:
            e_clamp = min(e, time_sec[-1])
            ax1.axvspan(s, e_clamp, color=col, alpha=0.6)
            mid = (s + e_clamp) / 2.0
            if (e_clamp - s) > 12.0:
                ax1.text(mid, 232, label, ha="center", va="top", fontsize=9.0, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, alpha=0.85))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.9)

    plt.title("Figure 6: Real-Time 10 Hz Telemetry Across GH200 Mandatory Protocol Workloads", fontweight="bold", pad=12)

    pdf_path = os.path.join(OUT_DIR, "fig6_realtime_nvml_power_clock_telemetry.pdf")
    png_path = os.path.join(OUT_DIR, "fig6_realtime_nvml_power_clock_telemetry.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Fig 6 to {pdf_path} and {png_path}")


# ==============================================================================
# Figure 7: Energy Efficiency & Cooperative CPU+GPU Partitioning (2 Panels)
# ==============================================================================
def plot_figure_7_energy_and_cooperative():
    print(">>> Generating Figure 7: Energy Efficiency & Cooperative CPU+GPU Partitioning...")
    coop_csv = os.path.join(CLEAN_DIR, "clean_cooperative_cpu_gpu.csv")
    power_csv = os.path.join(CLEAN_DIR, "clean_power_energy_telemetry.csv")

    fig, (ax_coop, ax_pwr) = plt.subplots(1, 2, figsize=(18, 6.2))

    # --------------------------------------------------------------------------
    # Panel (a): Cooperative CPU-GPU Workload Partitioning (alpha from 0.0 to 1.0)
    # --------------------------------------------------------------------------
    coop_rows = load_csv(coop_csv)
    saxpy_coop = [r for r in coop_rows if r["workload"] == "coop_partition_saxpy_128gb"]
    
    alphas = []
    for r in saxpy_coop:
        ps = r["problem_size"]
        if "alpha=" in ps:
            a_val = float(ps.split("alpha=")[1].split("_")[0])
            bw = parse_val(r["eff_bw_gbps"])
            alphas.append((a_val, bw))

    alphas.sort(key=lambda x: x[0])
    if alphas:
        a_x = [x[0] for x in alphas]
        b_y = [x[1] for x in alphas]
        ax_coop.plot(a_x, b_y, marker="o", color=PALETTE["gpu_zero_copy"], linewidth=2.4, label="128GB SAXPY Partitioning")

        ax_coop.scatter([0.0], [b_y[0]], color=PALETTE["gpu_explicit"], s=130, zorder=5, label=f"Pure GPU C2C ({b_y[0]:.1f} GB/s)")
        ax_coop.scatter([1.0], [b_y[-1]], color=PALETTE["cpu_local"], s=130, zorder=5, label=f"Pure CPU Local ({b_y[-1]:.1f} GB/s)")

    ax_coop.set_xlabel(r"Partition Ratio $\alpha$ (Fraction Processed by Grace CPU)")
    ax_coop.set_ylabel("Total Sustained Bandwidth (GB/s)")
    ax_coop.set_title(r"(a) Cooperative CPU+GPU Partitioning: Shared LPDDR5X Contention", fontweight="bold", pad=8)
    ax_coop.set_xticks(np.linspace(0.0, 1.0, 11))
    ax_coop.set_ylim(210, 420)
    ax_coop.axhline(y=384.0, color="#b2182b", linestyle="--", linewidth=1.6, label="LPDDR5X Shared Peak (384.0 GB/s)")
    ax_coop.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

    ax_coop.annotate("Contention Valley\n(Dual Access Throttles LPDDR5X)", xy=(0.3, 279.2), xytext=(0.35, 230),
                     arrowprops=dict(facecolor="#b2182b", shrink=0.08, width=1.4, headwidth=5),
                     fontsize=9.5, fontweight="bold", color="#b2182b")

    # --------------------------------------------------------------------------
    # Panel (b): Energy Efficiency Across Regimes (Proper GFLOPS/W vs GB/s/W)
    # --------------------------------------------------------------------------
    pwr_rows = load_csv(power_csv)
    
    # We display 4 representative workloads:
    # 1. GEMM FP32 (Compute Bound) -> GFLOPS / Watt
    # 2. SAXPY FP32 (Streaming Bound) -> GB/s / Watt
    # 3. SpMV FP32 (Sparse Irregular) -> GB/s / Watt
    # 4. AES-128 (Crypto Vectorized) -> GB/s / Watt
    labels_b = [
        "GEMM FP32\n(GFLOPS/W)",
        "SAXPY FP32\n(GB/s / W)",
        "SpMV FP32\n(GB/s / W)",
        "AES-128\n(GB/s / W)"
    ]
    
    # Accurate metrics from clean_power_energy_telemetry.csv:
    # GEMM: CPU=14.44 GFLOPS/W, GPU=69.13 GFLOPS/W
    # SAXPY: CPU=1.12 GB/s/W, GPU=5.44 GB/s/W
    # SpMV: CPU=0.005 GB/s/W, GPU=1.05 GB/s/W
    # AES-128: CPU=2.64 GB/s/W, GPU=4.45 GB/s/W
    cpu_vals = [14.44, 1.12, 0.005, 2.64]
    gpu_vals = [69.13, 5.44, 1.05, 4.45]

    x_p = np.arange(len(labels_b))
    w_bar = 0.35

    b_cpu = ax_pwr.bar(x_p - w_bar/2, cpu_vals, w_bar, label="Grace CPU (72 Cores)", color=PALETTE["cpu_local"], alpha=0.9, log=True)
    b_gpu = ax_pwr.bar(x_p + w_bar/2, gpu_vals, w_bar, label="Hopper GPU (HBM3 / C2C)", color=PALETTE["gpu_explicit"], alpha=0.9, log=True)

    ax_pwr.set_ylabel("Energy Efficiency (Log Scale: GFLOPS/W or GB/s/W)")
    ax_pwr.set_title("(b) Energy Efficiency: Hopper GPU vs Grace CPU", fontweight="bold", pad=8)
    ax_pwr.set_xticks(x_p)
    ax_pwr.set_xticklabels(labels_b, fontsize=9.5)
    ax_pwr.set_ylim(0.001, 150)
    ax_pwr.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    for r in b_cpu:
        h = r.get_height()
        ax_pwr.text(r.get_x() + r.get_width()/2., h * 1.35, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in b_gpu:
        h = r.get_height()
        ax_pwr.text(r.get_x() + r.get_width()/2., h * 1.35, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    pdf_path = os.path.join(OUT_DIR, "fig7_energy_efficiency_and_cooperative_partitioning.pdf")
    png_path = os.path.join(OUT_DIR, "fig7_energy_efficiency_and_cooperative_partitioning.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Fig 7 to {pdf_path} and {png_path}")


if __name__ == "__main__":
    plot_figure_5_protocol_characterization()
    plot_figure_6_telemetry_profiles()
    plot_figure_7_energy_and_cooperative()
    print("\nAll New Grace Hopper Sweep Figures Successfully Generated!")
