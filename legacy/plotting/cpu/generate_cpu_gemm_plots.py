#!/usr/bin/env python3
"""
results/plotting/cpu/generate_cpu_gemm_plots.py
Dedicated publication-grade visualization suite for CPU GEMM benchmark results.

Reads CSV files from: results/data/<CPU_MODEL>/sweep_gemm_*.csv
Outputs plots to:    results/plots/<CPU_MODEL>/gemm/
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.45,
        'grid.linestyle': '--'
    })

def load_data(filepath):
    df = pd.read_csv(filepath)
    ok = df['status'].astype(str).str.startswith('ok')
    df = df[ok].copy()

    # Normalise dimension column: M, N, K, or S
    if 'S' not in df.columns and 'M' in df.columns:
        df['S'] = df['M']

    numeric_cols = ['threads', 'M', 'N', 'K', 'S', 'agg_tflops', 'comp_ms',
                    'cpu_mhz_avg', 'power_w_avg', 'temp_c_max', 'mem_gb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'power_w_avg' in df.columns and 'agg_tflops' in df.columns:
        df['gflops_per_watt'] = (df['agg_tflops'] * 1000.0) / df['power_w_avg'].replace(0, np.nan)

    return df

def plot_fig1_tflops_vs_size(df, out_dir, cpu_name):
    """Figure 1: Achieved TFLOPS vs Matrix Dimension (S) at maximum thread count."""
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e', 'bf16': '#2ca02c'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['agg_tflops'], marker='o', linewidth=2.4,
                 markersize=7, label=f"{prec.upper()} ({max_threads} Threads)",
                 color=palette.get(prec, '#333333'))

    plt.title(f"CPU GEMM Compute Throughput vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (M = N = K = S)")
    plt.ylabel("Achieved Throughput (TFLOPS)")
    plt.xscale('log', base=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True, facecolor='white', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_gemm_tflops_vs_size.png"), dpi=300)
    plt.close()

def plot_fig2_tflops_by_threads(df, out_dir, cpu_name):
    """Figure 2: TFLOPS vs Matrix Size curves broken down by thread count."""
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(1, len(precisions), figsize=(6.5 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        p_df = df[df['precision'] == prec]
        threads = sorted(p_df['threads'].unique())
        palette = sns.color_palette("magma", len(threads))

        for tidx, T in enumerate(threads):
            t_df = p_df[p_df['threads'] == T].sort_values('S')
            ax.plot(t_df['S'], t_df['agg_tflops'], marker='o', linewidth=2, markersize=5,
                    label=f"{int(T)}T", color=palette[tidx])

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Matrix Dimension (S)")
        ax.set_ylabel("Throughput (TFLOPS)" if idx == 0 else "")
        ax.set_xscale('log', base=2)
        ax.legend(title="Threads", frameon=True)
        ax.grid(True, which="both", alpha=0.4)

    fig.suptitle(f"GEMM Throughput Scaling across Threads ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_gemm_tflops_by_threads.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig3_thread_scaling(df, out_dir, cpu_name):
    """Figure 3: Multi-core Thread Scaling (TFLOPS vs Threads) for key matrix sizes."""
    precisions = sorted(df['precision'].unique())
    sizes = sorted(df['S'].unique())
    if len(sizes) == 0: return
    selected_sizes = [sizes[0], sizes[len(sizes)//2], sizes[-1]] if len(sizes) >= 3 else sizes

    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        for S in selected_sizes:
            sub = df[(df['precision'] == prec) & (df['S'] == S)].sort_values('threads')
            if sub.empty: continue
            ax.plot(sub['threads'], sub['agg_tflops'], marker='s', linewidth=2.2,
                    markersize=6, label=f"S = {int(S)}")

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Achieved TFLOPS" if idx == 0 else "")
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=True)

    fig.suptitle(f"Multi-Core Scaling: TFLOPS vs. Thread Count ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_gemm_thread_scaling.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig4_speedup_scaling(df, out_dir, cpu_name):
    """Figure 4: Speedup relative to 1 thread against ideal linear scaling."""
    precisions = sorted(df['precision'].unique())
    sizes = sorted(df['S'].unique())
    if len(sizes) == 0: return
    target_s = sizes[-1]  # Largest matrix reaches closest to ideal compute bound

    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        p_df = df[(df['precision'] == prec) & (df['S'] == target_s)].sort_values('threads')
        if p_df.empty: continue

        threads = sorted(p_df['threads'].unique())
        ax.plot(threads, threads, 'k--', alpha=0.5, label="Ideal Linear Speedup")

        t1 = p_df[p_df['threads'] == 1]
        if not t1.empty and t1['agg_tflops'].values[0] > 0:
            base = t1['agg_tflops'].values[0]
            speedup = p_df['agg_tflops'] / base
            ax.plot(p_df['threads'], speedup, marker='^', linewidth=2.2, color='#1f77b4',
                    label=f"Achieved (S={int(target_s)})")

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Speedup over 1 Thread" if idx == 0 else "")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.4)

    fig.suptitle(f"GEMM Parallel Speedup vs. Core Count ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_gemm_speedup.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig5_heatmap(df, out_dir):
    """Figure 5: 2D Heatmap of TFLOPS (Matrix Size vs Threads)."""
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5.5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        sub = df[df['precision'] == prec]
        if sub.empty: continue

        pivot = sub.pivot_table(index='S', columns='threads', values='agg_tflops')
        pivot = pivot.sort_index(ascending=False)

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", ax=ax,
                    cbar_kws={'label': 'TFLOPS'} if idx == len(precisions) - 1 else None)
        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_ylabel("Matrix Size (S)" if idx == 0 else "")
        ax.set_xlabel("Threads (T)")

    fig.suptitle("GEMM Compute Throughput Heatmap (TFLOPS)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_gemm_heatmap.png"), dpi=300)
    plt.close()

def plot_fig6_power_vs_size(df, out_dir, cpu_name):
    """Figure 6: RAPL Power Consumption (Watts) vs Matrix Size."""
    if 'power_w_avg' not in df.columns or df['power_w_avg'].dropna().empty: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e', 'bf16': '#2ca02c'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['power_w_avg'], marker='o', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"CPU Package & DRAM Power Draw vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Average Power (Watts)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_gemm_power_vs_size.png"), dpi=300)
    plt.close()

def plot_fig7_energy_efficiency(df, out_dir, cpu_name):
    """Figure 7: Energy Efficiency (GFLOPS per Watt) vs Matrix Size."""
    if 'gflops_per_watt' not in df.columns or df['gflops_per_watt'].dropna().empty: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e', 'bf16': '#2ca02c'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['gflops_per_watt'], marker='s', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"GEMM Energy Efficiency (GFLOPS per Watt) vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Efficiency (GFLOPS / Watt)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_gemm_energy_efficiency.png"), dpi=300)
    plt.close()

def plot_fig8_fp32_vs_fp64_ratio(df, out_dir, cpu_name):
    """Figure 8: Ratio of FP32 to FP64 TFLOPS (SIMD 2x Vector Width Ratio)."""
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads]
    f32 = sub[sub['precision'] == 'fp32'].sort_values('S')
    f64 = sub[sub['precision'] == 'fp64'].sort_values('S')
    if f32.empty or f64.empty: return

    merged = pd.merge(f32[['S', 'agg_tflops']], f64[['S', 'agg_tflops']], on='S', suffixes=('_fp32', '_fp64'))
    if merged.empty: return

    ratio = merged['agg_tflops_fp32'] / merged['agg_tflops_fp64']

    plt.figure(figsize=(8.5, 5))
    plt.plot(merged['S'], ratio, marker='o', linewidth=2.4, color='#9467bd', label="FP32 / FP64 TFLOPS Ratio")
    plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.6, label="Ideal SIMD 2.0x Vector Width Peak")
    plt.title(f"FP32 vs. FP64 Compute Scaling Ratio ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("TFLOPS Ratio (FP32 / FP64)")
    plt.xscale('log', base=2)
    plt.ylim(0.8, 2.5)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_gemm_fp32_vs_fp64_ratio.png"), dpi=300)
    plt.close()

def plot_fig9_telemetry_freq_temp(df, out_dir, cpu_name):
    """Figure 9: CPU Core Frequency (MHz) and Peak Temperature (°C)."""
    if 'cpu_mhz_avg' not in df.columns or 'temp_c_max' not in df.columns: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads]
    f32 = sub[sub['precision'] == 'fp32'].sort_values('S')
    if f32.empty: return

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    color1 = '#1f77b4'
    ax1.set_xlabel("Matrix Dimension (S)")
    ax1.set_ylabel("Average CPU Frequency (MHz)", color=color1)
    ax1.plot(f32['S'], f32['cpu_mhz_avg'], color=color1, marker='o', linewidth=2.2, label="Core Frequency (MHz)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel("Peak Core Temperature (°C)", color=color2)
    ax2.plot(f32['S'], f32['temp_c_max'], color=color2, marker='^', linewidth=2.2, linestyle='--', label="Peak Temp (°C)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    fig.suptitle(f"CPU Telemetry: Frequency & Thermal Profile under Heavy GEMM ({cpu_name})", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig9_gemm_freq_temp.png"), dpi=300)
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # -> legacy/
    csv_pattern = os.path.join(base_dir, 'data', 'cpu', '*', 'sweep_gemm_*.csv')
    csv_files = [
        path for path in glob.glob(csv_pattern)
        if any(name in os.path.basename(os.path.dirname(path)).lower() for name in ('cpu', 'intel', 'amd'))
    ]

    if not csv_files:
        print(f"No CPU GEMM CSV files found matching: {csv_pattern}")
        return

    set_style()
    cleaned_dirs = set()

    for csv_file in csv_files:
        cpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', cpu_name, 'gemm')
        os.makedirs(out_dir, exist_ok=True)
        if out_dir not in cleaned_dirs:
            for old_plot in glob.glob(os.path.join(out_dir, 'fig*.png')):
                os.remove(old_plot)
            cleaned_dirs.add(out_dir)

        print(f"==> Processing {csv_file} for {cpu_name}...")
        df = load_data(csv_file)
        if df.empty: continue

        plot_fig1_tflops_vs_size(df, out_dir, cpu_name)
        plot_fig2_tflops_by_threads(df, out_dir, cpu_name)
        plot_fig3_thread_scaling(df, out_dir, cpu_name)
        plot_fig4_speedup_scaling(df, out_dir, cpu_name)
        plot_fig5_heatmap(df, out_dir)
        plot_fig6_power_vs_size(df, out_dir, cpu_name)
        plot_fig7_energy_efficiency(df, out_dir, cpu_name)
        plot_fig8_fp32_vs_fp64_ratio(df, out_dir, cpu_name)
        plot_fig9_telemetry_freq_temp(df, out_dir, cpu_name)
        print(f"    [SUCCESS] 9 publication-grade figures saved to: {out_dir}")

if __name__ == '__main__':
    main()
