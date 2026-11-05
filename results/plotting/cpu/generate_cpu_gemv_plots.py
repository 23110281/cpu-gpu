#!/usr/bin/env python3
"""
results/plotting/cpu/generate_cpu_gemv_plots.py
Dedicated publication-grade visualization suite for CPU GEMV / SpMV benchmark results.

Reads CSV files from: results/data/<CPU_MODEL>/sweep_gemv_*.csv
Outputs plots to:    results/plots/<CPU_MODEL>/gemv/
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

    numeric_cols = ['threads', 'S', 'density', 'nnz', 'agg_gbps', 'agg_tflops',
                    'comp_ms', 'wall_ms', 'cpu_mhz_avg', 'power_w_avg', 'temp_c_max', 'mem_gb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'power_w_avg' in df.columns and 'agg_gbps' in df.columns:
        df['gbps_per_watt'] = df['agg_gbps'] / df['power_w_avg'].replace(0, np.nan)

    return df

def plot_fig1_dense_bandwidth_vs_size(df, out_dir, cpu_name):
    """Figure 1: Achieved Bandwidth vs Matrix Dimension (S) for Dense GEMV."""
    dense = df[df['engine'] == 'dense'].copy()
    if dense.empty: return
    max_threads = int(dense['threads'].max())
    sub = dense[dense['threads'] == max_threads]
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['agg_gbps'], marker='o', linewidth=2.4,
                 markersize=7, label=f"Dense {prec.upper()} ({max_threads}T)",
                 color=palette.get(prec, '#333333'))

    plt.title(f"Dense GEMV Memory Bandwidth vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (M = N = S)")
    plt.ylabel("Achieved Bandwidth (GB/s)")
    plt.xscale('log', base=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_gemv_dense_bandwidth_vs_size.png"), dpi=300)
    plt.close()

def plot_fig2_dense_vs_sparse(df, out_dir, cpu_name):
    """Figure 2: Direct Comparison: Dense GEMV vs Sparse SpMV Bandwidth."""
    max_threads = int(df['threads'].max())
    sub = df[(df['threads'] == max_threads) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    dense = sub[sub['engine'] == 'dense'].sort_values('S')
    if not dense.empty:
        plt.plot(dense['S'], dense['agg_gbps'], marker='o', linewidth=2.5,
                 color='#1f77b4', label="Dense GEMV (100% Contiguous)")

    sparse = sub[sub['engine'] == 'sparse']
    if not sparse.empty:
        densities = sorted(sparse['density'].unique())
        colors = sns.color_palette("rocket", len(densities))
        for didx, d in enumerate(densities):
            s_df = sparse[sparse['density'] == d].sort_values('S')
            plt.plot(s_df['S'], s_df['agg_gbps'], marker='^', linewidth=2,
                     linestyle='--', color=colors[didx], label=f"SpMV CSR (Density: {d*100:.1f}%)")

    plt.title(f"Memory Bandwidth: Dense GEMV vs. Sparse SpMV ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Achieved Bandwidth (GB/s)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_gemv_dense_vs_sparse.png"), dpi=300)
    plt.close()

def plot_fig3_thread_scaling(df, out_dir, cpu_name):
    """Figure 3: Bandwidth Scaling across Thread Counts."""
    dense = df[df['engine'] == 'dense']
    if dense.empty: return
    precisions = sorted(dense['precision'].unique())
    sizes = sorted(dense['S'].unique())
    target_sizes = [sizes[len(sizes)//2], sizes[-1]] if len(sizes) >= 2 else sizes

    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        for S in target_sizes:
            sub = dense[(dense['precision'] == prec) & (dense['S'] == S)].sort_values('threads')
            if sub.empty: continue
            ax.plot(sub['threads'], sub['agg_gbps'], marker='s', linewidth=2.2,
                    markersize=6, label=f"S = {int(S)}")

        ax.set_title(f"Dense GEMV: {prec.upper()}")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Bandwidth (GB/s)" if idx == 0 else "")
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=True)

    fig.suptitle(f"Dense GEMV Bandwidth vs. Thread Count ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_gemv_thread_scaling.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig4_spmv_density_scaling(df, out_dir, cpu_name):
    """Figure 4: Sparse SpMV Throughput (GFLOPS) across Matrix Densities."""
    sp = df[df['engine'] == 'sparse']
    if sp.empty: return
    max_threads = int(sp['threads'].max())
    sub = sp[(sp['threads'] == max_threads) & (sp['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("viridis", len(densities))

    for didx, d in enumerate(densities):
        d_df = sub[sub['density'] == d].sort_values('S')
        gflops = d_df['agg_tflops'] * 1000.0
        plt.plot(d_df['S'], gflops, marker='^', linewidth=2.2,
                 color=palette[didx], label=f"Density {d*100:.1f}%")

    plt.title(f"Sparse SpMV Compute Throughput (GFLOPS) vs. Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Compute Throughput (GFLOPS)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_spmv_density_scaling.png"), dpi=300)
    plt.close()

def plot_fig5_heatmap(df, out_dir):
    """Figure 5: 2D Heatmap of Achieved GB/s (Matrix Size S vs Threads)."""
    dense = df[df['engine'] == 'dense']
    if dense.empty: return
    precisions = sorted(dense['precision'].unique())
    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5.5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        sub = dense[dense['precision'] == prec]
        if sub.empty: continue

        pivot = sub.pivot_table(index='S', columns='threads', values='agg_gbps')
        pivot = pivot.sort_index(ascending=False)

        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="mako", ax=ax,
                    cbar_kws={'label': 'GB/s'} if idx == len(precisions) - 1 else None)
        ax.set_title(f"Dense GEMV: {prec.upper()}")
        ax.set_ylabel("Matrix Size (S)" if idx == 0 else "")
        ax.set_xlabel("Threads (T)")

    fig.suptitle("Dense GEMV Memory Bandwidth Heatmap (GB/s)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_gemv_heatmap.png"), dpi=300)
    plt.close()

def plot_fig6_power_vs_size(df, out_dir, cpu_name):
    """Figure 6: RAPL Power Consumption (Watts) vs Matrix Size."""
    if 'power_w_avg' not in df.columns or df['power_w_avg'].dropna().empty: return
    dense = df[df['engine'] == 'dense']
    if dense.empty: return
    max_threads = int(dense['threads'].max())
    sub = dense[dense['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['power_w_avg'], marker='o', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"CPU Power Draw under GEMV Streaming ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Average Power (Watts)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_gemv_power_vs_size.png"), dpi=300)
    plt.close()

def plot_fig7_energy_efficiency(df, out_dir, cpu_name):
    """Figure 7: Bandwidth Energy Efficiency (GB/s per Watt)."""
    if 'gbps_per_watt' not in df.columns or df['gbps_per_watt'].dropna().empty: return
    dense = df[df['engine'] == 'dense']
    if dense.empty: return
    max_threads = int(dense['threads'].max())
    sub = dense[dense['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('S')
        plt.plot(p_df['S'], p_df['gbps_per_watt'], marker='s', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"GEMV Energy Efficiency (GB/s per Watt) vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Bandwidth Efficiency (GB/s / Watt)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_gemv_energy_efficiency.png"), dpi=300)
    plt.close()

def plot_fig8_freq_temp(df, out_dir, cpu_name):
    """Figure 8: CPU Core Frequency (MHz) and Peak Temperature (°C)."""
    if 'cpu_mhz_avg' not in df.columns or 'temp_c_max' not in df.columns: return
    dense = df[df['engine'] == 'dense']
    if dense.empty: return
    max_threads = int(dense['threads'].max())
    sub = dense[(dense['threads'] == max_threads) & (dense['precision'] == 'fp32')].sort_values('S')
    if sub.empty: return

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    color1 = '#1f77b4'
    ax1.set_xlabel("Matrix Dimension (S)")
    ax1.set_ylabel("Average CPU Frequency (MHz)", color=color1)
    ax1.plot(sub['S'], sub['cpu_mhz_avg'], color=color1, marker='o', linewidth=2.2, label="Core Frequency (MHz)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel("Peak Core Temperature (°C)", color=color2)
    ax2.plot(sub['S'], sub['temp_c_max'], color=color2, marker='^', linewidth=2.2, linestyle='--', label="Peak Temp (°C)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    fig.suptitle(f"CPU Telemetry: Frequency & Thermal Profile (GEMV) ({cpu_name})", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_gemv_freq_temp.png"), dpi=300)
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # -> results/
    csv_pattern = os.path.join(base_dir, 'data', 'legacy', 'cpu', '*', 'sweep_gemv_*.csv')
    csv_files = [
        path for path in glob.glob(csv_pattern)
        if any(name in os.path.basename(os.path.dirname(path)).lower() for name in ('cpu', 'intel', 'amd'))
    ]

    if not csv_files:
        print(f"No CPU GEMV CSV files found matching: {csv_pattern}")
        return

    set_style()
    cleaned_dirs = set()

    for csv_file in csv_files:
        cpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', cpu_name, 'gemv')
        os.makedirs(out_dir, exist_ok=True)
        if out_dir not in cleaned_dirs:
            for old_plot in glob.glob(os.path.join(out_dir, 'fig*.png')):
                os.remove(old_plot)
            cleaned_dirs.add(out_dir)

        print(f"==> Processing {csv_file} for {cpu_name}...")
        df = load_data(csv_file)
        if df.empty: continue

        plot_fig1_dense_bandwidth_vs_size(df, out_dir, cpu_name)
        plot_fig2_dense_vs_sparse(df, out_dir, cpu_name)
        plot_fig3_thread_scaling(df, out_dir, cpu_name)
        plot_fig4_spmv_density_scaling(df, out_dir, cpu_name)
        plot_fig5_heatmap(df, out_dir)
        plot_fig6_power_vs_size(df, out_dir, cpu_name)
        plot_fig7_energy_efficiency(df, out_dir, cpu_name)
        plot_fig8_freq_temp(df, out_dir, cpu_name)
        print(f"    [SUCCESS] 8 publication-grade figures saved to: {out_dir}")

if __name__ == '__main__':
    main()
