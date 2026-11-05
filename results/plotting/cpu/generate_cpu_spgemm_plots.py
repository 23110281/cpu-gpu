#!/usr/bin/env python3
"""
results/plotting/cpu/generate_cpu_spgemm_plots.py
Dedicated publication-grade visualization suite for CPU SpGEMM benchmark results.

Reads CSV files from: results/data/<CPU_MODEL>/sweep_spgemm_*.csv
Outputs plots to:    results/plots/<CPU_MODEL>/spgemm/
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

    numeric_cols = ['threads', 'S', 'density', 'nnz_A', 'nnz_B', 'nnz_C',
                    'agg_tflops', 'agg_gbps', 'comp_ms', 'wall_ms',
                    'cpu_mhz_avg', 'power_w_avg', 'temp_c_max', 'mem_gb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'agg_tflops' in df.columns:
        df['agg_gflops'] = df['agg_tflops'] * 1000.0

    if 'power_w_avg' in df.columns and 'agg_gflops' in df.columns:
        df['gflops_per_watt'] = df['agg_gflops'] / df['power_w_avg'].replace(0, np.nan)

    return df

def plot_fig1_tflops_vs_size(df, out_dir, cpu_name):
    """Figure 1: Achieved GFLOPS vs Matrix Dimension (S) across densities."""
    max_threads = int(df['threads'].max())
    sub = df[(df['threads'] == max_threads) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("viridis", len(densities))

    for didx, dens in enumerate(densities):
        d_df = sub[sub['density'] == dens].sort_values('S')
        plt.plot(d_df['S'], d_df['agg_gflops'], marker='o', linewidth=2.4,
                 markersize=7, label=f"Density: {dens*100:.1f}%", color=palette[didx])

    plt.title(f"CPU SpGEMM (Sparse x Sparse) Compute vs. Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (M = N = K = S)")
    plt.ylabel("Compute Throughput (GFLOPS)")
    plt.xscale('log', base=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_spgemm_gflops_vs_size.png"), dpi=300)
    plt.close()

def plot_fig2_bandwidth_vs_size(df, out_dir, cpu_name):
    """Figure 2: Achieved Memory Bandwidth (GB/s) vs Matrix Size."""
    if 'agg_gbps' not in df.columns: return
    max_threads = int(df['threads'].max())
    sub = df[(df['threads'] == max_threads) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("rocket", len(densities))

    for didx, dens in enumerate(densities):
        d_df = sub[sub['density'] == dens].sort_values('S')
        plt.plot(d_df['S'], d_df['agg_gbps'], marker='s', linewidth=2.2,
                 markersize=6, label=f"Density: {dens*100:.1f}%", color=palette[didx])

    plt.title(f"SpGEMM Effective Memory Bandwidth vs. Matrix Size ({cpu_name})", pad=12)
    plt.xlabel("Matrix Dimension (S)")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.xscale('log', base=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_spgemm_bandwidth_vs_size.png"), dpi=300)
    plt.close()

def plot_fig3_thread_scaling(df, out_dir, cpu_name):
    """Figure 3: Multi-Core Thread Scaling for SpGEMM."""
    max_size = df['S'].max()
    sub = df[(df['S'] == max_size) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("magma", len(densities))

    for didx, dens in enumerate(densities):
        d_df = sub[sub['density'] == dens].sort_values('threads')
        plt.plot(d_df['threads'], d_df['agg_gflops'], marker='^', linewidth=2.2,
                 markersize=7, label=f"Density: {dens*100:.1f}%", color=palette[didx])

    plt.title(f"SpGEMM Thread Scaling at S={int(max_size)} ({cpu_name})", pad=12)
    plt.xlabel("Number of Threads")
    plt.ylabel("Compute Throughput (GFLOPS)")
    plt.grid(True, alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_spgemm_thread_scaling.png"), dpi=300)
    plt.close()

def plot_fig4_speedup(df, out_dir, cpu_name):
    """Figure 4: Speedup relative to 1 thread for SpGEMM."""
    max_size = df['S'].max()
    sub = df[(df['S'] == max_size) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    threads = sorted(sub['threads'].unique())
    plt.plot(threads, threads, 'k--', alpha=0.5, label="Ideal Linear Speedup")

    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("Set2", len(densities))

    for didx, dens in enumerate(densities):
        d_df = sub[sub['density'] == dens].sort_values('threads')
        t1 = d_df[d_df['threads'] == 1]
        if t1.empty or t1['agg_gflops'].values[0] <= 0: continue
        speedup = d_df['agg_gflops'] / t1['agg_gflops'].values[0]
        plt.plot(d_df['threads'], speedup, marker='o', linewidth=2.2,
                 label=f"Density: {dens*100:.1f}%", color=palette[didx])

    plt.title(f"SpGEMM Parallel Speedup vs. Threads ({cpu_name})", pad=12)
    plt.xlabel("Thread Count")
    plt.ylabel("Speedup over 1 Thread")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_spgemm_speedup.png"), dpi=300)
    plt.close()

def plot_fig5_density_impact(df, out_dir, cpu_name):
    """Figure 5: Impact of Matrix Density on Compute Latency (comp_ms)."""
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    sizes = sorted(sub['S'].unique())
    palette = sns.color_palette("crest", len(sizes))

    for sidx, S in enumerate(sizes):
        s_df = sub[sub['S'] == S].sort_values('density')
        plt.plot(s_df['density'] * 100.0, s_df['comp_ms'], marker='^', linewidth=2.2,
                 markersize=7, label=f"S = {int(S)}", color=palette[sidx])

    plt.title(f"SpGEMM Execution Time vs. Nonzero Density ({cpu_name})", pad=12)
    plt.xlabel("Matrix Density (%)")
    plt.ylabel("Compute Time per Iteration (ms)")
    plt.grid(True, alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_spgemm_density_impact.png"), dpi=300)
    plt.close()

def plot_fig6_heatmap(df, out_dir):
    """Figure 6: 2D Heatmap of Achieved GFLOPS (Size S vs Threads)."""
    sub = df[df['precision'] == 'fp32']
    if sub.empty: return
    densities = sorted(sub['density'].unique())
    target_dens = densities[0]
    dens_df = sub[sub['density'] == target_dens]
    if dens_df.empty: return

    plt.figure(figsize=(7, 5.5))
    pivot = dens_df.pivot_table(index='S', columns='threads', values='agg_gflops')
    pivot = pivot.sort_index(ascending=False)

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="mako", cbar_kws={'label': 'GFLOPS'})
    plt.title(f"SpGEMM GFLOPS Heatmap (Density: {target_dens*100:.1f}%)", pad=12)
    plt.ylabel("Matrix Size (S)")
    plt.xlabel("Threads (T)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_spgemm_heatmap.png"), dpi=300)
    plt.close()

def plot_fig7_power_and_efficiency(df, out_dir, cpu_name):
    """Figure 7: Power Draw and Energy Efficiency vs Matrix Size."""
    if 'power_w_avg' not in df.columns or df['power_w_avg'].dropna().empty: return
    max_threads = int(df['threads'].max())
    sub = df[(df['threads'] == max_threads) & (df['precision'] == 'fp32')].copy()
    if sub.empty: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    densities = sorted(sub['density'].unique())
    palette = sns.color_palette("tab10", len(densities))

    for didx, dens in enumerate(densities):
        d_df = sub[sub['density'] == dens].sort_values('S')
        lbl = f"Density: {dens*100:.1f}%"
        ax1.plot(d_df['S'], d_df['power_w_avg'], marker='o', linewidth=2.2, label=lbl, color=palette[didx])
        if 'gflops_per_watt' in d_df.columns:
            ax2.plot(d_df['S'], d_df['gflops_per_watt'], marker='s', linewidth=2.2, label=lbl, color=palette[didx])

    ax1.set_title("Average Power Draw (Watts)")
    ax1.set_xlabel("Matrix Dimension (S)")
    ax1.set_ylabel("Power (Watts)")
    ax1.set_xscale('log', base=2)
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend(frameon=True)

    ax2.set_title("Energy Efficiency (GFLOPS / Watt)")
    ax2.set_xlabel("Matrix Dimension (S)")
    ax2.set_ylabel("GFLOPS / Watt")
    ax2.set_xscale('log', base=2)
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend(frameon=True)

    fig.suptitle(f"SpGEMM Energy & Efficiency Profile ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_spgemm_power_efficiency.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig8_freq_temp(df, out_dir, cpu_name):
    """Figure 8: CPU Core Frequency (MHz) and Peak Temperature (°C)."""
    if 'cpu_mhz_avg' not in df.columns or 'temp_c_max' not in df.columns: return
    max_threads = int(df['threads'].max())
    sub = df[(df['threads'] == max_threads) & (df['precision'] == 'fp32')].sort_values('S')
    if sub.empty: return
    # Pick first density to show thermal line
    sub = sub[sub['density'] == sub['density'].iloc[0]]

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

    fig.suptitle(f"CPU Telemetry: Frequency & Thermal Profile (SpGEMM) ({cpu_name})", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_spgemm_freq_temp.png"), dpi=300)
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # -> results/
    csv_pattern = os.path.join(base_dir, 'data', 'legacy', 'cpu', '*', 'sweep_spgemm_*.csv')
    csv_files = [
        path for path in glob.glob(csv_pattern)
        if any(name in os.path.basename(os.path.dirname(path)).lower() for name in ('cpu', 'intel', 'amd'))
    ]

    if not csv_files:
        print(f"No CPU SpGEMM CSV files found matching: {csv_pattern}")
        return

    set_style()
    cleaned_dirs = set()

    for csv_file in csv_files:
        cpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', cpu_name, 'spgemm')
        os.makedirs(out_dir, exist_ok=True)
        if out_dir not in cleaned_dirs:
            for old_plot in glob.glob(os.path.join(out_dir, 'fig*.png')):
                os.remove(old_plot)
            cleaned_dirs.add(out_dir)

        print(f"==> Processing {csv_file} for {cpu_name}...")
        df = load_data(csv_file)
        if df.empty: continue

        plot_fig1_tflops_vs_size(df, out_dir, cpu_name)
        plot_fig2_bandwidth_vs_size(df, out_dir, cpu_name)
        plot_fig3_thread_scaling(df, out_dir, cpu_name)
        plot_fig4_speedup(df, out_dir, cpu_name)
        plot_fig5_density_impact(df, out_dir, cpu_name)
        plot_fig6_heatmap(df, out_dir)
        plot_fig7_power_and_efficiency(df, out_dir, cpu_name)
        plot_fig8_freq_temp(df, out_dir, cpu_name)
        print(f"    [SUCCESS] 8 publication-grade figures saved to: {out_dir}")

if __name__ == '__main__':
    main()
