#!/usr/bin/env python3
"""
results/plotting/cpu/generate_cpu_saxpy_plots.py
Dedicated publication-grade visualization suite for CPU SAXPY / DAXPY benchmark results.

Reads CSV files from: results/data/<CPU_MODEL>/sweep_saxpy_*.csv
Outputs plots to:    results/plots/<CPU_MODEL>/saxpy/
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

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

    numeric_cols = ['threads', 'N', 'agg_gbps', 'agg_tflops', 'comp_ms', 'wall_ms',
                    'cpu_mhz_avg', 'power_w_avg', 'temp_c_max', 'mem_gb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'power_w_avg' in df.columns and 'agg_gbps' in df.columns:
        df['gbps_per_watt'] = df['agg_gbps'] / df['power_w_avg'].replace(0, np.nan)

    return df

def get_peak_dram_bw(cpu_name):
    """Estimate theoretical memory bandwidth for known Xeon families."""
    name_lower = cpu_name.lower()
    if 'd-1548' in name_lower or 'd1548' in name_lower:
        return 34.13, "m510 Dual-ch DDR4-2133 (34.1 GB/s)"
    elif 'c6620' in name_lower or 'sapphire' in name_lower or '8480' in name_lower or '6430' in name_lower or 'platinum' in name_lower or 'gold' in name_lower:
        return 307.2, "c6620 8-ch DDR5-4800 (~307 GB/s/socket)"
    elif 'e5-26' in name_lower:
        return 76.8, "Xeon E5 Quad-ch DDR4 (~76.8 GB/s)"
    return None, None

def plot_fig1_bandwidth_vs_n(df, out_dir, cpu_name):
    """Figure 1: Achieved Bandwidth vs Vector Size (N) showing Cache vs DRAM transition."""
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(9, 5.5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('N')
        plt.plot(p_df['N'], p_df['agg_gbps'], marker='o', linewidth=2.4,
                 markersize=7, label=f"{prec.upper()} ({max_threads} Threads)",
                 color=palette.get(prec, '#333333'))

    peak_bw, peak_lbl = get_peak_dram_bw(cpu_name)
    if peak_bw:
        plt.axhline(y=peak_bw, color='#d62728', linestyle=':', linewidth=2,
                    label=f"Theoretical DRAM Peak: {peak_lbl}")

    plt.title(f"CPU SAXPY Memory Bandwidth vs. Vector Size ({cpu_name})", pad=12)
    plt.xlabel("Vector Length (N Elements)")
    plt.ylabel("Achieved Bandwidth (GB/s)")
    plt.xscale('log', base=2)
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(frameon=True, facecolor='white', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_saxpy_bandwidth_vs_n.png"), dpi=300)
    plt.close()

def plot_fig2_bandwidth_by_threads(df, out_dir, cpu_name):
    """Figure 2: Bandwidth vs Vector Size curves broken down by thread count."""
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(1, len(precisions), figsize=(6.5 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        p_df = df[df['precision'] == prec]
        threads = sorted(p_df['threads'].unique())
        palette = sns.color_palette("viridis", len(threads))

        for tidx, T in enumerate(threads):
            t_df = p_df[p_df['threads'] == T].sort_values('N')
            ax.plot(t_df['N'], t_df['agg_gbps'], marker='o', linewidth=2, markersize=5,
                    label=f"{int(T)}T", color=palette[tidx])

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Vector Length (N Elements)")
        ax.set_ylabel("Bandwidth (GB/s)" if idx == 0 else "")
        ax.set_xscale('log', base=2)
        ax.legend(title="Threads", frameon=True)
        ax.grid(True, which="both", alpha=0.4)

    fig.suptitle(f"SAXPY Bandwidth Scaling across Thread Counts ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_saxpy_bandwidth_by_threads.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig3_thread_scaling(df, out_dir, cpu_name):
    """Figure 3: Bandwidth vs Thread count for cache-resident vs DRAM-resident vectors."""
    precisions = sorted(df['precision'].unique())
    sizes = sorted(df['N'].unique())
    if len(sizes) == 0: return

    # Pick smallest (cache) and largest (DRAM) sizes
    selected_sizes = [sizes[0], sizes[len(sizes)//2], sizes[-1]] if len(sizes) >= 3 else sizes

    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        for N in selected_sizes:
            sub = df[(df['precision'] == prec) & (df['N'] == N)].sort_values('threads')
            if sub.empty: continue
            label = f"N = 2^{int(np.log2(N))}" if (N & (N - 1) == 0) else f"N = {N/1e6:.1f}M"
            ax.plot(sub['threads'], sub['agg_gbps'], marker='s', linewidth=2.2,
                    markersize=6, label=label)

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Achieved Bandwidth (GB/s)" if idx == 0 else "")
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=True)

    fig.suptitle(f"Memory Bus Saturation vs. Thread Count ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_saxpy_thread_scaling.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig4_speedup_scaling(df, out_dir, cpu_name):
    """Figure 4: Parallel Speedup relative to 1 Thread."""
    precisions = sorted(df['precision'].unique())
    sizes = sorted(df['N'].unique())
    if len(sizes) == 0: return
    selected_sizes = [sizes[0], sizes[-1]] if len(sizes) >= 2 else sizes

    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        p_df = df[df['precision'] == prec]
        threads = sorted(p_df['threads'].unique())
        ax.plot(threads, threads, 'k--', alpha=0.5, label="Ideal Linear Speedup")

        for N in selected_sizes:
            sub = p_df[p_df['N'] == N].sort_values('threads')
            t1 = sub[sub['threads'] == 1]
            if t1.empty or t1['agg_gbps'].values[0] == 0: continue
            base_bw = t1['agg_gbps'].values[0]
            speedup = sub['agg_gbps'] / base_bw
            label = f"N = 2^{int(np.log2(N))}" if (N & (N - 1) == 0) else f"N = {N/1e6:.1f}M"
            ax.plot(sub['threads'], speedup, marker='^', linewidth=2, label=label)

        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Speedup over 1 Thread" if idx == 0 else "")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.4)

    fig.suptitle(f"SAXPY Multi-Thread Speedup ({cpu_name})", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_saxpy_speedup.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig5_heatmap(df, out_dir):
    """Figure 5: 2D Heatmap of Achieved GB/s (Vector Size N vs Threads)."""
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(1, len(precisions), figsize=(6 * len(precisions), 5.5), squeeze=False)

    for idx, prec in enumerate(precisions):
        ax = axes[0, idx]
        sub = df[df['precision'] == prec]
        if sub.empty: continue

        pivot = sub.pivot_table(index='N', columns='threads', values='agg_gbps')
        pivot = pivot.sort_index(ascending=False)
        pivot.index = [f"2^{int(np.log2(x))}" if (int(x) & (int(x) - 1) == 0) else f"{x/1e6:.1f}M" for x in pivot.index]

        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="mako", ax=ax,
                    cbar_kws={'label': 'GB/s'} if idx == len(precisions) - 1 else None)
        ax.set_title(f"Precision: {prec.upper()}")
        ax.set_ylabel("Vector Size (N)" if idx == 0 else "")
        ax.set_xlabel("Threads (T)")

    fig.suptitle("SAXPY Memory Bandwidth Heatmap (GB/s)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_saxpy_heatmap.png"), dpi=300)
    plt.close()

def plot_fig6_power_vs_n(df, out_dir, cpu_name):
    """Figure 6: RAPL Power Consumption (Watts) vs Vector Size."""
    if 'power_w_avg' not in df.columns or df['power_w_avg'].dropna().empty: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#1f77b4', 'fp64': '#ff7f0e'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('N')
        plt.plot(p_df['N'], p_df['power_w_avg'], marker='o', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"CPU Package & DRAM Power Draw vs. Vector Size ({cpu_name})", pad=12)
    plt.xlabel("Vector Length (N Elements)")
    plt.ylabel("Average Power (Watts)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_saxpy_power_vs_n.png"), dpi=300)
    plt.close()

def plot_fig7_energy_efficiency(df, out_dir, cpu_name):
    """Figure 7: Memory Bandwidth Energy Efficiency (GB/s per Watt)."""
    if 'gbps_per_watt' not in df.columns or df['gbps_per_watt'].dropna().empty: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads].copy()
    if sub.empty: return

    plt.figure(figsize=(8.5, 5))
    palette = {'fp32': '#2ca02c', 'fp64': '#d62728'}

    for prec in sorted(sub['precision'].unique()):
        p_df = sub[sub['precision'] == prec].sort_values('N')
        plt.plot(p_df['N'], p_df['gbps_per_watt'], marker='s', linewidth=2.2,
                 label=f"{prec.upper()} ({max_threads}T)", color=palette.get(prec, '#333333'))

    plt.title(f"SAXPY Energy Efficiency (GB/s per Watt) vs. Vector Size ({cpu_name})", pad=12)
    plt.xlabel("Vector Length (N Elements)")
    plt.ylabel("Bandwidth Efficiency (GB/s / Watt)")
    plt.xscale('log', base=2)
    plt.legend(frameon=True)
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_saxpy_energy_efficiency.png"), dpi=300)
    plt.close()

def plot_fig8_fp32_vs_fp64_convergence(df, out_dir, cpu_name):
    """Figure 8: Direct FP32 vs FP64 Bandwidth overlay demonstrating cache vs memory saturation."""
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads]
    f32 = sub[sub['precision'] == 'fp32'].sort_values('N')
    f64 = sub[sub['precision'] == 'fp64'].sort_values('N')
    if f32.empty or f64.empty: return

    plt.figure(figsize=(9, 5.5))
    plt.plot(f32['N'], f32['agg_gbps'], marker='o', linewidth=2.4, color='#1f77b4', label='FP32 Bandwidth (GB/s)')
    plt.plot(f64['N'], f64['agg_gbps'], marker='s', linewidth=2.4, color='#ff7f0e', label='FP64 Bandwidth (GB/s)')

    # Ratio on secondary axis
    merged = pd.merge(f32[['N', 'agg_gbps']], f64[['N', 'agg_gbps']], on='N', suffixes=('_fp32', '_fp64'))
    if not merged.empty:
        ax2 = plt.gca().twinx()
        ratio = merged['agg_gbps_fp32'] / merged['agg_gbps_fp64']
        ax2.plot(merged['N'], ratio, color='#7f7f7f', linestyle='--', linewidth=1.8, marker='x', label='Ratio (FP32 / FP64)')
        ax2.set_ylabel("Bandwidth Ratio (FP32 / FP64)", color='#7f7f7f')
        ax2.set_ylim(0.5, max(2.5, ratio.max() + 0.2))
        ax2.grid(False)

    plt.gca().set_xscale('log', base=2)
    plt.title(f"FP32 vs. FP64 Memory Bandwidth Convergence ({cpu_name})", pad=12)
    plt.gca().set_xlabel("Vector Length (N Elements)")
    plt.gca().set_ylabel("Achieved Bandwidth (GB/s)")
    plt.gca().legend(loc='upper left', frameon=True)
    if not merged.empty: ax2.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_saxpy_fp32_vs_fp64_convergence.png"), dpi=300)
    plt.close()

def plot_fig9_telemetry_freq_temp(df, out_dir, cpu_name):
    """Figure 9: CPU Core Frequency (MHz) and Peak Temperature (°C)."""
    if 'cpu_mhz_avg' not in df.columns or 'temp_c_max' not in df.columns: return
    max_threads = int(df['threads'].max())
    sub = df[df['threads'] == max_threads]
    f32 = sub[sub['precision'] == 'fp32'].sort_values('N')
    if f32.empty: return

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    color1 = '#1f77b4'
    ax1.set_xlabel("Vector Length (N Elements)")
    ax1.set_ylabel("Average CPU Frequency (MHz)", color=color1)
    ax1.plot(f32['N'], f32['cpu_mhz_avg'], color=color1, marker='o', linewidth=2.2, label="Core Frequency (MHz)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel("Peak Core Temperature (°C)", color=color2)
    ax2.plot(f32['N'], f32['temp_c_max'], color=color2, marker='^', linewidth=2.2, linestyle='--', label="Peak Temp (°C)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    fig.suptitle(f"CPU Telemetry: Frequency & Thermal Profile ({cpu_name})", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig9_saxpy_freq_temp.png"), dpi=300)
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # -> results/
    csv_pattern = os.path.join(base_dir, 'data', 'cpu', '*', 'sweep_saxpy_*.csv')
    csv_files = [
        path for path in glob.glob(csv_pattern)
        if any(name in os.path.basename(os.path.dirname(path)).lower() for name in ('cpu', 'intel', 'amd'))
    ]

    if not csv_files:
        print(f"No CPU SAXPY CSV files found matching: {csv_pattern}")
        return

    set_style()
    cleaned_dirs = set()

    for csv_file in csv_files:
        cpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', cpu_name, 'saxpy')
        os.makedirs(out_dir, exist_ok=True)
        if out_dir not in cleaned_dirs:
            for old_plot in glob.glob(os.path.join(out_dir, 'fig*.png')):
                os.remove(old_plot)
            cleaned_dirs.add(out_dir)

        print(f"==> Processing {csv_file} for {cpu_name}...")
        df = load_data(csv_file)
        if df.empty: continue

        plot_fig1_bandwidth_vs_n(df, out_dir, cpu_name)
        plot_fig2_bandwidth_by_threads(df, out_dir, cpu_name)
        plot_fig3_thread_scaling(df, out_dir, cpu_name)
        plot_fig4_speedup_scaling(df, out_dir, cpu_name)
        plot_fig5_heatmap(df, out_dir)
        plot_fig6_power_vs_n(df, out_dir, cpu_name)
        plot_fig7_energy_efficiency(df, out_dir, cpu_name)
        plot_fig8_fp32_vs_fp64_convergence(df, out_dir, cpu_name)
        plot_fig9_telemetry_freq_temp(df, out_dir, cpu_name)
        print(f"    [SUCCESS] 9 publication-grade figures saved to: {out_dir}")

if __name__ == '__main__':
    main()
