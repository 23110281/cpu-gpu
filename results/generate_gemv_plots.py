import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as ticker

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'figure.titlesize': 16, 'font.family': 'serif',
        'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--',
    })

def load_gemv_data(filepath, engine="dense"):
    """engine: 'dense' or 'sparse'. GEMV's CSV has no K column and its
    headline metric is bandwidth (agg_gbps/pct_peak_bw), not TFLOPS --
    this is why it's a separate loader from generate_plots.py's load_data(),
    not a reused one (see design spec 2026-07-11-gpu-gemv-bench-design.md §5.11)."""
    df = pd.read_csv(filepath)
    ok = df['status'].astype(str).str.startswith('ok')
    df = df[(df['engine'] == engine) & ok].copy()
    return df

def plot_gbps_heatmap(df, out_dir, filename_prefix):
    """Heatmap: rows=mode, cols=precision, cells=agg_gbps by (S, gpus)."""
    if df.empty:
        return
    modes = df['mode'].unique()
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(len(modes), len(precisions),
                              figsize=(5 * len(precisions), 5 * len(modes)))
    fig.suptitle('GEMV Achieved Bandwidth Heatmap (Rows: Mode, Cols: Precision)', y=0.98)
    axes = np.array(axes).reshape(len(modes), len(precisions))
    for i, mode in enumerate(modes):
        for j, prec in enumerate(precisions):
            ax = axes[i, j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = sub.pivot_table(index='S', columns='gpus', values='agg_gbps')
            pivot = pivot.sort_index(ascending=False)
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako", ax=ax,
                        cbar_kws={'label': 'GB/s'} if j == len(precisions) - 1 else None)
            ax.set_title(f'{mode.capitalize()} | {prec.upper()}')
            ax.set_ylabel('Matrix Size (S)' if j == 0 else '')
            ax.set_xlabel('Number of GPUs' if i == len(modes) - 1 else '')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f'{filename_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tflops_heatmap(df, out_dir, filename_prefix):
    """Heatmap: rows=mode, cols=precision, cells=agg_tflops by (S, gpus).

    Secondary/cross-reference metric only -- GEMV is bandwidth-bound at every
    size (see plot_gbps_heatmap), so this never becomes the headline plot the
    way the GEMM benchmark's fig1a/fig1b TFLOPS heatmaps are. Uses the same
    'viridis' cmap as generate_plots.py's plot_heatmap to signal "compute
    metric" vs plot_gbps_heatmap's 'mako' "bandwidth metric".
    """
    if df.empty:
        return
    modes = df['mode'].unique()
    precisions = sorted(df['precision'].unique())
    fig, axes = plt.subplots(len(modes), len(precisions),
                              figsize=(5 * len(precisions), 5 * len(modes)))
    fig.suptitle('GEMV Achieved TFLOPS Heatmap (Rows: Mode, Cols: Precision)', y=0.98)
    axes = np.array(axes).reshape(len(modes), len(precisions))
    for i, mode in enumerate(modes):
        for j, prec in enumerate(precisions):
            ax = axes[i, j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = sub.pivot_table(index='S', columns='gpus', values='agg_tflops')
            pivot = pivot.sort_index(ascending=False)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                        cbar_kws={'label': 'TFLOPS'} if j == len(precisions) - 1 else None)
            ax.set_title(f'{mode.capitalize()} | {prec.upper()}')
            ax.set_ylabel('Matrix Size (S)' if j == 0 else '')
            ax.set_xlabel('Number of GPUs' if i == len(modes) - 1 else '')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f'{filename_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gbps_vs_size(df, out_dir, suffix=''):
    """Line plot: achieved GB/s and % of peak vs matrix size, one line per gpu count."""
    if df.empty:
        return
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions),
                                  figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'GEMV Bandwidth vs Size ({mode.capitalize()})', y=0.98)
        axes = np.atleast_1d(axes)
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                continue
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].sort_values('S')
                ax.plot(g_df['S'], g_df['agg_gbps'], marker='o', color=gpu_palette[k],
                        label=f'{gpus} GPUs' if j == 0 else "")
            ax.axhline(y=1555, color='black', linestyle=':', alpha=0.5,
                       label='A100-SXM4-40GB HBM peak (~1555 GB/s)' if j == 0 else "")
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['S'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel('GB/s' if j == 0 else '')
            ax.set_xlabel('Matrix Size (S)')
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig2_gbps_vs_size_{mode}{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_power_util(df, out_dir, suffix=''):
    """Power and utilization vs size, mirroring generate_plots.py's plot_power_and_utilization
    but without the TFLOPS/W efficiency row (GEMV's headline is bandwidth, not TFLOPS)."""
    if df.empty:
        return
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(2, len(mode_precisions),
                                  figsize=(8 * len(mode_precisions), 12), sharex=True)
        fig.suptitle(f'GEMV Power and GPU Utilization ({mode.capitalize()})', y=0.98)
        axes = np.array(axes).reshape(2, len(mode_precisions))
        gpu_palette = sns.color_palette("Set1", len(df['gpus'].unique()))
        for j, prec in enumerate(mode_precisions):
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                continue
            ax_pwr = axes[0, j]
            sns.lineplot(data=sub, x='S', y='power_w_avg', hue='gpus', marker='o',
                         palette=gpu_palette, linewidth=2.5, ax=ax_pwr)
            ax_pwr.set_xscale('log', base=2)
            ax_pwr.set_title(f'{prec.upper()}')
            ax_pwr.set_ylabel('Power (W)' if j == 0 else '')
            ax_util = axes[1, j]
            sns.lineplot(data=sub, x='S', y='util_pct_avg', hue='gpus', marker='s',
                         palette=gpu_palette, linewidth=2.5, ax=ax_util)
            ax_util.set_xscale('log', base=2)
            ax_util.set_ylabel('Avg GPU Utilization (%)' if j == 0 else '')
            ax_util.set_xlabel('Matrix Size (S)')
            ax_util.set_ylim(0, 105)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig3_power_util_{mode}{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'sweep_gemv' in f]
    if not csv_files:
        print("No sweep_gemv CSV files found.")
        return
    csv_files.sort(reverse=True)
    csv_file = os.path.join(base_dir, csv_files[0])
    out_dir = os.path.join(base_dir, 'plots', 'gemv')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    set_style()
    print(f"Loading dense GEMV data from {csv_file}...")
    ddf = load_gemv_data(csv_file, engine="dense")
    if ddf.empty:
        print("No dense rows found.")
    else:
        plot_gbps_heatmap(ddf, out_dir, 'fig1_gbps_heatmap')
        plot_gbps_vs_size(ddf, out_dir)
        plot_power_util(ddf, out_dir)
        plot_tflops_heatmap(ddf, out_dir, 'fig4_tflops_heatmap')
        print(f"Dense GEMV plots written to {out_dir}")

    print(f"Loading sparse SpMV data from {csv_file}...")
    sdf = load_gemv_data(csv_file, engine="sparse")
    if sdf.empty:
        print("No sparse rows found -- skipping sparse plots.")
    else:
        plot_gbps_heatmap(sdf, out_dir, 'fig1_gbps_heatmap_sparse')
        plot_gbps_vs_size(sdf, out_dir, suffix='_sparse')
        plot_power_util(sdf, out_dir, suffix='_sparse')
        plot_tflops_heatmap(sdf, out_dir, 'fig4_tflops_heatmap_sparse')
        print(f"Sparse SpMV plots written to {out_dir}")

if __name__ == '__main__':
    main()
