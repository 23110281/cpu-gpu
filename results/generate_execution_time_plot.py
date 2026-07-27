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
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--'
    })

def plot_execution_time_comparison(csv_file, out_dir):
    df = pd.read_csv(csv_file)
    ok = df['status'].astype(str).str.startswith('ok')
    # Filter for split mode, 1 GPU dense GEMM
    sub = df[(df['mode'] == 'split') & (df['gpus'] == 1) & (df['density'] == 1.0) & ok].copy()
    if sub.empty:
        return

    # Convert compute_ms to seconds
    sub['compute_sec'] = sub['compute_ms'] / 1000.0
    sub['wall_sec'] = sub['wall_ms'] / 1000.0

    gpu_name = os.path.basename(os.path.dirname(csv_file))

    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {'fp64': '#d95f02', 'fp32': '#7570b3', 'bf16': '#1b9e77', 'fp64t': '#e7298a'}
    markers = {'fp64': 'o', 'fp32': 's', 'bf16': '^', 'fp64t': 'D'}

    precisions = [p for p in ['fp64', 'fp64t', 'fp32', 'bf16'] if p in sub['precision'].unique()]

    for prec in precisions:
        p_df = sub[sub['precision'] == prec].sort_values('M')
        ax.plot(
            p_df['M'], p_df['wall_sec'],
            marker=markers.get(prec, 'o'),
            color=palette.get(prec, '#333333'),
            linewidth=2.5,
            markersize=8,
            label=f"{prec.upper()} (Wall-clock time/batch)"
        )

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    ax.set_xticks(sorted(sub['M'].unique()))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('Execution Time per Run (Seconds, Log Scale)')
    ax.set_title(f'Execution Time Scaling Across Precisions ({gpu_name}, 1 GPU, Split Mode)')

    # Add callout annotations for FP64 vs FP32 vs BF16 at M=32768 if present
    max_m_df = sub[sub['M'] == 32768]
    if not max_m_df.empty:
        for prec in precisions:
            row = max_m_df[max_m_df['precision'] == prec]
            if not row.empty:
                val = row['wall_sec'].values[0]
                if prec == 'fp64':
                    ax.annotate(
                        f'FP64: {val:.1f}s (~{val/60:.1f} min)',
                        xy=(32768, val),
                        xytext=(32768 * 0.5, val * 1.5),
                        arrowprops=dict(facecolor='#d95f02', shrink=0.05, width=1.5, headwidth=8),
                        fontweight='bold', color='#d95f02'
                    )
                elif prec == 'bf16':
                    ax.annotate(
                        f'BF16: {val:.2f}s',
                        xy=(32768, val),
                        xytext=(32768 * 0.6, val * 0.3),
                        arrowprops=dict(facecolor='#1b9e77', shrink=0.05, width=1.5, headwidth=8),
                        fontweight='bold', color='#1b9e77'
                    )

    ax.legend(loc='upper left')
    plt.tight_layout()

    out_file = os.path.join(out_dir, 'fig6_execution_time_comparison.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated execution time comparison plot: {out_file}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pattern = os.path.join(base_dir, 'data', '*', 'sweep_20*.csv')
    csv_files = glob.glob(csv_pattern)
    for csv_file in csv_files:
        gpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', gpu_name, 'gemm')
        os.makedirs(out_dir, exist_ok=True)
        plot_execution_time_comparison(csv_file, out_dir)
