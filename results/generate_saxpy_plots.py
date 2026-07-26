#!/usr/bin/env python3
"""
generate_saxpy_plots.py

Generates scaling plots (GB/s, Power, Utilization, Bandwidth) for SAXPY sweeps.
Matches formatting conventions of other repo plotting scripts.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'figure.titlesize': 16, 'font.family': 'serif',
        'axes.grid': True, 'grid.alpha': 0.5, 'grid.linestyle': '--',
    })

def plot_gbps_heatmap(df, out_dir):
    """Heatmap: rows=mode, cols=precision, cells=agg_gbps by (N, gpus)."""
    modes = df['mode'].unique()
    precisions = df['precision'].unique()
    sizes = df['N'].unique()
    
    fig, axes = plt.subplots(len(modes), len(precisions), 
                             figsize=(max(5, 0.7 * len(sizes)) * len(precisions), 5 * len(modes)))
    if len(modes) == 1 and len(precisions) == 1: axes = np.array([[axes]])
    elif len(modes) == 1: axes = np.array([axes])
    elif len(precisions) == 1: axes = np.array([[ax] for ax in axes])
        
    fig.suptitle('SAXPY Aggregate Bandwidth (GB/s)', y=0.98)
    
    for i, mode in enumerate(modes):
        for j, prec in enumerate(precisions):
            ax = axes[i, j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                ax.set_visible(False)
                continue
            
            pivot = sub.pivot_table(index='gpus', columns='N', values='agg_gbps', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako", ax=ax, cbar_kws={'label': 'GB/s'})
            ax.set_title(f'{mode.capitalize()} | {prec.upper()}')
            ax.set_ylabel('GPUs' if j == 0 else '')
            ax.set_xlabel('Vector Size (N)' if i == len(modes)-1 else '')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, 'fig1_gbps_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gbps_vs_size(df, out_dir):
    """Line plot: achieved GB/s vs vector size, one line per gpu count."""
    # INTENTIONAL EXCEPTION: fp32 and fp64 are plotted on the same axes to show they converge to the same bandwidth ceiling.
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions),
                                  figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'SAXPY Bandwidth vs Vector Size ({mode.capitalize()})', y=0.98)
        axes = np.atleast_1d(axes)
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].sort_values('N')
                ax.plot(g_df['N'], g_df['agg_gbps'], marker='o', color=gpu_palette[k],
                        label=f'{gpus} GPUs' if j == 0 else "")
            
            # Theoretical peak for a single A100 GPU
            ax.axhline(y=1555, color='black', linestyle=':', alpha=0.5,
                       label='A100 HBM peak (~1555 GB/s/GPU)' if j == 0 else "")
            
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['N'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel('GB/s' if j == 0 else '')
            ax.set_xlabel('Vector Size (N)')
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig2_gbps_vs_size_{mode}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_power_util(df, out_dir):
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(2, len(mode_precisions), figsize=(6 * len(mode_precisions), 8), sharex=True)
        fig.suptitle(f'Power and Utilization vs Vector Size ({mode.capitalize()})', y=0.98)
        if len(mode_precisions) == 1: axes = axes.reshape(2, 1)
        
        for j, prec in enumerate(mode_precisions):
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            # Compute total node power (W) by multiplying avg power per GPU by num GPUs
            sub = sub.copy()
            sub['total_power_w'] = sub['power_w_avg'] * sub['gpus']
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            
            # Row 0: Power
            ax_pwr = axes[0, j]
            sns.lineplot(data=sub, x='N', y='total_power_w', hue='gpus', marker='o', palette=gpu_palette, linewidth=2.5, ax=ax_pwr)
            ax_pwr.set_xscale('log', base=2)
            ax_pwr.set_xticks(sorted(sub['N'].unique()))
            ax_pwr.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_pwr.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_pwr.set_title(f'{prec.upper()}')
            ax_pwr.set_ylabel('Total Node Power (W)' if j == 0 else '')
            ax_pwr.set_xlabel('')
            if j != len(mode_precisions) - 1:
                ax_pwr.get_legend().remove()
            else:
                ax_pwr.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')
                
            # Row 1: Utilization
            ax_util = axes[1, j]
            sns.lineplot(data=sub, x='N', y='util_pct_avg', hue='gpus', marker='s', palette=gpu_palette, linewidth=2.5, ax=ax_util)
            ax_util.set_xscale('log', base=2)
            ax_util.set_xticks(sorted(sub['N'].unique()))
            ax_util.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_util.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_util.set_ylabel('Avg GPU Utilization (%)' if j == 0 else '')
            ax_util.set_xlabel('Vector Size (N)')
            ax_util.set_ylim(0, 105)
            if j != len(mode_precisions) - 1:
                ax_util.get_legend().remove()
            else:
                ax_util.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig3_power_util_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_bandwidth(df, out_dir):
    """Plot PCIe Bandwidth Scaling for SAXPY transfers"""
    modes = df['mode'].unique()
    
    # Calculate bytes transferred
    # H2D: x and y vectors (each size N). D2H: y vector back
    # elem_bytes = 4 for fp32, 8 for fp64
    df = df.copy()
    df['elem_bytes'] = df['precision'].map({'fp32': 4, 'fp64': 8})
    df['h2d_bytes'] = df['N'] * 2 * df['elem_bytes']
    df['d2h_bytes'] = df['N'] * df['elem_bytes']
    
    # ms to seconds: / 1000. bytes to GB: / 1e9. combined: * 1e-6
    df['h2d_gbps'] = (df['h2d_bytes'] / (df['h2d_ms'] + 1e-9)) * 1e-6
    df['d2h_gbps'] = (df['d2h_bytes'] / (df['d2h_ms'] + 1e-9)) * 1e-6
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions), figsize=(5 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'PCIe Bandwidth (GB/s): H2D (Solid) vs D2H (Dashed) [{mode.capitalize()}]', y=0.98)
        
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].sort_values('N')
                color = gpu_palette[k]
                
                ax.plot(g_df['N'], g_df['h2d_gbps'], linestyle='-', marker='^', color=color, 
                        label=f'{gpus} GPUs (H2D)' if j==0 else "")
                ax.plot(g_df['N'], g_df['d2h_gbps'], linestyle='--', marker='v', color=color, 
                        label=f'{gpus} GPUs (D2H)' if j==0 else "")
                
            ax.axhline(y=24, color='black', linestyle=':', alpha=0.5, label='PCIe Gen4 (~24 GB/s)' if j==0 else "")
            
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['N'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            if j == 0: ax.set_ylabel('GB/s')
            ax.set_xlabel('Vector Size (N)')
            
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig4_pcie_bandwidth_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_time_distribution(df, out_dir):
    """Stacked bar chart for H2D/Compute/D2H percentages."""
    modes = df['mode'].unique()
    all_gpus = sorted(df['gpus'].unique())
    
    colors = sns.color_palette("muted")
    c_comp, c_h2d, c_d2h = colors[2], colors[0], colors[3]
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(len(all_gpus), len(mode_precisions), figsize=(6 * len(mode_precisions), 4 * len(all_gpus)))
        fig.suptitle(f'SAXPY Time Distribution Percentage vs Vector Size ({mode.capitalize()})', y=0.98)
        
        if len(all_gpus) == 1 and len(mode_precisions) == 1: axes = np.array([[axes]])
        elif len(all_gpus) == 1: axes = np.array([axes])
        elif len(mode_precisions) == 1: axes = np.array([[ax] for ax in axes])
            
        for i, gpus in enumerate(all_gpus):
            for j, prec in enumerate(mode_precisions):
                ax = axes[i, j]
                prec_df = df[(df['mode'] == mode) & (df['precision'] == prec) & (df['gpus'] == gpus)].sort_values('N')
                
                if prec_df.empty:
                    ax.set_visible(False)
                    continue
                
                total_time = prec_df['compute_ms'] + prec_df['h2d_ms'] + prec_df['d2h_ms']
                
                x = prec_df['N'].astype(str)
                y1 = (prec_df['compute_ms'] / total_time) * 100
                y2 = (prec_df['h2d_ms'] / total_time) * 100
                y3 = (prec_df['d2h_ms'] / total_time) * 100
                
                ax.bar(x, y1, color=c_comp, label='Compute' if i==0 and j==0 else "")
                ax.bar(x, y2, bottom=y1, color=c_h2d, label='Host-to-Device' if i==0 and j==0 else "")
                ax.bar(x, y3, bottom=y1+y2, color=c_d2h, label='Device-to-Host' if i==0 and j==0 else "")
                
                ax.set_title(f'{gpus} GPUs | {prec.upper()}')
                ax.set_ylim(0, 100)
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
                
                if j == 0: ax.set_ylabel('% of Total Time')
                if i == len(all_gpus) - 1: ax.set_xlabel('Vector Size (N)')
                
                if i == 0 and j == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.15), ncol=1)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig5_time_distribution_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
    else:
        csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'sweep_saxpy' in f]
        if not csv_files:
            print("No sweep_saxpy CSV files found in", base_dir)
            sys.exit(1)
        csv_files.sort(reverse=True)
        csv_file = os.path.join(base_dir, csv_files[0])

    print(f"Loading SAXPY data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        sys.exit(1)

    df = df[df["status"].astype(str).str.startswith("ok")]

    if df.empty:
        print("No valid 'ok' data in CSV to plot.")
        sys.exit(1)
        
    out_dir = os.path.join(base_dir, 'plots', 'saxpy')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    set_style()
    plot_gbps_heatmap(df, out_dir)
    plot_gbps_vs_size(df, out_dir)
    plot_power_util(df, out_dir)
    plot_bandwidth(df, out_dir)
    plot_time_distribution(df, out_dir)
    
    print(f"SAXPY plots written to {out_dir}")

if __name__ == "__main__":
    main()
