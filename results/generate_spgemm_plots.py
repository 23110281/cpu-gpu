#!/usr/bin/env python3
"""
generate_spgemm_plots.py

Generates scaling plots (TFLOP/s, GB/s, Density Scaling, Timing Distribution) for SpGEMM sweeps.
Matches formatting conventions of other repo plotting scripts while addressing unique SpGEMM phases.
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

def plot_metric_vs_density(df, out_dir, metric='agg_tflops', ylabel='TFLOP/s', filename_prefix='fig1_tflops_vs_density'):
    """Line plot: achieved metric vs Density, one line per gpu count. For a fixed large matrix size."""
    if df.empty: return
    modes = df['mode'].unique()
    
    # Pick the largest matrix size to show density scaling
    max_S = df['S'].max()
    sub_df = df[df['S'] == max_S]
    if sub_df.empty: return

    for mode in modes:
        mode_precisions = sorted(sub_df[sub_df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions),
                                  figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'SpGEMM {ylabel} vs Density at S={max_S} ({mode.capitalize()})', y=0.98)
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            # Use density_a (assuming symmetric density sweeps)
            sub = sub_df[(sub_df['mode'] == mode) & (sub_df['precision'] == prec)]
            if sub.empty: continue
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                # Group by density_a just in case
                g_df = sub[sub['gpus'] == gpus].groupby('density_a', as_index=False)[metric].mean().sort_values('density_a')
                ax.plot(g_df['density_a'], g_df[metric], marker='o', color=gpu_palette[k],
                        label=f'{gpus} GPUs' if j == 0 else "")
            
            ax.set_xscale('log', base=10)
            ax.set_xticks(sorted(sub['density_a'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel(ylabel if j == 0 else '')
            ax.set_xlabel('Matrix Density')
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_metric_vs_size(df, out_dir, metric='agg_gbps', ylabel='GB/s', filename_prefix='fig5_gbps_vs_size'):
    """Line plot: achieved metric vs Matrix Size, one line per density. For fixed 1 GPU."""
    if df.empty: return
    modes = df['mode'].unique()
    
    # We fix GPU count to 1 to clearly see how density scales over sizes
    sub_df = df[df['gpus'] == 1]
    if sub_df.empty: return
    
    for mode in modes:
        mode_precisions = sorted(sub_df[sub_df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions), figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'SpGEMM {ylabel} vs Matrix Size on 1 GPU ({mode.capitalize()})', y=0.98)
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = sub_df[(sub_df['mode'] == mode) & (sub_df['precision'] == prec)]
            if sub.empty: continue
            
            densities = sorted(sub['density_a'].unique())
            density_palette = sns.color_palette("plasma", len(densities))
            
            for k, dens in enumerate(densities):
                g_df = sub[sub['density_a'] == dens].groupby('S', as_index=False)[metric].mean().sort_values('S')
                ax.plot(g_df['S'], g_df[metric], marker='s', color=density_palette[k],
                        label=f'Density: {dens}' if j == 0 else "")
            
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['S'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel(ylabel if j == 0 else '')
            ax.set_xlabel('Matrix Size (S)')
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_compute_vs_effective_density(df, out_dir, filename_prefix='fig4_comp_eff_vs_density'):
    """Plot Compute vs Effective TFLOPS vs Density"""
    modes = df['mode'].unique()
    
    # Pick the largest matrix size to show density scaling
    max_S = df['S'].max()
    sub_df = df[df['S'] == max_S]
    if sub_df.empty: return

    for mode in modes:
        mode_precisions = sorted(sub_df[sub_df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions), figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'SpGEMM Compute vs Effective TFLOPS vs Density at S={max_S} ({mode.capitalize()})', y=0.98)
        
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = sub_df[(sub_df['mode'] == mode) & (sub_df['precision'] == prec)]
            if sub.empty: continue
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].groupby('density_a', as_index=False).mean(numeric_only=True).sort_values('density_a')
                color = gpu_palette[k]
                
                ax.plot(g_df['density_a'], g_df['agg_tflops'], linestyle='-', marker='o', color=color, 
                        label=f'{gpus} GPUs (Compute)' if j==0 else "")
                ax.plot(g_df['density_a'], g_df['eff_tflops'], linestyle='--', marker='s', color=color, 
                        label=f'{gpus} GPUs (Effective)' if j==0 else "")
                
            ax.set_xscale('log', base=10)
            ax.set_xticks(sorted(sub['density_a'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel('TFLOP/s' if j == 0 else '')
            ax.set_xlabel('Density')
            
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_timing_distribution(df, out_dir):
    """100% Stacked Bar Chart for Time Breakdown (H2D, Symbolic, Compute, D2H) vs Density for largest size"""
    if df.empty: return
    modes = df['mode'].unique()
    
    # Pick largest matrix size
    max_S = df['S'].max()
    sub_df = df[df['S'] == max_S]
    if sub_df.empty: return
    
    all_gpus = sorted(sub_df['gpus'].unique())
    
    colors = sns.color_palette("muted")
    c_comp, c_h2d, c_d2h, c_sym = colors[2], colors[0], colors[3], colors[1]
    
    for mode in modes:
        mode_precisions = sorted(sub_df[sub_df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(len(all_gpus), len(mode_precisions), figsize=(6 * len(mode_precisions), 4 * len(all_gpus)))
        fig.suptitle(f'SpGEMM Time Distribution vs Density at S={max_S} ({mode.capitalize()})', y=0.98)
        
        axes = np.array(axes).reshape(len(all_gpus), len(mode_precisions))
        
        for i, gpus in enumerate(all_gpus):
            for j, prec in enumerate(mode_precisions):
                ax = axes[i, j]
                prec_df = sub_df[(sub_df['mode'] == mode) & (sub_df['precision'] == prec) & (sub_df['gpus'] == gpus)].sort_values('density_a')
                
                if prec_df.empty:
                    ax.set_visible(False)
                    continue
                
                # Calculate percentages based on raw ms values
                # H2D/D2H are totals, but compute_ms is per-iteration. symbolic_ms is total (one-time).
                # To compare them fairly as latencies, we need to scale compute_ms by iters or just use total e2e.
                # Actually, the user script logs `h2d_ms`, `d2h_ms`, `compute_ms` (avg per iter), `symbolic_ms` (total one-time).
                # The time distribution compares phase times. Since symbolic is one-time, we compare total times for a workload:
                # Total H2D, Total Symbolic, Total Compute (compute_ms * iters), Total D2H.
                
                prec_df = prec_df.copy()
                total_compute = prec_df['compute_ms'] * prec_df['iters']
                total_time = prec_df['h2d_ms'] + prec_df['symbolic_ms'] + total_compute + prec_df['d2h_ms']
                
                pct_h2d = 100 * prec_df['h2d_ms'] / total_time
                pct_sym = 100 * prec_df['symbolic_ms'] / total_time
                pct_comp = 100 * total_compute / total_time
                pct_d2h = 100 * prec_df['d2h_ms'] / total_time
                
                x = prec_df['density_a'].astype(str)
                
                ax.bar(x, pct_comp, color=c_comp, label='Compute' if i==0 and j==0 else "")
                ax.bar(x, pct_sym, bottom=pct_comp, color=c_sym, label='Symbolic (Phase 1)' if i==0 and j==0 else "")
                ax.bar(x, pct_h2d, bottom=pct_comp+pct_sym, color=c_h2d, label='Host-to-Device' if i==0 and j==0 else "")
                ax.bar(x, pct_d2h, bottom=pct_comp+pct_sym+pct_h2d, color=c_d2h, label='Device-to-Host' if i==0 and j==0 else "")
                
                ax.set_title(f'{gpus} GPUs | {prec.upper()}')
                ax.set_ylim(0, 100)
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
                
                if j == 0: ax.set_ylabel('% of Workload Time')
                if i == len(all_gpus) - 1: ax.set_xlabel('Matrix Density')
                
                if i == 0 and j == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.15), ncol=1)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig3_time_distribution_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_power_util(df, out_dir, filename_prefix='fig6_power_util'):
    """Line plot of Power and Utilization vs Size (averaged over densities)"""
    if df.empty: return
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(2, len(mode_precisions), figsize=(6 * len(mode_precisions), 8), sharex=True)
        fig.suptitle(f'SpGEMM Power and Utilization vs Matrix Size ({mode.capitalize()})', y=0.98)
        if len(mode_precisions) == 1: axes = axes.reshape(2, 1)
        
        for j, prec in enumerate(mode_precisions):
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            sub = sub.copy()
            sub['total_power_w'] = sub['power_w_avg'] * sub['gpus']
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            
            # Row 0: Power
            ax_pwr = axes[0, j]
            sns.lineplot(data=sub, x='S', y='total_power_w', hue='gpus', marker='o', palette=gpu_palette, linewidth=2.5, ax=ax_pwr, errorbar=None)
            ax_pwr.set_xscale('log', base=2)
            ax_pwr.set_xticks(sorted(sub['S'].unique()))
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
            sns.lineplot(data=sub, x='S', y='util_pct_avg', hue='gpus', marker='s', palette=gpu_palette, linewidth=2.5, ax=ax_util, errorbar=None)
            ax_util.set_xscale('log', base=2)
            ax_util.set_xticks(sorted(sub['S'].unique()))
            ax_util.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_util.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_util.set_ylabel('Avg GPU Utilization (%)' if j == 0 else '')
            ax_util.set_xlabel('Matrix Size (S)')
            ax_util.set_ylim(0, 105)
            if j != len(mode_precisions) - 1:
                ax_util.get_legend().remove()
            else:
                ax_util.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
    else:
        csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'sweep_spgemm' in f]
        if not csv_files:
            print("No sweep_spgemm CSV files found in", base_dir)
            sys.exit(1)
        csv_files.sort(reverse=True)
        csv_file = os.path.join(base_dir, csv_files[0])

    print(f"Loading SpGEMM data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        sys.exit(1)

    df = df[df["status"].astype(str).str.startswith("ok")].copy()
    
    # Calculate eff_tflops exactly like GEMM does
    df['eff_tflops'] = (df['flops_theoretical'] / 1e12) / (df['e2e_ms'] / 1000.0)

    if df.empty:
        print("No valid 'ok' data in CSV to plot.")
        sys.exit(1)
        
    out_dir = os.path.join(base_dir, 'plots', 'spgemm')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    set_style()
    plot_metric_vs_density(df, out_dir, metric='agg_tflops', ylabel='TFLOP/s', filename_prefix='fig1_tflops_vs_density')
    plot_metric_vs_size(df, out_dir, metric='agg_tflops', ylabel='TFLOP/s', filename_prefix='fig2_tflops_vs_size')
    plot_timing_distribution(df, out_dir)
    plot_compute_vs_effective_density(df, out_dir)

    plot_metric_vs_size(df, out_dir, metric='agg_gbps', ylabel='GB/s', filename_prefix='fig5_gbps_vs_size')
    plot_power_util(df, out_dir, filename_prefix='fig6_power_util')
    
    print(f"SpGEMM plots written to {out_dir}")

if __name__ == "__main__":
    main()
