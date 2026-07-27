#!/usr/bin/env python3
"""
generate_spgemm_plots.py

Generates scaling plots (TFLOP/s, GB/s, Density Scaling, Timing Distribution) for SpGEMM sweeps.
Matches formatting conventions of other repo plotting scripts while addressing unique SpGEMM phases.
"""

import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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


def compute_tflops_and_status(df):
    df['status'] = df['status'].fillna('').astype(str)
    if 'flops_theoretical' in df.columns and 'wall_ms' in df.columns:
        df['recomputed_tflops'] = (df['flops_theoretical'] / (df['wall_ms'] / 1000.0)) / 1e12
        df['recomputed_tflops'] = df['recomputed_tflops'].fillna(0.0)
    else:
        df['recomputed_tflops'] = 0.0

    def get_cat(row):
        st = row['status']
        if st.startswith('ok'):
            if round(row['recomputed_tflops'], 2) == 0.0:
                return 'ok-underflow'
            return 'ok'
        if st == 'error':
            return 'error'
        if st == 'oom-skip':
            return 'oom-skip'
        return 'error'

    df['status_cat'] = df.apply(get_cat, axis=1)

    if 'flops_theoretical' in df.columns:
        df['agg_tflops'] = np.where(df['status'].str.startswith('ok'), df['recomputed_tflops'], 0.0)
        df['eff_tflops'] = np.where(df['status'].str.startswith('ok'), (df['flops_theoretical'] / 1e12) / (df['e2e_ms'] / 1000.0), 0.0)
    return df



def plot_status_heatmap(df, out_dir):
    color_dict = {'ok': '#2ca02c', 'ok-underflow': '#bcbd22', 'error': '#d62728', 'oom-skip': '#1f77b4'}
    modes = df['mode'].unique() if 'mode' in df.columns else ['unknown']
    for mode in modes:
        sub = df[df['mode'] == mode] if 'mode' in df.columns else df
        gpus_list = sorted(sub['gpus'].unique())
        precisions = sorted(sub['precision'].unique())

        fig, axes = plt.subplots(len(gpus_list), len(precisions),
                                 figsize=(5 * len(precisions), 4 * len(gpus_list)), squeeze=False)
        fig.suptitle(f'SpGEMM Pass/Fail Heatmap ({mode.capitalize()})', y=0.98)

        for i, gpus in enumerate(gpus_list):
            for j, prec in enumerate(precisions):
                ax = axes[i, j]
                sub2 = sub[(sub['gpus'] == gpus) & (sub['precision'] == prec)]
                if sub2.empty:
                    ax.set_visible(False)
                    continue

                pivot = sub2.pivot_table(index='S', columns='density_a', values='status_cat', aggfunc='first')

                cat_to_num = {'ok': 1, 'ok-underflow': 2, 'error': 3, 'oom-skip': 4}
                num_pivot = pivot.replace(cat_to_num).fillna(0).astype(float)

                cmap = mcolors.ListedColormap(['#ffffff', '#2ca02c', '#bcbd22', '#d62728', '#1f77b4'])
                bounds = [0, 1, 2, 3, 4, 5]
                norm = mcolors.BoundaryNorm(bounds, cmap.N)

                sns.heatmap(num_pivot, cmap=cmap, norm=norm, ax=ax, cbar=False,
                            annot=False, linewidths=.5, linecolor='lightgray')

                ax.set_title(f'{gpus} GPUs | {prec.upper()}')
                ax.set_ylabel('Matrix Size (S)' if j == 0 else '')
                ax.set_xlabel('Density' if i == len(gpus_list) - 1 else '')
                ax.invert_yaxis()

                if i == 0 and j == 0:
                    legend_elements = [Patch(facecolor=color_dict[k], edgecolor='gray', label=k)
                                       for k in color_dict.keys()]
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig0_status_heatmap_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_density_frontier(df, out_dir):
    modes = df['mode'].unique() if 'mode' in df.columns else ['unknown']
    for mode in modes:
        sub = df[df['mode'] == mode] if 'mode' in df.columns else df
        gpus_list = sorted(sub['gpus'].unique())
        precisions = sorted(sub['precision'].unique())

        fig, axes = plt.subplots(len(gpus_list), len(precisions),
                                 figsize=(5 * len(precisions), 4 * len(gpus_list)), squeeze=False)
        fig.suptitle(f'SpGEMM Max Successful Density Frontier ({mode.capitalize()})', y=0.98)

        for i, gpus in enumerate(gpus_list):
            for j, prec in enumerate(precisions):
                ax = axes[i, j]
                sub2 = sub[(sub['gpus'] == gpus) & (sub['precision'] == prec) &
                           (sub['status_cat'].isin(['ok', 'ok-underflow']))]
                if sub2.empty:
                    ax.set_visible(False)
                    continue

                frontier = sub2.groupby('S')['density_a'].max().reset_index().sort_values('S')

                ax.plot(frontier['S'].astype(str), frontier['density_a'],
                        marker='o', linestyle='-', color='#1f77b4')
                ax.fill_between(frontier['S'].astype(str), frontier['density_a'],
                                alpha=0.2, color='#1f77b4')

                ax.set_title(f'{gpus} GPUs | {prec.upper()}')
                ax.set_ylim(bottom=0)
                if j == 0:
                    ax.set_ylabel('Max Density')
                if i == len(gpus_list) - 1:
                    ax.set_xlabel('Matrix Size (S)')
                    ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig0b_density_frontier_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_metric_vs_density(df, out_dir, metric='agg_tflops', ylabel='TFLOP/s',
                           filename_prefix='fig1_tflops_vs_density'):
    if df.empty:
        return
    modes = df['mode'].unique()

    for mode in modes:
        sub = df[df['mode'] == mode]
        if sub.empty:
            continue

        mode_precisions = sorted(sub['precision'].unique())
        sizes = sorted(sub['S'].unique())

        fig, axes = plt.subplots(len(sizes), len(mode_precisions),
                                 figsize=(6 * len(mode_precisions), 4 * len(sizes)),
                                 sharex=True, sharey='col')
        fig.suptitle(f'SpGEMM {ylabel} vs Density across Size ({mode.capitalize()})', y=0.98)

        if len(sizes) == 1 and len(mode_precisions) == 1:
            axes = np.array([[axes]])
        elif len(sizes) == 1:
            axes = np.array([axes])
        elif len(mode_precisions) == 1:
            axes = np.array([[ax] for ax in axes])

        for i, s in enumerate(sizes):
            for j, prec in enumerate(mode_precisions):
                ax = axes[i, j]
                pan_df = sub[(sub['precision'] == prec) & (sub['S'] == s)]
                if pan_df.empty:
                    ax.set_visible(False)
                    continue

                gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
                for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                    g_df = (pan_df[pan_df['gpus'] == gpus]
                            .groupby('density_a', as_index=False)[metric].mean()
                            .sort_values('density_a'))
                    if not g_df.empty:
                        ax.plot(g_df['density_a'], g_df[metric], marker='o',
                                color=gpu_palette[k],
                                label=f'{gpus} GPUs' if i == 0 and j == 0 else "")

                ax.set_xscale('log', base=10)
                ax.set_xticks(sorted(sub['density_a'].unique()))
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
                ax.set_title(f'S={s}, {prec.upper()}' if i == 0 else f'S={s}')
                ax.set_ylabel(ylabel if j == 0 else '')
                if i == len(sizes) - 1:
                    ax.set_xlabel('Matrix Density')
                if i == 0 and j == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_metric_vs_size(df, out_dir, metric='agg_gbps', ylabel='GB/s',
                        filename_prefix='fig5_gbps_vs_size'):
    """Small Multiples Line plot: achieved metric vs Matrix Size, to avoid spaghetti plot."""
    if df.empty:
        return
    modes = df['mode'].unique()

    sub_df = df[df['gpus'] == 1]
    if sub_df.empty:
        return

    for mode in modes:
        mode_precisions = sorted(sub_df[sub_df['mode'] == mode]['precision'].unique())
        
        # Get all densities for this mode to determine rows
        densities = sorted(sub_df[sub_df['mode'] == mode]['density_a'].unique())
        if not densities:
            continue
            
        fig, axes = plt.subplots(len(densities), len(mode_precisions),
                                 figsize=(6 * len(mode_precisions), 3 * len(densities)), 
                                 sharex=True, sharey='col')
        fig.suptitle(f'SpGEMM {ylabel} vs Matrix Size on 1 GPU ({mode.capitalize()})', y=0.98)
        
        # Ensure 2D array
        axes = np.array(axes).reshape(len(densities), len(mode_precisions))

        density_palette = sns.color_palette("plasma", len(densities))

        for j, prec in enumerate(mode_precisions):
            sub = sub_df[(sub_df['mode'] == mode) & (sub_df['precision'] == prec)]
            
            for i, target_dens in enumerate(densities):
                ax = axes[i, j]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                
                # First, plot all other densities in gray (Background context)
                for k, bg_dens in enumerate(densities):
                    g_df = (sub[sub['density_a'] == bg_dens]
                            .groupby('S', as_index=False)[metric].mean()
                            .sort_values('S'))
                    
                    if bg_dens == target_dens:
                        # Highlight the target density
                        ax.plot(g_df['S'], g_df[metric], marker='s', color=density_palette[k], 
                                linewidth=2.5, zorder=5)
                    else:
                        # Background gray
                        ax.plot(g_df['S'], g_df[metric], marker='', color='lightgray', 
                                linewidth=1.5, alpha=0.5, zorder=1)

                ax.set_xscale('log', base=2)
                ax.set_xticks(sorted(sub['S'].unique()))
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.tick_params(axis='x', rotation=45, labelbottom=(i == len(densities) - 1))
                
                if i == 0:
                    ax.set_title(f'{prec.upper()}')
                if j == 0:
                    ax.set_ylabel(f'D: {target_dens}\n{ylabel}', rotation=0, labelpad=40, ha='center', va='center')
                if i == len(densities) - 1:
                    ax.set_xlabel('Matrix Size (S)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_compute_vs_effective_density(df, out_dir, filename_prefix='fig4_comp_eff_vs_density'):
    if df.empty:
        return
    modes = df['mode'].unique()

    for mode in modes:
        sub = df[df['mode'] == mode]
        if sub.empty:
            continue

        mode_precisions = sorted(sub['precision'].unique())
        sizes = sorted(sub['S'].unique())

        fig, axes = plt.subplots(len(sizes), len(mode_precisions),
                                 figsize=(6 * len(mode_precisions), 4 * len(sizes)),
                                 sharex=True, sharey='col')
        fig.suptitle(f'SpGEMM Compute vs Effective TFLOPS vs Density ({mode.capitalize()})', y=0.98)

        if len(sizes) == 1 and len(mode_precisions) == 1:
            axes = np.array([[axes]])
        elif len(sizes) == 1:
            axes = np.array([axes])
        elif len(mode_precisions) == 1:
            axes = np.array([[ax] for ax in axes])

        for i, s in enumerate(sizes):
            for j, prec in enumerate(mode_precisions):
                ax = axes[i, j]
                pan_df = sub[(sub['precision'] == prec) & (sub['S'] == s)]
                if pan_df.empty:
                    ax.set_visible(False)
                    continue

                gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
                for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                    g_sub = pan_df[pan_df['gpus'] == gpus]
                    ok_df = (g_sub[g_sub['status'].str.startswith('ok')]
                             .groupby('density_a', as_index=False).mean(numeric_only=True)
                             .sort_values('density_a'))
                    if not ok_df.empty:
                        ax.plot(ok_df['density_a'], ok_df['agg_tflops'],
                                linestyle='-', marker='o', color=gpu_palette[k],
                                label=f'{gpus} GPUs (Compute)' if i == 0 and j == 0 else "")
                        ax.plot(ok_df['density_a'], ok_df['eff_tflops'],
                                linestyle='--', marker='s', color=gpu_palette[k],
                                label=f'{gpus} GPUs (Effective)' if i == 0 and j == 0 else "")

                ax.set_xscale('log', base=10)
                ax.set_xticks(sorted(sub['density_a'].unique()))
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
                ax.set_title(f'S={s}, {prec.upper()}' if i == 0 else f'S={s}')
                ax.set_ylabel('TFLOP/s' if j == 0 else '')
                if i == len(sizes) - 1:
                    ax.set_xlabel('Density')
                if i == 0 and j == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'{filename_prefix}_{mode}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_timing_distribution(df, out_dir):
    if df.empty:
        return
    modes = df['mode'].unique()

    colors = sns.color_palette("muted")
    c_comp, c_h2d, c_d2h, c_sym = colors[2], colors[0], colors[3], colors[1]

    for mode in modes:
        sub = df[df['mode'] == mode]
        if sub.empty:
            continue

        mode_precisions = sorted(sub['precision'].unique())
        sizes = sorted(sub['S'].unique())
        all_gpus = sorted(sub['gpus'].unique())

        fig, axes = plt.subplots(len(sizes) * len(all_gpus), len(mode_precisions),
                                 figsize=(6 * len(mode_precisions), 4 * len(sizes) * len(all_gpus)))
        fig.suptitle(f'SpGEMM Time Distribution vs Density ({mode.capitalize()})', y=0.98)

        axes = np.array(axes).reshape(len(sizes) * len(all_gpus), len(mode_precisions))

        row_idx = 0
        for s in sizes:
            for gpus in all_gpus:
                for j, prec in enumerate(mode_precisions):
                    ax = axes[row_idx, j]
                    prec_df = sub[(sub['precision'] == prec) & (sub['gpus'] == gpus) &
                                  (sub['S'] == s)].sort_values('density_a')

                    if prec_df.empty:
                        ax.set_visible(False)
                        continue

                    prec_df = prec_df.copy()
                    total_compute = prec_df['compute_ms'] * prec_df['iters']
                    
                    densities = prec_df['density_a'].values
                    x_indices = np.arange(len(densities))
                    offsets = np.array([-0.3, -0.1, 0.1, 0.3])
                    
                    colors_list = [c_comp, c_sym, c_h2d, c_d2h]
                    labels_list = ['Compute', 'Symbolic (Phase 1)', 'Host-to-Device', 'Device-to-Host']
                    markers_list = ['o', 's', '^', 'v']
                    data_arrays = [total_compute.values, prec_df['symbolic_ms'].values, 
                                   prec_df['h2d_ms'].values, prec_df['d2h_ms'].values]
                    
                    for k in range(4):
                        x_pos = x_indices + offsets[k]
                        y_val = data_arrays[k]
                        ax.vlines(x_pos, 0, y_val, color=colors_list[k], alpha=0.7, linewidth=2)
                        ax.plot(x_pos, y_val, marker=markers_list[k], linestyle='None', 
                                color=colors_list[k], markersize=7,
                                label=labels_list[k] if row_idx == 0 and j == 0 else "")

                    ax.set_xticks(x_indices)
                    ax.set_xticklabels([str(d) for d in densities])
                    
                    ax.set_yscale('symlog', linthresh=0.1)
                    ax.set_ylim(bottom=0)

                    ax.set_title(f'S={s}, GPUs={gpus}, {prec.upper()}')
                    ax.set_ylabel('Absolute Time (ms, symlog)' if j == 0 else '')
                    if row_idx == (len(sizes) * len(all_gpus)) - 1:
                        ax.set_xlabel('Matrix Density')
                    ax.tick_params(axis='x', rotation=45)

                    if row_idx == 0 and j == 0:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig3_time_distribution_{mode}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_power_util(df, out_dir, filename_prefix='fig6_power_util'):
    """Line plot of Power and Utilization vs Size (averaged over densities)."""
    if df.empty:
        return
    modes = df['mode'].unique()
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(2, len(mode_precisions),
                                 figsize=(6 * len(mode_precisions), 8), sharex=True)
        fig.suptitle(f'SpGEMM Power and Utilization vs Matrix Size ({mode.capitalize()})', y=0.98)
        if len(mode_precisions) == 1:
            axes = axes.reshape(2, 1)

        for j, prec in enumerate(mode_precisions):
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                continue

            sub = sub.copy()
            sub['total_power_w'] = sub['power_w_avg'] * sub['gpus']

            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))

            ax_pwr = axes[0, j]
            sns.lineplot(data=sub, x='S', y='total_power_w', hue='gpus', marker='o',
                         palette=gpu_palette, linewidth=2.5, ax=ax_pwr, errorbar=None)
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

            ax_util = axes[1, j]
            sns.lineplot(data=sub, x='S', y='util_pct_avg', hue='gpus', marker='s',
                         palette=gpu_palette, linewidth=2.5, ax=ax_util, errorbar=None)
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
    csv_pattern = os.path.join(base_dir, 'data', '*', 'sweep_spgemm*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print("No sweep_spgemm CSV files found.")
        return

    set_style()
    for csv_file in csv_files:
        gpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', gpu_name, 'spgemm')
        os.makedirs(out_dir, exist_ok=True)

        print(f"Loading SpGEMM data from {csv_file}...")
        df = pd.read_csv(csv_file, engine='python', skipinitialspace=True)
        df = compute_tflops_and_status(df)

        if not df.empty:
            plot_status_heatmap(df, out_dir)
            plot_density_frontier(df, out_dir)

        df_ok = df[df["status_cat"].isin(["ok", "ok-underflow"])].copy()
        if not df_ok.empty:
            plot_metric_vs_density(df_ok, out_dir, metric='agg_tflops', ylabel='TFLOP/s',
                                   filename_prefix='fig1_tflops_vs_density')
            plot_metric_vs_size(df_ok, out_dir, metric='agg_tflops', ylabel='TFLOP/s',
                                filename_prefix='fig2_tflops_vs_size')
            plot_timing_distribution(df_ok, out_dir)
            plot_compute_vs_effective_density(df_ok, out_dir)
            plot_metric_vs_size(df_ok, out_dir, metric='agg_gbps', ylabel='GB/s',
                                filename_prefix='fig5_gbps_vs_size')
            plot_power_util(df_ok, out_dir, filename_prefix='fig6_power_util')

        print(f"SpGEMM plots written to {out_dir}")


if __name__ == '__main__':
    main()
