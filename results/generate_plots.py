import os
import gpu_specs
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gpu_specs
import matplotlib.ticker as ticker
import squarify
# Professional aesthetics
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

def _add_derived_columns(df):
    # Calculate time percentages with division-by-zero protection
    df['total_tracked_time'] = df['compute_ms'] + df['h2d_ms'] + df['d2h_ms']
    df['total_tracked_time'] = df['total_tracked_time'].replace(0, np.nan)

    df['pct_compute'] = df['compute_ms'] / df['total_tracked_time'] * 100
    df['pct_h2d'] = df['h2d_ms'] / df['total_tracked_time'] * 100
    df['pct_d2h'] = df['d2h_ms'] / df['total_tracked_time'] * 100

    # Calculate Total Node Power and Energy Efficiency with division-by-zero protection
    df['total_power_w'] = df['power_w_avg'] * df['gpus']
    df['tflops_per_watt'] = df['eff_tflops'] / df['total_power_w'].replace(0, np.nan)
    return df

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['density'] = df['density'].astype(float)
    # status is "ok", or "ok/val" / "ok/HIERR" when --validate was passed -- match all of them,
    # not just the bare "ok" (a plain == 'ok' silently drops every validated row).
    ok = df['status'].astype(str).str.startswith('ok')
    df = df[(df['density'] == 1.0) & (df['mode'].isin(['split', 'replicas'])) & ok].copy()
    return _add_derived_columns(df)

def load_sparse_data(filepath):
    """Like load_data(), but for cuSPARSELt structured-sparsity (engine=='sparse') rows.
    Sparse jobs always report density=0.5 (fixed by the 2:4 hardware pruning, independent
    of --density), so density isn't a meaningful filter here the way it is for dense."""
    df = pd.read_csv(filepath)
    ok = df['status'].astype(str).str.startswith('ok')
    df = df[(df['engine'] == 'sparse') & ok].copy()
    return _add_derived_columns(df)

def plot_heatmap(df, out_dir, metric, title_prefix, filename_prefix):
    """Plot Heatmap for a given metric across all modes and precisions"""
    modes = df['mode'].unique()
    precisions = sorted(df['precision'].unique())
    
    fig, axes = plt.subplots(len(modes), len(precisions), figsize=(5 * len(precisions), 5 * len(modes)))
    fig.suptitle(f'{title_prefix} Scaling Heatmap (Rows: Mode, Cols: Precision)', y=0.98)
    
    # Ensure axes is a 2D array even if len(modes) or len(precisions) is 1
    axes = np.array(axes).reshape(len(modes), len(precisions))
    
    for i, mode in enumerate(modes):
        for j, prec in enumerate(precisions):
            ax = axes[i, j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty:
                ax.set_visible(False)
                continue
            
            pivot = sub.pivot_table(index='M', columns='gpus', values=metric)
            pivot = pivot.sort_index(ascending=False)
            
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar_kws={'label': 'TFLOPS'} if j == len(precisions)-1 else None)
            ax.set_title(f'{mode.capitalize()} | {prec.upper()}')
            ax.set_ylabel('Matrix Size (M)' if j == 0 else '')
            ax.set_xlabel('Number of GPUs' if i == len(modes)-1 else '')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(os.path.join(out_dir, f'{filename_prefix}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f'{filename_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_compute_vs_effective(df, out_dir, suffix=''):
    """Plot B: Compute vs Effective TFLOPS (Line plot)"""
    modes = df['mode'].unique()
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions), figsize=(6 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'Compute vs Effective TFLOPS ({mode.capitalize()})', y=0.98)
        
        # Ensure axes is always indexable as a 1D array
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            gpu_palette = sns.color_palette("Set1", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].sort_values('M')
                color = gpu_palette[k]
                
                ax.plot(g_df['M'], g_df['agg_tflops'], linestyle='-', marker='o', color=color, 
                        label=f'{gpus} GPUs (Compute)' if j==0 else "")
                ax.plot(g_df['M'], g_df['eff_tflops'], linestyle='--', marker='s', color=color, 
                        label=f'{gpus} GPUs (Effective)' if j==0 else "")
                
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['M'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            ax.set_ylabel('TFLOPS' if j == 0 else '')
            ax.set_xlabel('Matrix Size (M)')
            
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(os.path.join(out_dir, f'fig2_compute_vs_effective_{mode}{suffix}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'fig2_compute_vs_effective_{mode}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_power_and_utilization(df, out_dir, suffix=''):
    """Plot C: Power, Utilization, and Efficiency over M"""
    modes = df['mode'].unique()
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(3, len(mode_precisions), figsize=(8 * len(mode_precisions), 18), sharex=True)
        fig.suptitle(f'Total Power, Efficiency, and GPU Utilization ({mode.capitalize()})', y=0.98)
        
        # Ensure axes is always indexable as a 2D array of size (3, len(mode_precisions))
        axes = np.array(axes).reshape(3, len(mode_precisions))
        
        # Use a distinct categorical palette so 1 GPU is highly visible
        gpu_palette = sns.color_palette("Set1", len(df['gpus'].unique()))
        
        for j, prec in enumerate(mode_precisions):
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            # Row 0: Total Power
            ax_pwr = axes[0, j]
            sns.lineplot(data=sub, x='M', y='total_power_w', hue='gpus', marker='o', palette=gpu_palette, linewidth=2.5, ax=ax_pwr)
            ax_pwr.set_xscale('log', base=2)
            ax_pwr.set_xticks(sorted(sub['M'].unique()))
            ax_pwr.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_pwr.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_pwr.set_title(f'{prec.upper()}')
            ax_pwr.set_ylabel('Total Node Power (W)' if j == 0 else '')
            ax_pwr.set_xlabel('')
            if j != len(mode_precisions) - 1:
                ax_pwr.get_legend().remove()
            else:
                ax_pwr.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')
                
            # Row 1: Energy Efficiency
            ax_eff = axes[1, j]
            sns.lineplot(data=sub, x='M', y='tflops_per_watt', hue='gpus', marker='D', palette=gpu_palette, linewidth=2.5, ax=ax_eff)
            ax_eff.set_xscale('log', base=2)
            ax_eff.set_xticks(sorted(sub['M'].unique()))
            ax_eff.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_eff.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_eff.set_ylabel('Efficiency (TFLOPS/W)' if j == 0 else '')
            ax_eff.set_xlabel('')
            if j != len(mode_precisions) - 1:
                ax_eff.get_legend().remove()
            else:
                ax_eff.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Row 2: Utilization
            ax_util = axes[2, j]
            sns.lineplot(data=sub, x='M', y='util_pct_avg', hue='gpus', marker='s', palette=gpu_palette, linewidth=2.5, ax=ax_util)
            ax_util.set_xscale('log', base=2)
            ax_util.set_xticks(sorted(sub['M'].unique()))
            ax_util.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_util.tick_params(axis='x', rotation=45, labelbottom=True)
            ax_util.set_ylabel('Avg GPU Utilization (%)' if j == 0 else '')
            ax_util.set_xlabel('Matrix Size (M)')
            ax_util.set_ylim(0, 105)
            if j != len(mode_precisions) - 1:
                ax_util.get_legend().remove()
            else:
                ax_util.legend(title='GPUs', bbox_to_anchor=(1.05, 1), loc='upper left')
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(os.path.join(out_dir, f'fig3_power_util_eff_{mode}{suffix}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'fig3_power_util_eff_{mode}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_timing_distribution(df, out_dir, suffix=''):
    """Plot D: 100% Stacked Bar for Time Breakdown (All GPUs)"""
    modes = df['mode'].unique()
    all_gpus = sorted(df['gpus'].unique())
    
    colors = sns.color_palette("muted")
    c_comp, c_h2d, c_d2h = colors[2], colors[0], colors[3]
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(len(all_gpus), len(mode_precisions), figsize=(6 * len(mode_precisions), 4 * len(all_gpus)))
        fig.suptitle(f'Time Distribution Percentage vs Problem Size ({mode.capitalize()})', y=0.98)
        
        # Ensure axes is always a 2D array
        axes = np.array(axes).reshape(len(all_gpus), len(mode_precisions))
        
        for i, gpus in enumerate(all_gpus):
            for j, prec in enumerate(mode_precisions):
                ax = axes[i, j]
                prec_df = df[(df['mode'] == mode) & (df['precision'] == prec) & (df['gpus'] == gpus)].sort_values('M')
                
                if prec_df.empty:
                    ax.set_visible(False)
                    continue
                
                x = prec_df['M'].astype(str)
                y1 = prec_df['pct_compute']
                y2 = prec_df['pct_h2d']
                y3 = prec_df['pct_d2h']
                
                ax.bar(x, y1, color=c_comp, label='Compute' if i==0 and j==0 else "")
                ax.bar(x, y2, bottom=y1, color=c_h2d, label='Host-to-Device' if i==0 and j==0 else "")
                ax.bar(x, y3, bottom=y1+y2, color=c_d2h, label='Device-to-Host' if i==0 and j==0 else "")
                
                ax.set_title(f'{gpus} GPUs | {prec.upper()}')
                ax.set_ylim(0, 100)
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
                
                if j == 0: ax.set_ylabel('% of Total Time')
                if i == len(all_gpus) - 1: ax.set_xlabel('Matrix Size (M)')
                
                if i == 0 and j == 0:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.15), ncol=1)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f'fig4_time_distribution_{mode}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_bandwidth(df, out_dir, spec, suffix=''):
    """Plot E: Bandwidth Scaling"""
    modes = df['mode'].unique()
    
    for mode in modes:
        mode_precisions = sorted(df[df['mode'] == mode]['precision'].unique())
        fig, axes = plt.subplots(1, len(mode_precisions), figsize=(5 * len(mode_precisions), 5), sharex=True)
        fig.suptitle(f'Bandwidth (GB/s): H2D (Solid) vs D2H (Dashed) [{mode.capitalize()}]', y=0.98)
        
        # Ensure axes is always indexable as a 1D array
        axes = np.atleast_1d(axes)
        
        for j, prec in enumerate(mode_precisions):
            ax = axes[j]
            sub = df[(df['mode'] == mode) & (df['precision'] == prec)]
            if sub.empty: continue
            
            gpu_palette = sns.color_palette("Set2", len(sub['gpus'].unique()))
            for k, gpus in enumerate(sorted(sub['gpus'].unique())):
                g_df = sub[sub['gpus'] == gpus].sort_values('M')
                color = gpu_palette[k]
                
                ax.plot(g_df['M'], g_df['h2d_gbps'], linestyle='-', marker='^', color=color, 
                        label=f'{gpus} GPUs (H2D)' if j==0 else "")
                ax.plot(g_df['M'], g_df['d2h_gbps'], linestyle='--', marker='v', color=color, 
                        label=f'{gpus} GPUs (D2H)' if j==0 else "")
                
                ax.axhline(y=spec["pcie_bandwidth_gbps"] * gpus, color=color, linestyle=':', alpha=0.5, label=f"PCIe Peak ({gpus}x)" if j==0 else "")
            
            ax.set_xscale('log', base=2)
            ax.set_xticks(sorted(sub['M'].unique()))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelbottom=True)
            ax.set_title(f'{prec.upper()}')
            if j == 0: ax.set_ylabel('GB/s')
            ax.set_xlabel('Matrix Size (M)')
            
            if j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(os.path.join(out_dir, f'fig5_bandwidth_{mode}{suffix}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'fig5_bandwidth_{mode}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_global_treemap(df, out_dir, suffix=''):
    """Plot: Global Hierarchical Treemap for Total Execution Time"""
    if df.empty or 'squarify' not in globals():
        return
        
    modes = df['mode'].unique()
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        if mode_df.empty: continue
        
        group = mode_df.groupby(['precision', 'M'])['wall_ms'].sum().reset_index()
        group = group[group['wall_ms'] > 0].sort_values('wall_ms', ascending=False)
        
        if group.empty: continue
        
        total_ms = group['wall_ms'].sum()
        
        # Combine items that are less than 0.5% into "Other" to prevent empty rendering artifacts in squarify
        threshold = total_ms * 0.005
        large = group[group['wall_ms'] >= threshold].copy()
        small = group[group['wall_ms'] < threshold]
        
        if not small.empty:
            other_ms = small['wall_ms'].sum()
            other_row = pd.DataFrame([{'precision': 'Other', 'M': 'Various', 'wall_ms': other_ms}])
            large = pd.concat([large, other_row], ignore_index=True)
            
            
        group = large.sort_values(by=['precision', 'wall_ms'], ascending=[True, False])
        group['pct'] = group['wall_ms'] / total_ms * 100
        
        labels = group.apply(lambda row: f"{row['precision'].upper()}{' S='+str(row['M']) if row['M'] != 'Various' else ''}\n{row['pct']:.1f}% ({row['wall_ms']/1000:.1f}s)", axis=1)
        sizes = group['wall_ms']
        
        palette = sns.color_palette("Set2", len(group['precision'].unique()))
        color_map = {prec: palette[i] for i, prec in enumerate(group['precision'].unique())}
        colors = [color_map[prec] for prec in group['precision']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
        
        ax.set_title(f"Total Sweep Execution Time Breakdown ({mode.capitalize()}) - Total: {total_ms/1000/60:.1f} min", fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fig7_global_treemap_{mode}{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pattern = os.path.join(base_dir, 'data', '*', 'sweep_20*.csv')
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        return
        
    set_style()
    for csv_file in csv_files:
        gpu_name = os.path.basename(os.path.dirname(csv_file))
        out_dir = os.path.join(base_dir, 'plots', gpu_name, 'gemm')
        os.makedirs(out_dir, exist_ok=True)
        spec = gpu_specs.get_gpu_spec(gpu_name)
        
        print(f"Loading data from {csv_file}...")
        df = load_data(csv_file)
        plot_heatmap(df, out_dir, 'eff_tflops', 'Effective TFLOPS', 'fig1a_eff_tflops_heatmap')
        plot_heatmap(df, out_dir, 'agg_tflops', 'Absolute/Compute TFLOPS', 'fig1b_abs_tflops_heatmap')
        plot_compute_vs_effective(df, out_dir)
        plot_power_and_utilization(df, out_dir)
        plot_timing_distribution(df, out_dir)
        plot_bandwidth(df, out_dir, spec)
        plot_global_treemap(df, out_dir)
        print(f"All dense plots generated in {out_dir}")
        sparse_df = load_sparse_data(csv_file)
        if not sparse_df.empty:
            plot_heatmap(sparse_df, out_dir, 'eff_tflops', 'Sparse Effective TFLOPS', 'fig1a_eff_tflops_heatmap_sparse')
            plot_heatmap(sparse_df, out_dir, 'agg_tflops', 'Sparse Absolute/Compute TFLOPS', 'fig1b_abs_tflops_heatmap_sparse')
            plot_compute_vs_effective(sparse_df, out_dir, suffix='_sparse')
            plot_power_and_utilization(sparse_df, out_dir, suffix='_sparse')
            plot_timing_distribution(sparse_df, out_dir, suffix='_sparse')
            plot_bandwidth(sparse_df, out_dir, spec, suffix='_sparse')
            plot_global_treemap(sparse_df, out_dir, suffix='_sparse')
            print(f"All sparse plots generated in {out_dir}")

if __name__ == '__main__':
    main()
