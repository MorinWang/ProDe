

import re
import ast
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def extract_metrics_from_line(line: str) -> Dict[str, Any]:
    
    
    dict_pattern = r"\{.*\}"
    match = re.search(dict_pattern, line)
    
    if not match:
        return None
    
    try:
        
        metrics = ast.literal_eval(match.group())
        return metrics
    except (ValueError, SyntaxError):
        return None


def collect_reward_metrics(log_file: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    
    reward_data = defaultdict(list)
    reward_timeseries = defaultdict(list)  
    
    with open(log_file, 'r', encoding='utf-8') as f:
        step = 0
        
        for line in f:
            metrics = extract_metrics_from_line(line)
            
            if metrics is None:
                continue
            
            step += 1
            
            for key, value in metrics.items():
                if ('reward' in key.lower() or 
                    'sasrec' in key.lower()) and isinstance(value, (int, float)):
                    reward_data[key].append(float(value))
                    reward_timeseries[key].append((step, float(value)))
    
    return reward_data, reward_timeseries


def compute_statistics(values: List[float]) -> Dict[str, float]:
    
    if not values:
        return {}
    
    arr = np.array(values)
    
    stats = {
        'count': len(values),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'q90': float(np.percentile(arr, 90)),
        'q95': float(np.percentile(arr, 95)),
        'q99': float(np.percentile(arr, 99)),
    }
    
    
    zero_count = np.sum(arr == 0)
    stats['zero_ratio'] = float(zero_count / len(values))
    
    
    unique_values = len(np.unique(arr))
    stats['unique_count'] = unique_values
    stats['discretization_ratio'] = float(unique_values / len(values))
    
    return stats


def generate_visualizations(reward_data: Dict[str, List[float]], 
                           reward_timeseries: Dict[str, List[Tuple[int, float]]],
                           stats_df: pd.DataFrame,
                           output_dir: Path,
                           log_name: str):
    
    
    main_metrics = []
    for metric in reward_data.keys():
        
        if ('/mean' in metric or '/mean:' in metric) and 'std' not in metric:
            if not any(m in metric for m in ['reward_components/', 'reward_std', 'frac_reward']):
                main_metrics.append(metric)
    
    
    if not main_metrics:
        main_metrics = list(reward_data.keys())[:6]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    fig, axes = plt.subplots(len(main_metrics), 1, figsize=(14, 3*len(main_metrics)))
    if len(main_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(main_metrics):
        if metric in reward_timeseries:
            steps, values = zip(*reward_timeseries[metric])
            axes[idx].plot(steps, values, alpha=0.6, linewidth=1)
            
            
            if len(values) > 10:
                window = min(50, len(values) // 10)
                values_ma = pd.Series(values).rolling(window=window, center=True).mean()
                axes[idx].plot(steps, values_ma, color='red', linewidth=2, label=f'Moving Avg (window={window})')
                axes[idx].legend()
            
            axes[idx].set_title(f'{metric} - Training Progress', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel('Reward Value')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{log_name}_reward_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    n_metrics = min(len(main_metrics), 6)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, metric in enumerate(main_metrics[:n_metrics]):
        if metric in reward_data:
            values = reward_data[metric]
            stats_row = stats_df[stats_df['metric_name'] == metric]
            
            if len(stats_row) > 0:
                mean_val = stats_row['mean'].values[0]
                std_val = stats_row['std'].values[0]
                
                axes[idx].hist(values, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                axes[idx].axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 Std')
                axes[idx].axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                
                axes[idx].set_title(f'{metric.split("/")[-1]}\nμ={mean_val:.3f}, σ={std_val:.3f}', 
                                  fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Reward Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)
    
    
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{log_name}_reward_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    std_metrics = [m for m in reward_data.keys() if '/std' in m and 'mean' not in m]
    
    if std_metrics:
        n_std = min(len(std_metrics), 6)
        fig, axes = plt.subplots(n_std, 1, figsize=(12, 3*n_std))
        if n_std == 1:
            axes = [axes]
        
        for idx, metric in enumerate(std_metrics[:n_std]):
            if metric in reward_timeseries:
                steps, values = zip(*reward_timeseries[metric])
                
                
                zero_mask = np.array(values) < 0.01
                non_zero_mask = ~zero_mask
                
                axes[idx].scatter(np.array(steps)[non_zero_mask], np.array(values)[non_zero_mask], 
                                alpha=0.5, s=10, color='blue', label='Non-zero std')
                if np.any(zero_mask):
                    axes[idx].scatter(np.array(steps)[zero_mask], np.array(values)[zero_mask], 
                                    alpha=0.8, s=15, color='red', marker='x', label='Zero std (no discrimination)')
                
                axes[idx].axhline(0.01, color='orange', linestyle='--', linewidth=1, 
                                label='Threshold (0.01)', alpha=0.7)
                
                axes[idx].set_title(f'{metric.split("/")[-1]} - Batch Discrimination Analysis', 
                                  fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel('Std Dev')
                axes[idx].set_yscale('log')
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{log_name}_std_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    plot_data = []
    plot_labels = []
    for metric in main_metrics[:8]:  
        if metric in reward_data:
            values = reward_data[metric]
            plot_data.append(values)
            
            label = metric.split('/')[-1].replace('_reward_func_x1.0/mean', '').replace('_mean', '')
            plot_labels.append(label)
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        
        colors = sns.color_palette("husl", len(plot_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Reward Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward Value')
        ax.set_xlabel('Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{log_name}_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    ax = axes[0, 0]
    for _, row in stats_df.iterrows():
        if row['count'] > 0:
            disc_ratio = row['discretization_ratio']
            color = 'red' if disc_ratio < 0.1 else ('orange' if disc_ratio < 0.3 else 'green')
            ax.scatter(row['count'], row['unique_count'], s=100, alpha=0.6, c=color)
            if row['discretization_ratio'] < 0.1:
                ax.annotate(row['metric_name'].split('/')[-1][:20], 
                          (row['count'], row['unique_count']), 
                          fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Unique Values')
    ax.set_title('Discretization Analysis\n(Red: <10%, Orange: <30%, Green: ≥30%)', 
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.plot([0, stats_df['count'].max()], [0, stats_df['count'].max()*0.1], 
           'r--', alpha=0.5, label='10% threshold')
    ax.legend()
    
    
    ax = axes[0, 1]
    zero_ratios = stats_df['zero_ratio'].values
    metric_names_short = [m.split('/')[-1][:25] for m in stats_df['metric_name']]
    
    bars = ax.barh(range(len(zero_ratios)), zero_ratios, 
                   color=['red' if z > 0.5 else ('orange' if z > 0.2 else 'green') 
                          for z in zero_ratios])
    ax.set_yticks(range(len(metric_names_short)))
    ax.set_yticklabels(metric_names_short, fontsize=8)
    ax.set_xlabel('Zero Value Ratio')
    ax.set_title('Zero Value Analysis\n(Red: >50%, Orange: >20%, Green: ≤20%)', 
                fontweight='bold')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    
    ax = axes[1, 0]
    for _, row in stats_df.iterrows():
        cv = row['std'] / row['mean'] if row['mean'] > 0 else 0
        color = 'red' if cv > 2 else ('orange' if cv > 1 else 'green')
        ax.scatter(row['mean'], row['std'], s=100, alpha=0.6, c=color)
    
    
    x_max = stats_df['mean'].max() * 1.1
    ax.plot([0, x_max], [0, x_max], 'b--', alpha=0.3, label='std = mean')
    ax.plot([0, x_max], [0, x_max*2], 'r--', alpha=0.5, label='std = 2*mean (high variance)')
    
    ax.set_xlabel('Mean')
    ax.set_ylabel('Std Dev')
    ax.set_title('Variability Analysis (Coefficient of Variation)\n(Red: CV>2, Orange: CV>1, Green: CV≤1)', 
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    
    ax = axes[1, 1]
    main_stats = stats_df[stats_df['metric_name'].isin(main_metrics)]
    if len(main_stats) > 0:
        x_pos = np.arange(len(main_stats))
        means = main_stats['mean'].values
        stds = main_stats['std'].values
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=sns.color_palette("husl", len(main_stats)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.split('/')[-1][:20] for m in main_stats['metric_name']], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Reward Value')
        ax.set_title('Main Reward Metrics Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{log_name}_diagnostic_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    

def analyze_reward_distribution(log_file: str, output_csv: str = None, enable_plot: bool = False):
    
    
    reward_data, reward_timeseries = collect_reward_metrics(log_file)
    
    if not reward_data:
        return
    
    results = []
    
    for metric_name, values in reward_data.items():
        stats = compute_statistics(values)
        stats['metric_name'] = metric_name
        results.append(stats)
    
    
    df = pd.DataFrame(results)
    
    
    column_order = [
        'metric_name',
        'count',
        'mean',
        'std',
        'min',
        'max',
        'median',
        'q25',
        'q75',
        'q90',
        'q95',
        'q99',
        'zero_ratio',
        'unique_count',
        'discretization_ratio'
    ]
    
    df = df[column_order]
    
    
    if output_csv is None:
        log_path = Path(log_file)
        output_csv = log_path.parent / f"{log_path.stem}_reward_distribution.csv"
    
    output_csv = Path(output_csv)
    df.to_csv(output_csv, index=False, float_format='%.6f')
    
    if enable_plot:
        plot_output_dir = output_csv.parent if output_csv else Path(log_file).parent
        generate_visualizations(reward_data, reward_timeseries, df, plot_output_dir, Path(log_file).stem)
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        sys.exit(1)
    
    try:
        analyze_reward_distribution(args.log_file, args.output, enable_plot=args.plot)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
