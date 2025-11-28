import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_hcast_speeds():
    """
    Analyze human vs AI speeds from the CSV file:
    1. Filter for rows with score_binarized = 1
    2. Calculate speed = human_minutes / ai_time_minutes for each row
    3. Output median speedup for each alias
    4. Create distribution plots for each alias
    """
    
    # Load the data
    data_path = Path(__file__).parent / 'inputs' / 'filtered_runs_for_collab.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Unique aliases: {df['alias'].nunique()}")
    
    # Filter for rows with score_binarized = 1 and exclude 'human' alias
    df_filtered = df[(df['score_binarized'] == 1) & (df['alias'] != 'human')].copy()
    print(f"After filtering for score_binarized=1 and excluding 'human': {len(df_filtered)} rows")
    
    # Convert ai_time from milliseconds to minutes
    # ai_time is in milliseconds, convert to minutes
    df_filtered['ai_time_minutes'] = df_filtered['ai_time'] / (1000 * 60)  # milliseconds to minutes
    
    # Calculate speedup = human_minutes / ai_time_minutes
    df_filtered['speedup'] = df_filtered['human_minutes'] / df_filtered['ai_time_minutes']
    
    # Calculate median speedup for each alias
    median_speedups = df_filtered.groupby('alias')['speedup'].median().sort_values(ascending=False)
    
    print("\n=== Median Speedups by Alias ===")
    for alias, median_speedup in median_speedups.items():
        count = len(df_filtered[df_filtered['alias'] == alias])
        print(f"{alias:<25}: {median_speedup:.2f}x (n={count})")
    
    # Create output directory
    output_path = Path(__file__).parent / 'outputs'
    output_path.mkdir(exist_ok=True)
    
    # Create distribution plots
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    # Get aliases sorted by median speedup (descending)
    aliases_sorted = median_speedups.index.tolist()
    
    for i, alias in enumerate(aliases_sorted):
        if i >= len(axes):
            break
            
        alias_data = df_filtered[df_filtered['alias'] == alias]['speedup']
        
        # Create histogram
        ax = axes[i]
        ax.hist(alias_data, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(alias_data.median(), color='red', linestyle='--', linewidth=2, 
                  label=f'Median: {alias_data.median():.2f}x')
        ax.set_title(f'{alias}\n(n={len(alias_data)})', fontsize=10)
        ax.set_xlabel('Speedup (human_minutes / ai_time_minutes)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits
        ax.set_xlim(0, min(alias_data.quantile(0.95) * 1.1, alias_data.max()))
    
    # Hide unused subplots
    for i in range(len(aliases_sorted), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Distribution of AI Speedups (Human Minutes / AI Minutes) by Model\nFiltered for Score Binarized = 1', 
                 fontsize=16, y=0.98)
    
    # Save the distribution plot
    plt.savefig(output_path / 'hcast_speedup_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\nDistribution plot saved to: {output_path / 'hcast_speedup_distributions.png'}")
    plt.close()
    
    # Create separate scatter plots for each model with log scales
    unique_aliases = df_filtered['alias'].unique()
    n_aliases = len(unique_aliases)
    
    # Calculate grid dimensions
    cols = 4
    rows = (n_aliases + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, alias in enumerate(unique_aliases):
        alias_data = df_filtered[df_filtered['alias'] == alias]
        
        ax = axes[i]
        ax.scatter(alias_data['human_minutes'], alias_data['speedup'], alpha=0.6, s=50)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Human Time (minutes, log scale)')
        ax.set_ylabel('Speedup (log scale)')
        ax.set_title(f'{alias}\n(n={len(alias_data)})')
        ax.grid(True, alpha=0.3)
        
        # Add median lines
        median_human_time = alias_data['human_minutes'].median()
        median_speedup = alias_data['speedup'].median()
        ax.axvline(median_human_time, color='red', linestyle='--', alpha=0.7, 
                  label=f'Median human time: {median_human_time:.1f}min')
        ax.axhline(median_speedup, color='orange', linestyle='--', alpha=0.7,
                  label=f'Median speedup: {median_speedup:.1f}x')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(len(unique_aliases), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Human Time vs AI Speedup by Model (Log-Log Scale)\nFiltered for Score Binarized = 1', 
                 fontsize=16, y=0.98)
    
    # Save the scatter plots
    plt.savefig(output_path / 'human_time_vs_speedup_scatter_by_model.png', dpi=300, bbox_inches='tight')
    print(f"Scatter plots by model saved to: {output_path / 'human_time_vs_speedup_scatter_by_model.png'}")
    plt.close()
    
    # Save summary statistics to CSV
    summary_stats = df_filtered.groupby('alias')['speedup'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),  # Q1
        lambda x: x.quantile(0.75),  # Q3
    ]).round(2)
    summary_stats.columns = ['count', 'mean', 'median', 'std', 'min', 'max', 'Q1', 'Q3']
    summary_stats = summary_stats.sort_values('median', ascending=False)
    
    summary_path = output_path / 'hcast_speedup_summary.csv'
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    return df_filtered, summary_stats

if __name__ == "__main__":
    df_filtered, summary_stats = analyze_hcast_speeds()
