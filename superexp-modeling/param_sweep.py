"""
Parameter Sweep Pipeline for AI vs Human Simulation

Tests different parameter values and evaluates against target metrics:
- Sigmoid R² (closer to 1 is better)
- t(50%)/t(80%) ratio (closer to 5x is better)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import pandas as pd

from superexp_sim import (
    h, run_simulation, compute_curves,
    sigmoid, fit_sigmoid, inverse_sigmoid
)

# Suppress tqdm output from run_simulation during sweeps
import functools


def run_simulation_quiet(p_H, p_AI, a, b, c, n_simulations=2000, S_min=1, S_max=1000):
    """Run simulation without tqdm progress bar."""
    np.random.seed(42)  # Reset seed for reproducibility

    log_S = np.random.uniform(np.log(S_min), np.log(S_max), size=n_simulations)
    S_array = np.exp(log_S).astype(int)
    S_array = np.maximum(S_array, 1)

    human_times = []
    ai_times = []
    S_human_list = []
    S_ai_list = []

    for S in S_array:
        h_time = np.random.geometric(p_H, size=S).sum()
        multiplier = h(h_time, a, b, c)
        S_ai = max(1, min(100000, int(round(S * multiplier))))
        a_time = np.random.geometric(p_AI, size=S_ai).sum()

        human_times.append(h_time)
        ai_times.append(a_time)
        S_human_list.append(S)
        S_ai_list.append(S_ai)

    return (np.array(human_times), np.array(ai_times),
            np.array(S_human_list), np.array(S_ai_list))


def evaluate_params(p_H, p_AI, a, b, c, n_sims=2000, S_min=1, S_max=1000):
    """
    Run simulation and compute metrics.

    Returns dict with:
    - r_squared: sigmoid fit R²
    - t50_t80_ratio: t(50%)/t(80%) ratio
    - params: dict of input parameters
    """
    try:
        ht, at, sh, sa = run_simulation_quiet(p_H, p_AI, a, b, c, n_sims, S_min, S_max)
        curves = compute_curves(ht, at)

        if len(curves['centers']) < 5:
            return None

        log_centers = np.log10(curves['centers'])
        ai_win = curves['ai_win_ties']

        popt, r_sq = fit_sigmoid(log_centers, ai_win)

        if popt is None:
            return None

        L, k, x0 = popt
        log_t_50 = inverse_sigmoid(50, L, k, x0)
        log_t_80 = inverse_sigmoid(80, L, k, x0)

        if log_t_50 is None or log_t_80 is None:
            t50_t80_ratio = None
        else:
            t_50 = 10 ** log_t_50
            t_80 = 10 ** log_t_80
            t50_t80_ratio = t_50 / t_80

        return {
            'p_H': p_H,
            'p_AI': p_AI,
            'a': a,
            'b': b,
            'c': c,
            'r_squared': r_sq,
            't50_t80_ratio': t50_t80_ratio,
            'sigmoid_L': L,
            'sigmoid_k': k,
            'sigmoid_x0': x0,
            'overall_ai_win': (at <= ht).mean() * 100
        }
    except Exception as e:
        print(f"Error with params p_AI={p_AI}, a={a}, b={b}, c={c}: {e}")
        return None


def compute_score(result, target_ratio=5.0):
    """
    Compute overall score from metrics.
    Score = R² * (1 - |log(ratio/target)|)
    Higher is better, max ~1.0
    """
    if result is None or result['t50_t80_ratio'] is None:
        return -np.inf

    r2 = result['r_squared']
    ratio = result['t50_t80_ratio']

    # Penalize deviation from target ratio (in log space)
    ratio_penalty = abs(np.log(ratio / target_ratio))
    ratio_score = max(0, 1 - ratio_penalty / 2)  # 0 to 1, with 1 being perfect

    return r2 * ratio_score


def run_parameter_sweep(param_grid, n_sims=2000, target_ratio=5.0):
    """
    Run parameter sweep over all combinations in param_grid.

    param_grid: dict with keys 'p_AI', 'a', 'b', 'c' and list values

    Returns: DataFrame with results sorted by score
    """
    p_H = 0.9  # Fixed
    S_min, S_max = 1, 1000

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    results = []

    print(f"Running {len(combinations)} parameter combinations...")

    for combo in tqdm(combinations, desc="Parameter sweep"):
        params = dict(zip(keys, combo))

        result = evaluate_params(
            p_H=p_H,
            p_AI=params.get('p_AI', 0.99),
            a=params.get('a', 0.108),
            b=params.get('b', 0.906),
            c=params.get('c', 2.0),
            n_sims=n_sims,
            S_min=S_min,
            S_max=S_max
        )

        if result is not None:
            result['score'] = compute_score(result, target_ratio)
            results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=False)

    return df


def plot_top_combinations(top_params, p_H, n_sims, S_min, S_max, output_path):
    """
    Plot detailed results for top parameter combinations.
    Similar to ht_simulation.png format.
    """
    from superexp_sim import sigmoid, fit_sigmoid, inverse_sigmoid

    n_top = len(top_params)
    fig, axes = plt.subplots(n_top, 3, figsize=(15, 4 * n_top))

    if n_top == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_top))

    for idx, params in enumerate(top_params):
        p_AI = params['p_AI']
        a = params['a']
        b = params['b']
        c = params['c']

        # Run simulation
        ht, at, sh, sa = run_simulation_quiet(p_H, p_AI, a, b, c, n_sims, S_min, S_max)
        curves = compute_curves(ht, at)

        log_centers = np.log10(curves['centers'])
        ai_win = curves['ai_win_ties']

        # Fit sigmoid
        popt, r_sq = fit_sigmoid(log_centers, ai_win)

        # Plot 1: P(AI wins) with sigmoid fit
        ax1 = axes[idx, 0]
        ax1.plot(log_centers, ai_win, 'o', color=colors[idx], markersize=5, label='Data')
        if popt is not None:
            x_fit = np.linspace(log_centers.min(), log_centers.max(), 100)
            y_fit = sigmoid(x_fit, *popt)
            ax1.plot(x_fit, y_fit, '-', color=colors[idx], linewidth=2,
                     label=f'Sigmoid (R²={r_sq:.4f})')
        ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(80, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('log₁₀(Human Time)')
        ax1.set_ylabel('P(AI ≤ Human) %')
        ax1.set_title(f'p_AI={p_AI}, a={a}, b={b}, c={c}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Plot 2: E[Human/AI] ratio
        ax2 = axes[idx, 1]
        ax2.plot(log_centers, curves['ratio_mean'], 'o-', color=colors[idx], markersize=5)
        ax2.axhline(1.0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('log₁₀(Human Time)')
        ax2.set_ylabel('E[Human Time / AI Time]')
        ax2.set_title(f'Speedup Ratio (overall: {(ht/at).mean():.2f})')
        ax2.grid(True, alpha=0.3)
        y_max = min(curves['ratio_mean'].max() * 1.1, 20)
        ax2.set_ylim(0, max(y_max, 2))

        # Plot 3: h(t) function
        ax3 = axes[idx, 2]
        t_range = np.logspace(0, 4, 100)
        h_vals = (1 + a * t_range**b) ** ((c - 1) / b)
        ax3.loglog(t_range, h_vals, '-', color=colors[idx], linewidth=2)
        ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Human Time t')
        ax3.set_ylabel('h(t) = S_AI / S_Human')
        ax3.set_title(f'h(t) multiplier function')
        ax3.grid(True, alpha=0.3)

        # Add t50/t80 annotation
        if popt is not None:
            L, k, x0 = popt
            log_t_50 = inverse_sigmoid(50, L, k, x0)
            log_t_80 = inverse_sigmoid(80, L, k, x0)
            if log_t_50 is not None and log_t_80 is not None:
                t_50 = 10 ** log_t_50
                t_80 = 10 ** log_t_80
                ratio = t_50 / t_80
                ax1.annotate(f't(50%)={t_50:.1f}\nt(80%)={t_80:.1f}\nratio={ratio:.2f}x',
                             xy=(0.02, 0.02), xycoords='axes fraction',
                             fontsize=8, verticalalignment='bottom',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sweep_results(df, param_name, output_path):
    """Plot how metrics vary with a single parameter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Group by the parameter and compute median metrics (robust to outliers)
    grouped = df.groupby(param_name).agg({
        'r_squared': 'median',
        't50_t80_ratio': 'median',
        'score': 'median'
    }).reset_index()

    ax1, ax2, ax3 = axes

    ax1.plot(grouped[param_name], grouped['r_squared'], 'o-', markersize=8)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('R² (sigmoid fit)')
    ax1.set_title('Sigmoid Fit Quality')
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Target')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(grouped[param_name], grouped['t50_t80_ratio'], 'o-', markersize=8, color='orange')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('t(50%)/t(80%)')
    ax2.set_title('Time Ratio (target: 5x)')
    ax2.axhline(5.0, color='green', linestyle='--', alpha=0.5, label='Target')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(grouped[param_name], grouped['score'], 'o-', markersize=8, color='purple')
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Combined Score')
    ax3.set_title('Overall Score')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return grouped


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)

    # Define parameter grid - fewer combinations, more trials
    # p_H = 0.9, so only try p_AI > p_H
    param_grid = {
        'p_AI': [0.92, 0.95, 0.99],        # Must be > p_H (0.9)
        'a': [0.02, 0.05, 0.1],            # Smaller a = gentler h(t) growth
        'b': [0.3, 0.5, 0.75],             # Smaller b = gentler h(t) growth
        'c': [1.1, 1.2, 1.3]               # Lower c = gentler sigmoid
    }
    n_sims = 10000  # More trials for stability

    print("="*60)
    print("PARAMETER SWEEP PIPELINE")
    print("="*60)
    print(f"Target metrics:")
    print(f"  - Sigmoid R² → 1.0")
    print(f"  - t(50%)/t(80%) → 5.0x")
    print()

    # Run full sweep
    df = run_parameter_sweep(param_grid, n_sims=n_sims, target_ratio=5.0)

    # Save full results
    df.to_csv('outputs/param_sweep_results.csv', index=False)
    print(f"\nFull results saved to outputs/param_sweep_results.csv")

    # Print top 10 results
    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*60)
    print(df[['p_AI', 'a', 'b', 'c', 'r_squared', 't50_t80_ratio', 'score']].head(10).to_string(index=False))

    # Print worst 5 results
    print("\n" + "="*60)
    print("BOTTOM 5 PARAMETER COMBINATIONS")
    print("="*60)
    print(df[['p_AI', 'a', 'b', 'c', 'r_squared', 't50_t80_ratio', 'score']].tail(5).to_string(index=False))

    # Plot individual parameter effects
    print("\nGenerating parameter effect plots...")

    for param in ['p_AI', 'a', 'b', 'c']:
        grouped = plot_sweep_results(df, param, f'outputs/sweep_{param}.png')
        print(f"\n{param} effect (averaged over other params):")
        print(grouped.to_string(index=False))

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total configurations tested: {len(df)}")
    print(f"Best R²: {df['r_squared'].max():.4f}")
    print(f"Best t50/t80 ratio: {df['t50_t80_ratio'].max():.2f}")
    print(f"Closest to target ratio (5x): {df.loc[(df['t50_t80_ratio'] - 5).abs().idxmin()]['t50_t80_ratio']:.2f}")
    print(f"Best overall score: {df['score'].max():.4f}")

    print("\nPlots saved to outputs/sweep_*.png")

    # Plot detailed results for top 5 parameter combinations
    print("\nGenerating detailed plots for top 5 combinations...")
    top_5 = df.head(5).to_dict('records')
    p_H = 0.9
    S_min, S_max = 1, 1000
    plot_top_combinations(top_5, p_H, n_sims, S_min, S_max, 'outputs/top_combinations.png')
    print("Detailed plots saved to outputs/top_combinations.png")
