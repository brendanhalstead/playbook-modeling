"""
AI vs Human Task Completion Simulation

Model:
- Task has S subtasks (drawn log-uniformly)
- Each subtask takes Geometric(p) trials, 1 minute per trial
- AI subtasks = S * h(human_time), where h(t) = (1 + a*t^b)^((c-1)/b)

Parameters to vary:
- p_H: human success probability per subtask attempt
- p_AI: AI success probability per subtask attempt  
- c: controls whether AI needs more (c>1) or fewer (c<1) subtasks for longer tasks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

np.random.seed(42)

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h(t, a=0.108, b=0.906, c=2.0):
    """
    Multiplier for AI subtasks relative to human subtasks.
    h(t) = (1 + a * t^b)^((c-1)/b)
    
    c > 1: AI needs MORE subtasks for longer tasks (overhead scales with complexity)
    c < 1: AI needs FEWER subtasks for longer tasks (AI efficiency improves)
    c = 1: h(t) = 1 for all t (AI has same subtasks as human)
    """
    return (1 + a * t**b) ** ((c - 1) / b)


def run_simulation(p_H, p_AI, a, b, c, n_simulations=5000, S_min=1, S_max=10000):
    """
    Run simulation with:
    - S drawn log-uniformly between S_min and S_max
    - AI subtasks = S * h(human_time)
    
    Returns: human_times, ai_times, S_human, S_ai (all arrays)
    """
    # Draw S log-uniformly
    log_S = np.random.uniform(np.log(S_min), np.log(S_max), size=n_simulations)
    S_array = np.exp(log_S).astype(int)
    S_array = np.maximum(S_array, 1)
    
    human_times = []
    ai_times = []
    S_human_list = []
    S_ai_list = []
    
    for S in tqdm(S_array, desc="Simulating tasks"):
        # Human completion time
        h_time = np.random.geometric(p_H, size=S).sum()
        
        # AI subtasks based on human time (capped at 100k for tractability)
        multiplier = h(h_time, a, b, c)
        S_ai = max(1, min(100000, int(round(S * multiplier))))
        
        # AI completion time
        a_time = np.random.geometric(p_AI, size=S_ai).sum()
        
        human_times.append(h_time)
        ai_times.append(a_time)
        S_human_list.append(S)
        S_ai_list.append(S_ai)
    
    return (np.array(human_times), np.array(ai_times), 
            np.array(S_human_list), np.array(S_ai_list))


def compute_curves(human_times, ai_times, n_bins=30):
    """Compute summary curves binned by human completion time."""
    log_min = np.log10(max(1, human_times.min()))
    log_max = np.log10(human_times.max())
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    
    results = {
        'centers': [],
        'ai_win_strict': [],
        'ai_win_ties': [],
        'ratio_mean': [],
        'ratio_median': [],
        'counts': []
    }
    
    for i in range(len(bin_edges) - 1):
        mask = (human_times >= bin_edges[i]) & (human_times < bin_edges[i+1])
        if mask.sum() > 30:
            h = human_times[mask]
            a = ai_times[mask]
            
            results['centers'].append(np.sqrt(bin_edges[i] * bin_edges[i+1]))
            results['ai_win_strict'].append((a < h).mean() * 100)
            results['ai_win_ties'].append((a <= h).mean() * 100)
            results['ratio_mean'].append((h / a).mean())
            results['ratio_median'].append(np.median(h / a))
            results['counts'].append(mask.sum())
    
    return {k: np.array(v) for k, v in results.items()}


def print_summary(human_times, ai_times, S_human, S_ai, label=""):
    """Print summary statistics."""
    print(f"\n--- {label} ---")
    print(f"Mean human subtasks: {S_human.mean():.1f}")
    print(f"Mean AI subtasks: {S_ai.mean():.1f}")
    print(f"Mean h(t) realized: {(S_ai/S_human).mean():.2f}")
    print(f"Mean human time: {human_times.mean():.1f}")
    print(f"Mean AI time: {ai_times.mean():.1f}")
    print(f"P(AI < Human): {(ai_times < human_times).mean()*100:.1f}%")
    print(f"P(AI ≤ Human): {(ai_times <= human_times).mean()*100:.1f}%")
    print(f"E[Human/AI]: {(human_times/ai_times).mean():.3f}")


def print_detailed_table(human_times, ai_times, label=""):
    """Print results by human time bins."""
    print(f"\n{label} - Results by human completion time:")
    print(f"{'Human Time':<20} {'P(AI≤H)':<15} {'E[H/AI]':<15} {'n':<10}")
    print("-"*60)
    
    log_bins = [1, 10, 100, 1000, 10000, 100000, np.inf]
    for i in range(len(log_bins)-1):
        lo, hi = log_bins[i], log_bins[i+1]
        mask = (human_times >= lo) & (human_times < hi)
        
        if mask.sum() > 50:
            p_win = (ai_times[mask] <= human_times[mask]).mean() * 100
            ratio = (human_times[mask] / ai_times[mask]).mean()
            hi_str = f"{hi:.0f}" if hi != np.inf else "∞"
            print(f"{lo:.0f}-{hi_str}".ljust(20) + 
                  f"{p_win:.1f}%".ljust(15) + 
                  f"{ratio:.2f}".ljust(15) + 
                  f"{mask.sum()}")


def sigmoid(x, L, k, x0):
    """Sigmoid function: L / (1 + exp(-k*(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def fit_sigmoid(log_human_times, ai_win_pct):
    """Fit a sigmoid to the data and return params and R^2."""
    try:
        # Initial guesses: L=100 (max %), k=-1 (decreasing), x0=median
        p0 = [100, -0.5, np.median(log_human_times)]
        bounds = ([0, -10, -10], [100, 10, 20])
        popt, _ = curve_fit(sigmoid, log_human_times, ai_win_pct, p0=p0, bounds=bounds, maxfev=5000)

        # Compute R^2
        y_pred = sigmoid(log_human_times, *popt)
        ss_res = np.sum((ai_win_pct - y_pred) ** 2)
        ss_tot = np.sum((ai_win_pct - np.mean(ai_win_pct)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return popt, r_squared
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        return None, None


def inverse_sigmoid(y, L, k, x0):
    """Inverse sigmoid: find x given y = L / (1 + exp(-k*(x - x0)))"""
    # y = L / (1 + exp(-k*(x - x0)))
    # 1 + exp(-k*(x - x0)) = L / y
    # exp(-k*(x - x0)) = L/y - 1
    # -k*(x - x0) = ln(L/y - 1)
    # x = x0 - ln(L/y - 1) / k
    if y <= 0 or y >= L:
        return None
    return x0 - np.log(L / y - 1) / k


# =============================================================================
# MAIN SIMULATION
# =============================================================================

if __name__ == "__main__":
    # Parameters
    p_H = 0.9
    p_AI = 0.99
    a, b = 0.108, 0.906
    n_sims = 2000
    S_min, S_max = 1, 1000

    c_values = [1.2, 2.0, 4.0]
    colors = ['green', 'blue', 'red']

    print("="*60)
    print("AI vs HUMAN TASK COMPLETION SIMULATION")
    print("="*60)
    print(f"p_H = {p_H}, p_AI = {p_AI}")
    print(f"a = {a}, b = {b}")
    print(f"S ~ LogUniform[{S_min}, {S_max}]")
    print(f"n_simulations = {n_sims}")
    print(f"c values tested: {c_values}")

    # Run simulations for each c value
    print("\nRunning simulations...")
    results = {}
    for c_val in c_values:
        print(f"\n--- c = {c_val} ---")
        ht, at, sh, sa = run_simulation(p_H, p_AI, a, b, c=c_val,
                                         n_simulations=n_sims,
                                         S_min=S_min, S_max=S_max)
        curves = compute_curves(ht, at)
        results[c_val] = {
            'human_times': ht,
            'ai_times': at,
            'S_human': sh,
            'S_ai': sa,
            'curves': curves
        }
        print_summary(ht, at, sh, sa, f"c={c_val}")

    # Fit sigmoids and report results
    print("\n" + "="*60)
    print("SIGMOID FIT RESULTS")
    print("="*60)
    print(f"{'c':<10} {'R²':<12} {'L (max %)':<12} {'k (slope)':<12} {'x0 (midpoint)':<15}")
    print("-"*60)

    sigmoid_fits = {}
    for c_val in c_values:
        curves = results[c_val]['curves']
        log_centers = np.log10(curves['centers'])
        ai_win = curves['ai_win_ties']

        popt, r_sq = fit_sigmoid(log_centers, ai_win)
        sigmoid_fits[c_val] = (popt, r_sq)

        if popt is not None:
            print(f"{c_val:<10} {r_sq:<12.4f} {popt[0]:<12.2f} {popt[1]:<12.4f} {popt[2]:<15.2f}")
        else:
            print(f"{c_val:<10} {'FAILED':<12}")

    # Compute 50% and 80% human times and their ratio
    print("\n" + "="*60)
    print("HUMAN TIME AT 50% vs 80% AI SUCCESS")
    print("="*60)
    print(f"{'c':<10} {'t(50%)':<15} {'t(80%)':<15} {'t(50%)/t(80%)':<15}")
    print("-"*60)

    for c_val in c_values:
        popt, _ = sigmoid_fits[c_val]
        if popt is not None:
            L, k, x0 = popt
            log_t_50 = inverse_sigmoid(50, L, k, x0)
            log_t_80 = inverse_sigmoid(80, L, k, x0)

            if log_t_50 is not None and log_t_80 is not None:
                t_50 = 10 ** log_t_50
                t_80 = 10 ** log_t_80
                ratio = t_50 / t_80
                print(f"{c_val:<10} {t_50:<15.1f} {t_80:<15.1f} {ratio:<15.2f}")
            else:
                print(f"{c_val:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        else:
            print(f"{c_val:<10} {'FAILED':<15}")

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, c_val in enumerate(c_values):
        ax = axes[idx]
        curves = results[c_val]['curves']
        log_centers = np.log10(curves['centers'])
        ai_win = curves['ai_win_ties']

        # Plot data points
        ax.plot(log_centers, ai_win, 'o', color=colors[idx], markersize=5, label='Data')

        # Plot sigmoid fit
        popt, r_sq = sigmoid_fits[c_val]
        if popt is not None:
            x_fit = np.linspace(log_centers.min(), log_centers.max(), 100)
            y_fit = sigmoid(x_fit, *popt)
            ax.plot(x_fit, y_fit, '-', color=colors[idx], linewidth=2,
                    label=f'Sigmoid (R²={r_sq:.4f})')

        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('log₁₀(Human Time)')
        ax.set_ylabel('P(AI ≤ Human) %')
        ax.set_title(f'c = {c_val}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.suptitle(f'Sigmoid Fits: p_H={p_H}, p_AI={p_AI}, a={a}, b={b}', fontsize=12)
    plt.tight_layout()

    # Save to outputs folder
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/sigmoid_fits.png', dpi=150, bbox_inches='tight')

    print("\nPlot saved to outputs/sigmoid_fits.png")