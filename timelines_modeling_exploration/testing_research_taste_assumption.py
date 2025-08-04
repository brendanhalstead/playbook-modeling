import pandas as pd
import numpy as np
from scipy.stats import norm, pareto
from scipy.optimize import milp, LinearConstraint
import time
import os


def get_lognorm_params(low, high, prob=0.8):
    """
    Calculates the mu and sigma for the underlying normal distribution
    of a lognormal distribution from a given confidence interval.
    """
    log_low = np.log(low)
    log_high = np.log(high)
    z = norm.ppf(1 - (1 - prob) / 2)
    mu = (log_high + log_low) / 2
    sigma = (log_high - log_low) / (2 * z)
    return mu, sigma


def solve_dp(experiments_df, time_budget, compute_budget, precision=100, verbose=True):
    """
    Solves the 0/1 knapsack problem with two constraints using dynamic programming.
    """
    if verbose:
        print(f"\n--- Running DP for Time: {time_budget}, Compute: {compute_budget} ---")

    # Discretize the problem
    time_budget_int = int(time_budget * precision)
    compute_budget_int = int(compute_budget * precision)

    dp_df = experiments_df.copy()
    dp_df['time_int'] = (dp_df['implementation_time'] * precision).round().astype(int)
    dp_df['compute_int'] = (dp_df['compute_required'] * precision).round().astype(int)
    dp_df['log_improvement'] = np.log(dp_df['software_improvement'])

    dp_df = dp_df[
        (dp_df['time_int'] > 0) & (dp_df['compute_int'] > 0) &
        (dp_df['time_int'] <= time_budget_int) &
        (dp_df['compute_int'] <= compute_budget_int)
    ]

    dp = [[(0.0, []) for _ in range(compute_budget_int + 1)] for _ in range(time_budget_int + 1)]

    if verbose:
        print("Running dynamic programming solver... (this may take a moment)")

    for index, row in dp_df.iterrows():
        item_time = int(row['time_int'])
        item_compute = int(row['compute_int'])
        item_log_improvement = row['log_improvement']
        
        for t in range(time_budget_int, item_time - 1, -1):
            for c in range(compute_budget_int, item_compute - 1, -1):
                val_without, items_without = dp[t][c]
                
                val_with_prev, items_with_prev = dp[t - item_time][c - item_compute]
                val_with = val_with_prev + item_log_improvement
                
                if val_with > val_without:
                    new_items = items_with_prev + [index]
                    dp[t][c] = (val_with, new_items)

    if verbose:
        print("Dynamic programming solver finished.")

    max_log_improvement, best_item_indices = dp[time_budget_int][compute_budget_int]
    selected_experiments = experiments_df.loc[best_item_indices]

    total_improvement = selected_experiments['software_improvement'].prod()
    
    return {
        'total_improvement': total_improvement,
        'log10_improvement': np.log10(total_improvement) if total_improvement > 0 else -np.inf,
        'num_experiments': len(selected_experiments),
        'time_used': selected_experiments['implementation_time'].sum(),
        'compute_used': selected_experiments['compute_required'].sum()
    }


def solve_greedy_1d(experiments_df, cost_col, budget_val, value_col='software_improvement', verbose=True):
    """
    Solves the 0/1 knapsack problem with a single constraint using a greedy heuristic.
    This is much faster than DP. The heuristic is to pick items with the highest log-value-to-cost ratio.
    """
    if verbose:
        print(f"\n--- Running 1D Greedy for Cost: {cost_col}, Budget: {budget_val} ---")
    
    df = experiments_df.copy()
    
    # Calculate score = log(value) / cost
    epsilon = 1e-9
    df['score'] = np.log(df[value_col]) / (df[cost_col] + epsilon)
    
    # Sort by score
    df_sorted = df.sort_values(by='score', ascending=False)
    
    selected_indices = []
    current_budget_used = 0.0
    
    for index, row in df_sorted.iterrows():
        cost = row[cost_col]
        if current_budget_used + cost <= budget_val:
            selected_indices.append(index)
            current_budget_used += cost
            
    selected_experiments = experiments_df.loc[selected_indices]
    
    total_improvement = selected_experiments[value_col].prod()
    
    return {
        'total_improvement': total_improvement,
        'log10_improvement': np.log10(total_improvement) if total_improvement > 0 else -np.inf,
        'num_experiments': len(selected_experiments),
        'time_used': selected_experiments['implementation_time'].sum(),
        'compute_used': selected_experiments['compute_required'].sum()
    }


# --- Data Generation Helpers ---
def generate_lognormal_data(mu, sigma, n_samples):
    """Generates standard independent lognormal data."""
    software_improvement = np.random.lognormal(mu, sigma, n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu, sigma, n_samples)
    compute_required = np.random.lognormal(mu, sigma, n_samples)
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required
    })

def generate_powerlaw_data(mu, sigma, n_samples, alpha=1.16):
    """Generates data where software improvement follows a power law."""
    software_improvement = pareto.rvs(b=alpha, size=n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu, sigma, n_samples)
    compute_required = np.random.lognormal(mu, sigma, n_samples)
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required
    })

def generate_correlated_lognormal_data(mu, sigma, n_samples, corr_matrix):
    """Generates correlated lognormal data from a given correlation matrix."""
    cov_matrix = np.diag([sigma, sigma, sigma]) @ corr_matrix @ np.diag([sigma, sigma, sigma])
    mean_vec = [mu, mu, mu]
    
    normal_samples = np.random.multivariate_normal(mean_vec, cov_matrix, n_samples)
    lognormal_samples = np.exp(normal_samples)
    
    # Ensure software improvements are >= 1
    lognormal_samples[:, 0] += 1.0
    
    return pd.DataFrame({
        'software_improvement': lognormal_samples[:, 0],
        'implementation_time': lognormal_samples[:, 1],
        'compute_required': lognormal_samples[:, 2]
    })


# --- Main Simulation Loop ---
experiment_configs = [(5000, 10), (50000, 2)]
all_trials_data = []

# Get parameters from the original script logic
CI_LOW = 0.1
CI_HIGH = 10
mu, sigma = get_lognorm_params(CI_LOW, CI_HIGH)

# Define the settings to sweep through
settings = {
    "current_lognormal": {},
    "power_law_sw": {},
    "corr_sw_compute_0.7": {"corr_matrix": np.array([[1.0, 0.0, 0.7], [0.0, 1.0, 0.0], [0.7, 0.0, 1.0]])},
    "corr_all_0.7": {"corr_matrix": np.full((3, 3), 0.7) + np.diag([0.3, 0.3, 0.3])},
    "corr_all_0.9": {"corr_matrix": np.full((3, 3), 0.9) + np.diag([0.1, 0.1, 0.1])}
}

for n_experiments, n_trials in experiment_configs:
    print(f"\n{'='*25} Running with {n_experiments} experiments and {n_trials} trials {'='*25}")
    for setting_name, setting_params in settings.items():
        print(f"\n{'#'*25} Running Setting: {setting_name} {'#'*25}")
        for trial in range(n_trials):
            print(f"\n--- Starting Trial {trial + 1}/{n_trials} for setting '{setting_name}' ---")

            # --- Generate Data based on Setting ---
            if setting_name == "current_lognormal":
                experiments_df = generate_lognormal_data(mu, sigma, n_experiments)
            elif setting_name == "power_law_sw":
                experiments_df = generate_powerlaw_data(mu, sigma, n_experiments)
            elif "corr" in setting_name:
                experiments_df = generate_correlated_lognormal_data(mu, sigma, n_experiments, setting_params["corr_matrix"])
            
            # --- Run simulations for this trial's data ---
            taste_levels = {
                "Optimal from all ideas (100% visibility)": 1.0,
                "Optimal from 10% of ideas (worse taste)": 0.1
            }
            base_time_budget = 1.0
            base_compute_budget = 1.0

            for taste_name, taste_fraction in taste_levels.items():
                if taste_fraction < 1.0:
                    ideas_pool = experiments_df.sample(frac=taste_fraction, random_state=trial)
                else:
                    ideas_pool = experiments_df
                
                s = {}
                # Default
                s['default'] = solve_dp(ideas_pool, base_time_budget, base_compute_budget, verbose=False)
                
                # Instant Time
                s['instant_time'] = solve_greedy_1d(ideas_pool, cost_col='compute_required', budget_val=base_compute_budget, verbose=False)
                
                # Infinite Compute
                s['infinite_compute'] = solve_greedy_1d(ideas_pool, cost_col='implementation_time', budget_val=base_time_budget, verbose=False)

                log10_default = s['default']['log10_improvement']
                q_instant_time_div_default = s['instant_time']['log10_improvement'] / log10_default if log10_default > 0 else np.nan
                q_infinite_compute_div_default = s['infinite_compute']['log10_improvement'] / log10_default if log10_default > 0 else np.nan
                
                all_trials_data.append({
                    'n_experiments': n_experiments,
                    'setting': setting_name,
                    'trial': trial, 
                    'taste_level': taste_name,
                    'q_instant_time_div_default': q_instant_time_div_default,
                    'q_infinite_compute_div_default': q_infinite_compute_div_default
                })


# --- Final Analysis ---
results_df = pd.DataFrame(all_trials_data)
pd.set_option('display.float_format', '{:.2f}'.format)

# Rename columns
column_renames = {
    'q_instant_time_div_default': 'Log10_InstantTime_vs_Default',
    'q_infinite_compute_div_default': 'Log10_InfiniteCompute_vs_Default'
}
results_df.rename(columns=column_renames, inplace=True)

# Console Output
print("\n\n--- Summary Statistics of Sweeps Over 10 Trials ---")
grouped_summary = results_df.groupby(['n_experiments', 'setting', 'taste_level']).describe()
print(grouped_summary)

# CSV Output
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'settings_sweep_summary.csv')
grouped_summary.to_csv(output_path)
print(f"\nSweep summary saved to {output_path}")
