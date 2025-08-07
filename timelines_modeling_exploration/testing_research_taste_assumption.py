import pandas as pd
import numpy as np
from scipy.stats import norm, pareto
from scipy.optimize import milp, LinearConstraint
import time
import os
from tqdm import tqdm


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


def get_lognorm_params_percentile(percentile_val, percentile=0.9):
    """
    Calculates mu and sigma for lognormal distribution where the given percentile equals percentile_val.
    Assumes median (50th percentile) is at 1.0 for consistency.
    """
    # For lognormal, if median = 1, then mu = 0
    mu = 0.0
    # sigma can be derived from the percentile condition
    z_percentile = norm.ppf(percentile)
    # log(percentile_val) = mu + sigma * z_percentile
    # Since mu = 0: sigma = log(percentile_val) / z_percentile
    sigma = np.log(percentile_val) / z_percentile
    return mu, sigma


def get_lognorm_params_two_percentiles(p10_val, p90_val):
    """
    Calculates mu and sigma for lognormal distribution given 10th and 90th percentile values.
    """
    log_p10 = np.log(p10_val)
    log_p90 = np.log(p90_val)
    z_10 = norm.ppf(0.1)
    z_90 = norm.ppf(0.9)
    
    # Solve system: log_p10 = mu + sigma * z_10, log_p90 = mu + sigma * z_90
    sigma = (log_p90 - log_p10) / (z_90 - z_10)
    mu = log_p10 - sigma * z_10
    return mu, sigma


def find_min_implementation_time_budget(ideas_pool, impl_time_budget_range, runtime_budget, compute_budget, target_improvement, base_time_budget):
    """
    Helper function to find minimum implementation time budget needed to achieve target improvement.
    """
    min_budget, max_budget = impl_time_budget_range
    tolerance = 0.02  # 2% tolerance
    
    # Work in log space for binary search
    log_min_budget = np.log(min_budget)
    log_max_budget = np.log(max_budget)
    
    for iteration in range(20):  # Max 20 iterations for binary search
        log_mid_budget = (log_min_budget + log_max_budget) / 2
        mid_budget = np.exp(log_mid_budget)
        
        result = solve_greedy_knapsack(ideas_pool, mid_budget, runtime_budget, compute_budget)
        achieved_improvement = result['total_improvement']
        
        if iteration < 3:  # Show first few iterations for debugging
            print(f"        Iteration {iteration}: impl_time_budget={mid_budget:.4f}, runtime_budget={runtime_budget:.4f}, improvement={achieved_improvement:.2e}, target={target_improvement:.2e}")
        
        if achieved_improvement >= target_improvement:
            log_max_budget = log_mid_budget
        else:
            log_min_budget = log_mid_budget
        
        if (np.exp(log_max_budget) - np.exp(log_min_budget)) / np.exp(log_max_budget) < tolerance:
            break
    
    return np.exp(log_max_budget)


# --- Solver Functions ---

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


def solve_greedy_knapsack(experiments_df, implementation_time_budget, runtime_budget, compute_budget, verbose=False):
    """
    Greedy solver for the knapsack problem with separate implementation time, runtime, and compute constraints.
    Each experiment must satisfy: implementation_time <= implementation_time_budget, runtime <= runtime_budget, compute <= compute_budget
    """
    df = experiments_df.copy()
    
    # Calculate resource usage score for efficiency ranking
    epsilon = 1e-9
    impl_time_usage = df['implementation_time'] / (implementation_time_budget + epsilon)
    runtime_usage = df['runtime'] / (runtime_budget + epsilon)  
    compute_usage = df['compute_required'] / (compute_budget + epsilon)
    resource_score = impl_time_usage + runtime_usage + compute_usage
    
    # Calculate efficiency score
    df['efficiency'] = np.log(df['software_improvement']) / (resource_score + epsilon)
    
    # Sort by efficiency
    df_sorted = df.sort_values(by='efficiency', ascending=False)
    
    selected_indices = []
    current_impl_time_used = 0.0
    current_runtime_used = 0.0
    current_compute_used = 0.0
    
    for index, row in df_sorted.iterrows():
        impl_time_needed = row['implementation_time']
        runtime_needed = row['runtime']
        compute_needed = row['compute_required']
        
        if (current_impl_time_used + impl_time_needed <= implementation_time_budget and 
            current_runtime_used + runtime_needed <= runtime_budget and
            current_compute_used + compute_needed <= compute_budget):
            selected_indices.append(index)
            current_impl_time_used += impl_time_needed
            current_runtime_used += runtime_needed
            current_compute_used += compute_needed
    
    if not selected_indices:
        return {'total_improvement': 1.0, 'log10_improvement': 0.0, 'num_experiments': 0, 
                'impl_time_used': 0.0, 'runtime_used': 0.0, 'compute_used': 0.0}
    
    selected_experiments = experiments_df.loc[selected_indices]
    total_improvement = selected_experiments['software_improvement'].prod()
    
    return {
        'total_improvement': total_improvement,
        'log10_improvement': np.log10(total_improvement) if total_improvement > 0 else -np.inf,
        'num_experiments': len(selected_experiments),
        'impl_time_used': selected_experiments['implementation_time'].sum(),
        'runtime_used': selected_experiments['runtime'].sum(),
        'compute_used': selected_experiments['compute_required'].sum()
    }


def find_min_time_budget_binary_search(experiments_df, compute_budget, target_improvement, scenario_type="default", baseline_time_budget=2.0):
    """
    Binary search to find minimum time budget to achieve target improvement.
    Searches between baseline_time_budget/100 and baseline_time_budget.
    """
    min_budget = baseline_time_budget / 100.0  # 1% of baseline
    max_budget = baseline_time_budget          # 100% of baseline
    tolerance = 0.02  # 2% tolerance to distinguish between small differences

    # Work in log space for binary search
    log_min_budget = np.log(min_budget)
    log_max_budget = np.log(max_budget)

    # Modify experiments based on scenario
    df_modified = experiments_df.copy()
    if scenario_type == "instant_time":
        df_modified['implementation_time'] = 1e-9
    elif scenario_type == "infinite_compute":
        compute_budget = 1e6  # Effectively infinite
        df_modified['runtime'] = 0.0  # Also set runtime to 0

    for _ in range(20):  # Max 20 iterations for binary search
        log_mid_budget = (log_min_budget + log_max_budget) / 2
        mid_budget = np.exp(log_mid_budget)

        result = solve_greedy_knapsack(df_modified, mid_budget, mid_budget, compute_budget)
        achieved_improvement = result['total_improvement']

        if achieved_improvement >= target_improvement:
            log_max_budget = log_mid_budget
        else:
            log_min_budget = log_mid_budget

        if (np.exp(log_max_budget) - np.exp(log_min_budget)) / np.exp(log_max_budget) < tolerance:
            break
    
    return np.exp(log_max_budget)


# --- Data Generation Helpers ---
def generate_lognormal_data(mu, sigma, n_samples):
    """Generates standard independent lognormal data."""
    software_improvement = np.random.lognormal(mu, sigma, n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu, sigma, n_samples)
    compute_required = np.random.lognormal(mu, sigma, n_samples)
    runtime = compute_required.copy()  # Runtime same as compute
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required,
        'runtime': runtime
    })


def generate_lognormal_data_percentile(percentile_val, percentile, n_samples):
    """Generates lognormal data with specified percentile value."""
    mu_sw, sigma_sw = get_lognorm_params_percentile(percentile_val, percentile)
    # Use original mu, sigma for implementation time and compute required
    mu_orig, sigma_orig = get_lognorm_params(CI_LOW, CI_HIGH)
    
    software_improvement = np.random.lognormal(mu_sw, sigma_sw, n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu_orig, sigma_orig, n_samples)
    compute_required = np.random.lognormal(mu_orig, sigma_orig, n_samples)
    runtime = compute_required.copy()  # Runtime same as compute
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required,
        'runtime': runtime
    })


def generate_lognormal_data_two_percentiles(p10_val, p90_val, n_samples):
    """Generates lognormal data with specified 10th and 90th percentile values for software improvement."""
    mu_sw, sigma_sw = get_lognorm_params_two_percentiles(p10_val, p90_val)
    # Use original mu, sigma for implementation time and compute required
    mu_orig, sigma_orig = get_lognorm_params(CI_LOW, CI_HIGH)
    
    software_improvement = np.random.lognormal(mu_sw, sigma_sw, n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu_orig, sigma_orig, n_samples)
    compute_required = np.random.lognormal(mu_orig, sigma_orig, n_samples)
    runtime = compute_required.copy()  # Runtime same as compute
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required,
        'runtime': runtime
    })


def generate_powerlaw_data(mu, sigma, n_samples, alpha=1.16):
    """Generates data where software improvement follows a power law."""
    software_improvement = pareto.rvs(b=alpha, size=n_samples) + 1.0  # Ensure all improvements >= 1
    implementation_time = np.random.lognormal(mu, sigma, n_samples)
    compute_required = np.random.lognormal(mu, sigma, n_samples)
    runtime = compute_required.copy()  # Runtime same as compute
    return pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required,
        'runtime': runtime
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
        'compute_required': lognormal_samples[:, 2],
        'runtime': lognormal_samples[:, 2].copy()  # Runtime same as compute
    })


# --- Main Simulation Loop ---
experiment_configs = [(5000, 10)] # , (50000, 2)]  # Commented out 50k experiment pools
all_trials_data = []

# Get parameters from the original script logic
CI_LOW = 0.1
CI_HIGH = 10
mu, sigma = get_lognorm_params(CI_LOW, CI_HIGH)

# Define the settings to sweep through - budget variations
settings = {
    "current_lognormal": {},
    # "budget_0.1": {"compute_budget": 0.1},  # time_budget will equal this
    # "budget_10": {"compute_budget": 10.0},   # time_budget will equal this
    # "zero_runtime": {"zero_runtime": True},  # runtime = 0 for all experiments - REMOVED
    # "power_law_sw": {},  # REMOVED
    # "corr_sw_compute_0.7": {"corr_matrix": np.array([[1.0, 0.0, 0.7], [0.0, 1.0, 0.0], [0.7, 0.0, 1.0]])},
    # "corr_all_0.7": {"corr_matrix": np.full((3, 3), 0.7) + np.diag([0.3, 0.3, 0.3])},
    # "corr_all_0.9": {"corr_matrix": np.full((3, 3), 0.9) + np.diag([0.1, 0.1, 0.1])},
    # "corr_sw_compute_0.8_time_0.7": {"corr_matrix": np.array([[1.0, 0.7, 0.8], [0.7, 1.0, 0.0], [0.8, 0.0, 1.0]])}  # NEW: sw-time 0.7, sw-compute 0.8
    
    # --- Previously tested settings (commented out) ---
    # "sw_90th_percentile_1": {"percentile_val": 1.0, "percentile": 0.9},
    # "sw_90th_percentile_100": {"percentile_val": 100.0, "percentile": 0.9},
    # "sw_p10_0.01_p90_10": {"p10_val": 0.01, "p90_val": 10.0}
}

for n_experiments, n_trials in experiment_configs:
    print(f"\n{'='*25} Running with {n_experiments} experiments and {n_trials} trials {'='*25}")
    for setting_name, setting_params in settings.items():
        print(f"\n{'#'*25} Running Setting: {setting_name} {'#'*25}")
        for trial in tqdm(range(n_trials), desc=f"Trials for {setting_name}"):

            # --- Generate Data based on Setting ---
            # if setting_name == "power_law_sw":  # REMOVED
            #     experiments_df = generate_powerlaw_data(mu, sigma, n_experiments)
            if "corr_matrix" in setting_params:
                experiments_df = generate_correlated_lognormal_data(mu, sigma, n_experiments, setting_params["corr_matrix"])
            else:
                experiments_df = generate_lognormal_data(mu, sigma, n_experiments)
            
            # Set budgets based on setting
            if "budget" in setting_name:
                base_compute_budget = setting_params["compute_budget"]
                base_time_budget = base_compute_budget  # Time budget equals compute budget
            else:
                base_compute_budget = 1.0
                base_time_budget = 1.0  # Time budget equals compute budget
            
            # Modify runtime if zero_runtime is specified - REMOVED
            # if setting_params.get("zero_runtime", False):
            #     experiments_df = experiments_df.copy()
            #     experiments_df['runtime'] = 0.0

            # --- Run simulations for this trial's data ---
            taste_levels = {
                "Optimal from all ideas (100% visibility)": 1.0,
                "Optimal from 10% of ideas (worse taste)": 0.1
            }
            
            for taste_name, taste_fraction in taste_levels.items():
                if taste_fraction < 1.0:
                    ideas_pool = experiments_df.sample(frac=taste_fraction, random_state=trial)
                else:
                    ideas_pool = experiments_df
                
                # Step 1: Get baseline improvement with default scenario
                print(f"    Running {taste_name} - default...")
                start_time = time.time()
                baseline_result = solve_greedy_knapsack(ideas_pool, base_time_budget, base_time_budget, base_compute_budget)
                baseline_improvement = baseline_result['total_improvement']
                print(f"      -> Finished in {time.time() - start_time:.2f}s, improvement: {baseline_improvement:.2e}")
                
                # Step 2: Test various compute multipliers
                compute_multipliers = [0.1, 1.0, 10, 100, 1e6]  # Added infinite compute as reference
                
                for compute_mult in tqdm(compute_multipliers, desc=f"Compute Multipliers for {taste_name}"):
                    print(f"    Finding min time budget for {compute_mult}x compute...")
                    start_time = time.time()
                    
                    # Create modified scenario with compute multiplier
                    modified_compute_budget = base_compute_budget * compute_mult
                    modified_runtime_budget = base_time_budget * compute_mult  # Scale runtime with compute
                    print(f"      -> Base compute budget: {base_compute_budget}, Modified: {modified_compute_budget}")
                    print(f"      -> Base runtime budget: {base_time_budget}, Modified: {modified_runtime_budget}")
                    print(f"      -> Implementation time budget (unchanged): {base_time_budget}")
                    print(f"      -> Target improvement: {baseline_improvement:.2e}")
                    
                    # Determine binary search range based on compute multiplier
                    if compute_mult > 1:
                        impl_time_budget_range = (base_time_budget / 100.0, base_time_budget)
                    else:
                        impl_time_budget_range = (base_time_budget, base_time_budget * 100)
                    
                    min_time_compute = find_min_implementation_time_budget(
                        ideas_pool, impl_time_budget_range, modified_runtime_budget, 
                        modified_compute_budget, baseline_improvement, base_time_budget
                    )
                    
                    speedup_compute = base_time_budget / min_time_compute
                    print(f"      -> Finished in {time.time() - start_time:.2f}s, speedup: {speedup_compute:.2f}x")
                    print(f"      -> Baseline budget: {base_time_budget}, Min budget needed: {min_time_compute:.4f}")
                    
                    all_trials_data.append({
                        'n_experiments': n_experiments,
                        'setting': setting_name,
                        'trial': trial, 
                        'taste_level': taste_name,
                        'scenario_type': 'compute_multiplier',
                        'multiplier': compute_mult,
                        'baseline_improvement': baseline_improvement,
                        'baseline_log10_improvement': baseline_result['log10_improvement'],
                        'baseline_num_experiments': baseline_result['num_experiments'],
                        'speedup': speedup_compute
                    })
                
                # Step 3: Test various implementation time multipliers (speed improvements)
                impl_speed_multipliers = [0.01, 1.0, 100]  # Simplified for debugging
                
                for impl_speed in tqdm(impl_speed_multipliers, desc=f"Implementation Time Multipliers for {taste_name}"):
                    print(f"    Finding min time budget for {impl_speed}x implementation speed...")
                    start_time = time.time()
                    
                    # Create modified scenario with implementation time multiplier
                    df_impl_modified = experiments_df.copy()
                    df_impl_modified['implementation_time'] = df_impl_modified['implementation_time'] * (1 / impl_speed)
                    
                    # Use the taste-filtered pool with modified implementation times
                    if taste_fraction < 1.0:
                        modified_ideas_pool = df_impl_modified.sample(frac=taste_fraction, random_state=trial)
                    else:
                        modified_ideas_pool = df_impl_modified
                    
                    print(f"      -> Implementation time multiplier: {impl_speed} (speed: {1/impl_speed}x)")
                    print(f"      -> Target improvement: {baseline_improvement:.2e}")
                    
                    # Determine binary search range based on implementation time multiplier
                    if impl_speed < 1:  # Faster implementation (smaller multiplier)
                        impl_time_budget_range = (base_time_budget / 100.0, base_time_budget)
                    else:  # Slower implementation (larger multiplier)
                        impl_time_budget_range = (base_time_budget, base_time_budget * 100)
                    
                    min_time_impl = find_min_implementation_time_budget(
                        modified_ideas_pool, impl_time_budget_range, base_time_budget, 
                        base_compute_budget, baseline_improvement, base_time_budget
                    )
                    
                    speedup_impl = base_time_budget / min_time_impl
                    print(f"      -> Finished in {time.time() - start_time:.2f}s, speedup: {speedup_impl:.2f}x")
                    print(f"      -> Baseline budget: {base_time_budget}, Min budget needed: {min_time_impl:.4f}")
                    
                    all_trials_data.append({
                        'n_experiments': n_experiments,
                        'setting': setting_name,
                        'trial': trial, 
                        'taste_level': taste_name,
                        'scenario_type': 'impl_time_speed',
                        'multiplier': impl_speed,
                        'baseline_improvement': baseline_improvement,
                        'baseline_log10_improvement': baseline_result['log10_improvement'],
                        'baseline_num_experiments': baseline_result['num_experiments'],
                        'speedup': speedup_impl
                    })

            # --- Inverted logic (commented out) ---
            # # --- New inverted logic: Test speedup from taste improvements across different scenarios ---
            # scenarios = {
            #     "default": {"time_budget": base_time_budget, "compute_budget": base_compute_budget},
            #     "instant_time": {"time_budget": base_time_budget, "compute_budget": base_compute_budget, "instant_time": True},
            #     "infinite_compute": {"time_budget": base_time_budget, "compute_budget": 1e6, "infinite_compute": True}
            # }
            # 
            # for scenario_name, scenario_params in scenarios.items():
            #     print(f"    Running scenario: {scenario_name}")
            #     
            #     # Modify experiments based on scenario
            #     df_modified = experiments_df.copy()
            #     if scenario_params.get("instant_time", False):
            #         df_modified['implementation_time'] = 1e-9
            #     if scenario_params.get("infinite_compute", False):
            #         df_modified['runtime'] = 0.0  # Set runtime to 0 for infinite compute
            #     
            #     # Step 1: Get baseline improvement with 10% taste at full budget
            #     print(f"      Testing 10% taste at full budget...")
            #     start_time = time.time()
            #     ideas_pool_10 = df_modified.sample(frac=0.1, random_state=trial)
            #     result_10 = solve_greedy_knapsack(
            #         ideas_pool_10, 
            #         scenario_params["time_budget"], 
            #         scenario_params["time_budget"], # Pass time_budget for runtime_budget
            #         scenario_params["compute_budget"]
            #     )
            #     baseline_improvement = result_10['total_improvement']
            #     print(f"        -> Finished in {time.time() - start_time:.2f}s, improvement: {baseline_improvement:.2e}")
            #     
            #     # Step 2: Find minimum time budget needed with 100% taste to achieve same improvement
            #     print(f"      Finding min time budget for 100% taste...")
            #     start_time = time.time()
            #     
            #     # Use all experiments for 100% taste
            #     ideas_pool_100 = df_modified
            #     
            #     # Binary search to find minimum time budget
            #     min_budget = scenario_params["time_budget"] / 100.0  # 1% of baseline
            #     max_budget = scenario_params["time_budget"]          # 100% of baseline
            #     tolerance = 0.02  # 2% tolerance
            #     
            #     # Work in log space for binary search
            #     log_min_budget = np.log(min_budget)
            #     log_max_budget = np.log(max_budget)
            #     
            #     for _ in range(20):  # Max 20 iterations for binary search
            #         log_mid_budget = (log_min_budget + log_max_budget) / 2
            #         mid_budget = np.exp(log_mid_budget)
            #         
            #         result = solve_greedy_knapsack(ideas_pool_100, mid_budget, mid_budget, scenario_params["compute_budget"])
            #         achieved_improvement = result['total_improvement']
            #         
            #         if achieved_improvement >= baseline_improvement:
            #             log_max_budget = log_mid_budget
            #         else:
            #             log_min_budget = log_mid_budget
            #         
            #         if (np.exp(log_max_budget) - np.exp(log_min_budget)) / np.exp(log_max_budget) < tolerance:
            #             break
            #     
            #     min_time_budget_100_taste = np.exp(log_max_budget)
            #     # Get final result with 100% taste at minimum budget to record num_experiments
            #     final_result_100 = solve_greedy_knapsack(ideas_pool_100, min_time_budget_100_taste, scenario_params["compute_budget"])
            #     taste_speedup = scenario_params["time_budget"] / min_time_budget_100_taste
            #     print(f"        -> Finished in {time.time() - start_time:.2f}s, speedup: {taste_speedup:.2f}x")
            #     
            #     all_trials_data.append({
            #         'n_experiments': n_experiments,
            #         'setting': setting_name,
            #         'trial': trial, 
            #         'scenario': scenario_name,
            #         'baseline_improvement_10_percent_taste': baseline_improvement,
            #         'baseline_log10_improvement_10_percent': result_10['log10_improvement'],
            #         'num_experiments_10_percent_taste': result_10['num_experiments'],
            #         'min_time_budget_100_percent_taste': min_time_budget_100_taste,
            #         'num_experiments_100_percent_taste': final_result_100['num_experiments'],
            #         'taste_speedup': taste_speedup,
            #         'log10_taste_speedup': np.log10(taste_speedup) if taste_speedup > 0 and taste_speedup != np.inf else np.inf
            #     })


# --- Final Analysis ---
results_df = pd.DataFrame(all_trials_data)
pd.set_option('display.float_format', '{:.2f}'.format)

# Console Output
print("\n\n--- Summary Statistics of Multiplier Speedup Analysis ---")
grouped_summary = results_df.groupby(['n_experiments', 'setting', 'taste_level', 'scenario_type', 'multiplier']).describe()
print(grouped_summary)

# CSV Output
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'multiplier_speedup_analysis_summary.csv')
grouped_summary.to_csv(output_path)
print(f"\nMultiplier speedup analysis summary saved to {output_path}")

# Also save the full results data
full_output_path = os.path.join(output_dir, 'multiplier_speedup_full_results.csv')
results_df.to_csv(full_output_path, index=False)
print(f"Full multiplier speedup results saved to {full_output_path}")
