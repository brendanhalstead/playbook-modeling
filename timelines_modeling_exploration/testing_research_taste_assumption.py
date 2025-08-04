import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import milp, LinearConstraint


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

# Part 1: Generate 100000 experiments
N_EXPERIMENTS = 2000
CI_LOW = 0.1
CI_HIGH = 10

mu, sigma = get_lognorm_params(CI_LOW, CI_HIGH)

software_improvement = np.random.lognormal(mu, sigma, N_EXPERIMENTS)
# Maybe make the next two correlated?
implementation_time = np.random.lognormal(mu, sigma, N_EXPERIMENTS)
compute_required = np.random.lognormal(mu, sigma, N_EXPERIMENTS)

experiments_df = pd.DataFrame({
    'software_improvement': software_improvement,
    'implementation_time': implementation_time,
    'compute_required': compute_required,
})
experiments_df['value'] = experiments_df['software_improvement'] / experiments_df['implementation_time']

# Part 2: Simulation for combinations of N and M
N_values = [1, 10, 100, 1000]
M_values = [1, 10, 100, 1000]
N_REPS = 3000

results = []

print("Running simulation...")

for M in M_values:
    # Adjust implementation time and value for the current M
    temp_df = experiments_df.copy()
    temp_df['implementation_time'] = temp_df['implementation_time'] / M
    temp_df['value'] = temp_df['software_improvement'] / temp_df['implementation_time']
    
    for N in N_values:
        rep_values = []
        for _ in range(N_REPS):
            # Pick N random experiments and find the one with the best value
            sample = temp_df.sample(n=N, replace=True)
            best_value = sample['value'].max()
            rep_values.append(best_value)
        
        # Calculate summary statistics for the 200 repetitions
        mean_val = np.mean(rep_values)
        std_val = np.std(rep_values)
        
        results.append({'M': M, 'N': N, 'mean_value': mean_val, 'std_value': std_val})

print("Simulation finished.")

# Part 3: Report summary statistics
results_df = pd.DataFrame(results)

# Pivot the results for better readability
pivot_mean_results = results_df.pivot(index='N', columns='M', values='mean_value')
pivot_std_results = results_df.pivot(index='N', columns='M', values='std_value')

print("\n--- Summary Statistics ---")
print("\nAverage value achieved:")
print(pivot_mean_results.to_string(float_format="%.2f"))

print("\nStandard deviation of value achieved:")
print(pivot_std_results.to_string(float_format="%.2f"))

# Part 4: Calculate and display relative value achieved
relative_value_df = pivot_mean_results.div(pivot_mean_results.loc[1], axis='columns')

print("\nRelative value achieved (compared to N=1):")
print(relative_value_df.to_string(float_format="%.2f"))


# Part 5: Budgeted simulation with a greedy heuristic
print("\n--- Heuristic Budgeted Simulation ---")
TIME_BUDGET = 1.0
COMPUTE_BUDGET = 1.0

# Create a copy to avoid SettingWithCopyWarning
heuristic_df = experiments_df.copy()

# Heuristic: score is log(improvement) / (time + compute)
# Add a small epsilon to the denominator to avoid division by zero
epsilon = 1e-9
heuristic_df['score'] = np.log(heuristic_df['software_improvement']) / \
                        (heuristic_df['implementation_time'] + heuristic_df['compute_required'] + epsilon)

# Sort experiments by the new score in descending order
sorted_df = heuristic_df.sort_values(by='score', ascending=False)

selected_experiments_indices = []
time_so_far = 0.0
compute_so_far = 0.0

# Greedily select experiments
for index, row in sorted_df.iterrows():
    if time_so_far + row['implementation_time'] <= TIME_BUDGET and \
       compute_so_far + row['compute_required'] <= COMPUTE_BUDGET:
        
        selected_experiments_indices.append(index)
        time_so_far += row['implementation_time']
        compute_so_far += row['compute_required']

selected_experiments = experiments_df.loc[selected_experiments_indices]

# Calculate total software improvement by multiplying them together
total_software_improvement = selected_experiments['software_improvement'].prod()
log10_total_software_improvement = np.log10(total_software_improvement) if total_software_improvement > 0 else -np.inf

print(f"\nResults for a time budget of {TIME_BUDGET} and compute budget of {COMPUTE_BUDGET}:")
print(f"Number of selected experiments: {len(selected_experiments)}")
print(f"Total time used: {time_so_far:.2f}")
print(f"Total compute used: {compute_so_far:.2f}")
print(f"Total software improvement: {total_software_improvement:e}")
print(f"Log10 of total software improvement: {log10_total_software_improvement:.2f}")


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


# --- Main Simulation Loop ---
N_TRIALS = 10
all_trials_data = []

# Get parameters from the original script logic
CI_LOW = 0.1
CI_HIGH = 10
N_EXPERIMENTS = 2000
mu, sigma = get_lognorm_params(CI_LOW, CI_HIGH)

for trial in range(N_TRIALS):
    print(f"\n--- Starting Trial {trial + 1}/{N_TRIALS} ---")

    # Generate a new set of experiments for each trial
    software_improvement = np.random.lognormal(mu, sigma, N_EXPERIMENTS)
    implementation_time = np.random.lognormal(mu, sigma, N_EXPERIMENTS)
    compute_required = np.random.lognormal(mu, sigma, N_EXPERIMENTS)

    experiments_df = pd.DataFrame({
        'software_improvement': software_improvement,
        'implementation_time': implementation_time,
        'compute_required': compute_required,
    })

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
            ideas_pool = ideas_pool = experiments_df

        # Run scenarios needed for calculations
        s = {}
        s['default'] = solve_dp(ideas_pool, base_time_budget, base_compute_budget, verbose=False)

        # 2x scenarios
        df_faster_2x = ideas_pool.copy()
        df_faster_2x['implementation_time'] /= 2
        s['faster_2x'] = solve_dp(df_faster_2x, base_time_budget, base_compute_budget, verbose=False)
        s['compute_2x'] = solve_dp(ideas_pool, base_time_budget, base_compute_budget * 2, verbose=False)
        s['both_2x'] = solve_dp(df_faster_2x, base_time_budget, base_compute_budget * 2, verbose=False)

        # 10x scenarios
        df_faster_10x = ideas_pool.copy()
        df_faster_10x['implementation_time'] /= 10
        s['faster_10x'] = solve_dp(df_faster_10x, base_time_budget, base_compute_budget, verbose=False)
        s['compute_10x'] = solve_dp(ideas_pool, base_time_budget, base_compute_budget * 10, verbose=False)
        s['both_10x'] = solve_dp(df_faster_10x, base_time_budget, base_compute_budget * 10, verbose=False)
        
        # Calculate derived quantities
        log10_default = s['default']['log10_improvement']
        
        denom_2x = s['both_2x']['log10_improvement'] - log10_default
        denom_10x = s['both_10x']['log10_improvement'] - log10_default
        
        q1_2x = (s['faster_2x']['log10_improvement'] - log10_default) / denom_2x if denom_2x != 0 else np.nan
        q1_10x = (s['faster_10x']['log10_improvement'] - log10_default) / denom_10x if denom_10x != 0 else np.nan
        
        q2_2x = (s['compute_2x']['log10_improvement'] - log10_default) / denom_2x if denom_2x != 0 else np.nan
        q2_10x = (s['compute_10x']['log10_improvement'] - log10_default) / denom_10x if denom_10x != 0 else np.nan
        
        denom_q3 = s['both_10x']['log10_improvement'] - log10_default
        q3 = (s['both_2x']['log10_improvement'] - log10_default) / denom_q3 if denom_q3 != 0 else np.nan
        
        q_both_2x_div_default = s['both_2x']['log10_improvement'] / log10_default if log10_default > 0 else np.nan
        q_both_10x_div_default = s['both_10x']['log10_improvement'] / log10_default if log10_default > 0 else np.nan
        
        all_trials_data.append({
            'trial': trial, 'taste_level': taste_name,
            'q1_2x': q1_2x, 'q1_10x': q1_10x,
            'q2_2x': q2_2x, 'q2_10x': q2_10x, 'q3': q3,
            'q_both_2x_div_default': q_both_2x_div_default,
            'q_both_10x_div_default': q_both_10x_div_default
        })

# --- Final Analysis ---
results_df = pd.DataFrame(all_trials_data)
pd.set_option('display.float_format', '{:.2f}'.format)

print("\n\n--- Summary Statistics Over 10 Trials ---")
for taste_name in taste_levels.keys():
    print(f"\n--- Taste Level: {taste_name} ---")
    taste_df = results_df[results_df['taste_level'] == taste_name]
    
    print("\nQ1: (Faster Impl - Default) / (Both - Default)")
    print(taste_df[['q1_2x', 'q1_10x']].describe())
    
    print("\nQ2: (More Compute - Default) / (Both - Default)")
    print(taste_df[['q2_2x', 'q2_10x']].describe())
    
    print("\nQ3: (Both 2x - Default) / (Both 10x - Default)")
    print(taste_df[['q3']].describe())
    
    print("\n(Log10 Both 2x) / (Log10 Default)")
    print(taste_df[['q_both_2x_div_default']].describe())
    
    print("\n(Log10 Both 10x) / (Log10 Default)")
    print(taste_df[['q_both_10x_div_default']].describe())
