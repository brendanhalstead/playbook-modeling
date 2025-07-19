import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from simple_forecasting_timelines_plotting import *
from timelines_common import *




def get_distribution_samples(config: dict, n_sims: int, correlation: float = 0.7) -> dict:
    """Generate samples from all input distributions."""
    samples = {}
    
    # First generate correlated standard normal variables for the two correlated parameters
    n_vars = 2  # horizon_doubling_time, cost_speed
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Sample initial software progress share from normal distribution
    lower, upper = config["initial_software_progress_share_ci"]
    # Convert 80% CI to normal distribution parameters
    z_low = -1.28  # norm.ppf(0.1)
    z_high = 1.28  # norm.ppf(0.9)
    mean = (lower + upper) / 2
    std = (upper - lower) / (z_high - z_low)
    # Generate samples and clip to [0.1, 0.9]
    samples["initial_software_progress_share"] = np.clip(
        np.random.normal(mean, std, n_sims),
        0.1, 0.9
    )
    
    # Sample horizon length needed for SC (in hours) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = dist.rvs(n_sims)
    
    # Sample doubling time (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["horizon_doubling_time_ci"][0],
        config["distributions"]["horizon_doubling_time_ci"][1]
    )
    samples["horizon_doubling_time"] = dist.ppf(uniform_samples[:, 0])
    
    # Sample cost and speed adjustment (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["cost_speed_ci"][0],
        config["distributions"]["cost_speed_ci"][1]
    )
    samples["cost_speed"] = dist.ppf(uniform_samples[:, 1])
    
    # Sample announcement delay (in months) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["announcement_delay_ci"][0],
        config["distributions"]["announcement_delay_ci"][1]
    )
    samples["announcement_delay"] = dist.rvs(n_sims)
    
    # Generate separate correlated samples for progress multipliers
    n_prog_vars = 2  # present_prog_multiplier, SC_prog_multiplier
    prog_corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
    prog_normal_samples = np.random.multivariate_normal(np.zeros(n_prog_vars), prog_corr_matrix, size=n_sims)
    prog_uniform_samples = norm.cdf(prog_normal_samples)
    
    # Sample present progress multiplier with correlation to SC multiplier only
    dist = get_lognormal_from_80_ci(
        config["distributions"]["present_prog_multiplier_ci"][0],
        config["distributions"]["present_prog_multiplier_ci"][1]
    )
    samples["present_prog_multiplier"] = dist.ppf(prog_uniform_samples[:, 0])
    
    # Sample SC progress multiplier with correlation to present multiplier only
    dist = get_lognormal_from_80_ci(
        config["distributions"]["SC_prog_multiplier_ci"][0],
        config["distributions"]["SC_prog_multiplier_ci"][1]
    )
    samples["SC_prog_multiplier"] = dist.ppf(prog_uniform_samples[:, 1])
    
    # Store the superexponential schedule for later use
    samples["superexponential_schedule_months"] = config["distributions"]["superexponential_schedule_months"]
    
    # Sample subexponential probability
    p_sub = config["distributions"]["p_subexponential"]
    
    # Generate independent uniform samples for growth type
    growth_type = np.random.uniform(0, 1, n_sims)
    samples["is_subexponential"] = growth_type > (1 - p_sub)
    samples["is_exponential"] = ~samples["is_subexponential"]
    samples["is_superexponential"] = ~samples["is_subexponential"]
    
    # For each simulation, determine if and when it becomes superexponential
    samples["superexponential_start_horizon"] = np.full(n_sims, np.inf)  # Default to never becoming superexponential
    for i in range(n_sims):
        if not samples["is_subexponential"][i]:  # Only consider non-subexponential cases
            # Generate a random number to determine if/when it becomes superexponential
            superexp_seed = np.random.uniform(0, 1)
            for horizon, prob in samples["superexponential_schedule_months"]:
                if superexp_seed < prob:
                    # Store the horizon directly since it's already in months
                    samples["superexponential_start_horizon"][i] = horizon
                    break
    
    # Sample se_doubling_decay_fraction from lognormal distribution
    dist = get_lognormal_from_80_ci(
        config["distributions"]["se_doubling_decay_fraction_ci"][0],
        config["distributions"]["se_doubling_decay_fraction_ci"][1]
    )
    samples["se_doubling_decay_fraction"] = np.clip(dist.rvs(n_sims), 0, 1)  # Clip to ensure decay is between 0 and 1
    
    # Add subexponential growth parameter
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]
    
    return samples

def calculate_base_time(samples: dict, current_horizon: float) -> tuple[np.ndarray, list]:
    """Calculate base time to reach SC without intermediate speedups and return time-to-horizon mappings."""
    # Convert current horizon from minutes to months
    h_current = current_horizon / (60 * 167)
    
    # Calculate number of doublings needed
    n_doublings = np.log2(samples["h_SC"]/h_current)
    
    # Print distribution statistics for the forecaster
    print("\nDistribution Statistics:")
    
    print(f"\nn_doublings:")
    print(f"  10th percentile: {np.percentile(n_doublings, 10):.2f}")
    print(f"  50th percentile: {np.percentile(n_doublings, 50):.2f}")
    print(f"  90th percentile: {np.percentile(n_doublings, 90):.2f}")
    print(f"  Mean: {np.mean(n_doublings):.2f}")
    print(f"  Std Dev: {np.std(n_doublings):.2f}")
    
    exp_mask = ~samples["is_subexponential"] & (samples["superexponential_start_horizon"] == np.inf)
    se_mask = ~samples["is_subexponential"] & (samples["superexponential_start_horizon"] < np.inf)
    sub_mask = samples["is_subexponential"]
    
    n_sims = len(n_doublings)
    # Calculate total time based on growth type
    total_times = np.zeros(n_sims)
    horizon_mappings = []  # List of progress-to-horizon mappings for each simulation
    dt_mapping = 1.0

    # For each simulation, calculate time based on growth type and superexponential transition
    for i in range(n_sims):

        #debug
        print(f"---------sample {i}-------------")
        print(f"is_subexponential: {samples['is_subexponential'][i]}")
        print(f"superexponential_start_horizon: {samples['superexponential_start_horizon'][i]}")

        mapping = []
        growth_time = 0
        horizons = []
        doubling_time = samples["horizon_doubling_time"][i]
        if samples["is_subexponential"][i]:
            # Subexponential case
            growth = samples["sub_doubling_growth_fraction"] #no index because it's the same for all simulations
            n = n_doublings[i]
            ratio = 1 + growth
            growth_time = doubling_time * (ratio**n - 1) / (ratio - 1)
            # Create time points during growth phase
            n_growth_points = max(int(growth_time / dt_mapping), 2)
            growth_times = np.linspace(0, growth_time, n_growth_points)
            # Calculate horizon at each time point for subexponential growth
            for t in growth_times:
                if t == 0:
                        horizons.append(h_current)
                else:
                    # Exact formula: solve for n doublings from t = T_t * ((1+growth)^n - 1) / growth
                    # Rearranging: (1+growth)^n = 1 + t*growth/T_t
                    # So: n = log(1 + t*growth/T_t) / log(1+growth)
                    ratio_term = 1 + t * growth / doubling_time
                    if ratio_term > 0:
                        n_doublings_at_t = np.log(ratio_term) / np.log(1 + growth)
                        horizons.append(h_current * (2 ** n_doublings_at_t))
                    else:
                        horizons.append(0.000000001) # Very small horizon
            horizons = np.array(horizons)
        else:
            # Exponential/superexponential case
            n = n_doublings[i]
            horizons = []
            
            # Check if/when it becomes superexponential
            superexponential_start = samples["superexponential_start_horizon"][i]
            if superexponential_start < h_current:
                print("WARNING: superexponential_start_horizon is less than h_current")
                print(f"superexponential_start_horizon: {superexponential_start}")
                print(f"h_current: {h_current}")
            if superexponential_start < np.inf:
                if superexponential_start > h_current:
                    # Calculate how many doublings happen before superexponential transition
                    n_before = np.log2(superexponential_start/h_current)
                    n_before = min(n_before, n)  # Can't exceed total doublings needed
                    
                    # Calculate time for exponential phase
                    time_before = n_before * doubling_time
                    n_before_points = max(int(time_before / dt_mapping), 2)
                    before_times = np.linspace(0, time_before, n_before_points)
                    # Calculate horizon at each time point for exponential growth
                    for t in before_times:
                        if t == 0:
                            horizons.append(h_current)
                        else:
                            n_doublings_at_t = t / doubling_time
                            horizons.append(h_current * (2 ** n_doublings_at_t))
                    horizon_at_se_start = horizons[-1]
                else:
                    n_before = 0
                    time_before = 0
                    before_times = np.array([0.0])  # include time 0 for alignment
                    horizon_at_se_start = h_current
                # Calculate remaining doublings after transition
                n_after = n - n_before
                if n_after > 0:
                    # Ensure we include the horizon at the start of the superexponential phase
                    if len(horizons) == 0:
                        horizons.append(horizon_at_se_start)
                    # Calculate time for superexponential phase
                    decay = samples["se_doubling_decay_fraction"][i]  # Get decay for this specific simulation
                    ratio = 1 - decay
                    time_after = doubling_time * (1 - ratio**n_after) / (1 - ratio)
                    growth_time = time_before + time_after
                    # Calculate time points during superexponential phase
                    n_after_points = max(int(time_after / dt_mapping), 2)
                    after_times_se = np.linspace(0, time_after, n_after_points)
                    # Calculate horizon at each time point for superexponential growth
                    for t in after_times_se[1:]:
                        if t == 0:
                            horizons.append(horizon_at_se_start)
                        else:
                            # Exact formula: solve for n doublings from t = T_t * (1 - (1-decay)^n) / decay
                            # Rearranging: (1-decay)^n = 1 - t*decay/T_t
                            # So: n = log(1 - t*decay/T_t) / log(1-decay)
                            ratio_term = 1 - t * decay / doubling_time
                            if ratio_term > 0:
                                n_se_doublings = np.log(ratio_term) / np.log(1 - decay)
                                horizons.append(horizon_at_se_start * (2 ** n_se_doublings))
                            else:
                                print("WARNING: singularity reached, superexponential is borked")
                                last_h = horizons[-1] if len(horizons) > 0 else horizon_at_se_start
                                print(f"most recent horizon: {last_h}")
                                assert np.isclose(t, time_after)
                                horizons.append(samples["h_SC"][i])
                    after_times = after_times_se + time_before
                    # Skip the first entry of `after_times` (which duplicates `time_before`).
                    growth_times = np.concatenate([before_times, np.asarray(after_times)[1:]])
                else:
                    print("superexponential_start_horizon is not inf but n_after is less than 0")
                    growth_time = time_before
                    growth_times = before_times
            else:
                # Pure exponential case
                growth_time = n * doubling_time
                n_growth_points = max(int(growth_time / dt_mapping), 2)
                growth_times = np.linspace(0, growth_time, n_growth_points)

                # Calculate horizon at each time point for exponential growth
                for t in growth_times:
                    if t == 0:
                        horizons.append(h_current)
                    else:
                        n_doublings_at_t = t / doubling_time
                        horizons.append(h_current * (2 ** n_doublings_at_t))
        
        
        # Ensure the mapping is strictly chronological so downstream interpolation works correctly.
        # If horizons is still empty (can occur when superexponential_start_horizon < h_current and n_after <= 0),
        # seed it with the current horizon so downstream code has at least one point.
        if len(horizons) == 0:
            horizons = [h_current]
            growth_times = [0.0]

        growth_times = np.array(growth_times)
        horizons = np.array(horizons)

        # If lengths mismatch (rare edge case), pad horizons with last known value.
        if len(horizons) < len(growth_times):
            pad_val = horizons[-1]
            horizons = np.append(horizons, np.full(len(growth_times) - len(horizons), pad_val))
        elif len(growth_times) < len(horizons):
            pad_val = growth_times[-1]
            growth_times = np.append(growth_times, np.full(len(horizons) - len(growth_times), pad_val))

        sort_idx = np.argsort(growth_times)
        growth_times = growth_times[sort_idx]
        horizons = horizons[sort_idx]

        for t, h in zip(growth_times, horizons):
            mapping.append((t, h * 60 * 167))

        print(f"h_SC: {samples['h_SC'][i]}")
        # print(f"horizons[-1]: {horizons[-1]}")
        print(f"len_growth_times: {len(growth_times)}")
        print(f"len_horizons: {len(horizons)}")
        print(f"len_mapping: {len(mapping)}")
        # if not np.isclose(horizons[-1], samples["h_SC"][i], atol=0.05*samples["h_SC"][i]):
        #     if horizons[-1] > samples["h_SC"][i]:
        #         print("WARNING: horizons[-1] is greater than h_SC")
        #     else:
        #         print("PROBLEM: horizons[-1] is less than h_SC")
        #         assert False
        assert len(horizons) == len(growth_times)
        assert len(mapping) == len(growth_times)
        # Add cost and speed adjustment
        total_times[i] = growth_time + samples["cost_speed"][i]
    
        # Add cost and speed adjustment to mappings
        cost_speed_time = samples["cost_speed"][i]
        if cost_speed_time > 0:
            final_horizon = samples["h_SC"][i] * 60 * 167  # Convert to minutes
            growth_time = total_times[i] - cost_speed_time  # Get the growth time
            cost_speed_times = np.linspace(growth_time, total_times[i], 
                                         max(int(cost_speed_time / dt_mapping), 2))
            for t in cost_speed_times[1:]:  # Skip first point to avoid duplication
                mapping.append((t, final_horizon))
        horizon_mappings.append(mapping)
    
    # Ensure all times are non-negative and finite
    total_times = np.where(np.isfinite(total_times), total_times, 0)
    total_times = np.maximum(total_times, 0)
    
    # Print time distribution by growth type
    print("\nTime Distribution by Growth Type:")
    for mask, name in [(exp_mask, "Exponential"), (se_mask, "Superexponential"), (sub_mask, "Subexponential")]:
        if np.any(mask):
            times = total_times[mask]
            print(f"\n{name}:")
            print(f"  10th percentile: {np.percentile(times, 10):.2f} months")
            print(f"  50th percentile: {np.percentile(times, 50):.2f} months")
            print(f"  90th percentile: {np.percentile(times, 90):.2f} months")
            print(f"  Mean: {np.mean(times):.2f} months")
            print(f"  Std Dev: {np.std(times):.2f} months")
    
    # Print overall time distribution
    print("\nOverall Time Distribution:")
    print(f"  10th percentile: {np.percentile(total_times, 10):.2f} months")
    print(f"  50th percentile: {np.percentile(total_times, 50):.2f} months")
    print(f"  90th percentile: {np.percentile(total_times, 90):.2f} months")
    print(f"  Mean: {np.mean(total_times):.2f} months")
    print(f"  Std Dev: {np.std(total_times):.2f} months")
    
    return total_times, horizon_mappings

def get_compute_rate(t: float, compute_schedule: list) -> float:
    """Calculate compute progress rate based on time and compute schedule.
    
    Args:
        t: Current time in years
        compute_schedule: List of [year, rate] pairs, sorted by year
    """
    # Default rate is 1.0
    current_rate = 1.0
    
    # Find the most recent schedule entry that applies
    for year, rate in compute_schedule:
        if t >= year:
            current_rate = rate
        else:
            break
            
    return current_rate

def get_labor_growth_rate(t: float, labor_growth_schedule: list) -> float:
    """Calculate labor growth rate based on time and labor growth schedule.
    
    Args:
        t: Current time in years
        labor_growth_schedule: List of [year, rate] pairs, sorted by year
    """
    # Default rate is 0.5 (same as before)
    current_rate = 0.5
    
    # Find the most recent schedule entry that applies
    for year, rate in labor_growth_schedule:
        if t >= year:
            current_rate = rate
        else:
            break
            
    return current_rate

def calculate_sc_arrival_year_with_trajectories(samples: dict, current_horizon: float, dt: float, human_alg_progress_decrease_date: float, max_simulation_years: float, forecaster_config: dict, simulation_config: dict) -> tuple[np.ndarray, list]:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling."""
    # First calculate base time including cost-and-speed adjustment
    base_time_in_months, horizon_mappings = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)
    
    # Get software progress share from samples
    software_progress_share = samples["initial_software_progress_share"]
    
    # Initialize array for actual times
    ending_times = np.zeros(n_sims)
    
    # Store trajectories for each simulation
    trajectories = []
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = simulation_config["start_year"]
    # Convert dt from days to months
    dt_in_months = dt / 30.5
    
    max_time = simulation_config["max_time"]
    baseline_growths = []
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
        progress = 0.0
        
        # Initialize trajectory list for this simulation
        trajectory = []
        
        # If base time is zero (already at target horizon), record current state and continue
        if base_time_in_months[i] == 0:
            ending_times[i] = time  # No additional time required
            trajectories.append(trajectory)
            continue
        
        # Initialize labor-based research variables
        labor_pool = simulation_config["initial_labor_pool"]
        research_stock = simulation_config["initial_research_stock"]
        labor_power = simulation_config["labor_power"]
        
        # Track previous labor growth rate to detect changes
        prev_labor_growth_rate = None
        
        # Counter for iteration tracking
        iteration_count = 0
        
        while progress < base_time_in_months[i] and time < max_time:

            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            current_horizon_minutes = get_horizon_at_progress(horizon_mappings[i], progress)
            trajectory.append((time+samples["announcement_delay"][i]/12, current_horizon_minutes))
            
            # Calculate software speedup based on intermediate speedup s(interpolate between present and SC rates)
            if forecaster_config["patch_rd_speedup"]:
                software_prog_multiplier = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** progress_fraction
            else:
                software_prog_multiplier = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # Get current labor growth rate from schedule
            current_labor_growth_rate = get_labor_growth_rate(time, forecaster_config["labor_growth_schedule"])
            # Convert annual growth rate to daily rate for the time step
            daily_growth_rate = (1 + current_labor_growth_rate) ** (dt/250) - 1

            # Calculate new labor added this period
            new_labor = labor_pool * daily_growth_rate
            labor_pool += new_labor
            
            # Calculate research contribution on a yearly basis, then divide
            research_contribution = ((((labor_pool+1) ** labor_power)-1) * software_prog_multiplier) / (250/dt)
            # Add to research stock
            new_research_stock = research_stock + research_contribution
            # Calculate actual growth rate (annualized)
            # actual_growth = (new_research_stock / research_stock) ** (250/dt) - 1
            actual_growth = new_research_stock / research_stock

            if progress == 0:
                baseline_growth = actual_growth
                baseline_growths.append(baseline_growth)
            
            # Calculate adjustment factor based on growth rate ratio
            # current RS-doubling-rate / baseline RS-doubling-rate
            # growth_ratio = np.log(1 + actual_growth) / np.log(1 + baseline_growth)
            growth_ratio = np.log(actual_growth) / np.log(baseline_growth)
            
            # Get compute rate for current time using compute schedule
            compute_rate = get_compute_rate(time, forecaster_config["compute_schedule"])
            # Total rate is weighted average of growth_ratio and compute rates
            total_rate = software_progress_share[i] * growth_ratio + (1 - software_progress_share[i]) * compute_rate
            # Update progress and time
            progress += dt_in_months * total_rate
            time += dt_in_months / 12  # Convert months to years
            
            # Update research stock
            research_stock = new_research_stock
            
            # Increment iteration counter
            iteration_count += 1

        # If we hit the time limit, set to max time
        if time >= max_time:
            time = max_time 
            # print(f"time is greater than max_time: {time} > {max_time}")
            
        ending_times[i] = time
        trajectories.append(trajectory)

    # Ensure baseline_growths has one entry per simulation (edge case: zero base time simulations)
    if len(baseline_growths) < n_sims:
        # Fill missing entries with the last observed baseline growth or a small positive fallback
        fallback_growth = baseline_growths[-1] if baseline_growths else 1e-6
        missing = n_sims - len(baseline_growths)
        baseline_growths.extend([fallback_growth] * missing)

    print(f"ending_times: {ending_times}")
    return ending_times, trajectories, baseline_growths

def backcast_base_time(samples: dict, current_horizon: float, dt: float, backcast_years: int = 5) -> list:
    """Backcast base time for each sample."""
    h_current = current_horizon / (60 * 167)
    n_sims = len(samples["h_SC"])
    dt_mapping = 1.0
    horizon_mappings = []
    for i in range(n_sims):
        mapping = []
        
        n_backcast_points = max(int(backcast_years*12 / dt_mapping), 2)
        
        doubling_time = samples["horizon_doubling_time"][i]
        if samples["is_subexponential"][i]:
            # Subexponential: properly inverted for backward time
            doubling_time = samples["horizon_doubling_time"][i]
            growth = samples["sub_doubling_growth_fraction"]
            horizons = []
            growth_times = np.linspace(-backcast_years*12, 0, n_backcast_points)
            for t in growth_times:
                if t == 0:
                    horizons.append(h_current)
                else:
                    # For backward time, we reverse the subexponential progression
                    # Instead of (1+growth)^n = 1 + t*growth/T_t, we use:
                    # We want to find what n_doublings gives us the current horizon when projected backward
                    # If forward: h(t) = h0 * 2^n where n comes from subexponential formula
                    # Then backward: we solve for how many doublings would have been achieved by time t
                    # For subexponential going backward: each step back requires MORE time
                    # So we use: (1+growth)^n = 1 / (1 + |t|*growth/T_t)
                    ratio_term = 1 + t * growth / doubling_time
                    if ratio_term > 0:
                        # This gives us a negative n_doublings for backward time
                        n_doublings = np.log(ratio_term) / np.log(1 + growth)
                        horizons.append(h_current * (2 ** n_doublings))
                    else:
                        horizons.append(0.000000001)  # Very small horizon
            
        
        else:
            # Exponential/superexponential case
            horizons = []
            decay = samples["se_doubling_decay_fraction"][i]
            ratio = 1 - decay
            # Check if/when it becomes superexponential
            superexponential_start = samples["superexponential_start_horizon"][i]
            if superexponential_start < h_current:
                # print("Superexp -> exp transition detected in backcast_base_time")
                # Calculate how many doublings happen after superexponential transition
                n_after = np.log2(superexponential_start/h_current)
                
                ratio = 1 - decay
                time_after = doubling_time * (1 - ratio**n_after) / (1 - ratio)
                time_after = max(time_after, -backcast_years*12)
                n_after_points = max(int(-time_after / dt_mapping), 2)

                # Time grid for the super-exponential segment (from the transition point up to today)
                after_times_full = np.linspace(time_after, 0, n_after_points)

                horizons_after = []
                for t in after_times_full:
                    if t == 0:
                        horizons_after.append(h_current)
                    else:
                        # Same formula as forward: solve for n doublings from
                        #   t = T_t * (1 - (1-decay)^n) / decay
                        # Rearranging: (1-decay)^n = 1 - t*decay/T_t
                        # Therefore: n = log(1 - t*decay/T_t) / log(1-decay)
                        ratio_term = 1 - t * decay / doubling_time
                        if ratio_term > 0:
                            n_se_doublings = np.log(ratio_term) / np.log(1 - decay)
                            horizons_after.append(h_current * (2 ** n_se_doublings))
                        else:
                            # If ratio_term <= 0, we're outside the valid range
                            assert False
                            horizons_after.append(0.000000001)  # Very small horizon

                # The first entry corresponds to `time_after` – the end of the exponential phase.
                horizon_at_exp_end = horizons_after[0]

                # Exclude the first entry when later concatenating, to avoid duplicates.
                after_times = after_times_full[1:]
                horizons_after = horizons_after[1:]

                # We'll build the full `horizons` list later by concatenating
                # the exponential-phase horizons with `horizons_after`.
                horizons = []

                # Earliest time in the backcast window
                time_before = -backcast_years * 12  # e.g. −60 months for 5 years
                assert time_before <= 0

                n_before_points = max(int((time_after - time_before) / dt_mapping), 2)

                # Create a time grid for the exponential phase that spans
                # `[time_before, time_after]`.  This removes the need for the
                # previous post-hoc time shift and keeps horizons properly aligned.
                before_times_exp = np.linspace(time_before, time_after, n_before_points)

                # Adjust doubling time to the point where the super-exponential phase ends
                doubling_time_before = doubling_time * (1 - decay * time_after / doubling_time)

                # Horizons during the purely-exponential phase.
                # We want horizons to grow as we move forward in time (toward 0).
                # Let Δt = t - time_after; at t = time_after, Δt = 0 ⇒ horizon = horizon_at_exp_end.
                # Each doubling takes `doubling_time_before`, so the number of doublings is Δt / doubling_time_before.
                n_doublings_exp = (before_times_exp - time_after) / doubling_time_before
                horizons_exp = horizon_at_exp_end * (2 ** n_doublings_exp)
                horizons.extend(horizons_exp.tolist())
                # Append super-exponential segment horizons, then build matching time array.
                horizons.extend(horizons_after)

                # Build time array (skip duplicated transition point)
                growth_times = np.concatenate([before_times_exp, np.asarray(after_times)])
            else:
                # print("Superexp -> exp transition not detected in backcast_base_time")
                after_times = []
                horizons = []
                time_after = 0
                horizon_at_exp_end = h_current
                horizons_after = []  # Ensure the variable exists so later code can safely reference it
            # Determine how far back the exponential phase extends before the
            # super-exponential transition.
            time_before = -backcast_years * 12  # e.g. −60 months for 5 years
            assert time_before <= 0

            n_before_points = max(int((time_after - time_before) / dt_mapping), 2)

            # Create a time grid for the exponential phase that spans
            # `[time_before, time_after]`.  This removes the need for the
            # previous post-hoc time shift and keeps horizons properly aligned.
            before_times_exp = np.linspace(time_before, time_after, n_before_points)

            # Adjust doubling time to the point where the super-exponential phase ends
            doubling_time_before = doubling_time * (1 - decay * time_after / doubling_time)

            # Horizons during the purely-exponential phase.
            # We want horizons to grow as we move forward in time (toward 0).
            # Let Δt = t - time_after; at t = time_after, Δt = 0 ⇒ horizon = horizon_at_exp_end.
            # Each doubling takes `doubling_time_before`, so the number of doublings is Δt / doubling_time_before.
            n_doublings_exp = (before_times_exp - time_after) / doubling_time_before
            horizons_exp = horizon_at_exp_end * (2 ** n_doublings_exp)
            horizons.extend(horizons_exp.tolist())
            # Append super-exponential segment horizons (empty when no SE phase).
            horizons.extend(horizons_after)

            # Build time array (skip duplicated transition point)
            growth_times = np.concatenate([before_times_exp, np.asarray(after_times)])
        horizons = np.array(horizons)
        for t, h in zip(growth_times, horizons):
            mapping.append((t, h * 60 * 167))
        horizon_mappings.append(mapping)
        # print(f"--------appending mapping {i}--------")
        # print(f"subexponential: {samples['is_subexponential'][i]}")
        # if not samples["is_subexponential"][i]:
        #     print(f"superexponential_start_horizon: {samples['superexponential_start_horizon'][i]}")
        #     print(f"doubling_time: {doubling_time}")
        #     print(f"decay: {decay}")
        #     print(f"after_times: {after_times}")
        #     print(f"before_times_exp: {before_times_exp}")
        #     print(f"growth_times: {growth_times}")
        #     print(f"doubling_time_before: {doubling_time_before}")
        #     print(f"time_after: {time_after}")
        #     print(f"time_before: {time_before}")
        # assert len(mapping) > 0
        # print(f"mapping: {mapping}")
    return horizon_mappings

def backcast_trajectories(samples: dict, current_horizon: float, dt: float, backcast_years: int = 5, forecaster_config: dict = None, simulation_config: dict = None, baseline_growths: list = None) -> list:
    """Backcast trajectories for each sample."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, _ = calculate_base_time(samples, current_horizon)
    horizon_mappings = backcast_base_time(samples, current_horizon, dt, backcast_years)
    n_sims = len(samples["h_SC"])
    trajectories = []
    current_year = 2025.25
    dt = -dt
    dt_in_months = dt / 30.5
    min_progress = - backcast_years * 12
    software_progress_share = samples["initial_software_progress_share"]

    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
        progress = 0.0
        trajectory = []

        # Skip simulations with zero base time to avoid divide-by-zero errors
        if base_time_in_months[i] == 0:
            trajectories.append(trajectory)
            continue

        # Initialize labor-based research variables
        labor_pool = simulation_config["initial_labor_pool"]
        research_stock = simulation_config["initial_research_stock"]
        labor_power = simulation_config["labor_power"]

        # Track previous labor growth rate to detect changes
        prev_labor_growth_rate = None
        baseline_growth = baseline_growths[i]

        # Counter for iteration tracking
        iteration_count = 0

        while progress > min_progress and time > 2020:
            # print(f"--------iteration {iteration_count}--------")
            # print(f"progress: {progress}")
            # print(f"time: {time}")
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            
            current_horizon_minutes = get_horizon_at_progress(horizon_mappings[i], progress)
            trajectory.append((time+samples["announcement_delay"][i]/12, current_horizon_minutes))

            # Calculate software speedup based on intermediate speedup s(interpolate between present and SC rates)
            if forecaster_config["patch_rd_speedup"]:
                software_prog_multiplier = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** progress_fraction
            else:
                software_prog_multiplier = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction
            # Get current labor growth rate from schedule
            current_labor_growth_rate = get_labor_growth_rate(time, forecaster_config["labor_growth_schedule"])
            # Convert annual growth rate to daily rate for the time step
            daily_growth_rate = (1 + current_labor_growth_rate) ** (dt/250) - 1
            # print(f"daily_growth_rate: {daily_growth_rate}")
            # Calculate new labor added this period
            new_labor = labor_pool * daily_growth_rate
            labor_pool += new_labor
            # Calculate research contribution on a yearly basis, then divide
            research_contribution = ((((labor_pool+1) ** labor_power)-1) * software_prog_multiplier) / (250/dt)
            # Add to research stock
            new_research_stock = research_stock + research_contribution
            # print(f"new_research_stock: {new_research_stock}")
            # Calculate actual growth rate (annualized)
            actual_growth = (new_research_stock / research_stock) ** (250/(dt)) - 1
            

            # Calculate adjustment factor based on growth rate ratio
            # Using log ratio to properly account for compound growth
            growth_ratio = np.log(1 + actual_growth) / np.log(1 + baseline_growth)
            # Get compute rate for current time using compute schedule
            compute_rate = 1

            # Total rate is weighted average of software and compute rates
            total_rate = software_progress_share[i] * growth_ratio + (1 - software_progress_share[i]) * compute_rate
            # Update progress and time
            progress += dt_in_months * total_rate
            time += dt_in_months / 12  # Convert months to years
            
            # Update research stock
            research_stock = new_research_stock
            
            # Increment iteration counter
            iteration_count += 1

        trajectories.append(trajectory)
    return trajectories
                

def run_simple_sc_simulation(config_path: str = "simple_params_may.yaml") -> tuple[plt.Figure, dict]:
    """Run simplified SC simulation and plot results."""
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year_decimal = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year_decimal = config["simulation"]["start_year"]
    # Store results for each forecaster
    all_forecaster_results = {}
    all_forecaster_samples = {}
    all_forecaster_trajectories = {}
    all_forecaster_backcast_trajectories = {}
    all_forecaster_baseline_growths = {}
    # Run simulations for each forecaster
    print("\nRunning simulations for each forecaster...")
    with tqdm(total=len(config["forecasters"]), desc="Processing forecasters") as pbar:
        for _, forecaster_config in config["forecasters"].items():
            name = forecaster_config["name"]
            pbar.set_description(f"Processing {name}")
            
            # Generate samples
            samples = get_distribution_samples(forecaster_config, config["simulation"]["n_sims"])
            all_forecaster_samples[name] = samples
            
            # Calculate time to SC
            all_forecaster_results[name], all_forecaster_trajectories[name], all_forecaster_baseline_growths[name] = calculate_sc_arrival_year_with_trajectories(
                samples, 
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                config["simulation"]["human_alg_progress_decrease_date"],
                config["simulation"]["max_simulation_years"],
                forecaster_config,
                config["simulation"]
            )

            # Generate backcasted trajectories
            print(f"Generating backcasted trajectories for {name}...")
            backcast_trajectories_result = backcast_trajectories(
                samples,
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                backcast_years=5,
                forecaster_config=forecaster_config,
                simulation_config=config["simulation"],
                baseline_growths=all_forecaster_baseline_growths[name]
            )
            all_forecaster_backcast_trajectories[name] = backcast_trajectories_result
            
            # Print percentage of subexponential simulations
            subexponential_percentage = np.mean(samples["is_subexponential"]) * 100
            print(f"\n{name} subexponential percentage: {subexponential_percentage:.1f}%")
            
            pbar.update(1)
    
    # Print debug information grouped by parameter
    # print("\nDebug Information by Parameter:")
    # print("=" * 80)
    
    # Define parameters to analyze
    # parameters = {
    #     "h_SC": "Time Horizon for SC (months)",
    #     "T_t": "Doubling Time (months)",
    #     "cost_speed": "Cost and Speed Adjustment (months)",
    #     "announcement_delay": "Announcement Delay (months)",
    # }
    
    # for param, description in parameters.items():
    #     print(f"\n{description}:")
    #     print("-" * 40)
    #     for name, samples in all_forecaster_samples.items():
    #         if param.startswith("is_"):
    #             # For boolean parameters, calculate percentage
    #             value = np.mean(samples[param]) * 100
    #             print(f"{name:>10}: {value:>6.1f}%")
    #         else:
    #             # For numeric parameters, show percentiles
    #             data = samples[param]
    #             print(f"{name:>10}:")
    #             print(f"          10th: {np.percentile(data, 10):>6.2f}")
    #             print(f"          50th: {np.percentile(data, 50):>6.2f}")
    #             print(f"          90th: {np.percentile(data, 90):>6.2f}")
    
    print("\nGenerating plot...")
    # Create and save plot
    fig = plot_results(all_forecaster_results, config)
    
    # Create output directory with current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # print("\nSaving plot...")
    # Save plot
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    
    # Close figure to free memory
    plt.close(fig)
    for forecaster_name in all_forecaster_results.keys():
        # --- Figures that are independent of a specific SC month ---
        fig_backcasted_colored = plot_backcasted_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=True,
            forecaster_filter=[forecaster_name],
        )

        fig_backcasted_red = plot_backcasted_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=False,
            forecaster_filter=[forecaster_name],
        )

        fig_combined_colored = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=True,
            forecaster_filter=[forecaster_name],
        )

        fig_combined_red = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=False,
            forecaster_filter=[forecaster_name],
        )

        # Save and close the month-independent figures
        fig_backcasted_colored.savefig(
            output_dir / f"backcasted_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_backcasted_red.savefig(
            output_dir / f"backcasted_trajectories_red_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_colored.savefig(
            output_dir / f"combined_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_red.savefig(
            output_dir / f"combined_trajectories_red_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig_backcasted_colored)
        plt.close(fig_backcasted_red)
        plt.close(fig_combined_colored)
        plt.close(fig_combined_red)

        # --- Figures filtered by specific SC arrival months ---
        target_months = [
            # March targets
            "February 2029",
            # "March 2028",
            # "March 2029",
            # "March 2030",
            # September targets
            # "September 2027",
            # "September 2028",
            # "September 2029",
            # "September 2030",
        ]

        # Collect central trajectories per month for later comparison
        central_trajs_by_month: dict[str, dict] = {}

        for sc_month_str in target_months:
            month_slug = sc_month_str.lower().replace(" ", "_")  # e.g. "march_2028"

            # Trajectory plot filtered by SC month
            fig_trajectories = plot_trajectories_sc_month(
                all_forecaster_results,
                all_forecaster_trajectories,
                all_forecaster_samples,
                config,
                sc_month_str=sc_month_str,
                forecaster_filter=[forecaster_name],
            )

            # Combined trajectory plots filtered by SC month
            fig_combined_month, central_path = plot_combined_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                forecaster_filter=[forecaster_name],
            )

            # Store central trajectory if available
            if central_path is not None:
                central_trajs_by_month[sc_month_str] = central_path

            fig_combined_month_median, _ = plot_combined_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                plot_median_curve=True,
                forecaster_filter=[forecaster_name],
            )

            fig_combined_month_illustrative, _ = plot_combined_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                overlay_illustrative_trend=True,
                forecaster_filter=[forecaster_name],
            )

            # --- Save the month-specific figures ---
            fig_trajectories.savefig(
                output_dir / f"{month_slug}_trajectories_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month.savefig(
                output_dir / f"combined_trajectories_{month_slug}_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_median.savefig(
                output_dir / f"combined_trajectories_{month_slug}_median_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_illustrative.savefig(
                output_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Close month-specific figures to free memory
            plt.close(fig_trajectories)
            plt.close(fig_combined_month)
            plt.close(fig_combined_month_median)
            plt.close(fig_combined_month_illustrative)

        print(f"\nSaved all trajectory plots (including month-specific versions) for {forecaster_name}.")

        # --------------------------------------------------------
        # Plot comparison of central trajectories across months
        # --------------------------------------------------------
        if central_trajs_by_month:
            central_list = sorted(central_trajs_by_month.items())  # list[(label, traj)]
            fig_cent_compare = plot_central_trajectories_comparison(
                central_list,
                config,
                overlay_external_data=True,
            )

            fig_cent_compare.savefig(
                output_dir / f"central_trajectories_comparison_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig_cent_compare)

    return fig, all_forecaster_results

if __name__ == "__main__":
    run_simple_sc_simulation()
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 