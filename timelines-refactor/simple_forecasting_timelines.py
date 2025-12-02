import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm, rankdata
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from simple_forecasting_timelines_plotting import *
from collections import OrderedDict
from timelines_common import *




def get_distribution_samples(config: dict, n_sims: int, correlation: float = 0.7) -> dict:
    """Generate samples from all input distributions."""
    samples = {}
    
    # First generate correlated standard normal variables for the three correlated parameters
    n_vars = 4  # T_t, cost_speed, inverse of p_superexponential, present_prog_multiplier, SC_prog_multiplier
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Sample horizon length needed for SC (in hours) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = dist.rvs(n_sims)
    
    # Sample doubling time (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["T_t_ci"][0],
        config["distributions"]["T_t_ci"][1]
    )
    samples["T_t"] = dist.ppf(uniform_samples[:, 0])
    
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
    
    # Sample growth types with correlation for p_superexponential
    p_super = config["distributions"]["p_superexponential"]
    p_sub = config["distributions"]["p_subexponential"]
    
    # Use the correlated uniform sample to determine p_superexponential
    growth_type = uniform_samples[:, 2]
    samples["is_superexponential"] = growth_type < p_super
    samples["is_subexponential"] = (growth_type >= p_super) & (growth_type < (p_super + p_sub))
    samples["is_exponential"] = ~(samples["is_superexponential"] | samples["is_subexponential"])
    
    # Add growth/decay parameters
    samples["se_doubling_decay_fraction"] = config["distributions"]["se_doubling_decay_fraction"]
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]

    # Patch R&D speedup
    samples["patch_rd_speedup"] = np.full(n_sims, config["distributions"]["patch_rd_speedup"])

    return samples


def get_median_samples(config: dict) -> dict:
    """Generate a single sample using median values for all parameters."""
    samples = {}

    # Get median (50th percentile) for each lognormal distribution
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["T_t_ci"][0],
        config["distributions"]["T_t_ci"][1]
    )
    samples["T_t"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["cost_speed_ci"][0],
        config["distributions"]["cost_speed_ci"][1]
    )
    samples["cost_speed"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["announcement_delay_ci"][0],
        config["distributions"]["announcement_delay_ci"][1]
    )
    samples["announcement_delay"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["present_prog_multiplier_ci"][0],
        config["distributions"]["present_prog_multiplier_ci"][1]
    )
    samples["present_prog_multiplier"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["SC_prog_multiplier_ci"][0],
        config["distributions"]["SC_prog_multiplier_ci"][1]
    )
    samples["SC_prog_multiplier"] = np.array([dist.median()])

    # Use exponential growth for median trajectory (most likely single outcome)
    samples["is_superexponential"] = np.array([False])
    samples["is_subexponential"] = np.array([False])
    samples["is_exponential"] = np.array([True])

    # Growth/decay parameters
    samples["se_doubling_decay_fraction"] = config["distributions"]["se_doubling_decay_fraction"]
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]

    # Patch R&D speedup
    samples["patch_rd_speedup"] = np.array([config["distributions"]["patch_rd_speedup"]])

    return samples


def format_horizon(h: float) -> str:
    """Format horizon value to human-readable string."""
    if h < 1:
        return f"{h*60:.1f} seconds"
    elif h < 60:
        return f"{h:.1f} minutes"
    elif h < 1440:
        return f"{h/60:.1f} hours"
    elif h < 10080:
        return f"{h/1440:.1f} days"
    elif h < 43200:
        return f"{h/10080:.1f} weeks"
    else:
        return f"{h/43200:.1f} months"


def run_median_trajectory(config: dict, forecaster_config: dict, forecaster_name: str) -> str:
    """Run trajectories with median parameters for both exponential and superexponential growth."""
    # Create config with this forecaster's distributions
    forecaster_dist_config = {"distributions": forecaster_config["distributions"]}

    # Get median samples (starts as exponential)
    median_samples = get_median_samples(forecaster_dist_config)

    # Build output string
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"MEDIAN TRAJECTORIES FOR: {forecaster_name}")
    lines.append(f"{'='*60}")
    lines.append(f"Median Parameters:")
    lines.append(f"  h_SC (work-months): {median_samples['h_SC'][0]:.2f}")
    lines.append(f"  T_t (doubling time, months): {median_samples['T_t'][0]:.2f}")
    lines.append(f"  cost_speed (months): {median_samples['cost_speed'][0]:.2f}")
    lines.append(f"  announcement_delay (months): {median_samples['announcement_delay'][0]:.2f}")
    lines.append(f"  present_prog_multiplier: {median_samples['present_prog_multiplier'][0]:.3f}")
    lines.append(f"  SC_prog_multiplier: {median_samples['SC_prog_multiplier'][0]:.2f}")
    lines.append(f"  patch_rd_speedup: {median_samples['patch_rd_speedup'][0]}")
    lines.append(f"  se_doubling_decay_fraction: {median_samples['se_doubling_decay_fraction']}")

    # Run simulation parameters
    current_horizon = config["simulation"]["current_horizon"]
    dt = config["simulation"]["dt"]
    compute_decrease_date = config["simulation"]["compute_decrease_date"]
    human_alg_progress_decrease_date = config["simulation"]["human_alg_progress_decrease_date"]
    max_simulation_years = config["simulation"]["max_simulation_years"]

    # Run both exponential and superexponential trajectories
    for growth_type in ["Exponential", "Superexponential"]:
        # Set growth type flags
        if growth_type == "Exponential":
            median_samples["is_exponential"] = np.array([True])
            median_samples["is_superexponential"] = np.array([False])
            median_samples["is_subexponential"] = np.array([False])
        else:
            median_samples["is_exponential"] = np.array([False])
            median_samples["is_superexponential"] = np.array([True])
            median_samples["is_subexponential"] = np.array([False])

        lines.append(f"\n--- {growth_type} Growth ---")

        # Run simulation
        ending_times, trajectories = calculate_sc_arrival_year_with_trajectories(
            median_samples, current_horizon, dt,
            compute_decrease_date, human_alg_progress_decrease_date, max_simulation_years
        )

        arrival_year = ending_times[0]
        trajectory = trajectories[0]

        lines.append(f"SC Arrival: {format_year_month(arrival_year)}")
        lines.append(f"\nTrajectory (selected points):")
        lines.append(f"  {'Year':<12} {'Horizon':<20}")
        lines.append(f"  {'-'*12} {'-'*20}")

        # Print trajectory at key points
        if trajectory:
            n_points = len(trajectory)
            # Show ~10 evenly spaced points
            step = max(1, n_points // 10)
            for i in range(0, n_points, step):
                t, h = trajectory[i]
                lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20}")

            # Always show final point
            if n_points > 1:
                t, h = trajectory[-1]
                lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20} (final)")

    lines.append(f"\n{'='*60}\n")

    result = "\n".join(lines)
    print(result)
    return result


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
    
    # Print growth type distribution
    exp_mask = samples["is_exponential"]
    se_mask = samples["is_superexponential"]
    sub_mask = samples["is_subexponential"]
    
    n_sims = len(n_doublings)
    
    total_time = np.zeros(n_sims)
    horizon_mappings = []  # List of time-to-horizon mappings for each simulation
    
    # Use 1 month resolution for efficiency
    dt_mapping = 1.0
    
    
    # Vectorized calculation of base growth time (before cost_speed)
    growth_time = np.zeros(n_sims)
    
    # For regular exponential cases
    growth_time[exp_mask] = n_doublings[exp_mask] * samples["T_t"][exp_mask]
    
    # For superexponential cases - use analytical formula
    if np.any(se_mask):
        decay = samples["se_doubling_decay_fraction"]
        first_doubling_time = samples["T_t"][se_mask]
        n = n_doublings[se_mask]
        ratio = 1 - decay
        # Sum of geometric series: T1 * (1 - r^n) / (1 - r)
        growth_time[se_mask] = first_doubling_time * (1 - ratio**n) / (1 - ratio)
    
    # For subexponential cases - use analytical formula
    if np.any(sub_mask):
        growth = samples["sub_doubling_growth_fraction"]
        first_doubling_time = samples["T_t"][sub_mask]
        n = n_doublings[sub_mask]
        ratio = 1 + growth
        # Sum of geometric series: T1 * (r^n - 1) / (r - 1)
        growth_time[sub_mask] = first_doubling_time * (ratio**n - 1) / (ratio - 1)
    
    # Total time includes cost_speed adjustment
    total_time = growth_time + samples["cost_speed"]
    
    # Create efficient horizon mappings
    for i in range(n_sims):
        mapping = []
        growth_time_i = growth_time[i]
        cost_speed_time = samples["cost_speed"][i]
        
        # Create time points during growth phase
        if growth_time_i > 0:
            n_growth_points = max(int(growth_time_i / dt_mapping), 2)
            growth_times = np.linspace(0, growth_time_i, n_growth_points)
            
            # Calculate horizon at each time point based on growth type
            if samples["is_exponential"][i]:
                # Exponential: h(t) = h0 * 2^(t/T)
                T_t = samples["T_t"][i]
                horizons = h_current * (2 ** (growth_times / T_t))
                
            elif samples["is_superexponential"][i]:
                # Superexponential: exact analytical formula
                T_t = samples["T_t"][i]
                decay = samples["se_doubling_decay_fraction"]
                horizons = []
                for t in growth_times:
                    if t == 0:
                        horizons.append(h_current)
                    else:
                        # Exact formula: solve for n doublings from t = T_t * (1 - (1-decay)^n) / decay
                        # Rearranging: (1-decay)^n = 1 - t*decay/T_t
                        # So: n = log(1 - t*decay/T_t) / log(1-decay)
                        ratio_term = 1 - t * decay / T_t
                        if ratio_term > 0:
                            n_doublings = np.log(ratio_term) / np.log(1 - decay)
                            horizons.append(h_current * (2 ** n_doublings))
                        else:
                            # If we've exceeded the theoretical limit, use the target
                            horizons.append(samples["h_SC"][i])
                horizons = np.array(horizons)
                
            elif samples["is_subexponential"][i]:
                # Subexponential: exact analytical formula
                T_t = samples["T_t"][i]
                growth = samples["sub_doubling_growth_fraction"]
                horizons = []
                for t in growth_times:
                    if t == 0:
                        horizons.append(h_current)
                    else:
                        # Exact formula: solve for n doublings from t = T_t * ((1+growth)^n - 1) / growth
                        # Rearranging: (1+growth)^n = 1 + t*growth/T_t
                        # So: n = log(1 + t*growth/T_t) / log(1+growth)
                        ratio_term = 1 + t * growth / T_t
                        if ratio_term > 0:
                            n_doublings = np.log(ratio_term) / np.log(1 + growth)
                            horizons.append(h_current * (2 ** n_doublings))
                        else:
                            # Fallback (shouldn't happen for subexponential)
                            horizons.append(samples["h_SC"][i])
                horizons = np.array(horizons)
            
            # Ensure we reach the target horizon
            horizons[-1] = samples["h_SC"][i]
            
            # Convert to minutes and add to mapping
            for t, h in zip(growth_times, horizons):
                mapping.append((t, h * 60 * 167))
        
        # Add cost_speed period (horizon stays constant)
        if cost_speed_time > 0:
            final_horizon = samples["h_SC"][i] * 60 * 167  # Convert to minutes
            cost_speed_times = np.linspace(growth_time_i, growth_time_i + cost_speed_time, 
                                         max(int(cost_speed_time / dt_mapping), 2))
            for t in cost_speed_times:
                mapping.append((t, final_horizon))
        
        horizon_mappings.append(mapping)
    
    # Print time distribution by growth type (using calculated total_time)
    print("\nTime Distribution by Growth Type:")
    for mask, name in [(exp_mask, "Exponential"), (se_mask, "Superexponential"), (sub_mask, "Subexponential")]:
        if np.any(mask):
            times = total_time[mask]
            print(f"\n{name}:")
            print(f"  10th percentile: {np.percentile(times, 10):.2f} months")
            print(f"  50th percentile: {np.percentile(times, 50):.2f} months")
            print(f"  90th percentile: {np.percentile(times, 90):.2f} months")
            print(f"  Mean: {np.mean(times):.2f} months")
            print(f"  Std Dev: {np.std(times):.2f} months")
    
    # Print overall time distribution
    print("\nOverall Time Distribution:")
    print(f"  10th percentile: {np.percentile(total_time, 10):.2f} months")
    print(f"  50th percentile: {np.percentile(total_time, 50):.2f} months")
    print(f"  90th percentile: {np.percentile(total_time, 90):.2f} months")
    print(f"  Mean: {np.mean(total_time):.2f} months")
    print(f"  Std Dev: {np.std(total_time):.2f} months")
    
    return total_time, horizon_mappings

def get_compute_rate(t: float, compute_decrease_date: float) -> float:
    """Calculate compute progress rate based on time."""
    return 0.5 if t >= compute_decrease_date else 1.0

    
def calculate_sc_arrival_year_with_trajectories(samples: dict, current_horizon: float, dt: float, compute_decrease_date: float, human_alg_progress_decrease_date: float, max_simulation_years: float) -> tuple[np.ndarray, list]:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling, returning both ending times and trajectories."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, horizon_mappings = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)
    
    # Initialize array for actual times
    ending_times = np.zeros(n_sims)
    
    # Store trajectories for each simulation
    trajectories = []
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = 2025.25
    
    # Convert dt from days to months
    dt = dt / 30.5
    
    max_time = 2050
    
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
    
        progress = 0.0
        
        # Store trajectory for this simulation
        trajectory = []
        
        while progress < base_time_in_months[i] and time < max_time:
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            
            # Get current horizon from the mapping
            current_horizon_minutes = get_horizon_at_progress(horizon_mappings[i], progress)
            
            # Store trajectory point (time, horizon in minutes)
            trajectory.append((time+samples["announcement_delay"][i]/12, current_horizon_minutes))
            
            # Calculate algorithmic speedup based on intermediate speedup s(interpolate between present and SC rates)
            if samples["patch_rd_speedup"][i]:
                v_algorithmic = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** progress_fraction
            else:
                v_algorithmic = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # adjust algorithmic rate if human alg progress has decreased, in between
            if time >= human_alg_progress_decrease_date:
                only_multiplier = v_algorithmic * 0.5
                only_additive = v_algorithmic - 0.5
                # geometric mean of only_multiplier and only_additive, aggregating between extremes of how AIs/humans could complement
                v_algorithmic = np.sqrt(only_multiplier * only_additive)
            
            # Get compute rate for current time (not affected by intermediate speedups)
            compute_rate = get_compute_rate(time, compute_decrease_date)
            # Total rate is mean of algorithmic and compute rates
            total_rate = (v_algorithmic + compute_rate) / 2
            
            # Update progress and time
            progress += dt * total_rate
            time += dt / 12  # Convert months to years
        
        # If we hit the time limit, set to max time
        if time >= max_time:
            time = max_time 
            
        ending_times[i] = time
        trajectories.append(trajectory)
    
    return ending_times, trajectories

def backcast_base_time(samples: dict, current_horizon: float, dt: float, backcast_years: int = 5) -> list:
    """Backcast base time for each forecaster."""
    # Convert current horizon from minutes to months
    h_current = current_horizon / (60 * 167)

    n_sims = len(samples["h_SC"])
    dt_mapping = 1.0
    horizon_mappings = []

    for i in range(n_sims):
        mapping = []
        
        # Always create backcast points - no growth_time check needed for backcasting
        n_backcast_points = max(int(backcast_years*12 / dt_mapping), 2)
        backcast_times = np.linspace(-backcast_years*12, 0, n_backcast_points)
        
        # Calculate horizon at each time point based on growth type
        if samples["is_exponential"][i]:
            # Exponential: h(t) = h0 * 2^(t/T)
            T_t = samples["T_t"][i]
            horizons = h_current * (2 ** (backcast_times / T_t))
            
        elif samples["is_superexponential"][i]:
            # Superexponential: exact analytical formula (same as forward)
            T_t = samples["T_t"][i]
            decay = samples["se_doubling_decay_fraction"]
            horizons = []
            for t in backcast_times:
                if t == 0:
                    horizons.append(h_current)
                else:
                    # Same formula as forward: solve for n doublings from t = T_t * (1 - (1-decay)^n) / decay
                    # Rearranging: (1-decay)^n = 1 - t*decay/T_t
                    # So: n = log(1 - t*decay/T_t) / log(1-decay)
                    ratio_term = 1 - t * decay / T_t
                    if ratio_term > 0:
                        n_doublings = np.log(ratio_term) / np.log(1 - decay)
                        horizons.append(h_current * (2 ** n_doublings))
                    else:
                        # If ratio_term <= 0, we're outside the valid range
                        horizons.append(0.001)  # Very small horizon
            horizons = np.array(horizons)
            
        elif samples["is_subexponential"][i]:
            # Subexponential: properly inverted for backward time
            T_t = samples["T_t"][i]
            growth = samples["sub_doubling_growth_fraction"]
            horizons = []
            for t in backcast_times:
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
                    ratio_term = 1 + t * growth / T_t
                    if ratio_term > 0:
                        # This gives us a negative n_doublings for backward time
                        n_doublings = np.log(ratio_term) / np.log(1 + growth)
                        horizons.append(h_current * (2 ** n_doublings))
                    else:
                        horizons.append(0.000000001)  # Very small horizon
            horizons = np.array(horizons)
        
        # Convert to minutes and add to mapping
        for t, h in zip(backcast_times, horizons):
            mapping.append((t, h * 60 * 167))
        
        horizon_mappings.append(mapping)
    
    return horizon_mappings

def backcast_trajectories(samples: dict, current_horizon: float, dt: float, backcast_years: int = 5) -> tuple[np.ndarray, list]:
    """Backcast trajectories for each forecaster."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, _ = calculate_base_time(samples, current_horizon)
    horizon_mappings = backcast_base_time(samples, current_horizon, dt, backcast_years)
    n_sims = len(base_time_in_months)
    
    # Store trajectories for each simulation
    trajectories = []

    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = 2025.25
    
    # Convert dt from days to months
    dt = dt / 30.5

    min_progress = - backcast_years * 12
    
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running backcasted simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12 # check if minus or plus
        
        progress = 0.0
        
        trajectory = []

        while progress > min_progress and time > 2020:
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]

            # Get current horizon from the mapping
            current_horizon_minutes = get_horizon_at_progress(horizon_mappings[i], progress)

            # Store trajectory point (time, horizon in minutes)
            trajectory.insert(0, (time+samples["announcement_delay"][i]/12, current_horizon_minutes))
            
            # Calculate algorithmic speedup based on intermediate speedup s(interpolate between present and SC rates)
            if samples["patch_rd_speedup"][i]:
                v_algorithmic = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** progress_fraction
            else:
                v_algorithmic = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # Update progress and time
            progress -= dt * v_algorithmic
            time -= dt / 12  # Convert months to years

        trajectories.append(trajectory)

    return trajectories

def format_year_month(year_decimal: float) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= 2050:
        return ">2050"
        
    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    return f"{month_name} {year}"

def run_simple_sc_simulation(config_path: str = "simple_params.yaml") -> tuple[plt.Figure, dict]:
    """Run simplified SC simulation and plot results."""
    print("Loading configuration...")
    config = load_config(config_path)

    # Apply parent-child inheritance (with topological sort)
    config["forecasters"] = apply_inheritance_to_forecasters(config["forecasters"])

    # Create output directory with current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for trajectory types
    combined_dir = output_dir / "combined_trajectories"
    backcasted_dir = output_dir / "backcasted_trajectories"
    central_dir = output_dir / "central_trajectories"
    combined_dir.mkdir(exist_ok=True)
    backcasted_dir.mkdir(exist_ok=True)
    central_dir.mkdir(exist_ok=True)

    # Run median trajectory for each forecaster first and save to file
    print("\n" + "="*60)
    print("RUNNING MEDIAN TRAJECTORIES")
    print("="*60)
    median_results = []
    for forecaster_key, forecaster_config in config["forecasters"].items():
        result = run_median_trajectory(config, forecaster_config, forecaster_config["name"])
        median_results.append(result)

    # Save median trajectory results to file
    with open(output_dir / "median_trajectories.txt", "w") as f:
        f.write("MEDIAN TRAJECTORY RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n".join(median_results))
    print(f"Saved median trajectories to {output_dir / 'median_trajectories.txt'}")

    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year_decimal = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year_decimal = 2025.25

    # Store results for each forecaster
    all_forecaster_results = {}
    all_forecaster_samples = {}
    all_forecaster_trajectories = {}
    all_forecaster_backcast_trajectories = {}

    # Run simulations for each forecaster
    print("\nRunning simulations for each forecaster...")
    with tqdm(total=len(config["forecasters"]), desc="Processing forecasters") as pbar:
        for _, forecaster_config in config["forecasters"].items():
            name = forecaster_config["name"]
            pbar.set_description(f"Processing {name}")

            # Generate samples
            samples = get_distribution_samples(forecaster_config, config["simulation"]["n_sims"])
            all_forecaster_samples[name] = samples
            print(f"Generated {len(samples)} samples for {name}")

            # Calculate time to SC with trajectories
            results, trajectories = calculate_sc_arrival_year_with_trajectories(
                samples,
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                config["simulation"]["compute_decrease_date"],
                config["simulation"]["human_alg_progress_decrease_date"],
                config["simulation"]["max_simulation_years"]
            )

            all_forecaster_results[name] = results
            all_forecaster_trajectories[name] = trajectories

            # Generate backcasted trajectories
            print(f"Generating backcasted trajectories for {name}...")
            backcast_trajectories_result = backcast_trajectories(
                samples,
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                backcast_years=5
            )
            all_forecaster_backcast_trajectories[name] = backcast_trajectories_result

            pbar.update(1)

    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    print("\nGenerating plots...")
    # Create and save original plot (PDF)
    fig = plot_results(all_forecaster_results, config)
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create and save CDF plot
    fig_cdf = plot_results_cdf(all_forecaster_results, config)
    fig_cdf.savefig(output_dir / "simple_combined_headline_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cdf)

    monthly_central_trajectories_by_forecaster = {}
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

        fig_combined_colored, unconditional_central_path = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=True,
            plot_median_curve=True,
            forecaster_filter=[forecaster_name],
        )

        fig_combined_red, _ = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=False,
            forecaster_filter=[forecaster_name],
        )

        # Save and close the month-independent figures
        fig_backcasted_colored.savefig(
            backcasted_dir / f"backcasted_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_backcasted_red.savefig(
            backcasted_dir / f"backcasted_trajectories_red_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_colored.savefig(
            combined_dir / f"combined_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_red.savefig(
            combined_dir / f"combined_trajectories_red_{forecaster_name}.png",
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
            "March 2027",
            "March 2028",
            "March 2029",
            "March 2030",
            # September targets
            "September 2027",
            "September 2028",
            "September 2029",
            "September 2030",
        ]

        # Collect central trajectories per month for later comparison
        central_trajs_by_month: dict[str, dict] = {}

        # Add the unconditional central trajectory
        if unconditional_central_path is not None:
            central_trajs_by_month["Unconditional"] = unconditional_central_path

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
                combined_dir / f"combined_trajectories_{month_slug}_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_median.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_median_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_illustrative.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Close month-specific figures to free memory
            plt.close(fig_trajectories)
            plt.close(fig_combined_month)
            plt.close(fig_combined_month_median)
            plt.close(fig_combined_month_illustrative)

        monthly_central_trajectories_by_forecaster[forecaster_name] = central_trajs_by_month

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
                central_dir / f"central_trajectories_comparison_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig_cent_compare)
        print(f"\nSaved all trajectory plots (including month-specific versions) for {forecaster_name}.")

    for sc_month_str in target_months:
        fig_cent_forecaster_comparison_month = plot_central_trajectories_comparison(
            [(forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str]) for forecaster_name in monthly_central_trajectories_by_forecaster.keys()],
            config,
            overlay_external_data=True,
            title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals",
        )
        fig_cent_forecaster_comparison_month.savefig(
            central_dir / f"forecaster_comparison_{sc_month_str}_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_cent_forecaster_comparison_month)

    return fig, all_forecaster_results, monthly_central_trajectories_by_forecaster

# NEW TOP-LEVEL FUNCTIONS FOR FORECASTER INHERITANCE

def _toposort_forecasters(forecasters: dict) -> list:
    """Return forecaster names sorted so that parents come before children.

    Raises:
        ValueError: if a cycle is detected or if a referenced parent is missing.
    """
    visited = {}
    order: list[str] = []

    def dfs(name: str):
        if name in visited:
            if visited[name] == "temp":
                raise ValueError(
                    f"Circular parent relationship detected involving '{name}'."
                )
            return
        visited[name] = "temp"
        parent_name = forecasters.get(name, {}).get("parent")
        if parent_name:
            if parent_name not in forecasters:
                raise ValueError(
                    f"Parent forecaster '{parent_name}' (referenced by '{name}') not found."
                )
            dfs(parent_name)
        visited[name] = "perm"
        order.append(name)

    for fname in forecasters:
        dfs(fname)
    return order


def apply_inheritance_to_forecasters(forecasters: dict) -> OrderedDict:
    """Resolve parent-based inheritance and return an OrderedDict in topological order."""
    sorted_names = _toposort_forecasters(forecasters)
    for name in sorted_names:
        cfg = forecasters[name]
        parent_name = cfg.get("parent")
        if parent_name:
            parent_cfg = forecasters[parent_name]
            parent_dist = parent_cfg.get("distributions", {})
            child_dist = cfg.setdefault("distributions", {})
            for param, value in parent_dist.items():
                if param not in child_dist:
                    child_dist[param] = value
                else:
                    print(
                        f"Applying change to {param} for {name} (overrides {parent_cfg.get('name', parent_name)})"
                    )
    return OrderedDict((n, forecasters[n]) for n in sorted_names)


def regenerate_plots(output_dir: str | Path):
    """Regenerate all plots from saved trajectory data.

    Parameters
    ----------
    output_dir : str or Path
        Path to an output directory containing trajectory_data.pkl and config.yaml
    """
    output_dir = Path(output_dir)

    # Create subdirectories for trajectory types
    combined_dir = output_dir / "combined_trajectories"
    backcasted_dir = output_dir / "backcasted_trajectories"
    central_dir = output_dir / "central_trajectories"
    combined_dir.mkdir(exist_ok=True)
    backcasted_dir.mkdir(exist_ok=True)
    central_dir.mkdir(exist_ok=True)

    # Load trajectory data
    with open(output_dir / "trajectory_data.pkl", "rb") as f:
        trajectory_data = pickle.load(f)

    all_forecaster_results = trajectory_data["all_forecaster_results"]
    all_forecaster_trajectories = trajectory_data["all_forecaster_trajectories"]
    all_forecaster_samples = trajectory_data["all_forecaster_samples"]
    all_forecaster_backcast_trajectories = trajectory_data["all_forecaster_backcast_trajectories"]

    # Load config
    with open(output_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded trajectory data from {output_dir}")
    print(f"Forecasters: {list(all_forecaster_results.keys())}")

    print("\nRegenerating plots...")

    # Create and save original plot (PDF)
    fig = plot_results(all_forecaster_results, config)
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create and save CDF plot
    fig_cdf = plot_results_cdf(all_forecaster_results, config)
    fig_cdf.savefig(output_dir / "simple_combined_headline_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cdf)

    monthly_central_trajectories_by_forecaster = {}
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

        fig_combined_colored, unconditional_central_path = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=True,
            plot_median_curve=True,
            forecaster_filter=[forecaster_name],
        )

        fig_combined_red, _ = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=False,
            forecaster_filter=[forecaster_name],
        )

        # Save and close the month-independent figures
        fig_backcasted_colored.savefig(
            backcasted_dir / f"backcasted_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_backcasted_red.savefig(
            backcasted_dir / f"backcasted_trajectories_red_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_colored.savefig(
            combined_dir / f"combined_trajectories_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig_combined_red.savefig(
            combined_dir / f"combined_trajectories_red_{forecaster_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig_backcasted_colored)
        plt.close(fig_backcasted_red)
        plt.close(fig_combined_colored)
        plt.close(fig_combined_red)

        # --- Figures filtered by specific SC arrival months ---
        target_months = [
            "March 2027",
            "March 2028",
            "March 2029",
            "March 2030",
            "September 2027",
            "September 2028",
            "September 2029",
            "September 2030",
        ]

        # Collect central trajectories per month for later comparison
        central_trajs_by_month: dict[str, dict] = {}

        # Add the unconditional central trajectory
        if unconditional_central_path is not None:
            central_trajs_by_month["Unconditional"] = unconditional_central_path

        for sc_month_str in target_months:
            month_slug = sc_month_str.lower().replace(" ", "_")

            fig_trajectories = plot_trajectories_sc_month(
                all_forecaster_results,
                all_forecaster_trajectories,
                all_forecaster_samples,
                config,
                sc_month_str=sc_month_str,
                forecaster_filter=[forecaster_name],
            )

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

            fig_trajectories.savefig(
                output_dir / f"{month_slug}_trajectories_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_median.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_median_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month_illustrative.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig_trajectories)
            plt.close(fig_combined_month)
            plt.close(fig_combined_month_median)
            plt.close(fig_combined_month_illustrative)

        monthly_central_trajectories_by_forecaster[forecaster_name] = central_trajs_by_month

        if central_trajs_by_month:
            central_list = sorted(central_trajs_by_month.items())
            fig_cent_compare = plot_central_trajectories_comparison(
                central_list,
                config,
                overlay_external_data=True,
            )

            fig_cent_compare.savefig(
                central_dir / f"central_trajectories_comparison_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig_cent_compare)
        print(f"\nSaved all trajectory plots (including month-specific versions) for {forecaster_name}.")

    for sc_month_str in target_months:
        fig_cent_forecaster_comparison_month = plot_central_trajectories_comparison(
            [(forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str])
             for forecaster_name in monthly_central_trajectories_by_forecaster.keys()],
            config,
            overlay_external_data=True,
            title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals",
        )
        fig_cent_forecaster_comparison_month.savefig(
            central_dir / f"forecaster_comparison_{sc_month_str}_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_cent_forecaster_comparison_month)

    print(f"\nPlot regeneration completed. Files saved to {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        # Regenerate plots from existing output directory
        if len(sys.argv) < 3:
            print("Usage: python simple_forecasting_timelines.py --replot <output_dir>")
            print("Example: python simple_forecasting_timelines.py --replot output/2024-01-15_10-30-00")
            sys.exit(1)
        regenerate_plots(sys.argv[2])
    else:
        # Run with closed-form solution (faster)
        print("=== Running with closed-form solution ===")
        run_simple_sc_simulation()

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 