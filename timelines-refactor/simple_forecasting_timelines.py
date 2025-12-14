import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm, rankdata
import yaml
import json
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

    # Sample software progress share (fraction of progress from software vs compute)
    # If not specified, default to 0.5 (equal weighting, equivalent to averaging)
    if "software_progress_share_ci" in config["distributions"]:
        lower, upper = config["distributions"]["software_progress_share_ci"]
        # Convert 80% CI to normal distribution parameters
        z_low = -1.28  # norm.ppf(0.1)
        z_high = 1.28  # norm.ppf(0.9)
        mean = (lower + upper) / 2
        std = (upper - lower) / (z_high - z_low)
        # Generate samples and clip to [0.1, 0.9]
        samples["software_progress_share"] = np.clip(
            np.random.normal(mean, std, n_sims),
            0.1, 0.9
        )
    else:
        # Default to 0.5 (equivalent to the original averaging behavior)
        samples["software_progress_share"] = np.full(n_sims, 0.5)

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

    # Software progress share - use median of CI if specified, else 0.5
    if "software_progress_share_ci" in config["distributions"]:
        lower, upper = config["distributions"]["software_progress_share_ci"]
        samples["software_progress_share"] = np.array([(lower + upper) / 2])
    else:
        samples["software_progress_share"] = np.array([0.5])

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


def run_median_trajectory(config: dict, forecaster_config: dict, forecaster_name: str) -> tuple[str, dict, dict]:
    """Run trajectories with median parameters for both exponential and superexponential growth.

    Returns
    -------
    tuple[str, dict, dict]
        A tuple of (text_output, trajectory_data, structured_data) where:
        - trajectory_data is a dict mapping growth_type ("exponential", "superexponential")
          to trajectory dicts with "times" and "horizons" arrays.
        - structured_data is a dict with all key values for CSV export.
    """
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

    # Run simulation parameters (allow per-forecaster overrides)
    current_horizon = forecaster_config.get("current_horizon", config["simulation"]["current_horizon"])
    current_year = forecaster_config.get("current_year", config["simulation"].get("current_year", 2025.25))
    dt = config["simulation"]["dt"]
    compute_decrease_date = config["simulation"]["compute_decrease_date"]
    human_alg_progress_decrease_date = config["simulation"]["human_alg_progress_decrease_date"]
    max_simulation_years = config["simulation"]["max_simulation_years"]

    # Store trajectory data for plotting
    trajectory_data = {}
    arrival_years = {}  # Store SC arrival year per growth type

    # Run exponential, superexponential, and subexponential trajectories
    for growth_type in ["Exponential", "Superexponential", "Subexponential"]:
        # Set growth type flags
        if growth_type == "Exponential":
            median_samples["is_exponential"] = np.array([True])
            median_samples["is_superexponential"] = np.array([False])
            median_samples["is_subexponential"] = np.array([False])
        elif growth_type == "Superexponential":
            median_samples["is_exponential"] = np.array([False])
            median_samples["is_superexponential"] = np.array([True])
            median_samples["is_subexponential"] = np.array([False])
        else:  # Subexponential
            median_samples["is_exponential"] = np.array([False])
            median_samples["is_superexponential"] = np.array([False])
            median_samples["is_subexponential"] = np.array([True])

        lines.append(f"\n--- {growth_type} Growth ---")

        # Run forward simulation
        ending_times, forward_trajectories = calculate_sc_arrival_year_with_trajectories(
            median_samples, current_horizon, dt,
            compute_decrease_date, human_alg_progress_decrease_date, max_simulation_years,
            current_year=current_year
        )

        # Run backcast
        backcast_trajs = backcast_trajectories(
            median_samples, current_horizon, dt, backcast_years=5,
            current_year=current_year
        )

        arrival_year = ending_times[0]
        arrival_years[growth_type.lower()] = arrival_year  # Store for CSV export
        fore_traj = forward_trajectories[0] if forward_trajectories else []
        back_traj = backcast_trajs[0] if backcast_trajs else []

        # Combine backcast and forward trajectories for plotting
        combined_t = []
        combined_h = []
        if back_traj:
            bt, bh = zip(*back_traj)
            combined_t.extend(bt)
            combined_h.extend(bh)
        if fore_traj:
            ft, fh = zip(*fore_traj)
            combined_t.extend(ft)
            combined_h.extend(fh)

        if combined_t:
            order = np.argsort(combined_t)
            trajectory_data[growth_type.lower()] = {
                'times': np.array(combined_t)[order],
                'horizons': np.array(combined_h)[order]
            }

        # arrival_year is internal time (without announcement delay)
        announcement_delay_years = median_samples['announcement_delay'][0] / 12
        lines.append(f"SC Arrival: {format_year_month(arrival_year)}")

        # Find human-cost-parity point (when horizon first reaches h_SC requirement)
        h_SC_threshold_minutes = median_samples['h_SC'][0] * 60 * 167  # Convert work-months to minutes
        human_cost_parity_year_announced = None  # Trajectory times include announcement delay
        if fore_traj:
            for t, h in fore_traj:
                if h >= h_SC_threshold_minutes:
                    human_cost_parity_year_announced = t
                    break

        if human_cost_parity_year_announced is not None:
            # Convert HCP to internal time (subtract announcement delay) to match SC arrival
            human_cost_parity_year = human_cost_parity_year_announced - announcement_delay_years
            lines.append(f"Human-Cost-Parity: {format_year_month(human_cost_parity_year)}")
            # Calculate days between human-cost-parity and SC arrival (both in internal time)
            days_to_sc = (arrival_year - human_cost_parity_year) * 365.25
            lines.append(f"Days from Human-Cost-Parity to SC: {days_to_sc:.1f} days")

        lines.append(f"\nTrajectory (selected points):")
        lines.append(f"  {'Year':<12} {'Horizon':<20}")
        lines.append(f"  {'-'*12} {'-'*20}")

        # Print trajectory at key points
        if fore_traj:
            n_points = len(fore_traj)
            # Show ~10 evenly spaced points
            step = max(1, n_points // 10)
            for i in range(0, n_points, step):
                t, h = fore_traj[i]
                lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20}")

            # Always show final point
            if n_points > 1:
                t, h = fore_traj[-1]
                lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20} (final)")

    lines.append(f"\n{'='*60}\n")

    result = "\n".join(lines)
    print(result)

    # Build structured data for CSV export
    structured_data = {
        "forecaster_name": forecaster_name,
        # SC arrival years
        "sc_arrival_year_exponential": arrival_years.get("exponential"),
        "sc_arrival_year_superexponential": arrival_years.get("superexponential"),
        "sc_arrival_year_subexponential": arrival_years.get("subexponential"),
        # Median parameters
        "h_SC_work_months": median_samples['h_SC'][0],
        "T_t_doubling_time_months": median_samples['T_t'][0],
        "cost_speed_months": median_samples['cost_speed'][0],
        "announcement_delay_months": median_samples['announcement_delay'][0],
        "present_prog_multiplier": median_samples['present_prog_multiplier'][0],
        "SC_prog_multiplier": median_samples['SC_prog_multiplier'][0],
        "patch_rd_speedup": median_samples['patch_rd_speedup'][0],
        "se_doubling_decay_fraction": median_samples['se_doubling_decay_fraction'],
        # Trajectory data will be added per growth type
        "trajectory_data": trajectory_data,
    }

    return result, trajectory_data, structured_data


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

    
def calculate_sc_arrival_year_with_trajectories(samples: dict, current_horizon: float, dt: float, compute_decrease_date: float, human_alg_progress_decrease_date: float, max_simulation_years: float, current_year: float = 2025.25) -> tuple[np.ndarray, list]:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling, returning both ending times and trajectories."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, horizon_mappings = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)

    # Initialize array for actual times
    ending_times = np.zeros(n_sims)

    # Store trajectories for each simulation
    trajectories = []
    
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
            # Total rate is weighted combination of algorithmic and compute rates
            # software_progress_share determines the weighting (0.5 = equal weighting = original behavior)
            sw_share = samples["software_progress_share"][i]
            total_rate = sw_share * v_algorithmic + (1 - sw_share) * compute_rate
            
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

def backcast_trajectories(samples: dict, current_horizon: float, dt: float, backcast_years: int = 5, current_year: float = 2025.25) -> tuple[np.ndarray, list]:
    """Backcast trajectories for each forecaster."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, _ = calculate_base_time(samples, current_horizon)
    horizon_mappings = backcast_base_time(samples, current_horizon, dt, backcast_years)
    n_sims = len(base_time_in_months) # EL: why len(base_time_in_months)?

    # Store trajectories for each simulation
    trajectories = []
    
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

            # For backcast, compute_rate is 1 (baseline)
            compute_rate = 1
            # Total rate is weighted combination of algorithmic and compute rates
            sw_share = samples["software_progress_share"][i]
            total_rate = sw_share * v_algorithmic + (1 - sw_share) * compute_rate

            # Update progress and time
            progress -= dt * total_rate
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


def format_year_month_full(year_decimal: float) -> str:
    """Convert decimal year to full Month Year format (e.g., 'January 2027')."""
    if year_decimal >= 2050:
        return ">2050"

    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    return f"{month_names[month-1]} {year}"


def get_forecaster_growth_type(samples: dict) -> str | None:
    """Determine if a forecaster has a single growth type across all samples.

    Returns
    -------
    str | None
        "exponential" if all samples are exponential,
        "superexponential" if all samples are superexponential,
        None if mixed.
    """
    is_exp = samples["is_exponential"]
    is_superexp = samples["is_superexponential"]

    if np.all(is_exp):
        return "exponential"
    elif np.all(is_superexp):
        return "superexponential"
    return None


def calculate_hcp_distributions(
    all_forecaster_trajectories: dict,
    all_forecaster_results: dict,
    all_forecaster_samples: dict,
    config: dict,
    output_dir: Path
) -> dict:
    """Calculate Human-Cost-Parity (HCP) distributions for all forecasters.

    For each simulation trajectory, finds when the horizon first reaches the h_SC
    threshold (HCP point) and calculates:
    - When HCP is achieved (calendar year)
    - Time gap from HCP to SC arrival (in days)

    Parameters
    ----------
    all_forecaster_trajectories : dict
        Dictionary mapping forecaster names to lists of trajectories
    all_forecaster_results : dict
        Dictionary mapping forecaster names to SC arrival times
    all_forecaster_samples : dict
        Dictionary mapping forecaster names to sampled parameters
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory for saving results

    Returns
    -------
    dict
        Dictionary with HCP distribution data for each forecaster
    """
    hcp_data = {}

    for forecaster_name in all_forecaster_trajectories.keys():
        trajectories = all_forecaster_trajectories[forecaster_name]
        sc_arrivals = all_forecaster_results[forecaster_name]
        samples = all_forecaster_samples[forecaster_name]

        # Get h_SC threshold in minutes for each simulation
        h_SC_thresholds = samples['h_SC'] * 60 * 167  # Convert work-months to minutes

        hcp_years = []
        hcp_to_sc_days = []

        for i, trajectory in enumerate(trajectories):
            if not trajectory:
                hcp_years.append(np.nan)
                hcp_to_sc_days.append(np.nan)
                continue

            h_SC_threshold = h_SC_thresholds[i]
            sc_arrival = sc_arrivals[i]  # Internal time (without announcement delay)
            announcement_delay_years = samples['announcement_delay'][i] / 12

            # Find when horizon first reaches h_SC threshold
            hcp_year_announced = None  # Trajectory times include announcement delay
            for t, h in trajectory:
                if h >= h_SC_threshold:
                    hcp_year_announced = t
                    break

            if hcp_year_announced is not None:
                # Convert HCP to internal time (subtract announcement delay) to match SC arrival
                hcp_year = hcp_year_announced - announcement_delay_years
                hcp_years.append(hcp_year)
                # Calculate days from HCP to SC arrival (both in internal time)
                days_to_sc = (sc_arrival - hcp_year) * 365.25
                hcp_to_sc_days.append(days_to_sc)
            else:
                hcp_years.append(np.nan)
                hcp_to_sc_days.append(np.nan)

        hcp_years = np.array(hcp_years)
        hcp_to_sc_days = np.array(hcp_to_sc_days)

        # Filter out NaN values for statistics
        valid_hcp_years = hcp_years[~np.isnan(hcp_years)]
        valid_hcp_to_sc_days = hcp_to_sc_days[~np.isnan(hcp_to_sc_days)]

        hcp_data[forecaster_name] = {
            'hcp_years': hcp_years,
            'hcp_to_sc_days': hcp_to_sc_days,
            'valid_hcp_years': valid_hcp_years,
            'valid_hcp_to_sc_days': valid_hcp_to_sc_days,
        }

    # Create plots
    fig_hcp_year, axes_year = plt.subplots(1, 2, figsize=(14, 5))
    fig_hcp_gap, axes_gap = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors

    # Plot HCP arrival year distributions
    ax_pdf_year = axes_year[0]
    ax_cdf_year = axes_year[1]

    for idx, (forecaster_name, data) in enumerate(hcp_data.items()):
        valid_years = data['valid_hcp_years']
        if len(valid_years) < 2:
            continue
        color = colors[idx % len(colors)]

        # PDF using KDE
        kde = gaussian_kde(valid_years)
        x_range = np.linspace(valid_years.min() - 0.5, valid_years.max() + 0.5, 200)
        ax_pdf_year.plot(x_range, kde(x_range), label=forecaster_name, color=color)
        ax_pdf_year.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

        # CDF
        sorted_years = np.sort(valid_years)
        cdf = np.arange(1, len(sorted_years) + 1) / len(sorted_years)
        ax_cdf_year.plot(sorted_years, cdf, label=forecaster_name, color=color)

    ax_pdf_year.set_xlabel('Year')
    ax_pdf_year.set_ylabel('Density')
    ax_pdf_year.set_title('Human-Cost-Parity Arrival Year (PDF)')
    ax_pdf_year.legend()
    ax_pdf_year.grid(True, alpha=0.3)

    ax_cdf_year.set_xlabel('Year')
    ax_cdf_year.set_ylabel('Cumulative Probability')
    ax_cdf_year.set_title('Human-Cost-Parity Arrival Year (CDF)')
    ax_cdf_year.legend()
    ax_cdf_year.grid(True, alpha=0.3)

    fig_hcp_year.suptitle('Distribution of Human-Cost-Parity (HCP) Arrival Year', fontsize=14)
    fig_hcp_year.tight_layout()
    fig_hcp_year.savefig(output_dir / "hcp_arrival_year_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig_hcp_year)

    # Plot HCP-to-SC gap distributions
    ax_pdf_gap = axes_gap[0]
    ax_cdf_gap = axes_gap[1]

    for idx, (forecaster_name, data) in enumerate(hcp_data.items()):
        valid_days = data['valid_hcp_to_sc_days']
        if len(valid_days) < 2:
            continue
        color = colors[idx % len(colors)]

        # PDF using KDE
        kde = gaussian_kde(valid_days)
        x_range = np.linspace(max(0, valid_days.min() - 10), valid_days.max() + 10, 200)
        ax_pdf_gap.plot(x_range, kde(x_range), label=forecaster_name, color=color)
        ax_pdf_gap.fill_between(x_range, kde(x_range), alpha=0.2, color=color)

        # CDF
        sorted_days = np.sort(valid_days)
        cdf = np.arange(1, len(sorted_days) + 1) / len(sorted_days)
        ax_cdf_gap.plot(sorted_days, cdf, label=forecaster_name, color=color)

    ax_pdf_gap.set_xlabel('Days')
    ax_pdf_gap.set_ylabel('Density')
    ax_pdf_gap.set_title('HCP to SC Gap (PDF)')
    ax_pdf_gap.legend()
    ax_pdf_gap.grid(True, alpha=0.3)

    ax_cdf_gap.set_xlabel('Days')
    ax_cdf_gap.set_ylabel('Cumulative Probability')
    ax_cdf_gap.set_title('HCP to SC Gap (CDF)')
    ax_cdf_gap.legend()
    ax_cdf_gap.grid(True, alpha=0.3)

    fig_hcp_gap.suptitle('Distribution of Time from Human-Cost-Parity to SC Arrival', fontsize=14)
    fig_hcp_gap.tight_layout()
    fig_hcp_gap.savefig(output_dir / "hcp_to_sc_gap_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig_hcp_gap)

    # Export to CSV
    csv_rows = []
    for forecaster_name, data in hcp_data.items():
        for i in range(len(data['hcp_years'])):
            csv_rows.append({
                'forecaster': forecaster_name,
                'simulation_id': i,
                'hcp_year': data['hcp_years'][i],
                'hcp_to_sc_days': data['hcp_to_sc_days'][i],
            })

    import pandas as pd
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_dir / "hcp_distributions.csv", index=False)

    # Also export summary statistics
    summary_rows = []
    for forecaster_name, data in hcp_data.items():
        valid_years = data['valid_hcp_years']
        valid_days = data['valid_hcp_to_sc_days']
        sc_arrivals = all_forecaster_results[forecaster_name]

        if len(valid_years) > 0:
            summary_rows.append({
                'forecaster': forecaster_name,
                'sc_year_10th': np.percentile(sc_arrivals, 10),
                'sc_year_25th': np.percentile(sc_arrivals, 25),
                'sc_year_50th': np.percentile(sc_arrivals, 50),
                'sc_year_75th': np.percentile(sc_arrivals, 75),
                'sc_year_90th': np.percentile(sc_arrivals, 90),
                'sc_year_mean': np.mean(sc_arrivals),
                'sc_year_std': np.std(sc_arrivals),
                'hcp_year_10th': np.percentile(valid_years, 10),
                'hcp_year_25th': np.percentile(valid_years, 25),
                'hcp_year_50th': np.percentile(valid_years, 50),
                'hcp_year_75th': np.percentile(valid_years, 75),
                'hcp_year_90th': np.percentile(valid_years, 90),
                'hcp_year_mean': np.mean(valid_years),
                'hcp_year_std': np.std(valid_years),
                'hcp_to_sc_days_10th': np.percentile(valid_days, 10),
                'hcp_to_sc_days_25th': np.percentile(valid_days, 25),
                'hcp_to_sc_days_50th': np.percentile(valid_days, 50),
                'hcp_to_sc_days_75th': np.percentile(valid_days, 75),
                'hcp_to_sc_days_90th': np.percentile(valid_days, 90),
                'hcp_to_sc_days_mean': np.mean(valid_days),
                'hcp_to_sc_days_std': np.std(valid_days),
                'n_valid_samples': len(valid_years),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "hcp_distributions_summary.csv", index=False)

    print(f"\nSaved HCP distribution plots and CSVs to {output_dir}")
    print("\nHCP Distribution Summary:")
    print("=" * 80)
    for forecaster_name, data in hcp_data.items():
        valid_years = data['valid_hcp_years']
        valid_days = data['valid_hcp_to_sc_days']
        if len(valid_years) > 0:
            print(f"\n{forecaster_name}:")
            print(f"  HCP Arrival Year: median={np.median(valid_years):.2f}, "
                  f"10th={np.percentile(valid_years, 10):.2f}, "
                  f"90th={np.percentile(valid_years, 90):.2f}")
            print(f"  HCP-to-SC Gap: median={np.median(valid_days):.1f} days, "
                  f"10th={np.percentile(valid_days, 10):.1f}, "
                  f"90th={np.percentile(valid_days, 90):.1f}")

    return hcp_data


def save_trajectories_jsonl(
    all_forecaster_trajectories: dict,
    all_forecaster_backcast_trajectories: dict,
    all_forecaster_samples: dict,
    all_forecaster_results: dict,
    output_dir: Path,
    downsample_step: int = 10
) -> None:
    """Save all time->horizon trajectories to JSONL files for post-run analysis.

    Each line in the JSONL file contains one simulation's trajectory data:
    - sample_idx: Index of this simulation
    - sc_arrival_year: SC arrival year for this simulation
    - growth_type: 'exponential', 'superexponential', or 'subexponential'
    - parameters: Dict of sample parameters for this simulation
    - forward_trajectory: List of [time, horizon_minutes] points (downsampled)
    - backcast_trajectory: List of [time, horizon_minutes] points (downsampled)

    Parameters
    ----------
    downsample_step : int
        Save every Nth point to reduce file size. Default 10 reduces ~2200 points to ~220.
    """
    trajectories_dir = output_dir / "trajectories_jsonl"
    trajectories_dir.mkdir(exist_ok=True)

    for forecaster_name in all_forecaster_trajectories.keys():
        forward_trajectories = all_forecaster_trajectories[forecaster_name]
        backcast_trajectories = all_forecaster_backcast_trajectories[forecaster_name]
        samples = all_forecaster_samples[forecaster_name]
        results = all_forecaster_results[forecaster_name]
        n_trajectories = len(forward_trajectories)

        # Pre-compute growth types as array for faster lookup
        growth_types = np.where(
            samples["is_exponential"], "exponential",
            np.where(samples["is_superexponential"], "superexponential", "subexponential")
        )

        # Build all records in memory first, then write in batch
        records = []
        for i in range(n_trajectories):
            # Downsample trajectories, always keeping first and last points
            fwd = forward_trajectories[i]
            bck = backcast_trajectories[i]
            fwd_downsampled = fwd[::downsample_step] if len(fwd) > downsample_step else fwd
            bck_downsampled = bck[::downsample_step] if len(bck) > downsample_step else bck
            # Ensure last point is included
            if len(fwd) > 1 and fwd[-1] != fwd_downsampled[-1]:
                fwd_downsampled = list(fwd_downsampled) + [fwd[-1]]
            if len(bck) > 1 and bck[-1] != bck_downsampled[-1]:
                bck_downsampled = list(bck_downsampled) + [bck[-1]]

            record = {
                "sample_idx": i,
                "sc_arrival_year": float(results[i]),
                "growth_type": str(growth_types[i]),
                "parameters": {
                    "h_SC": float(samples["h_SC"][i]),
                    "T_t": float(samples["T_t"][i]),
                    "cost_speed": float(samples["cost_speed"][i]),
                    "announcement_delay": float(samples["announcement_delay"][i]),
                    "present_prog_multiplier": float(samples["present_prog_multiplier"][i]),
                    "SC_prog_multiplier": float(samples["SC_prog_multiplier"][i]),
                    "patch_rd_speedup": bool(samples["patch_rd_speedup"][i]),
                    "software_progress_share": float(samples["software_progress_share"][i]),
                },
                "forward_trajectory": fwd_downsampled,
                "backcast_trajectory": bck_downsampled,
            }
            records.append(json.dumps(record, default=float))

        jsonl_path = trajectories_dir / f"{forecaster_name}_trajectories.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('\n'.join(records))

        print(f"Saved {n_trajectories} trajectories to {jsonl_path}")

    print(f"\nAll trajectory JSONL files saved to {trajectories_dir}")


def run_simple_sc_simulation(config_path: str = "simple_params.yaml", minimal: bool = False, n_sims: int | None = None) -> tuple[plt.Figure, dict, dict, dict]:
    """Run simplified SC simulation and plot results.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    minimal : bool
        If True, only generate: overall PDF/CDF plots and March 2027 illustrative
        combined trajectories (PNG + CSV) for each forecaster
    n_sims : int | None
        Override the number of Monte Carlo simulations from config

    Returns
    -------
    tuple[plt.Figure, dict, dict, dict]
        (fig, all_forecaster_results, monthly_central_trajectories_by_forecaster, median_trajectory_data)
    """
    print("Loading configuration...")
    config = load_config(config_path)

    # Override n_sims if provided via command line
    if n_sims is not None:
        config["simulation"]["n_sims"] = n_sims
        print(f"Overriding n_sims to {n_sims}")

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
    forecasted_dir = output_dir / "forecasted_trajectories"
    combined_dir.mkdir(exist_ok=True)
    backcasted_dir.mkdir(exist_ok=True)
    central_dir.mkdir(exist_ok=True)
    forecasted_dir.mkdir(exist_ok=True)

    # Run median trajectory for each forecaster first and save to file
    print("\n" + "="*60)
    print("RUNNING MEDIAN TRAJECTORIES")
    print("="*60)
    median_results = []
    median_trajectory_data = {}  # forecaster_name -> {growth_type: trajectory_dict}
    median_structured_data = []  # list of structured data dicts for CSV export
    for forecaster_key, forecaster_config in config["forecasters"].items():
        result, traj_data, structured_data = run_median_trajectory(config, forecaster_config, forecaster_config["name"])
        median_results.append(result)
        median_trajectory_data[forecaster_config["name"]] = traj_data
        median_structured_data.append(structured_data)

    # Save median trajectory results to txt file
    with open(output_dir / "median_trajectories.txt", "w") as f:
        f.write("MEDIAN TRAJECTORY RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n".join(median_results))
    print(f"Saved median trajectories to {output_dir / 'median_trajectories.txt'}")

    # Save median trajectory summary to CSV
    csv_rows = []
    prev_exp = None
    prev_superexp = None
    prev_subexp = None
    for data in median_structured_data:
        curr_exp = data["sc_arrival_year_exponential"]
        curr_superexp = data["sc_arrival_year_superexponential"]
        curr_subexp = data["sc_arrival_year_subexponential"]
        row = {
            "forecaster_name": data["forecaster_name"],
            "sc_arrival_year_exponential": curr_exp,
            "diff_exp_from_prev": curr_exp - prev_exp if prev_exp is not None else None,
            "sc_arrival_year_superexponential": curr_superexp,
            "diff_superexp_from_prev": curr_superexp - prev_superexp if prev_superexp is not None else None,
            "sc_arrival_year_subexponential": curr_subexp,
            "diff_subexp_from_prev": curr_subexp - prev_subexp if prev_subexp is not None else None,
            "h_SC_work_months": data["h_SC_work_months"],
            "T_t_doubling_time_months": data["T_t_doubling_time_months"],
            "cost_speed_months": data["cost_speed_months"],
            "announcement_delay_months": data["announcement_delay_months"],
            "present_prog_multiplier": data["present_prog_multiplier"],
            "SC_prog_multiplier": data["SC_prog_multiplier"],
            "patch_rd_speedup": data["patch_rd_speedup"],
            "se_doubling_decay_fraction": data["se_doubling_decay_fraction"],
        }
        csv_rows.append(row)
        prev_exp = curr_exp
        prev_superexp = curr_superexp
        prev_subexp = curr_subexp
    df_summary = pd.DataFrame(csv_rows)
    df_summary.to_csv(output_dir / "median_trajectories.csv", index=False)
    print(f"Saved median trajectories summary to {output_dir / 'median_trajectories.csv'}")

    # Save detailed trajectory points to separate CSV
    trajectory_rows = []
    for data in median_structured_data:
        forecaster = data["forecaster_name"]
        traj_data = data["trajectory_data"]
        for growth_type, traj in traj_data.items():
            times = traj["times"]
            horizons = traj["horizons"]
            for t, h in zip(times, horizons):
                trajectory_rows.append({
                    "forecaster_name": forecaster,
                    "growth_type": growth_type,
                    "year": t,
                    "horizon_minutes": h
                })
    df_points = pd.DataFrame(trajectory_rows)
    df_points.to_csv(output_dir / "median_trajectory_points.csv", index=False)
    print(f"Saved detailed trajectory points to {output_dir / 'median_trajectory_points.csv'}")

    # Generate simple median trajectory comparison plots for each growth type
    for growth_type in ["exponential", "superexponential"]:
        fig_median_simple = plot_median_trajectories_simple(
            median_trajectory_data,
            config,
            growth_type=growth_type,
            overlay_external_data=True,
        )
        fig_median_simple.savefig(
            output_dir / f"median_trajectories_{growth_type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_median_simple)
        print(f"Saved median trajectory plot ({growth_type}) to {output_dir / f'median_trajectories_{growth_type}.png'}")

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

            # Get per-forecaster overrides for current_horizon and current_year
            current_horizon = forecaster_config.get("current_horizon", config["simulation"]["current_horizon"])
            current_year = forecaster_config.get("current_year", config["simulation"].get("current_year", 2025.25))

            # Calculate time to SC with trajectories
            results, trajectories = calculate_sc_arrival_year_with_trajectories(
                samples,
                current_horizon,
                config["simulation"]["dt"],
                config["simulation"]["compute_decrease_date"],
                config["simulation"]["human_alg_progress_decrease_date"],
                config["simulation"]["max_simulation_years"],
                current_year=current_year
            )

            all_forecaster_results[name] = results
            all_forecaster_trajectories[name] = trajectories

            # Generate backcasted trajectories
            print(f"Generating backcasted trajectories for {name}...")
            backcast_trajectories_result = backcast_trajectories(
                samples,
                current_horizon,
                config["simulation"]["dt"],
                backcast_years=5,
                current_year=current_year
            )
            all_forecaster_backcast_trajectories[name] = backcast_trajectories_result

            pbar.update(1)

    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save all trajectories to JSONL for post-run analysis
    print("\nSaving trajectories to JSONL...")
    save_trajectories_jsonl(
        all_forecaster_trajectories,
        all_forecaster_backcast_trajectories,
        all_forecaster_samples,
        all_forecaster_results,
        output_dir
    )

    print("\nGenerating plots...")
    # Create and save original plot (PDF)
    fig = plot_results(all_forecaster_results, config)
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create and save CDF plot
    fig_cdf = plot_results_cdf(all_forecaster_results, config)
    fig_cdf.savefig(output_dir / "simple_combined_headline_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cdf)

    # Save PDF and CDF data to CSVs
    save_pdf_cdf_csvs(all_forecaster_results, config, output_dir)

    # Calculate and save HCP distributions
    print("\nCalculating HCP distributions...")
    hcp_data = calculate_hcp_distributions(
        all_forecaster_trajectories,
        all_forecaster_results,
        all_forecaster_samples,
        config,
        output_dir
    )

    monthly_central_trajectories_by_forecaster = {}
    for forecaster_name in all_forecaster_results.keys():
        # --- Figures that are independent of a specific SC month ---
        if not minimal:
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

            # # Commented out: unconditional combined trajectories
            # fig_combined_colored, unconditional_central_path = plot_combined_trajectories(
            #     all_forecaster_backcast_trajectories,
            #     all_forecaster_trajectories,
            #     all_forecaster_samples,
            #     config,
            #     color_by_growth_type=True,
            #     plot_median_curve=True,
            #     forecaster_filter=[forecaster_name],
            # )

            # fig_combined_red, _ = plot_combined_trajectories(
            #     all_forecaster_backcast_trajectories,
            #     all_forecaster_trajectories,
            #     all_forecaster_samples,
            #     config,
            #     color_by_growth_type=False,
            #     forecaster_filter=[forecaster_name],
            # )

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
            # # Commented out: saving unconditional combined trajectories
            # fig_combined_colored.savefig(
            #     combined_dir / f"combined_trajectories_{forecaster_name}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )
            # # Also save unconditional all trajectories to central_trajectories folder
            # fig_combined_colored.savefig(
            #     central_dir / f"all_trajectories_unconditional_{forecaster_name}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )
            # fig_combined_red.savefig(
            #     combined_dir / f"combined_trajectories_red_{forecaster_name}.png",
            #     dpi=300,
            #     bbox_inches="tight",
            # )

            plt.close(fig_backcasted_colored)
            plt.close(fig_backcasted_red)
            # plt.close(fig_combined_colored)
            # plt.close(fig_combined_red)

        # --- Figures filtered by specific SC arrival months ---
        if minimal:
            target_months = ["March 2027"]
        else:
            target_months = [
                "November 2026",
                "January 2027",
                "March 2027",
                "April 2027",
                "May 2027",
                "June 2027",
                "August 2027",
                "May 2028",
                "June 2028",
                "July 2028",
                "September 2028",
                "April 2029",
                "November 2029",
                "January 2030",
                "June 2030",
                "July 2030",
                # "March 2028",
                # "March 2029",
                # "March 2030",
                # "September 2027",
                # "September 2029",
                # "September 2030",
            ]

        # Collect central trajectories per month for later comparison
        central_trajs_by_month: dict[str, dict] = {}

        # # Commented out: Add the unconditional central trajectory
        # if unconditional_central_path is not None:
        #     central_trajs_by_month["Unconditional"] = unconditional_central_path

        for sc_month_str in target_months:
            month_slug = sc_month_str.lower().replace(" ", "_")  # e.g. "march_2028"

            if not minimal:
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
            fig_combined_month, all_trajectories = plot_combined_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                forecaster_filter=[forecaster_name],
            )

            # Extract central_incl_backcast as the primary central trajectory for backward compatibility
            central_path = all_trajectories.get('central_incl_backcast')

            # Store central trajectory if available
            if central_path is not None:
                central_trajs_by_month[sc_month_str] = central_path

            if not minimal:
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

            # Central-only plot (simplified: only Central incl. backcast curve)
            fig_central_only, _ = plot_central_only_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                forecaster_filter=[forecaster_name],
            )

            # --- Save the month-specific figures ---
            if not minimal:
                fig_trajectories.savefig(
                    forecasted_dir / f"{month_slug}_trajectories_{forecaster_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                # Save all_trajectories to central_trajectories folder
                fig_combined_month.savefig(
                    central_dir / f"all_trajectories_{month_slug}_{forecaster_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            fig_combined_month_illustrative.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_central_only.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_central_only_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Save all trajectory data as CSV (all 3 central trajectories + median)
            if all_trajectories:
                # Find the common time grid from median (most complete)
                median_traj = all_trajectories.get('median')
                if median_traj is not None:
                    times = median_traj['times']
                    csv_data = {'calendar_time': times}

                    # Add median trajectory
                    csv_data['median_time_horizon_minutes'] = median_traj['horizons']

                    # Add central trajectories (interpolated to common time grid)
                    for key, label in [
                        ('central_incl_backcast', 'central_incl_backcast'),
                        ('central_forecast_only', 'central_forecast_only'),
                        ('central_zscore', 'central_zscore'),
                    ]:
                        traj = all_trajectories.get(key)
                        if traj is not None:
                            # Interpolate to common time grid
                            interp_horizons = np.interp(
                                times,
                                traj['times'],
                                traj['horizons'],
                                left=np.nan,
                                right=np.nan,
                            )
                            csv_data[f'{label}_time_horizon_minutes'] = interp_horizons

                    central_df = pd.DataFrame(csv_data)
                    central_df.to_csv(
                        combined_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}_central_trajectory.csv",
                        index=False,
                    )

            # Close month-specific figures to free memory
            if not minimal:
                plt.close(fig_trajectories)
                plt.close(fig_combined_month_median)
            plt.close(fig_combined_month)
            plt.close(fig_combined_month_illustrative)
            plt.close(fig_central_only)

        monthly_central_trajectories_by_forecaster[forecaster_name] = central_trajs_by_month

        # --------------------------------------------------------
        # Plot comparison of central trajectories across months
        # --------------------------------------------------------
        if central_trajs_by_month and not minimal:
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

    # Save summary CSV with all central trajectory parameters
    central_params_rows = []
    for forecaster_name, trajs_by_month in monthly_central_trajectories_by_forecaster.items():
        for sc_month_str, central_path in trajs_by_month.items():
            if central_path is not None:
                row = {
                    'forecaster': forecaster_name,
                    'sc_month': sc_month_str,
                    'sample_idx': central_path.get('sample_idx'),
                    'h_SC_work_months': central_path.get('h_SC'),
                    'T_t_doubling_time_months': central_path.get('T_t'),
                    'cost_speed_months': central_path.get('cost_speed'),
                    'announcement_delay_months': central_path.get('announcement_delay'),
                    'present_prog_multiplier': central_path.get('present_prog_multiplier'),
                    'SC_prog_multiplier': central_path.get('SC_prog_multiplier'),
                    'is_exponential': central_path.get('is_exponential'),
                    'is_superexponential': central_path.get('is_superexponential'),
                    'is_subexponential': central_path.get('is_subexponential'),
                    'patch_rd_speedup': central_path.get('patch_rd_speedup'),
                    'software_progress_share': central_path.get('software_progress_share'),
                    'se_doubling_decay_fraction': central_path.get('se_doubling_decay_fraction'),
                    'sub_doubling_growth_fraction': central_path.get('sub_doubling_growth_fraction'),
                }
                central_params_rows.append(row)
    if central_params_rows:
        central_params_df = pd.DataFrame(central_params_rows)
        central_params_df.to_csv(combined_dir / "central_trajectory_parameters.csv", index=False)
        print(f"Saved central trajectory parameters to {combined_dir / 'central_trajectory_parameters.csv'}")

    if not minimal:
        # --------------------------------------------------------
        # Plot median params vs central trajectories comparison
        # Dynamically determine filter months based on median SC arrival times
        # This is plotted FIRST (after per-forecaster plots) for visibility
        # --------------------------------------------------------
        # Detect forecaster naming pattern: look for patched vs non-patched variants
        forecaster_names_list = list(all_forecaster_results.keys())

        # Check if we're using patched rd speedup variants
        has_patched = any("patched_rd_speedup" in name.lower() for name in forecaster_names_list)

        if has_patched:
            # Use patched variant naming
            base_name = "Eli_patched_rd_speedup"
            superexp_forecaster = "Eli_superexp_only_patched_rd_speedup"
            exp_forecaster = "Eli_exp_only_patched_rd_speedup"
            mixed_forecaster = "Eli_patched_rd_speedup"
            plot_title = "Backcasts with AI R&D interpolation fixed (Eli's distributions)"
        else:
            # Use original naming
            base_name = "Eli"
            superexp_forecaster = "Eli_superexp_only"
            exp_forecaster = "Eli_exp_only"
            mixed_forecaster = "Eli"
            plot_title = "Trajectory with Median Parameters vs Central Trajectories (Eli's distributions)"

        # Fixed median SC arrival months for Eli patched_rd_speedup forecasters
        if has_patched:
            superexp_month = "April 2027"
            exp_month = "April 2029"
            mixed_month = "May 2028"
        else:
            superexp_month = "November 2026"
            exp_month = "July 2028"
            mixed_month = "August 2027"

        print(f"Using SC arrival months: superexp={superexp_month}, exp={exp_month}, mixed={mixed_month}")

        fig_median_vs_central = plot_median_vs_central_comparison(
            median_trajectory_data,
            monthly_central_trajectories_by_forecaster,
            config,
            forecaster_name="Eli",
            superexp_forecaster_name=superexp_forecaster,
            exp_forecaster_name=exp_forecaster,
            mixed_forecaster_name=mixed_forecaster,
            superexp_sc_month=superexp_month,
            exp_sc_month=exp_month,
            mixed_sc_month=mixed_month,
            overlay_external_data=True,
            title=plot_title,
        )
        fig_median_vs_central.savefig(
            central_dir / f"median_vs_central_comparison_{base_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_median_vs_central)
        print("Saved median vs central comparison plot to central_trajectories folder")

        for sc_month_str in target_months:
            # Build list of trajectories for forecasters that have data for this month
            monthly_trajs = [
                (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str])
                for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
                if sc_month_str in monthly_central_trajectories_by_forecaster[forecaster_name]
            ]
            if not monthly_trajs:
                print(f"Skipping {sc_month_str} comparison plot - no forecasters have data for this month")
                continue
            fig_cent_forecaster_comparison_month = plot_central_trajectories_comparison(
                monthly_trajs,
                config,
                overlay_external_data=True,
                title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals",
                sc_month_str=sc_month_str,
            )
            fig_cent_forecaster_comparison_month.savefig(
                central_dir / f"forecaster_comparison_{sc_month_str}_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_cent_forecaster_comparison_month)

        # # Commented out: Compare unconditional central trajectories across forecasters
        # unconditional_central_trajs = [
        #     (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name]["Unconditional"])
        #     for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
        #     if "Unconditional" in monthly_central_trajectories_by_forecaster[forecaster_name]
        # ]
        # if unconditional_central_trajs:
        #     fig_unconditional_comparison = plot_central_trajectories_comparison(
        #         unconditional_central_trajs,
        #         config,
        #         overlay_external_data=True,
        #         title="Time Horizon Extension Central Trajectories – Unconditional",
        #     )
        #     fig_unconditional_comparison.savefig(
        #         central_dir / f"forecaster_comparison_unconditional_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
        #         dpi=300,
        #         bbox_inches="tight",
        #     )
        #     plt.close(fig_unconditional_comparison)

        # --------------------------------------------------------
        # Create separate plots for Nov 2026 and Jan 2027 with parameters
        # --------------------------------------------------------
        for key_month in ["November 2026", "January 2027"]:
            key_month_trajs = []
            for forecaster_name in monthly_central_trajectories_by_forecaster.keys():
                if key_month in monthly_central_trajectories_by_forecaster[forecaster_name]:
                    traj = monthly_central_trajectories_by_forecaster[forecaster_name][key_month]
                    key_month_trajs.append((forecaster_name, traj))
            if key_month_trajs:
                month_slug = key_month.lower().replace(" ", "_")
                fig_key_month = plot_central_trajectories_comparison(
                    key_month_trajs,
                    config,
                    overlay_external_data=True,
                    title=f"Central Trajectories – {key_month} (with parameters)",
                    sc_month_str=key_month,
                    show_params_in_legend=True,
                )
                fig_key_month.savefig(
                    central_dir / f"forecaster_comparison_{month_slug}_with_params.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_key_month)

        # --------------------------------------------------------
        # Get median parameter trajectories for pure growth type forecasters
        # (trajectories already computed by run_median_trajectory earlier)
        # --------------------------------------------------------
        median_param_trajectories = {}  # forecaster_name -> (trajectory, growth_type)
        for forecaster_name in all_forecaster_samples.keys():
            samples = all_forecaster_samples[forecaster_name]
            growth_type = get_forecaster_growth_type(samples)
            if growth_type is not None and forecaster_name in median_trajectory_data:
                traj_data = median_trajectory_data[forecaster_name]
                if growth_type in traj_data:
                    print(f"Using pre-computed median trajectory for {forecaster_name} ({growth_type})")
                    median_param_trajectories[forecaster_name] = (traj_data[growth_type], growth_type)

        # Create versions of central trajectory comparison plots with median param trajectories
        if median_param_trajectories:
            # Build list of median trajectories for plotting
            median_trajs_list = [
                (forecaster_name, traj, growth_type)
                for forecaster_name, (traj, growth_type) in median_param_trajectories.items()
            ]

            # Monthly comparisons with median trajectories
            for sc_month_str in target_months:
                # Build list of trajectories for forecasters that have data for this month
                monthly_trajs_median = [
                    (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str])
                    for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
                    if sc_month_str in monthly_central_trajectories_by_forecaster[forecaster_name]
                ]
                if not monthly_trajs_median:
                    continue
                fig_with_median = plot_central_trajectories_comparison(
                    monthly_trajs_median,
                    config,
                    overlay_external_data=True,
                    title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals (with Median Params)",
                    median_trajectories=median_trajs_list,
                    sc_month_str=sc_month_str,
                )
                fig_with_median.savefig(
                    central_dir / f"forecaster_comparison_{sc_month_str}_with_median_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_with_median)

            # # Commented out: Unconditional comparison with median trajectories
            # if unconditional_central_trajs:
            #     fig_unconditional_with_median = plot_central_trajectories_comparison(
            #         unconditional_central_trajs,
            #         config,
            #         overlay_external_data=True,
            #         title="Time Horizon Extension Central Trajectories – Unconditional (with Median Params)",
            #         median_trajectories=median_trajs_list,
            #     )
            #     fig_unconditional_with_median.savefig(
            #         central_dir / f"forecaster_comparison_unconditional_with_median_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
            #         dpi=300,
            #         bbox_inches="tight",
            #     )
            #     plt.close(fig_unconditional_with_median)

    return fig, all_forecaster_results, monthly_central_trajectories_by_forecaster, median_trajectory_data

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
    forecasted_dir = output_dir / "forecasted_trajectories"
    combined_dir.mkdir(exist_ok=True)
    backcasted_dir.mkdir(exist_ok=True)
    central_dir.mkdir(exist_ok=True)
    forecasted_dir.mkdir(exist_ok=True)

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

    # Save PDF and CDF data to CSVs
    save_pdf_cdf_csvs(all_forecaster_results, config, output_dir)

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

        # # Commented out: unconditional combined trajectories
        # fig_combined_colored, unconditional_central_path = plot_combined_trajectories(
        #     all_forecaster_backcast_trajectories,
        #     all_forecaster_trajectories,
        #     all_forecaster_samples,
        #     config,
        #     color_by_growth_type=True,
        #     plot_median_curve=True,
        #     forecaster_filter=[forecaster_name],
        # )

        # fig_combined_red, _ = plot_combined_trajectories(
        #     all_forecaster_backcast_trajectories,
        #     all_forecaster_trajectories,
        #     all_forecaster_samples,
        #     config,
        #     color_by_growth_type=False,
        #     forecaster_filter=[forecaster_name],
        # )

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
        # # Commented out: saving unconditional combined trajectories
        # fig_combined_colored.savefig(
        #     combined_dir / f"combined_trajectories_{forecaster_name}.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # # Also save unconditional all trajectories to central_trajectories folder
        # fig_combined_colored.savefig(
        #     central_dir / f"all_trajectories_unconditional_{forecaster_name}.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # fig_combined_red.savefig(
        #     combined_dir / f"combined_trajectories_red_{forecaster_name}.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )

        plt.close(fig_backcasted_colored)
        plt.close(fig_backcasted_red)
        # plt.close(fig_combined_colored)
        # plt.close(fig_combined_red)

        # --- Figures filtered by specific SC arrival months ---
        target_months = [
            "November 2026",
            "January 2027",
            "March 2027",
            "April 2027",
            "May 2027",
            "June 2027",
            "August 2027",
            "May 2028",
            "June 2028",
            "July 2028",
            "September 2028",
            "April 2029",
            "November 2029",
            "January 2030",
            "June 2030",
            "July 2030",
            # "March 2028",
            # "March 2029",
            # "March 2030",
            # "September 2027",
            # "September 2029",
            # "September 2030",
        ]

        # Collect central trajectories per month for later comparison
        central_trajs_by_month: dict[str, dict] = {}

        # # Commented out: Add the unconditional central trajectory
        # if unconditional_central_path is not None:
        #     central_trajs_by_month["Unconditional"] = unconditional_central_path

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

            fig_combined_month, all_trajectories = plot_combined_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                forecaster_filter=[forecaster_name],
            )

            # Extract central_incl_backcast as the primary central trajectory for backward compatibility
            central_path = all_trajectories.get('central_incl_backcast')

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

            # Central-only plot (simplified: only Central incl. backcast curve)
            fig_central_only, _ = plot_central_only_trajectories_sc_month(
                all_forecaster_backcast_trajectories,
                all_forecaster_trajectories,
                all_forecaster_samples,
                all_forecaster_results,
                config,
                sc_month_str=sc_month_str,
                color_by_growth_type=True,
                forecaster_filter=[forecaster_name],
            )

            fig_trajectories.savefig(
                forecasted_dir / f"{month_slug}_trajectories_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig_combined_month.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            # Also save to central_trajectories folder for easy access
            fig_combined_month.savefig(
                central_dir / f"all_trajectories_{month_slug}_{forecaster_name}.png",
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
            fig_central_only.savefig(
                combined_dir / f"combined_trajectories_{month_slug}_central_only_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Save all trajectory data as CSV (all 3 central trajectories + median)
            if all_trajectories:
                # Find the common time grid from median (most complete)
                median_traj = all_trajectories.get('median')
                if median_traj is not None:
                    times = median_traj['times']
                    csv_data = {'calendar_time': times}

                    # Add median trajectory
                    csv_data['median_time_horizon_minutes'] = median_traj['horizons']

                    # Add central trajectories (interpolated to common time grid)
                    for key, label in [
                        ('central_incl_backcast', 'central_incl_backcast'),
                        ('central_forecast_only', 'central_forecast_only'),
                        ('central_zscore', 'central_zscore'),
                    ]:
                        traj = all_trajectories.get(key)
                        if traj is not None:
                            # Interpolate to common time grid
                            interp_horizons = np.interp(
                                times,
                                traj['times'],
                                traj['horizons'],
                                left=np.nan,
                                right=np.nan,
                            )
                            csv_data[f'{label}_time_horizon_minutes'] = interp_horizons

                    central_df = pd.DataFrame(csv_data)
                    central_df.to_csv(
                        combined_dir / f"combined_trajectories_{month_slug}_illustrative_{forecaster_name}_central_trajectory.csv",
                        index=False,
                    )

            plt.close(fig_trajectories)
            plt.close(fig_combined_month)
            plt.close(fig_combined_month_median)
            plt.close(fig_combined_month_illustrative)
            plt.close(fig_central_only)

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

    # Save summary CSV with all central trajectory parameters
    central_params_rows = []
    for forecaster_name, trajs_by_month in monthly_central_trajectories_by_forecaster.items():
        for sc_month_str, central_path in trajs_by_month.items():
            if central_path is not None:
                row = {
                    'forecaster': forecaster_name,
                    'sc_month': sc_month_str,
                    'sample_idx': central_path.get('sample_idx'),
                    'h_SC_work_months': central_path.get('h_SC'),
                    'T_t_doubling_time_months': central_path.get('T_t'),
                    'cost_speed_months': central_path.get('cost_speed'),
                    'announcement_delay_months': central_path.get('announcement_delay'),
                    'present_prog_multiplier': central_path.get('present_prog_multiplier'),
                    'SC_prog_multiplier': central_path.get('SC_prog_multiplier'),
                    'is_exponential': central_path.get('is_exponential'),
                    'is_superexponential': central_path.get('is_superexponential'),
                    'is_subexponential': central_path.get('is_subexponential'),
                    'patch_rd_speedup': central_path.get('patch_rd_speedup'),
                    'software_progress_share': central_path.get('software_progress_share'),
                    'se_doubling_decay_fraction': central_path.get('se_doubling_decay_fraction'),
                    'sub_doubling_growth_fraction': central_path.get('sub_doubling_growth_fraction'),
                }
                central_params_rows.append(row)
    if central_params_rows:
        central_params_df = pd.DataFrame(central_params_rows)
        central_params_df.to_csv(combined_dir / "central_trajectory_parameters.csv", index=False)
        print(f"Saved central trajectory parameters to {combined_dir / 'central_trajectory_parameters.csv'}")

    # --------------------------------------------------------
    # Plot median params vs central trajectories comparison
    # Dynamically determine filter months based on median SC arrival times
    # This is plotted FIRST (after per-forecaster plots) for visibility
    # --------------------------------------------------------
    # Detect forecaster naming pattern: look for patched vs non-patched variants
    forecaster_names_list = list(all_forecaster_results.keys())

    # Check if we're using patched rd speedup variants
    has_patched = any("patched_rd_speedup" in name.lower() for name in forecaster_names_list)

    if has_patched:
        # Use patched variant naming
        base_name = "Eli_patched_rd_speedup"
        superexp_forecaster = "Eli_superexp_only_patched_rd_speedup"
        exp_forecaster = "Eli_exp_only_patched_rd_speedup"
        mixed_forecaster = "Eli_patched_rd_speedup"
        plot_title = "Backcasts with AI R&D interpolation fixed (Eli's distributions)"
    else:
        # Use original naming
        base_name = "Eli"
        superexp_forecaster = "Eli_superexp_only"
        exp_forecaster = "Eli_exp_only"
        mixed_forecaster = "Eli"
        plot_title = "Trajectory with Median Parameters vs Central Trajectories (Eli's distributions)"

    # Fixed median SC arrival months for Eli patched_rd_speedup forecasters
    if has_patched:
        superexp_month = "April 2027"
        exp_month = "April 2029"
        mixed_month = "May 2028"
    else:
        superexp_month = "November 2026"
        exp_month = "July 2028"
        mixed_month = "August 2027"

    print(f"Using SC arrival months: superexp={superexp_month}, exp={exp_month}, mixed={mixed_month}")

    fig_median_vs_central = plot_median_vs_central_comparison(
        median_trajectory_data,
        monthly_central_trajectories_by_forecaster,
        config,
        forecaster_name="Eli",
        superexp_forecaster_name=superexp_forecaster,
        exp_forecaster_name=exp_forecaster,
        mixed_forecaster_name=mixed_forecaster,
        superexp_sc_month=superexp_month,
        exp_sc_month=exp_month,
        mixed_sc_month=mixed_month,
        overlay_external_data=True,
        title=plot_title,
    )
    fig_median_vs_central.savefig(
        central_dir / f"median_vs_central_comparison_{base_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_median_vs_central)
    print("Saved median vs central comparison plot to central_trajectories folder")

    for sc_month_str in target_months:
        # Build list of trajectories for forecasters that have data for this month
        monthly_trajs = [
            (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str])
            for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
            if sc_month_str in monthly_central_trajectories_by_forecaster[forecaster_name]
        ]
        if not monthly_trajs:
            print(f"Skipping {sc_month_str} comparison plot - no forecasters have data for this month")
            continue
        fig_cent_forecaster_comparison_month = plot_central_trajectories_comparison(
            monthly_trajs,
            config,
            overlay_external_data=True,
            title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals",
            sc_month_str=sc_month_str,
        )
        fig_cent_forecaster_comparison_month.savefig(
            central_dir / f"forecaster_comparison_{sc_month_str}_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_cent_forecaster_comparison_month)

    # # Commented out: Compare unconditional central trajectories across forecasters
    # unconditional_central_trajs = [
    #     (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name]["Unconditional"])
    #     for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
    #     if "Unconditional" in monthly_central_trajectories_by_forecaster[forecaster_name]
    # ]
    # if unconditional_central_trajs:
    #     fig_unconditional_comparison = plot_central_trajectories_comparison(
    #         unconditional_central_trajs,
    #         config,
    #         overlay_external_data=True,
    #         title="Time Horizon Extension Central Trajectories – Unconditional",
    #     )
    #     fig_unconditional_comparison.savefig(
    #         central_dir / f"forecaster_comparison_unconditional_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
    #         dpi=300,
    #         bbox_inches="tight",
    #     )
    #     plt.close(fig_unconditional_comparison)

    # --------------------------------------------------------
    # Create separate plots for Nov 2026 and Jan 2027 with parameters
    # --------------------------------------------------------
    for key_month in ["November 2026", "January 2027"]:
        key_month_trajs = []
        for forecaster_name in monthly_central_trajectories_by_forecaster.keys():
            if key_month in monthly_central_trajectories_by_forecaster[forecaster_name]:
                traj = monthly_central_trajectories_by_forecaster[forecaster_name][key_month]
                key_month_trajs.append((forecaster_name, traj))
        if key_month_trajs:
            month_slug = key_month.lower().replace(" ", "_")
            fig_key_month = plot_central_trajectories_comparison(
                key_month_trajs,
                config,
                overlay_external_data=True,
                title=f"Central Trajectories – {key_month} (with parameters)",
                sc_month_str=key_month,
                show_params_in_legend=True,
            )
            fig_key_month.savefig(
                central_dir / f"forecaster_comparison_{month_slug}_with_params.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_key_month)

    # --------------------------------------------------------
    # Compute median parameter trajectories for pure growth type forecasters
    # --------------------------------------------------------
    print("\nComputing median trajectories for pure growth type forecasters...")
    median_trajectory_data = {}  # forecaster_name -> {growth_type: trajectory_dict}
    for _, forecaster_config in config["forecasters"].items():
        forecaster_name = forecaster_config["name"]
        if forecaster_name not in all_forecaster_samples:
            continue
        _, traj_data, _ = run_median_trajectory(config, forecaster_config, forecaster_name)
        median_trajectory_data[forecaster_name] = traj_data

    median_param_trajectories = {}  # forecaster_name -> (trajectory, growth_type)
    for forecaster_name in all_forecaster_samples.keys():
        samples = all_forecaster_samples[forecaster_name]
        growth_type = get_forecaster_growth_type(samples)
        if growth_type is not None and forecaster_name in median_trajectory_data:
            traj_data = median_trajectory_data[forecaster_name]
            if growth_type in traj_data:
                print(f"Using median trajectory for {forecaster_name} ({growth_type})")
                median_param_trajectories[forecaster_name] = (traj_data[growth_type], growth_type)

    # Create versions of central trajectory comparison plots with median param trajectories
    if median_param_trajectories:
        # Build list of median trajectories for plotting
        median_trajs_list = [
            (forecaster_name, traj, growth_type)
            for forecaster_name, (traj, growth_type) in median_param_trajectories.items()
        ]

        # Monthly comparisons with median trajectories
        for sc_month_str in target_months:
            # Build list of trajectories for forecasters that have data for this month
            monthly_trajs_median = [
                (forecaster_name, monthly_central_trajectories_by_forecaster[forecaster_name][sc_month_str])
                for forecaster_name in monthly_central_trajectories_by_forecaster.keys()
                if sc_month_str in monthly_central_trajectories_by_forecaster[forecaster_name]
            ]
            if not monthly_trajs_median:
                continue
            fig_with_median = plot_central_trajectories_comparison(
                monthly_trajs_median,
                config,
                overlay_external_data=True,
                title=f"Time Horizon Extension Central Trajectories – {sc_month_str} SC Arrivals (with Median Params)",
                median_trajectories=median_trajs_list,
                sc_month_str=sc_month_str,
            )
            fig_with_median.savefig(
                central_dir / f"forecaster_comparison_{sc_month_str}_with_median_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_with_median)

        # # Commented out: Unconditional comparison with median trajectories
        # if unconditional_central_trajs:
        #     fig_unconditional_with_median = plot_central_trajectories_comparison(
        #         unconditional_central_trajs,
        #         config,
        #         overlay_external_data=True,
        #         title="Time Horizon Extension Central Trajectories – Unconditional (with Median Params)",
        #         median_trajectories=median_trajs_list,
        #     )
        #     fig_unconditional_with_median.savefig(
        #         central_dir / f"forecaster_comparison_unconditional_with_median_{str(monthly_central_trajectories_by_forecaster.keys())}.png",
        #         dpi=300,
        #         bbox_inches="tight",
        #     )
        #     plt.close(fig_unconditional_with_median)

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

        # Parse arguments
        args = sys.argv[1:]
        minimal = "--minimal" in args
        if minimal:
            args.remove("--minimal")
            print("Running in MINIMAL mode: only PDF/CDF + March 2027 illustrative plots")

        # Parse --n_sims argument
        n_sims = None
        if "--n_sims" in args:
            idx = args.index("--n_sims")
            n_sims = int(args[idx + 1])
            args.pop(idx)  # Remove --n_sims
            args.pop(idx)  # Remove the value

        config_path = args[0] if args else "simple_params.yaml"
        run_simple_sc_simulation(config_path, minimal=minimal, n_sims=n_sims)

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 