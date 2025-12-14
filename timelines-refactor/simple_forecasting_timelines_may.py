import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm
import yaml
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
from simple_forecasting_timelines_plotting import *
from timelines_common import *

# Set the disclaimer variant for May plots
set_disclaimer_variant("may")


# Cache for new model interpolation data
_new_model_interpolation_cache = None


def get_new_model_interpolation_fraction(progress_fraction: float) -> float:
    """
    Transform progress_fraction using the New Model (Dec 2025) curve.

    Maps the model's progress_fraction to the normalized multiplier fraction
    from the New Model curve. For example, if progress_fraction=0.5 (50% of way to SC),
    this looks up what fraction of the normalized multiplier the New Model has at 50% progress
    (approximately 0.15 or 15%), and returns that value to use for interpolation.

    Uses formula: fraction = (log(cur)/log(start)) / (log(end)/log(start))
    """
    global _new_model_interpolation_cache

    if _new_model_interpolation_cache is None:
        # Load external CSV and build interpolation
        external_csv_path = Path(__file__).parent.parent / "external" / "inputs" / "ai_progress_results_20251211_034612.csv"

        if not external_csv_path.exists():
            # Fall back to linear interpolation if file doesn't exist
            return progress_fraction

        ext_df = pd.read_csv(external_csv_path, comment='#')

        # Find cumulative_progress at 2025.6 (start reference)
        ref_start_time = 2025.6
        closest_start_idx = (ext_df['time'] - ref_start_time).abs().idxmin()
        cumulative_progress_at_start = ext_df.loc[closest_start_idx, 'cumulative_progress']
        ext_start_mult = ext_df.loc[closest_start_idx, 'ai_software_progress_multiplier']

        # Find cumulative_progress at SC (use 2031.1 as the end point)
        ref_sc_time = 2031.1
        closest_sc_idx = (ext_df['time'] - ref_sc_time).abs().idxmin()
        cumulative_progress_at_sc = ext_df.loc[closest_sc_idx, 'cumulative_progress']
        ext_sc_mult = ext_df.loc[closest_sc_idx, 'ai_software_progress_multiplier']

        # Filter to time range from 2025.6 onwards
        ext_df = ext_df[ext_df['time'] >= ref_start_time].copy()

        # Calculate % of way to SC (progress_pct from 0 to 1)
        progress_range = cumulative_progress_at_sc - cumulative_progress_at_start
        ext_df['progress_fraction'] = (ext_df['cumulative_progress'] - cumulative_progress_at_start) / progress_range

        # Filter to 0-1 range
        ext_df = ext_df[(ext_df['progress_fraction'] >= 0) & (ext_df['progress_fraction'] <= 1)]

        # Calculate normalized multiplier using log scale normalization on (mult - 1):
        # fraction = log((cur-1)/(start-1)) / log((end-1)/(start-1))
        ext_df['mult_normalized'] = np.log((ext_df['ai_software_progress_multiplier'] - 1) / (ext_start_mult - 1)) / np.log((ext_sc_mult - 1) / (ext_start_mult - 1))

        # Build interpolation arrays
        _new_model_interpolation_cache = {
            'progress_fractions': ext_df['progress_fraction'].values,
            'mult_normalized': ext_df['mult_normalized'].values
        }

    # Interpolate to find the normalized multiplier at the given progress_fraction
    return np.interp(
        progress_fraction,
        _new_model_interpolation_cache['progress_fractions'],
        _new_model_interpolation_cache['mult_normalized']
    )


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


def get_median_samples(config: dict) -> dict:
    """Generate a single sample using median values for all parameters."""
    samples = {}

    # Get median for initial_software_progress_share (normal distribution)
    # Check multiple locations for the config key
    if "initial_software_progress_share_ci" in config:
        lower, upper = config["initial_software_progress_share_ci"]
    elif "distributions" in config and "initial_software_progress_share_ci" in config["distributions"]:
        lower, upper = config["distributions"]["initial_software_progress_share_ci"]
    else:
        # Default to 0.5 if not specified
        lower, upper = 0.5, 0.5
    samples["initial_software_progress_share"] = np.array([(lower + upper) / 2])

    # Get median (50th percentile) for each lognormal distribution
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = np.array([dist.median()])

    dist = get_lognormal_from_80_ci(
        config["distributions"]["horizon_doubling_time_ci"][0],
        config["distributions"]["horizon_doubling_time_ci"][1]
    )
    samples["horizon_doubling_time"] = np.array([dist.median()])

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

    # Use superexponential growth for median trajectory
    # Find the first horizon where cumulative probability >= 0.5
    superexp_schedule = config["distributions"]["superexponential_schedule_months"]
    median_superexp_start = np.inf
    for horizon, prob in superexp_schedule:
        if prob >= 0.5:
            median_superexp_start = horizon
            break

    samples["is_superexponential"] = np.array([median_superexp_start < np.inf])
    samples["is_subexponential"] = np.array([False])
    samples["is_exponential"] = np.array([median_superexp_start == np.inf])
    samples["superexponential_start_horizon"] = np.array([median_superexp_start])
    samples["superexponential_schedule_months"] = superexp_schedule

    # Get median for se_doubling_decay_fraction
    dist = get_lognormal_from_80_ci(
        config["distributions"]["se_doubling_decay_fraction_ci"][0],
        config["distributions"]["se_doubling_decay_fraction_ci"][1]
    )
    samples["se_doubling_decay_fraction"] = np.array([dist.median()])

    # Growth/decay parameters
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]

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


def run_median_trajectory_with_growth_dynamics(
    config: dict, forecaster_config: dict, forecaster_name: str
) -> tuple[list, float]:
    """Run a single median trajectory and track growth dynamics over time.

    Returns:
        growth_dynamics: List of dicts with keys:
            - year: calendar year
            - human_only_research_contribution: research contribution if software_prog_multiplier were 1
            - software_prog_multiplier: the actual software progress multiplier
            - research_contribution: actual research contribution
            - research_stock: cumulative research stock
            - baseline_growth: the baseline growth rate (stays constant)
            - actual_growth: the actual growth rate at this timestep
        baseline_growth_value: The constant baseline growth value
    """
    # Get median samples
    median_samples = get_median_samples(forecaster_config)

    # Get simulation parameters (forecaster can override current_horizon and start_year)
    current_horizon = forecaster_config.get("current_horizon", config["simulation"]["current_horizon"])
    dt = config["simulation"]["dt"]
    max_time = config["simulation"]["max_time"]
    start_year = forecaster_config.get("start_year", config["simulation"]["start_year"])

    # Calculate base time and horizon mappings
    base_time_in_months, horizon_mappings = calculate_base_time(median_samples, current_horizon)

    # Get software progress share from samples
    software_progress_share = median_samples["initial_software_progress_share"]

    # Initialize simulation variables
    time = start_year - median_samples["announcement_delay"][0] / 12
    progress = 0.0
    dt_in_months = dt / 30.5

    # Initialize labor-based research variables
    labor_pool = config["simulation"]["initial_labor_pool"]
    research_stock = config["simulation"]["initial_research_stock"]
    labor_power = config["simulation"]["labor_power"]

    # Track growth dynamics
    growth_dynamics = []
    baseline_growth_value = None

    i = 0  # Single simulation index

    while progress < base_time_in_months[i] and time < max_time:
        # Calculate progress fraction
        progress_fraction = progress / base_time_in_months[i]

        # Transform progress_fraction if using new model interpolation
        if forecaster_config.get("use_new_model_interpolation", False):
            interpolation_fraction = get_new_model_interpolation_fraction(progress_fraction)
        else:
            interpolation_fraction = progress_fraction

        # Calculate software speedup
        if forecaster_config.get("automation_off", False):
            software_prog_multiplier = 1.0
        elif forecaster_config.get("patch_rd_speedup", False):
            software_prog_multiplier = 1 + (median_samples["present_prog_multiplier"][i]) * (
                (median_samples["SC_prog_multiplier"][i]) / (median_samples["present_prog_multiplier"][i])
            ) ** interpolation_fraction
        else:
            software_prog_multiplier = (1 + median_samples["present_prog_multiplier"][i]) * (
                (1 + median_samples["SC_prog_multiplier"][i]) / (1 + median_samples["present_prog_multiplier"][i])
            ) ** interpolation_fraction

        # Get current labor growth rate from schedule
        current_labor_growth_rate = get_labor_growth_rate(time, forecaster_config["labor_growth_schedule"])
        # Convert annual growth rate to daily rate for the time step
        daily_growth_rate = (1 + current_labor_growth_rate) ** (dt / 250) - 1

        # Calculate new labor added this period
        new_labor = labor_pool * daily_growth_rate
        labor_pool += new_labor

        # Calculate research contribution (actual, with software multiplier) - per timestep
        research_contribution = ((((labor_pool + 1) ** labor_power) - 1) * software_prog_multiplier) / (250 / dt)

        # Calculate human-only research contribution (as if software_prog_multiplier were 1) - per timestep
        human_only_research_contribution = ((((labor_pool + 1) ** labor_power) - 1) * 1.0) / (250 / dt)

        # Annualize research contributions (250 working days per year)
        research_contribution_annualized = research_contribution * (250 / dt)
        human_only_research_contribution_annualized = human_only_research_contribution * (250 / dt)

        # Add to research stock
        new_research_stock = research_stock + research_contribution

        # Calculate actual growth rate (per timestep)
        actual_growth = new_research_stock / research_stock

        # Annualize growth rates (250 working days per year)
        actual_growth_annualized = actual_growth ** (250 / dt)

        # Set baseline growth on first iteration
        if progress == 0:
            baseline_growth_value = actual_growth
            baseline_growth_annualized = actual_growth_annualized

        # Calculate adjustment factor based on growth rate ratio
        growth_ratio = np.log(actual_growth) / np.log(baseline_growth_value)

        # Record growth dynamics
        growth_dynamics.append({
            "year": time + median_samples["announcement_delay"][0] / 12,
            "progress_fraction": progress_fraction,
            "human_only_research_contribution_annualized": human_only_research_contribution_annualized,
            "software_prog_multiplier": software_prog_multiplier,
            "research_contribution_annualized": research_contribution_annualized,
            "research_stock": research_stock,
            "baseline_growth_annualized": baseline_growth_annualized,
            "actual_growth_annualized": actual_growth_annualized,
            "relative_software_progress_rate": growth_ratio,
        })

        # Get compute rate for current time using compute schedule
        compute_rate = get_compute_rate(time, forecaster_config["compute_schedule"])

        # Total rate is weighted average of growth_ratio and compute rates
        total_rate = software_progress_share[i] * growth_ratio + (1 - software_progress_share[i]) * compute_rate

        # Update progress and time
        progress += dt_in_months * total_rate
        time += dt_in_months / 12  # Convert months to years

        # Update research stock
        research_stock = new_research_stock

    return growth_dynamics, baseline_growth_value


def plot_growth_dynamics(
    growth_dynamics: list,
    baseline_growth_value: float,
    forecaster_name: str,
    config: dict,
    output_path: Path,
) -> plt.Figure:
    """Plot growth dynamics quantities over time on a single graph.

    Plots:
    - human_only_research_contribution
    - software_prog_multiplier
    - research_contribution
    - research_stock
    - baseline_growth_annualized (constant, dotted line)
    - actual_growth_annualized
    - relative_software_progress_rate (growth ratio)
    """
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor=bg_rgb)
    ax.set_facecolor(bg_rgb)

    # Extract data
    years = [d["year"] for d in growth_dynamics]
    human_only = [d["human_only_research_contribution_annualized"] for d in growth_dynamics]
    software_mult = [d["software_prog_multiplier"] for d in growth_dynamics]
    research_contrib = [d["research_contribution_annualized"] for d in growth_dynamics]
    research_stock = [d["research_stock"] for d in growth_dynamics]
    baseline_growth = [d["baseline_growth_annualized"] for d in growth_dynamics]
    actual_growth = [d["actual_growth_annualized"] for d in growth_dynamics]
    relative_sw_progress = [d["relative_software_progress_rate"] for d in growth_dynamics]

    # Plot each quantity with different colors
    ax.plot(years, human_only, label="Human-Only Research Contribution (Annualized)", color="#1f77b4", linewidth=2)
    ax.plot(years, software_mult, label="Software Progress Multiplier", color="#ff7f0e", linewidth=2)
    ax.plot(years, research_contrib, label="Research Contribution (Annualized)", color="#2ca02c", linewidth=2)
    ax.plot(years, research_stock, label="Research Stock", color="#d62728", linewidth=2)
    ax.plot(years, baseline_growth, label="Baseline Growth (Annualized)", color="#9467bd", linewidth=2, linestyle=":")
    ax.plot(years, actual_growth, label="Actual Growth (Annualized)", color="#8c564b", linewidth=2)
    ax.plot(years, relative_sw_progress, label="Relative Software Progress Rate", color="#e377c2", linewidth=2)

    # Configure plot
    ax.set_title(f"Growth Dynamics Over Time - {forecaster_name}",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Value (log scale)", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_yscale("log")

    # Grid and spines
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.9)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_growth_dynamics_with_external_data(
    growth_dynamics: list,
    baseline_growth_value: float,
    forecaster_name: str,
    config: dict,
    output_path: Path,
    external_csv_path: str = None,
) -> plt.Figure:
    """Plot growth dynamics with external data overlay.

    Plots model data:
    - software_prog_multiplier
    - research_stock
    - research_effort (research_contribution_annualized)
    - relative_software_progress_rate

    Overlays external data (dashed lines):
    - ai_software_progress_multiplier
    - research_stock
    - research_effort
    - relative_software_progress_rate (computed as software_progress_rate / software_progress_rate_at_2025.6)
    """
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor=bg_rgb)
    ax.set_facecolor(bg_rgb)

    # Extract model data
    years = [d["year"] for d in growth_dynamics]
    software_mult = [d["software_prog_multiplier"] for d in growth_dynamics]
    research_stock = [d["research_stock"] for d in growth_dynamics]
    research_effort = [d["research_contribution_annualized"] for d in growth_dynamics]
    relative_sw_progress = [d["relative_software_progress_rate"] for d in growth_dynamics]

    # Plot model data (solid lines)
    ax.plot(years, software_mult, label="Previous Model (Apr 2025): Software Progress Multiplier", color="#ff7f0e", linewidth=2)
    ax.plot(years, research_stock, label="Previous Model (Apr 2025): Research Stock", color="#d62728", linewidth=2)
    ax.plot(years, research_effort, label="Previous Model (Apr 2025): Research Effort", color="#2ca02c", linewidth=2)
    ax.plot(years, relative_sw_progress, label="Previous Model (Apr 2025): Relative Software Progress Rate", color="#e377c2", linewidth=2)

    # Load and plot external data if provided
    if external_csv_path and Path(external_csv_path).exists():
        # Read external CSV, skipping comment lines
        ext_df = pd.read_csv(external_csv_path, comment='#')

        # Filter to relevant time range (around the model's time range)
        min_year = min(years) - 1
        max_year = max(years) + 1
        ext_df = ext_df[(ext_df['time'] >= min_year) & (ext_df['time'] <= max_year)]

        # Calculate relative software progress rate
        # Find software_progress_rate at 2025.6 (or closest time)
        ref_time = 2025.6
        closest_idx = (ext_df['time'] - ref_time).abs().idxmin()
        software_progress_rate_at_ref = ext_df.loc[closest_idx, 'software_progress_rate']

        ext_df['relative_software_progress_rate'] = ext_df['software_progress_rate'] / software_progress_rate_at_ref

        # Plot external data (dashed lines)
        ax.plot(ext_df['time'], ext_df['ai_software_progress_multiplier'],
                label="New Model (Dec 2025): AI Software Progress Multiplier", color="#ff7f0e", linewidth=2, linestyle='--')
        ax.plot(ext_df['time'], ext_df['research_stock'],
                label="New Model (Dec 2025): Research Stock", color="#d62728", linewidth=2, linestyle='--')
        ax.plot(ext_df['time'], ext_df['research_effort'],
                label="New Model (Dec 2025): Research Effort", color="#2ca02c", linewidth=2, linestyle='--')
        ax.plot(ext_df['time'], ext_df['relative_software_progress_rate'],
                label="New Model (Dec 2025): Relative Software Progress Rate", color="#e377c2", linewidth=2, linestyle='--')

    # Configure plot
    ax.set_title(f"Growth Dynamics Comparison - {forecaster_name}",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Value (log scale)", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_yscale("log")

    # Grid and spines
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"] - 2, framealpha=0.9)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_software_multiplier_vs_progress(
    growth_dynamics: list,
    forecaster_name: str,
    config: dict,
    output_path: Path,
    external_csv_path: str = None,
) -> plt.Figure:
    """Plot normalized software progress multipliers vs % of way to SC.

    X-axis: % of way to SC (0-100%)
    Y-axis: % of the way to SC progress multiplier (normalized 0-1, log scale)

    For model data: uses progress_fraction directly, normalizes by SC multiplier value
    For external data: calculates % as (cumulative_progress - cumulative_progress_at_2025.6) /
                       (cumulative_progress_at_SC - cumulative_progress_at_2025.6)
                       normalizes by SC multiplier value
    """
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax.set_facecolor(bg_rgb)

    # Extract model data
    progress_pct = [d["progress_fraction"] * 100 for d in growth_dynamics]  # Convert to percentage
    software_mult = [d["software_prog_multiplier"] for d in growth_dynamics]

    # Normalize model data using log scale normalization on (mult - 1):
    # fraction = log((cur-1)/(start-1)) / log((end-1)/(start-1))
    model_sc_mult = software_mult[-1]  # SC multiplier is the final value
    model_start_mult = software_mult[0]  # Starting multiplier

    # Handle edge case where start multiplier is 1 (no AI contribution yet)
    # Find first multiplier > 1 to use as reference, or fall back to linear normalization
    if model_start_mult <= 1.0:
        # Find first multiplier that's meaningfully > 1
        for i, m in enumerate(software_mult):
            if m > 1.001:
                model_start_mult = m
                break

    if model_start_mult <= 1.0 or model_sc_mult <= model_start_mult:
        # Fall back to linear normalization if we can't use log formula
        model_mult_normalized = [(m - 1) / (model_sc_mult - 1) for m in software_mult]
    else:
        log_denom = np.log((model_sc_mult - 1) / (model_start_mult - 1))
        model_mult_normalized = [
            np.log((m - 1) / (model_start_mult - 1)) / log_denom if m > model_start_mult else 0.0
            for m in software_mult
        ]

    # Plot model data (solid line)
    ax.plot(progress_pct, model_mult_normalized, label=f"Previous Model (Apr 2025) (SC mult: {model_sc_mult:.1f})", color="#ff7f0e", linewidth=2)

    # Load and plot external data if provided
    if external_csv_path and Path(external_csv_path).exists():
        # Read external CSV, skipping comment lines
        ext_df = pd.read_csv(external_csv_path, comment='#')

        # Find cumulative_progress at 2025.6 (start reference)
        ref_start_time = 2025.6
        closest_start_idx = (ext_df['time'] - ref_start_time).abs().idxmin()
        cumulative_progress_at_start = ext_df.loc[closest_start_idx, 'cumulative_progress']
        ext_start_mult = ext_df.loc[closest_start_idx, 'ai_software_progress_multiplier']

        # Find cumulative_progress at SC (use 2031.1 as the end point)
        ref_sc_time = 2031.1
        closest_sc_idx = (ext_df['time'] - ref_sc_time).abs().idxmin()
        cumulative_progress_at_sc = ext_df.loc[closest_sc_idx, 'cumulative_progress']
        ext_sc_mult = ext_df.loc[closest_sc_idx, 'ai_software_progress_multiplier']

        # Filter to time range from 2025.6 onwards
        ext_df = ext_df[ext_df['time'] >= ref_start_time].copy()

        # Calculate % of way to SC
        progress_range = cumulative_progress_at_sc - cumulative_progress_at_start
        ext_df['progress_pct'] = (ext_df['cumulative_progress'] - cumulative_progress_at_start) / progress_range * 100

        # Filter to 0-100% range
        ext_df = ext_df[(ext_df['progress_pct'] >= 0) & (ext_df['progress_pct'] <= 100)]

        # Normalize external data using log scale normalization on (mult - 1):
        # fraction = log((cur-1)/(start-1)) / log((end-1)/(start-1))
        ext_df['mult_normalized'] = np.log((ext_df['ai_software_progress_multiplier'] - 1) / (ext_start_mult - 1)) / np.log((ext_sc_mult - 1) / (ext_start_mult - 1))

        # Plot external data (dashed line)
        ax.plot(ext_df['progress_pct'], ext_df['mult_normalized'],
                label=f"New Model (Dec 2025) (SC mult: {ext_sc_mult:.1f})", color="#ff7f0e", linewidth=2, linestyle='--')

    # Configure plot
    ax.set_title(f"Normalized Software Progress Multiplier vs Progress to SC - {forecaster_name}",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("% of Way to SC", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("% of Way to SC Progress Multiplier", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)  # Linear scale, 0 to slightly above 1

    # Grid and spines
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.9)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])

    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_software_multiplier_vs_progress_absolute(
    growth_dynamics: list,
    forecaster_name: str,
    config: dict,
    output_path: Path,
    external_csv_path: str = None,
) -> plt.Figure:
    """Plot absolute software progress multipliers vs % of way to SC (log scale y-axis).

    X-axis: % of way to SC (0-100%)
    Y-axis: Software progress multiplier (absolute value, log scale)

    For model data: uses progress_fraction directly
    For external data: calculates % as (cumulative_progress - cumulative_progress_at_2025.6) /
                       (cumulative_progress_at_SC - cumulative_progress_at_2025.6)
    """
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax.set_facecolor(bg_rgb)

    # Extract model data
    progress_pct = [d["progress_fraction"] * 100 for d in growth_dynamics]  # Convert to percentage
    software_mult = [d["software_prog_multiplier"] for d in growth_dynamics]

    model_sc_mult = software_mult[-1]  # SC multiplier is the final value

    # Plot model data (solid line)
    ax.plot(progress_pct, software_mult, label=f"Previous Model (Apr 2025) (SC mult: {model_sc_mult:.1f})", color="#ff7f0e", linewidth=2)

    # Load and plot external data if provided
    if external_csv_path and Path(external_csv_path).exists():
        # Read external CSV, skipping comment lines
        ext_df = pd.read_csv(external_csv_path, comment='#')

        # Find cumulative_progress at 2025.6 (start reference)
        ref_start_time = 2025.6
        closest_start_idx = (ext_df['time'] - ref_start_time).abs().idxmin()
        cumulative_progress_at_start = ext_df.loc[closest_start_idx, 'cumulative_progress']

        # Find cumulative_progress at SC (use 2031.1 as the end point)
        ref_sc_time = 2031.1
        closest_sc_idx = (ext_df['time'] - ref_sc_time).abs().idxmin()
        cumulative_progress_at_sc = ext_df.loc[closest_sc_idx, 'cumulative_progress']
        ext_sc_mult = ext_df.loc[closest_sc_idx, 'ai_software_progress_multiplier']

        # Filter to time range from 2025.6 onwards
        ext_df = ext_df[ext_df['time'] >= ref_start_time].copy()

        # Calculate % of way to SC
        progress_range = cumulative_progress_at_sc - cumulative_progress_at_start
        ext_df['progress_pct'] = (ext_df['cumulative_progress'] - cumulative_progress_at_start) / progress_range * 100

        # Filter to 0-100% range
        ext_df = ext_df[(ext_df['progress_pct'] >= 0) & (ext_df['progress_pct'] <= 100)]

        # Plot external data (dashed line)
        ax.plot(ext_df['progress_pct'], ext_df['ai_software_progress_multiplier'],
                label=f"New Model (Dec 2025) (SC mult: {ext_sc_mult:.1f})", color="#ff7f0e", linewidth=2, linestyle='--')

    # Configure plot
    ax.set_title(f"Software Progress Multiplier vs Progress to SC - {forecaster_name}",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("% of Way to SC", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Software Progress Multiplier", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_xlim(0, 100)
    ax.set_yscale('log')

    # Grid and spines
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.9)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])

    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def run_median_trajectory(config: dict, forecaster_config: dict, forecaster_name: str) -> tuple[str, dict]:
    """Run a single trajectory with median parameters and return the results as a string and structured data."""
    # Get median samples
    median_samples = get_median_samples(forecaster_config)

    # Build output string
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"MEDIAN TRAJECTORY FOR: {forecaster_name}")
    lines.append(f"{'='*60}")
    lines.append(f"Median Parameters:")
    lines.append(f"  h_SC (work-months): {median_samples['h_SC'][0]:.2f}")
    lines.append(f"  horizon_doubling_time (months): {median_samples['horizon_doubling_time'][0]:.2f}")
    lines.append(f"  cost_speed (months): {median_samples['cost_speed'][0]:.2f}")
    lines.append(f"  announcement_delay (months): {median_samples['announcement_delay'][0]:.2f}")
    lines.append(f"  present_prog_multiplier: {median_samples['present_prog_multiplier'][0]:.3f}")
    lines.append(f"  SC_prog_multiplier: {median_samples['SC_prog_multiplier'][0]:.2f}")
    lines.append(f"  initial_software_progress_share: {median_samples['initial_software_progress_share'][0]:.3f}")
    if median_samples["is_superexponential"][0]:
        lines.append(f"  growth_type: Superexponential (starts at {median_samples['superexponential_start_horizon'][0]:.1f} month horizon)")
        lines.append(f"  se_doubling_decay_fraction: {median_samples['se_doubling_decay_fraction'][0]:.3f}")
    else:
        lines.append(f"  growth_type: Exponential")

    # Run simulation (forecaster can override current_horizon)
    current_horizon = forecaster_config.get("current_horizon", config["simulation"]["current_horizon"])
    dt = config["simulation"]["dt"]
    human_alg_progress_decrease_date = config["simulation"]["human_alg_progress_decrease_date"]
    max_simulation_years = config["simulation"]["max_simulation_years"]

    # Run simulation (May version has different signature)
    ending_times, trajectories, _, prog_multiplier_trajectories = calculate_sc_arrival_year_with_trajectories(
        median_samples, current_horizon, dt,
        human_alg_progress_decrease_date, max_simulation_years,
        forecaster_config, config["simulation"]
    )

    arrival_year = ending_times[0]  # Internal time (without announcement delay)
    trajectory = trajectories[0]
    prog_multiplier_trajectory = prog_multiplier_trajectories[0] if prog_multiplier_trajectories else []
    announcement_delay_years = median_samples['announcement_delay'][0] / 12

    # Display SC arrival in internal time (when it actually happens, not when announced)
    lines.append(f"\nMedian SC Arrival: {format_year_month(arrival_year)}")

    # Find human-cost-parity point (when horizon first reaches h_SC requirement)
    h_SC_threshold_minutes = median_samples['h_SC'][0] * 60 * 167  # Convert work-months to minutes
    human_cost_parity_year_announced = None  # Trajectory times include announcement delay
    if trajectory:
        for t, h in trajectory:
            if h >= h_SC_threshold_minutes:
                human_cost_parity_year_announced = t
                break

    if human_cost_parity_year_announced is not None:
        # Convert HCP to internal time (subtract announcement delay) to match SC arrival reference frame
        human_cost_parity_year = human_cost_parity_year_announced - announcement_delay_years
        lines.append(f"Human-Cost-Parity: {format_year_month(human_cost_parity_year)}")
        # Calculate days between human-cost-parity and SC arrival (both in internal time)
        days_to_sc = (arrival_year - human_cost_parity_year) * 365.25
        lines.append(f"Days from Human-Cost-Parity to SC: {days_to_sc:.1f} days")

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

    # --- Subexponential Median Trajectory ---
    lines.append(f"\n{'-'*60}")
    lines.append(f"SUBEXPONENTIAL MEDIAN TRAJECTORY")
    lines.append(f"{'-'*60}")

    # Create subexponential samples by copying and modifying
    sub_samples = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in median_samples.items()}
    sub_samples["is_subexponential"] = np.array([True])
    sub_samples["is_superexponential"] = np.array([False])
    sub_samples["is_exponential"] = np.array([False])

    lines.append(f"Subexponential Parameters:")
    lines.append(f"  sub_doubling_growth_fraction: {sub_samples['sub_doubling_growth_fraction']:.3f}")

    # Run subexponential simulation
    sub_ending_times, sub_trajectories, _, sub_prog_multiplier_trajectories = calculate_sc_arrival_year_with_trajectories(
        sub_samples, current_horizon, dt,
        human_alg_progress_decrease_date, max_simulation_years,
        forecaster_config, config["simulation"]
    )

    sub_arrival_year = sub_ending_times[0]
    sub_trajectory = sub_trajectories[0]
    sub_prog_multiplier_trajectory = sub_prog_multiplier_trajectories[0] if sub_prog_multiplier_trajectories else []

    lines.append(f"\nSubexponential SC Arrival: {format_year_month(sub_arrival_year)}")

    # Find HCP for subexponential trajectory
    sub_hcp_announced = None
    if sub_trajectory:
        for t, h in sub_trajectory:
            if h >= h_SC_threshold_minutes:
                sub_hcp_announced = t
                break

    if sub_hcp_announced is not None:
        sub_hcp = sub_hcp_announced - announcement_delay_years
        lines.append(f"Human-Cost-Parity: {format_year_month(sub_hcp)}")
        sub_days_to_sc = (sub_arrival_year - sub_hcp) * 365.25
        lines.append(f"Days from Human-Cost-Parity to SC: {sub_days_to_sc:.1f} days")

    lines.append(f"\nTrajectory (selected points):")
    lines.append(f"  {'Year':<12} {'Horizon':<20}")
    lines.append(f"  {'-'*12} {'-'*20}")

    if sub_trajectory:
        n_points = len(sub_trajectory)
        step = max(1, n_points // 10)
        for i in range(0, n_points, step):
            t, h = sub_trajectory[i]
            lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20}")

        if n_points > 1:
            t, h = sub_trajectory[-1]
            lines.append(f"  {format_year_month(t):<12} {format_horizon(h):<20} (final)")

    lines.append(f"{'='*60}\n")

    result = "\n".join(lines)
    print(result)

    # Build structured data for CSV export
    structured_data = {
        "forecaster_name": forecaster_name,
        # Median parameters
        "h_SC_work_months": median_samples['h_SC'][0],
        "horizon_doubling_time_months": median_samples['horizon_doubling_time'][0],
        "cost_speed_months": median_samples['cost_speed'][0],
        "announcement_delay_months": median_samples['announcement_delay'][0],
        "present_prog_multiplier": median_samples['present_prog_multiplier'][0],
        "SC_prog_multiplier": median_samples['SC_prog_multiplier'][0],
        "initial_software_progress_share": median_samples['initial_software_progress_share'][0],
        "growth_type": "superexponential" if median_samples["is_superexponential"][0] else "exponential",
        "superexponential_start_horizon": median_samples['superexponential_start_horizon'][0] if median_samples["is_superexponential"][0] else None,
        "se_doubling_decay_fraction": median_samples['se_doubling_decay_fraction'][0] if median_samples["is_superexponential"][0] else None,
        # Main trajectory results
        "sc_arrival_year": arrival_year,
        "human_cost_parity_year": human_cost_parity_year if human_cost_parity_year_announced is not None else None,
        "days_hcp_to_sc": days_to_sc if human_cost_parity_year_announced is not None else None,
        # Subexponential trajectory results
        "sub_sc_arrival_year": sub_arrival_year,
        "sub_human_cost_parity_year": sub_hcp if sub_hcp_announced is not None else None,
        "sub_days_hcp_to_sc": sub_days_to_sc if sub_hcp_announced is not None else None,
        # Full trajectories for detailed CSV
        "trajectory": trajectory,
        "sub_trajectory": sub_trajectory,
        # Progress multiplier trajectories
        "prog_multiplier_trajectory": prog_multiplier_trajectory,
        "sub_prog_multiplier_trajectory": sub_prog_multiplier_trajectory,
    }

    return result, structured_data


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
    # Store progress multiplier trajectories for each simulation
    prog_multiplier_trajectories = []

    # Get current date as decimal year (forecaster can override start_year)
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = forecaster_config.get("start_year", simulation_config["start_year"])
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
        # Initialize progress multiplier tracking list
        prog_multiplier_trajectory = []

        # If base time is zero (already at target horizon), record current state and continue
        if base_time_in_months[i] == 0:
            ending_times[i] = time  # No additional time required
            trajectories.append(trajectory)
            prog_multiplier_trajectories.append(prog_multiplier_trajectory)
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

            # Transform progress_fraction if using new model interpolation
            if forecaster_config.get("use_new_model_interpolation", False):
                interpolation_fraction = get_new_model_interpolation_fraction(progress_fraction)
            else:
                interpolation_fraction = progress_fraction

            # Calculate software speedup based on intermediate speedup s(interpolate between present and SC rates)
            if forecaster_config.get("automation_off", False):
                software_prog_multiplier = 1.0
            elif forecaster_config["patch_rd_speedup"]:
                software_prog_multiplier = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** interpolation_fraction
            else:
                software_prog_multiplier = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** interpolation_fraction

            # Track progress multiplier data
            prog_multiplier_trajectory.append((time + samples["announcement_delay"][i]/12, progress_fraction, software_prog_multiplier))

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
        prog_multiplier_trajectories.append(prog_multiplier_trajectory)

    # Ensure baseline_growths has one entry per simulation (edge case: zero base time simulations)
    if len(baseline_growths) < n_sims:
        # Fill missing entries with the last observed baseline growth or a small positive fallback
        fallback_growth = baseline_growths[-1] if baseline_growths else 1e-6
        missing = n_sims - len(baseline_growths)
        baseline_growths.extend([fallback_growth] * missing)

    print(f"ending_times: {ending_times}")
    return ending_times, trajectories, baseline_growths, prog_multiplier_trajectories

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

            # Transform progress_fraction if using new model interpolation
            if forecaster_config.get("use_new_model_interpolation", False):
                interpolation_fraction = get_new_model_interpolation_fraction(progress_fraction)
            else:
                interpolation_fraction = progress_fraction

            # Calculate software speedup based on intermediate speedup s(interpolate between present and SC rates)
            if forecaster_config.get("automation_off", False):
                software_prog_multiplier = 1.0
            elif forecaster_config["patch_rd_speedup"]:
                software_prog_multiplier = 1 + (samples["present_prog_multiplier"][i]) * ((samples["SC_prog_multiplier"][i])/(samples["present_prog_multiplier"][i])) ** interpolation_fraction
            else:
                software_prog_multiplier = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** interpolation_fraction
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
    import pandas as pd

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


# FORECASTER INHERITANCE FUNCTIONS

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
            # Copy non-distribution fields from parent if not present in child
            for key, value in parent_cfg.items():
                if key not in cfg and key != "distributions":
                    cfg[key] = value
            # Copy distribution fields from parent if not present in child
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


def run_simple_sc_simulation(config_path: str = "simple_params_may.yaml") -> tuple[plt.Figure, dict]:
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
    median_structured_data = []
    for forecaster_key, forecaster_config in config["forecasters"].items():
        result, structured_data = run_median_trajectory(config, forecaster_config, forecaster_config["name"])
        median_results.append(result)
        median_structured_data.append(structured_data)

    # Save median trajectory results to txt file
    with open(output_dir / "median_trajectories.txt", "w") as f:
        f.write("MEDIAN TRAJECTORY RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n".join(median_results))
    print(f"Saved median trajectories to {output_dir / 'median_trajectories.txt'}")

    # Save median trajectory summary to CSV
    csv_columns = [
        "forecaster_name", "sc_arrival_year", "diff_sc_from_prev", "sub_sc_arrival_year", "diff_sub_sc_from_prev",
        "growth_type",
        "h_SC_work_months", "horizon_doubling_time_months", "cost_speed_months",
        "announcement_delay_months", "present_prog_multiplier", "SC_prog_multiplier",
        "initial_software_progress_share", "superexponential_start_horizon", "se_doubling_decay_fraction",
        "human_cost_parity_year", "days_hcp_to_sc",
        "sub_human_cost_parity_year", "sub_days_hcp_to_sc"
    ]
    # Add diff columns by computing difference from previous row
    prev_sc = None
    prev_sub_sc = None
    for data in median_structured_data:
        curr_sc = data.get("sc_arrival_year")
        curr_sub_sc = data.get("sub_sc_arrival_year")
        data["diff_sc_from_prev"] = curr_sc - prev_sc if prev_sc is not None and curr_sc is not None else None
        data["diff_sub_sc_from_prev"] = curr_sub_sc - prev_sub_sc if prev_sub_sc is not None and curr_sub_sc is not None else None
        prev_sc = curr_sc
        prev_sub_sc = curr_sub_sc
    with open(output_dir / "median_trajectories.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        for data in median_structured_data:
            writer.writerow(data)
    print(f"Saved median trajectories summary to {output_dir / 'median_trajectories.csv'}")

    # Save detailed trajectory points to separate CSV
    with open(output_dir / "median_trajectory_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forecaster_name", "trajectory_type", "year", "horizon_minutes"])
        for data in median_structured_data:
            forecaster = data["forecaster_name"]
            # Main trajectory
            if data["trajectory"]:
                for t, h in data["trajectory"]:
                    writer.writerow([forecaster, "main", t, h])
            # Subexponential trajectory
            if data["sub_trajectory"]:
                for t, h in data["sub_trajectory"]:
                    writer.writerow([forecaster, "subexponential", t, h])
    print(f"Saved detailed trajectory points to {output_dir / 'median_trajectory_points.csv'}")

    # Save progress multiplier trajectories to separate CSV
    with open(output_dir / "median_prog_multiplier_trajectories.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["forecaster_name", "trajectory_type", "year", "progress_fraction", "software_prog_multiplier"])
        for data in median_structured_data:
            forecaster = data["forecaster_name"]
            # Main trajectory progress multipliers
            if data.get("prog_multiplier_trajectory"):
                for t, pf, pm in data["prog_multiplier_trajectory"]:
                    writer.writerow([forecaster, "main", t, pf, pm])
            # Subexponential trajectory progress multipliers
            if data.get("sub_prog_multiplier_trajectory"):
                for t, pf, pm in data["sub_prog_multiplier_trajectory"]:
                    writer.writerow([forecaster, "subexponential", t, pf, pm])
    print(f"Saved progress multiplier trajectories to {output_dir / 'median_prog_multiplier_trajectories.csv'}")

    # Run growth dynamics tracking for each forecaster's median trajectory
    print("\n" + "="*60)
    print("TRACKING GROWTH DYNAMICS FOR MEDIAN TRAJECTORIES")
    print("="*60)
    growth_dynamics_dir = output_dir / "growth_dynamics"
    growth_dynamics_dir.mkdir(exist_ok=True)

    for forecaster_key, forecaster_config in config["forecasters"].items():
        forecaster_name = forecaster_config["name"]
        print(f"Running growth dynamics for {forecaster_name}...")

        # Run growth dynamics simulation
        growth_dynamics, baseline_growth_value = run_median_trajectory_with_growth_dynamics(
            config, forecaster_config, forecaster_name
        )

        # Save growth dynamics to CSV
        csv_path = growth_dynamics_dir / f"growth_dynamics_{forecaster_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "year",
                "progress_fraction",
                "human_only_research_contribution_annualized",
                "software_prog_multiplier",
                "research_contribution_annualized",
                "research_stock",
                "baseline_growth_annualized",
                "actual_growth_annualized",
                "relative_software_progress_rate"
            ])
            writer.writeheader()
            for row in growth_dynamics:
                writer.writerow(row)
        print(f"  Saved growth dynamics CSV to {csv_path}")

        # Create and save growth dynamics plot
        plot_path = growth_dynamics_dir / f"growth_dynamics_{forecaster_name}.png"
        fig = plot_growth_dynamics(
            growth_dynamics,
            baseline_growth_value,
            forecaster_name,
            config,
            plot_path
        )
        plt.close(fig)
        print(f"  Saved growth dynamics plot to {plot_path}")

        # Check for external data file
        external_csv_path = Path(__file__).parent.parent / "external" / "inputs" / "ai_progress_results_20251211_034612.csv"
        external_csv_str = str(external_csv_path) if external_csv_path.exists() else None

        # Create and save comparison plot with external data (if available)
        if external_csv_str:
            comparison_plot_path = growth_dynamics_dir / f"growth_dynamics_comparison_{forecaster_name}.png"
            fig = plot_growth_dynamics_with_external_data(
                growth_dynamics,
                baseline_growth_value,
                forecaster_name,
                config,
                comparison_plot_path,
                external_csv_path=external_csv_str
            )
            plt.close(fig)
            print(f"  Saved growth dynamics comparison plot to {comparison_plot_path}")

        # Create and save software multiplier vs progress plot (normalized, linear scale)
        multiplier_vs_progress_path = growth_dynamics_dir / f"software_multiplier_vs_progress_{forecaster_name}.png"
        fig = plot_software_multiplier_vs_progress(
            growth_dynamics,
            forecaster_name,
            config,
            multiplier_vs_progress_path,
            external_csv_path=external_csv_str
        )
        plt.close(fig)
        print(f"  Saved software multiplier vs progress plot to {multiplier_vs_progress_path}")

        # Create and save software multiplier vs progress plot (absolute, log scale)
        multiplier_vs_progress_abs_path = growth_dynamics_dir / f"software_multiplier_vs_progress_absolute_{forecaster_name}.png"
        fig = plot_software_multiplier_vs_progress_absolute(
            growth_dynamics,
            forecaster_name,
            config,
            multiplier_vs_progress_abs_path,
            external_csv_path=external_csv_str
        )
        plt.close(fig)
        print(f"  Saved software multiplier vs progress (absolute) plot to {multiplier_vs_progress_abs_path}")

    print(f"Saved all growth dynamics files to {growth_dynamics_dir}")

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
            
            # Calculate time to SC (forecaster can override current_horizon)
            forecaster_current_horizon = forecaster_config.get("current_horizon", config["simulation"]["current_horizon"])
            all_forecaster_results[name], all_forecaster_trajectories[name], all_forecaster_baseline_growths[name], _ = calculate_sc_arrival_year_with_trajectories(
                samples,
                forecaster_current_horizon,
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
                forecaster_current_horizon,
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

    # Save config to output directory
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    # Save PDF plot
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create and save CDF plot
    fig_cdf = plot_results_cdf(all_forecaster_results, config, show_percentile_lines=False)
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

    # Collection for all central trajectory parameters across forecasters and months
    all_central_params = []

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

        fig_combined_colored, _ = plot_combined_trajectories(
            all_forecaster_backcast_trajectories,
            all_forecaster_trajectories,
            all_forecaster_samples,
            config,
            color_by_growth_type=True,
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
            "February 2029",
            "June 2028",
            "June 2030",
            "July 2030",
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

            # Save all trajectory data as CSV (all 3 central trajectories + median) and collect parameters
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

            # Collect parameters for summary CSV (from central_incl_backcast)
            if central_path is not None:
                all_central_params.append({
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
                })

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
                central_dir / f"central_trajectories_comparison_{forecaster_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig_cent_compare)

    # Save summary CSV with all central trajectory parameters
    if all_central_params:
        central_params_df = pd.DataFrame(all_central_params)
        central_params_df.to_csv(combined_dir / "central_trajectory_parameters.csv", index=False)
        print(f"Saved central trajectory parameters to {combined_dir / 'central_trajectory_parameters.csv'}")

    return fig, all_forecaster_results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "simple_params_may.yaml"
    run_simple_sc_simulation(config_path)
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 