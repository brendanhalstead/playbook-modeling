import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from forecasting_takeoff_plotting import *
from takeoff_compute_helpers import *

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_research_trajectory_data(trajectory_path: str) -> dict:
    """Load research trajectory data from JSON file."""
    with open(trajectory_path, 'r') as f:
        return json.load(f)

def get_lognormal_from_80_ci(lower_bound, upper_bound):
    """Get lognormal distribution parameters from 80% confidence interval."""
    # Convert to natural log space
    ln_lower = np.log(lower_bound)
    ln_upper = np.log(upper_bound)

    # Z-scores for 10th and 90th percentiles
    z_low = norm.ppf(0.1)  # ≈ -1.28
    z_high = norm.ppf(0.9)  # ≈ 1.28

    # Calculate mu and sigma in log space
    sigma = (ln_upper - ln_lower) / (z_high - z_low)
    mu = (ln_upper + ln_lower) / 2

    # Return the lognormal distribution
    return lognorm(s=sigma, scale=np.exp(mu))

def lookup_research_stock_by_date(trajectory_data: dict, target_date: datetime, trajectory_name: str = "Eli") -> float:
    """Look up median research stock value for a given date in the trajectory data."""
    trajectory = trajectory_data[trajectory_name]["trajectory"]
    
    # Convert target date to decimal year
    target_year = target_date.year + target_date.timetuple().tm_yday / 365.25
    
    # Find the closest time point in the trajectory
    times = [point["time"] for point in trajectory]
    closest_idx = np.argmin([abs(t - target_year) for t in times])
    
    return trajectory[closest_idx]["median_research_stock"]

def get_project_samples_with_correlations(config: dict, n_sims: int, trajectory_data: dict, sc_speedup_samples: np.ndarray | None = None) -> dict:
    """Generate correlated samples for project parameters."""
    project_samples = {}
    projects = list(config["projects"].keys())
    
    # Extract config values for present day to SC info
    present_day_to_sc_info = config["present_day_to_sc_info"]
    research_stock_at_present_day = present_day_to_sc_info["research_stock_at_present_day"]
    software_research_stock_at_sc = present_day_to_sc_info["software_research_stock_at_sc"]
    software_plus_hardware_equivalent_research_stock_at_sc = present_day_to_sc_info["software_plus_hardware_equivalent_research_stock_at_sc"]
    
    # Calculate reference doublings
    all_doublings = np.log2(software_plus_hardware_equivalent_research_stock_at_sc / research_stock_at_present_day)
    software_only_doublings = np.log2(software_research_stock_at_sc / research_stock_at_present_day)
    
    # Parse starting time
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    
    # Separate leading and non-leading projects
    leading_projects = [p for p in projects if p == "Leading Project"]
    non_leading_projects = [p for p in projects if p != "Leading Project"]
    
    # For non-leading projects, generate correlated software progress rates
    software_progress_correlation = 0.5  # Moderate correlation between non-leading projects
    hardware_software_correlation = 0.8  # High correlation within each project
    if len(non_leading_projects) > 1:
        # Create correlation matrix for software progress rates of non-leading projects
        n_non_leading = len(non_leading_projects)
        progress_corr_matrix = np.full((n_non_leading, n_non_leading), software_progress_correlation)
        np.fill_diagonal(progress_corr_matrix, 1.0)
        
        # Generate correlated normal samples
        correlated_normal = np.random.multivariate_normal(np.zeros(n_non_leading), progress_corr_matrix, n_sims)
        correlated_uniform_progress = norm.cdf(correlated_normal)
    else:
        correlated_uniform_progress = np.random.random((n_sims, len(non_leading_projects)))
    
    for _, project_name in enumerate(projects):
        params = config["projects"][project_name]
        
        # Sample the three main parameters with correlations
        # For the Leading Project, use fixed values
        if project_name == "Leading Project":
            # Use fixed values for leading project
            initial_software_months_behind = np.zeros(n_sims)  # Leading project is not behind
            initial_hardware_multiple = np.ones(n_sims)  # Leading project has 1x hardware
            software_progress_rate = np.ones(n_sims)  # Leading project has 1x progress rate
        else:
            # Sample correlated parameters for non-leading projects
            non_leading_idx = non_leading_projects.index(project_name)
            
            # Sample initial_software_months_behind (independent)
            months_behind_params = params["initial_software_months_behind"]
            p_zero, lower, upper, ceiling = months_behind_params
            
            # Generate samples
            is_nonzero = np.random.random(n_sims) >= p_zero
            if upper > lower:
                dist = get_lognormal_from_80_ci(lower, upper)
                samples = dist.rvs(n_sims)
                samples = np.minimum(samples, ceiling)
            else:
                samples = np.full(n_sims, lower)
            samples[~is_nonzero] = 0
            initial_software_months_behind = samples
            
            # Sample correlated hardware multiple and software progress rate
            corr_matrix = np.array([[1.0, hardware_software_correlation],
                                   [hardware_software_correlation, 1.0]])
            
            # Generate correlated uniform samples for this project's hardware and software
            normal_samples = np.random.multivariate_normal([0, 0], corr_matrix, n_sims)
            uniform_samples = norm.cdf(normal_samples)
            
            # Convert to hardware multiple
            hardware_params = params["initial_hardware_multiple"]
            p_zero_hw, lower_hw, upper_hw, ceiling_hw = hardware_params
            is_nonzero_hw = np.random.random(n_sims) >= p_zero_hw
            if upper_hw > lower_hw:
                hw_dist = get_lognormal_from_80_ci(lower_hw, upper_hw)
                hw_samples = hw_dist.ppf(uniform_samples[:, 0])
                hw_samples = np.minimum(hw_samples, ceiling_hw)
            else:
                hw_samples = np.full(n_sims, lower_hw)
            hw_samples[~is_nonzero_hw] = 0
            initial_hardware_multiple = hw_samples
            
            # Convert to software progress rate using the correlated samples across non-leading projects
            software_params = params["software_progress_rate"]
            p_zero_sw, lower_sw, upper_sw, ceiling_sw = software_params
            is_nonzero_sw = np.random.random(n_sims) >= p_zero_sw
            if upper_sw > lower_sw:
                sw_dist = get_lognormal_from_80_ci(lower_sw, upper_sw)
                # Use the cross-project correlated uniforms for software progress rate
                sw_samples = sw_dist.ppf(correlated_uniform_progress[:, non_leading_idx])
                sw_samples = np.minimum(sw_samples, ceiling_sw)
            else:
                sw_samples = np.full(n_sims, lower_sw)
            sw_samples[~is_nonzero_sw] = 0
            software_progress_rate = sw_samples
        
        # Calculate target SC research stock based on hardware multiple
        software_targets = np.zeros(n_sims)
        current_software_stocks = np.zeros(n_sims)
        remaining_human_only_years = np.zeros(n_sims)
        initial_speedups = np.zeros(n_sims)
        
        for i in range(n_sims):
            # (a) Calculate adjusted software target based on hardware multiple
            assert initial_hardware_multiple[i] > 0 # idk, the AI was concerned about this
            adjustment_factor = all_doublings - (all_doublings - software_only_doublings) * (np.log((4.5**2)*initial_hardware_multiple[i]) / np.log(4.5**2))
            software_targets[i] = research_stock_at_present_day * (2 ** adjustment_factor)
            
            # (b) Calculate current software stock based on months behind
            months_behind = initial_software_months_behind[i]
            # Use simple timedelta for date arithmetic
            lookup_date = start_date - timedelta(days=int(months_behind * 30.44))  # Average days per month
            current_research_stock = lookup_research_stock_by_date(trajectory_data, lookup_date)
            current_software_stocks[i] = current_research_stock / 9
            
            # (c) Calculate remaining human-only years and initial speedup
            remaining_human_only_years[i] = max(0, software_targets[i] - current_software_stocks[i])
            
            # Calculate initial speedup based on progress percentage
            if software_targets[i] > research_stock_at_present_day:
                total_doublings_needed = np.log2(software_targets[i] / research_stock_at_present_day)
                doublings_achieved = max(0, np.log2(max(current_software_stocks[i] / research_stock_at_present_day, 1)))
                progress_ratio = min(doublings_achieved / total_doublings_needed, 1.0) if total_doublings_needed > 0 else 0
                
                # Exponential interpolation between PRESENT_DAY speed (1.07) and SC speed
                present_day_speed = config["speedups"]["PRESENT_DAY"]
                
                sc_speed = sc_speedup_samples[i]

                initial_speedups[i] = present_day_speed * (sc_speed / present_day_speed) ** progress_ratio
            else:
                initial_speedups[i] = sc_speedup_samples[i] # Already at SC
        
        schedule_params = params["schedule_params"]
        
        # Sample scientist_talent_ratio for each simulation. If a list/tuple is provided we
        # treat it as an 80 % CI for a log-normal distribution (optionally with the
        # same [p_zero, lower, upper, ceiling] format used elsewhere). Otherwise we
        # fall back to a fixed scalar.
        st_ratio_cfg = schedule_params.get("scientist_talent_ratio", None)
        if isinstance(st_ratio_cfg, (list, tuple)):
            if len(st_ratio_cfg) == 2:
                lower_st, upper_st = st_ratio_cfg
                dist_st = get_lognormal_from_80_ci(lower_st, upper_st)
                scientist_talent_ratio_samples = dist_st.rvs(n_sims)
            elif len(st_ratio_cfg) == 4:
                p_zero_st, lower_st, upper_st, ceiling_st = st_ratio_cfg
                is_nonzero_st = np.random.random(n_sims) >= p_zero_st
                dist_st = get_lognormal_from_80_ci(lower_st, upper_st)
                scientist_talent_ratio_samples = dist_st.rvs(n_sims)
                scientist_talent_ratio_samples = np.minimum(scientist_talent_ratio_samples, ceiling_st)
                scientist_talent_ratio_samples[~is_nonzero_st] = 0
            else:
                raise ValueError("scientist_talent_ratio distribution must have 2 or 4 elements")
        else:
            # Treat missing or scalar value as fixed across simulations
            scientist_talent_ratio_samples = np.full(n_sims, st_ratio_cfg if st_ratio_cfg is not None else 1.0)
        
        # ------------------------------------------------------------------
        # Sample experiment_H100e and agent_H100be if provided as ranges
        # ------------------------------------------------------------------

        def _sample_lognorm_range(cfg_value):
            """Utility to sample log-normal given 2- or 4-element config."""
            if isinstance(cfg_value, (list, tuple)):
                if len(cfg_value) == 2:
                    lower, upper = cfg_value
                    dist = get_lognormal_from_80_ci(lower, upper)
                    return dist.rvs(n_sims)
                elif len(cfg_value) == 4:
                    p_zero, lower, upper, ceiling = cfg_value
                    is_nonzero = np.random.random(n_sims) >= p_zero
                    dist = get_lognormal_from_80_ci(lower, upper)
                    samples = dist.rvs(n_sims)
                    samples = np.minimum(samples, ceiling)
                    samples[~is_nonzero] = 0
                    return samples
                else:
                    raise ValueError("Distribution list must have 2 or 4 elements")
            else:
                return np.full(n_sims, cfg_value)

        # ------------------------------------------------------------------
        # Generic sampler that falls back to normal distribution for values
        # that are not strictly positive (since log-normal cannot handle 0/neg)
        # ------------------------------------------------------------------
        def _sample_generic_range(cfg_value):
            """Sample from cfg_value returning an array of length n_sims.

            Supports the same 2- or 4-element formats as _sample_lognorm_range.
            Uses log-normal when the bounds are > 0, otherwise a normal
            distribution where the 10th/90th percentiles match the bounds.
            Point estimates (scalars) are broadcast to length n_sims.
            """
            if isinstance(cfg_value, (list, tuple)):
                if len(cfg_value) == 2:
                    lower, upper = cfg_value
                    if lower > 0 and upper > 0:
                        return _sample_lognorm_range(cfg_value)
                    # Derive normal parameters from 10th/90th percentiles
                    z_low, z_high = norm.ppf(0.1), norm.ppf(0.9)
                    sigma = (upper - lower) / (z_high - z_low)
                    mu = (upper + lower) / 2.0
                    return np.random.normal(mu, sigma, n_sims)
                elif len(cfg_value) == 4:
                    p_zero, lower, upper, ceiling = cfg_value
                    if lower > 0 and upper > 0:
                        samples = _sample_lognorm_range(cfg_value)
                    else:
                        z_low, z_high = norm.ppf(0.1), norm.ppf(0.9)
                        sigma = (upper - lower) / (z_high - z_low)
                        mu = (upper + lower) / 2.0
                        samples = np.random.normal(mu, sigma, n_sims)
                        samples = np.minimum(samples, ceiling)
                    is_nonzero = np.random.random(n_sims) >= p_zero
                    samples[~is_nonzero] = 0
                    return samples
                else:
                    raise ValueError("Distribution list must have 2 or 4 elements")
            else:
                return np.full(n_sims, cfg_value)

        experiment_H100e_samples = _sample_lognorm_range(schedule_params.get("experiment_H100e", 0))
        agent_H100be_samples = _sample_lognorm_range(schedule_params.get("agent_H100be", 0))

        # ------------------------------------------------------------------
        # Sample parameters nested under SC_CES_params (if present)
        # ------------------------------------------------------------------
        sc_ces_cfg = schedule_params.get("SC_CES_params")
        sc_ces_param_samples = {}
        if sc_ces_cfg is not None:
            for ces_key, ces_cfg_val in sc_ces_cfg.items():
                sc_ces_param_samples[ces_key] = _sample_generic_range(ces_cfg_val)
            # Note: we DO NOT overwrite schedule_params["SC_CES_params"] here.
            # We keep the original config (which may be point estimates or ranges)
            # and instead store the sampled arrays separately in project_samples.

        schedule_params["start_date"] = start_date

        project_samples[project_name] = {
            "initial_software_months_behind": initial_software_months_behind,
            "initial_hardware_multiple": initial_hardware_multiple,
            "software_progress_rate": software_progress_rate,
            "schedule_params": schedule_params,
            "software_target": software_targets,
            "current_software_stock": current_software_stocks,
            # "remaining_human_only_years": remaining_human_only_years,
            "remaining_human_only_years": np.zeros(n_sims),
            "initial_speedup": initial_speedups,
            "scientist_talent_ratio": scientist_talent_ratio_samples,
            "experiment_H100e": experiment_H100e_samples,
            "agent_H100be": agent_H100be_samples,
            "SC_CES_params": sc_ces_param_samples,
            # new fields for hardware decay
            "no_resupply": params.get("no_resupply", False),
            "failure_model_params": (
                config.get("failure_models", {}).get(params.get("failure_model"))
                if params.get("failure_model") else None
            )
        }
    
    return project_samples

def get_milestone_samples(config: dict, n_sims: int, correlation: float = 0.7) -> dict:
    """
    Generate samples for milestone timings with correlation between gap sizes.
    Currently does NOT sample for speedups, just returns the fixed values defined in the config.
    """
    samples = {}
    
    # Parse starting time
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    samples["start_time"] = start_date
    
    # Get list of time gaps to model
    milestone_pairs = list(config["times"].keys())
    n_vars = len(milestone_pairs)
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Generate AMR to SAR samples with correlation to SC to SAR
    amr_sar_sc_sar_correlation = 0.8
    corr_matrix_2 = np.array([[1.0, amr_sar_sc_sar_correlation],
                             [amr_sar_sc_sar_correlation, 1.0]])
    mean_2 = np.zeros(2)
    correlated_normal_samples = np.random.multivariate_normal(mean_2, corr_matrix_2, size=n_sims)
    uniform_samples_corr = norm.cdf(correlated_normal_samples)
    
    # Generate AMR to SAR samples using the correlated uniforms
    amr_to_sar_dist = get_lognormal_from_80_ci(1, 25)
    amr_to_sar_samples = amr_to_sar_dist.ppf(uniform_samples_corr[:, 0])
    
    # Generate SAR to SIAR equivalent jumps samples
    sar_to_siar_equiv_jumps_dist = get_lognormal_from_80_ci(0.3, 7.5)
    sar_to_siar_equiv_jumps_samples = sar_to_siar_equiv_jumps_dist.ppf(np.random.random(n_sims))
    
    # Generate time gap samples
    samples["time_gaps"] = {}
    
    for idx, milestone_pair in enumerate(milestone_pairs):
        params = config["times"][milestone_pair]
        p_zero, lower, upper = params
        
        if milestone_pair == "SAR to SIAR":
            # Calculate SAR to SIAR time using the formula
            values = np.zeros(n_sims)
            for i in range(n_sims):
                x = 10 + amr_to_sar_samples[i]
                values[i] = (x * 2**(2*np.log(x/10)/np.log(2)))-x
            
            # Print distribution statistics for SAR to SIAR
            print("\nSAR to SIAR human-only time distribution (years):")
            print(f"10th percentile: {np.percentile(values, 10):.2f}")
            print(f"25th percentile: {np.percentile(values, 25):.2f}")
            print(f"50th percentile (median): {np.percentile(values, 50):.2f}")
            print(f"75th percentile: {np.percentile(values, 75):.2f}")
            print(f"90th percentile: {np.percentile(values, 90):.2f}")
            print(f"Min: {np.min(values):.2f}")
            print(f"Max: {np.max(values):.2f}")
        
        elif milestone_pair == "SIAR to ASI":
            # Calculate SIAR to ASI time using the new formula
            values = np.zeros(n_sims)
            for i in range(n_sims):
                stock_through_sar = 10 + amr_to_sar_samples[i]
                sar_to_siar_years = samples["time_gaps"]["SAR to SIAR"][i] / 365  # Convert back to years
                sar_to_siar_equiv_jumps = sar_to_siar_equiv_jumps_samples[i]
                
                x = stock_through_sar + sar_to_siar_years
                values[i] = (x * 2**(sar_to_siar_equiv_jumps*np.log(x/stock_through_sar)/np.log(2)))-x
            
            # Print distribution statistics for SIAR to ASI
            print("\nSIAR to ASI human-only time distribution (years):")
            print(f"10th percentile: {np.percentile(values, 10):.2f}")
            print(f"25th percentile: {np.percentile(values, 25):.2f}")
            print(f"50th percentile (median): {np.percentile(values, 50):.2f}")
            print(f"75th percentile: {np.percentile(values, 75):.2f}")
            print(f"90th percentile: {np.percentile(values, 90):.2f}")
            print(f"Min: {np.min(values):.2f}")
            print(f"Max: {np.max(values):.2f}")
        
        elif milestone_pair == "SC to SAR":
            # Generate lognormal samples using the correlated uniform samples
            dist = get_lognormal_from_80_ci(lower, upper)
            values = dist.ppf(uniform_samples_corr[:, 1])
            is_nonzero = np.random.random(n_sims) >= p_zero
            values[~is_nonzero] = 0
            
        elif milestone_pair == "PRESENT_DAY to SC":
            # Fixed 5-year gap (the baseline journey to SC)
            values = np.full(n_sims, 5.0)  # Always 5 years from present day to SC
            
        else:
            # Standard handling for other transitions
            is_nonzero = np.random.random(n_sims) >= p_zero
            dist = get_lognormal_from_80_ci(lower, upper)
            values = dist.ppf(uniform_samples[:, idx])
            values[~is_nonzero] = 0
        
        # Convert years to days
        samples["time_gaps"][milestone_pair] = values * 365
    
    # Store speedup values
    samples["speeds"] = {}
    
    # Handle fixed PRESENT_DAY speedup
    if "speedups" in config and "PRESENT_DAY" in config["speedups"]:
        samples["speeds"]["PRESENT_DAY"] = config["speedups"]["PRESENT_DAY"]

    # Sample correlated speedups for other milestones
    if "speedup_distributions" in config and "speedup_correlation_matrix" in config:
        speedup_dists_config = config["speedup_distributions"]
        speedup_corr_matrix = np.array(config["speedup_correlation_matrix"])
        
        milestone_ratios = list(speedup_dists_config.keys())
        n_speedup_vars = len(milestone_ratios)
        
        # Generate correlated uniform samples
        mean_speedups = np.zeros(n_speedup_vars)
        correlated_normal_speedups = np.random.multivariate_normal(mean_speedups, speedup_corr_matrix, size=n_sims)
        uniform_samples_speedups = norm.cdf(correlated_normal_speedups)
        
        # Create distributions and sample from them
        dists = {name: get_lognormal_from_80_ci(*speedup_dists_config[name]) for name in milestone_ratios}
        
        sc_samples = dists["SC"].ppf(uniform_samples_speedups[:, 0])
        sar_ratio_samples = dists["SAR_ratio"].ppf(uniform_samples_speedups[:, 1])
        siar_ratio_samples = dists["SIAR_ratio"].ppf(uniform_samples_speedups[:, 2])
        asi_ratio_samples = dists["ASI_ratio"].ppf(uniform_samples_speedups[:, 3])
        
        # Enforce non-decreasing speedups by ensuring ratios are >= 1, then calculate absolute values
        samples["speeds"]["SC"] = sc_samples
        samples["speeds"]["SAR"] = samples["speeds"]["SC"] * np.maximum(1, sar_ratio_samples)
        samples["speeds"]["SIAR"] = samples["speeds"]["SAR"] * np.maximum(1, siar_ratio_samples)
        samples["speeds"]["ASI"] = samples["speeds"]["SIAR"] * np.maximum(1, asi_ratio_samples)
    else:
        # Fallback to old fixed values if new config is not present
        for milestone, speed in config.get("speedups", {}).items():
            if milestone != "PRESENT_DAY":
                samples["speeds"][milestone] = speed

    return samples

def run_phase_simulation(gap: float, start_speed: float, end_speed: float, progress_rate_schedule_params: dict, phase_start_date: datetime, milestone_pair: str = None) -> float:
    """Run simulation for a single phase with exponential speedup.
    
    Args:
        gap: Required progress in days
        start_speed: Initial speed multiplier v for this phase
        end_speed: Final speed multiplier v for this phase
        progress_rate: Rate at which this actor makes progress (1.0 = normal, 0.3 = 30% speed)
        milestone_pair: String identifying which transition this is
        
    Returns:
        Calendar days taken to complete the phase
    """
    assert progress_rate_schedule_params is not None
    dt = 1  # One day timesteps
    elapsed_days = 0
    current_date = phase_start_date
    progress = 0
    # print(current_date)
    
    # Cap to prevent overflow
    MAX_CALENDAR_DAYS = 365 * 100
    
    while progress < gap:
        # Calculate current speedup based on progress through the phase
        progress_ratio = progress / gap
        prev_milestone = milestone_pair.split(" to ")[0]
        speed_with_prev_milestone = sw_progress_rate_schedule(current_date, progress_rate_schedule_params, prev_milestone, start_speed)
        next_milestone = milestone_pair.split(" to ")[1]
        speed_with_next_milestone = sw_progress_rate_schedule(current_date, progress_rate_schedule_params, next_milestone, end_speed)
        current_speedup = speed_with_prev_milestone * (speed_with_next_milestone/speed_with_prev_milestone)**progress_ratio
        if progress == -1:
            print("--------------------------------")
            print(f"Milestone pair: {milestone_pair}")
            sw_progress_rate_schedule(current_date, progress_rate_schedule_params, prev_milestone, start_speed, debug=True)
            sw_progress_rate_schedule(current_date, progress_rate_schedule_params, next_milestone, end_speed, debug=True)
            print(f"Speed with prev milestone: {speed_with_prev_milestone}")
            print(f"Speed with next milestone: {speed_with_next_milestone}")
            print(f"Progress ratio: {progress_ratio}")
            print(f"Progress: {progress}")
            print(f"Gap: {gap}")
            print(f"Current date: {current_date}")
        # Make progress at varying speed, adjusted by progress_rate
        assert elapsed_days < 366 * 1000

        progress += current_speedup * dt
        elapsed_days += dt
        if current_date < datetime(9999, 12, 31):
            try:
                current_date += timedelta(days=dt)
            except OverflowError:
                current_date = datetime(9999, 12, 31)
        
        if elapsed_days > MAX_CALENDAR_DAYS:
            # print(f"Warning: Phase duration capped at {MAX_CALENDAR_DAYS/365:.1f} years")
            return MAX_CALENDAR_DAYS
    
    return elapsed_days

def run_single_simulation_with_tracking(milestone_samples: dict, sim_idx: int, progress_rate_schedule_params: dict, remaining_years_to_sc: float = None, initial_speedup: float = None) -> tuple[list[datetime], list[float]]:
    """Run a single simulation and track both milestone dates and phase durations.
    
    Args:
        milestone_samples: Dictionary containing milestone samples
        sim_idx: Simulation index
        progress_rate: Rate at which this actor makes progress (1.0 = normal)
        remaining_years_to_sc: Calculated remaining human-only years to SC (new method)
        initial_speedup: Calculated initial speedup for this project (new method)
    """
    assert progress_rate_schedule_params is not None
    milestone_dates = []
    phase_calendar_days = []
    current_date = milestone_samples["start_time"]
    
    # List of milestones in order (start from present day, work toward SC)
    milestones = ["SC", "SAR", "SIAR", "ASI"] # "WS"
    
    # First, handle the journey from PRESENT_DAY to SC
    if remaining_years_to_sc is not None:
        # Use the new calculated method
        remaining_gap_to_sc = remaining_years_to_sc * 365  # Convert to days
        
        if remaining_gap_to_sc > 0:
            # Still need to complete journey to SC
            # Use the calculated initial speedup
            if initial_speedup is not None:
                present_day_speed = initial_speedup
            else:
                present_day_speed = milestone_samples["speeds"].get("PRESENT_DAY", 1.07)
            
            sc_speed = milestone_samples["speeds"]["SC"]
            if isinstance(sc_speed, np.ndarray):
                sc_speed = sc_speed[sim_idx]
            
            calendar_days = run_phase_simulation(remaining_gap_to_sc, present_day_speed, sc_speed, progress_rate_schedule_params, current_date, "PRESENT_DAY to SC")
            phase_calendar_days.append(calendar_days)
            
            try:
                current_date = current_date + timedelta(days=calendar_days)
                if current_date.year > 9999:
                    current_date = datetime(9999, 12, 31)
            except OverflowError:
                current_date = datetime(9999, 12, 31)
            
            milestone_dates.append(current_date)  # This is when SC is reached
        else:
            # Already at SC, no time needed
            milestone_dates.append(current_date)
            phase_calendar_days.append(0)
    else:
        # No PRESENT_DAY to SC gap defined, assume already at SC
        milestone_dates.append(current_date)
        phase_calendar_days.append(0)
    
    # Run through each remaining milestone gap
    for i, milestone in enumerate(milestones[:-1]):
        next_milestone = milestones[i + 1]
        milestone_pair = f"{milestone} to {next_milestone}"
        gap = milestone_samples["time_gaps"][milestone_pair][sim_idx]
        
        # Get speedup values, handling SC speedup samples
        if isinstance(milestone_samples["speeds"][milestone], np.ndarray):
            start_speed = milestone_samples["speeds"][milestone][sim_idx]
        else:
            start_speed = milestone_samples["speeds"][milestone]
            
        if isinstance(milestone_samples["speeds"][next_milestone], np.ndarray):
            end_speed = milestone_samples["speeds"][next_milestone][sim_idx]
        else:
            end_speed = milestone_samples["speeds"][next_milestone]
        
        # Run simulation for this phase with exponential speedup
        calendar_days = run_phase_simulation(gap, start_speed, end_speed, progress_rate_schedule_params, current_date, milestone_pair)
        phase_calendar_days.append(calendar_days)
        
        try:
            current_date = current_date + timedelta(days=calendar_days)
            if current_date.year > 9999:
                current_date = datetime(9999, 12, 31)
        except OverflowError:
            current_date = datetime(9999, 12, 31)
        
        milestone_dates.append(current_date)
        
        if current_date.year == 9999:
            for _ in range(i+1, len(milestones)-1):
                milestone_dates.append(current_date)
                phase_calendar_days.append(0)
            break
    
    return milestone_dates, phase_calendar_days

# def run_single_simulation(samples: dict, sim_idx: int, progress_rate: float = 1.0) -> list[datetime]:
    """Run a single simulation and return milestone dates."""
    milestone_dates, _ = run_single_simulation_with_tracking(samples, sim_idx, progress_rate)
    return milestone_dates

def run_multi_project_simulation_with_tracking(milestone_samples: dict, sim_idx: int, project_progress_samples: dict, project_samples: dict = None) -> tuple[dict, list[datetime], dict]:
    """Run multi-project simulation with detailed tracking.
    
    Args:
        milestone_samples: Milestone timing samples
        sim_idx: Simulation index
        project_progress_samples: Dictionary mapping project names to arrays of software progress rate samples
        project_samples: Full project samples with remaining years and initial speedups (optional)
    
    Returns:
        Tuple of (project_results, first_milestone_dates, project_phase_durations)
    """
    project_results = {}
    project_phase_durations = {}
    
    # Run simulation for each project with tracking
    for project_name, progress_rate_samples in project_progress_samples.items():
        # Deep copy schedule_params to avoid cross-sim contamination
        progress_rate_schedule_params = project_samples[project_name]["schedule_params"].copy()
        assert progress_rate_schedule_params is not None
        progress_rate_schedule_params["initial_sw_progress_rate"] = progress_rate_samples[sim_idx]  # Get initial rate for this simulation

        # Pass the sampled scientist talent ratio into the schedule for this simulation
        if "scientist_talent_ratio" in project_samples[project_name]:
            progress_rate_schedule_params["scientist_talent_ratio"] = project_samples[project_name]["scientist_talent_ratio"][sim_idx]

        # Inject compute bandwidth samples if present
        if "experiment_H100e" in project_samples[project_name]:
            progress_rate_schedule_params["experiment_H100e"] = project_samples[project_name]["experiment_H100e"][sim_idx]
        if "agent_H100be" in project_samples[project_name]:
            progress_rate_schedule_params["agent_H100be"] = project_samples[project_name]["agent_H100be"][sim_idx]
        # Inject SC_CES_params per simulation if present
        if "SC_CES_params" in project_samples[project_name]:
            ces_dict = project_samples[project_name]["SC_CES_params"]
            # Replace with a fresh dict to avoid bleed-through between simulations
            progress_rate_schedule_params["SC_CES_params"] = {
                ces_key: ces_samples[sim_idx] for ces_key, ces_samples in ces_dict.items()
            }

        # TODO: Add other project-specific parameters to the schedule params

        # Get remaining years and initial speedup if available
        remaining_years_to_sc = None
        initial_speedup = None
        if project_samples and project_name in project_samples:
            remaining_years_to_sc = project_samples[project_name]["remaining_human_only_years"][sim_idx]
            initial_speedup = project_samples[project_name]["initial_speedup"][sim_idx]
        
        # Run simulation with the project-specific parameters
        milestone_dates, phase_durations = run_single_simulation_with_tracking(
            milestone_samples, sim_idx, progress_rate_schedule_params,
            remaining_years_to_sc=remaining_years_to_sc,
            initial_speedup=initial_speedup
        )
        project_results[project_name] = milestone_dates
        project_phase_durations[project_name] = phase_durations
    
    # Find first achievement of each milestone across all projects
    milestones = ["SC", "SAR", "SIAR", "ASI"]  # These correspond to indices 0, 1, 2, 3 in milestone_dates
    first_milestone_dates = []
    
    for milestone_idx in range(len(milestones)):
        earliest_date = None
        for project_name in project_progress_samples:
            if milestone_idx < len(project_results[project_name]):
                project_date = project_results[project_name][milestone_idx]
                if earliest_date is None or project_date < earliest_date:
                    earliest_date = project_date
        
        if earliest_date is not None:
            first_milestone_dates.append(earliest_date)
        else:
            first_milestone_dates.append(datetime(9999, 12, 31))
    
    return project_results, first_milestone_dates, project_phase_durations

def run_multi_project_takeoff_simulation(config_path: str = "takeoff_params.yaml") -> tuple[plt.Figure, dict]:
    """Run multi-project takeoff simulation and create visualizations."""
    print("Loading configuration...")
    config = load_config(config_path)
    plotting_style = config["plotting_style"]
    
    # Check if projects are defined in config
    if "projects" not in config:
        print("No projects defined in config. Running single-project simulation instead.")
        return run_takeoff_simulation(config_path)
    
    # Load trajectory data and generate full project samples
    print("Loading research trajectory data...")
    trajectory_data = load_research_trajectory_data("research_trajectory_data.json")
    
    # Generate milestone samples (shared across all projects), including the new speedup distributions
    print("\nGenerating milestone samples...")
    milestone_samples = get_milestone_samples(config, config["simulation"]["n_sims"])

    print("\nGenerating project samples with correlations...")
    project_samples = get_project_samples_with_correlations(config, config["simulation"]["n_sims"], trajectory_data, milestone_samples["speeds"]["SC"])
    
    # Derive project progress samples from the full project_samples for compatibility
    print("\nDeriving project progress rate samples...")
    project_progress_samples = {name: data["software_progress_rate"] for name, data in project_samples.items()}
    
    # Print statistics for project parameters
    print("Project configurations:")
    for project_name, sample_data in project_samples.items():
        progress_rates = sample_data["software_progress_rate"]
        hardware_multiples = sample_data["initial_hardware_multiple"] 
        months_behind = sample_data["initial_software_months_behind"]
        remaining_years = sample_data["remaining_human_only_years"]
        
        print(f"  {project_name}:")
        print(f"    Progress rates: 10th={np.percentile(progress_rates, 10):.2f}x, 50th={np.percentile(progress_rates, 50):.2f}x, 90th={np.percentile(progress_rates, 90):.2f}x")
        print(f"    Hardware multiples: 10th={np.percentile(hardware_multiples, 10):.3f}x, 50th={np.percentile(hardware_multiples, 50):.3f}x, 90th={np.percentile(hardware_multiples, 90):.3f}x")
        print(f"    Months behind: 10th={np.percentile(months_behind, 10):.1f}, 50th={np.percentile(months_behind, 50):.1f}, 90th={np.percentile(months_behind, 90):.1f}")
        print(f"    Remaining years to SC: 10th={np.percentile(remaining_years, 10):.2f}, 50th={np.percentile(remaining_years, 50):.2f}, 90th={np.percentile(remaining_years, 90):.2f}")
    
    # Set up fonts
    fonts = setup_plotting_style(plotting_style)
    
    # Run multi-project simulations
    print("\nRunning multi-project simulations...")
    all_first_milestone_dates = []
    all_project_results = []
    all_project_phase_durations = []
    
    for i in tqdm(range(config["simulation"]["n_sims"]), desc="Simulations"):
        project_results, first_milestone_dates, project_phase_durations = run_multi_project_simulation_with_tracking(
            milestone_samples, i, project_progress_samples, project_samples
        )
        all_first_milestone_dates.append(first_milestone_dates)
        all_project_results.append(project_results)
        all_project_phase_durations.append(project_phase_durations)
    
    # Print summary statistics
    print("\nFirst-to-achieve milestone statistics:")
    milestones = ["SC", "SAR", "SIAR", "ASI"]
    for i, milestone in enumerate(milestones):
        milestone_years = [dates[i].year + dates[i].timetuple().tm_yday/365 
                          for dates in all_first_milestone_dates 
                          if i < len(dates) and dates[i].year < 9999]
        
        if milestone_years:
            p10 = np.percentile(milestone_years, 10)
            p50 = np.percentile(milestone_years, 50)
            p90 = np.percentile(milestone_years, 90)
            print(f"{milestone}: 10th={p10:.1f}, 50th={p50:.1f}, 90th={p90:.1f}")
        else:
            print(f"{milestone}: No valid data")
    
    # Analysis: Check which project wins SAR most often
    print("\nSAR Winner Analysis:")
    sar_winners = {}
    projects = list(config["projects"].keys())
    
    ties_found = 0
    
    for sim_idx, sim_results in enumerate(all_project_results):
        earliest_sar_date = None
        winner = None
        all_sar_dates = {}  # Track all SAR dates for this simulation
        
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    all_sar_dates[project_name] = sar_date
                    if earliest_sar_date is None or sar_date < earliest_sar_date:
                        earliest_sar_date = sar_date
                        winner = project_name
        
        # Check for ties
        if earliest_sar_date is not None:
            tied_winners = [name for name, date in all_sar_dates.items() if date == earliest_sar_date]
            if len(tied_winners) > 1:
                ties_found += 1
                print(f"  TIE in simulation {sim_idx}: {tied_winners} all achieved SAR on {earliest_sar_date}")
                print(f"    Main analysis chose: {winner}")
                
                # Print detailed parameters for tied projects
                for tied_project in tied_winners:
                    if tied_project in project_progress_samples and tied_project in project_samples:
                        print(f"    {tied_project} parameters:")
                        print(f"      Progress rate: {project_progress_samples[tied_project][sim_idx]:.3f}x")
                        print(f"      Hardware multiple: {project_samples[tied_project]['initial_hardware_multiple'][sim_idx]:.3f}x")
                        print(f"      Months behind: {project_samples[tied_project]['initial_software_months_behind'][sim_idx]:.1f}")
                        print(f"      Remaining years to SC: {project_samples[tied_project]['remaining_human_only_years'][sim_idx]:.3f}")
                        print(f"      Initial speedup: {project_samples[tied_project]['initial_speedup'][sim_idx]:.3f}x")
                        
                        # Show the actual stock calculations that determine remaining years
                        software_target = project_samples[tied_project]['software_target'][sim_idx]
                        current_software_stock = project_samples[tied_project]['current_software_stock'][sim_idx]
                        print(f"      Software target: {software_target:.3f}")
                        print(f"      Current software stock: {current_software_stock:.3f}")
                        print(f"      Stock difference: {software_target - current_software_stock:.3f}")
                        
                        # Show SC achievement date too
                        if tied_project in sim_results and len(sim_results[tied_project]) > 0:
                            sc_date = sim_results[tied_project][0]
                            sc_year = sc_date.year + sc_date.timetuple().tm_yday/365
                            print(f"      SC achieved: {sc_year:.3f}")
                            sc_to_sar_days = (all_sar_dates[tied_project] - sc_date).days
                            print(f"      SC to SAR time: {sc_to_sar_days/365:.3f} years")
                
                # Also show the SC to SAR gap sample for this simulation
                sc_to_sar_gap_years = milestone_samples["time_gaps"]["SC to SAR"][sim_idx] / 365
                print(f"    SC to SAR gap sample for this simulation: {sc_to_sar_gap_years:.3f} years")
        
        if winner:
            sar_winners[winner] = sar_winners.get(winner, 0) + 1
    
    print(f"Total ties found: {ties_found}")
    
    total_wins = sum(sar_winners.values())
    print(f"Total valid simulations: {total_wins}")
    for project, wins in sorted(sar_winners.items(), key=lambda x: x[1], reverse=True):
        percentage = (wins / total_wins) * 100 if total_wins > 0 else 0
        print(f"  {project}: {wins} wins ({percentage:.1f}%)")
    
    leading_lab_always_wins = sar_winners.get("Leading Project", 0) == total_wins
    print(f"\nDoes Leading Project always win SAR? {leading_lab_always_wins}")
    if not leading_lab_always_wins:
        print("-> Other projects sometimes beat Leading Project!")
    
    # Create plots
    print("\nGenerating plots...")
    fig_multi_timeline = create_multi_project_timeline_plot(all_first_milestone_dates, all_project_results, config, plotting_style, fonts)
    fig_project_delays = create_project_delay_plot(all_project_results, config, plotting_style, fonts, project_progress_samples)
    fig_sc_timeline = create_project_sc_timeline_plot(all_project_results, config, plotting_style, fonts)
    fig_sar_timeline = create_project_sar_timeline_plot(all_project_results, config, plotting_style, fonts)
    
    # Also create single-project plots for the fastest project (for comparison)
    fastest_project = min(project_progress_samples.keys(), key=lambda x: 1/np.mean(project_progress_samples[x]))  # Highest mean progress rate
    fastest_milestone_dates = [results[fastest_project] for results in all_project_results]
    fig_fastest_timeline = create_milestone_timeline_plot(fastest_milestone_dates, config, plotting_style, fonts)
    fig_fastest_phases = create_phase_duration_plot(fastest_milestone_dates, config, plotting_style, fonts)
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plots...")
    fig_multi_timeline.savefig(output_dir / "multi_project_takeoff_timeline.png", dpi=300, bbox_inches="tight")
    fig_project_delays.savefig(output_dir / "project_sar_delays.png", dpi=300, bbox_inches="tight")
    fig_sc_timeline.savefig(output_dir / "project_sc_timeline.png", dpi=300, bbox_inches="tight")
    fig_sar_timeline.savefig(output_dir / "project_sar_timeline.png", dpi=300, bbox_inches="tight")
    fig_fastest_timeline.savefig(output_dir / f"fastest_project_{fastest_project.replace(' ', '_')}_timeline.png", dpi=300, bbox_inches="tight")
    fig_fastest_phases.savefig(output_dir / f"fastest_project_{fastest_project.replace(' ', '_')}_phases.png", dpi=300, bbox_inches="tight")
    
    # Close figures to free memory
    plt.close("all")
    
    return fig_multi_timeline, {"milestone_dates": all_first_milestone_dates}

if __name__ == "__main__":
    run_multi_project_takeoff_simulation()
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 