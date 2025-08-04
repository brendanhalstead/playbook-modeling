# overall/overall_forecasting.py

import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.font_manager as fm
import sys

# --- Start of Copied/Adapted Utility Functions ---

def _load_config_impl(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _get_lognormal_from_80_ci_impl(lower_bound, upper_bound):
    # Given 80% CI (10th and 90th percentiles)
    # Convert to natural log space
    # Add small epsilon to lower_bound if it's zero or negative to avoid log errors
    epsilon = 1e-9
    ln_lower = np.log(lower_bound if lower_bound > 0 else epsilon)
    ln_upper = np.log(upper_bound if upper_bound > 0 else epsilon)

    # Ensure ln_lower < ln_upper after potential modifications
    if ln_lower >= ln_upper:
        # This might happen if upper_bound was also <=0 or very close to lower_bound
        # Fallback to a very small sigma or handle as an error
        # For now, let's make ln_upper slightly larger than ln_lower
        ln_upper = ln_lower + epsilon 

    # Z-scores for 10th and 90th percentiles
    z_low = norm.ppf(0.1)  # ≈ -1.28
    z_high = norm.ppf(0.9)  # ≈ 1.28

    # Calculate mu and sigma in log space
    sigma = (ln_upper - ln_lower) / (z_high - z_low)
    mu = (ln_upper + ln_lower) / 2
    
    # Ensure sigma is positive
    if sigma <= 0:
        sigma = epsilon # Fallback to a tiny positive sigma

    return lognorm(s=sigma, scale=np.exp(mu))

def _get_normal_from_80_ci_impl(lower_bound, upper_bound):
    # Z-scores for 10th and 90th percentiles
    z_low = norm.ppf(0.1)
    z_high = norm.ppf(0.9)

    # Calculate mu and sigma
    mu = (upper_bound + lower_bound) / 2
    sigma = (upper_bound - lower_bound) / (z_high - z_low)
    
    # Ensure sigma is positive
    if sigma <= 0:
        sigma = 1e-9 # Fallback to a tiny positive sigma

    return norm(loc=mu, scale=sigma)

# --- End of Copied/Adapted Utility Functions ---


# This is a common way to handle sibling directories in Python projects.
# The exact structure might need adjustment based on how the project is run.
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Will be removed later

# Attempt to import functions from timelines and takeoff
# We might need to copy/paste and adapt them if direct import is problematic
# due to their own internal dependencies or if they are not structured as libraries.

# From timelines/forecasting_timelines.py
# (We'll need to be careful about global variables or hardcoded paths if any)
try:
    from timelines.forecasting_timelines import (
        load_config as load_timelines_config, # Will be replaced by _load_config_impl
        get_lognormal_from_80_ci as get_lognormal_from_80_ci_tl, # Will be replaced
        get_normal_from_80_ci as get_normal_from_80_ci_tl, # Will be replaced
        get_distribution_samples as get_timelines_distribution_samples,
        calculate_gaps as calculate_timelines_gaps,
        run_single_scenario as run_timelines_single_scenario_logic, # This is the core logic part
        # setup_plotting_style # Plotting will be handled by overall
        # Other helper functions might be needed
    )
    print("Successfully imported from timelines.forecasting_timelines")
except ImportError as e:
    print(f"Error importing from timelines: {e}. Will need to copy/adapt functions.")
    # Placeholder if import fails - actual functions would be pasted here or loaded differently
    def load_timelines_config(p): return {}
    def get_timelines_distribution_samples(c, n): return {}
    def run_timelines_single_scenario_logic(s, p): return []


# From takeoff/forecasting_takeoff.py
try:
    from takeoff.forecasting_takeoff import (
        load_config as load_takeoff_config, # Will be replaced by _load_config_impl
        get_lognormal_from_80_ci as get_lognormal_from_80_ci_to, # Will be replaced
        get_milestone_samples as get_takeoff_milestone_samples,
        run_single_simulation_with_tracking as run_takeoff_single_simulation_logic, # Core logic
        # setup_plotting_style # Plotting will be handled by overall
        # Other helper functions might be needed (e.g. run_phase_simulation)
    )
    print("Successfully imported from takeoff.forecasting_takeoff")
except ImportError as e:
    print(f"Error importing from takeoff: {e}. Will need to copy/adapt functions.")
    # Placeholder
    def load_takeoff_config(p): return {}
    def get_takeoff_milestone_samples(c, n, corr): return {}
    def run_takeoff_single_simulation_logic(s, idx): return ([], [])


def year_float_to_datetime(year_float: float) -> datetime:
    """Converts a float year (e.g., 2027.5) to a datetime object."""
    year = int(year_float)
    rem = year_float - year
    base_date = datetime(year, 1, 1)
    num_days = rem * (366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365)
    return base_date + timedelta(days=num_days)

def datetime_to_year_float(dt: datetime) -> float:
    """Converts a datetime object to a float year."""
    year = dt.year

    if year == 9999:
        start_of_max_year = datetime(9999, 1, 1)
        days_passed = (dt - start_of_max_year).days
        # Check if MAXYEAR is a leap year
        is_leap = (9999 % 4 == 0 and 9999 % 100 != 0) or \
                  (9999 % 400 == 0)
        total_days_in_max_year = 366 if is_leap else 365
        return float(9999) + days_passed / total_days_in_max_year
    elif year > 9999:
        # This case should ideally not be reached if inputs are valid datetimes.
        # Represent as end of MAXYEAR or slightly beyond for plotting.
        return float(9999) + 0.9999999 

    # Standard case for years < 9999
    year_part = dt.year
    start_of_year = datetime(year_part, 1, 1)
    end_of_year = datetime(year_part + 1, 1, 1) # Safe here
    total_days_in_year = (end_of_year - start_of_year).days
    days_passed = (dt - start_of_year).days
    return year_part + days_passed / total_days_in_year


def generate_sc_arrival_dates(overall_cfg: dict, timelines_cfg_full: dict) -> list[float]:
    """
    Generates a list of SC (AGI) arrival dates (as year-floats)
    based on the timelines simulation.
    """
    print("Generating SC (AGI) arrival dates...")
    tc_params = overall_cfg['timelines_config']
    sim_params = timelines_cfg_full['simulation'] # From timelines/params.yaml
    forecaster_name = tc_params['forecaster_to_use']

    if forecaster_name not in timelines_cfg_full['forecasters']:
        raise ValueError(f"Forecaster '{forecaster_name}' not found in timelines/params.yaml")

    forecaster_specific_params = timelines_cfg_full['forecasters'][forecaster_name]
    n_sims = overall_cfg['overall_simulation']['n_sims']

    timeline_samples = get_timelines_distribution_samples(forecaster_specific_params, n_sims)
    sc_arrival_year_floats = run_timelines_single_scenario_logic(timeline_samples, sim_params)

    # Clamp SC arrival dates to be no earlier than t_0 from timelines_params
    t_0_timelines = float(sim_params['t_0'])
    
    # Handle cases where sc_arrival_year_floats might be empty or contain non-numeric data if imports fail badly
    if not sc_arrival_year_floats:
        print("Warning: sc_arrival_year_floats is empty. Cannot clamp.")
        return []

    original_min_sc_year = min(yr for yr in sc_arrival_year_floats if isinstance(yr, (int, float))) if any(isinstance(yr, (int, float)) for yr in sc_arrival_year_floats) else None

    clamped_sc_arrival_year_floats = []
    num_clamped = 0
    for year_val in sc_arrival_year_floats:
        if isinstance(year_val, (int, float)):
            if year_val < t_0_timelines:
                clamped_sc_arrival_year_floats.append(t_0_timelines)
                num_clamped += 1
            else:
                clamped_sc_arrival_year_floats.append(year_val)
        else:
            # Handle non-numeric if necessary, or let it error later / filter out
            clamped_sc_arrival_year_floats.append(year_val) # or skip/log
            print(f"Warning: Non-numeric SC arrival year found: {year_val}")

    print(f"Generated {len(sc_arrival_year_floats)} SC arrival dates (potentially clamped) using '{forecaster_name}'s parameters.")
    if num_clamped > 0 and original_min_sc_year is not None:
        print(f"Warning: Clamped {num_clamped} SC arrival dates to be no earlier than t_0 ({t_0_timelines}). Original min was {original_min_sc_year}.")
    
    return clamped_sc_arrival_year_floats


def generate_asi_arrival_from_sc(sc_arrival_date: datetime, overall_cfg: dict, takeoff_params_full: dict, sim_idx: int) -> datetime:
    """
    Generates an ASI arrival date given a single SC arrival date.
    This function will run one instance of the takeoff simulation.
    'sim_idx' is used if the takeoff sampling depends on the simulation index.
    """
    cfg_takeoff = overall_cfg['takeoff_config']
    
    # Prepare a temporary config for get_takeoff_milestone_samples
    # It needs 'starting_time', 'times', 'speedups'
    temp_takeoff_cfg_for_sampling = {
        "starting_time": sc_arrival_date.strftime("%B %d %Y"),
        "times": cfg_takeoff["times"], # From overall/params.yaml
        "speedups": cfg_takeoff["speedups"], # From overall/params.yaml
        # Potentially other params from takeoff_params_full if get_milestone_samples needs them
    }

    # Get samples for this single takeoff run.
    # The original get_milestone_samples generates n_sims worth of samples.
    # We need to adapt it or call it to get just one set of samples, or pick one.
    # For now, let's assume we generate N samples and pick the sim_idx one.
    # This might be inefficient if N is large.
    # A better approach would be to modify get_takeoff_milestone_samples to generate one sample set.
    
    n_total_sims = overall_cfg['overall_simulation']['n_sims']
    correlation = overall_cfg['overall_simulation']['takeoff_gap_correlation']

    # This generates all samples for all sims, we need to handle this carefully.
    # It might be better to call this once and then iterate through its samples.
    # For now, let's assume get_takeoff_milestone_samples can be called per SC date if it's light,
    # or we pre-generate and pass them.
    # The current structure of get_milestone_samples in takeoff_forecasting.py returns a dict
    # where each key (e.g., time_gaps) has an array of n_sims.
    # We need to pass the full takeoff_params_full (loaded from takeoff/params.yaml)
    # because get_milestone_samples might refer to parts of it not in overall_cfg['takeoff_config'] directly
    
    # To correctly use run_takeoff_single_simulation_logic(samples_dict, sim_idx),
    # the samples_dict must already contain all N simulations worth of sampled parameters.
    # So, get_takeoff_milestone_samples should be called *once* before the loop over SC dates.
    # This is a conceptual change from the current placeholder logic.
    
    # Let's adjust: The main loop will pre-generate takeoff_param_samples.
    # This function will then use one pre-generated sample.
    # This function signature needs to change to accept pre-generated takeoff_samples and sim_idx.
    
    # For now, this function's body is a placeholder for integrating with pre-sampled takeoff params.
    # The actual call to run_takeoff_single_simulation_logic will happen outside, in the main loop.
    # This function would then simplify to just converting an SC date to a start_time for a *single*
    # takeoff run using *pre-sampled* parameters for that run.

    # This placeholder will be replaced by logic that calls run_takeoff_single_simulation_logic
    # with the sc_arrival_date as the start_time and appropriate pre-sampled parameters.
    # The run_takeoff_single_simulation_logic returns (milestone_dates, phase_calendar_days)
    # milestone_dates is a list: [sar_date, siar_date, asi_date]
    
    # This function is difficult to implement correctly without modifying the imported takeoff functions
    # or copying and heavily adapting them.
    # The core issue is that `get_milestone_samples` in takeoff prepares samples for ALL sims at once,
    # and `run_single_simulation_with_tracking` picks one sample set using `sim_idx`.
    # If we call `get_milestone_samples` for each SC date, we are resampling takeoff parameters
    # for each SC date, which might not be what we want if takeoff params should be independent of SC date.
    
    # Correct approach:
    # 1. Generate N SC dates.
    # 2. Generate N sets of takeoff parameters (time gaps, etc.) using get_takeoff_milestone_samples *once*.
    # 3. For i from 0 to N-1:
    #    a. Get SC_date[i].
    #    b. Get takeoff_params_set[i].
    #    c. Run takeoff simulation with SC_date[i] as start and takeoff_params_set[i].
    #    d. Store resulting ASI_date[i].

    # This function, as is, is trying to do too much based on the current structure of imported functions.
    # It will be simplified in the main orchestrator.

    # For now, let's assume this function would be called with everything it needs for one run.
    # This is a placeholder as the main logic will be refactored.
    
    # Placeholder - to be replaced by actual call to takeoff logic
    # Example: derived_takeoff_duration_days = np.random.lognormal(mean=np.log(3*365), sigma=0.5)
    # asi_arrival_date = sc_arrival_date + timedelta(days=derived_takeoff_duration_days)
    # return asi_arrival_date
    raise NotImplementedError("Takeoff integration needs refactoring with pre-sampled parameters.")


def plot_overall_results(overall_cfg, sc_arrival_years, asi_arrival_dates, takeoff_durations_years):
    """Plots the AGI arrival distribution, takeoff duration distribution, and overall ASI distribution."""
    print("Generating plots...")

    # --- Debug prints for ranges ---
    print(f"Debug: SC Arrival Years (count: {len(sc_arrival_years)}):")
    if sc_arrival_years: # sc_arrival_years is already a list of floats
        # Filter out potential NaNs if any before min/max/median
        valid_sc_years = [yr for yr in sc_arrival_years if pd.notna(yr)]
        if valid_sc_years:
            print(f"  Min: {min(valid_sc_years)}, Max: {max(valid_sc_years)}, Median: {np.median(valid_sc_years)}")
            if len(valid_sc_years) < len(sc_arrival_years):
                 print(f"  NaN/None count in sc_arrival_years: {len(sc_arrival_years) - len(valid_sc_years)}")
        else:
            print("  No valid SC arrival years to stat.")
    else:
        print("  sc_arrival_years is empty.")
    
    print(f"Debug: Takeoff Durations (count: {len(takeoff_durations_years)}):")
    valid_takeoff_durations = [d for d in takeoff_durations_years if pd.notna(d)]
    if valid_takeoff_durations:
        print(f"  Min: {min(valid_takeoff_durations)}, Max: {max(valid_takeoff_durations)}, Median: {np.median(valid_takeoff_durations)}")
        print(f"  NaN count: {pd.isna(takeoff_durations_years).sum()}")
    else:
        print("  No valid takeoff durations.")

    print(f"Debug: ASI Arrival Datetimes (count: {len(asi_arrival_dates)}):")
    valid_asi_datetimes = [d for d in asi_arrival_dates if d is not None]
    if valid_asi_datetimes:
        min_asi_dt = min(valid_asi_datetimes)
        max_asi_dt = max(valid_asi_datetimes)
        # Calculate median ASI datetime carefully
        sorted_valid_asi_datetimes = sorted(valid_asi_datetimes)
        median_asi_dt = sorted_valid_asi_datetimes[len(sorted_valid_asi_datetimes) // 2]
        print(f"  Min DT: {min_asi_dt}, Max DT: {max_asi_dt}, Median DT: {median_asi_dt}")
        print(f"  None count: {sum(1 for d in asi_arrival_dates if d is None)}")

        # Convert to float years for ASI debug stats as well
        asi_arrival_float_years_for_debug = [datetime_to_year_float(d) for d in valid_asi_datetimes]
        if asi_arrival_float_years_for_debug:
            print(f"  Min Float Year: {min(asi_arrival_float_years_for_debug)}, Max Float Year: {max(asi_arrival_float_years_for_debug)}, Median Float Year: {np.median(asi_arrival_float_years_for_debug)}")
    else:
        print("  No valid ASI datetimes.")
    # --- End Debug prints ---

    style_cfg = overall_cfg['plotting_style']
    # setup_plotting_style(style_cfg) # Assuming a similar function can be adapted

    plt.style.use('default')
    bg_color = style_cfg['colors']['background']
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['savefig.facecolor'] = bg_color
    plt.rcParams['font.family'] = style_cfg['font']['family']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False) # Not sharing X as units differ initially
    
    # 1. SC (AGI) Arrival Dates
    ax1 = axes[0]
    ax1.hist(sc_arrival_years, bins=50, density=True, alpha=0.7, color=style_cfg['colors']['timelines_output'], label='SC (AGI) Arrival')
    ax1.set_title('Distribution of SC (AGI) Arrival Years', fontsize=style_cfg['font']['sizes']['title'])
    ax1.set_xlabel('Year', fontsize=style_cfg['font']['sizes']['axis_labels'])
    ax1.set_ylabel('Density', fontsize=style_cfg['font']['sizes']['axis_labels'])
    ax1.legend()

    # 2. Takeoff Durations (SC to ASI)
    ax2 = axes[1]
    ax2.hist(takeoff_durations_years, bins=50, density=True, alpha=0.7, color=style_cfg['colors']['takeoff_duration'], label='Takeoff Duration (SC to ASI)')
    ax2.set_title('Distribution of Takeoff Durations (Years from SC to ASI)', fontsize=style_cfg['font']['sizes']['title'])
    ax2.set_xlabel('Duration (Years)', fontsize=style_cfg['font']['sizes']['axis_labels'])
    ax2.set_ylabel('Density', fontsize=style_cfg['font']['sizes']['axis_labels'])
    ax2.legend()

    # 3. Overall ASI Arrival Dates
    asi_arrival_years = [datetime_to_year_float(d) for d in asi_arrival_dates if d is not None]
    ax3 = axes[2]
    if asi_arrival_years:
        ax3.hist(asi_arrival_years, bins=50, density=True, alpha=0.7, color=style_cfg['colors']['default'], label='Overall ASI Arrival')
        ax3.set_title('Overall Distribution of ASI Arrival Years', fontsize=style_cfg['font']['sizes']['title'])
        ax3.set_xlabel('Year', fontsize=style_cfg['font']['sizes']['axis_labels'])
        ax3.set_ylabel('Density', fontsize=style_cfg['font']['sizes']['axis_labels'])
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No ASI arrival dates generated.", ha='center', va='center')

    fig.suptitle('Overall ASI Arrival Simulation Results', fontsize=style_cfg['font']['sizes']['main_title'], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "overall_asi_arrival_simulation.png"
    fig.savefig(plot_path)
    print(f"Overall plot saved to {plot_path}")
    # plt.show() # Optional: display plot

    # Also save data
    results_df = pd.DataFrame({
        'sc_arrival_year': sc_arrival_years,
        'takeoff_duration_years': takeoff_durations_years,
        'asi_arrival_datetime': asi_arrival_dates # Store as datetime, convert to string if needed for CSV
    })
    csv_path = output_dir / "overall_simulation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Simulation results saved to {csv_path}")


def run_overall_simulation(overall_config_path: str = "params.yaml"):
    """Main function to run the chained timelines and takeoff simulation."""
    print(f"Starting overall simulation with config: {overall_config_path}")
    
    # --- 1. Load Configurations ---
    overall_cfg = yaml.safe_load(Path(overall_config_path).read_text())
    
    # Load the full timelines params.yaml (needed for forecaster details)
    # Assuming it's in ../timelines relative to this script's dir if not found directly
    timelines_params_path_str = overall_cfg['timelines_config'].get('params_file', '../timelines/params.yaml')
    timelines_params_path = Path(__file__).resolve().parent / timelines_params_path_str
    if not timelines_params_path.exists(): # Fallback if run from project root
         timelines_params_path = Path('timelines/params.yaml')
    print(f"Loading timelines parameters from: {timelines_params_path}")
    timelines_cfg_full = load_timelines_config(str(timelines_params_path))

    # Load the full takeoff params.yaml (needed for some defaults or structures if not overridden)
    takeoff_params_path_str = overall_cfg['takeoff_config'].get('params_file', '../takeoff/params.yaml')
    takeoff_params_path = Path(__file__).resolve().parent / takeoff_params_path_str
    if not takeoff_params_path.exists(): # Fallback
        takeoff_params_path = Path('takeoff/params.yaml')
    print(f"Loading takeoff parameters from: {takeoff_params_path}")
    takeoff_cfg_full = load_takeoff_config(str(takeoff_params_path)) # Assuming load_takeoff_config exists

    n_sims = overall_cfg['overall_simulation']['n_sims']

    # --- 2. Generate SC (AGI) Arrival Dates ---
    # This list will contain year-floats, e.g., [2027.5, 2028.1, ...]
    sc_arrival_year_floats = generate_sc_arrival_dates(overall_cfg, timelines_cfg_full)

    if not sc_arrival_year_floats:
        print("No SC arrival dates were generated by the timelines model. Exiting.")
        return

    # --- 3. Prepare Takeoff Parameter Samples (Once) ---
    print("Preparing takeoff parameter samples...")
    # The get_takeoff_milestone_samples function expects a config dict similar to takeoff/params.yaml
    # It needs 'starting_time' (can be a placeholder, will be overridden per sim),
    # 'times', 'speedups', and other simulation params if used internally.
    # It returns a dictionary of arrays, each of length n_sims.
    temp_takeoff_sampling_cfg = {
        "starting_time": datetime.now().strftime("%B %d %Y"), # Placeholder, not used for actual timing
        "times": overall_cfg['takeoff_config']["times"],
        "speedups": overall_cfg['takeoff_config']["speedups"],
        # Include other parameters from takeoff_cfg_full['simulation'] if needed by get_milestone_samples
        "simulation": takeoff_cfg_full.get("simulation", {"n_sims": n_sims, "dt": 1})
    }
    # Ensure n_sims in temp_takeoff_sampling_cfg matches overall_cfg
    temp_takeoff_sampling_cfg["simulation"]["n_sims"] = n_sims 
    
    takeoff_gap_correlation = overall_cfg['overall_simulation'].get('takeoff_gap_correlation', 0.7)
    # `get_takeoff_milestone_samples` needs to be callable with n_sims and correlation
    # Its signature in takeoff.py is: get_milestone_samples(config: dict, n_sims: int, correlation: float = 0.7)
    # The config it takes is the one loaded from its params.yaml. We need to ensure our temp_takeoff_sampling_cfg is compatible.
    all_takeoff_parameter_samples = get_takeoff_milestone_samples(temp_takeoff_sampling_cfg, n_sims, takeoff_gap_correlation)
    print(f"Generated {n_sims} sets of takeoff parameter samples.")


    # --- 4. Run Chained Simulation ---
    print("Running chained simulation for ASI arrival...")
    all_asi_arrival_dates = []
    all_takeoff_durations_years = []

    for i in tqdm(range(n_sims), desc="Overall Simulation Progress"):
        if i >= len(sc_arrival_year_floats):
            print(f"Warning: Not enough SC arrival dates ({len(sc_arrival_year_floats)}) for {n_sims} simulations. Using last available SC date.")
            current_sc_year_float = sc_arrival_year_floats[-1]
        else:
            current_sc_year_float = sc_arrival_year_floats[i]
        
        current_sc_datetime = year_float_to_datetime(current_sc_year_float)

        # Prepare the 'samples' dict for this specific simulation run (sim_idx = i)
        # This dict needs to mirror what run_takeoff_single_simulation_logic expects.
        # It expects a 'start_time' and then indexed access to pre-sampled parameters.
        single_run_takeoff_samples = {"start_time": current_sc_datetime}
        for key, value_array in all_takeoff_parameter_samples.items():
            if key == "start_time": continue # Already set
            if isinstance(value_array, dict): # e.g., time_gaps, speeds are dicts of arrays
                single_run_takeoff_samples[key] = {}
                for sub_key, sub_value_array in value_array.items():
                    single_run_takeoff_samples[key][sub_key] = sub_value_array # This should be sub_value_array[i]
            elif isinstance(value_array, np.ndarray) or isinstance(value_array, list):
                 # This part needs to be careful: get_milestone_samples returns dicts like:
                 # samples['time_gaps'][milestone_pair] = values * 365 (array of size n_sims)
                 # samples['speeds'][milestone] = speed (scalar or array)
                 # We need to ensure that run_takeoff_single_simulation_logic can pick the i-th sample correctly.
                 # The original takeoff code passes the *entire* `all_samples` dict and an `sim_idx`.
                 # So we should pass `all_takeoff_parameter_samples` directly, plus our `current_sc_datetime`
                 # which needs to be patched into `all_takeoff_parameter_samples['start_time']` for the i-th run,
                 # or `run_takeoff_single_simulation_logic` needs to accept `start_time` separately.

                 # Let's assume run_takeoff_single_simulation_logic will use all_takeoff_parameter_samples[key][i]
                 # and its internal current_date will be samples["start_time"]
                 pass # Parameters are already in all_takeoff_parameter_samples structured for indexing by sim_idx

        # Patch the start_time for the current simulation run into the samples structure expected by takeoff
        # The original run_single_simulation_with_tracking uses samples["start_time"]
        # which is set once in get_milestone_samples. We need to override it.
        # A cleaner way would be to modify run_takeoff_single_simulation_logic to accept start_time.
        # For now, we modify a copy or ensure it's used correctly.
        # Let's create a shallow copy of the samples for this run and set the start time.
        
        current_run_samples = {k: v for k, v in all_takeoff_parameter_samples.items()}
        current_run_samples["start_time"] = current_sc_datetime
        
        # milestone_dates_for_run is a list: [sar_dt, siar_dt, asi_dt, ...]
        milestone_dates_for_run, phase_durations_for_run = run_takeoff_single_simulation_logic(current_run_samples, i)
        
        if milestone_dates_for_run and len(milestone_dates_for_run) >= 3: # SC, SAR, SIAR, ASI (index 2 for ASI if SC is implicit start)
            # Assuming ASI is the 3rd date returned after SAR and SIAR (if SC is the implicit start)
            # The takeoff script considers milestones ["SC", "SAR", "SIAR", "ASI"].
            # run_single_simulation returns dates for SAR, SIAR, ASI. So ASI is index 2.
            asi_date = milestone_dates_for_run[2] # ASI is the 3rd milestone *date* in the list
            all_asi_arrival_dates.append(asi_date)
            
            takeoff_duration_days = (asi_date - current_sc_datetime).total_seconds() / (24 * 60 * 60)
            all_takeoff_durations_years.append(takeoff_duration_days / 365.25)
        else:
            all_asi_arrival_dates.append(None) # Handle cases where simulation might not reach ASI
            all_takeoff_durations_years.append(np.nan)

    # --- 5. Plot and Save Results ---
    plot_overall_results(overall_cfg, sc_arrival_year_floats, all_asi_arrival_dates, all_takeoff_durations_years)

    print("Overall simulation completed.")


if __name__ == "__main__":
    # Create a dummy figures directory if it doesn't exist, for local testing
    Path("figures").mkdir(exist_ok=True)
    
    # For the imports to work when running this script directly from `overall/`
    # and assuming `timelines` and `takeoff` are siblings:
    if not any('timelines' in p for p in sys.path if 'timelines' in str(p)): # crude check
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        print(f"Adjusted sys.path to include project root: {project_root}")

    # Due to the complexity of direct imports and potential for copied code,
    # this script will likely require careful adaptation of functions from
    # the timelines and takeoff scripts. The placeholders for imported functions
    # (like get_timelines_distribution_samples, run_timelines_single_scenario_logic, etc.)
    # would need to be replaced with their actual (potentially modified) implementations.

    # For a first run, one might copy the necessary functions directly into this file
    # or ensure the Python path is set up perfectly for relative imports across sibling folders.
    
    # Before running, ensure:
    # 1. `overall/params.yaml` is configured.
    # 2. The functions imported (or intended to be imported) from `timelines` and `takeoff`
    #    are accessible and their dependencies are met. This might mean copying
    #    and adapting them directly into this file initially.
    #    For example, helper functions like `get_lognormal_from_80_ci` are defined in both
    #    original scripts; a common version should be used or renamed.

    print("INFO: This script is a skeleton. Direct execution might fail if imports are not resolved or functions are not adapted.")
    print("INFO: The core logic for 'generate_sc_arrival_dates' and the takeoff loop needs robust implementation of imported/copied functions.")

    # Example of how to run:
    # Ensure you are in the `overall` directory or adjust paths in `overall/params.yaml`
    # `python overall_forecasting.py`
    
    # Given the import complexities, a safer first step is to manually copy
    # the required functions from the other two scripts into this one,
    # renaming them to avoid conflicts and adapting their internal calls.
    # For example, `load_config` from `timelines` could become `load_timelines_config_from_path`.

    run_overall_simulation() 