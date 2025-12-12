import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm, rankdata
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import json

def load_trajectories_from_jsonl(jsonl_dir: str | Path) -> tuple[dict, dict, dict, dict]:
    """
    Load trajectory data from JSONL files in a directory.

    Returns:
        Tuple of (all_forecaster_backcast_trajectories, all_forecaster_trajectories,
                  all_forecaster_samples, all_forecaster_results)
    """
    jsonl_dir = Path(jsonl_dir)

    all_forecaster_backcast_trajectories = {}
    all_forecaster_trajectories = {}
    all_forecaster_samples = {}
    all_forecaster_results = {}

    for jsonl_file in jsonl_dir.glob("*_trajectories.jsonl"):
        # Extract forecaster name from filename (e.g., "Eli_trajectories.jsonl" -> "Eli")
        forecaster_name = jsonl_file.stem.replace("_trajectories", "")

        backcast_list = []
        forecast_list = []
        results_list = []

        # Initialize sample arrays
        sample_keys = ['h_SC', 'T_t', 'cost_speed', 'announcement_delay',
                       'present_prog_multiplier', 'SC_prog_multiplier',
                       'patch_rd_speedup', 'software_progress_share',
                       'is_exponential', 'is_superexponential', 'is_subexponential',
                       'se_doubling_decay_fraction', 'sub_doubling_growth_fraction']
        samples_dict = {k: [] for k in sample_keys}

        with open(jsonl_file, 'r') as f:
            for line in f:
                record = json.loads(line)

                # Extract trajectories
                backcast_list.append(record.get('backcast_trajectory', []))
                forecast_list.append(record.get('forward_trajectory', []))
                results_list.append(record.get('sc_arrival_year', np.nan))

                # Extract parameters
                params = record.get('parameters', {})
                samples_dict['h_SC'].append(params.get('h_SC', np.nan))
                samples_dict['T_t'].append(params.get('T_t', np.nan))
                samples_dict['cost_speed'].append(params.get('cost_speed', np.nan))
                samples_dict['announcement_delay'].append(params.get('announcement_delay', np.nan))
                samples_dict['present_prog_multiplier'].append(params.get('present_prog_multiplier', np.nan))
                samples_dict['SC_prog_multiplier'].append(params.get('SC_prog_multiplier', np.nan))
                samples_dict['patch_rd_speedup'].append(params.get('patch_rd_speedup', False))
                samples_dict['software_progress_share'].append(params.get('software_progress_share', np.nan))

                # Determine growth type flags
                growth_type = record.get('growth_type', '')
                samples_dict['is_exponential'].append(growth_type == 'exponential')
                samples_dict['is_superexponential'].append(growth_type == 'superexponential')
                samples_dict['is_subexponential'].append(growth_type == 'subexponential')

                # These may not be in the JSONL, use defaults
                samples_dict['se_doubling_decay_fraction'].append(params.get('se_doubling_decay_fraction', 0.5))
                samples_dict['sub_doubling_growth_fraction'].append(params.get('sub_doubling_growth_fraction', 0.5))

        # Convert to numpy arrays
        for k in samples_dict:
            samples_dict[k] = np.array(samples_dict[k])

        all_forecaster_backcast_trajectories[forecaster_name] = backcast_list
        all_forecaster_trajectories[forecaster_name] = forecast_list
        all_forecaster_samples[forecaster_name] = samples_dict
        all_forecaster_results[forecaster_name] = np.array(results_list)

        print(f"Loaded {len(backcast_list)} trajectories for {forecaster_name} from {jsonl_file.name}")

    return (all_forecaster_backcast_trajectories, all_forecaster_trajectories,
            all_forecaster_samples, all_forecaster_results)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Global variable to control which disclaimer variant to use
_disclaimer_variant = "original"  # "original" or "may"

def set_disclaimer_variant(variant: str) -> None:
    """Set the disclaimer variant to use. Options: 'original' or 'may'."""
    global _disclaimer_variant
    _disclaimer_variant = variant

def add_disclaimer(fig: plt.Figure) -> None:
    """Add disclaimer text at the bottom of a figure."""
    if _disclaimer_variant == "may":
        disclaimer_text = (
            "This plot was generated using a revised version of our AI 2027 timelines model that we published in May 2025. "
            "Since then, we have created an improved model, which predicts somewhat longer timelines."
        )
    else:
        disclaimer_text = (
            "This plot was generated using the timelines model that was released alongside AI 2027. "
            "Since then, we have created an improved model, which predicts longer timelines."
        )
    fig.text(
        0.5, 0.02, disclaimer_text,
        ha='center', va='bottom',
        fontsize=11, color='red',
        wrap=True,
        transform=fig.transFigure
    )
    # Adjust bottom margin to make room for the disclaimer
    fig.subplots_adjust(bottom=0.15)

def load_external_data(yaml_path: str = "../external/benchmark_results.yaml") -> pd.DataFrame:
    """Load external METR benchmark data points for overlay on plots."""
    if not Path(yaml_path).exists():
        print(f"Warning: External data file {yaml_path} not found")
        return pd.DataFrame()
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    records = []
    
    # Extract data for each model
    for model_name, model_data in data.get('results', {}).items():
        release_date = model_data.get('release_date')
        if not release_date:
            continue
            
        # Parse release_date to decimal year format
        try:
            release_dt = pd.to_datetime(release_date)
            release_year_decimal = (release_dt.year + 
                                  (release_dt.month - 1) / 12 + 
                                  (release_dt.day - 1) / 365)
        except:
            print(f"Warning: Could not parse release_date for {model_name}: {release_date}")
            continue
        
        # Extract p80_horizon_length estimate from agents data
        agents_data = model_data.get('agents', {})
        for agent_name, agent_data in agents_data.items():
            p80_data = agent_data.get('p80_horizon_length', {})
            p80_estimate = p80_data.get('estimate')
            
            if p80_estimate is not None:
                records.append({
                    'model_name': model_name,
                    'agent_name': agent_name,
                    'release_date': release_date,
                    'release_year_decimal': release_year_decimal,
                    'p80': p80_estimate  # This is already in minutes according to the YAML
                })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} external METR data points from {yaml_path}")
    return df

def format_year_month(year_decimal: float) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= 2050:
        return ">2050"
        
    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    return f"{month_name} {year}"

def plot_results(all_forecaster_results: dict, config: dict) -> plt.Figure:
    """Create plot showing results from all forecasters."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    # Get current year for x-axis range
    current_year = 2025.25
    # current_year = datetime.now().year
    x_min = current_year
    x_max = current_year + 11
    
    # Plot each forecaster's results
    stats_text = []
    for name, results in all_forecaster_results.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Filter out >2050 points for density plot only
        valid_results = [r for r in results if r <= 2050]
        
        # Use KDE for smooth density estimation
        kde = gaussian_kde(valid_results)
        x_range = np.linspace(min(valid_results), max(valid_results), 200)
        density = kde(x_range)
        
        # Plot line with shaded area
        ax.plot(x_range, density, '-', color=color, label=name,
                linewidth=2, alpha=0.8, zorder=2)
        ax.fill_between(x_range, density, color=color, alpha=0.1)
        
        # Calculate statistics using all results to properly show >2050
        stats = (
            f"{name}:\n"
            f"  10th: {format_year_month(np.percentile(results, 10))}\n"
            f"  50th: {format_year_month(np.percentile(results, 50))}\n"
            f"  90th: {format_year_month(np.percentile(results, 90))}\n"
        )
        stats_text.append(stats)

    
    # Add statistics text box
    ax.text(0.7, 0.95, "\n\n".join(stats_text),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            # bbox=dict(facecolor=bg_rgb, alpha=0.9,
            #          edgecolor=config["plotting_style"]["colors"]["human"]["dark"],
            #          linewidth=0.5),
            fontsize=config["plotting_style"]["font"]["sizes"]["legend"])
    
    # Configure plot
    ax.set_title("Superhuman Coder Arrival, Time Horizon Extension",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("Probability Density", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
    # Set axis properties
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig


def plot_results_cdf(all_forecaster_results: dict, config: dict, *, show_percentile_lines: bool = True) -> plt.Figure:
    """Create CDF plot showing cumulative distribution of SC arrival times."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))

    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    # Get current year for x-axis range
    current_year = 2025.25
    x_min = current_year
    x_max = current_year + 11

    # Plot each forecaster's CDF
    for name, results in all_forecaster_results.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]

        # Sort results for CDF
        sorted_results = np.sort(results)
        cdf = np.arange(1, len(sorted_results) + 1) / len(sorted_results)

        # Plot CDF line
        ax.plot(sorted_results, cdf, '-', color=color, label=name,
                linewidth=2, alpha=0.8, zorder=2)


    # Configure plot
    ax.set_title("Superhuman Coder Arrival CDF, Time Horizon Extension",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("Cumulative Probability", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    # Set axis properties
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)

    # Add horizontal reference lines at key percentiles
    if show_percentile_lines:
        for pct in [0.1, 0.5, 0.9]:
            ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    legend = ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig


def save_pdf_cdf_csvs(all_forecaster_results: dict, config: dict, output_dir: Path) -> None:
    """Save PDF and CDF data points to CSV files for each forecaster.

    Creates a subfolder 'cdf_and_pdf_csvs' in output_dir containing one CSV per forecaster.
    Each CSV contains columns: x, pdf, cdf
    """
    csv_dir = output_dir / "cdf_and_pdf_csvs"
    csv_dir.mkdir(exist_ok=True)

    for name, results in all_forecaster_results.items():
        # Filter out >2050 points for PDF (matching plot_results behavior)
        valid_results = [r for r in results if r <= 2050]

        if len(valid_results) < 2:
            continue

        # Compute PDF using KDE (same as plot_results)
        kde = gaussian_kde(valid_results)
        x_range = np.linspace(min(valid_results), max(valid_results), 200)
        pdf_values = kde(x_range)

        # Compute CDF: for each x value, compute the fraction of results <= x
        # Using all results (not just valid_results) to match plot_results_cdf behavior
        sorted_results = np.sort(results)
        cdf_values = np.searchsorted(sorted_results, x_range, side='right') / len(sorted_results)

        # Create dataframe and save
        df = pd.DataFrame({
            'x': x_range,
            'pdf': pdf_values,
            'cdf': cdf_values
        })

        # Sanitize forecaster name for filename
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        df.to_csv(csv_dir / f"{safe_name}_pdf_cdf.csv", index=False)

    print(f"Saved PDF/CDF CSVs to {csv_dir}")


def _parse_month_year(month_year_str: str) -> tuple[float, float, float]:
    """Return (start_decimal, end_decimal, mid_decimal) for a given 'Month YYYY'.
    Accepts both full (e.g. 'March 2027') and abbreviated (e.g. 'Mar 2027') month names.
    """
    from datetime import datetime
    month_year_str = month_year_str.strip()
    dt = None
    for fmt in ("%B %Y", "%b %Y"):
        try:
            dt = datetime.strptime(month_year_str, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"Invalid month-year format: '{month_year_str}'. Expected e.g. 'March 2027'.")
    start_decimal = dt.year + (dt.month - 1) / 12.0
    end_decimal = dt.year + dt.month / 12.0
    mid_decimal = (start_decimal + end_decimal) / 2.0
    return start_decimal, end_decimal, mid_decimal

def plot_trajectories_sc_month(
    all_forecaster_results: dict,
    all_forecaster_trajectories: dict,
    all_forecaster_samples: dict,
    config: dict,
    *,
    sc_month_str: str = "March 2027",
    forecaster_filter: list[str] = None,
) -> plt.Figure:
    """Generalized version of `plot_march_2027_trajectories` that accepts an
    arbitrary month string such as 'April 2028'."""

    month_start, month_end, month_mid = _parse_month_year(sc_month_str)
    tolerance = 0.1  # same tolerance that was hard-coded for March 2027 (~1.2 months)

    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    # Determine horizontal span of plot
    current_year = 2025.25  # keeping existing assumption
    x_min = current_year
    x_max = max(current_year + 3, month_end + 1)  # extend at least 1 year beyond target month

    total_trajectories_plotted = 0
    all_final_horizons_target = []
    all_final_horizons_all_runs = []
    all_h_sc_samples = []
    best_path = None  # will hold central trajectory when computed

    for name, results in all_forecaster_results.items():
        if forecaster_filter and name not in forecaster_filter:
            continue

        trajectories = all_forecaster_trajectories[name]
        base_name = name.split(" (")[0].lower()
        color = config["forecasters"][base_name]["color"]

        # indices of runs that end within the chosen month window (using tolerance window identical to previous impl.)
        target_indices = [i for i, end_time in enumerate(results) if 0 < end_time - month_start <= tolerance]

        # Gather stats & prints similar to previous implementation
        final_horizons_this_forecaster = []
        final_horizons_all_runs_this_forecaster = []

        for i, traj in enumerate(trajectories):
            if traj:
                final_minutes = traj[-1][1]
                final_horizons_all_runs_this_forecaster.append(final_minutes)
                all_final_horizons_all_runs.append(final_minutes)

        for idx in target_indices:
            traj = trajectories[idx]
            if traj:
                final_minutes = traj[-1][1]
                final_horizons_this_forecaster.append(final_minutes)
                all_final_horizons_target.append(final_minutes)

        # print summary similar to original (but now dynamic month)
        print(f"{name}: Found {len(target_indices)} runs reaching SC in {sc_month_str}")
        h_sc_samples = all_forecaster_samples[name]["h_SC"]
        all_h_sc_samples.extend(h_sc_samples)
        if final_horizons_all_runs_this_forecaster:
            horizons_all_work_months = np.array(final_horizons_all_runs_this_forecaster) / (60 * 167)
            print(f"  ALL RUNS final horizon distribution (work months): 50th={np.percentile(horizons_all_work_months,50):.2f}")
        if final_horizons_this_forecaster:
            horizons_target_work_months = np.array(final_horizons_this_forecaster) / (60 * 167)
            print(f"  {sc_month_str} final horizon dist (work months): 50th={np.percentile(horizons_target_work_months,50):.2f}")

        # Plot trajectories
        for idx in target_indices:
            traj = trajectories[idx]
            if traj:
                times, horizons = zip(*traj)
                ax.plot(times, horizons, '-', color=color, alpha=0.3, linewidth=1)
                total_trajectories_plotted += 1

    print(f"\nTotal trajectories plotted: {total_trajectories_plotted}")

    # Reference / decoration lines identical to original but labels dynamic
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='--', alpha=0.7,
               label='Current Horizon (15 min)', linewidth=2)
    ax.axvline(x=month_mid, color='blue', linestyle='--', alpha=0.7,
               label=sc_month_str, linewidth=2)

    # Title & axes labels
    ax.set_title(f"Time Horizon Extension Trajectories for Runs Reaching SC in {sc_month_str}",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')

    # Use same tick setup as original implementation (simplified: inherit current settings)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -------------------------------------------------------------
    # Add reference trajectory line with specific points (same as
    # the March-2027 plot – keeps context for the reader)
    # -------------------------------------------------------------
    reference_times = [2025.25, 2026, 2026.5, month_mid]
    reference_horizons = [
        15,          # 15 minutes
        240,         # 4 work hours
        4800,        # 2 work weeks
        320640       # 32 work months
    ]
    ax.plot(reference_times, reference_horizons, 'o-', color='purple',
            linewidth=3, markersize=6, alpha=0.8, label='Reference Timeline', zorder=10)

    # -------------------------------------------------------------
    # Create dynamic y-axis tick labels (copied from original impl.)
    # -------------------------------------------------------------
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    min_time_minutes = 0.1 / 60  # 0.1 s expressed in minutes
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)

    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
    ]

    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    if len(valid_ticks) > 8:
        filtered_ticks = [valid_ticks[0]]
        for i in range(2, len(valid_ticks)-1, 2):
            filtered_ticks.append(valid_ticks[i])
        if len(valid_ticks) > 1:
            filtered_ticks.append(valid_ticks[-1])
        valid_ticks = filtered_ticks

    if valid_ticks:
        tick_positions, tick_labels = zip(*valid_ticks)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    # -------------------------------------------------------------
    # Final legend & tick configuration
    # -------------------------------------------------------------
    legend = ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)

    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig

def plot_combined_trajectories_sc_month(
    all_forecaster_backcast_trajectories: dict,
    all_forecaster_trajectories: dict,
    all_forecaster_samples: dict,
    all_forecaster_results: dict,
    config: dict,
    *,
    sc_month_str: str = "March 2027",
    color_by_growth_type: bool = True,
    overlay_external_data: bool = True,
    plot_central_trajectory: bool = True,
    plot_median_curve: bool = False,
    overlay_illustrative_trend: bool = False,
    add_agent_checkpoints: bool = False,
    forecaster_filter: list[str] = None,
    jsonl_dir: str | Path = None,
) -> tuple[plt.Figure, dict|None]:
    """Generalized version of `plot_combined_trajectories_march_2027`.

    Args:
        jsonl_dir: Optional path to directory containing JSONL trajectory files.
                   If provided, will load trajectories from JSONL instead of using
                   the passed-in dictionaries.
    """

    # If jsonl_dir is provided, load trajectories from JSONL files
    if jsonl_dir is not None:
        (all_forecaster_backcast_trajectories,
         all_forecaster_trajectories,
         all_forecaster_samples,
         all_forecaster_results) = load_trajectories_from_jsonl(jsonl_dir)

    month_start, month_end, month_mid = _parse_month_year(sc_month_str)

    # Re-use most of the original defaults for colours & fonts
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(14, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    current_year = 2025.25
    backcast_years = 5
    forecast_years = max(3, int(np.ceil(month_end - current_year)) + 1)
    x_min = current_year - backcast_years
    x_max = current_year + forecast_years

    total_trajectories_plotted = 0
    all_combined_paths = []
    best_path = None  # will hold central trajectory when computed

    assert len(all_forecaster_backcast_trajectories) == len(all_forecaster_trajectories) == len(all_forecaster_samples) == len(all_forecaster_results), "All dictionaries must have the same number of keys"

    for name in all_forecaster_backcast_trajectories.keys():
        if forecaster_filter and name not in forecaster_filter:
            continue

        backcast_traj_list = all_forecaster_backcast_trajectories[name]
        forecast_traj_list = all_forecaster_trajectories[name]
        samples = all_forecaster_samples[name]
        results = all_forecaster_results[name]

        mask = (results >= month_start) & (results < month_end)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # growth type colours
        if color_by_growth_type:
            growth_colors = {
                "exponential": "#2E8B57",
                "superexponential": "#FF6347",
                "subexponential": "#4169E1",
            }
        else:
            base_name = name.split(" (")[0].lower()
            forecaster_color = config["forecasters"][base_name]["color"]

        # limit number plotted to avoid clutter
        max_plot = min(500, len(indices))
        for i_idx in range(max_plot):
            idx = indices[i_idx]
            back_traj = backcast_traj_list[idx]
            fore_traj = forecast_traj_list[idx]

            # Skip samples with missing or trivially short forward trajectories
            if not fore_traj or len(fore_traj) <= 1:
                print("WARNING: omitting trajectory with missing/short forward part")
                continue

            if color_by_growth_type:
                if samples["is_exponential"][idx]:
                    color = growth_colors["exponential"]
                elif samples["is_superexponential"][idx]:
                    color = growth_colors["superexponential"]
                else:
                    color = growth_colors["subexponential"]
            else:
                color = forecaster_color

            if back_traj:
                bt, bh = zip(*back_traj)
                ax.plot(bt, bh, '-', color=color, alpha=0.15, linewidth=0.8)
            if fore_traj:
                ft, fh = zip(*fore_traj)
                ax.plot(ft, fh, '-', color=color, alpha=0.15, linewidth=0.8)
            total_trajectories_plotted += 1

            # collate for central trajectory logic
            combined_t = []
            combined_h = []
            if back_traj:
                combined_t.extend(bt)
                combined_h.extend(bh)
            if fore_traj:
                combined_t.extend(ft)
                combined_h.extend(fh)
            if combined_t:
                order = np.argsort(combined_t)
                all_combined_paths.append({
                    'times': np.array(combined_t)[order],
                    'horizons': np.array(combined_h)[order],
                    'forecaster_name': name,
                    'sample_idx': idx,
                })

    print(f"Total {sc_month_str} combined trajectories plotted: {total_trajectories_plotted}")

    # decoration lines
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='gray', linestyle=':', alpha=0.8, label='Current Horizon (15 min)', linewidth=3)
    ax.axvline(x=current_year, color='gray', linestyle=':', alpha=0.7, label='Current Time', linewidth=2)
    ax.axvline(x=month_mid, color='purple', linestyle=':', alpha=0.8, label=f'{sc_month_str} (SC Arrival)', linewidth=2)

    # External data overlay (reuse existing helper)
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible = external_df[mask]
            if not visible.empty:
                ax.scatter(visible['release_year_decimal'], visible['p80'], color='black', s=25, alpha=0.8, zorder=15, marker='o', label='External Benchmarks (p80)')

    # titles & axes
    # Determine title based on forecaster name
    active_forecaster = forecaster_filter[0] if forecaster_filter and len(forecaster_filter) == 1 else None
    if active_forecaster == "Eli_patched_rd_speedup":
        plot_title = "Comparing Bug-Fixed Model Trajectories to Graph Curves"
    else:
        plot_title = "Comparing Model Trajectories to Graph Curves"
    plot_subtitle = f"(Filtered for {sc_month_str} SC Arrivals)"
    ax.set_title(f"{plot_title}\n{plot_subtitle}", fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -------------------------------------------------------------
    # Y-axis tick labels (copied from original combined-plot impl.)
    # -------------------------------------------------------------
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    min_time_minutes = 0.1 / 60
    if y_min < min_time_minutes:
        y_min = min_time_minutes
    # Ensure y_max extends to at least 50 work years (6012000 minutes)
    if y_max < 6012000:
        y_max = 6012000
    ax.set_ylim(y_min, y_max)

    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
        (6012000, "50 work years"),
    ]

    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    if valid_ticks:
        pos, lab = zip(*valid_ticks)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)

    # -------------------------------------------------------------
    # Optional central trajectory & median curve (already implemented)
    # plus agent checkpoints and illustrative trend overlay
    # -------------------------------------------------------------
    # Central trajectory logic is above; we now add checkpoints & trend
    if plot_central_trajectory and all_combined_paths:
        x_grid = np.arange(x_min, x_max + 1e-6, 1/12)
        matrix = np.full((len(all_combined_paths), len(x_grid)), np.nan)
        for i, p in enumerate(all_combined_paths):
            mask = (x_grid >= p['times'][0]) & (x_grid <= p['times'][-1])
            if mask.any():
                matrix[i, mask] = np.log10(np.interp(x_grid[mask], p['times'], p['horizons']))
        median_curve = np.nanmedian(matrix, axis=0)
        valid = ~np.isnan(median_curve)
        if plot_median_curve and valid.any():
            ax.plot(x_grid[valid], 10**median_curve[valid], color='red', linewidth=2, linestyle=':', label='Median Trajectory', zorder=49)
        if valid.any():
            # Method 1: Old method - distances from 2025.25 (forecast start) up to the end of the SC month filter
            time_mask_old = (x_grid >= 2025.25) & (x_grid <= month_end)
            matrix_truncated_old = matrix[:, time_mask_old]
            median_truncated_old = median_curve[time_mask_old]
            distances_old = np.nanmean(np.abs(matrix_truncated_old - median_truncated_old), axis=1)
            best_idx_old = np.nanargmin(distances_old)
            best_path_old = all_combined_paths[int(best_idx_old)]
            ax.plot(best_path_old['times'], best_path_old['horizons'], color='gray', linewidth=2, linestyle='--', label='Central Trajectory (forecast only)', zorder=47)

            # Method 2: New method - distances from 2021.0 (backcast start) up to the end of the SC month filter
            time_mask = (x_grid >= 2021.0) & (x_grid <= month_end)
            matrix_truncated = matrix[:, time_mask]
            median_truncated = median_curve[time_mask]
            distances = np.nanmean(np.abs(matrix_truncated - median_truncated), axis=1)
            # Exclude trajectories that don't have points before 2021.0
            for i, p in enumerate(all_combined_paths):
                if p['times'][0] >= 2021.0:
                    distances[i] = np.inf
            best_idx = np.nanargmin(distances)
            best_path = all_combined_paths[int(best_idx)]
            ax.plot(best_path['times'], best_path['horizons'], color='black', linewidth=2, linestyle='--', label='Central Trajectory (incl. backcast)', zorder=48)

            # Method 3: Percentile-based z-score method from 2021.0
            # For each trajectory at each timestep, compute percentile rank, convert to z-score, average abs(z)
            n_traj, n_times = matrix_truncated.shape
            z_scores_sum = np.zeros(n_traj)
            z_scores_count = np.zeros(n_traj)
            for t_idx in range(n_times):
                col = matrix_truncated[:, t_idx]
                valid_mask = ~np.isnan(col)
                if valid_mask.sum() > 1:
                    # Compute percentile rank for each trajectory at this timestep
                    ranks = rankdata(col[valid_mask], method='average')
                    percentiles = (ranks - 0.5) / len(ranks)  # map to (0, 1)
                    # Convert percentile to z-score (number of SDs from mean in normal dist)
                    z_vals = np.abs(norm.ppf(percentiles))
                    # Add to running sum for valid trajectories
                    valid_indices = np.where(valid_mask)[0]
                    for i, z in zip(valid_indices, z_vals):
                        z_scores_sum[i] += z
                        z_scores_count[i] += 1
            # Compute average z-score (avoid division by zero)
            z_scores_avg = np.where(z_scores_count > 0, z_scores_sum / z_scores_count, np.inf)
            # Exclude trajectories that don't have points before 2021.0
            for i, p in enumerate(all_combined_paths):
                if p['times'][0] >= 2021.0:
                    z_scores_avg[i] = np.inf
            best_idx_zscore = np.nanargmin(z_scores_avg)
            best_path_zscore = all_combined_paths[int(best_idx_zscore)]
            ax.plot(best_path_zscore['times'], best_path_zscore['horizons'], color='blue', linewidth=2, linestyle='--', label='Central Trajectory (z-score)', zorder=46)

            # Add sample parameters to best_path for downstream use
            forecaster_name = best_path.get('forecaster_name')
            sample_idx = best_path.get('sample_idx')
            if forecaster_name is not None and sample_idx is not None and forecaster_name in all_forecaster_samples:
                samples = all_forecaster_samples[forecaster_name]
                best_path['h_SC'] = float(samples['h_SC'][sample_idx])
                best_path['T_t'] = float(samples['T_t'][sample_idx])
                best_path['cost_speed'] = float(samples['cost_speed'][sample_idx])
                best_path['announcement_delay'] = float(samples['announcement_delay'][sample_idx])
                best_path['present_prog_multiplier'] = float(samples['present_prog_multiplier'][sample_idx])
                best_path['SC_prog_multiplier'] = float(samples['SC_prog_multiplier'][sample_idx])
                best_path['is_exponential'] = bool(samples['is_exponential'][sample_idx])
                best_path['is_superexponential'] = bool(samples['is_superexponential'][sample_idx])
                best_path['is_subexponential'] = bool(samples['is_subexponential'][sample_idx])
                best_path['patch_rd_speedup'] = bool(samples['patch_rd_speedup'][sample_idx])
                best_path['software_progress_share'] = float(samples['software_progress_share'][sample_idx])
                best_path['se_doubling_decay_fraction'] = float(samples['se_doubling_decay_fraction']) if isinstance(samples['se_doubling_decay_fraction'], (int, float)) else float(samples['se_doubling_decay_fraction'][sample_idx])
                best_path['sub_doubling_growth_fraction'] = float(samples['sub_doubling_growth_fraction']) if isinstance(samples['sub_doubling_growth_fraction'], (int, float)) else float(samples['sub_doubling_growth_fraction'][sample_idx])

            # -----------------------------------------------------------------
            # Add labelled check-points on the median trajectory (if requested)
            # -----------------------------------------------------------------
            if add_agent_checkpoints:
                checkpoints = [
                    (2025 + 7/12, "Agent-0"),
                    (2026 + 2/12, "Agent-1"),
                    (2027 + 6/12, "Agent-3-mini"),
                ]
                t_arr = best_path['times']
                h_arr = best_path['horizons']
                for t_pt, label in checkpoints:
                    if t_pt < t_arr[0] or t_pt > t_arr[-1]:
                        continue
                    h_pt = np.interp(t_pt, t_arr, h_arr)
                    ax.scatter(t_pt, h_pt, color='green', s=40, zorder=49)
                    ax.annotate(label, (t_pt, h_pt), textcoords="offset points",
                                xytext=(5, -5), ha='left', fontsize=config["plotting_style"]["font"]["sizes"]["ticks"], color='black')

    # ---------------------------------------------------------------------
    # Overlay illustrative SE trends if requested
    # ---------------------------------------------------------------------
    if overlay_illustrative_trend:
        # Use script-relative paths so the CSVs are found regardless of cwd
        script_dir = Path(__file__).resolve().parent
        external_dir = script_dir.parent / "external"

        # Original illustrative graph trend (exp_power model)
        original_trend_path = external_dir / "original_graph_trend_generated.csv"
        if original_trend_path.exists():
            try:
                original_df = pd.read_csv(original_trend_path)
                mask = (original_df['year'] >= x_min) & (original_df['year'] <= x_max)
                if mask.any():
                    ax.plot(original_df.loc[mask, 'year'], original_df.loc[mask, 'horizon_minutes'],
                            color='green', linewidth=2, linestyle='--')
            except Exception as e:
                print(f"Warning: failed to plot original illustrative trend: {e}")

        # Fixed illustrative graph trend (horizon-doubling model)
        fixed_trend_path = external_dir / "fixed_illustrative_graph_trend.csv"
        if fixed_trend_path.exists():
            try:
                fixed_df = pd.read_csv(fixed_trend_path)
                mask = (fixed_df['year'] >= x_min) & (fixed_df['year'] <= x_max)
                if mask.any():
                    ax.plot(fixed_df.loc[mask, 'year'], fixed_df.loc[mask, 'horizon_minutes'],
                            color='blue', linewidth=2, linestyle='--')
            except Exception as e:
                print(f"Warning: failed to plot fixed illustrative trend: {e}")

    # -------------------------------------------------------------
    # Assemble custom legend (mirrors original combined plot)
    # -------------------------------------------------------------
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label=f'{sc_month_str} (SC Arrival)'),
    ]

    # Add METR line before growth type trajectories
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask2 = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            if external_df[mask2].shape[0] > 0:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=4, label='METR 80% time horizons', linestyle='None'))

    if color_by_growth_type:
        legend_elements.extend([
            Line2D([0], [0], color='#2E8B57', linewidth=2, alpha=0.15, label='Exponential Growth Trajectories'),
            Line2D([0], [0], color='#FF6347', linewidth=2, alpha=0.15, label='Superexponential Growth Trajectories'),
            Line2D([0], [0], color='#4169E1', linewidth=2, alpha=0.15, label='Subexponential Growth Trajectories')
        ])
    else:
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, label='Combined Trajectories'))

    # Add Central Trajectory after growth types
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Central Trajectory'))

    if plot_median_curve:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Median'))
    if overlay_illustrative_trend:
        legend_elements.append(Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Original illustrative graph curve'))
        legend_elements.append(Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Intended illustrative graph curve'))

    legend = ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)

    # Final tick configuration
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # -----------------------------------------------------------------
    # Prepare return value for central trajectory so caller can compare
    # -----------------------------------------------------------------
    central_traj: dict | None = None
    try:
        central_traj = best_path  # may raise if best_path undefined
    except Exception:
        central_traj = None

    add_disclaimer(fig)
    return fig, central_traj

# -----------------------------------------------------------------------------
# Compatibility wrappers for legacy function names (March 2027 specific)
# -----------------------------------------------------------------------------

def plot_march_2027_trajectories(
    all_forecaster_results: dict,
    all_forecaster_trajectories: dict,
    all_forecaster_samples: dict,
    config: dict,
    forecaster_filter: list[str] = None,
):
    """Backward-compatibility wrapper calling `plot_trajectories_sc_month` with March 2027."""
    return plot_trajectories_sc_month(
        all_forecaster_results,
        all_forecaster_trajectories,
        all_forecaster_samples,
        config,
        sc_month_str="March 2030",
        forecaster_filter=forecaster_filter,
    )

def plot_combined_trajectories_march_2027(
    all_forecaster_backcast_trajectories: dict,
    all_forecaster_trajectories: dict,
    all_forecaster_samples: dict,
    all_forecaster_results: dict,
    config: dict,
    *,
    color_by_growth_type: bool = True,
    overlay_external_data: bool = True,
    plot_central_trajectory: bool = True,
    plot_median_curve: bool = False,
    overlay_illustrative_trend: bool = False,
    add_agent_checkpoints: bool = False,
    forecaster_filter: list[str] = None,
    jsonl_dir: str | Path = None,
):
    """Backward-compatibility wrapper calling `plot_combined_trajectories_sc_month` with March 2027."""
    fig, _ = plot_combined_trajectories_sc_month(
        all_forecaster_backcast_trajectories,
        all_forecaster_trajectories,
        all_forecaster_samples,
        all_forecaster_results,
        config,
        sc_month_str="March 2030",
        color_by_growth_type=color_by_growth_type,
        overlay_external_data=overlay_external_data,
        plot_central_trajectory=plot_central_trajectory,
        plot_median_curve=plot_median_curve,
        overlay_illustrative_trend=overlay_illustrative_trend,
        add_agent_checkpoints=add_agent_checkpoints,
        forecaster_filter=forecaster_filter,
        jsonl_dir=jsonl_dir,
    )
    return fig

def plot_backcasted_trajectories(all_forecaster_backcast_trajectories: dict, all_forecaster_samples: dict, config: dict, color_by_growth_type: bool = True, overlay_external_data: bool = True, forecaster_filter: list[str] = None) -> plt.Figure:
    """Create plot showing backcasted time horizon trajectories."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Get current year for x-axis range
    current_year = 2025.25
    backcast_years = 5
    x_min = current_year - backcast_years
    x_max = current_year + 0.5
    
    total_trajectories_plotted = 0
    
    # Plot backcasted trajectories for each forecaster
    for name, trajectories in all_forecaster_backcast_trajectories.items():
        if forecaster_filter and name not in forecaster_filter:
            continue
        
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Get samples for this forecaster to filter out subexponential
        samples = all_forecaster_samples[name]
        
        # Include all growth types - we'll color-code by growth type instead
        filtered_trajectories = []
        growth_types = []
        for i, trajectory in enumerate(trajectories):
            filtered_trajectories.append(trajectory)
            # Determine growth type for this trajectory
            if samples["is_exponential"][i]:
                growth_types.append("exponential")
            elif samples["is_superexponential"][i]:
                growth_types.append("superexponential")
            else:  # is_subexponential
                growth_types.append("subexponential")
        
        # Plot a subset of trajectories to avoid overcrowding
        max_trajectories_to_plot = 500  # Show 500 trajectories per forecaster
        trajectories_to_plot = filtered_trajectories[:max_trajectories_to_plot]
        growth_types_to_plot = growth_types[:max_trajectories_to_plot]
        
        print(f"{name}: Plotting {len(trajectories_to_plot)} backcasted trajectories (all growth types)")
        
        if color_by_growth_type:
            # Define colors for each growth type
            growth_colors = {
                "exponential": "#2E8B57",        # Sea green
                "superexponential": "#FF6347",   # Tomato red  
                "subexponential": "#4169E1"      # Royal blue
            }
            
            # Plot trajectories with growth-type-specific colors
            for trajectory, growth_type in zip(trajectories_to_plot, growth_types_to_plot):
                if trajectory:  # Make sure trajectory is not empty
                    times, horizons = zip(*trajectory)
                    ax.plot(times, horizons, '-', color=growth_colors[growth_type], alpha=0.1, linewidth=0.8)
                    total_trajectories_plotted += 1
        else:
            # Plot all trajectories in red
            for trajectory in trajectories_to_plot:
                if trajectory:  # Make sure trajectory is not empty
                    times, horizons = zip(*trajectory)
                    ax.plot(times, horizons, '-', color='red', alpha=0.1, linewidth=0.8)
                    total_trajectories_plotted += 1
    
    print(f"\nTotal backcasted trajectories plotted: {total_trajectories_plotted}")
    
    # Add current horizon line (where all trajectories should converge)
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='-', alpha=0.8, 
               label='Current Horizon (15 min)', linewidth=3, zorder=10)
    
    # Add vertical line for current time
    ax.axvline(x=current_year, color='blue', linestyle='--', alpha=0.7, 
               label='Current Time', linewidth=2)
    
    # Overlay external data points if requested
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            # Filter to points within the plot's time range
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible_data = external_df[time_mask]
            
            if not visible_data.empty:
                ax.scatter(visible_data['release_year_decimal'], visible_data['p80'], 
                          color='black', s=25, alpha=0.8, zorder=15, marker='o',
                          label='External Benchmarks (p80)')
                print(f"Overlaid {len(visible_data)} external benchmark points")
    
    # Configure plot
    ax.set_title("Backcasted Time Horizon Trajectories\n(Historical development leading to current capabilities)",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')  # Log scale for time horizon
    
    # Create dynamic y-axis labels with human-readable time units
    def format_time_label(minutes):
        """Convert minutes to human-readable time labels."""
        if minutes < 1:
            return f"{minutes*60:.0f}s"
        elif minutes < 60:
            return f"{minutes:.0f}m"
        elif minutes < 1440:  # 24 hours
            hours = minutes / 60
            return f"{hours:.0f}h"
        elif minutes < 10080:  # 1 week
            days = minutes / 1440
            return f"{days:.0f}d"
        elif minutes < 43200:  # 1 month (30 days)
            weeks = minutes / 10080
            return f"{weeks:.0f}w"
        elif minutes < 525600:  # 1 year
            months = minutes / 43200
            return f"{months:.0f}mo"
        else:
            years = minutes / 525600
            return f"{years:.0f}y"
    
    # Get current y-axis limits to determine appropriate tick positions
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    # Ensure the y-axis never goes below 0.1 seconds (≈0.00167 minutes)
    min_time_minutes = 0.1 / 60  # Convert 0.1 sec to minutes
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)
    
    # Generate tick positions that make sense for time horizons
    tick_positions = []
    tick_labels = []
    
    # Define all possible tick positions with their labels
    all_ticks = [
        # Short times
        (0.01, "0.6s"),
        (0.1, "6s"),
        (0.5, "30s"),
        (1, "1 min"),
        (5, "5 min"),
        (15, "15 min"),
        (30, "30 min"),
        (60, "1 hour"),
        (240, "4 hours"),
        # Work-based time units
        (480, "1 work day"),      # 8 hours
        (2400, "1 work week"),    # 40 hours  
        (10020, "1 work month"),  # 167 hours
    ]
    
    # Filter ticks to be within range and well-spaced
    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    
    # If we have too many ticks, thin them out intelligently
    if len(valid_ticks) > 8:
        # Keep every other tick, but always keep the first and last
        filtered_ticks = [valid_ticks[0]]  # Always keep first
        for i in range(2, len(valid_ticks) - 1, 2):  # Take every other middle tick
            filtered_ticks.append(valid_ticks[i])
        if len(valid_ticks) > 1:
            filtered_ticks.append(valid_ticks[-1])  # Always keep last
        valid_ticks = filtered_ticks
    
    # Set the final ticks
    if valid_ticks:
        tick_positions, tick_labels = zip(*valid_ticks)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend for reference lines and growth types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='-', linewidth=3, label='Current Horizon (15 min)'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Current Time')
    ]
    
    if color_by_growth_type:
        legend_elements.extend([
            Line2D([0], [0], color='#2E8B57', linewidth=2, label='Exponential Growth'),
            Line2D([0], [0], color='#FF6347', linewidth=2, label='Superexponential Growth'), 
            Line2D([0], [0], color='#4169E1', linewidth=2, label='Subexponential Growth')
        ])
    else:
        legend_elements.append(
            Line2D([0], [0], color='red', linewidth=2, label='Backcasted Trajectories')
        )
    
    # Add external data to legend if present
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            if external_df[time_mask].shape[0] > 0:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                           markersize=4, label='External Benchmarks (p80)', linestyle='None')
                )
      
    legend = ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)

    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig

def plot_combined_trajectories(
    all_forecaster_backcast_trajectories: dict,
    all_forecaster_trajectories: dict,
    all_forecaster_samples: dict,
    config: dict,
    *,
    color_by_growth_type: bool = True,
    overlay_external_data: bool = True,
    plot_central_trajectory: bool = True,
    plot_median_curve: bool = False,
    add_agent_checkpoints: bool = False,
    forecaster_filter: list[str] = None,
    jsonl_dir: str | Path = None,
) -> plt.Figure:
    """Create plot showing both backcasted and forecasted time horizon trajectories.

    Args:
        jsonl_dir: Optional path to directory containing JSONL trajectory files.
                   If provided, will load trajectories from JSONL instead of using
                   the passed-in dictionaries.
    """
    # If jsonl_dir is provided, load trajectories from JSONL files
    if jsonl_dir is not None:
        (all_forecaster_backcast_trajectories,
         all_forecaster_trajectories,
         all_forecaster_samples,
         _) = load_trajectories_from_jsonl(jsonl_dir)

    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    fig = plt.figure(figsize=(14, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Get current year for x-axis range
    current_year = 2025.25
    backcast_years = 5
    forecast_years = 10
    x_min = current_year - backcast_years
    x_max = current_year + forecast_years
    
    total_trajectories_plotted = 0
    # Collect combined (backcast+forecast) paths for central/median computation
    all_combined_paths: list[dict] = []
    best_path: dict | None = None

    for name in all_forecaster_backcast_trajectories.keys():
        if forecaster_filter and name not in forecaster_filter:
            continue
        
        backcast_trajectories = all_forecaster_backcast_trajectories[name]
        forecast_trajectories = all_forecaster_trajectories[name]
        samples = all_forecaster_samples[name]
        
        # Get growth types for color coding
        growth_types = []
        for i in range(len(backcast_trajectories)):
            if samples["is_exponential"][i]:
                growth_types.append("exponential")
            elif samples["is_superexponential"][i]:
                growth_types.append("superexponential")
            else:  # is_subexponential
                growth_types.append("subexponential")
        
        # Plot a subset of trajectories to avoid overcrowding
        max_trajectories_to_plot = 1000  # Fewer trajectories for combined plot
        n_to_plot = min(max_trajectories_to_plot, len(backcast_trajectories), len(forecast_trajectories))
        
        print(f"{name}: Plotting {n_to_plot} combined trajectories (backcast + forecast)")
        
        if color_by_growth_type:
            # Define colors for each growth type
            growth_colors = {
                "exponential": "#2E8B57",        # Sea green
                "superexponential": "#FF6347",   # Tomato red  
                "subexponential": "#4169E1"      # Royal blue
            }
        else:
            # Use forecaster color for all trajectories
            base_name = name.split(" (")[0].lower()
            forecaster_color = config["forecasters"][base_name]["color"]
        
        # Plot combined trajectories
        for i in range(n_to_plot):
            # Skip samples with missing or trivially short forward trajectories
            if not forecast_trajectories[i] or len(forecast_trajectories[i]) <= 1:
                print("WARNING: omitting trajectory with missing/short forward part")
                continue

            backcast_traj = backcast_trajectories[i]
            forecast_traj = forecast_trajectories[i]
            growth_type = growth_types[i]
            
            if color_by_growth_type:
                color = growth_colors[growth_type]
            else:
                color = forecaster_color
            
            # Plot backcast trajectory (past)
            combined_t = []
            combined_h = []

            if backcast_traj:
                back_times, back_horizons = zip(*backcast_traj)
                ax.plot(back_times, back_horizons, '-', color=color, alpha=0.10, linewidth=0.8)
                combined_t.extend(back_times)
                combined_h.extend(back_horizons)

            # Plot forecast trajectory (future)
            if forecast_traj:
                fore_times, fore_horizons = zip(*forecast_traj)
                ax.plot(fore_times, fore_horizons, '-', color=color, alpha=0.10, linewidth=0.8)
                combined_t.extend(fore_times)
                combined_h.extend(fore_horizons)

            # Store path for central/median logic
            if combined_t:
                order = np.argsort(combined_t)
                all_combined_paths.append({
                    'times': np.array(combined_t)[order],
                    'horizons': np.array(combined_h)[order],
                    'forecaster_name': name,
                    'sample_idx': i,
                })

            total_trajectories_plotted += 1

    print(f"\nTotal combined trajectories plotted: {total_trajectories_plotted}")
    
    # Add current horizon line (where trajectories converge)
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='-', alpha=0.8, 
               label='Current Horizon (15 min)', linewidth=3, zorder=10)
    
    # Add vertical line for current time
    ax.axvline(x=current_year, color='blue', linestyle='--', alpha=0.7, 
               label='Current Time', linewidth=2)
    
    # Overlay external data points if requested
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            # Filter to points within the plot's time range
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible_data = external_df[time_mask]
            
            if not visible_data.empty:
                ax.scatter(visible_data['release_year_decimal'], visible_data['p80'], 
                          color='black', s=25, alpha=0.8, zorder=15, marker='o',
                          label='External Benchmarks (p80)')
                print(f"Overlaid {len(visible_data)} external benchmark points")
    
    # Configure plot
    ax.set_title("Complete Time Horizon Extension Trajectories\n(Historical development and future projections)",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')  # Log scale for time horizon
    
    # Create dynamic y-axis labels with human-readable time units
    def format_time_label(minutes):
        """Convert minutes to human-readable time labels."""
        if minutes < 1:
            return f"{minutes*60:.0f}s"
        elif minutes < 60:
            return f"{minutes:.0f}m"
        elif minutes < 1440:  # 24 hours
            hours = minutes / 60
            return f"{hours:.0f}h"
        elif minutes < 10080:  # 1 week
            days = minutes / 1440
            return f"{days:.0f}d"
        elif minutes < 43200:  # 1 month (30 days)
            weeks = minutes / 10080
            return f"{weeks:.0f}w"
        elif minutes < 525600:  # 1 year
            months = minutes / 43200
            return f"{months:.0f}mo"
        else:
            years = minutes / 525600
            return f"{years:.0f}y"
    
    # Get current y-axis limits to determine appropriate tick positions
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    # Ensure the y-axis never goes below 0.1 seconds (≈0.00167 minutes)
    min_time_minutes = 0.1 / 60  # Convert 0.1 sec to minutes
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)
    
    # Generate tick positions that make sense for time horizons
    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
    ]
    
    # Filter ticks to be within range
    valid_ticks = [(pos, lab) for pos, lab in all_ticks if y_min <= pos <= y_max]
    
    # Set the final ticks
    if valid_ticks:
        tick_positions, tick_labels = zip(*valid_ticks)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    # -------------------------------------------------------------
    # Median / Central trajectory plotting (borrowed from _sc_month)
    # -------------------------------------------------------------
    if plot_central_trajectory and all_combined_paths:
        x_grid = np.arange(x_min, x_max + 1e-6, 1/12)
        matrix = np.full((len(all_combined_paths), len(x_grid)), np.nan)
        for i, p in enumerate(all_combined_paths):
            mask = (x_grid >= p['times'][0]) & (x_grid <= p['times'][-1])
            if mask.any():
                matrix[i, mask] = np.log10(np.interp(x_grid[mask], p['times'], p['horizons']))

        median_curve = np.nanmedian(matrix, axis=0)
        valid = ~np.isnan(median_curve)

        if plot_median_curve and valid.any():
            ax.plot(x_grid[valid], 10**median_curve[valid], color='red', linewidth=2, linestyle=':', label='Median Trajectory', zorder=49)

        if valid.any():
            # Method 1: Old method - distances from 2025.25 (forecast start)
            time_mask_old = x_grid >= 2025.25
            matrix_truncated_old = matrix[:, time_mask_old]
            median_truncated_old = median_curve[time_mask_old]
            distances_old = np.nanmean(np.abs(matrix_truncated_old - median_truncated_old), axis=1)
            best_idx_old = np.nanargmin(distances_old)
            best_path_old = all_combined_paths[int(best_idx_old)]
            ax.plot(best_path_old['times'], best_path_old['horizons'], color='gray', linewidth=2, linestyle='--', label='Central Trajectory (forecast only)', zorder=47)

            # Method 2: New method - distances from 2021.0 (backcast start)
            time_mask = x_grid >= 2021.0
            matrix_truncated = matrix[:, time_mask]
            median_truncated = median_curve[time_mask]
            distances = np.nanmean(np.abs(matrix_truncated - median_truncated), axis=1)
            # Exclude trajectories that don't have points before 2021.0
            for i, p in enumerate(all_combined_paths):
                if p['times'][0] >= 2021.0:
                    distances[i] = np.inf
            best_idx = np.nanargmin(distances)
            best_path = all_combined_paths[int(best_idx)]
            ax.plot(best_path['times'], best_path['horizons'], color='black', linewidth=2, linestyle='--', label='Central Trajectory (incl. backcast)', zorder=48)

            # Method 3: Percentile-based z-score method from 2021.0
            # For each trajectory at each timestep, compute percentile rank, convert to z-score, average abs(z)
            n_traj, n_times = matrix_truncated.shape
            z_scores_sum = np.zeros(n_traj)
            z_scores_count = np.zeros(n_traj)
            for t_idx in range(n_times):
                col = matrix_truncated[:, t_idx]
                valid_mask = ~np.isnan(col)
                if valid_mask.sum() > 1:
                    # Compute percentile rank for each trajectory at this timestep
                    ranks = rankdata(col[valid_mask], method='average')
                    percentiles = (ranks - 0.5) / len(ranks)  # map to (0, 1)
                    # Convert percentile to z-score (number of SDs from mean in normal dist)
                    z_vals = np.abs(norm.ppf(percentiles))
                    # Add to running sum for valid trajectories
                    valid_indices = np.where(valid_mask)[0]
                    for i, z in zip(valid_indices, z_vals):
                        z_scores_sum[i] += z
                        z_scores_count[i] += 1
            # Compute average z-score (avoid division by zero)
            z_scores_avg = np.where(z_scores_count > 0, z_scores_sum / z_scores_count, np.inf)
            # Exclude trajectories that don't have points before 2021.0
            for i, p in enumerate(all_combined_paths):
                if p['times'][0] >= 2021.0:
                    z_scores_avg[i] = np.inf
            best_idx_zscore = np.nanargmin(z_scores_avg)
            best_path_zscore = all_combined_paths[int(best_idx_zscore)]
            ax.plot(best_path_zscore['times'], best_path_zscore['horizons'], color='blue', linewidth=2, linestyle='--', label='Central Trajectory (z-score)', zorder=46)

            # Add sample parameters to best_path for downstream use
            forecaster_name = best_path.get('forecaster_name')
            sample_idx = best_path.get('sample_idx')
            if forecaster_name is not None and sample_idx is not None and forecaster_name in all_forecaster_samples:
                samples = all_forecaster_samples[forecaster_name]
                best_path['h_SC'] = float(samples['h_SC'][sample_idx])
                best_path['T_t'] = float(samples['T_t'][sample_idx])
                best_path['cost_speed'] = float(samples['cost_speed'][sample_idx])
                best_path['announcement_delay'] = float(samples['announcement_delay'][sample_idx])
                best_path['present_prog_multiplier'] = float(samples['present_prog_multiplier'][sample_idx])
                best_path['SC_prog_multiplier'] = float(samples['SC_prog_multiplier'][sample_idx])
                best_path['is_exponential'] = bool(samples['is_exponential'][sample_idx])
                best_path['is_superexponential'] = bool(samples['is_superexponential'][sample_idx])
                best_path['is_subexponential'] = bool(samples['is_subexponential'][sample_idx])
                best_path['patch_rd_speedup'] = bool(samples['patch_rd_speedup'][sample_idx])
                best_path['software_progress_share'] = float(samples['software_progress_share'][sample_idx])
                best_path['se_doubling_decay_fraction'] = float(samples['se_doubling_decay_fraction']) if isinstance(samples['se_doubling_decay_fraction'], (int, float)) else float(samples['se_doubling_decay_fraction'][sample_idx])
                best_path['sub_doubling_growth_fraction'] = float(samples['sub_doubling_growth_fraction']) if isinstance(samples['sub_doubling_growth_fraction'], (int, float)) else float(samples['sub_doubling_growth_fraction'][sample_idx])

            # Optional annotated checkpoints
            if add_agent_checkpoints:
                checkpoints = [
                    (2025 + 7/12, 'Agent-0'),
                    (2026 + 2/12, 'Agent-1'),
                    (2027 + 6/12, 'Agent-3-mini'),
                ]
                t_arr = best_path['times']
                h_arr = best_path['horizons']
                for t_pt, label in checkpoints:
                    if t_pt < t_arr[0] or t_pt > t_arr[-1]:
                        continue
                    h_pt = np.interp(t_pt, t_arr, h_arr)
                    ax.scatter(t_pt, h_pt, color='green', s=40, zorder=49)
                    ax.annotate(label, (t_pt, h_pt), textcoords='offset points', xytext=(5, -5), ha='left', fontsize=config['plotting_style']['font']['sizes']['ticks'], color='black')

    # Grid and spines (drawn after trajectories & central path)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------------
    # Legend – keep ordering clear: reference lines first, then trajectories
    # ------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    # Ensure reference lines appear first
    ref_order = [label for label in labels if label in (
        "Current Horizon (15 min)",
        "Current Time",
        "External Benchmarks (p80)",
    )]
    traj_order = [label for label in labels if label not in ref_order]
    ordered_labels = ref_order + traj_order
    ordered_handles = [handles[labels.index(lab)] for lab in ordered_labels]
    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        fontsize=config["plotting_style"]["font"]["sizes"]["legend"],
        framealpha=0.5,
    )
    legend.set_zorder(50)

    # Final tick configuration
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig, best_path

# -----------------------------------------------------------------------------
# New helper: compare multiple central trajectories on a single figure
# -----------------------------------------------------------------------------

def plot_central_trajectories_comparison(
    central_trajectories: list[tuple[str, dict]],
    config: dict,
    *,
    overlay_external_data: bool = True,
    title: str = "Time Horizon Extension Central Trajectories Comparison",
    median_trajectories: list[tuple[str, dict, str]] | None = None,
    sc_month_str: str | None = None,
    show_params_in_legend: bool = False,
) -> plt.Figure:
    """Plot a set of *central trajectories* on a shared figure.

    Parameters
    ----------
    central_trajectories : list[tuple[str, dict]]
        A list where each element is ``(label, traj)`` with ``label`` a string
        (e.g. "March 2027") and ``traj`` a dict holding ``"times"`` and
        ``"horizons"`` arrays as returned by
        :pyfunc:`plot_combined_trajectories_sc_month`.
    config : dict
        Global configuration (same structure used by other plotting helpers).
    overlay_external_data : bool, default True
        Whether to scatter external METR benchmark p80 data points.
    median_trajectories : list[tuple[str, dict, str]] | None, default None
        Optional list of median parameter trajectories. Each element is
        ``(label, traj, growth_type)`` where ``growth_type`` is "exponential"
        or "superexponential". These are plotted with dashed lines.
    sc_month_str : str | None, default None
        Optional SC arrival month string (e.g. "January 2027"). If provided,
        a vertical dotted line is drawn at the end of that month.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.  The caller is responsible for saving/closing it.
    """
    # ------------------------------------------------------------------
    # Setup figure aesthetics identical to other helpers
    # ------------------------------------------------------------------
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    # ------------------------------------------------------------------
    # Determine overall x-range and y-range across all trajectories
    # ------------------------------------------------------------------
    if not central_trajectories:
        raise ValueError("central_trajectories list is empty – nothing to plot.")

    all_times = np.concatenate([np.asarray(traj["times"]) for _, traj in central_trajectories])
    _ = np.concatenate([np.asarray(traj["horizons"]) for _, traj in central_trajectories])  # unused, kept for future calc

    x_min = all_times.min()
    x_max = all_times.max()

    # Always include current time reference (assumed identical to other helpers)
    current_year = 2025.25
    x_min = min(x_min, current_year - 1)
    x_max = max(x_max, current_year + 1)

    # ------------------------------------------------------------------
    # Determine ordering by SC month (earlier = faster) and assign colours
    # using a continuous colormap so the reader can perceive the ordering.
    # Non-month labels (like "Unconditional") are placed at the end.
    # ------------------------------------------------------------------
    parsed_items = []
    non_month_items = []
    for lbl, traj in central_trajectories:
        try:
            month_mid = _parse_month_year(lbl)[2]
            parsed_items.append((month_mid, lbl, traj))
        except Exception:
            # Non-month labels (e.g., "Unconditional") go at the end
            non_month_items.append((float('inf'), lbl, traj))
    parsed_items.sort(key=lambda x: x[0])  # earliest first
    parsed_items.extend(non_month_items)

    n_items = len(parsed_items)
    cmap = plt.get_cmap("plasma")  # perceptually ordered colormap
    colours = cmap(np.linspace(0, 1, n_items))

    for idx, (_, label, traj) in enumerate(parsed_items):
        times = np.asarray(traj["times"])
        horizons = np.asarray(traj["horizons"])
        color = colours[idx]
        # Include parameters in label only if show_params_in_legend is True
        display_label = label
        if show_params_in_legend and 'cost_speed' in traj and 'announcement_delay' in traj:
            cs = traj['cost_speed']
            ad = traj['announcement_delay']
            display_label = f"{label} (cs={cs:.1f}mo, ad={ad:.1f}mo)"
        ax.plot(times, horizons, label=display_label, linewidth=1.5, color=color, alpha=0.6)

    # ------------------------------------------------------------------
    # Plot median parameter trajectories if provided
    # ------------------------------------------------------------------
    if median_trajectories:
        # Use distinct colors for growth types
        growth_colors = {
            "exponential": "#2E8B57",      # SeaGreen
            "superexponential": "#FF6347",  # Tomato
        }
        for label, traj, growth_type in median_trajectories:
            if traj is not None:
                times = np.asarray(traj["times"])
                horizons = np.asarray(traj["horizons"])
                color = growth_colors.get(growth_type, "gray")
                ax.plot(times, horizons, label=f"{label} (median params)",
                        linewidth=2.5, color=color, alpha=0.9, linestyle='--', zorder=20)

    # ------------------------------------------------------------------
    # Reference lines – keep in sync with other combined-plot helpers
    # ------------------------------------------------------------------
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Current Horizon (15 min)')
    ax.axvline(x=current_year, color='blue', linestyle='--', linewidth=2,
               alpha=0.7, label='Current Time')

    # SC arrival month vertical line (end of month)
    if sc_month_str is not None:
        try:
            _, end_decimal, _ = _parse_month_year(sc_month_str)
            ax.axvline(x=end_decimal, color='green', linestyle=':', linewidth=2,
                       alpha=0.7, label=f'SC Arrival (end of {sc_month_str})')
        except ValueError:
            pass  # Invalid month string, skip the line

    # ------------------------------------------------------------------
    # Overlay external METR data if requested
    # ------------------------------------------------------------------
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible = external_df[mask]
            if not visible.empty:
                ax.scatter(visible['release_year_decimal'], visible['p80'], color='black', s=25,
                           alpha=0.8, zorder=15, marker='o', label='External Benchmarks (p80)')

    # ------------------------------------------------------------------
    # Axes formatting (titles, labels, log-scale, ticks)
    # ------------------------------------------------------------------
    ax.set_title(title, fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')

    # Generate human-readable y-tick labels identical to other helpers
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    min_time_minutes = 0.1 / 60  # ≈0.00167 minutes (0.1 s)
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)

    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
    ]
    valid_ticks = [(pos, lab) for pos, lab in all_ticks if y_min <= pos <= y_max]
    if valid_ticks:
        pos, lab = zip(*valid_ticks)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)

    # Grid & spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------------
    # Legend – keep ordering clear: reference lines first, then trajectories
    # ------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    # Ensure reference lines appear first (including SC Arrival lines)
    ref_keywords = ("Current Horizon (15 min)", "Current Time", "External Benchmarks (p80)", "SC Arrival")
    ref_order = [label for label in labels if any(label.startswith(kw) or label == kw for kw in ref_keywords)]
    traj_order = [label for label in labels if label not in ref_order]
    ordered_labels = ref_order + traj_order
    ordered_handles = [handles[labels.index(lab)] for lab in ordered_labels]
    ax.legend(ordered_handles, ordered_labels,
              fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)

    # Rotate x-tick labels for readability
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig


def plot_median_vs_central_comparison(
    median_trajectory_data: dict,
    monthly_central_trajectories_by_forecaster: dict,
    config: dict,
    *,
    forecaster_name: str = "Eli",
    superexp_forecaster_name: str | None = None,
    exp_forecaster_name: str | None = None,
    mixed_forecaster_name: str | None = None,
    superexp_sc_month: str = "November 2026",
    exp_sc_month: str = "July 2028",
    mixed_sc_month: str | None = "August 2027",
    overlay_external_data: bool = True,
    show_params_in_legend: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """Plot comparison of median parameter trajectories vs central trajectories.

    This creates a plot comparing:
    - Median parameter trajectories (superexponential and exponential) for a forecaster
    - Central trajectories filtered by SC arrival month

    Parameters
    ----------
    median_trajectory_data : dict
        Dict mapping forecaster_name -> {growth_type: trajectory_dict}
        where trajectory_dict has 'times' and 'horizons' arrays.
    monthly_central_trajectories_by_forecaster : dict
        Dict mapping forecaster_name -> {sc_month_str: trajectory_dict}
    config : dict
        Global configuration.
    forecaster_name : str
        Base forecaster name for display in title (e.g., "Eli").
    superexp_forecaster_name : str | None
        Explicit name for the superexponential forecaster. If None, defaults to
        "{forecaster_name}_superexp_only".
    exp_forecaster_name : str | None
        Explicit name for the exponential forecaster. If None, defaults to
        "{forecaster_name}_exp_only".
    mixed_forecaster_name : str | None
        Explicit name for the mixed growth type forecaster. If None, defaults to
        forecaster_name.
    superexp_sc_month : str
        SC arrival month for superexponential central trajectory (e.g., "November 2026").
    exp_sc_month : str
        SC arrival month for exponential central trajectory (e.g., "July 2028").
    mixed_sc_month : str | None
        SC arrival month for mixed growth central trajectory (e.g., "August 2027").
    overlay_external_data : bool
        Whether to overlay external benchmark data points.
    title : str | None
        Custom title for the plot. If None, uses default title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.lines import Line2D

    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(14, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    # Define colors
    superexp_color = "#FF6347"  # Tomato red
    exp_color = "#2E8B57"       # Sea green
    mixed_color = "#4169E1"     # Royal blue

    current_year = 2025.25
    x_min = current_year - 5
    x_max = current_year + 5

    legend_elements = []

    # --- Determine forecaster names ---
    # Use explicit names if provided, otherwise fall back to default pattern
    superexp_forecaster = superexp_forecaster_name if superexp_forecaster_name else f"{forecaster_name}_superexp_only"
    exp_forecaster = exp_forecaster_name if exp_forecaster_name else f"{forecaster_name}_exp_only"
    mixed_forecaster = mixed_forecaster_name if mixed_forecaster_name else forecaster_name

    # Superexponential median trajectory
    if superexp_forecaster in median_trajectory_data:
        traj_data = median_trajectory_data[superexp_forecaster]
        if "superexponential" in traj_data:
            traj = traj_data["superexponential"]
            ax.plot(traj["times"], traj["horizons"], color=superexp_color, linewidth=2.5,
                    linestyle='-', alpha=0.9, zorder=20, label="Superexponential (median params)")
            legend_elements.append(
                Line2D([0], [0], color=superexp_color, linewidth=2.5, linestyle='-',
                       label="Superexponential (median params)")
            )

    # Exponential median trajectory
    if exp_forecaster in median_trajectory_data:
        traj_data = median_trajectory_data[exp_forecaster]
        if "exponential" in traj_data:
            traj = traj_data["exponential"]
            ax.plot(traj["times"], traj["horizons"], color=exp_color, linewidth=2.5,
                    linestyle='-', alpha=0.9, zorder=20, label="Exponential (median params)")
            legend_elements.append(
                Line2D([0], [0], color=exp_color, linewidth=2.5, linestyle='-',
                       label="Exponential (median params)")
            )

    # --- Plot central trajectories ---
    # Superexponential central trajectory (filtered to superexp_sc_month)
    if superexp_forecaster in monthly_central_trajectories_by_forecaster:
        central_by_month = monthly_central_trajectories_by_forecaster[superexp_forecaster]
        if superexp_sc_month in central_by_month:
            traj = central_by_month[superexp_sc_month]
            # Parse month for abbreviated format (e.g., "November 2026" -> "Nov 2026")
            superexp_month_abbrev = superexp_sc_month.replace("November", "Nov").replace("January", "Jan").replace("February", "Feb").replace("March", "Mar").replace("April", "Apr").replace("May", "May").replace("June", "Jun").replace("July", "Jul").replace("August", "Aug").replace("September", "Sep").replace("October", "Oct").replace("December", "Dec")
            label = f"Superexponential Central\n(filtered for SC in {superexp_month_abbrev})"
            if show_params_in_legend and 'cost_speed' in traj and 'announcement_delay' in traj:
                label += f" (cs={traj['cost_speed']:.1f}mo, ad={traj['announcement_delay']:.1f}mo)"
            ax.plot(traj["times"], traj["horizons"], color=superexp_color, linewidth=2,
                    linestyle='--', alpha=0.8, zorder=18)
            legend_elements.append(
                Line2D([0], [0], color=superexp_color, linewidth=2, linestyle='--',
                       label=label)
            )

    # Exponential central trajectory (filtered to exp_sc_month)
    if exp_forecaster in monthly_central_trajectories_by_forecaster:
        central_by_month = monthly_central_trajectories_by_forecaster[exp_forecaster]
        if exp_sc_month in central_by_month:
            traj = central_by_month[exp_sc_month]
            # Parse month for abbreviated format
            exp_month_abbrev = exp_sc_month.replace("November", "Nov").replace("January", "Jan").replace("February", "Feb").replace("March", "Mar").replace("April", "Apr").replace("May", "May").replace("June", "Jun").replace("July", "Jul").replace("August", "Aug").replace("September", "Sep").replace("October", "Oct").replace("December", "Dec")
            label = f"Exponential Central\n(filtered for SC in {exp_month_abbrev})"
            if show_params_in_legend and 'cost_speed' in traj and 'announcement_delay' in traj:
                label += f" (cs={traj['cost_speed']:.1f}mo, ad={traj['announcement_delay']:.1f}mo)"
            ax.plot(traj["times"], traj["horizons"], color=exp_color, linewidth=2,
                    linestyle='--', alpha=0.8, zorder=18)
            legend_elements.append(
                Line2D([0], [0], color=exp_color, linewidth=2, linestyle='--',
                       label=label)
            )

    # Mixed growth type central trajectory (filtered to mixed_sc_month)
    # This uses the base forecaster (e.g., "Eli") which has probability split among growth types
    if mixed_sc_month is not None and mixed_forecaster in monthly_central_trajectories_by_forecaster:
        central_by_month = monthly_central_trajectories_by_forecaster[mixed_forecaster]
        if mixed_sc_month in central_by_month:
            traj = central_by_month[mixed_sc_month]
            # Parse month for abbreviated format
            mixed_month_abbrev = mixed_sc_month.replace("November", "Nov").replace("January", "Jan").replace("February", "Feb").replace("March", "Mar").replace("April", "Apr").replace("May", "May").replace("June", "Jun").replace("July", "Jul").replace("August", "Aug").replace("September", "Sep").replace("October", "Oct").replace("December", "Dec")
            label = f"Central (mixed growth types)\n(filtered for SC in {mixed_month_abbrev})"
            if show_params_in_legend and 'cost_speed' in traj and 'announcement_delay' in traj:
                label += f" (cs={traj['cost_speed']:.1f}mo, ad={traj['announcement_delay']:.1f}mo)"
            ax.plot(traj["times"], traj["horizons"], color=mixed_color, linewidth=2,
                    linestyle='--', alpha=0.8, zorder=18)
            legend_elements.append(
                Line2D([0], [0], color=mixed_color, linewidth=2, linestyle='--',
                       label=label)
            )

    # --- Reference lines ---
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='gray', linestyle=':', alpha=0.8,
               linewidth=2, label='Anchor Horizon (15 mins)')
    ax.axvline(x=current_year, color='gray', linestyle=':', alpha=0.7,
               linewidth=2, label='Anchor Time (Apr 2025)')

    legend_elements.insert(0, Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Anchor Horizon (15 mins)'))
    legend_elements.insert(1, Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Anchor Time (Apr 2025)'))

    # --- Overlay external data ---
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible = external_df[mask]
            if not visible.empty:
                ax.scatter(visible['release_year_decimal'], visible['p80'], color='black', s=25,
                           alpha=0.8, zorder=15, marker='o')
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                           markersize=5, label='METR 80% time horizons', linestyle='None')
                )

    # --- Axes formatting ---
    if title is None:
        title = f"Trajectory with Median Parameters vs Central Trajectories ({forecaster_name}'s distributions)"
    ax.set_title(title, fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')

    # Y-axis ticks
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    min_time_minutes = 0.1 / 60
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)

    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
    ]
    valid_ticks = [(pos, lab) for pos, lab in all_ticks if y_min <= pos <= y_max]
    if valid_ticks:
        pos, lab = zip(*valid_ticks)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)

    # Grid & spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"],
              framealpha=0.5, loc='upper left')

    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig


def plot_median_trajectories_simple(
    median_trajectory_data: dict,
    config: dict,
    *,
    growth_type: str = "exponential",
    overlay_external_data: bool = True,
) -> plt.Figure:
    """Plot median parameter trajectories for all forecasters on a single simple graph.

    Parameters
    ----------
    median_trajectory_data : dict
        Dict mapping forecaster_name -> {growth_type: trajectory_dict}
        where trajectory_dict has 'times' and 'horizons' arrays.
    config : dict
        Global configuration.
    growth_type : str
        Which growth type to plot ("exponential" or "superexponential").
    overlay_external_data : bool
        Whether to overlay external benchmark data points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.lines import Line2D

    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family

    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)

    current_year = 2025.25
    x_min = current_year - 5
    x_max = current_year + 5

    legend_elements = []

    # Get a colormap for distinct colors
    n_forecasters = len(median_trajectory_data)
    cmap = plt.get_cmap("tab10")

    for idx, (forecaster_name, traj_data) in enumerate(median_trajectory_data.items()):
        if growth_type in traj_data:
            traj = traj_data[growth_type]
            color = cmap(idx % 10)
            ax.plot(traj["times"], traj["horizons"], color=color, linewidth=2.5,
                    linestyle='-', alpha=0.9, zorder=20)
            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=2.5, linestyle='-',
                       label=forecaster_name)
            )

    # Reference lines
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='gray', linestyle=':', alpha=0.8,
               linewidth=2, label='Current Horizon (15 min)')
    ax.axvline(x=current_year, color='gray', linestyle=':', alpha=0.7,
               linewidth=2, label='Current Time')

    legend_elements.insert(0, Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Current Horizon (15 min)'))
    legend_elements.insert(1, Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Current Time'))

    # Overlay external data
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible = external_df[mask]
            if not visible.empty:
                ax.scatter(visible['release_year_decimal'], visible['p80'], color='black', s=25,
                           alpha=0.8, zorder=15, marker='o')
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                           markersize=5, label='METR 80% time horizons', linestyle='None')
                )

    # Axes formatting
    growth_label = growth_type.capitalize()
    ax.set_title(f"Median Parameter Trajectories ({growth_label} Growth)",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_ylabel("80% Coding Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')

    # Y-axis ticks
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    min_time_minutes = 0.1 / 60
    if y_min < min_time_minutes:
        y_min = min_time_minutes
        ax.set_ylim(y_min, y_max)

    all_ticks = [
        (5/60, "5s"),
        (0.5, "30s"),
        (3, "3 mins"),
        (15, "15 mins"),
        (60, "1 hour"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (120240, "1 work year"),
        (601200, "5 work years"),
    ]
    valid_ticks = [(pos, lab) for pos, lab in all_ticks if y_min <= pos <= y_max]
    if valid_ticks:
        pos, lab = zip(*valid_ticks)
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)

    # Grid & spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"],
              framealpha=0.5, loc='upper left')

    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    add_disclaimer(fig)
    return fig