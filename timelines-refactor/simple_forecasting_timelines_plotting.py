import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm, rankdata
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    
    fig = plt.figure(figsize=(10, 6), dpi=150, facecolor=bg_rgb)
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
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
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
    
    return fig

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
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

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
        (0.1, "6s"),
        (0.5, "30s"),
        (1, "1 min"),
        (5, "5 min"),
        (15, "15 min"),
        (30, "30 min"),
        (60, "1 hour"),
        (240, "4 hours"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (60120, "6 work months"),
        (120240, "1 work year"),
        (240480, "2 work years"),
        (601200, "5 work years"),
        (1202400, "10 work years"),
        (2404800, "20 work years"),
        (4809600, "40 work years"),
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
) -> tuple[plt.Figure, dict|None]:
    """Generalized version of `plot_combined_trajectories_march_2027`."""

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
                all_combined_paths.append({'times': np.array(combined_t)[order], 'horizons': np.array(combined_h)[order]})

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
    ax.set_title(f"Complete Time Horizon Extension Trajectories – {sc_month_str} SC Arrivals", fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

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
        ax.set_ylim(y_min, y_max)

    all_ticks = [
        (0.01, "0.6s"),
        (0.1, "6s"),
        (0.5, "30s"),
        (1, "1 min"),
        (5, "5 min"),
        (15, "15 min"),
        (30, "30 min"),
        (60, "1 hour"),
        (240, "4 hours"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (60120, "1 work year"),
        (240480, "4 work years"),
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
            distances = np.nansum(np.abs(matrix - median_curve), axis=1)
            best_idx = np.nanargmin(distances)
            best_path = all_combined_paths[int(best_idx)]
            ax.plot(best_path['times'], best_path['horizons'], color='green', linewidth=2, linestyle='--', label='Central Trajectory', zorder=48)

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
    # Overlay illustrative SE trend if requested (same CSV as original impl.)
    # ---------------------------------------------------------------------
    if overlay_illustrative_trend:
        trend_path = Path("../external/illustrative_se_trend_converted.csv")
        if trend_path.exists():
            try:
                trend_df = pd.read_csv(trend_path)
                mask = (trend_df['year'] >= x_min) & (trend_df['year'] <= x_max)
                if mask.any():
                    ax.plot(trend_df.loc[mask, 'year'], trend_df.loc[mask, 'horizon_minutes'],
                            color='black', linewidth=2, alpha=0.7)
            except Exception as e:
                print(f"Warning: failed to plot illustrative trend: {e}")

    # -------------------------------------------------------------
    # Assemble custom legend (mirrors original combined plot)
    # -------------------------------------------------------------
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle=':', linewidth=3, label='Current Horizon (15 min)'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Current Time'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label=f'{sc_month_str} (SC Arrival)'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Central Trajectory'),
    ]

    if color_by_growth_type:
        legend_elements.extend([
            Line2D([0], [0], color='#2E8B57', linewidth=2, label='Exponential Growth'),
            Line2D([0], [0], color='#FF6347', linewidth=2, label='Superexponential Growth'),
            Line2D([0], [0], color='#4169E1', linewidth=2, label='Subexponential Growth')
        ])
    else:
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, label='Combined Trajectories'))

    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            mask2 = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            if external_df[mask2].shape[0] > 0:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=4, label='External Benchmarks (p80)', linestyle='None'))

    if plot_median_curve:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Median'))
    if overlay_illustrative_trend:
        legend_elements.append(Line2D([0], [0], color='black', linewidth=2, label='Previous Illustrative Trend'))

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
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
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
) -> plt.Figure:
    """Create plot showing both backcasted and forecasted time horizon trajectories."""
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
                    'horizons': np.array(combined_h)[order]
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
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
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
        (60120, "1 year"),        # 1 year of work
        (240480, "4 work years"),
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
            distances = np.nansum(np.abs(matrix - median_curve), axis=1)
            best_idx = np.nanargmin(distances)
            best_path = all_combined_paths[int(best_idx)]
            ax.plot(best_path['times'], best_path['horizons'], color='green', linewidth=2, linestyle='--', label='Central Trajectory', zorder=48)

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

    return fig

# -----------------------------------------------------------------------------
# New helper: compare multiple central trajectories on a single figure
# -----------------------------------------------------------------------------

def plot_central_trajectories_comparison(
    central_trajectories: list[tuple[str, dict]],
    config: dict,
    *,
    overlay_external_data: bool = True,
    title: str = "Time Horizon Extension Central Trajectories Comparison",
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
    # ------------------------------------------------------------------
    try:
        parsed_items = [(_parse_month_year(lbl)[2], lbl, traj) for lbl, traj in central_trajectories]
        parsed_items.sort(key=lambda x: x[0])  # earliest first
    except Exception:
        # Fallback: use original order if parsing fails for any label
        parsed_items = [(idx, lbl, traj) for idx, (lbl, traj) in enumerate(central_trajectories)]

    n_items = len(parsed_items)
    cmap = plt.get_cmap("plasma")  # perceptually ordered colormap
    colours = cmap(np.linspace(0, 1, n_items))

    for idx, (_, label, traj) in enumerate(parsed_items):
        times = np.asarray(traj["times"])
        horizons = np.asarray(traj["horizons"])
        color = colours[idx]
        ax.plot(times, horizons, label=label, linewidth=1.5, color=color, alpha=0.6)

    # ------------------------------------------------------------------
    # Reference lines – keep in sync with other combined-plot helpers
    # ------------------------------------------------------------------
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Current Horizon (15 min)')
    ax.axvline(x=current_year, color='blue', linestyle='--', linewidth=2,
               alpha=0.7, label='Current Time')

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
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])

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
        (0.01, "0.6s"),
        (0.1, "6s"),
        (0.5, "30s"),
        (1, "1 min"),
        (5, "5 min"),
        (15, "15 min"),
        (30, "30 min"),
        (60, "1 hour"),
        (240, "4 hours"),
        (480, "1 work day"),
        (2400, "1 work week"),
        (10020, "1 work month"),
        (60120, "1 year"),
        (240480, "4 work years"),
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
    # Ensure reference lines appear first
    ref_order = [label for label in labels if label in ("Current Horizon (15 min)", "Current Time", "External Benchmarks (p80)")]
    traj_order = [label for label in labels if label not in ref_order]
    ordered_labels = ref_order + traj_order
    ordered_handles = [handles[labels.index(lab)] for lab in ordered_labels]
    ax.legend(ordered_handles, ordered_labels,
              fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)

    # Rotate x-tick labels for readability
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    return fig