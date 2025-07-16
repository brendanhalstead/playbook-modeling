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

def load_external_data(csv_path: str = "../external/headline.csv") -> pd.DataFrame:
    """Load external benchmark data points for overlay on plots."""
    if not Path(csv_path).exists():
        print(f"Warning: External data file {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Parse release_date to decimal year format (matching our timeline plots)
    # Handle missing dates (like for 'human' row)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year_decimal'] = df['release_year'].dt.year + (df['release_year'].dt.month - 1) / 12 + (df['release_year'].dt.day - 1) / 365
    
    # Remove rows without valid dates
    df = df.dropna(subset=['release_year_decimal'])
    
    print(f"Loaded {len(df)} external data points from {csv_path}")
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
        assert len(valid_results) > 0
        assert len(valid_results) > 1
        if len(set(valid_results)) == 1:
            print(valid_results)
            assert False
        
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

def plot_march_2027_trajectories(all_forecaster_results: dict, all_forecaster_trajectories: dict, all_forecaster_samples: dict, config: dict, forecaster_filter: list[str] = None) -> plt.Figure:
    """Create plot showing time horizon trajectories for runs that reach SC in March 2027."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Define March 2027 as decimal year (2027.167)
    march_2027 = 2027 + (3 - 1) / 12  # March is month 3
    tolerance = 0.1  # Within 1.2 months after March 2027
    
    # Get current year for x-axis range
    # current_year = datetime.now().year
    current_year = 2025.25
    x_min = current_year
    x_max = 2028
    
    total_trajectories_plotted = 0
    all_final_horizons = []  # Collect final horizon times across all forecasters (March 2027 only)
    all_final_horizons_all_runs = []  # Collect final horizon times for ALL runs
    all_h_sc_samples = []  # Collect h_SC samples across all forecasters
    
    # Plot trajectories for each forecaster
    for name, results in all_forecaster_results.items():
        if forecaster_filter and name not in forecaster_filter:
            continue
        
        trajectories = all_forecaster_trajectories[name]
        
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Find runs that reach SC in March 2027 (within tolerance)
        march_2027_runs = []
        final_horizons_this_forecaster = []
        final_horizons_all_runs_this_forecaster = []
        
        # First, collect ALL final horizon times for this forecaster
        for i, trajectory in enumerate(trajectories):
            if trajectory:  # Make sure trajectory is not empty
                final_horizon_minutes = trajectory[-1][1]  # Last horizon value
                final_horizons_all_runs_this_forecaster.append(final_horizon_minutes)
                all_final_horizons_all_runs.append(final_horizon_minutes)
        
        # Then find March 2027 runs specifically
        for i, end_time in enumerate(results):
            if end_time - march_2027 > 0 and end_time - march_2027 <= tolerance:
                march_2027_runs.append(i)
                # Get the final horizon time from the trajectory
                trajectory = trajectories[i]
                if trajectory:  # Make sure trajectory is not empty
                    final_horizon_minutes = trajectory[-1][1]  # Last horizon value
                    final_horizons_this_forecaster.append(final_horizon_minutes)
                    all_final_horizons.append(final_horizon_minutes)
        
        print(f"{name}: Found {len(march_2027_runs)} runs reaching SC in March 2027")
        
        # Print h_SC distribution for this forecaster
        h_sc_samples = all_forecaster_samples[name]["h_SC"]
        all_h_sc_samples.extend(h_sc_samples)  # Collect for overall distribution
        h_sc_work_months = h_sc_samples  # Convert from minutes to work months
        print(f"  h_SC target distribution (work months):")
        print(f"    10th percentile: {np.percentile(h_sc_work_months, 10):,.2f}")
        print(f"    50th percentile: {np.percentile(h_sc_work_months, 50):,.2f}")
        print(f"    90th percentile: {np.percentile(h_sc_work_months, 90):,.2f}")
        print(f"    Mean: {np.mean(h_sc_work_months):,.2f}")
        
        # Print horizon distribution for ALL runs for this forecaster
        if final_horizons_all_runs_this_forecaster:
            horizons_all_array = np.array(final_horizons_all_runs_this_forecaster)
            print(f"  ALL RUNS final horizon distribution (work months):")
            horizons_all_work_months = horizons_all_array / (60 * 167)
            print(f"    10th percentile: {np.percentile(horizons_all_work_months, 10):,.2f}")
            print(f"    50th percentile: {np.percentile(horizons_all_work_months, 50):,.2f}")
            print(f"    90th percentile: {np.percentile(horizons_all_work_months, 90):,.2f}")
            print(f"    Mean: {np.mean(horizons_all_work_months):,.2f}")
        
        # Print horizon distribution for March 2027 runs for this forecaster
        if final_horizons_this_forecaster:
            horizons_array = np.array(final_horizons_this_forecaster)
            # Convert to work months (167 hours per work month)
            horizons_work_months = horizons_array / (60 * 167)
            print(f"  March 2027 final horizon distribution (work months):")
            print(f"    10th percentile: {np.percentile(horizons_work_months, 10):,.2f}")
            print(f"    50th percentile: {np.percentile(horizons_work_months, 50):,.2f}")
            print(f"    90th percentile: {np.percentile(horizons_work_months, 90):,.2f}")
            print(f"    Mean: {np.mean(horizons_work_months):,.2f}")
            print()
        
        # Plot trajectories for these runs
        for run_idx in march_2027_runs:
            trajectory = trajectories[run_idx]
            if trajectory:  # Make sure trajectory is not empty
                times, horizons = zip(*trajectory)
                ax.plot(times, horizons, '-', color=color, alpha=0.3, linewidth=1)
                total_trajectories_plotted += 1
    
    print(f"\nTotal trajectories plotted: {total_trajectories_plotted}")
    
    # Print overall h_SC distribution across all forecasters
    if all_h_sc_samples:
        print(f"\nOVERALL h_SC Target Distribution:")
        print(f"Total samples: {len(all_h_sc_samples)}")
        all_h_sc_array = np.array(all_h_sc_samples)
        all_h_sc_work_months = all_h_sc_array
        print(f"h_SC target distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_h_sc_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_h_sc_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_h_sc_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_h_sc_work_months):,.2f}")
        print()
    
    # Print overall horizon distribution for ALL runs across all forecasters
    if all_final_horizons_all_runs:
        print(f"OVERALL Final Horizon Distribution for ALL RUNS:")
        print(f"Total samples: {len(all_final_horizons_all_runs)}")
        all_horizons_all_array = np.array(all_final_horizons_all_runs)
        
        # Convert to work months (167 hours per work month)
        all_horizons_all_work_months = all_horizons_all_array / (60 * 167)
        print(f"Final horizon distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_horizons_all_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_horizons_all_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_horizons_all_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_horizons_all_work_months):,.2f}")
        print()
    
    # Print overall horizon distribution across all forecasters for March 2027 runs
    if all_final_horizons:
        print(f"OVERALL Final Horizon Distribution for March 2027 SC arrivals:")
        print(f"Total samples: {len(all_final_horizons)}")
        all_horizons_array = np.array(all_final_horizons)
        
        # Convert to work months (167 hours per work month)
        all_horizons_work_months = all_horizons_array / (60 * 167)
        print(f"Final horizon distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_horizons_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_horizons_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_horizons_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_horizons_work_months):,.2f}")
        print()
    
    # Add reference trajectory line with specific points
    reference_times = [2025.25, 2026, 2026.5, 2027]
    reference_horizons = [
        15,  # 15 minutes
        240,  # 4 work hours = 4 * 60 minutes
        4800,  # 2 work weeks = 2 * 40 hours * 60 minutes
        320640  # 32 work months = 32 * 167 hours * 60 minutes
    ]
    ax.plot(reference_times, reference_horizons, 'o-', color='purple', 
            linewidth=3, markersize=6, alpha=0.8, 
            label='Reference Timeline', zorder=10)
    
    # Add horizontal line for SC threshold (assuming it's the target horizon)
    # We'll use the current horizon as a reference point
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='--', alpha=0.7, 
               label='Current Horizon (15 min)', linewidth=2)
    
    # Add vertical line for March 2027
    ax.axvline(x=march_2027, color='blue', linestyle='--', alpha=0.7, 
               label='March 2027', linewidth=2)
    
    # Configure plot
    ax.set_title("Time Horizon Trajectories for Runs Reaching SC in March 2027",
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
    # Use a more intelligent spacing to avoid crowding
    tick_positions = []
    tick_labels = []
    
    # Define all possible tick positions with their labels
    all_ticks = [
        # Short times
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
        (60120, "6 work months"), # 6 * 167 hours
        (120240, "1 work year"),  # 12 * 167 hours
        (240480, "2 work years"), # 24 * 167 hours
        (601200, "5 work years"), # 60 * 167 hours
        (1202400, "10 work years"), # 120 * 167 hours
        (2404800, "20 work years"), # 240 * 167 hours
        (4809600, "40 work years"), # 480 * 167 hours
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
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
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
                          color='black', s=50, alpha=0.8, zorder=15, marker='o',
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
                           markersize=8, label='External Benchmarks (p80)', linestyle='None')
                )
      
    legend = ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    return fig

def plot_combined_trajectories(all_forecaster_backcast_trajectories: dict, all_forecaster_trajectories: dict, all_forecaster_samples: dict, config: dict, color_by_growth_type: bool = True, overlay_external_data: bool = True, forecaster_filter: list[str] = None) -> plt.Figure:
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
    
    # Plot trajectories for each forecaster
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
        max_trajectories_to_plot = 100  # Fewer trajectories for combined plot
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
            backcast_traj = backcast_trajectories[i]
            forecast_traj = forecast_trajectories[i]
            growth_type = growth_types[i]
            
            if color_by_growth_type:
                color = growth_colors[growth_type]
            else:
                color = forecaster_color
            
            # Plot backcast trajectory (past)
            if backcast_traj:
                back_times, back_horizons = zip(*backcast_traj)
                ax.plot(back_times, back_horizons, '-', color=color, alpha=0.15, linewidth=0.8)
            
            # Plot forecast trajectory (future)
            if forecast_traj:
                fore_times, fore_horizons = zip(*forecast_traj)
                ax.plot(fore_times, fore_horizons, '-', color=color, alpha=0.15, linewidth=0.8)
            
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
                          color='black', s=50, alpha=0.8, zorder=15, marker='o',
                          label='External Benchmarks (p80)')
                print(f"Overlaid {len(visible_data)} external benchmark points")
    
    # Configure plot
    ax.set_title("Complete Time Horizon Trajectories\n(Historical development and future projections)",
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
    ]
    
    # Filter ticks to be within range
    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    
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
    
    # Add legend
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
            Line2D([0], [0], color='gray', linewidth=2, label='Combined Trajectories')
        )
    
    # Add external data to legend if present
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            if external_df[time_mask].shape[0] > 0:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                           markersize=8, label='External Benchmarks (p80)', linestyle='None')
                )
    

    legend = ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    return fig

def plot_combined_trajectories_march_2027(all_forecaster_backcast_trajectories: dict, all_forecaster_trajectories: dict, all_forecaster_samples: dict, all_forecaster_results: dict, config: dict, color_by_growth_type: bool = True, overlay_external_data: bool = True, plot_central_trajectory: bool = True, plot_median_curve: bool = False, overlay_illustrative_trend: bool = False, add_agent_checkpoints: bool = False, forecaster_filter: list[str] = None) -> plt.Figure:
    """Create plot showing both backcasted and forecasted trajectories for March 2027 SC arrivals only."""
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
    forecast_years = 3  # Shorter forecast range for March 2027 focus
    x_min = current_year - backcast_years
    x_max = current_year + forecast_years  # Ends in 2028
    
    # Define March 2027 window (2027.167 to 2027.25)
    march_2027_start = 2027 + 2/12  # March 1st
    march_2027_end = 2027 + 3/12    # April 1st
    
    total_trajectories_plotted = 0
    
    # Collect combined (backcast + forecast) trajectories for later median-selection
    all_combined_paths = []
    
    assert len(all_forecaster_backcast_trajectories) == len(all_forecaster_trajectories) == len(all_forecaster_samples) == len(all_forecaster_results), "All dictionaries must have the same number of keys"

    # Plot trajectories for each forecaster
    for name in all_forecaster_backcast_trajectories.keys():
        if forecaster_filter and name not in forecaster_filter:
            continue
        
        backcast_trajectories = all_forecaster_backcast_trajectories[name]
        forecast_trajectories = all_forecaster_trajectories[name]
        samples = all_forecaster_samples[name]
        results = all_forecaster_results[name]
        
        # these should all have length n_sims, but samples has length equal to number of sampled parameters
        assert len(backcast_trajectories) == len(forecast_trajectories) == len(results), "All lists must have the same length"
        
        
        # Filter for March 2027 arrivals
        march_2027_mask = (results >= march_2027_start) & (results < march_2027_end)
        march_2027_indices = np.where(march_2027_mask)[0]
        
        if len(march_2027_indices) == 0:
            print(f"{name}: No trajectories reaching SC in March 2027")
            continue
        
        print(f"{name}: Found {len(march_2027_indices)} trajectories reaching SC in March 2027")
        
        # Get growth types for color coding
        growth_types = []
        for i in march_2027_indices:
            if samples["is_exponential"][i]:
                growth_types.append("exponential")
            elif samples["is_superexponential"][i]:
                growth_types.append("superexponential")
            else:  # is_subexponential
                growth_types.append("subexponential")
        
        # Plot all March 2027 trajectories (no additional limit since they're already filtered)
        max_trajectories_to_plot = min(500, len(march_2027_indices))  # Cap at 500 for more detail
        selected_indices = march_2027_indices[:max_trajectories_to_plot]
        selected_growth_types = growth_types[:max_trajectories_to_plot]
        
        print(f"{name}: Plotting {len(selected_indices)} March 2027 combined trajectories")
        
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
        for idx, (traj_idx, growth_type) in enumerate(zip(selected_indices, selected_growth_types)):
            backcast_traj = backcast_trajectories[traj_idx]
            forecast_traj = forecast_trajectories[traj_idx]
            
            if color_by_growth_type:
                color = growth_colors[growth_type]
            else:
                color = forecaster_color
            
            # Plot backcast trajectory (past)
            if backcast_traj:
                back_times, back_horizons = zip(*backcast_traj)
                ax.plot(back_times, back_horizons, '-', color=color, alpha=0.2, linewidth=0.8)
            
            # Plot forecast trajectory (future)
            if forecast_traj:
                fore_times, fore_horizons = zip(*forecast_traj)
                ax.plot(fore_times, fore_horizons, '-', color=color, alpha=0.2, linewidth=0.8)
            
            total_trajectories_plotted += 1
        
            # Collect combined (backcast + forecast) trajectories for later median-selection
            combined_times = []
            combined_horizons = []
            if backcast_traj:
                combined_times.extend(back_times)
                combined_horizons.extend(back_horizons)
            if forecast_traj:
                combined_times.extend(fore_times)
                combined_horizons.extend(fore_horizons)
            if combined_times:
                order = np.argsort(combined_times)
                all_combined_paths.append({
                    'times': np.array(combined_times)[order],
                    'horizons': np.array(combined_horizons)[order]
                })
    
    print(f"\nTotal March 2027 combined trajectories plotted: {total_trajectories_plotted}")
    
    # Add current horizon line (where trajectories converge)
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='gray', linestyle=':', alpha=0.8, 
               label='Current Horizon (15 min)', linewidth=3, zorder=10)
    
    # Add vertical line for current time
    ax.axvline(x=current_year, color='gray', linestyle=':', alpha=0.7, 
               label='Current Time', linewidth=2)
    
    # Add vertical line for March 2027
    march_2027_mid = (march_2027_start + march_2027_end) / 2
    ax.axvline(x=march_2027_mid, color='purple', linestyle=':', alpha=0.8, 
               label='March 2027 (SC Target)', linewidth=2)
    
    # Overlay external data points if requested
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            # Filter to points within the plot's time range
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            visible_data = external_df[time_mask]
            
            if not visible_data.empty:
                ax.scatter(visible_data['release_year_decimal'], visible_data['p80'], 
                          color='black', s=50, alpha=0.8, zorder=15, marker='o',
                          label='External Benchmarks (p80)')
                print(f"Overlaid {len(visible_data)} external benchmark points")
    
    # Configure plot
    ax.set_title("Complete Time Horizon Trajectories - March 2027 SC Arrivals\n(Historical development and future projections)",
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
        (60120, "1 work year"),        # 1 year of work
        (240480, "4 work years"),
    ]
    
    # Filter ticks to be within range
    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    
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
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle=':', linewidth=3, label='Current Horizon (15 min)'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Current Time'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='March 2027 (SC Arrival)'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Central AI 2027 Trajectory'),
    ]
    
    if color_by_growth_type:
        legend_elements.extend([
            Line2D([0], [0], color='#2E8B57', linewidth=2, label='Exponential Growth'),
            Line2D([0], [0], color='#FF6347', linewidth=2, label='Superexponential Growth'), 
            Line2D([0], [0], color='#4169E1', linewidth=2, label='Subexponential Growth')
        ])
    else:
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=2, label='March 2027 Trajectories')
        )
    
    # Add external data to legend if present
    if overlay_external_data:
        external_df = load_external_data()
        if not external_df.empty:
            time_mask = (external_df['release_year_decimal'] >= x_min) & (external_df['release_year_decimal'] <= x_max)
            if external_df[time_mask].shape[0] > 0:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                           markersize=8, label='METR 80% Time Horizon Data', linestyle='None')
                )
    
    if plot_median_curve:
        legend_elements.append(
            Line2D([0], [0], color='red', linewidth=2, linestyle=':', label='Median')
        )
    if overlay_illustrative_trend:
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=2, label='Previous Illustrative Trend')
        )
    
    legend = ax.legend(handles=legend_elements, fontsize=config["plotting_style"]["font"]["sizes"]["legend"], framealpha=0.5)
    legend.set_zorder(50)
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    # -------------------------------------------------------------
    # Identify and highlight the "median" trajectory (method 2B)
    # -------------------------------------------------------------
    if plot_central_trajectory:
        x_grid = np.arange(x_min, x_max + 1e-6, 1/12)  # monthly grid
        n_traj = len(all_combined_paths)
        n_x_grid = len(x_grid)
        matrix = np.full((n_traj, n_x_grid), np.nan)
        
        # print(f"length ofall_combined_paths: {len(all_combined_paths)}")
        for i, path in enumerate(all_combined_paths):
            t = path['times']
            h = path['horizons']
            if t.size == 0:
                assert False, "Empty trajectory"
            mask = (x_grid >= t[0]) & (x_grid <= t[-1])
            if mask.any():
                interp_vals = np.interp(x_grid[mask], t, h)
                # print(f"interp_vals shape: {interp_vals.shape}")
                # Work in log-space so differences reflect the plotted log scale
                interp_vals = np.log10(np.clip(interp_vals, 1e-6, None))
                matrix[i, mask] = interp_vals

        median_curve = np.nanmedian(matrix, axis=0)
        assert median_curve.size == n_x_grid, "Median curve has wrong length"
        # Plot the median curve
        valid_mask = ~np.isnan(median_curve)
        if valid_mask.any():
            # Convert back from log10 space to original scale
            median_horizons = 10**median_curve[valid_mask]
            if plot_median_curve:
                ax.plot(x_grid[valid_mask], median_horizons, 
                        color='red', linewidth=2, linestyle=':', 
                        label='Median Trajectory', zorder=49)
        distances = np.nansum(np.abs(matrix - median_curve), axis=1)
        # Guard against all-nan rows
        valid_mask = ~np.isnan(distances)
        best_idx = None
        if valid_mask.any():
            best_idx = int(np.nanargmin(np.where(valid_mask, distances, np.inf)))
            best_path = all_combined_paths[best_idx]
            ax.plot(best_path['times'], best_path['horizons'], color='green', linewidth=2,linestyle='--', label='Central Trajectory', zorder=48)
    


    # -----------------------------------------------------------------
    # Add labelled check-points on the median trajectory
    # -----------------------------------------------------------------
    if add_agent_checkpoints:
        checkpoints = [
            (2025 + 7/12, "Agent-0"),   # August 2025
            (2026 + 2/12, "Agent-1"),   # March 2026
            (2027 + 6/12, "Agent-3-mini")  # July 2027
        ]

        t_arr = best_path['times']
        h_arr = best_path['horizons']
        for t_point, label in checkpoints:
            if t_point < t_arr[0] or t_point > t_arr[-1]:
                continue  # outside trajectory range
            h_point = np.interp(t_point, t_arr, h_arr)
            ax.scatter(t_point, h_point, color='green', s=40, zorder=49)
            ax.annotate(label, (t_point, h_point), textcoords="offset points",
                        xytext=(5, -5), ha='left', fontsize=config["plotting_style"]["font"]["sizes"]["ticks"], color='black')
        
    # ---------------------------------------------------------------------
    # Overlay illustrative SE trend if requested
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
    return fig