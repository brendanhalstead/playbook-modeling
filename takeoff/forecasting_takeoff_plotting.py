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


def _year_to_date_str(year: float) -> str:
    """Convert a decimal year to a 'Month YYYY' string or '>2100'."""
    if year > 2100:
        return ">2100"
    year_int = int(year)
    month = int((year - year_int) * 12) + 1
    month = min(month, 12)  # Prevent month > 12 due to float precision
    month_name = datetime(year_int, month, 1).strftime('%b')
    return f"{month_name} {year_int}"


def setup_plotting_style(plotting_style: dict):
    """Set up matplotlib style according to config."""
    plt.style.use('default')  # Reset to default style
    
    # Set background color to cream (255, 250, 240)
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    plt.rcParams['figure.facecolor'] = bg_rgb
    plt.rcParams['axes.facecolor'] = bg_rgb
    plt.rcParams['savefig.facecolor'] = bg_rgb
    
    # Set font to monospace
    plt.rcParams['font.family'] = 'monospace'
    
    # Create font properties objects with sizes
    font_regular = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["title"])
    font_regular_small = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["axis_labels"])
    font_regular_legend = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["legend"])
    font_bold = fm.FontProperties(family='monospace', weight='bold', size=plotting_style["font"]["sizes"]["title"])
    font_medium = fm.FontProperties(family='monospace', weight='medium', size=plotting_style["font"]["sizes"]["title"])
    font_regular_xsmall = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["small"])
    
    # Return the font properties to be used in plotting functions
    return {
        'regular': font_regular,
        'regular_small': font_regular_small,
        'regular_legend': font_regular_legend,
        'bold': font_bold,
        'medium': font_medium,
        'small': font_regular_xsmall
    }


def create_milestone_timeline_plot(all_milestone_dates: list[list[datetime]], config: dict, plotting_style: dict, fonts: dict, title: str | None = None) -> plt.Figure:
    """Create timeline plot showing milestone achievement distributions."""
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    milestone_years = [[d.year + d.timetuple().tm_yday/365 for d in sim_dates] for sim_dates in all_milestone_dates]
    # Fix the indexing bug: milestone_dates contains [SC, SAR, SIAR, ASI], so we need to adjust indices
    milestones = ["SAR", "SIAR", "ASI"] #, "WS"]
    milestone_indices = [1, 2, 3]  # SAR=1, SIAR=2, ASI=3 (SC=0 is not plotted)
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Initialize stats text
    stats_text = ""
    
    # Plot distribution for each milestone
    for i, milestone in enumerate(milestones):
        MAX_GRAPH_YEAR = 2100
        start_year = float(config["starting_time"].split()[-1])

        # Use the correct index to get the right milestone data
        milestone_data = [years[milestone_indices[i]] for years in milestone_years if milestone_indices[i] < len(years)]
        
        # Calculate percentiles using full data
        p10 = np.percentile(milestone_data, 10)
        p50 = np.percentile(milestone_data, 50)
        p90 = np.percentile(milestone_data, 90)
        
        # Filter data to visible range for KDE
        visible_data = [x for x in milestone_data if start_year <= x <= MAX_GRAPH_YEAR]
        if not visible_data:
            print(f"Warning: No data in visible range for {milestone}")
            continue
        
        # Define colors for each milestone
        colors = ["#900000", "#004000", "#000090"]
        milestone_full = ["Superhuman\n  AI Researcher", "Superintelligent\n  AI Researcher", "Generally\n  Superintelligent"]
        
        # Try to calculate KDE, with fallback to histogram or vertical line if it fails
        try:
            # Check if data has sufficient variance for KDE
            data_variance = np.var(visible_data)
            unique_values = len(set(visible_data))
            
            if data_variance < 1e-10 or unique_values < 3:
                raise ValueError("Insufficient data variance for KDE")
            
            kde = gaussian_kde(visible_data)
            
            # Create x range for plotting
            x_range = np.linspace(start_year, MAX_GRAPH_YEAR, 200000)
            density = kde(x_range)
            
            # Normalize density to sum to 1 over visible range - fix division by zero
            density_sum = np.sum(density)
            if density_sum > 0:
                density = density / density_sum * (len(visible_data) / len(milestone_data))
            else:
                density = np.zeros_like(density)  # Fallback if density is all zeros
            
            # Plot with different colors for each milestone
            ax.plot(x_range, density, '-', color=colors[i], label=milestone,
                    linewidth=2, alpha=0.8, zorder=2)
            ax.fill_between(x_range, density, color=colors[i], alpha=0.1)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: KDE failed for {milestone} due to insufficient data variance. Using fallback visualization.")
            
            # Fallback 1: Try histogram if we have enough distinct values
            if len(set(visible_data)) > 1:
                # Create histogram
                data_min = min(visible_data)
                data_max = max(visible_data)
                if data_max > data_min:
                    bins = np.linspace(data_min, data_max, min(20, len(set(visible_data))))
                    hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as step function
                    ax.step(bin_centers, hist, where='mid', color=colors[i], 
                           label=milestone, linewidth=2, alpha=0.8, zorder=2)
                    ax.fill_between(bin_centers, hist, step='mid', color=colors[i], alpha=0.1)
                else:
                    # Fallback 2: All values are the same, plot a vertical line
                    ax.axvline(x=data_min, color=colors[i], linestyle='-', linewidth=3,
                             alpha=0.8, label=f"{milestone} (Consistent)", zorder=2)
            else:
                # Fallback 2: All values are the same, plot a vertical line  
                ax.axvline(x=visible_data[0], color=colors[i], linestyle='-', linewidth=3,
                         alpha=0.8, label=f"{milestone} (Consistent)", zorder=2)
        
        # Add statistics text using full data with month and year format
        stats = (
            f"{milestone}: {milestone_full[i]}\n"
            f"  10th: {_year_to_date_str(p10)}\n"
            f"  50th: {_year_to_date_str(p50)}\n"
            f"  90th: {_year_to_date_str(p90)}"
        )
        
        if i == 0:
            stats_text = stats
        else:
            stats_text += f"\n\n{stats}"
    
    # Add stats text with white background
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    plot_title = title if title else "AI Takeoff Forecast, Assuming Superhuman Coder in Mar 2027"
    title_obj = ax.set_title(plot_title,
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=10)
    title_obj.set_fontproperties(fonts['regular'])
    
    xlabel = ax.set_xlabel("Year",
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    xlabel.set_fontproperties(fonts['regular_small'])
    
    ylabel = ax.set_ylabel("Probability Density",
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ylabel.set_fontproperties(fonts['regular_small'])
    
    # Set axis properties
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both",
                   labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

def create_phase_duration_plot(all_milestone_dates: list[list[datetime]], config: dict, plotting_style: dict, fonts: dict, title: str | None = None) -> plt.Figure:
    """Create box plot showing the distribution of time spent in each phase."""
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    phase_durations = []
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    
    for sim_dates in all_milestone_dates:
        # sim_dates should contain [SC_date, SAR_date, SIAR_date, ASI_date]
        # We want to calculate durations for: SC to SAR, SAR to SIAR, SIAR to ASI
        durations = []
        
        # Calculate durations in years between consecutive milestones
        for i in range(len(sim_dates) - 1):
            if i + 1 < len(sim_dates):  # Ensure we don't go out of bounds
                delta = sim_dates[i+1] - sim_dates[i]
                years = delta.days / 365.0
                durations.append(years)
        
        # Ensure we have exactly 3 durations (SC to SAR, SAR to SIAR, SIAR to ASI)
        # Pad with NaN if we have fewer milestones due to capping
        while len(durations) < 3:
            durations.append(float('nan'))
        
        # Take only the first 3 durations to match our phase names
        phase_durations.append(durations[:3])
    
    # Transpose to get list of durations for each phase
    phase_durations = list(map(list, zip(*phase_durations)))
    
    # Filter out NaN values for each phase
    filtered_phase_durations = []
    for phase in phase_durations:
        filtered_phase = [d for d in phase if not np.isnan(d)]
        filtered_phase_durations.append(filtered_phase)
    
    # Set up figure with space for statistics on the right
    fig = plt.figure(figsize=(12, 6), facecolor=bg_rgb)
    
    # Create two subplots - one for the boxplot, one for the stats
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])
    ax_box = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax_box.set_facecolor(bg_rgb)
    ax_stats.set_facecolor(bg_rgb)
    
    # Define phases
    phase_names = ["SC to SAR", "SAR to SIAR", "SIAR to ASI"] # "ASI to WS"
    
    phase_full = ["Superhuman Coder", "Superhuman AI Researcher", "Superintelligent AI Researcher", "Artificial Superintelligence"]
    # Define shades of green for the boxes
    colors = ['#228B22', '#228B22', '#228B22']  # Dark green, Forest green, Lime green
    
    # Create box plot with log scale
    ax_box.set_yscale('log')
    
    # Create box plot with custom whiskers at 90th percentile - fix the deprecation warning
    bp = ax_box.boxplot(filtered_phase_durations, tick_labels=phase_names, patch_artist=True, 
                        whis=(10, 90))  # Set whiskers at 10th and 90th percentiles
    
    # Add color to boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Add some transparency for better visibility
    
    # Customize whiskers and medians for better visibility on log scale
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Add grid appropriate for log scale
    ax_box.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    plot_title = title if title else "Time Spent in Each Milestone Transition, assuming fixed training compute"
    ax_box.set_ylabel("Calendar Years (log scale)", fontproperties=fonts["regular_small"])
    ax_box.set_title(plot_title, fontproperties=fonts["regular"], loc='left')
    
    # Create statistics table in the right subplot
    ax_stats.axis('off')  # Hide axes for the stats panel
    
    # Prepare statistics text using filtered data
    stats_text = "Statistics (years):\n\n"
    for i, data in enumerate(filtered_phase_durations):
        if len(data) > 0:
            p10 = np.percentile(data, 10)
            p50 = np.percentile(data, 50)  # median
            p90 = np.percentile(data, 90)
            
            stats_text += f"{phase_full[i]} to \n {phase_full[i+1]}\n"
            stats_text += f"  10th: {p10:.2f}\n"
            stats_text += f"  50th: {p50:.2f}\n"
            if (p90 > 100): 
                stats_text += f"  90th: >100\n\n"
            else: 
                stats_text += f"  90th: {p90:.2f}\n\n"
        else:
            stats_text += f"{phase_full[i]} to \n {phase_full[i+1]}\n"
            stats_text += f"  No valid data\n\n"
    
    # Add statistics text to the right panel
    ax_stats.text(0.06, 0.95, stats_text, 
                 transform=ax_stats.transAxes,
                 verticalalignment='top',
                 fontproperties=fonts["small"],
                 bbox=dict(facecolor=bg_rgb, alpha=0.8, 
                          edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Set fixed y-axis ticks at 0.1, 1, 10, 100
    ax_box.set_yticks([0.01, 0.1, 1, 10, 100])
    ax_box.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    ax_box.set_ylim(0.01, 120)
    # Set the y-axis limits
    # ax_box.set_ylim(0.05, 200)  # Slightly beyond the display range for visual clarity
    
    # Add horizontal lines at each tick mark
    reference_lines = [0.01, 0.1, 1, 10, 100]
    for line_val in reference_lines:
        ax_box.axhline(y=line_val, color='gray', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    return fig

def print_median_comparison(all_milestone_dates: list[list[datetime]], config: dict):
    """Compare median of differences vs difference of medians."""
    phase_names = ["SC to SAR", "SAR to SIAR", "SIAR to ASI"] # "ASI to WS"

    # Convert dates to years
    milestone_years = [[d.year + d.timetuple().tm_yday/365 for d in sim_dates] for sim_dates in all_milestone_dates]
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    start_year = start_date.year + start_date.timetuple().tm_yday/365
    
    # Calculate medians of absolute years
    medians_absolute = [np.median([years[i] for years in milestone_years if i < len(years)]) for i in range(len(phase_names))]  # Fixed number of milestones
    medians_absolute.insert(0, start_year)
    
    # Calculate differences between consecutive medians
    diff_of_medians = [medians_absolute[i+1] - medians_absolute[i] for i in range(len(medians_absolute)-1)]
    
    # Calculate phase durations for each simulation, handling potential NaN or infinite values
    phase_durations = []
    
    for sim_dates in all_milestone_dates:
        years = [d.year + d.timetuple().tm_yday/365 for d in sim_dates]
        years.insert(0, start_year)
        
        # Handle simulations with capped dates by limiting to actual number of milestones
        durations = []
        for i in range(min(len(years)-1, len(phase_names))):
            durations.append(years[i+1] - years[i])
        
        # Pad with NaN if we don't have enough values
        while len(durations) < len(phase_names):
            durations.append(float('nan'))
            
        phase_durations.append(durations)
    
    # Transpose to get list of durations for each phase
    phase_durations = list(map(list, zip(*phase_durations)))
    
    # Calculate medians, filtering out NaN or infinite values
    median_of_diffs = []
    for phase in phase_durations:
        valid_durations = [d for d in phase if np.isfinite(d)]
        if valid_durations:
            median_of_diffs.append(np.median(valid_durations))
        else:
            median_of_diffs.append(float('nan'))

def create_multi_project_timeline_plot(all_first_milestone_dates: list[list[datetime]], all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create timeline plot showing milestone achievement distributions for multiple projects.
    
    Args:
        all_first_milestone_dates: List of first-to-achieve milestone dates for each simulation
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Project comparison - show SAR distributions for each project
    projects = list(config["projects"].keys())
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects)))
    
    MAX_GRAPH_YEAR = 2032
    start_year = float(config["starting_time"].split()[-1])
    
    # Plot SAR distributions for each project
    stats_text = ""
    for proj_idx, project_name in enumerate(projects):
        project_sar_times = []
        for sim_results in all_project_results:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    project_sar_times.append(sar_date.year + sar_date.timetuple().tm_yday/365)
        
        if project_sar_times and len(project_sar_times) > 10:  # Need enough data for KDE
            # Calculate percentiles for stats
            p10 = np.percentile(project_sar_times, 10)
            p50 = np.percentile(project_sar_times, 50)
            p90 = np.percentile(project_sar_times, 90)
            
            # Convert decimal years to month and year format for display
            def year_to_date(year):
                year_int = int(year)
                if year_int > 9999:
                    return f"{year_int}"
                month = int((year - year_int) * 12) + 1
                month_name = datetime(year_int, month, 1).strftime('%b')
                return f"{month_name} {year_int}"
            
            # Filter data to visible range for KDE
            visible_data = [x for x in project_sar_times if start_year <= x <= MAX_GRAPH_YEAR]
            if visible_data:
                # Calculate KDE on visible data using same approach as multi-project plot
                try:
                    # Special case: if most delays are 0, add small jitter for visualization
                    if len([d for d in visible_data if d == 0]) > len(visible_data) * 0.5:
                        # Add tiny random jitter to zero values for visualization
                        jittered_data = []
                        for d in visible_data:
                            if d == 0:
                                jittered_data.append(d + np.random.normal(0, 0.01))  # Small jitter
                            else:
                                jittered_data.append(d)
                        # Check if we have enough distinct values after jittering
                        if len(set(jittered_data)) > 1:
                            kde = gaussian_kde(jittered_data)
                        else:
                            raise ValueError("Insufficient data variation for KDE")
                    else:
                        kde = gaussian_kde(visible_data)
                    
                    # Create x range for plotting
                    max_delay_in_data = max(project_sar_times)
                    MAX_DELAY = max(2.0, min(max(5, max_delay_in_data * 1.2), 20))  # Ensure at least 2 years range
                    x_range = np.linspace(0, MAX_DELAY, 1000)
                    density = kde(x_range)
                    
                    # Normalize density - fix division by zero
                    density_sum = np.sum(density)
                    if density_sum > 0:
                        density = density / density_sum * (len(visible_data) / len(project_sar_times))
                    else:
                        density = np.zeros_like(density)  # Fallback if density is all zeros
                    
                    # Plot distribution
                    project_params = config["projects"][project_name]
                    if isinstance(project_params, dict) and "lower_bound" in project_params:
                        # New format - show median of range
                        median_rate = (project_params["lower_bound"] + project_params["upper_bound"]) / 2
                        rate_label = f"~{median_rate:.1f}x"
                    else:
                        # Old format - show fixed rate
                        progress_rate = project_params if isinstance(project_params, (int, float)) else 1.0
                        rate_label = f"{progress_rate:.1f}x"
                    
                    ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                           label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                    ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # KDE failed due to insufficient variance, use histogram instead
                    project_params = config["projects"][project_name]
                    if isinstance(project_params, dict) and "lower_bound" in project_params:
                        # New format - show median of range
                        median_rate = (project_params["lower_bound"] + project_params["upper_bound"]) / 2
                        rate_label = f"~{median_rate:.1f}x"
                    else:
                        # Old format - show fixed rate
                        progress_rate = project_params if isinstance(project_params, (int, float)) else 1.0
                        rate_label = f"{progress_rate:.1f}x"
                    
                    # Create histogram - fix MAX_DELAY undefined error
                    if len(visible_data) > 0:
                        data_min = min(visible_data)
                        data_max = max(visible_data)
                        if data_max > data_min:
                            bins = np.linspace(data_min, data_max, 20)
                            hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            
                            # Plot as step function
                            ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                                   label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                            ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                        else:
                            # All values are the same, plot a vertical line
                            ax.axvline(x=data_min, color=project_colors[proj_idx], linestyle='--', linewidth=2,
                                     alpha=0.8, label=f"{project_name} ({rate_label}) - Consistent")
                    else:
                        # No data to plot
                        print(f"Warning: No data to plot for {project_name}")
                        continue
                
                # Add to stats text
                if (p90 > 2100):
                    stats = (
                        f"{project_name} ({rate_label}):\n"
                        f"  10th: {year_to_date(p10)}\n"
                        f"  50th: {year_to_date(p50)}\n"
                        f"  90th: >2100"
                    )
                else:
                    stats = (
                        f"{project_name} ({rate_label}):\n"
                        f"  10th: {year_to_date(p10)}\n"
                        f"  50th: {year_to_date(p50)}\n"
                        f"  90th: {year_to_date(p90)}"
                    )
                
                if proj_idx == 0:
                    stats_text = stats
                else:
                    stats_text += f"\n\n{stats}"
    
    # Add stats text
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Multi-Project Comparison: Superhuman AI Researcher Achievement", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

def create_project_delay_plot(all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict, project_progress_samples: dict = None) -> plt.Figure:
    """Create plot showing the delay between each project's SAR achievement and the leading project's SAR achievement.
    
    Args:
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
        project_progress_samples: Dictionary mapping project names to arrays of progress rate samples
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    projects = list(config["projects"].keys())
    # Include all projects in delay plot since any project could win
    projects_for_delay = projects
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects_for_delay)))
    
    # Calculate delays for each simulation
    project_delays = {project: [] for project in projects_for_delay}
    
    # Track which project wins each simulation to determine actual winners
    project_wins = {project: 0 for project in projects_for_delay}
    total_valid_sims = 0
    
    for sim_idx, sim_results in enumerate(all_project_results):
        # Find the earliest SAR date across all projects in this simulation
        earliest_sar_date = None
        valid_projects = {}
        
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    valid_projects[project_name] = sar_date
                    if earliest_sar_date is None or sar_date < earliest_sar_date:
                        earliest_sar_date = sar_date
        
        # Calculate delays relative to earliest and track winner
        if earliest_sar_date is not None:
            total_valid_sims += 1
            # Find the winner(s) for this simulation
            winners = [name for name, date in valid_projects.items() if date == earliest_sar_date]
            
            for winner in winners:
                project_wins[winner] += 1 / len(winners)  # Split wins if there's a tie
            
            for project_name in projects_for_delay:
                if project_name in valid_projects:
                    delay_days = (valid_projects[project_name] - earliest_sar_date).days
                    delay_years = delay_days / 365.0
                    project_delays[project_name].append(delay_years)
    
    # Plot delay distributions for each project
    stats_text = ""
    has_plotted_data = False
    
    for proj_idx, project_name in enumerate(projects_for_delay):
        delays = project_delays[project_name]
        
        if delays and len(delays) > 1:  # Need enough data for KDE
            # Calculate percentiles for stats
            p10 = np.percentile(delays, 10)
            p50 = np.percentile(delays, 50)
            p90 = np.percentile(delays, 90)
            
            # Check if this project actually wins most/all simulations
            win_rate = project_wins[project_name] / total_valid_sims if total_valid_sims > 0 else 0
            
            # For plotting distributions, filter out zero delays (wins) to avoid visualization issues
            non_zero_delays = [d for d in delays if d > 0.01]  # Remove essentially zero delays
            
            # Special handling for projects that win almost all simulations (>95%) or have no non-zero delays
            if win_rate > 0.95 or len(non_zero_delays) < 5:
                # Add a vertical line at the median delay position instead of a distribution
                ax.axvline(x=p50, color=project_colors[proj_idx], linestyle='--', linewidth=2, 
                          alpha=0.8, label=f"{project_name} - Wins {win_rate:.0%}")
                
                # Add to stats text
                if p50 < 0.01:
                    stats = f"{project_name}:\n  Wins {win_rate:.0%} (0.00 yrs)"
                else:
                    stats = f"{project_name}:\n  Wins {win_rate:.0%}: {p50:.2f} yrs"
            else:
                # Plot normal distribution for projects with meaningful delay variation or don't always win
                # Use non-zero delays for better visualization
                visible_data = non_zero_delays
                
                if visible_data and len(visible_data) > 5:  # Need enough non-zero data for meaningful distribution
                    # Calculate percentiles based on non-zero delays for better stats
                    nz_p10 = np.percentile(visible_data, 10)
                    nz_p50 = np.percentile(visible_data, 50)
                    nz_p90 = np.percentile(visible_data, 90)
                    
                    # Determine appropriate x-axis range based on the non-zero data
                    max_delay_in_data = max(visible_data)
                    # Ensure reasonable range - extend to show more of the distribution
                    MAX_DELAY = max(5.0, min(max(10, max_delay_in_data * 1.2), 30))
                    
                    # Calculate KDE on non-zero data
                    try:
                        kde = gaussian_kde(visible_data)
                        
                        # Create x range for plotting (start from small positive value, not 0)
                        x_range = np.linspace(0.01, MAX_DELAY, 1000)
                        density = kde(x_range)
                        
                        # Normalize density
                        density_sum = np.sum(density)
                        if density_sum > 0:
                            density = density / density_sum * (len(visible_data) / len(delays))
                        else:
                            density = np.zeros_like(density)
                        
                        # Plot distribution without progress rate labels
                        ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                               label=f"{project_name} - Wins {win_rate:.0%}", linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                        has_plotted_data = True
                        
                    except (np.linalg.LinAlgError, ValueError):
                        # KDE failed, use histogram instead
                        bins = np.linspace(min(visible_data), max(visible_data), 15)
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name} - Wins {win_rate:.0%}", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                    
                    # Add to stats text (use non-zero delay percentiles for better insight)
                    if nz_p90 > 25:
                        stats = (
                            f"{project_name}:\n"
                            f"  Wins {win_rate:.0%}\n"
                            f"  When behind:\n"
                            f"    10th: {nz_p10:.2f} yrs\n"
                            f"    50th: {nz_p50:.2f} yrs\n"
                            f"    90th: >25 yrs"
                        )
                    else:
                        stats = (
                            f"{project_name}:\n"
                            f"  Wins {win_rate:.0%}\n"
                            f"  When behind:\n"
                            f"    10th: {nz_p10:.2f} yrs\n"
                            f"    50th: {nz_p50:.2f} yrs\n"
                            f"    90th: {nz_p90:.2f} yrs"
                        )
                else:
                    # Not enough non-zero data, fall back to vertical line
                    ax.axvline(x=p50, color=project_colors[proj_idx], linestyle='--', linewidth=2, 
                              alpha=0.8, label=f"{project_name} - Wins {win_rate:.0%}")
                    
                    stats = f"{project_name}:\n  Wins {win_rate:.0%}"
        
            # Add to stats text
            if proj_idx == 0:
                stats_text = stats
            else:
                stats_text += f"\n\n{stats}"
        elif delays:  # Has some data but not enough for full analysis
            win_rate = project_wins[project_name] / total_valid_sims if total_valid_sims > 0 else 0
            
            # Add basic stats without progress rate labels
            stats = f"{project_name}:\n  Wins {win_rate:.0%}"
            if proj_idx == 0:
                stats_text = stats
            else:
                stats_text += f"\n\n{stats}"
    
    # If no meaningful data to plot, show a message
    if not has_plotted_data and not stats_text:
        ax.text(0.5, 0.5, "No delay data available\n(All projects may achieve SAR simultaneously)", 
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        stats_text = "No delay data available"
    
    # Determine final x-axis limit based on actual data, excluding always-zero delays
    non_zero_delays = [delay for delays in project_delays.values() for delay in delays if delay > 0.01]
    if non_zero_delays:
        data_max = np.percentile(non_zero_delays, 95)  # Use 95th percentile to avoid outliers
        # For small delays, ensure minimum visible range but extend to show more data
        final_xlim = max(5.0, min(data_max * 1.3, 30))  # Extended reasonable range
    else:
        final_xlim = 10
    
    # Add stats text
    if stats_text:
        text = ax.text(0.68, 1, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=plotting_style["font"]["sizes"]["small"])
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(0, final_xlim)
    ax.set_ylim(0, None)
    ax.set_xlabel("Delay Behind Leading Project (Years)", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Project SAR Achievement Delays Relative to Leading Project", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig


def create_project_sc_timeline_plot(all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create timeline plot showing SC achievement distributions for multiple projects.
    
    Args:
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Project comparison - show SC distributions for each project
    projects = list(config["projects"].keys())
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects)))
    
    # Extend the graph range to show full distributions
    MAX_GRAPH_YEAR = 2050  # Extended to show more of the distributions
    start_year = float(config["starting_time"].split()[-1])
    
    # Plot SC distributions for each project
    stats_text = ""
    for proj_idx, project_name in enumerate(projects):
        project_sc_times = []
        for sim_results in all_project_results:
            if project_name in sim_results and len(sim_results[project_name]) > 0:
                sc_date = sim_results[project_name][0]  # SC is at index 0
                if sc_date.year < 9999:  # Filter out capped values
                    project_sc_times.append(sc_date.year + sc_date.timetuple().tm_yday/365)
        
        if project_sc_times and len(project_sc_times) > 5:  # Lower threshold
            # Calculate percentiles for stats
            p10 = np.percentile(project_sc_times, 10)
            p50 = np.percentile(project_sc_times, 50)
            p90 = np.percentile(project_sc_times, 90)
            
            # Convert decimal years to month and year format for display
            def year_to_date(year):
                year_int = int(year)
                if year_int > 9999:
                    return f"{year_int}"
                month = int((year - year_int) * 12) + 1
                month_name = datetime(year_int, month, 1).strftime('%b')
                return f"{month_name} {year_int}"
            
            # Filter data to visible range - use most of the data
            visible_data = [x for x in project_sc_times if x <= MAX_GRAPH_YEAR]
            
            if visible_data:
                # Check if we need special handling for Leading Project (all same values)
                unique_visible = len(set(visible_data))
                
                if unique_visible <= 1:
                    # All values are the same - plot vertical line
                    median_sc = visible_data[0]
                    ax.axvline(x=median_sc, color=project_colors[proj_idx], linestyle='-', linewidth=3,
                             alpha=0.8, label=f"{project_name}", zorder=2)
                    
                elif unique_visible < 5:
                    # Too few unique values for KDE, use histogram
                    data_min = min(visible_data)
                    data_max = max(visible_data)
                    if data_max > data_min:
                        bins = min(unique_visible, 10)
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                    else:
                        # Fallback to vertical line
                        ax.axvline(x=data_min, color=project_colors[proj_idx], linestyle='-', linewidth=3,
                                 alpha=0.8, label=f"{project_name}", zorder=2)
                        
                else:
                    # Enough unique values for KDE
                    try:
                        kde = gaussian_kde(visible_data)
                        
                        # Create x range for plotting
                        data_min = min(visible_data)
                        data_max = max(visible_data)
                        x_range = np.linspace(max(start_year, data_min - 0.5), min(MAX_GRAPH_YEAR, data_max + 0.5), 1000)
                        density = kde(x_range)
                        
                        # Normalize density
                        density_sum = np.sum(density)
                        if density_sum > 0:
                            density = density / density_sum * (len(visible_data) / len(project_sc_times))
                        else:
                            density = np.zeros_like(density)
                        
                        # Plot distribution
                        ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                        
                    except (np.linalg.LinAlgError, ValueError) as e:
                        # KDE failed, use histogram instead
                        bins = min(20, unique_visible)
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                
                # Add to stats text
                stats = (
                    f"{project_name}:\n"
                    f"  10th: {year_to_date(p10)}\n"
                    f"  50th: {year_to_date(p50)}\n"
                    f"  90th: {year_to_date(p90)}"
                )
                
                if proj_idx == 0:
                    stats_text = stats
                else:
                    stats_text += f"\n\n{stats}"
    
    # Add stats text
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Multi-Project Comparison: Superhuman Coder Achievement", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

def create_project_sar_timeline_plot(all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create timeline plot showing SAR achievement distributions for multiple projects.
    
    Args:
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Project comparison - show SAR distributions for each project
    projects = list(config["projects"].keys())
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects)))
    
    # Extend the graph range to show full distributions
    MAX_GRAPH_YEAR = 2050  # Extended to show more of the distributions
    start_year = float(config["starting_time"].split()[-1])
    
    # Plot SAR distributions for each project
    stats_text = ""
    for proj_idx, project_name in enumerate(projects):
        project_sar_times = []
        for sim_results in all_project_results:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is at index 1
                if sar_date.year < 9999:  # Filter out capped values
                    project_sar_times.append(sar_date.year + sar_date.timetuple().tm_yday/365)
        
        if project_sar_times and len(project_sar_times) > 5:  # Lower threshold
            # Calculate percentiles for stats
            p10 = np.percentile(project_sar_times, 10)
            p50 = np.percentile(project_sar_times, 50)
            p90 = np.percentile(project_sar_times, 90)
            
            # Convert decimal years to month and year format for display
            def year_to_date(year):
                year_int = int(year)
                if year_int > 9999:
                    return f"{year_int}"
                month = int((year - year_int) * 12) + 1
                month_name = datetime(year_int, month, 1).strftime('%b')
                return f"{month_name} {year_int}"
            
            # Filter data to visible range - use most of the data
            visible_data = [x for x in project_sar_times if x <= MAX_GRAPH_YEAR]
            
            if visible_data:
                # Check if we need special handling for Leading Project (all same values)
                unique_visible = len(set(visible_data))
                
                if unique_visible <= 1:
                    # All values are the same - plot vertical line
                    median_sar = visible_data[0]
                    ax.axvline(x=median_sar, color=project_colors[proj_idx], linestyle='-', linewidth=3,
                             alpha=0.8, label=f"{project_name}", zorder=2)
                    
                elif unique_visible < 5:
                    # Too few unique values for KDE, use histogram
                    data_min = min(visible_data)
                    data_max = max(visible_data)
                    if data_max > data_min:
                        bins = np.linspace(data_min, data_max, max(3, unique_visible))
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                    else:
                        # All values identical, just show vertical line
                        ax.axvline(x=data_min, color=project_colors[proj_idx], linestyle='-', linewidth=3,
                                 alpha=0.8, label=f"{project_name}", zorder=2)
                else:
                    # Use KDE for smooth distribution
                    try:
                        kde = gaussian_kde(visible_data)
                        x_range = np.linspace(min(visible_data), MAX_GRAPH_YEAR, 1000)
                        density = kde(x_range)
                        
                        # Plot distribution
                        ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                        
                    except (np.linalg.LinAlgError, ValueError):
                        # KDE failed, fall back to histogram
                        bins = min(15, max(5, unique_visible))
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function  
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name}", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                
                # Add to stats text
                stats = (
                    f"{project_name}:\n"
                    f"  10th: {year_to_date(p10)}\n"
                    f"  50th: {year_to_date(p50)}\n"
                    f"  90th: {year_to_date(p90)}"
                )
                
                if proj_idx == 0:
                    stats_text = stats
                else:
                    stats_text += f"\n\n{stats}"
    
    # Add stats text
    if stats_text:
        text = ax.text(0.68, 1, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=plotting_style["font"]["sizes"]["small"])
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Project SAR Achievement Timeline", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

