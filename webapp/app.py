import streamlit as st
import sys
import os
from pathlib import Path
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import copy
import json

# Add project root to sys.path to allow imports from takeoff
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from takeoff import forecasting_takeoff

st.set_page_config(layout="wide")

st.title("AI Takeoff Model")

@st.cache_data
def load_default_config():
    """Load the default configuration from the YAML file."""
    config_path = project_root / "takeoff/takeoff_params.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_projects_config():
    """Load project configurations from the dedicated JSON file."""
    projects_path = project_root / "webapp/projects.json"
    with open(projects_path, 'r') as f:
        return json.load(f)

def get_config():
    """Load default config and update with user inputs from the sidebar."""
    if 'config' not in st.session_state:
        st.session_state.config = copy.deepcopy(load_default_config())
    return st.session_state.config

config = get_config()
all_available_projects = load_projects_config()


# --- UI Sidebar ---
st.sidebar.header("Simulation Parameters")

if st.sidebar.button("Reset to Defaults"):
    st.session_state.clear() # Clear all session state
    st.rerun()

config['simulation']['n_sims'] = st.sidebar.slider(
    "Number of simulations", 100, 10000, config['simulation']['n_sims']
)
config['starting_time'] = st.sidebar.text_input(
    "Starting time", config['starting_time']
)

# --- Speedup Factors Configuration ---
st.sidebar.header("Speedup Factors (80% CI)")

if 'speedup_distributions' in config:
    speedup_dists_config = config['speedup_distributions']
    
    # Define bounds for sliders
    slider_bounds = {
        "SC": {"min": 1.0, "max": 20.0, "step": 0.1},
        "SAR_ratio": {"min": 1.0, "max": 15.0, "step": 0.1},
        "SIAR_ratio": {"min": 1.0, "max": 25.0, "step": 0.1},
        "ASI_ratio": {"min": 1.0, "max": 20.0, "step": 0.1}
    }

    for key, default_value in speedup_dists_config.items():
        bounds = slider_bounds.get(key, {"min": 1.0, "max": 30.0, "step": 0.1})
        
        # Get the current value from session state, or use the default from the config file.
        # The key on the slider widget ensures that user changes are automatically saved to session_state.
        current_value = st.session_state.get(f"speedup_{key}", (float(default_value[0]), float(default_value[1])))
        
        new_bounds = st.sidebar.slider(
            f"{key} CI",
            min_value=bounds['min'],
            max_value=bounds['max'],
            value=current_value,
            step=bounds['step'],
            key=f"speedup_{key}"
        )
        # Update the main config dictionary with the value from the slider (which is stateful).
        # This ensures the simulation always uses the most up-to-date value from the UI.
        config['speedup_distributions'][key] = [new_bounds[0], new_bounds[1]]

# --- Project Configuration ---
st.sidebar.header("Projects")

active_projects_config = {}

for project in all_available_projects:
    project_name = project['name']
    project_data = project['params']
    
    # Use session state to remember if a project is selected
    # Initialize with default value if not in state
    if f"include_{project_name}" not in st.session_state:
        st.session_state[f"include_{project_name}"] = project['enabled_by_default']

    is_active = st.sidebar.checkbox(f"Include {project_name}", key=f"include_{project_name}")
    
    if is_active:
        # Use a copy of the project data to avoid modifying the cached original
        active_project_data = copy.deepcopy(project_data)
        
        with st.sidebar.expander(f"Settings for {project_name}", expanded=False):
            # --- Project-specific Compute Decay ---
            schedule_params = active_project_data.get('schedule_params', {})
            default_use_decay = schedule_params.get('type') == 'decay'
            
            use_decay = st.checkbox("Enable compute decay", value=default_use_decay, key=f"decay_{project_name}")
            
            if use_decay:
                default_eta = schedule_params.get('eta_days', 5000)
                eta_days = st.slider(
                    "Decay Rate (eta_days)",
                    min_value=500,
                    max_value=20000,
                    value=default_eta,
                    step=500,
                    key=f"eta_{project_name}"
                )
                active_project_data['schedule_params'] = {
                    'type': 'decay',
                    'eta_days': eta_days
                }
            else:
                active_project_data['schedule_params'] = {
                    'type': 'constant'
                }
        
        active_projects_config[project_name] = active_project_data


# Update the main config with the user-selected projects
config['projects'] = active_projects_config

# Set start date for all active projects
for project_data in config['projects'].values():
    if 'schedule_params' not in project_data:
        project_data['schedule_params'] = {}
    project_data['schedule_params']['start_date'] = config['starting_time']

# --- Live YAML View ---
st.sidebar.header("Live Configuration (for simulation)")
st.sidebar.text_area(
    "The full configuration that will be used for the simulation is shown below.",
    value=yaml.dump(config),
    height=300,
    disabled=True
)

# --- Main App Logic ---
def run_simulation(config):
    """Run the simulation with the given configuration and return figures."""
    st.write("### Running simulation...")
    
    plotting_style = config["plotting_style"]

    # Load trajectory data
    trajectory_data_path = project_root / "takeoff/research_trajectory_data.json"
    trajectory_data = forecasting_takeoff.load_research_trajectory_data(str(trajectory_data_path))
    
    # Generate milestone samples
    milestone_samples = forecasting_takeoff.get_milestone_samples(config, config["simulation"]["n_sims"])

    # Generate project samples
    sc_speedup_samples = milestone_samples["speeds"]["SC"]
    project_samples = forecasting_takeoff.get_project_samples_with_correlations(config, config["simulation"]["n_sims"], trajectory_data, sc_speedup_samples)
    
    project_progress_samples = {name: data["software_progress_rate"] for name, data in project_samples.items()}
    
    # Setup plotting
    fonts = forecasting_takeoff.setup_plotting_style(plotting_style)
    
    # Run simulations
    all_first_milestone_dates = []
    all_project_results = []
    all_project_phase_durations = []
    
    progress_bar = st.progress(0, text="Running simulations...")
    n_sims = config["simulation"]["n_sims"]
    for i in range(n_sims):
        project_results, first_milestone_dates, project_phase_durations = forecasting_takeoff.run_multi_project_simulation_with_tracking(
            milestone_samples, i, project_progress_samples, project_samples
        )
        all_first_milestone_dates.append(first_milestone_dates)
        all_project_results.append(project_results)
        all_project_phase_durations.append(project_phase_durations)
        progress_bar.progress((i + 1) / n_sims, text=f"Simulations: {i+1}/{n_sims}")
    progress_bar.empty()

    st.write("### Simulation complete. Generating plots...")

    # Create plots
    figs = {}
    if not active_projects_config:
        st.warning("No projects selected. Nothing to plot.")
        return figs, {}

    # Fastest project plots (generate these first to display at the top)
    if project_progress_samples:
        fastest_project = min(project_progress_samples.keys(), key=lambda x: 1/np.mean(project_progress_samples[x]))
        fastest_milestone_dates = [results[fastest_project] for results in all_project_results]
        with st.spinner(f'Generating timeline for fastest project ({fastest_project})...'):
            title = f"Timeline for project: {fastest_project}"
            figs["fastest_project_timeline"] = forecasting_takeoff.create_milestone_timeline_plot(
                fastest_milestone_dates, config, plotting_style, fonts, title=title
            )

        with st.spinner(f'Generating phase durations for fastest project ({fastest_project})...'):
            title = f"Phase Durations for project: {fastest_project}"
            figs["fastest_project_phases"] = forecasting_takeoff.create_phase_duration_plot(
                fastest_milestone_dates, config, plotting_style, fonts, title=title
            )

    with st.spinner('Generating project delay plot...'):
        figs["project_delays"] = forecasting_takeoff.create_project_delay_plot(all_project_results, config, plotting_style, fonts, project_progress_samples)
    with st.spinner('Generating SC timeline plot...'):
        figs["sc_timeline"] = forecasting_takeoff.create_project_sc_timeline_plot(all_project_results, config, plotting_style, fonts)
    with st.spinner('Generating SAR timeline plot...'):
        figs["sar_timeline"] = forecasting_takeoff.create_project_sar_timeline_plot(all_project_results, config, plotting_style, fonts)
    
    plt.close("all")

    return figs, {}


if st.sidebar.button("Run Simulation", type="primary"):
    # Make a deep copy of the config to avoid issues with Streamlit's state
    config_to_run = copy.deepcopy(config)
    
    figs, captured_output = run_simulation(config_to_run)
    
    st.subheader("Resulting Plots")
    if figs:
        for name, fig in figs.items():
            st.pyplot(fig)
    else:
        st.write("No plots were generated.")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation'.") 