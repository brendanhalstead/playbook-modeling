import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory for saving figures
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'interpolation_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the CSV data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'new_model_trajectory.csv'), comment='#')

# Filter data: start at 2025.6 and end when ai_software_progress_multiplier reaches 19.5
start_time = 2025.6
target_multiplier = 19.5

# Find the row where multiplier first exceeds 19.5
df_filtered = df[df['time'] >= start_time].copy()

# Find where multiplier crosses 19.5 (interpolate if needed)
below_target = df_filtered[df_filtered['ai_software_progress_multiplier'] < target_multiplier]
above_target = df_filtered[df_filtered['ai_software_progress_multiplier'] >= target_multiplier]

if len(above_target) > 0:
    # Find the exact crossing point via interpolation
    last_below_idx = below_target.index[-1] if len(below_target) > 0 else None
    first_above_idx = above_target.index[0]

    if last_below_idx is not None:
        # Linear interpolation to find exact progress where multiplier = 19.5
        t1 = df.loc[last_below_idx, 'time']
        t2 = df.loc[first_above_idx, 'time']
        m1 = df.loc[last_below_idx, 'ai_software_progress_multiplier']
        m2 = df.loc[first_above_idx, 'ai_software_progress_multiplier']
        p1 = df.loc[last_below_idx, 'cumulative_progress']
        p2 = df.loc[first_above_idx, 'cumulative_progress']

        # Interpolate
        frac = (target_multiplier - m1) / (m2 - m1)
        end_time = t1 + frac * (t2 - t1)
        end_progress = p1 + frac * (p2 - p1)

        print(f"Target multiplier {target_multiplier} reached at time={end_time:.4f}, progress={end_progress:.4f}")
    else:
        end_time = df.loc[first_above_idx, 'time']
        end_progress = df.loc[first_above_idx, 'cumulative_progress']
else:
    # Use the last available data point
    end_time = df_filtered['time'].iloc[-1]
    end_progress = df_filtered['cumulative_progress'].iloc[-1]

# Filter data to the range
df_plot = df_filtered[df_filtered['time'] <= end_time + 0.1].copy()

# Get start and end values for interpolation
start_row = df_filtered[df_filtered['time'] >= start_time].iloc[0]
start_multiplier = start_row['ai_software_progress_multiplier']
start_progress = start_row['cumulative_progress']

# For the exponential interpolation in cumulative_progress:
# We want to interpolate (multiplier - 1) exponentially from start to end
# (multiplier - 1) = start_excess * exp(rate * (progress - start_progress))
# At end_progress: (target_multiplier - 1) = start_excess * exp(rate * (end_progress - start_progress))

start_excess = start_multiplier - 1
end_excess = target_multiplier - 1

# Calculate the exponential rate based on cumulative_progress
progress_span = end_progress - start_progress
rate = np.log(end_excess / start_excess) / progress_span

print(f"\nExponential interpolation parameters (based on cumulative_progress):")
print(f"  Start progress: {start_progress:.4f}, Start multiplier: {start_multiplier:.4f}")
print(f"  End progress: {end_progress:.4f}, Target multiplier: {target_multiplier}")
print(f"  Start excess (mult - 1): {start_excess:.4f}")
print(f"  End excess (mult - 1): {end_excess:.4f}")
print(f"  Progress span: {progress_span:.4f}")
print(f"  Exponential rate: {rate:.6f}")

# Generate interpolation curve based on cumulative_progress
interp_progress = np.linspace(start_progress, end_progress, 100)
interp_excess = start_excess * np.exp(rate * (interp_progress - start_progress))
interp_multipliers = interp_excess + 1

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot actual data
ax.plot(df_plot['cumulative_progress'], df_plot['ai_software_progress_multiplier'],
        'b-', linewidth=2, label='Actual trajectory')

# Plot exponential interpolation
ax.plot(interp_progress, interp_multipliers,
        'r--', linewidth=2, alpha=0.8,
        label=f'Exponential interpolation (progress-based)')

# Mark start and end points
ax.scatter([start_progress], [start_multiplier], color='green', s=100, zorder=5, label=f'Start (t={start_time})')
ax.scatter([end_progress], [target_multiplier], color='orange', s=100, zorder=5, label=f'End (mult={target_multiplier})')

# Labels and formatting
ax.set_xlabel('Cumulative Progress', fontsize=12)
ax.set_ylabel('AI Software Progress Multiplier', fontsize=12)
ax.set_title(f'AI Software Progress Multiplier vs Cumulative Progress\n(from t={start_time} to multiplier={target_multiplier})', fontsize=14)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_path = os.path.join(OUTPUT_DIR, 'new_model_trajectory_plot.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.close()
