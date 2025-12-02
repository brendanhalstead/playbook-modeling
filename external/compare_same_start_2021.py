"""Compare intended vs mistaken trends starting from same initial conditions in 2021.

Both models start with:
- Same initial horizon (h0) - the mistaken model's initial horizon
- Same initial horizon-doubling time - the mistaken model's actual first horizon doubling time
- Same d factor (0.85) - each doubling takes d times as long as the previous

The difference is:
- Intended: horizon doubles directly with shrinking doubling times
- Mistaken: y doubles with shrinking doubling times, then y is converted to horizon
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def y_to_horizon(y, y_min=0, y_max=23,
                 minutes_at_y_min=4/60, minutes_at_y_max=5*2000*60):
    """Convert y values to horizon_minutes using linear interpolation in log space."""
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)
    log_minutes = log_min + (log_max - log_min) * (y - y_min) / (y_max - y_min)
    return 10 ** log_minutes


def main():
    here = Path(__file__).resolve().parent

    # Common parameters
    base_year = 2021  # Start at 2021
    d = 0.85  # Each doubling takes d times as long as the previous

    # Starting conditions from mistaken model
    # From earlier analysis: y0=1.45, h0=0.1829 min, first horizon doubling took 1.51 years
    y0_mistaken = 1.45
    h0 = y_to_horizon(y0_mistaken)  # ~0.183 minutes
    T1_h = 1.51  # years - actual first horizon doubling time from mistaken model

    print(f"Starting horizon h0 = {h0:.4f} minutes = {h0*60:.2f} seconds")
    print(f"First horizon-doubling time T1_h = {T1_h:.2f} years = {T1_h*12:.1f} months")

    # Calculate singularities
    singularity_intended = base_year + T1_h / (1 - d)

    # Generate years - stop just before intended singularity
    years = np.linspace(2021, singularity_intended - 0.01, 500)

    # Model 1: Intended (horizon-doubling directly)
    intended_values = []
    for year in years:
        t = year - base_year
        if t <= 0:
            h = h0
        else:
            arg = 1 - t * (1 - d) / T1_h
            if arg <= 0:
                h = np.inf  # At singularity
            else:
                n = np.log(arg) / np.log(d)
                h = h0 * (2 ** n)
        intended_values.append(h)
    intended_values = np.array(intended_values)

    # Model 2: Mistaken formula (y-doubling, then convert to horizon)
    y_min = 0
    y_max = 23
    minutes_at_y_min = 4/60
    minutes_at_y_max = 5*2000*60
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)

    # For mistaken formula, T1_y is the y-doubling time
    # We want the actual horizon-doubling time to match T1_h
    # Calibrate T1_y so that first horizon doubling happens at T1_h

    def simulate_mistaken_formula(T1_y):
        fixed_values = []
        for year in years:
            t = year - base_year
            if t <= 0:
                y = y0_mistaken
            else:
                arg = 1 - t * (1 - d) / T1_y
                if arg <= 0:
                    y = np.inf  # At singularity
                else:
                    n = np.log(arg) / np.log(d)
                    y = y0_mistaken * (2 ** n)
            fixed_values.append(y_to_horizon(y))
        return np.array(fixed_values)

    # Find T1_y such that first horizon doubling happens at T1_h
    # Binary search
    def find_first_doubling_time(values):
        target = 2 * values[0]
        for i, v in enumerate(values):
            if v >= target:
                return years[i] - base_year
        return float('inf')

    T1_y_low, T1_y_high = 0.5, 5.0
    for _ in range(30):
        T1_y_mid = (T1_y_low + T1_y_high) / 2
        values = simulate_mistaken_formula(T1_y_mid)
        doubling_time = find_first_doubling_time(values)
        if doubling_time < T1_h:
            T1_y_low = T1_y_mid
        else:
            T1_y_high = T1_y_mid

    T1_y = T1_y_mid
    print(f"Calibrated T1_y for first horizon doubling at {T1_h:.2f} years: T1_y={T1_y:.4f} years")

    mistaken_values = simulate_mistaken_formula(T1_y)

    # Calculate singularities
    singularity_mistaken = base_year + T1_y / (1 - d)

    print(f"Singularity for intended model: {singularity_intended:.2f}")
    print(f"Singularity for mistaken model: {singularity_mistaken:.2f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both models
    ax.plot(years, intended_values, 'r-', linewidth=2.5,
            label=f'Intended (horizon-doubling, T1_h={T1_h:.2f}yr)', alpha=0.8)
    ax.plot(years, mistaken_values, 'g--', linewidth=2,
            label=f'Mistaken (y-doubling, T1_y={T1_y:.2f}yr)', alpha=0.8)

    # Add vertical lines for singularities
    ax.axvline(x=singularity_intended, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=singularity_mistaken, color='green', linestyle=':', alpha=0.5, linewidth=1)

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Horizon (work time)', fontsize=12)
    ax.set_title(f'Same Starting Conditions: h0={h0:.3f}min ({h0*60:.1f}s), first horizon-doubling={T1_h:.2f}yr, d={d}\n'
                 f'Singularities: Intended={singularity_intended:.2f}, Mistaken={singularity_mistaken:.2f}',
                 fontsize=12)

    # Custom y-axis ticks with human-readable labels (work time units)
    work_month = 167 * 60  # 10020 minutes
    work_year = 12 * work_month  # 120240 minutes
    y_ticks = [1/60, 1, 60, 60*8, work_month, work_year, 10*work_year, 100*work_year]
    y_labels = ['1s', '1m', '1h', '1 work day', '1 work mo', '1 work yr', '10 work yr', '100 work yr']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Grid
    ax.grid(True, alpha=0.3, which='both')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Set axis limits - show up to just past the intended singularity
    ax.set_xlim(2021, singularity_intended + 0.1)
    # Bound y-axis: min 1 second, max 100 work years
    ax.set_ylim(1/60, 100 * work_year)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.1)

    # Save
    output_path = here / "same_start_comparison_2021.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
