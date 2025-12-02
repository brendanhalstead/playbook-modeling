"""Compare intended vs mistaken trends starting from same initial conditions.

Both models start with:
- Same initial horizon (h0)
- Same initial horizon-doubling time (T1_h)
- Same d factor (0.85) - each doubling takes d times as long as the previous

The difference is:
- Intended: horizon doubles directly with shrinking doubling times
- Mistaken: y doubles with shrinking doubling times, then y is converted to horizon
"""

import numpy as np
import pandas as pd
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
    base_year = 2025.25  # Start at 2025.25
    d = 0.85  # Each doubling takes d times as long as the previous

    # Starting conditions - 15 minutes at 2025.25, 3.6 month doubling time
    h0 = 15.0  # minutes (same as intended model at 2025.25)
    T1_h = 3.6 / 12  # 3.6 months in years

    # Generate years - stop just before intended singularity
    singularity_intended = base_year + T1_h / (1 - d)
    years = np.linspace(2025.25, singularity_intended - 0.01, 500)

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

    # Model 2: Fixed formula (y-doubling, then convert to horizon)
    # We need to find y0 such that y_to_horizon(y0) = h0
    # This is the inverse of y_to_horizon
    y_min = 0
    y_max = 23
    minutes_at_y_min = 4/60
    minutes_at_y_max = 5*2000*60
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)

    # h0 = 10^(log_min + (log_max - log_min) * y0 / y_max)
    # log10(h0) = log_min + (log_max - log_min) * y0 / y_max
    # y0 = (log10(h0) - log_min) * y_max / (log_max - log_min)
    y0 = (np.log10(h0) - log_min) * y_max / (log_max - log_min)
    print(f"Calculated y0 for h0={h0:.4f} min: y0={y0:.4f}")

    # For mistaken formula, T1_y is the y-doubling time
    # We want the actual horizon-doubling time to match T1_h
    # From the analysis: instantaneous horizon-doubling time ≈ y-doubling time * (some factor)
    # The factor depends on the y-to-horizon conversion slope
    B = (log_max - log_min) / y_max
    # Initial horizon-doubling time = log10(2) / (B * y0 * ln(2) * dn/dt_0)
    # where dn/dt_0 = (1-d) / (T1_y * ln(d)) (taking absolute value)
    # Simplifying: T_h = log10(2) / (B * y0 * ln(2)) * T1_y * |ln(d)| / (1-d)
    # So: T1_y = T_h * B * y0 * ln(2) * (1-d) / (log10(2) * |ln(d)|)

    # Actually let's use a simpler approach - calibrate T1_y so that
    # the first horizon doubling happens at T1_h
    # From the CSV analysis, with T1_y=1.91 years, first horizon doubling took 1.51 years
    # Ratio: 1.51/1.91 = 0.79
    # So T1_y ≈ T1_h / 0.79

    # But let's be more precise - simulate and adjust
    def simulate_fixed_formula(T1_y):
        fixed_values = []
        for year in years:
            t = year - base_year
            if t <= 0:
                y = y0
            else:
                arg = 1 - t * (1 - d) / T1_y
                if arg <= 0:
                    y = np.inf  # At singularity
                else:
                    n = np.log(arg) / np.log(d)
                    y = y0 * (2 ** n)
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
        values = simulate_fixed_formula(T1_y_mid)
        doubling_time = find_first_doubling_time(values)
        if doubling_time < T1_h:
            T1_y_low = T1_y_mid
        else:
            T1_y_high = T1_y_mid

    T1_y = T1_y_mid
    print(f"Calibrated T1_y for first horizon doubling at {T1_h:.2f} years: T1_y={T1_y:.4f} years")

    fixed_values = simulate_fixed_formula(T1_y)

    # Calculate singularities
    singularity_fixed = base_year + T1_y / (1 - d)

    print(f"Singularity for intended model: {singularity_intended:.2f}")
    print(f"Singularity for mistaken model: {singularity_fixed:.2f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both models
    ax.plot(years, intended_values, 'r-', linewidth=2.5,
            label=f'Intended (horizon-doubling, T1_h={T1_h:.2f}yr)', alpha=0.8)
    ax.plot(years, fixed_values, 'g--', linewidth=2,
            label=f'Mistaken (y-doubling, T1_y={T1_y:.2f}yr)', alpha=0.8)

    # Add vertical lines for singularities
    ax.axvline(x=singularity_intended, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=singularity_fixed, color='green', linestyle=':', alpha=0.5, linewidth=1)

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Horizon (work time)', fontsize=12)
    ax.set_title(f'Same Starting Conditions: h0={h0:.1f}min, first horizon-doubling={T1_h*12:.1f}mo, d={d}\n'
                 f'Singularities: Intended={singularity_intended:.2f}, Mistaken={singularity_fixed:.2f}',
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
    ax.set_xlim(2025.25, singularity_intended + 0.1)
    # Bound y-axis: min 1 second, max 100 work years
    ax.set_ylim(1/60, 100 * work_year)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.1)

    # Save
    output_path = here / "same_start_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
