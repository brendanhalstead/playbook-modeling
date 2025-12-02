"""Generate doubling time comparison table for intended vs mistaken curves.

Uses the same parameters as plot_three_models.py (80p_curves_compared_to_intended.png):
- base_year = 2021
- q = 0.85
- T1_h calibrated for 3.6 month doubling time at 2025.25
- h0 calibrated for 15 minutes horizon at 2025.25
"""

import numpy as np
import pandas as pd
from pathlib import Path


def y_to_horizon(y, y_min=0, y_max=23,
                 minutes_at_y_min=4/60, minutes_at_y_max=5*2000*60):
    """Convert y values to horizon_minutes using linear interpolation in log space."""
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)
    log_minutes = log_min + (log_max - log_min) * (y - y_min) / (y_max - y_min)
    return 10 ** log_minutes


def format_horizon(minutes):
    """Format horizon as human-readable string."""
    if minutes < 1:
        return f"{minutes * 60:.1f}s"
    elif minutes < 60:
        return f"{minutes:.1f}min"
    elif minutes < 480:
        return f"{minutes / 60:.1f}hr"
    elif minutes < 10020:  # work month
        return f"{minutes / 480:.1f} work days"
    elif minutes < 120240:  # work year
        return f"{minutes / 10020:.1f} work mo"
    else:
        return f"{minutes / 120240:.1f} work yr"


def main():
    here = Path(__file__).resolve().parent

    # Parameters from plot_three_models.py
    base_year = 2021
    q = 0.85
    T1_h = 3.6/12 + 4.25 * (1 - 0.85)  # Calibrated for 3.6 months doubling time at 2025.25

    # Calculate h0 so that horizon = 15 minutes at year 2025.25
    t_target = 4.25  # years from 2021 to 2025.25
    arg_target = 1 - t_target * (1 - q) / T1_h
    n_target = np.log(arg_target) / np.log(q)
    h0 = 15.0 / (2 ** n_target)

    # Singularity
    singularity = base_year + T1_h / (1 - q)

    print(f"Parameters:")
    print(f"  base_year = {base_year}")
    print(f"  q = {q}")
    print(f"  T1_h = {T1_h:.4f} years = {T1_h * 12:.2f} months")
    print(f"  h0 = {h0:.4f} minutes")
    print(f"  Singularity = {singularity:.2f}")
    print()

    # For mistaken model, use parameters from illustrative_se_trend
    y0_mistaken = 1.45
    T1_y = 1.91  # years (y-doubling time)
    singularity_mistaken = base_year + T1_y / (1 - q)

    # Generate fine-grained time series for both models
    years = np.linspace(base_year, singularity - 0.001, 10000)

    # Intended model (horizon-doubling)
    intended_values = []
    for year in years:
        t = year - base_year
        if t <= 0:
            h = h0
        else:
            arg = 1 - t * (1 - q) / T1_h
            if arg <= 0:
                h = np.inf
            else:
                n = np.log(arg) / np.log(q)
                h = h0 * (2 ** n)
        intended_values.append(h)
    intended_values = np.array(intended_values)

    # Mistaken model (y-doubling then convert)
    mistaken_values = []
    for year in years:
        t = year - base_year
        if t <= 0:
            y = y0_mistaken
        else:
            arg = 1 - t * (1 - q) / T1_y
            if arg <= 0:
                y = np.inf
            else:
                n = np.log(arg) / np.log(q)
                y = y0_mistaken * (2 ** n)
        mistaken_values.append(y_to_horizon(y))
    mistaken_values = np.array(mistaken_values)

    # Find horizon doublings for intended model
    print("=" * 80)
    print("Intended Model: Horizon Doublings")
    print("(Each horizon doubling takes q = 0.85 times as long as the previous)")
    print("=" * 80)

    intended_doublings = []
    current_h = intended_values[0]
    target_h = current_h * 2
    doubling_num = 0
    prev_year = base_year

    for i, (year, h) in enumerate(zip(years, intended_values)):
        if h >= target_h and not np.isinf(h):
            doubling_num += 1
            time_for_doubling = year - prev_year
            intended_doublings.append({
                'Doubling #': doubling_num,
                'Duration (months)': time_for_doubling * 12,
                'Year': year,
                'Horizon': format_horizon(h),
                'Horizon (min)': h,
            })
            prev_year = year
            target_h = h * 2
            if doubling_num >= 20:
                break

    print(f"\n{'#':<4} {'Duration':<14} {'Year':<10} {'Horizon':<20}")
    print("-" * 50)
    for row in intended_doublings:
        print(f"{row['Doubling #']:<4} {row['Duration (months)']:<14.2f} {row['Year']:<10.2f} {row['Horizon']:<20}")

    # Find log-horizon doublings for mistaken model
    # (equivalent to horizon doublings - when horizon doubles)
    print("\n" + "=" * 80)
    print("Mistaken Model: Log-Horizon Doublings (= Horizon Doublings)")
    print("(y doubles with shrinking time, but horizon doubling time stays more constant)")
    print("=" * 80)

    mistaken_doublings = []
    current_h = mistaken_values[0]
    target_h = current_h * 2
    doubling_num = 0
    prev_year = base_year

    for i, (year, h) in enumerate(zip(years, mistaken_values)):
        if h >= target_h and not np.isinf(h):
            doubling_num += 1
            time_for_doubling = year - prev_year
            mistaken_doublings.append({
                'Doubling #': doubling_num,
                'Duration (months)': time_for_doubling * 12,
                'Year': year,
                'Horizon': format_horizon(h),
                'Horizon (min)': h,
            })
            prev_year = year
            target_h = h * 2
            if doubling_num >= 20:
                break

    print(f"\n{'#':<4} {'Duration':<14} {'Year':<10} {'Horizon':<20}")
    print("-" * 50)
    for row in mistaken_doublings:
        print(f"{row['Doubling #']:<4} {row['Duration (months)']:<14.2f} {row['Year']:<10.2f} {row['Horizon']:<20}")

    # Create combined comparison table
    print("\n" + "=" * 80)
    print("Side-by-Side Comparison")
    print("=" * 80)

    max_rows = max(len(intended_doublings), len(mistaken_doublings))
    print(f"\n{'#':<4} {'Intended':<20} {'Mistaken':<20} {'Ratio':<10}")
    print(f"{'':4} {'Duration (mo)':<20} {'Duration (mo)':<20} {'M/I':<10}")
    print("-" * 60)

    for i in range(max_rows):
        int_dur = intended_doublings[i]['Duration (months)'] if i < len(intended_doublings) else None
        mis_dur = mistaken_doublings[i]['Duration (months)'] if i < len(mistaken_doublings) else None

        int_str = f"{int_dur:.2f}" if int_dur else "-"
        mis_str = f"{mis_dur:.2f}" if mis_dur else "-"
        ratio_str = f"{mis_dur / int_dur:.2f}" if (int_dur and mis_dur) else "-"

        print(f"{i+1:<4} {int_str:<20} {mis_str:<20} {ratio_str:<10}")

    # Save to CSV
    df_intended = pd.DataFrame(intended_doublings)
    df_mistaken = pd.DataFrame(mistaken_doublings)

    # Combined table
    combined_data = []
    for i in range(max_rows):
        row = {'Doubling #': i + 1}
        if i < len(intended_doublings):
            row['Intended Duration (months)'] = intended_doublings[i]['Duration (months)']
            row['Intended Year'] = intended_doublings[i]['Year']
            row['Intended Horizon'] = intended_doublings[i]['Horizon']
        if i < len(mistaken_doublings):
            row['Mistaken Duration (months)'] = mistaken_doublings[i]['Duration (months)']
            row['Mistaken Year'] = mistaken_doublings[i]['Year']
            row['Mistaken Horizon'] = mistaken_doublings[i]['Horizon']
        if i < len(intended_doublings) and i < len(mistaken_doublings):
            row['Ratio (M/I)'] = mistaken_doublings[i]['Duration (months)'] / intended_doublings[i]['Duration (months)']
        combined_data.append(row)

    df_combined = pd.DataFrame(combined_data)
    output_path = here / "doubling_comparison_table.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\nSaved combined table to: {output_path}")


if __name__ == "__main__":
    main()
