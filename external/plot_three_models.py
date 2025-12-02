"""Plot three superexponential models for comparison.

Models:
1. Original (exp_power): y = e^(r * t^p), r=0.395, p=1.15
2. Mistaken illustrative graph trend (y-doubling): y = y0 * 2^n, where T_n = T1 * q^(n-1), q=0.85
3. Intended illustrative graph trend (horizon-doubling): horizon = h0 * 2^n, where T_n = T1_h * q_h^(n-1)

The horizon-doubling model is calibrated so that at year 2025.25, the doubling
time is 3.6 months.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Y to horizon conversion (same as in generate_illustrative_trend.py)
def y_to_horizon(y, y_min=0, y_max=23,
                 minutes_at_y_min=4/60, minutes_at_y_max=5*2000*60):
    """Convert y values to horizon_minutes using linear interpolation in log space."""
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)
    log_minutes = log_min + (log_max - log_min) * (y - y_min) / (y_max - y_min)
    return 10 ** log_minutes


def main():
    here = Path(__file__).resolve().parent

    # Load the two generated CSVs
    original_df = pd.read_csv(here / "original_graph_trend_generated.csv")
    fixed_df = pd.read_csv(here / "illustrative_se_trend_generated.csv")

    years = original_df['year'].values

    # Model 3: Horizon-doubling with q=0.85
    # Calibrated so doubling time = 3.6 months at year 2025.25
    #
    # Formula: T_current(t) = T1 - t*(1-q)
    # We want: 0.3 = T1 - 4.25*0.15
    # So: T1 = 0.3 + 0.6375 = 0.9375 years

    q_h = 0.85
    T1_h = 3.6/12 + 4.25 * (1 - 0.85)  # Calibrated for 3.6 months doubling time at 2025.25
    base_year = 2021

    # Calculate h0 so that horizon = 15 minutes at year 2025.25
    # h(t) = h0 * 2^n where n = log(1 - t*(1-q)/T1) / log(q)
    # At t=4.25: n = log(1 - 4.25*0.15/0.9708) / log(0.85) = 6.585
    # So h0 = 15 / 2^6.585 = 0.156 minutes
    t_target = 4.25  # years from 2021 to 2025.25
    arg_target = 1 - t_target * (1 - q_h) / T1_h
    n_target = np.log(arg_target) / np.log(q_h)
    h0 = 15.0 / (2 ** n_target)  # h0 such that h(2025.25) = 15 minutes

    max_horizon = y_to_horizon(30)  # ~150 years, high enough to not cap the curve

    horizon_doubling_values = []
    for year in years:
        t = year - base_year
        if t <= 0:
            h = h0
        else:
            arg = 1 - t * (1 - q_h) / T1_h
            if arg <= 0:
                h = max_horizon  # Past singularity
            else:
                n = np.log(arg) / np.log(q_h)
                h = h0 * (2 ** n)
                h = min(h, max_horizon)
        horizon_doubling_values.append(h)

    horizon_doubling_values = np.array(horizon_doubling_values)

    # Save the fixed illustrative graph trend to CSV
    fixed_trend_df = pd.DataFrame({
        'year': years,
        'horizon_minutes': horizon_doubling_values
    })
    fixed_trend_path = here / "fixed_illustrative_graph_trend.csv"
    fixed_trend_df.to_csv(fixed_trend_path, index=False)
    print(f"Saved fixed illustrative graph trend to: {fixed_trend_path}")

    # Verify the doubling time and horizon at 2025.25
    t_check = 2025.25 - base_year
    T_at_check = T1_h - t_check * (1 - q_h)
    h_at_check = np.interp(2025.25, years, horizon_doubling_values)
    print(f"Doubling time at year 2025.25: {T_at_check:.4f} years = {T_at_check*12:.2f} months")
    print(f"Horizon at year 2025.25: {h_at_check:.2f} minutes")
    print(f"h0 (starting horizon): {h0:.4f} minutes")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all three models
    ax.plot(years, original_df['horizon_minutes'],
            'b-', linewidth=2, label='Original (exp_power)', alpha=0.8)
    ax.plot(years, fixed_df['horizon_minutes'],
            'g--', linewidth=2, label='Mistaken illustrative graph trend', alpha=0.8)
    ax.plot(years, horizon_doubling_values,
            'r:', linewidth=2.5, label='Intended illustrative graph trend (3.6mo @ 2025.25)', alpha=0.8)

    # Add vertical line at 2025.25
    ax.axvline(x=2025.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(2025.25, ax.get_ylim()[0], '2025.25', ha='center', va='bottom', fontsize=9, color='gray')

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Horizon (work time)', fontsize=12)
    ax.set_title('Three Superexponential Models Comparison\n(Red: 3.6 month doubling time at 2025.25)', fontsize=14)

    # Custom y-axis ticks with human-readable labels (work time units)
    # 1 work month = 167 hours = 10020 minutes
    # 1 work year = 12 work months = 120240 minutes
    work_month = 167 * 60  # 10020 minutes
    work_year = 12 * work_month  # 120240 minutes
    y_ticks = [1/60, 1, 60, 60*8, work_month, work_year, 10*work_year]
    y_labels = ['1s', '1m', '1h', '1 work day', '1 work mo', '1 work yr', '10 work yr']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Grid
    ax.grid(True, alpha=0.3, which='both')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Set axis limits
    ax.set_xlim(2021, 2027.2)
    ax.set_ylim(0.01, 1e7)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.1)

    # Save
    output_path = here / "80p_curves_compared_to_intended.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")

    # Also print singularity time
    singularity = base_year + T1_h / (1 - q_h)
    print(f"Singularity for horizon-doubling model: {singularity:.2f}")


if __name__ == "__main__":
    main()
