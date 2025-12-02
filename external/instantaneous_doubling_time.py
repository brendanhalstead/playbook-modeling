"""Calculate and plot instantaneous horizon-doubling times for intended vs mistaken curves.

Instantaneous doubling time = ln(2) / (d(ln h)/dt)

For the intended model (horizon-doubling):
    h = h0 * 2^n where n = log(1 - t*(1-q)/T1) / log(q)
    T_inst(t) = T1 - t*(1-q)  (linear decrease to 0 at singularity)

For the mistaken model (y-doubling then convert):
    y = y0 * 2^n, then h = 10^(log_min + B*y) where B = (log_max - log_min)/y_max
    The instantaneous horizon-doubling time depends on y(t)
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

    # Parameters from plot_three_models.py
    base_year = 2021
    q = 0.85
    T1_h = 3.6/12 + 4.25 * (1 - 0.85)  # Calibrated for 3.6 months doubling time at 2025.25

    # Calculate h0 so that horizon = 15 minutes at year 2025.25
    t_target = 4.25  # years from 2021 to 2025.25
    arg_target = 1 - t_target * (1 - q) / T1_h
    n_target = np.log(arg_target) / np.log(q)
    h0 = 15.0 / (2 ** n_target)

    # Singularity for intended model
    singularity_intended = base_year + T1_h / (1 - q)

    # For mistaken model
    y0_mistaken = 1.45
    T1_y = 1.91  # years (y-doubling time)
    singularity_mistaken = base_year + T1_y / (1 - q)

    # Y-to-horizon conversion parameters
    y_min = 0
    y_max = 23
    minutes_at_y_min = 4/60
    minutes_at_y_max = 5*2000*60
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)
    B = (log_max - log_min) / y_max  # slope in log space

    print(f"Parameters:")
    print(f"  Intended: T1_h = {T1_h:.4f} yr = {T1_h*12:.2f} mo, singularity = {singularity_intended:.2f}")
    print(f"  Mistaken: T1_y = {T1_y:.4f} yr = {T1_y*12:.2f} mo, singularity = {singularity_mistaken:.2f}")
    print(f"  B (log-space slope) = {B:.6f}")
    print()

    # Generate time points
    years = np.linspace(base_year, singularity_intended - 0.05, 1000)

    # -------------------------------------------------------------------------
    # Intended model: Instantaneous doubling time
    # T_inst(t) = T1 - t*(1-q)
    # -------------------------------------------------------------------------
    intended_doubling_times = []
    for year in years:
        t = year - base_year
        T_inst = T1_h - t * (1 - q)
        if T_inst <= 0:
            T_inst = np.nan
        intended_doubling_times.append(T_inst * 12)  # Convert to months
    intended_doubling_times = np.array(intended_doubling_times)

    # -------------------------------------------------------------------------
    # Mistaken model: Instantaneous doubling time
    #
    # h = 10^(log_min + B*y) where y = y0 * 2^n
    # ln(h) = ln(10) * (log_min + B*y)
    # d(ln h)/dt = ln(10) * B * dy/dt
    #
    # For y = y0 * 2^n: dy/dt = y * ln(2) * dn/dt
    # where dn/dt = (1-q) / ((T1 - t*(1-q)) * |ln(q)|)
    #
    # So: d(ln h)/dt = ln(10) * B * y * ln(2) * (1-q) / ((T1 - t*(1-q)) * |ln(q)|)
    #
    # Instantaneous horizon-doubling time = ln(2) / (d(ln h)/dt)
    #   = |ln(q)| * (T1 - t*(1-q)) / (ln(10) * B * y * (1-q))
    # -------------------------------------------------------------------------
    mistaken_doubling_times = []
    mistaken_y_values = []

    for year in years:
        t = year - base_year

        # Calculate y at this time
        if t <= 0:
            y = y0_mistaken
        else:
            arg = 1 - t * (1 - q) / T1_y
            if arg <= 0:
                y = np.inf
            else:
                n = np.log(arg) / np.log(q)
                y = y0_mistaken * (2 ** n)

        mistaken_y_values.append(y)

        # Calculate instantaneous horizon-doubling time
        # T_y_inst = T1_y - t*(1-q) is the instantaneous y-doubling time
        T_y_inst = T1_y - t * (1 - q)

        if T_y_inst <= 0 or np.isinf(y):
            T_h_inst = np.nan
        else:
            # T_h_inst = |ln(q)| * T_y_inst / (ln(10) * B * y * (1-q))
            # Simplifying: T_h_inst = T_y_inst * |ln(q)| / (ln(10) * B * y * (1-q))
            # But actually easier: T_h_inst = T_y_inst / (ln(10) * ln(2) * B * y)
            # Because horizon doubling time = y-doubling time / (factor that converts y-doubling to h-doubling)
            # The factor is: ln(10) * B * y * ln(2) (from the derivative)
            T_h_inst = 1 / (np.log(10) * B * y * np.log(2) / T_y_inst)
            # Simplify: T_h_inst = T_y_inst / (ln(10) * B * y * ln(2))
            T_h_inst = T_y_inst / (np.log(10) * B * y * np.log(2))

        mistaken_doubling_times.append(T_h_inst * 12 if not np.isnan(T_h_inst) else np.nan)  # Convert to months

    mistaken_doubling_times = np.array(mistaken_doubling_times)
    mistaken_y_values = np.array(mistaken_y_values)

    # -------------------------------------------------------------------------
    # Calculate mistaken log-horizon doubling times for the plot
    # -------------------------------------------------------------------------
    mistaken_log_h_doubling_times = []
    for i, year in enumerate(years):
        t = year - base_year
        y = mistaken_y_values[i]

        if np.isinf(y) or y <= 0:
            T_log_h = np.nan
        else:
            T_y_inst = T1_y - t * (1 - q)
            if T_y_inst <= 0:
                T_log_h = np.nan
            else:
                # Time for log10(h) to increase by log10(2)
                T_log_h = (np.log10(2) * T_y_inst / (B * y * np.log(2))) * 12
        mistaken_log_h_doubling_times.append(T_log_h)

    mistaken_log_h_doubling_times = np.array(mistaken_log_h_doubling_times)

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(years, intended_doubling_times, 'r-', linewidth=2.5,
            label='Intended (horizon-doubling)', alpha=0.8)
    ax.plot(years, mistaken_doubling_times, 'g--', linewidth=2,
            label='Mistaken (horizon-doubling)', alpha=0.8)
    ax.plot(years, mistaken_log_h_doubling_times, 'b:', linewidth=2,
            label='Mistaken (log-horizon doubling)', alpha=0.8)

    # Add vertical line at 2025.25
    ax.axvline(x=2025.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add horizontal line at 3.6 months
    ax.axhline(y=3.6, color='blue', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(2021.1, 3.8, '3.6 months', fontsize=9, color='blue')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Instantaneous Horizon-Doubling Time (months)', fontsize=12)
    ax.set_title('Instantaneous Horizon-Doubling Time: Intended vs Mistaken\n'
                 '(Both calibrated with same parameters)', fontsize=14)

    ax.set_xlim(2021, singularity_intended)
    ax.set_ylim(0, 20)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = here / "instantaneous_doubling_time.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")

    # -------------------------------------------------------------------------
    # Create table at specific years
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Instantaneous Horizon-Doubling Time at Specific Years")
    print("=" * 80)

    table_years = [2021, 2022, 2023, 2024, 2025, 2025.25, 2026, 2026.5, 2027]
    table_data = []

    print(f"\n{'Year':<10} {'Intended (mo)':<16} {'Mistaken (mo)':<16} {'Mistaken log-h (mo)':<20} {'Ratio (M/I)':<12}")
    print("-" * 75)

    for yr in table_years:
        if yr >= singularity_intended:
            continue

        t = yr - base_year

        # Intended
        T_int = (T1_h - t * (1 - q)) * 12

        # Mistaken - horizon doubling time
        arg = 1 - t * (1 - q) / T1_y
        if arg > 0:
            n = np.log(arg) / np.log(q)
            y = y0_mistaken * (2 ** n)
            T_y_inst = T1_y - t * (1 - q)
            T_mis = (T_y_inst / (np.log(10) * B * y * np.log(2))) * 12
            # Mistaken - log-horizon doubling time (= y doubling time, since log(h) is linear in y)
            # Time for log10(h) to increase by log10(2) = time for y to increase by log10(2)/B
            # This equals the y-doubling time scaled by log10(2)/(B * y_doubling_amount)
            # But simpler: log10(h) = log_min + B*y, so d(log10(h))/dt = B * dy/dt
            # Time for log10(h) to increase by log10(2) = log10(2) / (B * dy/dt)
            # dy/dt = y * ln(2) / T_y_inst
            # So: T_log_h = log10(2) / (B * y * ln(2) / T_y_inst) = log10(2) * T_y_inst / (B * y * ln(2))
            T_log_h = (np.log10(2) * T_y_inst / (B * y * np.log(2))) * 12
        else:
            T_mis = np.nan
            T_log_h = np.nan

        ratio = T_mis / T_int if (T_int > 0 and not np.isnan(T_mis)) else np.nan

        table_data.append({
            'Year': yr,
            'Intended (months)': T_int,
            'Mistaken (months)': T_mis,
            'Mistaken log-h (months)': T_log_h,
            'Ratio (M/I)': ratio,
        })

        print(f"{yr:<10} {T_int:<16.2f} {T_mis:<16.2f} {T_log_h:<20.2f} {ratio:<12.2f}")

    # Save table to CSV
    df_table = pd.DataFrame(table_data)
    table_path = here / "instantaneous_doubling_time_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"\nSaved table to: {table_path}")

    # -------------------------------------------------------------------------
    # Also save full time series to CSV
    # -------------------------------------------------------------------------
    df_series = pd.DataFrame({
        'year': years,
        'intended_doubling_time_months': intended_doubling_times,
        'mistaken_doubling_time_months': mistaken_doubling_times,
    })
    series_path = here / "instantaneous_doubling_time_series.csv"
    df_series.to_csv(series_path, index=False)
    print(f"Saved time series to: {series_path}")


if __name__ == "__main__":
    main()
