"""
Illustrates why when lowering the initial time horizon of a superexponential curve,
you should increase the initial doubling time.

This is a simplified version that shows years on the x-axis, assuming constant
effective compute growth (1 2024-year per calendar year).

This script generates 3 curves:
a) "With median parameters": Initial horizon of 15, initial doubling time of 4.74, doubling decay fraction of 0.095
b) "Only lowering initial time horizon": Initial horizon of 15 nanoseconds, otherwise same as (a)
c) "Lowering initial time horizon and increasing initial doubling time": Same as (b) except the initial
   doubling time is 4.74*1.095^(log2(15 minutes / 15 nanoseconds))
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_superexponential_curve(
    initial_horizon_minutes: float,
    initial_doubling_time_months: float,
    doubling_decay_fraction: float,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a superexponential horizon growth curve.

    The model:
    - Each doubling takes q times as long as the previous (q = 1 - doubling_decay_fraction)
    - Time for n doublings: t = T_0 * (1 - q^n) / (1 - q)
    - Finite time singularity at t_sing = T_0 / (1 - q)
    - Horizon at time t: h(t) = h_0 * 2^n(t)

    Returns:
        x_years: array of years values (starting at 2025)
        horizons: array of horizon values in minutes
        singularity_year: calendar year when singularity occurs
    """
    q = 1 - doubling_decay_fraction

    # Singularity time in months
    singularity_time_months = initial_doubling_time_months / (1 - q)
    # Convert to years from start, then to calendar year (starting at 2025)
    singularity_years_from_start = singularity_time_months / 12
    singularity_year = 2025 + singularity_years_from_start

    # Generate time points from start to just before singularity
    # Stop at 99.9% of the way to the singularity
    max_time_months = singularity_time_months * 0.999
    time_months = np.linspace(0, max_time_months, num_points)

    # Convert to calendar years (starting at 2025)
    x_years = 2025 + time_months / 12

    # Calculate number of doublings at each time point
    # From t = T_0 * (1 - q^n) / (1 - q), we get:
    # q^n = 1 - t * (1 - q) / T_0
    # n = log(1 - t * (1 - q) / T_0) / log(q)

    ratio_term = 1 - time_months * (1 - q) / initial_doubling_time_months
    # Clip to avoid log of non-positive numbers
    ratio_term = np.clip(ratio_term, 1e-15, None)
    n_doublings = np.log(ratio_term) / np.log(q)

    # Calculate horizon at each time
    horizons = initial_horizon_minutes * (2 ** n_doublings)

    return x_years, horizons, singularity_year


def add_disclaimer(fig: plt.Figure) -> None:
    """Add disclaimer text at the bottom of a figure."""
    disclaimer_text = (
        "This plot was generated using the median parameter estimates of a timelines model that was released alongside AI 2027. "
        "Since then, we have created an improved model, which predicts longer timelines."
    )
    fig.text(
        0.5, 0.01, disclaimer_text,
        ha='center', va='bottom',
        fontsize=11, color='red',
        wrap=True,
        transform=fig.transFigure
    )


def main():
    # (a) Median parameters
    initial_horizon_a = 15  # minutes
    initial_doubling_time_a = 4.74  # months
    doubling_decay_fraction = 0.095

    # (b) Only lowering initial time horizon
    # 1 ms = 1e-3 seconds = 1e-3 / 60 minutes
    initial_horizon_b = 1e-3 / 60  # 1 ms in minutes
    initial_doubling_time_b = initial_doubling_time_a  # same

    # (c) Lowering initial time horizon AND increasing initial doubling time
    # The adjustment factor: (1/q)^n where q = 1 - doubling_decay_fraction
    # This ensures the green curve's doubling time when it reaches 15 min
    # equals the blue curve's initial doubling time (exact horizontal translation)
    q = 1 - doubling_decay_fraction
    ratio = initial_horizon_a / initial_horizon_b  # = 15 minutes / 1 ms
    adjustment_exponent = np.log2(ratio)
    adjustment_factor = (1 / q) ** adjustment_exponent  # = 1.105^n instead of 1.095^n
    initial_horizon_c = initial_horizon_b
    initial_doubling_time_c = initial_doubling_time_a * adjustment_factor

    print(f"Parameters:")
    print(f"  (a) Initial horizon: {initial_horizon_a} min, Initial doubling time: {initial_doubling_time_a} months")
    print(f"  (b) Initial horizon: {initial_horizon_b:.2e} min ({initial_horizon_b * 60 * 1e3:.0f} ms), Initial doubling time: {initial_doubling_time_b} months")
    print(f"  (c) Initial horizon: {initial_horizon_c:.2e} min, Initial doubling time: {initial_doubling_time_c:.2f} months")
    print(f"\n  Ratio of horizons: {ratio:.2e}")
    print(f"  log2(ratio): {adjustment_exponent:.2f}")
    print(f"  Adjustment factor for doubling time: {adjustment_factor:.2f}")

    # Compute curves (x-axis is now calendar years starting at 2025)
    x_a, horizons_a, sing_a = compute_superexponential_curve(
        initial_horizon_a, initial_doubling_time_a, doubling_decay_fraction
    )
    x_b, horizons_b, sing_b = compute_superexponential_curve(
        initial_horizon_b, initial_doubling_time_b, doubling_decay_fraction
    )
    x_c, horizons_c, sing_c = compute_superexponential_curve(
        initial_horizon_c, initial_doubling_time_c, doubling_decay_fraction
    )

    print(f"\nSingularity (calendar year):")
    print(f"  (a) {sing_a:.2f}")
    print(f"  (b) {sing_b:.2f}")
    print(f"  (c) {sing_c:.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot curves
    ax.plot(x_a, horizons_a, 'b-', linewidth=2, label='(a) With median parameters')
    ax.plot(x_b, horizons_b, 'r-', linewidth=2, label="(b) Only lowering initial time horizon (titotal's concern)")
    ax.plot(x_c, horizons_c, 'g-', linewidth=2, label='(c) As in (b), plus slowing initial doubling time (our suggestion)')

    # Plot singularity lines
    ax.axvline(x=sing_a, color='b', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=sing_b, color='r', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=sing_c, color='g', linestyle=':', linewidth=1.5, alpha=0.7)

    # SC threshold (10 work years) - used for arrow positioning
    sc_threshold_minutes = 10 * 2000 * 60  # 10 work years in minutes (2000 hours/year)

    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Year (simplified, assuming constant effective compute growth)', fontsize=12)
    ax.set_ylabel('Coding time horizon (log scale)', fontsize=14)
    ax.set_title('If you decrease the initial time horizon of a superexponential curve,\nyou should slow the initial doubling time', fontsize=14)

    # Set reasonable y-axis limits - start just below 1 ms, end above 10 work years
    # 1 ms = 1e-3 seconds = 1e-3/60 minutes = 1.67e-5 minutes
    ax.set_ylim(1e-6, 5e7)

    # Custom y-axis ticks in MINUTES, but labels in work time
    # Work time: 1 work day = 8 hours, 1 work week = 40 hours, 1 work year = 2000 hours
    # 1 ms = 1e-3 sec = 1e-3/60 min
    # 1 work hour = 60 min
    # 1 work day = 8 * 60 = 480 min
    # 1 work week = 40 * 60 = 2400 min
    # 1 work month = 2000/12 * 60 = 10000 min
    # 1 work year = 2000 * 60 = 120000 min
    # 10 work years = 1200000 min
    y_ticks = [1e-3/60, 1/60, 1, 60, 8*60, 40*60, 2000/12*60, 2000*60, 10*2000*60]
    y_labels = ['1 ms', '1 sec', '1 min', '1 work hr', '1 work day', '1 work wk', '1 work mo', '1 work yr', '10 work yr']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Set x-axis limits (in years)
    ax.set_xlim(2024.9, max(sing_a, sing_b, sing_c) + 0.5)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Add horizontal arrows showing the shift between blue and green curves
    # Use 15 min and SC threshold as reference points
    arrow_y_values = [15, sc_threshold_minutes]  # 15 min and SC threshold
    arrow_labels = ['(at present time horizon)', '(at median human-cost-parity SC threshold)']

    for arrow_y, extra_label in zip(arrow_y_values, arrow_labels):
        # Find x where blue curve crosses this y value
        idx_a = np.argmin(np.abs(horizons_a - arrow_y))
        x_blue = x_a[idx_a]

        # Find x where green curve crosses this y value
        idx_c = np.argmin(np.abs(horizons_c - arrow_y))
        x_green = x_c[idx_c]

        # Calculate the actual shift at this horizon (in years)
        actual_shift = x_green - x_blue

        # Only draw if both points are within plot bounds
        if x_blue >= 2025 and x_green <= max(sing_a, sing_b, sing_c) + 0.5:
            # Draw arrow from blue to green, extending slightly past on the right
            arrow_extension = 0.05  # Small extension on the right side
            ax.annotate('', xy=(x_green + arrow_extension, arrow_y), xytext=(x_blue, arrow_y),
                       arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
            # Add label with shift amount and context
            mid_x = (x_blue + x_green + arrow_extension) / 2
            ax.text(mid_x, arrow_y * 1.5, f'{actual_shift:.1f} years {extra_label}',
                   ha='center', va='bottom', fontsize=9, color='purple')

    # Adjust layout: less space at top, more space at bottom for x-axis label and disclaimer
    plt.subplots_adjust(bottom=0.155, top=0.92)

    # Add disclaimer - position it lower to avoid covering x-axis label
    add_disclaimer(fig)
    plt.savefig('timelines_modeling_exploration/initial_doubling_time_adjustment_years.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Commented out to not block

    print(f"\nPlot saved to timelines_modeling_exploration/initial_doubling_time_adjustment_years.png")


if __name__ == "__main__":
    main()
