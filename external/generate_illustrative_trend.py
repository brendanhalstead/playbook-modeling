"""Generate illustrative SE trends with superexponential growth.

This script generates y-values using superexponential growth models.

Formula 1 (doubling time model):
    t = year - baseYear
    n = log(1 - t*(1-q)/T1) / log(q)
    y = y0 * 2^n

Formula 2 (exponential with power model):
    t = year - baseYear
    y = a + b * e^(r * t^p)

Supports three configurations:
1. current_website_trend: baseYear=2021, y0=1.45, T1=1.91, q=0.85
   y=0 -> 4 sec, y=23 -> 5 work years
2. 50p_update_tweeted: baseYear=2019, y0=2.4, T1=2.91, q=0.85
   y=0 -> 1 sec, y=26 -> 10 work years
3. original_graph: baseYear=2021, a=0, b=1, r=0.395, p=1.15
   y=0 -> 4 sec, y=23 -> 5 work years
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------------------------------------------------------
# Time conversion: linear interpolation in log space
# -----------------------------------------------------------------------------

def convert_y_to_horizon_minutes(
    y_values: np.ndarray,
    y_min: float = 0,
    y_max: float = 23,
    minutes_at_y_min: float = 4 / 60,  # 4 seconds
    minutes_at_y_max: float = 5 * 2000 * 60,  # 5 work years
) -> np.ndarray:
    """Convert y values to horizon_minutes using linear interpolation in log space."""
    log_min = np.log10(minutes_at_y_min)
    log_max = np.log10(minutes_at_y_max)

    # Linear interpolation in log space
    log_minutes = log_min + (log_max - log_min) * (y_values - y_min) / (y_max - y_min)
    return 10 ** log_minutes


def generate_trend(
    base_year: float = 2021,
    start_value: float = 1.45,
    initial_doubling_time: float = 1.91,
    doubling_factor_q: float = 0.85,
    max_y_value: float = 24,
    y_min: float = 0,
    y_max: float = 23,
    minutes_at_y_min: float = 4 / 60,
    minutes_at_y_max: float = 5 * 2000 * 60,
) -> pd.DataFrame:
    """Generate trend data using superexponential growth model.

    Args:
        base_year: Starting year for t calculation
        start_value: Initial y-value (y0)
        initial_doubling_time: Time for first doubling in years (T1)
        doubling_factor_q: Each doubling takes this fraction of the previous time (q)
        max_y_value: Upper bound for y values
        y_min: y value corresponding to minutes_at_y_min
        y_max: y value corresponding to minutes_at_y_max
        minutes_at_y_min: horizon_minutes at y_min
        minutes_at_y_max: horizon_minutes at y_max

    Returns:
        DataFrame with 'year', 'y', and 'horizon_minutes' columns
    """
    # Read the original trend file to get the year values
    here = Path(__file__).resolve().parent
    original_path = here / "illustrative_se_trend.csv"
    original_df = pd.read_csv(original_path)
    years = original_df['year'].values

    # Calculate y values using the superexponential formula
    y_values = []
    for year in years:
        t = year - base_year

        # Handle edge case at t=0
        if t == 0:
            y = start_value
        else:
            # n = log(1 - t*(1-q)/T1) / log(q)
            # y = y0 * 2^n
            q = doubling_factor_q
            T1 = initial_doubling_time
            y0 = start_value

            argument = 1 - t * (1 - q) / T1

            # Check if we've hit the singularity (argument <= 0)
            if argument <= 0:
                y = max_y_value
            else:
                n = np.log(argument) / np.log(q)
                y = y0 * (2 ** n)

                # Clamp to max value
                if y > max_y_value:
                    y = max_y_value

        y_values.append(y)

    df = pd.DataFrame({'year': years, 'y': y_values})

    # Convert y to horizon_minutes
    df['horizon_minutes'] = convert_y_to_horizon_minutes(
        np.array(y_values),
        y_min=y_min,
        y_max=y_max,
        minutes_at_y_min=minutes_at_y_min,
        minutes_at_y_max=minutes_at_y_max,
    )

    return df


def generate_trend_exp_power(
    base_year: float = 2021,
    a: float = 0,
    b: float = 1,
    r: float = 0.395,
    p: float = 1.15,
    max_y_value: float = 24,
    y_min: float = 0,
    y_max: float = 23,
    minutes_at_y_min: float = 4 / 60,
    minutes_at_y_max: float = 5 * 2000 * 60,
) -> pd.DataFrame:
    """Generate trend data using exponential with power model.

    Formula: y = a + b * e^(r * t^p)
    where t = year - base_year

    Args:
        base_year: Starting year for t calculation
        a: Vertical shift
        b: Scale factor
        r: Base growth rate
        p: Curvature parameter (superexponential when p > 1)
        max_y_value: Upper bound for y values
        y_min: y value corresponding to minutes_at_y_min
        y_max: y value corresponding to minutes_at_y_max
        minutes_at_y_min: horizon_minutes at y_min
        minutes_at_y_max: horizon_minutes at y_max

    Returns:
        DataFrame with 'year', 'y', and 'horizon_minutes' columns
    """
    # Read the original trend file to get the year values
    here = Path(__file__).resolve().parent
    original_path = here / "illustrative_se_trend.csv"
    original_df = pd.read_csv(original_path)
    years = original_df['year'].values

    # Calculate y values using the exponential power formula
    y_values = []
    for year in years:
        t = year - base_year

        # y = a + b * e^(r * t^p)
        if t <= 0:
            y = a + b  # e^0 = 1
        else:
            y = a + b * np.exp(r * (t ** p))

        # Clamp to max value
        if y > max_y_value:
            y = max_y_value

        y_values.append(y)

    df = pd.DataFrame({'year': years, 'y': y_values})

    # Convert y to horizon_minutes
    df['horizon_minutes'] = convert_y_to_horizon_minutes(
        np.array(y_values),
        y_min=y_min,
        y_max=y_max,
        minutes_at_y_min=minutes_at_y_min,
        minutes_at_y_max=minutes_at_y_max,
    )

    return df


# -----------------------------------------------------------------------------
# Configuration presets
# -----------------------------------------------------------------------------

CURRENT_WEBSITE_CONFIG = {
    'base_year': 2021,
    'start_value': 1.45,
    'initial_doubling_time': 1.91,
    'doubling_factor_q': 0.85,
    'max_y_value': 24,
    'y_min': 0,
    'y_max': 23,
    'minutes_at_y_min': 4 / 60,  # 4 seconds
    'minutes_at_y_max': 5 * 2000 * 60,  # 5 work years (2000 hours/year)
    'comparison_csv': 'current_website_trend.csv',
    'output_csv': 'illustrative_se_trend_generated.csv',
}

FIFTY_PERCENT_UPDATE_CONFIG = {
    'base_year': 2019,
    'start_value': 2.4,
    'initial_doubling_time': 2.91,
    'doubling_factor_q': 0.85,
    'max_y_value': 30,
    'y_min': 0,
    'y_max': 26,
    'minutes_at_y_min': 1 / 60,  # 1 second
    'minutes_at_y_max': 10 * 50 * 40 * 60,  # 10 work years
    'comparison_csv': '50p_update_tweeted.csv',
    'output_csv': '50p_update_trend_generated.csv',
    'model': 'doubling',
}

ORIGINAL_GRAPH_CONFIG = {
    'base_year': 2021,
    'a': 0,
    'b': 1,
    'r': 0.395,
    'p': 1.15,
    'max_y_value': 24,
    'y_min': 0,
    'y_max': 23,
    'minutes_at_y_min': 4 / 60,  # 4 seconds
    'minutes_at_y_max': 5 * 2000 * 60,  # 5 work years (2000 hours/year)
    'comparison_csv': 'original_graph.csv',
    'output_csv': 'original_graph_trend_generated.csv',
    'model': 'exp_power',
}


def generate_and_save(config: dict, verbose: bool = True) -> pd.DataFrame:
    """Generate trend using config and save to CSV."""
    here = Path(__file__).resolve().parent

    model = config.get('model', 'doubling')

    if model == 'exp_power':
        df = generate_trend_exp_power(
            base_year=config['base_year'],
            a=config['a'],
            b=config['b'],
            r=config['r'],
            p=config['p'],
            max_y_value=config['max_y_value'],
            y_min=config['y_min'],
            y_max=config['y_max'],
            minutes_at_y_min=config['minutes_at_y_min'],
            minutes_at_y_max=config['minutes_at_y_max'],
        )
    else:
        df = generate_trend(
            base_year=config['base_year'],
            start_value=config['start_value'],
            initial_doubling_time=config['initial_doubling_time'],
            doubling_factor_q=config['doubling_factor_q'],
            max_y_value=config['max_y_value'],
            y_min=config['y_min'],
            y_max=config['y_max'],
            minutes_at_y_min=config['minutes_at_y_min'],
            minutes_at_y_max=config['minutes_at_y_max'],
        )

    output_path = here / config['output_csv']
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"Generated trend saved to: {output_path}")
        print(f"\nParameters used:")
        print(f"  baseYear = {config['base_year']}")
        if model == 'exp_power':
            print(f"  a = {config['a']}")
            print(f"  b = {config['b']}")
            print(f"  r = {config['r']}")
            print(f"  p = {config['p']}")
        else:
            print(f"  startValue (y0) = {config['start_value']}")
            print(f"  initialDoublingTime (T1) = {config['initial_doubling_time']}")
            print(f"  doublingFactorQ (q) = {config['doubling_factor_q']}")
        print(f"  y range for conversion: {config['y_min']} -> {config['y_max']}")
        print(f"\nFirst few rows:")
        print(df.head(5).to_string(index=False))
        print(f"\nLast few rows:")
        print(df.tail(5).to_string(index=False))

    return df


def plot_comparison(generated: pd.DataFrame, config: dict) -> None:
    """Generate and save comparison plot."""
    here = Path(__file__).resolve().parent

    # Load reference CSV
    reference_path = here / config['comparison_csv']
    if not reference_path.exists():
        print(f"Warning: Reference file not found: {reference_path}")
        return

    reference = pd.read_csv(reference_path)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot generated trend
    ax.plot(generated['year'], generated['horizon_minutes'],
            'b-', linewidth=2, label='Generated trend')

    # Plot reference points
    ax.scatter(reference['year'], reference['horizon_minutes'],
               c='red', s=100, zorder=5, label='Reference points')

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Horizon (work time)', fontsize=12)
    csv_name = Path(config['comparison_csv']).stem
    ax.set_title(f'Illustrative SE Trend: {csv_name}', fontsize=14)

    # Custom y-axis ticks with work time labels
    # Work time: 1 work year = 2000 hours, 1 work month = 167 hours, 1 work week = 40 hours, 1 work day = 8 hours
    y_ticks = [1/60, 1, 8*60, 40*60, 167*60, 2000*60, 20000*60]
    y_labels = ['1s', '1m', '1 work day', '1 work week', '1 work month', '1 work year', '10 work years']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Grid
    ax.grid(True, alpha=0.3, which='both')

    # Legend
    ax.legend(loc='upper left')

    # Set reasonable y-axis limits
    y_min = min(generated['horizon_minutes'].min(), reference['horizon_minutes'].min())
    y_max = max(generated['horizon_minutes'].max(), reference['horizon_minutes'].max())
    ax.set_ylim(y_min * 0.5, y_max * 2)

    plt.tight_layout()

    # Save
    output_name = f"{csv_name}_comparison.png"
    output_path = here / output_name
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


def main():
    print("=" * 60)
    print("Generating current_website_trend comparison")
    print("=" * 60)
    df1 = generate_and_save(CURRENT_WEBSITE_CONFIG)
    plot_comparison(df1, CURRENT_WEBSITE_CONFIG)

    print("\n")
    print("=" * 60)
    print("Generating 50p_update_tweeted comparison")
    print("=" * 60)
    df2 = generate_and_save(FIFTY_PERCENT_UPDATE_CONFIG)
    plot_comparison(df2, FIFTY_PERCENT_UPDATE_CONFIG)

    print("\n")
    print("=" * 60)
    print("Generating original_graph comparison")
    print("=" * 60)
    df3 = generate_and_save(ORIGINAL_GRAPH_CONFIG)
    plot_comparison(df3, ORIGINAL_GRAPH_CONFIG)


if __name__ == "__main__":
    main()
