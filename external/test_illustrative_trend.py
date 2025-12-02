"""Test script for comparing generated illustrative trend against reference points.

Usage:
    python test_illustrative_trend.py <reference_csv>

The reference CSV should have columns: year, horizon_minutes
Each row is a reference point that will be compared against the generated trend.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from generate_illustrative_trend import generate_trend


def load_reference(reference_path: Path) -> pd.DataFrame:
    """Load reference CSV with year and horizon_minutes columns."""
    df = pd.read_csv(reference_path)
    if 'year' not in df.columns or 'horizon_minutes' not in df.columns:
        raise ValueError("Reference CSV must have 'year' and 'horizon_minutes' columns")
    return df


def compare_at_reference_points(generated: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Compare generated trend against each reference point.

    For each reference point, interpolates the generated trend to find the
    expected horizon_minutes at that year.
    """
    results = []

    for _, ref_row in reference.iterrows():
        year = ref_row['year']
        ref_minutes = ref_row['horizon_minutes']

        # Interpolate generated trend at this year
        gen_minutes = np.interp(year, generated['year'], generated['horizon_minutes'])

        ratio = gen_minutes / ref_minutes
        log_diff = np.log10(gen_minutes) - np.log10(ref_minutes)

        results.append({
            'year': year,
            'reference': ref_minutes,
            'generated': gen_minutes,
            'ratio': ratio,
            'log_diff': log_diff,
        })

    return pd.DataFrame(results)


def format_minutes(minutes: float) -> str:
    """Format minutes as human-readable time."""
    if minutes < 1:
        return f"{minutes * 60:.1f} sec"
    elif minutes < 60:
        return f"{minutes:.1f} min"
    elif minutes < 1440:
        return f"{minutes / 60:.1f} hr"
    elif minutes < 10080:
        return f"{minutes / 1440:.1f} days"
    elif minutes < 43200:
        return f"{minutes / 10080:.1f} weeks"
    elif minutes < 525600:
        return f"{minutes / 43200:.1f} months"
    else:
        return f"{minutes / 525600:.1f} years"


def print_comparison(comparison: pd.DataFrame):
    """Print comparison results."""
    print("=" * 80)
    print("COMPARISON AT REFERENCE POINTS")
    print("=" * 80)

    print(f"\n{'Year':>8}  {'Reference':>14}  {'Generated':>14}  {'Ratio':>8}  {'Log Diff':>10}")
    print(f"{'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*10}")

    for _, row in comparison.iterrows():
        print(f"{row['year']:8.2f}  {format_minutes(row['reference']):>14}  {format_minutes(row['generated']):>14}  {row['ratio']:8.3f}  {row['log_diff']:+10.3f}")

    print(f"\nSummary:")
    print(f"  Mean ratio:     {comparison['ratio'].mean():.3f}")
    print(f"  Mean log diff:  {comparison['log_diff'].mean():+.3f} ({10**comparison['log_diff'].mean():.2f}x)")
    print(f"  Max |log diff|: {comparison['log_diff'].abs().max():.3f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo reference CSV provided. Generating trend only...")
        df = generate_trend()
        print(f"\nGenerated {len(df)} points")
        print(f"Year range: {df['year'].min():.3f} - {df['year'].max():.3f}")
        print(f"y range: {df['y'].min():.3f} - {df['y'].max():.3f}")
        print(f"horizon_minutes range: {format_minutes(df['horizon_minutes'].min())} - {format_minutes(df['horizon_minutes'].max())}")
        print("\nFirst few rows:")
        print(df.head(5).to_string(index=False))
        print("\nLast few rows:")
        print(df.tail(5).to_string(index=False))
        return

    reference_path = Path(sys.argv[1])
    if not reference_path.exists():
        print(f"Error: Reference file not found: {reference_path}")
        sys.exit(1)

    print(f"Loading reference from: {reference_path}")
    reference = load_reference(reference_path)

    print("Generating trend...")
    generated = generate_trend()

    print("Comparing at reference points...\n")
    comparison = compare_at_reference_points(generated, reference)

    print_comparison(comparison)


if __name__ == "__main__":
    main()
