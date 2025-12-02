"""Plot generated illustrative trend against reference points.

Usage:
    python plot_illustrative_trend.py [reference_csv]
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generate_illustrative_trend import generate_trend


def format_minutes_label(minutes: float) -> str:
    """Format minutes as human-readable time for axis labels."""
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    elif minutes < 60:
        return f"{minutes:.0f}m"
    elif minutes < 1440:
        return f"{minutes / 60:.0f}h"
    elif minutes < 10080:
        return f"{minutes / 1440:.0f}d"
    elif minutes < 43200:
        return f"{minutes / 10080:.0f}w"
    elif minutes < 525600:
        return f"{minutes / 43200:.0f}mo"
    else:
        return f"{minutes / 525600:.1f}y"


def main():
    # Generate trend
    generated = generate_trend()

    # Load reference if provided
    reference = None
    if len(sys.argv) >= 2:
        reference_path = Path(sys.argv[1])
        if reference_path.exists():
            reference = pd.read_csv(reference_path)
            print(f"Loaded reference from: {reference_path}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot generated trend
    ax.plot(generated['year'], generated['horizon_minutes'],
            'b-', linewidth=2, label='Generated trend')

    # Plot reference points if available
    if reference is not None:
        ax.scatter(reference['year'], reference['horizon_minutes'],
                   c='red', s=100, zorder=5, label='Reference points')

    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Horizon (minutes)', fontsize=12)
    ax.set_title('Illustrative SE Trend: Generated vs Reference', fontsize=14)

    # Custom y-axis ticks with human-readable labels
    y_ticks = [1/60, 1, 60, 1440, 10080, 43200, 525600, 5256000]  # 1s, 1m, 1h, 1d, 1w, 1mo, 1y, 10y
    y_labels = ['1s', '1m', '1h', '1d', '1w', '1mo', '1y', '10y']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Grid
    ax.grid(True, alpha=0.3, which='both')

    # Legend
    ax.legend(loc='upper left')

    # Set reasonable y-axis limits
    y_min = min(generated['horizon_minutes'].min(),
                reference['horizon_minutes'].min() if reference is not None else float('inf'))
    y_max = max(generated['horizon_minutes'].max(),
                reference['horizon_minutes'].max() if reference is not None else 0)
    ax.set_ylim(y_min * 0.5, y_max * 2)

    plt.tight_layout()

    # Save with name based on reference CSV
    if len(sys.argv) >= 2:
        csv_name = Path(sys.argv[1]).stem
        output_name = f"{csv_name}_comparison.png"
    else:
        output_name = "illustrative_trend_comparison.png"
    output_path = Path(__file__).parent / output_name
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
