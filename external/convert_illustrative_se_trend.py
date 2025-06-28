import pandas as pd
import numpy as np
import re
from pathlib import Path

"""convert_illustrative_se_trend.py

This utility reads the illustrative scale mapping (categorical labels -> yIndex) and the
raw illustrative_se_trend.csv file, then translates the arbitrary ``y`` units into an
approximate *time-horizon in minutes* so that the resulting curve can be plotted on the
same axes as the trajectories produced by ``simple_forecasting_timelines.py``.

The mapping from ``yIndex`` to a true duration is only *approximately* logarithmic
because the human-readable labels (e.g. ``"5 years"``) are rounded.  We therefore:

1.  Parse each ``time`` label in ``illustrative_scale.csv`` into an estimated duration
    expressed in minutes.  (The parsing logic is deliberately simple – the errors
    introduced by rounding in the original labels dominate.)
2.  Take the base-10 logarithm of these durations.
3.  Fit a simple linear regression ``log10(minutes) = a + b * yIndex``.  This gives a
    smooth, strictly monotonic mapping that respects the approximate log nature of the
    original scale while smoothing out the rounding noise.
4.  Apply that mapping to the continuous ``y`` values in ``illustrative_se_trend.csv``
    to obtain the corresponding time horizon in minutes.
5.  Write a new CSV file alongside the originals named
    ``illustrative_se_trend_converted.csv`` containing the columns:
        year, horizon_minutes

The resulting file is intentionally minimal so that downstream plotting code can read
it with one line such as:

>>> df = pd.read_csv("external/illustrative_se_trend_converted.csv")

and then scatter ``df['year']`` against ``df['horizon_minutes']`` on a logarithmic
"minutes" axis.
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_TIME_PATTERN = re.compile(r"^(?P<value>[0-9.]+)\s*(?P<unit>[A-Za-z]+)s?$", flags=re.I)

_UNIT_TO_MINUTES = {
    # seconds
    "sec": 1.0 / 60,
    "second": 1.0 / 60,
    "seconds": 1.0 / 60,
    # minutes
    "min": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    # hours
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    # days
    "day": 1440.0,
    "days": 1440.0,
    # weeks
    "week": 7 * 1440.0,
    "weeks": 7 * 1440.0,
    # months – use a 30-day month for simplicity (≈ calendar month)
    "month": 30 * 1440.0,
    "months": 30 * 1440.0,
    # years – use 365-day year
    "year": 365 * 1440.0,
    "years": 365 * 1440.0,
}


def _label_to_minutes(label: str) -> float:
    """Convert a human-readable time label (e.g. "4 sec", "1 hr") into minutes."""
    m = _TIME_PATTERN.match(label.strip())
    if not m:
        raise ValueError(f"Unrecognised time label: {label!r}")
    value = float(m.group("value"))
    unit = m.group("unit").lower()
    if unit not in _UNIT_TO_MINUTES:
        raise ValueError(f"Unsupported time unit in label {label!r}")
    return value * _UNIT_TO_MINUTES[unit]


# -----------------------------------------------------------------------------
# Main conversion logic
# -----------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent

    scale_path = here / "illustrative_scale.csv"
    trend_path = here / "illustrative_se_trend.csv"
    out_path = here / "illustrative_se_trend_converted.csv"

    if not scale_path.exists():
        raise FileNotFoundError(f"Cannot find scale file: {scale_path}")
    if not trend_path.exists():
        raise FileNotFoundError(f"Cannot find trend file: {trend_path}")

    # ---------------------------------------------------------------------
    # 1.  Load and parse the scale mapping
    # ---------------------------------------------------------------------
    scale_df = pd.read_csv(scale_path)
    # Expect columns: time, yIndex
    minutes = scale_df["time"].apply(_label_to_minutes)
    y_index = scale_df["yIndex"].astype(float)

    # ---------------------------------------------------------------------
    # 2.  Regress log10(minutes) ~ yIndex
    # ---------------------------------------------------------------------
    log_minutes = np.log10(minutes.values)
    slope, intercept = np.polyfit(y_index, log_minutes, deg=1)

    # ---------------------------------------------------------------------
    # 3.  Load continuous trend curve and convert
    # ---------------------------------------------------------------------
    trend_df = pd.read_csv(trend_path)
    # Expect columns: year, y
    y_values = trend_df["y"].astype(float).values
    log_minutes_trend = intercept + slope * y_values
    minutes_trend = 10 ** log_minutes_trend

    trend_df["horizon_minutes"] = minutes_trend

    # ---------------------------------------------------------------------
    # 4.  Save result
    # ---------------------------------------------------------------------
    trend_df[["year", "horizon_minutes"]].to_csv(out_path, index=False)
    print(f"Wrote converted trend to {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main() 