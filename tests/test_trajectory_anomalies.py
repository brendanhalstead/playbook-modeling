import numpy as np
from collections import defaultdict


def detect_anomalous_trajectories(
    trajectories,
    *,
    current_year: float = 2025.25,
    horizon_min_threshold: float = 15.0,
):
    """Identify anomalous trajectories.

    A trajectory is a list of ``(time_decimal_year, horizon_minutes)`` tuples.

    The following anomaly classes are detected:
    1. ``empty_trajectory``           – no samples in the list.
    2. ``short_initial_horizon``      – first horizon value is below the provided *horizon_min_threshold* (≤15 min by default).
    3. ``no_forecast_part``           – all timestamps are strictly *before* *current_year*, i.e. the path never enters the forecast period.
    4. ``non_monotonic_horizon``      – horizon decreases at any point (horizon should be non-decreasing over time).

    Parameters
    ----------
    trajectories
        Iterable of trajectories (each itself an iterable of time / horizon tuples).
    current_year
        Decimal year that marks the boundary between back-cast and forecast portions.
    horizon_min_threshold
        Minimum allowed initial horizon in **minutes**.

    Returns
    -------
    dict[int, list[str]]
        Mapping from *trajectory index* → list of detected anomaly tags.
    """

    anomalies = defaultdict(list)

    for idx, traj in enumerate(trajectories):
        # 1. Empty trajectory
        if not traj:
            anomalies[idx].append("empty_trajectory")
            continue

        times = np.asarray([t for t, _ in traj], dtype=float)
        horizons = np.asarray([h for _, h in traj], dtype=float)

        # 2. Initial horizon too small
        if horizons[0] < horizon_min_threshold:
            anomalies[idx].append("short_initial_horizon")

        # 3. Never moves into the forecast window
        if np.all(times < current_year):
            anomalies[idx].append("no_forecast_part")

        # 4. Horizon decreases (should be monotonic increasing)
        if np.any(np.diff(horizons) < -1e-6):
            anomalies[idx].append("non_monotonic_horizon")

    return dict(anomalies)


# -------------------------------------------------------------
#                        Unit Tests
# -------------------------------------------------------------


def _make_demo_trajectories():
    """Craft a small set of trajectories with known pathologies."""
    good = [(2025.25, 15), (2025.50, 30)]
    short_horizon = [(2025.25, 10), (2025.50, 20)]
    backcast_only = [(2024.80, 15), (2024.60, 30)]
    non_monotonic = [(2025.25, 15), (2025.50, 14)]
    empty = []
    return [good, short_horizon, backcast_only, non_monotonic, empty]


def test_detect_anomalies_on_synthetic_examples():
    trajectories = _make_demo_trajectories()
    anomalies = detect_anomalous_trajectories(trajectories)

    assert set(anomalies.keys()) == {1, 2, 3, 4}
    assert "short_initial_horizon" in anomalies[1]
    assert "no_forecast_part" in anomalies[2]
    assert "non_monotonic_horizon" in anomalies[3]
    assert "empty_trajectory" in anomalies[4]


if __name__ == "__main__":
    # Quick manual run – helpful when invoking directly
    from pprint import pprint

    pprint(detect_anomalous_trajectories(_make_demo_trajectories())) 