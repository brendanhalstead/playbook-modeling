#!/usr/bin/env python
"""Analyze simulation trajectories for anomalies.

This script executes the simplified SC simulation defined in
``timelines-rd-speedup-fix/simple_forecasting_timelines.py`` (or a custom
config), gathers *all* generated trajectories, and feeds them into the
``detect_anomalous_trajectories`` utility (defined under
``tests/test_trajectory_anomalies.py``).

It then prints a concise summary showing:
    • total number of trajectories examined,
    • how many were flagged as anomalous,
    • counts per anomaly category,
    • and optional verbose details mapping trajectory indices back to their
      forecaster names.

Example
-------
$ python trajectory_anomaly_analysis.py --config simple_params.yaml --verbose
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure project root (two levels up) is on the import path so that the `tests`
# package can be discovered when installed in editable mode, or locate the test
# module manually otherwise.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
#  Import simulation machinery from the local module
# ---------------------------------------------------------------------------
from simple_forecasting_timelines import (  # noqa: E402  pylint: disable=wrong-import-position
    load_config,
    apply_inheritance_to_forecasters,
    get_distribution_samples,
    calculate_sc_arrival_year_with_trajectories,
)

# ---------------------------------------------------------------------------
#  Import anomaly detector – try the normal import first, fall back to manual
#  loading if the `tests` directory is not a Python package.
# ---------------------------------------------------------------------------
try:
    from tests.test_trajectory_anomalies import (  # type: ignore  # noqa: E402
        detect_anomalous_trajectories,
    )
except ModuleNotFoundError:
    import importlib.util

    test_file = PROJECT_ROOT / "tests" / "test_trajectory_anomalies.py"
    if not test_file.exists():
        raise FileNotFoundError(
            "Could not locate test_trajectory_anomalies.py – "
            "make sure the tests directory exists."
        )

    spec = importlib.util.spec_from_file_location("test_trajectory_anomalies", test_file)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None  # mypy assurance
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    detect_anomalous_trajectories = module.detect_anomalous_trajectories  # type: ignore

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def summarize_anomalies(anomaly_dict: dict[int, List[str]]) -> Counter:
    """Return a counter with the frequency of each anomaly label."""
    return Counter(label for labels in anomaly_dict.values() for label in labels)


# ---------------------------------------------------------------------------
#  Main routine
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401 (imperative mood)
    """Run simulation, detect and report anomalous trajectories."""
    parser = argparse.ArgumentParser(description="Detect anomalous trajectories in SC simulation output.")
    parser.add_argument(
        "--config",
        default="simple_params.yaml",
        help="Path to a YAML config (relative to the trajectory script directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed mapping of anomalous trajectory indices → (forecaster, simulation_index, anomaly_tags).",
    )
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------
    #  Load config & resolve inheritance
    # ---------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).with_name(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))
    config["forecasters"] = apply_inheritance_to_forecasters(config["forecasters"])

    # ---------------------------------------------------------------------
    #  Generate trajectories for every forecaster & simulation
    # ---------------------------------------------------------------------
    all_trajectories: List[List[Tuple[float, float]]] = []
    trajectory_meta: List[Tuple[str, int]] = []  # (forecaster_name, simulation_idx)
    trajectory_params: List[dict[str, float | bool]] = []  # flattened sample values per trajectory

    sim_cfg = config["simulation"]

    for forecaster_key, fc_cfg in config["forecasters"].items():
        name = fc_cfg.get("name", forecaster_key)
        print(f"[+] Generating trajectories for forecaster: {name}")

        samples = get_distribution_samples(fc_cfg, sim_cfg["n_sims"])

        # The function returns (ending_times, trajectories)
        _, trajectories = calculate_sc_arrival_year_with_trajectories(
            samples,
            sim_cfg["current_horizon"],
            sim_cfg["dt"],
            sim_cfg["compute_decrease_date"],
            sim_cfg["human_alg_progress_decrease_date"],
            sim_cfg["max_simulation_years"],
        )

        for idx, traj in enumerate(trajectories):
            all_trajectories.append(traj)
            trajectory_meta.append((name, idx))

            # Capture parameter values for this simulation index
            param_snapshot: dict[str, float | bool] = {}
            for key, arr in samples.items():
                if isinstance(arr, np.ndarray):
                    # Extract the scalar value for this simulation; ensure it's python-native for readability
                    val = arr[idx]
                    if isinstance(val, np.generic):  # type: ignore[attr-defined]
                        val = val.item()
                    param_snapshot[key] = val
                else:
                    param_snapshot[key] = arr  # constant across sims
            trajectory_params.append(param_snapshot)

    # ---------------------------------------------------------------------
    #  Detect anomalies & summarise
    # ---------------------------------------------------------------------
    anomalies = detect_anomalous_trajectories(all_trajectories)

    print("\n=================== Anomaly Summary ===================")
    print(f"Total trajectories examined : {len(all_trajectories):,}")
    print(f"Trajectories with anomalies : {len(anomalies):,}")

    counts = summarize_anomalies(anomalies)
    for label, count in counts.items():
        print(f"  • {label:24s}: {count}")

    if args.verbose and anomalies:
        print("\nDetailed list (index, forecaster, sim_idx, tags):")
        for t_idx, tags in sorted(anomalies.items()):
            forecaster, sim_idx = trajectory_meta[t_idx]
            params = trajectory_params[t_idx]

            # Format params: key=value (floats rounded for brevity)
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:.4g}"
                return str(v)

            param_str = ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())

            print(
                f"  #{t_idx:6d} | {forecaster:20s} | {sim_idx:5d} | {', '.join(tags)}\n"
                f"             params: {param_str}"
            )


if __name__ == "__main__":  # pragma: no cover
    main() 