"""Load training run history from debug_metrics.csv files.

Scans logs/skrl/ for all runs with CSV data, extracts key metrics,
and returns JSON-serializable data for the dashboard.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================
LOGS_DIR = Path(__file__).resolve().parents[4] / "logs" / "skrl"

# Minimum rows to consider a run meaningful
MIN_ROWS = 50

# Key metrics to extract (CSV column → short key)
METRIC_MAP = {
    "timestep": "timestep",
    "Curriculum / success_rate": "success_rate",
    "Curriculum / collision_rate": "collision_rate",
    "Curriculum / timeout_rate": "timeout_rate",
    "Curriculum / stage": "stage",
    "Curriculum / num_goals": "num_goals",
    "Curriculum / num_obstacles_static": "num_obs_static",
    "Curriculum / num_obstacles_dynamic": "num_obs_dynamic",
    "Curriculum / gamma": "gamma",
    "Curriculum / episode_length_s": "episode_length_s",
    "Curriculum / upgrade_pass_count": "upgrade_pass",
    "Reward / Total reward (mean)": "reward_mean",
    "Reward / Total reward (max)": "reward_max",
    "Loss / Policy loss": "policy_loss",
    "Loss / Value loss": "value_loss",
    "Loss / Entropy loss": "entropy_loss",
    "Learning / Learning rate": "learning_rate",
    "Episode / Total timesteps (mean)": "ep_len_mean",
    "nav/success_rate": "nav_sr",
    "nav/collision_rate": "nav_cr",
    "nav/timeout_rate": "nav_to",
    "robot/speed_mean": "speed_mean",
    "robot/min_obstacle_dist_mean": "min_obs_dist",
    "robot/progress_per_step": "progress_per_step",
    "train/fps": "fps",
    "train/elapsed_time": "elapsed_time",
    "train/actor_grad_norm": "actor_grad_norm",
    "train/critic_grad_norm": "critic_grad_norm",
}

# Downsample target: max points per run for frontend performance
MAX_POINTS = 500


def _downsample(data: list[dict], max_points: int = MAX_POINTS) -> list[dict]:
    """Downsample time series to max_points using strided selection."""
    n = len(data)
    if n <= max_points:
        return data
    stride = n / max_points
    return [data[int(i * stride)] for i in range(max_points)]


def _parse_float(val: str) -> float | None:
    """Safe float parse."""
    if not val or val in ("", "nan", "NaN", "inf", "-inf"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _run_name(task_dir: str, run_dir: str) -> str:
    """Generate human-readable run name."""
    task_short = task_dir.replace("Isaac-Navigation-Charge-", "")
    # Named runs (not timestamp-based)
    if not run_dir[:4].isdigit():
        return f"{task_short}/{run_dir}"
    # Timestamp runs: use shorter format
    return f"{task_short}/{run_dir}"


def load_run_history() -> dict:
    """Scan all training runs and load key metrics.

    Returns:
        dict with:
          - runs: list of run summaries (name, task, rows, final metrics)
          - series: dict of run_id → time series data
    """
    if not LOGS_DIR.exists():
        return {"runs": [], "series": {}}

    runs = []
    series = {}

    for task_dir in sorted(LOGS_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "play":
            continue

        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            csv_path = run_dir / "debug_metrics.csv"
            if not csv_path.exists():
                continue

            # Count rows first (fast check)
            try:
                with open(csv_path) as f:
                    row_count = sum(1 for _ in f) - 1  # minus header
                if row_count < MIN_ROWS:
                    continue
            except Exception:
                continue

            # Parse CSV
            try:
                data = _load_csv(csv_path)
            except Exception:
                continue

            if not data:
                continue

            run_id = f"{task_dir.name}/{run_dir.name}"
            display_name = _run_name(task_dir.name, run_dir.name)

            # Final metrics (last row)
            last = data[-1]
            first = data[0]

            # Compute elapsed time
            elapsed_s = last.get("elapsed_time")
            elapsed_str = ""
            if elapsed_s is not None:
                h = int(elapsed_s // 3600)
                m = int((elapsed_s % 3600) // 60)
                elapsed_str = f"{h}h{m:02d}m"

            run_info = {
                "id": run_id,
                "name": display_name,
                "task": task_dir.name,
                "run_dir": run_dir.name,
                "rows": row_count,
                "timesteps": int(last.get("timestep", 0) or 0),
                "elapsed": elapsed_str,
                "final_sr": last.get("success_rate"),
                "final_cr": last.get("collision_rate"),
                "final_to": last.get("timeout_rate"),
                "final_stage": last.get("stage"),
                "final_reward": last.get("reward_mean"),
                "max_stage": max((d.get("stage") for d in data if d.get("stage") is not None), default=None),
            }
            runs.append(run_info)

            # Store downsampled series
            series[run_id] = _downsample(data)

    # Sort: most recent first, then by row count
    runs.sort(key=lambda r: (-r["rows"], r["id"]))

    return {"runs": runs, "series": series}


def _load_csv(path: Path) -> list[dict]:
    """Load CSV and extract key metrics."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for csv_col, key in METRIC_MAP.items():
                val = _parse_float(raw.get(csv_col, ""))
                if val is not None:
                    row[key] = val
            if row.get("timestep") is not None:
                rows.append(row)
    return rows
