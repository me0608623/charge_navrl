"""Shared vehicle ``speed_rate`` semantics for training and evaluation.

The deployed vehicle applies the rate in two places: it reduces all four
action limits and rescales selected raw observation channels before the frozen
observation normalizer. LiDAR remains unchanged in the deployed ``ego`` mode.
"""

from __future__ import annotations

import math
from typing import Any

import torch


ACTION_LIMIT_FIELDS = (
    "max_linear_velocity",
    "max_linear_accel",
    "max_angular_vel",
    "max_angular_accel",
)
OBS_MODES = frozenset({"none", "ego", "ego_lidar"})
LIDAR_START = 6
LIDAR_END = 78
ACTION_HISTORY_START = 79
ACTION_HISTORY_END = 83
LIDAR_MAX_RANGE_M = 20.0
ROBOT_RADIUS_M = 0.35


def validate_vehicle_speed_rate(rate: float, obs_mode: str) -> tuple[float, str]:
    """Validate and canonicalize the deployment speed-rate contract."""
    rate = float(rate)
    obs_mode = str(obs_mode)
    if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("speed_rate must be finite and in (0, 1]")
    if obs_mode not in OBS_MODES:
        raise ValueError(
            f"speed_rate_obs must be one of {sorted(OBS_MODES)}, got {obs_mode!r}"
        )
    return rate, obs_mode


def speed_rate_is_active(rate: float) -> bool:
    """Match the existing vehicle/evaluator activation threshold."""
    return float(rate) < 0.999


def apply_vehicle_speed_rate_action_limits(
    action_cfg: Any,
    rate: float,
) -> dict[str, Any]:
    """Scale the four action limits exactly as the deployed vehicle does."""
    rate, _ = validate_vehicle_speed_rate(rate, "none")
    deployment_scale = float(
        getattr(action_cfg, "deployment_speed_scale", 1.0)
    )
    if speed_rate_is_active(rate) and not math.isclose(
        deployment_scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "speed_rate requires deployment_speed_scale=1.0 to avoid double scaling; "
            f"got {deployment_scale}"
        )

    before: dict[str, float] = {}
    after: dict[str, float] = {}
    for field in ACTION_LIMIT_FIELDS:
        value = getattr(action_cfg, field, None)
        if value is None:
            raise ValueError(f"diff_drive action config is missing {field}")
        before[field] = float(value)
        after[field] = (
            before[field] * rate if speed_rate_is_active(rate) else before[field]
        )
        setattr(action_cfg, field, after[field])

    return {
        "rate": rate,
        "deployment_speed_scale": deployment_scale,
        "before": before,
        "after": after,
    }


def apply_vehicle_speed_rate_observation(
    observation: torch.Tensor,
    rate: float,
    obs_mode: str,
) -> torch.Tensor:
    """Apply vehicle raw-observation scaling before normalization.

    ``ego`` reproduces the current vehicle behavior. ``ego_lidar`` is retained
    only for the pre-existing evaluator ablation and is not used by the 0.7
    training pilot.
    """
    rate, obs_mode = validate_vehicle_speed_rate(rate, obs_mode)
    if not speed_rate_is_active(rate) or obs_mode == "none":
        return observation
    if observation.shape[-1] < LIDAR_END:
        raise ValueError(
            "speed_rate observation transform requires at least 78 channels; "
            f"got {observation.shape[-1]}"
        )

    inv = 1.0 / rate
    transformed = observation.clone()
    transformed[..., 0:3] = torch.clamp(
        transformed[..., 0:3] * inv, -1.0, 1.0
    )

    goal = transformed[..., 4:6] * inv
    goal_norm = goal.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
    transformed[..., 4:6] = torch.where(
        goal_norm > 18.0,
        goal * (18.0 / goal_norm),
        goal,
    )

    if transformed.shape[-1] >= ACTION_HISTORY_END:
        transformed[..., ACTION_HISTORY_START:ACTION_HISTORY_END] *= inv

    if obs_mode == "ego_lidar":
        lidar_m = (
            transformed[..., LIDAR_START:LIDAR_END] * LIDAR_MAX_RANGE_M
            + ROBOT_RADIUS_M
        )
        lidar_m *= inv
        transformed[..., LIDAR_START:LIDAR_END] = torch.clamp(
            (lidar_m - ROBOT_RADIUS_M) / LIDAR_MAX_RANGE_M,
            0.0,
            1.0,
        )

    return transformed
