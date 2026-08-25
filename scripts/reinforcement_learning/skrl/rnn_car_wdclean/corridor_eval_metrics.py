"""Pure metrics for deployment-corridor action quality."""

from __future__ import annotations

import torch


STOP_SPEED_MPS = 0.10
LOW_SPEED_MPS = 0.12
REVERSE_SPEED_MPS = -0.02
HIGH_TURN_RAD_S = 0.80
EXTREME_TURN_RAD_S = 1.00


def corridor_clear_mask(
    policy_obs: torch.Tensor,
    *,
    goal_ahead_deg: float = 15.0,
    front_clear_m: float = 2.5,
    lidar_scale_m: float = 20.0,
) -> torch.Tensor:
    """Select frames where the goal is ahead and the front cone is clear."""
    if policy_obs.ndim != 2 or policy_obs.shape[1] < 49:
        raise ValueError(
            "policy_obs must have shape [frames, >=49], got "
            f"{tuple(policy_obs.shape)}"
        )
    goal_bearing = torch.atan2(
        policy_obs[:, 5],
        policy_obs[:, 4],
    ).abs()
    # LiDAR starts at policy index 6; bins 30:43 are the +/-30 degree cone.
    front = policy_obs[:, 6 + 30:6 + 43]
    front = torch.where(
        front < 0.02,
        torch.full_like(front, float("inf")),
        front,
    )
    front_m = front.min(dim=1).values * float(lidar_scale_m)
    return (
        goal_bearing < (float(goal_ahead_deg) * torch.pi / 180.0)
    ) & (front_m > float(front_clear_m))


def summarize_corridor_actions(
    actions: torch.Tensor,
    *,
    stop_speed_mps: float = STOP_SPEED_MPS,
    low_speed_mps: float = LOW_SPEED_MPS,
    reverse_speed_mps: float = REVERSE_SPEED_MPS,
    high_turn_rad_s: float = HIGH_TURN_RAD_S,
    extreme_turn_rad_s: float = EXTREME_TURN_RAD_S,
) -> dict[str, float | int | None]:
    """Summarize applied ``(v, omega)`` commands without simulator state."""
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError(
            "actions must have shape [frames, >=2], got "
            f"{tuple(actions.shape)}"
        )
    finite = torch.isfinite(actions[:, :2]).all(dim=1)
    values = actions[finite, :2].float()
    if values.shape[0] == 0:
        return {
            "action_frames": 0,
            "linear_speed_mean_mps": None,
            "linear_speed_abs_mean_mps": None,
            "angular_abs_mean_rad_s": None,
            "angular_abs_p50_rad_s": None,
            "angular_abs_p90_rad_s": None,
            "angular_abs_p95_rad_s": None,
            "high_turn_fraction": None,
            "extreme_turn_fraction": None,
            "stop_command_fraction": None,
            "reverse_command_fraction": None,
            "low_speed_high_turn_fraction": None,
        }

    linear = values[:, 0]
    angular_abs = values[:, 1].abs()
    quantiles = torch.quantile(
        angular_abs,
        torch.tensor(
            [0.50, 0.90, 0.95],
            dtype=angular_abs.dtype,
            device=angular_abs.device,
        ),
    )
    high_turn = angular_abs > float(high_turn_rad_s)
    return {
        "action_frames": int(values.shape[0]),
        "linear_speed_mean_mps": float(linear.mean().item()),
        "linear_speed_abs_mean_mps": float(linear.abs().mean().item()),
        "angular_abs_mean_rad_s": float(angular_abs.mean().item()),
        "angular_abs_p50_rad_s": float(quantiles[0].item()),
        "angular_abs_p90_rad_s": float(quantiles[1].item()),
        "angular_abs_p95_rad_s": float(quantiles[2].item()),
        "high_turn_fraction": float(high_turn.float().mean().item()),
        "extreme_turn_fraction": float(
            (angular_abs > float(extreme_turn_rad_s)).float().mean().item()
        ),
        "stop_command_fraction": float(
            (linear.abs() < float(stop_speed_mps)).float().mean().item()
        ),
        "reverse_command_fraction": float(
            (linear < float(reverse_speed_mps)).float().mean().item()
        ),
        "low_speed_high_turn_fraction": float(
            (
                (linear.abs() < float(low_speed_mps))
                & high_turn
            ).float().mean().item()
        ),
    }
