"""Scripted direct-crossing teacher for randomised narrow-gap scenes.

07-27 verdict: the SA5 narrow teacher itself detours to |y|~3.9 m before
crossing (3-seed n=2094, direct=0), so its KL term has been distilling the
detour. Curriculum on start distance is ruled out (direct=0 even from
1.5 m). What imitation needs is a teacher that provably crosses straight.

The first version of this module hard-coded the fixed Gate5 layout
(barrier x=0, gap y=0, goal +3, west->east only). Production narrow replay
(`events/narrow_passage_bridge.py`) randomises the gap center in
y in [-1,+1], the barrier in x in [-0.5,+0.5], and flips direction with
probability 0.5, so the teacher now reads the real per-env geometry:

    d               = sign(goal_x - barrier_x)
    pre-cross aim   = (barrier_x + d * cross_clear_x, gap_center_y)
    post-cross aim  = goal
    crossed         <=> d * (x - barrier_x) >= cross_clear_x

Heading is regulated by a proportional law on bearing error, and speed
drops to a crawl while that error is large. Desired (v', omega') are
snapped to the discrete action grid through the same decode the privileged
corridor teacher uses, so the scripted action is always reachable this
step.

Verification protocol before N1 training: three-seed randomised-replay
override with direct_crossing >= 0.99 and collision <= 0.01.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)


@dataclass(frozen=True)
class ScriptedNarrowTeacherSpec:
    # 過牆判定/瞄準點與牆的距離：拉到牆後 0.6m 讓穿越瞬間 bearing 不突變
    cross_clear_x_m: float = 0.6
    heading_gain: float = 2.0
    slow_heading_rad: float = 0.35
    crawl_speed_mps: float = 0.15
    cruise_speed_mps: float = 1.0


def _validate_spec(spec: ScriptedNarrowTeacherSpec) -> None:
    if spec.heading_gain <= 0.0:
        raise ValueError("heading_gain must be positive")
    if spec.slow_heading_rad <= 0.0:
        raise ValueError("slow_heading_rad must be positive")
    if spec.cross_clear_x_m <= 0.0:
        raise ValueError("cross_clear_x_m must be positive")
    if spec.crawl_speed_mps < 0.0 or spec.cruise_speed_mps <= 0.0:
        raise ValueError("teacher speeds must be non-negative/positive")


def narrow_bridge_teacher_geometry(env) -> dict[str, torch.Tensor]:
    """Read per-env barrier/gap/goal from the narrow-bridge replay state.

    Returns env-local coordinates so the teacher never sees env origins.
    Envs without an installed narrow bridge come back with active=False and
    zeroed geometry; callers must gate on ``active``.
    """

    num_envs = int(env.num_envs)
    device = getattr(env, "device", "cpu")
    zeros = torch.zeros(num_envs, device=device)
    inactive = {
        "active": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "barrier_x_m": zeros,
        "gap_center_y_m": zeros,
        "goal_xy_m": torch.zeros(num_envs, 2, device=device),
    }
    centers = getattr(env, "_narrow_bridge_wall_centers", None)
    goal_w = getattr(env, "_narrow_bridge_goal_w", None)
    active = getattr(env, "_narrow_bridge_active", None)
    if centers is None or goal_w is None or active is None:
        return inactive

    barrier_x = centers[:, 0, 0]
    # 兩段牆分別貼在缺口上下緣、長度相同，故其中心的中點 = 缺口中心。
    gap_center_y = 0.5 * (centers[:, 0, 1] + centers[:, 1, 1])
    goal_xy = goal_w[:, :2] - env.scene.env_origins[:, :2]
    return {
        "active": active.to(dtype=torch.bool),
        "barrier_x_m": barrier_x,
        "gap_center_y_m": gap_center_y,
        "goal_xy_m": goal_xy,
    }


def scripted_narrow_gap_action_indices(
    robot_xy_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    current_velocity: torch.Tensor,
    current_omega: torch.Tensor,
    *,
    barrier_x_m: torch.Tensor,
    gap_center_y_m: torch.Tensor,
    goal_xy_m: torch.Tensor,
    num_bins: int,
    dt: float,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
    spec: ScriptedNarrowTeacherSpec = ScriptedNarrowTeacherSpec(),
) -> torch.Tensor:
    """Return [E,2] long (linear_idx, angular_idx) for the direct crossing."""

    _validate_spec(spec)
    if robot_xy_m.ndim != 2 or robot_xy_m.shape[-1] != 2:
        raise ValueError("robot_xy_m must have shape [E,2]")
    E = robot_xy_m.shape[0]
    for name, tensor in (
        ("robot_yaw_rad", robot_yaw_rad),
        ("current_velocity", current_velocity),
        ("current_omega", current_omega),
        ("barrier_x_m", barrier_x_m),
        ("gap_center_y_m", gap_center_y_m),
    ):
        if tensor.reshape(-1).shape[0] != E:
            raise ValueError(f"{name} must have {E} entries")
    if goal_xy_m.shape != (E, 2):
        raise ValueError("goal_xy_m must have shape [E,2]")

    xy = robot_xy_m.float()
    yaw = robot_yaw_rad.reshape(-1).float()
    barrier_x = barrier_x_m.reshape(-1).float()
    gap_y = gap_center_y_m.reshape(-1).float()
    goal = goal_xy_m.float()

    # 穿越方向由 goal 與牆的相對位置決定，左→右 / 右→左皆可。
    direction = torch.sign(goal[:, 0] - barrier_x)
    direction = torch.where(
        direction == 0.0, torch.ones_like(direction), direction
    )
    signed_progress = direction * (xy[:, 0] - barrier_x)
    crossed = signed_progress >= spec.cross_clear_x_m

    # 兩段瞄準：過牆前瞄喉道後方的軸心點（y 用缺口中心，不是 goal 的 y），
    # 過牆後才換成 goal。
    throat_x = barrier_x + direction * spec.cross_clear_x_m
    target_x = torch.where(crossed, goal[:, 0], throat_x)
    target_y = torch.where(crossed, goal[:, 1], gap_y)

    bearing = torch.atan2(target_y - xy[:, 1], target_x - xy[:, 0])
    heading_err = torch.atan2(
        torch.sin(bearing - yaw), torch.cos(bearing - yaw)
    )

    omega_des = (spec.heading_gain * heading_err).clamp(
        -max_angular_velocity, max_angular_velocity
    )
    v_des = torch.where(
        heading_err.abs() > spec.slow_heading_rad,
        torch.full_like(yaw, spec.crawl_speed_mps),
        torch.full_like(yaw, spec.cruise_speed_mps),
    )

    if E == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=xy.device)

    linear_grid, angular_grid = decode_discrete_drive_action_grid(
        current_velocity.reshape(-1),
        current_omega.reshape(-1),
        num_bins=num_bins,
        dt=dt,
        max_linear_velocity=max_linear_velocity,
        reverse_velocity_scale=reverse_velocity_scale,
        max_linear_accel=max_linear_accel,
        max_angular_velocity=max_angular_velocity,
        max_angular_accel=max_angular_accel,
    )
    # 網格沿 dim1 只隨 linear idx 變、沿 dim2 只隨 angular idx 變。
    next_v = linear_grid[:, :, 0]
    next_omega = angular_grid[:, 0, :]
    linear_idx = (next_v - v_des[:, None]).abs().argmin(dim=1)
    angular_idx = (next_omega - omega_des[:, None]).abs().argmin(dim=1)
    return torch.stack([linear_idx, angular_idx], dim=1).long()
