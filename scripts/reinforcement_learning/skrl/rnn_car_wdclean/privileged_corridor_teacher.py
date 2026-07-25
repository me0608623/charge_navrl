"""Privileged short-horizon teacher for controlled corridor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)


@dataclass(frozen=True)
class CorridorTeacherSpec:
    horizon_s: float = 2.0
    samples: int = 10
    robot_half_length_m: float = 0.35
    robot_half_width_m: float = 0.30
    robot_buffer_m: float = 0.10
    robot_obb_offset_x_m: float = -0.128
    hard_obstacle_clearance_m: float = 0.10
    obstacle_clearance_margin_m: float = 0.40
    obstacle_clearance_weight: float = 0.50
    goal_distance_weight: float = 3.0
    goal_heading_weight: float = 0.50
    action_smoothness_weight: float = 0.05
    stall_weight: float = 1.0
    stall_speed_mps: float = 0.10
    allow_reverse: bool = False


def select_teacher_rollout_actions(
    policy_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_feasible: torch.Tensor,
    *,
    override: bool,
) -> torch.Tensor:
    """Select executed actions without conflating labeling and control."""

    if policy_actions.ndim != 2 or policy_actions.shape[-1] != 2:
        raise ValueError("policy_actions must have shape [E,2]")
    if teacher_actions.shape != policy_actions.shape:
        raise ValueError("teacher_actions must match policy_actions")
    if teacher_feasible.shape != policy_actions.shape[:1]:
        raise ValueError("teacher_feasible must have shape [E]")
    if not override:
        return policy_actions
    return torch.where(
        teacher_feasible[:, None],
        teacher_actions.to(dtype=policy_actions.dtype),
        policy_actions,
    )


def _validate_teacher_spec(spec: CorridorTeacherSpec) -> None:
    if spec.horizon_s <= 0.0:
        raise ValueError("teacher horizon must be positive")
    if spec.samples < 1:
        raise ValueError("teacher samples must be positive")
    if spec.robot_half_length_m <= 0.0 or spec.robot_half_width_m <= 0.0:
        raise ValueError("robot half extents must be positive")
    if spec.robot_buffer_m < 0.0:
        raise ValueError("robot buffer must be non-negative")
    if spec.hard_obstacle_clearance_m < 0.0:
        raise ValueError("hard obstacle clearance must be non-negative")
    if spec.obstacle_clearance_margin_m <= 0.0:
        raise ValueError("obstacle clearance margin must be positive")


def predict_patrol_obstacle_paths(
    positions_m: torch.Tensor,
    behavior_type: torch.Tensor,
    patrol_waypoints_m: torch.Tensor,
    patrol_wp_index: torch.Tensor,
    patrol_num_waypoints: torch.Tensor,
    patrol_speed_mps: torch.Tensor,
    patrol_pause_remaining: torch.Tensor,
    *,
    dt: float,
    samples: int,
    patrol_behavior_id: int = 2,
    inactive_behavior_id: int = 0,
    new_waypoint_pause_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict scheduler patrol motion, including waypoint reversals.

    Existing pauses are honored exactly. ``new_waypoint_pause_steps`` selects
    a deterministic branch for future, currently unknowable random pauses.
    """

    if positions_m.ndim != 3 or positions_m.shape[-1] != 2:
        raise ValueError("positions must have shape [E,N,2]")
    if behavior_type.shape != positions_m.shape[:2]:
        raise ValueError("behavior_type must have shape [E,N]")
    if (
        patrol_waypoints_m.ndim != 4
        or patrol_waypoints_m.shape[:2] != positions_m.shape[:2]
        or patrol_waypoints_m.shape[-1] != 2
    ):
        raise ValueError("patrol waypoints must have shape [E,N,W,2]")
    if samples < 1 or dt <= 0.0:
        raise ValueError("samples and dt must be positive")
    if new_waypoint_pause_steps < 0:
        raise ValueError("new waypoint pause steps must be non-negative")

    pos = positions_m.clone()
    wp_index = patrol_wp_index.clone().long()
    pause = patrol_pause_remaining.clone().long()
    valid = behavior_type != int(inactive_behavior_id)
    patrol = behavior_type == int(patrol_behavior_id)
    paths: list[torch.Tensor] = []

    for _ in range(int(samples)):
        pausing = patrol & (pause > 0)
        pause = torch.where(pausing, pause - 1, pause)
        moving = patrol & ~pausing

        safe_num_waypoints = patrol_num_waypoints.clamp_min(1).long()
        safe_index = torch.remainder(wp_index, safe_num_waypoints)
        target = torch.gather(
            patrol_waypoints_m,
            2,
            safe_index[:, :, None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        delta = target - pos
        distance = delta.norm(dim=-1)
        direction = delta / distance.clamp_min(1e-6)[..., None]
        step_distance = torch.minimum(
            patrol_speed_mps * float(dt), distance
        )
        pos = torch.where(
            moving[..., None],
            pos + direction * step_distance[..., None],
            pos,
        )

        reached = moving & (distance < 0.3)
        wp_index = torch.where(
            reached,
            torch.remainder(wp_index + 1, safe_num_waypoints),
            wp_index,
        )
        pause = torch.where(
            reached,
            torch.full_like(pause, int(new_waypoint_pause_steps)),
            pause,
        )
        # Static and inactive obstacle positions remain fixed; valid masks keep
        # inactive slots out of all collision calculations.
        paths.append(pos.clone())

    return torch.stack(paths, dim=2), valid


def _unicycle_paths(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    robot_xy_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    *,
    horizon_s: float,
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    times = torch.linspace(
        float(horizon_s) / int(samples),
        float(horizon_s),
        int(samples),
        dtype=linear_velocity.dtype,
        device=linear_velocity.device,
    )
    omega_safe = torch.where(
        angular_velocity.abs() < 1e-4,
        torch.ones_like(angular_velocity),
        angular_velocity,
    )
    angle = angular_velocity[..., None] * times
    radius = linear_velocity / omega_safe
    body_x = radius[..., None] * torch.sin(angle)
    body_y = radius[..., None] * (1.0 - torch.cos(angle))
    straight = angular_velocity.abs() < 1e-4
    body_x = torch.where(
        straight[..., None],
        linear_velocity[..., None] * times,
        body_x,
    )
    body_y = torch.where(
        straight[..., None], torch.zeros_like(body_y), body_y
    )

    c0 = torch.cos(robot_yaw_rad)[:, None, None, None]
    s0 = torch.sin(robot_yaw_rad)[:, None, None, None]
    world_x = (
        robot_xy_m[:, None, None, 0, None]
        + c0 * body_x
        - s0 * body_y
    )
    world_y = (
        robot_xy_m[:, None, None, 1, None]
        + s0 * body_x
        + c0 * body_y
    )
    path = torch.stack([world_x, world_y], dim=-1)
    yaw = (
        robot_yaw_rad[:, None, None, None]
        + angular_velocity[..., None] * times
    )
    return path, yaw


def _obb_circle_clearance(
    robot_path_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    obstacle_paths_m: torch.Tensor,
    obstacle_radii_m: torch.Tensor,
    obstacle_valid: torch.Tensor,
    spec: CorridorTeacherSpec,
) -> torch.Tensor:
    # [E,L,A,N,T,2], obstacle and robot samples share the same time axis.
    delta = (
        obstacle_paths_m[:, None, None, :, :, :]
        - robot_path_m[:, :, :, None, :, :]
    )
    c = torch.cos(robot_yaw_rad)[:, :, :, None, :]
    s = torch.sin(robot_yaw_rad)[:, :, :, None, :]
    dx = delta[..., 0] - spec.robot_obb_offset_x_m * c
    dy = delta[..., 1] - spec.robot_obb_offset_x_m * s
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    closest_x = local_x.clamp(
        -spec.robot_half_length_m, spec.robot_half_length_m
    )
    closest_y = local_y.clamp(
        -spec.robot_half_width_m, spec.robot_half_width_m
    )
    distance_to_obb = torch.sqrt(
        (local_x - closest_x).square()
        + (local_y - closest_y).square()
    )
    clearance = distance_to_obb - (
        obstacle_radii_m[:, None, None, :, None]
        + spec.robot_buffer_m
    )
    clearance = torch.where(
        obstacle_valid[:, None, None, :, None],
        clearance,
        torch.full_like(clearance, float("inf")),
    )
    return clearance


def _obb_wall_collision(
    robot_path_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    wall_centers_m: torch.Tensor,
    wall_sizes_m: torch.Tensor,
    wall_valid: torch.Tensor,
    spec: CorridorTeacherSpec,
) -> torch.Tensor:
    c = torch.cos(robot_yaw_rad)
    s = torch.sin(robot_yaw_rad)
    obb_center = robot_path_m + spec.robot_obb_offset_x_m * torch.stack(
        [c, s], dim=-1
    )
    delta = (
        wall_centers_m[:, None, None, None, :, :]
        - obb_center[:, :, :, :, None, :]
    )
    dx = delta[..., 0]
    dy = delta[..., 1]
    ex = wall_sizes_m[:, None, None, None, :, 0] * 0.5
    ey = wall_sizes_m[:, None, None, None, :, 1] * 0.5
    cc = c[..., None]
    ss = s[..., None]
    ac = cc.abs()
    ass = ss.abs()
    half_length = spec.robot_half_length_m + spec.robot_buffer_m
    half_width = spec.robot_half_width_m + spec.robot_buffer_m
    separated = (
        (dx.abs() > (half_length * ac + half_width * ass + ex))
        | (dy.abs() > (half_length * ass + half_width * ac + ey))
        | (
            (dx * cc + dy * ss).abs()
            > (half_length + ex * ac + ey * ass)
        )
        | (
            (-dx * ss + dy * cc).abs()
            > (half_width + ex * ass + ey * ac)
        )
    )
    collision = (~separated) & wall_valid[
        :, None, None, None, :
    ]
    return collision.any(dim=-1)


@torch.no_grad()
def corridor_teacher_action_grid(
    *,
    current_velocity: torch.Tensor,
    current_omega: torch.Tensor,
    robot_xy_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    goal_xy_m: torch.Tensor,
    obstacle_paths_m: torch.Tensor,
    obstacle_radii_m: torch.Tensor,
    obstacle_valid: torch.Tensor,
    wall_centers_m: torch.Tensor,
    wall_sizes_m: torch.Tensor,
    wall_valid: torch.Tensor,
    num_bins: int,
    dt: float,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
    spec: CorridorTeacherSpec = CorridorTeacherSpec(),
) -> dict[str, torch.Tensor]:
    """Rank all reachable action pairs by hard safety then local goal cost."""

    _validate_teacher_spec(spec)
    linear, angular = decode_discrete_drive_action_grid(
        current_velocity,
        current_omega,
        num_bins=num_bins,
        dt=dt,
        max_linear_velocity=max_linear_velocity,
        reverse_velocity_scale=reverse_velocity_scale,
        max_linear_accel=max_linear_accel,
        max_angular_velocity=max_angular_velocity,
        max_angular_accel=max_angular_accel,
    )
    path, yaw = _unicycle_paths(
        linear,
        angular,
        robot_xy_m,
        robot_yaw_rad,
        horizon_s=spec.horizon_s,
        samples=spec.samples,
    )
    obstacle_clearance = _obb_circle_clearance(
        path,
        yaw,
        obstacle_paths_m,
        obstacle_radii_m,
        obstacle_valid,
        spec,
    )
    min_obstacle_clearance = obstacle_clearance.amin(dim=(-1, -2))
    obstacle_collision = (
        min_obstacle_clearance < spec.hard_obstacle_clearance_m
    )
    wall_collision_samples = _obb_wall_collision(
        path,
        yaw,
        wall_centers_m,
        wall_sizes_m,
        wall_valid,
        spec,
    )
    wall_collision = wall_collision_samples.any(dim=-1)
    feasible = ~obstacle_collision & ~wall_collision
    if not spec.allow_reverse:
        feasible &= linear >= -1e-4

    endpoint = path[..., -1, :]
    endpoint_yaw = yaw[..., -1]
    goal_delta = goal_xy_m[:, None, None, :] - endpoint
    goal_distance = goal_delta.norm(dim=-1)
    initial_goal_distance = (
        goal_xy_m - robot_xy_m
    ).norm(dim=-1).clamp_min(1.0)
    goal_distance_cost = goal_distance / initial_goal_distance[:, None, None]
    goal_heading = torch.atan2(goal_delta[..., 1], goal_delta[..., 0])
    heading_error = torch.atan2(
        torch.sin(goal_heading - endpoint_yaw),
        torch.cos(goal_heading - endpoint_yaw),
    ).abs() / torch.pi
    clearance_cost = (
        (
            spec.obstacle_clearance_margin_m
            - min_obstacle_clearance
        )
        / spec.obstacle_clearance_margin_m
    ).clamp(0.0, 1.0)
    smoothness = (
        (linear - current_velocity[:, None, None]).abs()
        / max(float(max_linear_velocity), 1e-6)
        + (angular - current_omega[:, None, None]).abs()
        / max(float(max_angular_velocity), 1e-6)
    )
    stall = (linear.abs() < spec.stall_speed_mps).float()
    cost = (
        spec.obstacle_clearance_weight * clearance_cost
        + spec.goal_distance_weight * goal_distance_cost
        + spec.goal_heading_weight * heading_error
        + spec.action_smoothness_weight * smoothness
        + spec.stall_weight * stall
    )
    ranked_cost = torch.where(
        feasible, cost, torch.full_like(cost, float("inf"))
    )
    flat_index = ranked_cost.flatten(1).argmin(dim=-1)
    any_feasible = feasible.flatten(1).any(dim=-1)
    linear_index = torch.div(
        flat_index, int(num_bins), rounding_mode="floor"
    )
    angular_index = torch.remainder(flat_index, int(num_bins))
    actions = torch.stack([linear_index, angular_index], dim=-1)
    actions = torch.where(
        any_feasible[:, None],
        actions,
        torch.full_like(actions, int(num_bins) // 2),
    )
    return {
        "actions": actions,
        "any_feasible": any_feasible,
        "feasible_fraction": feasible.float().mean(dim=(1, 2)),
        "selected_cost": ranked_cost.flatten(1).amin(dim=-1),
        "cost_grid": ranked_cost,
        "linear_velocity_grid": linear,
        "angular_velocity_grid": angular,
        "min_obstacle_clearance_grid": min_obstacle_clearance,
        "wall_collision_grid": wall_collision,
        "obstacle_collision_grid": obstacle_collision,
    }
