"""Pure geometry for the deployment-corridor replay scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LongCorridorSpec:
    """Frozen geometry for a 4 m free-width, 10 m long corridor."""

    free_width: float = 4.0
    length: float = 10.0
    wall_thickness: float = 1.0
    wall_height: float = 3.0
    robot_start_y: float = -4.1
    goal_y: float = 4.1
    static_x: float = 1.25
    static_y: tuple[float, ...] = (-3.0, -0.6, 0.6, 3.0)
    dynamic_y: tuple[float, ...] = (-1.8, 1.8)
    dynamic_x_limit: float = 1.30
    obstacle_radius: float = 0.35
    wall_clearance: float = 0.10
    robot_half_length: float = 0.35
    robot_half_width: float = 0.30
    robot_buffer: float = 0.10
    static_xy_jitter: float = 0.10
    dynamic_y_jitter: float = 0.08

    @property
    def wall_center_offset(self) -> float:
        return 0.5 * (self.free_width + self.wall_thickness)

    @property
    def inner_half_width(self) -> float:
        return 0.5 * self.free_width

    @property
    def robot_conservative_radius(self) -> float:
        return math.hypot(
            self.robot_half_length + self.robot_buffer,
            self.robot_half_width + self.robot_buffer,
        )

    @property
    def centerline_static_clearance(self) -> float:
        return (
            self.static_x
            - self.static_xy_jitter
            - self.obstacle_radius
            - self.robot_conservative_radius
        )


def validate_spec(spec: LongCorridorSpec) -> None:
    """Reject geometry that cannot guarantee the intended constructive route."""
    if spec.free_width <= 0.0 or spec.length <= 0.0:
        raise ValueError("corridor width and length must be positive")
    if spec.wall_thickness <= 0.0:
        raise ValueError("wall thickness must be positive")
    if len(spec.static_y) != 4 or len(spec.dynamic_y) != 2:
        raise ValueError("deployment corridor requires exactly 4 static and 2 dynamic obstacles")
    if abs(spec.robot_start_y) >= 0.5 * spec.length:
        raise ValueError("robot start must stay inside the corridor")
    if abs(spec.goal_y) >= 0.5 * spec.length:
        raise ValueError("goal must stay inside the corridor")
    max_center_x = (
        spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    )
    if spec.static_x + spec.static_xy_jitter > max_center_x:
        raise ValueError("static obstacle can overlap a corridor wall")
    if spec.dynamic_x_limit > max_center_x:
        raise ValueError("dynamic patrol can overlap a corridor wall")
    if spec.centerline_static_clearance <= 0.0:
        raise ValueError("static obstacles do not leave a conservative centerline route")

    max_center_y = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    all_y = (*spec.static_y, *spec.dynamic_y)
    if max(abs(y) for y in all_y) + spec.static_xy_jitter > max_center_y:
        raise ValueError("obstacle template can leave the corridor ends")

    min_crossing_separation = min(
        abs(dynamic_y - static_y)
        for dynamic_y in spec.dynamic_y
        for static_y in spec.static_y
    )
    required = 2.0 * spec.obstacle_radius + (
        spec.static_xy_jitter + spec.dynamic_y_jitter
    )
    if min_crossing_separation <= required:
        raise ValueError("dynamic patrol lane can overlap a static obstacle")


def validate_obstacle_counts(
    static_obstacles: int,
    dynamic_obstacles: int,
) -> None:
    """Validate a curriculum subset of the frozen 4S+2D template."""
    if not 0 <= int(static_obstacles) <= 4:
        raise ValueError("corridor static obstacle count must be in [0, 4]")
    if not 0 <= int(dynamic_obstacles) <= 2:
        raise ValueError("corridor dynamic obstacle count must be in [0, 2]")


def wall_geometry(
    count: int,
    spec: LongCorridorSpec,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local wall centers/sizes for `count` mirrored-identical corridors."""
    validate_spec(spec)
    centers = torch.zeros(count, 2, 2, device=device)
    centers[:, 0, 0] = -spec.wall_center_offset
    centers[:, 1, 0] = spec.wall_center_offset
    sizes = torch.zeros(count, 2, 2, device=device)
    sizes[:, :, 0] = spec.wall_thickness
    sizes[:, :, 1] = spec.length
    return centers, sizes


def sample_obstacle_layout(
    count: int,
    spec: LongCorridorSpec,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample safe static positions, dynamic starts and two-point patrol paths."""
    validate_spec(spec)
    mirror = torch.where(
        torch.rand(count, device=device) < 0.5,
        torch.full((count,), -1.0, device=device),
        torch.ones(count, device=device),
    )

    base_static_x = torch.tensor(
        (-spec.static_x, spec.static_x, -spec.static_x, spec.static_x),
        device=device,
    )
    base_static_y = torch.tensor(spec.static_y, device=device)
    static = torch.zeros(count, 4, 2, device=device)
    static[:, :, 0] = mirror[:, None] * base_static_x[None, :]
    static[:, :, 1] = base_static_y[None, :]
    static += torch.empty_like(static).uniform_(
        -spec.static_xy_jitter, spec.static_xy_jitter
    )

    dynamic_y = torch.tensor(spec.dynamic_y, device=device)[None, :].expand(count, -1)
    dynamic_y = dynamic_y + torch.empty_like(dynamic_y).uniform_(
        -spec.dynamic_y_jitter, spec.dynamic_y_jitter
    )
    dynamic = torch.zeros(count, 2, 2, device=device)
    dynamic[:, :, 0] = torch.empty(count, 2, device=device).uniform_(
        -spec.dynamic_x_limit, spec.dynamic_x_limit
    )
    dynamic[:, :, 1] = dynamic_y

    waypoints = torch.zeros(count, 2, 2, 2, device=device)
    waypoints[:, :, 0, 0] = -spec.dynamic_x_limit
    waypoints[:, :, 1, 0] = spec.dynamic_x_limit
    waypoints[:, :, :, 1] = dynamic_y[:, :, None]
    return static, dynamic, waypoints


def layout_is_constructively_solvable(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    waypoints: torch.Tensor,
    spec: LongCorridorSpec,
) -> torch.Tensor:
    """Check wall bounds and the static centerline route for each sampled env."""
    validate_spec(spec)
    if static.ndim != 3 or static.shape[1:] != (4, 2):
        raise ValueError(f"expected static [N,4,2], got {tuple(static.shape)}")
    if dynamic.shape != (static.shape[0], 2, 2):
        raise ValueError(f"expected dynamic [N,2,2], got {tuple(dynamic.shape)}")
    if waypoints.shape != (static.shape[0], 2, 2, 2):
        raise ValueError(f"expected waypoints [N,2,2,2], got {tuple(waypoints.shape)}")

    max_x = spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    max_y = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    all_points = torch.cat(
        [static, dynamic, waypoints.reshape(static.shape[0], -1, 2)], dim=1
    )
    inside = (
        (all_points[..., 0].abs() <= max_x)
        & (all_points[..., 1].abs() <= max_y)
    ).all(dim=1)

    centerline_clearance = (
        static[..., 0].abs()
        - spec.obstacle_radius
        - spec.robot_conservative_radius
    )
    centerline_open = (centerline_clearance > 0.0).all(dim=1)
    return inside & centerline_open
