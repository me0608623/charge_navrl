"""Pure geometry for the deployment-corridor replay scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MOTION_LATERAL = 0
MOTION_LONGITUDINAL = 1
MOTION_RANDOM_2D = 2
DYNAMIC_MOTION_MODES = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed",
    "env_stratified",
)


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


def validate_dynamic_motion_mode(mode: str) -> str:
    """Return a normalized deployment-corridor dynamic motion mode."""
    normalized = str(mode).strip().lower()
    if normalized not in DYNAMIC_MOTION_MODES:
        raise ValueError(
            f"unsupported corridor motion mode {mode!r}; "
            f"expected one of {DYNAMIC_MOTION_MODES}"
        )
    return normalized


def normalize_dynamic_motion_weights(
    weights, mode: str
) -> tuple[float, float, float] | None:
    """Validate optional per-family env weights for ``env_stratified``.

    Weights are ``(lateral, longitudinal, random_2d)`` and are renormalized to
    sum to one. ``None`` keeps the balanced 1:1:1 stratification. Supplying
    weights for any other motion mode raises ``ValueError`` so that a
    misconfigured experiment fails loudly instead of silently ignoring them.
    """
    if weights is None:
        return None
    normalized_mode = validate_dynamic_motion_mode(mode)
    if normalized_mode != "env_stratified":
        raise ValueError(
            "corridor motion weights are only supported for "
            f"mode='env_stratified', got mode={normalized_mode!r}"
        )
    values = tuple(float(w) for w in weights)
    if len(values) != 3:
        raise ValueError(
            f"expected three corridor motion weights, got {len(values)}"
        )
    if any(not math.isfinite(w) or w < 0.0 for w in values):
        raise ValueError(
            f"corridor motion weights must be finite and non-negative, got {values}"
        )
    total = sum(values)
    if total <= 0.0:
        raise ValueError("corridor motion weights must sum to a positive value")
    return tuple(w / total for w in values)


def sample_motion_families(
    count: int,
    weights: tuple[float, float, float],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random-phase systematic stratification of ``count`` envs across families.

    Resets arrive in small, variable batches, so a deterministic largest-
    remainder quota would bias every batch the same way and systematically
    starve low-weight families (with weights 0.30/0.10/0.60 a single-env reset
    always produced random_2d, and longitudinal never appeared below seven
    envs). Random-phase systematic sampling keeps the per-call allocation within
    one env of ``count * weight`` while staying unbiased in expectation for any
    batch size, including ``count == 1``. It is fully determined by the torch
    RNG, so runs stay reproducible.
    """
    if count <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float64, device=device)
    boundaries = torch.cumsum(weights_tensor, dim=0)[:-1]
    phase = torch.rand((), dtype=torch.float64, device=device)
    points = (
        torch.arange(count, device=device, dtype=torch.float64) + phase
    ) / count
    families = torch.bucketize(points, boundaries, right=True)
    return families[torch.randperm(count, device=device)].to(torch.long)


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


def sample_dynamic_trajectories(
    dynamic: torch.Tensor,
    lateral_waypoints: torch.Tensor,
    spec: LongCorridorSpec,
    mode: str,
    motion_weights: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build controlled dynamic paths for one corridor motion family.

    Returns starts, two-point patrol paths, per-obstacle motion type IDs, and
    the first target waypoint index. ``mixed`` samples the three controlled
    families independently per obstacle and per environment.
    ``env_stratified`` assigns one family to both dynamic obstacles in an
    environment while balancing families across the batch. This matches the
    pure-family regression gates without removing heterogeneous mixed scenes.
    """
    normalized = validate_dynamic_motion_mode(mode)
    weights = normalize_dynamic_motion_weights(motion_weights, normalized)
    if dynamic.ndim != 3 or dynamic.shape[1:] != (2, 2):
        raise ValueError(f"expected dynamic [N,2,2], got {tuple(dynamic.shape)}")
    if lateral_waypoints.shape != (dynamic.shape[0], 2, 2, 2):
        raise ValueError(
            "expected lateral_waypoints [N,2,2,2], got "
            f"{tuple(lateral_waypoints.shape)}"
        )

    count = dynamic.shape[0]
    device = dynamic.device
    lateral_starts = dynamic.clone()
    lateral_targets = (
        torch.rand(count, 2, device=device) < 0.5
    ).long()

    # One lane approaches the robot while the other initially travels with it.
    longitudinal_starts = torch.zeros_like(dynamic)
    longitudinal_starts[:, 0, 0] = -0.40
    longitudinal_starts[:, 0, 1] = abs(float(spec.dynamic_y[1]))
    longitudinal_starts[:, 1, 0] = 0.40
    longitudinal_starts[:, 1, 1] = -abs(float(spec.dynamic_y[0]))
    longitudinal_waypoints = torch.zeros_like(lateral_waypoints)
    longitudinal_y_limit = min(
        3.4,
        0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance,
    )
    longitudinal_waypoints[:, :, 0, 1] = -longitudinal_y_limit
    longitudinal_waypoints[:, :, 1, 1] = longitudinal_y_limit
    longitudinal_waypoints[:, 0, :, 0] = -0.40
    longitudinal_waypoints[:, 1, :, 0] = 0.40
    longitudinal_targets = torch.tensor(
        [0, 1], dtype=torch.long, device=device
    ).expand(count, -1).clone()

    # Randomized diagonal patrols stay in separate longitudinal bands and in
    # the center strip, so they cannot tunnel through the static side obstacles.
    random_waypoints = torch.zeros_like(lateral_waypoints)
    x_left = torch.empty(count, 2, device=device).uniform_(-0.35, -0.15)
    x_right = torch.empty(count, 2, device=device).uniform_(0.15, 0.35)
    swap_x = torch.rand(count, 2, device=device) < 0.5
    random_waypoints[:, :, 0, 0] = torch.where(
        swap_x, x_right, x_left
    )
    random_waypoints[:, :, 1, 0] = torch.where(
        swap_x, x_left, x_right
    )
    random_outer_y = min(
        3.3,
        0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance,
    )
    random_waypoints[:, 0, 0, 1] = torch.empty(
        count, device=device
    ).uniform_(-random_outer_y, -2.1)
    random_waypoints[:, 0, 1, 1] = torch.empty(
        count, device=device
    ).uniform_(-1.8, -0.8)
    random_waypoints[:, 1, 0, 1] = torch.empty(
        count, device=device
    ).uniform_(0.8, 1.8)
    random_waypoints[:, 1, 1, 1] = torch.empty(
        count, device=device
    ).uniform_(2.1, random_outer_y)
    random_starts = random_waypoints[:, :, 0].clone()
    random_targets = torch.ones(count, 2, dtype=torch.long, device=device)

    if normalized == "lateral":
        motion_types = torch.full(
            (count, 2), MOTION_LATERAL, dtype=torch.long, device=device
        )
    elif normalized == "longitudinal":
        motion_types = torch.full(
            (count, 2), MOTION_LONGITUDINAL, dtype=torch.long, device=device
        )
    elif normalized == "random_2d":
        motion_types = torch.full(
            (count, 2), MOTION_RANDOM_2D, dtype=torch.long, device=device
        )
    elif normalized == "mixed":
        flat_count = count * 2
        offset = int(torch.randint(0, 3, (1,), device=device).item())
        balanced = (
            torch.arange(flat_count, device=device) + offset
        ) % (MOTION_RANDOM_2D + 1)
        motion_types = balanced[
            torch.randperm(flat_count, device=device)
        ].reshape(count, 2)
    elif weights is None:
        offset = int(torch.randint(0, 3, (1,), device=device).item())
        balanced = (
            torch.arange(count, device=device) + offset
        ) % (MOTION_RANDOM_2D + 1)
        env_motion_types = balanced[torch.randperm(count, device=device)]
        motion_types = env_motion_types[:, None].expand(-1, 2).clone()
    else:
        # Weighted env-level stratification. Random-phase systematic sampling
        # keeps small reset batches unbiased; both obstacles of an env share the
        # drawn family.
        env_motion_types = sample_motion_families(count, weights, device)
        motion_types = env_motion_types[:, None].expand(-1, 2).clone()

    starts = lateral_starts
    waypoints = lateral_waypoints.clone()
    target_indices = lateral_targets
    longitudinal = motion_types == MOTION_LONGITUDINAL
    random_2d = motion_types == MOTION_RANDOM_2D
    starts = torch.where(
        longitudinal[..., None], longitudinal_starts, starts
    )
    starts = torch.where(random_2d[..., None], random_starts, starts)
    waypoints = torch.where(
        longitudinal[..., None, None], longitudinal_waypoints, waypoints
    )
    waypoints = torch.where(
        random_2d[..., None, None], random_waypoints, waypoints
    )
    target_indices = torch.where(
        longitudinal, longitudinal_targets, target_indices
    )
    target_indices = torch.where(
        random_2d, random_targets, target_indices
    )
    return starts, waypoints, motion_types, target_indices


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

    # Controlled patrol segments must not intersect the fixed side obstacles.
    point = static[:, :, None, :]
    segment_start = waypoints[:, None, :, 0, :]
    segment_delta = (
        waypoints[:, None, :, 1, :]
        - segment_start
    )
    projection = (
        ((point - segment_start) * segment_delta).sum(dim=-1)
        / segment_delta.square().sum(dim=-1).clamp_min(1e-9)
    ).clamp(0.0, 1.0)
    closest = segment_start + projection[..., None] * segment_delta
    dynamic_static_clear = (
        torch.linalg.vector_norm(point - closest, dim=-1)
        > (2.0 * spec.obstacle_radius)
    ).all(dim=(1, 2))
    return inside & centerline_open & dynamic_static_clear
