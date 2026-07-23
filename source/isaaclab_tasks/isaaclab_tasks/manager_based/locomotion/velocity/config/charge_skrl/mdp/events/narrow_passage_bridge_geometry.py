"""Pure geometry and schedule helpers for the SA5 narrow-passage bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NarrowPassageSchedule:
    """Difficulty values used when a narrow-passage episode is reset."""

    progress: float
    width_min: float
    width_max: float
    yaw_limit_deg: float
    stress_ratio: float


def schedule_at(
    progress: float,
    final_stress_ratio: float = 0.25,
    fixed_width_range: tuple[float, float] | None = None,
    fixed_yaw_limit_deg: float | None = None,
) -> NarrowPassageSchedule:
    """Return a continuous easy-to-hard bridge schedule.

    First half:
        gap [1.6, 1.4] -> [1.4, 1.2] m, yaw +/-2 -> +/-6 deg.
    Second half:
        regular gap remains [1.4, 1.2] m, yaw grows to +/-10 deg, while a
        small [1.2, 1.0] m stress subset ramps from 0 to ``final_stress_ratio``.
    """
    p = min(max(float(progress), 0.0), 1.0)
    if (fixed_width_range is None) != (fixed_yaw_limit_deg is None):
        raise ValueError("fixed width range and yaw limit must be configured together")
    if fixed_width_range is not None:
        width_min, width_max = map(float, fixed_width_range)
        yaw_limit = float(fixed_yaw_limit_deg)
        if not 0.0 < width_min <= width_max:
            raise ValueError(f"invalid fixed narrow-passage width range: {fixed_width_range}")
        if yaw_limit <= 0.0:
            raise ValueError(f"fixed narrow-passage yaw limit must be positive: {yaw_limit}")
        return NarrowPassageSchedule(
            progress=p,
            width_min=width_min,
            width_max=width_max,
            yaw_limit_deg=yaw_limit,
            stress_ratio=0.0,
        )

    if p <= 0.5:
        q = p / 0.5
        return NarrowPassageSchedule(
            progress=p,
            width_min=1.4 - 0.2 * q,
            width_max=1.6 - 0.2 * q,
            yaw_limit_deg=2.0 + 4.0 * q,
            stress_ratio=0.0,
        )

    q = (p - 0.5) / 0.5
    return NarrowPassageSchedule(
        progress=p,
        width_min=1.2,
        width_max=1.4,
        yaw_limit_deg=6.0 + 4.0 * q,
        stress_ratio=max(0.0, float(final_stress_ratio)) * q,
    )


def minimum_solvable_gap(robot_half_width: float = 0.30, collision_buffer: float = 0.10) -> float:
    """Return the exact straight-line OBB width required by wall termination."""
    return 2.0 * (float(robot_half_width) + float(collision_buffer))


def validate_constructed_scene(
    *,
    gap_width: float,
    gap_center: float,
    barrier_x: float,
    start_x: float,
    goal_x: float,
    room_half_extent: float,
    boundary_wall_width: float,
    segment_length: float,
    robot_half_length: float = 0.35,
    robot_half_width: float = 0.30,
    collision_buffer: float = 0.10,
    epsilon: float = 1e-6,
) -> tuple[bool, str]:
    """Prove that the constructed center-line path is physically traversable.

    This is a constructive check, not a center-point heuristic. It verifies that
    the measured 0.70 x 0.60 m OBB plus the configured 0.10 m collision buffer
    fits through the opening, both barrier segments reach the arena boundaries,
    and the start/goal poses fit inside the room.
    """
    required = minimum_solvable_gap(robot_half_width, collision_buffer)
    if gap_width + epsilon < required:
        return False, f"gap {gap_width:.3f}m < OBB requirement {required:.3f}m"

    inner = float(room_half_extent) - 0.5 * float(boundary_wall_width)
    gap_lo = float(gap_center) - 0.5 * float(gap_width)
    gap_hi = float(gap_center) + 0.5 * float(gap_width)
    if gap_lo <= -inner or gap_hi >= inner:
        return False, "gap overlaps an arena boundary"

    if gap_hi + float(segment_length) < inner - epsilon:
        return False, "upper barrier segment does not reach the boundary"
    if gap_lo - float(segment_length) > -inner + epsilon:
        return False, "lower barrier segment does not reach the boundary"

    longitudinal_margin = float(robot_half_length) + float(collision_buffer)
    if max(abs(float(start_x)), abs(float(goal_x))) + longitudinal_margin >= inner:
        return False, "start or goal OBB does not fit inside the room"
    if not min(float(start_x), float(goal_x)) < float(barrier_x) < max(float(start_x), float(goal_x)):
        return False, "barrier is not between start and goal"

    return True, "constructive center-line path is solvable"
