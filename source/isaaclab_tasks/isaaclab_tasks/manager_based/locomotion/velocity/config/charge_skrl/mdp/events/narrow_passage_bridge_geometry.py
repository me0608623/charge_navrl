"""Pure geometry and schedule helpers for the SA5 narrow-passage bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math


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


def validate_closed_range(
    name: str,
    values: tuple[float, float] | None,
    *,
    fallback: float,
    minimum: float | None = None,
) -> tuple[float, float]:
    """Validate a training-time closed range without silently reordering it."""
    if values is None:
        lo = hi = float(fallback)
    else:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lo, hi = (float(value) for value in values)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"{name} must contain finite values: {(lo, hi)}")
    if lo > hi:
        raise ValueError(f"{name} must be ordered low <= high: {(lo, hi)}")
    if minimum is not None and lo < float(minimum):
        raise ValueError(f"{name} minimum must be >= {minimum}: {(lo, hi)}")
    return lo, hi


def validate_constructed_scene(
    *,
    gap_width: float,
    gap_center: float,
    barrier_x: float,
    start_x: float,
    goal_x: float,
    goal_y: float | None = None,
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
    if goal_y is not None:
        # Goal arrival has no prescribed yaw, so reserve the larger OBB half
        # extent instead of assuming the body remains axis-aligned.
        lateral_margin = max(
            float(robot_half_length), float(robot_half_width)
        ) + float(collision_buffer)
        if abs(float(goal_y)) + lateral_margin >= inner:
            return False, "goal OBB does not fit inside the room laterally"
    if not min(float(start_x), float(goal_x)) < float(barrier_x) < max(float(start_x), float(goal_x)):
        return False, "barrier is not between start and goal"

    return True, "constructive center-line path is solvable"


# ---------------------------------------------------------------------------
# Replay layout resolution
# ---------------------------------------------------------------------------

#: Historical bridge defaults. N1 explicitly overrides only the two goal ranges;
#: D0/W1/W2 continue to use these pinned values for reproducibility.
TRAINING_REPLAY_LAYOUT: dict = {
    "gap_center_range": (-1.0, 1.0),
    "barrier_x_range": (-0.5, 0.5),
    "start_distance": 3.0,
    "goal_distance": 3.0,
    "goal_lateral_offset": 0.0,
    "goal_distance_range": (3.0, 3.0),
    "goal_lateral_offset_range": (0.0, 0.0),
    "segment_length": 9.0,
    "direction_mode": "random",
}

DIRECTION_MODES = ("random", "forward", "backward")


@dataclass(frozen=True)
class NarrowReplayLayout:
    """A narrow-replay sampling box that is solvable at every corner.

    ``resolve_replay_layout`` clamps a requested layout into this form. Every
    combination of the range endpoints is guaranteed to pass
    :func:`validate_constructed_scene`, so the reset injector never has to
    reject a scene it was asked to build.
    """

    gap_center_range: tuple[float, float]
    barrier_x_range: tuple[float, float]
    width_range: tuple[float, float]
    start_distance: float
    goal_distance: float
    goal_lateral_offset: float
    goal_distance_range: tuple[float, float]
    goal_lateral_offset_range: tuple[float, float]
    segment_length: float
    direction_mode: str
    warnings: tuple[str, ...]


def _ordered(pair) -> tuple[float, float]:
    lo, hi = (float(v) for v in pair)
    return (lo, hi) if lo <= hi else (hi, lo)


def resolve_replay_layout(
    *,
    gap_center_range,
    barrier_x_range,
    width_range,
    start_distance: float,
    goal_distance: float,
    goal_lateral_offset: float = 0.0,
    goal_distance_range: tuple[float, float] | None = None,
    goal_lateral_offset_range: tuple[float, float] | None = None,
    segment_length: float = 9.0,
    direction_mode: str = "random",
    room_half_extent: float = 7.0,
    boundary_wall_width: float = 1.0,
    barrier_thickness: float = 1.0,
    robot_half_length: float = 0.35,
    robot_half_width: float = 0.30,
    collision_buffer: float = 0.10,
    epsilon: float = 1e-3,
) -> NarrowReplayLayout:
    """Clamp a requested replay layout into a fully solvable one.

    Manual play-side overrides are easy to push past the arena geometry (a goal
    outside the room, a gap flush against a boundary wall). The reset injector
    treats that as a hard error, which for training is the right call but for an
    interactive play session means crashing a minute after Isaac Sim booted.
    This resolver instead pulls every value back to the nearest legal one and
    reports what it changed, so callers can warn rather than abort.

    Raises:
        ValueError: only for a ``direction_mode`` that has no legal meaning.
    """
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(
            f"unknown direction_mode {direction_mode!r}; expected one of {DIRECTION_MODES}"
        )

    warnings: list[str] = []

    def _note(name: str, before: float, after: float, why: str) -> None:
        if abs(before - after) > 1e-9:
            warnings.append(f"{name} {before:.3f} -> {after:.3f} ({why})")

    inner = float(room_half_extent) - 0.5 * float(boundary_wall_width)
    obb_margin = float(robot_half_length) + float(collision_buffer)

    gap_lo_req, gap_hi_req = _ordered(gap_center_range)
    barrier_lo_req, barrier_hi_req = _ordered(barrier_x_range)
    width_lo_req, width_hi_req = _ordered(width_range)

    # 1. The gap must at least fit the measured OBB.
    required_width = minimum_solvable_gap(robot_half_width, collision_buffer)
    width_lo = max(width_lo_req, required_width)
    width_hi = max(width_hi_req, width_lo)
    _note("width_min", width_lo_req, width_lo, f"OBB needs {required_width:.2f}m")
    _note("width_max", width_hi_req, width_hi, "kept above width_min")

    # 2. The widest gap must not touch a boundary wall at the extreme centers.
    gap_center_limit = max(inner - 0.5 * width_hi - epsilon, 0.0)
    gap_lo = max(gap_lo_req, -gap_center_limit)
    gap_hi = min(gap_hi_req, gap_center_limit)
    if gap_lo > gap_hi:  # both endpoints collapsed onto the same clamp
        gap_lo = gap_hi = max(min(gap_lo_req, gap_center_limit), -gap_center_limit)
    _note("gap_center_min", gap_lo_req, gap_lo, "gap would overlap the arena boundary")
    _note("gap_center_max", gap_hi_req, gap_hi, "gap would overlap the arena boundary")

    # 3. Both barrier segments must still reach the boundary at the extreme
    #    centers. Lengthening the wall is the fix; shrinking the gap range is not.
    needed_segment = max(
        inner - (gap_lo + 0.5 * width_lo),
        inner + (gap_hi - 0.5 * width_lo),
    )
    segment = max(float(segment_length), needed_segment + epsilon)
    _note("segment_length", float(segment_length), segment, "segments must reach both boundaries")

    # 4. Leave room for a legal start/goal on both sides of the barrier. The
    #    poses must clear half the barrier thickness plus the OBB margin.
    min_distance = 0.5 * float(barrier_thickness) + obb_margin
    barrier_limit = max(inner - obb_margin - min_distance - epsilon, 0.0)
    barrier_lo = max(barrier_lo_req, -barrier_limit)
    barrier_hi = min(barrier_hi_req, barrier_limit)
    if barrier_lo > barrier_hi:
        barrier_lo = barrier_hi = max(min(barrier_lo_req, barrier_limit), -barrier_limit)
    _note("barrier_x_min", barrier_lo_req, barrier_lo, "no room left for start/goal")
    _note("barrier_x_max", barrier_hi_req, barrier_hi, "no room left for start/goal")

    # 5. Distances are measured from the barrier, so the worst case adds the
    #    largest |barrier_x| the range can produce.
    worst_barrier = max(abs(barrier_lo), abs(barrier_hi))
    distance_max = max(inner - obb_margin - worst_barrier - epsilon, min_distance)
    start = min(max(float(start_distance), min_distance), distance_max)
    goal_req_lo, goal_req_hi = _ordered(
        goal_distance_range
        if goal_distance_range is not None
        else (goal_distance, goal_distance)
    )
    goal_lo = min(max(goal_req_lo, min_distance), distance_max)
    goal_hi = min(max(goal_req_hi, goal_lo), distance_max)
    goal = 0.5 * (goal_lo + goal_hi)
    _note(
        "start_distance",
        float(start_distance),
        start,
        f"legal range [{min_distance:.2f}, {distance_max:.2f}]m",
    )
    _note(
        "goal_distance_min",
        goal_req_lo,
        goal_lo,
        f"legal range [{min_distance:.2f}, {distance_max:.2f}]m",
    )
    _note(
        "goal_distance_max",
        goal_req_hi,
        goal_hi,
        f"legal range [{min_distance:.2f}, {distance_max:.2f}]m",
    )

    # 6. The off-axis goal must also stay inside the room, worst case sitting on
    #    top of the most extreme gap center.
    offset_limit = max(inner - obb_margin - max(abs(gap_lo), abs(gap_hi)), 0.0)
    offset_req_lo, offset_req_hi = _ordered(
        goal_lateral_offset_range
        if goal_lateral_offset_range is not None
        else (goal_lateral_offset, goal_lateral_offset)
    )
    offset_lo = min(max(offset_req_lo, -offset_limit), offset_limit)
    offset_hi = min(max(offset_req_hi, offset_lo), offset_limit)
    offset = 0.5 * (offset_lo + offset_hi)
    _note(
        "goal_lateral_offset_min",
        offset_req_lo,
        offset_lo,
        f"goal would leave the room (limit +/-{offset_limit:.2f}m)",
    )
    _note(
        "goal_lateral_offset_max",
        offset_req_hi,
        offset_hi,
        f"goal would leave the room (limit +/-{offset_limit:.2f}m)",
    )

    return NarrowReplayLayout(
        gap_center_range=(gap_lo, gap_hi),
        barrier_x_range=(barrier_lo, barrier_hi),
        width_range=(width_lo, width_hi),
        start_distance=start,
        goal_distance=goal,
        goal_lateral_offset=offset,
        goal_distance_range=(goal_lo, goal_hi),
        goal_lateral_offset_range=(offset_lo, offset_hi),
        segment_length=segment,
        direction_mode=direction_mode,
        warnings=tuple(warnings),
    )
