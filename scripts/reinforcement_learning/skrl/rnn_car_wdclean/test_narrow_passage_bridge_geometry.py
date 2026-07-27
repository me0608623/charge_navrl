"""Unit tests for the SA5 narrow-passage bridge schedule and solvability proof."""

import importlib.util
from pathlib import Path
import sys

import pytest


_REPO = Path(__file__).resolve().parents[4]
_MODULE_PATH = (
    _REPO
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "charge_skrl"
    / "mdp"
    / "events"
    / "narrow_passage_bridge_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("narrow_passage_bridge_geometry", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

minimum_solvable_gap = _MODULE.minimum_solvable_gap
schedule_at = _MODULE.schedule_at
validate_constructed_scene = _MODULE.validate_constructed_scene
validate_closed_range = _MODULE.validate_closed_range
resolve_replay_layout = _MODULE.resolve_replay_layout
TRAINING_REPLAY_LAYOUT = _MODULE.TRAINING_REPLAY_LAYOUT


def test_schedule_matches_requested_curriculum() -> None:
    start = schedule_at(0.0)
    middle = schedule_at(0.5)
    end = schedule_at(1.0)

    assert (start.width_min, start.width_max, start.yaw_limit_deg, start.stress_ratio) == (
        1.4,
        1.6,
        2.0,
        0.0,
    )
    assert middle.width_min == pytest.approx(1.2)
    assert middle.width_max == pytest.approx(1.4)
    assert middle.yaw_limit_deg == pytest.approx(6.0)
    assert middle.stress_ratio == pytest.approx(0.0)
    assert end.width_min == pytest.approx(1.2)
    assert end.width_max == pytest.approx(1.4)
    assert end.yaw_limit_deg == pytest.approx(10.0)
    assert end.stress_ratio == pytest.approx(0.25)


def test_fixed_retention_schedule_does_not_ramp_or_add_stress() -> None:
    start = schedule_at(
        0.0,
        final_stress_ratio=0.25,
        fixed_width_range=(1.2, 1.4),
        fixed_yaw_limit_deg=4.0,
    )
    end = schedule_at(
        1.0,
        final_stress_ratio=0.25,
        fixed_width_range=(1.2, 1.4),
        fixed_yaw_limit_deg=4.0,
    )

    for schedule in (start, end):
        assert schedule.width_min == pytest.approx(1.2)
        assert schedule.width_max == pytest.approx(1.4)
        assert schedule.yaw_limit_deg == pytest.approx(4.0)
        assert schedule.stress_ratio == pytest.approx(0.0)


def test_fixed_retention_schedule_requires_complete_valid_override() -> None:
    with pytest.raises(ValueError):
        schedule_at(0.5, fixed_width_range=(1.2, 1.4))
    with pytest.raises(ValueError):
        schedule_at(
            0.5,
            fixed_width_range=(1.4, 1.2),
            fixed_yaw_limit_deg=4.0,
        )


def test_measured_obb_requires_point_eight_meter_gap() -> None:
    assert minimum_solvable_gap(robot_half_width=0.30, collision_buffer=0.10) == 0.8


def test_one_meter_stress_scene_is_constructively_solvable() -> None:
    ok, reason = validate_constructed_scene(
        gap_width=1.0,
        gap_center=1.0,
        barrier_x=0.5,
        start_x=-2.5,
        goal_x=3.5,
        room_half_extent=7.5,
        boundary_wall_width=1.0,
        segment_length=9.0,
    )
    assert ok, reason


def test_sub_obb_gap_is_rejected() -> None:
    ok, reason = validate_constructed_scene(
        gap_width=0.79,
        gap_center=0.0,
        barrier_x=0.0,
        start_x=-3.0,
        goal_x=3.0,
        room_half_extent=7.5,
        boundary_wall_width=1.0,
        segment_length=9.0,
    )
    assert not ok
    assert "OBB requirement" in reason


def test_short_segments_are_rejected() -> None:
    ok, reason = validate_constructed_scene(
        gap_width=1.4,
        gap_center=0.0,
        barrier_x=0.0,
        start_x=-3.0,
        goal_x=3.0,
        room_half_extent=7.5,
        boundary_wall_width=1.0,
        segment_length=5.0,
    )
    assert not ok
    assert "segment" in reason


# ---------------------------------------------------------------------------
# Replay layout resolution (play-side manual overrides)
# ---------------------------------------------------------------------------


def _corners(layout):
    """Enumerate the extreme sampled scenes a layout can produce."""
    for gap_center in layout.gap_center_range:
        for barrier_x in layout.barrier_x_range:
            for gap_width in layout.width_range:
                for direction in (1.0, -1.0):
                    for goal_distance in layout.goal_distance_range:
                        for goal_offset in layout.goal_lateral_offset_range:
                            yield dict(
                                gap_width=gap_width,
                                gap_center=gap_center,
                                barrier_x=barrier_x,
                                start_x=barrier_x - direction * layout.start_distance,
                                goal_x=barrier_x + direction * goal_distance,
                                goal_y=gap_center + goal_offset,
                            )


def test_training_layout_constant_matches_event_term_defaults() -> None:
    # These are the values hardcoded in the narrow_passage_bridge EventTerm and
    # never overridden by configure_narrow_passage_assets. Play-side defaults
    # must not silently drift from them.
    assert TRAINING_REPLAY_LAYOUT["gap_center_range"] == (-1.0, 1.0)
    assert TRAINING_REPLAY_LAYOUT["barrier_x_range"] == (-0.5, 0.5)
    assert TRAINING_REPLAY_LAYOUT["start_distance"] == 3.0
    assert TRAINING_REPLAY_LAYOUT["goal_distance"] == 3.0
    assert TRAINING_REPLAY_LAYOUT["goal_lateral_offset"] == 0.0
    assert TRAINING_REPLAY_LAYOUT["goal_distance_range"] == (3.0, 3.0)
    assert TRAINING_REPLAY_LAYOUT["goal_lateral_offset_range"] == (0.0, 0.0)
    assert TRAINING_REPLAY_LAYOUT["segment_length"] == 9.0
    assert TRAINING_REPLAY_LAYOUT["direction_mode"] == "random"


def test_training_defaults_pass_through_unclamped() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-1.0, 1.0),
        barrier_x_range=(-0.5, 0.5),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        segment_length=9.0,
        room_half_extent=7.0,
    )
    assert layout.warnings == ()
    assert layout.gap_center_range == (-1.0, 1.0)
    assert layout.barrier_x_range == (-0.5, 0.5)
    assert layout.start_distance == 3.0
    assert layout.goal_distance == 3.0
    assert layout.goal_distance_range == (3.0, 3.0)
    assert layout.goal_lateral_offset_range == (0.0, 0.0)
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_reversed_ranges_are_normalized() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(1.0, -1.0),
        barrier_x_range=(0.5, -0.5),
        width_range=(1.4, 1.2),
        start_distance=3.0,
        goal_distance=3.0,
        room_half_extent=7.0,
    )
    assert layout.gap_center_range == (-1.0, 1.0)
    assert layout.barrier_x_range == (-0.5, 0.5)
    assert layout.width_range == (1.2, 1.4)


def test_pinned_single_values_are_preserved() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(0.8, 0.8),
        barrier_x_range=(-0.25, -0.25),
        width_range=(1.0, 1.0),
        start_distance=2.5,
        goal_distance=4.0,
        room_half_extent=7.0,
    )
    assert layout.warnings == ()
    assert layout.gap_center_range == (0.8, 0.8)
    assert layout.barrier_x_range == (-0.25, -0.25)
    assert layout.width_range == (1.0, 1.0)
    assert layout.start_distance == 2.5
    assert layout.goal_distance == 4.0


def test_sub_obb_width_is_clamped_up_with_a_warning() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(0.0, 0.0),
        barrier_x_range=(0.0, 0.0),
        width_range=(0.5, 0.7),
        start_distance=3.0,
        goal_distance=3.0,
        room_half_extent=7.0,
    )
    assert layout.width_range[0] >= minimum_solvable_gap()
    assert layout.width_range[1] >= layout.width_range[0]
    assert any("width" in w for w in layout.warnings)


def test_far_goal_is_clamped_inside_the_room() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(0.0, 0.0),
        barrier_x_range=(-0.5, 0.5),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=9.0,
        room_half_extent=7.0,
    )
    assert layout.goal_distance < 9.0
    assert any("goal_distance" in w for w in layout.warnings)
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_zero_distance_is_pushed_off_the_barrier() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(0.0, 0.0),
        barrier_x_range=(0.0, 0.0),
        width_range=(1.2, 1.4),
        start_distance=0.0,
        goal_distance=0.0,
        room_half_extent=7.0,
    )
    # Both poses must clear half the 1.0 m barrier thickness plus the OBB margin.
    assert layout.start_distance >= 0.95
    assert layout.goal_distance >= 0.95
    assert any("start_distance" in w for w in layout.warnings)
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_gap_center_is_clamped_off_the_arena_boundary() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-6.5, 6.5),
        barrier_x_range=(0.0, 0.0),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        room_half_extent=7.0,
    )
    assert layout.gap_center_range[1] < 6.5
    assert layout.gap_center_range[0] > -6.5
    assert any("gap_center" in w for w in layout.warnings)
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_segment_length_grows_so_offset_gaps_still_reach_the_boundary() -> None:
    # A gap pinned at y=-4.0 leaves 4.0+ m above it; the default 9.0 m segment
    # is fine, but a deliberately short one must be lengthened, not rejected.
    layout = resolve_replay_layout(
        gap_center_range=(-4.0, 4.0),
        barrier_x_range=(0.0, 0.0),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        segment_length=4.0,
        room_half_extent=7.0,
    )
    assert layout.segment_length > 4.0
    assert any("segment_length" in w for w in layout.warnings)
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_goal_lateral_offset_is_clamped_inside_the_room() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-1.0, 1.0),
        barrier_x_range=(0.0, 0.0),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        goal_lateral_offset=9.0,
        room_half_extent=7.0,
    )
    assert any("goal_lateral_offset" in w for w in layout.warnings)
    worst_goal_y = layout.gap_center_range[1] + layout.goal_lateral_offset
    assert worst_goal_y <= 7.0 - 0.5 * 1.0 - 0.45 + 1e-9


def test_goal_lateral_offset_within_room_is_untouched() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-1.0, 1.0),
        barrier_x_range=(-0.5, 0.5),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        goal_lateral_offset=-1.5,
        room_half_extent=7.0,
    )
    assert layout.warnings == ()
    assert layout.goal_lateral_offset == -1.5


def test_n1_goal_ranges_are_preserved_and_solvable_at_every_corner() -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-1.0, 1.0),
        barrier_x_range=(-0.5, 0.5),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        goal_lateral_offset=0.0,
        goal_distance_range=(2.0, 4.0),
        goal_lateral_offset_range=(-1.5, 1.5),
        room_half_extent=7.0,
    )
    assert layout.warnings == ()
    assert layout.goal_distance_range == (2.0, 4.0)
    assert layout.goal_lateral_offset_range == (-1.5, 1.5)
    assert layout.goal_distance == 3.0
    assert layout.goal_lateral_offset == 0.0
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


def test_training_range_validation_rejects_reversed_or_nonfinite_values() -> None:
    assert validate_closed_range(
        "goal_distance_range", None, fallback=3.0, minimum=0.0
    ) == (3.0, 3.0)
    with pytest.raises(ValueError, match="ordered"):
        validate_closed_range(
            "goal_distance_range", (4.0, 2.0), fallback=3.0, minimum=0.0
        )
    with pytest.raises(ValueError, match="finite"):
        validate_closed_range(
            "goal_lateral_offset_range",
            (float("-inf"), 1.0),
            fallback=0.0,
        )


def test_wide_barrier_range_shrinks_before_distances_collapse() -> None:
    # barrier_x is free across almost the whole room: there is no room left for
    # a legal start/goal distance unless the barrier range itself is clamped.
    layout = resolve_replay_layout(
        gap_center_range=(0.0, 0.0),
        barrier_x_range=(-6.0, 6.0),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        room_half_extent=7.0,
    )
    assert layout.barrier_x_range[1] < 6.0
    assert any("barrier_x" in w for w in layout.warnings)
    assert layout.start_distance >= 0.95
    for scene in _corners(layout):
        ok, reason = validate_constructed_scene(
            room_half_extent=7.0,
            boundary_wall_width=1.0,
            segment_length=layout.segment_length,
            **scene,
        )
        assert ok, f"{reason} for {scene}"


@pytest.mark.parametrize("mode", ["random", "forward", "backward"])
def test_direction_modes_are_accepted(mode: str) -> None:
    layout = resolve_replay_layout(
        gap_center_range=(-1.0, 1.0),
        barrier_x_range=(-0.5, 0.5),
        width_range=(1.2, 1.4),
        start_distance=3.0,
        goal_distance=3.0,
        direction_mode=mode,
        room_half_extent=7.0,
    )
    assert layout.direction_mode == mode


def test_unknown_direction_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction_mode"):
        resolve_replay_layout(
            gap_center_range=(-1.0, 1.0),
            barrier_x_range=(-0.5, 0.5),
            width_range=(1.2, 1.4),
            start_distance=3.0,
            goal_distance=3.0,
            direction_mode="sideways",
            room_half_extent=7.0,
        )
