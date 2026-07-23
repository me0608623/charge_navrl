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
