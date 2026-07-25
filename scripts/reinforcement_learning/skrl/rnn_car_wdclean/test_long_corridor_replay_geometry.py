"""Sim-free tests for the frozen deployment-corridor geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
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
    / "long_corridor_replay_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("long_corridor_replay_geometry", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

LongCorridorSpec = _MOD.LongCorridorSpec
layout_is_constructively_solvable = _MOD.layout_is_constructively_solvable
sample_obstacle_layout = _MOD.sample_obstacle_layout
validate_spec = _MOD.validate_spec
validate_obstacle_counts = _MOD.validate_obstacle_counts
wall_geometry = _MOD.wall_geometry


def test_wall_inner_faces_are_exactly_four_metres_apart() -> None:
    spec = LongCorridorSpec()
    centers, sizes = wall_geometry(3, spec, "cpu")
    left_inner = centers[:, 0, 0] + 0.5 * sizes[:, 0, 0]
    right_inner = centers[:, 1, 0] - 0.5 * sizes[:, 1, 0]
    assert torch.allclose(right_inner - left_inner, torch.full((3,), 4.0))
    assert torch.allclose(sizes[:, :, 1], torch.full((3, 2), 10.0))


def test_sampled_layouts_stay_inside_and_keep_static_centerline_open() -> None:
    torch.manual_seed(42)
    spec = LongCorridorSpec()
    static, dynamic, waypoints = sample_obstacle_layout(4096, spec, "cpu")
    valid = layout_is_constructively_solvable(static, dynamic, waypoints, spec)
    assert valid.all()
    assert spec.centerline_static_clearance > 0.0


def test_invalid_wall_clearance_is_rejected() -> None:
    spec = LongCorridorSpec(free_width=2.5)
    try:
        validate_spec(spec)
    except ValueError:
        return
    raise AssertionError("too-narrow deployment corridor was accepted")


def test_curriculum_obstacle_subsets_stay_within_frozen_capacity() -> None:
    for static, dynamic in ((2, 0), (3, 1), (4, 2)):
        validate_obstacle_counts(static, dynamic)

    for static, dynamic in ((-1, 0), (5, 0), (0, -1), (0, 3)):
        try:
            validate_obstacle_counts(static, dynamic)
        except ValueError:
            continue
        raise AssertionError(
            f"invalid corridor obstacle subset accepted: {static}S+{dynamic}D"
        )
