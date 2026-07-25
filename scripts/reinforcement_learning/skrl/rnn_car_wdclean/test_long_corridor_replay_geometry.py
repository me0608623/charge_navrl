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
sample_dynamic_trajectories = _MOD.sample_dynamic_trajectories
sample_obstacle_layout = _MOD.sample_obstacle_layout
validate_dynamic_motion_mode = _MOD.validate_dynamic_motion_mode
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


def test_all_motion_families_are_bounded_and_constructively_solvable() -> None:
    torch.manual_seed(43)
    spec = LongCorridorSpec()
    static, dynamic, lateral_waypoints = sample_obstacle_layout(
        4096, spec, "cpu"
    )

    for mode in ("lateral", "longitudinal", "random_2d", "mixed"):
        starts, waypoints, motion_types, target_indices = (
            sample_dynamic_trajectories(
                dynamic, lateral_waypoints, spec, mode
            )
        )
        valid = layout_is_constructively_solvable(
            static, starts, waypoints, spec
        )
        assert valid.all(), mode
        assert target_indices.shape == (4096, 2)
        assert torch.isin(
            target_indices, torch.tensor([0, 1])
        ).all()
        assert motion_types.shape == (4096, 2)


def test_motion_family_axes_match_their_names() -> None:
    torch.manual_seed(44)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(256, spec, "cpu")

    _, lateral, lateral_types, _ = sample_dynamic_trajectories(
        dynamic, lateral_waypoints, spec, "lateral"
    )
    lateral_delta = (lateral[:, :, 1] - lateral[:, :, 0]).abs()
    assert (lateral_types == _MOD.MOTION_LATERAL).all()
    assert (lateral_delta[..., 0] > 2.0).all()
    assert torch.equal(lateral_delta[..., 1], torch.zeros_like(lateral_delta[..., 1]))

    starts, longitudinal, longitudinal_types, targets = (
        sample_dynamic_trajectories(
            dynamic, lateral_waypoints, spec, "longitudinal"
        )
    )
    longitudinal_delta = (
        longitudinal[:, :, 1] - longitudinal[:, :, 0]
    ).abs()
    assert (longitudinal_types == _MOD.MOTION_LONGITUDINAL).all()
    assert torch.equal(
        longitudinal_delta[..., 0],
        torch.zeros_like(longitudinal_delta[..., 0]),
    )
    assert (longitudinal_delta[..., 1] > 6.0).all()
    target = torch.gather(
        longitudinal,
        2,
        targets[..., None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    assert (target[:, 0, 1] < starts[:, 0, 1]).all()
    assert (target[:, 1, 1] > starts[:, 1, 1]).all()

    _, random_2d, random_types, _ = sample_dynamic_trajectories(
        dynamic, lateral_waypoints, spec, "random_2d"
    )
    random_delta = (random_2d[:, :, 1] - random_2d[:, :, 0]).abs()
    assert (random_types == _MOD.MOTION_RANDOM_2D).all()
    assert (random_delta[..., 0] >= 0.30).all()
    assert (random_delta[..., 1] >= 0.30).all()


def test_mixed_mode_is_balanced_over_a_gate_batch() -> None:
    torch.manual_seed(45)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(64, spec, "cpu")

    _, _, motion_types, _ = sample_dynamic_trajectories(
        dynamic, lateral_waypoints, spec, "mixed"
    )
    counts = torch.bincount(motion_types.flatten(), minlength=3)

    assert counts.min().item() >= 42
    assert counts.max().item() <= 43


def test_invalid_motion_mode_is_rejected() -> None:
    try:
        validate_dynamic_motion_mode("unbounded_random_walk")
    except ValueError:
        return
    raise AssertionError("unsupported corridor motion mode was accepted")


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
