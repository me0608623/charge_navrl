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
normalize_dynamic_motion_weights = _MOD.normalize_dynamic_motion_weights
sample_motion_families = _MOD.sample_motion_families
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

    for mode in (
        "lateral",
        "longitudinal",
        "random_2d",
        "mixed",
        "env_stratified",
    ):
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


def test_env_stratified_mode_balances_pure_motion_environments() -> None:
    torch.manual_seed(46)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(96, spec, "cpu")

    _, _, motion_types, _ = sample_dynamic_trajectories(
        dynamic, lateral_waypoints, spec, "env_stratified"
    )
    assert torch.equal(motion_types[:, 0], motion_types[:, 1])

    counts = torch.bincount(motion_types[:, 0], minlength=3)
    assert torch.equal(counts, torch.tensor([32, 32, 32]))


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


# ---------------------------------------------------------------------------
# Weighted env-level stratification (D1)
# ---------------------------------------------------------------------------


def test_motion_weights_normalize_to_one():
    weights = normalize_dynamic_motion_weights((30.0, 10.0, 60.0), "env_stratified")
    assert weights is not None
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(weights[0] - 0.30) < 1e-9
    assert abs(weights[2] - 0.60) < 1e-9


def test_motion_weights_none_is_allowed_for_every_mode():
    for mode in ("lateral", "longitudinal", "random_2d", "mixed", "env_stratified"):
        assert normalize_dynamic_motion_weights(None, mode) is None


def test_motion_weights_rejected_outside_env_stratified():
    for mode in ("lateral", "longitudinal", "random_2d", "mixed"):
        try:
            normalize_dynamic_motion_weights((0.3, 0.1, 0.6), mode)
        except ValueError:
            continue
        raise AssertionError(f"weights should be rejected for mode={mode}")


def test_motion_weights_reject_malformed_values():
    for bad in ((0.5, 0.5), (-1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (float("nan"), 1.0, 1.0)):
        try:
            normalize_dynamic_motion_weights(bad, "env_stratified")
        except ValueError:
            continue
        raise AssertionError(f"weights should be rejected: {bad}")


def test_family_draw_sums_to_count_and_stays_within_one_env():
    weights = normalize_dynamic_motion_weights((0.30, 0.10, 0.60), "env_stratified")
    torch.manual_seed(0)
    for count in (1, 2, 3, 5, 7, 10, 16, 84, 100, 1024):
        for _ in range(50):
            families = sample_motion_families(count, weights)
            counts = torch.bincount(families, minlength=3)
            assert int(counts.sum()) == count
            for index, weight in enumerate(weights):
                assert abs(int(counts[index]) - count * weight) < 1.0


def test_single_env_resets_stay_unbiased_over_time():
    """Regression: a deterministic quota starved longitudinal on small batches.

    With weights 0.30/0.10/0.60 a largest-remainder quota returned [0, 0, 1] for
    every single-env reset, so longitudinal never appeared. Reset batches are
    usually small, so the long-run mix must hold at count == 1.
    """
    weights = normalize_dynamic_motion_weights((0.30, 0.10, 0.60), "env_stratified")
    torch.manual_seed(1234)
    totals = torch.zeros(3, dtype=torch.long)
    for _ in range(8000):
        totals += torch.bincount(sample_motion_families(1, weights), minlength=3)
    observed = (totals.double() / int(totals.sum())).tolist()
    for index, weight in enumerate(weights):
        assert abs(observed[index] - weight) < 0.02


def test_mixed_small_batches_stay_unbiased_over_time():
    weights = normalize_dynamic_motion_weights((0.30, 0.10, 0.60), "env_stratified")
    torch.manual_seed(99)
    totals = torch.zeros(3, dtype=torch.long)
    sizes = [1, 2, 3, 5, 7]
    for step in range(4000):
        totals += torch.bincount(
            sample_motion_families(sizes[step % len(sizes)], weights), minlength=3
        )
    observed = (totals.double() / int(totals.sum())).tolist()
    for index, weight in enumerate(weights):
        assert abs(observed[index] - weight) < 0.02


def test_zero_weight_family_is_never_drawn():
    weights = normalize_dynamic_motion_weights((0.5, 0.0, 0.5), "env_stratified")
    torch.manual_seed(7)
    totals = torch.zeros(3, dtype=torch.long)
    for _ in range(2000):
        totals += torch.bincount(sample_motion_families(1, weights), minlength=3)
    assert int(totals[1]) == 0
    assert int(totals[0]) > 0 and int(totals[2]) > 0


def test_env_stratified_weights_drive_family_counts():
    spec = LongCorridorSpec()
    count = 84
    weights = normalize_dynamic_motion_weights((0.30, 0.10, 0.60), "env_stratified")
    _, dynamic, waypoints = sample_obstacle_layout(count, spec, torch.device("cpu"))
    _, _, motion_types, _ = sample_dynamic_trajectories(
        dynamic, waypoints, spec, "env_stratified", (0.30, 0.10, 0.60)
    )
    # both obstacles of an env share one family
    assert bool((motion_types[:, 0] == motion_types[:, 1]).all())
    counts = torch.bincount(motion_types[:, 0], minlength=3).tolist()
    assert sum(counts) == count
    for index, weight in enumerate(weights):
        assert abs(counts[index] - count * weight) < 1.0


def test_env_stratified_without_weights_stays_balanced():
    spec = LongCorridorSpec()
    count = 84
    _, dynamic, waypoints = sample_obstacle_layout(count, spec, torch.device("cpu"))
    _, _, motion_types, _ = sample_dynamic_trajectories(
        dynamic, waypoints, spec, "env_stratified"
    )
    counts = torch.bincount(motion_types[:, 0], minlength=3).tolist()
    assert sum(counts) == count
    assert max(counts) - min(counts) <= 1


def test_cumulative_family_counters_match_long_run_weights():
    """Simulate many small reset batches and check the cumulative audit.

    This mirrors what the runtime counters accumulate: env-level counts, slot
    counts (two obstacles per env) and pure-env count. Only a cumulative view
    can reveal a small-batch sampling bias.
    """
    weights = normalize_dynamic_motion_weights((0.30, 0.10, 0.60), "env_stratified")
    spec = LongCorridorSpec()
    torch.manual_seed(2026)
    env_counts_total = torch.zeros(3, dtype=torch.long)
    slot_counts_total = torch.zeros(3, dtype=torch.long)
    pure_env_total = 0
    batches = [1, 1, 2, 1, 3, 1, 5, 2, 1, 7] * 60
    for count in batches:
        _, dynamic, waypoints = sample_obstacle_layout(
            count, spec, torch.device("cpu")
        )
        _, _, motion_types, _ = sample_dynamic_trajectories(
            dynamic, waypoints, spec, "env_stratified", (0.30, 0.10, 0.60)
        )
        active = motion_types[:, :2]
        env_counts_total += torch.bincount(active[:, 0], minlength=3)
        slot_counts_total += torch.bincount(active.flatten(), minlength=3)
        pure_env_total += int((active == active[:, :1]).all(dim=1).sum())

    total = int(env_counts_total.sum())
    assert total == sum(batches)
    # slots are exactly two per env because both obstacles share one family
    assert torch.equal(slot_counts_total, env_counts_total * 2)
    # every env is pure under env_stratified
    assert pure_env_total == total
    observed = (env_counts_total.double() / total).tolist()
    for index, weight in enumerate(weights):
        assert abs(observed[index] - weight) < 0.02
