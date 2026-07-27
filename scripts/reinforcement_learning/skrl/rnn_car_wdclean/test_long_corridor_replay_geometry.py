"""Sim-free tests for the frozen deployment-corridor geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
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
        "mixed_iid",
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
    for mode in (
        "lateral", "longitudinal", "random_2d", "mixed", "mixed_iid",
        "env_stratified",
    ):
        assert normalize_dynamic_motion_weights(None, mode) is None


def test_motion_weights_rejected_outside_env_stratified():
    for mode in ("lateral", "longitudinal", "random_2d", "mixed", "mixed_iid"):
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


# ---------------------------------------------------------------------------
# mixed_iid: the unbiased per-obstacle variant. `mixed` must stay untouched as
# the legacy regression gate, so both are asserted side by side.
# ---------------------------------------------------------------------------


def _pair_histogram(motion_types) -> dict:
    """Counts of unordered family pairs across a [count, 2] motion tensor."""
    hist: dict = {}
    for row in motion_types.tolist():
        key = tuple(sorted(row))
        hist[key] = hist.get(key, 0) + 1
    return hist


def test_mixed_iid_is_a_registered_mode() -> None:
    assert validate_dynamic_motion_mode("mixed_iid") == "mixed_iid"


def test_legacy_mixed_still_forces_distinct_families_at_count_one() -> None:
    """Pins the legacy behaviour so a future edit cannot silently change it."""
    torch.manual_seed(7)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(1, spec, "cpu")
    for _ in range(200):
        _, _, motion_types, _ = sample_dynamic_trajectories(
            dynamic, lateral_waypoints, spec, "mixed"
        )
        assert motion_types[0, 0].item() != motion_types[0, 1].item()


def test_mixed_iid_small_batches_are_unbiased_over_time() -> None:
    """count == 1 is the common auto-reset batch; pair shares must hit IID."""
    torch.manual_seed(11)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(1, spec, "cpu")
    hist: dict = {}
    trials = 30000
    for _ in range(trials):
        _, _, motion_types, _ = sample_dynamic_trajectories(
            dynamic, lateral_waypoints, spec, "mixed_iid"
        )
        for key, value in _pair_histogram(motion_types).items():
            hist[key] = hist.get(key, 0) + value

    homogeneous = [(0, 0), (1, 1), (2, 2)]
    heterogeneous = [(0, 1), (0, 2), (1, 2)]
    for key in homogeneous:
        assert abs(hist.get(key, 0) / trials - 1 / 9) < 0.01, key
    for key in heterogeneous:
        assert abs(hist.get(key, 0) / trials - 2 / 9) < 0.01, key
    assert sum(hist.get(k, 0) for k in homogeneous) / trials == pytest.approx(
        1 / 3, abs=0.015
    )


def test_mixed_iid_marginal_family_share_is_uniform() -> None:
    torch.manual_seed(13)
    spec = LongCorridorSpec()
    _, dynamic, lateral_waypoints = sample_obstacle_layout(4096, spec, "cpu")
    _, _, motion_types, _ = sample_dynamic_trajectories(
        dynamic, lateral_waypoints, spec, "mixed_iid"
    )
    total = motion_types.numel()
    for family in (0, 1, 2):
        share = (motion_types == family).sum().item() / total
        assert abs(share - 1 / 3) < 0.02, family


def test_mixed_iid_rejects_weights() -> None:
    try:
        normalize_dynamic_motion_weights((0.3, 0.1, 0.6), "mixed_iid")
    except ValueError:
        return
    raise AssertionError("mixed_iid must not accept env-stratified weights")


# ---------------------------------------------------------------------------
# eval-only pause override. `None` must be a strict no-op so training and every
# historical gate keep the stock 0-5 step behaviour.
# ---------------------------------------------------------------------------

normalize_pause_steps_range = _MOD.normalize_pause_steps_range
apply_pause_override = _MOD.apply_pause_override


class _FakePatrol:
    def __init__(self, value=(0, 5)):
        self.pause_steps_range = value


class _FakeEnv:
    def __init__(self, patrol=None):
        class _Cfg:
            pass

        class _Sched:
            pass

        self.unwrapped = self
        if patrol is None:
            self._behavior_scheduler = None
        else:
            sched = _Sched()
            cfg = _Cfg()
            cfg.patrol = patrol
            sched.cfg = cfg
            self._behavior_scheduler = sched


def test_pause_none_is_a_no_op() -> None:
    patrol = _FakePatrol((0, 5))
    assert apply_pause_override(_FakeEnv(patrol), None) == (0, 5)
    assert patrol.pause_steps_range == (0, 5)


def test_pause_zero_override_applies_and_is_reported() -> None:
    patrol = _FakePatrol((0, 5))
    assert apply_pause_override(_FakeEnv(patrol), (0, 0)) == (0, 0)
    assert patrol.pause_steps_range == (0, 0)


def test_pause_override_without_scheduler_does_not_crash() -> None:
    assert apply_pause_override(_FakeEnv(None), (0, 0)) == (0, 0)
    assert apply_pause_override(_FakeEnv(None), None) is None


def test_pause_range_validation_rejects_malformed_input() -> None:
    assert normalize_pause_steps_range(None) is None
    assert normalize_pause_steps_range((0, 0)) == (0, 0)
    for bad in ((1,), (0, 1, 2), (-1, 3), (4, 2)):
        try:
            normalize_pause_steps_range(bad)
        except ValueError:
            continue
        raise AssertionError(f"pause range should be rejected: {bad}")


# ---------------------------------------------------------------------------
# random_2d wander kinematics. `patrol` stays the frozen default; `wander`
# replaces the fixed ~1.5 m ping-pong with a bounded random walk that has no
# turnaround point.
# ---------------------------------------------------------------------------

validate_random_2d_kinematics = _MOD.validate_random_2d_kinematics
corridor_wander_bounds = _MOD.corridor_wander_bounds


def test_random_2d_kinematics_validation() -> None:
    assert validate_random_2d_kinematics("patrol") == "patrol"
    assert validate_random_2d_kinematics("WANDER") == "wander"
    for bad in ("", "ping_pong", "random"):
        try:
            validate_random_2d_kinematics(bad)
        except ValueError:
            continue
        raise AssertionError(f"kinematics should be rejected: {bad}")


def test_wander_bounds_keep_the_obstacle_off_the_walls() -> None:
    spec = LongCorridorSpec()
    x_min, x_max, y_min, y_max = corridor_wander_bounds(spec)
    # Symmetric about the corridor centreline.
    assert x_min == pytest.approx(-x_max)
    assert y_min == pytest.approx(-y_max)
    # An obstacle sitting exactly on the bound must not overlap a wall.
    assert x_max + spec.obstacle_radius <= spec.inner_half_width
    assert y_max + spec.obstacle_radius <= 0.5 * spec.length
    # The wander area must be wide enough to be worth calling a corridor.
    assert x_max > 0.5
    assert y_max > 4.0


# ---------------------------------------------------------------------------
# Boundary reflection. step_random_walk documented a bounce it never
# implemented; without it a wandering obstacle walks straight through a 4 m
# corridor wall. These tests drive the real function via a minimal fake
# scheduler, so containment is proven rather than inferred from a smoke run.
# ---------------------------------------------------------------------------

import math as _math  # noqa: E402

_RB_PATH = _MODULE_PATH.parent / "rule_behaviors.py"


def _load_rule_behaviors():
    """Load rule_behaviors with its package-relative imports stubbed out."""
    import types

    pkg = types.ModuleType("_rb_pkg")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules["_rb_pkg"] = pkg
    spec = importlib.util.spec_from_file_location(
        "_rb_pkg.rule_behaviors", _RB_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeRandomWalkCfg:
    speed_range = (0.3, 0.6)
    direction_change_interval_range = (5, 25)
    max_turn_angle = _math.pi / 2
    smooth_turn_steps = 3


class _FakeSched:
    """Minimal stand-in exposing only what step_random_walk touches."""

    def __init__(self, n_env, n_obs, bounds=None, heading=0.0, speed=0.5):
        self.device = "cpu"
        shape = (n_env, n_obs)
        self.positions = torch.zeros(*shape, 2)
        self.velocities = torch.zeros(*shape, 2)
        self.rw_heading = torch.full(shape, float(heading))
        self.rw_target_heading = torch.full(shape, float(heading))
        self.rw_speed = torch.full(shape, float(speed))
        self.rw_timer = torch.zeros(*shape, dtype=torch.long)
        self.rw_change_interval = torch.full(shape, 10_000, dtype=torch.long)
        self.rw_turn_remaining = torch.zeros(*shape, dtype=torch.long)
        self.rw_bounds = torch.zeros(*shape, 4)
        # Static-conflict resolver participation: zero = legacy blind walk.
        self.pairwise_clearance = torch.zeros(*shape)
        self.behavior_type = torch.full(shape, 3, dtype=torch.long)  # active
        self.max_obstacles = n_obs
        # Patrol-capsule inputs read by the avoidance block; zero waypoints
        # means "not a two-point patrol", i.e. plain point-disc treatment.
        self.patrol_waypoints = torch.zeros(*shape, 4, 2)
        self.patrol_num_waypoints = torch.zeros(*shape, dtype=torch.long)
        if bounds is None:
            self.rw_bounds[..., 0] = float("-inf")
            self.rw_bounds[..., 1] = float("inf")
            self.rw_bounds[..., 2] = float("-inf")
            self.rw_bounds[..., 3] = float("inf")
        else:
            self.rw_bounds[...] = torch.tensor(bounds, dtype=torch.float32)

        class _Cfg:
            random_walk = _FakeRandomWalkCfg()

        self.cfg = _Cfg()


def test_unbounded_random_walk_behaviour_is_unchanged() -> None:
    """Infinite bounds must reproduce the original integrate-only path."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 1, bounds=None, heading=0.0, speed=0.5)
    mask = torch.ones(1, 1, dtype=torch.bool)
    for _ in range(50):
        rb.step_random_walk(sched, mask, 0.2)
    # Heading 0 => pure +x travel, 50 steps * 0.5 m/s * 0.2 s = 5 m, no clamp.
    assert sched.positions[0, 0, 0].item() == pytest.approx(5.0, abs=1e-4)


def test_wander_never_leaves_the_corridor_bounds() -> None:
    """The safety-critical property: a bounded walker cannot exit the box."""
    rb = _load_rule_behaviors()
    spec = LongCorridorSpec()
    x_min, x_max, y_min, y_max = corridor_wander_bounds(spec)
    torch.manual_seed(17)
    sched = _FakeSched(32, 2, bounds=(x_min, x_max, y_min, y_max))
    # Random headings and short change intervals to exercise the walls hard.
    sched.rw_heading.uniform_(-_math.pi, _math.pi)
    sched.rw_target_heading.copy_(sched.rw_heading)
    sched.rw_speed.uniform_(0.3, 0.6)
    sched.rw_change_interval.fill_(3)
    mask = torch.ones(32, 2, dtype=torch.bool)
    for _ in range(2000):
        rb.step_random_walk(sched, mask, 0.2)
        assert sched.positions[..., 0].min().item() >= x_min - 1e-5
        assert sched.positions[..., 0].max().item() <= x_max + 1e-5
        assert sched.positions[..., 1].min().item() >= y_min - 1e-5
        assert sched.positions[..., 1].max().item() <= y_max + 1e-5


def test_reflection_reverses_the_normal_component() -> None:
    """Hitting the +x wall must send the obstacle back, not slide it along."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 1, bounds=(-1.0, 1.0, -5.0, 5.0), heading=0.0, speed=0.6)
    mask = torch.ones(1, 1, dtype=torch.bool)
    for _ in range(12):
        rb.step_random_walk(sched, mask, 0.2)
    assert sched.positions[0, 0, 0].item() <= 1.0 + 1e-6
    # After bouncing off +x the velocity must point back toward -x.
    assert sched.velocities[0, 0, 0].item() < 0.0


def test_wander_keeps_moving_every_step() -> None:
    """No pause state: unlike patrol, speed never drops to zero."""
    rb = _load_rule_behaviors()
    spec = LongCorridorSpec()
    sched = _FakeSched(8, 2, bounds=corridor_wander_bounds(spec))
    sched.rw_change_interval.fill_(4)
    mask = torch.ones(8, 2, dtype=torch.bool)
    for _ in range(400):
        rb.step_random_walk(sched, mask, 0.2)
        assert sched.velocities.norm(dim=-1).min().item() > 1e-6


def test_configure_event_params_carry_random_2d_kinematics() -> None:
    """Regression: the reset event re-invokes setup on every auto-reset.

    If ``random_2d_kinematics`` is missing from the event params, the default
    ("patrol") silently overwrites the direct call's setting after the first
    reset — the measured symptom was a "wander" gate whose CR and paused-frame
    share matched patrol exactly. The event params are the only channel that
    survives auto-reset, so the key must be present there.
    """
    import re

    replay_src = (_MODULE_PATH.parent / "long_corridor_replay.py").read_text(
        encoding="utf-8"
    )
    # Locate the configure function body and its event.params.update block.
    match = re.search(
        r"def configure_long_corridor_assets\(.*?\n(def |\Z)",
        replay_src,
        re.S,
    )
    assert match is not None
    body = match.group(0)
    update = re.search(r"event\.params\.update\(\s*\{(.*?)\}\s*\)", body, re.S)
    assert update is not None, "configure must update the reset event params"
    assert '"random_2d_kinematics"' in update.group(1)
    # And the setup function must accept it, so the event call does not crash.
    assert "random_2d_kinematics: str = \"patrol\"" in replay_src


# ---------------------------------------------------------------------------
# Physical-penetration masks. Solvability is only checked at install time and
# wander is blind to the other obstacles, so overlap must be measured.
# ---------------------------------------------------------------------------

corridor_penetration_masks = _MOD.corridor_penetration_masks


def test_penetration_masks_detect_static_overlap() -> None:
    dyn = torch.tensor([[[0.0, 0.0], [3.0, 3.0]]])          # [1, 2, 2]
    static = torch.tensor([[[0.5, 0.0], [-2.0, -2.0]]])     # [1, 2, 2]
    ds, dd = corridor_penetration_masks(dyn, static, 0.35)
    # slot 0 sits 0.5 m from a static disc (< 0.7 threshold); slot 1 is clear.
    assert ds.tolist() == [[True, False]]
    assert dd.tolist() == [[False, False]]


def test_penetration_masks_detect_dynamic_pair_overlap_symmetrically() -> None:
    dyn = torch.tensor([[[0.0, 0.0], [0.6, 0.0]]])
    static = torch.zeros(1, 0, 2)
    ds, dd = corridor_penetration_masks(dyn, static, 0.35)
    assert ds.tolist() == [[False, False]]
    assert dd.tolist() == [[True, True]]  # overlap is mutual by construction


def test_penetration_masks_clear_at_exact_threshold() -> None:
    """Centres exactly 2r apart are touching, not penetrating."""
    dyn = torch.tensor([[[0.0, 0.0], [0.7, 0.0]]])
    static = torch.tensor([[[0.0, 0.7]]])
    ds, dd = corridor_penetration_masks(dyn, static, 0.35)
    assert not ds.any()
    assert not dd.any()


def test_penetration_masks_ignore_self_pairs_and_reject_bad_radius() -> None:
    dyn = torch.rand(4, 2, 2) * 10 + 100  # far from everything, incl. origin
    static = torch.zeros(4, 4, 2)
    ds, dd = corridor_penetration_masks(dyn, static, 0.35)
    assert not ds.any() and not dd.any()
    try:
        corridor_penetration_masks(dyn, static, 0.0)
    except ValueError:
        return
    raise AssertionError("non-positive radius must be rejected")


# ---------------------------------------------------------------------------
# Penetration hard gate: zero counts alone must not pass — the audit has to
# have actually run.
# ---------------------------------------------------------------------------

corridor_penetration_pass = _MOD.corridor_penetration_pass


def test_penetration_gate_passes_only_on_clean_nonempty_audit() -> None:
    assert corridor_penetration_pass(153600, 0) is True


def test_penetration_gate_fails_on_static_penetration_only() -> None:
    """Ruling lock: dynamic-vs-static penetration fails the gate; the gate
    takes no dynamic-dynamic argument at all — pedestrian crossings are
    recorded by the audit but never block a checkpoint."""
    assert corridor_penetration_pass(153600, 1) is False
    assert corridor_penetration_pass(153600, 7) is False
    import inspect

    params = list(inspect.signature(corridor_penetration_pass).parameters)
    assert params == ["audited_slot_frames", "dynamic_static_count"]


def test_penetration_gate_fails_when_audit_never_ran() -> None:
    """An empty audit reporting 'no penetrations' is a wiring failure, not a
    clean run — exactly the failure mode that produced the fake-wander gate."""
    assert corridor_penetration_pass(0, 0) is False


# ---------------------------------------------------------------------------
# Post-behavior static-conflict resolver. Ruling: dynamic obstacles may cross
# each other (recorded, not blocked); what stays forbidden is a dynamic
# obstacle passing through a static one. No capsule keep-outs.
# ---------------------------------------------------------------------------


def _step_with_resolver(rb, sched, walk_mask, dt=0.2, patrol_mask=None):
    prev = sched.positions.clone()
    if patrol_mask is not None:
        rb.step_patrol(sched, patrol_mask, dt)
    rb.step_random_walk(sched, walk_mask, dt)
    rb.resolve_static_conflicts(sched, prev)


def test_zero_clearance_keeps_legacy_blind_walk() -> None:
    """pairwise_clearance=0 must reproduce the blind walk bit-for-bit."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 2, bounds=(-5, 5, -5, 5), heading=0.0, speed=0.5)
    sched.behavior_type[0, 1] = 1  # BEHAVIOR_STATIC in slot 0's path
    sched.positions[0, 1] = torch.tensor([1.0, 0.0])
    mask = torch.zeros(1, 2, dtype=torch.bool)
    mask[0, 0] = True
    for _ in range(20):
        _step_with_resolver(rb, sched, mask)
    assert sched.positions[0, 0, 0].item() == pytest.approx(2.0, abs=1e-4)


def test_wander_never_penetrates_a_static_disc() -> None:
    rb = _load_rule_behaviors()
    torch.manual_seed(23)
    sched = _FakeSched(16, 3, bounds=(-1.5, 1.5, -4.5, 4.5))
    sched.rw_heading.uniform_(-_math.pi, _math.pi)
    sched.rw_target_heading.copy_(sched.rw_heading)
    sched.rw_speed.uniform_(0.3, 0.6)
    sched.rw_change_interval.fill_(4)
    sched.behavior_type[:, 2] = 1  # static disc at the centre
    sched.positions[:, 2] = 0.0
    sched.pairwise_clearance[:, 0] = 0.7
    sched.pairwise_clearance[:, 1] = 0.7
    sched.positions[:, 0] = torch.tensor([0.0, -3.0])
    sched.positions[:, 1] = torch.tensor([0.0, 3.0])
    mask = torch.zeros(16, 3, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = True
    for _ in range(2000):
        _step_with_resolver(rb, sched, mask)
        for wander_slot in (0, 1):
            d = (sched.positions[:, wander_slot] - sched.positions[:, 2]).norm(dim=-1)
            assert d.min().item() >= 0.7 - 1e-5


def test_dynamic_obstacles_may_cross_each_other() -> None:
    """Ruling lock: independent pedestrians cross; no invisible walls between
    them. Two head-on walkers with straight headings must pass through."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 2, bounds=(-5, 5, -5, 5), speed=0.5)
    sched.rw_heading[0, 0] = 0.0            # slot 0 heads +x
    sched.rw_heading[0, 1] = _math.pi       # slot 1 heads -x
    sched.rw_target_heading.copy_(sched.rw_heading)
    sched.positions[0, 0] = torch.tensor([-2.0, 0.0])
    sched.positions[0, 1] = torch.tensor([2.0, 0.0])
    sched.pairwise_clearance[0, :] = 0.7
    mask = torch.ones(1, 2, dtype=torch.bool)
    crossed = False
    for _ in range(60):
        _step_with_resolver(rb, sched, mask)
        if (
            sched.positions[0, 0, 0].item() > sched.positions[0, 1, 0].item()
        ):
            crossed = True
            break
    assert crossed, "walkers must be able to pass through each other"


def test_resolver_reverts_a_blind_patrol_only_against_statics() -> None:
    """A managed patrol stalls at a static disc instead of passing through,
    but is NOT deflected by the other dynamic slot."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 3, bounds=(-5, 5, -5, 5))
    # Slot 0: patrol marching +x through a static at x=1.5 and a dynamic at x=0.5.
    sched.behavior_type[0, 0] = 2
    sched.patrol_waypoints[0, 0, 0] = torch.tensor([-2.0, 0.0])
    sched.patrol_waypoints[0, 0, 1] = torch.tensor([4.0, 0.0])
    sched.patrol_num_waypoints[0, 0] = 2
    sched.patrol_wp_index = torch.zeros(1, 3, dtype=torch.long)
    sched.patrol_wp_index[0, 0] = 1
    sched.patrol_speed = torch.zeros(1, 3)
    sched.patrol_speed[0, 0] = 0.5
    sched.patrol_pause_remaining = torch.zeros(1, 3, dtype=torch.long)
    sched.positions[0, 0] = torch.tensor([-2.0, 0.0])

    class _PatrolCfg:
        pause_steps_range = (0, 0)

    sched.cfg.patrol = _PatrolCfg()
    sched.behavior_type[0, 1] = 3            # dynamic (random walk, inert here)
    sched.positions[0, 1] = torch.tensor([0.5, 0.0])
    sched.rw_speed[0, 1] = 0.0
    sched.behavior_type[0, 2] = 1            # static
    sched.positions[0, 2] = torch.tensor([1.5, 0.0])
    sched.pairwise_clearance[0, 0] = 0.7
    patrol_mask = torch.zeros(1, 3, dtype=torch.bool)
    patrol_mask[0, 0] = True
    walk_mask = torch.zeros(1, 3, dtype=torch.bool)
    for _ in range(200):
        _step_with_resolver(rb, sched, walk_mask, patrol_mask=patrol_mask)
    x = sched.positions[0, 0, 0].item()
    # Passed straight through the dynamic at 0.5 but stalled before the
    # static's 0.7 clearance at 1.5.
    assert x > 0.5
    assert x <= 1.5 - 0.7 + 1e-5


def test_resolver_catches_a_graze_that_endpoint_checks_miss() -> None:
    """Codex counterexample: both endpoints outside the clearance, but the
    midpath dips inside. The swept-segment test must revert; an endpoint test
    provably cannot."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 2, bounds=(-50, 50, -50, 50))
    sched.behavior_type[0, 1] = 1  # static at origin
    sched.positions[0, 1] = torch.tensor([0.0, 0.0])
    sched.pairwise_clearance[0, 0] = 0.7
    prev = sched.positions.clone()
    prev[0, 0] = torch.tensor([-0.5, 0.5])          # dist 0.707 >= 0.7
    sched.positions[0, 0] = torch.tensor([0.5, 0.5])  # dist 0.707 >= 0.7
    # midpoint (0, 0.5): dist 0.5 < 0.7 -> the step grazes the disc
    rb.resolve_static_conflicts(sched, prev)
    assert torch.allclose(sched.positions[0, 0], prev[0, 0])


def test_resolver_blocks_full_tunnelling_at_any_step_size() -> None:
    """A step long enough to jump clean across the disc must still revert;
    with segment checks, no step-length guard is needed."""
    rb = _load_rule_behaviors()
    sched = _FakeSched(1, 2, bounds=(-50, 50, -50, 50))
    sched.behavior_type[0, 1] = 1
    sched.positions[0, 1] = torch.tensor([0.0, 0.0])
    sched.pairwise_clearance[0, 0] = 0.7
    prev = sched.positions.clone()
    prev[0, 0] = torch.tensor([-5.0, 0.0])
    sched.positions[0, 0] = torch.tensor([5.0, 0.0])  # straight through
    rb.resolve_static_conflicts(sched, prev)
    assert torch.allclose(sched.positions[0, 0], prev[0, 0])


# ---------------------------------------------------------------------------
# 2026-07-27：slot 形狀可變（預設維持歷史 4S+2D）
# ---------------------------------------------------------------------------


def test_default_shapes_are_the_legacy_template():
    s, d, w = sample_obstacle_layout(8, LongCorridorSpec(), "cpu")
    assert tuple(s.shape) == (8, _MOD.LEGACY_MAX_STATIC, 2)
    assert tuple(d.shape) == (8, _MOD.LEGACY_MAX_DYNAMIC, 2)
    assert tuple(w.shape) == (8, _MOD.LEGACY_MAX_DYNAMIC, 2, 2)


def test_mixed_shapes_expand_to_five_by_five():
    s, d, w = sample_obstacle_layout(
        8, LongCorridorSpec(), "cpu",
        max_static=_MOD.MIXED_MAX_STATIC, max_dynamic=_MOD.MIXED_MAX_DYNAMIC,
    )
    assert tuple(s.shape) == (8, 5, 2)
    assert tuple(d.shape) == (8, 5, 2)
    assert tuple(w.shape) == (8, 5, 2, 2)


def test_fifth_static_pairs_with_an_existing_row():
    """第 5 個靜態補在既有列的對側，形成左右各一的『門口』。"""
    spec = LongCorridorSpec()
    rows = len(spec.static_y)
    s, _, _ = sample_obstacle_layout(64, spec, "cpu", max_static=5, max_dynamic=2)
    for env in s:
        paired, extra = env[0], env[rows]
        assert float(extra[0]) * float(paired[0]) < 0.0, "配對 slot 須在該列反側"
        assert abs(float(extra[1]) - float(paired[1])) < 2 * spec.static_xy_jitter + 1e-6


def test_paired_gate_leaves_a_traversable_central_channel():
    spec = LongCorridorSpec()
    channel = 2 * spec.static_x - 2 * spec.obstacle_radius
    assert channel > 2 * spec.robot_conservative_radius


def test_legacy_default_is_bit_identical_to_explicit_legacy_caps():
    torch.manual_seed(0)
    a = sample_obstacle_layout(16, LongCorridorSpec(), "cpu")
    torch.manual_seed(0)
    b = sample_obstacle_layout(
        16, LongCorridorSpec(), "cpu",
        max_static=_MOD.LEGACY_MAX_STATIC, max_dynamic=_MOD.LEGACY_MAX_DYNAMIC,
    )
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_validate_counts_defaults_still_reject_high_density():
    """預設路徑不得默默接受 5S+5D —— 呼叫端必須明確傳上限。"""
    with pytest.raises(ValueError):
        _MOD.validate_obstacle_counts(5, 5)
    _MOD.validate_obstacle_counts(
        5, 5, max_static=_MOD.MIXED_MAX_STATIC, max_dynamic=_MOD.MIXED_MAX_DYNAMIC,
    )


def test_permute_slots_default_off_is_bit_identical():
    """預設不打散 —— 既有 gate 與 SA1–SA6 血緣逐位元不變。"""
    spec = LongCorridorSpec()
    torch.manual_seed(4242)
    a = sample_obstacle_layout(64, spec, "cpu")
    torch.manual_seed(4242)
    b = sample_obstacle_layout(64, spec, "cpu", permute_slots=False)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_permute_slots_lets_a_prefix_mask_reach_every_static_row():
    """高密度場用前綴 mask 關 slot；不打散的話 S=3 永遠碰不到 y=3.0 那列。"""
    spec = LongCorridorSpec()
    torch.manual_seed(7)
    static, _, _ = sample_obstacle_layout(
        512, spec, "cpu", max_static=_MOD.MIXED_MAX_STATIC,
        max_dynamic=_MOD.MIXED_MAX_DYNAMIC, permute_slots=True,
    )
    prefix_y = static[:, :3, 1]
    for row_y in spec.static_y:
        hit = (prefix_y - row_y).abs() <= spec.static_xy_jitter + 1e-6
        assert bool(hit.any()), f"row y={row_y} never appears in a 3-static prefix"


def test_permute_slots_keeps_dynamic_start_on_its_own_patrol_lane():
    """動態起點與其 waypoint 必須同一條橫排 —— 兩者要用同一個排列。"""
    spec = LongCorridorSpec()
    torch.manual_seed(11)
    _, dynamic, waypoints = sample_obstacle_layout(
        256, spec, "cpu", max_static=_MOD.MIXED_MAX_STATIC,
        max_dynamic=_MOD.MIXED_MAX_DYNAMIC, permute_slots=True,
    )
    assert torch.allclose(dynamic[:, :, 1], waypoints[:, :, 0, 1])
    assert torch.allclose(dynamic[:, :, 1], waypoints[:, :, 1, 1])
