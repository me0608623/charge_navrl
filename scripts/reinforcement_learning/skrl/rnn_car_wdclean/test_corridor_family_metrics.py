"""CPU tests for training-time corridor motion-family accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SKRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKRL_ROOT))

from rnn_car_wdclean.corridor_family_metrics import (  # noqa: E402
    CORRIDOR_FAMILY_NAMES,
    NON_CORRIDOR,
    CorridorFamilyMetrics,
    CorridorFamilySnapshot,
    snapshot_corridor_families,
)


def _family_id(name: str) -> int:
    return CORRIDOR_FAMILY_NAMES.index(name)


class FakeEnv:
    def __init__(
        self,
        active,
        ready=None,
        motion=None,
        episode_length=None,
        profile=None,
    ):
        self._long_corridor_active = torch.tensor(active, dtype=torch.bool)
        if ready is not None:
            self._long_corridor_obstacles_ready = torch.tensor(
                ready, dtype=torch.bool
            )
        if motion is not None:
            self._long_corridor_dynamic_motion_type = torch.as_tensor(
                motion, dtype=torch.long
            ).clone()
        if episode_length is not None:
            self.episode_length_buf = torch.tensor(
                episode_length, dtype=torch.long
            )
        if profile is not None:
            self._long_corridor_speed_density_profile = torch.tensor(
                profile, dtype=torch.long
            )


def _snapshot(names, *, reset=None):
    ids = [
        NON_CORRIDOR if name is None else _family_id(name)
        for name in names
    ]
    return CorridorFamilySnapshot(
        family_id=torch.tensor(ids, dtype=torch.long),
        is_reset=torch.tensor(
            reset if reset is not None else [False] * len(ids),
            dtype=torch.bool,
        ),
    )


def _breakdown(
    n,
    *,
    goal=(),
    wall=(),
    obstacle=(),
    other=(),
    static=(),
    dynamic=(),
):
    def mask(indices):
        value = torch.zeros(n, dtype=torch.bool)
        value[list(indices)] = True
        return value

    return {
        "goal_reached": mask(goal),
        "wall_collision": mask(wall),
        "obs_collision": mask(obstacle),
        "other_death": mask(other),
        "static_obs_collision": mask(static),
        "dynamic_obs_collision": mask(dynamic),
        "progress_reward": torch.arange(n, dtype=torch.float32) + 0.5,
        "goal_reward": mask(goal).float() * 10.0,
    }


def _actions(n):
    return {
        "v_x": torch.linspace(-0.1, 0.4, n),
        "omega": torch.linspace(0.0, 1.2, n),
    }


def _step(
    metrics,
    snapshot,
    *,
    breakdown,
    done,
    terminated,
    truncated,
    corridor_mask=None,
    reward=None,
    actions=None,
):
    n = metrics.num_envs
    metrics.step(
        snapshot=snapshot,
        corridor_scene_mask=(
            snapshot.family_id >= 0
            if corridor_mask is None
            else torch.tensor(corridor_mask, dtype=torch.bool)
        ),
        reward=(
            torch.arange(n, dtype=torch.float32)
            if reward is None
            else torch.tensor(reward, dtype=torch.float32)
        ),
        reward_breakdown=breakdown,
        charge_actions=_actions(n) if actions is None else actions,
        done=torch.tensor(done, dtype=torch.bool),
        terminated_flat=torch.tensor(terminated, dtype=torch.bool),
        truncated_flat=torch.tensor(truncated, dtype=torch.bool),
    )


def test_snapshot_classifies_every_supported_family_without_silent_fallback():
    env = FakeEnv(
        active=[False, True, True, True, True, True, True, False],
        ready=[False, True, True, True, True, True, False, False],
        motion=[
            [-1, -1],
            [0, -1],
            [1, 1],
            [2, -1],
            [0, 1],
            [-1, -1],
            [-1, -1],
            [99, -1],
        ],
        episode_length=[0] * 8,
        profile=[-1, 0, 1, 2, 3, 4, 4, 99],
    )
    result = snapshot_corridor_families(env, 8, "cpu")
    expected = [
        NON_CORRIDOR,
        _family_id("lateral"),
        _family_id("longitudinal"),
        _family_id("random_2d"),
        _family_id("mixed"),
        _family_id("no_dynamic"),
        _family_id("unready"),
        NON_CORRIDOR,
    ]
    assert result.family_id.tolist() == expected
    assert result.is_reset.tolist() == [True] * 8
    assert result.profile_id.tolist() == [-1, 0, 1, 2, 3, 4, 4, -1]


def test_snapshot_supports_a_ready_corridor_with_zero_motion_slots():
    env = FakeEnv(
        active=[True],
        ready=[True],
        motion=torch.empty((1, 0), dtype=torch.long),
        episode_length=[0],
    )
    result = snapshot_corridor_families(env, 1, "cpu")
    assert result.family_id.item() == _family_id("no_dynamic")


def test_snapshot_without_corridor_state_is_all_non_corridor():
    class NoCorridor:
        episode_length_buf = torch.tensor([0, 4])

    result = snapshot_corridor_families(NoCorridor(), 2, "cpu")
    assert result.family_id.tolist() == [NON_CORRIDOR, NON_CORRIDOR]
    assert result.is_reset.tolist() == [False, False]


def test_active_corridor_requires_ready_motion_and_reset_metadata():
    env = FakeEnv(active=[True])
    with pytest.raises(RuntimeError, match="missing family accounting state"):
        snapshot_corridor_families(env, 1, "cpu")


def test_unknown_active_motion_family_is_rejected():
    env = FakeEnv(
        active=[True],
        ready=[True],
        motion=[[7]],
        episode_length=[0],
    )
    with pytest.raises(RuntimeError, match="unknown corridor motion"):
        snapshot_corridor_families(env, 1, "cpu")


def test_snapshot_rejects_wrong_motion_shape():
    env = FakeEnv(
        active=[True, False],
        ready=[True, False],
        motion=[[0]],
        episode_length=[0, 0],
    )
    with pytest.raises(RuntimeError, match="must have shape"):
        snapshot_corridor_families(env, 2, "cpu")


def test_all_families_emit_outcomes_actions_signals_and_shares():
    names = list(CORRIDOR_FAMILY_NAMES)
    n = len(names)
    metrics = CorridorFamilyMetrics(n, "cpu")
    breakdown = _breakdown(
        n,
        goal=[0],
        wall=[1],
        obstacle=[3, 4],
        other=[5],
        static=[3],
        dynamic=[4],
    )
    _step(
        metrics,
        _snapshot(names, reset=[True] * n),
        breakdown=breakdown,
        done=[True] * n,
        terminated=[True, True, False, True, True, True],
        truncated=[False, False, True, False, False, False],
    )
    result = metrics.collect(expected_corridor_episodes=n)

    assert result["corridor_family/accounting/reconciliation_ok"] == 1.0
    assert result["corridor_family/accounting/active_steps_total"] == n
    assert result["corridor_family/accounting/resets_total"] == n
    assert result["corridor_family/accounting/episodes_total"] == n
    assert result["corridor_family/accounting/active_step_share_sum"] == 1.0
    assert result["corridor_family/accounting/reset_share_sum"] == 1.0
    assert (
        result[
            "corridor_family/accounting/completed_episode_share_sum"
        ]
        == 1.0
    )

    for name in names:
        prefix = f"corridor_family/{name}"
        assert result[f"{prefix}/active_step_share"] == pytest.approx(1 / n)
        assert result[f"{prefix}/reset_share"] == pytest.approx(1 / n)
        assert result[f"{prefix}/completed_episode_share"] == pytest.approx(
            1 / n
        )
        assert result[f"{prefix}/action_coverage"] == 1.0
        assert f"{prefix}/signal/progress_reward_mean_per_step" in result
        assert f"{prefix}/signal/total_reward_mean_per_step" in result

    assert result["corridor_family/lateral/sr"] == 1.0
    assert result["corridor_family/longitudinal/cr"] == 1.0
    assert result["corridor_family/longitudinal/wall_cr"] == 1.0
    assert result["corridor_family/random_2d/timeout"] == 1.0
    assert result["corridor_family/mixed/static_obstacle_cr"] == 1.0
    assert result["corridor_family/no_dynamic/dynamic_obstacle_cr"] == 1.0
    assert result["corridor_family/unready/other_death"] == 1.0


def test_action_threshold_metrics_use_shared_corridor_gate_semantics():
    metrics = CorridorFamilyMetrics(2, "cpu")
    actions = {
        "v_x": torch.tensor([0.09, -0.03]),
        "omega": torch.tensor([0.81, 1.01]),
    }
    _step(
        metrics,
        _snapshot(["lateral", "lateral"], reset=[True, True]),
        breakdown=_breakdown(2),
        done=[False, False],
        terminated=[False, False],
        truncated=[False, False],
        actions=actions,
    )
    result = metrics.collect(expected_corridor_episodes=0)
    prefix = "corridor_family/lateral"
    assert result[f"{prefix}/stop_command_fraction"] == 1.0
    assert result[f"{prefix}/reverse_command_fraction"] == 0.5
    assert result[f"{prefix}/high_turn_fraction"] == 1.0
    assert result[f"{prefix}/extreme_turn_fraction"] == 0.5
    assert result[f"{prefix}/low_speed_high_turn_fraction"] == 1.0


def test_iteration_reset_preserves_open_episode_identity():
    metrics = CorridorFamilyMetrics(1, "cpu")
    snapshot = _snapshot(["lateral"], reset=[False])
    _step(
        metrics,
        snapshot,
        breakdown=_breakdown(1),
        done=[False],
        terminated=[False],
        truncated=[False],
    )
    first = metrics.collect(expected_corridor_episodes=0)
    assert first["corridor_family/accounting/late_attach_count"] == 1.0

    metrics.reset_iteration()
    _step(
        metrics,
        snapshot,
        breakdown=_breakdown(1, goal=[0]),
        done=[True],
        terminated=[True],
        truncated=[False],
    )
    second = metrics.collect(expected_corridor_episodes=1)
    assert second["corridor_family/lateral/sr"] == 1.0
    assert second["corridor_family/accounting/late_attach_count"] == 0.0


def test_unready_episode_can_upgrade_once_installation_finishes():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["unready"], reset=[True]),
        breakdown=_breakdown(1),
        done=[False],
        terminated=[False],
        truncated=[False],
    )
    _step(
        metrics,
        _snapshot(["longitudinal"], reset=[False]),
        breakdown=_breakdown(1, goal=[0]),
        done=[True],
        terminated=[True],
        truncated=[False],
    )
    result = metrics.collect(expected_corridor_episodes=1)
    assert result["corridor_family/unready/reset_count"] == 1.0
    assert result["corridor_family/longitudinal/episodes"] == 1.0
    assert result["corridor_family/accounting/unready_upgrade_count"] == 1.0


def test_family_change_before_episode_end_is_rejected():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["lateral"], reset=[True]),
        breakdown=_breakdown(1),
        done=[False],
        terminated=[False],
        truncated=[False],
    )
    with pytest.raises(RuntimeError, match="changed or disappeared"):
        _step(
            metrics,
            _snapshot(["longitudinal"], reset=[False]),
            breakdown=_breakdown(1),
            done=[False],
            terminated=[False],
            truncated=[False],
        )


def test_scene_and_family_masks_must_match_exactly():
    metrics = CorridorFamilyMetrics(1, "cpu")
    with pytest.raises(RuntimeError, match="masks disagree"):
        _step(
            metrics,
            _snapshot(["lateral"], reset=[True]),
            corridor_mask=[False],
            breakdown=_breakdown(1),
            done=[False],
            terminated=[False],
            truncated=[False],
        )


def test_completed_episodes_must_reconcile_with_scene_corridor():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["lateral"], reset=[True]),
        breakdown=_breakdown(1, goal=[0]),
        done=[True],
        terminated=[True],
        truncated=[False],
    )
    with pytest.raises(RuntimeError, match="do not reconcile"):
        metrics.collect(expected_corridor_episodes=2)


def test_terminal_outcome_requires_exactly_one_class():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["lateral"], reset=[True]),
        breakdown=_breakdown(1),
        done=[True],
        terminated=[True],
        truncated=[False],
    )
    with pytest.raises(RuntimeError, match="exactly one class"):
        metrics.collect(expected_corridor_episodes=1)


def test_obstacle_subtype_without_obstacle_collision_is_rejected():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["lateral"], reset=[True]),
        breakdown=_breakdown(1, goal=[0], static=[0]),
        done=[True],
        terminated=[True],
        truncated=[False],
    )
    with pytest.raises(RuntimeError, match="subtype occurred without"):
        metrics.collect(expected_corridor_episodes=1)


def test_intentional_nonfinite_aux_signal_is_covered_not_fatal():
    metrics = CorridorFamilyMetrics(2, "cpu")
    breakdown = _breakdown(2)
    breakdown["diagnostic_distance"] = torch.tensor(
        [float("inf"), 2.0]
    )
    _step(
        metrics,
        _snapshot(["lateral", "lateral"], reset=[True, True]),
        breakdown=breakdown,
        done=[False, False],
        terminated=[False, False],
        truncated=[False, False],
    )
    result = metrics.collect(expected_corridor_episodes=0)
    prefix = "corridor_family/lateral/signal/diagnostic_distance"
    assert result[f"{prefix}_coverage"] == 0.5
    assert result[f"{prefix}_nonfinite_fraction"] == 0.5
    assert result[f"{prefix}_mean_per_step"] == 2.0


def test_nonfinite_total_reward_is_fatal_at_collection():
    metrics = CorridorFamilyMetrics(1, "cpu")
    _step(
        metrics,
        _snapshot(["lateral"], reset=[True]),
        breakdown=_breakdown(1),
        reward=[float("nan")],
        done=[False],
        terminated=[False],
        truncated=[False],
    )
    with pytest.raises(RuntimeError, match="total_reward"):
        metrics.collect(expected_corridor_episodes=0)
