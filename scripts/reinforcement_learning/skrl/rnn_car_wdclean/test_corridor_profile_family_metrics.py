"""Contract tests for profile x family training accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SKRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKRL_ROOT))

from rnn_car_wdclean.corridor_family_metrics import (  # noqa: E402
    CORRIDOR_FAMILY_NAMES,
    CorridorFamilySnapshot,
)
from rnn_car_wdclean.corridor_profile_family_metrics import (  # noqa: E402
    CorridorProfileFamilyMetrics,
    normalize_profile_specs,
)


MIX = (
    ((0, 1), (0.50, 0.70), 0.40),
    ((1, 1), (0.50, 0.70), 0.60),
)


def _family(name: str) -> int:
    return CORRIDOR_FAMILY_NAMES.index(name)


def _snapshot(profiles, families, *, reset):
    return CorridorFamilySnapshot(
        family_id=torch.tensor([_family(name) for name in families]),
        is_reset=torch.tensor(reset, dtype=torch.bool),
        profile_id=torch.tensor(profiles, dtype=torch.long),
    )


def _breakdown(n, *, goal=(), collision=(), other=()):
    def mask(indices):
        result = torch.zeros(n, dtype=torch.bool)
        result[list(indices)] = True
        return result

    return {
        "goal_reached": mask(goal),
        "wall_collision": torch.zeros(n, dtype=torch.bool),
        "obs_collision": mask(collision),
        "other_death": mask(other),
    }


def _step(
    metrics,
    snapshot,
    *,
    done,
    terminated,
    truncated,
    breakdown,
):
    metrics.step(
        snapshot=snapshot,
        corridor_scene_mask=torch.ones(metrics.num_envs, dtype=torch.bool),
        reward_breakdown=breakdown,
        done=torch.tensor(done, dtype=torch.bool),
        terminated_flat=torch.tensor(terminated, dtype=torch.bool),
        truncated_flat=torch.tensor(truncated, dtype=torch.bool),
    )


def test_profile_labels_preserve_p060_density_identity():
    specs = normalize_profile_specs(MIX)
    assert [spec.label for spec in specs] == ["p060_0s1d", "p060_1s1d"]


def test_profile_and_family_metrics_emit_exact_rates_and_shares():
    metrics = CorridorProfileFamilyMetrics(4, "cpu", MIX)
    snapshot = _snapshot(
        [0, 0, 1, 1],
        ["lateral", "longitudinal", "random_2d", "mixed"],
        reset=[True] * 4,
    )
    _step(
        metrics,
        snapshot,
        done=[True] * 4,
        terminated=[True, True, False, True],
        truncated=[False, False, True, False],
        breakdown=_breakdown(4, goal=[0], collision=[1], other=[3]),
    )
    result = metrics.collect(
        expected_corridor_episodes=4,
        expected_active_steps=4,
        expected_resets=4,
    )

    p0 = "corridor_profile/p060_0s1d"
    p1 = "corridor_profile/p060_1s1d"
    assert result[f"{p0}/reset_count"] == 2.0
    assert result[f"{p0}/active_steps"] == 2.0
    assert result[f"{p0}/episodes"] == 2.0
    assert result[f"{p0}/sr"] == 0.5
    assert result[f"{p0}/cr"] == 0.5
    assert result[f"{p0}/timeout"] == 0.0
    assert result[f"{p1}/sr"] == 0.0
    assert result[f"{p1}/cr"] == 0.0
    assert result[f"{p1}/timeout"] == 0.5
    assert result["corridor_profile_family/p060_0s1d/lateral/sr"] == 1.0
    assert result["corridor_profile_family/p060_0s1d/longitudinal/cr"] == 1.0
    assert result[
        "corridor_profile_family/p060_1s1d/random_2d/timeout"
    ] == 1.0
    assert result["corridor_profile_family/accounting/reconciliation_ok"] == 1.0


def test_episode_profile_change_is_rejected():
    metrics = CorridorProfileFamilyMetrics(1, "cpu", MIX)
    _step(
        metrics,
        _snapshot([0], ["lateral"], reset=[True]),
        done=[False],
        terminated=[False],
        truncated=[False],
        breakdown=_breakdown(1),
    )
    with pytest.raises(RuntimeError, match="changed/disappeared"):
        _step(
            metrics,
            _snapshot([1], ["lateral"], reset=[False]),
            done=[False],
            terminated=[False],
            truncated=[False],
            breakdown=_breakdown(1),
        )


def test_unknown_active_profile_fails_closed():
    metrics = CorridorProfileFamilyMetrics(1, "cpu", MIX)
    with pytest.raises(RuntimeError, match="unknown speed-density profile"):
        _step(
            metrics,
            _snapshot([-1], ["lateral"], reset=[True]),
            done=[False],
            terminated=[False],
            truncated=[False],
            breakdown=_breakdown(1),
        )


def test_collect_rejects_any_family_accounting_mismatch():
    metrics = CorridorProfileFamilyMetrics(1, "cpu", MIX)
    _step(
        metrics,
        _snapshot([0], ["lateral"], reset=[True]),
        done=[True],
        terminated=[True],
        truncated=[False],
        breakdown=_breakdown(1, goal=[0]),
    )
    with pytest.raises(RuntimeError, match="does not reconcile"):
        metrics.collect(
            expected_corridor_episodes=1,
            expected_active_steps=2,
            expected_resets=1,
        )


def test_trainer_wires_monitoring_without_policy_or_reward_mutation():
    trainer = (
        SKRL_ROOT / "train" / "train_rnn_car_wdclip.py"
    ).read_text(encoding="utf-8")
    assert "CorridorProfileFamilyMetrics" in trainer
    assert "corridor_profile_family/accounting/reconciliation_ok" not in trainer
    assert "corridor_speed_density_mix=getattr(" in trainer
    assert "self._corridor_profile_family_metrics.step(" in trainer
    assert "expected_active_steps=int(" in trainer
