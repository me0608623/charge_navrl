"""Fail-closed contracts for the paired stage-2 exposure experiment."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SKRL = HERE.parent
sys.path.insert(0, str(SKRL))

from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c50_equalweight_control_p25 as parent,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c50_stage2_equal_p25 as a2,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c50_stage2_mild040_p25 as b2,
)


REPO = Path(__file__).resolve().parents[4]
METADATA_FIELDS = {"name", "description", "tags", "notes"}


def _diff(left, right) -> set[str]:
    return {
        field.name
        for field in fields(left)
        if field.name not in METADATA_FIELDS
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_authorization_and_evidence_are_hash_locked() -> None:
    authorization = json.loads(a2.AUTHORIZATION_RECORD.read_text())
    assert hashlib.sha256(a2.AUTHORIZATION_RECORD.read_bytes()).hexdigest() == (
        a2.AUTHORIZATION_RECORD_SHA256
    )
    assert authorization["decision"] == (
        "RUN_PAIRED_STAGE2_EQUAL_VS_MILD_030_030_040_P25"
    )
    assert authorization["lineage"]["sa6"] == "HOLD_NOT_AUTHORIZED"
    assert authorization["post_training_screen"]["sa6_auto_start"] is False


def test_parent_checkpoint_identity_and_optimizer_are_present() -> None:
    path = Path(a2.PARENT_CHECKPOINT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        a2.PARENT_CHECKPOINT_SHA256
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 24
    assert payload["total_steps"] == 3200
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert a2.CONFIG.no_resume_optimizer is False
    assert b2.CONFIG.no_resume_optimizer is False


def test_a2_changes_only_the_parent_checkpoint() -> None:
    assert _diff(a2.CONFIG, parent.CONFIG) == {"checkpoint"}


def test_b2_changes_only_checkpoint_and_motion_weights_from_parent() -> None:
    assert _diff(b2.CONFIG, parent.CONFIG) == {
        "checkpoint",
        "long_corridor_dynamic_motion_weights",
    }


def test_only_behavioral_difference_between_paired_arms_is_weight() -> None:
    assert _diff(b2.CONFIG, a2.CONFIG) == {
        "long_corridor_dynamic_motion_weights"
    }
    assert a2.CONFIG.long_corridor_dynamic_motion_weights is None
    assert b2.CONFIG.long_corridor_dynamic_motion_weights == (0.30, 0.30, 0.40)


def test_runtime_contract_is_identical() -> None:
    fixed = (
        "initial_stage",
        "fixed_stage",
        "num_envs",
        "seed",
        "rollout_length",
        "timesteps",
        "save_interval",
        "long_corridor_fraction",
        "narrow_passage_fraction",
        "long_corridor_speed_density_mix",
        "speed_rate",
        "speed_rate_obs",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "actuator_delay_range",
    )
    for name in fixed:
        assert getattr(a2.CONFIG, name) == getattr(b2.CONFIG, name)
    assert a2.CONFIG.initial_stage == 5
    assert a2.CONFIG.fixed_stage is True
    assert a2.CONFIG.num_envs == 1024
    assert a2.CONFIG.seed == 42
    assert a2.CONFIG.rollout_length == 128
    assert a2.CONFIG.timesteps == 3200
    assert a2.CONFIG.speed_rate == 0.7
    assert a2.CONFIG.actuator_delay_range == (1, 2)


def test_reward_action_observation_network_and_ppo_do_not_drift() -> None:
    forbidden_fragments = (
        "reward",
        "penalty",
        "weight",
        "action",
        "observation",
        "hidden",
        "rnn",
        "learning_rate",
        "optimizer",
        "mini_batch",
        "epochs",
        "grad_clip",
    )
    allowed = {"long_corridor_dynamic_motion_weights"}
    for field in fields(a2.CONFIG):
        if field.name in allowed:
            continue
        if any(fragment in field.name for fragment in forbidden_fragments):
            assert getattr(a2.CONFIG, field.name) == getattr(b2.CONFIG, field.name)


def test_branch_is_bounded_and_cannot_authorize_sa6() -> None:
    assert a2.TRAINING_ITERATIONS == 25
    assert a2.CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
    source = (
        Path(a2.__file__).read_text()
        + Path(b2.__file__).read_text()
    )
    assert "sa6_hold" in source
    assert "HOLD_NOT_AUTHORIZED" in source
    assert "systemd-run" not in source
