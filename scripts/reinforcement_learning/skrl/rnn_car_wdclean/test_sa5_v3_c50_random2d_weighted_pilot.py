from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_modular.configs import (
    e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 as source,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_random2d_weighted_p25 as pilot,
)


REPO = Path(__file__).resolve().parents[4]
AUTH = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_random2d_weighted_p25_authorization_20260824.json"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_conditional_authorization_and_evidence_are_frozen() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    assert _sha256(AUTH) == pilot.AUTHORIZATION_RECORD_SHA256
    assert auth["decision"] == (
        "ACTIVATE_SHORT_HORIZON_ACTION_OPPORTUNITY_BRANCH"
    )
    analysis = REPO / auth["evidence"]["analysis"]
    manifest = REPO / auth["evidence"]["manifest"]
    assert _sha256(analysis) == auth["evidence"]["analysis_sha256"]
    assert _sha256(manifest) == auth["evidence"]["manifest_sha256"]
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    assert payload["decision"] == "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED"


def test_parent_hash_metadata_and_optimizer_are_exact() -> None:
    path = Path(pilot.PARENT_CHECKPOINT)
    assert _sha256(path) == pilot.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 49
    assert payload["total_steps"] == 6400
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_pilot_changes_only_parent_motion_weights_and_budget() -> None:
    metadata = {"name", "description", "tags", "notes"}
    drifted = {
        field.name
        for field in fields(pilot.CONFIG)
        if field.name not in metadata
        and getattr(pilot.CONFIG, field.name) != getattr(source.CONFIG, field.name)
    }
    assert drifted == {
        "checkpoint",
        "long_corridor_dynamic_motion_weights",
        "timesteps",
    }


def test_motion_intervention_is_bounded_and_normalized() -> None:
    assert source.CONFIG.long_corridor_dynamic_motion_weights is None
    assert pilot.MOTION_WEIGHTS == (0.15, 0.15, 0.70)
    assert sum(pilot.MOTION_WEIGHTS) == 1.0
    assert pilot.CONFIG.long_corridor_dynamic_motion_mode == "env_stratified"
    assert pilot.CONFIG.long_corridor_random_2d_kinematics == "patrol"


def test_budget_and_optimizer_resume_are_exact() -> None:
    assert pilot.TRAINING_ITERATIONS == 25
    assert pilot.CONFIG.timesteps == 25 * 128
    assert pilot.CONFIG.save_interval == 25
    assert pilot.CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
    assert pilot.CONFIG.no_resume_optimizer is False


def test_scene_speed_density_sensor_and_actuator_contracts_are_preserved() -> None:
    protected = (
        "long_corridor_fraction",
        "narrow_passage_fraction",
        "long_corridor_speed_density_mix",
        "long_corridor_dynamic_speed_range",
        "long_corridor_free_width",
        "long_corridor_length",
        "speed_rate",
        "speed_rate_obs",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "lidar_frame_stack",
        "actuator_delay_range",
        "actuator_velocity_scale",
        "actuator_motor_lag",
    )
    for name in protected:
        assert getattr(pilot.CONFIG, name) == getattr(source.CONFIG, name), name
    assert 1.0 - pilot.CONFIG.long_corridor_fraction - pilot.CONFIG.narrow_passage_fraction == 0.78


def test_reward_model_and_ppo_contracts_are_preserved() -> None:
    protected_tokens = (
        "reward",
        "penalty",
        "optimizer",
        "learning_rate",
        "lr",
        "hidden",
        "rnn",
        "encoder",
        "critic",
        "policy",
        "clip",
        "epoch",
        "mini_batch",
    )
    for field in fields(pilot.CONFIG):
        if any(token in field.name for token in protected_tokens):
            assert getattr(pilot.CONFIG, field.name) == getattr(
                source.CONFIG, field.name
            ), field.name


def test_lineage_remains_ungraduated_and_sa6_forbidden() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    assert auth["lineage"] == {
        "sa5": "HOLD_NOT_GRADUATED",
        "anchor": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
        "sa6": "HOLD_NOT_AUTHORIZED",
    }
    assert auth["post_training_screen"]["extension_auto_authorized"] is False
    assert auth["post_training_screen"]["graduation_auto_authorized"] is False
