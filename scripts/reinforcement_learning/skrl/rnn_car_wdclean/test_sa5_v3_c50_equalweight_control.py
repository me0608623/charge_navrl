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
    e2e_sa5_v3_c50_equalweight_control_p25 as control,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_random2d_weighted_p25 as weighted,
)


REPO = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _behavioral_diffs(left: object, right: object) -> set[str]:
    metadata = {"name", "description", "tags", "notes"}
    return {
        field.name
        for field in fields(left)
        if field.name not in metadata
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_authorization_and_trigger_evidence_are_frozen() -> None:
    auth = json.loads(control.AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
    assert _sha256(control.AUTHORIZATION_RECORD) == control.AUTHORIZATION_RECORD_SHA256
    assert auth["decision"] == "RUN_MATCHED_CONTINUATION_CONTROL_P25"
    assert _sha256(control.WEIGHTED_SCREEN_SUMMARY) == (
        control.WEIGHTED_SCREEN_SUMMARY_SHA256
    )
    assert _sha256(control.WEIGHTED_COMPLETION) == (
        control.WEIGHTED_COMPLETION_SHA256
    )


def test_parent_and_embedded_optimizer_are_exact() -> None:
    path = Path(control.PARENT_CHECKPOINT)
    assert _sha256(path) == control.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 49
    assert payload["total_steps"] == 6400
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_control_changes_only_parent_and_budget_from_source() -> None:
    assert _behavioral_diffs(control.CONFIG, source.CONFIG) == {
        "checkpoint",
        "timesteps",
    }


def test_control_and_weighted_are_exactly_paired() -> None:
    assert _behavioral_diffs(control.CONFIG, weighted.CONFIG) == {
        "long_corridor_dynamic_motion_weights"
    }
    assert control.CONFIG.long_corridor_dynamic_motion_weights is None
    assert weighted.CONFIG.long_corridor_dynamic_motion_weights == (
        0.15,
        0.15,
        0.70,
    )


def test_budget_and_optimizer_resume_are_exact() -> None:
    assert control.TRAINING_ITERATIONS == 25
    assert control.CONFIG.timesteps == 25 * 128
    assert control.CONFIG.save_interval == 25
    assert control.CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
    assert control.CONFIG.no_resume_optimizer is False


def test_scene_sensor_actuator_reward_and_ppo_contracts_are_paired() -> None:
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
        "speed",
        "lidar",
        "actuator",
        "corridor",
        "narrow",
    )
    for field in fields(control.CONFIG):
        if any(token in field.name for token in protected_tokens):
            if field.name == "long_corridor_dynamic_motion_weights":
                continue
            assert getattr(control.CONFIG, field.name) == getattr(
                weighted.CONFIG, field.name
            ), field.name


def test_lineage_stays_ungraduated_and_sa6_is_forbidden() -> None:
    auth = json.loads(control.AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
    assert auth["lineage"]["sa5"] == "HOLD_NOT_GRADUATED"
    assert auth["lineage"]["control"] == (
        "MATCHED_DEVELOPMENT_CONTROL_NOT_PARENT"
    )
    assert auth["lineage"]["sa6"] == "HOLD_NOT_AUTHORIZED"
    assert auth["post_training_screen"]["extension_auto_authorized"] is False
    assert auth["post_training_screen"]["graduation_auto_authorized"] is False
