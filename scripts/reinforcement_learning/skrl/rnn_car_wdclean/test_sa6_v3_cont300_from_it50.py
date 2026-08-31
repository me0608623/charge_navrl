"""Fail-closed contract tests for the SA6 it50 +300 continuation."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_modular.configs import e2e_sa6_v3_from_sa5_c600_p50 as source
from rnn_car_modular.configs import e2e_sa6_v3_cont300_from_it50 as continuation
from rnn_car_modular.configs import e2e_sa6_v3_cont300_from_it50_smoke as smoke
from rnn_car_modular.configs.registry import get_experiment_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authorization_matches_the_exact_continuation() -> None:
    record = json.loads(
        continuation.AUTHORIZATION_RECORD.read_text(encoding="utf-8")
    )
    assert _sha256(continuation.AUTHORIZATION_RECORD) == continuation.AUTHORIZATION_RECORD_SHA256
    assert record["decision"] == "HUMAN_AUTHORIZED_EXACT_CONTINUATION_IT50_PLUS_300"
    assert record["automatic_actions"]["training_launch_authorized"] is True
    assert record["automatic_actions"]["extend_beyond_conceptual_it350"] is False
    assert record["automatic_actions"]["start_sa7"] is False


def test_parent_is_completed_it50_with_resumable_ppo_optimizer() -> None:
    assert _sha256(continuation.PARENT_CHECKPOINT) == continuation.PARENT_CHECKPOINT_SHA256
    payload = torch.load(
        continuation.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 6_400
    assert payload["iteration"] == 49
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert continuation.CONFIG.no_resume_optimizer is False


def test_budget_and_conceptual_checkpoints_are_exact() -> None:
    assert continuation.PARENT_CONCEPTUAL_ITERATION == 50
    assert continuation.ADDITIONAL_TRAINING_ITERATIONS == 300
    assert continuation.CONCEPTUAL_END_ITERATION == 350
    assert continuation.CONFIG.num_envs == 1024
    assert continuation.CONFIG.seed == 42
    assert continuation.CONFIG.rollout_length == 128
    assert continuation.CONFIG.timesteps == 38_400
    assert continuation.CONFIG.save_interval == 50
    assert continuation.CHECKPOINTS == (
        (100, "checkpoint_6400.pt"),
        (150, "checkpoint_12800.pt"),
        (200, "checkpoint_19200.pt"),
        (250, "checkpoint_25600.pt"),
        (300, "checkpoint_32000.pt"),
        (350, "checkpoint_38400.pt"),
    )


def test_only_checkpoint_budget_and_save_interval_change() -> None:
    changed = {
        field.name
        for field in fields(continuation.CONFIG)
        if field.name not in continuation.METADATA_FIELDS
        and getattr(continuation.CONFIG, field.name) != getattr(source.CONFIG, field.name)
    }
    assert changed == {"checkpoint", "timesteps", "save_interval"}


def test_sim2real_scene_reward_network_and_teacher_contracts_are_unchanged() -> None:
    for field in fields(continuation.CONFIG):
        if field.name in continuation.METADATA_FIELDS | continuation.EXPECTED_DIFFS:
            continue
        assert getattr(continuation.CONFIG, field.name) == getattr(source.CONFIG, field.name)
    assert continuation.CONFIG.initial_stage == 6
    assert continuation.CONFIG.lidar_frame_stack == 8
    assert continuation.CONFIG.vlp16_noise_mode == "full"
    assert continuation.CONFIG.lidar_distractor_eligibility == "valid_return_only"
    assert continuation.CONFIG.enable_actuator_dr is True
    assert continuation.CONFIG.actuator_delay_range == (1, 2)
    assert continuation.CONFIG.speed_rate == 0.7
    assert continuation.CONFIG.teacher_retention_checkpoint is None
    assert continuation.CONFIG.teacher_retention_weight == 0.0
    assert continuation.CONFIG.teacher_retention_rollout_override is False
    assert continuation.CONFIG.previous_stage_teacher_checkpoint is None
    assert continuation.CONFIG.corridor_teacher_distill_epochs == 0


def test_source_and_completion_evidence_are_sha_locked() -> None:
    for path, expected in (
        (continuation.SOURCE_CONFIG, continuation.SOURCE_CONFIG_SHA256),
        (continuation.COMPLETION_RECORD, continuation.COMPLETION_RECORD_SHA256),
        (continuation.TRAINER, continuation.TRAINER_SHA256),
        (continuation.CORRIDOR_FAMILY_METRICS, continuation.CORRIDOR_FAMILY_METRICS_SHA256),
        (
            continuation.CORRIDOR_PROFILE_FAMILY_METRICS,
            continuation.CORRIDOR_PROFILE_FAMILY_METRICS_SHA256,
        ),
        (continuation.LONG_CORRIDOR_REPLAY, continuation.LONG_CORRIDOR_REPLAY_SHA256),
        (
            continuation.LONG_CORRIDOR_GEOMETRY,
            continuation.LONG_CORRIDOR_GEOMETRY_SHA256,
        ),
    ):
        assert _sha256(path) == expected


def test_smoke_changes_only_runtime_scale() -> None:
    changed = {
        field.name
        for field in fields(smoke.CONFIG)
        if field.name not in smoke.METADATA_FIELDS
        and getattr(smoke.CONFIG, field.name) != getattr(continuation.CONFIG, field.name)
    }
    assert changed == {"num_envs", "timesteps", "save_interval"}
    assert smoke.CONFIG.num_envs == 64
    assert smoke.CONFIG.timesteps == 128
    assert smoke.CONFIG.save_interval == 1


def test_registry_resolves_continuation_and_smoke() -> None:
    resolved = get_experiment_config("e2e_sa6_v3_cont300_from_it50")
    resolved_smoke = get_experiment_config("e2e_sa6_v3_cont300_from_it50_smoke")
    assert resolved.name == continuation.CONFIG.name
    assert resolved.checkpoint == str(continuation.PARENT_CHECKPOINT)
    assert resolved_smoke.num_envs == 64


def test_config_import_has_no_launch_side_effect() -> None:
    source_text = Path(continuation.__file__).read_text(encoding="utf-8")
    for forbidden in ("systemctl", "systemd-run", "subprocess", "Popen("):
        assert forbidden not in source_text
    assert continuation.TRAINING_STARTED is False
