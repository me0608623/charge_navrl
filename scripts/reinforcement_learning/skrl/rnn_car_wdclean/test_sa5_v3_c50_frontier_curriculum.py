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
    e2e_sa5_v3_c50_frontier_curriculum_phase1_p25 as phase1,
)


REPO = Path(__file__).resolve().parents[4]
AUTH = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_authorization_and_frontier_evidence_are_frozen() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    assert _sha256(AUTH) == phase1.AUTHORIZATION_RECORD_SHA256
    assert auth["decision"] == (
        "HUMAN_AUTHORIZED_SA5_V3_C50_JOINT_FRONTIER_CURRICULUM_P50"
    )
    summary = REPO / auth["evidence"]["summary"]
    assert _sha256(summary) == auth["evidence"]["summary_sha256"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["recommended_next_action"] == (
        "BUILD_JOINT_1D_2D_AND_1S_2S_4S_CURRICULUM_PILOT_FROM_C50"
    )
    assert payload["source_fingerprint_stable"] is True


def test_anchor_hash_metadata_and_optimizer_are_exact() -> None:
    path = Path(phase1.PARENT_CHECKPOINT)
    assert _sha256(path) == phase1.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 49
    assert payload["total_steps"] == 6400
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_anchor_remains_ungraduated_and_sa6_is_forbidden() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    assert auth["lineage_status"] == {
        "sa5": "HOLD_NOT_GRADUATED",
        "anchor": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
        "sa6": "HOLD_NOT_AUTHORIZED",
    }
    assert auth["pilot"]["auto_start_sa6"] is False


def test_phase1_changes_only_parent_mix_and_budget() -> None:
    metadata = {"name", "description", "tags", "notes"}
    drifted = {
        field.name
        for field in fields(phase1.CONFIG)
        if field.name not in metadata
        and getattr(phase1.CONFIG, field.name) != getattr(source.CONFIG, field.name)
    }
    assert drifted == {
        "checkpoint",
        "long_corridor_speed_density_mix",
        "timesteps",
    }


def test_phase1_budget_and_optimizer_resume_are_exact() -> None:
    assert phase1.TRAINING_ITERATIONS == 25
    assert phase1.CONFIG.timesteps == 25 * 128
    assert phase1.CONFIG.save_interval == 25
    assert phase1.CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
    assert phase1.CONFIG.no_resume_optimizer is False


def test_phase1_preserves_outer_scene_replay() -> None:
    assert phase1.CONFIG.long_corridor_fraction == 0.10
    assert phase1.CONFIG.narrow_passage_fraction == 0.12
    assert 1.0 - 0.10 - 0.12 == 0.78


def test_phase1_preserves_sensor_actuator_and_vehicle_contracts() -> None:
    for name in (
        "speed_rate",
        "speed_rate_obs",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "lidar_frame_stack",
        "actuator_delay_range",
        "actuator_velocity_scale",
        "actuator_motor_lag",
    ):
        assert getattr(phase1.CONFIG, name) == getattr(source.CONFIG, name)


def test_phase1_preserves_reward_model_and_ppo_contracts() -> None:
    protected_tokens = (
        "reward",
        "penalty",
        "weight",
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
    allowed = {"no_resume_optimizer"}
    for field in fields(phase1.CONFIG):
        if field.name in allowed:
            continue
        if any(token in field.name for token in protected_tokens):
            assert getattr(phase1.CONFIG, field.name) == getattr(
                source.CONFIG, field.name
            ), field.name


def test_phase1_mix_is_normalized_and_has_no_duplicate_profiles() -> None:
    mix = phase1.PHASE_1_SPEED_DENSITY_MIX
    assert abs(sum(weight for _, _, weight in mix) - 1.0) < 1e-12
    keys = [(counts, speed) for counts, speed, _ in mix]
    assert len(keys) == len(set(keys))


def test_phase1_retains_exact_low_density_speed_profiles() -> None:
    mix = phase1.PHASE_1_SPEED_DENSITY_MIX
    retained = mix[:3]
    assert retained == (
        ((0, 1), (0.70, 0.90), 0.10),
        ((1, 1), (0.70, 0.90), 0.10),
        ((2, 1), (0.50, 0.70), 0.15),
    )
    assert sum(row[2] for row in retained) == 0.35


def test_phase1_builds_the_single_pedestrian_static_ladder() -> None:
    p035_1d = {
        counts: weight
        for counts, speed, weight in phase1.PHASE_1_SPEED_DENSITY_MIX
        if speed == (0.25, 0.45) and counts[1] == 1
    }
    assert p035_1d == {(1, 1): 0.20, (2, 1): 0.15, (4, 1): 0.10}


def test_phase1_introduces_only_the_first_two_pedestrian_boundary() -> None:
    p035_2d = {
        counts: weight
        for counts, speed, weight in phase1.PHASE_1_SPEED_DENSITY_MIX
        if speed == (0.25, 0.45) and counts[1] == 2
    }
    assert p035_2d == {(1, 2): 0.20}


def test_phase2_is_preregistered_but_not_auto_started() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    profiles = auth["pilot"]["phase_2"]["profiles"]
    counts = {tuple(row[0]) for row in profiles}
    assert {(1, 2), (2, 2), (4, 2)} <= counts
    assert auth["pilot"]["auto_extend"] is False
    assert auth["pilot"]["auto_start_sa6"] is False


def test_acceptance_requires_sr_not_collision_to_timeout_conversion() -> None:
    acceptance = json.loads(AUTH.read_text(encoding="utf-8"))["acceptance"]
    assert acceptance["target_sr_must_increase"] is True
    assert acceptance["target_to_max"] == 0.05
    assert acceptance["native_degradation_max_pp"] == 2.0
    assert acceptance["native_graduation_cr_max"] == 0.10
