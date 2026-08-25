from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_frontier_curriculum_phase1_p25 as phase1,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_frontier_curriculum_phase2_p25 as phase2,
)


REPO = Path(__file__).resolve().parents[4]
RESULT = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_phase1_complete_v1.json"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_phase1_completion_evidence_is_exact() -> None:
    assert _sha256(RESULT) == phase2.PHASE_1_RESULT_SHA256
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE_VALID_PHASE1_DIAGNOSTIC_CURRICULUM"
    assert payload["integrity"]["strict_errors"] == 0
    assert payload["lineage"]["sa6"] == "HOLD_NOT_AUTHORIZED"


def test_phase2_parent_hash_metadata_and_optimizer_are_exact() -> None:
    path = Path(phase2.PARENT_CHECKPOINT)
    assert _sha256(path) == phase2.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 24
    assert payload["total_steps"] == 3200
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_phase2_changes_only_parent_and_curriculum_mix() -> None:
    metadata = {"name", "description", "tags", "notes"}
    drifted = {
        field.name
        for field in fields(phase2.CONFIG)
        if field.name not in metadata
        and getattr(phase2.CONFIG, field.name) != getattr(phase1.CONFIG, field.name)
    }
    assert drifted == {"checkpoint", "long_corridor_speed_density_mix"}


def test_phase2_budget_and_optimizer_resume_are_exact() -> None:
    assert phase2.CONFIG.no_resume_optimizer is False
    assert phase2.CONFIG.timesteps == 25 * 128
    assert phase2.CONFIG.save_interval == 25
    assert phase2.CHECKPOINTS == ((50, "checkpoint_3200.pt"),)


def test_phase2_mix_is_normalized_and_has_no_duplicate_profiles() -> None:
    mix = phase2.PHASE_2_SPEED_DENSITY_MIX
    assert abs(sum(weight for _, _, weight in mix) - 1.0) < 1e-12
    keys = [(counts, speed) for counts, speed, _ in mix]
    assert len(keys) == len(set(keys))


def test_phase2_retains_low_density_speed_replay() -> None:
    mix = phase2.PHASE_2_SPEED_DENSITY_MIX
    retained = mix[:3]
    assert retained == (
        ((0, 1), (0.70, 0.90), 0.05),
        ((1, 1), (0.70, 0.90), 0.05),
        ((2, 1), (0.50, 0.70), 0.10),
    )
    assert sum(row[2] for row in retained) == 0.20


def test_phase2_retains_single_pedestrian_ladder() -> None:
    p035_1d = {
        counts: weight
        for counts, speed, weight in phase2.PHASE_2_SPEED_DENSITY_MIX
        if speed == (0.25, 0.45) and counts[1] == 1
    }
    assert p035_1d == {(1, 1): 0.05, (2, 1): 0.05, (4, 1): 0.10}


def test_phase2_adds_ordered_two_pedestrian_ladder() -> None:
    p035_2d = {
        counts: weight
        for counts, speed, weight in phase2.PHASE_2_SPEED_DENSITY_MIX
        if speed == (0.25, 0.45) and counts[1] == 2
    }
    assert p035_2d == {(1, 2): 0.15, (2, 2): 0.20, (4, 2): 0.25}


def test_phase2_preserves_outer_scene_sensor_and_actuator_contracts() -> None:
    protected = (
        "long_corridor_fraction",
        "narrow_passage_fraction",
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
        assert getattr(phase2.CONFIG, name) == getattr(phase1.CONFIG, name)
    assert phase2.CONFIG.long_corridor_fraction == 0.10
    assert phase2.CONFIG.narrow_passage_fraction == 0.12


def test_phase2_preserves_all_non_curriculum_behavioral_fields() -> None:
    excluded = {
        "name",
        "description",
        "tags",
        "notes",
        "checkpoint",
        "long_corridor_speed_density_mix",
    }
    for field in fields(phase2.CONFIG):
        if field.name not in excluded:
            assert getattr(phase2.CONFIG, field.name) == getattr(
                phase1.CONFIG, field.name
            ), field.name


def test_phase2_does_not_authorize_extension_or_sa6() -> None:
    auth = json.loads(phase2.AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
    assert auth["pilot"]["auto_extend"] is False
    assert auth["pilot"]["auto_start_sa6"] is False
    assert auth["lineage_status"]["sa6"] == "HOLD_NOT_AUTHORIZED"

