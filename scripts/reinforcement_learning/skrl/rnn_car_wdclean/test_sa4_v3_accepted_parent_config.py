"""Fail-closed lineage tests for the accepted c300 SA4-v3 config."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SKRL = REPO / "scripts/reinforcement_learning/skrl"
if str(SKRL) not in sys.path:
    sys.path.insert(0, str(SKRL))

from rnn_car_modular.configs.e2e_sa4_k8_obb_speed_density_v3_from_sa3 import (  # noqa: E402
    CONFIG,
    CURRICULUM_FREEZE,
    CURRICULUM_FREEZE_SHA256,
    P060_SUMMARY,
    P060_SUMMARY_SHA256,
    PARENT_ACCEPTANCE_RECORD,
    PARENT_ACCEPTANCE_RECORD_SHA256,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    RETENTION_SUMMARY,
    RETENTION_SUMMARY_SHA256,
    SAVE_INTERVAL,
    TRAINING_ITERATIONS,
    TRAINING_STARTED,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (  # noqa: E402
    SA4_SPEED_DENSITY_MIX,
    make_speed_density_curriculum_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_acceptance_record_is_exact_and_not_a_launch_record() -> None:
    assert _sha256(PARENT_ACCEPTANCE_RECORD) == PARENT_ACCEPTANCE_RECORD_SHA256
    record = json.loads(PARENT_ACCEPTANCE_RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "sa3_v3_parent_acceptance/v1"
    assert record["decision"] == "ACCEPT_C300_AS_SA4_V3_PARENT"
    assert record["accepted_checkpoint"] == {
        "name": "c300",
        "conceptual_iteration": 300,
        "path": (
            "logs/rnn_car/sa3_speed_density_v3_from_sa2r1_c100_"
            "ne1024_s42_p300_r1/checkpoint_38400.pt"
        ),
        "sha256": PARENT_CHECKPOINT_SHA256,
        "training_seed": 42,
    }
    assert record["sa4_config"]["config_ready"] is True
    assert record["sa4_config"]["training_started"] is False


def test_parent_file_and_evidence_files_are_sha_locked() -> None:
    assert _sha256(Path(PARENT_CHECKPOINT)) == PARENT_CHECKPOINT_SHA256
    assert _sha256(CURRICULUM_FREEZE) == CURRICULUM_FREEZE_SHA256
    assert _sha256(P060_SUMMARY) == P060_SUMMARY_SHA256
    assert _sha256(RETENTION_SUMMARY) == RETENTION_SUMMARY_SHA256


def test_upstream_summaries_authorize_c300() -> None:
    phase_a = json.loads(P060_SUMMARY.read_text(encoding="utf-8"))
    assert phase_a["status"] == "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN"
    assert phase_a["promoted_for_retention"] == ["c300", "c200"]
    assert phase_a["ranked_candidates"][0]["checkpoint_name"] == "c300"
    assert phase_a["ranked_candidates"][0]["hard_gate_pass"] is True

    retention = json.loads(RETENTION_SUMMARY.read_text(encoding="utf-8"))
    assert retention["status"] == (
        "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN"
    )
    assert retention["source_fingerprint_stable"] is True
    assert retention["recommended_parent_candidate"] == "c300"
    c300 = next(
        row
        for row in retention["candidate_results"]
        if row["checkpoint_name"] == "c300"
    )
    assert c300["retention_pass"] is True
    assert set(c300["cells"]) == {
        "nav_native",
        "narrow_range",
        "p035_0s1d_lateral",
        "p035_1s1d_lateral",
    }
    assert all(cell["gate_pass"] for cell in c300["cells"].values())


def test_sa4_config_keeps_the_frozen_stage_four_contract() -> None:
    assert CONFIG.checkpoint == PARENT_CHECKPOINT
    assert CONFIG.initial_stage == 4
    assert CONFIG.fixed_stage is True
    assert CONFIG.no_resume_optimizer is True
    assert TRAINING_ITERATIONS == 300
    assert CONFIG.timesteps == 300 * 128
    assert SAVE_INTERVAL == CONFIG.save_interval == 50
    assert CONFIG.speed_rate == 0.7
    assert CONFIG.speed_rate_obs == "ego"
    assert CONFIG.long_corridor_speed_density_mix == SA4_SPEED_DENSITY_MIX
    assert CONFIG.long_corridor_obstacle_count_mix is None
    assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
    assert CONFIG.actuator_delay_range == (1, 2)
    assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
    assert CONFIG.actuator_motor_lag == 1.0


def test_config_diff_from_stage_four_factory_is_allowlisted() -> None:
    base = make_speed_density_curriculum_config(
        4,
        checkpoint=PARENT_CHECKPOINT,
    )
    allowed = {
        "name",
        "description",
        "no_resume_optimizer",
        "timesteps",
        "save_interval",
        "tags",
        "notes",
    }
    drifted = sorted(
        field.name
        for field in fields(CONFIG)
        if field.name not in allowed
        and getattr(CONFIG, field.name) != getattr(base, field.name)
    )
    assert drifted == []


def test_config_import_has_no_training_side_effect() -> None:
    assert TRAINING_STARTED is False
    source = (
        REPO
        / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
        "e2e_sa4_k8_obb_speed_density_v3_from_sa3.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("systemctl", "systemd-run", "subprocess", "Popen("):
        assert forbidden not in source
