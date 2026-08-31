"""Contract tests for the READY_NOT_RUN c250 profile-curriculum pilot."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from rnn_car_modular.configs import (
    e2e_sa6_v3_cont300_from_it50 as source,
)
from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_control_p50 as control,
)
from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_rung1_p50 as intervention,
)


def _behavioral_diffs(left, right):
    return {
        field.name
        for field in fields(left)
        if field.name not in control.METADATA_FIELDS
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_control_only_changes_parent_and_budget_from_source():
    assert _behavioral_diffs(control.CONFIG, source.CONFIG) == {
        "checkpoint",
        "timesteps",
        "save_interval",
    }


def test_intervention_only_changes_profile_mix_from_fresh_control():
    assert _behavioral_diffs(intervention.CONFIG, control.CONFIG) == {
        "long_corridor_speed_density_mix"
    }


def test_both_arms_use_exact_c250_checkpoint_and_optimizer():
    with control.PARENT_CHECKPOINT.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == (
            control.PARENT_CHECKPOINT_SHA256
        )
    payload = torch.load(
        control.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 25_600
    assert payload["iteration"] == 199
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert control.CONFIG.no_resume_optimizer is False
    assert intervention.CONFIG.no_resume_optimizer is False


def test_shared_training_and_sim2real_contract_is_exactly_matched():
    for name in (
        "checkpoint",
        "timesteps",
        "save_interval",
        "num_envs",
        "seed",
        "rollout_length",
        "initial_stage",
        "speed_rate",
        "speed_rate_obs",
        "lidar_frame_stack",
        "lidar_no_noise",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "enable_actuator_dr",
        "actuator_delay_range",
        "actuator_velocity_scale",
        "actuator_motor_lag",
        "reward_profile",
        "reward_mode",
        "encoder_profile",
        "critic_profile",
        "long_corridor_dynamic_motion_mode",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_random_2d_kinematics",
    ):
        assert getattr(control.CONFIG, name) == getattr(intervention.CONFIG, name)
    assert control.CONFIG.timesteps == intervention.CONFIG.timesteps == 6_400
    assert control.CONFIG.save_interval == intervention.CONFIG.save_interval == 25


def test_rung1_preserves_low_density_and_only_reweights_existing_profiles():
    before = control.CONTROL_MIX
    after = intervention.CURRICULUM_RUNG1_MIX
    assert before[:3] == after[:3]
    assert tuple(row[2] for row in before) == (0.10, 0.10, 0.15, 0.20, 0.30, 0.15)
    assert tuple(row[2] for row in after) == (0.10, 0.10, 0.15, 0.35, 0.25, 0.05)
    assert sum(row[2] for row in before[:3]) == pytest.approx(0.35)
    assert sum(row[2] for row in after[:3]) == pytest.approx(0.35)
    assert sum(row[2] for row in before[3:]) == pytest.approx(0.65)
    assert sum(row[2] for row in after[3:]) == pytest.approx(0.65)
    assert [row[:2] for row in before] == [row[:2] for row in after]


def test_outer_scene_mix_and_teacher_contract_are_unchanged():
    for name in (
        "narrow_passage_fraction",
        "long_corridor_fraction",
        "teacher_retention_checkpoint",
        "teacher_retention_rollout_override",
        "corridor_teacher_distill_epochs",
    ):
        assert getattr(control.CONFIG, name) == getattr(intervention.CONFIG, name)
    assert intervention.CONFIG.teacher_retention_checkpoint is None
    assert intervention.CONFIG.teacher_retention_rollout_override is False
    assert intervention.CONFIG.corridor_teacher_distill_epochs == 0


def test_design_is_ready_not_run_and_forbids_automatic_progression():
    design = json.loads(control.DESIGN_RECORD.read_text(encoding="utf-8"))
    assert design["decision"] == "ESTABLISH_READY_NOT_RUN"
    assert design["parent"]["status"] == "SA6_NOT_GRADUATED_DIAGNOSTIC_ANCHOR"
    assert design["evaluation_contract"]["fresh_control_required"] is True
    assert design["evaluation_contract"]["primary_offset_iterations"] == 50
    assert design["automatic_actions"] == {
        "config_build_authorized": True,
        "cpu_tests_authorized": True,
        "gpu_smoke_authorized": False,
        "training_launch_authorized": False,
        "auto_extend_authorized": False,
        "start_sa7_authorized": False,
        "enable_teacher_distillation_or_override": False,
    }
    assert control.TRAINING_STARTED is False
    assert intervention.TRAINING_STARTED is False


def test_design_does_not_repeat_the_rejected_random2d_weight_only_pilot():
    assert control.CONFIG.long_corridor_dynamic_motion_weights == (0.3, 0.3, 0.4)
    assert intervention.CONFIG.long_corridor_dynamic_motion_weights == (0.3, 0.3, 0.4)
    assert intervention.CONFIG.long_corridor_dynamic_motion_weights != (0.15, 0.15, 0.70)

