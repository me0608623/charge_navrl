"""Locks for the READY_NOT_RUN c550 profile-ratio matched pilot."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_c550_p080replay10_p50 as intervention,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_c550_profilemix_control_p50 as control,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont500_from_it100 as source,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P060,
    P080,
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


def test_both_arms_use_exact_c550_optimizer_parent():
    with control.PARENT_CHECKPOINT.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == (
            control.PARENT_CHECKPOINT_SHA256
        )
    payload = torch.load(
        control.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 57600
    assert payload["iteration"] == 449
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert control.CONFIG.no_resume_optimizer is False
    assert intervention.CONFIG.no_resume_optimizer is False


def test_shared_training_contract_is_exactly_matched():
    for name in (
        "checkpoint",
        "timesteps",
        "save_interval",
        "num_envs",
        "seed",
        "rollout_length",
        "speed_rate",
        "speed_rate_obs",
        "actuator_delay_range",
        "lidar_distractor_eligibility",
        "reward_profile",
        "encoder_profile",
        "critic_profile",
    ):
        assert getattr(control.CONFIG, name) == getattr(intervention.CONFIG, name)
    assert control.CONFIG.timesteps == intervention.CONFIG.timesteps == 6400
    assert control.CONFIG.save_interval == intervention.CONFIG.save_interval == 25


def test_intervention_preserves_every_p060_profile_and_adds_p080_replay():
    control_p060 = tuple(
        row
        for row in control.CONFIG.long_corridor_speed_density_mix
        if row[1] == P060
    )
    intervention_p060 = tuple(
        row
        for row in intervention.CONFIG.long_corridor_speed_density_mix
        if row[1] == P060
    )
    intervention_p080 = tuple(
        row
        for row in intervention.CONFIG.long_corridor_speed_density_mix
        if row[1] == P080
    )
    assert intervention_p060 == control_p060
    assert intervention_p080 == (
        ((0, 1), P080, 0.05),
        ((1, 1), P080, 0.05),
    )
    assert sum(row[2] for row in intervention.P080_REPLAY_MIX) == pytest.approx(1.0)


def test_design_record_keeps_both_arms_ready_not_run_and_sa6_on_hold():
    design = json.loads(control.DESIGN_RECORD.read_text(encoding="utf-8"))
    assert design["decision"] == "ESTABLISH_READY_NOT_RUN"
    assert design["automatic_actions"] == {
        "gpu_launch_authorized": False,
        "sa6_start_authorized": False,
        "auto_advance_authorized": False,
    }
    assert design["interpretation"]["fresh_control_required"] is True
    assert design["interpretation"]["historical_c600_is_formal_control"] is False
    evaluation = design["evaluation_contract"]
    assert evaluation["primary_comparison_offset"] == 50
    assert evaluation["offset_25_role"] == "diagnostic_trajectory_only"
    assert evaluation["phase_b_cr_degradation_limit"] == 0.02
