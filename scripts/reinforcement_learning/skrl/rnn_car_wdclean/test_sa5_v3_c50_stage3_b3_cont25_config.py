"""Locks for the human-authorized B3 it25-to-it50 continuation."""

from __future__ import annotations

from dataclasses import fields
import json

import pytest

torch = pytest.importorskip("torch")

from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont25_from_it25 as cont,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont25_smoke as smoke,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_p060aligned_p25 as source,
)


def test_only_parent_and_save_interval_change_as_budget_is_same_size():
    changed = {
        field.name
        for field in fields(cont.CONFIG)
        if field.name not in cont.METADATA_FIELDS
        and getattr(cont.CONFIG, field.name) != getattr(source.CONFIG, field.name)
    }
    assert changed == {"checkpoint", "save_interval"}
    assert cont.CONFIG.timesteps == source.CONFIG.timesteps == 3200
    assert cont.AUTHORIZED_BEHAVIORAL_DIFFS == {
        "checkpoint",
        "timesteps",
        "save_interval",
    }


def test_exact_parent_contains_resumable_rl_optimizer():
    payload = torch.load(
        cont.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 3200
    assert payload["iteration"] == 24
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert cont.CONFIG.no_resume_optimizer is False


def test_budget_maps_to_conceptual_it30_through_it50():
    assert cont.ADDITIONAL_TRAINING_ITERATIONS == 25
    assert cont.CONFIG.timesteps == 25 * 128
    assert cont.CONFIG.save_interval == 5
    assert cont.CHECKPOINTS == (
        (30, "checkpoint_640.pt"),
        (35, "checkpoint_1280.pt"),
        (40, "checkpoint_1920.pt"),
        (45, "checkpoint_2560.pt"),
        (50, "checkpoint_3200.pt"),
    )


def test_authorization_holds_phase_b_and_sa6():
    authorization = json.loads(
        cont.AUTHORIZATION_RECORD.read_text(encoding="utf-8")
    )
    assert authorization["decision"] == (
        "HUMAN_AUTHORIZED_B3_EXACT_CONTINUATION_IT25_TO_IT50"
    )
    assert authorization["lineage"]["phase_b_auto_start"] is False
    assert authorization["lineage"]["sa6_auto_start"] is False
    assert authorization["monitoring_contract"]["fail_closed"] is True


def test_full_behavior_contract_is_retained():
    for name in (
        "reward_profile",
        "encoder_profile",
        "critic_profile",
        "use_action_history",
        "speed_rate",
        "speed_rate_obs",
        "lidar_distractor_eligibility",
        "actuator_delay_range",
        "long_corridor_fraction",
        "long_corridor_speed_density_mix",
        "long_corridor_dynamic_motion_weights",
        "narrow_passage_fraction",
        "rollout_length",
        "num_envs",
        "seed",
    ):
        assert getattr(cont.CONFIG, name) == getattr(source.CONFIG, name)


def test_smoke_changes_only_runtime_scale():
    changed = {
        field.name
        for field in fields(smoke.CONFIG)
        if field.name not in smoke.METADATA_FIELDS
        and getattr(smoke.CONFIG, field.name) != getattr(cont.CONFIG, field.name)
    }
    assert changed == {"num_envs", "timesteps", "save_interval"}
