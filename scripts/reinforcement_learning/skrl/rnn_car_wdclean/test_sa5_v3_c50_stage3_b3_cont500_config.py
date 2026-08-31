"""Locks for the user-authorized B3 it100-to-it600 continuation."""

from __future__ import annotations

from dataclasses import fields
import json

import pytest

torch = pytest.importorskip("torch")

from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont50_from_it50 as source,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont500_from_it100 as cont,
)
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_cont500_from_it100_smoke as smoke,
)


def test_only_parent_budget_and_save_interval_change():
    changed = {
        field.name
        for field in fields(cont.CONFIG)
        if field.name not in cont.METADATA_FIELDS
        and getattr(cont.CONFIG, field.name) != getattr(source.CONFIG, field.name)
    }
    assert changed == {"checkpoint", "timesteps", "save_interval"}


def test_exact_parent_contains_resumable_rl_optimizer():
    payload = torch.load(
        cont.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert payload["total_steps"] == 6400
    assert payload["iteration"] == 49
    assert len(payload["charge_opt_rl"]["state"]) == 38
    assert cont.CONFIG.no_resume_optimizer is False


def test_budget_maps_to_conceptual_it150_through_it600():
    assert cont.ADDITIONAL_TRAINING_ITERATIONS == 500
    assert cont.CONFIG.timesteps == 500 * 128
    assert cont.CONFIG.save_interval == 50
    assert cont.CHECKPOINTS == (
        (150, "checkpoint_6400.pt"),
        (200, "checkpoint_12800.pt"),
        (250, "checkpoint_19200.pt"),
        (300, "checkpoint_25600.pt"),
        (350, "checkpoint_32000.pt"),
        (400, "checkpoint_38400.pt"),
        (450, "checkpoint_44800.pt"),
        (500, "checkpoint_51200.pt"),
        (550, "checkpoint_57600.pt"),
        (600, "checkpoint_64000.pt"),
    )


def test_authorization_is_explicit_and_keeps_sa6_on_hold():
    authorization = json.loads(
        cont.AUTHORIZATION_RECORD.read_text(encoding="utf-8")
    )
    assert authorization["decision"] == (
        "HUMAN_AUTHORIZED_B3_EXACT_CONTINUATION_IT100_TO_IT600"
    )
    assert authorization["request"] == "繼續訓練500iter"
    assert authorization["automatic_actions"]["sa6_auto_start"] is False
    assert authorization["evidence_context"][
        "formal_fixed_checkpoint_screen_complete"
    ] is False


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
