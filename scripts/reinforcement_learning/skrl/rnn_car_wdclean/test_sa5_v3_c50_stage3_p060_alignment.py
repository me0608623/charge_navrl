"""Contract tests for the paired stage-3 P060-alignment experiment."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path

import torch

from rnn_car_modular.configs import e2e_sa5_v3_c50_stage3_a3_control_p25 as a3
from rnn_car_modular.configs import e2e_sa5_v3_c50_stage3_b3_p060aligned_p25 as b3
from rnn_car_modular.configs import (
    e2e_sa5_v3_c50_stage3_b3_p060aligned_smoke as smoke,
)


METADATA = {"name", "description", "tags", "notes"}


def test_parent_hash_and_optimizer_are_exact() -> None:
    path = Path(a3.PARENT_CHECKPOINT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == a3.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 24
    assert payload["total_steps"] == 3_200
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_a3_changes_only_checkpoint_from_b2() -> None:
    changed = {
        field.name
        for field in fields(a3.CONFIG)
        if field.name not in METADATA
        and getattr(a3.CONFIG, field.name) != getattr(a3._B2, field.name)
    }
    assert changed == {"checkpoint"}


def test_b3_changes_only_speed_density_mix_from_a3() -> None:
    changed = {
        field.name
        for field in fields(b3.CONFIG)
        if field.name not in METADATA
        and getattr(b3.CONFIG, field.name) != getattr(b3._A3, field.name)
    }
    assert changed == {"long_corridor_speed_density_mix"}


def test_b3_only_replaces_first_two_speed_ranges() -> None:
    old = b3._A3.long_corridor_speed_density_mix
    new = b3.P060_ALIGNED_MIX
    assert [row[0] for row in new] == [row[0] for row in old]
    assert [row[2] for row in new] == [row[2] for row in old]
    assert new[0][1] == b3.P060 and new[1][1] == b3.P060
    assert new[2:] == old[2:]


def test_runtime_contract_is_identical_between_arms() -> None:
    for key in (
        "checkpoint",
        "no_resume_optimizer",
        "num_envs",
        "seed",
        "rollout_length",
        "timesteps",
        "save_interval",
        "speed_rate",
        "speed_rate_obs",
        "vlp16_noise_mode",
        "lidar_distractor_eligibility",
        "actuator_delay_range",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_fraction",
        "narrow_passage_fraction",
    ):
        assert getattr(a3.CONFIG, key) == getattr(b3.CONFIG, key)


def test_no_arm_can_auto_advance() -> None:
    assert "sa6_hold" in a3.CONFIG.tags
    assert a3.CONFIG.no_resume_optimizer is False
    assert b3.CONFIG.no_resume_optimizer is False
    assert a3.CHECKPOINTS == ((25, "checkpoint_3200.pt"),)


def test_smoke_changes_only_budget_and_env_count() -> None:
    changed = {
        field.name
        for field in fields(smoke.CONFIG)
        if field.name not in METADATA
        and getattr(smoke.CONFIG, field.name) != getattr(b3.CONFIG, field.name)
    }
    assert changed == {"num_envs", "timesteps", "save_interval"}
    assert smoke.CONFIG.long_corridor_speed_density_mix == b3.P060_ALIGNED_MIX
