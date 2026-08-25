"""CPU-only locks for the bounded SA4-R3 c6400 continuation."""

from dataclasses import fields
import hashlib
from pathlib import Path

import torch

from rnn_car_modular.configs import e2e_sa4_r3_cont50_from_r3_c6400 as cont_mod
from rnn_car_modular.configs import (
    e2e_sa4_r3_valid_return_noise_from_sa3_r1_c100 as r3_mod,
)


CONT = cont_mod.CONFIG
R3 = r3_mod.CONFIG


def test_parent_hash_and_optimizer_state_are_available():
    parent = Path(cont_mod.PARENT_CHECKPOINT)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == cont_mod.PARENT_CHECKPOINT_SHA256
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    assert "charge_opt_rl" in checkpoint
    assert checkpoint["charge_opt_rl"]["state"]


def test_continuation_changes_only_lineage_and_metadata():
    drifted = sorted(
        field.name
        for field in fields(CONT)
        if field.name not in cont_mod.METADATA_FIELDS
        and getattr(CONT, field.name) != getattr(R3, field.name)
    )
    assert drifted == ["checkpoint", "no_resume_optimizer"]


def test_budget_and_conceptual_checkpoint_mapping_are_fixed():
    assert cont_mod.PARENT_TOTAL_ITERATION == 50
    assert cont_mod.CONTINUATION_ITERATIONS == 50
    assert CONT.timesteps == 6_400
    assert CONT.save_interval == 25
    assert cont_mod.CONCEPTUAL_CHECKPOINTS == (
        (75, "checkpoint_3200.pt"),
        (100, "checkpoint_6400.pt"),
    )


def test_optimizer_is_resumed_without_a_stage_change():
    assert CONT.initial_stage == R3.initial_stage == 4
    assert CONT.fixed_stage is R3.fixed_stage is True
    assert R3.no_resume_optimizer is True
    assert CONT.no_resume_optimizer is False


def test_noise_delay_reward_scene_network_and_ppo_are_identical():
    for field in fields(CONT):
        if field.name in cont_mod.ALLOWED_DIFFS:
            continue
        assert getattr(CONT, field.name) == getattr(R3, field.name), field.name
    assert CONT.vlp16_noise_mode == "full"
    assert CONT.lidar_distractor_eligibility == "valid_return_only"
    assert CONT.actuator_delay_range == (0, 2)
    assert CONT.lidar_frame_stack == 8
    assert CONT.future_occupancy_weight == 0.15


def test_continuation_cannot_name_or_launch_sa5():
    text = " ".join((CONT.name, CONT.description, " ".join(CONT.tags), CONT.notes)).lower()
    assert "do not auto-launch sa5" in text
    assert "sa5_started" not in text
