"""CPU-only locks for the SA4-R3 conceptual it100 to it125 continuation."""

from dataclasses import fields
import hashlib
from pathlib import Path

import torch

from rnn_car_modular.configs import e2e_sa4_r3_cont25_from_it100 as cont_mod
from rnn_car_modular.configs import (
    e2e_sa4_r3_cont50_from_r3_c6400 as prior_mod,
)


CONT = cont_mod.CONFIG
PRIOR = prior_mod.CONFIG


def test_parent_hash_iteration_and_optimizer_state_are_locked():
    parent = Path(cont_mod.PARENT_CHECKPOINT)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == (
        cont_mod.PARENT_CHECKPOINT_SHA256
    )
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    assert checkpoint["total_steps"] == 6_400
    assert checkpoint["iteration"] == 49
    assert checkpoint["charge_opt_rl"]["state"]


def test_continuation_changes_only_budget_lineage_and_metadata():
    drifted = sorted(
        field.name
        for field in fields(CONT)
        if field.name not in cont_mod.METADATA_FIELDS
        and getattr(CONT, field.name) != getattr(PRIOR, field.name)
    )
    assert drifted == ["checkpoint", "timesteps"]


def test_budget_and_conceptual_iteration_mapping_are_fixed():
    assert cont_mod.PARENT_CONCEPTUAL_ITERATION == 100
    assert cont_mod.CONTINUATION_ITERATIONS == 25
    assert CONT.timesteps == 3_200
    assert CONT.save_interval == 25
    assert cont_mod.CONCEPTUAL_CHECKPOINTS == ((125, "checkpoint_3200.pt"),)


def test_same_83d_k8_recipe_and_optimizer_resume_are_preserved():
    assert CONT.initial_stage == PRIOR.initial_stage == 4
    assert CONT.fixed_stage is PRIOR.fixed_stage is True
    assert CONT.no_resume_optimizer is PRIOR.no_resume_optimizer is False
    assert CONT.lidar_frame_stack == PRIOR.lidar_frame_stack == 8
    assert CONT.use_action_history is PRIOR.use_action_history is True
    assert CONT.vlp16_noise_mode == PRIOR.vlp16_noise_mode == "full"
    assert CONT.lidar_distractor_eligibility == "valid_return_only"
    assert CONT.actuator_delay_range == PRIOR.actuator_delay_range == (0, 2)


def test_teacher_r10_is_not_mixed_into_this_rl_continuation():
    assert CONT.corridor_teacher_distill_epochs == 0
    assert CONT.corridor_teacher_goal_denominator_floor_m == 1.0


def test_all_non_lineage_recipe_fields_match_prior_continuation():
    for field in fields(CONT):
        if field.name in cont_mod.ALLOWED_DIFFS:
            continue
        assert getattr(CONT, field.name) == getattr(PRIOR, field.name), field.name


def test_parent_is_training_checkpoint_and_sa5_cannot_launch():
    assert CONT.checkpoint.endswith(".pt")
    assert not CONT.checkpoint.endswith(".ts")
    text = " ".join(
        (CONT.name, CONT.description, " ".join(CONT.tags), CONT.notes)
    ).lower()
    assert "do not launch sa5" in text

