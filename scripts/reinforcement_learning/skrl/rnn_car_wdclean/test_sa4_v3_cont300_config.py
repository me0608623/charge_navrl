"""CPU-only locks for the exact SA4-v3 c300-to-c600 continuation."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_modular.configs import e2e_sa4_v3_cont300_from_c300 as cont_mod
from rnn_car_modular.configs import (
    e2e_sa4_k8_obb_speed_density_v3_from_sa3 as source_mod,
)


CONT = cont_mod.CONFIG
SOURCE = source_mod.CONFIG


def test_parent_hash_and_nonempty_rl_optimizer_state_are_locked():
    parent = Path(cont_mod.PARENT_CHECKPOINT)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == (
        cont_mod.PARENT_CHECKPOINT_SHA256
    )
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 299
    assert checkpoint["total_steps"] == 38_400
    assert checkpoint["charge_opt_rl"]["state"]
    assert checkpoint["charge_opt_rl"]["param_groups"]


def test_human_authorization_and_selection_evidence_are_exact():
    authorization = json.loads(
        cont_mod.AUTHORIZATION_RECORD.read_text(encoding="utf-8")
    )
    assert authorization["decision"] == (
        "HUMAN_AUTHORIZED_C300_TO_C600_EXACT_CONTINUATION"
    )
    selection = json.loads(cont_mod.SELECTION_SUMMARY.read_text(encoding="utf-8"))
    assert selection["ranked_candidates"][0]["checkpoint_name"] == "c300"
    assert selection["ranked_candidates"][0]["hard_gate_pass"] is False


def test_continuation_changes_only_lineage_and_metadata():
    drifted = sorted(
        field.name
        for field in fields(CONT)
        if field.name not in cont_mod.METADATA_FIELDS
        and getattr(CONT, field.name) != getattr(SOURCE, field.name)
    )
    assert drifted == ["checkpoint", "no_resume_optimizer"]


def test_budget_and_conceptual_checkpoint_mapping_are_fixed():
    assert cont_mod.PARENT_TOTAL_ITERATION == 300
    assert cont_mod.CONTINUATION_ITERATIONS == 300
    assert CONT.timesteps == 38_400
    assert CONT.save_interval == 50
    assert cont_mod.CONCEPTUAL_CHECKPOINTS == (
        (350, "checkpoint_6400.pt"),
        (400, "checkpoint_12800.pt"),
        (450, "checkpoint_19200.pt"),
        (500, "checkpoint_25600.pt"),
        (550, "checkpoint_32000.pt"),
        (600, "checkpoint_38400.pt"),
    )


def test_optimizer_resumes_without_stage_or_recipe_change():
    assert CONT.initial_stage == SOURCE.initial_stage == 4
    assert CONT.fixed_stage is SOURCE.fixed_stage is True
    assert SOURCE.no_resume_optimizer is True
    assert CONT.no_resume_optimizer is False
    for field in fields(CONT):
        if field.name in cont_mod.ALLOWED_DIFFS:
            continue
        assert getattr(CONT, field.name) == getattr(SOURCE, field.name), field.name


def test_sensor_actuator_speed_scene_network_and_ppo_contract_stays_fixed():
    assert CONT.speed_rate == SOURCE.speed_rate == 0.7
    assert CONT.speed_rate_obs == SOURCE.speed_rate_obs == "ego"
    assert CONT.actuator_delay_range == SOURCE.actuator_delay_range == (1, 2)
    assert CONT.vlp16_noise_mode == SOURCE.vlp16_noise_mode == "full"
    assert CONT.lidar_distractor_eligibility == "valid_return_only"
    assert SOURCE.lidar_distractor_eligibility == "valid_return_only"
    assert CONT.long_corridor_speed_density_mix == (
        SOURCE.long_corridor_speed_density_mix
    )
    assert CONT.reward_profile == SOURCE.reward_profile
    assert CONT.lr == SOURCE.lr
    assert CONT.rnn_lr == SOURCE.rnn_lr


def test_config_forbids_automatic_parent_acceptance_or_sa5_launch():
    text = " ".join(
        (CONT.name, CONT.description, " ".join(CONT.tags), CONT.notes)
    ).lower()
    assert "do not auto-accept" in text
    assert "do not auto-launch sa5" in text
