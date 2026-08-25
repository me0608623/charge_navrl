"""CPU-only contract tests for the SA5 run from SA4-R3 it125."""

from dataclasses import fields
import ast
import hashlib
from pathlib import Path

import torch

from rnn_car_modular.configs import (
    e2e_sa4_r3_cont25_from_it100 as parent_mod,
)
from rnn_car_modular.configs import (
    e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver as sa5_mod,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
    make_sim2real_curriculum_config,
)


PARENT = parent_mod.CONFIG
SA5 = sa5_mod.CONFIG
SPEC4 = STAGE_SCENE_CURRICULUM[4]
SPEC5 = STAGE_SCENE_CURRICULUM[5]


def test_parent_path_hash_and_training_identity_are_locked():
    parent = Path(sa5_mod.PARENT_CHECKPOINT)
    assert parent.name == "checkpoint_3200.pt"
    assert "sa4_r3_cont25_from_it100" in str(parent)
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == (
        sa5_mod.PARENT_CHECKPOINT_SHA256
    )
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 24
    assert checkpoint["total_steps"] == 3_200
    assert len(checkpoint["charge_opt_rl"]["state"]) == 38


def test_human_waiver_is_recorded_without_claiming_a_machine_pass():
    text = " ".join((SA5.description, " ".join(SA5.tags), SA5.notes)).lower()
    assert "human_waiver_parent" in SA5.tags
    assert "human waiver" in text
    assert "1.0465" in text
    assert "three-scene screen passed" not in text


def test_budget_is_300_iterations_with_six_checkpoints():
    assert sa5_mod.TRAINING_ITERATIONS == 300
    assert SA5.rollout_length == 128
    assert SA5.timesteps == 38_400
    assert SA5.save_interval == 50
    assert sa5_mod.CHECKPOINTS == (
        (50, "checkpoint_6400.pt"),
        (100, "checkpoint_12800.pt"),
        (150, "checkpoint_19200.pt"),
        (200, "checkpoint_25600.pt"),
        (250, "checkpoint_32000.pt"),
        (300, "checkpoint_38400.pt"),
    )


def test_stage_transition_loads_weights_but_resets_optimizer():
    assert PARENT.initial_stage == 4
    assert PARENT.no_resume_optimizer is False
    assert SA5.initial_stage == 5
    assert SA5.no_resume_optimizer is True
    assert SA5.checkpoint == sa5_mod.PARENT_CHECKPOINT


def test_sa5_scene_fields_match_the_canonical_builder():
    canonical = make_sim2real_curriculum_config(
        5, checkpoint=sa5_mod.PARENT_CHECKPOINT
    )
    for field_name in sa5_mod.STAGE_SCENE_FIELDS:
        assert getattr(SA5, field_name) == getattr(canonical, field_name), field_name


def test_sa5_scene_contract_values():
    assert SA5.room_size == SPEC5.room_half_extent == 7.5
    assert SPEC5.native_fraction == 0.78
    assert SA5.narrow_passage_fraction == 0.12
    assert SA5.narrow_passage_fixed_width_range == (1.30, 1.50)
    assert SA5.narrow_passage_fixed_yaw_limit_deg == 4.0
    assert SA5.long_corridor_fraction == 0.10
    assert SA5.long_corridor_free_width == 4.2
    assert SA5.long_corridor_obstacle_count_mix == (
        ((3, 2), 0.50),
        ((4, 2), 0.50),
    )
    assert SA5.long_corridor_dynamic_speed_range == (0.25, 0.45)
    assert SA5.long_corridor_dynamic_motion_mode == "env_stratified"
    assert SA5.long_corridor_dynamic_motion_weights is None
    assert SA5.long_corridor_random_2d_kinematics == "patrol"


def test_sa5_is_harder_than_sa4_in_the_intended_dimensions():
    assert SPEC5.room_half_extent < SPEC4.room_half_extent
    assert SPEC5.narrow_width_range[0] < SPEC4.narrow_width_range[0]
    assert SPEC5.corridor_free_width < SPEC4.corridor_free_width
    assert SPEC5.corridor_speed_range[1] > SPEC4.corridor_speed_range[1]
    assert SPEC4.corridor_density_mix is None
    assert SPEC5.corridor_density_mix is not None
    assert SPEC4.corridor_motion_weights == (0.5, 0.5, 0.0)
    assert SPEC5.corridor_motion_weights is None


def test_sensor_actuator_reward_model_and_ppo_are_preserved():
    for field in fields(SA5):
        if field.name in sa5_mod.ALLOWED_DIFFS:
            continue
        assert getattr(SA5, field.name) == getattr(PARENT, field.name), field.name
    assert SA5.lidar_frame_stack == 8
    assert SA5.use_action_history is True
    assert SA5.vlp16_noise_mode == "full"
    assert SA5.lidar_distractor_eligibility == "valid_return_only"
    assert SA5.actuator_delay_range == (0, 2)
    assert SA5.obs_delay_steps == (0, 0)
    assert SA5.future_occupancy_weight == 0.15


def test_actual_nonmetadata_changes_are_exactly_preregistered():
    changed = frozenset(
        field.name
        for field in fields(SA5)
        if field.name not in sa5_mod.METADATA_FIELDS
        and getattr(SA5, field.name) != getattr(PARENT, field.name)
    )
    assert changed == sa5_mod.EXPECTED_CHANGED_FIELDS


def test_teacher_and_rejected_r10_branch_stay_disabled():
    assert SA5.corridor_teacher_distill_epochs == 0
    assert SA5.corridor_teacher_goal_denominator_floor_m == 1.0


def test_run_shape_is_fixed():
    assert SA5.fixed_stage is True
    assert SA5.num_envs == 1024
    assert SA5.seed == 42
    assert SA5.rollout_length == 128


def test_config_cannot_launch_sa6_or_other_processes():
    tree = ast.parse(Path(sa5_mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "os", "shutil", "socket"):
        assert forbidden not in imported
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("system", "run", "Popen", "spawn"):
        assert forbidden not in calls
    assert "do not auto-launch sa6" in SA5.notes.lower()

