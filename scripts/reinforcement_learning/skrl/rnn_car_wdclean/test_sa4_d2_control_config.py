"""SA4-D2 Control config 的鎖定測試（CPU only）。

Control 的唯一價值是「除了預算以外什麼都沒變」。這裡逐欄位驗證那句話，
而不是相信 docstring。
"""

import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

from rnn_car_modular.configs import (
    e2e_sa4_d2_control_from_sa3_r1_c100 as control_mod,
)
from rnn_car_modular.configs.e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100 import (
    CONFIG as PILOT,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


CONTROL = control_mod.CONFIG


# --------------------------------------------------------------------------
# 只有預算能不同 —— 這是 control 存在的全部理由
# --------------------------------------------------------------------------
def test_only_budget_fields_differ_from_the_pilot():
    drifted = sorted(
        f.name for f in fields(CONTROL)
        if f.name not in control_mod.BUDGET_FIELDS
        and getattr(CONTROL, f.name) != getattr(PILOT, f.name)
    )
    assert drifted == [], f"非預算欄位也變了：{drifted}"


def test_budget_field_set_is_explicit_and_minimal():
    assert control_mod.BUDGET_FIELDS == frozenset(
        {"name", "description", "timesteps", "save_interval", "tags", "notes"}
    )


def test_reward_scene_and_optimizer_are_the_pilot_values():
    for attr in (
        "long_corridor_fraction", "long_corridor_free_width",
        "long_corridor_static_obstacles", "long_corridor_dynamic_obstacles",
        "long_corridor_dynamic_speed_range", "long_corridor_dynamic_motion_weights",
        "narrow_passage_fraction", "narrow_passage_fixed_width_range",
        "room_size", "num_envs", "seed", "rollout_length",
    ):
        assert getattr(CONTROL, attr) == getattr(PILOT, attr), attr


def test_config_is_derived_not_restated():
    """用 replace 從 pilot 衍生，避免手抄常數漂移。"""
    source = Path(control_mod.__file__).read_text()
    assert "replace(" in source
    assert "_SA4_PILOT" in source


# --------------------------------------------------------------------------
# 預算
# --------------------------------------------------------------------------
def test_budget_is_fifty_iterations():
    assert control_mod.CONTROL_ITERATIONS == 50
    assert CONTROL.timesteps == 50 * 128 == 6_400


def test_checkpoints_land_on_it25_and_it50():
    assert CONTROL.save_interval == 25
    assert control_mod.CHECKPOINT_STEPS == (3200, 6400)


def test_checkpoint_steps_match_the_save_schedule():
    """存檔點必須真的是 save_interval 的倍數，否則檔名不會是這兩個。"""
    for step in control_mod.CHECKPOINT_STEPS:
        iteration = step // CONTROL.rollout_length
        assert iteration % CONTROL.save_interval == 0
    assert control_mod.CHECKPOINT_STEPS[-1] == CONTROL.timesteps


# --------------------------------------------------------------------------
# 共同起點
# --------------------------------------------------------------------------
def test_parent_is_sa3_c100_not_the_old_sa4():
    assert CONTROL.checkpoint.endswith(
        "sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1/checkpoint_12800.pt"
    )
    assert "sa4_sim2real_v2" not in CONTROL.checkpoint


def test_parent_hash_verified_at_import():
    source = Path(control_mod.__file__).read_text()
    assert "hashlib.file_digest" in source
    assert "raise RuntimeError" in source


def test_parent_file_on_disk_matches():
    digest = hashlib.sha256(Path(CONTROL.checkpoint).read_bytes()).hexdigest()
    assert digest == control_mod.PARENT_CHECKPOINT_SHA256


def test_optimizer_state_is_not_resumed():
    assert CONTROL.no_resume_optimizer is True


# --------------------------------------------------------------------------
# 場景與 sim2real 不變
# --------------------------------------------------------------------------
def test_stage_four_scene_still_from_the_spec():
    spec = STAGE_SCENE_CURRICULUM[4]
    assert CONTROL.initial_stage == 4 and CONTROL.fixed_stage is True
    assert CONTROL.room_size == spec.room_half_extent
    assert CONTROL.long_corridor_free_width == spec.corridor_free_width
    assert CONTROL.long_corridor_dynamic_obstacles == spec.corridor_dynamic_obstacles


def test_corridor_families_stay_fifty_fifty():
    """A/B 這一輪不得改 corridor 權重。"""
    assert CONTROL.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)


def test_sim2real_mechanisms_retained():
    assert CONTROL.lidar_frame_stack == 8
    assert CONTROL.actuator_delay_range == (0, 2)
    assert CONTROL.lidar_no_noise is False
    assert CONTROL.lidar_hole_rate > 0.0


def test_observation_delay_stays_off():
    assert tuple(getattr(CONTROL, "obs_delay_steps", (0, 0))) == (0, 0)


# --------------------------------------------------------------------------
# 這一輪不得帶任何 intervention
# --------------------------------------------------------------------------
def test_control_carries_no_intervention():
    """control 就是 pilot 的配方；future-occupancy 權重等一律不得先動。"""
    for attr in dir(PILOT):
        if attr.startswith("_"):
            continue
        if "future" in attr or "occupancy" in attr:
            assert getattr(CONTROL, attr) == getattr(PILOT, attr), attr


def test_config_module_is_pure_declaration():
    import ast

    tree = ast.parse(Path(control_mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "os", "shutil", "socket"):
        assert forbidden not in imported
