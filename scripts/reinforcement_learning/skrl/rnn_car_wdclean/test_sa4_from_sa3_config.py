"""SA4 warm-start config 的鎖定測試（CPU only）。

這支 config 的錯法都是安靜的：接錯 parent、場景漂到別的階段、sim2real 機制被
warm start 洗掉。全部在 import 期或這裡就要吵出來。
"""

import hashlib
from pathlib import Path

import pytest

from rnn_car_modular.configs import e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100 as mod
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


CONFIG = mod.CONFIG
SPEC4 = STAGE_SCENE_CURRICULUM[4]

#: The checkpoint the SA3 graduation gate selected. Not c300.
GATE_SELECTED_SHA256 = (
    "e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88"
)


# --------------------------------------------------------------------------
# parent 血緣
# --------------------------------------------------------------------------
def test_parent_is_the_gate_selected_sa3_c100():
    assert mod.PARENT_CHECKPOINT.endswith(
        "sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1/checkpoint_12800.pt"
    )
    assert mod.PARENT_CHECKPOINT_SHA256 == GATE_SELECTED_SHA256


def test_parent_hash_is_verified_at_import_not_merely_recorded():
    source = Path(mod.__file__).read_text()
    assert "hashlib.file_digest" in source
    assert "raise RuntimeError" in source


def test_parent_file_on_disk_still_matches():
    digest = hashlib.sha256(Path(mod.PARENT_CHECKPOINT).read_bytes()).hexdigest()
    assert digest == GATE_SELECTED_SHA256


def test_parent_is_not_the_final_sa3_checkpoint():
    """c300 的 lateral CR 是六顆最差（0.0472 vs c100 0.0171）。"""
    assert "checkpoint_38400" not in mod.PARENT_CHECKPOINT
    assert CONFIG.checkpoint == mod.PARENT_CHECKPOINT


# --------------------------------------------------------------------------
# pilot 預算
# --------------------------------------------------------------------------
def test_pilot_is_exactly_one_hundred_iterations():
    assert mod.PILOT_ITERATIONS == 100
    assert CONFIG.timesteps == 100 * 128 == 12_800


def test_checkpoints_every_fifty_iterations():
    assert CONFIG.save_interval == 50


def test_optimizer_state_is_not_resumed():
    assert CONFIG.no_resume_optimizer is True


def test_run_shape_is_pinned():
    assert CONFIG.num_envs == 1024
    assert CONFIG.seed == 42
    assert CONFIG.rollout_length == 128


# --------------------------------------------------------------------------
# 階段與場景：只能來自 STAGE_SCENE_CURRICULUM[4]
# --------------------------------------------------------------------------
def test_stage_is_four_and_fixed():
    assert CONFIG.initial_stage == 4
    assert CONFIG.fixed_stage is True


def test_room_matches_stage_four_spec():
    assert CONFIG.room_size == SPEC4.room_half_extent == 8.0


@pytest.mark.parametrize(
    "config_attr, spec_attr",
    [
        ("narrow_passage_fraction", "narrow_fraction"),
        ("narrow_passage_fixed_width_range", "narrow_width_range"),
        ("narrow_passage_fixed_yaw_limit_deg", "narrow_yaw_limit_deg"),
        ("long_corridor_fraction", "corridor_fraction"),
        ("long_corridor_free_width", "corridor_free_width"),
        ("long_corridor_static_obstacles", "corridor_static_obstacles"),
        ("long_corridor_dynamic_obstacles", "corridor_dynamic_obstacles"),
        ("long_corridor_dynamic_speed_range", "corridor_speed_range"),
    ],
)
def test_every_scene_value_comes_from_the_stage_spec(config_attr, spec_attr):
    assert getattr(CONFIG, config_attr) == getattr(SPEC4, spec_attr)


def test_sa4_is_harder_than_sa3_in_the_expected_directions():
    """接錯階段的典型症狀是場景沒變難。"""
    spec3 = STAGE_SCENE_CURRICULUM[3]
    assert SPEC4.room_half_extent < spec3.room_half_extent
    assert SPEC4.narrow_width_range[0] < spec3.narrow_width_range[0]
    assert SPEC4.corridor_free_width < spec3.corridor_free_width
    assert SPEC4.corridor_dynamic_obstacles > spec3.corridor_dynamic_obstacles


# --------------------------------------------------------------------------
# sim2real 機制必須活過 warm start
# --------------------------------------------------------------------------
def test_frame_stack_k8_retained():
    assert CONFIG.lidar_frame_stack == 8


def test_actuator_delay_is_the_uniform_zero_one_two():
    assert CONFIG.actuator_delay_range == (0, 2)


def test_lidar_full_noise_retained():
    assert CONFIG.lidar_no_noise is False
    assert CONFIG.lidar_hole_rate > 0.0
    assert CONFIG.lidar_distractor_rate > 0.0
    assert CONFIG.lidar_displacement_std_soft > 0.0


def test_observation_delay_stays_off():
    """obs_delay_steps 是先前確診的滿舵塌縮兇手，永遠不得開啟。"""
    assert tuple(getattr(CONFIG, "obs_delay_steps", (0, 0))) == (0, 0)


# --------------------------------------------------------------------------
# 不得自動延長或啟動下一階段
# --------------------------------------------------------------------------
def test_config_module_is_pure_declaration():
    """config 只描述一次 run，不得含任何啟動或執行手段。

    注意不能用子字串找 'sa5' —— notes 正當地寫著「SA5 is not started by this
    run」。要驗的是「有沒有啟動的能力」，不是「有沒有提到名字」。
    """
    import ast

    tree = ast.parse(Path(mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "os", "shutil", "socket"):
        assert forbidden not in imported
    calls = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("system", "run", "Popen", "spawn"):
        assert forbidden not in calls
