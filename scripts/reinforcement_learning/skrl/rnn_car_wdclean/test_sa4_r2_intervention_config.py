"""SA4-R2 intervention config 的鎖定測試（CPU only）。

單變因實驗的價值完全建立在「真的只有一個變因」。這裡逐欄位驗證兩件事：
R2 相對 pilot 只多了 future_occupancy_weight，且 R2 相對 control 只差
metadata 與那一個欄位 —— 後者才是 A/B 的實際契約。
"""

import ast
import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

from rnn_car_modular.configs import (
    e2e_sa4_r2_futureocc015_from_sa3_r1_c100 as r2_mod,
)
from rnn_car_modular.configs import (
    e2e_sa4_d2_control_from_sa3_r1_c100 as control_mod,
)
from rnn_car_modular.configs.e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100 import (
    CONFIG as PILOT,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


R2 = r2_mod.CONFIG
CONTROL = control_mod.CONFIG
METADATA_FIELDS = {"name", "description", "tags", "notes"}


# --------------------------------------------------------------------------
# A/B 契約：R2 vs Control 只差一個行為欄位
# --------------------------------------------------------------------------
def test_intervention_is_the_only_behavioural_diff_against_control():
    """兩臂預算相同，所以除 metadata 外只准差 future_occupancy_weight。"""
    drifted = sorted(
        f.name for f in fields(R2)
        if f.name not in METADATA_FIELDS
        and getattr(R2, f.name) != getattr(CONTROL, f.name)
    )
    assert drifted == [r2_mod.INTERVENTION_FIELD], f"非預期差異：{drifted}"


def test_both_arms_share_budget_and_checkpoints():
    assert R2.timesteps == CONTROL.timesteps == 6_400
    assert R2.save_interval == CONTROL.save_interval == 25
    assert r2_mod.CHECKPOINT_STEPS == control_mod.CHECKPOINT_STEPS == (3200, 6400)


def test_both_arms_share_the_same_parent():
    assert R2.checkpoint == CONTROL.checkpoint
    assert R2.seed == CONTROL.seed == 42


# --------------------------------------------------------------------------
# 相對 pilot：只准 budget + metadata + intervention
# --------------------------------------------------------------------------
def test_only_allowed_fields_differ_from_the_pilot():
    drifted = sorted(
        f.name for f in fields(R2)
        if f.name not in r2_mod.ALLOWED_DIFFS
        and getattr(R2, f.name) != getattr(PILOT, f.name)
    )
    assert drifted == [], f"非允許欄位也變了：{drifted}"


def test_allowed_diff_set_is_explicit():
    assert r2_mod.ALLOWED_DIFFS == frozenset(
        {"name", "description", "timesteps", "save_interval", "tags", "notes",
         "future_occupancy_weight"}
    )


def test_drift_lock_runs_at_import():
    source = Path(r2_mod.__file__).read_text()
    assert "_drifted" in source
    assert "raise RuntimeError" in source


# --------------------------------------------------------------------------
# intervention 本身
# --------------------------------------------------------------------------
def test_future_occupancy_weight_is_the_probe_value():
    assert PILOT.future_occupancy_weight == 0.10
    assert R2.future_occupancy_weight == 0.15
    assert r2_mod.INTERVENTION_FUTURE_OCCUPANCY_WEIGHT == 0.15


def test_probe_step_is_guarded_against_a_moved_baseline():
    """pilot 的基準值若被改動，0.15 這個步長就失去依據，必須 raise。"""
    source = Path(r2_mod.__file__).read_text()
    assert "BASELINE_FUTURE_OCCUPANCY_WEIGHT" in source
    assert "must be re-derived" in source


@pytest.mark.parametrize(
    "attr",
    ["future_occupancy_horizon_s", "future_occupancy_samples",
     "future_occupancy_safe_distance_m", "future_occupancy_near_distance_m",
     "future_occupancy_move_threshold_mps"],
)
def test_rest_of_the_future_occupancy_family_is_unchanged(attr):
    """只改幅度，不改該項的形狀。"""
    assert getattr(R2, attr) == getattr(PILOT, attr)


def test_no_other_reward_term_moved():
    for f in fields(R2):
        name = f.name
        if name == r2_mod.INTERVENTION_FIELD:
            continue
        if any(tag in name for tag in ("reward", "penalty", "weight", "anti_spin")):
            assert getattr(R2, name) == getattr(PILOT, name), name


# --------------------------------------------------------------------------
# 明確禁止的同輪變更
# --------------------------------------------------------------------------
def test_corridor_family_weights_untouched():
    assert R2.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)
    assert (
        R2.long_corridor_dynamic_motion_weights
        == PILOT.long_corridor_dynamic_motion_weights
    )


def test_optimizer_and_learning_rate_untouched():
    for f in fields(R2):
        name = f.name
        if any(tag in name for tag in ("lr", "learning_rate", "optimizer",
                                       "grad_clip", "epochs", "mini_batch")):
            assert getattr(R2, name) == getattr(PILOT, name), name


# --------------------------------------------------------------------------
# 共同起點與場景
# --------------------------------------------------------------------------
def test_parent_is_sa3_c100_and_no_sa4_checkpoint_is_inherited():
    assert R2.checkpoint.endswith(
        "sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1/checkpoint_12800.pt"
    )
    assert "sa4_" not in Path(R2.checkpoint).parent.name
    assert R2.no_resume_optimizer is True


def test_parent_hash_verified_at_import_and_on_disk():
    source = Path(r2_mod.__file__).read_text()
    assert "hashlib.file_digest" in source
    digest = hashlib.sha256(Path(R2.checkpoint).read_bytes()).hexdigest()
    assert digest == r2_mod.PARENT_CHECKPOINT_SHA256


def test_stage_four_scene_still_from_the_spec():
    spec = STAGE_SCENE_CURRICULUM[4]
    assert R2.initial_stage == 4 and R2.fixed_stage is True
    assert R2.room_size == spec.room_half_extent
    assert R2.long_corridor_free_width == spec.corridor_free_width
    assert R2.long_corridor_dynamic_obstacles == spec.corridor_dynamic_obstacles


def test_sim2real_mechanisms_retained():
    assert R2.lidar_frame_stack == 8
    assert R2.actuator_delay_range == (0, 2)
    assert R2.lidar_no_noise is False
    assert R2.lidar_hole_rate > 0.0
    assert tuple(getattr(R2, "obs_delay_steps", (0, 0))) == (0, 0)


# --------------------------------------------------------------------------
# 不得自行啟動任何東西
# --------------------------------------------------------------------------
def test_config_module_is_pure_declaration():
    tree = ast.parse(Path(r2_mod.__file__).read_text())
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
