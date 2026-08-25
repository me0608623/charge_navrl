"""CPU-only locks for the SA4-R3 valid-return-only pilot."""

import ast
import hashlib
from dataclasses import fields
from pathlib import Path

from rnn_car_modular.experiment_config import ExperimentConfig
from rnn_car_modular.configs import (
    e2e_sa4_r2_futureocc015_from_sa3_r1_c100 as r2_mod,
)
from rnn_car_modular.configs import (
    e2e_sa4_r3_valid_return_noise_from_sa3_r1_c100 as r3_mod,
)


R2 = r2_mod.CONFIG
R3 = r3_mod.CONFIG
TRAIN = (
    Path(__file__).resolve().parents[1]
    / "train"
    / "train_rnn_car_wdclip.py"
)


def test_historical_default_is_preserved():
    assert ExperimentConfig().lidar_distractor_eligibility == "all_rays"
    assert R2.lidar_distractor_eligibility == "all_rays"


def test_valid_return_only_is_the_only_behavioral_diff_from_r2():
    drifted = sorted(
        field.name
        for field in fields(R3)
        if field.name not in r3_mod.METADATA_FIELDS
        and getattr(R3, field.name) != getattr(R2, field.name)
    )
    assert drifted == [r3_mod.ELIGIBILITY_FIELD]
    assert R3.lidar_distractor_eligibility == "valid_return_only"


def test_budget_parent_and_checkpoint_schedule_are_fixed():
    assert R3.timesteps == R2.timesteps == 6_400
    assert R3.save_interval == R2.save_interval == 25
    assert r3_mod.CHECKPOINT_STEPS == (3200, 6400)
    assert R3.checkpoint == R2.checkpoint == r3_mod.PARENT_CHECKPOINT
    assert R3.no_resume_optimizer is True
    assert "sa3_sim2real_v2_from_sa2r1_c100" in R3.checkpoint
    assert "sa4_" not in Path(R3.checkpoint).parent.name


def test_parent_hash_matches_disk():
    digest = hashlib.sha256(Path(R3.checkpoint).read_bytes()).hexdigest()
    assert digest == r3_mod.PARENT_CHECKPOINT_SHA256


def test_reward_optimizer_scene_and_sim2real_bundle_are_unchanged():
    for field in fields(R3):
        name = field.name
        if name in r3_mod.ALLOWED_DIFFS:
            continue
        if any(
            token in name
            for token in (
                "reward",
                "penalty",
                "weight",
                "optimizer",
                "lr",
                "epochs",
                "mini_batch",
                "corridor",
                "actuator",
            )
        ):
            assert getattr(R3, name) == getattr(R2, name), name
    assert R3.future_occupancy_weight == 0.15
    assert R3.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)
    assert R3.lidar_frame_stack == 8
    assert R3.actuator_delay_range == (0, 2)


def test_trainer_exposes_and_applies_eligibility_before_env_creation():
    source = TRAIN.read_text(encoding="utf-8")
    assert '"--lidar_distractor_eligibility"' in source
    assert 'choices=["all_rays", "valid_return_only"]' in source
    import_pos = source.index("_apply_lidar_distractor_eligibility_config,")
    call_pos = source.index("_apply_lidar_distractor_eligibility_config(env_cfg, args_cli)")
    noise_pos = source.index("_apply_lidar_noise_config(env_cfg, args_cli)", call_pos - 500)
    actuator_pos = source.index("_apply_actuator_dr_config(env_cfg, args_cli)", call_pos)
    assert import_pos < call_pos
    assert noise_pos < call_pos < actuator_pos


def test_config_module_is_pure_declaration():
    tree = ast.parse(Path(r3_mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "os", "shutil", "socket"):
        assert forbidden not in imported

