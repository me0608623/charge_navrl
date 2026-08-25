"""CPU-only contract tests for the SA5-R2 sealed low-density mix."""

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

from rnn_car_modular.configs import (
    e2e_sa5_r2_sealed_densitymix_from_sa4r3_it125_actdelay12 as mixed,
)
from rnn_car_modular.configs import (
    e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12 as geometry_only,
)


def test_only_declared_density_mix_behavior_changes():
    changed = {
        field.name
        for field in fields(mixed.CONFIG)
        if getattr(mixed.CONFIG, field.name)
        != getattr(geometry_only.CONFIG, field.name)
    }
    assert changed == mixed.ALLOWED_DIFFS
    assert mixed.INTENDED_BEHAVIOR_FIELDS == {
        "long_corridor_obstacle_count_mix"
    }


def test_low_density_mix_is_exact_and_normalized():
    assert mixed.CORRIDOR_DENSITY_MIX == (
        ((0, 1), 0.10),
        ((1, 1), 0.10),
        ((2, 1), 0.15),
        ((3, 2), 0.30),
        ((4, 2), 0.35),
    )
    assert math.isclose(
        sum(weight for _, weight in mixed.CORRIDOR_DENSITY_MIX),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_single_dynamic_and_hard_case_shares_are_preserved():
    single_dynamic = sum(
        weight
        for (_, dynamic), weight in mixed.CORRIDOR_DENSITY_MIX
        if dynamic == 1
    )
    two_dynamic = sum(
        weight
        for (_, dynamic), weight in mixed.CORRIDOR_DENSITY_MIX
        if dynamic == 2
    )
    assert math.isclose(single_dynamic, 0.35, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(two_dynamic, 0.65, rel_tol=0.0, abs_tol=1.0e-12)


def test_parent_optimizer_budget_sensor_and_actuator_do_not_drift():
    cfg = mixed.CONFIG
    base = geometry_only.CONFIG
    assert cfg.checkpoint == base.checkpoint
    assert cfg.no_resume_optimizer is True
    assert cfg.timesteps == 38_400
    assert cfg.save_interval == 50
    assert cfg.num_envs == 1024
    assert cfg.seed == 42
    assert cfg.vlp16_noise_mode == "full"
    assert cfg.lidar_distractor_eligibility == "valid_return_only"
    assert cfg.actuator_delay_range == (1, 2)
    assert cfg.actuator_velocity_scale == (1.0, 1.0)


def test_geometry_and_corridor_reset_share_do_not_drift():
    cfg = mixed.CONFIG
    assert cfg.room_size == 7.5
    assert cfg.long_corridor_length == 10.0
    assert mixed.PHYSICAL_WALL_SPAN_M == 15.0
    assert mixed.BOUNDARY_OVERLAP_M == 0.5
    assert cfg.long_corridor_fraction == 0.10


def test_freeze_matches_config_parent_and_density_mix():
    repo = Path(__file__).resolve().parents[4]
    freeze = json.loads(
        (repo / "docs/freeze/sa5_r2_sealed_densitymix_correction_v1.json")
        .read_text(encoding="utf-8")
    )
    config_path = Path(mixed.__file__).resolve()
    with config_path.open("rb") as config_file:
        config_sha256 = hashlib.file_digest(config_file, "sha256").hexdigest()

    assert freeze["status"] == "AUTHORIZED_NOT_STARTED"
    assert freeze["run_name"] == mixed.RUN_NAME
    assert freeze["parent"]["checkpoint"] == mixed.PARENT_CHECKPOINT
    assert freeze["parent"]["sha256"] == mixed.PARENT_CHECKPOINT_SHA256
    assert freeze["source_sha256"]["densitymix_config"] == config_sha256
    assert freeze["training"]["corridor_density_reset_mix"] == {
        f"{static}S{dynamic}D": weight
        for (static, dynamic), weight in mixed.CORRIDOR_DENSITY_MIX
    }
