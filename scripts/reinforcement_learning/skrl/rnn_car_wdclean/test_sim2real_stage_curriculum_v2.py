"""Contract tests for the canonical SA1-SA8 sim-to-real scene curriculum."""

import ast
from dataclasses import fields
import importlib
import math
import pathlib
import sys


_REPO = pathlib.Path(__file__).resolve().parents[4]
_SKRL = _REPO / "scripts" / "reinforcement_learning" / "skrl"
if str(_SKRL) not in sys.path:
    sys.path.insert(0, str(_SKRL))


from rnn_car_modular.configs.e2e_k8_future_frozen_base import (  # noqa: E402
    ROOM_SIZE_BY_STAGE,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (  # noqa: E402
    SA7_DEPLOYMENT_DENSITY_MIX,
    SA8_STRESS_DENSITY_MIX,
    STAGE_SCENE_CURRICULUM,
    pending_parent_checkpoint,
)


def _load_stage_config(stage: int):
    module = importlib.import_module(
        "rnn_car_modular.configs."
        f"e2e_sa{stage}_k8_obb_sim2real_curriculum_v2"
    )
    return module.CONFIG


CONFIGS = tuple(_load_stage_config(stage) for stage in range(1, 9))

SCENE_VARIANT_FIELDS = {
    "name",
    "description",
    "initial_stage",
    "room_size",
    "checkpoint",
    "previous_stage_replay_fraction",
    "narrow_passage_fraction",
    "narrow_passage_segment_length",
    "narrow_passage_final_stress_ratio",
    "narrow_passage_fixed_width_range",
    "narrow_passage_fixed_yaw_limit_deg",
    "narrow_passage_exact_width",
    "narrow_passage_exact_width_ratio",
    "long_corridor_fraction",
    "long_corridor_free_width",
    "long_corridor_length",
    "long_corridor_static_obstacles",
    "long_corridor_dynamic_obstacles",
    "long_corridor_obstacle_count_mix",
    "long_corridor_gate_aligned_share",
    "long_corridor_dynamic_speed_range",
    "long_corridor_dynamic_motion_mode",
    "long_corridor_dynamic_motion_weights",
    "long_corridor_random_2d_kinematics",
    "tags",
    "notes",
}


def _density_mean(spec) -> tuple[float, float]:
    if spec.corridor_density_mix is None:
        return (
            float(spec.corridor_static_obstacles),
            float(spec.corridor_dynamic_obstacles),
        )
    static = sum(
        counts[0] * weight
        for counts, weight in spec.corridor_density_mix
    )
    dynamic = sum(
        counts[1] * weight
        for counts, weight in spec.corridor_density_mix
    )
    return static, dynamic


def test_stage_table_is_complete_and_uses_frozen_room_sizes():
    assert tuple(STAGE_SCENE_CURRICULUM) == tuple(range(1, 9))
    assert [config.initial_stage for config in CONFIGS] == list(range(1, 9))
    assert [config.room_size for config in CONFIGS] == [
        ROOM_SIZE_BY_STAGE[stage] for stage in range(1, 9)
    ]
    assert CONFIGS[0].room_size == 10.0
    assert CONFIGS[1].room_size == 9.0
    assert CONFIGS[-1].room_size == 6.0


def test_sa1_is_navigation_dominant_and_sa2_is_easy_3s1d():
    sa1, sa2 = STAGE_SCENE_CURRICULUM[1], STAGE_SCENE_CURRICULUM[2]
    assert (sa1.native_fraction, sa1.narrow_fraction, sa1.corridor_fraction) == (
        0.94,
        0.04,
        0.02,
    )
    assert sa1.narrow_width_range == (1.8, 2.0)
    assert (
        sa1.corridor_static_obstacles,
        sa1.corridor_dynamic_obstacles,
    ) == (2, 1)
    assert (
        sa2.corridor_static_obstacles,
        sa2.corridor_dynamic_obstacles,
    ) == (3, 1)
    assert sa2.corridor_motion_weights == (0.5, 0.5, 0.0)
    assert sa2.random_2d_kinematics == "patrol"


def test_scene_difficulty_progresses_monotonically():
    specs = [STAGE_SCENE_CURRICULUM[stage] for stage in range(1, 9)]
    native = [spec.native_fraction for spec in specs]
    narrow = [spec.narrow_fraction for spec in specs]
    corridor = [spec.corridor_fraction for spec in specs]
    width_lo = [spec.narrow_width_range[0] for spec in specs]
    width_hi = [spec.narrow_width_range[1] for spec in specs]
    corridor_width = [spec.corridor_free_width for spec in specs]
    speed_lo = [spec.corridor_speed_range[0] for spec in specs]
    speed_hi = [spec.corridor_speed_range[1] for spec in specs]
    density = [_density_mean(spec) for spec in specs]

    assert all(a >= b for a, b in zip(native, native[1:]))
    assert all(a <= b for a, b in zip(narrow, narrow[1:]))
    assert all(a <= b for a, b in zip(corridor, corridor[1:]))
    assert all(a >= b for a, b in zip(width_lo, width_lo[1:]))
    assert all(a >= b for a, b in zip(width_hi, width_hi[1:]))
    assert all(a >= b for a, b in zip(corridor_width, corridor_width[1:]))
    assert all(a <= b for a, b in zip(speed_lo, speed_lo[1:]))
    assert all(a <= b for a, b in zip(speed_hi, speed_hi[1:]))
    assert all(a[0] <= b[0] for a, b in zip(density, density[1:]))
    assert all(a[1] <= b[1] for a, b in zip(density, density[1:]))


def test_random_2d_begins_at_sa5_and_wander_waits_until_sa7():
    for stage in range(1, 5):
        spec = STAGE_SCENE_CURRICULUM[stage]
        assert spec.corridor_density_mix is None
        if spec.corridor_motion_weights is not None:
            assert spec.corridor_motion_weights[2] == 0.0
    assert STAGE_SCENE_CURRICULUM[5].corridor_density_mix is not None
    assert STAGE_SCENE_CURRICULUM[5].random_2d_kinematics == "patrol"
    assert STAGE_SCENE_CURRICULUM[6].random_2d_kinematics == "patrol"
    assert STAGE_SCENE_CURRICULUM[7].random_2d_kinematics == "wander"
    assert STAGE_SCENE_CURRICULUM[8].random_2d_kinematics == "wander"


def test_native_curriculum_introduces_dedicated_crossing_at_sa3():
    source = (
        _REPO
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "charge_skrl"
        / "curriculum"
        / "phases"
        / "e2e_final20_v1.py"
    )
    tree = ast.parse(source.read_text())
    crossing_fraction = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "_CORRIDOR_FRACTION"
            for target in node.targets
        ):
            crossing_fraction = ast.literal_eval(node.value)
            break

    assert crossing_fraction is not None
    assert crossing_fraction["SA1_nav_bootstrap"] == 0.0
    assert crossing_fraction["SA2_nav_static"] == 0.0
    assert crossing_fraction["SA3_walls_crossing"] > 0.0
    assert crossing_fraction["SA4_spatial_plan"] > 0.0


def test_sa6_reaches_deployment_gap_and_sa7_sa8_use_full_density_tables():
    assert STAGE_SCENE_CURRICULUM[6].narrow_width_range == (1.2, 1.4)
    assert STAGE_SCENE_CURRICULUM[7].corridor_density_mix == (
        SA7_DEPLOYMENT_DENSITY_MIX
    )
    assert STAGE_SCENE_CURRICULUM[8].corridor_density_mix == (
        SA8_STRESS_DENSITY_MIX
    )
    assert STAGE_SCENE_CURRICULUM[7].narrow_exact_width_ratio == 0.25
    assert STAGE_SCENE_CURRICULUM[8].narrow_exact_width_ratio == 0.50


def test_sensor_actuator_model_and_optimizer_contract_is_invariant():
    base = CONFIGS[0]
    invariant_fields = {
        field.name for field in fields(base)
    } - SCENE_VARIANT_FIELDS
    for config in CONFIGS[1:]:
        for field_name in invariant_fields:
            assert getattr(config, field_name) == getattr(base, field_name), (
                config.name,
                field_name,
            )

    for config in CONFIGS:
        assert config.lidar_frame_stack == 8
        assert config.end_to_end_frame_stack is True
        assert config.use_action_history is True
        assert config.lidar_no_noise is False
        assert config.vlp16_noise_mode == "full"
        assert config.enable_actuator_dr is True
        assert config.actuator_delay_range == (0, 2)
        assert config.actuator_velocity_scale == (1.0, 1.0)
        assert config.actuator_motor_lag == 1.0
        assert config.obs_delay_steps == (0, 0)
        assert config.no_domain_randomization is True
        assert config.long_corridor_gate_aligned_share == 0.0


def test_sa2_sa8_fail_closed_until_each_parent_passes_gates():
    assert CONFIGS[0].checkpoint is None
    assert "from_scratch" in CONFIGS[0].tags
    for stage, config in enumerate(CONFIGS[1:], start=2):
        sentinel = pending_parent_checkpoint(stage)
        assert config.checkpoint == sentinel
        assert not (_REPO / sentinel).exists()
        assert config.no_resume_optimizer is True
        assert "warm_start" in config.tags


def test_nominal_reset_shares_are_explicit_and_sum_to_one():
    expected = {
        1: (0.94, 0.04, 0.02),
        2: (0.90, 0.06, 0.04),
        3: (0.86, 0.08, 0.06),
        4: (0.82, 0.10, 0.08),
        5: (0.78, 0.12, 0.10),
        6: (0.78, 0.12, 0.10),
        7: (0.78, 0.12, 0.10),
        8: (0.78, 0.12, 0.10),
    }
    for stage, shares in expected.items():
        spec = STAGE_SCENE_CURRICULUM[stage]
        actual = (
            spec.native_fraction,
            spec.narrow_fraction,
            spec.corridor_fraction,
        )
        assert all(
            math.isclose(value, expected_value, abs_tol=1e-12)
            for value, expected_value in zip(actual, shares)
        )
        assert abs(sum(actual) - 1.0) < 1e-12


def test_trainer_accepts_early_stage_narrow_replay_shares():
    trainer = (
        _REPO
        / "scripts"
        / "reinforcement_learning"
        / "skrl"
        / "train"
        / "train_rnn_car_wdclip.py"
    ).read_text()

    assert "0.0 < _narrow_fraction <= 0.20" in trainer
    assert "0.10 <= _narrow_fraction <= 0.15" not in trainer
    assert all(
        0.0 < STAGE_SCENE_CURRICULUM[stage].narrow_fraction <= 0.20
        for stage in range(1, 9)
    )
