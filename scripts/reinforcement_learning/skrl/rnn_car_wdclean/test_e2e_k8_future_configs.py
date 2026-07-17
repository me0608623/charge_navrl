from dataclasses import asdict

from rnn_car_modular.configs.e2e_k8_future_frozen_base import (
    BASE_CONFIG,
    ROOM_SIZE_BY_STAGE,
    make_stage_config,
)


_ALLOWED_STAGE_DIFFERENCES = {"name", "description", "initial_stage", "room_size", "tags"}


def test_frozen_stage_configs_only_change_stage_metadata() -> None:
    base = asdict(BASE_CONFIG)
    for stage, room_size in ROOM_SIZE_BY_STAGE.items():
        config = make_stage_config(stage)
        actual = asdict(config)
        assert config.initial_stage == stage
        assert config.room_size == room_size
        for key, value in base.items():
            if key not in _ALLOWED_STAGE_DIFFERENCES:
                assert actual[key] == value, f"SA{stage} unexpectedly changes {key}"


def test_frozen_recipe_matches_accepted_future_arm() -> None:
    config = make_stage_config(1)
    assert config.fixed_stage is True
    assert config.lidar_frame_stack == 8
    assert config.end_to_end_frame_stack is True
    assert config.algorithm == "ppo"
    assert config.anti_spin_weight == 0.15
    assert config.future_occupancy_weight == 0.10
    assert config.future_occupancy_horizon_s == 1.5
    assert config.future_occupancy_samples == 8
    assert config.future_occupancy_safe_distance_m == 1.0
    assert config.future_occupancy_near_distance_m == 3.0
    assert config.future_occupancy_move_threshold_mps == 0.1
    assert config.checkpoint is None
    assert config.no_resume_optimizer is True
