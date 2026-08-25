from dataclasses import fields
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_r2_c250_speed0p7_adapt_p50 import (
    CHECKPOINTS,
    CONFIG,
    EXPECTED_DIFFS,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    RATE_1P0_CONTINUATION_CONTROL,
    TRAINING_ITERATIONS,
    _SA5_R2,
    _sha256,
)


def test_parent_and_rate_one_control_are_frozen_files():
    assert Path(PARENT_CHECKPOINT).is_file()
    assert _sha256(PARENT_CHECKPOINT) == PARENT_CHECKPOINT_SHA256
    assert Path(RATE_1P0_CONTINUATION_CONTROL).is_file()


def test_exact_optimizer_continuation_budget_and_speed_contract():
    assert CONFIG.checkpoint == PARENT_CHECKPOINT
    assert CONFIG.no_resume_optimizer is False
    assert CONFIG.algorithm == "ppo"
    assert CONFIG.use_a2c is False
    assert CONFIG.speed_rate == 0.7
    assert CONFIG.speed_rate_obs == "ego"
    assert CONFIG.timesteps == TRAINING_ITERATIONS * CONFIG.rollout_length == 6_400
    assert CONFIG.save_interval == 25
    assert CHECKPOINTS == (
        (25, "checkpoint_3200.pt"),
        (50, "checkpoint_6400.pt"),
    )


def test_only_preregistered_fields_differ_from_sa5_r2():
    drifted = {
        field.name
        for field in fields(CONFIG)
        if getattr(CONFIG, field.name) != getattr(_SA5_R2, field.name)
    }
    assert drifted == EXPECTED_DIFFS


def test_noise_delay_scene_reward_and_network_are_inherited():
    assert CONFIG.vlp16_noise_mode == "full"
    assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
    assert CONFIG.actuator_delay_range == (1, 2)
    assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
    assert CONFIG.actuator_motor_lag == 1.0
    assert CONFIG.end_to_end_frame_stack is True
    assert CONFIG.lidar_frame_stack == 8
    assert CONFIG.reward_profile == _SA5_R2.reward_profile
    assert CONFIG.long_corridor_obstacle_count_mix == _SA5_R2.long_corridor_obstacle_count_mix


def test_trainer_applies_speed_rate_before_normalization_and_buffers_policy_obs():
    trainer = Path(
        "/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train/"
        "train_rnn_car_wdclip.py"
    ).read_text(encoding="utf-8")
    rollout_start = trainer.index("for step in range(RL):")
    transform = trainer.index(
        "policy_obs = apply_vehicle_speed_rate_observation(", rollout_start
    )
    normalizer = trainer.index("obs_normalizer.update(policy_obs)", transform)
    assert transform < normalizer
    assert "value, done, policy_obs, hidden" in trainer
    assert "_bootstrap_policy_obs = apply_vehicle_speed_rate_observation(" in trainer
