"""CPU-only contract tests for the measured-delay SA5 replacement config."""

from dataclasses import fields
from statistics import median

import pytest

from rnn_car_modular.configs import (
    e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver as old_mod,
)
from rnn_car_modular.configs import (
    e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12 as new_mod,
)


OLD = old_mod.CONFIG
NEW = new_mod.CONFIG


def test_replacement_changes_only_delay_and_metadata():
    changed = frozenset(
        field.name
        for field in fields(NEW)
        if field.name not in new_mod.METADATA_FIELDS
        and getattr(NEW, field.name) != getattr(OLD, field.name)
    )
    assert changed == new_mod.EXPECTED_CHANGED_FIELDS == {"actuator_delay_range"}


def test_delay_center_changes_from_200_to_300_ms():
    assert OLD.actuator_delay_range == (0, 2)
    assert NEW.actuator_delay_range == (1, 2)
    assert sum(OLD.actuator_delay_range) / 2 * 0.2 == pytest.approx(0.2)
    assert sum(NEW.actuator_delay_range) / 2 * 0.2 == pytest.approx(0.3)


def test_aggregate_028_044_is_not_encoded_as_steady_gain():
    assert min(new_mod.CLEAN_WINDOW_STATIC_GAINS) >= 0.98
    assert max(new_mod.CLEAN_WINDOW_STATIC_GAINS) <= 1.05
    assert 0.99 <= median(new_mod.CLEAN_WINDOW_STATIC_GAINS) <= 1.02
    assert NEW.actuator_velocity_scale == OLD.actuator_velocity_scale == (1.0, 1.0)


def test_real_delay_evidence_and_bounded_approximation_are_recorded():
    assert new_mod.REAL_ROBOT_DELAY_QUANTILES_MS == (0, 300, 400)
    assert min(new_mod.CLEAN_WINDOW_BEST_LAGS_MS) >= 350
    assert max(new_mod.CLEAN_WINDOW_BEST_LAGS_MS) <= 400
    text = " ".join((NEW.description, " ".join(NEW.tags), NEW.notes)).lower()
    assert "bounded approximation" in text
    assert "not an exact empirical distribution" in text


def test_scale_lag_parent_budget_and_scene_are_unchanged():
    for field in fields(NEW):
        if field.name in new_mod.ALLOWED_DIFFS:
            continue
        assert getattr(NEW, field.name) == getattr(OLD, field.name), field.name
    assert NEW.actuator_motor_lag == OLD.actuator_motor_lag == 1.0
    assert NEW.actuator_motor_lag_by_channel is OLD.actuator_motor_lag_by_channel is None
    assert NEW.checkpoint == OLD.checkpoint == new_mod.PARENT_CHECKPOINT
    assert NEW.no_resume_optimizer is OLD.no_resume_optimizer is True
    assert NEW.initial_stage == OLD.initial_stage == 5
    assert NEW.timesteps == OLD.timesteps == 38_400
    assert NEW.save_interval == OLD.save_interval == 50


def test_observation_covers_the_two_step_maximum_delay():
    assert NEW.use_action_history is True
    assert NEW.lidar_frame_stack == 8
    assert NEW.actuator_delay_range[1] == 2


def test_sensor_reward_model_and_teacher_contract_remain_frozen():
    assert NEW.vlp16_noise_mode == "full"
    assert NEW.lidar_distractor_eligibility == "valid_return_only"
    assert NEW.future_occupancy_weight == 0.15
    assert NEW.corridor_teacher_distill_epochs == 0
    assert NEW.corridor_teacher_goal_denominator_floor_m == 1.0
