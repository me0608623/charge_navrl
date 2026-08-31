"""CPU contracts for the c600 K8 interaction observability probe."""

from __future__ import annotations

import numpy as np
import torch

from rnn_car_wdclean.analyze_sa5_c600_k8_observability import (
    _binary_metrics,
    _split_masks,
    build_lateral_dataset,
    build_reversal_dataset,
    decision,
)


def _trace(steps: int = 30, envs: int = 1, slots: int = 1) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {
        "episode_step": np.tile(np.arange(steps)[:, None], (1, envs)),
        "dynamic_positions_m": np.zeros((steps, envs, slots, 2), np.float32),
        "dynamic_velocities_mps": np.zeros((steps, envs, slots, 2), np.float32),
        "dynamic_valid": np.ones((steps, envs, slots), bool),
        "patrol_active": np.ones((steps, envs, slots), bool),
        "patrol_waypoint_index": np.zeros((steps, envs, slots), np.int16),
        "robot_xy_m": np.zeros((steps, envs, 2), np.float32),
        "pre_delay_command_mps_rad_s": np.zeros((steps, envs, 2), np.float32),
        "post_delay_command_mps_rad_s": np.zeros((steps, envs, 2), np.float32),
        "done": np.zeros((steps, envs), bool),
        "termination_cause": np.zeros((steps, envs), np.int8),
        "policy_current_input": np.zeros((steps, envs, 83), np.float16),
        "policy_k8_input": np.zeros((steps, envs, 587), np.float16),
        "policy_representation": np.zeros((steps, envs, 179), np.float16),
    }
    return data


def test_binary_metrics_are_balanced():
    metrics = _binary_metrics(
        torch.tensor([2.0, -2.0, 1.0, -1.0]),
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )
    assert metrics["positive_recall"] == 0.5
    assert metrics["negative_recall"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_environment_split_has_no_overlap():
    env_ids = np.arange(30)
    masks = _split_masks(env_ids)
    assert not np.any(masks["train"] & masks["validation"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["validation"] & masks["test"])
    assert np.all(masks["train"] | masks["validation"] | masks["test"])


def test_lateral_label_uses_side_before_crossing():
    data = _trace()
    data["dynamic_positions_m"][:, 0, 0, 1] = 1.0
    data["dynamic_positions_m"][:9, 0, 0, 0] = np.linspace(0.5, 0.03, 9)
    data["dynamic_positions_m"][9:, 0, 0, 0] = np.linspace(-0.03, -0.5, 21)
    data["dynamic_velocities_mps"][:, 0, 0, 0] = -0.2
    dataset = build_lateral_dataset(data)
    assert dataset.targets.tolist() == [1]
    assert dataset.metadata["positive_label"] == "+local_x side was just vacated"


def test_reversal_label_is_exactly_one_second_before_actual_motion_reversal():
    data = _trace()
    data["dynamic_positions_m"][:, 0, 0, 1] = 1.0
    data["dynamic_velocities_mps"][:15, 0, 0, 0] = 0.2
    data["dynamic_velocities_mps"][15:17, 0, 0, 0] = 0.0
    data["dynamic_velocities_mps"][17:, 0, 0, 0] = -0.2
    data["patrol_waypoint_index"][15:, 0, 0] = 1
    dataset = build_reversal_dataset(data)
    assert int(dataset.targets.sum()) == 1
    positive_index = int(np.flatnonzero(dataset.targets == 1)[0])
    assert dataset.features["policy_k8_input"].shape[1] == 587
    assert dataset.env_ids[positive_index] == 0
    assert dataset.metadata["positive_samples"] == 1


def _task_report(balanced_accuracy: float) -> dict:
    metrics = {
        "balanced_accuracy": balanced_accuracy,
        "positive_recall": balanced_accuracy,
        "negative_recall": balanced_accuracy,
    }
    return {
        "feature_sets": {
            name: {"selected_by_validation": {"test": metrics}}
            for name in (
                "policy_current_input",
                "policy_k8_input",
                "policy_representation",
            )
        }
    }


def test_reversal_failure_is_explicitly_non_blocking_for_sa6():
    result = decision(_task_report(0.8), _task_report(0.5))
    assert result["vacated_side"]["policy_representation_observable"] is True
    assert result["reversal_1s_ahead"]["policy_representation_observable"] is False
    assert result["reversal_1s_ahead"]["blocking_for_sa6"] is False
    assert result["sa6_blocked_by_this_probe"] is False
