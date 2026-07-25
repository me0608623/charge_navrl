import json

import numpy as np
import pytest
import torch
from torch import nn

from rnn_car_wdclean.corridor_teacher_distillation import (
    _soft_neighbor_targets,
    post_update_corridor_action_projection,
    select_corridor_interventions,
)
from rnn_car_wdclean.fit_corridor_residual_adapter import _load_datasets


def _write_teacher_dataset(
    path,
    *,
    source: str,
    feasible: list[bool],
) -> None:
    frame_count = len(feasible)
    np.savez_compressed(
        path,
        policy_input=np.zeros((frame_count, 179), dtype=np.float16),
        teacher_action=np.zeros((frame_count, 2), dtype=np.uint8),
        feasible=np.asarray(feasible, dtype=bool),
        env_id=np.asarray(
            [index % 2 for index in range(frame_count)],
            dtype=np.int16,
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "trajectory_source": source,
                    "teacher_replaced_policy_actions": (
                        source == "privileged_teacher"
                    ),
                }
            )
        ),
    )


def test_load_datasets_aggregates_sources_and_offsets_env_ids(
    tmp_path,
) -> None:
    teacher_path = tmp_path / "teacher.npz"
    student_path = tmp_path / "student.npz"
    _write_teacher_dataset(
        teacher_path,
        source="privileged_teacher",
        feasible=[True, False, True, True],
    )
    _write_teacher_dataset(
        student_path,
        source="student_policy",
        feasible=[True, True, False, True],
    )

    features, actions, env_ids, dataset_ids, summaries = _load_datasets(
        [str(teacher_path), str(student_path)]
    )

    assert features.shape == (6, 179)
    assert actions.shape == (6, 2)
    assert env_ids[:3].max() < env_ids[3:].min()
    assert dataset_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert summaries[0]["trajectory_source"] == "privileged_teacher"
    assert summaries[1]["trajectory_source"] == "student_policy"
    assert summaries[0]["feasible_frames"] == 3


def test_soft_neighbor_targets_preserve_mass_at_edges() -> None:
    indices = torch.tensor([0, 3, 4])
    targets = _soft_neighbor_targets(
        indices,
        num_bins=5,
        neighbor_mass=0.2,
        dtype=torch.float32,
    )
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(
        targets[0], torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        targets[1], torch.tensor([0.0, 0.0, 0.1, 0.8, 0.1])
    )
    torch.testing.assert_close(
        targets[2], torch.tensor([0.0, 0.0, 0.0, 0.2, 0.8])
    )


def test_projection_improves_teacher_action_agreement() -> None:
    torch.manual_seed(7)
    model = nn.Linear(4, 10)
    inputs = torch.randn(256, 4)
    teacher_linear = (inputs[:, 0] > 0).long() * 4
    teacher_angular = (inputs[:, 1] > 0).long() * 4
    teacher_actions = torch.stack(
        [teacher_linear, teacher_angular], dim=-1
    )
    indices = torch.arange(inputs.shape[0])

    stats = post_update_corridor_action_projection(
        lambda selected: model(inputs[selected]),
        model.parameters(),
        indices,
        teacher_actions,
        num_bins=5,
        epochs=20,
        learning_rate=0.2,
        batch_size=64,
        max_grad_norm=1.0,
        neighbor_mass=0.1,
    )

    assert stats.active_count == 256
    assert stats.optimizer_steps == 80
    assert stats.loss_after < stats.loss_before
    assert (
        stats.agreement_after_joint
        > stats.agreement_before_joint + 0.25
    )
    assert stats.within_one_after_joint >= stats.agreement_after_joint


def test_intervention_mask_requires_conflict_feasible_teacher_and_difference() -> None:
    policy = torch.tensor([[0, 0], [1, 1], [2, 2], [1, 0], [0, 2]])
    teacher = torch.tensor([[0, 1], [1, 2], [2, 2], [2, 0], [1, 2]])
    feasible = torch.tensor([True, True, True, True, False])
    obstacle_collision = torch.zeros(5, 3, 3, dtype=torch.bool)
    wall_collision = torch.zeros_like(obstacle_collision)
    clearance = torch.ones(5, 3, 3)
    obstacle_collision[0, 0, 0] = True
    wall_collision[2, 2, 2] = True
    clearance[3, 1, 0] = 0.15
    obstacle_collision[4, 0, 2] = True

    result = select_corridor_interventions(
        policy,
        teacher,
        feasible,
        obstacle_collision,
        wall_collision,
        clearance,
        clearance_threshold_m=0.20,
    )

    assert result["intervention"].tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]
    assert result["policy_obstacle_collision"].tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert result["policy_wall_collision"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert result["policy_low_clearance"].tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_intervention_mask_rejects_invalid_action_indices() -> None:
    with pytest.raises(ValueError, match="outside the teacher grid"):
        select_corridor_interventions(
            torch.tensor([[3, 0]]),
            torch.tensor([[0, 0]]),
            torch.tensor([True]),
            torch.zeros(1, 3, 3, dtype=torch.bool),
            torch.zeros(1, 3, 3, dtype=torch.bool),
            torch.ones(1, 3, 3),
            clearance_threshold_m=0.20,
        )
