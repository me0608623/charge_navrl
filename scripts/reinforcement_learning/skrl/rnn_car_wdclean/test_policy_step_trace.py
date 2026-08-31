"""CPU contracts for the SA5 c600 policy-only step trace."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rnn_car_wdclean.policy_step_trace import PolicyStepTraceRecorder


def _batch(step: int, *, episode_step: int, done: bool = False) -> dict:
    pre = torch.tensor([[0.2 + 0.1 * step, 0.05 * step]])
    post = torch.zeros_like(pre) if step == 0 else torch.tensor(
        [[0.2 + 0.1 * (step - 1), 0.05 * (step - 1)]]
    )
    return {
        "step": step,
        "episode_step": torch.tensor([episode_step]),
        "policy_actions": torch.tensor([[8, 11]]),
        "effective_actions": torch.tensor([[8, 11]]),
        "pre_delay_command": pre,
        "post_delay_command": post,
        "robot_xy_m": torch.tensor([[0.0, 0.2 * step]]),
        "robot_yaw_rad": torch.tensor([0.0]),
        "robot_velocity_body_mps": torch.tensor([[0.3, 0.0]]),
        "dynamic_positions_m": torch.tensor(
            [[[0.2 - 0.1 * step, 1.0], [-0.5, 2.0]]]
        ),
        "dynamic_velocities_mps": torch.tensor(
            [[[-0.5, 0.0], [0.0, 0.0]]]
        ),
        "dynamic_valid": torch.tensor([[True, True]]),
        "patrol_waypoint_index": torch.tensor([[0, 1]]),
        "patrol_pause_remaining": torch.tensor([[0, 0]]),
        "patrol_target_m": torch.tensor([[[-1.0, 1.0], [-0.5, 3.0]]]),
        "patrol_active": torch.tensor([[True, True]]),
        "done": torch.tensor([done]),
        "termination_cause": torch.tensor([3 if done else 0]),
    }


def test_trace_preserves_policy_identity_and_d1_alignment(tmp_path: Path):
    recorder = PolicyStepTraceRecorder(1, 2, expected_delay_steps=1)
    for step in range(3):
        recorder.record_tensor_batch(
            **_batch(step, episode_step=step, done=step == 2)
        )
    output = tmp_path / "trace.npz"
    report = recorder.write(output, metadata={"scenario": "lateral"})

    assert report["self_check"]["policy_identity_ok"] is True
    assert report["self_check"]["delay_alignment_ok"] is True
    assert report["self_check"]["delay_alignment_samples"] == 2
    assert report["self_check"]["reconciliation_ok"] is True
    with np.load(output, allow_pickle=False) as trace:
        assert trace["policy_action_indices"].shape == (3, 1, 2)
        assert trace["dynamic_positions_m"].shape == (3, 1, 2, 2)
        metadata = json.loads(str(trace["metadata_json"].item()))
        assert metadata["teacher_replaced_policy_actions"] is False


def test_trace_fails_closed_if_policy_action_is_changed():
    recorder = PolicyStepTraceRecorder(1, 2, expected_delay_steps=1)
    row = _batch(0, episode_step=0)
    row["effective_actions"] = torch.tensor([[8, 12]])
    with pytest.raises(RuntimeError, match="modified policy action"):
        recorder.record_tensor_batch(**row)


def test_trace_reports_immediate_application_as_delay_failure(tmp_path: Path):
    recorder = PolicyStepTraceRecorder(1, 2, expected_delay_steps=1)
    recorder.record_tensor_batch(**_batch(0, episode_step=0))
    row = _batch(1, episode_step=1)
    row["post_delay_command"] = row["pre_delay_command"].clone()
    recorder.record_tensor_batch(**row)
    report = recorder.write(tmp_path / "bad.npz", metadata={})
    assert report["self_check"]["delay_alignment_ok"] is False
    assert report["self_check"]["reconciliation_ok"] is False


def test_trace_optionally_preserves_exact_k8_policy_inputs(tmp_path: Path):
    recorder = PolicyStepTraceRecorder(
        1, 2, expected_delay_steps=1, include_policy_inputs=True
    )
    row = _batch(0, episode_step=0)
    row.update(
        policy_current_input=torch.arange(83).reshape(1, 83).float(),
        policy_k8_input=torch.arange(587).reshape(1, 587).float(),
        policy_representation=torch.arange(179).reshape(1, 179).float(),
    )
    recorder.record_tensor_batch(**row)
    output = tmp_path / "k8.npz"
    recorder.write(output, metadata={})
    with np.load(output, allow_pickle=False) as trace:
        assert trace["policy_current_input"].shape == (1, 1, 83)
        assert trace["policy_k8_input"].shape == (1, 1, 587)
        assert trace["policy_representation"].shape == (1, 1, 179)
        assert trace["policy_k8_input"].dtype == np.float16
        metadata = json.loads(str(trace["metadata_json"].item()))
        assert metadata["schema"] == "policy_step_trace/v2-k8-observability"
        assert metadata["policy_inputs"]["dimensions"] == {
            "policy_current_input": 83,
            "policy_k8_input": 587,
            "policy_representation": 179,
        }


def test_k8_trace_fails_closed_when_any_policy_input_is_missing():
    recorder = PolicyStepTraceRecorder(
        1, 2, expected_delay_steps=1, include_policy_inputs=True
    )
    row = _batch(0, episode_step=0)
    row.update(
        policy_current_input=torch.zeros(1, 83),
        policy_k8_input=torch.zeros(1, 587),
    )
    with pytest.raises(ValueError, match="missing policy inputs"):
        recorder.record_tensor_batch(**row)
