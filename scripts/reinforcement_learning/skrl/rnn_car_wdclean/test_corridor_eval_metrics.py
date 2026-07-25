from __future__ import annotations

import pytest
import torch

from corridor_eval_metrics import (
    corridor_clear_mask,
    summarize_corridor_actions,
)


def test_corridor_clear_mask_uses_goal_and_front_cone() -> None:
    obs = torch.full((3, 83), 0.20)
    obs[:, 4:6] = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    obs[1, 6 + 36] = 0.10

    result = corridor_clear_mask(obs)

    assert result.tolist() == [True, False, False]


def test_corridor_action_summary_reports_turn_and_stop_behavior() -> None:
    actions = torch.tensor(
        [
            [0.00, 0.00],
            [0.50, 0.40],
            [-0.10, 0.90],
            [0.05, -1.20],
        ]
    )

    result = summarize_corridor_actions(actions)

    assert result["action_frames"] == 4
    assert result["linear_speed_mean_mps"] == pytest.approx(0.1125)
    assert result["linear_speed_abs_mean_mps"] == pytest.approx(0.1625)
    assert result["angular_abs_mean_rad_s"] == pytest.approx(0.625)
    assert result["high_turn_fraction"] == pytest.approx(0.5)
    assert result["extreme_turn_fraction"] == pytest.approx(0.25)
    assert result["stop_command_fraction"] == pytest.approx(0.5)
    assert result["reverse_command_fraction"] == pytest.approx(0.25)
    assert result["low_speed_high_turn_fraction"] == pytest.approx(0.5)


def test_corridor_action_summary_ignores_nonfinite_frames() -> None:
    result = summarize_corridor_actions(
        torch.tensor([[float("nan"), 0.0], [0.5, 0.4]])
    )
    assert result["action_frames"] == 1
    assert result["linear_speed_mean_mps"] == pytest.approx(0.5)


def test_corridor_action_summary_handles_empty_input() -> None:
    result = summarize_corridor_actions(torch.empty(0, 2))
    assert result["action_frames"] == 0
    assert result["angular_abs_p95_rad_s"] is None


def test_corridor_action_summary_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="actions must have shape"):
        summarize_corridor_actions(torch.zeros(4))


def test_corridor_clear_mask_rejects_short_observation() -> None:
    with pytest.raises(ValueError, match="policy_obs must have shape"):
        corridor_clear_mask(torch.zeros(4, 48))
