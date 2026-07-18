from __future__ import annotations

import sys
from pathlib import Path

import torch


SKRL_ROOT = Path(__file__).resolve().parent.parent
for path in (SKRL_ROOT, SKRL_ROOT / "models"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modular_rnn_models import (  # noqa: E402
    LidarStateExtractor,
    PolicyHead,
    adapt_lidar_frame_stack_state_dict,
)
from rnn_car_modular.rewards.clean_progress import CleanProgressReward  # noqa: E402


class _TerminationManager:
    def __init__(self, terms: dict[str, torch.Tensor]):
        self._terms = terms
        self._term_names = list(terms)

    def get_term(self, name: str) -> torch.Tensor:
        return self._terms[name]


class _Env:
    def __init__(self, terms: dict[str, torch.Tensor]):
        self.termination_manager = _TerminationManager(terms)


def _obs_with_goal_distance(*distances: float) -> torch.Tensor:
    obs = torch.zeros(len(distances), 79)
    obs[:, 4] = torch.tensor(distances)
    return obs


def test_clean_reward_progress_and_terminal_semantics() -> None:
    reward_fn = CleanProgressReward()
    env = _Env({
        "goal_reached": torch.tensor([False, True, False]),
        "wall_collision": torch.tensor([False, False, True]),
    })
    obs = _obs_with_goal_distance(5.0, 1.0, 2.0)
    next_obs = _obs_with_goal_distance(4.5, 8.0, 9.0)  # terminal rows are reset observations
    actions = torch.tensor([[9, 9], [9, 9], [9, 9]])
    previous = actions.clone()
    terminated = torch.tensor([[False], [True], [True]])
    truncated = torch.zeros(3, 1, dtype=torch.bool)

    total, terms = reward_fn.compute(
        env, actions, terminated, truncated,
        context={"obs": obs, "next_obs": next_obs, "prev_actions": previous},
    )

    torch.testing.assert_close(total[0], torch.tensor(0.49))
    torch.testing.assert_close(total[1], torch.tensor(9.99))
    torch.testing.assert_close(total[2], torch.tensor(-15.01))
    assert terms["progress_reward"][1:].eq(0).all()
    assert terms["goal_reached"].tolist() == [False, True, False]
    assert terms["wall_collision"].tolist() == [False, False, True]


def test_clean_reward_action_delta_is_small_and_normalized() -> None:
    reward_fn = CleanProgressReward()
    env = _Env({"goal_reached": torch.tensor([False])})
    obs = _obs_with_goal_distance(3.0)
    total, terms = reward_fn.compute(
        env,
        torch.tensor([[18, 0]]),
        torch.tensor([[False]]),
        torch.tensor([[False]]),
        context={
            "obs": obs,
            "next_obs": obs.clone(),
            "prev_actions": torch.tensor([[0, 18]]),
        },
    )
    torch.testing.assert_close(terms["smoothness_reward"], torch.tensor([-0.015]))
    torch.testing.assert_close(total, torch.tensor([-0.025]))


def test_anti_spin_penalizes_moving_spin_only_after_same_sign_yaw_grace() -> None:
    reward_fn = CleanProgressReward(
        anti_spin_weight=0.15,
        anti_spin_grace_steps=5,
        anti_spin_ramp_steps=1,
    )
    env = _Env({"goal_reached": torch.tensor([False])})
    obs = _obs_with_goal_distance(3.0)
    common = {
        "obs": obs,
        "next_obs": obs.clone(),
        "prev_actions": torch.tensor([[9, 9]]),
        "near_obs_dist_m": torch.tensor([1.0]),
        "v_forward_actual_m": torch.tensor([0.6]),
        "omega_actual_rad_s": torch.tensor([1.0]),
    }
    actions = torch.tensor([[9, 9]])
    not_done = torch.tensor([[False]])

    # 15 steps at 1 rad/s and dt=0.2 is 171.9 degrees: a normal avoidance
    # arc remains below the 180-degree pathology threshold.
    for expected_run in range(1, 16):
        _, terms = reward_fn.compute(env, actions, not_done, not_done, context=common)
        torch.testing.assert_close(terms["anti_spin"], torch.tensor([0.0]))
        torch.testing.assert_close(
            terms["anti_spin_run_steps"], torch.tensor([float(expected_run)])
        )

    # The moving turn crosses 180 degrees on step 16 and is now penalized.
    total, terms = reward_fn.compute(env, actions, not_done, not_done, context=common)
    assert terms["anti_spin"].item() < 0.0
    assert total.item() < -0.01
    assert terms["anti_spin_same_sign_yaw_deg"].item() > 180.0

    # Reversing turn direction starts a fresh yaw accumulator and is not penalized.
    reversed_turn = dict(common, omega_actual_rad_s=torch.tensor([-1.0]))
    _, terms = reward_fn.compute(env, actions, not_done, not_done, context=reversed_turn)
    torch.testing.assert_close(terms["anti_spin"], torch.tensor([0.0]))
    assert terms["anti_spin_same_sign_yaw_deg"].item() < 12.0

    # Stopping clears all persistence state.
    stopped = dict(common, omega_actual_rad_s=torch.tensor([0.0]))
    _, terms = reward_fn.compute(env, actions, not_done, not_done, context=stopped)
    torch.testing.assert_close(terms["anti_spin_run_steps"], torch.tensor([0.0]))
    torch.testing.assert_close(terms["anti_spin_same_sign_yaw_deg"], torch.tensor([0.0]))


def test_policy_loss_reaches_frame_stack_encoder() -> None:
    torch.manual_seed(7)
    encoder = LidarStateExtractor(include_act_hist=False, frame_stack=4)
    policy = PolicyHead(input_dim=79 + encoder.output_dim)
    encoder_input = torch.randn(16, 79 + 3 * 72)

    features = encoder(encoder_input)
    logits = policy(torch.cat([encoder_input[:, :79], features], dim=-1))
    loss = logits.square().mean()
    loss.backward()

    grad_norm = torch.sqrt(sum(
        parameter.grad.square().sum()
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ))
    assert grad_norm.item() > 0.0


def test_frame_stack_expansion_preserves_existing_four_frame_features() -> None:
    torch.manual_seed(11)
    source = LidarStateExtractor(include_act_hist=False, frame_stack=4).eval()
    target = LidarStateExtractor(include_act_hist=False, frame_stack=8).eval()
    migrated, stack_change = adapt_lidar_frame_stack_state_dict(target, source.state_dict())
    target.load_state_dict(migrated)
    assert stack_change == (4, 8)

    current = torch.randn(8, 79)
    history_3 = torch.randn(8, 3 * 72)
    extra_history_4 = torch.randn(8, 4 * 72)
    source_input = torch.cat([current, history_3], dim=1)
    target_input = torch.cat([current, history_3, extra_history_4], dim=1)
    torch.testing.assert_close(source(source_input), target(target_input), rtol=0.0, atol=0.0)


def test_future_occupancy_prefers_passing_behind_crossing_obstacle() -> None:
    reward_fn = CleanProgressReward(
        future_occupancy_weight=0.1,
        future_occupancy_horizon_s=1.5,
        future_occupancy_samples=8,
        future_occupancy_safe_distance_m=1.2,
    )
    env = _Env({"goal_reached": torch.tensor([False])})
    obs = _obs_with_goal_distance(5.0)
    base_context = {
        "obs": obs,
        "next_obs": obs.clone(),
        "prev_actions": torch.tensor([[9, 9]]),
        "v_forward_actual_m": torch.tensor([0.8]),
        "dynamic_obstacle_positions_body_m": torch.tensor([[[2.0, -1.0]]]),
        "dynamic_obstacle_velocities_body_mps": torch.tensor([[[0.0, 1.0]]]),
    }
    not_done = torch.tensor([[False]])
    _, left_terms = reward_fn.compute(
        env,
        torch.tensor([[9, 9]]),
        not_done,
        not_done,
        context=dict(base_context, omega_actual_rad_s=torch.tensor([0.8])),
    )
    _, right_terms = reward_fn.compute(
        env,
        torch.tensor([[9, 9]]),
        not_done,
        not_done,
        context=dict(base_context, omega_actual_rad_s=torch.tensor([-0.8])),
    )
    assert left_terms["future_occupancy"].item() < right_terms["future_occupancy"].item()
    assert left_terms["future_occupancy_risk"].item() > 0.0

    _, static_terms = reward_fn.compute(
        env,
        torch.tensor([[9, 9]]),
        not_done,
        not_done,
        context=dict(
            base_context,
            omega_actual_rad_s=torch.tensor([0.8]),
            dynamic_obstacle_velocities_body_mps=torch.zeros(1, 1, 2),
        ),
    )
    torch.testing.assert_close(static_terms["future_occupancy"], torch.tensor([0.0]))
