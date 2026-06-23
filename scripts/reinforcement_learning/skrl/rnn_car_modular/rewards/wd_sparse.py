"""WD sparse reward module -- wraps the canonical compute_wd_charge_reward.

This is a thin facade that delegates to the existing implementation in
rnn_car_wdclean/rewards.py to avoid logic duplication and drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure the wdclean package is importable
_skrl_root = Path(__file__).resolve().parent.parent.parent
if str(_skrl_root) not in sys.path:
    sys.path.insert(0, str(_skrl_root))

from rnn_car_wdclean.rewards import compute_wd_charge_reward  # noqa: E402


class WDSparseReward:
    """Warp Drive style sparse reward for charge navigation.

    Breakdown keys (unchanged):
        goal_reward, wall_hit_reward, obs_hit_reward, floor_reward,
        action_reward, goal_reached, wall_collision, obs_collision, other_death
    """

    name: str = "wd_sparse"

    def __init__(
        self,
        penalty_hit: float = -5.0,
        reward_get_goal: float = 40.0,
        cost_operate: float = 0.03,
        penalty_timeout: float = 0.0,
        rl_fps: float = 5.0,
        cost_turn_rate: float = 0.5,
        penalty_smoothness: float = 0.0,
        penalty_speed_near_obs: float = 0.0,
        near_obs_d_react: float = 1.2,
        near_obs_d_stop: float = 0.45,
    ) -> None:
        self.penalty_hit = penalty_hit
        self.reward_get_goal = reward_get_goal
        self.cost_operate = cost_operate
        self.penalty_timeout = penalty_timeout
        self.rl_fps = rl_fps
        self.cost_turn_rate = cost_turn_rate
        self.penalty_smoothness = penalty_smoothness
        # v3f-react: clearance-gated 減速懲罰（抗動態障礙晚反應碰撞）
        self.penalty_speed_near_obs = penalty_speed_near_obs
        self.near_obs_d_react = near_obs_d_react
        self.near_obs_d_stop = near_obs_d_stop

    def update_params(self, curriculum_info: dict) -> None:
        """Sync reward params from curriculum phase config.

        Expected keys (matching Phase Config flat schema):
            spot_penalty_hit, spot_reward_get_goal, spot_cost_operate,
            spot_penalty_timeout, spot_penalty_smoothness
        """
        if "spot_penalty_hit" in curriculum_info:
            self.penalty_hit = curriculum_info["spot_penalty_hit"]
        if "spot_reward_get_goal" in curriculum_info:
            self.reward_get_goal = curriculum_info["spot_reward_get_goal"]
        if "spot_cost_operate" in curriculum_info:
            self.cost_operate = curriculum_info["spot_cost_operate"]
        if "spot_penalty_timeout" in curriculum_info:
            self.penalty_timeout = curriculum_info["spot_penalty_timeout"]
        if "spot_penalty_smoothness" in curriculum_info:
            self.penalty_smoothness = curriculum_info["spot_penalty_smoothness"]
        if "spot_penalty_speed_near_obs" in curriculum_info:
            self.penalty_speed_near_obs = curriculum_info["spot_penalty_speed_near_obs"]

    def compute(
        self,
        env_unwrapped: object,
        actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        context: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute WD sparse reward by delegating to canonical implementation.

        context (optional):
            - prev_actions: torch.Tensor [N, 2] — previous step's actions, used for
              frame-to-frame smoothness penalty (v3, anti-jitter)
        """
        prev_actions = None
        near_obs_dist_m = None
        v_forward_m = None
        if context is not None:
            prev_actions = context.get("prev_actions")
            near_obs_dist_m = context.get("near_obs_dist_m")
            v_forward_m = context.get("v_forward_m")
        return compute_wd_charge_reward(
            env_unwrapped=env_unwrapped,
            actions=actions,
            terminated=terminated,
            truncated=truncated,
            penalty_hit=self.penalty_hit,
            reward_get_goal=self.reward_get_goal,
            cost_operate=self.cost_operate,
            penalty_timeout=self.penalty_timeout,
            rl_fps=self.rl_fps,
            cost_turn_rate=self.cost_turn_rate,
            penalty_smoothness=self.penalty_smoothness,
            prev_actions=prev_actions,
            penalty_speed_near_obs=self.penalty_speed_near_obs,
            near_obs_dist_m=near_obs_dist_m,
            v_forward_m=v_forward_m,
            near_obs_d_react=self.near_obs_d_react,
            near_obs_d_stop=self.near_obs_d_stop,
        )
