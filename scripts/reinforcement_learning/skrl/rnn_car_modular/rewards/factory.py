"""Reward module factory -- create RewardModule from profile name."""

from __future__ import annotations

from rnn_car_modular.rewards.base import RewardModule
from rnn_car_modular.rewards.wd_sparse import WDSparseReward


def create_reward_module(
    reward_profile: str,
    penalty_hit: float = -5.0,
    reward_get_goal: float = 40.0,
    cost_operate: float = 0.03,
) -> RewardModule:
    """Create a reward module from profile name.

    Args:
        reward_profile: One of "wd_sparse", "navrl_dense", "hybrid_progress", "ttc_risk".
        penalty_hit: Initial collision penalty (synced from curriculum later).
        reward_get_goal: Initial goal reward.
        cost_operate: Initial action cost.

    Returns:
        A RewardModule instance.

    Raises:
        NotImplementedError: If profile is not yet implemented.
    """
    if reward_profile == "wd_sparse":
        return WDSparseReward(
            penalty_hit=penalty_hit,
            reward_get_goal=reward_get_goal,
            cost_operate=cost_operate,
        )

    raise NotImplementedError(
        f"reward_profile={reward_profile!r} is not yet implemented. "
        f"Available: wd_sparse. "
        f"Future: navrl_dense, hybrid_progress, ttc_risk."
    )
