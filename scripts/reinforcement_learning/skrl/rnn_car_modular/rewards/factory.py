"""Reward module factory -- create RewardModule from profile name."""

from __future__ import annotations

from rnn_car_modular.rewards.base import RewardModule
from rnn_car_modular.rewards.clean_progress import CleanProgressReward
from rnn_car_modular.rewards.wd_sparse import WDSparseReward


def create_reward_module(
    reward_profile: str,
    penalty_hit: float = -5.0,
    reward_get_goal: float = 40.0,
    cost_operate: float = 0.03,
    anti_spin_weight: float = 0.0,
    anti_spin_hazard_distance: float = 1.5,
    anti_spin_omega_threshold: float = 0.8,
    anti_spin_progress_threshold: float = 0.02,
    anti_spin_grace_steps: int = 5,
    anti_spin_ramp_steps: int = 5,
    anti_spin_yaw_grace_deg: float = 180.0,
    anti_spin_yaw_ramp_deg: float = 180.0,
    anti_spin_dt: float = 0.2,
    future_occupancy_weight: float = 0.0,
    future_occupancy_horizon_s: float = 1.5,
    future_occupancy_samples: int = 8,
    future_occupancy_safe_distance_m: float = 1.0,
    future_occupancy_near_distance_m: float = 3.0,
    future_occupancy_move_threshold_mps: float = 0.1,
) -> RewardModule:
    """Create a reward module from profile name.

    Args:
        reward_profile: One of "wd_sparse", "navrl_dense_v8", ...
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

    if reward_profile == "clean_progress":
        return CleanProgressReward(
            anti_spin_weight=anti_spin_weight,
            anti_spin_hazard_distance=anti_spin_hazard_distance,
            anti_spin_omega_threshold=anti_spin_omega_threshold,
            anti_spin_progress_threshold=anti_spin_progress_threshold,
            anti_spin_grace_steps=anti_spin_grace_steps,
            anti_spin_ramp_steps=anti_spin_ramp_steps,
            anti_spin_yaw_grace_deg=anti_spin_yaw_grace_deg,
            anti_spin_yaw_ramp_deg=anti_spin_yaw_ramp_deg,
            anti_spin_dt=anti_spin_dt,
            future_occupancy_weight=future_occupancy_weight,
            future_occupancy_horizon_s=future_occupancy_horizon_s,
            future_occupancy_samples=future_occupancy_samples,
            future_occupancy_safe_distance_m=future_occupancy_safe_distance_m,
            future_occupancy_near_distance_m=future_occupancy_near_distance_m,
            future_occupancy_move_threshold_mps=future_occupancy_move_threshold_mps,
        )

    if reward_profile == "navrl_dense_v8":
        from rnn_car_modular.rewards.navrl_dense_v8 import NavRLDenseV8Reward
        return NavRLDenseV8Reward()

    raise NotImplementedError(
        f"reward_profile={reward_profile!r} is not yet implemented. "
        f"Available: wd_sparse, clean_progress, navrl_dense_v8. "
        f"Future: hybrid_progress, ttc_risk."
    )
