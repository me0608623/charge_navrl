"""Expected Value Reward Functions

Based on the Expected Value philosophy for autonomous navigation:
  - Pull Force: risk_aware_progress  = v_proj * (1 - exp(-alpha * TTC_min))
  - Push Force: ttc_defensive_penalty = -(tau - TTC_min) if TTC_min < tau else 0

Design principles:
  1. Only TTC_min (most dangerous obstacle) matters, NOT summed across all
  2. Minimal parameters: one tuning knob per function (alpha or tau)
  3. Pure vectorized PyTorch, zero for-loops over environments
  4. Compatible with Asymmetric Actor-Critic (privileged obstacle info in Critic)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term
from .ttc_defensive_driving import compute_ttc


def _compute_ttc_min(
    env: ManagerBasedRLEnv,
    robot_pos: torch.Tensor,
    robot_vel: torch.Tensor,
    default_ttc: float = 100.0,
) -> torch.Tensor:
    """Compute TTC_min: minimum TTC across all visible obstacles per env.

    Args:
        env: Environment instance.
        robot_pos: Robot 2D position [num_envs, 2].
        robot_vel: Robot 2D velocity [num_envs, 2].
        default_ttc: Default TTC when no dangerous obstacle exists.

    Returns:
        TTC_min tensor [num_envs]. Large value means safe.
    """
    num_envs = env.num_envs
    device = env.device
    ttc_min = torch.full((num_envs,), default_ttc, device=device)

    if not hasattr(env, '_dynamic_obstacle_pos') or env._dynamic_obstacle_pos is None:
        return ttc_min

    obs_pos = env._dynamic_obstacle_pos   # [num_envs, N, 2]
    obs_vel = env._dynamic_obstacle_vel   # [num_envs, N, 2]

    if obs_pos.shape[1] == 0:
        return ttc_min

    # Visibility mask: obstacles with z > 0 are above ground (not hidden)
    if hasattr(env, '_dynamic_obstacle_pos_w'):
        visible = env._dynamic_obstacle_pos_w[:, :, 2] > 0  # [num_envs, N]
    else:
        visible = torch.ones(
            num_envs, obs_pos.shape[1], device=device, dtype=torch.bool
        )

    # Compute TTC for all obstacle pairs: [num_envs, N]
    ttc_all = compute_ttc(robot_pos, robot_vel, obs_pos, obs_vel)

    # Filter: only approaching (finite positive TTC) + visible obstacles
    valid = (ttc_all > 0) & torch.isfinite(ttc_all) & visible
    ttc_filtered = torch.where(valid, ttc_all, torch.full_like(ttc_all, default_ttc))

    # TTC_min: most dangerous obstacle per env
    ttc_min, _ = ttc_filtered.min(dim=1)  # [num_envs]

    return ttc_min


def risk_aware_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 1.0,
) -> torch.Tensor:
    """Risk-Aware Progress Reward (Pull Force).

    R_prog = v_proj * (1 - exp(-alpha * TTC_min))

    Physical meaning:
      - v_proj: velocity projection onto goal direction (positive = toward goal)
      - (1 - exp(-alpha * TTC_min)): risk modulation factor in [0, 1)
        - TTC_min large (safe): factor -> 1, full speed rewarded
        - TTC_min small (danger): factor -> 0, reward suppressed

    This solves the "Freezing Robot" problem by coupling progress with safety.

    Args:
        env: Environment instance.
        robot_cfg: Robot scene entity config.
        alpha: Risk sensitivity coefficient. Larger = faster decay near danger.

    Returns:
        Reward tensor [num_envs]. Can be negative if moving away from goal.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    # Robot state
    robot_pos = robot.data.root_pos_w[:, :2]      # [num_envs, 2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]   # [num_envs, 2]

    # Goal position
    if hasattr(env, '_local_goal_world') and env._local_goal_world is not None:
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # Goal direction (unit vector)
    goal_dir = goal_pos - robot_pos                           # [num_envs, 2]
    goal_dist = torch.norm(goal_dir, dim=1, keepdim=True)     # [num_envs, 1]
    goal_unit = goal_dir / (goal_dist + 1e-6)                 # [num_envs, 2]

    # v_proj: velocity projection onto goal direction
    # Clamp to [-max_v, max_v] to guard against physics explosions
    max_v = 2.0  # Charge robot max ~1.5 m/s, allow slight overshoot
    v_proj = (robot_vel * goal_unit).sum(dim=1).clamp(-max_v, max_v)  # [num_envs]

    # TTC_min
    ttc_min = _compute_ttc_min(env, robot_pos, robot_vel)  # [num_envs]

    # Risk modulation: (1 - exp(-alpha * TTC_min))
    risk_factor = 1.0 - torch.exp(-alpha * ttc_min)  # [num_envs], in [0, 1)

    # Final reward: R in [-max_v, max_v) since risk_factor in [0, 1)
    reward = v_proj * risk_factor  # [num_envs]

    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    return reward


def ttc_defensive_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tau: float = 1.5,
) -> torch.Tensor:
    """TTC Defensive Driving Penalty (Push Force).

    R_ttc = -(tau - TTC_min)   if TTC_min < tau
          = 0                  otherwise

    Only the MOST dangerous obstacle (TTC_min) contributes.
    This avoids over-penalizing in crowded scenes (no sum over all obstacles).

    Physical meaning:
      - tau: safety time horizon (seconds). TTC below this triggers penalty.
      - Penalty is linear: closer collision time = stronger penalty.
      - Maximum penalty = -tau (when TTC_min = 0, i.e., imminent collision).

    Args:
        env: Environment instance.
        robot_cfg: Robot scene entity config.
        tau: Safety time threshold (seconds).

    Returns:
        Penalty tensor [num_envs], in [-tau, 0].
    """
    robot: Articulation = env.scene[robot_cfg.name]

    robot_pos = robot.data.root_pos_w[:, :2]      # [num_envs, 2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]   # [num_envs, 2]

    # TTC_min
    ttc_min = _compute_ttc_min(env, robot_pos, robot_vel, default_ttc=tau + 1.0)

    # Penalty: -(tau - TTC_min) when TTC_min < tau, else 0
    penalty = torch.where(
        ttc_min < tau,
        -(tau - ttc_min),
        torch.zeros_like(ttc_min),
    )  # [num_envs], in [-tau, 0]

    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-tau)
    penalty = _check_reward_term("ttc_defensive_penalty", penalty, env, raise_on_error=True)

    return penalty


__all__ = [
    "risk_aware_progress",
    "ttc_defensive_penalty",
]
