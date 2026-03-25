"""Potential-Based Reward Shaping for Phase 0

Implements principled reward functions based on:
  1. Potential-Based Shaping: R = Phi(s') - Phi(s), guarantees policy invariance
  2. Decoupled collision penalty: warning zone [min_dist, warn_dist] separate from termination
  3. Terminal reward dominance: reaching_goal=+100, collision=-100

Sign convention:
  All functions return NON-NEGATIVE values (or values with clear physical meaning).
  The SIGN is controlled by the weight in RewardsCfgPhase0.
  IsaacLab RewardManager computes: reward = func(...) * weight * dt

References:
  - Ng et al. (1999) "Policy invariance under reward transformations"
  - NavRL (Xu et al., 2025) for log-distance safety
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_lidar_min_distance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Helper: compute per-env minimum LiDAR 2D distance.

    Uses the same 2D projection as collision_occurred / collision_penalty
    for consistency across the reward system.

    Returns:
        [num_envs] minimum 2D distance to nearest obstacle per env.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_pos_2d = sensor.data.pos_w[:, :2]  # [num_envs, 2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]  # [num_envs, num_rays, 2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance,
    )
    return torch.min(distances_2d, dim=1)[0]  # [num_envs]


def potential_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Potential-Based Reward Shaping for goal progress.

    Math:
        Phi(s) = -d(robot, goal)
        R(t) = Phi(s_{t+1}) - Phi(s_t)
             = -d_{t+1} + d_t
             = d_t - d_{t+1}          (positive when approaching)

    Properties:
        - Telescoping sum: sum(R) = Phi(s_final) - Phi(s_0) = d_0 - d_final
        - Bounded total: max = d_0 (initial distance, typically 5-14m)
        - Uses env._local_goal_world (fixes CRITICAL-2 target split)
        - Stores prev distance in env._prev_goal_dist per env

    Returns:
        [num_envs] -- positive when approaching goal, negative when retreating.
        Typical range per step: [-0.03, +0.03] at 50Hz with max 1.5m/s.
        Weight should be POSITIVE (e.g., +12.0).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device

    # Robot position (2D)
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    # Goal position -- unified source (fixes CRITICAL-2)
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world  # [num_envs, 2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # Current distance to goal
    d_curr = torch.norm(goal_pos - robot_pos, dim=1)  # [num_envs]

    # Initialize prev distance on first call or after episode reset
    if not hasattr(env, "_prev_goal_dist") or env._prev_goal_dist is None:
        env._prev_goal_dist = d_curr.clone()
        return torch.zeros(env.num_envs, device=device)

    # Detect episode resets: episode_length_buf == 0 means just reset
    # After reset, prev distance is stale -- reinitialize to current
    just_reset = env.episode_length_buf == 0
    env._prev_goal_dist[just_reset] = d_curr[just_reset]

    d_prev = env._prev_goal_dist

    # Potential-based shaping: R = d_prev - d_curr
    reward = d_prev - d_curr  # [num_envs]

    # Update stored distance for next step
    env._prev_goal_dist = d_curr.clone()

    # Safety: clamp to prevent physics explosions
    reward = reward.clamp(-1.0, 1.0)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    return reward


def smooth_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    warn_dist: float = 0.8,
    min_dist: float = 0.3,
) -> torch.Tensor:
    """Smooth repulsive collision penalty (decoupled from termination).

    Math:
        d_min = min(LiDAR readings) per env
        if d_min >= warn_dist:  penalty = 0
        elif d_min <= min_dist: penalty = 1.0 (maximum)
        else: penalty = ((warn_dist - d_min) / (warn_dist - min_dist))^2

    Properties:
        - Warning zone: [min_dist, warn_dist] = [0.3m, 0.8m]
        - Smooth quadratic ramp (no discontinuous jump)
        - Returns [0, 1] -- always non-negative
        - Weight should be NEGATIVE in config (e.g., -3.0)
        - Decoupled: termination at 0.3m, penalty starts at 0.8m

    Returns:
        [num_envs] in [0, 1].
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]

    # Quadratic ramp in warning zone
    range_size = warn_dist - min_dist  # 0.5m
    normalized = ((warn_dist - d_min) / (range_size + 1e-6)).clamp(0.0, 1.0)
    penalty = normalized ** 2  # [num_envs], in [0, 1]

    return penalty


def collision_terminal_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """Binary collision terminal penalty (one-shot).

    Returns 1.0 when any LiDAR reading <= threshold, else 0.0.
    Weight should be NEGATIVE (e.g., -100.0).

    This is aligned with the termination threshold (0.3m) so that
    the terminal penalty fires on the same step as episode termination.

    Returns:
        [num_envs] -- 0.0 or 1.0.
    """
    d_min = _get_lidar_min_distance(env, sensor_cfg)  # [num_envs]
    return (d_min <= threshold).float()


def per_step_time_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Constant per-step time penalty to encourage efficiency.

    Returns 1.0 every step. Weight should be negative (e.g., -0.03).
    At 50Hz x 20s = 1000 steps, total = -0.03 * 1000 * dt ~ -0.6.

    Returns:
        [num_envs] -- always 1.0.
    """
    return torch.ones(env.num_envs, device=env.device)


__all__ = [
    "potential_progress_reward",
    "smooth_collision_penalty",
    "collision_terminal_penalty",
    "per_step_time_penalty",
]
