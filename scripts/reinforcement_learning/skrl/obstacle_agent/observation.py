"""Obstacle agent 觀測建構。

每個 obstacle 看到 9D 觀測 (env local frame):
  [0:2]  own_local_xy   — 自己在 env 中的局部座標
  [2:4]  own_vel        — 自己的速度 (vx, vy)
  [4:6]  robot_rel_xy   — robot 相對於自己的位置 (robot_pos - obs_pos)
  [6:8]  robot_vel      — robot 的速度 (vx, vy)
  [8]    d_wall         — 到最近 env 邊界的距離

所有 obstacle 共享同一個 policy (parameter sharing)，
所以觀測是 self-centric (以自己為中心)。
"""

import torch

from .config import OBS_DIM


def build_obstacle_obs(env_unwrapped, max_obstacles: int, device) -> torch.Tensor:
    """讀取 obstacle + robot state，建構 [num_envs, N, 9] obs tensor。

    Args:
        env_unwrapped: Isaac Lab env (unwrapped)
        max_obstacles: 最大 obstacle 數量 (N)
        device: torch device

    Returns:
        [num_envs, N, 9] tensor。Inactive obstacles (Z=-10) 填 0。
    """
    num_envs = env_unwrapped.num_envs
    env_origins = env_unwrapped.scene.env_origins  # [E, 3]

    # Robot state
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :3]  # [E, 3]
    robot_local = robot_pos_w - env_origins  # [E, 3]
    robot_vel_w = env_unwrapped.scene["robot"].data.root_lin_vel_w[:, :2]  # [E, 2]

    # Velocity cache (由 apply_obstacle_actions 更新)
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)

    obs_all = torch.zeros(num_envs, max_obstacles, OBS_DIM, device=device)

    # Scene entity cache (避免每步重新查找)
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    # Per-env scene bounds
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((num_envs,), 7.0, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # [E, 3]
        obs_local = obs_pos_w - env_origins  # [E, 3]

        # Active = 在場景中可見 (Z > 0)
        active = obs_pos_w[:, 2] > 0.0  # [E]

        obs_xy = obs_local[:, :2]  # [E, 2]
        obs_vel = env_unwrapped._obstacle_velocities[:, i, :]  # [E, 2]
        robot_rel = robot_local[:, :2] - obs_xy  # [E, 2]

        # d_wall: 到最近邊界的距離
        d_left = obs_xy[:, 0] + bound_limits
        d_right = bound_limits - obs_xy[:, 0]
        d_bottom = obs_xy[:, 1] + bound_limits
        d_top = bound_limits - obs_xy[:, 1]
        d_wall = torch.stack([d_left, d_right, d_bottom, d_top], dim=-1).min(dim=-1).values
        d_wall = torch.max(torch.min(d_wall, bound_limits), torch.zeros_like(d_wall))

        obs_all[:, i, 0:2] = obs_xy
        obs_all[:, i, 2:4] = obs_vel
        obs_all[:, i, 4:6] = robot_rel
        obs_all[:, i, 6:8] = robot_vel_w
        obs_all[:, i, 8] = d_wall

        # Inactive → zero
        inactive = ~active
        if inactive.any():
            obs_all[inactive, i, :] = 0.0

    return obs_all  # [num_envs, N, 9]
