"""Privileged observation extraction for Asymmetric Critic.

Extracts 50D privileged information from the running environment:
  - Obstacles (50D): 10 × 5D [x, y, dir, v, size] in body-frame

Removed (v2 2026-05-27):
  - Walls 40D: Actor 360° LiDAR 已覆蓋大部分靜態牆壁，且背後/未掃描牆壁
    會造成 gradient confusion（Actor 無法推斷 → Critic 評分不一致）。
  - Goal Queue 6D: 目前 MultiGoalCommand 到達最近 1 個就結束 episode，
    第 2-3 近 goal 不影響 outcome。

All operations are GPU-vectorized (no Python loops over envs).
"""

from __future__ import annotations

import torch

# Fixed scale factors (no running normalization needed)
_OBSTACLE_XY_SCALE = 1.0 / 10.0   # max_distance = 10m
_OBSTACLE_V_SCALE = 1.0 / 2.0     # max obstacle speed ~2 m/s
_OBSTACLE_SIZE_SCALE = 1.0 / 1.5  # max obstacle radius ~1.5m

PRIVILEGED_OBS_DIM = 50  # obstacles only
MAX_OBSTACLES = 10


def extract_privileged_obs(env_unwrapped, device: torch.device | None = None) -> torch.Tensor:
    """Extract 50D privileged obs from running env (obstacles only).

    Args:
        env_unwrapped: The unwrapped Isaac Lab env (env.unwrapped).
        device: Target device. If None, uses env's device.

    Returns:
        Tensor [num_envs, 50] — normalized privileged observations.
    """
    if device is None:
        device = env_unwrapped.device
    num_envs = env_unwrapped.num_envs

    robot = env_unwrapped.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # [E, 3]
    robot_quat_w = robot.data.root_quat_w       # [E, 4]

    return _extract_obstacles(env_unwrapped, robot_pos_w, robot_quat_w, num_envs, device)


def _extract_obstacles(
    env, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor,
    num_envs: int, device: torch.device,
) -> torch.Tensor:
    """Extract 50D obstacle privileged obs (10 × 5D body-frame)."""
    from isaaclab.utils import math as math_utils

    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)
    cos_yaw = torch.cos(-robot_yaw)  # [E]
    sin_yaw = torch.sin(-robot_yaw)  # [E]

    result = torch.zeros(num_envs, MAX_OBSTACLES, 5, device=device)

    obstacle_sizes = getattr(env, "_obstacle_sizes", None)
    obstacle_velocities = getattr(env, "_obstacle_velocities", None)

    for i in range(MAX_OBSTACLES):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obstacle = env.scene[obs_name]
        obs_pos_w = torch.nan_to_num(obstacle.data.root_pos_w[:, :3], nan=0.0)

        # Hidden check (Z < 0 = underground)
        hidden = obs_pos_w[:, 2] < 0.0  # [E]

        # Relative position in body-frame
        dx = obs_pos_w[:, 0] - robot_pos_w[:, 0]
        dy = obs_pos_w[:, 1] - robot_pos_w[:, 1]
        rel_x = cos_yaw * dx + sin_yaw * dy
        rel_y = -sin_yaw * dx + cos_yaw * dy

        # Velocity
        if obstacle_velocities is not None and obstacle_velocities.shape[1] > i:
            vel_w = obstacle_velocities[:, i, :]  # [E, 2]
            speed = torch.linalg.norm(vel_w, dim=1)
            vel_yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])
            rel_dir = vel_yaw - robot_yaw
        else:
            speed = torch.zeros(num_envs, device=device)
            obs_quat = obstacle.data.root_quat_w
            _, _, obs_yaw = math_utils.euler_xyz_from_quat(obs_quat)
            rel_dir = obs_yaw - robot_yaw

        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        # Size
        if obstacle_sizes is not None and i < len(obstacle_sizes):
            size_val = float(obstacle_sizes[i])
        else:
            size_val = 0.0

        # Write (zeroed for hidden obstacles)
        result[:, i, 0] = torch.where(hidden, torch.zeros_like(rel_x), rel_x * _OBSTACLE_XY_SCALE)
        result[:, i, 1] = torch.where(hidden, torch.zeros_like(rel_y), rel_y * _OBSTACLE_XY_SCALE)
        result[:, i, 2] = torch.where(hidden, torch.zeros_like(rel_dir), rel_dir / torch.pi)
        result[:, i, 3] = torch.where(hidden, torch.zeros_like(speed), speed * _OBSTACLE_V_SCALE)
        result[:, i, 4] = torch.where(hidden, torch.zeros_like(speed), torch.full_like(speed, size_val * _OBSTACLE_SIZE_SCALE))

    return result.reshape(num_envs, MAX_OBSTACLES * 5)  # [E, 50]
