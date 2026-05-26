"""Privileged observation extraction for Asymmetric Critic.

Extracts 96D privileged information from the running environment:
  - Obstacles (50D): 10 × 5D [x, y, dir, v, size] in body-frame
  - Walls (40D): 8 × 4D [cx, cy, sx, sy] body-frame + 8D mask
  - Goal queue (6D): 3 × 2D [gx, gy] body-frame

All operations are GPU-vectorized (no Python loops over envs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Fixed scale factors (no running normalization needed)
_OBSTACLE_XY_SCALE = 1.0 / 10.0   # max_distance = 10m
_OBSTACLE_V_SCALE = 1.0 / 2.0     # max obstacle speed ~2 m/s
_OBSTACLE_SIZE_SCALE = 1.0 / 1.5  # max obstacle radius ~1.5m
_WALL_XY_SCALE = 1.0 / 12.0       # arena half-size ~12m
_WALL_SIZE_SCALE = 1.0 / 10.0     # max wall dimension ~10m
_GOAL_XY_SCALE = 1.0 / 12.0       # same as wall

PRIVILEGED_OBS_DIM = 96  # 50 + 40 + 6
MAX_OBSTACLES = 10
MAX_WALL_SLOTS = 8
MAX_GOAL_QUEUE = 3


def extract_privileged_obs(env_unwrapped, device: torch.device | None = None) -> torch.Tensor:
    """Extract 96D privileged obs from running env.

    Args:
        env_unwrapped: The unwrapped Isaac Lab env (env.unwrapped).
        device: Target device. If None, uses env's device.

    Returns:
        Tensor [num_envs, 96] — normalized privileged observations.
    """
    if device is None:
        device = env_unwrapped.device
    num_envs = env_unwrapped.num_envs

    # Robot state for body-frame transforms
    robot = env_unwrapped.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # [E, 3]
    robot_quat_w = robot.data.root_quat_w       # [E, 4]

    obs_part = _extract_obstacles(env_unwrapped, robot_pos_w, robot_quat_w, num_envs, device)
    wall_part = _extract_walls(env_unwrapped, robot_pos_w, robot_quat_w, num_envs, device)
    goal_part = _extract_goal_queue(env_unwrapped, robot_pos_w, robot_quat_w, num_envs, device)

    return torch.cat([obs_part, wall_part, goal_part], dim=-1)  # [E, 96]


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


def _extract_walls(
    env, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor,
    num_envs: int, device: torch.device,
) -> torch.Tensor:
    """Extract 40D wall privileged obs (8 × 4D body-frame + 8D mask)."""
    from isaaclab.utils import math as math_utils

    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)
    cos_yaw = torch.cos(-robot_yaw)  # [E]
    sin_yaw = torch.sin(-robot_yaw)  # [E]

    # Per-env wall data
    wall_centers = getattr(env, "_maze_wall_centers", None)  # [E, 8, 2]
    wall_sizes = getattr(env, "_maze_wall_sizes", None)      # [E, 8, 2]
    wall_mask = getattr(env, "_maze_wall_mask", None)        # [E, 8]

    if wall_centers is None:
        return torch.zeros(num_envs, 40, device=device)

    num_slots = min(wall_centers.shape[1], MAX_WALL_SLOTS)

    # Transform centers to body-frame
    dx = wall_centers[:, :num_slots, 0] - robot_pos_w[:, 0:1]  # [E, W]
    dy = wall_centers[:, :num_slots, 1] - robot_pos_w[:, 1:2]  # [E, W]
    rel_cx = cos_yaw.unsqueeze(1) * dx + sin_yaw.unsqueeze(1) * dy   # [E, W]
    rel_cy = -sin_yaw.unsqueeze(1) * dx + cos_yaw.unsqueeze(1) * dy  # [E, W]

    # Sizes (axis-aligned, don't need rotation — just scale)
    sx = wall_sizes[:, :num_slots, 0] * _WALL_SIZE_SCALE  # [E, W]
    sy = wall_sizes[:, :num_slots, 1] * _WALL_SIZE_SCALE  # [E, W]

    # Normalize positions
    rel_cx = rel_cx * _WALL_XY_SCALE
    rel_cy = rel_cy * _WALL_XY_SCALE

    # Apply mask (zero out hidden walls)
    if wall_mask is not None:
        mask_f = wall_mask[:, :num_slots].float()  # [E, W] 1.0=visible
    else:
        mask_f = torch.ones(num_envs, num_slots, device=device)

    # Stack: [E, W, 4] then flatten
    wall_obs = torch.stack([
        rel_cx * mask_f,
        rel_cy * mask_f,
        sx * mask_f,
        sy * mask_f,
    ], dim=-1)  # [E, W, 4]

    # Pad to MAX_WALL_SLOTS if needed
    if num_slots < MAX_WALL_SLOTS:
        pad = torch.zeros(num_envs, MAX_WALL_SLOTS - num_slots, 4, device=device)
        wall_obs = torch.cat([wall_obs, pad], dim=1)
        mask_f = F.pad(mask_f, (0, MAX_WALL_SLOTS - num_slots))

    wall_flat = wall_obs.reshape(num_envs, MAX_WALL_SLOTS * 4)  # [E, 32]
    return torch.cat([wall_flat, mask_f], dim=-1)  # [E, 40]


def _extract_goal_queue(
    env, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor,
    num_envs: int, device: torch.device,
) -> torch.Tensor:
    """Extract 6D goal queue privileged obs (3 × 2D body-frame)."""
    from isaaclab.utils import math as math_utils

    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)
    cos_yaw = torch.cos(-robot_yaw)  # [E]
    sin_yaw = torch.sin(-robot_yaw)  # [E]

    result = torch.zeros(num_envs, MAX_GOAL_QUEUE, 2, device=device)

    # Access goal command term
    cmd_mgr = getattr(env, "command_manager", None)
    if cmd_mgr is None:
        return result.reshape(num_envs, MAX_GOAL_QUEUE * 2)

    # Get active command term (first one)
    cmd_terms = list(cmd_mgr._terms.values()) if hasattr(cmd_mgr, '_terms') else []
    if not cmd_terms:
        return result.reshape(num_envs, MAX_GOAL_QUEUE * 2)

    cmd = cmd_terms[0]

    # Multi-goal: has all_goals_pos_w [E, max_goals, 3]
    if hasattr(cmd, 'all_goals_pos_w'):
        all_goals = cmd.all_goals_pos_w  # [E, max_goals, 3]
        num_goals = getattr(cmd.cfg, 'num_goals', all_goals.shape[1])
        num_to_use = min(num_goals, MAX_GOAL_QUEUE)

        # Sort by distance to get nearest goals first
        goals_xy = all_goals[:, :num_goals, :2]  # [E, ng, 2]
        dists = torch.norm(goals_xy - robot_pos_w[:, :2].unsqueeze(1), dim=2)  # [E, ng]
        _, sorted_idx = torch.sort(dists, dim=1)  # [E, ng]

        for g in range(num_to_use):
            idx = sorted_idx[:, g]  # [E]
            gx = torch.gather(goals_xy[:, :, 0], 1, idx.unsqueeze(1)).squeeze(1)  # [E]
            gy = torch.gather(goals_xy[:, :, 1], 1, idx.unsqueeze(1)).squeeze(1)  # [E]

            # Transform to body-frame
            dx = gx - robot_pos_w[:, 0]
            dy = gy - robot_pos_w[:, 1]
            rel_x = cos_yaw * dx + sin_yaw * dy
            rel_y = -sin_yaw * dx + cos_yaw * dy

            result[:, g, 0] = rel_x * _GOAL_XY_SCALE
            result[:, g, 1] = rel_y * _GOAL_XY_SCALE
    else:
        # Single goal fallback
        goal_pos = cmd.goal_pos_w[:, :2]  # [E, 2]
        dx = goal_pos[:, 0] - robot_pos_w[:, 0]
        dy = goal_pos[:, 1] - robot_pos_w[:, 1]
        result[:, 0, 0] = (cos_yaw * dx + sin_yaw * dy) * _GOAL_XY_SCALE
        result[:, 0, 1] = (-sin_yaw * dx + cos_yaw * dy) * _GOAL_XY_SCALE

    return result.reshape(num_envs, MAX_GOAL_QUEUE * 2)  # [E, 6]
