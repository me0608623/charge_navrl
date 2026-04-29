"""Obstacle agent environment interface -- extracted from train_rnn_car_wdclip.py."""

import torch

from models import OBS_POLICY_OBS_DIM


def build_obstacle_obs(env_unwrapped, max_obstacles: int, device) -> torch.Tensor:
    """Read obstacle + robot state, build [num_envs, N, 9] obs tensor.

    Per-obstacle obs (body-frame of env):
      own_local_xy(2) + own_vel(2) + robot_rel_xy(2) + robot_vel(2) + d_wall(1) = 9D
    """
    num_envs = env_unwrapped.num_envs
    env_origins = env_unwrapped.scene.env_origins  # [num_envs, 3]

    # Robot state
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :3]  # [E, 3]
    robot_local = robot_pos_w - env_origins  # [E, 3]
    robot_vel_w = env_unwrapped.scene["robot"].data.root_lin_vel_w[:, :2]  # [E, 2]

    # Ensure velocity cache exists
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)

    obs_all = torch.zeros(num_envs, max_obstacles, OBS_POLICY_OBS_DIM, device=device)

    # Build obstacle cache if needed
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((num_envs,), 7.0, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # [E, 3]
        obs_local = obs_pos_w - env_origins  # [E, 3]

        # Check if active (Z > 0)
        active = obs_pos_w[:, 2] > 0.0  # [E]

        obs_xy = obs_local[:, :2]  # [E, 2]
        obs_vel = env_unwrapped._obstacle_velocities[:, i, :]  # [E, 2]
        robot_rel = robot_local[:, :2] - obs_xy  # [E, 2] relative robot pos

        # d_wall: min distance to any boundary (per-env bound)
        d_left = obs_xy[:, 0] + bound_limits
        d_right = bound_limits - obs_xy[:, 0]
        d_bottom = obs_xy[:, 1] + bound_limits
        d_top = bound_limits - obs_xy[:, 1]
        d_wall = torch.stack([d_left, d_right, d_bottom, d_top], dim=-1).min(dim=-1).values  # [E]
        d_wall = torch.max(torch.min(d_wall, bound_limits), torch.zeros_like(d_wall))

        obs_all[:, i, 0:2] = obs_xy
        obs_all[:, i, 2:4] = obs_vel
        obs_all[:, i, 4:6] = robot_rel
        obs_all[:, i, 6:8] = robot_vel_w
        obs_all[:, i, 8] = d_wall

        # Zero out inactive obstacles
        inactive = ~active
        if inactive.any():
            obs_all[inactive, i, :] = 0.0

    return obs_all  # [num_envs, N, 9]


def apply_obstacle_actions(env_unwrapped, actions: torch.Tensor, max_obstacles: int,
                           dt: float = 0.2, speed_limit: float = 0.8, bound_limit: float = 7.0):
    """Apply obstacle policy actions: velocity -> position update -> write_root_pose_to_sim.

    Only moves ACTIVE obstacles (Z > 0). Hidden obstacles (Z=-10) are skipped.
    Uses per-env _scene_bounds if available (scene size randomization).

    Args:
        actions: [num_envs, N, 2] velocity commands in [-1, 1], scaled by speed_limit
    """
    env_origins = env_unwrapped.scene.env_origins  # [E, 3]

    velocity = actions * speed_limit  # [E, N, 2], per-axis scale
    # Vector norm clamp: limit actual speed (L2 norm) to speed_limit
    speed_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    excess = speed_norm > speed_limit
    velocity = torch.where(excess, velocity * (speed_limit / speed_norm), velocity)

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((env_unwrapped.num_envs,), bound_limit, device=actions.device)

    # Update velocity cache for obs_functions.py
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(
            env_unwrapped.num_envs, max_obstacles, 2, device=actions.device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue

        pos_w = obstacle.data.root_pos_w.clone()  # [E, 3]
        active = pos_w[:, 2] > 0.0  # only move visible obstacles

        if not active.any():
            # Zero velocity for inactive obstacles
            env_unwrapped._obstacle_velocities[:, i, :] = 0.0
            continue

        # Local frame
        local_xy = pos_w[:, :2] - env_origins[:, :2]

        # Apply velocity only to active envs
        vel_i = velocity[:, i, :] * active.unsqueeze(-1).float()  # zero for inactive
        local_xy = local_xy + vel_i * dt

        # Geofence with soft bounce: reflect when hitting per-env boundary
        for dim in range(2):
            over_max = local_xy[:, dim] > bound_limits
            under_min = local_xy[:, dim] < -bound_limits
            local_xy[over_max, dim] = 2 * bound_limits[over_max] - local_xy[over_max, dim]
            local_xy[under_min, dim] = -2 * bound_limits[under_min] - local_xy[under_min, dim]
            # Final safety clamp (element-wise with per-env bounds)
            local_xy[:, dim] = torch.max(torch.min(local_xy[:, dim], bound_limits), -bound_limits)

        # Write back to world frame (only active)
        new_pos = pos_w.clone()
        new_pos[:, :2] = local_xy + env_origins[:, :2]
        # Preserve Z for inactive (keep at -10)
        new_pos[:, :2] = torch.where(active.unsqueeze(-1), new_pos[:, :2], pos_w[:, :2])

        quat = torch.zeros(env_unwrapped.num_envs, 4, device=actions.device)
        quat[:, 0] = 1.0  # identity quaternion
        pose = torch.cat([new_pos, quat], dim=-1)  # [E, 7]

        obstacle.write_root_pose_to_sim(pose)

        # Update velocity cache (only for active)
        env_unwrapped._obstacle_velocities[:, i, :] = vel_i
