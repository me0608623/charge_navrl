"""Obstacle agent 動作執行。

Policy 輸出 [-1, 1]² 速度命令 → 乘以 speed_limit → position update。
邊界碰撞用 soft bounce (反射)。
"""

import torch


def apply_obstacle_actions(
    env_unwrapped,
    actions: torch.Tensor,
    max_obstacles: int,
    dt: float = 0.2,
    speed_limit: float = 0.8,
    bound_limit: float = 7.0,
):
    """執行 obstacle policy 動作: velocity → position update → write_root_pose_to_sim。

    只移動 ACTIVE obstacles (Z > 0)。Hidden obstacles (Z=-10) 不動。

    Args:
        env_unwrapped: Isaac Lab env (unwrapped)
        actions: [num_envs, N, 2] velocity commands ∈ [-1, 1]
        max_obstacles: N
        dt: 時間步長 (s)
        speed_limit: 最大速度 (m/s)，由 curriculum obstacle_speed_rate 覆蓋
        bound_limit: 活動範圍 fallback (由 _scene_bounds 覆蓋)
    """
    env_origins = env_unwrapped.scene.env_origins  # [E, 3]

    # Scale to physical velocity + L2 norm clamp
    velocity = actions * speed_limit  # [E, N, 2]
    speed_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    excess = speed_norm > speed_limit
    velocity = torch.where(excess, velocity * (speed_limit / speed_norm), velocity)

    # Per-env scene bounds
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((env_unwrapped.num_envs,), bound_limit, device=actions.device)

    # Velocity cache
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(
            env_unwrapped.num_envs, max_obstacles, 2, device=actions.device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue

        pos_w = obstacle.data.root_pos_w.clone()  # [E, 3]
        active = pos_w[:, 2] > 0.0

        if not active.any():
            env_unwrapped._obstacle_velocities[:, i, :] = 0.0
            continue

        # Local frame position update
        local_xy = pos_w[:, :2] - env_origins[:, :2]
        vel_i = velocity[:, i, :] * active.unsqueeze(-1).float()
        local_xy = local_xy + vel_i * dt

        # Geofence: soft bounce (reflect at per-env boundary)
        for dim in range(2):
            over_max = local_xy[:, dim] > bound_limits
            under_min = local_xy[:, dim] < -bound_limits
            local_xy[over_max, dim] = 2 * bound_limits[over_max] - local_xy[over_max, dim]
            local_xy[under_min, dim] = -2 * bound_limits[under_min] - local_xy[under_min, dim]
            local_xy[:, dim] = torch.max(torch.min(local_xy[:, dim], bound_limits), -bound_limits)

        # Write back to world frame (only active)
        new_pos = pos_w.clone()
        new_pos[:, :2] = local_xy + env_origins[:, :2]
        new_pos[:, :2] = torch.where(active.unsqueeze(-1), new_pos[:, :2], pos_w[:, :2])

        quat = torch.zeros(env_unwrapped.num_envs, 4, device=actions.device)
        quat[:, 0] = 1.0  # identity quaternion
        pose = torch.cat([new_pos, quat], dim=-1)  # [E, 7]

        obstacle.write_root_pose_to_sim(pose)
        env_unwrapped._obstacle_velocities[:, i, :] = vel_i
