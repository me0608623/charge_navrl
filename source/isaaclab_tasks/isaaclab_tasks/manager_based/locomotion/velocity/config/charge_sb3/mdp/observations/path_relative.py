"""
相對路徑觀測函數 (Relative Path Observations)

為 AIT* + RL 層級式導航提供相對路徑觀測特徵：
- 最近路徑點的相對位移
- 路徑切線方向
- AIT* 啟發式場
- 進度跟蹤
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def relative_path_displacement(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
) -> torch.Tensor:
    """計算機器人到最近路徑點的相對位移

    觀測: [displacement_x, displacement_y] 在機器人坐標系中

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 生成的路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        [num_envs, 2] 相對位移 (機器人坐標系)
    """
    robot = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_yaw = robot.data.root_quat_w  # [num_envs, 4]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 計算機器人朝向（從四元數提取 yaw）
    # q = [w, x, y, z] -> yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = robot_yaw[:, 0], robot_yaw[:, 1], robot_yaw[:, 2], robot_yaw[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 計算旋轉矩陣（世界坐標 -> 機器人坐標）
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # 找每個環境對應的路徑
    if path_points.shape[0] == 1:
        # 所有環境共用同一條路徑
        path = path_points[0]  # [num_waypoints, 2]
    else:
        path = path_points[0]  # 簡化：使用第一條路徑

    # 找到最近的路徑點
    # path: [num_waypoints, 2], robot_pos: [num_envs, 2]
    # 擴展維度進行廣播
    path_expanded = path.unsqueeze(0)  # [1, num_waypoints, 2]
    robot_expanded = robot_pos.unsqueeze(1)  # [num_envs, 1, 2]

    distances = torch.norm(path_expanded - robot_expanded, dim=2)  # [num_envs, num_waypoints]
    nearest_idx = torch.argmin(distances, dim=1)  # [num_envs]

    # 獲取最近路徑點的世界坐標
    nearest_points = path[nearest_idx]  # [num_envs, 2]

    # 計算相對位移（世界坐標）
    relative_world = nearest_points - robot_pos  # [num_envs, 2]

    # 旋轉到機器人坐標系
    relative_x = relative_world[:, 0] * cos_yaw + relative_world[:, 1] * sin_yaw
    relative_y = -relative_world[:, 0] * sin_yaw + relative_world[:, 1] * cos_yaw

    relative_robot = torch.stack([relative_x, relative_y], dim=1)  # [num_envs, 2]

    return relative_robot


def path_tangent_direction(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
) -> torch.Tensor:
    """計算路徑切線方向（相對於機器人朝向）

    觀測: [tangent_x, tangent_y] 在機器人坐標系中

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        [num_envs, 2] 切線方向（機器人坐標系，已歸一化）
    """
    robot = env.scene[robot_cfg]
    robot_yaw = robot.data.root_quat_w  # [num_envs, 4]

    num_envs = env.num_envs
    device = robot_yaw.device

    # 計算機器人朝向
    w, x, y, z = robot_yaw[:, 0], robot_yaw[:, 1], robot_yaw[:, 2], robot_yaw[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    # 找到最近路徑點的索引
    robot_pos = robot.data.root_pos_w[:, :2]
    path_expanded = path.unsqueeze(0)
    robot_expanded = robot_pos.unsqueeze(1)
    distances = torch.norm(path_expanded - robot_expanded, dim=2)
    nearest_idx = torch.argmin(distances, dim=1)

    # 計算切線方向（指向下一個路徑點）
    tangents_world = torch.zeros((num_envs, 2), device=device)

    for i in range(num_envs):
        idx = nearest_idx[i].item()
        if idx < len(path) - 1:
            # 使用下一段的方向
            tangent = path[idx + 1] - path[idx]
        else:
            # 在最後一點，使用最後一段的方向
            tangent = path[-1] - path[-2] if len(path) > 1 else torch.zeros(2)

        # 歸一化
        norm = torch.norm(tangent)
        if norm > 1e-6:
            tangent = tangent / norm

        tangents_world[i] = tangent

    # 旋轉到機器人坐標系
    tangent_x = tangents_world[:, 0] * cos_yaw + tangents_world[:, 1] * sin_yaw
    tangent_y = -tangents_world[:, 0] * sin_yaw + tangents_world[:, 1] * cos_yaw

    tangents_robot = torch.stack([tangent_x, tangent_y], dim=1)

    return tangents_robot


def path_progress(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
) -> torch.Tensor:
    """計算沿路徑的進度

    觀測: [progress] 範圍 [0, 1]，0 表示起點，1 表示終點

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        [num_envs, 1] 進度值
    """
    robot = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    # 找到最近路徑點的索引
    path_expanded = path.unsqueeze(0)
    robot_expanded = robot_pos.unsqueeze(1)
    distances = torch.norm(path_expanded - robot_expanded, dim=2)
    nearest_idx = torch.argmin(distances, dim=1)

    # 計算進度
    progress = nearest_idx.float() / (len(path) - 1)

    return progress.unsqueeze(-1)  # [num_envs, 1]


def cross_track_error_obs(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
) -> torch.Tensor:
    """計算偏離路徑的垂直距離（Cross-Track Error）

    觀測: [cte] 偏離距離（米）

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        [num_envs, 1] CTE 值
    """
    robot = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    cte = torch.zeros(num_envs, 1, device=device)

    for i in range(num_envs):
        min_cte = float('inf')

        # 遍歷所有路徑段
        for j in range(len(path) - 1):
            p1 = path[j]
            p2 = path[j + 1]

            segment = p2 - p1
            segment_length_sq = torch.sum(segment ** 2)

            if segment_length_sq < 1e-6:
                continue

            robot_to_p1 = robot_pos[i] - p1
            t = torch.clamp(
                torch.sum(robot_to_p1 * segment) / segment_length_sq,
                min=0.0, max=1.0
            )

            projection = p1 + t * segment
            dist_to_segment = torch.norm(robot_pos[i] - projection)

            min_cte = min(min_cte, dist_to_segment.item())

        cte[i, 0] = min_cte

    return cte


def aitstar_heuristic_field(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    goal_pos: torch.Tensor,
    grid_size: int = 20,
    map_range: float = 10.0,
) -> torch.Tensor:
    """計算 AIT* 啟發式場（降採樣版本）

    為了減少觀測維度，將啟發式場降採樣為小網格。

    觀測: [grid_size * grid_size] 展平的啟發式場

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        goal_pos: [num_envs, 2] 目標位置
        grid_size: 網格大小
        map_range: 地圖半徑（米）

    Returns:
        [num_envs, grid_size * grid_size] 啟發式場
    """
    robot = env.scene[robot_cfg := "robot"]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑和目標
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    if goal_pos.shape[0] == 1:
        goal = goal_pos[0]
    else:
        goal = goal_pos[0]

    # 計算相對於機器人的網格
    fields = torch.zeros((num_envs, grid_size * grid_size), device=device)

    for env_id in range(num_envs):
        robot_center = robot_pos[env_id]

        # 為每個網格單元計算啟發式值
        for i in range(grid_size):
            for j in range(grid_size):
                # 網格位置（相對於機器人）
                offset_x = (i - grid_size / 2) * (2 * map_range / grid_size)
                offset_y = (j - grid_size / 2) * (2 * map_range / grid_size)

                grid_pos = robot_center + torch.tensor([offset_x, offset_y])

                # 計算到路徑的距離
                distances = torch.norm(path - grid_pos, dim=1)
                path_dist = distances.min().item()

                # 計算到目標的距離
                goal_dist = torch.norm(goal - grid_pos).item()

                # 啟發式值（路徑偏離 + 目標距離）
                heuristic = path_dist + 0.5 * goal_dist

                idx = i * grid_size + j
                fields[env_id, idx] = heuristic

    # 歸一化（每個環境獨立）
    for env_id in range(num_envs):
        field = fields[env_id]
        min_val = field.min()
        max_val = field.max()
        if max_val > min_val:
            fields[env_id] = (field - min_val) / (max_val - min_val)

    return fields


def combined_path_observations(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    goal_pos: torch.Tensor,
    robot_cfg: str = "robot",
    include_heuristic: bool = False,
) -> torch.Tensor:
    """組合所有路徑觀測

    觀測向量包含：
    - [2] 相對路徑點位移
    - [2] 路徑切線方向
    - [1] 路徑進度
    - [1] Cross-track error
    - [H] 啟發式場（可選，H = grid_size^2）

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        goal_pos: [num_envs, 2] 目標位置
        robot_cfg: 機器人配置名稱
        include_heuristic: 是否包含啟發式場

    Returns:
        [num_envs, obs_dim] 組合觀測
    """
    # 基礎觀測
    rel_disp = relative_path_displacement(env, path_points, robot_cfg)
    tangent = path_tangent_direction(env, path_points, robot_cfg)
    progress = path_progress(env, path_points, robot_cfg)
    cte = cross_track_error_obs(env, path_points, robot_cfg)

    # 拼接
    obs = torch.cat([rel_disp, tangent, progress, cte], dim=1)  # [num_envs, 6]

    # 可選：添加啟發式場
    if include_heuristic:
        heuristic = aitstar_heuristic_field(env, path_points, goal_pos)
        obs = torch.cat([obs, heuristic], dim=1)

    return obs


__all__ = [
    "relative_path_displacement",
    "path_tangent_direction",
    "path_progress",
    "cross_track_error_obs",
    "aitstar_heuristic_field",
    "combined_path_observations",
]
