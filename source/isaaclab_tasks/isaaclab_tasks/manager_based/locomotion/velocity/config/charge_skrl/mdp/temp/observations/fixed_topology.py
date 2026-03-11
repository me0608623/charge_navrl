"""
固定拓撲觀測系統 (Fixed Topology Observation System)

設計目標：
- 122 維固定觀測空間，所有 Phase 共享
- 支持連續邊界（牆壁 LiDAR）與離散物件（障礙物槽位）

觀測層級：
┌─────────────────────────────────────────────────────────────────────┐
│ 第一層防線：LiDAR (The Wall Hunter)                               │
│   - 72 維，360° 牆壁/角落掃描                                      │
│   - 連續邊界檢測，不需要預留槽位                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 第二層防線：Static Slots (靜態障礙物槽位)                          │
│   - 8 個障礙物 × 3 維 (x, y, radius) = 24 維                      │
│   - KNN 選擇最近的 8 個，不足補 0                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 第三層防線：Dynamic Slots (動態障礙物槽位)                         │
│   - 5 個障礙物 × 4 維 (x, y, vx, vy) = 20 維                      │
│   - KNN 選擇最近的 5 個，不足補 0                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 指揮層：Navigation Command (導航資訊)                             │
│   - 3 維 (goal_distance, goal_angle, goal_relative_speed)         │
├─────────────────────────────────────────────────────────────────────┤
│ 本體層：Proprioception (本體感覺)                                 │
│   - 3 維 (vx, vy, omega)                                          │
├─────────────────────────────────────────────────────────────────────┤
│ 總計: 122 維 (固定，所有 Phase 一致)                               │
└─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


# ============================================================================
# KNN 障礙物觀測函數
# ============================================================================

def nearest_static_obstacles(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_slots: int = 8,
    max_distance: float = 8.0,
) -> torch.Tensor:
    """KNN 靜態障礙物觀測 - 選取最近的 N 個障礙物

    使用 K-Nearest Neighbors 邏輯，只返回離機器人最近的 num_slots 個障礙物。
    不足的槽位用零填充。

    輸出維度: [num_envs, num_slots * 3]
    - 每個障礙物: [relative_x, relative_y, radius]
    - 座標系: 機器人座標系（機器人前方為 +X）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        num_slots: 預留槽位數量（預設 8）
        max_distance: 只觀測此距離內的障礙物（米）

    Returns:
        [num_envs, num_slots * 3] 障礙物狀態，不足的填 0
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 機器人位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_quat = robot.data.root_quat_w  # [num_envs, 4]

    # 計算機器人朝向 (yaw)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    # 初始化輸出：全部為 0
    obstacle_states = torch.zeros(num_envs, num_slots * 3, device=device)

    # 如果環境中有靜態障礙物數據
    if hasattr(env, '_static_obstacle_pos') and env._static_obstacle_pos is not None:
        obs_pos = env._static_obstacle_pos  # [num_envs, num_obstacles, 2]
        obs_radius = env._static_obstacle_radius  # [num_envs, num_obstacles]

        # 批量處理所有環境
        for env_idx in range(num_envs):
            this_robot_pos = robot_pos[env_idx]
            this_cos = cos_yaw[env_idx]
            this_sin = sin_yaw[env_idx]

            # 計算相對位置
            rel_pos = obs_pos[env_idx] - this_robot_pos  # [num_obstacles, 2]

            # 轉到機器人座標系
            rel_x = rel_pos[:, 0] * this_cos + rel_pos[:, 1] * this_sin
            rel_y = -rel_pos[:, 0] * this_sin + rel_pos[:, 1] * this_cos

            # 計算距離
            distances = torch.norm(rel_pos, dim=1)

            # 過濾：只考慮 max_distance 內的障礙物
            valid_mask = distances <= max_distance
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue  # 附近沒有障礙物，保持全 0

            # KNN: 找到最近的 num_slots 個
            valid_distances = distances[valid_indices]
            valid_rel_x = rel_x[valid_indices]
            valid_rel_y = rel_y[valid_indices]
            valid_radius = obs_radius[env_idx][valid_indices]

            # 排序並取前 num_slots 個
            num_to_take = min(num_slots, len(valid_indices))
            sorted_indices = torch.argsort(valid_distances)[:num_to_take]

            # 填充槽位
            for i in range(num_to_take):
                idx = sorted_indices[i]
                slot_offset = i * 3
                obstacle_states[env_idx, slot_offset + 0] = valid_rel_x[idx]
                obstacle_states[env_idx, slot_offset + 1] = valid_rel_y[idx]
                obstacle_states[env_idx, slot_offset + 2] = valid_radius[idx]

    return obstacle_states


def nearest_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_slots: int = 5,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """KNN 動態障礙物觀測 - 選取最近的 N 個移動障礙物

    使用 K-Nearest Neighbors 邏輯，只返回離機器人最近的 num_slots 個動態障礙物。
    不足的槽位用零填充。

    輸出維度: [num_envs, num_slots * 4]
    - 每個障礙物: [relative_x, relative_y, relative_vx, relative_vy]
    - 座標系: 機器人座標系（機器人前方為 +X）
    - 速度: 相對於機器人的速度

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        num_slots: 預留槽位數量（預設 5）
        max_distance: 只觀測此距離內的障礙物（米）

    Returns:
        [num_envs, num_slots * 4] 動態障礙物狀態，不足的填 0
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 機器人位置、朝向和速度
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w
    robot_vel = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2] 機器人座標系

    # 計算機器人朝向
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    # 初始化輸出：全部為 0
    obstacle_states = torch.zeros(num_envs, num_slots * 4, device=device)

    # 如果環境中有動態障礙物數據
    if hasattr(env, '_dynamic_obstacle_pos') and env._dynamic_obstacle_pos is not None:
        obs_pos = env._dynamic_obstacle_pos  # [num_envs, num_obstacles, 2]
        obs_vel = env._dynamic_obstacle_vel  # [num_envs, num_obstacles, 2]

        # 批量處理所有環境
        for env_idx in range(num_envs):
            this_robot_pos = robot_pos[env_idx]
            this_robot_vel = robot_vel[env_idx]
            this_cos = cos_yaw[env_idx]
            this_sin = sin_yaw[env_idx]

            # 相對位置
            rel_pos = obs_pos[env_idx] - this_robot_pos
            distances = torch.norm(rel_pos, dim=1)

            # 相對速度（世界座標系）
            rel_vel_w = obs_vel[env_idx] - robot_vel[env_idx]

            # 轉到機器人座標系
            rel_x = rel_pos[:, 0] * this_cos + rel_pos[:, 1] * this_sin
            rel_y = -rel_pos[:, 0] * this_sin + rel_pos[:, 1] * this_cos
            rel_vx = rel_vel_w[:, 0] * this_cos + rel_vel_w[:, 1] * this_sin
            rel_vy = -rel_vel_w[:, 0] * this_sin + rel_vel_w[:, 1] * this_cos

            # 過濾：只考慮 max_distance 內的障礙物
            valid_mask = distances <= max_distance
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # KNN: 找到最近的 num_slots 個
            valid_distances = distances[valid_indices]
            valid_rel_x = rel_x[valid_indices]
            valid_rel_y = rel_y[valid_indices]
            valid_rel_vx = rel_vx[valid_indices]
            valid_rel_vy = rel_vy[valid_indices]

            # 排序並取前 num_slots 個
            num_to_take = min(num_slots, len(valid_indices))
            sorted_indices = torch.argsort(valid_distances)[:num_to_take]

            # 填充槽位
            for i in range(num_to_take):
                idx = sorted_indices[i]
                slot_offset = i * 4
                obstacle_states[env_idx, slot_offset + 0] = valid_rel_x[idx]
                obstacle_states[env_idx, slot_offset + 1] = valid_rel_y[idx]
                obstacle_states[env_idx, slot_offset + 2] = valid_rel_vx[idx]
                obstacle_states[env_idx, slot_offset + 3] = valid_rel_vy[idx]

    return obstacle_states


# ============================================================================
# 導航指令觀測
# ============================================================================

def navigation_command(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """導航指令觀測 - 3 維壓縮資訊

    返回關於目標的關鍵導航資訊：
    - distance: 到目標的距離（米）
    - angle: 目標相對於機器人朝向的角度（弧度）
    - relative_speed: 機器人朝向目標的速度分量

    輸出維度: [num_envs, 3]
    - [distance, angle, relative_speed]

    AIT* 整合說明：
    - 如果 env._local_goal_world 存在，使用 AIT* 規劃的局部目標
    - 否則使用終點目標（向後兼容）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置

    Returns:
        [num_envs, 3] 導航指令
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # 優先使用 AIT* 規劃的局部目標（Carrot-on-stick）
    if hasattr(env, "_local_goal_world"):
        goal_pos_w = env._local_goal_world  # [num_envs, 2]
    else:
        # 向後兼容：使用終點目標
        goal_pos_w = env.command_manager.get_command("goal_command")[:, :2]

    robot_pos_w = robot.data.root_pos_w[:, :2]
    robot_quat_w = robot.data.root_quat_w
    robot_vel_b = robot.data.root_lin_vel_b[:, :2]  # 機器人座標系

    # 計算機器人朝向
    w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 計算到目標的向量
    to_goal = goal_pos_w - robot_pos_w  # [num_envs, 2]

    # distance: 歐幾里得距離
    distance = torch.norm(to_goal, dim=1, keepdim=True)

    # angle: 目標相對於機器人朝向的角度
    goal_angle = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    relative_angle = goal_angle - robot_yaw
    # 歸一化到 [-pi, pi]
    relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle))
    angle = relative_angle.unsqueeze(1)

    # relative_speed: 朝向目標的速度分量
    # 先將速度轉到世界座標系
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    vel_w_x = robot_vel_b[:, 0] * cos_yaw - robot_vel_b[:, 1] * sin_yaw
    vel_w_y = robot_vel_b[:, 0] * sin_yaw + robot_vel_b[:, 1] * cos_yaw
    vel_w = torch.stack([vel_w_x, vel_w_y], dim=1)

    # 計算朝向目標的速度分量
    to_goal_norm = to_goal / (distance.squeeze(1).unsqueeze(1) + 1e-6)
    relative_speed = (vel_w * to_goal_norm).sum(dim=1, keepdim=True)

    # 組合輸出
    result = torch.cat([distance, angle, relative_speed], dim=1)

    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = torch.clamp(result, -10.0, 10.0)

    return result


# ============================================================================
# 本體感覺觀測 (Proprioception)
# ============================================================================

def proprioception(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """本體感覺觀測 - 3 維機器人狀態

    返回機器人的基本運動狀態：
    - vx: 前向速度（機器人座標系）
    - vy: 側向速度（機器人座標系）
    - omega: 旋轉角速度

    輸出維度: [num_envs, 3]

    Args:
        env: 環境實例
        robot_cfg: 機器人配置

    Returns:
        [num_envs, 3] 本體感覺
    """
    from isaaclab.sensors import RayCaster

    robot: Articulation = env.scene[robot_cfg.name]

    # 線速度（機器人座標系）
    vel_xy = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]

    # 角速度（繞 Z 軸）
    omega = robot.data.root_ang_vel_b[:, 2:3]  # [num_envs, 1]

    # 組合
    result = torch.cat([vel_xy, omega], dim=1)

    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
    result = torch.clamp(result, -5.0, 5.0)

    return result


# ============================================================================
# 維度常量
# ============================================================================

LIDAR_DIM = 72
STATIC_SLOTS_DIM = 8 * 3  # x, y, radius
DYNAMIC_SLOTS_DIM = 5 * 4  # x, y, vx, vy
NAV_DIM = 3  # distance, angle, relative_speed
PROPRIOCEPTION_DIM = 3  # vx, vy, omega

TOTAL_OBS_DIM = LIDAR_DIM + STATIC_SLOTS_DIM + DYNAMIC_SLOTS_DIM + NAV_DIM + PROPRIOCEPTION_DIM
# TOTAL_OBS_DIM = 122


__all__ = [
    # KNN 障礙物觀測
    "nearest_static_obstacles",
    "nearest_dynamic_obstacles",
    # 導航指令
    "navigation_command",
    # 本體感覺
    "proprioception",
    # 維度常量
    "LIDAR_DIM",
    "STATIC_SLOTS_DIM",
    "DYNAMIC_SLOTS_DIM",
    "NAV_DIM",
    "PROPRIOCEPTION_DIM",
    "TOTAL_OBS_DIM",
]
