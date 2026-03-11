"""障礙物觀測函數 (Obstacle Observations)

為未來 Phase 1/2/3 提供障礙物相關觀測功能：
- 靜態障礙物：固定位置的障礙物（Phase 1）
- 動態障礙物：移動的障礙物（Phase 2+）

設計原則：
1. 所有觀測都在機器人座標系（relative to robot）
2. 障礙物狀態包含：位置、速度（動態）、半徑/尺寸
3. 只觀測「最近 N 個」障礙物，避免觀測空間爆炸
4. 與 LiDAR 觀測互補：LiDAR 提供距離，這裡提供精確位置
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


# ============================================================================
# 靜態障礙物觀測 (Phase 1)
# ============================================================================

def static_obstacles_relative_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_obstacles: int = 5,
    max_distance: float = 5.0,
) -> torch.Tensor:
    """靜態障礙物相對狀態觀測

    返回最近的 N 個靜態障礙物的相對狀態。

    輸出維度: [max_obstacles * 3]
    - 障礙物 0: [relative_x, relative_y, radius]
    - 障礙物 1: [relative_x, relative_y, radius]
    - ...
    - 若不足 N 個，剩餘填 0

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        max_obstacles: 最大觀測障礙物數量
        max_distance: 只觀測此距離內的障礙物

    Returns:
        [num_envs, max_obstacles * 3] 障礙物狀態
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

    # 初始化輸出
    obstacle_states = torch.zeros(num_envs, max_obstacles * 3, device=device)

    # 如果環境中有障礙物數據
    if hasattr(env, '_static_obstacle_pos'):
        obs_pos = env._static_obstacle_pos  # [num_envs, num_obstacles, 2]
        obs_radius = env._static_obstacle_radius  # [num_envs, num_obstacles]

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

            # 只保留最近的障礙物
            sorted_indices = torch.argsort(distances)[:max_obstacles]

            for i, idx in enumerate(sorted_indices):
                if distances[idx] <= max_distance:
                    obstacle_states[env_idx, i * 3] = rel_x[idx]
                    obstacle_states[env_idx, i * 3 + 1] = rel_y[idx]
                    obstacle_states[env_idx, i * 3 + 2] = obs_radius[env_idx, idx]

    return obstacle_states


def static_obstacles_polar(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_obstacles: int = 5,
    max_distance: float = 5.0,
) -> torch.Tensor:
    """靜態障礙物極坐標觀測

    輸出維度: [max_obstacles * 2]
    - 每個障礙物: [距離, 角度]

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        max_obstacles: 最大觀測障礙物數量
        max_distance: 只觀測此距離內的障礙物

    Returns:
        [num_envs, max_obstacles * 2] 障礙物極坐標
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 機器人位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w

    # 計算機器人朝向
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 初始化輸出
    obstacle_polar = torch.zeros(num_envs, max_obstacles * 2, device=device)

    if hasattr(env, '_static_obstacle_pos'):
        obs_pos = env._static_obstacle_pos

        for env_idx in range(num_envs):
            this_robot_pos = robot_pos[env_idx]
            this_yaw = robot_yaw[env_idx]

            # 計算相對位置
            rel_pos = obs_pos[env_idx] - this_robot_pos
            distances = torch.norm(rel_pos, dim=1)

            # 計算角度
            angles = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]) - this_yaw
            angles = torch.atan2(torch.sin(angles), torch.cos(angles))

            # 排序並取最近的
            sorted_indices = torch.argsort(distances)[:max_obstacles]

            for i, idx in enumerate(sorted_indices):
                if distances[idx] <= max_distance:
                    # 歸一化距離到 [0, 1]
                    norm_dist = torch.clamp(distances[idx] / max_distance, 0.0, 1.0)
                    obstacle_polar[env_idx, i * 2] = norm_dist
                    obstacle_polar[env_idx, i * 2 + 1] = angles[idx]

    return obstacle_polar


# ============================================================================
# 動態障礙物觀測 (Phase 2+)
# ============================================================================

def dynamic_obstacles_relative_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_obstacles: int = 3,
    max_distance: float = 6.0,
) -> torch.Tensor:
    """動態障礙物相對狀態觀測

    返回最近的 N 個動態障礙物的相對狀態（包含速度）。

    輸出維度: [max_obstacles * 4]
    - 障礙物 0: [relative_x, relative_y, vx, vy]
    - 障礙物 1: [relative_x, relative_y, vx, vy]
    - ...

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        max_obstacles: 最大觀測障礙物數量
        max_distance: 只觀測此距離內的障礙物

    Returns:
        [num_envs, max_obstacles * 4] 動態障礙物狀態
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 機器人位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w

    # 計算機器人朝向
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    # 機器人速度
    robot_vel = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]

    # 初始化輸出
    obstacle_states = torch.zeros(num_envs, max_obstacles * 4, device=device)

    if hasattr(env, '_dynamic_obstacle_pos'):
        obs_pos = env._dynamic_obstacle_pos  # [num_envs, num_obstacles, 2]
        obs_vel = env._dynamic_obstacle_vel  # [num_envs, num_obstacles, 2]

        for env_idx in range(num_envs):
            this_robot_pos = robot_pos[env_idx]
            this_robot_vel = robot_vel[env_idx]
            this_cos = cos_yaw[env_idx]
            this_sin = sin_yaw[env_idx]

            # 相對位置
            rel_pos = obs_pos[env_idx] - this_robot_pos
            distances = torch.norm(rel_pos, dim=1)

            # 相對速度（考慮機器人運動）
            rel_vel = obs_vel[env_idx] - this_robot_vel

            # 轉到機器人座標系
            rel_x = rel_pos[:, 0] * this_cos + rel_pos[:, 1] * this_sin
            rel_y = -rel_pos[:, 0] * this_sin + rel_pos[:, 1] * this_cos
            rel_vx = rel_vel[:, 0] * this_cos + rel_vel[:, 1] * this_sin
            rel_vy = -rel_vel[:, 0] * this_sin + rel_vel[:, 1] * this_cos

            # 排序
            sorted_indices = torch.argsort(distances)[:max_obstacles]

            for i, idx in enumerate(sorted_indices):
                if distances[idx] <= max_distance:
                    obstacle_states[env_idx, i * 4] = rel_x[idx]
                    obstacle_states[env_idx, i * 4 + 1] = rel_y[idx]
                    obstacle_states[env_idx, i * 4 + 2] = rel_vx[idx]
                    obstacle_states[env_idx, i * 4 + 3] = rel_vy[idx]

    return obstacle_states


# ============================================================================
# 輔助函數：從環境收集障礙物信息
# ============================================================================

def collect_obstacle_info_from_scene(
    env: ManagerBasedRLEnv,
    obstacle_prefix: str = "obstacle_",
    max_obstacles: int = 10,
) -> dict:
    """從場景中收集障礙物信息

    此函數應在環境初始化或重置時調用，
    將障礙物信息存儲到環境變數中供觀測函數使用。

    Args:
        env: 環境實例
        obstacle_prefix: 障礙物 prim 前綴
        max_obstacles: 最大障礙物數量

    Returns:
        包含障礙物位置和速度的字典
    """
    num_envs = env.num_envs
    device = env.device

    positions = torch.zeros(num_envs, max_obstacles, 2, device=device)
    velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)
    radii = torch.zeros(num_envs, max_obstacles, device=device)
    count = torch.zeros(num_envs, dtype=torch.long, device=device)

    # 遍歷場景中的障礙物
    for name in env.scene.keys():
        if name.startswith(obstacle_prefix):
            obj = env.scene[name]
            if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                # 假設單例障礙物，複製到所有環境
                pos = obj.data.root_pos_w[0, :2]  # [2]
                positions[:, 0] = pos
                count[:] += 1

    return {
        'positions': positions,
        'velocities': velocities,
        'radii': radii,
        'count': count,
    }


# ============================================================================
# 觀測空間總結文檔
# ============================================================================

"""
╔════════════════════════════════════════════════════════════════════════════╗
║             零填充觀測空間設計 (Zero-Padding Observation Space)           ║
╠════════════════════════════════════════════════════════════════════════════╣

核心思想：Phase 0 就定義所有未來 Phase 可能用到的最大維度，
          未使用的維度填零。這樣模型可以在 Phase 間無縫遷移。

════════════════════════════════════════════════════════════════════════════
                    固定觀測空間: 108 維 (所有 Phase 一致)
════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ 基礎觀測 (所有 Phase): 81 維                                      │
├─────────────────────────────────────────────────────────────────────┤
│ lidar_scan        72 │ 360° LiDAR 掃描                                  │
│ local_goal         2 │ AIT* 局部目標 (前向, 側向)                       │
│ base_velocity_xy   2 │ 線速度 (vx, vy)                                  │
│ angular_velocity_z  1 │ 角速度 (ω)                                      │
│ time_remaining     1 │ 剩餘時間比例                                     │
│ alive_flag         1 │ 存活標誌                                         │
│ actions            2 │ 上一步動作                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 零填充: 靜態障礙物: 15 維 (預留給 Phase 1)                        │
│   5 個障礙物 × 3 維 (relative_x, relative_y, radius)               │
│   Phase 0/2: 全為 0 (無靜態障礙物)                                │
│   Phase 1:   實際障礙物數據                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 零填充: 動態障礙物: 12 維 (預留給 Phase 2+)                       │
│   3 個障礙物 × 4 維 (relative_x, relative_y, vx, vy)               │
│   Phase 0/1: 全為 0 (無動態障礙物)                                │
│   Phase 2+:  實際動態障礙物數據                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 總計: 81 + 15 + 12 = 108 維 (固定，所有 Phase 一致)               │
└─────────────────────────────────────────────────────────────────────┘

零填充設計優勢：
════════════════
1. 無縫遷移：Phase 0 訓練的模型可以直接載入 Phase 1/2/3 使用
2. 維度穩定：不需要重新定義觀測空間或修改網絡架構
3. 學習效率：RL 在 Phase 0 就學會「忽略」零填充維度
4. 漸進式：可以逐步啟用障礙物觀測，觀察學習曲線變化

Phase 演進：
═══════════
Phase 0 (車輛動力學校準):
  - 空地，無障礙物
  - static_obstacles 全為 0
  - dynamic_obstacles 全為 0
  - 專注於車輛控制基礎

Phase 1 (靜態障礙物導航):
  - 添加固定位置的障礙物
  - static_obstacles 有實際數據
  - dynamic_obstacles 全為 0
  - 可以直接載入 Phase 0 的模型作為初始化

Phase 2+ (動態障礙物導航):
  - 添加移動的障礙物
  - static_obstacles 有實際數據
  - dynamic_obstacles 有實際數據
  - 可以載入 Phase 0/1 的模型

╚════════════════════════════════════════════════════════════════════════════╝
"""


def topk_obstacles_goal_centric(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
) -> torch.Tensor:
    """Top-K 障礙物觀測（Goal-Centric Frame，NavRL 風格）

    選擇最近 K 個可見障礙物，在 goal-centric 座標系下計算相對位置和速度。
    Goal-centric frame: 目標方向為 X 軸，垂直方向為 Y 軸。

    每個障礙物 6 維: [rel_x_goal, rel_y_goal, distance, vel_x_goal, vel_y_goal, size]

    全向量化實現（torch.topk + batch 投影），無 Python per-env 迴圈。
    Empty 環境自動零填充。

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        top_k: 選取最近的 K 個障礙物
        max_obstacles: 場景中最大障礙物數量
        max_distance: 最大觀測距離（超過此距離視為不可見）

    Returns:
        [num_envs, top_k * 6] 障礙物觀測（30 維 for K=5）
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # 輸出 shape: [num_envs, top_k * 6]
    output = torch.zeros(num_envs, top_k * 6, device=device)

    # --- 收集所有障礙物位置、速度、大小 ---
    # positions: [num_envs, max_obstacles, 3] (世界座標)
    # velocities: [num_envs, max_obstacles, 2]
    all_pos = torch.zeros(num_envs, max_obstacles, 3, device=device)
    all_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_size = torch.zeros(num_envs, max_obstacles, device=device)

    num_found = 0
    for i in range(max_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue
        obstacle = env.scene[obstacle_name]
        all_pos[:, num_found, :] = obstacle.data.root_pos_w[:, :3]
        num_found += 1

    if num_found == 0:
        return output

    # 截斷到實際障礙物數量
    all_pos = all_pos[:, :num_found, :]  # [num_envs, num_found, 3]
    all_vel = all_vel[:, :num_found, :]
    all_size = all_size[:, :num_found]

    # 從緩存讀取速度和大小
    if hasattr(env, "_obstacle_velocities"):
        vel_cache = env._obstacle_velocities  # [num_envs, ?, 2]
        n_copy = min(num_found, vel_cache.shape[1])
        all_vel[:, :n_copy, :] = vel_cache[:, :n_copy, :]

    if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
        sizes_raw = env._obstacle_sizes  # list[float] 或 Tensor
        sizes = torch.as_tensor(sizes_raw, device=device, dtype=torch.float32)
        n_copy = min(num_found, len(sizes))
        # _obstacle_sizes is per-obstacle (not per-env), broadcast
        all_size[:, :n_copy] = sizes[:n_copy].unsqueeze(0).expand(num_envs, -1)

    # --- 可見性判斷：Z > 0 表示可見（隱藏的在 Z = -10） ---
    visible = all_pos[:, :, 2] > 0.0  # [num_envs, num_found]

    # --- 機器人位置和目標位置 ---
    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    # 統一目標來源：優先使用局部航點（分層架構核心）
    try:
        if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
            goal_xy = env._local_goal_world[:, :2]  # [num_envs, 2]
        else:
            # Fallback: 系統尚未生成局部航點時暫用全域目標
            goal_pos = env.command_manager.get_command("goal_command")
            goal_xy = goal_pos[:, :2]  # [num_envs, 2]
    except (AttributeError, KeyError, IndexError):
        # 無目標時使用機器人前方作為 X 軸（退化為 robot frame）
        robot_quat = robot.data.root_quat_w
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        goal_xy = robot_pos_xy + torch.stack(
            [torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=1
        )

    # --- 計算 goal-centric 座標系 ---
    # X 軸 = 目標方向（從機器人指向目標）
    goal_dir = goal_xy - robot_pos_xy  # [num_envs, 2]
    goal_dist = torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)  # [num_envs, 1]
    x_axis = goal_dir / goal_dist  # [num_envs, 2] 單位向量
    # Y 軸 = 垂直於 X 軸（逆時針旋轉 90 度）
    y_axis = torch.stack([-x_axis[:, 1], x_axis[:, 0]], dim=1)  # [num_envs, 2]

    # --- 計算障礙物相對位置（世界座標系） ---
    rel_pos = all_pos[:, :, :2] - robot_pos_xy.unsqueeze(1)  # [num_envs, num_found, 2]

    # 計算距離
    distances = torch.norm(rel_pos, dim=2)  # [num_envs, num_found]

    # 不可見或太遠的設為大值（不會被 topk 選中）
    large_val = max_distance * 10.0
    masked_distances = torch.where(
        visible & (distances < max_distance),
        distances,
        torch.full_like(distances, large_val),
    )

    # --- 選取最近 K 個 ---
    actual_k = min(top_k, num_found)
    topk_distances, topk_indices = torch.topk(
        masked_distances, k=actual_k, dim=1, largest=False
    )  # [num_envs, actual_k]

    # 有效掩碼：距離小於 large_val
    topk_valid = topk_distances < large_val  # [num_envs, actual_k]

    # --- batch gather ---
    # 擴展 indices 以收集位置、速度、大小
    idx_expanded = topk_indices.unsqueeze(2).expand(-1, -1, 2)  # [num_envs, K, 2]

    topk_rel_pos = torch.gather(rel_pos, 1, idx_expanded)  # [num_envs, K, 2]
    topk_vel = torch.gather(all_vel, 1, idx_expanded)  # [num_envs, K, 2]
    topk_size = torch.gather(all_size, 1, topk_indices)  # [num_envs, K]
    topk_dist = topk_distances  # [num_envs, K]

    # --- 投影到 goal-centric frame ---
    # rel_x_goal = dot(rel_pos, x_axis), rel_y_goal = dot(rel_pos, y_axis)
    x_axis_exp = x_axis.unsqueeze(1).expand(-1, actual_k, -1)  # [num_envs, K, 2]
    y_axis_exp = y_axis.unsqueeze(1).expand(-1, actual_k, -1)  # [num_envs, K, 2]

    rel_x_goal = (topk_rel_pos * x_axis_exp).sum(dim=2)  # [num_envs, K]
    rel_y_goal = (topk_rel_pos * y_axis_exp).sum(dim=2)  # [num_envs, K]
    vel_x_goal = (topk_vel * x_axis_exp).sum(dim=2)  # [num_envs, K]
    vel_y_goal = (topk_vel * y_axis_exp).sum(dim=2)  # [num_envs, K]

    # --- 組裝輸出 [rel_x, rel_y, distance, vel_x, vel_y, size] ---
    for k in range(actual_k):
        offset = k * 6
        output[:, offset + 0] = torch.where(topk_valid[:, k], rel_x_goal[:, k], torch.zeros_like(rel_x_goal[:, k]))
        output[:, offset + 1] = torch.where(topk_valid[:, k], rel_y_goal[:, k], torch.zeros_like(rel_y_goal[:, k]))
        output[:, offset + 2] = torch.where(topk_valid[:, k], topk_dist[:, k], torch.zeros_like(topk_dist[:, k]))
        output[:, offset + 3] = torch.where(topk_valid[:, k], vel_x_goal[:, k], torch.zeros_like(vel_x_goal[:, k]))
        output[:, offset + 4] = torch.where(topk_valid[:, k], vel_y_goal[:, k], torch.zeros_like(vel_y_goal[:, k]))
        output[:, offset + 5] = torch.where(topk_valid[:, k], topk_size[:, k], torch.zeros_like(topk_size[:, k]))

    # 如果 actual_k < top_k，剩餘已經是 0（output 初始化為 0）

    return output


def topk_obstacles_paper_format(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
) -> torch.Tensor:
    """Paper-style obstacle observation (Eq. 3.8 + Table 3.2)

    每個障礙物 9 維: [dx_norm, dy_norm, cos_θ, sin_θ, ρ, o, r, vx_norm, vy_norm]

    座標系: robot frame（非 goal-centric）
    歸一化: pos / sensing_range, vel / v_max

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        top_k: 選取最近的 K 個障礙物
        max_obstacles: 場景中最大障礙物數量
        max_distance: 最大觀測距離 (= sensing_range)
        v_max: 速度歸一化常數

    Returns:
        [num_envs, top_k * 9] 障礙物觀測（45 維 for K=5）
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device
    sensing_range = max_distance

    # 輸出 shape: [num_envs, top_k * 9]
    output = torch.zeros(num_envs, top_k * 9, device=device)

    # --- 收集所有障礙物位置、速度、大小 ---
    all_pos = torch.zeros(num_envs, max_obstacles, 3, device=device)
    all_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_size = torch.zeros(num_envs, max_obstacles, device=device)

    num_found = 0
    for i in range(max_obstacles):
        obstacle_name = f"obstacle_{i}"
        if obstacle_name not in env.scene.keys():
            continue
        obstacle = env.scene[obstacle_name]
        all_pos[:, num_found, :] = obstacle.data.root_pos_w[:, :3]
        num_found += 1

    if num_found == 0:
        return output

    # 截斷到實際障礙物數量
    all_pos = all_pos[:, :num_found, :]
    all_vel = all_vel[:, :num_found, :]
    all_size = all_size[:, :num_found]

    # 從緩存讀取速度和大小
    if hasattr(env, "_obstacle_velocities"):
        vel_cache = env._obstacle_velocities
        n_copy = min(num_found, vel_cache.shape[1])
        all_vel[:, :n_copy, :] = vel_cache[:, :n_copy, :]

    if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
        sizes_raw = env._obstacle_sizes
        sizes = torch.as_tensor(sizes_raw, device=device, dtype=torch.float32)
        n_copy = min(num_found, len(sizes))
        all_size[:, :n_copy] = sizes[:n_copy].unsqueeze(0).expand(num_envs, -1)

    # --- 可見性判斷：Z > 0 表示可見（隱藏的在 Z = -10）---
    visible = all_pos[:, :, 2] > 0.0  # [num_envs, num_found]

    # --- 機器人狀態 ---
    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
    robot_quat = robot.data.root_quat_w
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_yaw = torch.cos(robot_yaw)  # [num_envs]
    sin_yaw = torch.sin(robot_yaw)  # [num_envs]

    # --- 世界座標系相對位置 ---
    rel_pos_w = all_pos[:, :, :2] - robot_pos_xy.unsqueeze(1)  # [num_envs, num_found, 2]
    distances = torch.norm(rel_pos_w, dim=2)  # [num_envs, num_found]

    # 不可見或太遠的設為大值
    large_val = max_distance * 10.0
    masked_distances = torch.where(
        visible & (distances < max_distance),
        distances,
        torch.full_like(distances, large_val),
    )

    # --- 選取最近 K 個 ---
    actual_k = min(top_k, num_found)
    topk_distances, topk_indices = torch.topk(
        masked_distances, k=actual_k, dim=1, largest=False
    )
    topk_valid = topk_distances < large_val  # [num_envs, actual_k]

    # --- batch gather ---
    idx_expanded = topk_indices.unsqueeze(2).expand(-1, -1, 2)  # [num_envs, K, 2]
    topk_rel_pos_w = torch.gather(rel_pos_w, 1, idx_expanded)  # [num_envs, K, 2]
    topk_vel_w = torch.gather(all_vel, 1, idx_expanded)  # [num_envs, K, 2]
    topk_size = torch.gather(all_size, 1, topk_indices)  # [num_envs, K]

    # --- 轉 robot frame: rotation by -yaw ---
    cos_y = cos_yaw.unsqueeze(1).unsqueeze(2)  # [num_envs, 1, 1]
    sin_y = sin_yaw.unsqueeze(1).unsqueeze(2)  # [num_envs, 1, 1]

    # 位置: robot frame
    rel_x_r = topk_rel_pos_w[:, :, 0:1] * cos_y + topk_rel_pos_w[:, :, 1:2] * sin_y
    rel_y_r = -topk_rel_pos_w[:, :, 0:1] * sin_y + topk_rel_pos_w[:, :, 1:2] * cos_y
    # [num_envs, K, 1] each

    # 相對速度 (世界座標系)
    rel_vel_w = topk_vel_w - robot_vel_xy.unsqueeze(1)  # [num_envs, K, 2]
    # 速度: robot frame
    rel_vx_r = rel_vel_w[:, :, 0:1] * cos_y + rel_vel_w[:, :, 1:2] * sin_y
    rel_vy_r = -rel_vel_w[:, :, 0:1] * sin_y + rel_vel_w[:, :, 1:2] * cos_y

    # --- 歸一化 ---
    dx_norm = rel_x_r.squeeze(2) / sensing_range  # [num_envs, K]
    dy_norm = rel_y_r.squeeze(2) / sensing_range
    vx_norm = rel_vx_r.squeeze(2) / v_max
    vy_norm = rel_vy_r.squeeze(2) / v_max

    # --- bearing angle ---
    bearing = torch.atan2(rel_y_r.squeeze(2), rel_x_r.squeeze(2))  # [num_envs, K]
    cos_theta = torch.cos(bearing)
    sin_theta = torch.sin(bearing)

    # --- 障礙物類型 ρ (暫用 0=unknown) ---
    rho = torch.zeros(num_envs, actual_k, device=device)

    # --- 運動狀態 o: 速度 > 0.05 m/s 為 moving ---
    obs_speed = torch.sqrt(rel_vx_r.squeeze(2) ** 2 + rel_vy_r.squeeze(2) ** 2)
    o_flag = (obs_speed > 0.05).float()

    # --- 組裝 9 維 × K ---
    for k in range(actual_k):
        offset = k * 9
        valid_k = topk_valid[:, k]
        zero = torch.zeros(num_envs, device=device)
        output[:, offset + 0] = torch.where(valid_k, dx_norm[:, k], zero)
        output[:, offset + 1] = torch.where(valid_k, dy_norm[:, k], zero)
        output[:, offset + 2] = torch.where(valid_k, cos_theta[:, k], zero)
        output[:, offset + 3] = torch.where(valid_k, sin_theta[:, k], zero)
        output[:, offset + 4] = torch.where(valid_k, rho[:, k], zero)
        output[:, offset + 5] = torch.where(valid_k, o_flag[:, k], zero)
        output[:, offset + 6] = torch.where(valid_k, topk_size[:, k], zero)
        output[:, offset + 7] = torch.where(valid_k, vx_norm[:, k], zero)
        output[:, offset + 8] = torch.where(valid_k, vy_norm[:, k], zero)

    return output


__all__ = [
    # 靜態障礙物觀測
    "static_obstacles_relative_state",
    "static_obstacles_polar",
    # 動態障礙物觀測
    "dynamic_obstacles_relative_state",
    # Top-K 障礙物觀測（NavRL 風格）
    "topk_obstacles_goal_centric",
    # Top-K 障礙物觀測（論文式 9 維）
    "topk_obstacles_paper_format",
    # 輔助函數
    "collect_obstacle_info_from_scene",
]
