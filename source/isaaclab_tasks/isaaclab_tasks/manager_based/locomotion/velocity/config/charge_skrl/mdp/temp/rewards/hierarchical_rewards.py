"""層級式導航獎勵函數 (Hierarchical Navigation Rewards)

為 AIT* + RL (PPO) 層級式導航提供獎勵函數：

獎勵公式：
    R_total = R_reach + R_progress + R_tracking - P_collision - P_unstable

設計理念：
- 鼓勵跟隨局部目標（Carrot-on-stick）
- 懲罰碰撞
- 鼓勵平滑控制
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


# ============================================================================
# 核心獎勵函數
# ============================================================================

def local_goal_reached_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
    threshold: float = 0.5,
    reward: float = 10.0,
) -> torch.Tensor:
    """局部目標抵達獎勵

    當機器人到達局部目標時給予大獎勵。

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置
        threshold: 抵達閾值（米）
        reward: 獎勵值

    Returns:
        [num_envs] 獎勵張量
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]

    # 獲取目標位置
    if hasattr(env, '_local_goal_world'):
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 計算距離
    distance = torch.norm(goal_pos - robot_pos, dim=1)

    # 抵達獎勵
    reward_tensor = torch.where(distance < threshold, torch.tensor(reward, device=env.device), torch.zeros_like(distance))

    return reward_tensor


def progress_to_local_goal(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """前進獎勵（距離變化）

    每一幀比上一幀更靠近目標給正分，遠離給負分。

    公式: R = d_{t-1} - d_t

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs] 獎勵張量
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]

    # 獲取目標位置
    if hasattr(env, '_local_goal_world'):
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 計算當前距離
    current_distance = torch.norm(goal_pos - robot_pos, dim=1)

    # 獲取上一次的距離
    if not hasattr(env, '_prev_goal_distance'):
        env._prev_goal_distance = current_distance.clone()

    # 計算進度
    progress = env._prev_goal_distance - current_distance

    # 更新歷史
    env._prev_goal_distance = current_distance.clone()

    return progress


def heading_alignment_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """航向對齊獎勵

    機器人朝向與目標方向對齊時給予獎勵。

    公式: R = cos(heading_error)
    - heading_error = 0 時 R = 1（完全對齊）
    - heading_error = ±π/2 時 R = 0（垂直）
    - heading_error = ±π 時 R = -1（相反）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs] 獎勵張量
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 獲取機器人朝向
    robot_quat = robot.data.root_quat_w
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 獲取目標位置
    robot_pos = robot.data.root_pos_w[:, :2]
    if hasattr(env, '_local_goal_world'):
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 目標方向
    to_goal = goal_pos - robot_pos
    goal_direction = torch.atan2(to_goal[:, 1], to_goal[:, 0])

    # 航向誤差
    heading_error = goal_direction - robot_yaw
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

    # 對齊獎勵（cos 值）
    alignment = torch.cos(heading_error)

    return alignment


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    threshold: float = 0.3,
    penalty: float = -20.0,
) -> torch.Tensor:
    """碰撞懲罰

    當 LiDAR 檢測到距離低於閾值時給予懲罰。

    Args:
        env: 環境實例
        sensor_cfg: LiDAR 感測器配置
        threshold: 碰撞閾值（米）
        penalty: 懲罰值

    Returns:
        [num_envs] 懲罰張量
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取 LiDAR 數據
    sensor_pos = sensor.data.pos_w[:, :2]
    hit_points = sensor.data.ray_hits_w[:, :, :2]
    distances = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=2)
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)

    # 最小距離
    min_distance = torch.min(distances, dim=1)[0]

    # 碰撞懲罰
    penalty_tensor = torch.where(min_distance < threshold, torch.tensor(penalty, device=env.device), torch.zeros_like(min_distance))

    return penalty_tensor


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """動作平滑懲罰

    懲罰劇烈的動作變化，防止機器人抖動。

    公式: P = -||action_t - action_{t-1}||^2

    Args:
        env: 環境實例

    Returns:
        [num_envs] 懲罰張量
    """
    # 獲取當前動作
    current_action = env.action_manager.get_current_action()

    # 獲取上一次動作
    if not hasattr(env, '_prev_action'):
        env._prev_action = current_action.clone()

    # 計算變化量
    action_change = current_action - env._prev_action

    # 平滑懲罰
    penalty = -torch.norm(action_change, dim=1) ** 2

    # 更新歷史
    env._prev_action = current_action.clone()

    return penalty


def proximity_to_obstacle_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    safe_distance: float = 1.0,
    penalty_scale: float = -1.0,
) -> torch.Tensor:
    """障礙物接近懲罰

    越靠近障礙物，懲罰越大（但在安全距離內）。

    公式: P = -scale * (1 - min_dist / safe_distance)^2
           當 min_dist >= safe_distance 時 P = 0

    Args:
        env: 環境實例
        sensor_cfg: LiDAR 感測器配置
        safe_distance: 安全距離（米）
        penalty_scale: 懲罰係數

    Returns:
        [num_envs] 懲罰張量
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取 LiDAR 數據
    sensor_pos = sensor.data.pos_w[:, :2]
    hit_points = sensor.data.ray_hits_w[:, :, :2]
    distances = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=2)
    distances = torch.nan_to_num(distances, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)

    # 最小距離
    min_distance = torch.min(distances, dim=1)[0]

    # 只對安全距離內的障礙物懲罰
    unsafe_mask = min_distance < safe_distance

    # 計算懲罰（距離越近，懲罰越大）
    penalty = torch.zeros_like(min_distance)

    if unsafe_mask.any():
        # 歸一化距離 (0 = 很危險, 1 = 安全邊界)
        normalized_dist = min_distance[unsafe_mask] / safe_distance
        # 懲罰函數：(1 - d)^2
        penalty[unsafe_mask] = penalty_scale * (1 - normalized_dist) ** 2

    return penalty


# ============================================================================
# 組合獎勵函數
# ============================================================================

def hierarchical_navigation_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    weights: dict | None = None,
) -> torch.Tensor:
    """層級式導航完整獎勵函數

    組合所有獎勵項，給出最終的獎勵值。

    R_total = w_reach * R_reach
             + w_progress * R_progress
             + w_tracking * R_tracking
             + w_collision * P_collision
             + w_proximity * P_proximity
             + w_smooth * P_smooth

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置
        sensor_cfg: LiDAR 感測器配置
        weights: 獎勵權重字典

    Returns:
        [num_envs] 總獎勵張量
    """
    # 默認權重
    if weights is None:
        weights = {
            'reach': 1.0,        # 抵達獎勵
            'progress': 2.0,     # 前進獎勵
            'tracking': 0.5,     # 航向對齊獎勵
            'collision': 1.0,    # 碰撞懲罰
            'proximity': 0.5,    # 接近懲罰
            'smooth': 0.1,       # 平滑懲罰
        }

    # 計算各項獎勵
    r_reach = local_goal_reached_reward(env, robot_cfg, goal_cfg)
    r_progress = progress_to_local_goal(env, robot_cfg, goal_cfg)
    r_tracking = heading_alignment_reward(env, robot_cfg, goal_cfg)
    p_collision = collision_penalty(env, sensor_cfg)
    p_proximity = proximity_to_obstacle_penalty(env, sensor_cfg)
    p_smooth = action_smoothness_penalty(env)

    # 加權總和
    total_reward = (
        weights['reach'] * r_reach +
        weights['progress'] * r_progress +
        weights['tracking'] * r_tracking +
        weights['collision'] * p_collision +
        weights['proximity'] * p_proximity +
        weights['smooth'] * p_smooth
    )

    return total_reward


# ============================================================================
# 輔助函數
# ============================================================================

def reset_reward_tracking(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """重置獎勵追蹤狀態

    在環境重置時調用，清除歷史狀態。

    Args:
        env: 環境實例
        env_ids: 要重置的環境 ID
    """
    if hasattr(env, '_prev_goal_distance'):
        env._prev_goal_distance[env_ids] = 0.0

    if hasattr(env, '_prev_action'):
        num_actions = env._prev_action.shape[1]
        env._prev_action[env_ids] = torch.zeros((len(env_ids), num_actions), device=env.device)


__all__ = [
    # 核心獎勵
    "local_goal_reached_reward",
    "progress_to_local_goal",
    "heading_alignment_reward",
    "collision_penalty",
    "action_smoothness_penalty",
    "proximity_to_obstacle_penalty",
    # 組合獎勵
    "hierarchical_navigation_reward",
    # 輔助函數
    "reset_reward_tracking",
]
