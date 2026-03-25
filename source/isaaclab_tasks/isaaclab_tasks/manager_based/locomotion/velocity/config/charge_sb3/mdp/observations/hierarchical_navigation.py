"""層級式導航觀測函數 (Hierarchical Navigation Observations)

為 AIT* + RL (PPO) 層級式導航提供觀測：
- LiDAR 感知（避障）
- 局部目標信息（跟隨 AIT* 路徑）
- 機器人狀態（控制）

設計理念：
- AIT* = 大腦（全域規劃）
- RL = 小腦（局部控制）
- 觀測只包含局部目標，不包含完整 AIT* 路徑
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster

from isaaclab.managers import SceneEntityCfg


# ============================================================================
# 核心觀測函數
# ============================================================================

def local_goal_polar(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """局部目標極坐標觀測

    返回目標相對於機器人的距離和角度（極坐標表示）。
    這是「Carrot-on-stick」的核心觀測。

    輸出: [num_envs, 2]
    - [0]: 距離 (米)，歸一化到 [0, 1]
    - [1]: 角度 (弧度)，範圍 [-π, π]，0 表示正前方

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs, 2] 極坐標觀測
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 獲取機器人位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    robot_quat = robot.data.root_quat_w  # [num_envs, 4]

    # 計算機器人朝向 (yaw)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # [num_envs]

    # 獲取目標位置（從命令管理器）
    goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]  # [num_envs, 2]

    # 如果使用局部目標提取器
    if hasattr(env, '_local_goal_world'):
        # 使用局部目標而非全局目標
        goal_pos = env._local_goal_world  # [num_envs, 2]

    # 計算相對位置（世界坐標）
    relative_pos = goal_pos - robot_pos  # [num_envs, 2]

    # 計算極坐標
    distance = torch.norm(relative_pos, dim=1, keepdim=True)  # [num_envs, 1]

    # 計算相對角度（世界坐標系）
    angle_world = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])  # [num_envs]

    # 轉換到機器人坐標系
    angle_relative = angle_world - robot_yaw  # [num_envs]

    # 歸一化角度到 [-π, π]
    angle_relative = torch.atan2(torch.sin(angle_relative), torch.cos(angle_relative))

    # 歸一化距離到 [0, 1]（假設最大距離 10 米）
    distance_normalized = torch.clamp(distance / 10.0, 0.0, 1.0)

    # 拼接
    obs = torch.cat([distance_normalized, angle_relative.unsqueeze(-1)], dim=1)  # [num_envs, 2]

    return obs


def local_goal_cartesian(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """局部目標笛卡爾坐標觀測

    返回目標在機器人坐標系中的笛卡爾坐標。
    機器人坐標系：X軸 = 前方，Y軸 = 左側

    輸出: [num_envs, 2]
    - [0]: 前向分量 (米)，正值 = 前方
    - [1]: 側向分量 (米)，正值 = 左側

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs, 2] 笛卡爾坐標觀測
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 獲取機器人位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w

    # 計算機器人朝向 (yaw)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # 獲取目標位置
    if hasattr(env, '_local_goal_world'):
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 計算相對位置（世界坐標）
    dx = goal_pos[:, 0] - robot_pos[:, 0]
    dy = goal_pos[:, 1] - robot_pos[:, 1]

    # 旋轉到機器人坐標系
    forward = dx * cos_yaw + dy * sin_yaw   # 前方分量
    lateral = -dx * sin_yaw + dy * cos_yaw  # 側向分量

    # 歸一化
    forward_normalized = torch.clamp(forward / 10.0, -1.0, 1.0)
    lateral_normalized = torch.clamp(lateral / 10.0, -1.0, 1.0)

    obs = torch.stack([forward_normalized, lateral_normalized], dim=1)

    return obs


def navigation_features(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """完整導航特征觀測

    結合極坐標和笛卡爾坐標，提供給 RL 最完整的導航信息。

    輸出: [num_envs, 6]
    - [0]: 距離 (歸一化)
    - [1]: 角度誤差 (弧度)
    - [2]: 前向分量 (歸一化)
    - [3]: 側向分量 (歸一化)
    - [4]: 當前線速度 (歸一化)
    - [5]: 當前角速度 (歸一化)

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs, 6] 導航特征觀測
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 極坐標
    polar = local_goal_polar(env, robot_cfg, goal_cfg)  # [num_envs, 2]

    # 笛卡爾坐標
    cartesian = local_goal_cartesian(env, robot_cfg, goal_cfg)  # [num_envs, 2]

    # 機器人速度
    velocity = robot.data.root_lin_vel_b  # [num_envs, 3]
    linear_vel = velocity[:, 0]  # 前進速度
    angular_vel = robot.data.root_ang_vel_b[:, 2]  # 偏航角速度

    # 歸一化速度
    linear_vel_norm = torch.clamp(linear_vel / 1.0, -1.0, 1.0)  # 假設最大 1 m/s
    angular_vel_norm = torch.clamp(angular_vel / 1.0, -1.0, 1.0)  # 假設最大 1 rad/s

    # 拼接所有特征
    obs = torch.cat([
        polar,           # [num_envs, 2] - 距離, 角度
        cartesian,       # [num_envs, 2] - 前向, 側向
        linear_vel_norm.unsqueeze(-1),
        angular_vel_norm.unsqueeze(-1),
    ], dim=1)  # [num_envs, 6]

    return obs


def lidar_with_navigation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """LiDAR + 導航信息組合觀測

    這是層級式導航的主要觀測函數，結合感知和導航信息。

    輸出: [num_envs, lidar_dim + nav_dim]
    - [:72]: LiDAR 掃描（360度）
    - [72:78]: 導航特征

    Args:
        env: 環境實例
        sensor_cfg: LiDAR 感測器配置
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        組合觀測張量
    """
    from isaaclab.sensors import RayCaster
    from ..functions import lidar_scan_2d_sweep

    # LiDAR 觀測
    lidar = lidar_scan_2d_sweep(env, sensor_cfg)  # [num_envs, 72]

    # 導航特征
    nav = navigation_features(env, robot_cfg, goal_cfg)  # [num_envs, 6]

    # 拼接
    obs = torch.cat([lidar, nav], dim=1)  # [num_envs, 78]

    return obs


# ============================================================================
# 高級觀測函數
# ============================================================================

def cross_track_error_with_heading(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
) -> torch.Tensor:
    """Cross-Track Error (CTE) + 航向誤差觀測

    用於路徑跟隨的經典觀測組合。

    輸出: [num_envs, 2]
    - [0]: 側向偏差 (米)，正 = 在路徑左側
    - [1]: 航向誤差 (弧度)，正 = 需要左轉

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置

    Returns:
        [num_envs, 2] CTE + 航向誤差
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 獲取位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 獲取目標
    if hasattr(env, '_local_goal_world'):
        goal_pos = env._local_goal_world
    else:
        goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 計算相對向量
    to_goal = goal_pos - robot_pos  # [num_envs, 2]
    distance = torch.norm(to_goal, dim=1, keepdim=True)

    # 目標方向
    goal_direction = torch.atan2(to_goal[:, 1], to_goal[:, 0])

    # 航向誤差
    heading_error = goal_direction - robot_yaw
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

    # 側向偏差（使用叉積）
    cross_product = to_goal[:, 0] * torch.sin(robot_yaw) - to_goal[:, 1] * torch.cos(robot_yaw)
    lateral_error = cross_product / (distance.squeeze(-1) + 1e-6)

    # 歸一化
    lateral_norm = torch.clamp(lateral_error / 2.0, -1.0, 1.0)  # 假設走廊寬度 2m

    obs = torch.stack([lateral_norm, heading_error], dim=1)

    return obs


def lookahead_goal_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_command"),
    lookahead_distance: float = 2.0,
) -> torch.Tensor:
    """前瞻目標觀測 (Lookahead Goal Observation)

    沿著當前到目標的方向，向前看 lookahead_distance 米，
    返回那個點在機器人坐標系中的位置。

    這是一種簡化的局部目標提取，不需要 AIT* 路徑。

    輸出: [num_envs, 3]
    - [0]: 前向距離 (米)
    - [1]: 側向距離 (米)
    - [2]: 到前瞻點的總距離 (米)

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        goal_cfg: 目標命令配置
        lookahead_distance: 前瞻距離（米）

    Returns:
        [num_envs, 3] 前瞻目標觀測
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]

    # 獲取位置和朝向
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w

    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)

    # 獲取目標
    goal_pos = env.command_manager.get_command(goal_cfg.name)[:, :2]

    # 計算到目標的距離
    to_goal = goal_pos - robot_pos
    distance_to_goal = torch.norm(to_goal, dim=1, keepdim=True)

    # 計算前瞻點（限制在目標距離內）
    lookahead_dist = torch.clamp(
        torch.tensor(lookahead_distance, device=env.device),
        max=distance_to_goal.max()
    )

    # 單位方向向量
    goal_direction = to_goal / (distance_to_goal + 1e-6)

    # 前瞻點位置
    lookahead_point = robot_pos + goal_direction * lookahead_dist

    # 轉換到機器人坐標系
    dx = lookahead_point[:, 0] - robot_pos[:, 0]
    dy = lookahead_point[:, 1] - robot_pos[:, 1]

    forward = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw

    # 歸一化
    forward_norm = torch.clamp(forward / 10.0, -1.0, 1.0)
    lateral_norm = torch.clamp(lateral / 10.0, -1.0, 1.0)
    distance_norm = torch.clamp(lookahead_dist / 10.0, 0.0, 1.0)

    obs = torch.stack([forward_norm, lateral_norm, distance_norm], dim=1)

    return obs


# ============================================================================
# 更新局部目標（供環境調用）
# ============================================================================

def update_local_goal_from_aitstar(
    env: ManagerBasedRLEnv,
    aitstar_path: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead_distance: float = 2.0,
) -> None:
    """從 AIT* 路徑更新局部目標

    這個函數應該在環境的 step() 開始時調用，
    用於更新環境緩存的局部目標位置。

    Args:
        env: 環境實例
        aitstar_path: [N, 2] AIT* 路徑點
        robot_cfg: 機器人配置
        lookahead_distance: 前瞻距離
    """
    from isaaclab.assets import Articulation
    from ..path_planner.local_goal_extractor import extract_local_goals_batch

    if not hasattr(env, '_local_goal_extractor'):
        from ..path_planner.local_goal_extractor import LocalGoalExtractor, LocalGoalConfig
        cfg = LocalGoalConfig(lookahead_distance=lookahead_distance)
        env._local_goal_extractor = LocalGoalExtractor(cfg, env.num_envs)

    # 提取局部目標
    local_goals = extract_local_goals_batch(
        env,
        aitstar_path,
        robot_cfg.name,
    )  # [num_envs, 2]

    # 保存到環境（供觀測函數使用）
    env._local_goal_world = local_goals


__all__ = [
    # 核心觀測
    "local_goal_polar",
    "local_goal_cartesian",
    "navigation_features",
    "lidar_with_navigation",
    # 高級觀測
    "cross_track_error_with_heading",
    "lookahead_goal_obs",
    # 工具函數
    "update_local_goal_from_aitstar",
]
