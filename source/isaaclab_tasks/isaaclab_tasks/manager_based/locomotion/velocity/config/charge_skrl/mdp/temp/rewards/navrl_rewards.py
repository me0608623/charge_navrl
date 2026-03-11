"""NavRL 風格獎勵函數

移植自 NavRL 專案 (https://github.com/Zhefan-Xu/NavRL)
NavRL: Learning Safe Flight in Dynamic Environments (IEEE RA-L 2025)

核心特點：
1. 使用 LiDAR 距離的對數作為安全獎勵
2. 動作平滑度懲罰（防止抖動）
3. 速度方向獎勵（鼓勵向目標移動）

原始實現基於四旋翼，此處適配為車輛。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def navrl_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,
    target_cfg: object = None,
    max_reward: float = 2.0,
) -> torch.Tensor:
    """NavRL 風格的速度獎勵

    獎勵機器人在目標方向上的速度分量

    R_vel = (v · r_hat) + 1.0

    其中 v 是速度向量，r_hat 是指向目標的單位向量
    +1.0 是基本獎勵（即使不移動也給點分）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置 (SceneEntityCfg)
        target_cfg: 目標配置
        max_reward: 最大獎勵（用於裁剪）

    Returns:
        獎勵張量，形狀為 (num_envs, 1)
    """
    # 獲取機器人資產
    if robot_cfg is None:
        robot = env.scene["robot"]
    elif isinstance(robot_cfg, SceneEntityCfg):
        robot = env.scene[robot_cfg.name]
    else:
        robot = robot_cfg

    # 統一目標來源：優先使用局部航點（分層架構核心）
    if target_cfg is None:
        if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
            target_cfg = env._local_goal_world
        else:
            # Fallback: 系統尚未生成局部航點時暫用全域目標
            target_cfg = env.command_manager.get_command("goal_command")

    # 機器人速度 (世界座標系)
    robot_vel_w = robot.data.root_vel_w[:, :2]  # (num_envs, 2)

    # 目標位置（世界座標系）
    target_pos_w = target_cfg[:, :2]

    # 機器人位置
    robot_pos_w = robot.data.root_pos_w[:, :2]

    # 計算相對位置向量
    r_vec = target_pos_w - robot_pos_w

    # 計算距離（避免除零）
    distance = torch.norm(r_vec, dim=-1, keepdim=True).clamp(min=1e-6)

    # 歸一化的目標方向
    r_hat = r_vec / distance

    # 速度在目標方向的投影
    vel_toward_goal = (robot_vel_w * r_hat).sum(dim=-1)

    # 獎勵 = 純速度投影（符合原版 NavRL Eq.8，無偏移）
    reward = vel_toward_goal

    # 裁剪到合理範圍
    reward = reward.clamp(min=-max_reward, max=max_reward)

    # NaN 防護：物理模擬偶爾產生 NaN 速度，防止汙染整個 reward signal
    reward = torch.nan_to_num(reward, nan=0.0)

    return reward


def navrl_safety_reward_lidar(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = None,
    lidar_range: float = 10.0,
) -> torch.Tensor:
    """NavRL 風格的 LiDAR 安全獎勵（原版 Eq.9）

    R_safety = (1/N) * Σ log(d_i)

    其中 d_i 是每條 LiDAR 射線的實際距離。
    - 遠離障礙物 → log(大值) → 高分（正獎勵）
    - 靠近障礙物 → log(小值) → 低分（負獎勵，梯度陡峭）
    - 能偵測所有物體（牆壁 + 障礙物），因為使用 LiDAR 射線

    權重由 RewardManager config 控制，不在函數內部。

    Args:
        env: 環境實例
        sensor_cfg: LiDAR 傳感器配置 (SceneEntityCfg)
        lidar_range: LiDAR 最大探測距離（用於 clamp 上限）

    Returns:
        獎勵張量，形狀為 (num_envs,)
    """
    # 獲取 LiDAR 傳感器
    if sensor_cfg is None:
        sensor = env.scene.sensors["lidar"]
    elif isinstance(sensor_cfg, SceneEntityCfg):
        sensor = env.scene.sensors[sensor_cfg.name]
    else:
        sensor = sensor_cfg

    # 正確計算 2D 距離：ray_hits_w 是世界座標，需與 pos_w 做差再取 norm
    sensor_pos_2d = sensor.data.pos_w[:, :2]  # [num_envs, 2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]  # [num_envs, num_rays, 2]
    distances_2d = torch.norm(
        hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1
    )  # [num_envs, num_rays]

    # 處理 NaN/inf（射線未命中時）
    distances_2d = torch.nan_to_num(
        distances_2d, nan=lidar_range, posinf=lidar_range
    )

    # Clamp 到合理範圍
    valid_distances = distances_2d.clamp(min=1e-6, max=lidar_range)

    # 原版 NavRL Eq.9: R_ss = mean(log(distance))
    log_rewards = torch.log(valid_distances)

    # 平均所有射線，返回 (num_envs,)
    reward = log_rewards.mean(dim=-1)

    return reward


def navrl_dynamic_obstacle_safety_reward(
    env: ManagerBasedRLEnv,
    obstacle_prefix: str = "obstacle_",
    num_dynamic_obstacles: int = 8,
    robot_cfg: SceneEntityCfg = None,
    lidar_range: float = 4.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """動態障礙物安全獎勵

    與 NavRL 類似，使用最近障礙物的距離對數作為獎勵

    Args:
        env: 環境實例
        obstacle_prefix: 障礙物前綴
        num_dynamic_obstacles: 動態障礙物數量
        robot_cfg: 機器人配置 (SceneEntityCfg)
        lidar_range: LiDAR 最大探測距離（用於超出範圍的障礙物）
        weight: 獎勵權重

    Returns:
        獎勵張量，形狀為 (num_envs,)
    """
    # 獲取機器人資產
    if robot_cfg is None:
        robot = env.scene["robot"]
    elif isinstance(robot_cfg, SceneEntityCfg):
        robot = env.scene[robot_cfg.name]
    else:
        robot = robot_cfg

    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2)

    # 初始化最近距離
    closest_distances = torch.full((env.num_envs,), lidar_range,
                                   device=env.device, dtype=torch.float32)

    # 遍歷動態障礙物
    for i in range(num_dynamic_obstacles):
        obstacle_name = f"{obstacle_prefix}{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue

        obstacle = env.scene[obstacle_name]
        obs_pos = obstacle.data.root_pos_w[:, :2]

        # 計算距離
        distances = torch.norm(obs_pos - robot_pos, dim=-1)

        # 取最小距離
        closest_distances = torch.minimum(closest_distances, distances)

    # 對數獎勵
    valid_distances = closest_distances.clamp(min=1e-6, max=lidar_range)
    reward = torch.log(valid_distances) * weight

    return reward


def navrl_smoothness_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """NavRL 風格的動作平滑度懲罰

    P_smooth = ||v_t - v_{t-1}||

    懲罰速度變化，鼓勵平滑移動。
    回傳 >= 0 的值，權重由 RewardManager config 控制。

    Args:
        env: 環境實例
        robot_cfg: 機器人配置 (SceneEntityCfg)

    Returns:
        懲罰張量，形狀為 (num_envs,)，值 >= 0
    """
    # 獲取機器人資產
    if robot_cfg is None:
        robot = env.scene["robot"]
    elif isinstance(robot_cfg, SceneEntityCfg):
        robot = env.scene[robot_cfg.name]
    else:
        robot = robot_cfg

    # 當前速度
    current_vel = robot.data.root_vel_w[:, :2]

    # 上一步速度（從緩存獲取）
    if not hasattr(env, "_navrl_prev_vel"):
        env._navrl_prev_vel = torch.zeros_like(current_vel)

    prev_vel = env._navrl_prev_vel

    # 計算速度變化
    vel_diff = current_vel - prev_vel
    penalty = torch.norm(vel_diff, dim=-1)

    # 更新緩存（使用 nan_to_num 防止 NaN 傳播到下一步）
    env._navrl_prev_vel = torch.nan_to_num(current_vel.clone(), nan=0.0)

    # NaN 防護
    penalty = torch.nan_to_num(penalty, nan=0.0)

    return penalty


def navrl_height_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: object = None,
    target_height_min: float = 0.2,
    target_height_max: float = 0.5,
    weight: float = 8.0,
) -> torch.Tensor:
    """高度懲罰（車輛版，簡化版）

    懲罰機器人偏離預定高度範圍

    對於車輛，我們只需要確保它在地面高度，不需要高度懲罰。
    這個函數返回零懲罰，保持與 NavRL API 一致。

    Args:
        env: 環境實例
        robot_cfg: 機人配置
        target_height_min: 目標高度最小值
        target_height_max: 目標高度最大值
        weight: 懲罰權重

    Returns:
        懲罰張量，形狀為 (num_envs,)，全部為 0（車輛不需要高度懲罰）
    """
    # 車輛在地面上，不需要高度懲罰
    num_envs = env.num_envs
    return torch.zeros((num_envs,), device=env.device, dtype=torch.float32)


def navrl_total_reward(
    env: ManagerBasedRLEnv,
    # 獎勵組成
    velocity_weight: float = 1.0,
    safety_static_weight: float = 1.0,
    safety_dynamic_weight: float = 1.0,
    smoothness_weight: float = 0.1,
    # 配置
    lidar_range: float = 4.0,
    num_dynamic_obstacles: int = 8,
    # 獎勵裁剪
    clip_reward: bool = True,
    max_reward: float = 10.0,
    min_reward: float = -10.0,
) -> torch.Tensor:
    """NavRL 總獎勵函數

    完整實現 NavRL 論文的獎勵函數：

    R_total = R_vel + 1.0
             + R_safety_static * 1.0
             + R_safety_dynamic * 1.0
             - P_smooth * 0.1
             - P_height * 8.0

    Args:
        env: 環境實例
        velocity_weight: 速度獎勵權重
        safety_static_weight: 靜態障礙物安全獎勵權重
        safety_dynamic_weight: 動態障礙物安全獎勵權重
        smoothness_weight: 平滑度懲罰權重
        lidar_range: LiDAR 最大探測距離
        num_dynamic_obstacles: 動態障礙物數量
        clip_reward: 是否裁剪獎勵
        max_reward: 最大獎勵
        min_reward: 最小獎勵

    Returns:
        總獎勵張量，形狀為 (num_envs, 1)
    """
    # 各個獎勵組件
    reward_vel = navrl_velocity_reward(env)
    reward_safety_static = navrl_safety_reward_lidar(env, lidar_range=lidar_range)

    # 檢查是否有動態障礙物
    has_dynamic = num_dynamic_obstacles > 0
    if has_dynamic:
        reward_safety_dynamic = navrl_dynamic_obstacle_safety_reward(
            env,
            num_dynamic_obstacles=num_dynamic_obstacles,
            lidar_range=lidar_range
        )
    else:
        reward_safety_dynamic = torch.zeros_like(reward_safety_static)

    penalty_smooth = navrl_smoothness_penalty(env)
    penalty_height = navrl_height_penalty(env)  # 車輛版本為 0

    # 總獎勵
    total_reward = (
        reward_vel * velocity_weight
        + reward_safety_static * safety_static_weight
        + reward_safety_dynamic * safety_dynamic_weight
        - penalty_smooth * smoothness_weight
        - penalty_height * 8.0
    )

    # 裁剪獎勵
    if clip_reward:
        total_reward = total_reward.clamp(min=min_reward, max=max_reward)

    return total_reward


__all__ = [
    "navrl_velocity_reward",
    "navrl_safety_reward_lidar",
    "navrl_dynamic_obstacle_safety_reward",
    "navrl_smoothness_penalty",
    "navrl_height_penalty",
    "navrl_total_reward",
]
