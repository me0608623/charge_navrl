"""感知感知獎勵 (Perception-Aware Rewards)

用於 Stage 2：強迫 RL 使用 LiDAR 數據而非僅依賴路徑追蹤。
這能防止 Agent 過度依賴全域規劃器 (AIT*)。

設計理念：
- 路徑是參考，LiDAR 才是保命符
- 離障礙物太近時給懲罰，即使走在「正確」的路徑上
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def lidar_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
    comfort_distance: float = 1.5,
    min_distance_for_reward: float = 0.5,
) -> torch.Tensor:
    """LiDAR 清晰度獎勵：獎勵與障礙物保持安全距離

    即使走在「正確」的路徑上，如果太靠近障礙物也會被懲罰。
    這能教會 RL：「路徑是參考，LiDAR 才是保命符」

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        safe_distance: 安全距離（米），超過此距離給獎勵
        comfort_distance: 舒適距離（米），最佳獎勵距離
        min_distance_for_reward: 最小獎勵距離（米），太近不給獎勵

    Returns:
        shape [num_envs]：獎勵值 [0, 1]
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取雷達數據
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]

    # 計算獎勵
    # 距離 < min_distance_for_reward：無獎勵（太危險）
    # min_distance_for_reward < 距離 < comfort_distance：線性增加
    # 距離 >= comfort_distance：滿分獎勵

    reward = torch.zeros_like(min_distance)

    # 安全區域：給予獎勵
    safe_mask = min_distance >= min_distance_for_reward
    if safe_mask.any():
        # 🔥 添加 epsilon 保護（防止 comfort_distance == min_distance_for_reward 時除以零）
        normalized_dist = (min_distance[safe_mask] - min_distance_for_reward) / (comfort_distance - min_distance_for_reward + 1e-6)
        reward[safe_mask] = torch.clamp(normalized_dist, 0.0, 1.0)

    # 太靠近障礙物：給予懲罰（即使走對路徑）
    danger_mask = min_distance < min_distance_for_reward
    if danger_mask.any():
        reward[danger_mask] = -0.5  # 固定懲罰

    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
    reward = torch.clamp(reward, -1.0, 1.0)
    reward = _check_reward_term("lidar_clearance_reward", reward, env, raise_on_error=False)

    return reward


def too_close_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.2,
    penalty: float = -1.0,
) -> torch.Tensor:
    """太靠近懲罰：非常接近障礙物時給予懲罰

    這是一個「硬邊界」懲罰，防止 Agent 貼著障礙物邊緣擦過去。

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        threshold: 臨界距離（米），低於此距離給予懲罰
        penalty: 懲罰值

    Returns:
        shape [num_envs]：懲罰值
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )
    min_distance = torch.min(distances_2d, dim=1)[0]

    # 太靠近時給懲罰
    too_close = min_distance < threshold
    reward = torch.where(too_close, torch.tensor(penalty, device=env.device), torch.zeros_like(min_distance))

    reward = _check_reward_term("too_close_penalty", reward, env, raise_on_error=False)

    return reward


def lidar_utilization_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    action_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """LiDAR 使用獎勵：根據 Agent 是否根據 LiDAR 數據調整行為給予獎勵

    這是一個高階獎勵，檢測 Agent 是否實際使用 LiDAR 數據：
    - 當 LiDAR 顯示前方有障礙物時，Agent 是否減速或轉向
    - 當 LiDAR 顯示前方清空時，Agent 是否加速

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        action_cfg: 機器人動作配置（用於獲取速度）

    Returns:
        shape [num_envs]：獎勵值 [0, 1]
    """
    from isaaclab.assets import Articulation

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[action_cfg.name]

    # 獲取 LiDAR 數據
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )

    # 計算前方扇形區域的平均距離（假設前 36 條射線是前方 180 度）
    front_distances = distances_2d[:, :36]  # 前半部分
    avg_front_distance = torch.mean(front_distances, dim=1)

    # 獲取機器人速度
    robot_vel_b = asset.data.root_lin_vel_b  # [num_envs, 3]
    forward_speed = robot_vel_b[:, 0]  # 前進速度（機器人座標系）

    # 獎勵邏輯：
    # 1. 前方有障礙物且減速 → 獎勵
    # 2. 前方清空且加速 → 獎勵
    # 3. 前方有障礙物但仍加速 → 懲罰

    reward = torch.zeros(env.num_envs, device=env.device)

    # 前方有障礙物時
    obstacle_ahead = avg_front_distance < 3.0
    slowing_down = forward_speed < 0.5

    reward[obstacle_ahead & slowing_down] = 0.5  # 減速獎勵
    reward[obstacle_ahead & (~slowing_down)] = -0.5  # 沒減速懲罰

    # 前方清空時
    clear_ahead = avg_front_distance > 5.0
    speeding_up = forward_speed > 0.8

    reward[clear_ahead & speeding_up] = 0.3  # 加速獎勵

    reward = torch.clamp(reward, -1.0, 1.0)
    reward = _check_reward_term("lidar_utilization_bonus", reward, env, raise_on_error=False)

    return reward


def safety_field_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    collision_threshold: float = 0.25,
    safe_distance: float = 0.6,
) -> torch.Tensor:
    """安全場懲罰 (Safety Field Penalty) - Push Force

    實現 Pull-Push Theory 中的「推力」：
    - 當障礙物距離 < collision_threshold (0.25m) → 最大懲罰 -1.0
    - 當障礙物距離 > safe_distance (0.6m) → 無懲罰 0.0
    - 中間距離線性插值

    公式：
    ```
    P_safety = -1.0 × (safe_distance - d) / (safe_distance - collision_threshold)
    ```
    其中 d 是 LiDAR 最小距離

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        collision_threshold: 碰撞臨界距離（米），低於此距離給予最大懲罰
        safe_distance: 安全距離（米），超過此距離無懲罰

    Returns:
        shape [num_envs]：懲罰值 [-1.0, 0.0]
        -1.0 = 碰撞危險（距離 < 0.25m）
        0.0 = 安全（距離 > 0.6m）
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取雷達數據
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]

    # 計算懲罰：線性插值
    # d < collision_threshold → -1.0 (最大懲罰)
    # d > safe_distance → 0.0 (無懲罰)
    # 中間線性插值
    penalty = torch.where(
        min_distance < collision_threshold,
        torch.full_like(min_distance, -1.0),  # 碰撞危險：最大懲罰
        torch.where(
            min_distance < safe_distance,
            # 中間距離：線性插值 [0, safe_distance - collision_threshold]
            -(safe_distance - min_distance) / (safe_distance - collision_threshold + 1e-6),
            torch.zeros_like(min_distance)  # 安全距離：無懲罰
        )
    )

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-1.0)
    penalty = torch.clamp(penalty, -1.0, 0.0)
    penalty = _check_reward_term("safety_field_penalty", penalty, env, raise_on_error=False)

    return penalty


def collision_penalty_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.25,
) -> torch.Tensor:
    """連續碰撞懲罰 (Continuous Collision Penalty)

    當 LiDAR 檢測到距離 < threshold 時給予懲罰 -1.0。
    這是一個「每步檢查」的懲罰，與終止條件分開。

    設計理念：
    - 終止條件：碰撞後立即終止（不給予更多學習機會）
    - 連續懲罰：在每一步檢查接近程度，給予負獎勵

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        threshold: 臨界距離（米），低於此距離給予懲罰

    Returns:
        shape [num_envs]：懲罰值
        -1.0 = 危險接近
        0.0 = 安全
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )
    min_distance = torch.min(distances_2d, dim=1)[0]

    # 危險接近時給予懲罰
    too_close = min_distance < threshold
    reward = torch.where(too_close, torch.tensor(-1.0, device=env.device), torch.zeros_like(min_distance))

    reward = _check_reward_term("collision_penalty_reward", reward, env, raise_on_error=False)

    return reward


def wall_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    penalty_scale: float = 5.0,
) -> torch.Tensor:
    """牆壁過近懲罰 (Wall Proximity Penalty)

    避免機器人在空房間裡產生「只敢走中間」的偏差行為 (Center Bias)。

    邏輯：
        if min_lidar < threshold:
            reward = -penalty_scale * (threshold - min_lidar)
        else:
            reward = 0

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        threshold: 臨界距離（米），低於此距離開始懲罰
        penalty_scale: 懲罰強度係數

    Returns:
        shape [num_envs]：懲罰值 [≤0, 0]
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取雷達數據
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )
    min_distance = torch.min(distances_2d, dim=1)[0]

    # 計算懲罰：只在距離 < threshold 時懲罰
    too_close = min_distance < threshold
    penalty = torch.where(
        too_close,
        -penalty_scale * (threshold - min_distance),
        torch.zeros_like(min_distance)
    )

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-10.0)
    penalty = torch.clamp(penalty, -5.0, 0.0)

    return penalty


__all__ = [
    "lidar_clearance_reward",
    "too_close_penalty",
    "lidar_utilization_bonus",
    "safety_field_penalty",
    "collision_penalty_reward",
    "wall_proximity_penalty",
]
