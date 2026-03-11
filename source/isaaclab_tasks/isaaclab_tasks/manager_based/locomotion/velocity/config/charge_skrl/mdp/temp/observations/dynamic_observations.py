"""動態環境觀測函數 (Dynamic Environment Observations)

用於 Stage 4：動態與未知環境
包含 Frame Stamping 和動作預測功能
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ObservationTermCfg


def frame_stack_lidar(env: ManagerBasedRLEnv, term_cfg: ObservationTermCfg, stack_size: int = 4) -> torch.Tensor:
    """LiDAR Frame Stacking：堆疊過去幀的 LiDAR 數據

    這讓 RL 能夠：
    1. 感知障礙物的運動方向和速度
    2. 預測障礙物的未來位置
    3. 補償 AIT* 全域規劃的更新延遲

    Args:
        env: 環境實例
        term_cfg: 觀測配置
        stack_size: 堆疊幀數（默認 4 幀）

    Returns:
        shape [num_envs, 72 * stack_size]：堆疊後的 LiDAR 數據
    """
    # 從環境獲取當前 LiDAR 數據
    # 假設 lidar_scan 是 72 維
    lidar_current = env.sensors[term_cfg.params["sensor_name"]].data.out_hits

    # 計算距離並歸一化
    lidar_pos = env.sensors[term_cfg.params["sensor_name"]].data.pos_w[:, :2]
    hit_points = lidar_current[:, :, :2]
    distances = torch.norm(hit_points - lidar_pos.unsqueeze(1), dim=-1)
    max_dist = env.sensors[term_cfg.params["sensor_name"]].cfg.max_distance
    normalized_lidar = 1.0 - (distances / max_dist)

    # 獲取歷史幀（從環境緩衝區）
    if not hasattr(env, "_lidar_history"):
        env._lidar_history = []
    history = env._lidar_history

    # 將當前幀加入歷史
    history.append(normalized_lidar.clone())

    # 保持歷史長度
    if len(history) > stack_size:
        history.pop(0)

    # 填充（如果歷史不足）
    while len(history) < stack_size:
        history.insert(0, normalized_lidar.clone())

    # 堆疊幀
    stacked = torch.stack(history, dim=1)  # [num_envs, stack_size, 72]
    flattened = torch.flatten(stacked, start_dim=1)  # [num_envs, 72 * stack_size]

    return flattened


def obstacle_velocity_hint(
    env: ManagerBasedRLEnv,
    term_cfg: ObservationTermCfg,
) -> torch.Tensor:
    """障礙物速度提示：計算障礙物的運動趨勢

    這是一個簡化的實現，通過比較前後幀的 LiDAR 數據來估計運動。
    對於完整的動態障礙物，應該直接從環境獲取障礙物速度。

    Args:
        env: 環境實例
        term_cfg: 觀測配置

    Returns:
        shape [num_envs, 4]：[left_approaching, right_approaching, front_approaching, avg_speed]
    """
    sensor_name = term_cfg.params["sensor_name"]
    lidar_current = env.sensors[sensor_name].data.out_hits

    # 計算距離
    lidar_pos = env.sensors[sensor_name].data.pos_w[:, :2]
    hit_points = lidar_current[:, :, :2]
    distances = torch.norm(hit_points - lidar_pos.unsqueeze(1), dim=-1)

    # 獲取歷史距離
    if hasattr(env, "_lidar_distances_history"):
        prev_distances = env._lidar_distances_history
        # 計算距離變化（正值 = 靠近，負值 = 遠離）
        distance_changes = prev_distances - distances

        # 計算各方向的接近趨勢
        num_rays = distances.shape[1]
        quarter = num_rays // 4

        left_zone = slice(0, quarter)
        right_zone = slice(quarter * 3, num_rays)
        front_zone = slice(quarter, quarter * 3)

        left_approaching = torch.mean(distance_changes[:, left_zone], dim=1)
        right_approaching = torch.mean(distance_changes[:, right_zone], dim=1)
        front_approaching = torch.mean(distance_changes[:, front_zone], dim=1)

        # 平均接近速度
        avg_speed = torch.mean(torch.abs(distance_changes), dim=1)

        # 組合特徵
        features = torch.stack([
            torch.clamp(left_approaching / 0.5, 0, 1),    # 左側接近程度
            torch.clamp(right_approaching / 0.5, 0, 1),   # 右側接近程度
            torch.clamp(front_approaching / 0.5, 0, 1),   # 前方接近程度
            torch.clamp(avg_speed / 1.0, 0, 1),           # 整體運動程度
        ], dim=1)

    else:
        # 沒有歷史數據，返回零
        features = torch.zeros(env.num_envs, 4, device=env.device)

    # 保存當前距離到歷史
    env._lidar_distances_history = distances.clone()

    return features


__all__ = [
    "frame_stack_lidar",
    "obstacle_velocity_hint",
]
