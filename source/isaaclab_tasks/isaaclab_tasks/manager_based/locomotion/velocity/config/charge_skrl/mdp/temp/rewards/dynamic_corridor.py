"""動態廊道獎勵 (Dynamic Corridor Rewards)

用於 Stage 3：窄門與精確控制
根據周圍障礙物密度動態調整容許的追蹤誤差。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def dynamic_corridor_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    path_cfg: SceneEntityCfg,
    base_corridor_width: float = 2.0,
    min_corridor_width: float = 0.5,
) -> torch.Tensor:
    """動態廊道獎勵：根據障礙物密度調整容許誤差

    在寬廣區域允許較大的路徑偏離，在窄門區域要求精確跟隨。

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        path_cfg: 路徑配置（用於獲取路徑位置）
        base_corridor_width: 基礎廊道寬度（米）
        min_corridor_width: 最小廊道寬度（米）

    Returns:
        shape [num_envs]：獎勵值
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 計算障礙物密度（周圍障礙物的緊密程度）
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )

    # 計算障礙物密度（近處障礙物越多，密度越高）
    close_obstacles = (distances_2d < 3.0).float()  # 3米內的障礙物
    obstacle_density = torch.sum(close_obstacles, dim=1) / distances_2d.shape[1]

    # 動態調整容許廊道寬度
    # 密度越高，允許的誤差越小
    density_factor = torch.clamp(obstacle_density / 5.0, 0.0, 1.0)
    dynamic_width = base_corridor_width * (1.0 - density_factor * 0.75)  # 最小到 base * 0.25
    dynamic_width = torch.clamp(dynamic_width, min_corridor_width, base_corridor_width)

    # TODO: 計算實際的路徑追蹤誤差（需要路徑數據）
    # 這裡先返回一個基於密度的獎勵

    # 寬敞區域：小幅獎勵（因為容易）
    # 窄窄區域：大幅獎勵（因為困難）
    reward = 0.1 * (1.0 - density_factor)  # 寬敞區域 0.1，窄區域接近 0

    reward = torch.clamp(reward, 0.0, 1.0)
    reward = _check_reward_term("dynamic_corridor_reward", reward, env, raise_on_error=False)

    return reward


def narrow_gate_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    narrow_threshold: float = 1.0,
    bonus: float = 1.0,
) -> torch.Tensor:
    """窄門通過獎勵：成功通過窄門區域時給予額外獎勵

    這是一個稀疏獎勵，用於激勵 Agent 克服「窄門恐懼」。

    Args:
        env: 環境實例
        sensor_cfg: 雷達感測器配置
        narrow_threshold: 窄門閾值（米），周圍空間小於此值視為窄門
        bonus: 通過獎勵值

    Returns:
        shape [num_envs]：獎勵值
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

    # 計算各方向的可用空間
    min_distance = torch.min(distances_2d, dim=1)[0]
    max_distance = torch.max(distances_2d, dim=1)[0]

    # 判斷是否在窄門區域（最短距離很近，但有至少一個方向是通暢的）
    in_narrow_gate = (min_distance < narrow_threshold) & (max_distance > narrow_threshold * 2)

    # 根據是否在窄門區域以及是否安全調整獎勵
    reward = torch.zeros(env.num_envs, device=env.device)

    # 在窄門區域且沒有碰撞 → 給予獎勵
    # 這裡簡化處理：實際應該檢查是否真的在通過窄門
    reward[in_narrow_gate] = bonus * 0.1  # 每步給予小幅獎勵

    reward = torch.clamp(reward, 0.0, bonus)
    reward = _check_reward_term("narrow_gate_bonus", reward, env, raise_on_error=False)

    return reward


__all__ = [
    "dynamic_corridor_reward",
    "narrow_gate_bonus",
]
