"""平滑獎勵函數 — 改進版加速度/角速度懲罰

與原始 potential_based_rewards.py 中的 discrete_acceleration_squared_penalty / angular_velocity_squared_penalty 差異：
- deadzone_acceleration_penalty: 微小加速度（|a| < dead_zone）不被懲罰，允許精細調整
- context_aware_angular_velocity_penalty: 靠近障礙物時降低轉彎懲罰，允許閃避
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def deadzone_acceleration_penalty(
    env: ManagerBasedRLEnv,
    dead_zone: float = 0.1,
) -> torch.Tensor:
    """Dead-zone 加速度懲罰：|a| < dead_zone 時零懲罰。

    Math:
        penalty = max(0, |applied_a| - dead_zone)²

    與 discrete_acceleration_squared_penalty（純 a²）的差異：
    微小加速度不被懲罰，agent 可以做精細速度調整而不付出代價。

    Args:
        env: 環境實例
        dead_zone: 免懲罰區間 (m/s²)

    Returns:
        [num_envs] — 非負值
    """
    term = list(env.action_manager._terms.values())[0]
    if hasattr(term, "applied_accelerations"):
        applied_a = term.applied_accelerations[:, 0]  # [N]
    else:
        applied_a = term.processed_actions[:, 0]

    excess = (applied_a.abs() - dead_zone).clamp(min=0.0)
    penalty = excess ** 2

    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    return penalty.clamp(0.0, 10.0)


def context_aware_angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    proximity_distance: float = 1.5,
    max_reduction: float = 0.7,
) -> torch.Tensor:
    """Context-aware 角速度懲罰：靠近障礙物時降低轉彎懲罰。

    Math:
        d_min = LiDAR 最近距離
        proximity_factor = clamp((proximity_distance - d_min) / proximity_distance, 0, max_reduction)
        penalty = ω² × (1 - proximity_factor)

    與 angular_velocity_squared_penalty（純 ω²）的差異：
    - 遠離障礙物：完整 ω² 懲罰（鼓勵直線）
    - 靠近障礙物：懲罰降低最多 max_reduction（允許閃避轉彎）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        sensor_cfg: LiDAR 感測器配置
        proximity_distance: 障礙物接近距離閾值 (m)
        max_reduction: 最大懲罰降低比例 (0-1)

    Returns:
        [num_envs] — 非負值
    """
    robot: Articulation = env.scene[robot_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 角速度
    omega_z = torch.nan_to_num(robot.data.root_ang_vel_w[:, 2], nan=0.0)  # [N]
    omega_sq = omega_z ** 2

    # LiDAR 最近距離
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance,
    )
    d_min = torch.min(distances_2d, dim=1)[0]  # [N]

    # 接近因子：0（遠離）→ max_reduction（靠近）
    proximity_factor = ((proximity_distance - d_min) / proximity_distance).clamp(0.0, max_reduction)

    # 調整後的懲罰
    penalty = omega_sq * (1.0 - proximity_factor)

    return penalty.clamp(0.0, 10.0)


__all__ = [
    "deadzone_acceleration_penalty",
    "context_aware_angular_velocity_penalty",
]
