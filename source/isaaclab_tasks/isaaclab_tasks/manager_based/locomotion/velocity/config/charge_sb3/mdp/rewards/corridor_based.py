"""
走廊式獎勵函數 (Corridor-Based Rewards)

解決「Cross-track Error 懲罰」與「避障」衝突的問題。

核心思想：
- 不懲罰機器人偏離「線」
- 而是設定一個「走廊」寬度
- 只要在走廊內移動，就能得到獎勵
- **活著比聽話更重要**：當風險高時，降低路徑跟隨權重
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class CorridorRewardConfig:
    """走廊獎勵配置

    Attributes:
        corridor_half_width: 走廊半寬（米）
        progress_weight: 前進獎勵權重
        corridor_bonus: 在走廊內的固定獎勵
        out_of_corridor_penalty: 超出走廊的懲罰（每米）
        collision_risk_threshold: 碰撞風險閾值（距離障礙物小於此值視為高風險）
        dynamic_weighting: 是否啟用動態權重（高風險時降低跟隨權重）
        min_corridor_width_dynamic: 動態情況下的最小走廊寬度
    """
    corridor_half_width: float = 0.5  # 默認 ±0.5m 走廊
    progress_weight: float = 1.0
    corridor_bonus: float = 0.1
    out_of_corridor_penalty: float = 2.0  # 每超出 1m 懲罰 2
    collision_risk_threshold: float = 0.8  # 距離障礙物 < 0.8m 視為高風險
    dynamic_weighting: bool = True  # 啟用動態權重調整
    min_corridor_width_dynamic: float = 1.0  # 高風險時走廊寬度擴大到 ±1.0m


def corridor_following_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    cfg: Optional[CorridorRewardConfig] = None,
    obstacle_distances: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """走廊式路徑跟隨獎勵

    與傳統 cross-track error 不同：
    - 在走廊內：給固定 bonus
    - 超出走廊：按超出距離懲罰
    - 高風險時：自動放寬走廊要求

    Args:
        env: 環境實例
        path_points: [num_envs, N, 2] 或 [N, 2] 路徑點
        robot_cfg: 機器人配置名稱
        cfg: 走廊獎勵配置
        obstacle_distances: [num_envs] 到最近障礙物的距離（用於動態權重）

    Returns:
        [num_envs] 走廊跟隨獎勵
    """
    from isaaclab.assets import Articulation

    if cfg is None:
        cfg = CorridorRewardConfig()

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑
    if path_points.dim() == 2:
        path = path_points
    else:
        path = path_points[0]

    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        # 確定當前走廊寬度
        current_corridor_width = cfg.corridor_half_width

        # 動態權重：如果靠近障礙物，放寬走廊要求
        if cfg.dynamic_weighting and obstacle_distances is not None:
            if obstacle_distances[env_id] < cfg.collision_risk_threshold:
                current_corridor_width = cfg.min_corridor_width_dynamic

        # 找到最近的路徑段
        min_lateral = float('inf')
        min_lateral_signed = 0.0

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment = p2 - p1
            segment_length_sq = torch.sum(segment ** 2)

            if segment_length_sq < 1e-6:
                continue

            robot_to_p1 = robot_pos[env_id] - p1
            t = torch.clamp(
                torch.sum(robot_to_p1 * segment) / segment_length_sq,
                min=0.0, max=1.0
            )

            projection = p1 + t * segment
            lateral_vec = robot_pos[env_id] - projection
            lateral_dist = torch.norm(lateral_vec)

            # 判斷左右（叉積）
            cross = segment[0] * lateral_vec[1] - segment[1] * lateral_vec[0]
            lateral_signed = lateral_dist if cross > 0 else -lateral_dist

            if abs(lateral_signed) < abs(min_lateral_signed):
                min_lateral_signed = lateral_signed.item()
                min_lateral = abs(lateral_signed)

        # 計算獎勵
        if min_lateral <= current_corridor_width:
            # 在走廊內
            rewards[env_id] = cfg.corridor_bonus
        else:
            # 超出走廊，按超出距離懲罰
            excess = min_lateral - current_corridor_width
            rewards[env_id] = -cfg.out_of_corridor_penalty * excess

    return rewards


def dynamic_corridor_width(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    obstacle_distances: torch.Tensor,
    robot_cfg: str = "robot",
    base_width: float = 0.5,
    max_width: float = 2.0,
    risk_threshold: float = 0.8,
) -> torch.Tensor:
    """動態走廊寬度

    根據障礙物距離動態調整走廊寬度：
    - 安全時：使用 base_width
    - 危險時：擴大到 max_width

    Args:
        env: 環境實例
        path_points: 路徑點
        obstacle_distances: [num_envs] 到障礙物距離
        robot_cfg: 機器人配置
        base_width: 基礎走廊寬度（半寬）
        max_width: 最大走廊寬度（半寬）
        risk_threshold: 風險閾值

    Returns:
        [num_envs] 動態走廊寬度（半寬）
    """
    num_envs = obstacle_distances.shape[0]
    device = obstacle_distances.device

    # 計算風險因子 (0 = 安全, 1 = 非常危險)
    risk_factor = torch.clamp(
        (risk_threshold - obstacle_distances) / risk_threshold,
        min=0.0, max=1.0
    )

    # 動態寬度
    dynamic_width = base_width + risk_factor * (max_width - base_width)

    return dynamic_width


def corridor_guided_total_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    previous_progress: Optional[torch.Tensor] = None,
    obstacle_distances: Optional[torch.Tensor] = None,
    cfg: Optional[CorridorRewardConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """走廊引導的綜合獎勵

    解決獎勵衝突：
    - 當風險高時，大幅降低路徑跟隨權重
    - 優先考慮生存（避障）而非跟隨

    Args:
        env: 環境實例
        path_points: 路徑點
        robot_cfg: 機器人配置
        previous_progress: 上次進度
        obstacle_distances: 到障礙物距離（用於動態權重）
        cfg: 走廊配置

    Returns:
        ([num_envs] 總獎勵, [num_envs] 當前進度)
    """
    from isaaclab.assets import Articulation
    from .aitstar_guided import aitstar_progress_reward

    if cfg is None:
        cfg = CorridorRewardConfig()

    # 1. 計算進步獎勵
    progress_reward, current_progress = aitstar_progress_reward(
        env, path_points, robot_cfg, previous_progress
    )

    # 2. 計算走廊獎勵
    corridor_reward = corridor_following_reward(
        env, path_points, robot_cfg, cfg, obstacle_distances
    )

    # 3. 動態調整進步權重
    progress_weights = torch.ones(env.num_envs, device=progress_reward.device)

    if cfg.dynamic_weighting and obstacle_distances is not None:
        for i in range(env.num_envs):
            if obstacle_distances[i] < cfg.collision_risk_threshold:
                # 高風險：降低進步權重，讓專注於避障
                progress_weights[i] = 0.3

    # 綜合獎勵
    total_reward = (
        progress_weights * cfg.progress_weight * progress_reward +
        corridor_reward
    )

    return total_reward, current_progress


def survival_priority_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    obstacle_distances: torch.Tensor,
    robot_cfg: str = "robot",
    safe_distance: float = 1.0,
    danger_distance: float = 0.5,
) -> torch.Tensor:
    """生存優先的獎勵調整

    當接近障礙物時，強制降低所有路徑跟隨相關獎勵的權重。
    「活著比聽話更重要！」

    Args:
        env: 環境實例
        path_points: 路徑點
        obstacle_distances: [num_envs] 到障礙物距離
        robot_cfg: 機器人配置
        safe_distance: 安全距離
        danger_distance: 危險距離

    Returns:
        [num_envs] 權重調整因子 [0, 1]
    """
    num_envs = obstacle_distances.shape[0]
    device = obstacle_distances.device

    # 計算調整因子
    # safe_distance 以外：factor = 1.0 (正常)
    # danger_distance 以內：factor = 0.2 (大幅降低)
    weights = torch.zeros(num_envs, device=device)

    for i in range(num_envs):
        dist = obstacle_distances[i]

        if dist >= safe_distance:
            weights[i] = 1.0
        elif dist <= danger_distance:
            weights[i] = 0.2
        else:
            # 線性插值
            t = (dist - danger_distance) / (safe_distance - danger_distance)
            weights[i] = 0.2 + 0.8 * t

    return weights


@dataclass
class AdaptiveRewardWeights:
    """自適應獎勵權重

    根據當前狀態動態調整各項獎勵的權重。
    """
    # 基礎權重
    base_progress: float = 1.0
    base_corridor: float = 1.0
    base_velocity: float = 0.1
    base_alignment: float = 0.3

    # 安全情況下的權重
    safe_progress: float = 1.0
    safe_corridor: float = 0.5
    safe_velocity: float = 0.1
    safe_alignment: float = 0.3

    # 危險情況下的權重（優先避障）
    danger_progress: float = 0.2  # 降低進步權重
    danger_corridor: float = 0.1  # 降低跟隨權重
    danger_velocity: float = 0.0  # 不關心速度
    danger_alignment: float = 0.0  # 不關心方向

    # 權重插值速度
    interpolation_rate: float = 0.1

    def get_weights(
        self,
        obstacle_distances: torch.Tensor,
        safe_threshold: float = 1.0,
        danger_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """根據障礙物距離獲取當前權重

        Args:
            obstacle_distances: [num_envs] 障礙物距離
            safe_threshold: 安全閾值
            danger_threshold: 危險閾值

        Returns:
            權重字典
        """
        num_envs = obstacle_distances.shape[0]
        device = obstacle_distances.device

        # 計算危險因子 [0, 1]
        danger_factor = torch.zeros(num_envs, device=device)

        for i in range(num_envs):
            dist = obstacle_distances[i]
            if dist >= safe_threshold:
                danger_factor[i] = 0.0
            elif dist <= danger_threshold:
                danger_factor[i] = 1.0
            else:
                # 線性插值
                danger_factor[i] = 1.0 - (dist - danger_threshold) / (safe_threshold - danger_threshold)

        # 計算權重
        weights = {
            'progress': torch.zeros(num_envs, device=device),
            'corridor': torch.zeros(num_envs, device=device),
            'velocity': torch.zeros(num_envs, device=device),
            'alignment': torch.zeros(num_envs, device=device),
        }

        for name in weights.keys():
            base_val = getattr(self, f'base_{name}')
            safe_val = getattr(self, f'safe_{name}')
            danger_val = getattr(self, f'danger_{name}')

            # 線性插值
            weights[name] = safe_val + danger_factor * (danger_val - safe_val)

        return weights


__all__ = [
    "CorridorRewardConfig",
    "corridor_following_reward",
    "dynamic_corridor_width",
    "corridor_guided_total_reward",
    "survival_priority_reward",
    "AdaptiveRewardWeights",
]
