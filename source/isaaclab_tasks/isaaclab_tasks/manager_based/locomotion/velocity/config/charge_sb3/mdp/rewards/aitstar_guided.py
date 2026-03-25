"""
AIT* 引導獎勵函數 (AIT*-Guided Rewards)

專為 AIT* + RL 層級式導航設計的獎勵函數：
- Progress Along Path: 沿 AIT* 路徑前進獎勵
- Cross-Track Error: 偏離路徑懲罰
- Path Alignment: 與路徑方向一致性
- Dynamic Goal Adjustment: 根據路徑進度調整目標權重

獎勵公式：
    R = w1 * progress - w2 * cte - w3 * collision + w4 * alignment
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class AITStarRewardWeights:
    """AIT* 獎勵權重配置

    Attributes:
        progress: 沿路徑前進獎勵權重
        cross_track: 偏離路徑懲罰權重
        collision: 碰撞懲罰權重
        alignment: 方向一致性獎勵權重
        velocity: 速度獎勵權重（鼓勵移動）
        time_penalty: 時間懲罰權重（鼓勵快速完成）
    """
    progress: float = 1.0
    cross_track: float = 0.5
    collision: float = 10.0
    alignment: float = 0.3
    velocity: float = 0.1
    time_penalty: float = 0.01


def aitstar_progress_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    previous_progress: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿 AIT* 路徑前進的獎勵

    計算機器人沿著 AIT* 生成路徑的前進距離增量。
    使用「最近路徑點投影」方法。

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱
        previous_progress: [num_envs] 上一時刻的進度值

    Returns:
        ([num_envs] 進步獎勵, [num_envs] 當前進度值)
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    # 計算累積路徑長度
    path_lengths = torch.zeros(len(path), device=device)
    for i in range(1, len(path)):
        path_lengths[i] = path_lengths[i-1] + torch.norm(path[i] - path[i-1])

    # 找到每個機器人的最近路徑點
    path_expanded = path.unsqueeze(0)
    robot_expanded = robot_pos.unsqueeze(1)
    distances = torch.norm(path_expanded - robot_expanded, dim=2)
    nearest_idx = torch.argmin(distances, dim=1)

    # 計算進度（考慮到最近點的累積路徑長度）
    current_progress = torch.zeros(num_envs, device=device)
    for i in range(num_envs):
        idx = nearest_idx[i]
        current_progress[i] = path_lengths[idx].item()

    # 計算進步（與上一時刻的差值）
    if previous_progress is None:
        progress_reward = torch.zeros(num_envs, device=device)
    else:
        progress_delta = current_progress - previous_progress
        progress_reward = torch.clamp(progress_delta, min=-1.0, max=2.0)

    return progress_reward, current_progress


def aitstar_cross_track_error(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    safe_distance: float = 0.5,
) -> torch.Tensor:
    """偏離 AIT* 路徑的垂直距離懲罰

    使用線段到點的垂直距離公式計算 cross-track error。

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱
        safe_distance: 安全距離，在此範圍內不懲罰

    Returns:
        [num_envs] CTE 懲罰（負值）
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    cte = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        min_cte = float('inf')

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
            dist_to_segment = torch.norm(robot_pos[env_id] - projection)

            if dist_to_segment > safe_distance:
                adjusted_dist = dist_to_segment - safe_distance
            else:
                adjusted_dist = torch.tensor(0.0, device=device)

            min_cte = min(min_cte, adjusted_dist.item())

        cte[env_id] = -min_cte

    return cte


def aitstar_alignment_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
) -> torch.Tensor:
    """與 AIT* 路徑方向的一致性獎勵

    計算機器人移動方向與路徑切線方向的餘弦相似度。

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        [num_envs] 方向一致性獎勵 [-1, 1]
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 歸一化速度
    speed = torch.norm(robot_vel, dim=-1, keepdim=True)
    speed = torch.clamp(speed, min=1e-6)
    velocity_direction = robot_vel / speed

    # 獲取路徑
    if path_points.shape[0] == 1:
        path = path_points[0]
    else:
        path = path_points[0]

    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        if speed[env_id] < 1e-3:
            continue  # 機器人靜止，不給獎勵

        # 找最近路徑段
        min_dist = float('inf')
        best_direction = None

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment = p2 - p1
            segment_length = torch.norm(segment)

            if segment_length < 1e-6:
                continue

            segment_direction = segment / segment_length

            robot_to_p1 = robot_pos[env_id] - p1
            dist_to_segment = torch.norm(
                robot_to_p1 - torch.clamp(
                    torch.sum(robot_to_p1 * segment) / (segment_length ** 2),
                    min=0.0, max=1.0
                ) * segment
            )

            if dist_to_segment < min_dist:
                min_dist = dist_to_segment
                best_direction = segment_direction

        if best_direction is not None:
            direction_similarity = torch.sum(
                velocity_direction[env_id] * best_direction
            )
            rewards[env_id] = direction_similarity

    return rewards


def aitstar_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: str = "robot",
    target_speed: float = 1.0,
) -> torch.Tensor:
    """速度獎勵（鼓勵機器人以目標速度移動）

    Args:
        env: 環境實例
        robot_cfg: 機器人配置名稱
        target_speed: 目標速度 (m/s)

    Returns:
        [num_envs] 速度獎勵
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    speed = torch.norm(robot_vel, dim=-1)

    # 使用高斯函數：在目標速度附近給予高獎勵
    diff = speed - target_speed
    reward = torch.exp(-0.5 * (diff / 0.5) ** 2) - 0.5  # 最大值 0.5，最小值 -0.5

    return reward


def aitstar_guided_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    weights: Optional[AITStarRewardWeights] = None,
    previous_progress: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AIT* 引導的綜合獎勵

    結合多個獎勵組成部分，為 RL 提供清晰的学习信號。

    Args:
        env: 環境實例
        path_points: [num_envs, num_waypoints, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱
        weights: 獎勵權重配置
        previous_progress: [num_envs] 上一時刻的進度值

    Returns:
        ([num_envs] 總獎勵, [num_envs] 當前進度值)
    """
    if weights is None:
        weights = AITStarRewardWeights()

    # 1. 沿路徑前進獎勵
    progress_reward, current_progress = aitstar_progress_reward(
        env, path_points, robot_cfg, previous_progress
    )

    # 2. Cross-track error 懲罰
    cte = aitstar_cross_track_error(env, path_points, robot_cfg)

    # 3. 方向一致性獎勵
    alignment = aitstar_alignment_reward(env, path_points, robot_cfg)

    # 4. 速度獎勵
    velocity = aitstar_velocity_reward(env, robot_cfg)

    # 5. 時間懲罰
    time_penalty = -weights.time_penalty * torch.ones(env.num_envs, device=progress_reward.device)

    # 綜合獎勵
    total_reward = (
        weights.progress * progress_reward +
        weights.cross_track * cte +
        weights.alignment * alignment +
        weights.velocity * velocity +
        time_penalty
    )

    # 檢查碰撞（額外懲罰）
    # 這需要從環境中獲取碰撞信息
    # collision_penalty = detect_collisions(env) * weights.collision

    return total_reward, current_progress


def curriculum_adjusted_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_cfg: str = "robot",
    stage: int = 1,
    previous_progress: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根據課程階段調整獎勵權重

    不同階段使用不同的獎勵權重配置，
    幫助 RL 從基礎到高級逐步學習。

    Args:
        env: 環境實例
        path_points: AIT* 路徑點
        robot_cfg: 機器人配置名稱
        stage: 當前課程階段 (1-4)
        previous_progress: 上一時刻的進度值

    Returns:
        ([num_envs] 總獎勵, [num_envs] 當前進度值)
    """
    # 根據階段選擇權重
    stage_weights = {
        1: AITStarRewardWeights(
            progress=1.0,
            cross_track=0.2,
            collision=10.0,
            alignment=0.1,
            velocity=0.2,
            time_penalty=0.005,
        ),
        2: AITStarRewardWeights(
            progress=1.0,
            cross_track=0.5,
            collision=10.0,
            alignment=0.3,
            velocity=0.1,
            time_penalty=0.01,
        ),
        3: AITStarRewardWeights(
            progress=1.2,
            cross_track=0.8,
            collision=15.0,
            alignment=0.4,
            velocity=0.05,
            time_penalty=0.02,
        ),
        4: AITStarRewardWeights(
            progress=1.5,
            cross_track=1.0,
            collision=20.0,
            alignment=0.5,
            velocity=0.0,
            time_penalty=0.03,
        ),
    }

    weights = stage_weights.get(stage, stage_weights[1])

    return aitstar_guided_reward(
        env, path_points, robot_cfg, weights, previous_progress
    )


# 導出
__all__ = [
    "AITStarRewardWeights",
    "aitstar_progress_reward",
    "aitstar_cross_track_error",
    "aitstar_alignment_reward",
    "aitstar_velocity_reward",
    "aitstar_guided_reward",
    "curriculum_adjusted_reward",
]
