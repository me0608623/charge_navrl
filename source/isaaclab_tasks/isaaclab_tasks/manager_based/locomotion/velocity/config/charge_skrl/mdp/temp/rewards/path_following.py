"""
路徑跟隨獎勵函數 (Path Following Rewards)

用於 A* + RL 層級式導航的獎勵函數：
- progress_along_path: 沿著 A* 路徑前進的距離
- cross_track_error: 偏離路徑的垂直距離
- path_direction_reward: 與路徑方向一致的獎勵

設計理念：
- A* (全局規劃器) 生成路徑
- RL (局部策略) 學習跟隨路徑
- 獎勵函數引導 RL 朝向路徑前進，同時懲罰偏離
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def progress_along_path(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_position_cfg: str = "robot",
) -> torch.Tensor:
    """沿著 A* 路徑前進的獎勵

    計算機器人沿著 A* 生成路徑的前進距離。
    使用「最近路徑點投影」方法：找到機器人在路徑上的投影點，
    計算從起點到投影點的累積距離。

    Args:
        env: 環境實例
        path_points: A* 生成的路徑點，shape [num_envs, num_path_points, 2]
        robot_position_cfg: 機器人配置名稱

    Returns:
        shape [num_envs]: 沿著路徑前進的距離獎勵

    計算公式：
        1. 找到機器人在路徑上的最近點
        2. 計算機器人到投影點的距離
        3. 返回前進距離（後退為負）
    """
    # 獲取機器人位置
    robot: Articulation = env.scene[robot_position_cfg]
    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    num_envs = robot_pos_xy.shape[0]
    device = robot_pos_xy.device

    # 初始化獎勵
    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]  # [num_path_points, 2]

        # 計算機器人到每個路徑點的距離
        distances = torch.norm(
            path - robot_pos_xy[env_id], dim=-1
        )  # [num_path_points]

        # 找到最近路徑點的索引
        closest_idx = torch.argmin(distances)
        closest_point = path[closest_idx]

        # 計算機器人到投影點的距離
        distance_to_path = distances[closest_idx]

        # 判斷是否前進（沿著路徑方向）
        if closest_idx > 0:
            # 有前一個路徑點，檢查是否更接近
            prev_point = path[closest_idx - 1]
            dist_to_prev = torch.norm(
                prev_point - robot_pos_xy[env_id], dim=-1
            )

            # 如果距離前一個點更近，說明在後退
            if dist_to_prev < distance_to_path:
                rewards[env_id] = distance_to_prev * 0.5  # 後退懲罰
            else:
                rewards[env_id] = distance_to_path  # 前進獎勵
        else:
            # 在第一個路徑點之前，簡單給予距離獎勵
            rewards[env_id] = distance_to_path

    return rewards


def cross_track_error(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_position_cfg: str = "robot",
    safe_distance: float = 0.5,
) -> torch.Tensor:
    """偏離路徑的垂直距離懲罰 (Cross-Track Error)

    計算機器人到 A* 路徑的垂直距離（偏差）。
    距離路徑越遠，懲罰越大。

    Args:
        env: 環境實例
        path_points: A* 生成的路徑點，shape [num_envs, num_path_points, 2]
        robot_position_cfg: 機器人配置名稱
        safe_distance: 安全距離（米），在此範圍內不懲罰

    Returns:
        shape [num_envs]: 偏離路徑的垂直距離（負值 = 懲罰）
        範圍：[-inf, 0]，越接近 0 表示懲罰越小

    計算方法：
        1. 將路徑段視為線段
        2. 計算點到線段的垂直距離
        3. 取最小距離作為 cross-track error
    """
    robot: Articulation = env.scene[robot_position_cfg]
    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    num_envs = robot_pos_xy.shape[0]
    device = robot_pos_xy.device

    # 初始化 cross-track error
    cte = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]  # [num_path_points, 2]
        min_cte = float('inf')

        # 遍歷每個路徑段
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            # 路徑段向量
            segment = p2 - p1
            segment_length_sq = torch.sum(segment ** 2)

            if segment_length_sq < 1e-6:
                # 路徑段太短，跳過
                continue

            # 機器人相對於 p1 的位置
            robot_to_p1 = robot_pos_xy[env_id] - p1

            # 投影到路徑段上的參數 t
            t = torch.clamp(
                torch.sum(robot_to_p1 * segment) / segment_length_sq,
                min=0.0, max=1.0
            )

            # 投影點
            projection = p1 + t * segment

            # 垂直距離（機器人到投影點）
            dist_to_segment = torch.norm(robot_pos_xy[env_id] - projection)

            # 如果在安全距離內，不懲罰
            if dist_to_segment > safe_distance:
                adjusted_dist = dist_to_segment - safe_distance
            else:
                adjusted_dist = torch.tensor(0.0, device=device)

            # 更新最小距離
            min_cte = min(min_cte, adjusted_dist.item())

        cte[env_id] = -min_cte  # 負值表示懲罰

    return cte


def path_direction_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_position_cfg: str = "robot",
    robot_velocity_cfg: str = "robot",
) -> torch.Tensor:
    """路徑方向一致性獎勵

    獎勵機器人的移動方向與 A* 路徑切線方向的一致性。
    方向越一致，獎勵越高。

    Args:
        env: 環境實例
        path_points: A* 生成的路徑點
        robot_position_cfg: 機器人位置配置
        robot_velocity_cfg: 機器人速度配置

    Returns:
        shape [num_envs]: 方向一致性獎勵 [-1, 1]
        1.0 = 完全一致，0.0 = 垂直，-1.0 = 相反
    """
    robot: Articulation = env.scene[robot_position_cfg]
    robot_pos_xy = robot.data.root_pos_w[:, :2]  # [num_envs, 2]

    # 獲取機器人速度方向
    robot_vel: Articulation = env.scene[robot_velocity_cfg]
    robot_vel_xy = robot_vel.data.root_lin_vel_w[:, :2]  # [num_envs, 2]

    # 歸一化速度向量
    speed = torch.norm(robot_vel_xy, dim=-1, keepdim=True)
    # 避免除以零
    speed = torch.clamp(speed, min=1e-6, max=None)

    velocity_direction = robot_vel_xy / speed  # [num_envs, 2]

    num_envs = robot_pos_xy.shape[0]
    device = robot_pos_xy.device

    rewards = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        path = path_points[env_id]
        min_dist = float('inf')
        best_direction = None

        # 找到最近的路徑段及其方向
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            # 路徑段方向
            segment = p2 - p1
            segment_length = torch.norm(segment)

            if segment_length < 1e-6:
                continue

            segment_direction = segment / segment_length

            # 機器人到路徑段的距離
            robot_to_p1 = robot_pos_xy[env_id] - p1
            dist_to_segment = torch.norm(
                robot_to_p1 - torch.clamp(
                    torch.sum(robot_to_p1 * segment) / (segment_length ** 2),
                    min=0.0, max=1.0
                ) * segment
            )

            if dist_to_segment < min_dist:
                min_dist = dist_to_segment.item()
                best_direction = segment_direction

        if best_direction is not None:
            # 計算方向一致性（cosine similarity）
            direction_similarity = torch.sum(
                velocity_direction[env_id] * best_direction
            )
            rewards[env_id] = direction_similarity

    return rewards


# 組合獎勵函數（包含所有組成部分）
def path_following_reward(
    env: ManagerBasedRLEnv,
    path_points: torch.Tensor,
    robot_position_cfg: str = "robot",
    robot_velocity_cfg: str = "robot",
    weights: dict = None,
) -> torch.Tensor:
    """綜合路徑跟隨獎勵（進階版）

    結合多個路徑跟隨獎勵的組成部分：
    - progress_along_path: 前進距離獎勵
    - cross_track_error: 偏離路徑懲罰
    - path_direction_reward: 方向一致性獎勵

    Args:
        env: 環境實例
        path_points: A* 生成的路徑點
        weights: 各組成部分的權重
            {
                "progress": 1.0,        # 前進獎勵權重
                "cross_track": 0.5,     # cross-track error 權重
                "direction": 0.3,        # 方向一致性權重
            }

    Returns:
        shape [num_envs]: 總獎勵
    """
    if weights is None:
        weights = {
            "progress": 1.0,
            "cross_track": 0.5,
            "direction": 0.3,
        }

    # 1. 沿路徑前進獎勵
    progress = progress_along_path(env, path_points, robot_position_cfg)

    # 2. Cross-track error 懲罰
    cte = cross_track_error(env, path_points, robot_position_cfg)

    # 3. 方向一致性獎勵
    direction = path_direction_reward(env, path_points, robot_position_cfg, robot_velocity_cfg)

    # 綜合獎勵
    total_reward = (
        weights["progress"] * progress +
        weights["cross_track"] * cte +
        weights["direction"] * direction
    )

    return total_reward


# A* 路徑相關的輔助函數
def generate_straight_path(
    env: ManagerBasedRLEnv,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
) -> torch.Tensor:
    """生成直線路徑（用於測試或 Phase 0）

    在沒有障礙物的情況下，A* 會生成一條直線路徑。
    這個函數模擬 A* 的輸出，用於測試路徑跟隨獎勵函數。

    Args:
        env: 環境實例
        start_pos: 起點位置，shape [num_envs, 2]
        goal_pos: 終點位置，shape [num_envs, 2]

    Returns:
        shape [num_envs, 2, 2]: 包含起點和終點的簡單路徑
        每個環境的路徑為 [[start, goal]]
    """
    num_envs = start_pos.shape[0]
    device = start_pos.device

    # 創建路徑：每個環境包含起點和終點
    path_points = torch.stack([
        start_pos,  # [num_envs, 2]
        goal_pos,   # [num_envs, 2]
    ], dim=1)  # [2, num_envs, 2]

    # 交換維度以匹配 [num_envs, num_points, 2]
    path_points = path_points.transpose(0, 1)  # [num_envs, 2, 2]

    return path_points


__all__ = [
    "progress_along_path",
    "cross_track_error",
    "path_direction_reward",
    "path_following_reward",
    "generate_straight_path",
]
