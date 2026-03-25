"""
局部目標提取器 (Local Goal / Carrot-on-stick)

解決 AIT* 路徑「跳變」導致 RL 震盪的問題。

核心思想：
- 不把整條 AIT* 路徑直接餵給 RL
- 只取出前方 x 公尺（例如 2m）的點作為 RL 的當前目標
- 使用平滑過濾防止目標跳變
"""

from __future__ import annotations

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class LocalGoalConfig:
    """局部目標配置

    Attributes:
        lookahead_distance: 前瞻距離（米），提取路徑前方這麼遠的點
        min_update_distance: 最小更新距離，目標移動超過此距離才更新
        smoothing_factor: 平滑因子 (0-1)，越小越平滑
        corridor_half_width: 走廊半寬（米），在這範圍內視為「跟隨良好」
        max_goal_change: 最大目標變化量（米/步），防止劇烈跳變
    """
    lookahead_distance: float = 2.0  # 前方 2 米
    min_update_distance: float = 0.3  # 目標移動超過 0.3m 才更新
    smoothing_factor: float = 0.3  # 平滑因子
    corridor_half_width: float = 0.5  # 走廊半寬 0.5m
    max_goal_change: float = 0.5  # 每步最多變化 0.5m


class LocalGoalExtractor:
    """局部目標提取器

    從 AIT* 全局路徑中提取局部目標點，給 RL 提供穩定的追隨目標。
    實現 Carrot-on-stick 策略。
    """

    def __init__(self, cfg: LocalGoalConfig, num_envs: int = 1):
        """初始化局部目標提取器

        Args:
            cfg: 局部目標配置
            num_envs: 環境數量
        """
        self.cfg = cfg
        self.num_envs = num_envs

        # 當前目標（每個環境）
        self.current_goals: list[Optional[torch.Tensor]] = [None] * num_envs

        # 目標歷史（用於平滑）
        self.goal_history: list[deque] = [
            deque(maxlen=5) for _ in range(num_envs)
        ]

        # 上次更新的路徑版本
        self._path_version: list[int] = [0] * num_envs

    def extract_local_goal(
        self,
        robot_pos: torch.Tensor,
        aitstar_path: torch.Tensor,
        path_version: int = 0,
        env_id: int = 0,
    ) -> torch.Tensor:
        """從 AIT* 路徑提取局部目標

        Args:
            robot_pos: [2] 機器人當前位置（全局坐標）
            aitstar_path: [N, 2] AIT* 生成的全局路徑
            path_version: 路徝版本號（用於檢測路徑是否更新）
            env_id: 環境 ID

        Returns:
            [2] 局部目標位置（全局坐標）
        """
        if len(aitstar_path) < 2:
            # 路徑太短，直接用終點
            return aitstar_path[-1] if len(aitstar_path) > 0 else robot_pos

        # 找到機器人在路徑上的位置
        distances = torch.norm(aitstar_path - robot_pos, dim=1)
        nearest_idx = torch.argmin(distances)

        # 從最近點開始，向前尋找 lookahead_distance 處的點
        accumulated_dist = 0.0
        target_idx = nearest_idx

        for i in range(nearest_idx.item(), len(aitstar_path) - 1):
            segment_length = torch.norm(aitstar_path[i + 1] - aitstar_path[i]).item()
            accumulated_dist += segment_length

            if accumulated_dist >= self.cfg.lookahead_distance:
                target_idx = i + 1
                break
            target_idx = i + 1

        # 如果到路徑末點還不夠，使用最後一點
        if target_idx >= len(aitstar_path):
            target_idx = len(aitstar_path) - 1

        raw_goal = aitstar_path[target_idx]

        # 檢查是否需要更新
        if self.current_goals[env_id] is not None:
            old_goal = self.current_goals[env_id]
            change = torch.norm(raw_goal - old_goal).item()

            # 路徑版本相同且變化很小，使用平滑後的目標
            if path_version == self._path_version[env_id] and change < self.cfg.min_update_distance:
                return self._smooth_goal(env_id, raw_goal)

        # 更新目標
        self._path_version[env_id] = path_version
        smoothed_goal = self._smooth_goal(env_id, raw_goal)
        self.current_goals[env_id] = smoothed_goal

        return smoothed_goal

    def _smooth_goal(self, env_id: int, raw_goal: torch.Tensor) -> torch.Tensor:
        """平滑目標更新

        使用移動平均和最大變化限制來防止目標跳變。

        Args:
            env_id: 環境 ID
            raw_goal: 原始目標位置

        Returns:
            平滑後的目標位置
        """
        # 添加到歷史
        self.goal_history[env_id].append(raw_goal)

        if len(self.goal_history[env_id]) < 2:
            return raw_goal

        # 取歷史平均
        history_tensor = torch.stack(list(self.goal_history[env_id]))
        avg_goal = history_tensor.mean(dim=0)

        # 如果有當前目標，限制變化幅度
        if self.current_goals[env_id] is not None:
            old_goal = self.current_goals[env_id]
            change = avg_goal - old_goal
            change_mag = torch.norm(change)

            if change_mag > self.cfg.max_goal_change:
                # 限制變化幅度
                change = change / change_mag * self.cfg.max_goal_change
                avg_goal = old_goal + change

        # 混合舊目標和新目標
        if self.current_goals[env_id] is not None:
            alpha = self.cfg.smoothing_factor
            smoothed = (1 - alpha) * self.current_goals[env_id] + alpha * avg_goal
        else:
            smoothed = avg_goal

        return smoothed

    def get_corridor_status(
        self,
        robot_pos: torch.Tensor,
        aitstar_path: torch.Tensor,
        env_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """獲取機器人相對於路徑走廊的狀態

        Args:
            robot_pos: [2] 機器人位置
            aitstar_path: [N, 2] AIT* 路徑
            env_id: 環境 ID

        Returns:
            字典包含：
            - 'in_corridor': 是否在走廊內
            - 'lateral_error': 側向誤差（米）
            - 'distance_to_corridor_center': 到走廊中心的距離
        """
        # 找最近路徑段
        distances = torch.norm(aitstar_path - robot_pos, dim=1)
        nearest_idx = torch.argmin(distances)
        nearest_point = aitstar_path[nearest_idx]

        # 計算側向距離
        lateral_error = robot_pos - nearest_point

        # 找路徑段方向
        if nearest_idx < len(aitstar_path) - 1:
            path_direction = aitstar_path[nearest_idx + 1] - nearest_point
        else:
            path_direction = nearest_point - aitstar_path[max(0, nearest_idx - 1)]

        path_direction = path_direction / (torch.norm(path_direction) + 1e-6)

        # 計算側向分量（使用叉積判斷左右）
        lateral_mag = torch.norm(lateral_error)
        cross = path_direction[0] * lateral_error[1] - path_direction[1] * lateral_error[0]
        lateral_signed = lateral_mag if cross > 0 else -lateral_mag

        in_corridor = abs(lateral_signed) <= self.cfg.corridor_half_width

        return {
            'in_corridor': torch.tensor(in_corridor),
            'lateral_error': lateral_signed,
            'distance_to_center': lateral_signed,
        }

    def reset_env(self, env_id: int):
        """重置特定環境的狀態

        Args:
            env_id: 環境 ID
        """
        self.current_goals[env_id] = None
        self.goal_history[env_id].clear()
        self._path_version[env_id] = 0


def extract_local_goals_batch(
    env: ManagerBasedRLEnv,
    aitstar_paths: torch.Tensor,
    robot_cfg: str = "robot",
    cfg: Optional[LocalGoalConfig] = None,
) -> torch.Tensor:
    """批量提取局部目標

    Args:
        env: 環境實例
        aitstar_paths: [num_envs, N, 2] 或 [N, 2] AIT* 路徑
        robot_cfg: 機器人配置名稱
        cfg: 局部目標配置

    Returns:
        [num_envs, 2] 局部目標位置
    """
    from isaaclab.assets import Articulation

    if cfg is None:
        cfg = LocalGoalConfig()

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]

    num_envs = env.num_envs

    # 創建提取器（如果還沒有）
    if not hasattr(env, '_local_goal_extractor'):
        env._local_goal_extractor = LocalGoalExtractor(cfg, num_envs)

    extractor = env._local_goal_extractor

    # 處理路徑
    if aitstar_paths.dim() == 2:
        # 單一路徑，複製到所有環境
        aitstar_paths = aitstar_paths.unsqueeze(0).expand(num_envs, -1, -1)

    local_goals = []

    for env_id in range(num_envs):
        local_goal = extractor.extract_local_goal(
            robot_pos[env_id],
            aitstar_paths[env_id],
            env_id=env_id,
        )
        local_goals.append(local_goal)

    return torch.stack(local_goals)


__all__ = [
    "LocalGoalConfig",
    "LocalGoalExtractor",
    "extract_local_goals_batch",
]
