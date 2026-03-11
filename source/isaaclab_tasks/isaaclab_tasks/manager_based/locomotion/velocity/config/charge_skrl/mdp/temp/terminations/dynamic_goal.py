"""動態目標相關終止條件

包含與動態 Goal 重置功能兼容的終止條件：
- goal_reached_dynamic: 僅在完成足夠數量的 Goal 後才終止
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_goal_pos_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    """統一目標來源：優先局部航點，fallback 全域目標。

    分層架構核心：RL 只追蹤局部航點 (_local_goal_world)，
    全域目標由 AIT* 管理，RL 不應知道。

    Returns:
        [num_envs, 2] 目標 XY 座標。
    """
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        return env._local_goal_world[:, :2]
    # Fallback: 系統尚未生成局部航點時暫用全域目標
    return env.command_manager.get_command("goal_command")[:, :2]


def goal_reached_dynamic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    body_radius: float = 0.0,
    min_goals: int = 3,
) -> torch.Tensor:
    """動態 Goal 模式終止條件：僅在完成足夠數量的 Goal 後才終止

    分層架構設計：判定到達局部航點 (_local_goal_world)，而非全域目標。

    行為邏輯：
    1. 檢測 Agent 是否抵達當前局部航點 (distance < threshold)
    2. 如果已啟用動態 Goal 重置，檢查是否已完成 min_goals 數量
    3. 只有在「抵達航點」且「已完成足夠數量」時才終止

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米）
        body_radius: 機器人本體半徑（米），用於調整距離計算
        min_goals: 最少需要完成的 Goal 數量（默認 3）

    Returns:
        shape [num_envs]：布林張量
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = _get_goal_pos_2d(env)
    robot_pos_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=1)

    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)

    reached_goal = distance < threshold

    # 未啟用動態 Goal 重置，直接返回
    if not hasattr(env, "_dynamic_goal_count"):
        return reached_goal

    # 已啟用：檢查是否已完成足夠數量的 Goal
    goal_count = env._dynamic_goal_count.long()

    return reached_goal & (goal_count >= min_goals)


def goal_reached_with_count(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """動態 Goal 模式終止條件：使用配置的目標數量

    分層架構設計：判定到達局部航點 (_local_goal_world)，而非全域目標。

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米）
        body_radius: 機器人本體半徑（米）

    Returns:
        shape [num_envs]：布林張量
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = _get_goal_pos_2d(env)
    robot_pos_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=1)

    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)

    reached_goal = distance < threshold

    # 未啟用動態 Goal 重置，直接返回
    if not hasattr(env, "_dynamic_goal_count"):
        return reached_goal

    # 使用配置的目標數量（隨機化）
    goal_count = env._dynamic_goal_count.long()
    goal_target = env._dynamic_goal_target.long()

    return reached_goal & (goal_count >= goal_target)


__all__ = [
    "goal_reached_dynamic",
    "goal_reached_with_count",
]
