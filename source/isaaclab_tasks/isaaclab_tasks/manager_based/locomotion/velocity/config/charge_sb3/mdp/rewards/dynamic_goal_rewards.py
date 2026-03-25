"""動態目標重置獎勵 (Dynamic Goal Respawn Rewards)

為 Episode 內動態 Goal 重置功能提供獎勵函數：
- dynamic_goal_respawn_reward: Goal 重置時給予獎勵
- dynamic_goal_bonus_reward: 基於已抵達 Goal 數量的累進獎勵
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dynamic_goal_respawn_reward(
    env: ManagerBasedRLEnv,
    reward: float = 10.0,
) -> torch.Tensor:
    """🔥 動態 Goal 重置獎勵

    當 Agent 抵達 Goal 並觸發重置時給予獎勵。
    這個獎勵由 respawn_goal_on_reach() event 函數設置。

    Args:
        env: 環境實例
        reward: 基礎獎勵值（默認 10.0）

    Returns:
        [num_envs] 獎勵張量
    """
    if not hasattr(env, "_dynamic_goal_reward"):
        return torch.zeros(env.num_envs, device=env.device)

    reward_tensor = env._dynamic_goal_reward.clone()

    # 🔥 清理 NaN/Inf
    reward_tensor = torch.nan_to_num(reward_tensor, nan=0.0, posinf=reward, neginf=-reward)

    # 清零已消耗的獎勵（避免重複計算）
    env._dynamic_goal_reward.zero_()

    return reward_tensor


def dynamic_goal_bonus_reward(
    env: ManagerBasedRLEnv,
    base_reward: float = 10.0,
    bonus_multiplier: float = 2.0,
) -> torch.Tensor:
    """累進獎勵：抵達的 Goal 越多，獎勵越高

    獎勵公式：R = base_reward * (1 + bonus_multiplier * (count - 1))
    - 第 1 個 Goal: base_reward
    - 第 2 個 Goal: base_reward * (1 + bonus_multiplier)
    - 第 3 個 Goal: base_reward * (1 + 2 * bonus_multiplier)
    - ...

    Args:
        env: 環境實例
        base_reward: 基礎獎勵值
        bonus_multiplier: 累進係數

    Returns:
        [num_envs] 獎勵張量
    """
    if not hasattr(env, "_dynamic_goal_reward"):
        return torch.zeros(env.num_envs, device=env.device)

    # 獲取當前計數（重置前）
    if not hasattr(env, "_dynamic_goal_count"):
        return torch.zeros(env.num_envs, device=env.device)

    # 檢測哪些環境在這一幀重置了 Goal
    has_respawn = env._dynamic_goal_reward > 0

    if not has_respawn.any():
        return torch.zeros(env.num_envs, device=env.device)

    # 計算累進獎勵
    count = env._dynamic_goal_count.long()
    bonus = 1.0 + bonus_multiplier * (count - 1).clamp(min=0).float()
    adjusted_reward = env._dynamic_goal_reward * bonus

    # 🔥 清理 NaN/Inf
    adjusted_reward = torch.nan_to_num(adjusted_reward, nan=0.0, posinf=base_reward * 10, neginf=-base_reward)

    # 清零已消耗的獎勵
    reward_tensor = adjusted_reward.clone()
    env._dynamic_goal_reward.zero_()

    return reward_tensor


def dynamic_goal_progress_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 5.0,
) -> torch.Tensor:
    """持續追蹤獎勵（動態 Goal 模式）

    在動態 Goal 模式下，Agent 持續追蹤 Goal 時給予獎勵。
    這與普通的 progress_to_goal 不同，因為 Goal 會移動。

    分層架構設計：
    - RL 只負責追蹤局部航點 (_local_goal_world)
    - 全域目標由 AIT* 規劃器管理，RL 不應知道
    - Fallback: 系統尚未初始化局部航點時，暫用 goal_command

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        max_distance: 最大距離（用於歸一化）

    Returns:
        [num_envs] 獎勵張量，[0, 1]
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]

    # 統一目標來源：優先使用局部航點（分層架構核心）
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world[:, :2]
    else:
        # Fallback: 系統尚未生成局部航點時暫用全域目標
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    # 清理輸入數據的 NaN/Inf
    robot_pos = torch.nan_to_num(robot_pos, nan=0.0, posinf=100.0, neginf=-100.0)
    goal_pos = torch.nan_to_num(goal_pos, nan=0.0, posinf=100.0, neginf=-100.0)

    # 計算距離
    distance = torch.norm(goal_pos - robot_pos, dim=1)
    distance = torch.nan_to_num(distance, nan=0.0, posinf=max_distance, neginf=0.0)

    # 距離越近，獎勵越高（歸一化到 [0, 1]）
    normalized = 1.0 - torch.clamp(distance / max_distance, 0.0, 1.0)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = torch.clamp(normalized, 0.0, 1.0)

    return normalized


__all__ = [
    "dynamic_goal_respawn_reward",
    "dynamic_goal_bonus_reward",
    "dynamic_goal_progress_reward",
]
