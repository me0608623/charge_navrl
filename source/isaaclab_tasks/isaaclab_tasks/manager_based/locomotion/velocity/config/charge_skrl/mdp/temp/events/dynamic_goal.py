"""動態目標重置事件 (Dynamic Goal Respawn)

實現 Episode 內動態 Goal 重置功能：
- 當 agent 抵達 goal 時，將 goal 重置到新的隨機位置
- Episode 繼續，不終止
- 讓 agent 學會「持續追蹤 goal」而不是「記住固定路徑」

設計理念：
=========
訓練時：Agent 抵達 Goal → Goal 重置到新位置 → 繼續追蹤
推理時：Agent 抵達 Way-point → AIT* 更新下一個 Way-point → 繼續追蹤

這樣訓練和推理的行為模式一致，都是「持續追蹤動態更新的目標點」。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initialize_dynamic_goal_respawn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_goals: int = 3,
    max_goals: int = 5,
) -> dict:
    """初始化動態 Goal 重置功能

    為每個環境設置 Goal 抵達計數器，用於追蹤已完成的 Goal 數量。

    Args:
        env: 環境實例
        env_ids: 環境 ID（startup 模式下為 None，忽略）
        min_goals: 每個 episode 最少要完成的 Goal 數量
        max_goals: 每個 episode 最多要完成的 Goal 數量（之後允許終止）

    Returns:
        初始化字典
    """
    num_envs = env.num_envs
    device = env.device

    # Goal 抵達計數器：[num_envs]
    env._dynamic_goal_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    # 每個 episode 的 Goal 抵達目標（隨機化）
    env._dynamic_goal_target = torch.randint(
        min_goals, max_goals + 1, (num_envs,), device=device
    )

    # 配置參數
    env._dynamic_goal_min = min_goals
    env._dynamic_goal_max = max_goals

    return {}


def respawn_goal_on_reach(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    threshold: float = 0.5,
    min_goals: int = 3,
    max_goals: int = 5,
    reward_on_respawn: float = 10.0,
    enable_randomization: bool = True,
    enable_progressive_reward: bool = True,  # 🆕 啟用遞增獎勵
    progressive_multiplier: float = 0.5,    # 🆕 遞增係數（每多 1 個 Goal 增加 50%）
) -> dict:
    """🔥 Episode 內動態 Goal 重置（核心功能）

    當 Agent 抵達 Goal 時，將 Goal 重置到新的隨機位置，Episode 繼續。

    行為流程：
    1. 檢測 Agent 到 Goal 的距離
    2. 如果距離 < threshold，觸發 Goal 重置
    3. Goal 重置到新的隨機位置
    4. 更新 _local_goal_world（保持一致性）
    5. 累積計數器，Episode 繼續

    Args:
        env: 環境實例
        env_ids: 要檢查的環境 ID
        threshold: 抵達閾值（米）
        min_goals: 每個 episode 最少要完成的 Goal 數量
        max_goals: 每個 episode 最多要完成的 Goal 數量
        reward_on_respawn: Goal 重置時給予的獎勵
        enable_randomization: 是否啟用隨機目標距離/角度範圍

    Returns:
        {
            "respawned_envs": tensor of env IDs that had goal respawned,
            "total_respawns": total number of respawns this call,
        }
    """
    # 處理 env_ids 參數
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    if len(env_ids) == 0:
        return {"respawned_envs": torch.tensor([], device=env.device), "total_respawns": 0}

    device = env.device

    # 初始化計數器（如果不存在）
    if not hasattr(env, "_dynamic_goal_count"):
        env._dynamic_goal_count = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        env._dynamic_goal_target = torch.randint(
            min_goals, max_goals + 1, (env.num_envs,), device=device
        )
        env._dynamic_goal_min = min_goals
        env._dynamic_goal_max = max_goals

    # 獲取機器人和目標位置
    # 統一目標來源：優先使用局部航點（分層架構核心）
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[env_ids, :2]
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world[env_ids, :2]
    else:
        # Fallback: 系統尚未生成局部航點時暫用全域目標
        goal_pos = env.command_manager.get_command("goal_command")[env_ids, :2]

    # 計算距離
    distance = torch.norm(goal_pos - robot_pos, dim=1)

    # 檢測哪些環境需要重置 Goal
    needs_respawn = distance < threshold

    if not needs_respawn.any():
        return {"respawned_envs": torch.tensor([], device=device), "total_respawns": 0}

    # 獲取需要重置的環境 ID
    respawn_indices = torch.where(needs_respawn)[0]
    respawned_env_ids = env_ids[respawn_indices]

    # 🆕 計算遞增獎勵：活得越久，獎勵越高
    # 第 1 個 Goal: base_reward
    # 第 2 個 Goal: base_reward * (1 + 0.5) = 1.5x
    # 第 3 個 Goal: base_reward * (1 + 2*0.5) = 2.0x
    respawn_rewards = torch.zeros(len(env_ids), device=device)

    if enable_progressive_reward:
        # 獲取當前計數（在 +1 之前）
        current_counts = env._dynamic_goal_count[respawned_env_ids].float()
        # 計算遞增係數：1.0, 1.5, 2.0, 2.5, ...
        bonus_multiplier = 1.0 + progressive_multiplier * current_counts
        # 計算實際獎勵
        actual_rewards = reward_on_respawn * bonus_multiplier
        respawn_rewards[respawn_indices] = actual_rewards
    else:
        # 固定獎勵（原始行為）
        respawn_rewards[respawn_indices] = reward_on_respawn

    # 通知 reward 系統（通過設置臨時標記）
    env._dynamic_goal_reward = getattr(env, "_dynamic_goal_reward", torch.zeros(env.num_envs, device=device))
    env._dynamic_goal_reward[respawned_env_ids] += respawn_rewards[respawn_indices]

    # 🔥 重置 Goal 到新的隨機位置
    goal_command_manager = env.command_manager.get_term("goal_command")
    goal_command_manager._resample_command(respawned_env_ids.cpu().tolist())

    # 🔥 同步更新 _local_goal_world（保持一致性）
    # 這很重要！因為觀測函數使用 _local_goal_world 而不是直接使用 goal_command
    if hasattr(env, "_local_goal_world"):
        new_goal_pos = env.command_manager.get_command("goal_command")[respawned_env_ids, :2]
        env._local_goal_world[respawned_env_ids] = new_goal_pos

    # 🔥 更新可視化標記
    if hasattr(goal_command_manager, "_update_goal_markers"):
        goal_command_manager._update_goal_markers()

    # 更新計數器
    env._dynamic_goal_count[respawned_env_ids] += 1

    return {
        "respawned_envs": respawned_env_ids,
        "total_respawns": len(respawned_env_ids),
        "rewards": respawn_rewards,
    }


def should_allow_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """檢查是否允許 Episode 終止

    只有當 Agent 已經完成足夠數量的 Goal 後，才允許 Episode 終止。
    這用於修改 goal_reached 終止條件的行為。

    Args:
        env: 環境實例
        env_ids: 要檢查的環境 ID

    Returns:
        [num_envs] bool 張量，True 表示允許終止
    """
    if env_ids is None:
        env_ids = slice(None)

    if not hasattr(env, "_dynamic_goal_count"):
        # 如果未啟用動態 Goal 重置，允許所有終止
        return torch.ones(len(env_ids) if isinstance(env_ids, slice) else len(env_ids), device=env.device, dtype=torch.bool)

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    # 檢查是否已達到目標數量
    goal_count = env._dynamic_goal_count[env_ids]
    goal_target = env._dynamic_goal_target[env_ids]

    # 只有當已完成 >= 目標數量時，才允許終止
    return goal_count >= goal_target


def reset_dynamic_goal_count(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> dict:
    """重置動態 Goal 計數器（Episode 結束時調用）

    Args:
        env: 環境實例
        env_ids: 要重置的環境 ID

    Returns:
        空字典
    """
    if env_ids is None:
        env_ids = slice(None)

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    if not hasattr(env, "_dynamic_goal_count"):
        return {}

    # 重置計數器
    env._dynamic_goal_count[env_ids] = 0

    # 隨機化新的目標數量
    min_goals = getattr(env, "_dynamic_goal_min", 3)
    max_goals = getattr(env, "_dynamic_goal_max", 5)
    new_targets = torch.randint(
        min_goals, max_goals + 1, (len(env_ids),), device=env.device
    )
    env._dynamic_goal_target[env_ids] = new_targets

    return {}


def get_dynamic_goal_stats(env: ManagerBasedRLEnv) -> dict:
    """獲取動態 Goal 統計信息（用於調試和日誌）

    Args:
        env: 環境實例

    Returns:
        {
            "mean_goals_per_episode": 平均每個 episode 完成的 Goal 數量,
            "current_counts": 當前每個環境的計數,
            "target_counts": 當前每個環境的目標,
        }
    """
    if not hasattr(env, "_dynamic_goal_count"):
        return {
            "mean_goals_per_episode": 0.0,
            "current_counts": None,
            "target_counts": None,
        }

    return {
        "mean_goals_per_episode": env._dynamic_goal_count.float().mean().item(),
        "current_counts": env._dynamic_goal_count.cpu().clone(),
        "target_counts": env._dynamic_goal_target.cpu().clone(),
    }


__all__ = [
    "initialize_dynamic_goal_respawn",
    "respawn_goal_on_reach",
    "should_allow_termination",
    "reset_dynamic_goal_count",
    "get_dynamic_goal_stats",
]
