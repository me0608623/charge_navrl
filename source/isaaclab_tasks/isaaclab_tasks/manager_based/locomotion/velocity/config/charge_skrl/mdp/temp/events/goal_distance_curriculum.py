"""目標距離課程學習事件

Phase 3 專用：根據成功率動態調整目標距離範圍。
設計理念：讓 agent 深刻記住抵達目的地能帶來巨大正向回報。

課程學習邏輯：
- 成功率 > 80%：增加目標距離（升級）
- 成功率 < 50%：減少目標距離（降級）
- 否則：保持當前難度
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 全局變量：追蹤訓練統計
_goal_distance_curriculum_stats = {
    "success_rate_history": [],
    "window_size": 100,  # 滑動窗口大小（episodes）
    "current_difficulty": 1,  # 當前難度等級（1-10）
    "min_difficulty": 1,  # 最小難度
    "max_difficulty": 10,  # 最大難度
}

# 難度等級對應的目標距離範圍與目標數量
# 格式：(min_dist, max_dist, num_goals)
# 策略：初期目標多且近（廣撒網），後期目標少且遠（精準導航）
DIFFICULTY_DISTANCE_MAP = {
    1: (2.0, 4.0, 5),    # 新手：5 個目標，2-4 米（隨便撞都能贏）
    2: (2.5, 5.0, 5),    # 進階：5 個目標，距離稍遠
    3: (3.0, 6.0, 3),    # 熟練：3 個目標，開始減少數量
    4: (3.5, 7.0, 3),
    5: (4.0, 8.0, 1),    # 專家：1 個目標，4-8 米（標準 Phase 2 難度）
    6: (5.0, 9.0, 1),
    7: (6.0, 10.0, 1),
    8: (7.0, 12.0, 1),
    9: (8.0, 14.0, 1),
    10: (9.0, 15.0, 1),  # 大師：1 個目標，9-15 米
}


def get_success_rate_for_distance(env: "ManagerBasedRLEnv") -> float:
    """計算當前成功率（用於目標距離課程學習）

    從終止管理器的統計信息中獲取成功率。
    成功率 = 到達目標的環境數 / 總環境數

    Args:
        env: 環境實例

    Returns:
        成功率（0.0-1.0）
    """
    termination_manager = env.termination_manager

    # 方法 1：從終止管理器的 _last_episode_dones 中獲取
    if hasattr(termination_manager, "_term_names") and hasattr(termination_manager, "_last_episode_dones"):
        # 查找 goal_reached 終止條件
        goal_reached_idx = None
        for i, name in enumerate(termination_manager._term_names):
            if name == "goal_reached":
                goal_reached_idx = i
                break

        if goal_reached_idx is not None:
            # 獲取最近一個 episode 的終止狀態
            goal_reached = termination_manager._last_episode_dones[:, goal_reached_idx]
            if goal_reached.numel() > 0 and goal_reached.any():
                success_rate = goal_reached.float().mean().item()
                return success_rate

    # 方法 2：從 extras 中獲取（如果有的話）
    if "log" in env.extras:
        # 檢查是否有 Episode_Termination 統計
        for key, value in env.extras["log"].items():
            if "Episode_Termination" in key and "goal_reached" in key:
                if isinstance(value, (int, float)):
                    return float(value)
        # 檢查是否有直接的 success_rate
        if "success_rate" in env.extras["log"]:
            return float(env.extras["log"]["success_rate"])

    # 默認返回 0.0（沒有統計信息）
    return 0.0


def initialize_goal_distance_curriculum(env: "ManagerBasedRLEnv", initial_difficulty: int = 1):
    """初始化目標距離課程學習"""
    global _goal_distance_curriculum_stats
    
    if hasattr(env, "num_envs"):
        required_len = int(env.num_envs * 0.1)
        _goal_distance_curriculum_stats["window_size"] = max(100, required_len + 50)

    _goal_distance_curriculum_stats["current_difficulty"] = initial_difficulty
    
    # 設置初始目標距離範圍和目標數量
    min_dist, max_dist, num_goals = DIFFICULTY_DISTANCE_MAP[initial_difficulty]
    env._goal_distance_range = (min_dist, max_dist)
    
    # 初始化 goal command 的 num_goals
    if hasattr(env, "command_manager"):
        goal_cmd = env.command_manager.get_term("goal_command")
        if hasattr(goal_cmd, "cfg"):
            goal_cmd.cfg.num_goals = num_goals
    
    print(f"[Curriculum] 初始難度: {initial_difficulty} | 距離: {min_dist}-{max_dist}m | 目標數: {num_goals}")


def update_goal_distance_curriculum(
    env: "ManagerBasedRLEnv",
    success_rate: float,
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.5,
) -> dict:
    """更新目標距離課程學習難度"""
    global _goal_distance_curriculum_stats
    
    _goal_distance_curriculum_stats["success_rate_history"].append(success_rate)
    
    if len(_goal_distance_curriculum_stats["success_rate_history"]) > _goal_distance_curriculum_stats["window_size"]:
        _goal_distance_curriculum_stats["success_rate_history"].pop(0)
    
    calculated_min_len = max(20, int(env.num_envs * 0.1)) if hasattr(env, "num_envs") else 20
    min_history_length = min(calculated_min_len, _goal_distance_curriculum_stats["window_size"])
    
    current_difficulty = _goal_distance_curriculum_stats["current_difficulty"]
    min_dist, max_dist, num_goals = DIFFICULTY_DISTANCE_MAP[current_difficulty]
    
    if len(_goal_distance_curriculum_stats["success_rate_history"]) < min_history_length:
        return {
            "difficulty": current_difficulty,
            "min_distance": min_dist,
            "max_distance": max_dist,
            "num_goals": num_goals,
            "success_rate": success_rate,
        }

    avg_success_rate = sum(_goal_distance_curriculum_stats["success_rate_history"]) / len(_goal_distance_curriculum_stats["success_rate_history"])
    
    min_difficulty = _goal_distance_curriculum_stats["min_difficulty"]
    max_difficulty = _goal_distance_curriculum_stats["max_difficulty"]
    
    should_increase = avg_success_rate >= difficulty_increase_threshold and current_difficulty < max_difficulty
    should_decrease = avg_success_rate < difficulty_decrease_threshold and current_difficulty > min_difficulty
    
    if should_increase:
        new_difficulty = min(current_difficulty + 1, max_difficulty)
        _goal_distance_curriculum_stats["current_difficulty"] = new_difficulty
        _goal_distance_curriculum_stats["success_rate_history"].clear()
        
        new_min_dist, new_max_dist, new_num_goals = DIFFICULTY_DISTANCE_MAP[new_difficulty]
        print(f"[Curriculum] ⬆️ 升級: {current_difficulty}->{new_difficulty} (SR: {avg_success_rate:.2%}) "
              f"| 距離: {new_min_dist}-{new_max_dist}m | 目標數: {new_num_goals}")
    elif should_decrease:
        new_difficulty = max(current_difficulty - 1, min_difficulty)
        _goal_distance_curriculum_stats["current_difficulty"] = new_difficulty
        _goal_distance_curriculum_stats["success_rate_history"].clear()
        
        new_min_dist, new_max_dist, new_num_goals = DIFFICULTY_DISTANCE_MAP[new_difficulty]
        print(f"[Curriculum] ⬇️ 降級: {current_difficulty}->{new_difficulty} (SR: {avg_success_rate:.2%}) "
              f"| 距離: {new_min_dist}-{new_max_dist}m | 目標數: {new_num_goals}")
    else:
        new_difficulty = current_difficulty
        new_min_dist, new_max_dist, new_num_goals = min_dist, max_dist, num_goals
    
    return {
        "difficulty": new_difficulty,
        "min_distance": new_min_dist,
        "max_distance": new_max_dist,
        "num_goals": new_num_goals,
        "success_rate": avg_success_rate,
    }


def adaptive_goal_distance_update(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.5,
):
    """課程學習更新函數"""
    global _goal_distance_curriculum_stats
    
    current_difficulty = _goal_distance_curriculum_stats.get("current_difficulty", 1)
    
    if not hasattr(env, "reset_buf") or not env.reset_buf.any():
        min_dist, max_dist, num_goals = DIFFICULTY_DISTANCE_MAP[current_difficulty]
        return {
            "difficulty": float(current_difficulty),
            "min_distance": float(min_dist),
            "max_distance": float(max_dist),
            "num_goals": float(num_goals),
            "success_rate": 0.0,
        }
    
    success_rate = get_success_rate_for_distance(env)
    
    if env.reset_buf.any():
        difficulty_params = update_goal_distance_curriculum(
            env,
            success_rate,
            difficulty_increase_threshold,
            difficulty_decrease_threshold,
        )
        
        # 更新目標距離範圍
        env._goal_distance_range = (
            difficulty_params["min_distance"],
            difficulty_params["max_distance"],
        )
        
        # 更新目標數量（關鍵！）
        if hasattr(env, "command_manager"):
            goal_cmd = env.command_manager.get_term("goal_command")
            if hasattr(goal_cmd, "cfg") and hasattr(goal_cmd.cfg, "num_goals"):
                goal_cmd.cfg.num_goals = difficulty_params["num_goals"]
        
        return {
            "difficulty": float(difficulty_params["difficulty"]),
            "min_distance": float(difficulty_params["min_distance"]),
            "max_distance": float(difficulty_params["max_distance"]),
            "num_goals": float(difficulty_params["num_goals"]),
            "success_rate": float(difficulty_params["success_rate"]),
        }
    
    min_dist, max_dist, num_goals = DIFFICULTY_DISTANCE_MAP[current_difficulty]
    return {
        "difficulty": float(current_difficulty),
        "min_distance": float(min_dist),
        "max_distance": float(max_dist),
        "num_goals": float(num_goals),
        "success_rate": float(success_rate),
    }
