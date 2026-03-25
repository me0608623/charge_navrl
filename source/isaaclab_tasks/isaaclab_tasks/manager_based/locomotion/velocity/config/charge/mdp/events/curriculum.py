"""自適應課程學習事件

包含自適應課程學習相關的事件函數：
- initialize_adaptive_curriculum: 初始化課程學習
- get_success_rate: 獲取成功率
- get_collision_rate: 獲取碰撞率
- update_adaptive_curriculum: 更新課程學習難度
- adaptive_curriculum_update: 課程學習更新函數（用於 CurriculumTerm）

全局變量：
- _adaptive_curriculum_stats: 追蹤訓練統計
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 全局變量：追蹤訓練統計
_adaptive_curriculum_stats = {
    "success_rate_history": [],
    "collision_rate_history": [],
    "window_size": 100,  # 滑動窗口大小（episodes）
    "current_difficulty": 3,  # 當前難度等級（障礙物數量）
    "min_difficulty": 3,  # 最小難度（3 個障礙物）
    "max_difficulty": 10,  # 最大難度（10 個障礙物）
}


def initialize_adaptive_curriculum(env: "ManagerBasedRLEnv", initial_difficulty: int = 3):
    """初始化自適應課程學習
    
    Args:
        env: 環境實例
        initial_difficulty: 初始難度等級（默認 3）
    """
    global _adaptive_curriculum_stats
    
    # [Fix] 根據環境數量動態調整窗口大小
    # 這是為了確保我們可以收集到足夠的樣本 (num_envs * 0.1) 而不會被窗口截斷
    if hasattr(env, "num_envs"):
        required_len = int(env.num_envs * 0.1)
        # 確保窗口至少是 100，或者是需求量 + 50 的緩衝
        _adaptive_curriculum_stats["window_size"] = max(100, required_len + 50)
        print(f"[Adaptive Curriculum] 初始化: Window Size 設為 {_adaptive_curriculum_stats['window_size']} (Envs: {env.num_envs})")

    _adaptive_curriculum_stats["current_difficulty"] = initial_difficulty
    
    # 設置環境中的初始障礙物數量
    env._num_obstacles = initial_difficulty
    
    # [NEW] 初始化期望值追蹤器
    if not hasattr(env, "_expected_value_tracker"):
        from ..rewards import ExpectedValueTracker
        capability_weights = {
            "survival": 1.0,    # w₁: 生存/導航 (最重要)
            "posture": 0.3,     # w₂: 姿態控制 (次要)
            "efficiency": 0.5,  # w₃: 效率 (次要)
        }
        env._expected_value_tracker = ExpectedValueTracker(env, capability_weights)
        print(f"[Adaptive Curriculum] 初始化期望值追蹤器: w₁={capability_weights['survival']}, w₂={capability_weights['posture']}, w₃={capability_weights['efficiency']}")
    
    # 初始化課程學習參數
    if not hasattr(env, "_adaptive_curriculum_params"):
        env._adaptive_curriculum_params = {
            "num_obstacles": initial_difficulty,
            "min_robot_distance": 3.5,
            "min_goal_distance": 2.0,
            "min_obstacle_spacing": 2.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "difficulty": initial_difficulty,
        }


def get_success_rate(env: "ManagerBasedRLEnv") -> float:
    """計算當前成功率
    
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


def get_collision_rate(env: "ManagerBasedRLEnv") -> float:
    """計算當前碰撞率
    
    從終止管理器的統計信息中獲取碰撞率。
    碰撞率 = (collision_contact 或 collision) 的環境數 / 總環境數
    
    Args:
        env: 環境實例
        
    Returns:
        碰撞率（0.0-1.0）
    """
    termination_manager = env.termination_manager
    
    # 方法 1：從終止管理器的 _last_episode_dones 中獲取
    if hasattr(termination_manager, "_term_names") and hasattr(termination_manager, "_last_episode_dones"):
        # 查找 collision 或 collision_contact 終止條件
        collision_idx = None
        collision_contact_idx = None
        
        for i, name in enumerate(termination_manager._term_names):
            if name == "collision":
                collision_idx = i
            elif name == "collision_contact":
                collision_contact_idx = i
        
        # 合併兩個碰撞終止條件（任一觸發都算碰撞）
        collision_combined = None
        if collision_idx is not None and collision_contact_idx is not None:
            collision_combined = (
                termination_manager._last_episode_dones[:, collision_idx] |
                termination_manager._last_episode_dones[:, collision_contact_idx]
            )
        elif collision_idx is not None:
            collision_combined = termination_manager._last_episode_dones[:, collision_idx]
        elif collision_contact_idx is not None:
            collision_combined = termination_manager._last_episode_dones[:, collision_contact_idx]
        
        if collision_combined is not None and collision_combined.any():
            collision_rate = collision_combined.float().mean().item()
            return collision_rate
    
    # 方法 2：從 extras 中獲取（如果有的話）
    if "log" in env.extras:
        # 檢查是否有 Episode_Termination 統計
        collision_rate = 0.0
        for key, value in env.extras["log"].items():
            if "Episode_Termination" in key and ("collision" in key or "collision_contact" in key):
                if isinstance(value, (int, float)):
                    collision_rate = max(collision_rate, float(value))  # 取最大值（任一觸發都算）
        if collision_rate > 0.0:
            return collision_rate
        # 檢查是否有直接的 collision_rate
        if "collision_rate" in env.extras["log"]:
            return float(env.extras["log"]["collision_rate"])
    
    # 默認返回 0.0（沒有統計信息）
    return 0.0


def update_adaptive_curriculum(
    env: "ManagerBasedRLEnv",
    success_rate: float,
    collision_rate: float,
    target_success_rate: float = 0.7,
    max_collision_rate: float = 0.3,
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.55,
) -> dict:
    """更新自適應課程學習難度
    
    根據成功率和碰撞率動態調整環境難度：
    - 如果成功率 > 80% 且碰撞率 < 20%：增加難度（增加障礙物數量）
    - 如果成功率 < 50% 或碰撞率 > 40%：降低難度（減少障礙物數量）
    - 否則：保持當前難度
    
    Args:
        env: 環境實例
        success_rate: 當前成功率（0.0-1.0）
        collision_rate: 當前碰撞率（0.0-1.0）
        target_success_rate: 目標成功率（默認 0.7）
        max_collision_rate: 最大可接受碰撞率（默認 0.3）
        difficulty_increase_threshold: 難度增加閾值（默認 0.8）
        difficulty_decrease_threshold: 難度降低閾值（默認 0.5）
        
    Returns:
        包含更新後難度參數的字典
    """
    global _adaptive_curriculum_stats
    
    # 更新歷史記錄
    _adaptive_curriculum_stats["success_rate_history"].append(success_rate)
    _adaptive_curriculum_stats["collision_rate_history"].append(collision_rate)
    
    # 保持歷史記錄在窗口大小內
    if len(_adaptive_curriculum_stats["success_rate_history"]) > _adaptive_curriculum_stats["window_size"]:
        _adaptive_curriculum_stats["success_rate_history"].pop(0)
        _adaptive_curriculum_stats["collision_rate_history"].pop(0)
    
    # 1. 強制數據收集期：只有當收集到足夠的歷史數據後才允許調整難度
    # 這防止了單次幸運的 batch 導致難度跳變
    # 根據環境數量動態調整等待的 episodes 數
    # 如果環境數量很大，數據收集很快，但我們希望看到穩定的表現
    # 至少 20 個 episodes，或者環境數量的 10%（例如 128 個環境 -> 需收集 12 個，取 max 就是 20）
    # ⚠️ [Fix] 限制最大值不超過 window_size，否則會導致無限等待 (e.g. 4096 envs -> 409 > 100)
    calculated_min_len = max(20, int(env.num_envs * 0.1)) if hasattr(env, "num_envs") else 20
    min_history_length = min(calculated_min_len, _adaptive_curriculum_stats["window_size"])
    
    if len(_adaptive_curriculum_stats["success_rate_history"]) < min_history_length:
        # 數據不足，打印日誌（可選，為了不洗屏可以只在特定數量打印）
        if len(_adaptive_curriculum_stats["success_rate_history"]) % 5 == 0:
            print(f"[Adaptive Curriculum] 收集數據中: {len(_adaptive_curriculum_stats['success_rate_history'])}/{min_history_length}")
        
        # 返回當前狀態，不進行難度調整
        return {
            "num_obstacles": _adaptive_curriculum_stats["current_difficulty"],
            "difficulty": _adaptive_curriculum_stats["current_difficulty"],
            "success_rate": success_rate, # 返回當前單次成功率供記錄
            "collision_rate": collision_rate,
            # 其他參數保持默認計算或不返回（調用者通常只關心 difficulty）
            "min_robot_distance": 3.5 * (1.0 - (_adaptive_curriculum_stats["current_difficulty"] - 3) / 7 * 0.3),
            "min_goal_distance": 2.0 * (1.0 - (_adaptive_curriculum_stats["current_difficulty"] - 3) / 7 * 0.3),
            "min_obstacle_spacing": 2.0 * (1.0 - (_adaptive_curriculum_stats["current_difficulty"] - 3) / 7 * 0.3),
        }

    # 計算滑動平均
    avg_success_rate = sum(_adaptive_curriculum_stats["success_rate_history"]) / len(_adaptive_curriculum_stats["success_rate_history"])
    avg_collision_rate = sum(_adaptive_curriculum_stats["collision_rate_history"]) / len(_adaptive_curriculum_stats["collision_rate_history"])
    
    current_difficulty = _adaptive_curriculum_stats["current_difficulty"]
    min_difficulty = _adaptive_curriculum_stats["min_difficulty"]
    max_difficulty = _adaptive_curriculum_stats["max_difficulty"]
    
    # 決定是否調整難度
    should_increase = (
        avg_success_rate >= difficulty_increase_threshold and
        avg_collision_rate < max_collision_rate / 2 and
        current_difficulty < max_difficulty
    )
    
    should_decrease = (
        (avg_success_rate < difficulty_decrease_threshold or avg_collision_rate > max_collision_rate) and
        current_difficulty > min_difficulty
    )
    
    # 調整難度
    if should_increase:
        new_difficulty = min(current_difficulty + 1, max_difficulty)
        _adaptive_curriculum_stats["current_difficulty"] = new_difficulty
        # 2. 升級後清空歷史：迫使系統在新的難度下重新收集數據
        _adaptive_curriculum_stats["success_rate_history"].clear()
        _adaptive_curriculum_stats["collision_rate_history"].clear()
        
        print(f"[Adaptive Curriculum] 增加難度: {current_difficulty} -> {new_difficulty} "
              f"(成功率: {avg_success_rate:.2%}, 碰撞率: {avg_collision_rate:.2%}) | 歷史記錄已重置")
    elif should_decrease:
        new_difficulty = max(current_difficulty - 1, min_difficulty)
        _adaptive_curriculum_stats["current_difficulty"] = new_difficulty
        # 降級後也清空歷史，給予重新適應的機會
        _adaptive_curriculum_stats["success_rate_history"].clear()
        _adaptive_curriculum_stats["collision_rate_history"].clear()
        
        print(f"[Adaptive Curriculum] 降低難度: {current_difficulty} -> {new_difficulty} "
              f"(成功率: {avg_success_rate:.2%}, 碰撞率: {avg_collision_rate:.2%}) | 歷史記錄已重置")
    else:
        new_difficulty = current_difficulty
    
    # 根據難度等級計算參數
    # 難度 3-10 對應障礙物數量 3-10
    num_obstacles = new_difficulty
    
    # 根據難度調整距離參數
    # 難度越高，距離參數越小（更困難）
    difficulty_factor = (new_difficulty - min_difficulty) / (max_difficulty - min_difficulty)
    
    # 基礎距離參數（Phase 2 的設置）
    base_min_robot_distance = 3.5
    base_min_goal_distance = 2.0
    base_min_obstacle_spacing = 2.0
    
    # 根據難度調整（難度越高，距離越小）
    min_robot_distance = base_min_robot_distance * (1.0 - difficulty_factor * 0.3)  # 3.5 -> 2.45
    min_goal_distance = base_min_goal_distance * (1.0 - difficulty_factor * 0.3)  # 2.0 -> 1.4
    min_obstacle_spacing = base_min_obstacle_spacing * (1.0 - difficulty_factor * 0.3)  # 2.0 -> 1.4
    
    return {
        "num_obstacles": num_obstacles,
        "min_robot_distance": min_robot_distance,
        "min_goal_distance": min_goal_distance,
        "min_obstacle_spacing": min_obstacle_spacing,
        "success_rate": avg_success_rate,
        "collision_rate": avg_collision_rate,
        "difficulty": new_difficulty,
    }


def adaptive_curriculum_update(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    target_success_rate: float = 0.7,
    max_collision_rate: float = 0.3,
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.55,
):
    """自適應課程學習更新函數（用於 CurriculumTerm）
    
    此函數在每個 step 結束時被調用，檢查是否有環境重置，並更新統計和難度。
    實際的參數調整會在 reset 事件中進行。
    
    返回值：
        dict[str, float]: 包含課程學習狀態的字典，用於日誌記錄
        - difficulty: 當前難度等級
        - success_rate: 當前成功率
        - collision_rate: 當前碰撞率
    
    Args:
        env: 環境實例
        env_ids: 環境 ID 列表（Isaac Lab 課程學習管理器自動傳入）
        target_success_rate: 目標成功率
        max_collision_rate: 最大可接受碰撞率
        difficulty_increase_threshold: 難度增加閾值
        difficulty_decrease_threshold: 難度降低閾值
    """
    global _adaptive_curriculum_stats
    
    # [FIX] 確保期望值追蹤器已初始化
    if not hasattr(env, "_expected_value_tracker"):
        from ..rewards import ExpectedValueTracker
        capability_weights = {
            "survival": 1.0,    # w₁: 生存/導航 (最重要)
            "posture": 0.3,     # w₂: 姿態控制 (次要)
            "efficiency": 0.5,  # w₃: 效率 (次要)
        }
        env._expected_value_tracker = ExpectedValueTracker(env, capability_weights)
        print(f"[Adaptive Curriculum] 初始化期望值追蹤器: w₁={capability_weights['survival']}, w₂={capability_weights['posture']}, w₃={capability_weights['efficiency']}")
    
    # 獲取當前難度等級（如果尚未初始化，使用默認值）
    current_difficulty = _adaptive_curriculum_stats.get("current_difficulty", 3)
    
    # [FIX] 確保期望值追蹤在每次調用時都更新（包括沒有 reset 的情況）
    if hasattr(env, "_expected_value_tracker"):
        tracker = env._expected_value_tracker
        tracker.log_to_extras()
    
    # 只在有環境重置時更新統計（episode 結束時）
    if not hasattr(env, "reset_buf") or not env.reset_buf.any():
        # 返回當前狀態（不更新）
        return {
            "difficulty": float(current_difficulty),
            "success_rate": 0.0,
            "collision_rate": 0.0,
        }
    
    # 獲取當前統計（從終止管理器的 _last_episode_dones 中獲取）
    success_rate = get_success_rate(env)
    collision_rate = get_collision_rate(env)
    
    # 更新難度（只在有有效統計時，或者至少有一些環境重置了）
    # 注意：即使統計為 0，也可能是因為沒有環境成功/碰撞，這本身也是信息
    if env.reset_buf.any():
        # 記錄更新前的難度
        previous_difficulty = _adaptive_curriculum_stats.get("current_difficulty", 3)
        
        difficulty_params = update_adaptive_curriculum(
            env,
            success_rate,
            collision_rate,
            target_success_rate,
            max_collision_rate,
            difficulty_increase_threshold,
            difficulty_decrease_threshold,
        )
        
        # 將更新後的參數存儲在環境中，供 reset 事件使用
        if not hasattr(env, "_adaptive_curriculum_params"):
            env._adaptive_curriculum_params = {}
        env._adaptive_curriculum_params.update(difficulty_params)
        
        # 更新當前難度
        current_difficulty = difficulty_params.get("difficulty", current_difficulty)
        
        # [Fix] 移除強制重置邏輯
        # 讓難度變更在下一個自然 Reset 時生效，避免 Agent 在 Episode 中途被突然生成的障礙物撞到
        if previous_difficulty != current_difficulty:
            print(f"[Adaptive Curriculum] 難度參數已更新 (Level {previous_difficulty} -> {current_difficulty})，將在各環境下一次 Reset 時生效")
    
    # [NEW] 計算並記錄期望值
    if hasattr(env, "_expected_value_tracker"):
        tracker = env._expected_value_tracker
        tracker.log_to_extras()
    
    # 返回課程學習狀態字典（用於日誌記錄）
    # RSL-RL 會將這些值記錄到日誌中，方便監控訓練進度
    result = {
        "difficulty": float(current_difficulty),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
    }
    
    # [NEW] 添加期望值到返回字典
    if hasattr(env, "_expected_value_tracker"):
        tracker = env._expected_value_tracker
        result["V_survival"] = tracker.compute_expected_value("survival")
        result["V_posture"] = tracker.compute_expected_value("posture")
        result["V_efficiency"] = tracker.compute_expected_value("efficiency")
        result["V_total"] = result["V_survival"] + result["V_posture"] + result["V_efficiency"]
    
    return result
