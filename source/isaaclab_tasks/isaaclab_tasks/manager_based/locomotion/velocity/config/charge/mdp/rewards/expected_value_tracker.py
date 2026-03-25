"""期望值追蹤系統 (優化版本)

直接從 RewardManager 的 _step_reward 讀取數據,解決日誌延遲與鍵名翻譯問題。
實現論文的期望值設計理念: V = R × p
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ExpectedValueTracker:
    """期望值追蹤器
    
    追蹤各能力的期望值 V = R × p,並記錄到訓練日誌。
    
    能力分類 (使用配置中的原始名稱):
    - Survival (生存/導航): 核心能力,權重 w₁ = 1.0
    - Posture (姿態控制): 輔助能力,權重 w₂ = 0.3
    - Efficiency (效率): 輔助能力,權重 w₃ = 0.5
    """
    
    # 獎勵項到能力的映射 (使用 RewardsCfg 中的原始變數名)
    REWARD_TO_CAPABILITY = {
        # Survival (生存/導航)
        "progressive_collision": "survival",
        "reaching_goal": "survival",
        "safe_navigation": "survival",
        # Posture (姿態控制)
        "forward_motion": "posture",
        # Efficiency (效率)
        "distance_to_goal": "efficiency",
        "speed_control_near_obstacles": "efficiency",
    }
    
    def __init__(self, env: "ManagerBasedRLEnv", capability_weights: Dict[str, float]):
        """初始化期望值追蹤器
        
        Args:
            env: 環境實例
            capability_weights: 能力權重字典 {"survival": 1.0, "posture": 0.3, "efficiency": 0.5}
        """
        self.env = env
        self.capability_weights = capability_weights
        
        # 歷史記錄 (用於計算滑動平均)
        self.history = {
            "survival": [],
            "posture": [],
            "efficiency": [],
        }
        
        # 名稱到索引的緩存
        self._term_name_to_index = None
    
    def _get_term_index_mapping(self) -> Dict[str, int]:
        """建立獎勵項名稱到索引的映射"""
        if self._term_name_to_index is not None:
            return self._term_name_to_index
            
        if not hasattr(self.env, "reward_manager"):
            return {}
            
        # 從 RewardManager 獲取所有項目的名稱
        term_names = self.env.reward_manager._term_names
        self._term_name_to_index = {name: i for i, name in enumerate(term_names)}
        return self._term_name_to_index
    
    def update_history(self):
        """更新歷史記錄
        
        直接從 env.reward_manager._step_reward 讀取當前的獎勵值。
        這解決了 env.extras["log"] 可能存在的延遲或過濾問題。
        """
        if not hasattr(self.env, "reward_manager"):
            return

        # 獲取索引映射
        index_mapping = self._get_term_index_mapping()
        # 獲取當前步的獎勵張量 (num_envs, num_terms)
        step_rewards = self.env.reward_manager._step_reward
        
        # 計算各能力的當前總獎勵
        current_rewards = {
            "survival": 0.0,
            "posture": 0.0,
            "efficiency": 0.0,
        }
        
        # 遍歷映射表
        for term_name, capability in self.REWARD_TO_CAPABILITY.items():
            if term_name in index_mapping:
                idx = index_mapping[term_name]
                # 加總所有環境的該項獎勵並取平均
                term_value = torch.mean(step_rewards[:, idx]).item()
                current_rewards[capability] += term_value
        
        # 記錄到歷史 (保持最近 100 個 steps)
        for capability in ["survival", "posture", "efficiency"]:
            self.history[capability].append(current_rewards[capability])
            if len(self.history[capability]) > 100:
                self.history[capability].pop(0)
    
    def compute_expected_value(self, capability: str) -> float:
        """計算指定能力的期望值
        
        V_capability = (Σ rewards) × success_probability × capability_weight
        """
        if capability not in self.history or len(self.history[capability]) == 0:
            return 0.0
        
        # 1. 計算歷史平均獎勵 (最近 100 步)
        avg_reward = sum(self.history[capability]) / len(self.history[capability])
        
        # 2. 獲取成功機率
        success_prob = self._get_success_probability()
        
        # 3. 應用能力權重
        capability_weight = self.capability_weights.get(capability, 1.0)
        
        return avg_reward * success_prob * capability_weight
    
    def _get_success_probability(self) -> float:
        """獲取成功機率 (從 Curriculum 參數中獲取)"""
        if hasattr(self.env, "_adaptive_curriculum_params"):
            params = self.env._adaptive_curriculum_params
            if "success_rate" in params:
                return float(params["success_rate"])
        return 0.5
    
    def log_to_extras(self):
        """將期望值記錄到 env.extras["log"]"""
        if "log" not in self.env.extras:
            self.env.extras["log"] = {}
        
        # 先更新物理數據
        self.update_history()
        
        # 計算期望值
        v_surv = self.compute_expected_value("survival")
        v_post = self.compute_expected_value("posture")
        v_effi = self.compute_expected_value("efficiency")
        v_total = v_surv + v_post + v_effi
        
        # 寫入日誌
        self.env.extras["log"]["ExpectedValue/survival"] = v_surv
        self.env.extras["log"]["ExpectedValue/posture"] = v_post
        self.env.extras["log"]["ExpectedValue/efficiency"] = v_effi
        self.env.extras["log"]["ExpectedValue/total"] = v_total
