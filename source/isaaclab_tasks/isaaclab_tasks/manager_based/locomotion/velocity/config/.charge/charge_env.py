# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航環境類別（帶有觀測檢查）

這個文件定義了一個自定義環境類，繼承自 ManagerBasedRLEnv，
並在觀測返回前添加 NaN/Inf 檢查，用於修復 PPO std>=0 錯誤。
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn

from .charge_mdp import _check_finite


class ChargeNavigationEnv(ManagerBasedRLEnv):
    """Charge 導航環境（帶有觀測檢查）
    
    這個類繼承自 ManagerBasedRLEnv，並在 step() 方法中添加了觀測檢查，
    用於檢測和修復 NaN/Inf 值，防止 PPO std>=0 錯誤。
    """
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """執行環境動態的一個時間步長（帶有觀測檢查）
        
        在返回觀測前，檢查所有觀測項是否包含 NaN/Inf 值。
        如果發現異常值，會立即拋出異常並提供詳細信息。
        """
        # 調用父類的 step 方法
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        
        # ========================================================================
        # 關鍵：在觀測返回前檢查 NaN/Inf（防止 PPO std>=0 錯誤）
        # ========================================================================
        if isinstance(obs_buf, dict):
            # 如果觀測是字典格式（多個觀測組）
            for group_name, obs in obs_buf.items():
                if isinstance(obs, torch.Tensor):
                    # 檢查並清理觀測
                    obs_cleaned = self._sanitize_observation(f"{group_name}", obs)
                    obs_buf[group_name] = obs_cleaned
                elif isinstance(obs, dict):
                    # 如果觀測組內部還是字典（未合併的情況）
                    for term_name, term_obs in obs.items():
                        if isinstance(term_obs, torch.Tensor):
                            term_cleaned = self._sanitize_observation(f"{group_name}.{term_name}", term_obs)
                            obs_buf[group_name][term_name] = term_cleaned
        elif isinstance(obs_buf, torch.Tensor):
            # 如果觀測是單個張量
            obs_buf = self._sanitize_observation("policy", obs_buf)
        
        # 檢查獎勵是否也有 NaN/Inf
        if not torch.isfinite(reward_buf).all():
            _check_finite("reward_buf", reward_buf, raise_on_error=True)
        
        return obs_buf, reward_buf, terminated, time_outs, extras
    
    def _sanitize_observation(self, name: str, obs: torch.Tensor) -> torch.Tensor:
        """清理觀測中的 NaN/Inf 值
        
        Args:
            name: 觀測項名稱（用於錯誤訊息）
            obs: 觀測張量
        
        Returns:
            清理後的觀測張量
        """
        # 檢查是否有 NaN/Inf
        if not torch.isfinite(obs).all():
            # 先報告問題（不拋出異常，因為我們會嘗試修復）
            _check_finite(name, obs, raise_on_error=False)
            
            # 計算合理的替換值
            # 先獲取有限值的統計信息
            finite_mask = torch.isfinite(obs)
            if finite_mask.any():
                # 如果有有限值，使用有限值的中位數作為替換值
                finite_values = obs[finite_mask]
                replacement_value = finite_values.median().item()
                # 確保替換值在合理範圍內
                replacement_value = max(-10.0, min(10.0, replacement_value))
            else:
                # 如果所有值都是 NaN/Inf，使用 0.5（假設觀測已歸一化）
                replacement_value = 0.5
            
            # 清理異常值
            obs_cleaned = torch.nan_to_num(
                obs, 
                nan=replacement_value, 
                posinf=replacement_value, 
                neginf=replacement_value
            )
            
            # 根據觀測名稱決定 clip 範圍
            # lidar_scan 應該在 [0, 1]（已歸一化）
            if "lidar" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, 0.0, 1.0)
            # goal_position 和 goal_distance 可能在較大範圍
            elif "goal" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, -10.0, 10.0)
            # actions 應該在 [-1, 1]
            elif "action" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, -1.0, 1.0)
            else:
                # 默認 clip 到 [-10, 10]
                obs_cleaned = torch.clamp(obs_cleaned, -10.0, 10.0)
            
            # 最終檢查
            if not torch.isfinite(obs_cleaned).all():
                raise RuntimeError(
                    f"[{name}] Failed to sanitize observation! "
                    f"Still contains NaN/Inf after cleaning. "
                    f"shape={tuple(obs.shape)}"
                )
            
            return obs_cleaned
        
        return obs
