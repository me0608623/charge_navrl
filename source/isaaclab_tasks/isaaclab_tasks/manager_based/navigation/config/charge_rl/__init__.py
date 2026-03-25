# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge RL 機器人的導航環境配置和註冊模組

此模組負責：
1. 導入訓練代理配置（agents 子模組）
2. 註冊 Gymnasium 環境，使導航任務可以通過標準的 Gym API 使用

註冊的環境：
- Isaac-Navigation-Flat-Charge-RL-v0：訓練環境（大量並行環境，啟用域隨機化）
- Isaac-Navigation-Flat-Charge-RL-Play-v0：測試/遊玩環境（少量環境，禁用噪聲）
"""

import gymnasium as gym  # Gymnasium 庫，用於註冊和創建強化學習環境

from . import agents  # 導入訓練代理配置模組（包含 PPO 等算法配置）

##
# 註冊 Gymnasium 環境 (Register Gym environments)
##

# 註冊訓練環境：用於訓練導航策略
gym.register(
    id="Isaac-Navigation-Flat-Charge-RL-v0",  # 環境 ID，用於 gym.make() 創建環境
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 環境類的入口點
    disable_env_checker=True,  # 禁用環境檢查器（提高啟動速度）
    kwargs={
        # 環境配置入口點：指向 NavigationEnvCfg 類
        # 此配置定義了環境的所有參數（動作、觀測、獎勵、終止條件等）
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationEnvCfg",
        # RSL-RL 訓練配置入口點：指向 PPO 算法的訓練配置
        # 此配置定義了網絡結構、超參數、訓練設置等
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
        # SKRL 訓練配置入口點：SKRL 庫的配置文件（YAML 格式）
        # SKRL 是另一個強化學習庫，提供不同的訓練選項
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# 註冊測試/遊玩環境：用於測試訓練好的策略或人工控制
gym.register(
    id="Isaac-Navigation-Flat-Charge-RL-Play-v0",  # 環境 ID（Play 版本）
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 環境類的入口點（與訓練環境相同）
    disable_env_checker=True,  # 禁用環境檢查器
    kwargs={
        # 環境配置入口點：指向 NavigationEnvCfg_PLAY 類
        # 此配置繼承自 NavigationEnvCfg，但進行了優化：
        # - 減少環境數量（從數千個減少到 50 個）以提高渲染性能
        # - 增大環境間距以避免視覺干擾
        # - 禁用觀測噪聲以獲得穩定的性能表現
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationEnvCfg_PLAY",
        # 訓練配置與訓練環境相同（雖然在 Play 模式下通常不會進行訓練）
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
