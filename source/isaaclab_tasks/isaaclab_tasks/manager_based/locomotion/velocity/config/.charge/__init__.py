# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務模組初始化文件

此模組負責將 Charge 機器人的導航環境註冊到 Gymnasium 環境註冊表中，
使其可以通過標準的 gym.make() 接口創建和使用。
"""

import gymnasium as gym  # Gymnasium 環境註冊庫（OpenAI Gym 的後繼者）

from . import agents  # 導入 agents 子模組（包含強化學習演算法配置）
from .charge_env_cfg import ChargeNavigationEnvCfg, ChargeNavigationEnvCfg_PLAY  # 導入環境配置類別
from .charge_env_cfg_v2 import ChargeNavigationEnvCfgV2, ChargeNavigationEnvCfgV2_PLAY  # Phase 2 配置
from .charge_env_cfg_v2_5 import ChargeNavigationEnvCfgV2_5, ChargeNavigationEnvCfgV2_5_PLAY  # Phase 2.5 配置（自適應課程學習）
from .charge_env import ChargeNavigationEnv  # 導入自定義環境類（帶有觀測檢查）

##
# 註冊 Gymnasium 環境 (Register Gymnasium Environments)
##

# 註冊訓練環境：用於強化學習訓練
# 使用自定義環境類 ChargeNavigationEnv（帶有觀測檢查，防止 PPO std>=0 錯誤）
gym.register(
    id="Isaac-Navigation-Charge-v0",  # 環境 ID：用於 gym.make() 的唯一標識符
    entry_point=f"{__name__}.charge_env:ChargeNavigationEnv",  # 環境類別的入口點：自定義環境類（帶有觀測檢查）
    disable_env_checker=True,  # 禁用環境檢查器：跳過環境規範檢查（提高啟動速度）
    kwargs={
        # 環境配置入口點：指向 ChargeNavigationEnvCfg 類別
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg:ChargeNavigationEnvCfg",
        # 強化學習演算法配置入口點：指向 PPO 訓練配置
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChargeNavigationPPORunnerCfg",
    },
)

# 註冊 Phase 2 訓練環境：5 個靜態障礙物
gym.register(
    id="Isaac-Navigation-Charge-v1",
    entry_point=f"{__name__}.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg_v2:ChargeNavigationEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2:ChargeNavigationPPORunnerCfgPhase2",
    },
)

# 註冊 Phase 2.5 訓練環境：自適應課程學習（根據成功率和碰撞率動態調整難度）
gym.register(
    id="Isaac-Navigation-Charge-v2.5",
    entry_point=f"{__name__}.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg_v2_5:ChargeNavigationEnvCfgV2_5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2_5:ChargeNavigationPPORunnerCfgPhase2_5",
    },
)

# 註冊測試/演示環境：用於測試訓練好的策略或進行演示
gym.register(
    id="Isaac-Navigation-Charge-Play-v0",  # 環境 ID：Play 版本用於測試和演示
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 環境類別入口點：與訓練環境相同
    disable_env_checker=True,  # 禁用環境檢查器
    kwargs={
        # 環境配置入口點：使用 Play 版本的配置（減少環境數量、關閉觀測腐化）
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg:ChargeNavigationEnvCfg_PLAY",
        # 強化學習演算法配置入口點：與訓練環境相同（用於載入訓練好的模型）
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChargeNavigationPPORunnerCfg",
    },
)

# 註冊 Phase 2 測試/展示環境
gym.register(
    id="Isaac-Navigation-Charge-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg_v2:ChargeNavigationEnvCfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2:ChargeNavigationPPORunnerCfgPhase2",
    },
)

# 註冊 Phase 2.5 測試/展示環境：自適應課程學習
gym.register(
    id="Isaac-Navigation-Charge-Play-v2.5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charge_env_cfg_v2_5:ChargeNavigationEnvCfgV2_5_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2_5:ChargeNavigationPPORunnerCfgPhase2_5",
    },
)