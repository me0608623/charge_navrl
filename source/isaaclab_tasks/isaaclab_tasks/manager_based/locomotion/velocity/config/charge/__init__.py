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
from .cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    ChargeNavigationEnvCfgV2,
    ChargeNavigationEnvCfgV2_PLAY,
    ChargeNavigationEnvCfgV3,
    ChargeNavigationEnvCfgV3_PLAY,
    ChargeNavigationEnv,
)  # 導入環境配置類別和環境類

##
# 註冊 Gymnasium 環境 (Register Gymnasium Environments)
##

# 註冊訓練環境：用於強化學習訓練
# 使用自定義環境類 ChargeNavigationEnv（帶有觀測檢查，防止 PPO std>=0 錯誤）

# v0：無障礙物（使用 Phase 3 配置：只有牆壁，無障礙物）
gym.register(
    id="Isaac-Navigation-Charge-v0",
    entry_point=f"{__name__}.cfg.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:ChargeNavigationPPORunnerCfgPhase3",
    },
)

# v1：Phase 1（3 個靜態障礙物）
gym.register(
    id="Isaac-Navigation-Charge-v1",
    entry_point=f"{__name__}.cfg.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg:ChargeNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChargeNavigationPPORunnerCfg",
    },
)

# v2：Phase 2（5 個靜態障礙物）
gym.register(
    id="Isaac-Navigation-Charge-v2",
    entry_point=f"{__name__}.cfg.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v2:ChargeNavigationEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2:ChargeNavigationPPORunnerCfgPhase2",
    },
)

# v3：Phase 3（目標距離課程學習，無障礙物）- 保留作為向後兼容
gym.register(
    id="Isaac-Navigation-Charge-v3",
    entry_point=f"{__name__}.cfg.charge_env:ChargeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:ChargeNavigationPPORunnerCfgPhase3",
    },
)

# 註冊測試/演示環境：用於測試訓練好的策略或進行演示

# v0：無障礙物（使用 Phase 3 配置：只有牆壁，無障礙物）
gym.register(
    id="Isaac-Navigation-Charge-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:ChargeNavigationPPORunnerCfgPhase3",
    },
)

# v1：Phase 1（3 個靜態障礙物）
gym.register(
    id="Isaac-Navigation-Charge-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg:ChargeNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChargeNavigationPPORunnerCfg",
    },
)

# v2：Phase 2（5 個靜態障礙物）
gym.register(
    id="Isaac-Navigation-Charge-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v2:ChargeNavigationEnvCfgV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2:ChargeNavigationPPORunnerCfgPhase2",
    },
)

# v3：Phase 3（目標距離課程學習，無障礙物）- 保留作為向後兼容
gym.register(
    id="Isaac-Navigation-Charge-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:ChargeNavigationPPORunnerCfgPhase3",
    },
)