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
    # ========================================
    # 新地形 Curriculum 配置
    # ========================================
    ChargeNavigationEnvCfgPhase0,
    ChargeNavigationEnvCfgPhase0_PLAY,
    ChargeNavigationEnvCfgPhase1,
    ChargeNavigationEnvCfgPhase1_PLAY,
    ChargeNavigationEnvCfgPhase2,
    ChargeNavigationEnvCfgPhase2_PLAY,
    ChargeNavigationEnvCfgPhase3,
    ChargeNavigationEnvCfgPhase3_PLAY,
    # ========================================
    # 🆕 NavRL 風格配置
    # ========================================
    ChargeNavigationEnvCfgPhase0NavRL,
    # ========================================
    # 舊版配置（保留兼容性）
    # ========================================
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    ChargeNavigationEnvCfgV2,
    ChargeNavigationEnvCfgV2_PLAY,
    ChargeNavigationEnvCfgV3,
    ChargeNavigationEnvCfgV3_PLAY,
    # 環境類
    ChargeNavigationEnv,
    HierarchicalChargeNavigationEnv,
)  # 導入環境配置類別和環境類

##
# 註冊 Gymnasium 環境 (Register Gymnasium Environments)
##

# 註冊訓練環境：用於強化學習訓練
# 使用自定義環境類 ChargeNavigationEnv（帶有觀測檢查，防止 PPO std>=0 錯誤）

# ============================================================================
# 新地形 Curriculum - 推薦使用
# ============================================================================
# 設計理念：增加地形複雜度，而不只是障礙物密度
# - Phase 0: 空房間（車輛動力學校準）
# - Phase 1: 內牆結構（U型牆、隔間）
# - Phase 2: 走廊與窄通道
# - Phase 3: 複雜地形 + 動態障礙物

# Phase 0: 空房間（車輛動力學校準）
# 🔥 修復：使用硬編碼模組路徑代替 agents.__name__，確保 YAML 配置正確加載
# 🚀 突破配置：使用突破版 PPO 配置（三管齊下策略）
gym.register(
    id="Isaac-Navigation-Charge-Phase0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_breakthrough.yaml",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:skrl_ppo_cfg_phase0.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0_PLAY",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase0.yaml",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:skrl_ppo_cfg_phase0.yaml",
    },
)

# 🚀 Phase 0: 突破 13% 瓶頸配置（三管齊下策略）
gym.register(
    id="Isaac-Navigation-Charge-Phase0-Breakthrough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_breakthrough.yaml",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:skrl_ppo_cfg_phase0.yaml",
    },
)

# 🆕 Phase 0 NavRL: 使用 NavRL 風格獎勵函數
# 移植自 NavRL (IEEE RA-L 2025)
gym.register(
    id="Isaac-Navigation-Charge-Phase0-NavRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0_navrl:ChargeNavigationEnvCfgPhase0NavRL",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase0.yaml",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:skrl_ppo_cfg_phase0.yaml",
    },
)

# Phase 1: 內牆結構（U型牆、隔間）
gym.register(
    id="Isaac-Navigation-Charge-Phase1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase1:ChargeNavigationEnvCfgPhase1",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase1.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase1:ChargeNavigationEnvCfgPhase1_PLAY",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase1.yaml",
    },
)

# Phase 2: 走廊與窄通道
gym.register(
    id="Isaac-Navigation-Charge-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase2:ChargeNavigationEnvCfgPhase2",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase2.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase2:ChargeNavigationEnvCfgPhase2_PLAY",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase2.yaml",
    },
)

# Phase 3: 複雜地形 + 動態障礙物
gym.register(
    id="Isaac-Navigation-Charge-Phase3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase3:ChargeNavigationEnvCfgPhase3",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase3.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase3-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase3:ChargeNavigationEnvCfgPhase3_PLAY",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents:sb3_ppo_cfg_phase3.yaml",
    },
)

# ============================================================================
# 舊版 Curriculum（保留兼容性）
# ============================================================================

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

## ==============================================================================
## Stable Baselines 3 (SB3) 環境註冊
## ==============================================================================

# 🔥 修復：使用硬編碼模組路徑代替 agents.__name__，確保 YAML 配置正確加載
_agents_module = "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.agents"

# v0-SB3：無障礙物（使用 Phase 3 配置，SB3 PPO 算法，基礎配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg.yaml",
    },
)

# v1-SB3：Phase 1（3 個靜態障礙物，SB3 PPO 算法，基礎配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg:ChargeNavigationEnvCfg",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg.yaml",
    },
)

# v2-SB3：Phase 2（5 個靜態障礙物，SB3 PPO 算法，Phase 2 配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v2:ChargeNavigationEnvCfgV2",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg_v2.yaml",
    },
)

# v3-SB3：Phase 3（目標距離課程學習，無障礙物，SB3 PPO 算法，Phase 3 配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg_v3.yaml",
    },
)

# 註冊測試/演示環境（SB3）

# v0-SB3-Play：無障礙物（SB3 PPO 算法，基礎配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3_PLAY",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg.yaml",
    },
)

# v1-SB3-Play：Phase 1（3 個靜態障礙物，SB3 PPO 算法，基礎配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg:ChargeNavigationEnvCfg_PLAY",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg.yaml",
    },
)

# v2-SB3-Play：Phase 2（5 個靜態障礙物，SB3 PPO 算法，Phase 2 配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v2:ChargeNavigationEnvCfgV2_PLAY",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg_v2.yaml",
    },
)

# v3-SB3-Play：Phase 3（目標距離課程學習，無障礙物，SB3 PPO 算法，Phase 3 配置）
gym.register(
    id="Isaac-Navigation-Charge-SB3-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v3:ChargeNavigationEnvCfgV3_PLAY",
        "sb3_cfg_entry_point": f"{_agents_module}:sb3_ppo_cfg_v3.yaml",
    },
)

## ==============================================================================
## 層級式導航環境註冊 (Hierarchical Navigation with AIT*)
## ==============================================================================
# 層級式導航環境：整合 AIT* 全局路徑規劃 + RL 局部控制器
# - AIT* 規劃全局路徑 (1-5 Hz)
# - RL 跟隨局部目標 (50-100 Hz)
# - 視覺化 AIT* 路徑（綠色線條和點）
# - 支持動態重規劃（卡住、偏離路徑、無進步時觸發）

# 工廠函數：創建層級式導航環境
def make_hierarchical_env_v0(**kwargs):
    """工廠函數：創建帶 AIT* 的層級式導航環境 v0"""
    from .cfg.charge_env_cfg_v3 import ChargeNavigationEnvCfgV3
    from .cfg.charge_env import HierarchicalChargeNavigationEnv

    # 處理 num_envs 參數
    num_envs = kwargs.pop('num_envs', None)

    # 創建配置
    cfg = ChargeNavigationEnvCfgV3()
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return HierarchicalChargeNavigationEnv(cfg=cfg, **kwargs)

def make_hierarchical_env_v1(**kwargs):
    """工廠函數：創建帶 AIT* 的層級式導航環境 v1"""
    from .cfg.charge_env_cfg import ChargeNavigationEnvCfg
    from .cfg.charge_env import HierarchicalChargeNavigationEnv

    # 處理 num_envs 參數
    num_envs = kwargs.pop('num_envs', None)

    # 創建配置
    cfg = ChargeNavigationEnvCfg()
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return HierarchicalChargeNavigationEnv(cfg=cfg, **kwargs)

# v0-Hierarchical：無障礙物（使用 Phase 3 配置，帶 AIT* 全局路徑規劃）
gym.register(
    id="Isaac-Navigation-Charge-Hierarchical-v0",
    entry_point=make_hierarchical_env_v0,
    disable_env_checker=True,
)

# v1-Hierarchical：Phase 1（3 個靜態障礙物，帶 AIT* 全局路徑規劃）
gym.register(
    id="Isaac-Navigation-Charge-Hierarchical-v1",
    entry_point=make_hierarchical_env_v1,
    disable_env_checker=True,
)

## ==============================================================================
## Domain Randomization 環境註冊 (實驗性功能 - 待實現)
## ==============================================================================
# Domain Randomization (域隨機化) 可以幫助 PPO 模型更好地泛化到真實機器人
#
# TODO: 實現域隨機化功能
# - 需要創建專門的環境配置類 (ChargeNavigationEnvCfgDR)
# - 需要實現隨機化管理器 (physics_randomization, sensor_noise, external_disturbances)
# - 需要在環境 reset 時應用隨機化
#
# 目前的 domain_randomization/ 目錄只有接口聲明，沒有實際實現
# 如果需要使用域隨機化，請：
# 1. 創建 cfg/charge_env_cfg_dr.py 繼承 ChargeNavigationEnvCfg
# 2. 在其中配置 randomize 屬性
# 3. 註冊新環境並替換下面的註冊代碼
#
# 示例配置（未實現）：
# # Domain Randomization - Light 輕微
# gym.register(
#     id="Isaac-Navigation-Charge-DR-Light-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_dr:ChargeNavigationEnvCfgDRLight",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
# )
# ==============================================================================