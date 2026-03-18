# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務模組（SKRL 專用）

此模組負責將 Charge 機器人的導航環境註冊到 Gymnasium 環境註冊表中，
使用 SKRL 作為強化學習框架。

Phase 0 配置：
- 16x16m 房間，四面牆壁，混合障礙物
- 單 Goal：到達即終止 episode
- Episode 長度 20 秒
"""

import gymnasium as gym

from . import agents
from .cfg import (
    ChargeNavigationEnvCfgPhase0,
    ChargeNavigationEnvCfgPhase0_PLAY,
    ChargeNavigationEnvCfgPhase0NavRL,
    ChargeNavigationEnvCfgCompetitive,
    ChargeNavigationEnvCfgVLP16,
    ChargeNavigationEnvCfgVLP16Phase2,
    ChargeNavigationEnvCfgVLP16Curriculum,
    ChargeNavigationEnvCfgVLP16CurriculumNavRL,
)

# 實驗性配置（依賴模組可能尚未完成）
try:
    from .cfg.charge_env_cfg_maze import ChargeNavigationEnvCfgMaze
    _MAZE_AVAILABLE = True
except ImportError:
    ChargeNavigationEnvCfgMaze = None  # type: ignore[assignment, misc]
    _MAZE_AVAILABLE = False

##
# 註冊 Gymnasium 環境 (SKRL)
##

# SKRL agent config 模組路徑
_skrl_agents = "isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.agents"

# ============================================================================
# Phase 0: 空房間 + 混合障礙物（車輛動力學校準，10-Goal 動態重生）
# ============================================================================

# Phase 0: 標準 MLP 架構
gym.register(
    id="Isaac-Navigation-Charge-Phase0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

gym.register(
    id="Isaac-Navigation-Charge-Phase0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0_PLAY",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

# Phase 0 NavRL: NavRL 風格獎勵函數
gym.register(
    id="Isaac-Navigation-Charge-Phase0-NavRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0_navrl:ChargeNavigationEnvCfgPhase0NavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
    },
)

# Phase 0 NavRL CNN: NavRL 風格獎勵 + CNN 架構
gym.register(
    id="Isaac-Navigation-Charge-Phase0-NavRL-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0_navrl:ChargeNavigationEnvCfgPhase0NavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_navrl.yaml",
    },
)

# Phase 0 CNN: 多分支 CNN 特徵萃取器（LiDAR Conv1d + 排列不變障礙物 + State MLP）
gym.register(
    id="Isaac-Navigation-Charge-Phase0-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_phase0:ChargeNavigationEnvCfgPhase0",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_charge_cnn.yaml",
    },
)

# ============================================================================
# Competitive: 競爭式多 Agent 訓練（實驗性）
# ============================================================================
if ChargeNavigationEnvCfgCompetitive is not None:
    gym.register(
        id="Isaac-Navigation-Charge-Competitive",
        entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.competition.competitive_env:CompetitiveNavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_competitive:ChargeNavigationEnvCfgCompetitive",
            "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_competitive.yaml",
        },
    )

# ============================================================================
# Maze: 迷宮導航（實驗性）
# ============================================================================
if _MAZE_AVAILABLE:
    gym.register(
        id="Isaac-Navigation-Charge-Maze",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_maze:ChargeNavigationEnvCfgMaze",
            "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_phase0.yaml",
        },
    )

# ============================================================================
# VLP-16 v2: 離散動作 + 79D 觀測 baseline
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

# ============================================================================
# VLP-16 Phase 2: 避障微調（從 Phase 1 checkpoint 繼續訓練）
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16:ChargeNavigationEnvCfgVLP16Phase2",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16_phase2.yaml",
    },
)

# ============================================================================
# VLP-16 Curriculum: 20×20m 場景 + 4 階段 Goal-Obstacle 聯動課程學習
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Curriculum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_curriculum:ChargeNavigationEnvCfgVLP16Curriculum",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)

# ============================================================================
# VLP-16 Curriculum NavRL: NavRL-Style Dense Rewards（安全導航密集獎勵）
# ============================================================================
gym.register(
    id="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_vlp16_curriculum:ChargeNavigationEnvCfgVLP16CurriculumNavRL",
        "skrl_cfg_entry_point": f"{_skrl_agents}:skrl_ppo_cfg_vlp16.yaml",
    },
)
