# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
競爭式多 Agent 導航環境配置（SKRL 版本）

核心特點：
- 5 agents per group，Parameter Sharing
- 競爭式獎勵：第一個到達目標的 Agent 獲得額外獎勵
- 使用 CompetitiveNavigationEnv 環境類
"""

import math

from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from .charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0,
    COLLISION_THRESHOLD,
    GOAL_REACH_THRESHOLD,
    ROBOT_BODY_RADIUS,
)

from ..mdp.rewards import (
    reaching_goal,
    potential_progress_reward,
    collision_terminal_penalty,
    per_step_time_penalty,
)
from ..mdp.temp.rewards.navrl_rewards import (
    navrl_velocity_reward,
    navrl_smoothness_penalty,
)
from ..mdp.rewards.competitive_rewards import (
    competitive_reaching_goal,
    competitive_loser_penalty,
)


@configclass
class RewardsCfgCompetitive:
    """競爭式獎勵配置

    在 Phase 0 基礎上增加競爭式獎勵：
    - competitive_reaching_goal: 第一個到達的 Agent 獲得額外獎勵
    - competitive_loser_penalty: 落後的 Agent 受到懲罰
    """

    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=500.0,
    )
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-500.0,
    )
    potential_progress = RewTerm(
        func=potential_progress_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=8.0,
    )
    navrl_velocity = RewTerm(
        func=navrl_velocity_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=1.0,
    )
    navrl_smoothness = RewTerm(
        func=navrl_smoothness_penalty,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=-0.1,
    )
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=-0.1,
    )

    # 競爭式獎勵
    competitive_goal = RewTerm(
        func=competitive_reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD},
        weight=200.0,
    )
    competitive_loser = RewTerm(
        func=competitive_loser_penalty,
        params={},
        weight=-50.0,
    )


@configclass
class ChargeNavigationEnvCfgCompetitive(ChargeNavigationEnvCfgPhase0):
    """競爭式導航環境配置

    繼承 Phase 0，覆蓋獎勵為競爭式。
    """
    rewards: RewardsCfgCompetitive = RewardsCfgCompetitive()
