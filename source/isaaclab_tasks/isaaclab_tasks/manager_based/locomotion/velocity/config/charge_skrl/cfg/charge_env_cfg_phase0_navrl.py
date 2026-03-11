# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Phase 0 NavRL 變體：使用 NavRL（IEEE RA-L 2025）風格獎勵函數

與 Phase 0 共享場景、觀測、終止條件和事件配置，
僅覆蓋獎勵為純 NavRL 風格（更強調速度投影和 LiDAR 安全獎勵）。
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
    smooth_collision_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
)
from ..mdp.temp.rewards.navrl_rewards import (
    navrl_velocity_reward,
    navrl_smoothness_penalty,
    navrl_safety_reward_lidar,
)


@configclass
class RewardsCfgPhase0NavRL:
    """NavRL 風格獎勵配置

    強調 NavRL 核心獎勵：
    - navrl_velocity: v dot r_hat（速度在目標方向的投影）
    - navrl_safety: log-distance LiDAR 安全獎勵
    - navrl_smoothness: 速度變化量懲罰

    Terminal + PBRS 與 Phase 0 一致。
    """

    # Terminal Rewards
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

    # PBRS: 距離縮減
    potential_progress = RewTerm(
        func=potential_progress_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=8.0,
    )

    # NavRL Core: 速度方向投影（權重提升至 2.0，強調 NavRL 風格）
    navrl_velocity = RewTerm(
        func=navrl_velocity_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=2.0,
    )

    # NavRL Core: LiDAR 對數安全距離（權重提升至 1.0）
    lidar_log_safety = RewTerm(
        func=navrl_safety_reward_lidar,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "lidar_range": 10.0},
        weight=1.0,
    )

    # 連續碰撞懲罰
    smooth_collision = RewTerm(
        func=smooth_collision_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "warn_dist": 1.8, "min_dist": COLLISION_THRESHOLD},
        weight=-3.0,
    )

    # NavRL Core: 平滑度懲罰
    navrl_smoothness = RewTerm(
        func=navrl_smoothness_penalty,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=-0.2,
    )

    # 時間懲罰
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=-0.1,
    )


@configclass
class ChargeNavigationEnvCfgPhase0NavRL(ChargeNavigationEnvCfgPhase0):
    """Phase 0 NavRL 變體：使用 NavRL 風格獎勵"""

    rewards: RewardsCfgPhase0NavRL = RewardsCfgPhase0NavRL()
