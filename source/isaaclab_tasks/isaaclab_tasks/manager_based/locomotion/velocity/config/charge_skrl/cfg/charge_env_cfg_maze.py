# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
迷宮導航環境配置（SKRL 版本）

核心特點：
- 非競爭式，5 目標，論文式 9 維障礙物觀測
- 使用標準 ManagerBasedRLEnv（非 CompetitiveNavigationEnv）
- 基於 Phase 0 場景但使用 MultiGoalCommand
"""

import math

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0,
    MySceneCfgPhase0,
    RewardsCfgPhase0,
    TerminationsCfgPhase0,
    EventCfgPhase0,
    MAX_OBSTACLES,
)

from ..multi_goal_command import MultiGoalCommandCfg
from ..goal_command import GoalCommandCfg

from ..mdp.observations import (
    base_velocity_xy,
    base_angular_velocity_z,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    lidar_scan_2d_sweep,
    safe_last_action,
    topk_obstacles_goal_centric,
    dynamic_obstacles_state,
)
from ..mdp.observations.obstacle_observations import topk_obstacles_paper_format


# ============================================================================
# Maze 命令配置（MultiGoalCommand，5 個目標）
# ============================================================================
@configclass
class CommandsCfgMaze:
    """迷宮命令：5 個同時存在的目標"""
    goal_command = MultiGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(3.0, 8.0),
            angle=(-math.pi, math.pi),
        ),
        wall_boundary=7.5,
        wall_safe_margin=0.5,
        obstacle_safe_distance=1.0,
        num_obstacles=0,
        num_goals=5,
    )


# ============================================================================
# Maze 觀測配置（論文式 9 維障礙物觀測）
# ============================================================================
@configclass
class ObservationsCfgMaze:
    """迷宮觀測配置

    Policy: 與 Phase 0 相同但使用 paper-format 障礙物觀測
    Critic: Policy + obstacles_state 特權資訊
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組"""
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_velocity_xy = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_angular_velocity_z = ObsTerm(
            func=base_angular_velocity_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        goal_distance = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        time_remaining_ratio = ObsTerm(func=time_remaining_ratio)
        alive_flag = ObsTerm(func=alive_flag)
        safe_last_action = ObsTerm(func=safe_last_action)
        # 論文式障礙物觀測（9 維 per obstacle）
        topk_obstacles = ObsTerm(
            func=topk_obstacles_paper_format,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "top_k": 5,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 8.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 觀測組 - Policy + 特權資訊"""
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_velocity_xy = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_angular_velocity_z = ObsTerm(
            func=base_angular_velocity_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        goal_distance = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        time_remaining_ratio = ObsTerm(func=time_remaining_ratio)
        alive_flag = ObsTerm(func=alive_flag)
        safe_last_action = ObsTerm(func=safe_last_action)
        topk_obstacles = ObsTerm(
            func=topk_obstacles_paper_format,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "top_k": 5,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 8.0,
            },
        )
        # Critic 特權資訊
        obstacles_state = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 0,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 15.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================
# Maze 完整環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgMaze(ChargeNavigationEnvCfgPhase0):
    """迷宮導航環境配置

    繼承 Phase 0，覆蓋：
    - 觀測：使用論文式障礙物觀測
    - 命令：MultiGoalCommand（5 個目標）
    """
    observations: ObservationsCfgMaze = ObservationsCfgMaze()
    commands: CommandsCfgMaze = CommandsCfgMaze()
