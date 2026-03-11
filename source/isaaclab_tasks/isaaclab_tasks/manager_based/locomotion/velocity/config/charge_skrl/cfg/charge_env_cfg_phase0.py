# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Phase 0: 車輛動力學與控制校準（SKRL 版本）

核心目標：
1. 車輛動力學校準：理解 (v, omega) 與輸出速度的關係
2. 旋轉 vs 前進：學會何時原地旋轉、何時前進
3. 煞車距離：學會接近目標時減速
4. 動作平滑：避免輪子左右狂抖

環境特點：
- 16x16m 房間，四面牆壁，混合障礙物
- 機器人重生在 ±5m 範圍（確保離牆 3m 以上）
- 目標距離 3-8m
- 非對稱 Actor-Critic (AAC) 架構
  - Policy: 112 維（72 LiDAR + 2 vxy + 1 wz + 2 goal + 1 dist + 1 time + 1 alive + 2 action + 30 topk）
  - Critic: 162 維（112 + 50 obstacles_state 特權資訊）

獎勵設計（NavRL + PBRS）：
- reaching_goal (weight=+500): 到達目標 sparse reward
- collision_terminal (weight=-500): 碰撞 sparse penalty
- potential_progress (weight=+8.0): PBRS 距離縮減
- navrl_velocity (weight=+1.0): 速度方向投影
- lidar_log_safety (weight=+0.5): LiDAR 對數安全距離
- smooth_collision (weight=-3.0): 二次方接近懲罰
- navrl_smoothness (weight=-0.1): 速度變化量懲罰
- time_penalty (weight=-0.1): 每步懲罰
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .charge_env_cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    CommandsCfg,
    EventCfg,
    GoalCommandCfg,
    COLLISION_THRESHOLD,
    GOAL_REACH_THRESHOLD,
    ROBOT_BODY_RADIUS,
    TerminationsCfg,
    MySceneCfg,
    MAX_OBSTACLES,
)

# 觀測函數
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

# 獎勵函數
from ..mdp.rewards import (
    reaching_goal,
    potential_progress_reward,
    smooth_collision_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    collision_occurred,
)
from ..mdp.temp.rewards.navrl_rewards import (
    navrl_velocity_reward,
    navrl_smoothness_penalty,
    navrl_safety_reward_lidar,
)

# 終止條件
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
)

# 事件
from ..mdp.events import (
    reset_root_state_fixed_per_env,
    randomize_obstacles_by_difficulty,
    move_obstacles_vectorized,
)

from isaaclab.envs.mdp.terminations import time_out

# 域隨機化
from ..domain_randomization import apply_domain_randomization


# ============================================================================
# Phase 0 觀測配置（非對稱 AAC 架構）
# ============================================================================
@configclass
class ObservationsCfgPhase0:
    """Phase 0 觀測配置 - 非對稱 Actor-Critic

    Policy: 112 維 = 72 LiDAR + 2 vxy + 1 wz + 2 goal + 1 dist + 1 time + 1 alive + 2 action + 30 topk
    Critic: 162 維 = 112 + 50 obstacles_state（特權資訊）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組 - 112 維"""
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
            func=topk_obstacles_goal_centric,
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
        """Critic 觀測組 - 162 維（112 + 50 特權）"""
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
            func=topk_obstacles_goal_centric,
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
# Phase 0 獎勵配置（NavRL + PBRS）
# ============================================================================
@configclass
class RewardsCfgPhase0:
    """Phase 0 NavRL-Style Reward Configuration

    Terminal: reaching_goal(+500), collision_terminal(-500)
    Shaping: potential_progress(+8.0), navrl_velocity(+1.0)
    Safety: lidar_log_safety(+0.5), smooth_collision(-3.0)
    Regularization: navrl_smoothness(-0.1), time_penalty(-0.1)
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
    lidar_log_safety = RewTerm(
        func=navrl_safety_reward_lidar,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "lidar_range": 10.0},
        weight=0.5,
    )
    smooth_collision = RewTerm(
        func=smooth_collision_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "warn_dist": 1.8, "min_dist": COLLISION_THRESHOLD},
        weight=-3.0,
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


# ============================================================================
# Phase 0 終止條件
# ============================================================================
@configclass
class TerminationsCfgPhase0:
    """Phase 0 終止條件

    - time_out: Episode 超時
    - goal_reached: 到達目標（邊緣距離 < 0.5m）
    - robot_tipped_over: 翻倒
    - collision: LiDAR 碰撞（threshold = 1.0m）
    """
    time_out = DoneTerm(func=time_out, time_out=True)
    goal_reached = DoneTerm(
        func=goal_reached,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
    )
    robot_tipped_over = DoneTerm(
        func=robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    collision = DoneTerm(
        func=collision_occurred,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
    )


# ============================================================================
# Phase 0 場景（16x16m 房間 + 四面牆 + 10 個障礙物）
# ============================================================================
@configclass
class MySceneCfgPhase0(MySceneCfg):
    """Phase 0 場景 - 16x16m 房間，四面牆壁，10 個混合障礙物

    混合平行環境：
    - 50% empty, 30% static(5), 20% dynamic(8)
    - 障礙物初始隱藏在 Z = -10.0
    """

    def __post_init__(self):
        InteractiveSceneCfg.__post_init__(self)

        room_size = 8.0
        wall_thickness = 0.2
        wall_height = 1.5
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, metallic=0.1)

        # 四面牆壁
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, room_size, wall_height / 2)),
        )
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -room_size, wall_height / 2)),
        )
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(room_size, 0.0, wall_height / 2)),
        )
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-room_size, 0.0, wall_height / 2)),
        )

        # LiDAR 配置更新：包含牆壁
        self.lidar.mesh_prim_paths = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Wall_.*", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Obstacle_.*", track_mesh_transforms=True),
        ]

        # 10 個混合障礙物（初始隱藏在 Z = -10.0）
        HIDDEN_Z = -10.0
        obstacle_configs = [
            {"type": "cuboid", "size": (0.5, 0.5, 1.2), "color": (0.8, 0.2, 0.2)},
            {"type": "cylinder", "radius": 0.3, "height": 1.0, "color": (0.8, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.7, 0.7, 1.4), "color": (0.2, 0.4, 0.8)},
            {"type": "cylinder", "radius": 0.25, "height": 0.8, "color": (0.2, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.6, 0.6, 1.0), "color": (0.8, 0.2, 0.8)},
            {"type": "cylinder", "radius": 0.35, "height": 1.2, "color": (0.8, 0.5, 0.2)},
            {"type": "cuboid", "size": (0.4, 0.4, 0.9), "color": (0.2, 0.8, 0.8)},
            {"type": "cylinder", "radius": 0.2, "height": 1.5, "color": (0.5, 0.5, 0.5)},
            {"type": "cuboid", "size": (0.55, 0.55, 1.1), "color": (0.9, 0.9, 0.9)},
            {"type": "cylinder", "radius": 0.28, "height": 1.1, "color": (0.3, 0.3, 0.3)},
        ]

        obstacle_sizes: list[float] = []
        for i, cfg in enumerate(obstacle_configs):
            if cfg["type"] == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=cfg["size"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                height_i = cfg["size"][2] / 2
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"], height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                height_i = cfg["height"] / 2
                size_scalar = cfg["radius"] * 2.0

            setattr(self, f"obstacle_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
            ))
            obstacle_sizes.append(size_scalar)

        from ..mdp.events.state import set_obstacle_metadata
        set_obstacle_metadata(10, obstacle_sizes)


# ============================================================================
# Phase 0 事件配置
# ============================================================================
@configclass
class EventCfgPhase0:
    """Phase 0 事件：重置 + 混合平行 + 障礙物移動 + 域隨機化"""
    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={},
    )
    reset_obstacles = None  # 由 randomize_obstacles 處理
    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={"enable_physics": True, "enable_sensor_noise": True, "enable_external_force": True},
    )
    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 0.5, "static_ratio": 0.3, "dynamic_ratio": 0.2,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50, "boundary": 7.5,
        },
    )
    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 0.5, "static_ratio": 0.3, "dynamic_ratio": 0.2,
            "num_obstacles_static": 5, "num_obstacles_dynamic": 8,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0, "max_spawn_attempts": 50, "boundary": 7.5,
        },
    )
    move_dynamic_obstacles = EventTerm(
        func=move_obstacles_vectorized,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "move_dt": 0.2, "speed_min": 0.3, "speed_max": 1.2,
            "goal_reach_threshold": 0.5, "speed_resample_steps": 10,
            "area_limit": 6.0, "max_obstacles": 10, "bound_limit": 7.0,
        },
    )


# ============================================================================
# Phase 0 命令配置
# ============================================================================
@configclass
class CommandsCfgPhase0:
    """Phase 0 命令：目標距離 3-8m，全方位"""
    goal_command = GoalCommandCfg(
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
    )


# ============================================================================
# Phase 0 完整環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgPhase0(ChargeNavigationEnvCfg):
    """Phase 0 完整環境配置"""

    scene: MySceneCfgPhase0 = MySceneCfgPhase0(num_envs=256, env_spacing=18.0)
    observations: ObservationsCfgPhase0 = ObservationsCfgPhase0()
    rewards: RewardsCfgPhase0 = RewardsCfgPhase0()
    terminations: TerminationsCfgPhase0 = TerminationsCfgPhase0()
    events: EventCfgPhase0 = EventCfgPhase0()
    commands: CommandsCfgPhase0 = CommandsCfgPhase0()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.seed = 42


@configclass
class ChargeNavigationEnvCfgPhase0_PLAY(ChargeNavigationEnvCfgPhase0):
    """Phase 0 播放配置"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 18.0
