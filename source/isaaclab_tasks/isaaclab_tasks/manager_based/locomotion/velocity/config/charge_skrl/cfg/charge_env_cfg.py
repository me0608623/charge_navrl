# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航環境基礎配置（SKRL 版本）

此文件定義：
1. 場景配置（地面、機器人、LiDAR、障礙物）
2. 觀測配置（機器人能「看到」什麼）
3. 動作配置（機器人能「做」什麼）
4. 獎勵配置（什麼行為會得分/扣分）
5. 終止條件（什麼情況下 episode 結束）
6. 事件配置（重置時發生什麼）
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .charge_cfg import CHARGE_CFG
from ..goal_command import GoalCommandCfg

# 動作類（離散）
from ..mdp.actions import DiscreteDifferentialDriveActionCfg

# 觀測函數
from ..mdp.observations import (
    lidar_scan_2d_sweep,
    base_velocity_xy,
    base_angular_velocity_z,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    dynamic_obstacles_state,
    safe_last_action,
    charge_dies_at_birth_probability,
)

# 獎勵函數
from ..mdp.rewards import (
    heading_to_goal_distance_weighted,
    progress_to_goal,
    reaching_goal,
    approaching_goal_bonus,
    progressive_collision_penalty,
    collision_occurred,
    alignment_reward,
    collision_contact_occurred,
    # Potential-Based
    potential_progress_reward,
    smooth_collision_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
)
# Base config 額外使用的函數（來自 temp 模組）
from ..mdp.temp.rewards.motion_rewards import time_out_penalty, move_reward
from ..mdp.temp.rewards.navrl_rewards import (
    navrl_velocity_reward,
    navrl_safety_reward_lidar,
    navrl_smoothness_penalty,
)

# 終止條件函數
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    robot_flying,
)

# 事件函數
from ..mdp.events import reset_obstacles, reset_root_state_fixed_per_env

# 核心狀態管理
from ..mdp.events.state import set_obstacle_metadata

# ============================================================================
# 環境常數
# ============================================================================
ROBOT_BODY_RADIUS = 0.35  # 實際機器人半徑 [m]
GOAL_REACH_THRESHOLD = ROBOT_BODY_RADIUS
COLLISION_BUFFER = 0.35  # 安全裕量：防止物理穿透導致 Warp CUDA error [m]
COLLISION_THRESHOLD = ROBOT_BODY_RADIUS + COLLISION_BUFFER  # 0.7m

SAFE_DISTANCE = 1.5
DANGER_DISTANCE = 0.8
COLLISION_DISTANCE = 0.4

MIN_ROBOT_DISTANCE = 1.5
MIN_GOAL_DISTANCE = 1.0
MIN_OBSTACLE_SPACING = 1.0

MAX_OBSTACLES = 10


# ============================================================================
# 場景配置
# ============================================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """基礎場景配置：地面、機器人、LiDAR（2D 平面掃描 72 射線）、3 個障礙物"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = CHARGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=10.0,
        debug_vis=True,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
                track_mesh_transforms=True,
            ),
        ],
        update_period=0.04,
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"],
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )

    def __post_init__(self):
        """動態生成 3 個障礙物"""
        super().__post_init__()

        obstacle_configs = [
            {"type": "cuboid", "size": (0.5, 0.5, 1.2), "color": (0.8, 0.2, 0.2), "pos": (0.0, 0.0, 0.6)},
            {"type": "cylinder", "radius": 0.3, "height": 1.0, "color": (0.8, 0.8, 0.2), "pos": (0.0, 0.0, 0.5)},
            {"type": "cuboid", "size": (0.7, 0.7, 1.4), "color": (0.2, 0.4, 0.8), "pos": (0.0, 0.0, 0.7)},
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
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"],
                    height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = cfg["radius"] * 2.0

            setattr(
                self,
                f"obstacle_{i}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=spawn_cfg,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=cfg["pos"]),
                ),
            )
            obstacle_sizes.append(size_scalar)

        set_obstacle_metadata(3, obstacle_sizes)


# ============================================================================
# 命令配置
# ============================================================================
@configclass
class CommandsCfg:
    """命令配置：目標位置生成"""
    goal_command = GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(MIN_ROBOT_DISTANCE, 3.0),
            angle=(-math.pi, math.pi),
        ),
    )


# ============================================================================
# 動作配置
# ============================================================================
@configclass
class ActionsCfg:
    """離散差速驅動動作：Discrete(361) = 19×19 中心對稱 + 動態加速度邊界"""
    diff_drive = DiscreteDifferentialDriveActionCfg(
        asset_name="robot",
        num_bins=19,
        max_linear_velocity=1.0,
        max_linear_accel=0.5,
        max_angular_vel=2.0,    # 2026-05-28: 0.25π → 2.0 rad/s（與 vlp16_curriculum.py 同步）
        debug_vis=True,
    )


# ============================================================================
# 觀測配置（131 維統一觀測空間）
# ============================================================================
@configclass
class ObservationsCfg:
    """基礎觀測配置 - 131 維"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組 - 131 維
        [72 LiDAR + 2 速度 + 2 目標位置 + 1 目標距離 + 1 時間 + 1 alive + 50 障礙物 + 2 動作]
        """
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        speed = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        goal_distance = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        time_remaining = ObsTerm(func=time_remaining_ratio)
        alive = ObsTerm(func=alive_flag)
        obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 3,
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 10.0,
            },
        )
        actions = ObsTerm(func=safe_last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# 獎勵配置
# ============================================================================
@configclass
class RewardsCfg:
    """基礎獎勵配置"""
    move = RewTerm(
        func=move_reward,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 1.0, "b": 0.0},
    )
    heading_to_goal = RewTerm(
        func=heading_to_goal_distance_weighted,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "close_distance": 2.0, "far_distance": 5.0},
    )
    alignment = RewTerm(
        func=alignment_reward,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=500.0,
    )
    progress = RewTerm(
        func=progress_to_goal,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    collision_penalty = RewTerm(
        func=progressive_collision_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": SAFE_DISTANCE,
            "danger_distance": DANGER_DISTANCE,
            "collision_distance": COLLISION_DISTANCE,
        },
    )
    collision = RewTerm(
        func=collision_occurred,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
    )
    time_penalty = RewTerm(
        func=time_out_penalty,
        weight=-0.05,
        params={},
    )


# ============================================================================
# 終止條件配置
# ============================================================================
@configclass
class TerminationsCfg:
    """基礎終止條件"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
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
# 事件配置
# ============================================================================
@configclass
class EventCfg:
    """基礎事件配置"""
    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={},
    )
    reset_obstacles = EventTerm(
        func=reset_obstacles,
        mode="reset",
        params={
            "num_obstacles": 3,
            "boundary": 4.5,
            "min_robot_distance": MIN_ROBOT_DISTANCE,
            "min_goal_distance": MIN_GOAL_DISTANCE,
            "min_obstacle_spacing": MIN_OBSTACLE_SPACING,
        },
    )


# ============================================================================
# 完整環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Charge 導航基礎環境配置"""

    scene: MySceneCfg = MySceneCfg(num_envs=64, env_spacing=12.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.seed = 42


@configclass
class ChargeNavigationEnvCfg_PLAY(ChargeNavigationEnvCfg):
    """播放用配置（少量環境、啟用可視化）"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 12.0
