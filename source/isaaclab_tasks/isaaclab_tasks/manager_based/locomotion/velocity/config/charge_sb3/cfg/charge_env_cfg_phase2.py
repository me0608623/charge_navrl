# ============================================================================
# Phase 2: 走廊與窄通道導航 (Corridor & Narrow Passage Navigation)
# ============================================================================
"""
Phase 2 - 走廊與窄通道導航環境

這是第三個訓練階段，引入走廊、門口和窄通道，測試精確控制能力。

核心目標：
1. 窄通道控制：學會通過比機器人寬不了多少的通道
2. 走廊導航：在長走廊中保持方向，不碰撞牆壁
3. 門口識別：找到並通過窄門口
4. 精確轉向：在狹窄空間中精確轉向

環境特點：
- 16x16m 房間，四面有牆
- 內部有走廊結構（牆壁形成的通道）
- 多個窄門口（1.2-1.5m 寬）
- T 型路口和十字路口

AIT* 規劃：
- 必須規劃穿過窄門口的路徑
- 路徑會沿著走廊前進
- 需要精確對準門口

預期訓練結果：
- 成功率 > 80%
- 能夠通過 1.2m 寬的門口
- 能夠在走廊中保持方向
- 能夠在 T 型路口選擇正確方向

訓練命令：
    ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase2 \\
        --num_envs 256 \\
        --headless \\
        --agent sb3_cfg_entry_point
"""

import math

from isaaclab.assets import AssetBaseCfg
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
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .charge_env_cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    CommandsCfg,
    EventCfg,
    GoalCommandCfg,
    GOAL_REACH_THRESHOLD,
    ROBOT_BODY_RADIUS,
    TerminationsCfg,
    MySceneCfg,
)

from ..mdp.observations import (
    base_velocity_xy,
    base_angular_velocity_z,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    lidar_scan_2d_sweep,
    safe_last_action,
)

from ..mdp.observations.hierarchical_navigation import (
    local_goal_cartesian,
)

from ..mdp.observations.fixed_topology import (
    nearest_static_obstacles,
    nearest_dynamic_obstacles,
    navigation_command,
    proprioception,
    LIDAR_DIM,
    STATIC_SLOTS_DIM,
    DYNAMIC_SLOTS_DIM,
    NAV_DIM,
    PROPRIOCEPTION_DIM,
    TOTAL_OBS_DIM,
)

from ..mdp.rewards import (
    progress_to_goal,
    reaching_goal,
    velocity_toward_goal,
    forward_velocity_reward,
    time_out_penalty,
    lidar_clearance_reward,
    action_rate_penalty,
    collision_occurred,  # 🔥 LiDAR 碰撞檢測
)

from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    wall_collision,
)

from ..mdp.events import plan_aitstar_and_update_local_goal
from ..mdp.events import reset_root_state_fixed_per_env

# 導入域隨機化事件 🆕
from ..domain_randomization import apply_domain_randomization


# ============================================================================
# Phase 2 觀測配置
# ============================================================================

@configclass
class ObservationsCfgPhase2:
    """Phase 2 觀測配置 - 固定拓撲觀測系統 (122 維)"""

    @configclass
    class PolicyCfg(ObsGroup):
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        static_obstacles = ObsTerm(
            func=nearest_static_obstacles,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "num_slots": 8,
                "max_distance": 8.0,
            },
        )

        dynamic_obstacles = ObsTerm(
            func=nearest_dynamic_obstacles,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "num_slots": 5,
                "max_distance": 10.0,
            },
        )

        navigation_command = ObsTerm(
            func=navigation_command,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        proprioception = ObsTerm(
            func=proprioception,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Phase 2 獎勵配置
# ============================================================================

@configclass
class RewardsCfgPhase2:
    """Phase 2 獎勵配置 - 走廊與窄通道導航"""

    # 主要獎勵
    progress_to_goal = RewTerm(
        func=progress_to_goal,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=20.0,
    )

    reaching_goal = RewTerm(
        func=reaching_goal,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
        },
        weight=10.0,
    )

    velocity_toward_goal = RewTerm(
        func=velocity_toward_goal,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=1.0,
    )

    # 懲罰
    action_smoothness = RewTerm(
        func=action_rate_penalty,
        params={},
        weight=-0.5,
    )

    time_out_penalty = RewTerm(
        func=time_out_penalty,
        weight=-0.05,
    )

    # Phase 2 關鍵：牆壁接近懲罰權重更高（窄通道需要更小心）
    wall_proximity_penalty = RewTerm(
        func=lidar_clearance_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
        },
        weight=1.0,  # 從 0.5 提高到 1.0
    )


# ============================================================================
# Phase 2 終止條件配置
# ============================================================================

@configclass
class TerminationsCfgPhase2:
    """Phase 2 終止條件配置"""

    goal_reached = DoneTerm(
        func=goal_reached,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
        },
    )

    robot_tipped_over = DoneTerm(
        func=robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 🔥 LiDAR 碰撞終止（死亡機制）
    lidar_collision = DoneTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "threshold": 0.5,
        },
    )

    # 牆壁碰撞（位置檢測，作為備用）
    wall_collision = DoneTerm(
        func=wall_collision,
        params={"asset_cfg": SceneEntityCfg("robot"), "boundary": 8.0, "robot_radius": 0.5},
    )


# ============================================================================
# Phase 2 場景配置（走廊與窄通道）
# ============================================================================

@configclass
class MySceneCfgPhase2(MySceneCfg):
    """Phase 2 場景配置 - 16x16m 房間 + 走廊與窄通道

    地形設計：
    - 外牆：四面牆
    - 中央走廊：橫貫中央的主走廊
    - 窄門口：多個 1.2-1.5m 寬的門口
    - T 型路口：走廊交叉路口
    """

    def __post_init__(self):
        """初始化場景（走廊與窄通道版本）"""
        InteractiveSceneCfg.__post_init__(self)

        # ============================================================================
        # 基礎配置
        # ============================================================================
        room_size = 8.0
        wall_thickness = 0.2
        wall_height = 1.5
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=wall_color,
            metallic=0.1,
        )

        # ============================================================================
        # 外牆（四面牆）
        # ============================================================================
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, room_size, wall_height / 2),
            ),
        )

        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -room_size, wall_height / 2),
            ),
        )

        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(room_size, 0.0, wall_height / 2),
            ),
        )

        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-room_size, 0.0, wall_height / 2),
            ),
        )

        # ============================================================================
        # 走廊結構
        # ============================================================================
        corridor_color = (0.5, 0.6, 0.5)  # 綠灰色

        corridor_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=corridor_color,
            metallic=0.1,
        )

        # 中央走廊（橫向）- 上牆
        # 留出兩個門口：左側門口和右側門口
        door_width = 1.3  # 窄門口寬度（機器人寬度 ~1m）

        # 上走廊左段
        self.corridor_top_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Top_Left",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-5.0, 3.0, wall_height / 2),
            ),
        )

        # 上走廊中段（門口之間）
        self.corridor_top_middle = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Top_Middle",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 3.0, wall_height / 2),
            ),
        )

        # 上走廊右段
        self.corridor_top_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Top_Right",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(5.0, 3.0, wall_height / 2),
            ),
        )

        # 中央走廊（橫向）- 下牆
        self.corridor_bottom_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Bottom_Left",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-5.0, -3.0, wall_height / 2),
            ),
        )

        self.corridor_bottom_middle = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Bottom_Middle",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -3.0, wall_height / 2),
            ),
        )

        self.corridor_bottom_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Bottom_Right",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(5.0, -3.0, wall_height / 2),
            ),
        )

        # ============================================================================
        # T 型路口結構（縱向走廊）
        # ============================================================================
        tee_color = (0.6, 0.5, 0.4)  # 棕灰色

        tee_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tee_color,
            metallic=0.1,
        )

        # T 型走廊 - 上段
        self.ee_corridor_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tee_Corridor_Top",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 3.0, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=tee_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 4.5, wall_height / 2),
            ),
        )

        # T 型走廊 - 下段（左右分叉）
        # 左分叉
        self.tee_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tee_Left",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 4.0, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=tee_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2.0, -5.0, wall_height / 2),
            ),
        )

        # 右分叉
        self.tee_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tee_Right",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 4.0, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=tee_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(2.0, -5.0, wall_height / 2),
            ),
        )

        # ============================================================================
        # 窄門口通道（額外的窄通道挑戰）
        # ============================================================================
        narrow_color = (0.7, 0.5, 0.3)  # 深棕色

        narrow_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=narrow_color,
            metallic=0.1,
        )

        # 窄通道 1（右上角）
        narrow1_door_width = 1.2  # 非常窄的門口

        self.narrow1_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Narrow1_Left",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=narrow_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(6.0, 5.5, wall_height / 2),
            ),
        )

        self.narrow1_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Narrow1_Right",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=narrow_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(6.0, 6.7, wall_height / 2),
            ),
        )

        # 窄通道 2（左下角）
        self.narrow2_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Narrow2_Left",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=narrow_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-6.0, -6.0, wall_height / 2),
            ),
        )

        self.narrow2_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Narrow2_Right",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=narrow_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-6.0, -4.8, wall_height / 2),
            ),
        )

        # ============================================================================
        # LiDAR 配置（檢測所有牆壁）
        # ============================================================================
        self.lidar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            ray_alignment="yaw",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=5.0,  # 72 條射線
            ),
            max_distance=10.0,
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/ground",
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Corridor_.*",
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Tee_.*",
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Narrow.*",
                    track_mesh_transforms=False,
                ),
            ],
            update_period=0.04,
            debug_vis=False,  # 关闭 LiDAR 射线可视化以提升性能（红色长线）
        )


# ============================================================================
# Phase 2 命令配置
# ============================================================================

@configclass
class CommandsCfgPhase2(CommandsCfg):
    """Phase 2 命令配置 - 走廊與窄通道導航"""

    goal_command = GoalCommandCfg(
        asset_name="robot",
        debug_vis=True,  # 顯示紅色箭頭（GUI 模式，headless 自動禁用）
        resampling_time_range=(5.0, 10.0),
        # Phase 2 規格：目標可能在走廊另一端
        ranges=GoalCommandCfg.Ranges(
            distance=(5.0, 12.0),  # 5-12 米（需要穿過走廊）
            angle=(-math.pi, math.pi),
        ),
    )


# ============================================================================
# Phase 2 事件配置
# ============================================================================

@configclass
class EventCfgPhase2(EventCfg):
    """Phase 2 事件配置 - 走廊與窄通道導航"""

    # ========== 域隨機化事件 🆕 ==========
    # 在環境重置時應用域隨機化
    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={
            "enable_physics": True,  # 物理參數隨機化
            "enable_sensor_noise": True,  # 傳感器噪聲
            "enable_external_force": True,  # 外部擾動
        },
    )

    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, 6.0),
                "y": (-6.0, 6.0),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )

    # 🔥 訓練模式：禁用 AIT* 路徑規劃
    # 讓 RL Agent 直接學會往目標點前進，同時避開障礙物
    # Play 模式會啟用 AIT* 展示完整導航能力
    #
    # plan_aitstar_path = EventTerm(
    #     func=plan_aitstar_and_update_local_goal,
    #     mode="reset",
    #     params={
    #         "lookahead_distance": 2.0,
    #         "map_size": (16.0, 16.0),
    #         "robot_radius": 0.3,
    #         "visualize_path": True,
    #         "sample_walls_from_usd": True,
    #         "run_verification": True,
    #         "phase": "phase2",
    #     },
    # )


# ============================================================================
# Phase 2 環境配置主類
# ============================================================================

@configclass
class ChargeNavigationEnvCfgPhase2(ChargeNavigationEnvCfg):
    """Phase 2 導航環境配置 - 走廊與窄通道導航

    環境規格：
    - 地圖：16x16m 房間
    - 走廊結構：中央走廊 + T 型路口
    - 窄門口：1.2-1.3m 寬

    畢業標準：
    - 成功率 > 80%
    - 能夠通過 1.2m 寬門口
    - 能夠在走廊中保持方向
    """

    scene: MySceneCfgPhase2 = MySceneCfgPhase2(num_envs=256, env_spacing=18.0)

    observations: ObservationsCfgPhase2 = ObservationsCfgPhase2()

    rewards: RewardsCfgPhase2 = RewardsCfgPhase2()

    terminations: TerminationsCfgPhase2 = TerminationsCfgPhase2()

    commands: CommandsCfgPhase2 = CommandsCfgPhase2()

    events: EventCfgPhase2 = EventCfgPhase2()

    episode_length_s = 30.0  # 增加到 30 秒，走廊路徑更長


@configclass
class EventCfgPhase2_PLAY(EventCfg):
    """Phase 2 Play 模式事件配置 - 啟用 AIT* 路徑規劃"""

    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, 6.0),
                "y": (-6.0, 6.0),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )

    # 🔥 Play 模式：使用虛擬規劃器（不執行 AIT*）
    plan_aitstar_path = EventTerm(
        func=plan_aitstar_and_update_local_goal,
        mode="reset",
        params={
            "lookahead_distance": 2.0,
            "map_size": (16.0, 16.0),
            "robot_radius": 0.3,
            "visualize_path": False,         # 虛擬規劃器不需要路徑可視化
            "sample_walls_from_usd": False,  # 虛擬規劃器不需要牆壁採樣
            "run_verification": False,       # 關閉驗收測試（提升性能）
            "phase": "phase2",
            "use_astar": False,
            "use_virtual_planner": True,     # 🔥 使用虛擬規劃器（不執行 AIT*）
        },
    )


@configclass
class ChargeNavigationEnvCfgPhase2_PLAY(ChargeNavigationEnvCfgPhase2):
    """Phase 2 演示環境配置（使用虛擬規劃器）"""

    scene: MySceneCfgPhase2 = MySceneCfgPhase2(num_envs=1, env_spacing=18.0)
    observations: ObservationsCfgPhase2 = ObservationsCfgPhase2()
    events: EventCfgPhase2_PLAY = EventCfgPhase2_PLAY()  # 覆蓋事件配置


__all__ = [
    "ChargeNavigationEnvCfgPhase2",
    "ChargeNavigationEnvCfgPhase2_PLAY",
]
