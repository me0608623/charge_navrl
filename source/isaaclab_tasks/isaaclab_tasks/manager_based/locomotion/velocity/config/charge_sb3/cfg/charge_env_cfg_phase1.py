# ============================================================================
# Phase 1: 內牆結構導航 (Internal Wall Navigation)
# ============================================================================
"""
Phase 1 - 內牆結構導航環境

這是第二個訓練階段，引入內部牆壁結構，強迫機器人學會「繞路」。

核心目標：
1. 繞路策略：理解無法直線到達，必須繞過牆壁
2. U型牆：學會進入死胡同後返回
3. 隔間導航：在房間之間找到門口

環境特點：
- 16x16m 房間，四面有牆
- 內部有 U 型牆結構（產生死胡同）
- 內部有隔間牆（需要找門口）
- 無動態障礙物（專注於靜態地形）

AIT* 規劃：
- 必須規劃繞過內牆的路徑
- 路徑會包含轉彎和繞行
- 測試 Global Planner 在複雜地形的表現

預期訓練結果：
- 成功率 > 85%
- 能夠繞過 U 型牆
- 能夠找到並穿過門口
訓練命令：
    ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase1 \\
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

# 導入基礎觀測和獎勵函數
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

# 導入層級式導航觀測（用於 AIT* 局部目標）
from ..mdp.observations.hierarchical_navigation import (
    local_goal_cartesian,
)

# 導入固定拓撲觀測系統 (122 維固定觀測空間)
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

# 導入 AIT* 整合事件
from ..mdp.events import plan_aitstar_and_update_local_goal
from ..mdp.events import reset_root_state_fixed_per_env

# 導入域隨機化事件 🆕
from ..domain_randomization import apply_domain_randomization


# ============================================================================
# Phase 1 觀測配置（與 Phase 0 保持一致）
# ============================================================================

@configclass
class ObservationsCfgPhase1:
    """Phase 1 觀測配置 - 固定拓撲觀測系統 (122 維)"""

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
# Phase 1 獎勵配置
# ============================================================================

@configclass
class RewardsCfgPhase1:
    """Phase 1 獎勵配置 - 內牆結構導航"""

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

    wall_proximity_penalty = RewTerm(
        func=lidar_clearance_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
        },
        weight=0.5,
    )


# ============================================================================
# Phase 1 終止條件配置
# ============================================================================

@configclass
class TerminationsCfgPhase1:
    """Phase 1 終止條件配置"""

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
# Phase 1 場景配置（內牆結構）
# ============================================================================

@configclass
class MySceneCfgPhase1(MySceneCfg):
    """Phase 1 場景配置 - 16x16m 房間 + 內牆結構

    地形設計：
    - 外牆：四面牆（與 Phase 0 相同）
    - U 型牆：產生死胡同效果
    - 隔間牆：帶門口的隔間
    """

    def __post_init__(self):
        """初始化場景（內牆結構版本）"""
        InteractiveSceneCfg.__post_init__(self)

        # ============================================================================
        # 外牆配置（與 Phase 0 相同）
        # ============================================================================
        room_size = 8.0  # 半徑 8m = 16x16m 房間
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

        # 北牆 (Y = +8m)
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

        # 南牆 (Y = -8m)
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

        # 東牆 (X = +8m)
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

        # 西牆 (X = -8m)
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
        # 內牆結構：U 型牆（產生死胡同）
        # ============================================================================
        # U 型牆開口朝向東邊，機器人必須進入後從同一側退出
        u_wall_color = (0.6, 0.4, 0.3)  # 棕色

        u_wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=u_wall_color,
            metallic=0.1,
        )

        # U 型牆 - 上段（橫向）
        self.u_wall_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/U_Wall_Top",
            spawn=sim_utils.CuboidCfg(
                size=(4.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=u_wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2.0, 3.0, wall_height / 2),
            ),
        )

        # U 型牆 - 左段（縱向）
        self.u_wall_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/U_Wall_Left",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 3.0, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=u_wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-4.0, 1.5, wall_height / 2),
            ),
        )

        # U 型牆 - 下段（橫向，留出開口）
        self.u_wall_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/U_Wall_Bottom",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=u_wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-2.5, 0.0, wall_height / 2),
            ),
        )

        # ============================================================================
        # 內牆結構：隔間牆（帶門口）
        # ============================================================================
        # 在右側創建一個隔間，門口在左側
        partition_color = (0.4, 0.5, 0.6)  # 藍灰色

        partition_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=partition_color,
            metallic=0.1,
        )

        # 隔間上牆（帶門口）
        door_width = 1.5  # 門口寬度
        partition_top_length = 2.5  # 上段長度

        self.partition_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Partition_Top",
            spawn=sim_utils.CuboidCfg(
                size=(partition_top_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=partition_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(4.0 + partition_top_length / 2, 2.0, wall_height / 2),
            ),
        )

        # 隔間下牆（帶門口）
        partition_bottom_length = 2.5

        self.partition_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Partition_Bottom",
            spawn=sim_utils.CuboidCfg(
                size=(partition_bottom_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=partition_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(4.0 + partition_bottom_length / 2, -2.0, wall_height / 2),
            ),
        )

        # 隔間側牆（封閉右側）
        self.partition_side = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Partition_Side",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 5.0, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=partition_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(4.0 + 2.5 + door_width + wall_thickness / 2, 0.0, wall_height / 2),
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
                # 檢測外牆
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,
                ),
                # 檢測 U 型牆
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/U_Wall_.*",
                    track_mesh_transforms=False,
                ),
                # 檢測隔間牆
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Partition_.*",
                    track_mesh_transforms=False,
                ),
            ],
            update_period=0.04,
            debug_vis=False,  # 关闭 LiDAR 射线可视化以提升性能（红色长线）
        )


# ============================================================================
# Phase 1 命令配置
# ============================================================================

@configclass
class CommandsCfgPhase1(CommandsCfg):
    """Phase 1 命令配置 - 內牆結構導航"""

    goal_command = GoalCommandCfg(
        asset_name="robot",
        debug_vis=True,  # 顯示紅色箭頭（GUI 模式，headless 自動禁用）
        resampling_time_range=(5.0, 10.0),
        # Phase 1 規格：目標距離需要更遠，因為可能需要繞路
        ranges=GoalCommandCfg.Ranges(
            distance=(4.0, 10.0),  # 4-10 米（考慮繞路）
            angle=(-math.pi, math.pi),  # 全方向 360 度
        ),
    )


# ============================================================================
# Phase 1 事件配置
# ============================================================================

@configclass
class EventCfgPhase1(EventCfg):
    """Phase 1 事件配置 - 內牆結構導航"""

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

    # 重置機器人位置
    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
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
    #         "phase": "phase1",
    #     },
    # )


# ============================================================================
# Phase 1 環境配置主類
# ============================================================================

@configclass
class ChargeNavigationEnvCfgPhase1(ChargeNavigationEnvCfg):
    """Phase 1 導航環境配置 - 內牆結構導航

    這是第二個訓練階段，引入內部牆壁結構。

    環境規格：
    - 地圖：16x16m 房間
    - 外牆：四面牆
    - 內牆：U 型牆 + 隔間牆
    - 無動態障礙物

    畢業標準：
    - 成功率 > 85%
    - 能夠繞過 U 型牆
    - 能夠找到並穿過門口
    """

    scene: MySceneCfgPhase1 = MySceneCfgPhase1(num_envs=256, env_spacing=18.0)

    observations: ObservationsCfgPhase1 = ObservationsCfgPhase1()

    rewards: RewardsCfgPhase1 = RewardsCfgPhase1()

    terminations: TerminationsCfgPhase1 = TerminationsCfgPhase1()

    commands: CommandsCfgPhase1 = CommandsCfgPhase1()

    events: EventCfgPhase1 = EventCfgPhase1()

    episode_length_s = 25.0  # 增加到 25 秒，因為可能需要繞路


@configclass
class EventCfgPhase1_PLAY(EventCfg):
    """Phase 1 Play 模式事件配置 - 啟用 AIT* 路徑規劃"""

    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={
            "pose_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
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
            "phase": "phase1",
            "use_astar": False,
            "use_virtual_planner": True,     # 🔥 使用虛擬規劃器（不執行 AIT*）
        },
    )


@configclass
class ChargeNavigationEnvCfgPhase1_PLAY(ChargeNavigationEnvCfgPhase1):
    """Phase 1 演示環境配置（使用虛擬規劃器）"""

    scene: MySceneCfgPhase1 = MySceneCfgPhase1(num_envs=1, env_spacing=18.0)
    observations: ObservationsCfgPhase1 = ObservationsCfgPhase1()
    events: EventCfgPhase1_PLAY = EventCfgPhase1_PLAY()  # 覆蓋事件配置


__all__ = [
    "ChargeNavigationEnvCfgPhase1",
    "ChargeNavigationEnvCfgPhase1_PLAY",
]
