# ============================================================================
# Phase 3: 複雜地形與動態障礙物 (Complex Terrain + Dynamic Obstacles)
# ============================================================================
"""
Phase 3 - 複雜地形與動態障礙物導航環境

這是最後一個訓練階段，結合複雜地形和動態障礙物，測試完整導航能力。

核心目標：
1. Global + Local 協作：AIT* 規劃繞過靜態牆壁，RL 閃避動態障礙
2. 即時避障：使用 LiDAR 檢測並閃移動障礙物
3. 路徑重評估：當動態障礙物擋住路徑時，等待或繞行
4. 魯棒性：在複雜動態環境中保持穩定導航

環境特點：
- 16x16m 房間，複雜內牆結構（繼承 Phase 1 和 2）
- 動態障礙物：在 episode 中隨機移動
- 障礙物會擋住 AIT* 規劃的路徑

AIT* 規劃：
- 只規劃靜態牆壁的繞路路徑
- **不考慮**動態障礙物（避免 Planner Thrashing）
- 動態障礙物交給 LiDAR + RL 即時處理

RL Policy：
- 使用 LiDAR 檢測動態障礙物
- 即時閃避，不影響 AIT* 全局規劃
- 當障礙物擋路時，等待或重新規劃局部路徑

預期訓練結果：
- 成功率 > 75%（動態環境更難）
- 能夠同時處理靜態牆壁和動態障礙
- 能夠在障礙物擋路時等待或繞行

訓練命令：
    ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase3 \\
        --num_envs 256 \\
        --headless \\
        --agent sb3_cfg_entry_point
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
    dynamic_obstacles_state,
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
    collision_occurred,
)

from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    wall_collision,
)

from ..mdp.events import plan_aitstar_and_update_local_goal, reset_obstacles
from ..mdp.events import reset_root_state_fixed_per_env
from ..mdp.core import set_obstacle_metadata

# 導入域隨機化事件 🆕
from ..domain_randomization import apply_domain_randomization


# ============================================================================
# Phase 3 觀測配置
# ============================================================================

@configclass
class ObservationsCfgPhase3:
    """Phase 3 觀測配置 - 固定拓撲觀測系統 (122 維)

    Phase 3 關鍵：動態障礙物觀測必須準確
    - LiDAR：檢測動態障礙物位置
    - Dynamic Slots：跟蹤移動障礙物速度
    """

    @configclass
    class PolicyCfg(ObsGroup):
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Phase 3 關鍵：動態障礙物現在是真實的，需要檢測
        dynamic_obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 5,  # Phase 3: 5 個動態障礙物
            },
        )

        static_obstacles = ObsTerm(
            func=nearest_static_obstacles,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "num_slots": 8,
                "max_distance": 8.0,
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
# Phase 3 獎勵配置
# ============================================================================

@configclass
class RewardsCfgPhase3:
    """Phase 3 獎勵配置 - 複雜地形與動態障礙物"""

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

    # Phase 3 關鍵：動態障礙物懲罰（降低權重，讓 Agent 敢衝）
    dynamic_collision = RewTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),  # 🔥 使用 LiDAR 檢測
            "threshold": 0.5,
        },
        weight=-1.0,  # 🔥 降低：-10.0 → -1.0（讓 Agent 敢撞）
    )

    wall_proximity_penalty = RewTerm(
        func=lidar_clearance_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
        },
        weight=1.0,
    )


# ============================================================================
# Phase 3 終止條件配置
# ============================================================================

@configclass
class TerminationsCfgPhase3:
    """Phase 3 終止條件配置"""

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

    # 🔥 Phase 3：LiDAR 碰撞終止（死亡機制保留）
    # 策略：碰撞後死亡（重置），但獎勵懲罰很輊
    dynamic_collision = DoneTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),  # 🔥 使用 LiDAR
            "threshold": 0.5,
        },
    )


# ============================================================================
# Phase 3 場景配置（複雜地形 + 動態障礙物）
# ============================================================================

@configclass
class MySceneCfgPhase3(MySceneCfg):
    """Phase 3 場景配置 - 複雜地形 + 動態障礙物

    地形設計：
    - 外牆：四面牆
    - 內牆：結合 Phase 1 和 2 的複雜結構
    - 動態障礙物：5 個移動的圓柱體
    """

    def __post_init__(self):
        """初始化場景（複雜地形 + 動態障礙物版本）"""
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
        # 內牆結構（結合 Phase 1 和 2）
        # ============================================================================
        u_wall_color = (0.6, 0.4, 0.3)
        corridor_color = (0.5, 0.6, 0.5)

        u_wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=u_wall_color,
            metallic=0.1,
        )
        corridor_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=corridor_color,
            metallic=0.1,
        )

        # U 型牆（左上角）
        self.u_wall_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/U_Wall_Top",
            spawn=sim_utils.CuboidCfg(
                size=(3.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=u_wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-4.0, 2.0, wall_height / 2),
            ),
        )

        self.u_wall_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/U_Wall_Left",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, 2.5, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=u_wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-5.5, 0.75, wall_height / 2),
            ),
        )

        # 中央走廊段
        self.corridor_segment = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Corridor_Segment",
            spawn=sim_utils.CuboidCfg(
                size=(5.0, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=corridor_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(2.5, 0.0, wall_height / 2),
            ),
        )

        # ============================================================================
        # 動態障礙物（Phase 3 新增）
        # ============================================================================
        # 5 個動態移動的圓柱體障礙物
        num_dynamic_obstacles = 5

        dynamic_obstacle_configs = [
            # Obstacle 0: 中等大小，紅色
            {
                "radius": 0.4,
                "height": 1.2,
                "color": (0.8, 0.2, 0.2),
                "pos": (0.0, 0.0, 0.6),
            },
            # Obstacle 1: 較小，黃色
            {
                "radius": 0.3,
                "height": 1.0,
                "color": (0.8, 0.8, 0.2),
                "pos": (0.0, 0.0, 0.5),
            },
            # Obstacle 2: 較大，藍色
            {
                "radius": 0.5,
                "height": 1.4,
                "color": (0.2, 0.4, 0.8),
                "pos": (0.0, 0.0, 0.7),
            },
            # Obstacle 3: 中等，綠色
            {
                "radius": 0.35,
                "height": 1.1,
                "color": (0.2, 0.8, 0.3),
                "pos": (0.0, 0.0, 0.55),
            },
            # Obstacle 4: 小型，橙色
            {
                "radius": 0.25,
                "height": 0.9,
                "color": (0.9, 0.6, 0.2),
                "pos": (0.0, 0.0, 0.45),
            },
        ]

        obstacle_sizes = []

        for i, cfg in enumerate(dynamic_obstacle_configs):
            # Phase 3 關鍵：動態障礙物是 dynamic（非 kinematic）
            # 它們會在 reset_obstacles 事件中移動
            obstacle_rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 動態物體
                disable_gravity=False,
            )

            obstacle_visual = sim_utils.PreviewSurfaceCfg(
                diffuse_color=cfg["color"],
                metallic=0.2,
                roughness=0.7,
            )

            setattr(
                self,
                f"obstacle_{i}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/obstacle_{i}",
                    spawn=sim_utils.CylinderCfg(
                        radius=cfg["radius"],
                        height=cfg["height"],
                        rigid_props=obstacle_rigid_props,
                        visual_material=obstacle_visual,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=cfg["pos"],
                    ),
                ),
            )

            obstacle_sizes.append(cfg["radius"] * 2.0)  # 直徑

        # 設置障礙物元數據（用於獎勵和終止條件）
        set_obstacle_metadata(num_dynamic_obstacles, obstacle_sizes)

        # ============================================================================
        # LiDAR 配置（檢測所有牆壁 + 動態障礙物）
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
                # 外牆
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,
                ),
                # 內牆
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/U_Wall_.*",
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Corridor_.*",
                    track_mesh_transforms=False,
                ),
                # 🔥 Phase 3 關鍵：檢測動態障礙物
                # obstacle_.* 會匹配所有動態障礙物
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/obstacle_.*",
                    track_mesh_transforms=True,  # 動態物體，追蹤變換
                ),
            ],
            update_period=0.04,
            debug_vis=False,  # 关闭 LiDAR 射线可视化以提升性能（红色长线）
        )


# ============================================================================
# Phase 3 命令配置
# ============================================================================

@configclass
class CommandsCfgPhase3(CommandsCfg):
    """Phase 3 命令配置 - 複雜地形與動態障礙物"""

    goal_command = GoalCommandCfg(
        asset_name="robot",
        debug_vis=True,  # 顯示紅色箭頭（GUI 模式，headless 自動禁用）
        resampling_time_range=(5.0, 10.0),
        # Phase 3 規格：目標距離考慮複雜繞路
        ranges=GoalCommandCfg.Ranges(
            distance=(5.0, 12.0),
            angle=(-math.pi, math.pi),
        ),
    )


# ============================================================================
# Phase 3 事件配置
# ============================================================================

@configclass
class EventCfgPhase3(EventCfg):
    """Phase 3 事件配置 - 複雜地形與動態障礙物"""

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
    #         "phase": "phase3",
    #     },
    # )

    # 🔥 Phase 3 關鍵：動態障礙物重置
    # 障礙物會隨機移動到新位置，但每個 episode 內保持相對靜態
    # 這測試 policy 在不同障礙物配置下的適應能力
    reset_dynamic_obstacles = EventTerm(
        func=reset_obstacles,
        mode="reset",
        params={
            "num_obstacles": 5,
            "velocity_range": (0.0, 0.0),  # Phase 3A: 靜態位置
            # 未來 Phase 3B 可以啟用速度：(0.1, 0.3) 讓障礙物緩慢移動
        },
    )


# ============================================================================
# Phase 3 環境配置主類
# ============================================================================

@configclass
class ChargeNavigationEnvCfgPhase3(ChargeNavigationEnvCfg):
    """Phase 3 導航環境配置 - 複雜地形與動態障礙物

    環境規格：
    - 地圖：16x16m 房間
    - 地形：複雜內牆結構
    - 障礙物：5 個動態圓柱體

    架構說明：
    - Global Planner (AIT*): 只規劃靜態牆壁的繞路
    - Local Policy (RL): 使用 LiDAR 檢測並閃避動態障礙

    畢業標準：
    - 成功率 > 75%
    - 能夠同時處理靜態牆壁和動態障礙
    - 能夠在障礙物擋路時等待或繞行
    """

    scene: MySceneCfgPhase3 = MySceneCfgPhase3(num_envs=256, env_spacing=18.0)

    observations: ObservationsCfgPhase3 = ObservationsCfgPhase3()

    rewards: RewardsCfgPhase3 = RewardsCfgPhase3()

    terminations: TerminationsCfgPhase3 = TerminationsCfgPhase3()

    commands: CommandsCfgPhase3 = CommandsCfgPhase3()

    events: EventCfgPhase3 = EventCfgPhase3()

    episode_length_s = 30.0


@configclass
class EventCfgPhase3_PLAY(EventCfg):
    """Phase 3 Play 模式事件配置 - 啟用 AIT* 路徑規劃"""

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
            "phase": "phase3",
            "use_astar": False,
            "use_virtual_planner": True,     # 🔥 使用虛擬規劃器（不執行 AIT*）
        },
    )

    # Play 模式也需要動態障礙物
    reset_dynamic_obstacles = EventTerm(
        func=reset_obstacles,
        mode="reset",
        params={
            "num_obstacles": 5,
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ChargeNavigationEnvCfgPhase3_PLAY(ChargeNavigationEnvCfgPhase3):
    """Phase 3 演示環境配置（使用虛擬規劃器）"""

    scene: MySceneCfgPhase3 = MySceneCfgPhase3(num_envs=1, env_spacing=18.0)
    observations: ObservationsCfgPhase3 = ObservationsCfgPhase3()
    events: EventCfgPhase3_PLAY = EventCfgPhase3_PLAY()  # 覆蓋事件配置


__all__ = [
    "ChargeNavigationEnvCfgPhase3",
    "ChargeNavigationEnvCfgPhase3_PLAY",
]
