"""
Charge 導航環境配置（Phase 3）

此版本專注於「目標距離課程學習」，讓 Agent 學會直直朝向目標前進。

設計理念：
- 根據成功率動態調整目標距離（而非障礙物數量）
- 鼓勵 Agent 直直朝 goal 前進（velocity_to_goal 獎勵）
- 暫時移除障礙物相關配置（未來會使用）

課程學習邏輯：
- 難度等級：1-10
- 目標距離範圍：
  - 等級 1: 2.0-4.0 米（簡單）
  - 等級 5: 4.0-8.0 米（中等）
  - 等級 10: 9.0-15.0 米（困難）
- 調整規則：
  - 成功率 > 80%: 增加距離（升級）
  - 成功率 < 50%: 減少距離（降級）

預期效果：
- Agent 深刻「記住」抵達目的地能帶來巨大正向回報
- 學會直直朝向 goal 前進以獲得最大獎勵
"""

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg  # 導入基類，用於顯式調用初始化
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
    MAX_OBSTACLES,  # 導入 MAX_OBSTACLES (10)
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    ROBOT_BODY_RADIUS,
    TerminationsCfg,
)

# 觀測函數
from ..mdp.observations import (
    lidar_scan_2d_sweep,
    base_velocity_xy,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    safe_last_action,
    dynamic_obstacles_state,
)

# 獎勵函數（只導入目標導向獎勵）
from ..mdp.rewards import (
    progress_to_goal,
    reaching_goal,
    velocity_toward_goal,
    collision_occurred,
    collision_contact_occurred,
)

# 終止條件函數
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
)

# 目標距離課程學習
from ..mdp.events import (
    adaptive_goal_distance_update,
    DIFFICULTY_DISTANCE_MAP,
)

# 多目標命令
from ..multi_goal_command import MultiGoalCommand, MultiGoalCommandCfg

# 從 Phase 2 導入配置類
from .charge_env_cfg_v2 import (
    MySceneCfgV2,
    ObservationsCfgV2,
    CommandsCfgV2,
    TerminationsCfgV2,
    EventCfgV2,
)


@configclass
class MySceneCfgV3(MySceneCfg):
    """場景配置（Phase 3）
    
    簡化場景：只有地面、機器人、牆壁，無障礙物。
    專注於讓 Agent 學會直直朝目標前進。
    """
    
    def __post_init__(self):
        """初始化場景（無障礙物版本）
        
        注意：直接調用 InteractiveSceneCfg.__post_init__，而不是 super().__post_init__。
        原因：MySceneCfg (父類) 的 __post_init__ 會創建 3 個障礙物。
        我們Phase 3 不需要這些障礙物，所以跳過父類的初始化邏輯。
        """
        InteractiveSceneCfg.__post_init__(self)
        
        # ====================================================================
        # 創建邊界牆壁（8.0 米邊界）
        # ====================================================================
        boundary = 8.0
        wall_thickness = 0.1
        wall_height = 1.5
        wall_length = boundary * 2 + wall_thickness
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
        
        # 北牆
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, boundary + wall_thickness / 2, wall_height / 2)
            ),
        )
        
        # 南牆
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -boundary - wall_thickness / 2, wall_height / 2)
            ),
        )
        
        # 東牆
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(boundary + wall_thickness / 2, 0.0, wall_height / 2)
            ),
        )
        
        # 西牆
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-boundary - wall_thickness / 2, 0.0, wall_height / 2)
            ),
        )
        
        # 覆蓋 lidar 配置（只檢測牆壁，不檢測障礙物）
        self.lidar = MultiMeshRayCasterCfg(
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
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,
                ),
            ],
            update_period=0.04,
        )


@configclass
class ObservationsCfgV3(ObservationsCfg):
    """觀測配置（Phase 3）
    
    簡化觀測：移除障礙物資訊，專注於目標導航。
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組（Phase 3：無障礙物版本）"""
        
        # 2D 平面掃描（72 個角度）
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # 速度資訊
        speed = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # 目標相對位置
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # 目標距離
        goal_distance_obs = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # 時間剩餘比例
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )
        
        # 存活狀態
        alive = ObsTerm(
            func=alive_flag,
        )
        
        # 上一步動作
        actions = ObsTerm(func=safe_last_action)
        
        # 觀測項 7：障礙物資訊（Phase 3 雖無障礙物，但為了匹配 Checkpoint 的 131 維，需保留 50 維 Padding）
        # num_obstacles=0 表示無真實障礙物，全為 Padding
        obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 0,  # 無障礙物
                "max_obstacles": MAX_OBSTACLES,  # 固定 10 個（產生 50 維 Padding）
                "max_distance": 15.0,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfgV3(CommandsCfg):
    """命令配置（Phase 3）

    目標距離會根據課程學習動態調整。
    初始設置為簡單難度（2.0-4.0 米）。
    """

    goal_command = MultiGoalCommandCfg(
        class_type=MultiGoalCommand,  # 必須顯式設置 class_type
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),
        debug_vis=True,
        ranges=MultiGoalCommandCfg.Ranges(
            distance=(2.0, 4.0),  # 等級 1：2-4 米（廣撒網）
            angle=(-math.pi, math.pi),
        ),
        num_goals=5,  # 初始多目標數量
        # 無障礙物相關配置
        wall_boundary=8.0,
        wall_safe_margin=1.0,
        obstacle_safe_distance=1.0,  # 保留兼容性
        max_resample_attempts=50,
        num_obstacles=0,  # 無障礙物
    )


@configclass
class RewardsCfgV3(RewardsCfg):
    """獎勵配置（Phase 3）
    
    專注於目標導向獎勵，移除障礙物相關獎勵。
    
    當前獎勵結構：
    ┌─────────────────────────────────────────────────────────────┐
    │ 目標導向獎勵（核心）                                          │
    │ - reaching_goal: +500（到達目標，一次性大獎勵）               │
    │ - distance_to_goal: +5.0 × 距離變化（每步）                  │
    │ - velocity_to_goal: +3.0 × 速度投影（⭐核心：朝目標前進）     │
    └─────────────────────────────────────────────────────────────┘
    
    設計理念：
    - 讓 Agent 深刻「記住」抵達目的地能帶來巨大正向回報
    - 鼓勵直直朝向 goal 前進以獲得最大獎勵
    """
    
    # ------------------------------------------------------------------------
    # 到達目標獎勵（最重要！大獎勵讓 Agent 記住目標的價值）
    # ------------------------------------------------------------------------
    reaching_goal = RewTerm(
        func=reaching_goal,
        weight=500,  # 大獎勵：讓 Agent 深刻記住抵達目標的價值
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,
        },
    )
    
    # ------------------------------------------------------------------------
    # 進度獎勵（距離變化）
    # ------------------------------------------------------------------------
    distance_to_goal = RewTerm(
        func=progress_to_goal,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ------------------------------------------------------------------------
    # 直線朝目標前進獎勵（⭐核心：鼓勵直直往 goal 前進）
    # ------------------------------------------------------------------------
    velocity_to_goal = RewTerm(
        func=velocity_toward_goal,
        weight=3.0,  # 較高權重，強化直線前進的重要性
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_dist": 0.5,  # 距離 < 0.5m 時不給獎勵（避免衝撞）
        },
    )
    
    # ========================================================================
    # 以下獎勵已暫時移除（障礙物相關，未來會使用）
    # ========================================================================
    # progressive_collision = RewTerm(...)  # 漸進式碰撞懲罰
    # collision_terminal_penalty = RewTerm(...)  # 碰撞終止懲罰
    # safe_navigation = RewTerm(...)  # 安全導航獎勵
    # speed_control_near_obstacles = RewTerm(...)  # 速度控制


@configclass
class TerminationsCfgV3(TerminationsCfg):
    """終止條件配置（Phase 3）
    
    簡化終止條件：只保留目標到達和翻倒判定。
    移除碰撞終止（無障礙物）。
    """
    
    # 到達目標（成功）
    goal_reached = DoneTerm(
        func=goal_reached,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
        },
    )
    
    # 機器人翻倒（失敗）
    tipped_over = DoneTerm(
        func=robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ------------------------------------------------------------------------
    # 碰撞終止（牆壁）
    # ------------------------------------------------------------------------
    # 雖然沒有障礙物，但撞到牆壁仍需終止
    
    # 1. 雷達碰撞判定（檢測牆壁）
    collision = DoneTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "threshold": 0.5,  # 距離牆壁 < 0.5m 終止
        },
    )
    
    # 2. 接觸傳感器判定（真實碰撞）
    collision_contact = DoneTerm(
        func=collision_contact_occurred,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )


@configclass
class CurriculumsCfgV3:
    """目標距離課程學習配置（Phase 3）
    
    根據成功率動態調整目標距離範圍：
    - 難度等級：1-10
    - 目標距離：2-15 米
    - 成功率 > 80%：增加距離（升級）
    - 成功率 < 50%：減少距離（降級）
    """
    
    adaptive_distance = CurrTerm(
        func=adaptive_goal_distance_update,
        params={
            "difficulty_increase_threshold": 0.8,  # 成功率 > 80% 時升級
            "difficulty_decrease_threshold": 0.5,  # 成功率 < 50% 時降級
        },
    )


@configclass
class EventCfgV3(EventCfg):
    """事件配置（Phase 3）
    
    簡化事件：移除障礙物重置（無障礙物）。
    """
    pass  # 使用基礎配置的事件（無障礙物相關事件）


@configclass
class ChargeNavigationEnvCfgV3(ChargeNavigationEnvCfg):
    """Phase 3 訓練配置：目標距離課程學習
    
    Curriculum Learning 階段：
    - Phase 1: 3 個靜態障礙物 ✅
    - Phase 2: 5 個靜態障礙物 ✅
    - Phase 3: 目標距離課程學習（當前）
      - 無障礙物，專注於目標導航
      - 根據成功率動態調整目標距離
      - 鼓勵 Agent 直直朝 goal 前進
    
    訓練目標：
    - 讓 Agent 深刻「記住」抵達目的地能帶來巨大正向回報
    - 學會直直朝向 goal 前進以獲得最大獎勵
    """

    scene: MySceneCfgV3 = MySceneCfgV3(num_envs=128, env_spacing=25.0)
    observations: ObservationsCfgV3 = ObservationsCfgV3()
    commands: CommandsCfgV3 = CommandsCfgV3()
    rewards: RewardsCfgV3 = RewardsCfgV3()
    terminations: TerminationsCfgV3 = TerminationsCfgV3()
    events: EventCfgV3 = EventCfgV3()
    curriculum: CurriculumsCfgV3 = CurriculumsCfgV3()


@configclass
class ChargeNavigationEnvCfgV3_PLAY(ChargeNavigationEnvCfg_PLAY):
    """Phase 3 測試/展示配置"""

    scene: MySceneCfgV3 = MySceneCfgV3(num_envs=16, env_spacing=25.0)
    observations: ObservationsCfgV3 = ObservationsCfgV3()
    commands: CommandsCfgV3 = CommandsCfgV3()
    rewards: RewardsCfgV3 = RewardsCfgV3()
    terminations: TerminationsCfgV3 = TerminationsCfgV3()
    events: EventCfgV3 = EventCfgV3()
    curriculum: CurriculumsCfgV3 = CurriculumsCfgV3()


# ============================================================================
# 備忘：障礙物相關配置（未來會使用）
# ============================================================================
"""
以下是 Phase 3 原本使用的障礙物相關配置，已暫時移除但保留備用：

1. 獎勵函數：
   - progressive_collision_penalty: 漸進式碰撞懲罰
   - collision_occurred: 碰撞終止懲罰
   - safe_navigation_bonus: 安全導航獎勵
   - speed_control_near_obstacles: 速度控制

2. 終止條件：
   - collision: 雷達距離碰撞判定
   - collision_contact: 接觸感測器碰撞判定

3. 事件函數：
   - reset_obstacles_with_adaptive_params: 自適應障礙物重置
   - hide_unused_obstacles: 隱藏未使用的障礙物

4. 課程學習：
   - adaptive_curriculum_update: 根據成功率和碰撞率調整障礙物數量

這些配置可以在未來的 Phase 4 或其他訓練階段重新啟用。
"""
