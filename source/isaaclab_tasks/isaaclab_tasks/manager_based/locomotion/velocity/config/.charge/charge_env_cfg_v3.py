"""
Charge 導航環境配置（Phase 3）

此版本專注於「靜態障礙物迴避」訓練，強調平衡安全性和效率。
主要特點：
1. 障礙物數量：10 個靜態障礙物
2. 障礙物之間最小間距：2.0 米
3. 障礙物設為靜態（先不使用動態障礙物）
4. 目標距離範圍：4.0-8.0 米
5. 場景邊界：8.0 米
6. 強化迴避障礙物獎勵機制：
   - 平衡安全性和效率（既要安全，也要能到達目標）
   - 學習提前減速和迴避策略
   - 增加碰撞懲罰權重
   - 強化安全導航獎勵
   - 增加速度控制重要性
"""

import math  # 用於角度計算
import random  # 用於隨機生成障礙物

import isaaclab.sim as sim_utils  # 模擬工具
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # 資產基礎配置
from isaaclab.managers import EventTermCfg as EventTerm  # 事件配置類
from isaaclab.sensors import ContactSensorCfg  # 接觸感測器配置
from isaaclab.utils import configclass  # 配置類裝飾器

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 噪音配置

from .charge_env_cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    CommandsCfg,
    EventCfg,
    GoalCommandCfg,
    GOAL_REACH_THRESHOLD,  # 目標到達閾值
    MAX_OBSTACLES,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    ROBOT_BODY_RADIUS,  # 機器人身體半徑
    TerminationsCfg,
    charge_mdp,
)

# 從 Phase 2 導入配置類
from .charge_env_cfg_v2 import (
    MySceneCfgV2,
    ObservationsCfgV2,
    CommandsCfgV2,
    TerminationsCfgV2,
    EventCfgV2,
)


@configclass
class MySceneCfgV3(MySceneCfgV2):
    """場景配置（Phase 3）
    
    Phase 3：10 個靜態障礙物（先不使用動態障礙物）
    1. 障礙物數量：10 個（從 Phase 2 的 5 個增加到 10 個）
    2. 障礙物之間最小間距：2.0 米
    3. 場景邊界：8.0 米
    4. 障礙物設為靜態（kinematic_enabled=True）
    """
    
    def __post_init__(self):
        """在配置初始化後自動執行（Phase 3：10 個靜態障礙物）
        
        注意：不調用父類的 __post_init__，因為父類會創建 5 個靜態障礙物。
        我們直接創建 10 個靜態障礙物來覆蓋。
        """
        # 不調用 super().__post_init__()，避免創建重複的障礙物
        # 父類的 __post_init__ 只創建障礙物，其他配置（地形、機器人等）已經在類定義中完成
        
        # Phase 3：創建 10 個靜態障礙物
        num_obstacles = 10  # Phase 3：10 個障礙物
        obstacle_sizes: list[float] = []
        
        for i in range(num_obstacles):
            # 隨機選擇障礙物類型（立方體 or 圓柱體）
            is_cube = random.random() > 0.5
            
            if is_cube:
                # 立方體障礙物
                size = (
                    random.uniform(0.3, 0.8),  # X 軸長度：0.3-0.8 米
                    random.uniform(0.3, 0.8),  # Y 軸寬度：0.3-0.8 米
                    random.uniform(0.8, 1.5),  # Z 軸高度：0.8-1.5 米
                )
                
                spawn_cfg = sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # Phase 3：靜態障礙物（不會移動）
                        disable_gravity=True,  # 靜態物體不需要重力
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            random.random(),
                            random.random(),
                            random.random(),
                        ),
                        metallic=0.2,
                    ),
                )
                height_i = size[2] / 2
                size_scalar = max(size[0], size[1])
                
            else:
                # 圓柱體障礙物
                radius = random.uniform(0.2, 0.5)
                height_val = random.uniform(0.8, 1.5)
                
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,
                    height=height_val,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # Phase 3：靜態障礙物（不會移動）
                        disable_gravity=True,  # 靜態物體不需要重力
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            random.random(),
                            random.random(),
                            random.random(),
                        ),
                        metallic=0.2,
                    ),
                )
                height_i = height_val / 2
                size_scalar = radius * 2.0
            
            # 創建障礙物資產並添加到場景
            # 注意：初始位置範圍改為 ±8 米，與移動邊界一致
            setattr(
                self,
                f"obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(
                            random.uniform(-8, 8),  # ±8 米（與移動邊界一致）
                            random.uniform(-8, 8),  # ±8 米（與移動邊界一致）
                            height_i,
                        )
                    ),
                ),
            )
            
            # 記錄障礙物尺寸
            obstacle_sizes.append(size_scalar)
        
        # 記錄障礙物設定（供觀測和事件使用）
        charge_mdp.set_obstacle_metadata(num_obstacles, obstacle_sizes)
        
        # ====================================================================
        # 創建邊界牆壁（Phase 3：8.0 米邊界）
        # ====================================================================
        # 牆壁參數
        boundary = 8.0  # 邊界距離（米）
        wall_thickness = 0.1  # 牆壁厚度（米）
        wall_height = 1.5  # 牆壁高度（米）
        wall_length = boundary * 2 + wall_thickness  # 牆壁長度（覆蓋整個邊界）
        wall_color = (0.5, 0.5, 0.5)  # 灰色牆壁
        
        # 牆壁通用配置（靜態、不可移動）
        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # 靜態牆壁（不會被推動）
            disable_gravity=True,  # 不需要重力
        )
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=wall_color,
            metallic=0.1,
        )
        
        # 北牆（Y = +boundary）
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
        
        # 南牆（Y = -boundary）
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
        
        # 東牆（X = +boundary）
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
        
        # 西牆（X = -boundary）
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
        
        # ====================================================================
        # 覆蓋接觸感測器：加入牆壁碰撞檢測（Phase 3）
        # ====================================================================
        # 需要同時監測障礙物和牆壁的碰撞
        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",  # 只監測機器人底盤
            update_period=0.0,  # 更新週期：0.0 = 每步都更新（最快）
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Obstacle_.*",  # 監測與障礙物的碰撞
                "{ENV_REGEX_NS}/Wall_.*",       # 監測與牆壁的碰撞
            ],
            debug_vis=False,  # 調試可視化：False = 不顯示接觸點（提升性能）
        )


@configclass
class ObservationsCfgV3(ObservationsCfgV2):
    """觀測配置（Phase 3）
    
    Phase 3：10 個靜態障礙物
    需要更新障礙物數量從 5 個到 10 個。
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組（Phase 3）"""
        
        # 繼承 Phase 2 的所有觀測項，但更新障礙物數量
        # 觀測項 1：2D 平面掃描（72 個角度）
        lidar_scan = ObsTerm(
            func=charge_mdp.lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # 觀測項 2：速度資訊（機器人座標系）
        speed = ObsTerm(
            func=charge_mdp.base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # 觀測項 3：目標相對位置
        goal_position = ObsTerm(
            func=charge_mdp.goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # 觀測項 4：目標距離
        goal_distance = ObsTerm(
            func=charge_mdp.goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # 觀測項 5：時間剩餘比例
        time_remaining = ObsTerm(
            func=charge_mdp.time_remaining_ratio,
        )
        
        # 觀測項 6：存活/死亡狀態
        alive = ObsTerm(
            func=charge_mdp.alive_flag,
        )
        
        # 觀測項 7：靜態障礙物資訊（Phase 3：10 個障礙物）
        obstacles = ObsTerm(
            func=charge_mdp.dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 10,  # Phase 3：10 個障礙物
                "max_obstacles": MAX_OBSTACLES,  # 固定最大數量（用於 padding）
                "max_distance": 15.0,  # 最大觀測距離
            },
        )
        
        # 觀測項 8：上一步動作
        actions = ObsTerm(func=charge_mdp.safe_last_action)
        
        def __post_init__(self):
            """觀測組初始化配置"""
            self.enable_corruption = True  # 啟用數據損壞（加噪音）
            self.concatenate_terms = True  # 串接所有觀測項
    
    policy: PolicyCfg = PolicyCfg()  # 實例化策略觀測組


@configclass
class CommandsCfgV3(CommandsCfgV2):
    """命令配置（Phase 3）
    
    Phase 3：10 個靜態障礙物
    需要更新障礙物數量從 5 個到 10 個。
    """
    
    goal_command = GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),  # 只在環境重置時生成新目標
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(4.0, 8.0),  # Phase 3：目標距離範圍 4.0-8.0 米
            angle=(-math.pi, math.pi),  # 全方位（360度）
        ),
        # 碰撞檢查配置（Phase 3，更新障礙物數量）
        wall_boundary=8.0,        # 牆壁邊界距離（與場景配置一致）
        wall_safe_margin=1.0,     # 牆壁安全邊距（目標離牆至少 1.0 米）
        obstacle_safe_distance=1.0,  # 障礙物安全距離（目標離障礙物至少 1.0 米）
        max_resample_attempts=50,    # 最大重試次數
        num_obstacles=10,         # 障礙物數量（Phase 3：10 個）
    )


@configclass
class RewardsCfgV3(RewardsCfg):
    """Phase 3 獎勵配置：10 個靜態障礙物（平衡安全性和效率）
    
    專注於平衡安全性和效率的訓練策略：
    1. 平衡安全性和效率：既要安全，也要能到達目標
    2. 學習提前減速和迴避策略：鼓勵在接近障礙物時提前減速
    3. 強化碰撞懲罰：提高權重，讓機器人更重視安全
    4. 強化安全導航：增加獎勵權重，鼓勵保持安全距離
    5. 強化速度控制：提高權重，鼓勵在接近障礙物時減速
    6. 保持目標導向：維持足夠的目標獎勵，確保機器人仍能到達目標
    
    設計理念：
    - 10 個靜態障礙物提供足夠的挑戰性
    - 需要機器人具備規劃和預測能力
    - 平衡安全性和效率（既要安全，也要能到達目標）
    - 學習提前減速和迴避策略，避免最後一刻才減速
    """
    
    # ------------------------------------------------------------------------
    # 核心調整 1：碰撞懲罰（強化，提高安全性，但不過度）
    # ------------------------------------------------------------------------
    # Phase 3：10 個靜態障礙物，需要適度的碰撞懲罰來確保安全
    # 平衡安全性和效率：既要安全，也要能到達目標
    progressive_collision = RewTerm(
        func=charge_mdp.progressive_collision_penalty,
        weight=-2.0,  # 適度提高（從 -1.0 增加到 -2.0），強化安全意識但不過度
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": 2.0,      # 2.0米開始懲罰（提前警告，學習提前減速）
            "danger_distance": 1.2,    # 1.2米中等懲罰
            "collision_distance": 0.6,  # 0.6米最大懲罰
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 2：進度獎勵（適度降低，平衡安全與效率）
    # ------------------------------------------------------------------------
    # 降低權重，避免機器人為了快速到達目標而忽略安全
    # 但保持足夠的權重，確保機器人仍能積極向目標前進
    distance_to_goal = RewTerm(
        func=charge_mdp.progress_to_goal,
        weight=4.0,  # 適度降低（從 5.0 降到 4.0），平衡安全與效率
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 3：到達目標獎勵（保持高權重）
    # ------------------------------------------------------------------------
    # 保持高權重，確保機器人仍能完成任務
    reaching_goal = RewTerm(
        func=charge_mdp.reaching_goal,
        weight=500,  # 保持高權重（500），強烈鼓勵到達目標
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 4：安全導航獎勵（強化，鼓勵提前減速和迴避）
    # ------------------------------------------------------------------------
    # Phase 3：10 個靜態障礙物需要更強的安全導航獎勵
    # 鼓勵保持安全距離，學習提前減速和迴避策略
    safe_navigation = RewTerm(
        func=charge_mdp.safe_navigation_bonus,
        weight=0.8,  # 大幅提高（從 0.1 增加到 0.8），強化安全行為和提前迴避
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "comfort_distance": 2.5,  # 增加舒適距離（從 2.0 米增加到 2.5 米），鼓勵提前減速
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 5：速度控制（強化，學習提前減速）
    # ------------------------------------------------------------------------
    # Phase 3：10 個靜態障礙物需要積極的速度控制
    # 增加警告距離，鼓勵在接近障礙物時提前減速，而不是最後一刻才減速
    speed_control_near_obstacles = RewTerm(
        func=charge_mdp.speed_control_near_obstacles,
        weight=0.4,  # 大幅提高（從 0.1 增加到 0.4），強化速度控制和提前減速
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "warning_distance": 2.5,  # 增加警告距離（從 1.5 米增加到 2.5 米），鼓勵提前減速
            "max_safe_speed": 0.9,    # 適度降低安全速度（從 1.0 降到 0.9 m/s），更保守但不過度
        },
    )
    
    # 注意：牆壁碰撞已統一通過 Lidar 的 progressive_collision 處理
    # 不再需要獨立的 wall_collision 懲罰


@configclass
class TerminationsCfgV3(TerminationsCfgV2):
    """Phase 3 終止條件配置
    
    繼承 Phase 2 的終止條件，動態障礙物需要相同的碰撞檢測機制。
    """
    pass  # 使用 Phase 2 的終止條件


@configclass
class EventCfgV3(EventCfgV2):
    """事件配置（Phase 3）

    Phase 3：10 個靜態障礙物（先不使用動態障礙物）
    - 障礙物數量：10 個
    - 障礙物是靜態的（不會移動）
    - 不添加 move_obstacles 事件，保持靜態
    """
    
    # Phase 3 重置事件：10 個靜態障礙物，使用 8.0 米邊界
    reset_obstacles = EventTerm(
        func=charge_mdp.reset_obstacles,  # 使用靜態障礙物重置函數
        mode="reset",
        params={
            "speed_range": 0.0,  # 靜態障礙物：不需要速度
            "min_speed": 0.0,
            "min_robot_distance": 3.5,  # 機器人與障礙物最小距離：3.5 米
            "min_goal_distance": 2.0,  # 目標與障礙物最小距離：2.0 米
            "min_obstacle_spacing": 2.0,  # 障礙物之間最小間距：2.0 米
            "max_spawn_attempts": 50,  # 最大生成嘗試次數
            "boundary": 8.0,  # 場景邊界：8.0 米（Phase 3 使用更大的場景）
        },
    )
    
    # 注意：Phase 3 不使用動態障礙物移動事件
    # 先專注於靜態障礙物的迴避訓練


@configclass
class ChargeNavigationEnvCfgV3(ChargeNavigationEnvCfg):
    """Phase 3 訓練配置：10 個靜態障礙物（平衡安全性和效率）
    
    Curriculum Learning 階段：
    - Phase 1: 3 個靜態障礙物 ✅（已完成）
    - Phase 2: 5 個靜態障礙物 ✅（已完成）
    - Phase 3: 10 個靜態障礙物（當前，專注於平衡安全性和效率）
    
    主要特點：
    1. 障礙物數量：10 個靜態障礙物（先不使用動態障礙物）
    2. 障礙物之間最小間距：2.0 米
    3. 目標距離範圍：4.0-8.0 米
    4. 場景邊界：8.0 米
    5. 強化迴避獎勵機制（平衡安全性和效率）：
       - 碰撞懲罰權重：-2.0（適度強化）
       - 安全導航獎勵權重：0.8（強化，鼓勵提前減速和迴避）
       - 速度控制權重：0.4（強化，學習提前減速）
       - 進度獎勵權重：4.0（適度降低，平衡安全與效率）
       - 到達目標獎勵權重：500（保持高權重）
    
    訓練目標：
    - 平衡安全性和效率（既要安全，也要能到達目標）
    - 學習提前減速和迴避策略（避免最後一刻才減速）
    - 在複雜的靜態障礙物環境中安全導航
    """

    scene: MySceneCfgV3 = MySceneCfgV3(num_envs=128, env_spacing=25.0)  # 使用 Phase 3 場景配置
    observations: ObservationsCfgV3 = ObservationsCfgV3()  # 使用 Phase 3 觀測配置
    commands: CommandsCfgV3 = CommandsCfgV3()  # 使用 Phase 3 命令配置
    rewards: RewardsCfgV3 = RewardsCfgV3()  # ⭐ 使用 Phase 3 獎勵配置（強化迴避）
    terminations: TerminationsCfgV3 = TerminationsCfgV3()  # 使用 Phase 3 終止條件
    events: EventCfgV3 = EventCfgV3()  # ⭐ 使用 Phase 3 事件配置（動態移動）


@configclass
class ChargeNavigationEnvCfgV3_PLAY(ChargeNavigationEnvCfg_PLAY):
    """Phase 3 測試/展示配置"""

    scene: MySceneCfgV3 = MySceneCfgV3(num_envs=16, env_spacing=25.0)  # 測試時使用較少環境
    observations: ObservationsCfgV3 = ObservationsCfgV3()  # 使用 Phase 3 的觀測配置
    commands: CommandsCfgV3 = CommandsCfgV3()  # 使用 Phase 3 的目標距離配置
    rewards: RewardsCfgV3 = RewardsCfgV3()  # 使用 Phase 3 的獎勵配置
    terminations: TerminationsCfgV3 = TerminationsCfgV3()  # 使用 Phase 3 的終止條件
    events: EventCfgV3 = EventCfgV3()  # 使用 Phase 3 的事件配置（動態移動）
