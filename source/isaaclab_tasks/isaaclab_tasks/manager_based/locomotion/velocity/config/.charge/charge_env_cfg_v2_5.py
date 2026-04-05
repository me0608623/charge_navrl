"""
Charge 導航環境配置（Phase 2.5）

此版本實現「自適應課程學習框架」，根據導航成功率與碰撞率動態調整環境難度。

主要特點：
1. 自適應障礙物數量：根據成功率和碰撞率動態調整（3-10 個）
2. 自適應距離參數：根據難度等級動態調整距離參數
3. 基於統計的難度調整：
   - 成功率 > 80% 且碰撞率 < 20%：增加難度
   - 成功率 < 50% 或碰撞率 > 40%：降低難度
   - 否則：保持當前難度
4. 場景邊界：8.0 米（繼承 Phase 2）
5. 目標距離範圍：4.0-8.0 米（繼承 Phase 2）
"""

import math  # 用於角度計算
import random  # 用於隨機生成障礙物

import isaaclab.sim as sim_utils  # 模擬工具
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # 資產基礎配置
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,  # 課程學習項配置
    EventTermCfg as EventTerm,  # 事件配置類
)
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
    RewardsCfgV2,
    EventCfgV2,  # 事件配置（Phase 2）
)


@configclass
class MySceneCfgV2_5(MySceneCfgV2):
    """場景配置（Phase 2.5）
    
    繼承 Phase 2 的場景配置，但障礙物數量會根據自適應課程學習動態調整。
    初始設置為 3 個障礙物（最小難度）。
    """
    
    def __post_init__(self):
        """在配置初始化後自動執行（Phase 2.5：自適應障礙物數量）
        
        創建 MAX_OBSTACLES 個障礙物，但初始只啟用 3 個（最小難度）。
        後續會根據課程學習動態調整啟用的障礙物數量。
        """
        # 不調用 super().__post_init__()，避免創建固定數量的障礙物
        # 創建 MAX_OBSTACLES 個障礙物，但根據難度等級只啟用前 N 個
        
        # 創建最大數量的障礙物（以支持動態調整）
        max_obstacles = MAX_OBSTACLES  # 10 個
        obstacle_sizes: list[float] = []
        
        # 創建所有障礙物（最多 10 個），但初始只啟用 3 個
        for i in range(max_obstacles):
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
                        kinematic_enabled=True,  # Phase 2.5：靜態障礙物
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
                        kinematic_enabled=True,  # Phase 2.5：靜態障礙物
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
        # 初始設置為 3 個障礙物（最小難度），但實際創建了 max_obstacles 個
        initial_num_obstacles = 3  # 初始難度：3 個障礙物
        charge_mdp.set_obstacle_metadata(initial_num_obstacles, obstacle_sizes[:initial_num_obstacles])
        
        # 創建邊界牆壁（繼承 Phase 2 的配置）
        boundary = 8.0  # 邊界距離（米）
        wall_thickness = 0.1  # 牆壁厚度（米）
        wall_height = 1.5  # 牆壁高度（米）
        wall_length = boundary * 2 + wall_thickness  # 牆壁長度
        wall_color = (0.5, 0.5, 0.5)  # 灰色牆壁
        
        # 牆壁通用配置
        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # 靜態牆壁
            disable_gravity=True,
        )
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=wall_color,
            metallic=0.1,
        )
        
        # 創建 4 面牆壁
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
        
        # 接觸感測器配置
        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
            update_period=0.0,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Obstacle_.*",
                "{ENV_REGEX_NS}/Wall_.*",
            ],
            debug_vis=False,
        )


@configclass
class ObservationsCfgV2_5(ObservationsCfgV2):
    """觀測配置（Phase 2.5）
    
    繼承 Phase 2 的觀測配置，但障礙物數量會動態調整。
    觀測維度會根據當前障礙物數量自動適應。
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組（Phase 2.5）"""
        
        # 繼承 Phase 2 的所有觀測項
        lidar_scan = ObsTerm(
            func=charge_mdp.lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        speed = ObsTerm(
            func=charge_mdp.base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        goal_position = ObsTerm(
            func=charge_mdp.goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        goal_distance = ObsTerm(
            func=charge_mdp.goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        time_remaining = ObsTerm(
            func=charge_mdp.time_remaining_ratio,
        )
        
        alive = ObsTerm(
            func=charge_mdp.alive_flag,
        )
        
        # 障礙物觀測（數量會動態調整，但使用 MAX_OBSTACLES 作為上限）
        obstacles = ObsTerm(
            func=charge_mdp.dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": MAX_OBSTACLES,  # 使用最大值，實際數量會動態調整
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 15.0,
            },
        )
        
        actions = ObsTerm(func=charge_mdp.safe_last_action)
        
        def __post_init__(self):
            """觀測組初始化配置"""
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfgV2_5(CommandsCfgV2):
    """命令配置（Phase 2.5）
    
    繼承 Phase 2 的命令配置，保持相同的目標距離範圍。
    """
    pass


@configclass
class RewardsCfgV2_5(RewardsCfgV2):
    """獎勵配置（Phase 2.5）
    
    根據 Phase 2 訓練問題重新平衡獎勵機制：
    - 問題：進度獎勵過高（5.0），安全獎勵過低（0.1），導致機器人學到「高速衝刺」策略
    - 解決：降低進度獎勵，提高安全獎勵和碰撞懲罰，強化速度控制
    
    調整策略：
    1. progressive_collision: -1.0 → -2.5（提高碰撞懲罰）
    2. distance_to_goal: 5.0 → 2.5（降低進度獎勵）
    3. reaching_goal: 500 → 300（降低目標獎勵）
    4. safe_navigation: 0.1 → 0.8（提高安全獎勵）
    5. speed_control_near_obstacles: 0.1 → 0.5（強化速度控制）
    """
    
    # ------------------------------------------------------------------------
    # 核心調整 1：碰撞懲罰（提高，強化安全意識）
    # ------------------------------------------------------------------------
    # Phase 2 問題：weight=-1.0 太低，機器人敢於高速衝刺
    # Phase 2.5 修復：提高懲罰，讓機器人更重視安全
    progressive_collision = RewTerm(
        func=charge_mdp.progressive_collision_penalty,
        weight=-2.5,  # 提高懲罰（-1.0 → -2.5），強化安全意識
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": 1.5,      # 保持 1.5 米開始懲罰
            "danger_distance": 1.0,    # 保持 1.0 米中等懲罰
            "collision_distance": 0.6,  # 保持 0.6 米最大懲罰
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 2：進度獎勵（降低，避免過度激勵）
    # ------------------------------------------------------------------------
    # Phase 2 問題：weight=5.0 太高，導致機器人過度追求速度
    # Phase 2.5 修復：降低進度獎勵，平衡安全與效率
    distance_to_goal = RewTerm(
        func=charge_mdp.progress_to_goal,
        weight=2.5,  # 降低進度獎勵（5.0 → 2.5），避免過度激勵
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 3：到達目標獎勵（降低，避免過度激勵）
    # ------------------------------------------------------------------------
    # Phase 2 問題：weight=500 太高，導致機器人不顧安全衝刺
    # Phase 2.5 修復：降低目標獎勵，平衡安全與效率
    reaching_goal = RewTerm(
        func=charge_mdp.reaching_goal,
        weight=300,  # 降低目標獎勵（500 → 300），避免過度激勵
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 4：安全導航獎勵（大幅提高，強化安全行為）
    # ------------------------------------------------------------------------
    # Phase 2 問題：weight=0.1 太低，safe_navigation 獎勵從 0.01 降至 0
    # Phase 2.5 修復：大幅提高安全獎勵，鼓勵安全導航行為
    safe_navigation = RewTerm(
        func=charge_mdp.safe_navigation_bonus,
        weight=0.8,  # 大幅提高（0.1 → 0.8），強化安全行為
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "comfort_distance": 2.0,  # 保持 2.0 米舒適距離
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 5：速度控制（大幅提高，強化提前減速）
    # ------------------------------------------------------------------------
    # Phase 2 問題：weight=0.1 太低，speed_control 持續為負值
    # Phase 2.5 修復：大幅提高速度控制權重，強化提前減速策略
    speed_control_near_obstacles = RewTerm(
        func=charge_mdp.speed_control_near_obstacles,
        weight=0.5,  # 大幅提高（0.1 → 0.5），強化速度控制
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "warning_distance": 1.5,  # 保持 1.5 米警告距離
            "max_safe_speed": 0.8,    # 降低最大安全速度（1.0 → 0.8 m/s），強化提前減速
        },
    )


@configclass
class TerminationsCfgV2_5(TerminationsCfgV2):
    """終止條件配置（Phase 2.5）
    
    繼承 Phase 2 的終止條件配置。
    """
    pass


@configclass
class CurriculumsCfgV2_5:
    """自適應課程學習配置（Phase 2.5）
    
    根據導航成功率與碰撞率動態調整環境難度。
    
    根據 Phase 2 訓練問題調整策略：
    - Phase 2 問題：起始難度過高（5 個障礙物），成功率在 10k 步後崩潰
    - Phase 2.5 修復：更保守的難度調整策略，確保穩定性
    
    調整策略：
    1. 降低目標成功率：70% → 65%（更現實的目標）
    2. 降低最大可接受碰撞率：30% → 25%（更嚴格的安全要求）
    3. 降低難度增加閾值：80% → 75%（更保守的難度提升）
    4. 提高難度降低閾值：50% → 60%（更積極的難度降低）
    """
    
    # 自適應課程學習：根據成功率和碰撞率調整難度
    adaptive_difficulty = CurrTerm(
        func=charge_mdp.adaptive_curriculum_update,
        params={
            "target_success_rate": 0.65,  # 目標成功率：65%（從 70% 降低，更現實）
            "max_collision_rate": 0.25,  # 最大可接受碰撞率：25%（從 30% 降低，更嚴格）
            "difficulty_increase_threshold": 0.75,  # 難度增加閾值：75%（從 80% 降低，更保守）
            "difficulty_decrease_threshold": 0.60,  # 難度降低閾值：60%（從 50% 提高，更積極）
        },
    )


@configclass
class EventCfgV2_5(EventCfgV2):
    """事件配置（Phase 2.5）

    Phase 2.5：自適應障礙物數量
    - 障礙物數量會根據課程學習動態調整（3-10 個）
    - 距離參數會根據難度等級動態調整
    - 障礙物是靜態的（不會移動）
    """
    
    # 重置障礙物位置（使用動態參數）
    # 注意：實際的參數值會在運行時根據課程學習結果動態設置
    reset_obstacles = EventTerm(
        func=charge_mdp.reset_obstacles_with_adaptive_params,  # 使用自適應參數版本
        mode="reset",
        params={
            "speed_range": 0.0,  # 靜態障礙物：不需要速度
            "min_speed": 0.0,
            # 這些參數會在運行時根據課程學習結果動態設置
            "min_robot_distance": 3.5,  # 初始值，會被動態調整
            "min_goal_distance": 2.0,  # 初始值，會被動態調整
            "min_obstacle_spacing": 2.0,  # 初始值，會被動態調整
            "max_spawn_attempts": 50,
            "boundary": 8.0,  # 場景邊界：8.0 米
        },
    )


@configclass
class ChargeNavigationEnvCfgV2_5(ChargeNavigationEnvCfg):
    """Phase 2.5 訓練配置：自適應課程學習
    
    Curriculum Learning 階段：
    - Phase 1: 3 個靜態障礙物 ✅（已完成）
    - Phase 2: 5 個靜態障礙物 ✅（已完成）
    - Phase 2.5: 自適應課程學習（當前，根據成功率和碰撞率動態調整難度）
    
    主要特點：
    1. 自適應障礙物數量：3-10 個（根據訓練表現動態調整）
    2. 自適應距離參數：根據難度等級動態調整
    3. 基於統計的難度調整：
       - 成功率 > 80% 且碰撞率 < 20%：增加難度
       - 成功率 < 50% 或碰撞率 > 40%：降低難度
       - 否則：保持當前難度
    4. 場景邊界：8.0 米
    5. 目標距離範圍：4.0-8.0 米
    """

    scene: MySceneCfgV2_5 = MySceneCfgV2_5(num_envs=128, env_spacing=25.0)
    observations: ObservationsCfgV2_5 = ObservationsCfgV2_5()
    commands: CommandsCfgV2_5 = CommandsCfgV2_5()
    rewards: RewardsCfgV2_5 = RewardsCfgV2_5()
    terminations: TerminationsCfgV2_5 = TerminationsCfgV2_5()
    events: EventCfgV2_5 = EventCfgV2_5()
    curriculum: CurriculumsCfgV2_5 = CurriculumsCfgV2_5()  # ⭐ 自適應課程學習


@configclass
class ChargeNavigationEnvCfgV2_5_PLAY(ChargeNavigationEnvCfg_PLAY):
    """Phase 2.5 測試/展示配置"""

    scene: MySceneCfgV2_5 = MySceneCfgV2_5(num_envs=16, env_spacing=25.0)
    observations: ObservationsCfgV2_5 = ObservationsCfgV2_5()
    commands: CommandsCfgV2_5 = CommandsCfgV2_5()
    rewards: RewardsCfgV2_5 = RewardsCfgV2_5()
    terminations: TerminationsCfgV2_5 = TerminationsCfgV2_5()
    events: EventCfgV2_5 = EventCfgV2_5()
    curriculum: CurriculumsCfgV2_5 = CurriculumsCfgV2_5()
