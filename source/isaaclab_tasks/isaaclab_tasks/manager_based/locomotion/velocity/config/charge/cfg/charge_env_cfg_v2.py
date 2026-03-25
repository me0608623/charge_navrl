"""
Charge 導航環境配置（Phase 2）

此版本在原本配置上，加入「動態障礙物移動」功能。
主要改進：
1. 障礙物數量設定為 5 個（減少到 5 個，增加距離設置）
2. 障礙物之間最小間距增加到 2.0 米（增加間距）
3. 障礙物設為動態（使用 RigidObjectCfg + 速度控制）
4. 目標距離範圍提高到 4.0-8.0 米（增加範圍）
5. 場景邊界增加到 8.0 米（從 5.0 米增加）
"""

import math  # 用於角度計算
import random  # 用於隨機生成障礙物

import isaaclab.sim as sim_utils  # 模擬工具
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # 資產基礎配置
from isaaclab.managers import EventTermCfg as EventTerm  # 事件配置類
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns  # 感測器配置
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
)

# ============================================================================
# 從 mdp 模組直接導入所需函數和類（模組化後直接導入）
# ============================================================================
# 觀測函數
from ..mdp.observations import (
    lidar_scan_2d_sweep,
    base_velocity_xy,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    dynamic_obstacles_state,
    safe_last_action,
)

# 獎勵函數
from ..mdp.rewards import (
    progressive_collision_penalty,
    progress_to_goal,
    reaching_goal,
    safe_navigation_bonus,
    speed_control_near_obstacles,
    collision_occurred,
)

# 終止條件函數
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    robot_flying,
    wall_collision,
)
from ..mdp.rewards import collision_occurred, collision_contact_occurred

# 事件函數
from ..mdp.events import reset_obstacles

# 狀態管理函數
from ..mdp.core import set_obstacle_metadata


@configclass
class MySceneCfgV2(MySceneCfg):
    """場景配置（Phase 2）
    
    覆蓋 Phase 1 的場景配置：
    1. 障礙物數量設定為 5 個（減少到 5 個，增加距離設置）
    2. 障礙物之間最小間距增加到 2.0 米（增加間距）
    3. 障礙物設為「靜態」（kinematic_enabled=True）
    4. 場景邊界增加到 8.0 米（從 5.0 米增加）
    5. Phase 3 才會啟用動態障礙物
    """
    
    def __post_init__(self):
        """在配置初始化後自動執行（Phase 2：5 個動態障礙物）
        
        注意：不調用父類的 __post_init__，因為父類會創建 3 個靜態障礙物。
        我們直接創建 5 個動態障礙物來覆蓋，並增加所有距離設置。
        """
        # 不調用 super().__post_init__()，避免創建重複的障礙物
        # 父類的 __post_init__ 只創建障礙物，其他配置（地形、機器人等）已經在類定義中完成
        
        # Phase 2：創建 5 個動態障礙物（增加距離設置）
        num_obstacles = 5  # Phase 2：5 個障礙物（減少到 5 個，增加距離）
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
                        kinematic_enabled=True,  # Phase 2：靜態障礙物（不會移動）
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
                        kinematic_enabled=True,  # Phase 2：靜態障礙物（不會移動）
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
            # ⭐ 使用 RigidObjectCfg 才能動態控制速度
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
        set_obstacle_metadata(num_obstacles, obstacle_sizes)
        
        # ====================================================================
        # 創建邊界牆壁（Phase 2：擴大場地）
        # ====================================================================
        # 牆壁參數
        boundary = 8.0  # 邊界距離（米）（從 5.0 米增加到 8.0 米）
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
        # 覆蓋 lidar 配置：加入牆壁 raycast target（Phase 2 新增）
        # ====================================================================
        # Phase 1 的 lidar 配置不包含牆壁（因為 Phase 1 沒有牆壁）
        # Phase 2 創建了牆壁，所以需要在 lidar 配置中加入牆壁的 raycast target
        from isaaclab.sensors import MultiMeshRayCasterCfg
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
                    prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
                    track_mesh_transforms=True,
                ),
                # Phase 2 新增：牆壁 raycast target
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,  # 牆壁是靜態的，不需要追蹤變換
                ),
            ],
            update_period=0.04,
        )
        
        # ====================================================================
        # 覆蓋 lidar 配置：加入牆壁 raycast target（Phase 2 新增）
        # ====================================================================
        # Phase 1 的 lidar 配置不包含牆壁（因為 Phase 1 沒有牆壁）
        # Phase 2 創建了牆壁，所以需要在 lidar 配置中加入牆壁的 raycast target
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
                    prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
                    track_mesh_transforms=True,
                ),
                # Phase 2 新增：牆壁 raycast target
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",
                    track_mesh_transforms=False,  # 牆壁是靜態的，不需要追蹤變換
                ),
            ],
            update_period=0.04,
        )
        
        # ====================================================================
        # 覆蓋接觸感測器：加入牆壁碰撞檢測（Phase 2 修復）
        # ====================================================================
        # Phase 1 的接觸感測器只監測障礙物（Obstacle_.*），沒有包含牆壁
        # Phase 2 需要同時監測障礙物和牆壁的碰撞
        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",  # 只監測機器人底盤
            update_period=0.0,  # 更新週期：0.0 = 每步都更新（最快）
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Obstacle_.*",  # 監測與障礙物的碰撞
                "{ENV_REGEX_NS}/Wall_.*",       # 監測與牆壁的碰撞（Phase 2 新增）
            ],
            debug_vis=False,  # 調試可視化：False = 不顯示接觸點（提升性能）
        )


@configclass
class ObservationsCfgV2(ObservationsCfg):
    """觀測配置（Phase 2）
    
    覆蓋觀測配置，將障礙物數量從 3 個更新為 10 個。
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組（Phase 2）"""
        
        # ----------------------------------------------------------------
        # 觀測項 1：2D 平面掃描（72 個角度）
        # ----------------------------------------------------------------
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # ----------------------------------------------------------------
        # 觀測項 2：速度資訊（機器人座標系）
        # ----------------------------------------------------------------
        speed = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # ----------------------------------------------------------------
        # 觀測項 3：目標相對位置
        # ----------------------------------------------------------------
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # ----------------------------------------------------------------
        # 觀測項 4：目標距離
        # ----------------------------------------------------------------
        goal_distance = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        # ----------------------------------------------------------------
        # 觀測項 5：時間剩餘比例
        # ----------------------------------------------------------------
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )
        
        # ----------------------------------------------------------------
        # 觀測項 6：存活/死亡狀態（關鍵特徵）
        # ----------------------------------------------------------------
        # 作用 1：幫助 Critic Network 正確估算期望值
        #   - 死亡（碰撞）後 Value = 0，因為無法獲得未來獎勵
        #   - 存活時 Value = 正常估算的期望獎勵
        # 作用 2：強制融合「避障」與「抵達目標」的能力
        #   - 讓 Agent 明白：「活著」是獲得任何獎勵的前提
        #   - 即使抵達目標的獎勵很高，碰撞後也無法獲得
        alive = ObsTerm(
            func=alive_flag,
        )
        
        # ----------------------------------------------------------------
        # 觀測項 7：動態障礙物資訊（Phase 2：5 個障礙物）
        # ----------------------------------------------------------------
        obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 5,  # Phase 2：5 個障礙物
                "max_obstacles": MAX_OBSTACLES,  # 固定最大數量（用於 padding）
                "max_distance": 15.0,  # 增加最大觀測距離（從 10.0 米增加到 15.0 米）
            },
        )
        
        # ----------------------------------------------------------------
        # 觀測項 8：上一步動作
        # ----------------------------------------------------------------
        actions = ObsTerm(func=safe_last_action)
        
        def __post_init__(self):
            """觀測組初始化配置"""
            self.enable_corruption = True  # 啟用數據損壞（加噪音）
            self.concatenate_terms = True  # 串接所有觀測項
    
    policy: PolicyCfg = PolicyCfg()  # 實例化策略觀測組


@configclass
class CommandsCfgV2(CommandsCfg):
    """命令配置（Phase 2）
    
    覆蓋目標距離範圍，確保 agent 重生時相對目標位置至少 3 米遠。
    這適合 Phase 2 訓練，因為：
    1. Phase 1 已經學會了基本導航（1.5-3.0 米）
    2. Phase 2 需要更遠的距離來應對動態障礙物
    3. 更遠的距離提供更多時間和空間來規劃路徑
    
    新增碰撞檢查：
    - 目標不會生成在牆壁邊界附近（保持 0.5 米安全邊距）
    - 目標不會生成在障礙物上或附近（保持 1.0 米安全距離）
    """
    goal_command = GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),  # 只在環境重置時生成新目標
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(4.0, 8.0),  # Phase 2：目標距離範圍 4.0-8.0 米（增加範圍）
            # 從原本的 (3.0, 5.0) 提高到 (4.0, 8.0)
            # 最小值 4.0 確保 agent 重生時相對目標位置至少 4 米遠
            # 最大值 8.0 提供更大的挑戰性，適應更大的場景
            angle=(-math.pi, math.pi),  # 全方位（360度）
        ),
        # 碰撞檢查配置（Phase 2 新增，增加距離設置）
        wall_boundary=8.0,        # 牆壁邊界距離（與場景配置一致，從 5.0 米增加到 8.0 米）
        wall_safe_margin=1.0,     # 牆壁安全邊距（目標離牆至少 1.0 米，從 0.5 米增加）
        obstacle_safe_distance=1.0,  # 障礙物安全距離（目標離障礙物至少 1.0 米，從 0.5 米增加）
        max_resample_attempts=50,    # 最大重試次數
        num_obstacles=5,         # 障礙物數量（與場景配置一致，從 10 個減少到 5 個）
    )


@configclass
class RewardsCfgV2(RewardsCfg):
    """Phase 2 獎勵配置：10 個靜態障礙物（目標導向優化版）
    
    根據訓練曲線分析的問題：
    - time_out 從 0.1 上升到 0.5+（機器人越來越難到達目標）
    - goal_reached 從 0.5 下降到 0.3（成功率下降）
    - speed_control 持續負值（阻礙機器人移動）
    
    優化策略：大幅強化目標導向，放寬避障限制
    - progressive_collision: -1.0（大幅降低懲罰）
    - safe_distance: 0.8m（縮小到只在很近時才懲罰）
    - distance_to_goal: 5.0（大幅提高進度獎勵）
    - reaching_goal: 500（大幅提高目標獎勵）
    - safe_navigation: 0.1（最小化干擾）
    - speed_control: 0.1, warning_distance=0.8m（只在很近時限速）
    """
    
    # ------------------------------------------------------------------------
    # 核心調整 1：碰撞懲罰（大幅放寬，讓機器人更敢移動）
    # ------------------------------------------------------------------------
    # 問題：訓練曲線顯示 time_out 上升、goal_reached 下降
    # 解決：大幅降低碰撞懲罰，只在真正接近碰撞時才懲罰
    progressive_collision = RewTerm(
        func=progressive_collision_penalty,
        weight=-1.0,  # 大幅降低（-2.0 → -1.0），讓機器人更敢移動
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "safe_distance": 1.5,      # 1.5米開始懲罰（從 0.8 米增加）
            "danger_distance": 1.0,    # 1.0米中等懲罰（從 0.5 米增加）
            "collision_distance": 0.6, # 0.6米最大懲罰（從 0.35 米增加）
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 2：進度獎勵（大幅提高，強化目標導向）
    # ------------------------------------------------------------------------
    # 問題：approaching_goal 獎勵太小（0.0006-0.0014），不足以驅動機器人
    # 解決：大幅提高進度獎勵權重
    distance_to_goal = RewTerm(
        func=progress_to_goal,
        weight=5.0,  # 大幅提高（2.5 → 5.0），強化目標導向
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 3：到達目標獎勵（大幅提高）
    # ------------------------------------------------------------------------
    reaching_goal = RewTerm(
        func=reaching_goal,
        weight=500,  # 大幅提高（300 → 500），強烈鼓勵到達目標
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 4：安全導航獎勵（進一步降低干擾）
    # ------------------------------------------------------------------------
    safe_navigation = RewTerm(
        func=safe_navigation_bonus,
        weight=0.1,  # 進一步降低（0.2 → 0.1），減少干擾
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "comfort_distance": 2.0,  # 增加舒適距離（從 1.0 米增加到 2.0 米）
        },
    )
    
    # ------------------------------------------------------------------------
    # 核心調整 5：速度控制（大幅放寬，避免阻礙移動）
    # ------------------------------------------------------------------------
    # 問題：訓練曲線顯示 speed_control 是負值（約 -0.004），在阻礙機器人
    # 解決：大幅縮小警告範圍，只在非常接近時才限速
    speed_control_near_obstacles = RewTerm(
        func=speed_control_near_obstacles,
        weight=0.1,  # 大幅降低（0.3 → 0.1），減少對移動的阻礙
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "warning_distance": 1.5,  # 增加警告距離（從 0.8 米增加到 1.5 米）
            "max_safe_speed": 1.0,    # 提高到 1.0 m/s，允許正常速度
        },
    )
    
    # 注意：牆壁碰撞已統一通過 Lidar 的 progressive_collision 處理
    # 不再需要獨立的 wall_collision 懲罰


@configclass
class TerminationsCfgV2(TerminationsCfg):
    """Phase 2 終止條件配置
    
    碰撞檢測機制（兩種方式同時生效，任一觸發即終止）：
    1. collision_contact：接觸感測器（繼承自 Phase 1，已更新加入牆壁監測）
    2. collision：雷達距離判定（增加閾值到 0.5 米）
    
    修復內容：
    - 閾值從 0.28 → 0.5 米（更靈敏的碰撞檢測）
    - 接觸感測器已在場景配置中加入牆壁監測（Wall_.*）
    """
    
    # ------------------------------------------------------------------------
    # 覆蓋：雷達距離碰撞（增加閾值，更靈敏）
    # ------------------------------------------------------------------------
    # 問題：原閾值 0.28~0.4 米太小，機器人可能已經碰到障礙物但還沒觸發終止
    # 原因：
    #   1. 雷達在機器人中心上方 0.5 米，測量的是 2D 水平距離
    #   2. 機器人底盤半徑約 0.28 米
    #   3. 雷達每 5 度發射一條射線，可能錯過某些角度的障礙物
    # 修復：增加閾值到 0.5 米，提供更大的安全緩衝
    collision = DoneTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "threshold": 0.5,  # ⭐ 增加閾值：0.28 → 0.5 米（更靈敏的碰撞檢測）
        },
    )
    
    # ------------------------------------------------------------------------
    # 繼承：接觸感測器碰撞（collision_contact）
    # ------------------------------------------------------------------------
    # 從 TerminationsCfg 繼承，使用物理引擎的真實碰撞檢測
    # 注意：場景配置（MySceneCfgV2）中已更新 contact_sensor，加入牆壁監測
    # 兩個終止條件是「或」關係，任一觸發都會終止


@configclass
class EventCfgV2(EventCfg):
    """事件配置（Phase 2）

    Phase 2：5 個靜態障礙物（減少數量，增加距離）
    - 障礙物數量設定為 5 個（從 10 個減少）
    - 障礙物之間最小間距增加到 2.0 米（從 0.5 米增加）
    - 障礙物是靜態的（不會移動）
    - Phase 3 才會啟用動態障礙物
    """

    # ------------------------------------------------------------------------
    # 重置障礙物位置（Phase 2：靜態障礙物）
    # ------------------------------------------------------------------------
    reset_obstacles = EventTerm(
        func=reset_obstacles,  # 使用靜態障礙物重置函數
        mode="reset",
        params={
            "speed_range": 0.0,  # 靜態障礙物：不需要速度
            "min_speed": 0.0,
            "min_robot_distance": 3.5,  # 增加距離（從 2.5 米增加到 3.5 米）
            "min_goal_distance": 2.0,  # 增加距離（從 1.0 米增加到 2.0 米）
            "min_obstacle_spacing": 2.0,  # 增加間距（從 0.5 米增加到 2.0 米）
            "max_spawn_attempts": 50,
            "boundary": 8.0,  # 場景邊界：8.0 米（Phase 2 使用更大的場景）
        },
    )
    
    # 注意：Phase 2 不需要 move_obstacles 事件，因為障礙物是靜態的
    # Phase 3 會添加動態移動事件


@configclass
class ChargeNavigationEnvCfgV2(ChargeNavigationEnvCfg):
    """Phase 2 訓練配置：5 個靜態障礙物（減少數量，增加距離）
    
    Curriculum Learning 階段：
    - Phase 1: 3 個靜態障礙物 ✅（已完成）
    - Phase 2: 5 個靜態障礙物（當前，減少數量並增加距離）
    - Phase 3: 5 個動態障礙物（之後）
    
    主要改進：
    1. 障礙物數量設定為 5 個（從 10 個減少）
    2. 障礙物之間最小間距增加到 2.0 米（從 0.5 米增加）
    3. 目標距離範圍提高到 4.0-8.0 米（從 3.0-5.0 米增加）
    4. 場景邊界增加到 8.0 米（從 5.0 米增加）
    5. 新增邊界牆壁和牆壁碰撞檢測
    """

    scene: MySceneCfgV2 = MySceneCfgV2(num_envs=128, env_spacing=25.0)  # 覆蓋場景配置，使用 5 個障礙物，增加環境間隔
    observations: ObservationsCfgV2 = ObservationsCfgV2()  # 覆蓋觀測配置，使用 10 個障礙物
    commands: CommandsCfgV2 = CommandsCfgV2()  # 覆蓋命令配置，使用更大的目標距離
    rewards: RewardsCfgV2 = RewardsCfgV2()  # ⭐ 覆蓋獎勵配置，使用 Phase 2 專屬設計
    terminations: TerminationsCfgV2 = TerminationsCfgV2()  # ⭐ 覆蓋終止條件，新增牆壁碰撞
    events: EventCfgV2 = EventCfgV2()


@configclass
class ChargeNavigationEnvCfgV2_PLAY(ChargeNavigationEnvCfg_PLAY):
    """Phase 2 測試/展示配置"""

    scene: MySceneCfgV2 = MySceneCfgV2(num_envs=16, env_spacing=25.0)  # 測試時使用較少環境，增加環境間隔
    observations: ObservationsCfgV2 = ObservationsCfgV2()  # 使用 Phase 2 的觀測配置
    commands: CommandsCfgV2 = CommandsCfgV2()  # 使用 Phase 2 的目標距離配置
    rewards: RewardsCfgV2 = RewardsCfgV2()  # 使用 Phase 2 的獎勵配置
    terminations: TerminationsCfgV2 = TerminationsCfgV2()  # 使用 Phase 2 的終止條件
    events: EventCfgV2 = EventCfgV2()
