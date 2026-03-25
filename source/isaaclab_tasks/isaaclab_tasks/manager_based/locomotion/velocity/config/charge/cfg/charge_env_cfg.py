# ============================================================================
# 文件說明
# ============================================================================
"""
Charge 激光雷達導航環境配置

這個文件定義了完整的強化學習訓練環境，包括：
1. 場景配置（地面、機器人、雷達、障礙物）
2. 觀測配置（機器人能「看到」什麼）
3. 動作配置（機器人能「做」什麼）
4. 獎勵配置（什麼行為會得分/扣分）
5. 終止條件（什麼情況下遊戲結束）
6. 事件配置（重置時發生什麼）
"""

# ============================================================================
# 模塊導入
# ============================================================================
import math  # 數學函數（用於角度計算，例如 math.pi）

# Isaac Lab 核心模塊
import isaaclab.sim as sim_utils  # 模擬工具：物理引擎、3D物體生成
from isaaclab.assets import AssetBaseCfg  # 資產基礎配置：場景物體的通用類別
from isaaclab.envs import ManagerBasedRLEnvCfg  # 強化學習環境配置基類

# 管理器模塊（MDP 元件）
from isaaclab.managers import (
    SceneEntityCfg,  # 場景實體配置：引用場景中的物體（機器人、雷達等）
    ObservationGroupCfg as ObsGroup,  # 觀測組配置：將多個觀測項組合
    ObservationTermCfg as ObsTerm,  # 觀測項配置：定義單個觀測（雷達數據、目標位置等）
    RewardTermCfg as RewTerm,  # 獎勵項配置：定義單個獎勵函數
    TerminationTermCfg as DoneTerm,  # 終止條件配置：定義遊戲結束條件
    EventTermCfg as EventTerm,  # 事件配置：定義特定時機觸發的事件（如重置）
)

# 場景和傳感器
from isaaclab.scene import InteractiveSceneCfg  # 交互場景配置：可以包含動態物體的場景
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns  # 感測器配置
from isaaclab.terrains import TerrainImporterCfg  # 地形導入器：生成地面
from isaaclab.utils import configclass  # 配置類裝飾器：標記為配置類
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 噪音配置：模擬傳感器噪音

# Isaac Lab 預設的運動控制函數庫
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# 本專案的自定義模塊
from .charge_cfg import CHARGE_CFG  # Charge 機器人配置
from ..goal_command import GoalCommandCfg  # 目標位置生成器配置

# ============================================================================
# 從 mdp 模組直接導入所需函數和類（模組化後直接導入）
# ============================================================================
# 動作類
from ..mdp.actions import DifferentialDriveActionCfg

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
    time_out_penalty,
    move_reward,
    alignment_reward,
)

# 終止條件函數
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    robot_flying,
)
from ..mdp.rewards import collision_contact_occurred  # 同時作為獎勵和終止條件

# 事件函數
from ..mdp.events import reset_obstacles, reset_root_state_fixed_per_env

# 狀態管理函數
from ..mdp.core import set_obstacle_metadata

# ============================================================================
# 環境常數定義（Environment Constants）
# ============================================================================
# 2024-01 修正：統一定義常數，避免硬編碼魔法數字
# 這些常數在多處使用，統一管理便於維護和調整

# 機器人物理參數
ROBOT_BODY_RADIUS = 0.5  # 機器人身體半徑（米），用於碰撞判定和目標到達判定

# 目標和碰撞判定閾值
GOAL_REACH_THRESHOLD = ROBOT_BODY_RADIUS  # 到達目標的距離閾值（米）
COLLISION_THRESHOLD = ROBOT_BODY_RADIUS  # 碰撞判定的距離閾值（米）

# 安全距離參數
SAFE_DISTANCE = 1.5  # 障礙物安全距離（米），用於漸進式碰撞懲罰
DANGER_DISTANCE = 0.8  # 危險距離（米），低於此距離開始中等懲罰
COLLISION_DISTANCE = 0.4  # 碰撞距離（米），低於此距離最大懲罰

# 障礙物重置參數
MIN_ROBOT_DISTANCE = 1.5  # 障礙物與機器人的最小距離（米）
MIN_GOAL_DISTANCE = 1.0  # 障礙物與目標的最小距離（米）
MIN_OBSTACLE_SPACING = 1.0  # 障礙物之間的最小距離（米）

# 觀測空間固定維度參數（2024-01 架構修復）
MAX_OBSTACLES = 10  # 最大障礙物數量（用於固定觀測維度）
# 說明：為了支持權重遷移（Phase 1 → Phase 2），觀測維度必須固定
# Phase 1: 3 個障礙物 → 使用 padding 填充到 10 個
# Phase 2: 10 個障礙物 → 直接使用
# 不存在的障礙物用特殊標記填充：size = -1.0（明確標記「不存在」）

# ============================================================================
# 場景配置（Scene Configuration）
# ============================================================================
# 定義模擬世界中有哪些物體：地面、機器人、雷達、光照、障礙物
# ============================================================================

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """場景配置
    
    這個類定義了模擬世界的所有物體和它們的初始狀態。
    InteractiveSceneCfg 表示這是一個可交互的場景（物體會動、會碰撞）。
    """
    
    # ------------------------------------------------------------------------
    # 地形配置
    # ------------------------------------------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 場景路徑：在 USD 場景樹中的位置
        # "/World/ground" 表示這是全局唯一的地面（所有環境共享）
        
        terrain_type="plane",  # 地形類型：平面
        # 其他選項："generator"（程序生成地形）、"usd"（從文件加載）
        
        collision_group=-1,  # 碰撞組：-1 表示與所有物體碰撞
        # 碰撞組用於控制哪些物體之間會碰撞（-1 = 全部碰撞）
        
        # 物理材質配置（摩擦力、彈性）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦力組合模式：相乘
            # 當兩個物體接觸時，最終摩擦力 = 物體A的摩擦力 × 物體B的摩擦力
            # 其他選項："average"（平均）、"min"（取最小值）、"max"（取最大值）
            
            restitution_combine_mode="multiply",  # 彈性組合模式：相乘
            # restitution（恢復係數）決定碰撞後的反彈程度
            # 0 = 完全不反彈（黏土），1 = 完全彈性碰撞（超級球）
            
            static_friction=1.0,  # 靜摩擦係數：1.0（正常摩擦）
            # 靜摩擦：物體從靜止開始移動需要的力
            # 1.0 = 標準摩擦（輪子不會打滑），0.1 = 冰面
            
            dynamic_friction=1.0,  # 動摩擦係數：1.0（正常摩擦）
            # 動摩擦：物體移動中的摩擦力
            # 通常 dynamic_friction ≤ static_friction
        ),
    )
    
    # ------------------------------------------------------------------------
    # 機器人配置
    # ------------------------------------------------------------------------
    robot = CHARGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 使用之前定義的 CHARGE_CFG（機器人物理模型）
    # .replace() 修改其中的參數
    # prim_path="{ENV_REGEX_NS}/Robot"：設定機器人在場景中的路徑
    #   {ENV_REGEX_NS} 是正則表達式，會被替換成 /World/envs/env_0、env_1...
    #   這樣每個並行環境都有自己的機器人實例
    
    # ------------------------------------------------------------------------
    # 激光雷達配置
    # ------------------------------------------------------------------------
    
    # ========================================================================
    # 舊配置（單層水平掃描雷達）- 已註釋保留
    # ========================================================================
    # lidar = MultiMeshRayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
    #     offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=1,  # 單通道（只有一層水平掃描）
    #         vertical_fov_range=(0.0, 0.0),  # 垂直視野：0 度（純水平掃描）
    #         horizontal_fov_range=(-180.0, 180.0),  # 水平視野：360 度
    #         horizontal_res=1.0,  # 水平解析度：1 度/射線（總共 360 條射線）
    #     ),
    #     max_distance=10.0,
    #     debug_vis=False,
    #     mesh_prim_paths=[
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             prim_expr="/World/ground",
    #             track_mesh_transforms=False,
    #         ),
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
    #             track_mesh_transforms=True,
    #         ),
    #     ],
    #     update_period=0.04,
    # )
    
    # ========================================================================
    # 舊配置（Velodyne VLP-16 16 線激光雷達）- 已註釋保留
    # ========================================================================
    # VLP-16 是 Velodyne 的 16 線旋轉式激光雷達，廣泛用於自動駕駛和機器人導航
    # 參考規格：https://velodynelidar.com/wp-content/uploads/2019/12/63-9229_Rev-K_Puck-_Datasheet_Web.pdf
    # lidar = MultiMeshRayCasterCfg(
    #     # 感測器安裝位置：機器人底盤連結（base_link）
    #     # /World/envs/env_0/Robot/charger_rover_urdf5/base_link
    #     prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
    #     # 感測器偏移量：相對於 base_link 的位置
    #     # pos=(0.0, 0.0, 0.2) 表示：
    #     #   X: 0.0 米（前後方向，0 = 不偏移）
    #     #   Y: 0.0 米（左右方向，0 = 不偏移）
    #     #   Z: 0.2 米（上下方向，0.2 = 向上偏移 20 公分）
    #     # 這樣雷達會安裝在機器人上方 20 公分處，避免被機器人本體遮擋
    #     offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
    #
    #     # 感測器姿態跟隨模式：只跟隨機器人的偏航角（Yaw）
    #     # True = 雷達只會隨機器人左右旋轉（Z 軸旋轉），不會隨俯仰（Pitch）和滾轉（Roll）傾斜
    #     # 這模擬了雷達安裝在穩定平台上的情況（例如使用萬向節或穩定器）
    #     # False = 雷達會完全跟隨機器人的所有旋轉（包括翻轉時也會跟著翻）
    #     attach_yaw_only=True,
    #     # ====================================================================
    #     # VLP-16 掃描模式配置
    #     # ====================================================================
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         # 通道數：16（VLP-16 的核心特徵）
    #         channels=16,
    #         # 垂直視場角範圍：-15° 到 +15°（總共 30°）
    #         # 這表示：
    #         #   -15° = 最下方的掃描線（向下傾斜 15 度）
    #         #   +15° = 最上方的掃描線（向上傾斜 15 度）
    #         #   16 條掃描線在這 30 度範圍內均勻分佈
    #         #   每條線之間的垂直間隔約為 30° / 15 = 2°（因為有 16 個點，15 個間隔）
    #         # 實際 VLP-16 的垂直角度分佈不是完全均勻的，但這裡用均勻分佈來近似
    #         vertical_fov_range=(-15.0, 15.0),
    #         
    #         # 水平視場角範圍：-180° 到 +180°（總共 360°）
    #         # 這表示雷達可以掃描完整的一圈（全方位掃描）
    #         #   -180° = 正後方
    #         #   0° = 正前方
    #         #   +180° = 正後方（與 -180° 重合，形成完整圓圈）
    #         horizontal_fov_range=(-180.0, 180.0),
    #         
    #         # 水平解析度：0.2 度/射線
    #         # 這表示每 0.2 度發射一條射線
    #         # 總水平射線數 = 360° / 0.2° = 1800 條射線
    #         # 每層掃描線都有 1800 條水平射線
    #         # 總射線數 = 16 層 × 1800 條/層 = 28,800 條射線（每幀）
    #         # 這是 VLP-16 在 10 Hz 旋轉速度下的典型解析度
    #         # 注意：實際 VLP-16 的水平解析度會根據旋轉速度變化（5-20 Hz），這裡使用典型值
    #         horizontal_res=0.2,
    #     ),
    #     
    #     # 最大檢測距離：10 米
    #     # 超過這個距離的物體不會被檢測到（返回最大距離值）
    #     # VLP-16 的實際最大檢測距離可達 100 米，但這裡設為 10 米以適應室內導航任務
    #     # 如果任務需要檢測更遠的物體，可以增加到 20-30 米
    #     max_distance=10.0,
    #     
    #     # 調試可視化：關閉
    #     # False = 不顯示紅色點雲（射線碰撞點的可視化）
    #     # True = 顯示紅色點雲，可以用來調試雷達是否正常工作
    #     # 開啟後會看到紅色點標示出所有被檢測到的物體表面
    #     debug_vis=True,
    #     
    #     # ====================================================================
    #     # 射線投射目標配置（檢測哪些物體）
    #     # ====================================================================
    #     # 雷達會向這些目標發射射線，檢測碰撞距離
    #     mesh_prim_paths=[
    #         # 目標 1：全局地面（所有環境共享的靜態地面）
    #         # prim_expr="/World/ground" 表示檢測路徑為 /World/ground 的所有物體
    #         # track_mesh_transforms=False 表示不追蹤變換（因為地面是靜態的，不會移動）
    #         # 這可以提升性能，因為不需要每幀更新地面的位置
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             prim_expr="/World/ground",
    #             track_mesh_transforms=False,  # 地面是靜態的，不需要追蹤變換
    #         ),
    #         
    #         # 目標 2：環境中的動態障礙物
    #         # prim_expr="{ENV_REGEX_NS}/Obstacle_.*" 表示檢測每個環境中的障礙物
    #         # 這個正則表達式會匹配：
    #         #   /World/envs/env_0/Obstacle_0
    #         #   /World/envs/env_0/Obstacle_1
    #         #   /World/envs/env_1/Obstacle_0
    #         #   ... 等等
    #         # track_mesh_transforms=True 表示需要追蹤變換（因為障礙物可能會移動或重置位置）
    #         # 這確保雷達能正確檢測到移動中的障礙物
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             prim_expr="{ENV_REGEX_NS}/Obstacle_.*",  # 匹配所有障礙物（Obstacle_0, Obstacle_1, ...）
    #             track_mesh_transforms=True,  # 障礙物可能移動，需要追蹤變換
    #         ),
    #     ],
    #     
    #     # 更新週期：0.04 秒（25 Hz）
    #     # 這表示雷達每 0.04 秒更新一次掃描數據
    #     # 25 Hz 的更新頻率對於導航任務來說已經足夠（人類反應時間約 0.1-0.2 秒）
    #     # 實際 VLP-16 的旋轉速度通常是 5-20 Hz（對應 0.2-0.05 秒/轉）
    #     # 這裡設為 0.04 秒（25 Hz）是為了與控制頻率同步（見 __post_init__ 中的 decimation=4）
    #     update_period=0.04,
    # )
    
    # ========================================================================
    # 新配置（2D 平面掃描 - 72 個角度）
    # ========================================================================
    # 設計：將機器人周圍 360 度切分為 72 個等份（每 5 度一格）
    # 內容：每個角度紀錄「從機器狗中心點到最近障礙物的距離」
    # 來源：由 3D 點雲資料轉換為鳥瞰圖的可行駛區域後計算得出
    # 這代表機器人只知道「哪個方向多遠有牆」，而不知道那個牆是什麼顏色的
    # 從 36 個角度增加到 72 個角度，提高掃描精度（射線數從 36 增加到 72）
    lidar = MultiMeshRayCasterCfg(
        # 感測器安裝位置：機器人底盤連結（base_link）
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
        
        # 感測器偏移量：相對於 base_link 的位置
        # pos=(0.0, 0.0, 0.2) 表示安裝在機器人上方 20 公分處
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        
        # 感測器姿態跟隨模式：只跟隨機器人的偏航角（Yaw）
        ray_alignment="yaw",  # 使用新的參數替代已棄用的 attach_yaw_only
        # attach_yaw_only=True → ray_alignment="yaw"（只跟隨偏航角）
        
        # ====================================================================
        # 2D 平面掃描模式配置
        # ====================================================================
        pattern_cfg=patterns.LidarPatternCfg(
            # 通道數：1（單層水平掃描）
            channels=1,
            
            # 垂直視場角範圍：0 度（純水平掃描）
            # 這表示只在水平面上掃描，不檢測上下方向的障礙物
            vertical_fov_range=(0.0, 0.0),
            
            # 水平視場角範圍：-180° 到 +180°（總共 360°）
            # 這表示雷達可以掃描完整的一圈（全方位掃描）
            horizontal_fov_range=(-180.0, 180.0),
            
            # 水平解析度：5 度/射線
            # 這表示每 5 度發射一條射線
            # 總水平射線數 = 360° / 5° = 72 條射線
            # 每條射線對應一個角度區間（5 度扇形區域）
            # 從 10 度改為 5 度，提高掃描精度（射線數從 36 增加到 72）
            horizontal_res=5.0,
        ),
        
        # 最大檢測距離：10 米
        max_distance=10.0,
        
        # 調試可視化：開啟（顯示點雲掃描可視化）
        # True = 顯示紅色點雲（射線碰撞點的可視化），可以用來調試雷達是否正常工作
        # 開啟後會看到紅色點標示出所有被檢測到的物體表面
        # 這對於觀察 2D 平面掃描的效果非常有用
        debug_vis=True,
        
        # ====================================================================
        # 射線投射目標配置（檢測哪些物體）
        # ====================================================================
        mesh_prim_paths=[
            # 目標 1：全局地面（所有環境共享的靜態地面）
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                track_mesh_transforms=False,  # 地面是靜態的，不需要追蹤變換
            ),
            
            # 目標 2：環境中的動態障礙物
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle_.*",  # 匹配所有障礙物
                track_mesh_transforms=True,  # 障礙物可能移動，需要追蹤變換
            ),
            
            # 注意：Phase 1 沒有牆壁，所以不包含 Wall_.* 的 raycast target
            # Phase 2+ 會在場景配置中創建牆壁，並在各自的配置中覆蓋 lidar 配置以包含牆壁
        ],
        
        # 更新週期：0.04 秒（25 Hz）
        update_period=0.04,
    )
    
    # ------------------------------------------------------------------------
    # 接觸感測器配置（用於檢測機器人與障礙物的真實碰撞）
    # ------------------------------------------------------------------------
    # 注意：當使用 filter_prim_paths_expr 時，prim_path 必須指向單個具體的部件
    # 因此我們只監測機器人的底盤（base_link），這是主要的碰撞體
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",  # 只監測機器人底盤
        # 使用機器人的主體（base_link）作為接觸感測器
        # 這是機器人的主要碰撞體，可以代表整個機器人與障礙物的碰撞
        # 路徑格式：/World/envs/env_0/Robot/charger_rover_urdf5/base_link
        
        update_period=0.0,  # 更新週期：0.0 = 每步都更新（最快）
        
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"],  # 只監測與障礙物的碰撞
        # 過濾器：只報告與障礙物的接觸，忽略與地面、目標等的接觸
        # 這樣可以精確檢測機器人是否撞到障礙物
        
        debug_vis=False,  # 調試可視化：False = 不顯示接觸點（提升性能）
        # True = 顯示紅色接觸點（用於調試）
    )
    # 注意：這個感測器會檢測機器人底盤與障礙物的碰撞
    # 比雷達距離判定更準確（使用真實的碰撞框）
    # 雖然只監測底盤，但這已經足夠檢測大部分碰撞情況
    
    # ------------------------------------------------------------------------
    # 光照配置
    # ------------------------------------------------------------------------
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",  # 光源路徑：全局唯一
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
        # Dome Light（穹頂光）：來自四面八方的環境光
        # intensity=1000.0：光照強度（標準值）
        # 如果場景太暗，可以增加到 2000-3000
    )
    
    # ------------------------------------------------------------------------
    # 動態生成障礙物
    # ------------------------------------------------------------------------
    def __post_init__(self):
        """在配置初始化後自動執行
        
        這個函數會在場景配置創建後自動調用，用於動態添加障礙物。
        為什麼不在上面直接定義？因為需要循環生成多個障礙物。
        
        注意（2024-01 架構問題）：
        - 障礙物的形狀和大小在配置創建時就固定了（使用 random）
        - 這意味著所有環境實例會有相同的障礙物形狀和大小
        - 只有位置會在 reset_obstacles 事件中重新隨機
        - 如果要每次重置時都改變形狀，需要動態修改 USD 場景（較複雜）
        - 當前設計：形狀固定但位置隨機，已足夠提供多樣性
        """
        super().__post_init__()  # 調用父類的初始化
        import random  # 導入隨機數生成器（用於隨機障礙物）
        
        num_obstacles = 3  # Phase 1A：降低到 3 個障礙物（原本 10 個）
        # Curriculum Learning 策略：
        # Phase 1A: 3 個障礙物 → 訓練到成功率 > 0.7
        # Phase 1B: 6 個障礙物 → 訓練到成功率 > 0.6
        # Phase 1C: 10 個障礙物 → 最終目標
        # 這樣可以避免 agent 一開始就面對過於複雜的場景
        obstacle_sizes: list[float] = []
        for i in range(num_obstacles):
            # 循環生成 4 個障礙物（obstacle_0, obstacle_1, obstacle_2, obstacle_3）
            # ----------------------------------------------------------------
            # 隨機選擇障礙物類型（立方體 or 圓柱體）
            # ----------------------------------------------------------------
            is_cube = random.random() > 0.5
            # random.random() 生成 0-1 之間的隨機數
            # > 0.5 表示有 50% 機率是立方體，50% 機率是圓柱體
            
            if is_cube:
                # ============================================================
                # 立方體障礙物
                # ============================================================
                size = (
                    random.uniform(0.3, 0.8),  # X 軸長度：0.3-0.8 米
                    random.uniform(0.3, 0.8),  # Y 軸寬度：0.3-0.8 米
                    random.uniform(0.8, 1.5),  # Z 軸高度：0.8-1.5 米（高於 ray_caster 0.5m）
                )
                # uniform(a, b) 在 [a, b] 範圍內生成隨機數
                # 這樣每個立方體的大小都不同
                
                spawn_cfg = sim_utils.CuboidCfg(
                    size=size,  # 使用上面生成的隨機大小
                    
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    # kinematic_enabled=True：運動學物體（不受力影響，固定不動）
                    # True = 障礙物是「死物」（機器人撞不動它）
                    # False = 障礙物會被推動（像箱子）
                    
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    # 碰撞屬性：使用默認值（會阻擋機器人）
                    
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            random.random(),  # R（紅色通道）：0-1
                            random.random(),  # G（綠色通道）：0-1
                            random.random(),  # B（藍色通道）：0-1
                        ),
                        # 隨機顏色：每個障礙物顏色不同（方便區分）
                        
                        metallic=0.2,  # 金屬度：0.2（略帶金屬質感）
                        # 0.0 = 完全啞光, 1.0 = 像鏡子
                    ),
                )
                height_i = size[2] / 2  # 計算質心高度（用於設定初始位置）
                size_scalar = max(size[0], size[1])  # 以 XY 最大邊長作為尺寸近似
                # 立方體的質心在中心，所以是高度的一半
                
            else:
                # ============================================================
                # 圓柱體障礙物
                # ============================================================
                radius = random.uniform(0.2, 0.5)  # 半徑：0.2-0.5 米
                height_val = random.uniform(0.8, 1.5)  # 高度：0.8-1.5 米（高於 ray_caster 0.5m）
                
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,  # 圓柱體半徑
                    height=height_val,  # 圓柱體高度
                    
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    # 同樣設為運動學物體（固定不動）
                    
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    # 使用默認碰撞屬性
                    
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            random.random(),
                            random.random(),
                            random.random(),
                        ),
                        # 隨機顏色
                        metallic=0.2,
                    ),
                )
                height_i = height_val / 2  # 圓柱體質心也在中心
                size_scalar = radius * 2.0  # 以直徑作為尺寸近似
            
            # ----------------------------------------------------------------
            # 創建障礙物資產並添加到場景
            # ----------------------------------------------------------------
            setattr(
                self,  # 將障礙物添加到當前配置類（self）
                f"obstacle_{i}",  # 屬性名稱：obstacle_0, obstacle_1, ...
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    # 場景路徑：每個環境都有自己的障礙物
                    # /World/envs/env_0/Obstacle_0
                    # /World/envs/env_1/Obstacle_0 ...
                    
                    spawn=spawn_cfg,  # 使用上面定義的生成配置（立方體或圓柱體）
                    
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(
                            random.uniform(-6, 6),  # X 座標：-6 到 6 米
                            random.uniform(-6, 6),  # Y 座標：-6 到 6 米
                            height_i,  # Z 座標：質心高度（讓障礙物貼地）- 使用當前迭代的 height_i
                        )
                        # 隨機位置：在 12m × 12m 的範圍內
                        # 每次重置環境時，障礙物會重新隨機擺放
                    ),
                )
            )
            
            # 記錄障礙物尺寸（提供觀測用）
            obstacle_sizes.append(size_scalar)
            # setattr(object, name, value) 等同於：self.obstacle_0 = AssetBaseCfg(...)

        # 記錄障礙物設定（避免加入 scene cfg 欄位，造成資產解析錯誤）
        set_obstacle_metadata(num_obstacles, obstacle_sizes)


# ============================================================================
# MDP 配置（Markov Decision Process）
# ============================================================================
# 強化學習的核心元件：觀測、動作、獎勵、終止條件
# ============================================================================

# ----------------------------------------------------------------------------
# 命令配置（Commands）
# ----------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """命令配置
    
    定義機器人接收的「任務指令」（這裡是目標位置）。
    在導航任務中，命令就是「去哪裡」。
    """
    goal_command = GoalCommandCfg(
        asset_name="robot",  # 關聯的資產：機器人
        # 命令生成器需要知道機器人的位置來生成相對目標
        
        resampling_time_range=(1e10, 1e10),  # 重新採樣時間範圍：極大值（幾乎不重採樣）
        # (最小時間, 最大時間)：何時生成新目標
        # 1e10 秒 ≈ 317 年（實際上只在環境重置時才生成新目標）
        # 如果設為 (10, 20)，則每 10-20 秒會生成新目標（移動目標）
        
        debug_vis=True,  # 調試可視化：顯示綠色箭頭指示目標位置
        # True = 在模擬器中顯示綠色箭頭標記目標位置（方便調試和觀察）
        # False = 不顯示（可以提升性能）
        
        ranges=GoalCommandCfg.Ranges(
            distance=(MIN_ROBOT_DISTANCE, 3.0),  # 目標距離範圍：1.5-3.0 米
            # 目標會生成在距離機器人 MIN_ROBOT_DISTANCE-3.0 米的位置
            # 太近（<1米）：任務太簡單
            # 太遠（>5米）：任務太難，可能被障礙物擋住
            
            angle=(-math.pi, math.pi),  # 目標角度範圍：-180° 到 +180°（全方位）
            # math.pi ≈ 3.14159（180度）
            # (-π, π) 表示目標可以在機器人的任何方向
            # (0, π/2) 則只會在右前方 90 度扇形區域
        ),
    )

# ----------------------------------------------------------------------------
# 動作配置（Actions）
# ----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """動作配置
    
    定義機器人能執行的動作類型和範圍。
    """
    
    diff_drive = DifferentialDriveActionCfg(
        # 差速驅動動作：控制前進速度和旋轉速度
        
        asset_name="robot",  # 控制的資產：機器人
        
        # ====================================================================
        # 速度限制
        # ====================================================================
        max_linear_velocity=1.5,  # 最大前進速度：1.5 m/s
        # 神經網絡輸出 [-1, 1] 會被縮放到 [-1.5, 1.5] m/s
        # 1.5 m/s ≈ 5.4 km/h（慢跑速度，適合室內導航）
        # 輸入 1.0 → 目標前進速度 1.5 m/s（實際速度受加速度限制）
        # 輸入 -0.5 → 目標後退速度 0.75 m/s
        
        max_angular_velocity=1.5,  # 最大旋轉速度：1.5 rad/s
        # 1.5 rad/s ≈ 86 度/秒（約 4 秒轉一圈）
        # 輸入 1.0 → 目標右轉速度 1.5 rad/s（實際速度受加速度限制）
        # 輸入 -1.0 → 目標左轉速度 1.5 rad/s
        
        # ====================================================================
        # 加速度限制（新增）
        # ====================================================================
        max_linear_acceleration=2.0,  # 最大線性加速度：2.0 m/s²
        # 限制速度變化率，確保平滑運動
        # 2.0 m/s² 表示每秒最多改變 2.0 m/s 的速度
        # 控制頻率 25 Hz（0.04 秒/步）→ 每步最多改變 0.08 m/s
        # 例如：從 0 m/s 加速到 1.5 m/s 需要約 0.75 秒（19 步）
        # 
        # 建議值：
        #   - 2.0 m/s²：中等加速度（適合大多數情況）
        #   - 1.0 m/s²：較低加速度（更平滑，但響應較慢）
        #   - 3.0 m/s²：較高加速度（響應快，但可能不夠平滑）
        #   - None：無限制（直接達到目標速度，可能導致急轉急停）
        
        max_angular_acceleration=3.0,  # 最大角加速度：3.0 rad/s²
        # 限制角速度變化率，確保平滑旋轉
        # 3.0 rad/s² 表示每秒最多改變 3.0 rad/s 的角速度
        # 控制頻率 25 Hz（0.04 秒/步）→ 每步最多改變 0.12 rad/s
        # 例如：從 0 rad/s 加速到 1.5 rad/s 需要約 0.5 秒（13 步）
        # 
        # 建議值：
        #   - 3.0 rad/s²：中等角加速度（適合大多數情況）
        #   - 2.0 rad/s²：較低角加速度（更平滑的旋轉）
        #   - 5.0 rad/s²：較高角加速度（快速響應）
        #   - None：無限制（直接達到目標角速度）
        
        debug_vis=True,  # 啟用調試可視化：顯示速度箭頭
        # True = 顯示綠色箭頭（目標速度命令）和藍色箭頭（當前實際速度）
        # False = 不顯示（可以提升性能）
    )


# ----------------------------------------------------------------------------
# 觀測配置（Observations）
# ----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """觀測配置
    
    定義機器人在每一步能「感知」到什麼信息。
    這些信息會被餵給神經網絡，用於決策。
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組
        
        PolicyCfg 是給神經網絡（策略網絡）的觀測。
        所有觀測項會被串接成一個向量。
        """
        # ----------------------------------------------------------------
        # 觀測項 1：2D 平面掃描（72 個角度）
        # ----------------------------------------------------------------
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,  # 觀測函數：處理 2D 掃描數據
            params={"sensor_cfg": SceneEntityCfg("lidar")},  # 參數：引用場景中的雷達
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 噪音：±0.05（模擬真實傳感器）
        )
        # 輸出：[72] 個歸一化距離值（0-1）
        # 0 = 非常近（0米），1 = 最遠（10米）
        # 每個值對應一個 5 度扇形區域的最近障礙物距離
        # 由 3D 點雲資料轉換為鳥瞰圖的可行駛區域後計算得出
        # 從 36 個角度增加到 72 個角度（每 5 度一條射線），提高掃描精度
        
        # ----------------------------------------------------------------
        # 觀測項 2：速度資訊（機器人座標系）
        # ----------------------------------------------------------------
        speed = ObsTerm(
            func=base_velocity_xy,  # 觀測函數：機器人座標系速度
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 訓練噪音：±0.1 m/s
        )
        # 輸出：[2] 個值（vx, vy）
        # 例如：[0.8, 0.1] 表示前進 0.8 m/s、橫向 0.1 m/s
        
        # ----------------------------------------------------------------
        # 觀測項 3：目標相對位置
        # ----------------------------------------------------------------
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,  # 觀測函數：計算目標相對位置
            params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：引用機器人
        )
        # 輸出：[2] 個值（X, Y）在機器人坐標系
        # 例如：[2.5, -1.0] 表示目標在「右前方 2.5 米，左邊 1 米」
        
        # ----------------------------------------------------------------
        # 觀測項 4：目標距離
        # ----------------------------------------------------------------
        goal_distance = ObsTerm(
            func=goal_distance,  # 觀測函數：計算到目標的直線距離
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 輸出：[1] 個值（距離，單位：米）
        # 例如：2.69 表示距離目標 2.69 米
        
        # ----------------------------------------------------------------
        # 觀測項 5：時間剩餘比例
        # ----------------------------------------------------------------
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )
        # 輸出：[1]（1 = 剛開始，0 = 即將超時）
        
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
        # 輸出：[1]（1 = 存活，0 = 因碰撞/翻倒死亡）
        
        # ----------------------------------------------------------------
        # 觀測項 7：動態障礙物資訊（固定 MAX_OBSTACLES 個 × 5 維 = 50 維）
        # ----------------------------------------------------------------
        # 架構修復說明（2024-01）：
        # - 核心問題：觀測維度必須固定，否則無法在不同階段之間遷移權重
        # - Phase 1: 3 個障礙物 → 觀測 96 維（無法遷移到 Phase 2）
        # - Phase 2: 10 個障礙物 → 觀測 131 維（輸入層大小不匹配）
        # - 解決方案：固定使用 MAX_OBSTACLES = 10，不存在的障礙物用 padding 填充
        # - Padding 策略：size = -1.0 標記「不存在」（避免誤解為「原點有障礙物」）
        obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 3,  # 實際障礙物數量（Phase 1: 3, Phase 2: 10）
                "max_obstacles": MAX_OBSTACLES,  # 固定最大數量（用於 padding）
                "max_distance": 10.0,
            },
        )
        # 每個障礙物：[x, y, dir, v, size]
        # x,y = 相對位置；dir = 方向；v = 速度；size = 尺寸
        # 存在的障礙物：正常值
        # 不存在的障礙物（padding）：[0, 0, 0, 0, -1]（size = -1 標記「不存在」）
        # 觀測維度：固定 50 維（MAX_OBSTACLES × 5）
        
        # ----------------------------------------------------------------
        # 觀測項 8：上一步動作
        # ----------------------------------------------------------------
        actions = ObsTerm(func=safe_last_action)
        # 觀測上一步的動作（記憶）
        # 輸出：[2] 個值（上一步的前進速度、旋轉速度）
        # 這有助於動作平滑（避免突然急轉）
        
        # ----------------------------------------------------------------
        # 觀測項 9：Charge 在重置後一定時間步內撞到障礙物的機率（診斷觀測）
        # ----------------------------------------------------------------
        charge_dies_at_birth = ObsTerm(
            func=charge_dies_at_birth_probability,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),  # 使用雷達傳感器檢測碰撞
                "threshold": 0.3,  # 碰撞閾值：0.3 米（與 collision_occurred 保持一致）
                "birth_window_steps": 10,  # 出生窗口期：重置後的前 10 步內碰撞才算「出生即死」
            },
        )
        # 診斷觀測：監控機器人在重置後的「出生窗口期」內是否會與障礙物發生碰撞
        # 輸出：[1] 個值（0.0 = 安全，1.0 = 在出生窗口期內碰撞）
        # 設計理念：
        #   - 只在重置後的特定時間窗口內（例如前 10 步）檢測碰撞
        #   - 如果在這個時間窗口內發生碰撞，則標記為「出生即死」（1.0）
        #   - 一旦標記為 1.0，在該 episode 的剩餘時間內保持為 1.0
        # 用途：
        #   - 識別障礙物生成邏輯是否有問題
        #   - 檢查機器人起始位置是否合理
        #   - 驗證訓練環境配置是否正確
        # 注意：這是診斷性觀測，通常不應包含在策略輸入中
        # 如果需要記錄此值，可以通過 env.extras["log"] 記錄
        
        def __post_init__(self):
            """觀測組初始化配置"""
            self.enable_corruption = True  # 啟用數據損壞（加噪音）
            # True = 訓練時加噪音（提高魯棒性）
            # False = 完全乾淨的數據（可能過擬合）
            
            self.concatenate_terms = True  # 串接所有觀測項
            # True = 將所有觀測拼成一個長向量：
            # [72 + 2 + 2 + 1 + 1 + 1 + 50 + 2] = [131]
            # 觀測維度固定：131 維（支持權重遷移）
            # - lidar_scan: 72 維
            # - speed: 2 維
            # - goal_position: 2 維
            # - goal_distance: 1 維
            # - time_remaining: 1 維
            # - alive: 1 維
            # - obstacles: 50 維（MAX_OBSTACLES × 5，固定維度）
            # - actions: 2 維
            # False = 分開處理（需要多頭神經網絡）

    policy: PolicyCfg = PolicyCfg()  # 實例化策略觀測組


# ----------------------------------------------------------------------------
# 獎勵配置（Rewards）
# ----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """獎勵配置
    
    定義什麼行為會得分/扣分（強化學習的核心！）。
    機器人通過最大化總獎勵來學習正確的行為。
    """
    
    # ------------------------------------------------------------------------
    # 獎勵項 1：朝目標移動速度（已移除 forward_velocity，改用 velocity_toward_goal）
    # ------------------------------------------------------------------------
    # 移除 forward_velocity 的原因：
    # - forward_velocity 獎勵機器人沿自身 X 軸移動（不管目標在哪）
    # - 這會與 heading_to_goal 產生衝突（當目標不在前方時）
    # - 改用 velocity_toward_goal 直接獎勵「朝目標方向的速度」，更符合任務目標
    
    # ------------------------------------------------------------------------
    # 獎勵項 1.1：漸進式速度獎勵（替代 forward_velocity）【暫時關閉】
    # ------------------------------------------------------------------------
    # 使用漸進式速度獎勵，讓機器人在接近目標時自動減速
    # 這比簡單的「0/1 開關」更平滑，避免衝撞目標
    # velocity_toward_goal = RewTerm(
    #     func=charge_mdp.velocity_toward_goal_smooth,  # 漸進式速度獎勵函數
    #     weight=0.3,  # 權重：0.3（可以稍微提高，因為獎勵範圍是 [0, 1]）
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "slow_distance": 1.0,    # 開始減速的距離：1.0 米
    #         "stop_distance": GOAL_REACH_THRESHOLD,   # 停止獎勵的距離（對齊目標大小）
    #         "max_reward_speed": 1.5, # 遠距離目標速度：1.5 m/s（快速接近）
    #         "min_reward_speed": 0.3, # 近距離目標速度：0.3 m/s（低速接近）
    #     },
    # )
    # 獎勵機制：
    # - 距離 > 1.0m：獎勵高速（1.5 m/s），速度越接近 1.5 m/s 獎勵越高
    # - GOAL_REACH_THRESHOLD < 距離 < 1.0m：線性插值期望速度（0.3-1.5 m/s），自動減速
    # - 距離 < GOAL_REACH_THRESHOLD：不給獎勵（避免衝撞目標）
    # 
    # 獎勵計算：
    # - 使用指數衰減：速度越接近「期望速度」，獎勵越高
    # - 完美匹配期望速度 → 獎勵 1.0 × 0.3 = +0.3 分/步
    # - 偏離期望速度 → 獎勵指數衰減
    # 
    # 效果：
    #   距離 = 3.0m → 期望速度 1.5 m/s → 以 1.5 m/s 前進得最高獎勵
    #   距離 = 1.0m → 期望速度 0.3 m/s → 開始自動減速
    #   距離 = 0.5m → 期望速度 0.15 m/s → 繼續減速
    #   距離 = GOAL_REACH_THRESHOLD → 期望速度 0 m/s → 停止獎勵
    
    # ------------------------------------------------------------------------
    # 獎勵項 1.2：移動獎勵 (R_move)：鼓勵車輛保持移動，避免因恐懼碰撞而停滯
    # ------------------------------------------------------------------------
    move = RewTerm(
        func=move_reward,  # 獎勵函數：移動獎勵
        weight=0.2,  # 權重：0.2（鼓勵移動但不過度）
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "k": 1.0,  # 係數：1.0
            "b": 0.0,  # 偏移量：0.0（可設為負值以鼓勵更高速度）
        },
    )
    # 獎勵計算：k × (speed_x + b) × weight
    # 公式：1.0 × (speed_x + 0.0) × 0.2 = 0.2 × speed_x
    # 效果：
    #   - speed_x = 0.0 m/s → 0 分（靜止，無獎勵）
    #   - speed_x = 0.5 m/s → +0.1 分/步
    #   - speed_x = 1.0 m/s → +0.2 分/步
    #   - speed_x = 1.5 m/s → +0.3 分/步
    # 目的：鼓勵車輛保持移動，避免因恐懼碰撞而停滯
    
    # ------------------------------------------------------------------------
    # 獎勵項 1.5：朝向目標的朝向獎勵（新增）
    # ------------------------------------------------------------------------
    heading_to_goal = RewTerm(
        func=heading_to_goal_distance_weighted,  # 獎勵函數：根據距離動態調整
        weight=0.1,  # 權重：0.1（小幅行為約束）
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "close_distance": 2.0,  # 接近距離閾值：< 2米時朝向獎勵權重最大
            "far_distance": 5.0,  # 遠距離閾值：> 5米時朝向獎勵權重最小
        },
    )
    # 獎勵計算：cos(朝向角度) × 動態權重 × 2.0
    # 接近目標時（< 2米）：朝向獎勵權重 100%（強烈獎勵朝向目標）
    # 遠離目標時（> 5米）：朝向獎勵權重 20%（速度更重要）
    # 中間距離：線性插值
    # 正對目標（0°）→ +2.0 分/步（接近時）或 +0.4 分/步（遠離時）
    # 側對目標（90°）→ 0 分
    
    # ------------------------------------------------------------------------
    # 獎勵項 1.6：對齊獎勵 (Alignment)：車頭朝向目標的獎勵（改進版）
    # ------------------------------------------------------------------------
    alignment = RewTerm(
        func=alignment_reward,  # 獎勵函數：對齊獎勵
        weight=0.15,  # 權重：0.15（中等重要性）
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "misalignment_penalty_scale": 0.5,  # 未對準時的懲罰係數
            "high_speed_threshold": 1.0,  # 高速閾值（m/s），超過此速度且未對準時給予懲罰
        },
    )
    # 獎勵計算：
    # 1. 基礎對齊獎勵：cos(朝向角度) × weight
    #    - 正對目標（0°）→ +1.0 × 0.15 = +0.15 分/步
    #    - 側對目標（90°）→ 0.0 × 0.15 = 0 分
    #    - 背對目標（180°）→ 0.0（不懲罰，因為速度獎勵會處理）
    # 
    # 2. 未對準懲罰（防止衝過頭）：
    #    - 條件：未對準（> 72°）且高速前進（speed_x > 1.0 m/s）
    #    - 懲罰：未對準程度 × 速度超出閾值 × 0.5 × weight
    #    - 例如：90° 未對準 + 1.5 m/s 速度 → -0.075 分/步
    # 
    # 設計理念：
    # - 由於車輛不能側移，「車頭朝向目標」變得至關重要
    # - 在 ROS 系統測試中，為了讓 Agent 準確抵達，加入了「比例控制器」概念
    # - 在 RL 獎勵中，加入對齊獎勵可以引導 Agent 先對準方向再加速
    # - 防止 Agent「衝過頭」：高速前進但方向不對
    
    # ------------------------------------------------------------------------
    # 獎勵項 1.6：倒車靠近目標（已取消）
    # ------------------------------------------------------------------------
    # reverse_toward_goal = RewTerm(
    #     func=charge_mdp.reverse_toward_goal_distance_weighted,
    #     weight=0.6,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "min_dist": 0.28,
    #         "close_distance": 2.0,
    #         "far_distance": 5.0,
    #     },
    # )
    
    # ------------------------------------------------------------------------
    # 獎勵項 2：接近目標的進度（輔助獎勵）
    # ------------------------------------------------------------------------
    distance_to_goal = RewTerm(
        func=progress_to_goal,  # 獎勵函數：進度獎勵
        weight=2.0,  # 權重：2.0（主導 shaping）
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 獎勵計算：(上一步距離 - 這一步距離) × 3.0（原值：× 2.0）
    # 每接近目標 1 米 → +3.0 分（原值：+2.0 分）
    # 遠離目標 1 米 → -3.0 分（原值：-2.0 分）
    # 這避免了機器人「在目標附近繞圈」來刷速度獎勵
    
    # ------------------------------------------------------------------------
    # 獎勵項 3：到達目標（最重要！）
    # ------------------------------------------------------------------------
    # 修復說明：
    # - 原問題：權重 100 太小，無法對抗累積的碰撞懲罰（-20 ~ -60）
    # - 修復：提高到 200，讓成功獎勵更有吸引力
    # - 效果：agent 會更積極地嘗試到達目標，而不是「原地轉圈等超時」
    reaching_goal = RewTerm(
        func=reaching_goal,  # 獎勵函數：檢查是否到達
        weight=200,  # 權重：200（提高權重，增強成功獎勵的吸引力）
        # 原值：100 → 新值：200（提高 100%）
        # 這樣可以對抗累積的碰撞懲罰（-5 ~ -15），讓 agent 更積極地嘗試到達目標
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,  # 使用身體半徑做補償（以「身體到目標」判定）
        },
        # threshold=GOAL_REACH_THRESHOLD：距離目標 < GOAL_REACH_THRESHOLD 米算成功（對齊 Goal 大小）
        # 注意：到達判定只看距離，不要求特定朝向
    )
    # 獎勵計算：到達目標 → +200 分（一次性）
    # 這是任務的最終目標，必須給大獎勵
    # 修復後：+200 分可以對抗累積懲罰（-5 ~ -15），讓 agent 更積極地嘗試到達目標
    
    # ------------------------------------------------------------------------
    # 獎勵項 3.5：接近目標獎勵（修復：大幅降低權重）
    # ------------------------------------------------------------------------
    # 修復說明（2024-01 修正）：
    # - 原問題：權重 5.0 太大，累積獎勵（5000 分/episode）遠超過 reaching_goal（200 分）
    # - 這導致策略學會「持續接近但不真的到達」比「實際到達」更划算
    # - 修復：權重從 5.0 降到 0.5，並縮小觸發距離到 1.0 米
    # - 效果：只在最後衝刺階段才給獎勵，不會干擾長距離導航
    approaching_goal = RewTerm(
        func=approaching_goal_bonus,  # 獎勵函數：接近目標獎勵
        weight=0.5,  # 權重：0.5（大幅降低，從 5.0 → 0.5，降低 90%）
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "close_threshold": 1.0,  # 接近閾值：1.0 米（縮小範圍，只在最後衝刺才觸發）
            "decay_scale": 0.5,  # 衰減尺度：0.5（較平緩的衰減曲線）
        },
    )
    # 修復後獎勵計算：
    # scale = close_threshold * decay_scale = 1.0 * 0.5 = 0.5
    # 距離 >= 1.0m → reward = 0（不觸發）
    # 距離 = 0.5m → reward = exp(-0.5/0.5) ≈ 0.37 × 0.5 = +0.18 分/步
    # 距離 = 0.0m → reward = exp(0) = 1.0 × 0.5 = +0.5 分/步
    # 
    # 修復後效果：
    # - 累積獎勵大幅降低（從 5000 分降到約 50 分）
    # - reaching_goal (+200) 重新成為主導獎勵
    # - 策略會更積極地「真正到達」目標
    
    # ------------------------------------------------------------------------
    # 懲罰項 1：靠近障礙物的軟懲罰（只保留一種 near-collision shaping）
    # ------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------
    # 懲罰項 1.5：漸進式碰撞懲罰（修復：降低權重和擴大安全距離）
    # ------------------------------------------------------------------------
    # 修復說明：
    # - 原問題：權重 -2.0 太大，累積懲罰（-20 ~ -60）壓垮了成功獎勵（+100）
    # - 原問題：safe_distance 1.0 太小，10 個障礙物在 12×12 場景很難保持 1 米
    # - 修復：降低權重到 -0.5，擴大安全距離到 1.5 米
    # - 效果：讓 agent 學到「保持距離但不必恐慌」，而不是「遠離一切」
    progressive_collision = RewTerm(
        func=progressive_collision_penalty,  # 懲罰函數：漸進式碰撞懲罰（改進版）
        weight=-0.5,  # 權重：-0.5（降低權重，避免累積懲罰過大）
        # 原值：-2.0 → 新值：-0.5（降低 75%）
        # 這樣累積懲罰會從 -20 ~ -60 降低到 -5 ~ -15，不會壓垮成功獎勵
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "asset_cfg": SceneEntityCfg("robot"),  # 新增：用於計算速度方向
            "safe_distance": SAFE_DISTANCE,   # 安全距離：使用常數 SAFE_DISTANCE
            "danger_distance": DANGER_DISTANCE, # 危險距離：使用常數 DANGER_DISTANCE
            "collision_distance": COLLISION_DISTANCE,  # 碰撞距離：使用常數 COLLISION_DISTANCE
            "use_directional_penalty": True,  # 啟用方向性懲罰：只有「朝障礙物前進」才重罰
            "use_nonlinear_penalty": True,    # 啟用非線性懲罰：危險區域使用平方/反比函數
        },
    )
    # 改進後的懲罰計算（三項核心改進）：
    # 
    # 1. 非線性懲罰（建議三）：
    #    - 危險區域（0.8m > 距離 >= 0.4m）：使用平方函數，懲罰曲線更陡峭
    #    - 碰撞區域（距離 < 0.4m）：使用反比函數，距離越近懲罰急遽增加
    #    - 讓 0.6m 以內「非常不舒服」，直接瓦解「一頭撞上去」策略
    # 
    # 2. 方向性懲罰（建議二）：
    #    - 只有「朝障礙物前進」才重罰
    #    - 距離近 + 朝向障礙物的速度大 → 懲罰急遽增加（平方函數，最大 3.0 倍）
    #    - 距離近但側向移動 → 懲罰變小（接近基礎懲罰）
    #    - 直接瓦解「一頭撞上去」策略
    # 
    # 3. 基礎距離懲罰：
    #    距離 >= 1.5m：0 分（安全，無懲罰）
    #    1.5m > 距離 >= 0.8m：-0 到 -0.25 分（輕微懲罰，線性）
    #    0.8m > 距離 >= 0.4m：-0.25 到 -0.5 分（中等懲罰，平方）
    #    距離 < 0.4m：-0.5 到 -2.5 分（最大懲罰，反比 + 方向性）
    # 
    # 預期效果：
    # - 朝障礙物高速前進時，懲罰會急遽增加（方向性 × 非線性）
    # - 側向移動時，懲罰較小，允許更靈活的避障
    # - 在危險區域「心理崩潰」，避免「一頭撞上去」
    
    # ------------------------------------------------------------------------
    # 懲罰項 1.7：真正碰撞的一次性大懲罰（與終止同步）
    # ------------------------------------------------------------------------
    hard_collision = RewTerm(
        func=collision_occurred,  # 使用雷達距離判定
        weight=-100.0,  # 權重：-100（硬碰撞懲罰）
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
    )
    # 距離 < COLLISION_THRESHOLD 當步給一次性大懲罰，並與 DoneTerm 同步終止
    
    # ------------------------------------------------------------------------
    # 懲罰項 2：動作抖動懲罰
    # ------------------------------------------------------------------------
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,  # 懲罰函數：動作變化率的 L2 範數
        weight=-0.05  # 權重：-0.05（輕微懲罰）
    )
    # 懲罰計算：|| 這步動作 - 上步動作 ||² × -0.05
    # 動作變化越大，懲罰越大
    # 這鼓勵平滑運動（避免急轉急停）
    
    # ------------------------------------------------------------------------
    # 懲罰項 2.5：翻倒懲罰（新增 - 防止激進動作導致翻倒）
    # ------------------------------------------------------------------------
    # 修復說明（2024-01 新增）：
    # - 原問題：tipped_over 只是終止條件，沒有懲罰
    # - 策略可能學會激進動作來追求其他獎勵，導致翻倒
    # - 修復：加入翻倒懲罰，讓策略避免不穩定的動作
    tipped_over_penalty = RewTerm(
        func=robot_tipped_over,  # 使用相同的翻倒判定函數
        weight=-200.0,  # 權重：-200（翻倒懲罰，大幅增加權重）
        # 原值：-50.0 → 新值：-200.0（增加 300%）
        # 強制策略學會避免不穩定的動作，翻倒 = 嚴重失敗
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 懲罰計算：翻倒時 → -50 分（一次性）
    # 這讓策略學會避免激進動作，保持物理穩定性

    # ------------------------------------------------------------------------
    # 懲罰項 3：超時懲罰（修復：降低權重）
    # ------------------------------------------------------------------------
    # 修復說明：
    # - 原問題：權重 -10 太大，加劇了 agent 的「恐懼」（不敢靠近任何東西）
    # - 修復：降低到 -5，減少對 agent 的負面影響
    # - 效果：agent 不會因為害怕超時而「原地轉圈」，會更積極地嘗試到達目標
    time_out = RewTerm(
        func=time_out_penalty,  # 懲罰函數：超時檢查
        weight=-5.0,  # 權重：-5.0（降低權重，減少負面影響）
        # 原值：-10.0 → 新值：-5.0（降低 50%）
        # 這樣不會加劇 agent 的「恐懼」，讓它更積極地嘗試到達目標
    )
    # 懲罰計算：超過 45 秒還沒完成 → -5 分（一次性）
    # 原值：-10 分 → 新值：-5 分
    # 降低超時懲罰，鼓勵機器人積極嘗試而不是「原地轉圈等超時」


# ----------------------------------------------------------------------------
# 終止條件配置（Terminations）
# ----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """終止條件配置
    
    定義什麼情況下「遊戲結束」，環境會被重置。
    """
    
    # ------------------------------------------------------------------------
    # 終止條件 1：超時
    # ------------------------------------------------------------------------
    time_out = DoneTerm(
        func=mdp.time_out,  # 終止函數：檢查是否超時
        time_out=True  # 標記為「超時」類型的終止
    )
    # 判斷：episode 時長 > 45 秒 → 終止（原值：30 秒）
    # 45 秒在 __post_init__ 中設定（episode_length_s=45.0）
    
    # ------------------------------------------------------------------------
    # 終止條件 2：到達目標（成功！）
    # ------------------------------------------------------------------------
    goal_reached = DoneTerm(
        func=goal_reached,  # 終止函數：檢查是否到達目標
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
            "body_radius": ROBOT_BODY_RADIUS,  # 使用身體半徑做補償（以「身體到目標」判定）
        },
        # threshold=GOAL_REACH_THRESHOLD：距離 < GOAL_REACH_THRESHOLD 米算成功（對齊 Goal 大小）
        # 注意：到達判定只看距離，不要求特定朝向
    )
    
    # ------------------------------------------------------------------------
    # 終止條件 3：碰撞（失敗 - 絕對不能撞到！）
    # ------------------------------------------------------------------------
    # 同時使用兩個判定方式，更嚴格地檢測碰撞（任一條件觸發都會終止）
    
    # 判定方式 A：接觸感測器（真實碰撞框接觸）
    # 使用物理引擎的真實碰撞檢測，檢測機器人任何部件與障礙物的碰撞
    collision_contact = DoneTerm(
        func=collision_contact_occurred,  # 終止函數：接觸感測器判定
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
        # 優勢：使用真實的碰撞框，比雷達距離更準確
        # 可以檢測機器人任何部件（底盤、輪子等）與障礙物的碰撞
    )
    
    # 判定方式 B：雷達距離判定（2D 平面距離）
    # 使用雷達掃描的最近距離，作為接觸感測器的補充
    collision = DoneTerm(
        func=collision_occurred,  # 終止函數：雷達距離判定
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        # 以雷達距離判定碰撞（使用 2D 平面距離）
        # 優勢：可以提前檢測到接近的障礙物（即使還沒真正接觸）
        # 補充：接觸感測器可能在某些邊緣情況下漏檢，雷達可以提供額外保護
    )
    # 注意：兩個終止條件是「或」關係，任一觸發都會終止
    # 這樣可以確保碰撞檢測的可靠性
    
    # ------------------------------------------------------------------------
    # 終止條件 4：機器人翻倒
    # ------------------------------------------------------------------------
    tipped_over = DoneTerm(
        func=robot_tipped_over,  # 終止函數：檢查翻倒
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 判斷：機器人的 Z 軸傾斜 > 某個角度 → 終止
    # 防止機器人倒下還在地上「游泳」
    
    # ------------------------------------------------------------------------
    # 終止條件 5：機器人飛起來（異常）
    # ------------------------------------------------------------------------
    flying = DoneTerm(
        func=robot_flying,  # 終止函數：檢查飛行
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 判斷：機器人高度 > 1 米 → 終止
    # 防止物理引擎出錯導致機器人「升天」


# ----------------------------------------------------------------------------
# 事件配置（Events）
# ----------------------------------------------------------------------------
@configclass
class EventCfg:
    """事件配置
    
    定義在特定時機（如環境重置）觸發的事件。
    """
    
    # ------------------------------------------------------------------------
    # 事件 1：重置機器人位置（固定每個環境編號的位置）
    # ------------------------------------------------------------------------
    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,  # 事件函數：固定每個環境編號的位置
        mode="reset",  # 觸發模式："reset"（環境重置時）
        # 其他模式："startup"（首次啟動）、"interval"（定期）
        
        params={
            "pose_range": {
                "x": (-4.0, 4.0),  # X 座標範圍：-4 到 4 米
                "y": (-4.0, 4.0),  # Y 座標範圍：-4 到 4 米
                "yaw": (-3.14, 3.14)  # 偏航角範圍：-180° 到 +180°（任意朝向）
            },
            # 機器人會在 8m × 8m 的區域內隨機出生
            # 注意：每個環境編號的位置在訓練過程中保持固定
            # 不同環境編號之間位置不同，提升訓練效果和穩定性
            
            "velocity_range": {
                "x": (0.0, 0.0),  # X 方向初始速度：0（靜止）
                "y": (0.0, 0.0),  # Y 方向初始速度：0
                "z": (0.0, 0.0),  # Z 方向初始速度：0
            },
            # 每次重置時機器人都是靜止的
            # 如果設為 (-0.5, 0.5)，則會有初始速度（更難）
        },
    )
    
    # ------------------------------------------------------------------------
    # 事件 2：重置障礙物位置（Phase 1：完全靜態）
    # ------------------------------------------------------------------------
    reset_obstacles = EventTerm(
        func=reset_obstacles,  # 事件函數：重置所有障礙物
        mode="reset",  # 觸發模式：只在環境重置時觸發
        params={
            "min_robot_distance": MIN_ROBOT_DISTANCE,
            "min_goal_distance": MIN_GOAL_DISTANCE,
            "min_obstacle_spacing": MIN_OBSTACLE_SPACING,
        },
        # Phase 1 設計：障礙物完全靜態，只在每個 episode 開始時隨機擺放
        # 這樣可以：
        # 1. 避免「瞬移」問題（不會在 episode 中間突然移動）
        # 2. 讓機器人專注學習「抵達目的地」而非「追蹤移動障礙物」
        # 3. 觀測中的速度資訊會正確設為 0（與物理引擎一致）
        # 4. 碰撞檢查確保障礙物不會與機器人、目標重疊
        # 
        # Phase 2 升級：如需動態障礙物，可以：
        # 1. 將 rigid_props.kinematic_enabled 設為 False
        # 2. 添加 move_obstacles 事件（使用 interval 模式，每 0.1 秒更新）
    )


# ============================================================================
# 環境總配置（Environment Configuration）
# ============================================================================
# 將所有組件組合成完整的訓練環境
# ============================================================================

@configclass
class ChargeNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Charge 激光雷達導航環境
    
    這是最終的環境配置，組合了所有組件。
    ManagerBasedRLEnvCfg 是 Isaac Lab 的強化學習環境基類。
    """
    
    # ------------------------------------------------------------------------
    # 組件配置
    # ------------------------------------------------------------------------
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=15.0)
    # 場景配置：
    #   num_envs=128：同時運行 128 個並行環境（GPU 加速）
    #   env_spacing=15.0：每個環境間隔 15 米（避免互相干擾）
    #   間距設定可視需求調整：
    #   - 太小：不同環境的機器人可能互相干擾
    #   - 太大：場景總大小增加
    
    observations: ObservationsCfg = ObservationsCfg()  # 觀測配置
    actions: ActionsCfg = ActionsCfg()  # 動作配置
    commands: CommandsCfg = CommandsCfg()  # 命令配置
    rewards: RewardsCfg = RewardsCfg()  # 獎勵配置
    terminations: TerminationsCfg = TerminationsCfg()  # 終止條件配置
    events: EventCfg = EventCfg()  # 事件配置
    
    # ------------------------------------------------------------------------
    # 環境參數
    # ------------------------------------------------------------------------
    def __post_init__(self):
        """環境初始化後自動執行"""
        
        # ====================================================================
        # 控制頻率配置
        # ====================================================================
        self.decimation = 4  # 降採樣因子：4
        # 物理模擬以 100 Hz 運行（0.01秒/步）
        # 但控制頻率是 100/4 = 25 Hz（0.04秒/步）
        # 意思：物理模擬 4 步，神經網絡才決策 1 次
        # 為什麼？因為神經網絡決策很慢，不需要每步都算
        
        self.episode_length_s = 45.0  # Episode 最大時長：45 秒（原值：30.0）- 延長超時時間以減少超時終止率
        # 45 秒內沒完成任務 → 超時終止
        # 45 秒 ÷ 0.04 秒/步 = 1125 步（最多 1125 次決策）
        
        # ====================================================================
        # 模擬參數配置
        # ====================================================================
        self.sim.dt = 0.01  # 物理模擬時間步長：0.01 秒（100 Hz）
        # 越小越精確但越慢
        # 0.01 秒是標準值（平衡精度和速度）
        
        self.sim.render_interval = self.decimation  # 渲染間隔：4
        # 不需要每個物理步都渲染畫面
        # 每 4 步渲染一次 → 25 FPS（夠流暢了）
        
        self.sim.use_gpu_pipeline = True  # 使用 GPU 物理管線：開啟
        # True = 用 GPU 加速物理模擬（快很多！）
        # False = 用 CPU（慢，只適合單環境）
        
        self.sim.physx.use_gpu = True  # PhysX 使用 GPU：開啟
        # NVIDIA PhysX 物理引擎的 GPU 加速
        # 必須搭配 use_gpu_pipeline=True
        
        # ====================================================================
        # 傳感器更新頻率配置
        # ====================================================================
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt
            # 雷達更新週期 = 4 × 0.01 = 0.04 秒（25 Hz）
            # 和控制頻率同步（每次決策時都有新的雷達數據）


# ============================================================================
# 測試/展示配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfg_PLAY(ChargeNavigationEnvCfg):
    """測試/展示用的環境配置
    
    繼承自訓練配置，但調整了一些參數以便展示。
    """
    def __post_init__(self):
        super().__post_init__()  # 先執行訓練配置的初始化
        
        self.scene.num_envs = 9 # 並行環境數：16（訓練時是 128）
        # 展示時不需要太多環境（看不過來）
        # 16 個環境剛好可以排成 4×4 網格觀看
        
        self.observations.policy.enable_corruption = False  # 關閉觀測噪音
        # 展示時不加噪音（數據更乾淨，行為更穩定）
        # 訓練時是 True（加噪音提高魯棒性）
