"""
Carter 激光雷达导航环境配置 / Carter LiDAR Navigation Environment Configuration

此模組定義了 Carter 機器人使用雷射雷達進行導航任務的完整環境配置。
This module defines the complete environment configuration for Carter robot navigation using LiDAR.

主要組件包括：
Main components include:
- 場景配置（地形、機器人、感測器、障礙物）
  Scene configuration (terrain, robot, sensors, obstacles)
- MDP 設定（動作、觀測、獎勵、終止條件、事件）
  MDP settings (actions, observations, rewards, terminations, events)
- 環境參數（時間步長、回合長度、並行環境數等）
  Environment parameters (time step, episode length, number of parallel environments, etc.)
"""

import math  # 數學運算模組 / Math module for mathematical operations
import isaaclab.sim as sim_utils  # Isaac Lab 模擬工具 / Isaac Lab simulation utilities
from isaaclab.assets import AssetBaseCfg  # 資產基礎配置類別 / Base asset configuration class
from isaaclab.envs import ManagerBasedRLEnvCfg  # 基於管理器的強化學習環境配置 / Manager-based RL environment configuration
from isaaclab.managers import (
    SceneEntityCfg,  # 場景實體配置 / Scene entity configuration
    ObservationGroupCfg as ObsGroup,  # 觀測群組配置 / Observation group configuration
    ObservationTermCfg as ObsTerm,  # 觀測項配置 / Observation term configuration
    RewardTermCfg as RewTerm,  # 獎勵項配置 / Reward term configuration
    TerminationTermCfg as DoneTerm,  # 終止條件配置 / Termination condition configuration
    EventTermCfg as EventTerm,  # 事件項配置 / Event term configuration
)
from isaaclab.scene import InteractiveSceneCfg  # 互動場景配置 / Interactive scene configuration
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns  # 射線投射感測器配置 / Ray caster sensor configurations
from isaaclab.terrains import TerrainImporterCfg  # 地形匯入配置 / Terrain importer configuration
from isaaclab.utils import configclass  # 配置類別裝飾器 / Configuration class decorator
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 加性均勻雜訊配置 / Additive uniform noise configuration
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # 速度導航 MDP 模組 / Velocity navigation MDP module

from .carter_cfg import CARTER_CFG  # Carter 機器人配置 / Carter robot configuration
from . import carter_mdp  # Carter MDP 函數模組（動作、觀測、獎勵、終止條件等） / Carter MDP functions module (actions, observations, rewards, terminations, etc.)
from .goal_command import GoalCommandCfg  # 目標命令配置 / Goal command configuration

##
# Scene definition / 場景定義
##
# 此區塊定義模擬場景中的所有實體：地形、機器人、感測器、障礙物等
# This section defines all entities in the simulation scene: terrain, robot, sensors, obstacles, etc.


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """
    場景配置 / Scene Configuration

    定義模擬環境中的所有實體，包括：
    Defines all entities in the simulation environment, including:
    - 地形（平坦地面）
      Terrain (flat ground)
    - 機器人（Carter）
      Robot (Carter)
    - 感測器（雷射雷達）
      Sensors (LiDAR)
    - 障礙物（動態創建的立方體和圓柱體）
      Obstacles (dynamically created cubes and cylinders)
    - 光照（環境光）
      Lighting (ambient light)
    """

    # 平坦地面 / Flat ground terrain
    # 使用平面地形作為機器人的移動表面
    # Uses a plane terrain as the movement surface for the robot
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 場景圖中的路徑 / Path in the scene graph
        terrain_type="plane",  # 地形類型：平面 / Terrain type: plane
        collision_group=-1,  # 碰撞群組：-1 表示與所有物體碰撞 / Collision group: -1 means collides with all objects
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦係數組合模式：相乘 / Friction combine mode: multiply
            restitution_combine_mode="multiply",  # 恢復係數組合模式：相乘 / Restitution combine mode: multiply
            static_friction=1.0,  # 靜摩擦係數 / Static friction coefficient
            dynamic_friction=1.0,  # 動摩擦係數 / Dynamic friction coefficient
        ),
    )

    # Carter 機器人 / Carter robot
    # 使用預定義的 Carter 配置，並設置其在場景中的路徑
    # Uses the predefined Carter configuration and sets its path in the scene
    # {ENV_REGEX_NS} 會被替換為每個環境的命名空間（例如 /World/envs/env_0）
    # {ENV_REGEX_NS} will be replaced with each environment's namespace (e.g., /World/envs/env_0)
    robot = CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 激光雷达 / LiDAR sensor
    # 使用 MultiMeshRayCaster 以檢測多個物體（包括障礙物和地面）
    # Uses MultiMeshRayCaster to detect multiple objects (including obstacles and ground)
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",  # 感測器安裝在機器人底盤上 / Sensor mounted on robot chassis
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),  # 偏移量：向上 0.2 公尺 / Offset: 0.2m upward
        attach_yaw_only=True,  # 只跟隨機器人的偏航角（Z 軸旋轉），不跟隨俯仰和滾轉 / Only follows robot's yaw (Z-axis rotation), not pitch and roll
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # 通道數：1 層掃描 / Number of channels: 1 layer scan
            vertical_fov_range=(0.0, 0.0),  # 垂直視野範圍：0 度（水平掃描） / Vertical FOV range: 0 degrees (horizontal scan)
            horizontal_fov_range=(-180.0, 180.0),  # 水平視野範圍：360 度（完整圓周） / Horizontal FOV range: 360 degrees (full circle)
            horizontal_res=1.0,  # 水平解析度：1 度（每度一條射線） / Horizontal resolution: 1 degree (one ray per degree)
        ),
        max_distance=10.0,  # 最大檢測距離：10 公尺 / Maximum detection distance: 10 meters
        debug_vis=True,  # 啟用紅色點雲可視化（顯示射線碰撞點） / Enable red point cloud visualization (shows ray hit points)
        # 檢測環境中的所有物體：障礙物和地面 / Detect all objects in environment: obstacles and ground
        mesh_prim_paths=[
            # 檢測全局地面（靜態，不需要追蹤變換） / Detect global ground (static, no need to track transforms)
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",  # 地面路徑表達式 / Ground path expression
                track_mesh_transforms=False,  # 不追蹤變換（地面是靜態的） / Don't track transforms (ground is static)
            ),
            # 檢測環境中的所有障礙物（動態創建，需要追蹤變換） / Detect all obstacles in environment (dynamically created, need to track transforms)
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle_.*",  # 障礙物路徑正則表達式：匹配所有 Obstacle_0, Obstacle_1 等 / Obstacle path regex: matches all Obstacle_0, Obstacle_1, etc.
                track_mesh_transforms=True,  # 追蹤變換（障礙物可能被重置移動） / Track transforms (obstacles may be reset/moved)
            ),
        ],
        update_period=0.04,  # 更新週期：0.04 秒（25 Hz） / Update period: 0.04 seconds (25 Hz)
    )

    # 光照 / Lighting
    # 環境光（穹頂光），提供場景的基礎照明
    # Ambient light (dome light), provides basic illumination for the scene
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",  # 光照在場景圖中的路徑 / Light path in scene graph
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),  # 光照強度：1000.0 / Light intensity: 1000.0
    )

    def __post_init__(self):
        """
        動態創建多個障礙物 / Dynamically create multiple obstacles

        在場景初始化後，動態創建多個隨機障礙物（立方體或圓柱體）。
        After scene initialization, dynamically create multiple random obstacles (cubes or cylinders).

        這個方法在配置類別初始化後自動調用，用於：
        This method is automatically called after the configuration class initialization, used for:
        - 創建隨機形狀和大小的障礙物
          Creating obstacles with random shapes and sizes
        - 設置障礙物的隨機位置和顏色
          Setting random positions and colors for obstacles
        - 確保每個環境都有不同的障礙物配置
          Ensuring each environment has a different obstacle configuration
        """
        super().__post_init__()  # 調用父類的初始化方法 / Call parent class initialization
        import random  # 隨機數生成模組 / Random number generation module

        num_obstacles = 4  # 障礙物數量：每個環境 4 個 / Number of obstacles: 4 per environment

        for i in range(num_obstacles):
            # 隨機選擇類型：50% 機率為立方體，50% 機率為圓柱體
            # Randomly select type: 50% chance cube, 50% chance cylinder
            is_cube = random.random() > 0.5

            if is_cube:
                # 立方体 / Cube
                # 生成隨機尺寸（寬、深、高）
                # Generate random dimensions (width, depth, height)
                size = (
                    random.uniform(0.3, 0.8),  # X 方向尺寸：0.3-0.8 公尺 / X dimension: 0.3-0.8 meters
                    random.uniform(0.3, 0.8),  # Y 方向尺寸：0.3-0.8 公尺 / Y dimension: 0.3-0.8 meters
                    random.uniform(0.5, 1.5),  # Z 方向尺寸（高度）：0.5-1.5 公尺 / Z dimension (height): 0.5-1.5 meters
                )
                spawn_cfg = sim_utils.CuboidCfg(
                    size=size,  # 立方體尺寸 / Cube size
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 運動學模式：不受物理力影響，但可以移動 / Kinematic mode: not affected by physics forces, but can be moved
                    collision_props=sim_utils.CollisionPropertiesCfg(),  # 碰撞屬性：使用預設值 / Collision properties: use defaults
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(random.random(), random.random(), random.random()),  # 隨機顏色（RGB） / Random color (RGB)
                        metallic=0.2,  # 金屬度：0.2（稍微有金屬光澤） / Metallic: 0.2 (slight metallic sheen)
                    ),
                )
                height = size[2] / 2  # 高度的一半（用於設置位置，使底部貼地） / Half of height (for positioning, so bottom touches ground)
            else:
                # 圆柱体 / Cylinder
                # 生成隨機半徑和高度
                # Generate random radius and height
                radius = random.uniform(0.2, 0.5)  # 半徑：0.2-0.5 公尺 / Radius: 0.2-0.5 meters
                height_val = random.uniform(0.5, 1.5)  # 高度：0.5-1.5 公尺 / Height: 0.5-1.5 meters
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,  # 圓柱體半徑 / Cylinder radius
                    height=height_val,  # 圓柱體高度 / Cylinder height
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 運動學模式 / Kinematic mode
                    collision_props=sim_utils.CollisionPropertiesCfg(),  # 碰撞屬性 / Collision properties
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(random.random(), random.random(), random.random()),  # 隨機顏色 / Random color
                        metallic=0.2,  # 金屬度 / Metallic
                    ),
                )
                height = height_val / 2  # 高度的一半 / Half of height

            # 創建障礙物資產 / Create obstacle asset
            # 使用 setattr 動態添加屬性到配置類別
            # Use setattr to dynamically add attributes to the configuration class
            setattr(
                self,
                f"obstacle_{i}",  # 屬性名稱：obstacle_0, obstacle_1, ... / Attribute name: obstacle_0, obstacle_1, ...
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",  # 障礙物在場景圖中的路徑 / Obstacle path in scene graph
                    spawn=spawn_cfg,  # 生成配置（立方體或圓柱體） / Spawn configuration (cube or cylinder)
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(
                            random.uniform(-6, 6),  # X 位置：-6 到 6 公尺 / X position: -6 to 6 meters
                            random.uniform(-6, 6),  # Y 位置：-6 到 6 公尺 / Y position: -6 to 6 meters
                            height,  # Z 位置：高度的一半（使底部貼地） / Z position: half height (so bottom touches ground)
                        )
                    ),
                )
            )

##
# MDP settings / MDP 設定
##
# MDP (Markov Decision Process) 設定：定義動作、觀測、獎勵、終止條件和事件
# MDP (Markov Decision Process) settings: define actions, observations, rewards, terminations, and events


@configclass
class CommandsCfg:
    """
    命令配置 / Command Configuration

    定義環境中的命令生成器，用於生成任務目標。
    Defines command generators in the environment for generating task goals.

    在此配置中，使用目標位置命令生成器，隨機生成目標點供機器人導航。
    In this configuration, uses a goal position command generator to randomly generate target points for robot navigation.
    """
    goal_command = GoalCommandCfg(
        asset_name="robot",  # 目標相對於的資產名稱 / Asset name relative to which goal is generated
        resampling_time_range=(1e10, 1e10),  # 重新採樣時間範圍：極大值表示只在重置時生成新目標 / Resampling time range: very large values mean only generate new goal on reset
        debug_vis=True,  # 啟用可視化：在場景中顯示目標位置標記 / Enable visualization: show goal position marker in scene
        ranges=GoalCommandCfg.Ranges(
            distance=(1.5, 3.0),  # 目標距離範圍：1.5-3.0 公尺 / Goal distance range: 1.5-3.0 meters
            angle=(-math.pi, math.pi),  # 目標角度範圍：-π 到 π（360 度） / Goal angle range: -π to π (360 degrees)
        ),
    )


@configclass
class ActionsCfg:
    """
    動作配置 - 差速驅動 / Action Configuration - Differential Drive

    定義機器人可以執行的動作空間。
    Defines the action space that the robot can execute.

    差速驅動（Differential Drive）：
    Differential Drive:
    - 動作空間為 2 維：[線性速度, 角速度]
      Action space is 2D: [linear velocity, angular velocity]
    - 線性速度：控制前進/後退速度（m/s）
      Linear velocity: controls forward/backward speed (m/s)
    - 角速度：控制旋轉速度（rad/s）
      Angular velocity: controls rotation speed (rad/s)
    - 動作值範圍為 [-1, 1]，會被縮放到實際速度範圍
      Action values range [-1, 1], will be scaled to actual velocity range
    """

    diff_drive = carter_mdp.DifferentialDriveActionCfg(
        asset_name="robot",  # 控制的機器人資產名稱 / Name of robot asset to control
        max_linear_velocity=1.5,   # 最大前進速度：1.5 m/s / Maximum forward speed: 1.5 m/s
        max_angular_velocity=1.5,  # 最大旋轉速度：1.5 rad/s（約 86 度/秒） / Maximum rotation speed: 1.5 rad/s (approximately 86 deg/s)
    )


@configclass
class ObservationsCfg:
    """
    觀測配置 / Observation Configuration

    定義策略網絡可以觀察到的環境資訊。
    Defines the environmental information that the policy network can observe.

    觀測空間包括：
    Observation space includes:
    - 雷射雷達掃描：360 度距離測量
      LiDAR scan: 360-degree distance measurements
    - 目標位置：目標相對於機器人的位置
      Goal position: target position relative to robot
    - 目標距離：機器人到目標的距離
      Goal distance: distance from robot to goal
    - 上一步動作：用於學習動作序列的連續性
      Previous action: for learning action sequence continuity
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        策略觀測 / Policy Observations

        定義輸入到策略網絡的所有觀測項。
        Defines all observation terms input to the policy network.

        這些觀測會被連接（concatenate）成一個向量，然後輸入到神經網絡。
        These observations will be concatenated into a single vector, then fed to the neural network.
        """

        lidar_scan = ObsTerm(
            func=carter_mdp.lidar_scan,  # 觀測函數：獲取雷射雷達掃描數據 / Observation function: get LiDAR scan data
            params={"sensor_cfg": SceneEntityCfg("lidar")},  # 參數：指定使用 "lidar" 感測器 / Parameters: specify using "lidar" sensor
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 添加均勻雜訊：±0.05（增加魯棒性） / Add uniform noise: ±0.05 (increases robustness)
        )

        goal_position = ObsTerm(
            func=carter_mdp.goal_position_in_robot_frame,  # 觀測函數：獲取目標在機器人座標系中的位置 / Observation function: get goal position in robot frame
            params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：指定使用 "robot" 資產 / Parameters: specify using "robot" asset
        )

        goal_distance = ObsTerm(
            func=carter_mdp.goal_distance,  # 觀測函數：獲取機器人到目標的距離 / Observation function: get distance from robot to goal
            params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：指定使用 "robot" 資產 / Parameters: specify using "robot" asset
        )

        actions = ObsTerm(func=carter_mdp.safe_last_action)  # 觀測函數：獲取上一步的動作（帶安全檢查） / Observation function: get previous action (with safety checks)

        def __post_init__(self):
            """
            後處理初始化 / Post-initialization processing

            設置觀測群組的屬性：
            Sets observation group attributes:
            - enable_corruption: 啟用觀測腐化（用於域隨機化）
              Enable observation corruption (for domain randomization)
            - concatenate_terms: 連接所有觀測項成一個向量
              Concatenate all observation terms into a single vector
            """
            self.enable_corruption = True  # 啟用觀測腐化 / Enable observation corruption
            self.concatenate_terms = True  # 連接所有觀測項 / Concatenate all observation terms

    policy: PolicyCfg = PolicyCfg()  # 策略觀測配置實例 / Policy observation configuration instance


@configclass
class RewardsCfg:
    """
    獎勵配置 / Reward Configuration

    定義強化學習的獎勵函數，用於指導策略學習。
    Defines reward functions for reinforcement learning to guide policy learning.

    獎勵設計原則：
    Reward design principles:
    - 正獎勵：鼓勵期望行為（朝向目標移動、到達目標）
      Positive rewards: encourage desired behaviors (moving toward goal, reaching goal)
    - 負獎勵（懲罰）：阻止不良行為（碰撞、動作不平滑、超時）
      Negative rewards (penalties): discourage bad behaviors (collision, non-smooth actions, timeout)
    - 權重：調整各項獎勵的重要性
      Weights: adjust the importance of each reward term
    """

    # ✅ 改用速度獎勵（核心） / Use velocity reward (core)
    # 獎勵機器人朝向目標移動的速度分量，這比距離獎勵更直接有效
    # Rewards the velocity component of robot moving toward goal, more direct and effective than distance reward
    velocity_toward_goal = RewTerm(
        func=carter_mdp.velocity_toward_goal,  # 獎勵函數：朝向目標的速度 / Reward function: velocity toward goal
        weight=0.8,  # 權重：0.8（核心獎勵項） / Weight: 0.8 (core reward term)
        params={"asset_cfg": SceneEntityCfg("robot"), "min_dist": 1.0},  # 參數：最小距離閾值 1.0 公尺（避免在目標附近振盪） / Parameters: min distance threshold 1.0m (avoid oscillation near goal)
    )

    # ✅ 距離獎勵（輔助） / Distance reward (auxiliary)
    # 只有在本時間步比上一步更接近目標時才有正獎勵（進度獎勵）
    # Only gives positive reward when closer to goal than previous step (progress reward)
    distance_to_goal = RewTerm(
        func=carter_mdp.progress_to_goal,  # 獎勵函數：朝向目標的進度 / Reward function: progress toward goal
        weight=2.0,  # 權重：2.0（輔助獎勵項） / Weight: 2.0 (auxiliary reward term)
        params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：指定使用 "robot" 資產 / Parameters: specify using "robot" asset
    )

    # 到達目標（最重要） / Reaching goal (most important)
    # 當機器人成功到達目標時給予大獎勵，確保策略優先學習到達目標
    # Gives large reward when robot successfully reaches goal, ensures policy prioritizes learning to reach goal
    reaching_goal = RewTerm(
        func=carter_mdp.reaching_goal,  # 獎勵函數：到達目標 / Reward function: reaching goal
        weight=100,  # 權重：100（最重要的獎勵項） / Weight: 100 (most important reward term)
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},  # 參數：到達閾值 0.7 公尺 / Parameters: reaching threshold 0.7 meters
    )

    # 碰撞懲罰 / Collision penalty
    # 當機器人過於接近障礙物時給予懲罰，鼓勵避障行為
    # Penalizes robot when too close to obstacles, encourages obstacle avoidance behavior
    collision = RewTerm(
        func=carter_mdp.collision_penalty,  # 獎勵函數：碰撞懲罰 / Reward function: collision penalty
        weight=-50.0,  # 權重：-50.0（強烈懲罰） / Weight: -50.0 (strong penalty)
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.5},  # 參數：碰撞距離閾值 0.5 公尺 / Parameters: collision distance threshold 0.5 meters
    )

    # 動作平滑 / Action smoothness
    # 懲罰動作變化過大，鼓勵平滑的動作序列
    # Penalizes large action changes, encourages smooth action sequences
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # 權重：-0.05（輕微懲罰） / Weight: -0.05 (light penalty)

    # 超時懲罰 / Timeout penalty
    # 對超時的環境施加懲罰，鼓勵策略在時間限制內完成任務
    # Penalizes timed-out environments, encourages policy to complete task within time limit
    time_out = RewTerm(
        func=carter_mdp.time_out_penalty,  # 獎勵函數：超時懲罰 / Reward function: timeout penalty
        weight=-10.0,  # 權重：-10.0 / Weight: -10.0
    )


@configclass
class TerminationsCfg:
    """
    終止條件配置 / Termination Condition Configuration

    定義何時結束一個回合（episode）。
    Defines when to end an episode.

    終止條件包括：
    Termination conditions include:
    - 超時：回合時間達到上限
      Timeout: episode time reaches limit
    - 到達目標：成功完成任務
      Goal reached: successfully completed task
    - 碰撞：與障礙物發生碰撞
      Collision: collided with obstacle
    - 翻倒：機器人傾斜過大
      Tipped over: robot tilted too much
    - 懸空：機器人離開地面（異常狀態）
      Flying: robot left ground (abnormal state)
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超時終止：回合時間達到上限 / Timeout termination: episode time reaches limit

    goal_reached = DoneTerm(
        func=carter_mdp.goal_reached,  # 終止函數：到達目標 / Termination function: goal reached
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.5},  # 參數：到達閾值 0.5 公尺 / Parameters: reaching threshold 0.5 meters
    )

    collision = DoneTerm(
        func=carter_mdp.collision_occurred,  # 終止函數：碰撞發生 / Termination function: collision occurred
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.3},  # 參數：碰撞距離閾值 0.3 公尺 / Parameters: collision distance threshold 0.3 meters
    )
    tipped_over = DoneTerm(
        func=carter_mdp.robot_tipped_over,  # 終止函數：機器人翻倒 / Termination function: robot tipped over
        params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：指定使用 "robot" 資產 / Parameters: specify using "robot" asset
    )

    flying = DoneTerm(
        func=carter_mdp.robot_flying,  # 終止函數：機器人懸空 / Termination function: robot flying
        params={"asset_cfg": SceneEntityCfg("robot")},  # 參數：指定使用 "robot" 資產 / Parameters: specify using "robot" asset
    )


@configclass
class EventCfg:
    """
    事件配置 / Event Configuration

    定義環境重置時的事件處理。
    Defines event handling when environment resets.

    事件包括：
    Events include:
    - 重置機器人基座：隨機設置機器人的初始位置和朝向
      Reset robot base: randomly set robot's initial position and orientation
    - 重置障礙物：隨機重新生成障礙物位置
      Reset obstacles: randomly regenerate obstacle positions
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # 事件函數：重置根狀態（位置和速度） / Event function: reset root state (position and velocity)
        mode="reset",  # 模式：重置模式 / Mode: reset mode
        params={
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},  # 位置範圍：X/Y 在 -2 到 2 公尺，偏航角在 -π 到 π / Position range: X/Y in -2 to 2 meters, yaw in -π to π
            "velocity_range": {
                "x": (0.0, 0.0),  # X 速度範圍：0（靜止開始） / X velocity range: 0 (start stationary)
                "y": (0.0, 0.0),  # Y 速度範圍：0 / Y velocity range: 0
                "z": (0.0, 0.0),  # Z 速度範圍：0 / Z velocity range: 0
            },
        },
    )

    # 重置障礙物 / Reset obstacles
    # 在環境重置時，隨機重新生成所有障礙物的位置
    # On environment reset, randomly regenerate positions of all obstacles
    reset_obstacles = EventTerm(
        func=carter_mdp.reset_obstacles,  # 事件函數：重置障礙物 / Event function: reset obstacles
        mode="reset",  # 模式：重置模式 / Mode: reset mode
    )


##
# Environment configuration / 環境配置
##

@configclass
class CarterNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """
    Carter 激光雷达导航环境 / Carter LiDAR Navigation Environment

    完整的環境配置類別，整合所有組件：
    Complete environment configuration class that integrates all components:
    - 場景配置（地形、機器人、感測器、障礙物）
      Scene configuration (terrain, robot, sensors, obstacles)
    - 觀測配置（策略輸入）
      Observation configuration (policy inputs)
    - 動作配置（策略輸出）
      Action configuration (policy outputs)
    - 命令配置（任務目標）
      Command configuration (task goals)
    - 獎勵配置（學習信號）
      Reward configuration (learning signals)
    - 終止條件配置（回合結束條件）
      Termination configuration (episode end conditions)
    - 事件配置（重置行為）
      Event configuration (reset behavior)
    """

    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=15.0)  # 場景配置：128 個並行環境，間距 15 公尺 / Scene config: 128 parallel environments, 15m spacing
    observations: ObservationsCfg = ObservationsCfg()  # 觀測配置 / Observation configuration
    actions: ActionsCfg = ActionsCfg()  # 動作配置 / Action configuration
    commands: CommandsCfg = CommandsCfg()  # 命令配置 / Command configuration
    rewards: RewardsCfg = RewardsCfg()  # 獎勵配置 / Reward configuration
    terminations: TerminationsCfg = TerminationsCfg()  # 終止條件配置 / Termination configuration
    events: EventCfg = EventCfg()  # 事件配置 / Event configuration

    def __post_init__(self):
        """
        後處理初始化 / Post-initialization processing

        設置環境的運行參數：
        Sets environment runtime parameters:
        - 時間步長和更新頻率
          Time step and update frequency
        - 回合長度
          Episode length
        - GPU 加速設置
          GPU acceleration settings
        - 感測器更新週期
          Sensor update period
        """
        self.decimation = 4  # 降採樣率：每 4 個物理步驟執行一次策略 / Decimation: execute policy every 4 physics steps
        self.episode_length_s = 30.0  # 回合長度：30 秒 / Episode length: 30 seconds

        self.sim.dt = 0.01  # 物理時間步長：0.01 秒（100 Hz） / Physics time step: 0.01 seconds (100 Hz)
        self.sim.render_interval = self.decimation  # 渲染間隔：與降採樣率相同 / Render interval: same as decimation
        self.sim.use_gpu_pipeline = True  # 使用 GPU 管線：加速模擬 / Use GPU pipeline: accelerate simulation
        self.sim.physx.use_gpu = True  # 使用 GPU 物理引擎：加速物理計算 / Use GPU physics engine: accelerate physics computation

        # 設置雷射雷達更新週期，使其與策略更新頻率同步
        # Set LiDAR update period to synchronize with policy update frequency
        if self.scene.lidar is not None:
            # 更新週期 = 降採樣率 × 物理時間步長 = 4 × 0.01 = 0.04 秒（25 Hz）
            # Update period = decimation × physics time step = 4 × 0.01 = 0.04 seconds (25 Hz)
            self.scene.lidar.update_period = self.decimation * self.sim.dt


@configclass
class CarterNavigationEnvCfg_PLAY(CarterNavigationEnvCfg):
    """
    Carter 導航環境配置 - 遊戲/測試模式 / Carter Navigation Environment Configuration - Play/Test Mode

    用於測試和可視化的環境配置變體。
    Environment configuration variant for testing and visualization.

    與訓練配置的主要差異：
    Main differences from training configuration:
    - 較少的並行環境數（16 vs 128）：減少計算負擔，便於觀察
      Fewer parallel environments (16 vs 128): reduce computational burden, easier to observe
    - 禁用觀測腐化：使用真實觀測，不添加雜訊
      Disable observation corruption: use true observations, no noise added
    - 啟用可視化：顯示雷射雷達點雲等視覺輔助
      Enable visualization: show LiDAR point cloud and other visual aids
    """
    def __post_init__(self):
        """
        後處理初始化 - 遊戲模式設置 / Post-initialization - Play mode settings
        """
        super().__post_init__()  # 調用父類初始化 / Call parent class initialization
        self.scene.num_envs = 16  # 並行環境數：16（訓練時為 128） / Number of parallel environments: 16 (128 during training)
        self.observations.policy.enable_corruption = False  # 禁用觀測腐化：使用真實觀測 / Disable observation corruption: use true observations
        # 在 PLAY 模式下啟用 ray_caster 紅色點雲可視化 / Enable ray_caster red point cloud visualization in PLAY mode
        if self.scene.lidar is not None:
            self.scene.lidar.debug_vis = True  # 啟用除錯視覺化：顯示射線碰撞點 / Enable debug visualization: show ray hit points
