# ============================================================================
# Phase 0 NavRL: 使用 NavRL 風格獎勵函數的車輛導航訓練
# ============================================================================
"""
Phase 0 NavRL - 使用 NavRL 獎勵函數的導航環境

移植自 NavRL 專案 (https://github.com/Zhefan-Xu/NavRL)
NavRL: Learning Safe Flight in Dynamic Environments (IEEE RA-L 2025)

核心特點：
1. 使用 LiDAR 距離的對數作為安全獎勵
2. 速度方向獎勵（鼓勵向目標移動）
3. 動作平滑度懲罰（防止抖動）

與原始 NavRL 的差異：
- 原始：四旋翼飛行器，3D LiDAR
- 此處：地面車輛，2D LiDAR
- 保留：相同的獎勵函數形式和權重比例

獎勵組成：
    R_total = R_vel + 1.0              # 速度獎勵
             + R_safety_static * 1.0   # 靜態障礙物安全（LiDAR）
             + R_safety_dynamic * 1.0  # 動態障礙物安全
             - P_smooth * 0.1          # 平滑度懲罰
             - P_height * 8.0          # 高度懲罰（車輛為 0）

訓練命令：
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase0-NavRL \\
        --num_envs 256 \\
        --headless
"""

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import (
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
    CommandsCfg,
    EventCfg,
    GoalCommandCfg,
    GOAL_REACH_THRESHOLD,
    TerminationsCfg,
    MySceneCfg,
)

# 導入基礎觀測函數
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

from ..mdp.rewards import (
    reaching_goal,
    collision_penalty_reward,
    collision_occurred,
)

# 🆕 導入 NavRL 風格獎勵函數
from ..mdp.rewards import (
    navrl_velocity_reward,
    navrl_safety_reward_lidar,
    navrl_dynamic_obstacle_safety_reward,
    navrl_smoothness_penalty,
    navrl_height_penalty,
    navrl_total_reward,
)

from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
)

# 導入事件函數
from ..mdp.events import randomize_obstacles_by_difficulty
from ..mdp.events import reset_root_state_fixed_per_env
from ..domain_randomization import apply_domain_randomization
from .charge_env_cfg_phase0 import MAX_OBSTACLES


# ============================================================================
# NavRL 觀測配置（與 Phase 0 一致）
# ============================================================================

@configclass
class ObservationsCfgNavRL:
    """NavRL 觀測配置 - 與 Phase 0 一致的 81/131 維 AAC 架構

    NavRL 原始實現使用：
    - LiDAR: 360° × N 條射線（原始 3D LiDAR，此處 2D）
    - 目標相對位置
    - 機器人速度
    - 動作（用於平滑度懲罰）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組 - 81 維無特權資訊"""

        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_velocity_xy = ObsTerm(
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

        time_remaining_ratio = ObsTerm(
            func=time_remaining_ratio,
        )

        alive_flag = ObsTerm(
            func=alive_flag,
        )

        safe_last_action = ObsTerm(func=safe_last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 觀測組 - 131 維上帝視角"""

        # Policy 觀測
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_velocity_xy = ObsTerm(
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

        time_remaining_ratio = ObsTerm(
            func=time_remaining_ratio,
        )

        alive_flag = ObsTerm(
            func=alive_flag,
        )

        safe_last_action = ObsTerm(func=safe_last_action)

        # 障礙物資訊
        obstacles_state = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 0,  # 可在混合環境中啟用
                "max_obstacles": MAX_OBSTACLES,
                "max_distance": 15.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================
# NavRL 獎勵配置
# ============================================================================

@configclass
class RewardsCfgNavRL:
    """NavRL 風格獎勵配置

    ═══════════════════════════════════════════════════════════════════════════
                            NavRL 獎勵函數設計
    ═══════════════════════════════════════════════════════════════════════════

    移植自 NavRL (IEEE RA-L 2025)：

    R_vel = (v · r_hat) + 1.0
    -------------------------------------
    - v: 機器人速度向量
    - r_hat: 指向目標的單位向量
    - +1.0: 基本獎勵（即使不移動也給點分）

    R_safety = mean(log(lidar_range - lidar_scan))
    -------------------------------------
    - lidar_range: LiDAR 最大探測距離
    - lidar_scan: 實際測量距離
    - 對數使獎勵平滑，遠離障礙物時給高分

    P_smooth = ||v_t - v_{t-1}|| * 0.1
    -------------------------------------
    - 懲罰速度變化
    - 鼓勵平滑移動，防止抖動

    總獎勵公式：
    R_total = R_vel * 1.0
             + R_safety_static * 1.0
             + R_safety_dynamic * 1.0
             - P_smooth * 0.1
             - P_height * 8.0 (車輛版本 = 0)
    """

    # ========== 速度獎勵 ==========
    # R_vel: 速度在目標方向的投影 + 基本獎勵
    navrl_velocity = RewTerm(
        func=navrl_velocity_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "max_reward": 2.0,
        },
        weight=1.0,  # NavRL 原始權重
    )

    # ========== 安全獎勵（靜態障礙物） ==========
    # R_safety_static: LiDAR 對數獎勵
    navrl_safety_lidar = RewTerm(
        func=navrl_safety_reward_lidar,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "lidar_range": 4.0,  # 與 NavRL 一致
            "safety_margin": 0.3,
        },
        weight=1.0,  # NavRL 原始權重
    )

    # ========== 安全獎勵（動態障礙物） ==========
    # R_safety_dynamic: 動態障礙物距離對數獎勵
    navrl_safety_dynamic = RewTerm(
        func=navrl_dynamic_obstacle_safety_reward,
        params={
            "obstacle_prefix": "obstacle_",
            "num_dynamic_obstacles": 0,  # Phase 0 預設無動態障礙物
            "lidar_range": 4.0,
        },
        weight=1.0,
    )

    # ========== 平滑度懲罰 ==========
    # P_smooth: 速度變化懲罰
    navrl_smoothness = RewTerm(
        func=navrl_smoothness_penalty,
        params={},
        weight=-0.1,  # 負權重 = 懲罰
    )

    # ========== 高度懲罰 ==========
    # P_height: 車輛版本為 0（保留與 NavRL API 一致）
    navrl_height = RewTerm(
        func=navrl_height_penalty,
        params={},
        weight=-8.0,  # 負權重，但函數返回 0
    )

    # ========== 目標抵達獎勵 ==========
    # 保留原始的抵達目標獎勵（給予額外獎勵）
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
        weight=10.0,  # 抵達目標給予額外獎勵
    )

    # ========== 碰撞懲罰 ==========
    # 保留碰撞懲罰（終止條件已處理，這裡給予額外懲罰）
    collision_penalty = RewTerm(
        func=collision_penalty_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "threshold": 0.5,
        },
        weight=-1.0,
    )


# ============================================================================
# NavRL 終止條件配置
# ============================================================================

@configclass
class TerminationsCfgNavRL:
    """NavRL 終止條件配置"""

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

    collision = DoneTerm(
        func=collision_occurred,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "threshold": 0.5,
        },
    )


# ============================================================================
# NavRL 場景配置（與 Phase 0 相同）
# ============================================================================

@configclass
class MySceneCfgNavRL(MySceneCfg):
    """NavRL 場景配置 - 與 Phase 0 一致的 16x16m 房間"""

    def __post_init__(self):
        """初始化場景（四面牆壁 + 10個障礙物版本）"""
        InteractiveSceneCfg.__post_init__(self)

        # 房間配置
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
                pos=(0.0, room_size, wall_height / 2),
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
                pos=(0.0, -room_size, wall_height / 2),
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
                pos=(room_size, 0.0, wall_height / 2),
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
                pos=(-room_size, 0.0, wall_height / 2),
            ),
        )

        # 障礙物配置
        HIDDEN_Z = -10.0

        obstacle_configs = [
            {"type": "cuboid", "size": (0.5, 0.5, 1.2), "color": (0.8, 0.2, 0.2)},
            {"type": "cylinder", "radius": 0.3, "height": 1.0, "color": (0.8, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.7, 0.7, 1.4), "color": (0.2, 0.4, 0.8)},
            {"type": "cylinder", "radius": 0.25, "height": 0.8, "color": (0.2, 0.8, 0.2)},
            {"type": "cuboid", "size": (0.6, 0.6, 1.0), "color": (0.8, 0.2, 0.8)},
            {"type": "cylinder", "radius": 0.35, "height": 1.2, "color": (0.8, 0.5, 0.2)},
            {"type": "cuboid", "size": (0.4, 0.4, 0.9), "color": (0.2, 0.8, 0.8)},
            {"type": "cylinder", "radius": 0.2, "height": 1.5, "color": (0.5, 0.5, 0.5)},
            {"type": "cuboid", "size": (0.55, 0.55, 1.1), "color": (0.9, 0.9, 0.9)},
            {"type": "cylinder", "radius": 0.28, "height": 1.1, "color": (0.3, 0.3, 0.3)},
        ]

        obstacle_sizes = []

        for i, cfg in enumerate(obstacle_configs):
            if cfg["type"] == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=cfg["size"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=cfg["color"],
                        metallic=0.2,
                    ),
                )
                height_i = cfg["size"][2] / 2
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"],
                    height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=cfg["color"],
                        metallic=0.2,
                    ),
                )
                height_i = cfg["height"] / 2
                size_scalar = cfg["radius"] * 2.0

            setattr(
                self,
                f"obstacle_{i}",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=spawn_cfg,
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(0.0, 0.0, HIDDEN_Z),
                    ),
                )
            )

            obstacle_sizes.append(size_scalar)

        # 記錄障礙物元數據
        from ..mdp.core import set_obstacle_metadata
        set_obstacle_metadata(MAX_OBSTACLES, obstacle_sizes)

        # LiDAR 配置
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
                    prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
            update_period=0.04,
            debug_vis=True,
        )


# ============================================================================
# NavRL 命令配置
# ============================================================================

@configclass
class CommandsCfgNavRL(CommandsCfg):
    """NavRL 命令配置"""

    goal_command = GoalCommandCfg(
        asset_name="robot",
        debug_vis=True,
        resampling_time_range=(5.0, 10.0),
        ranges=GoalCommandCfg.Ranges(
            distance=(3.0, 8.0),
            angle=(-math.pi, math.pi),
        ),
        wall_boundary=7.5,
        wall_safe_margin=0.5,
        num_obstacles=0,
    )


# ============================================================================
# NavRL 事件配置
# ============================================================================

@configclass
class EventCfgNavRL(EventCfg):
    """NavRL 事件配置"""

    # 域隨機化
    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={
            "enable_physics": True,
            "enable_sensor_noise": True,
            "enable_external_force": True,
        },
    )

    # 混合平行環境事件
    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 0.2,
            "static_ratio": 0.5,
            "dynamic_ratio": 0.3,
            "num_obstacles_static": 5,
            "num_obstacles_dynamic": 8,
            "max_obstacles": MAX_OBSTACLES,
            "speed_range": 0.5,
            "min_speed": 0.05,
            "min_robot_distance": 1.5,
            "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.0,
            "max_spawn_attempts": 50,
            "boundary": 7.5,
        },
    )

    # 重置機器人
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
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )


# ============================================================================
# NavRL 環境配置主類
# ============================================================================

@configclass
class ChargeNavigationEnvCfgPhase0NavRL(ChargeNavigationEnvCfg):
    """NavRL 風格的 Phase 0 導航環境配置

    使用 NavRL 獎勵函數進行訓練：
    - 速度獎勵：鼓勵向目標移動
    - LiDAR 安全獎勵：對數距離獎勵
    - 平滑度懲罰：防止抖動

    訓練建議：
    - num_envs: 256-512
    - 訓練步數: 2M-5M steps
    """

    scene: MySceneCfgNavRL = MySceneCfgNavRL(num_envs=256, env_spacing=18.0)
    observations: ObservationsCfgNavRL = ObservationsCfgNavRL()
    rewards: RewardsCfgNavRL = RewardsCfgNavRL()
    terminations: TerminationsCfgNavRL = TerminationsCfgNavRL()
    commands: CommandsCfgNavRL = CommandsCfgNavRL()
    events: EventCfgNavRL = EventCfgNavRL()
    episode_length_s = 20.0


# ============================================================================
# 導出
# ============================================================================

__all__ = [
    "ChargeNavigationEnvCfgPhase0NavRL",
]
