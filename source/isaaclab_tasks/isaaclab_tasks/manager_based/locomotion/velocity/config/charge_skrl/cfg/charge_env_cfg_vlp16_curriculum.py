"""VLP-16 Curriculum 環境配置 — 20×20m 場景 + 課程學習

繼承 ChargeNavigationEnvCfgVLP16，變更：
- 場景：16×16m → 20×20m，4 面內部牆（密度更低）
- 課程學習：4 階段 goal-obstacle 聯動
- 獎勵：deadzone 加速度 + context-aware 角速度懲罰
- Episode：45s → 60s（更大場景）
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .charge_env_cfg_vlp16 import (
    ChargeNavigationEnvCfgVLP16,
    MySceneCfgVLP16,
    RewardsCfgVLP16,
    EventCfgVLP16,
    ROBOT_BODY_RADIUS,
    GOAL_REACH_THRESHOLD,
    COLLISION_THRESHOLD,
)
from ..multi_goal_command import MultiGoalCommandCfg
from ..goal_command import GoalCommandCfg

# 獎勵函數
from ..mdp.rewards import (
    reaching_goal,
)
from ..mdp.rewards.potential_based_rewards import (
    potential_progress_reward,
    exponential_obstacle_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    velocity_too_low_penalty,
)
from ..mdp.rewards.smoothness_rewards import (
    deadzone_acceleration_penalty,
    context_aware_angular_velocity_penalty,
)

# 事件
from ..mdp.events import (
    randomize_obstacles_by_difficulty,
    move_obstacles_vectorized,
)
from ..mdp.events.reset import reset_root_state_random_safe
from ..domain_randomization import apply_domain_randomization
from ..mdp.events.state import set_obstacle_metadata

# 課程
from ..curriculum.goal_obstacle_curriculum import goal_obstacle_curriculum

# 牆壁
from ..mdp.wall_layout import get_wall_tensors_20x20


def _set_wall_fn_20x20(env, env_ids):
    """Startup event: 設定 env._wall_tensor_fn 為 20×20 牆壁版本。"""
    env._wall_tensor_fn = get_wall_tensors_20x20


# ============================================================================
# 20×20m 場景配置
# ============================================================================
@configclass
class MySceneCfgVLP16_20x20(MySceneCfgVLP16):
    """20×20m 場景 — 4 面內部牆（匹配 MAZE_WALLS_20x20）"""

    def __post_init__(self):
        # 呼叫祖父類 __post_init__（跳過 MySceneCfgVLP16 的 16×16 牆壁設定）
        from isaaclab.scene import InteractiveSceneCfg
        InteractiveSceneCfg.__post_init__(self)

        room_size = 10.0  # ±10m = 20×20m
        wall_thickness = 0.2
        wall_height = 1.5
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, metallic=0.1)

        # 四面外牆（20×20m）
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, room_size, wall_height / 2)),
        )
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -room_size, wall_height / 2)),
        )
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(room_size, 0.0, wall_height / 2)),
        )
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-room_size, 0.0, wall_height / 2)),
        )

        # 4 面內部牆壁（匹配 wall_layout.py MAZE_WALLS_20x20）
        internal_walls = [
            ((4.0, 0.2, 1.5), (-7.5, 5.0, 0.75)),    # wall_internal_0: Top-left horizontal
            ((0.2, 3.5, 1.5), (4.0, 7.5, 0.75)),      # wall_internal_1: Top-right vertical
            ((0.2, 4.5, 1.5), (-4.0, -1.5, 0.75)),    # wall_internal_2: Center-left vertical
            ((3.5, 0.2, 1.5), (2.0, -6.0, 0.75)),     # wall_internal_3: Bottom-center horizontal
        ]

        # 先刪除繼承的 16×16 內部牆（index 4, 5 不存在於 20×20）
        for i in range(6):
            attr_name = f"wall_internal_{i}"
            if i < len(internal_walls):
                size, pos = internal_walls[i]
                setattr(self, attr_name, AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Wall_Internal_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        rigid_props=wall_rigid_props,
                        collision_props=wall_collision_props,
                        visual_material=wall_visual,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                ))
            else:
                # 移除多餘的內部牆（16×16 有 6 面，20×20 只有 4 面）
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

        # 10 個混合障礙物（與 16×16 相同，初始隱藏在 Z = -10.0）
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

        obstacle_sizes: list[float] = []
        for i, cfg in enumerate(obstacle_configs):
            if cfg["type"] == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=cfg["size"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"], height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = cfg["radius"] * 2.0

            setattr(self, f"obstacle_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
            ))
            obstacle_sizes.append(size_scalar)

        set_obstacle_metadata(10, obstacle_sizes)


# ============================================================================
# 課程版命令配置（Stage 1 初始值）
# ============================================================================
@configclass
class CommandsCfgVLP16Curriculum:
    """課程命令：Phase 1 初始 8 goals，距離 2-5m（密集探索）"""
    goal_command = MultiGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(2.0, 5.0),
            angle=(-math.pi, math.pi),
        ),
        wall_boundary=9.5,
        wall_safe_margin=0.5,
        obstacle_safe_distance=1.0,
        num_obstacles=0,
        num_goals=8,
        max_goals=10,
    )


# ============================================================================
# 課程版事件配置（Stage 1 初始值）
# ============================================================================
@configclass
class EventCfgVLP16Curriculum:
    """課程事件：20×20m 場景 + Stage 1 初始障礙物分布"""

    reset_base = EventTerm(
        func=reset_root_state_random_safe,
        mode="reset",
        params={
            "pose_range": {
                "x": (-7.0, 7.0),
                "y": (-7.0, 7.0),
                "yaw": (-3.14, 3.14),
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

    # 設定 20×20 牆壁 dispatch（必須在 goal_command 之前執行）
    set_wall_fn = EventTerm(
        func=_set_wall_fn_20x20,
        mode="startup",
        params={},
    )

    reset_obstacles = None  # 由 randomize_obstacles 處理

    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={"enable_physics": True, "enable_sensor_noise": True, "enable_external_force": True},
    )

    # Phase 1 初始分布：100% empty（純導航學習，無障礙物）
    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": 9.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )

    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 10, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": 9.5, "active_obstacle_ratio": 0.25, "debug": False,
        },
    )

    move_dynamic_obstacles = EventTerm(
        func=move_obstacles_vectorized,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "move_dt": 0.2, "speed_min": 0.3, "speed_max": 1.2,
            "goal_reach_threshold": 0.5, "speed_resample_steps": 10,
            "area_limit": 8.0, "max_obstacles": 10, "bound_limit": 9.0,
        },
    )


# ============================================================================
# 課程版獎勵配置（改進平滑懲罰）
# ============================================================================
@configclass
class RewardsCfgVLP16Curriculum(RewardsCfgVLP16):
    """課程獎勵 — deadzone 加速度 + context-aware 角速度

    相對 Phase 1 變更：
    - acceleration_penalty → deadzone_acceleration_penalty（dead_zone=0.1）
    - angular_velocity_penalty → context_aware_angular_velocity_penalty
    """

    # 平滑：dead-zone 加速度懲罰（|a| < 0.1 免懲罰）
    acceleration_penalty = RewTerm(
        func=deadzone_acceleration_penalty,
        params={"dead_zone": 0.1},
        weight=-0.15,
    )

    # 平滑：context-aware 角速度懲罰（靠近障礙物時降低懲罰）
    angular_velocity_penalty = RewTerm(
        func=context_aware_angular_velocity_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "proximity_distance": 1.5,
            "max_reduction": 0.7,
        },
        weight=-0.15,
    )


# ============================================================================
# 課程管理配置
# ============================================================================
@configclass
class CurriculumCfgVLP16:
    """4 階段 goal-obstacle 聯動課程"""
    goal_obstacle_curriculum = CurriculumTermCfg(
        func=goal_obstacle_curriculum,
        params={
            "window_size": 2000,
            "min_stage_episodes": 5000,
            "initial_stage": 1,
        },
    )


# ============================================================================
# 完整課程環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgVLP16Curriculum(ChargeNavigationEnvCfgVLP16):
    """VLP-16 Curriculum 環境 — 20×20m + 4 階段課程學習

    繼承 VLP16 Phase 1 的觀測/動作/終止設定，
    覆蓋 scene/commands/events/rewards/curriculum。
    """

    scene: MySceneCfgVLP16_20x20 = MySceneCfgVLP16_20x20(num_envs=1024, env_spacing=22.0)
    commands: CommandsCfgVLP16Curriculum = CommandsCfgVLP16Curriculum()
    events: EventCfgVLP16Curriculum = EventCfgVLP16Curriculum()
    rewards: RewardsCfgVLP16Curriculum = RewardsCfgVLP16Curriculum()
    curriculum: CurriculumCfgVLP16 = CurriculumCfgVLP16()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0  # 20×20 更大，給 60s
        self.viewer.eye = (9.0, 9.0, 9.0)
        # 設定 wall dispatch 使用 20×20 牆壁
        # （在環境初始化後由 startup event 設定 env._wall_tensor_fn）
