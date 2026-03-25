"""
Carter 激光雷达导航环境配置
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfRandomUniformTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
)

from .carter_cfg import CARTER_CFG
from . import carter_mdp
from .goal_command import GoalCommandCfg
from . import carter_mdp

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """场景配置"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=0.0,
            num_rows=4,
            num_cols=4,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            seed=42,                # ✅ 设置种子
            use_cache=True,         # ✅ 启用缓存
            cache_dir="/tmp/isaaclab/terrains",  # 缓存目录（可选）
            sub_terrains={
                "flat": HfRandomUniformTerrainCfg(
                    size=(8.0, 8.0),
                    proportion=0.3,
                    noise_range=(0.0, 0.0),
                    noise_step=0.01,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    border_width=0.0,
                    slope_threshold=0.0,
                ),
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(8.0, 8.0),
                    proportion=0.4,
                    platform_width=1.5,
                    num_obstacles=20,
                    obstacle_height_range=(0.6, 0.8),
                    obstacle_width_range=(0.4, 0.8),
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    border_width=0.0,
                ),
                "rough": HfRandomUniformTerrainCfg(
                    size=(8.0, 8.0),
                    proportion=0.3,
                    noise_range=(0.01, 0.05),
                    noise_step=0.01,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    border_width=0.0,
                    slope_threshold=0.75,
                ),
            },
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            color_scheme="height",
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        debug_vis=True,
    )

    # Carter机器人
    robot = CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 激光雷达
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        debug_vis=False,
        #mesh_prim_paths=["{ENV_REGEX_NS}"], 
        mesh_prim_paths=["/World/ground"], 
        update_period=0.04,
    )
    
    # 光照
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )
    
"""     def __post_init__(self): """
"""         super().__post_init__() """
"""         import random """
"""          """
"""         num_obstacles = 8  # 障碍物数量 """
"""          """
"""         for i in range(num_obstacles): """
"""             # 随机选择类型 """
"""             is_cube = random.random() > 0.5 """
"""              """
"""             if is_cube: """
"""                 # 立方体 """
"""                 size = ( """
"""                     random.uniform(0.3, 0.8), """
"""                     random.uniform(0.3, 0.8), """
"""                     random.uniform(0.5, 1.5), """
"""                 ) """
"""                 spawn_cfg = sim_utils.CuboidCfg( """
"""                     size=size, """
"""                     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), """
"""                     collision_props=sim_utils.CollisionPropertiesCfg(), """
"""                     visual_material=sim_utils.PreviewSurfaceCfg( """
"""                         diffuse_color=(random.random(), random.random(), random.random()), """
"""                         metallic=0.2, """
"""                     ), """
"""                 ) """
"""                 height = size[2] / 2 """
"""             else: """
"""                 # 圆柱体 """
"""                 radius = random.uniform(0.2, 0.5) """
"""                 height_val = random.uniform(0.5, 1.5) """
"""                 spawn_cfg = sim_utils.CylinderCfg( """
"""                     radius=radius, """
"""                     height=height_val, """
"""                     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), """
"""                     collision_props=sim_utils.CollisionPropertiesCfg(), """
"""                     visual_material=sim_utils.PreviewSurfaceCfg( """
"""                         diffuse_color=(random.random(), random.random(), random.random()), """
"""                         metallic=0.2, """
"""                     ), """
"""                 ) """
"""                 height = height_val / 2 """
"""              """
"""             # 创建障碍物资产 """
"""             setattr( """
"""                 self, """
"""                 f"obstacle_{i}", """
"""                 AssetBaseCfg( """
"""                     prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}", """
"""                     spawn=spawn_cfg, """
"""                     init_state=AssetBaseCfg.InitialStateCfg( """
"""                         pos=( """
"""                             random.uniform(-6, 6), """
"""                             random.uniform(-6, 6), """
"""                             height, """
"""                         ) """
"""                     ), """
"""                 ) """
"""             ) """
"""  """

##
# MDP settings
##

@configclass
class CommandsCfg:
    """命令配置"""
    goal_command = GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(1.5, 3.0),
            angle=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """动作配置 - 差速驱动"""
    
    diff_drive = carter_mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        max_linear_velocity=1.5,   # 最大前进速度 2 m/s
        max_angular_velocity=1.5,  # 最大旋转速度 2 rad/s
    )


@configclass
class ObservationsCfg:
    """观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测"""
        
        lidar_scan = ObsTerm(
            func=carter_mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        goal_position = ObsTerm(
            func=carter_mdp.goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        goal_distance = ObsTerm(
            func=carter_mdp.goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        actions = ObsTerm(func=carter_mdp.safe_last_action) 

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """奖励配置"""
    
    # ✅ 改用速度奖励（核心）
    velocity_toward_goal = RewTerm(
        func=carter_mdp.velocity_toward_goal,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_dist": 1.0},
    )
    
    # ✅ 距离奖励（辅助）
    distance_to_goal = RewTerm(
        func=carter_mdp.progress_to_goal,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 到达目标（最重要）
    reaching_goal = RewTerm(
        func=carter_mdp.reaching_goal,
        weight=100,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    )
    
    # 碰撞惩罚
    collision = RewTerm(
        func=carter_mdp.collision_penalty,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.5},
    )
    
    # 动作平滑
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # 超时惩罚
    time_out = RewTerm(
    func=carter_mdp.time_out_penalty,
    weight=-10.0,
)



@configclass
class TerminationsCfg:
    """终止条件配置"""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    goal_reached = DoneTerm(
        func=carter_mdp.goal_reached,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.5},
    )
    
    collision = DoneTerm(
        func=carter_mdp.collision_occurred,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.3},
    )
    tipped_over = DoneTerm(
        func=carter_mdp.robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    flying = DoneTerm(
        func=carter_mdp.robot_flying,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """事件配置"""
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )
    
    # 重置障碍物
    reset_obstacles = EventTerm(
        func=carter_mdp.reset_obstacles,
        mode="reset",
    )


##
# Environment configuration
##

@configclass
class CarterNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Carter 激光雷达导航环境"""
    
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=15.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.use_gpu_pipeline = True
        self.sim.physx.use_gpu = True
        
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt


@configclass
class CarterNavigationEnvCfg_PLAY(CarterNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False