"""DirectMARLEnv for Charge Navigation — Phase 2 (N cars)

Phase 2: N 台車 DirectMARLEnv，WarpDrive-style 連續 episode。
- N agents: "car_0" .. "car_{N-1}" (default N=6)
- 79D obs per car: ego(4) + goal(2) + LiDAR(72) + time(1)  [WD-aligned, no TopK]
- MultiDiscrete([19,19]) action per car
- NavRL dense rewards + car-to-car collision penalty
- WarpDrive-style: collision/goal → respawn car, episode 不終止
- 只有 shared timeout 終止 episode
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    MultiMeshRayCaster,
    MultiMeshRayCasterCfg,
    patterns,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim_utils

from ..cfg.charge_cfg import CHARGE_CFG
from ..mdp.events.state import set_obstacle_metadata
from ..mdp.wall_layout import WALL_SLOT_SPECS


# ============================================================================
# Constants
# ============================================================================
ROBOT_BODY_RADIUS = 0.35
COLLISION_BUFFER = 0.10
COLLISION_THRESHOLD = round(ROBOT_BODY_RADIUS + COLLISION_BUFFER, 2)  # 0.45m
MAX_OBSTACLES = 15  # WD max is 10 per phase; 15 gives headroom
HIDDEN_Z = -10.0
DEFAULT_NUM_CARS = 6


# ============================================================================
# Scene Configuration — 20×20m, N cars
# ============================================================================
def _build_scene_cfg(num_envs: int = 1024, env_spacing: float = 22.0,
                     num_cars: int = DEFAULT_NUM_CARS) -> InteractiveSceneCfg:
    """Factory function to build scene cfg (avoids putting num_cars as a field
    which InteractiveScene._add_entities_from_cfg() would reject)."""
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    N = num_cars

    # Dome light
    cfg.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )

    # ==== N cars (Articulation) + N LiDARs + N contact sensors ====
    _lidar_targets = [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/ground", track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Wall_.*", track_mesh_transforms=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_.*", track_mesh_transforms=True,
        ),
        # Pedestrian colliders — only when pedestrians are spawned
        # DISABLED: causes RuntimeError if no Pedestrian prims in scene
        # MultiMeshRayCasterCfg.RaycastTargetCfg(
        #     prim_expr="{ENV_REGEX_NS}/Pedestrian_.*/collision_capsule",
        #     track_mesh_transforms=True,
        # ),
        # NOTE: Car_* NOT included as LiDAR target — Articulation mesh causes
        # GfVec3d TypeError in MultiMeshRayCaster. Inter-car visibility handled
        # by TopK obstacle obs (other cars appear as obstacles in _compute_topk).
    ]

    for i in range(N):
        # Articulation
        setattr(cfg, f"car_{i}", CHARGE_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/Car_{i}",
        ))

        # LiDAR (VLP-16)
        setattr(cfg, f"lidar_{i}", MultiMeshRayCasterCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Car_{i}/charger_rover_urdf5/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.6)),
            ray_alignment="yaw",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=16,
                vertical_fov_range=(-15.0, 15.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=1.0,
            ),
            max_distance=20.0,
            debug_vis=False,
            mesh_prim_paths=_lidar_targets,
            update_period=0.2,
        ))

        # Contact sensor (obstacles only — car-to-car uses geometric check)
        setattr(cfg, f"contact_{i}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Car_{i}/charger_rover_urdf5/base_link",
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"],
            debug_vis=False,
        ))

    # ==== 4 external boundary walls ====
    room_size = 10.0
    wall_thickness = 1.0
    wall_height = 3.0   # ★ 對齊 VLP16 可觀測性
    wall_length = room_size * 2 + wall_thickness
    wall_color = (0.5, 0.5, 0.5)

    wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True, disable_gravity=True,
    )
    wall_collision_props = sim_utils.CollisionPropertiesCfg()
    wall_visual = sim_utils.PreviewSurfaceCfg(
        diffuse_color=wall_color, metallic=0.1,
    )

    for name, pos in [
        ("Wall_North", (0.0, room_size, wall_height / 2)),
        ("Wall_South", (0.0, -room_size, wall_height / 2)),
    ]:
        sz = (wall_length, wall_thickness, wall_height)
        setattr(cfg, name.lower(), AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=sz,
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        ))
    for name, pos in [
        ("Wall_East", (room_size, 0.0, wall_height / 2)),
        ("Wall_West", (-room_size, 0.0, wall_height / 2)),
    ]:
        sz = (wall_thickness, wall_length, wall_height)
        setattr(cfg, name.lower(), AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=sz,
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        ))

    # ==== 8 internal wall slots (RigidObject) ====
    internal_wall_visual = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.6, 0.55, 0.5), metallic=0.1,
    )
    for i, (length, width, height) in enumerate(WALL_SLOT_SPECS):
        setattr(cfg, f"wall_internal_{i}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Wall_Internal_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(length, width, height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=internal_wall_visual,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
        ))

    # ==== 100 obstacle slots (kinematic, hidden at z=-10) ====
    # ★ 行人高度 (1.6~1.8m)：確保 VLP16 LiDAR 可觀測
    _templates = [
        {"type": "cuboid",   "size": (0.5, 0.5, 1.7),   "color": (0.8, 0.2, 0.2)},
        {"type": "cylinder", "radius": 0.3, "height": 1.6, "color": (0.8, 0.8, 0.2)},
        {"type": "cuboid",   "size": (0.7, 0.7, 1.8),   "color": (0.2, 0.4, 0.8)},
        {"type": "cylinder", "radius": 0.25, "height": 1.7, "color": (0.2, 0.8, 0.2)},
        {"type": "cuboid",   "size": (0.6, 0.6, 1.6),   "color": (0.8, 0.2, 0.8)},
        {"type": "cylinder", "radius": 0.35, "height": 1.8, "color": (0.8, 0.5, 0.2)},
        {"type": "cuboid",   "size": (0.4, 0.4, 1.7),   "color": (0.2, 0.8, 0.8)},
        {"type": "cylinder", "radius": 0.2, "height": 1.6, "color": (0.5, 0.5, 0.5)},
        {"type": "cuboid",   "size": (0.55, 0.55, 1.8),  "color": (0.9, 0.9, 0.9)},
        {"type": "cylinder", "radius": 0.28, "height": 1.7, "color": (0.3, 0.3, 0.3)},
    ]
    obstacle_sizes: list[float] = []
    for i in range(MAX_OBSTACLES):
        tmpl = _templates[i % len(_templates)]
        if tmpl["type"] == "cuboid":
            spawn_cfg = sim_utils.CuboidCfg(
                size=tmpl["size"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tmpl["color"], metallic=0.2,
                ),
            )
            size_scalar = max(tmpl["size"][0], tmpl["size"][1])
        else:
            spawn_cfg = sim_utils.CylinderCfg(
                radius=tmpl["radius"], height=tmpl["height"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tmpl["color"], metallic=0.2,
                ),
            )
            size_scalar = tmpl["radius"] * 2.0

        setattr(cfg, f"obstacle_{i}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
        ))
        obstacle_sizes.append(size_scalar)

    set_obstacle_metadata(MAX_OBSTACLES, obstacle_sizes)

    return cfg


# ============================================================================
# Environment Configuration
# ============================================================================
@configclass
class DirectMARLChargeCfg(DirectMARLEnvCfg):
    """Phase 2: Multi-car DirectMARLEnv (WarpDrive-style)."""

    num_cars: int = DEFAULT_NUM_CARS

    # -- MARL agents (default N=6) --
    possible_agents = [f"car_{i}" for i in range(DEFAULT_NUM_CARS)]
    observation_spaces = {f"car_{i}": 79 for i in range(DEFAULT_NUM_CARS)}
    action_spaces = {f"car_{i}": 2 for i in range(DEFAULT_NUM_CARS)}
    state_space = 0

    # -- Timing --
    decimation = 20        # env dt = 0.2s (physics dt = 0.01s)
    episode_length_s = 45.0

    # -- Simulation --
    sim = sim_utils.SimulationCfg(dt=0.01, render_interval=20)

    # -- Scene (built by factory to avoid InteractiveScene rejecting int fields) --
    scene: InteractiveSceneCfg = _build_scene_cfg(num_envs=1024, env_spacing=22.0)

    # -- Action: discrete differential drive --
    num_action_bins: int = 19
    max_linear_velocity: float = 1.0
    max_linear_accel: float = 0.5
    max_angular_vel: float = 0.25 * math.pi

    # -- LiDAR processing --
    lidar_num_channels: int = 16
    lidar_num_horizontal: int = 360
    lidar_num_bins: int = 72
    lidar_r_max: float = 20.0
    lidar_r_robot: float = ROBOT_BODY_RADIUS
    lidar_r_min: float = 0.5
    lidar_z_filter: float = 0.5

    # -- Obstacle observation --
    obs_top_k: int = 10
    obs_max_obstacles: int = MAX_OBSTACLES
    obs_max_distance: float = 8.0
    obs_v_max: float = 1.5

    # -- Goal --
    goal_distance_range: tuple[float, float] = (2.0, 13.0)
    goal_boundary: float = 9.5
    goal_reach_threshold: float = ROBOT_BODY_RADIUS

    # -- Reward weights (NavRL) --
    reward_reaching_goal: float = 500.0
    reward_collision: float = -100.0
    reward_car_collision: float = -50.0
    reward_velocity_to_goal: float = 15.0
    reward_safety_log: float = 3.0
    reward_safe_progress: float = 20.0
    reward_time_penalty: float = -0.2

    # -- Obstacle spawning (fixed difficulty for Phase 2) --
    num_static_obstacles: int = 5
    num_dynamic_obstacles: int = 5
    obstacle_speed_range: tuple[float, float] = (0.3, 1.2)
    obstacle_area_limit: float = 8.0

    # -- Safety thresholds --
    collision_threshold: float = COLLISION_THRESHOLD
    body_radius: float = ROBOT_BODY_RADIUS

    # -- Car spawn --
    car_min_separation: float = 2.0  # min distance between cars at spawn

    # -- Obstacle control mode --
    use_learned_obstacles: bool = False  # True = obstacle agent 控制; False = scripted waypoint

    # -- WD per-phase randomization (set by curriculum) --
    obs_count_randomize: bool = False  # True = randint(1, N+1) per episode
    obs_size_rand: float = 0.0        # obstacle size randomization factor
    scene_bound_rand: float = 0.0     # scene boundary randomization factor

    def __post_init__(self):
        # Sync MARL agent lists with num_cars
        self.possible_agents = [f"car_{i}" for i in range(self.num_cars)]
        self.observation_spaces = {f"car_{i}": 79 for i in range(self.num_cars)}
        self.action_spaces = {f"car_{i}": 2 for i in range(self.num_cars)}


# ============================================================================
# Environment Implementation
# ============================================================================
class DirectMARLChargeEnv(DirectMARLEnv):
    """Phase 2: Multi-car DirectMARLEnv with WarpDrive-style continuous episodes."""

    cfg: DirectMARLChargeCfg

    def __init__(self, cfg: DirectMARLChargeCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = cfg.num_cars
        E = self.num_envs

        # -- Robot references --
        self.robots: list[Articulation] = [
            self.scene[f"car_{i}"] for i in range(N)
        ]
        self.lidars: list[MultiMeshRayCaster] = [
            self.scene.sensors[f"lidar_{i}"] for i in range(N)
        ]
        self.contacts: list[ContactSensor] = [
            self.scene.sensors[f"contact_{i}"] for i in range(N)
        ]

        # -- Active cars (1 charge + K virtual spots, adjustable per phase) --
        # n_active_cars: only cars [0..n_active-1] participate; rest are parked
        self.n_active_cars = N  # default: all active
        self._park_pos = torch.tensor([9.0, 9.0], device=self.device)  # corner

        # -- Per-car action state [E, N] --
        self._current_velocity = torch.zeros(E, N, device=self.device)
        self._a_bar = torch.zeros(E, N, device=self.device)
        self._omega_bar = torch.zeros(E, N, device=self.device)
        self._next_velocity = torch.zeros(E, N, device=self.device)
        self._next_omega = torch.zeros(E, N, device=self.device)
        self._center = cfg.num_action_bins // 2  # 9

        # -- Per-car goal state [E, N, 2] --
        self._goal_pos_w = torch.zeros(E, N, 2, device=self.device)
        self._prev_goal_dist = torch.zeros(E, N, device=self.device)

        # -- Obstacle state --
        self._obstacle_sizes = torch.zeros(cfg.obs_max_obstacles, device=self.device)
        self._obstacle_velocities = torch.zeros(E, cfg.obs_max_obstacles, 2, device=self.device)
        self._obstacle_goals = torch.zeros(E, cfg.obs_max_obstacles, 2, device=self.device)
        self._load_obstacle_sizes()

        # -- Collect obstacle rigid objects --
        self._obstacle_entities: list[RigidObject] = []
        for i in range(cfg.obs_max_obstacles):
            name = f"obstacle_{i}"
            if name in self.scene.rigid_objects:
                self._obstacle_entities.append(self.scene.rigid_objects[name])
            else:
                break
        self._num_obstacle_slots = len(self._obstacle_entities)

        # -- Respawn mask (exposed via extras for RNN hidden reset) --
        self._respawn_mask = torch.zeros(E, N, dtype=torch.bool, device=self.device)
        self.preprocess_target_dim = 8

        # -- Episode stats (for logging) --
        self._ep_goals_reached = torch.zeros(E, N, device=self.device)
        self._ep_collisions = torch.zeros(E, N, device=self.device)
        self._last_reward_components: list[dict] = [{} for _ in range(N)]
        self._ep_car_collisions = torch.zeros(E, N, device=self.device)

        # -- WD-aligned: steps since last respawn (for dies_at_birth tracking) --
        self._steps_since_respawn = torch.zeros(E, N, dtype=torch.long, device=self.device)

        # -- Rolling event counters (read & reset by training loop for curriculum) --
        self._iter_goals = 0
        self._iter_obs_collisions = 0
        self._iter_car_collisions = 0
        self._iter_born_died = 0  # WD: agents that collide on first step after respawn
        self._iter_timeouts = 0  # number of env-episodes that reached timeout

        print(
            f"[DirectMARLChargeEnv] num_envs={E}, num_cars={N}, "
            f"obstacle_slots={self._num_obstacle_slots}, step_dt={self.step_dt:.3f}s",
            flush=True,
        )

        # Disable OmniGraph controllers AFTER super().__init__() completes
        # (which calls sim.reset() → all OG graphs now registered)
        self._disable_omnigraph_controllers(None)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Spawn ground plane + clone environments."""
        from pxr import UsdGeom, UsdPhysics
        from omni.isaac.core.utils.stage import get_current_stage

        stage = get_current_stage()
        ground_prim = stage.DefinePrim("/World/ground", "Xform")
        plane_prim = stage.DefinePrim("/World/ground/CollisionPlane", "Plane")
        UsdGeom.Plane(plane_prim).GetAxisAttr().Set("Y")
        UsdPhysics.CollisionAPI.Apply(plane_prim)

        self.scene.clone_environments(copy_from_source=False)

        # OmniGraph will be disabled in __init__ after super().__init__ completes
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _disable_omnigraph_controllers(_stage):
        """Disable ALL OmniGraph action graphs to prevent error spam.

        Called once from first _pre_physics_step (after sim.reset() so all
        graphs are registered). Disables every graph containing '/Car_'.
        """
        try:
            import omni.graph.core as og
            graphs = og.get_all_graphs()
            disabled = 0
            for graph in graphs:
                path = graph.get_path_to_graph()
                if "/Car_" in path:
                    graph.set_disabled(True)
                    disabled += 1
            # Also disable sub-graphs
            for graph in graphs:
                for sub in graph.get_subgraphs():
                    path = sub.get_path_to_graph()
                    if "/Car_" in path:
                        sub.set_disabled(True)
                        disabled += 1
            print(f"[DirectMARLChargeEnv] Disabled {disabled} OmniGraph controllers", flush=True)
        except Exception as e:
            print(f"[DirectMARLChargeEnv] OmniGraph disable failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Decode MultiDiscrete [accel_idx, omega_idx] → velocity for each car."""
        center = self._center
        dt = self.step_dt
        a_max = self.cfg.max_linear_accel
        v_max = self.cfg.max_linear_velocity

        for i in range(self.cfg.num_cars):
            raw = actions[f"car_{i}"]  # [E, 2]

            accel_idx = raw[:, 0].round().long().clamp(0, self.cfg.num_action_bins - 1)
            omega_idx = raw[:, 1].round().long().clamp(0, self.cfg.num_action_bins - 1)

            ratio_linear = (accel_idx.float() - center) / center
            ratio_angular = (omega_idx.float() - center) / center

            v = self._current_velocity[:, i]
            allow_max = torch.min(torch.full_like(v, a_max), (v_max - v) / dt)
            allow_min = torch.max(torch.full_like(v, -a_max), (-v_max - v) / dt)

            actual_accel = torch.where(
                ratio_linear >= 0,
                ratio_linear * allow_max,
                -ratio_linear * allow_min,
            ).clamp(allow_min, allow_max)

            actual_omega = ratio_angular * self.cfg.max_angular_vel
            next_vel = (v + actual_accel * dt).clamp(-v_max, v_max)

            self._current_velocity[:, i] = next_vel
            self._a_bar[:, i] = actual_accel / a_max
            self._omega_bar[:, i] = ratio_angular
            self._next_velocity[:, i] = next_vel
            self._next_omega[:, i] = actual_omega

        # Update dynamic obstacle velocities before physics sim
        # (skip if obstacle agent controls them externally)
        if not self.cfg.use_learned_obstacles:
            self._move_dynamic_obstacles()

    def _apply_action(self) -> None:
        """Write velocity to sim for all cars (called every physics substep)."""
        for i in range(self.cfg.num_cars):
            robot = self.robots[i]
            quat_w = robot.data.root_quat_w
            local_vel = torch.zeros(self.num_envs, 3, device=self.device)
            local_vel[:, 0] = self._next_velocity[:, i]
            global_vel = quat_apply(quat_w, local_vel)

            root_vel = torch.zeros(self.num_envs, 6, device=self.device)
            root_vel[:, :3] = global_vel
            root_vel[:, 5] = self._next_omega[:, i]
            robot.write_root_velocity_to_sim(root_vel)

    # ------------------------------------------------------------------
    # Observations (139D per car)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Build 79D obs for each car: ego(4) + goal(2) + lidar(72) + time(1).

        WD-aligned: 只用 LiDAR angle bins 感知環境，不用分離的 TopK obstacle 特徵。
        WD Spot: Sweep_angle 36 bins / 我們: VLP-16 → 72 bins。
        """
        E = self.num_envs
        time_ratio = 1.0 - self.episode_length_buf.float() / max(self.max_episode_length - 1, 1)
        time_feat = time_ratio.unsqueeze(-1)  # [E, 1]

        obs_dict: dict[str, torch.Tensor] = {}
        for i in range(self.cfg.num_cars):
            robot = self.robots[i]

            # Ego (4D)
            ego = torch.stack([
                self._a_bar[:, i],
                self._current_velocity[:, i] / self.cfg.max_linear_velocity,
                self._omega_bar[:, i],
                torch.full((E,), self.cfg.body_radius, device=self.device),
            ], dim=-1)  # [E, 4]

            # Goal in robot frame (2D)
            robot_pos = robot.data.root_pos_w[:, :2]
            goal_diff = self._goal_pos_w[:, i] - robot_pos
            yaw = self._quat_to_yaw(robot.data.root_quat_w)
            cos_yaw, sin_yaw = torch.cos(-yaw), torch.sin(-yaw)
            goal_robot = torch.stack([
                goal_diff[:, 0] * cos_yaw - goal_diff[:, 1] * sin_yaw,
                goal_diff[:, 0] * sin_yaw + goal_diff[:, 1] * cos_yaw,
            ], dim=-1)  # [E, 2]

            # LiDAR bins (72D)
            lidar_bins = self._compute_lidar_bins(i)

            obs = torch.cat([ego, goal_robot, lidar_bins, time_feat], dim=-1)  # [E, 79]
            obs_dict[f"car_{i}"] = obs

        return obs_dict

    # ------------------------------------------------------------------
    # Rewards (NavRL + car-to-car + WarpDrive respawn)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """NavRL dense rewards for each car. Respawn on collision/goal (no episode termination)."""
        N = self.cfg.num_cars
        E = self.num_envs
        reward_dict: dict[str, torch.Tensor] = {}
        self._respawn_mask.zero_()

        for i in range(N):
            robot = self.robots[i]
            robot_pos = robot.data.root_pos_w[:, :2]
            robot_vel = robot.data.root_lin_vel_w[:, :2]
            d_curr = torch.norm(self._goal_pos_w[:, i] - robot_pos, dim=1)

            reward = torch.zeros(E, device=self.device)

            # --- Event detection ---
            goal_reached = d_curr < self.cfg.goal_reach_threshold
            obs_collided = self._check_obstacle_collision(i)
            car_collided = self._check_car_collision(i)

            # WD-aligned: track steps since respawn for dies_at_birth
            self._steps_since_respawn[:, i] += 1

            # 1. Goal bonus
            reward += goal_reached.float() * self.cfg.reward_reaching_goal

            # 2. Obstacle collision penalty
            reward += obs_collided.float() * self.cfg.reward_collision

            # 3. Car-to-car collision penalty
            reward += car_collided.float() * self.cfg.reward_car_collision

            # 4. Velocity to goal (with safety gate)
            goal_diff = self._goal_pos_w[:, i] - robot_pos
            goal_dist = d_curr.clamp(min=1e-6)
            goal_dir = goal_diff / goal_dist.unsqueeze(1)
            v_toward = (robot_vel * goal_dir).sum(dim=1)
            v_reward = (v_toward / self.cfg.max_linear_velocity).clamp(-1, 1)
            far_enough = (d_curr > self.cfg.goal_reach_threshold).float()
            d_safe = self._get_d_safe(i)
            v_gate = ((d_safe - 1.0) / (2.5 - 1.0 + 1e-6)).clamp(0.0, 1.0)
            reward += v_reward * far_enough * v_gate * self.cfg.reward_velocity_to_goal

            # 5. Safety log distance
            raw_log = torch.log(d_safe).clamp(-6.0, 2.0)
            speed = torch.norm(robot_vel, dim=-1)
            speed_factor = (speed / 0.1).clamp(0.1, 1.0)
            reward += raw_log * speed_factor * self.cfg.reward_safety_log

            # 6. Safe progress (PBRS)
            progress = (self._prev_goal_dist[:, i] - d_curr).clamp(-1.0, 1.0)
            gate = torch.where(
                d_safe < 0.8,
                torch.full_like(d_safe, -0.5),
                ((d_safe - 0.8) / (2.0 - 0.8 + 1e-6)).clamp(0.0, 1.0),
            )
            reward += progress * gate * self.cfg.reward_safe_progress
            self._prev_goal_dist[:, i] = d_curr

            # 7. Time penalty
            reward += self.cfg.reward_time_penalty

            reward_dict[f"car_{i}"] = torch.nan_to_num(reward, nan=0.0)

            # WD-aligned: dies_at_birth = collided on first step after respawn
            any_collided = obs_collided | car_collided
            born_died = any_collided & (self._steps_since_respawn[:, i] <= 1)

            # --- Expose reward components for WandB logging (WD-aligned) ---
            self._last_reward_components[i] = {
                "goal_bonus": (goal_reached.float() * self.cfg.reward_reaching_goal).mean().item(),
                "obs_collision": (obs_collided.float() * self.cfg.reward_collision).mean().item(),
                "car_collision": (car_collided.float() * self.cfg.reward_car_collision).mean().item(),
                "velocity_to_goal": (v_reward * far_enough * v_gate * self.cfg.reward_velocity_to_goal).mean().item(),
                "safety_log": (raw_log * speed_factor * self.cfg.reward_safety_log).mean().item(),
                "safe_progress": (progress * gate * self.cfg.reward_safe_progress).mean().item(),
                "time_penalty": self.cfg.reward_time_penalty,
                "hit_probability": obs_collided.float().mean().item(),
                "goal_reach_probability": goal_reached.float().mean().item(),
                "born_died_probability": born_died.float().mean().item(),
                "d_safe_mean": d_safe.mean().item(),
                "speed_mean": speed.mean().item(),
                "v_toward_mean": v_toward.mean().item(),
            }

            # --- WarpDrive-style: respawn on collision or goal ---
            needs_respawn = goal_reached | any_collided
            self._respawn_mask[:, i] = needs_respawn
            if needs_respawn.any():
                respawn_ids = needs_respawn.nonzero(as_tuple=False).squeeze(-1)
                # Track born_died before respawn resets the counter
                n_born_died = born_died[respawn_ids].sum().item()
                self._respawn_car(i, respawn_ids)
                self._steps_since_respawn[respawn_ids, i] = 0  # reset after respawn
                # Track stats
                n_goals = goal_reached[respawn_ids].sum().item()
                n_obs_coll = obs_collided[respawn_ids].sum().item()
                n_car_coll = car_collided[respawn_ids].sum().item()
                self._ep_goals_reached[respawn_ids, i] += goal_reached[respawn_ids].float()
                self._ep_collisions[respawn_ids, i] += obs_collided[respawn_ids].float()
                self._ep_car_collisions[respawn_ids, i] += car_collided[respawn_ids].float()
                # Rolling counters for curriculum
                self._iter_goals += int(n_goals)
                self._iter_obs_collisions += int(n_obs_coll)
                self._iter_car_collisions += int(n_car_coll)
                self._iter_born_died += int(n_born_died)

        # Expose respawn mask for RNN hidden state reset
        self.extras["respawn_mask"] = self._respawn_mask.clone()

        return reward_dict

    # ------------------------------------------------------------------
    # Done signals (WarpDrive-style: only shared timeout)
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated_dict: dict[str, torch.Tensor] = {}
        timeout_dict: dict[str, torch.Tensor] = {}
        for i in range(self.cfg.num_cars):
            agent = f"car_{i}"
            terminated_dict[agent] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device,
            )
            timeout_dict[agent] = timed_out

        # Log per-episode metrics on timeout
        if timed_out.any():
            self._iter_timeouts += int(timed_out.sum().item())
            mask = timed_out
            for i in range(self.cfg.num_cars):
                agent = f"car_{i}"
                self.extras[agent] = {
                    "log": {
                        "goals_per_episode": self._ep_goals_reached[mask, i].mean().item(),
                        "collisions_per_episode": self._ep_collisions[mask, i].mean().item(),
                        "car_collisions_per_episode": self._ep_car_collisions[mask, i].mean().item(),
                    }
                }

        return terminated_dict, timeout_dict

    # ------------------------------------------------------------------
    # States (unused — state_space=0)
    # ------------------------------------------------------------------
    def _get_states(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    # ------------------------------------------------------------------
    # Reset (full episode reset on timeout)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robots[0]._ALL_INDICES
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # Reset active cars; park inactive ones
        for i in range(self.cfg.num_cars):
            if i < self.n_active_cars:
                self._reset_car_pose(i, env_ids)
                self._randomize_goal(i, env_ids)
            else:
                self._park_car(i, env_ids)

        # Reset action state
        self._current_velocity[env_ids] = 0.0
        self._a_bar[env_ids] = 0.0
        self._omega_bar[env_ids] = 0.0

        # Reset obstacles
        self._randomize_obstacles(env_ids)

        # Reset episode stats
        self._ep_goals_reached[env_ids] = 0.0
        self._ep_collisions[env_ids] = 0.0
        self._ep_car_collisions[env_ids] = 0.0
        self._steps_since_respawn[env_ids] = 0

    # ------------------------------------------------------------------
    # Car pose reset (used by both _reset_idx and _respawn_car)
    # ------------------------------------------------------------------
    def _reset_car_pose(self, car_idx: int, env_ids: torch.Tensor):
        """Place car at random XY with min separation from other cars."""
        robot = self.robots[car_idx]
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        default_state = robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += origins

        # Random XY in [-7, 7] with min separation
        xy = torch.empty(n, 2, device=self.device).uniform_(-7, 7)
        for _ in range(20):
            too_close = torch.zeros(n, dtype=torch.bool, device=self.device)
            for j in range(self.cfg.num_cars):
                if j == car_idx:
                    continue
                other_pos = self.robots[j].data.root_pos_w[env_ids, :2]
                other_local = other_pos - origins[:, :2]
                dist = torch.norm(xy - other_local, dim=1)
                too_close |= dist < self.cfg.car_min_separation
            if not too_close.any():
                break
            # Resample only conflicting positions
            m = too_close.sum()
            xy[too_close] = torch.empty(m, 2, device=self.device).uniform_(-7, 7)

        default_state[:, 0] += xy[:, 0]
        default_state[:, 1] += xy[:, 1]
        default_state[:, 2] = 0.0

        # Random yaw
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        half = yaw * 0.5
        default_state[:, 3] = torch.cos(half)  # w
        default_state[:, 4] = 0.0              # x
        default_state[:, 5] = 0.0              # y
        default_state[:, 6] = torch.sin(half)  # z
        default_state[:, 7:] = 0.0

        robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

    def _park_car(self, car_idx: int, env_ids: torch.Tensor):
        """Park inactive car at corner, zero velocity."""
        robot = self.robots[car_idx]
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        pose = robot.data.default_root_state[env_ids].clone()
        pose[:, :3] += origins
        pose[:, 0] += self._park_pos[0]
        pose[:, 1] += self._park_pos[1]
        pose[:, 2] = 0.0
        pose[:, 3] = 1.0
        pose[:, 4:7] = 0.0
        pose[:, 7:] = 0.0
        robot.write_root_pose_to_sim(pose[:, :7], env_ids)
        robot.write_root_velocity_to_sim(pose[:, 7:], env_ids)
        self._current_velocity[env_ids, car_idx] = 0.0

    def _respawn_car(self, car_idx: int, env_ids: torch.Tensor):
        """WarpDrive-style mid-episode respawn: new position + new goal."""
        self._reset_car_pose(car_idx, env_ids)
        self._current_velocity[env_ids, car_idx] = 0.0
        self._a_bar[env_ids, car_idx] = 0.0
        self._omega_bar[env_ids, car_idx] = 0.0
        self._randomize_goal(car_idx, env_ids)

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------
    def get_and_reset_event_counts(self) -> tuple[int, int, int, int, int]:
        """Return (goals, obs_collisions, car_collisions, born_died, timeouts) since last call, then reset."""
        g = self._iter_goals
        oc = self._iter_obs_collisions
        cc = self._iter_car_collisions
        bd = self._iter_born_died
        t = self._iter_timeouts
        self._iter_goals = 0
        self._iter_obs_collisions = 0
        self._iter_car_collisions = 0
        self._iter_born_died = 0
        self._iter_timeouts = 0
        return g, oc, cc, bd, t

    # ------------------------------------------------------------------
    # Obstacle agent interface (for learned obstacle policy)
    # ------------------------------------------------------------------
    def build_obstacle_obs(self, n_active: int | None = None) -> torch.Tensor:
        """Build 9D obs for each active obstacle: [E, N_obs, 9].

        Per-obstacle layout:
          [0:2] own_local_xy — position in env-local frame
          [2:4] own_vel — cached velocity (vx, vy)
          [4:6] robot_rel — nearest car relative position (car_xy - obs_xy)
          [6:8] robot_vel — nearest car velocity
          [8]   d_wall — min distance to boundary walls
        """
        E = self.num_envs
        if n_active is None:
            n_active = self.cfg.num_static_obstacles + self.cfg.num_dynamic_obstacles
        n_active = min(n_active, self._num_obstacle_slots)

        obs = torch.zeros(E, n_active, 9, device=self.device)
        origins = self.scene.env_origins[:, :2]
        boundary = self.cfg.goal_boundary + 0.5  # room_size

        # Collect all car positions for nearest-car computation
        all_car_pos = torch.stack(
            [r.data.root_pos_w[:, :2] for r in self.robots], dim=1,
        )  # [E, N_cars, 2]
        all_car_vel = torch.stack(
            [r.data.root_lin_vel_w[:, :2] for r in self.robots], dim=1,
        )  # [E, N_cars, 2]

        for i in range(n_active):
            entity = self._obstacle_entities[i]
            pos_w = entity.data.root_pos_w  # [E, 3]
            visible = pos_w[:, 2] > 0.0  # hidden at z=-10

            # Own local position
            local_xy = pos_w[:, :2] - origins
            obs[:, i, 0:2] = local_xy * visible.unsqueeze(1).float()

            # Own velocity (cached)
            obs[:, i, 2:4] = self._obstacle_velocities[:, i] * visible.unsqueeze(1).float()

            # Nearest car: relative position and velocity
            obs_xy = pos_w[:, :2].unsqueeze(1)  # [E, 1, 2]
            dist_to_cars = torch.norm(all_car_pos - obs_xy, dim=2)  # [E, N_cars]
            nearest_idx = dist_to_cars.argmin(dim=1)  # [E]
            nearest_car_pos = all_car_pos[torch.arange(E, device=self.device), nearest_idx]
            nearest_car_vel = all_car_vel[torch.arange(E, device=self.device), nearest_idx]
            rel_pos = nearest_car_pos - pos_w[:, :2]
            obs[:, i, 4:6] = rel_pos * visible.unsqueeze(1).float()
            obs[:, i, 6:8] = nearest_car_vel * visible.unsqueeze(1).float()

            # Min distance to boundary
            d_left = local_xy[:, 0] + boundary
            d_right = boundary - local_xy[:, 0]
            d_bottom = local_xy[:, 1] + boundary
            d_top = boundary - local_xy[:, 1]
            d_wall = torch.stack([d_left, d_right, d_bottom, d_top], dim=1).min(dim=1).values
            obs[:, i, 8] = d_wall.clamp(min=0.0) * visible.float()

        return obs

    def apply_obstacle_actions(
        self, actions: torch.Tensor, n_active: int | None = None,
        speed_limit: float = 0.8,
    ) -> None:
        """Apply learned obstacle velocity actions: [E, N_obs, 2] in [-1, 1].

        Kinematic position integration with geofence bounce.
        """
        E = self.num_envs
        if n_active is None:
            n_active = self.cfg.num_static_obstacles + self.cfg.num_dynamic_obstacles
        n_active = min(n_active, self._num_obstacle_slots)
        dt = self.step_dt
        origins = self.scene.env_origins
        boundary = self.cfg.obstacle_area_limit

        for i in range(n_active):
            entity = self._obstacle_entities[i]
            pos_w = entity.data.root_pos_w.clone()
            visible = pos_w[:, 2] > 0.0

            # Scale action to velocity
            vel = actions[:, i] * speed_limit  # [E, 2]

            # Integrate position (local frame)
            local_xy = pos_w[:, :2] - origins[:, :2]
            new_local = local_xy + vel * dt

            # Geofence: soft bounce at boundary
            for dim in range(2):
                over = new_local[:, dim] > boundary
                under = new_local[:, dim] < -boundary
                new_local[over, dim] = 2 * boundary - new_local[over, dim]
                new_local[under, dim] = -2 * boundary - new_local[under, dim]
            new_local = new_local.clamp(-boundary, boundary)

            # Write back
            new_pos_w = new_local + origins[:, :2]
            pose = torch.zeros(E, 7, device=self.device)
            pose[:, :2] = new_pos_w
            pose[:, 2] = torch.where(visible, pos_w[:, 2], torch.full_like(pos_w[:, 2], HIDDEN_Z))
            pose[:, 3] = 1.0
            entity.write_root_pose_to_sim(pose)

            # Cache velocity for obs
            self._obstacle_velocities[:, i] = vel * visible.unsqueeze(1).float()

    # ------------------------------------------------------------------
    # Goal management (per-car)
    # ------------------------------------------------------------------
    def _randomize_goal(self, car_idx: int, env_ids: torch.Tensor):
        """Random goal for a specific car in specified environments."""
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids, :2]
        robot_pos = self.robots[car_idx].data.root_pos_w[env_ids, :2]
        rng = self.cfg.goal_distance_range
        boundary = self.cfg.goal_boundary

        for _ in range(50):
            angle = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
            dist = torch.empty(n, device=self.device).uniform_(rng[0], rng[1])
            goal_local = torch.stack([
                dist * torch.cos(angle), dist * torch.sin(angle),
            ], dim=-1)
            goal_w = robot_pos + goal_local
            goal_local_env = goal_w - origins
            in_bounds = (goal_local_env.abs() < boundary).all(dim=-1)
            if in_bounds.all():
                break
            oob = ~in_bounds
            goal_local_env[oob] = goal_local_env[oob].clamp(-boundary, boundary)
            goal_w[oob] = origins[oob] + goal_local_env[oob]

        self._goal_pos_w[env_ids, car_idx] = goal_w
        self._prev_goal_dist[env_ids, car_idx] = torch.norm(goal_w - robot_pos, dim=1)

    # ------------------------------------------------------------------
    # Obstacle management (shared across all cars)
    # ------------------------------------------------------------------
    def _load_obstacle_sizes(self):
        _templates = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.4, 0.4, 0.55, 0.56]
        for i in range(min(self.cfg.obs_max_obstacles, len(self._obstacle_sizes))):
            self._obstacle_sizes[i] = _templates[i % len(_templates)]

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """Place static + dynamic obstacles randomly.

        WD-style: when obs_count_randomize=True, actual count is randint(1, N+1).
        """
        n = len(env_ids)
        n_static = self.cfg.num_static_obstacles
        n_dynamic = self.cfg.num_dynamic_obstacles
        # WD: self.num_obstacle = int(num_obstacle * rand + 1)
        if self.cfg.obs_count_randomize and n_dynamic > 0:
            import random
            n_dynamic = random.randint(1, n_dynamic)
        n_total = n_static + n_dynamic
        limit = self.cfg.obstacle_area_limit
        origins = self.scene.env_origins[env_ids]

        for i in range(self._num_obstacle_slots):
            entity = self._obstacle_entities[i]
            if i < n_total:
                pos = torch.zeros(n, 3, device=self.device)
                pos[:, 0] = torch.empty(n, device=self.device).uniform_(-limit, limit)
                pos[:, 1] = torch.empty(n, device=self.device).uniform_(-limit, limit)
                pos[:, 2] = 0.5

                # Ensure min distance from all cars
                for ci in range(self.cfg.num_cars):
                    car_pos_local = (
                        self.robots[ci].data.root_pos_w[env_ids, :2] - origins[:, :2]
                    )
                    delta = pos[:, :2] - car_pos_local
                    too_close = torch.norm(delta, dim=1) < 1.5
                    if too_close.any():
                        push_dir = delta[too_close] / (
                            torch.norm(delta[too_close], dim=1, keepdim=True) + 1e-6
                        )
                        pos[too_close, :2] = car_pos_local[too_close] + push_dir * 1.5

                pos_w = pos.clone()
                pos_w[:, :2] += origins[:, :2]
                pos_w[:, 2] += origins[:, 2]

                pose = torch.zeros(n, 7, device=self.device)
                pose[:, :3] = pos_w
                pose[:, 3] = 1.0
                entity.write_root_pose_to_sim(pose, env_ids)

                if i >= n_static:
                    speed_min, speed_max = self.cfg.obstacle_speed_range
                    speed = torch.empty(n, device=self.device).uniform_(speed_min, speed_max)
                    angle = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
                    vel = torch.zeros(n, 6, device=self.device)
                    vel[:, 0] = speed * torch.cos(angle)
                    vel[:, 1] = speed * torch.sin(angle)
                    entity.write_root_velocity_to_sim(vel, env_ids)
                    self._obstacle_velocities[env_ids, i, 0] = vel[:, 0]
                    self._obstacle_velocities[env_ids, i, 1] = vel[:, 1]
                    self._obstacle_goals[env_ids, i, 0] = torch.empty(
                        n, device=self.device,
                    ).uniform_(-limit, limit)
                    self._obstacle_goals[env_ids, i, 1] = torch.empty(
                        n, device=self.device,
                    ).uniform_(-limit, limit)
                else:
                    vel = torch.zeros(n, 6, device=self.device)
                    entity.write_root_velocity_to_sim(vel, env_ids)
                    self._obstacle_velocities[env_ids, i] = 0.0
            else:
                pose = torch.zeros(n, 7, device=self.device)
                pose[:, :2] += origins[:, :2]
                pose[:, 2] = origins[:, 2] + HIDDEN_Z
                pose[:, 3] = 1.0
                entity.write_root_pose_to_sim(pose, env_ids)

    def _move_dynamic_obstacles(self):
        """Update dynamic obstacle velocities toward their goals."""
        n_static = self.cfg.num_static_obstacles
        n_dynamic = self.cfg.num_dynamic_obstacles
        limit = self.cfg.obstacle_area_limit
        bound = limit + 1.0

        for i in range(n_static, n_static + n_dynamic):
            if i >= self._num_obstacle_slots:
                break
            entity = self._obstacle_entities[i]
            pos_w = entity.data.root_pos_w.clone()
            pos_local = pos_w[:, :2] - self.scene.env_origins[:, :2]

            goal = self._obstacle_goals[:, i]
            dist_to_goal = torch.norm(pos_local - goal, dim=1)
            reached = dist_to_goal < 0.5
            if reached.any():
                m = reached.sum()
                self._obstacle_goals[reached, i, 0] = torch.empty(
                    m, device=self.device,
                ).uniform_(-limit, limit)
                self._obstacle_goals[reached, i, 1] = torch.empty(
                    m, device=self.device,
                ).uniform_(-limit, limit)

            goal = self._obstacle_goals[:, i]
            direction = goal - pos_local
            dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            direction = direction / dist
            speed = self._obstacle_velocities[:, i].norm(dim=1, keepdim=True).clamp(
                min=self.cfg.obstacle_speed_range[0],
            )
            new_vel = direction * speed

            oob = pos_local.abs() > bound
            if oob.any():
                bounce_dir = -pos_local.sign()
                new_vel[oob[:, 0], 0] = (
                    bounce_dir[oob[:, 0], 0].abs() * speed[oob[:, 0]].squeeze()
                )
                new_vel[oob[:, 1], 1] = (
                    bounce_dir[oob[:, 1], 1].abs() * speed[oob[:, 1]].squeeze()
                )

            self._obstacle_velocities[:, i] = new_vel
            vel_6d = torch.zeros(self.num_envs, 6, device=self.device)
            vel_6d[:, :2] = new_vel
            entity.write_root_velocity_to_sim(vel_6d)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def _check_obstacle_collision(self, car_idx: int) -> torch.Tensor:
        """Check car vs obstacles + walls (contact sensor + geometric)."""
        E = self.num_envs
        collided = torch.zeros(E, dtype=torch.bool, device=self.device)

        robot = self.robots[car_idx]
        robot_pos = robot.data.root_pos_w[:, :2]

        # Contact sensor
        contact = self.contacts[car_idx]
        net_forces = contact.data.net_forces_w
        force_mag = torch.norm(net_forces, dim=-1).max(dim=-1).values
        collided |= force_mag > 0.1

        # Geometric — obstacle center distance
        n_active = self.cfg.num_static_obstacles + self.cfg.num_dynamic_obstacles
        for j in range(min(n_active, self._num_obstacle_slots)):
            obs_pos = self._obstacle_entities[j].data.root_pos_w
            visible = obs_pos[:, 2] > 0.0
            dist = torch.norm(obs_pos[:, :2] - robot_pos, dim=1)
            r_obs = self._obstacle_sizes[min(j, len(self._obstacle_sizes) - 1)]
            threshold = self.cfg.body_radius + r_obs * 0.5 + COLLISION_BUFFER
            collided |= (dist < threshold) & visible

        # Wall collision (AABB)
        local_pos = robot_pos - self.scene.env_origins[:, :2]
        wall_dist = self.cfg.goal_boundary + 0.5
        collided |= (local_pos[:, 0].abs() > wall_dist) | (local_pos[:, 1].abs() > wall_dist)

        return collided

    def _check_car_collision(self, car_idx: int) -> torch.Tensor:
        """Check car vs other cars (geometric center distance)."""
        E = self.num_envs
        collided = torch.zeros(E, dtype=torch.bool, device=self.device)

        my_pos = self.robots[car_idx].data.root_pos_w[:, :2]
        car_threshold = self.cfg.body_radius * 2 + COLLISION_BUFFER

        for j in range(self.cfg.num_cars):
            if j == car_idx:
                continue
            other_pos = self.robots[j].data.root_pos_w[:, :2]
            dist = torch.norm(my_pos - other_pos, dim=1)
            collided |= dist < car_threshold

        return collided

    # ------------------------------------------------------------------
    # LiDAR processing (per-car)
    # ------------------------------------------------------------------
    def _compute_lidar_bins(self, car_idx: int) -> torch.Tensor:
        """VLP-16 5760 rays → 72 bins (normalized [0, 1])."""
        lidar = self.lidars[car_idx]
        sensor_pos = lidar.data.pos_w         # [E, 3]
        hit_points = lidar.data.ray_hits_w    # [E, 5760, 3]

        sensor_xy = sensor_pos[:, :2].unsqueeze(1)
        hits_xy = hit_points[:, :, :2]
        dist_2d = torch.norm(hits_xy - sensor_xy, dim=-1)  # [E, 5760]

        # Z-filter: exclude hidden obstacles (z=-10)
        if self.cfg.lidar_z_filter > 0:
            sensor_z = sensor_pos[:, 2:3].unsqueeze(1)
            hit_z = hit_points[:, :, 2:3]
            z_diff = (hit_z - sensor_z).abs().squeeze(-1)
            invalid_z = z_diff > self.cfg.lidar_z_filter
            dist_2d = torch.where(invalid_z, self.cfg.lidar_r_max, dist_2d)

        dist_2d = torch.where(torch.isfinite(dist_2d), dist_2d, self.cfg.lidar_r_max)
        dist_2d = dist_2d.clamp(max=self.cfg.lidar_r_max)

        # Min range filter (excludes self-hits from car's own mesh)
        if self.cfg.lidar_r_min > 0:
            dist_2d = torch.where(
                dist_2d < self.cfg.lidar_r_min, self.cfg.lidar_r_max, dist_2d,
            )

        # Min pooling: [E, 16, 72, 5] → [E, 72]
        rays_per_bin = self.cfg.lidar_num_horizontal // self.cfg.lidar_num_bins
        x = dist_2d.view(
            self.num_envs, self.cfg.lidar_num_channels,
            self.cfg.lidar_num_bins, rays_per_bin,
        )
        x = x.min(dim=3).values.min(dim=1).values  # [E, 72]

        # Safe margin & normalize
        x = (x - self.cfg.lidar_r_robot).clamp(min=0.0) / self.cfg.lidar_r_max
        return x

    # ------------------------------------------------------------------
    # TopK obstacles (per-car, includes other cars)
    # ------------------------------------------------------------------
    def _compute_topk_obstacles(
        self, car_idx: int, obstacles_only: bool = False,
    ) -> torch.Tensor:
        """Top-K nearest objects in body frame: [E, K*6] = [E, 60].

        Object pool = obstacle slots (+ other N-1 cars unless obstacles_only).
        Each object: [dx_body, dy_body, vx_body, vy_body, radius, valid_mask].
        """
        E = self.num_envs
        K = self.cfg.obs_top_k
        N_cars = self.cfg.num_cars
        device = self.device

        robot = self.robots[car_idx]
        robot_pos = robot.data.root_pos_w[:, :2]
        robot_vel = robot.data.root_lin_vel_w[:, :2]
        yaw = self._quat_to_yaw(robot.data.root_quat_w)
        cos_yaw, sin_yaw = torch.cos(-yaw), torch.sin(-yaw)

        n_obs_slots = min(self._num_obstacle_slots, self.cfg.obs_max_obstacles)
        n_other_cars = 0 if obstacles_only else N_cars - 1
        n_total = n_obs_slots + n_other_cars

        if n_total == 0:
            return torch.zeros(E, K * 6, device=device)

        all_pos = torch.zeros(E, n_total, 2, device=device)
        all_vel = torch.zeros(E, n_total, 2, device=device)
        all_valid = torch.zeros(E, n_total, dtype=torch.bool, device=device)
        all_sizes = torch.zeros(n_total, device=device)

        # --- Obstacle slots ---
        for i in range(n_obs_slots):
            entity = self._obstacle_entities[i]
            pos_w = entity.data.root_pos_w
            all_pos[:, i] = pos_w[:, :2]
            all_valid[:, i] = pos_w[:, 2] > 0.0
            all_vel[:, i] = self._obstacle_velocities[:, i]
            all_sizes[i] = self._obstacle_sizes[i]

        # --- Other cars (skipped when obstacles_only) ---
        if not obstacles_only:
            other_idx = 0
            for j in range(N_cars):
                if j == car_idx:
                    continue
                slot = n_obs_slots + other_idx
                other = self.robots[j]
                all_pos[:, slot] = other.data.root_pos_w[:, :2]
                all_vel[:, slot] = other.data.root_lin_vel_w[:, :2]
                all_valid[:, slot] = True
                all_sizes[slot] = ROBOT_BODY_RADIUS * 2  # diameter
                other_idx += 1

        # Relative position in world frame
        delta_w = all_pos - robot_pos.unsqueeze(1)  # [E, S, 2]
        dist = torch.norm(delta_w, dim=2)            # [E, S]

        # Rotate to body frame
        dx_b = delta_w[:, :, 0] * cos_yaw.unsqueeze(1) - delta_w[:, :, 1] * sin_yaw.unsqueeze(1)
        dy_b = delta_w[:, :, 0] * sin_yaw.unsqueeze(1) + delta_w[:, :, 1] * cos_yaw.unsqueeze(1)

        # Relative velocity in body frame
        rel_vel_w = all_vel - robot_vel.unsqueeze(1)
        vx_b = rel_vel_w[:, :, 0] * cos_yaw.unsqueeze(1) - rel_vel_w[:, :, 1] * sin_yaw.unsqueeze(1)
        vy_b = rel_vel_w[:, :, 0] * sin_yaw.unsqueeze(1) + rel_vel_w[:, :, 1] * cos_yaw.unsqueeze(1)

        # Invalidate far / hidden
        dist_sort = dist.clone()
        dist_sort[~all_valid] = 1e6
        dist_sort[dist > self.cfg.obs_max_distance] = 1e6

        # TopK selection
        if n_total >= K:
            _, topk_idx = torch.topk(dist_sort, k=K, dim=1, largest=False)
        else:
            _, sort_idx = torch.sort(dist_sort, dim=1)
            pad = sort_idx[:, :1].expand(-1, K - n_total)
            topk_idx = torch.cat([sort_idx, pad], dim=1)

        # Gather & normalize
        topk_dx = torch.gather(dx_b, 1, topk_idx) / self.cfg.obs_max_distance
        topk_dy = torch.gather(dy_b, 1, topk_idx) / self.cfg.obs_max_distance
        topk_vx = torch.gather(vx_b, 1, topk_idx) / self.cfg.obs_v_max
        topk_vy = torch.gather(vy_b, 1, topk_idx) / self.cfg.obs_v_max
        topk_r = torch.gather(
            all_sizes.unsqueeze(0).expand(E, -1), 1, topk_idx,
        )
        topk_valid = torch.gather(all_valid.long(), 1, topk_idx).float()

        if n_total < K:
            topk_valid[:, n_total:] = 0.0

        # Zero out invalid
        mask = topk_valid
        topk_dx = topk_dx * mask
        topk_dy = topk_dy * mask
        topk_vx = topk_vx * mask
        topk_vy = topk_vy * mask
        topk_r = topk_r * mask

        feat = torch.stack([topk_dx, topk_dy, topk_vx, topk_vy, topk_r, mask], dim=2)
        return feat.reshape(E, K * 6)

    def build_preprocess_targets(self, n_active: int | None = None) -> torch.Tensor:
        """Build WD-style privileged geometry targets: [E, N_active, 8].

        Layout per car:
          [near_1_x, near_1_y, near_1_d,
           near_2_x, near_2_y, near_2_d,
           near_1_vx, near_1_vy]

        Positions are body-frame surface-point coordinates in meters, not raw
        center-to-center offsets. Velocity is the nearest object's relative
        body-frame velocity in m/s. These are training-only targets.

        Pool = obstacles only (no other cars). Policy obs cannot see other
        cars (LiDAR excludes Car_*, no TopK for cars), so including them
        would make aux targets partially unobservable.
        """
        if n_active is None:
            n_active = self.cfg.num_cars
        n_active = min(n_active, self.cfg.num_cars)

        E = self.num_envs
        device = self.device
        out = torch.zeros(E, n_active, self.preprocess_target_dim, device=device)

        for car_idx in range(n_active):
            topk = self._compute_topk_obstacles(car_idx, obstacles_only=True).reshape(E, self.cfg.obs_top_k, 6)
            nearest = topk[:, :2]

            dx = nearest[:, :, 0] * self.cfg.obs_max_distance
            dy = nearest[:, :, 1] * self.cfg.obs_max_distance
            vx = nearest[:, :, 2] * self.cfg.obs_v_max
            vy = nearest[:, :, 3] * self.cfg.obs_v_max
            diam = nearest[:, :, 4]
            valid = nearest[:, :, 5] > 0.5

            center_d = torch.sqrt(dx * dx + dy * dy)
            obj_radius = diam * 0.5
            surface_d = torch.clamp(center_d - obj_radius - ROBOT_BODY_RADIUS, min=0.0)
            scale = torch.where(
                center_d > 1e-6,
                surface_d / center_d,
                torch.zeros_like(center_d),
            )
            surf_x = dx * scale
            surf_y = dy * scale

            mask = valid.float()
            surf_x = surf_x * mask
            surf_y = surf_y * mask
            surface_d = surface_d * mask
            vx = vx * mask
            vy = vy * mask

            out[:, car_idx, 0] = surf_x[:, 0]
            out[:, car_idx, 1] = surf_y[:, 0]
            out[:, car_idx, 2] = surface_d[:, 0]
            out[:, car_idx, 3] = surf_x[:, 1]
            out[:, car_idx, 4] = surf_y[:, 1]
            out[:, car_idx, 5] = surface_d[:, 1]
            out[:, car_idx, 6] = vx[:, 0]
            out[:, car_idx, 7] = vy[:, 0]

        return out

    # ------------------------------------------------------------------
    # Safety distance (per-car)
    # ------------------------------------------------------------------
    def _get_d_safe(self, car_idx: int, bottom_k: int = 36) -> torch.Tensor:
        """LiDAR bottom-K safety distance."""
        lidar = self.lidars[car_idx]
        sensor_pos = lidar.data.pos_w[:, :2]
        hit_points = lidar.data.ray_hits_w[:, :, :2]
        dist_2d = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)
        dist_2d = torch.nan_to_num(
            dist_2d, nan=self.cfg.lidar_r_max, posinf=self.cfg.lidar_r_max,
        )

        actual_k = min(bottom_k, dist_2d.shape[1])
        bottom_vals = torch.topk(dist_2d, k=actual_k, dim=1, largest=False).values
        return (bottom_vals.mean(dim=1) - self.cfg.body_radius).clamp(min=1e-4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        """Quaternion [w, x, y, z] → yaw angle."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
