"""
rvo2_safety_filter.py — RVO2 (ORCA) Safety Filter for Charge RL (Isaac Lab)

Performance-optimized architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  GPU PRE-FILTERING                                                  │
  │  1. Batch-gather all obstacle pos/vel into [N, M, 2] tensors       │
  │  2. Compute robot-obstacle distances on GPU: [N, M]                │
  │  3. Boolean mask: dist < CULLING_RADIUS & active (z >= 0)          │
  │  4. Transfer ONLY nearby obstacles to CPU (minimal .cpu() calls)   │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CPU ORCA COMPUTATION                                               │
  │  5. Per-env: build RVO2 sim with culled obstacles only              │
  │  6. doStep() → v_safe per env                                      │
  │  7. Project v_safe → body frame (non-holonomic constraint)          │
  │  8. Write back to action_term tensors (GPU)                        │
  └─────────────────────────────────────────────────────────────────────┘

Multi-env strategy:
  - num_envs == 1: single-thread, zero pool overhead
  - num_envs > 1: ThreadPoolExecutor (RVO2 C++ releases GIL)

Usage:
  from rvo2_safety_filter import RVO2SafetyFilter
  filt = RVO2SafetyFilter(raw_env, num_envs=N, culling_radius=5.0)
  filt.install(action_term)
  # BEV (env 0 only): filt.draw_on_bev(ax, yaw, frame)
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import matplotlib.patches as mpatches
import numpy as np
import rvo2
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerEnvORCAInput:
    """CPU-side data for one environment's ORCA computation."""
    env_idx: int
    robot_pos: tuple[float, float]
    robot_vel: tuple[float, float]
    yaw: float
    v_pref: tuple[float, float]
    v_next_body: float
    omega_rl: float
    # Nearby obstacles only (after GPU culling)
    obs_pos: np.ndarray     # [K, 2]
    obs_vel: np.ndarray     # [K, 2]
    obs_radii: np.ndarray   # [K]
    obs_is_static: np.ndarray  # [K] bool — True = static (|vel| < threshold)


@dataclass
class PerEnvORCAOutput:
    """Result of one environment's ORCA computation."""
    env_idx: int
    v_safe_linear: float
    v_safe_omega: float
    orca_active: bool
    fallback_active: bool
    # For visualization (env 0 only)
    v_pref_world: np.ndarray | None
    v_safe_world: np.ndarray | None
    vo_cone_data: list[tuple[np.ndarray, float]] | None


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class RVO2SafetyFilter:
    """ORCA Safety Filter with GPU pre-filtering and multi-env parallelism."""

    def __init__(
        self,
        raw_env,
        num_envs: int = 1,
        robot_radius: float = 0.35,
        safety_margin: float = 0.10,
        max_speed: float = 1.0,
        culling_radius: float = 5.0,
        neighbor_dist: float = 10.0,
        max_neighbors: int = 20,
        time_horizon_dynamic: float = 3.5,
        time_horizon_static: float = 0.5,
        static_vel_threshold: float = 0.15,
        angle_threshold_deg: float = 60.0,
        omega_max: float = 0.25 * math.pi,
        max_workers: int | None = None,
        # Non-cooperative obstacle inflation
        obs_inflation: float = 1.8,
        # Responsibility: fraction of avoidance ego must handle (1.0 = 100%)
        ego_responsibility: float = 1.0,
        # Tactical retreat (Zone B: forced reverse on lateral threat)
        tactical_retreat: bool = True,
        # Recovery behavior parameters
        stuck_speed_threshold: float = 0.1,
        stuck_pref_threshold: float = 0.05,
        stuck_duration_steps: int = 10,
        recovery_reverse_steps: int = 5,
        recovery_rotate_steps: int = 5,
        # Displacement-based stuck detection (catches oscillation + any stuck)
        disp_window: int = 15,
        disp_threshold: float = 0.3,  # if moved < 0.3m in 15 steps → stuck
        # Legacy alias
        time_horizon: float | None = None,
        time_horizon_obst: float | None = None,
    ):
        self.raw_env = raw_env
        self.num_envs = num_envs
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.rvo_radius = robot_radius + safety_margin
        self.max_speed = max_speed
        self.culling_radius = culling_radius
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        # Solution 3: separate time horizons for dynamic vs static obstacles
        self.time_horizon_dynamic = time_horizon or time_horizon_dynamic
        self.time_horizon_static = time_horizon_obst or time_horizon_static
        self.static_vel_threshold = static_vel_threshold
        self.angle_threshold = math.radians(angle_threshold_deg)
        self.omega_max = omega_max
        # Non-cooperative obstacle inflation (scripted obs don't share ORCA 50/50)
        self.obs_inflation = obs_inflation
        # Ego responsibility: 1.0 = ego takes 100% avoidance (obstacles non-cooperative)
        self.ego_responsibility = ego_responsibility
        # Tactical retreat: Zone B 側向威脅強制倒車
        self.tactical_retreat = tactical_retreat

        # Solution 2: Recovery behavior — stuck detection + escape maneuver
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_pref_threshold = stuck_pref_threshold
        self.stuck_duration_steps = stuck_duration_steps
        self.recovery_reverse_steps = recovery_reverse_steps
        self.recovery_rotate_steps = recovery_rotate_steps
        self._stuck_counter = np.zeros(num_envs, dtype=np.int32)
        self._recovery_counter = np.zeros(num_envs, dtype=np.int32)
        self._recovery_mode = np.zeros(num_envs, dtype=bool)
        self._recovery_phase = np.zeros(num_envs, dtype=np.int32)  # 0=reverse, 1=rotate
        self.recovery_active = False  # env 0 visualization
        self.recovery_count = 0  # total recoveries triggered

        # Displacement-based stuck detection (catches oscillation + any stuck pattern)
        self.disp_window = disp_window
        self.disp_threshold = disp_threshold
        self._pos_history = [[] for _ in range(num_envs)]  # ring buffer of (x, y)
        self.displacement_stuck = False  # env 0 visualization
        self.displacement_recovery_count = 0

        # ── Obstacle entity references (built once at init) ──
        self._num_obstacles = getattr(raw_env, "_num_obstacles", None) or 10
        self._obstacle_entities = []
        for i in range(self._num_obstacles):
            try:
                self._obstacle_entities.append(raw_env.scene[f"obstacle_{i}"])
            except KeyError:
                break
        self._actual_num_obs = len(self._obstacle_entities)

        # Obstacle radii tensor on GPU [M] — constant across steps
        sizes = getattr(raw_env, "_obstacle_sizes", None)
        if sizes is not None and len(sizes) >= self._actual_num_obs:
            self._obs_radii_gpu = torch.tensor(
                [float(sizes[i]) for i in range(self._actual_num_obs)],
                device=raw_env.device, dtype=torch.float32,
            )
        else:
            self._obs_radii_gpu = torch.full(
                (self._actual_num_obs,), 0.3,
                device=raw_env.device, dtype=torch.float32,
            )

        # ── Multi-env thread pool ──
        if num_envs > 1:
            workers = max_workers or min(num_envs, 8)
            self._pool = ThreadPoolExecutor(max_workers=workers)
        else:
            self._pool = None

        # ── Visualization (env 0 only) ──
        self.v_pref_world = np.zeros(2)
        self.v_safe_world = np.zeros(2)
        self.last_agent_linear = 0.0
        self.last_agent_omega = 0.0
        self.orca_active = False
        self.fallback_active = False
        self._vo_cone_data: list[tuple[np.ndarray, float]] = []

        # ── Statistics ──
        self.total_steps = 0
        self.intervention_steps = 0
        self.fallback_steps = 0

        # ── Decimation guard (per-env cached results) ──
        self._cached_v_safe_linear = torch.zeros(num_envs, device=raw_env.device)
        self._cached_v_safe_omega = torch.zeros(num_envs, device=raw_env.device)
        self._new_decision: bool = True
        self._has_cache: bool = False

    # ══════════════════════════════════════════════════════════════════════════
    # GPU PRE-FILTERING
    # ══════════════════════════════════════════════════════════════════════════

    def _gather_obstacle_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-gather all obstacle states into GPU tensors.

        Returns:
            obs_pos: [num_envs, M, 2] — XY positions
            obs_vel: [num_envs, M, 2] — XY velocities
            obs_z:   [num_envs, M]    — Z positions (for active check)
        """
        N = self.num_envs
        M = self._actual_num_obs

        # Pre-allocate on GPU
        device = self.raw_env.device
        obs_pos = torch.zeros(N, M, 2, device=device)
        obs_vel = torch.zeros(N, M, 2, device=device)
        obs_z = torch.full((N, M), -10.0, device=device)

        for i, entity in enumerate(self._obstacle_entities):
            pos_w = entity.data.root_pos_w  # [N, 3]
            vel_w = entity.data.root_lin_vel_w  # [N, 3]
            obs_pos[:, i, :] = pos_w[:, :2]
            obs_vel[:, i, :] = vel_w[:, :2]
            obs_z[:, i] = pos_w[:, 2]

        return obs_pos, obs_vel, obs_z

    def _gpu_distance_culling(
        self,
        robot_pos_xy: torch.Tensor,
        obs_pos: torch.Tensor,
        obs_z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance mask on GPU — only nearby & active obstacles pass.

        Filters out:
          - Inactive obstacles (z < 0)
          - Obstacles beyond culling radius
          - Self-obstacles (dist < 0.01m) — prevents ghost intervention

        Args:
            robot_pos_xy: [N, 2] robot XY positions
            obs_pos:      [N, M, 2] obstacle XY positions
            obs_z:        [N, M] obstacle Z (active if >= 0)

        Returns:
            mask: [N, M] boolean — True = obstacle passes culling
        """
        # [N, M, 2] = obs_pos - robot_pos[:, None, :]
        rel = obs_pos - robot_pos_xy.unsqueeze(1)
        # [N, M] Euclidean distance
        dist = torch.norm(rel, dim=2)
        # Combined mask: active (z >= 0) AND within culling radius AND not self
        mask = (obs_z >= 0.0) & (dist < self.culling_radius) & (dist > 0.01)
        return mask

    # ══════════════════════════════════════════════════════════════════════════
    # CPU ORCA COMPUTATION (per-env worker)
    # ══════════════════════════════════════════════════════════════════════════

    # Intervention threshold: norm(v_safe - v_pref) must exceed this to count
    INTERVENTION_EPSILON = 0.05  # m/s

    def _solve_orca_single_env(self, inp: PerEnvORCAInput, dt: float) -> PerEnvORCAOutput:
        """Run ORCA for one environment (designed to run in thread pool).

        ORCA only processes dynamic obstacles. Static obstacles are excluded
        entirely and left to the RL policy. If ORCA outputs near-zero velocity
        (freeze), lateral evasion injects perpendicular escape velocity when
        a fast dynamic obstacle is on collision course.
        """
        # Dynamic maxSpeed: must be >= |v_pref| to avoid RVO2 truncation ghost
        v_pref_mag = math.sqrt(inp.v_pref[0]**2 + inp.v_pref[1]**2)
        agent_max_speed = max(self.max_speed, v_pref_mag + 0.1)

        # Build RVO2 simulator — default params use dynamic time horizon
        sim = rvo2.PyRVOSimulator(
            dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon_dynamic,
            self.time_horizon_static,
            self.rvo_radius,
            agent_max_speed,
        )

        # Robot agent
        robot_agent = sim.addAgent(inp.robot_pos)
        sim.setAgentVelocity(robot_agent, inp.robot_vel)
        sim.setAgentPrefVelocity(robot_agent, inp.v_pref)

        # Culled obstacle agents — ORCA only handles dynamic obstacles.
        # Static obstacles are left entirely to the RL policy.
        K = len(inp.obs_radii)
        valid_count = 0
        dynamic_indices = []  # track which k are dynamic (for lateral evasion)
        for k in range(K):
            is_static = bool(inp.obs_is_static[k])
            if is_static:
                continue  # 靜態交給 RL，ORCA 不處理

            obs_pos_k = (float(inp.obs_pos[k, 0]), float(inp.obs_pos[k, 1]))
            obs_vel_k = (float(inp.obs_vel[k, 0]), float(inp.obs_vel[k, 1]))
            obs_r = float(inp.obs_radii[k])

            th = self.time_horizon_dynamic
            th_obst = self.time_horizon_static

            # Break reciprocal assumption: scripted obstacles are non-cooperative.
            # Setting their maxSpeed near zero tells ORCA they won't deviate from
            # their current trajectory, forcing ego to take full avoidance responsibility.
            # Combined with radius inflation, this produces one-sided detour trajectories.
            effective_r = obs_r + self.safety_margin
            if self.obs_inflation > 1.0:
                effective_r *= self.obs_inflation

            # Non-cooperative maxSpeed: ego_responsibility=1.0 → obs maxSpeed≈0
            # ego_responsibility=0.5 → standard 50/50 (obs maxSpeed matches actual)
            obs_speed_mag = math.sqrt(obs_vel_k[0]**2 + obs_vel_k[1]**2)
            obs_max_speed = obs_speed_mag * (1.0 - self.ego_responsibility) + 0.01

            sim.addAgent(
                obs_pos_k,
                self.neighbor_dist,
                self.max_neighbors,
                th,
                th_obst,
                effective_r,
                obs_max_speed,
                obs_vel_k,
            )
            sim.setAgentPrefVelocity(valid_count + 1, obs_vel_k)
            dynamic_indices.append(k)
            valid_count += 1

        # ORCA solve (skip if no dynamic obstacles in sim)
        if valid_count == 0:
            # No dynamic obstacles → pure pass-through
            return PerEnvORCAOutput(
                env_idx=inp.env_idx,
                v_safe_linear=inp.v_next_body,
                v_safe_omega=inp.omega_rl,
                orca_active=False,
                fallback_active=False,
                v_pref_world=np.array(inp.v_pref) if inp.env_idx == 0 else None,
                v_safe_world=np.array(inp.v_pref) if inp.env_idx == 0 else None,
                vo_cone_data=[] if inp.env_idx == 0 else None,
            )

        sim.doStep()
        v_safe = sim.getAgentVelocity(robot_agent)
        v_safe_arr = np.array(v_safe)

        # ── Lateral evasion: if ORCA says "stop" but ego is on collision
        # course with a fast approaching obstacle, inject perpendicular escape
        # velocity so the ego drifts out of the threat's trajectory. ──
        v_safe_mag = math.sqrt(v_safe[0]**2 + v_safe[1]**2)
        if v_safe_mag < self.LATERAL_EVASION_V_THRESH:
            v_safe_arr = self._lateral_evasion_if_frozen(
                v_safe_arr, inp, dynamic_indices
            )
            v_safe = (float(v_safe_arr[0]), float(v_safe_arr[1]))

        # Check intervention with proper epsilon
        v_pref_arr = np.array(inp.v_pref)
        v_diff = float(np.linalg.norm(v_safe_arr - v_pref_arr))
        orca_active = v_diff > self.INTERVENTION_EPSILON

        # Debug log on intervention (throttle: 每 10 次介入才印一次，避免 stdout I/O 拖慢 loop)
        if orca_active and inp.env_idx == 0:
            self._debug_intervene_count = getattr(self, "_debug_intervene_count", 0) + 1
            if self._debug_intervene_count % 10 == 1:
                print(
                    f"[RVO2 intervene #{self._debug_intervene_count}] "
                    f"v_pref=({inp.v_pref[0]:+.3f},{inp.v_pref[1]:+.3f}) "
                    f"v_safe=({v_safe[0]:+.3f},{v_safe[1]:+.3f}) "
                    f"diff={v_diff:.3f} obs_in_sim={valid_count}"
                )

        # Project to body frame or pass-through
        if orca_active:
            v_safe_linear, v_safe_omega = self._project_to_body_frame(
                v_safe, inp.yaw, dt
            )
        else:
            v_safe_linear = inp.v_next_body
            v_safe_omega = inp.omega_rl
        fallback_active = False  # 連續投影無硬切 fallback

        # Visualization data (env 0 only)
        vis_v_pref = None
        vis_v_safe = None
        vis_cones = None
        if inp.env_idx == 0:
            vis_v_pref = v_pref_arr.copy()
            vis_v_safe = v_safe_arr.copy()
            # VO cone data (dynamic obstacles only, matching ORCA sim)
            vis_cones = []
            robot_xy = np.array(inp.robot_pos)
            for k in dynamic_indices:
                rel_xy = inp.obs_pos[k] - robot_xy
                combined_r = self.rvo_radius + float(inp.obs_radii[k]) + self.safety_margin
                vis_cones.append((rel_xy.copy(), combined_r))

        return PerEnvORCAOutput(
            env_idx=inp.env_idx,
            v_safe_linear=v_safe_linear,
            v_safe_omega=v_safe_omega,
            orca_active=orca_active,
            fallback_active=fallback_active,
            v_pref_world=vis_v_pref,
            v_safe_world=vis_v_safe,
            vo_cone_data=vis_cones,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Core filter step (orchestrator)
    # ══════════════════════════════════════════════════════════════════════════

    def filter_step(self, action_term) -> None:
        """Main entry: GPU pre-filter → CPU ORCA → write back.

        Called inside monkey-patched apply_actions(), between process_actions()
        and the actual sim write.
        """
        # ── Decimation guard ──
        if not self._new_decision and self._has_cache:
            action_term._processed_actions[:, 0] = self._cached_v_safe_linear
            action_term._processed_actions[:, 1] = self._cached_v_safe_omega
            action_term._current_velocity[:] = self._cached_v_safe_linear
            return

        self._new_decision = False
        self.total_steps += 1
        N = self.num_envs
        dt = action_term._dt

        # ════════════════════════════════════════════════════════════════════
        # GPU PRE-FILTERING SECTION
        # ════════════════════════════════════════════════════════════════════

        # 1. Robot states (stay on GPU as long as possible)
        robot = self.raw_env.scene["robot"]
        robot_pos_w = robot.data.root_pos_w[:, :2]      # [N, 2] GPU
        robot_vel_w = robot.data.root_lin_vel_w[:, :2]   # [N, 2] GPU
        robot_quat_w = robot.data.root_quat_w            # [N, 4] GPU

        # Yaw from quaternion (vectorized on GPU)
        w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
        yaw_gpu = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # [N]

        # 2. v_pref from action_term (vectorized for all envs)
        v_next_body_gpu = action_term._processed_actions[:, 0]  # [N] GPU
        omega_rl_gpu = action_term._processed_actions[:, 1]     # [N] GPU
        yaw_next = yaw_gpu + omega_rl_gpu * dt
        v_pref_gpu = torch.stack([
            v_next_body_gpu * torch.cos(yaw_next),
            v_next_body_gpu * torch.sin(yaw_next),
        ], dim=1)  # [N, 2] GPU

        # 3. Batch-gather obstacle tensors (GPU)
        obs_pos, obs_vel, obs_z = self._gather_obstacle_tensors()  # [N,M,2], [N,M,2], [N,M]

        # 4. Distance culling (GPU) — boolean mask [N, M]
        mask = self._gpu_distance_culling(robot_pos_w, obs_pos, obs_z)

        # 5. Check if ANY env has nearby obstacles
        any_nearby = mask.any().item()

        if not any_nearby:
            # No obstacles near any robot → pure pass-through, skip CPU entirely
            self._cached_v_safe_linear = v_next_body_gpu.clone()
            self._cached_v_safe_omega = omega_rl_gpu.clone()
            self._has_cache = True
            self.orca_active = False
            self.fallback_active = False
            self.v_pref_world[:] = v_pref_gpu[0].detach().cpu().numpy()
            self.v_safe_world[:] = self.v_pref_world
            self._vo_cone_data.clear()
            self.last_agent_linear = float(v_next_body_gpu[0])
            self.last_agent_omega = float(omega_rl_gpu[0])
            return

        # ════════════════════════════════════════════════════════════════════
        # CPU TRANSFER (minimal — only culled data)
        # ════════════════════════════════════════════════════════════════════

        # Transfer needed data to CPU (single batch transfer)
        robot_pos_cpu = robot_pos_w.detach().cpu().numpy()       # [N, 2]
        robot_vel_cpu = robot_vel_w.detach().cpu().numpy()       # [N, 2]
        yaw_cpu = yaw_gpu.detach().cpu().numpy()                 # [N]
        v_pref_cpu = v_pref_gpu.detach().cpu().numpy()           # [N, 2]
        v_next_body_cpu = v_next_body_gpu.detach().cpu().numpy() # [N]
        omega_rl_cpu = omega_rl_gpu.detach().cpu().numpy()       # [N]

        obs_pos_cpu = obs_pos.detach().cpu().numpy()   # [N, M, 2]
        obs_vel_cpu = obs_vel.detach().cpu().numpy()   # [N, M, 2]
        mask_cpu = mask.detach().cpu().numpy()          # [N, M] bool
        obs_radii_cpu = self._obs_radii_gpu.detach().cpu().numpy()  # [M]

        # Build per-env ORCA inputs (only envs with nearby obstacles need ORCA)
        orca_inputs: list[PerEnvORCAInput] = []
        passthrough_envs: list[int] = []

        for e in range(N):
            nearby_mask = mask_cpu[e]  # [M] bool
            if not nearby_mask.any():
                passthrough_envs.append(e)
                continue

            nearby_idx = np.where(nearby_mask)[0]
            nearby_vel = obs_vel_cpu[e, nearby_idx, :]  # [K, 2]
            # Classify static vs dynamic by velocity magnitude
            vel_mag = np.linalg.norm(nearby_vel, axis=1)  # [K]
            is_static = vel_mag < self.static_vel_threshold  # [K] bool

            orca_inputs.append(PerEnvORCAInput(
                env_idx=e,
                robot_pos=(float(robot_pos_cpu[e, 0]), float(robot_pos_cpu[e, 1])),
                robot_vel=(float(robot_vel_cpu[e, 0]), float(robot_vel_cpu[e, 1])),
                yaw=float(yaw_cpu[e]),
                v_pref=(float(v_pref_cpu[e, 0]), float(v_pref_cpu[e, 1])),
                v_next_body=float(v_next_body_cpu[e]),
                omega_rl=float(omega_rl_cpu[e]),
                obs_pos=obs_pos_cpu[e, nearby_idx, :],    # [K, 2]
                obs_vel=nearby_vel,                        # [K, 2]
                obs_radii=obs_radii_cpu[nearby_idx],       # [K]
                obs_is_static=is_static,                   # [K] bool
            ))

        # ════════════════════════════════════════════════════════════════════
        # CPU ORCA COMPUTATION SECTION
        # ════════════════════════════════════════════════════════════════════

        # Initialize results with pass-through values
        result_linear = v_next_body_cpu.copy()   # [N]
        result_omega = omega_rl_cpu.copy()       # [N]

        if orca_inputs:
            if self._pool is not None and len(orca_inputs) > 1:
                # Multi-env: parallel ORCA via thread pool
                futures = [
                    self._pool.submit(self._solve_orca_single_env, inp, dt)
                    for inp in orca_inputs
                ]
                outputs = [f.result() for f in futures]
            else:
                # Single-env or single input: direct call (no pool overhead)
                outputs = [self._solve_orca_single_env(inp, dt) for inp in orca_inputs]

            # Aggregate results
            any_intervention = False
            any_fallback = False
            for out in outputs:
                e = out.env_idx
                result_linear[e] = out.v_safe_linear
                result_omega[e] = out.v_safe_omega
                if out.orca_active:
                    any_intervention = True
                if out.fallback_active:
                    any_fallback = True

                # Env 0 visualization
                if out.env_idx == 0:
                    self.orca_active = out.orca_active
                    self.fallback_active = out.fallback_active
                    if out.v_pref_world is not None:
                        self.v_pref_world[:] = out.v_pref_world
                    if out.v_safe_world is not None:
                        self.v_safe_world[:] = out.v_safe_world
                    if out.vo_cone_data is not None:
                        self._vo_cone_data = out.vo_cone_data
                    self.last_agent_linear = float(v_next_body_cpu[0])
                    self.last_agent_omega = float(omega_rl_cpu[0])

            if any_intervention:
                self.intervention_steps += 1
            if any_fallback:
                self.fallback_steps += 1
        else:
            # All envs passed through
            self.orca_active = False
            self.fallback_active = False
            self.v_pref_world[:] = v_pref_cpu[0]
            self.v_safe_world[:] = v_pref_cpu[0]
            self._vo_cone_data.clear()
            self.last_agent_linear = float(v_next_body_cpu[0])
            self.last_agent_omega = float(omega_rl_cpu[0])

        # ════════════════════════════════════════════════════════════════════
        # ANTI-OSCILLATION + RECOVERY BEHAVIOR
        # Layer 1: Detect v_body sign flipping → force consistent direction
        # Layer 2: Detect stuck (low speed for N steps) → reverse + rotate
        # ════════════════════════════════════════════════════════════════════

        robot_speed_cpu = np.linalg.norm(robot_vel_cpu, axis=1)  # [N]

        for e in range(N):
            # ── Displacement-based stuck detection ──
            pos_xy = (float(robot_pos_cpu[e, 0]), float(robot_pos_cpu[e, 1]))
            phist = self._pos_history[e]
            phist.append(pos_xy)
            if len(phist) > self.disp_window:
                phist.pop(0)

            disp_stuck = False
            if len(phist) >= self.disp_window and not self._recovery_mode[e]:
                dx = phist[-1][0] - phist[0][0]
                dy = phist[-1][1] - phist[0][1]
                displacement = math.sqrt(dx * dx + dy * dy)
                if displacement < self.disp_threshold:
                    disp_stuck = True
                    # Directly trigger recovery
                    self._recovery_mode[e] = True
                    self._recovery_counter[e] = 0
                    self._recovery_phase[e] = 0
                    self._stuck_counter[e] = 0
                    self._pos_history[e].clear()
                    self.recovery_count += 1
                    self.displacement_recovery_count += 1
                    if e == 0:
                        self.displacement_stuck = True
                        print(
                            f"[RVO2 DISP→REC] env {e}: moved only {displacement:.3f}m in "
                            f"{self.disp_window} steps → reverse+rotate"
                        )

            if e == 0 and not disp_stuck:
                self.displacement_stuck = False

            # ── Recovery behavior: stuck detection + escape maneuver ──
            if self._recovery_mode[e]:
                # Currently in recovery — execute escape maneuver
                rc = self._recovery_counter[e]
                if self._recovery_phase[e] == 0:
                    # Phase 0: reverse
                    result_linear[e] = -0.3
                    result_omega[e] = 0.0
                    self._recovery_counter[e] = rc + 1
                    if rc + 1 >= self.recovery_reverse_steps:
                        self._recovery_phase[e] = 1
                        self._recovery_counter[e] = 0
                else:
                    # Phase 1: rotate 90 degrees
                    result_linear[e] = 0.0
                    result_omega[e] = self.omega_max
                    self._recovery_counter[e] = rc + 1
                    if rc + 1 >= self.recovery_rotate_steps:
                        # Recovery complete — hand back to RL
                        self._recovery_mode[e] = False
                        self._recovery_counter[e] = 0
                        self._recovery_phase[e] = 0
                        self._stuck_counter[e] = 0
                        self._pos_history[e].clear()

                if e == 0:
                    self.recovery_active = True
            else:
                # Check if stuck: RL wants to move but robot isn't
                v_pref_mag_e = abs(v_next_body_cpu[e])
                actual_speed_e = robot_speed_cpu[e]
                if v_pref_mag_e > self.stuck_pref_threshold and actual_speed_e < self.stuck_speed_threshold:
                    self._stuck_counter[e] += 1
                else:
                    self._stuck_counter[e] = max(0, self._stuck_counter[e] - 1)

                if self._stuck_counter[e] >= self.stuck_duration_steps:
                    # Trigger recovery
                    self._recovery_mode[e] = True
                    self._recovery_counter[e] = 0
                    self._recovery_phase[e] = 0
                    self._stuck_counter[e] = 0
                    self._pos_history[e].clear()
                    self.recovery_count += 1
                    if e == 0:
                        print(f"[RVO2 RECOVERY] env {e}: stuck detected, "
                              f"v_pref={v_pref_mag_e:.2f} actual={actual_speed_e:.2f} → reverse+rotate")

                if e == 0:
                    self.recovery_active = False

        # ════════════════════════════════════════════════════════════════════
        # VELOCITY CLAMPING — 確保所有路徑的輸出都在物理極限內
        # ════════════════════════════════════════════════════════════════════
        np.clip(result_linear, -self.max_speed, self.max_speed, out=result_linear)
        np.clip(result_omega, -self.omega_max, self.omega_max, out=result_omega)

        # ════════════════════════════════════════════════════════════════════
        # WRITE BACK TO GPU
        # ════════════════════════════════════════════════════════════════════

        device = action_term._processed_actions.device
        safe_linear_gpu = torch.from_numpy(result_linear).to(device=device, dtype=torch.float32)
        safe_omega_gpu = torch.from_numpy(result_omega).to(device=device, dtype=torch.float32)

        # Only overwrite envs where ORCA actually intervened
        # For pass-through envs, values are already == original RL values
        action_term._processed_actions[:, 0] = safe_linear_gpu
        action_term._processed_actions[:, 1] = safe_omega_gpu
        action_term._current_velocity[:] = safe_linear_gpu

        # Cache for decimation guard
        self._cached_v_safe_linear = safe_linear_gpu.clone()
        self._cached_v_safe_omega = safe_omega_gpu.clone()
        self._has_cache = True

    # ══════════════════════════════════════════════════════════════════════════
    # Body-frame projection (non-holonomic)
    # ══════════════════════════════════════════════════════════════════════════

    # Steering proportional gain: ω = Kp * angle_err
    STEER_KP: float = 3.0
    # Reverse is only allowed when angle_err > this threshold (radians)
    REVERSE_ANGLE_THRESHOLD: float = 2.6  # ~150°
    # Minimum forward speed maintained during lateral detour (fraction of v_safe_mag)
    FORWARD_DETOUR_BIAS: float = 0.25

    # ── Lateral evasion constants ──
    # v_safe magnitude below this → ego is "frozen" (ORCA says stop)
    LATERAL_EVASION_V_THRESH: float = 0.08
    # Obstacle must be approaching faster than this to trigger evasion
    LATERAL_EVASION_OBS_SPEED: float = 0.3
    # Time-to-collision threshold: only evade if TTC < this
    LATERAL_EVASION_TTC: float = 4.0
    # Perpendicular miss distance: ego is "in the path" if < this
    LATERAL_EVASION_MISS_DIST: float = 1.2  # robot_r + obs_r + margin
    # Escape speed injected perpendicular to threat trajectory
    LATERAL_EVASION_SPEED: float = 0.4

    def _lateral_evasion_if_frozen(
        self,
        v_safe_arr: np.ndarray,
        inp: PerEnvORCAInput,
        dynamic_indices: list[int],
    ) -> np.ndarray:
        """When ORCA outputs ≈ 0 velocity, check if ego sits on a collision
        course with an approaching dynamic obstacle. If so, inject a lateral
        escape velocity perpendicular to the threat's trajectory.

        This prevents the "sitting duck" failure mode where ego brakes in-place
        and lets a fast side-approaching obstacle crash into it.

        Returns:
            Modified v_safe_arr (or unchanged if no threat detected).
        """
        robot_xy = np.array(inp.robot_pos)
        best_ttc = float("inf")
        best_perp = None  # perpendicular escape direction

        for k in dynamic_indices:
            obs_xy = inp.obs_pos[k]
            obs_v = inp.obs_vel[k]
            obs_speed = float(np.linalg.norm(obs_v))
            if obs_speed < self.LATERAL_EVASION_OBS_SPEED:
                continue  # slow obstacle — no urgent threat

            # Vector from obstacle to robot
            rel_pos = robot_xy - obs_xy  # points obs → ego
            obs_dir = obs_v / obs_speed  # unit velocity direction

            # Project rel_pos onto obstacle's velocity to get TTC
            along = float(np.dot(rel_pos, obs_dir))
            if along < 0:
                continue  # obstacle is moving away from ego

            ttc = along / obs_speed
            if ttc > self.LATERAL_EVASION_TTC:
                continue  # plenty of time

            # Perpendicular miss distance: how far will obs pass from ego
            perp_dist = abs(float(rel_pos[0] * (-obs_dir[1]) + rel_pos[1] * obs_dir[0]))
            combined_r = float(inp.obs_radii[k]) + self.rvo_radius
            if perp_dist > self.LATERAL_EVASION_MISS_DIST + combined_r:
                continue  # will miss — no danger

            if ttc < best_ttc:
                best_ttc = ttc
                # Perpendicular direction: rotate obs_dir by +90° or -90°
                # Choose the side closer to v_pref so we dodge toward our goal
                perp_left = np.array([-obs_dir[1], obs_dir[0]])
                perp_right = np.array([obs_dir[1], -obs_dir[0]])
                v_pref_arr = np.array(inp.v_pref)
                # Pick the perpendicular that aligns better with v_pref
                if np.dot(perp_left, v_pref_arr) >= np.dot(perp_right, v_pref_arr):
                    best_perp = perp_left
                else:
                    best_perp = perp_right

        if best_perp is not None:
            # Inject lateral escape velocity
            escape_v = best_perp * self.LATERAL_EVASION_SPEED
            if inp.env_idx == 0:
                self._lateral_evasion_count = getattr(self, "_lateral_evasion_count", 0) + 1
                if self._lateral_evasion_count % 5 == 1:
                    print(
                        f"[RVO2 LATERAL] ttc={best_ttc:.2f}s "
                        f"escape=({escape_v[0]:+.3f},{escape_v[1]:+.3f})"
                    )
            return v_safe_arr + escape_v

        return v_safe_arr

    def _project_to_body_frame(
        self, v_safe: tuple[float, float], yaw: float, dt: float
    ) -> tuple[float, float]:
        """World-frame v_safe → body-frame (v_linear, omega_z).

        Forward-Biased Non-Holonomic Projection (NH-ORCA style):

        Instead of naive cos(angle_err) which causes passive braking at 90° and
        reversing at >90°, this approach maintains forward momentum during lateral
        detours and only reverses as an absolute last resort:

          angle_err   行為
          ──────────  ────────────────────────────────────────
            0°-60°    Full forward + gentle steering
           60°-120°   Sustained forward (≥ 25% of |v_safe|) + aggressive steering
          120°-150°   Reduced forward speed + max steering (strong turn-in-place)
          150°-180°   Reverse allowed (emergency only, obstacle directly behind goal)

        Pure-pursuit inspired: compute yaw rate to steer toward v_safe direction,
        while maintaining forward speed to keep the vehicle progressing laterally.
        This converts "dodge left/right" into "forward-arc" maneuvers.
        """
        vx_safe, vy_safe = v_safe
        v_safe_mag = math.sqrt(vx_safe**2 + vy_safe**2)

        if v_safe_mag < 0.01:
            return 0.0, 0.0

        # 1. angle_err: signed angle from heading to v_safe, in [-π, π]
        angle_v_safe = math.atan2(vy_safe, vx_safe)
        angle_err = math.atan2(
            math.sin(angle_v_safe - yaw),
            math.cos(angle_v_safe - yaw),
        )
        abs_angle_err = abs(angle_err)

        # 2. Forward-biased speed mapping
        #    Key insight: for lateral detours (60°-150°), the robot should maintain
        #    forward speed while steering, NOT brake to zero.
        if abs_angle_err < math.radians(60):
            # Zone A: Small deviation — full forward speed, gentle steering
            v_linear = v_safe_mag * math.cos(angle_err)
        elif abs_angle_err < self.REVERSE_ANGLE_THRESHOLD:
            # Zone B: Large lateral deviation — maintain forward bias + aggressive steer
            # cos gives 0 or negative here, but we override with a forward floor
            cos_component = v_safe_mag * math.cos(angle_err)
            forward_floor = v_safe_mag * self.FORWARD_DETOUR_BIAS
            v_linear = max(cos_component, forward_floor)
        else:
            # Zone C: Nearly opposite direction (>150°) — allow reverse as emergency
            v_linear = v_safe_mag * math.cos(angle_err)

        # 3. ω = Kp * angle_err (P-Controller steering toward v_safe direction)
        #    Higher gain in the detour zone for faster turning
        if abs_angle_err > math.radians(60):
            # Aggressive steering during detour — use 1.5x gain
            omega_z = self.STEER_KP * 1.5 * angle_err
        else:
            omega_z = self.STEER_KP * angle_err

        # 4. Clamp to physical limits
        v_linear = max(-self.max_speed, min(self.max_speed, v_linear))
        omega_z = max(-self.omega_max, min(self.omega_max, omega_z))

        return v_linear, omega_z

    # ══════════════════════════════════════════════════════════════════════════
    # Installation (monkey-patch)
    # ══════════════════════════════════════════════════════════════════════════

    def install(self, action_term) -> None:
        """Monkey-patch action_term to inject RVO2 filter.

        Patches:
          - process_actions: set _new_decision flag
          - apply_actions: run ORCA filter (first decimation step only)
        """
        filter_ref = self

        original_process = action_term.process_actions

        def flagged_process(actions):
            filter_ref._new_decision = True
            original_process(actions)

        action_term.process_actions = flagged_process

        original_apply = action_term.apply_actions

        def rvo2_filtered_apply():
            filter_ref.filter_step(action_term)
            original_apply()

        action_term.apply_actions = rvo2_filtered_apply

        print(
            f"[RVO2] Safety filter installed: "
            f"num_envs={self.num_envs}, "
            f"radius={self.rvo_radius:.2f}m, "
            f"culling={self.culling_radius:.1f}m, "
            f"horizon_dyn={self.time_horizon_dynamic:.1f}s (dynamic only, static excluded), "
            f"ego_responsibility={self.ego_responsibility:.0%}, "
            f"projection=NH-forward-biased(Kp={self.STEER_KP:.1f}, "
            f"reverse_thresh={math.degrees(self.REVERSE_ANGLE_THRESHOLD):.0f}°), "
            f"lateral_evasion=ON(ttc<{self.LATERAL_EVASION_TTC:.1f}s, "
            f"v_escape={self.LATERAL_EVASION_SPEED:.1f}m/s), "
            f"workers={'single' if self._pool is None else self._pool._max_workers}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # BEV Visualization (env 0 only)
    # ══════════════════════════════════════════════════════════════════════════

    def draw_on_bev(self, ax, yaw: float, frame: str = "world", show_overlay_text: bool = True) -> None:
        """Draw v_pref (blue), v_safe (green), VO cones on BEV axes."""
        scale = 3.0

        v_pref = self.v_pref_world * scale
        v_safe = self.v_safe_world * scale

        if frame == "body":
            v_pref = self._world_to_body_2d(v_pref, yaw)
            v_safe = self._world_to_body_2d(v_safe, yaw)

        # VO cones: semi-transparent wedges
        for rel_xy, combined_r in self._vo_cone_data:
            rel = rel_xy * scale if frame == "world" else self._world_to_body_2d(rel_xy, yaw) * scale
            dist = float(np.linalg.norm(rel_xy))
            if dist < combined_r or dist < 0.01:
                continue
            half_angle_rad = math.asin(min(combined_r / dist, 1.0))
            center_dir = math.atan2(float(rel[1]), float(rel[0]))
            theta1 = math.degrees(center_dir - half_angle_rad)
            theta2 = math.degrees(center_dir + half_angle_rad)
            cone_range = min(dist * scale, 6.0)
            wedge = mpatches.Wedge(
                (0, 0), cone_range, theta1, theta2,
                facecolor="#ff6600", edgecolor="#ff8800",
                alpha=0.15, linewidth=0.5, zorder=3,
            )
            ax.add_patch(wedge)

        # v_pref: blue arrow (wide + white outline for visibility)
        pref_mag = float(np.linalg.norm(v_pref))
        if pref_mag > 0.05:
            ax.arrow(
                0, 0, float(v_pref[0]), float(v_pref[1]),
                color="white", width=0.18, head_width=0.45,
                length_includes_head=True, zorder=7, alpha=0.6,
            )
            ax.arrow(
                0, 0, float(v_pref[0]), float(v_pref[1]),
                color="#4488ff", width=0.12, head_width=0.38,
                length_includes_head=True, zorder=8, alpha=0.95,
            )

        # v_safe: green arrow (narrower, on top)
        safe_mag = float(np.linalg.norm(v_safe))
        if safe_mag > 0.05:
            ax.arrow(
                0, 0, float(v_safe[0]), float(v_safe[1]),
                color="#44ff44", width=0.06, head_width=0.25,
                length_includes_head=True, zorder=9, alpha=0.9,
            )

        # Deviation line (red dashed)
        if self.orca_active:
            ax.plot(
                [float(v_pref[0]), float(v_safe[0])],
                [float(v_pref[1]), float(v_safe[1])],
                color="#ff4444", linestyle="--", linewidth=1.5,
                zorder=7, alpha=0.8,
            )

        if not show_overlay_text:
            return

        # Status indicator: purple "OSC" / yellow "REC" / red "RVO" / green "RL"
        rate = self.intervention_steps / max(1, self.total_steps)
        if self.recovery_active:
            label, label_color = "REC", "#ffaa00"
            bg_color, edge_color = "#403000", "#cc8800"
        elif self.displacement_stuck:
            label, label_color = "DISP", "#cc77ff"
            bg_color, edge_color = "#301040", "#9944cc"
        elif self.orca_active:
            label, label_color = "RVO", "#ff4444"
            bg_color, edge_color = "#401010", "#cc3333"
        else:
            label, label_color = "RL", "#44ff44"
            bg_color, edge_color = "#104010", "#33cc33"

        ax.text(
            0.02, 0.96, f" {label} ",
            transform=ax.transAxes, va="top", ha="left",
            color=label_color, fontsize=14, fontweight="bold",
            bbox={"facecolor": bg_color, "edgecolor": edge_color,
                  "alpha": 0.9, "boxstyle": "round,pad=0.3"},
        )
        info_text = (
            f"rate: {rate:.1%}  |v_pref|={pref_mag / scale:.2f}  |v_safe|={safe_mag / scale:.2f}"
        )
        ax.text(
            0.02, 0.86, info_text,
            transform=ax.transAxes, va="top", ha="left",
            color="#88ccff", fontsize=7,
            bbox={"facecolor": "#202040", "edgecolor": "#4466aa", "alpha": 0.75},
        )

    @staticmethod
    def _world_to_body_2d(vec_world: np.ndarray, yaw: float) -> np.ndarray:
        """World 2D → body 2D (inverse rotation). BEV body: x=left, y=fwd."""
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_fwd = cos_y * vec_world[0] + sin_y * vec_world[1]
        y_left = -sin_y * vec_world[0] + cos_y * vec_world[1]
        return np.array([y_left, x_fwd])

    # ══════════════════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        total = max(1, self.total_steps)
        return {
            "total_steps": self.total_steps,
            "intervention_steps": self.intervention_steps,
            "intervention_rate": self.intervention_steps / total,
            "fallback_steps": self.fallback_steps,
            "fallback_rate": self.fallback_steps / total,
            "recovery_count": self.recovery_count,
            "disp_recovery_count": self.displacement_recovery_count,
        }

    def print_stats(self) -> None:
        s = self.get_stats()
        print(f"\n[RVO2] Safety Filter Stats:")
        print(f"  Total steps:     {s['total_steps']}")
        print(f"  ORCA intervened: {s['intervention_steps']} ({s['intervention_rate']:.1%})")
        print(f"  Fallback (turn): {s['fallback_steps']} ({s['fallback_rate']:.1%})")
        print(f"  Recovery trigg:  {s['recovery_count']}")
        print(f"  Disp→recovery:   {s['disp_recovery_count']}")
        print(f"  Culling radius:  {self.culling_radius:.1f}m")
        print(f"  TimeHorizon:     dynamic={self.time_horizon_dynamic:.1f}s static={self.time_horizon_static:.1f}s")
        print(f"  Obstacles slots: {self._actual_num_obs}")

    def shutdown(self) -> None:
        """Clean up thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
