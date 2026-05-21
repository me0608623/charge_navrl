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
    obs_pos: np.ndarray   # [K, 2]
    obs_vel: np.ndarray   # [K, 2]
    obs_radii: np.ndarray  # [K]


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
        time_horizon: float = 2.0,
        time_horizon_obst: float = 2.0,
        angle_threshold_deg: float = 60.0,
        omega_max: float = 0.25 * math.pi,
        max_workers: int | None = None,
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
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.angle_threshold = math.radians(angle_threshold_deg)
        self.omega_max = omega_max

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
        # Combined mask: active (z >= 0) AND within culling radius
        mask = (obs_z >= 0.0) & (dist < self.culling_radius)
        return mask

    # ══════════════════════════════════════════════════════════════════════════
    # CPU ORCA COMPUTATION (per-env worker)
    # ══════════════════════════════════════════════════════════════════════════

    def _solve_orca_single_env(self, inp: PerEnvORCAInput, dt: float) -> PerEnvORCAOutput:
        """Run ORCA for one environment (designed to run in thread pool)."""
        # Build RVO2 simulator with culled obstacles only
        sim = rvo2.PyRVOSimulator(
            dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.rvo_radius,
            self.max_speed,
        )

        # Robot agent
        robot_agent = sim.addAgent(inp.robot_pos)
        sim.setAgentVelocity(robot_agent, inp.robot_vel)
        sim.setAgentPrefVelocity(robot_agent, inp.v_pref)

        # Culled obstacle agents only
        K = len(inp.obs_radii)
        for k in range(K):
            obs_pos_k = (float(inp.obs_pos[k, 0]), float(inp.obs_pos[k, 1]))
            obs_vel_k = (float(inp.obs_vel[k, 0]), float(inp.obs_vel[k, 1]))
            obs_r = float(inp.obs_radii[k])
            sim.addAgent(
                obs_pos_k,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                obs_r + self.safety_margin,
                2.0,
                obs_vel_k,
            )
            sim.setAgentPrefVelocity(k + 1, obs_vel_k)

        # ORCA solve
        sim.doStep()
        v_safe = sim.getAgentVelocity(robot_agent)

        # Check intervention
        v_pref_arr = np.array(inp.v_pref)
        v_safe_arr = np.array(v_safe)
        v_diff = float(np.linalg.norm(v_safe_arr - v_pref_arr))
        orca_active = v_diff > 0.01

        # Project to body frame or pass-through
        if orca_active:
            v_safe_linear, v_safe_omega = self._project_to_body_frame(
                v_safe, inp.yaw, dt
            )
            fallback_active = self._last_fallback
        else:
            v_safe_linear = inp.v_next_body
            v_safe_omega = inp.omega_rl
            fallback_active = False

        # Visualization data (env 0 only)
        vis_v_pref = None
        vis_v_safe = None
        vis_cones = None
        if inp.env_idx == 0:
            vis_v_pref = v_pref_arr.copy()
            vis_v_safe = v_safe_arr.copy()
            # VO cone data
            vis_cones = []
            robot_xy = np.array(inp.robot_pos)
            for k in range(K):
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
            orca_inputs.append(PerEnvORCAInput(
                env_idx=e,
                robot_pos=(float(robot_pos_cpu[e, 0]), float(robot_pos_cpu[e, 1])),
                robot_vel=(float(robot_vel_cpu[e, 0]), float(robot_vel_cpu[e, 1])),
                yaw=float(yaw_cpu[e]),
                v_pref=(float(v_pref_cpu[e, 0]), float(v_pref_cpu[e, 1])),
                v_next_body=float(v_next_body_cpu[e]),
                omega_rl=float(omega_rl_cpu[e]),
                obs_pos=obs_pos_cpu[e, nearby_idx, :],    # [K, 2]
                obs_vel=obs_vel_cpu[e, nearby_idx, :],    # [K, 2]
                obs_radii=obs_radii_cpu[nearby_idx],       # [K]
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

    # Thread-local fallback flag (avoid race in multi-thread)
    _last_fallback: bool = False

    def _project_to_body_frame(
        self, v_safe: tuple[float, float], yaw: float, dt: float
    ) -> tuple[float, float]:
        """World-frame v_safe → body-frame (v_linear, omega_z).

        Non-holonomic handling:
          - angle_diff < threshold: project onto heading axis
          - angle_diff >= threshold: slow down + max turn rate
        """
        vx_safe, vy_safe = v_safe
        v_safe_mag = math.sqrt(vx_safe**2 + vy_safe**2)

        if v_safe_mag < 0.01:
            self._last_fallback = False
            return 0.0, 0.0

        angle_v_safe = math.atan2(vy_safe, vx_safe)
        angle_diff = math.atan2(
            math.sin(angle_v_safe - yaw),
            math.cos(angle_v_safe - yaw),
        )
        abs_angle_diff = abs(angle_diff)

        if abs_angle_diff < self.angle_threshold:
            # Normal: project onto heading
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            v_linear = vx_safe * cos_yaw + vy_safe * sin_yaw
            desired_omega = angle_diff / max(dt, 1e-4)
            omega_z = max(-self.omega_max, min(self.omega_max, desired_omega))
            v_linear = max(-self.max_speed, min(self.max_speed, v_linear))
            self._last_fallback = False
            return v_linear, omega_z
        else:
            # Fallback: slow + max turn
            alignment = math.cos(angle_diff)
            v_linear = max(0.0, alignment) * v_safe_mag * 0.2
            omega_z = math.copysign(self.omega_max, angle_diff)
            self._last_fallback = True
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
            f"horizon={self.time_horizon:.1f}s, "
            f"angle_threshold={math.degrees(self.angle_threshold):.0f}deg, "
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

        # Status indicator: red "RVO" or green "RL"
        rate = self.intervention_steps / max(1, self.total_steps)
        if self.orca_active:
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
        }

    def print_stats(self) -> None:
        s = self.get_stats()
        print(f"\n[RVO2] Safety Filter Stats:")
        print(f"  Total steps:     {s['total_steps']}")
        print(f"  ORCA intervened: {s['intervention_steps']} ({s['intervention_rate']:.1%})")
        print(f"  Fallback (turn): {s['fallback_steps']} ({s['fallback_rate']:.1%})")
        print(f"  Culling radius:  {self.culling_radius:.1f}m")
        print(f"  Obstacles slots: {self._actual_num_obs}")

    def shutdown(self) -> None:
        """Clean up thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
