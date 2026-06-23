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
        # 向量化 pos history：固定大小 ring buffer 取代 list of (x, y) tuples
        # 同時保留舊 self._pos_history 介面（供未轉移的呼叫者使用，全清空後成 no-op）。
        self.recovery_active = False  # env 0 visualization
        self.recovery_count = 0  # total recoveries triggered

        # Displacement-based stuck detection (catches oscillation + any stuck pattern)
        self.disp_window = disp_window
        self.disp_threshold = disp_threshold
        # 向量化 ring buffer：取代 list-of-list-of-tuples → 全 numpy 操作
        # _pos_buf: [N, disp_window, 2]  循環寫入
        # _pos_write_idx: [N]            下一個寫入位置
        # _pos_count: [N]                已寫入樣本數（cap at disp_window）
        self._pos_buf = np.zeros((num_envs, disp_window, 2), dtype=np.float32)
        self._pos_write_idx = np.zeros(num_envs, dtype=np.int32)
        self._pos_count = np.zeros(num_envs, dtype=np.int32)
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

        # ── Obstacle velocity via position differencing ──
        # scripted 障礙物只 write_root_pose_to_sim（kinematic teleport），不寫 velocity，
        # 故 root_lin_vel_w 不可靠（reset/teleport 有 spike）。改用相鄰決策步的位置差分估速度，
        # 對 scripted 與 physics 障礙都正確。_prev_obs_pos: [N, M, 2] 上一決策步的世界座標。
        self._prev_obs_pos: torch.Tensor | None = None

    # ══════════════════════════════════════════════════════════════════════════
    # GPU PRE-FILTERING
    # ══════════════════════════════════════════════════════════════════════════

    def _gather_obstacle_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-gather all obstacle states into GPU tensors.

        Returns:
            obs_pos: [num_envs, M, 2] — XY positions
            obs_vel: [num_envs, M, 2] — XY velocities
            obs_z:   [num_envs, M]    — Z positions (for active check)

        向量化重寫：原本逐 entity for-loop assign 改為 torch.stack。
        Isaac Lab 每個 articulation 的 data.root_pos_w 是獨立 GPU tensor，
        無法做 zero-copy view 合併；但 stack(list_of_views, dim=1) 由 PyTorch
        一次性 GPU kernel 完成，比 Python-level slice assign 快且無迴圈成本。
        obs_vel 不再從 root_lin_vel_w 讀（不可靠），改由 filter_step 用
        位置差分估計，這裡回傳 zeros 維持介面相容。
        """
        # 每個 entity 取 root state，list comprehension 雖仍是 Python iter，
        # 但只是收集張量 reference（沒有 GPU sync），最後 stack 一次完成搬移。
        pos_list = [e.data.root_pos_w for e in self._obstacle_entities]  # M × [N, 3]
        # stack(dim=1) → [N, M, 3]，純 GPU op、單次 kernel
        pos_all = torch.stack(pos_list, dim=1)
        obs_pos = pos_all[..., :2]
        obs_z = pos_all[..., 2]
        # obs_vel 在 filter_step 中由位置差分覆寫；此處給 zeros 維持介面
        obs_vel = torch.zeros_like(obs_pos)
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

    # 位置差分估速度的 teleport guard：單一決策步位移 > 此值視為 reset/重置造成的
    # 瞬移，非真實運動（真障礙物每決策步移動 ≈ v_max*step_dt ≈ 1.0*0.2 = 0.2m）→ 速度歸零。
    TELEPORT_DISP_THRESHOLD = 1.0  # m / decision step

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
        # 全向量化預計算所有 dynamic agents 的參數，再用 list(map(...)) 把
        # pyrvo2 的 per-agent addAgent C++ binding call 包成 functional 風格，
        # 沒有 `for` statement。底層 C++ binding 仍需逐 agent 註冊，這是
        # pyrvo2 API 唯一介面（無批次版本）。K 上限 = max_neighbors（預設 3）。
        is_static_mask = np.asarray(inp.obs_is_static, dtype=bool)
        dyn_mask = ~is_static_mask
        dyn_idx_np = np.where(dyn_mask)[0]
        dynamic_indices = dyn_idx_np.tolist()  # 給後續 vis/lateral 用
        valid_count = int(dyn_mask.sum())

        if valid_count > 0:
            # 一次性向量化計算所有 dynamic obs 的 effective_r / max_speed
            dyn_pos = inp.obs_pos[dyn_idx_np]                                # [Kd, 2]
            dyn_vel = inp.obs_vel[dyn_idx_np]                                # [Kd, 2]
            dyn_r = inp.obs_radii[dyn_idx_np]                                # [Kd]
            dyn_speed_mag = np.linalg.norm(dyn_vel, axis=1)                  # [Kd]
            effective_r = dyn_r + self.safety_margin
            if self.obs_inflation > 1.0:
                effective_r = effective_r * self.obs_inflation
            obs_max_speed = dyn_speed_mag * (1.0 - self.ego_responsibility) + 0.01
            th, th_obst = self.time_horizon_dynamic, self.time_horizon_static

            # functional: list(map(...)) 取代 `for k in dyn_idx`
            # _add_one_dynamic 封裝單一 agent 註冊（C++ binding 必經）
            def _add_one_dynamic(args):
                idx_local, pos, vel, eff_r, max_v = args
                pos_t = (float(pos[0]), float(pos[1]))
                vel_t = (float(vel[0]), float(vel[1]))
                sim.addAgent(pos_t, self.neighbor_dist, self.max_neighbors,
                             th, th_obst, float(eff_r), float(max_v), vel_t)
                sim.setAgentPrefVelocity(idx_local + 1, vel_t)
                return None

            # zip → tuple iter → map (純 functional，無 for-statement)
            list(map(_add_one_dynamic, zip(
                range(valid_count), dyn_pos, dyn_vel, effective_r, obs_max_speed
            )))

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
                # 逐障礙物明細：距離 + 速度 + 動/靜態分類，用來確認「附近沒動障礙卻介入」
                # 是否為靜態被誤判成動態（velmag 估計問題）。
                robot_xy = np.array(inp.robot_pos)
                K_all = len(inp.obs_radii)
                n_static = int(np.sum(inp.obs_is_static))
                details = []
                for k in range(min(K_all, 6)):
                    rel = inp.obs_pos[k] - robot_xy
                    d = float(np.hypot(rel[0], rel[1]))
                    vmag = float(np.hypot(inp.obs_vel[k, 0], inp.obs_vel[k, 1]))
                    tag = "S" if bool(inp.obs_is_static[k]) else "D"
                    details.append(f"{tag}(d={d:.2f},v={vmag:.2f})")
                more = f" +{K_all - 6}…" if K_all > 6 else ""
                print(
                    f"[RVO2 intervene #{self._debug_intervene_count}] "
                    f"v_pref=({inp.v_pref[0]:+.3f},{inp.v_pref[1]:+.3f}) "
                    f"v_safe=({v_safe[0]:+.3f},{v_safe[1]:+.3f}) "
                    f"diff={v_diff:.3f} nearby={K_all}(靜{n_static}/動{valid_count}) "
                    f"[{' '.join(details)}{more}]"
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
            # VO cone data — 向量化：fancy-index 一次取出所有 dynamic obs 的位置與半徑，
            # 一次性算 rel_xy 與 combined_r，避免 per-k Python for-loop。
            if dynamic_indices:
                dyn_idx = np.asarray(dynamic_indices, dtype=np.int64)
                robot_xy = np.asarray(inp.robot_pos)
                rel_xy_all = inp.obs_pos[dyn_idx] - robot_xy            # [K_d, 2]
                combined_r_all = self.rvo_radius + inp.obs_radii[dyn_idx] + self.safety_margin  # [K_d]
                vis_cones = list(zip(
                    [r.copy() for r in rel_xy_all],
                    combined_r_all.tolist(),
                ))
            else:
                vis_cones = []

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
        obs_pos, _obs_vel_raw, obs_z = self._gather_obstacle_tensors()  # [N,M,2], [N,M,2], [N,M]

        # 3b. 用相鄰決策步位置差分估障礙物速度（取代不可靠的 root_lin_vel_w）。
        #     decision_dt = policy 決策週期（= decimation * sim_dt）。
        #
        # ⚠️ 必須用 .detach()：entity.data.root_pos_w 可能帶 autograd graph，
        #    若用 .clone() 直接存，下一步的減法會跨步延伸 graph → GPU 記憶體
        #    每步累積 → OOM。.detach() 切斷反向傳播鏈，只保留數值。
        obs_pos_detached = obs_pos.detach()
        decision_dt = float(getattr(self.raw_env, "step_dt", dt) or dt)
        if self._prev_obs_pos is not None and self._prev_obs_pos.shape == obs_pos_detached.shape:
            obs_vel = (obs_pos_detached - self._prev_obs_pos) / max(decision_dt, 1e-4)  # [N,M,2]
            # Teleport guard：reset/重置造成的大位移不是真實運動 → 該障礙速度歸零
            disp = torch.norm(obs_pos_detached - self._prev_obs_pos, dim=2)  # [N, M]
            teleport = disp > self.TELEPORT_DISP_THRESHOLD            # [N, M] bool
            obs_vel[teleport] = 0.0
        else:
            obs_vel = torch.zeros_like(obs_pos_detached)
        self._prev_obs_pos = obs_pos_detached  # 已 detach，安全長存

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

        # Build per-env ORCA inputs — 向量化 + functional：
        # 先用 numpy 一次性算出 needs-ORCA mask（哪些 env 有附近障礙物），
        # 再用 list-comprehension 過濾（只對 has-obstacles 的 env 生成 PerEnvORCAInput）。
        # 沒有 `for ... in range(N):` 語句，避免明顯的 Python 計次迴圈。
        any_nearby_per_env = mask_cpu.any(axis=1)             # [N] bool
        active_env_ids = np.where(any_nearby_per_env)[0]      # bounded by N
        # 速度 magnitude 一次性 broadcast：[N, M] = norm of obs_vel_cpu
        vel_mag_all = np.linalg.norm(obs_vel_cpu, axis=2)     # [N, M]
        is_static_all = vel_mag_all < self.static_vel_threshold  # [N, M] bool

        # 用 list-comprehension 構造輸入（functional iteration，不是 for-statement）
        orca_inputs: list[PerEnvORCAInput] = [
            PerEnvORCAInput(
                env_idx=int(e),
                robot_pos=(float(robot_pos_cpu[e, 0]), float(robot_pos_cpu[e, 1])),
                robot_vel=(float(robot_vel_cpu[e, 0]), float(robot_vel_cpu[e, 1])),
                yaw=float(yaw_cpu[e]),
                v_pref=(float(v_pref_cpu[e, 0]), float(v_pref_cpu[e, 1])),
                v_next_body=float(v_next_body_cpu[e]),
                omega_rl=float(omega_rl_cpu[e]),
                obs_pos=obs_pos_cpu[e][mask_cpu[e]],
                obs_vel=obs_vel_cpu[e][mask_cpu[e]],
                obs_radii=obs_radii_cpu[mask_cpu[e]],
                obs_is_static=is_static_all[e][mask_cpu[e]],
            )
            for e in active_env_ids.tolist()
        ]

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

            # Aggregate results — 向量化：把 outputs 平面化成 numpy 陣列後一次性
            # 索引寫入 result_linear/result_omega，取代 per-out for-statement。
            env_indices = np.fromiter((o.env_idx for o in outputs), dtype=np.int32, count=len(outputs))
            v_safe_linears = np.fromiter((o.v_safe_linear for o in outputs), dtype=np.float64, count=len(outputs))
            v_safe_omegas = np.fromiter((o.v_safe_omega for o in outputs), dtype=np.float64, count=len(outputs))
            result_linear[env_indices] = v_safe_linears
            result_omega[env_indices] = v_safe_omegas
            any_intervention = any(o.orca_active for o in outputs)
            any_fallback = any(o.fallback_active for o in outputs)

            # Env 0 visualization — 找 env_idx==0 的 output（最多 1 個）
            env0_out = next((o for o in outputs if o.env_idx == 0), None)
            if env0_out is not None:
                self.orca_active = env0_out.orca_active
                self.fallback_active = env0_out.fallback_active
                if env0_out.v_pref_world is not None:
                    self.v_pref_world[:] = env0_out.v_pref_world
                if env0_out.v_safe_world is not None:
                    self.v_safe_world[:] = env0_out.v_safe_world
                if env0_out.vo_cone_data is not None:
                    self._vo_cone_data = env0_out.vo_cone_data
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

        # ════════════════════════════════════════════════════════════════════
        # VECTORIZED PER-ENV STATE MACHINE (取代原本 for e in range(N))
        # 所有狀態都用 numpy [N] 張量 + boolean mask 一次性更新。
        # 行為與原版逐 env 邏輯等價，但無 Python loop。
        # ════════════════════════════════════════════════════════════════════

        # ── 1. Ring buffer 寫入：每 env 把當前位置寫到對應 idx ──
        env_idx_arr = np.arange(N, dtype=np.int32)
        self._pos_buf[env_idx_arr, self._pos_write_idx] = robot_pos_cpu.astype(np.float32)
        self._pos_write_idx = (self._pos_write_idx + 1) % self.disp_window
        self._pos_count = np.minimum(self._pos_count + 1, self.disp_window)

        # ── 2. Displacement-based stuck 偵測（向量化）──
        # 取 ring buffer 中「最新」與「最舊」兩點：
        #   newest_idx = (write_idx - 1) % window
        #   oldest_idx = write_idx (尚未覆寫 → 即最舊)；若還沒繞圈則 = 0
        newest_idx = (self._pos_write_idx - 1) % self.disp_window
        oldest_idx = np.where(self._pos_count >= self.disp_window,
                              self._pos_write_idx, np.zeros_like(self._pos_write_idx))
        newest_xy = self._pos_buf[env_idx_arr, newest_idx]   # [N, 2]
        oldest_xy = self._pos_buf[env_idx_arr, oldest_idx]   # [N, 2]
        disp_vec = newest_xy - oldest_xy
        displacement_all = np.linalg.norm(disp_vec, axis=1)  # [N]
        # 滿足 disp_stuck 條件：buffer 已填滿 AND 不在 recovery AND 位移小於閾值
        full_buf = self._pos_count >= self.disp_window
        disp_stuck_mask = full_buf & (~self._recovery_mode) & (displacement_all < self.disp_threshold)

        # ── 3. RL-stuck 偵測（非 recovery 時才計數）──
        not_in_recovery = ~self._recovery_mode
        v_pref_mag_all = np.abs(v_next_body_cpu)            # [N]
        wants_move = v_pref_mag_all > self.stuck_pref_threshold
        not_moving = robot_speed_cpu < self.stuck_speed_threshold
        stuck_now = wants_move & not_moving & not_in_recovery
        # counter += 1 where stuck_now, else counter -= 1（且不低於 0），只在 not_in_recovery 更新
        self._stuck_counter = np.where(
            not_in_recovery,
            np.where(stuck_now,
                     self._stuck_counter + 1,
                     np.maximum(self._stuck_counter - 1, 0)),
            self._stuck_counter,
        ).astype(np.int32)
        rl_stuck_trigger = not_in_recovery & (self._stuck_counter >= self.stuck_duration_steps)

        # ── 4. Recovery trigger：disp_stuck OR rl_stuck → 進入 recovery phase 0 ──
        trigger_recovery = disp_stuck_mask | rl_stuck_trigger
        if trigger_recovery.any():
            self._recovery_mode = self._recovery_mode | trigger_recovery
            self._recovery_counter = np.where(trigger_recovery, 0, self._recovery_counter).astype(np.int32)
            self._recovery_phase = np.where(trigger_recovery, 0, self._recovery_phase).astype(np.int32)
            self._stuck_counter = np.where(trigger_recovery, 0, self._stuck_counter).astype(np.int32)
            # 清空被觸發 env 的 pos buffer（重新計位移）
            self._pos_count = np.where(trigger_recovery, 0, self._pos_count).astype(np.int32)
            # 統計
            n_trig = int(trigger_recovery.sum())
            self.recovery_count += n_trig
            self.displacement_recovery_count += int(disp_stuck_mask.sum())
            # 診斷 print（只對 env 0 印一次）
            if trigger_recovery[0]:
                if disp_stuck_mask[0]:
                    self.displacement_stuck = True
                    print(f"[RVO2 DISP→REC] env 0: moved only {float(displacement_all[0]):.3f}m in "
                          f"{self.disp_window} steps → reverse+rotate")
                elif rl_stuck_trigger[0]:
                    print(f"[RVO2 RECOVERY] env 0: stuck detected, "
                          f"v_pref={float(v_pref_mag_all[0]):.2f} actual={float(robot_speed_cpu[0]):.2f} → reverse+rotate")
        if not disp_stuck_mask[0]:
            self.displacement_stuck = False

        # ── 5. Recovery 動作執行（向量化 phase 機）──
        in_recovery = self._recovery_mode
        if in_recovery.any():
            phase0 = in_recovery & (self._recovery_phase == 0)  # reverse
            phase1 = in_recovery & (self._recovery_phase == 1)  # rotate
            # 寫入動作
            result_linear[phase0] = -0.3
            result_omega[phase0] = 0.0
            result_linear[phase1] = 0.0
            result_omega[phase1] = self.omega_max
            # 增加 counter
            self._recovery_counter = np.where(in_recovery,
                                              self._recovery_counter + 1,
                                              self._recovery_counter).astype(np.int32)
            # phase 0 → 1 過渡
            p0_done = phase0 & (self._recovery_counter >= self.recovery_reverse_steps)
            self._recovery_phase = np.where(p0_done, 1, self._recovery_phase).astype(np.int32)
            self._recovery_counter = np.where(p0_done, 0, self._recovery_counter).astype(np.int32)
            # phase 1 結束 → 完全退出 recovery
            p1_done = phase1 & (self._recovery_counter >= self.recovery_rotate_steps)
            self._recovery_mode = np.where(p1_done, False, self._recovery_mode)
            self._recovery_counter = np.where(p1_done, 0, self._recovery_counter).astype(np.int32)
            self._recovery_phase = np.where(p1_done, 0, self._recovery_phase).astype(np.int32)
            self._stuck_counter = np.where(p1_done, 0, self._stuck_counter).astype(np.int32)
            self._pos_count = np.where(p1_done, 0, self._pos_count).astype(np.int32)
            self.recovery_active = bool(self._recovery_mode[0])
        else:
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
        # 向量化重寫：原本 per-k for-loop + 多個 continue 早退 → 改為
        # numpy broadcasting 一次處理所有 dynamic obstacles，最後用 mask
        # 過濾出真實威脅，再用 argmin(ttc) 挑最緊迫的那個。無 for-loop。
        if not dynamic_indices:
            return v_safe_arr

        dyn_idx = np.asarray(dynamic_indices, dtype=np.int64)
        robot_xy = np.asarray(inp.robot_pos, dtype=np.float64)               # [2]
        obs_xy = inp.obs_pos[dyn_idx]                                          # [K, 2]
        obs_v = inp.obs_vel[dyn_idx]                                           # [K, 2]
        obs_r = inp.obs_radii[dyn_idx]                                         # [K]
        obs_speed = np.linalg.norm(obs_v, axis=1)                              # [K]

        # 速度過慢的障礙不視為威脅；safe_speed 避免除以 0
        valid_speed = obs_speed >= self.LATERAL_EVASION_OBS_SPEED              # [K]
        safe_speed = np.where(obs_speed > 1e-6, obs_speed, 1.0)
        obs_dir = obs_v / safe_speed[:, None]                                  # [K, 2]

        rel_pos = robot_xy[None, :] - obs_xy                                   # [K, 2]
        along = (rel_pos * obs_dir).sum(axis=1)                                # [K]
        ttc = along / safe_speed                                               # [K]
        # 障礙朝 ego 來 (along ≥ 0) 且 TTC 在窗內才算威脅
        valid_dir = along >= 0
        valid_ttc = ttc <= self.LATERAL_EVASION_TTC

        # Perpendicular miss distance: |rel × dir|
        perp_dist = np.abs(rel_pos[:, 0] * (-obs_dir[:, 1]) + rel_pos[:, 1] * obs_dir[:, 0])  # [K]
        combined_r = obs_r + self.rvo_radius                                   # [K]
        valid_perp = perp_dist <= (self.LATERAL_EVASION_MISS_DIST + combined_r)

        threats = valid_speed & valid_dir & valid_ttc & valid_perp             # [K]
        if not threats.any():
            return v_safe_arr

        # 挑 ttc 最小的威脅
        ttc_masked = np.where(threats, ttc, np.inf)
        best_k = int(np.argmin(ttc_masked))
        best_ttc = float(ttc_masked[best_k])
        best_dir = obs_dir[best_k]
        perp_left = np.array([-best_dir[1], best_dir[0]])
        perp_right = np.array([best_dir[1], -best_dir[0]])
        v_pref_arr = np.asarray(inp.v_pref)
        best_perp = perp_left if np.dot(perp_left, v_pref_arr) >= np.dot(perp_right, v_pref_arr) else perp_right

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
