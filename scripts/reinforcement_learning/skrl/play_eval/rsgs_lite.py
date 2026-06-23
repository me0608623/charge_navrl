"""RSGS-Lite: Recovery-only Safer-Gap-style Goal Selector (Play-only module)

卡住偵測 -> LiDAR gap 搜尋 -> 恢復目標選取 -> 脫困後還原原始目標。
非侵入式：不改 obs 維度、不改 action decode、不改 reward、預設 OFF。

Integration point:
    env._local_goal_world [N, 2] (world frame) — obs function
    goal_position_in_robot_frame() checks hasattr(env, '_local_goal_world')
    and uses it instead of goal_command when present.

Usage in play_rnn_car.py:
    rsgs = RSGSLite(raw_env, num_envs, device, cfg=RSGSConfig(...))
    # after env.step():
    rsgs.step(obs_tensor)
    # on episode done:
    rsgs.reset(done_ids)
    # at end:
    rsgs.print_stats()
"""

from __future__ import annotations

import math

import numpy as np
import torch


class RSGSConfig:
    """RSGS-Lite parameters"""

    __slots__ = (
        "stuck_window",
        "stuck_threshold",
        "gap_min_width",
        "gap_clear_threshold",
        "recovery_distance",
        "max_recovery_steps",
        "exit_displacement",
        "goal_bias_weight",
    )

    def __init__(
        self,
        stuck_window: int = 15,
        stuck_threshold: float = 0.3,
        gap_min_width: int = 3,
        gap_clear_threshold: float = 0.10,
        recovery_distance: float = 2.0,
        max_recovery_steps: int = 50,
        exit_displacement: float = 1.0,
        goal_bias_weight: float = 0.3,
    ):
        self.stuck_window = stuck_window            # 位移偵測視窗 (步, 15 = 3s @5Hz)
        self.stuck_threshold = stuck_threshold      # 視窗內位移 < 此值 = stuck (m)
        self.gap_min_width = gap_min_width          # 最小 gap 寬度 (bins)
        self.gap_clear_threshold = gap_clear_threshold  # LiDAR normalized > 此值 = clear
        self.recovery_distance = recovery_distance  # 恢復目標距離 (m)
        self.max_recovery_steps = max_recovery_steps  # 恢復最長步數
        self.exit_displacement = exit_displacement  # 脫困成功位移門檻 (m)
        self.goal_bias_weight = goal_bias_weight    # gap 選擇時偏向 final goal 的權重 [0,1]


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
_NORMAL = 0
_RECOVERY = 1

# LiDAR layout in 139D obs
_LIDAR_START = 6
_LIDAR_END = 78
_N_BINS = 72


class RSGSLite:
    """Recovery-only Safer-Gap-style Goal Selector

    State machine (per-env):
        NORMAL   — 不介入，policy 直接追 goal_command
        RECOVERY — 設定 _local_goal_world 導向安全 gap
    """

    def __init__(
        self,
        raw_env,
        num_envs: int,
        device: torch.device,
        cfg: RSGSConfig | None = None,
    ):
        self.raw_env = raw_env
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg or RSGSConfig()

        # Per-env state
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.recovery_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.recovery_start_pos = torch.zeros(num_envs, 2, device=device)
        self.recovery_goal = torch.zeros(num_envs, 2, device=device)

        # Circular position buffer for stuck detection
        self.pos_history = torch.zeros(num_envs, self.cfg.stuck_window, 2, device=device)
        self._buf_idx = 0
        self._buf_filled = 0

        # Cooldown: steps since last recovery exit (avoid re-trigger)
        self._cooldown = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._cooldown_steps = self.cfg.stuck_window  # must fill buffer before next check

        # Pre-compute bin center angles: bin i -> body-frame angle
        # bin 0 ~ -pi (behind-left), bin 36 ~ 0 (forward), bin 71 ~ +pi
        bin_size = 2.0 * math.pi / _N_BINS
        self.bin_centers = torch.tensor(
            [-math.pi + (i + 0.5) * bin_size for i in range(_N_BINS)],
            dtype=torch.float32,
            device=device,
        )

        # Statistics
        self.stats_activations = 0
        self.stats_exits_success = 0
        self.stats_exits_timeout = 0
        self.stats_total_recovery_steps = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, obs_tensor: torch.Tensor) -> None:
        """Call once per step, AFTER env.step() returns next_obs.

        Sets raw_env._local_goal_world so that the NEXT obs computation
        (inside the next env.step) picks up the recovery goal.
        """
        robot_pos, yaw = self._get_robot_state()
        final_goal = self._get_final_goal()  # [N, 2] world frame
        lidar = obs_tensor[:, _LIDAR_START:_LIDAR_END]  # [N, 72] normalized

        # Update position buffer
        self.pos_history[:, self._buf_idx, :] = robot_pos
        self._buf_idx = (self._buf_idx + 1) % self.cfg.stuck_window
        self._buf_filled = min(self._buf_filled + 1, self.cfg.stuck_window)

        # Tick cooldown
        self._cooldown = torch.clamp(self._cooldown - 1, min=0)

        # --- Per-env logic ---
        any_recovery = False
        lidar_np = lidar.detach().cpu().numpy()

        for ei in range(self.num_envs):
            if self.state[ei] == _NORMAL:
                self._check_stuck(ei, robot_pos, yaw, final_goal, lidar_np)
            else:  # _RECOVERY
                self._update_recovery(ei, robot_pos)

            if self.state[ei] == _RECOVERY:
                any_recovery = True

        # --- Sync _local_goal_world ---
        if any_recovery:
            # Ensure attribute exists with correct shape
            if not hasattr(self.raw_env, "_local_goal_world") or self.raw_env._local_goal_world is None:
                self.raw_env._local_goal_world = final_goal.clone()
            # NORMAL envs: track goal_command; RECOVERY envs: use recovery_goal
            for ei in range(self.num_envs):
                if self.state[ei] == _NORMAL:
                    self.raw_env._local_goal_world[ei] = final_goal[ei]
                else:
                    self.raw_env._local_goal_world[ei] = self.recovery_goal[ei]
        else:
            # All NORMAL — remove override so obs reads goal_command directly
            if hasattr(self.raw_env, "_local_goal_world"):
                del self.raw_env._local_goal_world

    def reset(self, done_ids: torch.Tensor) -> None:
        """Call when episodes end (done_ids = env indices that reset)."""
        if done_ids.numel() == 0:
            return
        self.state[done_ids] = _NORMAL
        self.recovery_step[done_ids] = 0
        self.pos_history[done_ids] = 0.0
        self._cooldown[done_ids] = 0

        # Clean up _local_goal_world if no env in recovery
        if (self.state == _NORMAL).all() and hasattr(self.raw_env, "_local_goal_world"):
            del self.raw_env._local_goal_world

    def print_stats(self) -> None:
        """Print end-of-play summary."""
        print("\n--- RSGS-Lite 統計 ---")
        print(f"  啟動次數:       {self.stats_activations}")
        print(f"  脫困成功:       {self.stats_exits_success}")
        print(f"  恢復超時:       {self.stats_exits_timeout}")
        if self.stats_activations > 0:
            sr = self.stats_exits_success / self.stats_activations * 100
            avg = self.stats_total_recovery_steps / self.stats_activations
            print(f"  脫困成功率:     {sr:.1f}%")
            print(f"  平均恢復步數:   {avg:.1f}")
        print("---")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_robot_state(self):
        """Return (pos_world [N,2], yaw [N])."""
        robot = self.raw_env.scene["robot"]
        pos_w = robot.data.root_pos_w[:, :2].clone()
        quat_w = robot.data.root_quat_w
        qw, qx, qy, qz = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return pos_w, yaw

    def _get_final_goal(self) -> torch.Tensor:
        """Return goal_command world-frame [N, 2]."""
        return self.raw_env.command_manager.get_command("goal_command")[:, :2].clone()

    # --- Stuck detection ---

    def _check_stuck(self, ei: int, robot_pos: torch.Tensor,
                     yaw: torch.Tensor, final_goal: torch.Tensor,
                     lidar_np: np.ndarray) -> None:
        if self._buf_filled < self.cfg.stuck_window:
            return
        if self._cooldown[ei] > 0:
            return

        oldest_idx = self._buf_idx  # oldest entry (circular)
        oldest_pos = self.pos_history[ei, oldest_idx]
        current_pos = robot_pos[ei]
        disp = float((current_pos - oldest_pos).norm().item())

        if disp >= self.cfg.stuck_threshold:
            return

        # Stuck detected — find gaps and select recovery goal
        gaps = self._find_gaps(lidar_np[ei])
        if not gaps:
            return  # No safe gap, cannot recover

        rgoal = self._select_best_gap(
            gaps, robot_pos[ei], float(yaw[ei].item()), final_goal[ei]
        )
        if rgoal is None:
            return

        self.state[ei] = _RECOVERY
        self.recovery_step[ei] = 0
        self.recovery_start_pos[ei] = current_pos.clone()
        self.recovery_goal[ei] = rgoal
        self.stats_activations += 1

        # 計算 gap 角度 (body frame, degrees) 用於 logging
        gap_rel = rgoal - current_pos
        gap_angle_w = float(torch.atan2(gap_rel[1], gap_rel[0]).item())
        gap_angle_body_deg = math.degrees(gap_angle_w - float(yaw[ei].item()))
        # Normalize to [-180, 180]
        gap_angle_body_deg = (gap_angle_body_deg + 180) % 360 - 180
        print(
            f"[RSGS] env={ei} RECOVERY start "
            f"(disp={disp:.3f}m < {self.cfg.stuck_threshold}m, "
            f"{len(gaps)} gaps) "
            f"gap_dir={gap_angle_body_deg:+.0f}° "
            f"goal=({rgoal[0].item():+.2f}, {rgoal[1].item():+.2f})"
        )

    # --- Recovery update ---

    def _update_recovery(self, ei: int, robot_pos: torch.Tensor) -> None:
        self.recovery_step[ei] += 1
        self.stats_total_recovery_steps += 1

        current_pos = robot_pos[ei]
        disp = float((current_pos - self.recovery_start_pos[ei]).norm().item())

        if disp >= self.cfg.exit_displacement:
            self._exit_recovery(ei, "success")
            self.stats_exits_success += 1
        elif int(self.recovery_step[ei].item()) >= self.cfg.max_recovery_steps:
            self._exit_recovery(ei, "timeout")
            self.stats_exits_timeout += 1

    def _exit_recovery(self, ei: int, reason: str) -> None:
        steps = int(self.recovery_step[ei].item())
        self.state[ei] = _NORMAL
        self.recovery_step[ei] = 0
        # Cooldown: refill position buffer before next stuck check
        self._cooldown[ei] = self._cooldown_steps
        self.pos_history[ei] = 0.0  # clear stale history

        print(f"[RSGS] env={ei} RECOVERY end ({reason}, {steps} steps)")

    # --- Gap finding (circular LiDAR) ---

    def _find_gaps(self, lidar_1d: np.ndarray) -> list[dict]:
        """Find contiguous clear gaps in 72-bin circular LiDAR.

        Returns list of dicts with keys: width, center_bin, center_angle.
        Sorted by width descending.
        """
        n = _N_BINS
        threshold = self.cfg.gap_clear_threshold
        min_width = self.cfg.gap_min_width
        clear = lidar_1d > threshold

        if not clear.any():
            return []
        if clear.all():
            return [{
                "width": n,
                "center_bin": n // 2,
                "center_angle": float(self.bin_centers[n // 2]),
            }]

        # Start scan at a blocked bin to avoid splitting a gap
        first_blocked = int(np.argmin(clear))

        gaps = []
        gap_start = None

        for k in range(n):
            i = (first_blocked + k) % n
            if clear[i]:
                if gap_start is None:
                    gap_start = k
            else:
                if gap_start is not None:
                    width = k - gap_start
                    if width >= min_width:
                        center_k = gap_start + width // 2
                        center_bin = (first_blocked + center_k) % n
                        gaps.append({
                            "width": width,
                            "center_bin": int(center_bin),
                            "center_angle": float(self.bin_centers[center_bin]),
                        })
                    gap_start = None

        # Handle gap that runs to the end of the scan
        if gap_start is not None:
            width = n - gap_start
            if width >= min_width:
                center_k = gap_start + width // 2
                center_bin = (first_blocked + center_k) % n
                gaps.append({
                    "width": width,
                    "center_bin": int(center_bin),
                    "center_angle": float(self.bin_centers[center_bin]),
                })

        gaps.sort(key=lambda g: g["width"], reverse=True)
        return gaps

    # --- Goal selection ---

    def _select_best_gap(self, gaps: list[dict], robot_pos: torch.Tensor,
                         yaw: float, final_goal: torch.Tensor) -> torch.Tensor | None:
        """Score gaps and return world-frame recovery goal for the best one.

        Score = forward_factor * ((1 - bias) * width_norm + bias * goal_alignment)

        forward_factor 基於 gap 相對於目標方向的角度：
        - 目標方向 ±60°: factor ≈ 1.0 (最優)
        - 側向 ±90°:     factor ≈ 0.5 (可接受)
        - 後方 ±180°:    factor ≈ 0.1 (幾乎不選，除非唯一)
        """
        if not gaps:
            return None

        # Goal direction in body frame
        goal_rel = final_goal - robot_pos  # [2]
        goal_angle_world = float(torch.atan2(goal_rel[1], goal_rel[0]).item())
        goal_angle_body = goal_angle_world - yaw
        # Normalize to [-pi, pi]
        goal_angle_body = (goal_angle_body + math.pi) % (2 * math.pi) - math.pi

        max_width = max(g["width"] for g in gaps)
        bias = self.cfg.goal_bias_weight

        best_score = -1e9
        best_gap = None

        for g in gaps:
            width_score = g["width"] / max(max_width, 1)

            # Goal alignment: gap 中心與目標方向的 cos 相似度 [0, 1]
            angle_diff = g["center_angle"] - goal_angle_body
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            alignment_score = (math.cos(angle_diff) + 1.0) / 2.0  # [0, 1]

            # Forward factor: 基於 gap 與目標方向的夾角，抑制背離目標的 gap
            # cos(angle_diff)=1 → factor=1.0, cos=0 → 0.55, cos=-1 → 0.1
            forward_factor = max(0.1, (math.cos(angle_diff) + 1.0) / 2.0 * 0.9 + 0.1)

            base_score = (1.0 - bias) * width_score + bias * alignment_score
            score = forward_factor * base_score
            if score > best_score:
                best_score = score
                best_gap = g

        if best_gap is None:
            return None

        # Convert gap center angle (body frame) to world-frame goal position
        gap_angle_world = yaw + best_gap["center_angle"]
        dist = self.cfg.recovery_distance
        goal_world = torch.tensor(
            [
                float(robot_pos[0].item()) + dist * math.cos(gap_angle_world),
                float(robot_pos[1].item()) + dist * math.sin(gap_angle_world),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return goal_world
