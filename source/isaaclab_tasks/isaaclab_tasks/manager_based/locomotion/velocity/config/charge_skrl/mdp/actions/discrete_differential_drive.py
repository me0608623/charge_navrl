"""離散差速驅動動作 (Discrete Differential Drive Action)

動作空間：MultiDiscrete([19, 19]) — 線加速度 × 角速度，獨立採樣。

v3 變更：Discrete(361) → MultiDiscrete([19, 19])
  - NN 輸出 38 個 logits（19+19），兩組獨立 Categorical
  - 降低維度詛咒：從 361 種組合→19+19 獨立選擇
  - action_dim=2，process_actions 直接接收 [accel_idx, omega_idx]

設計特點：
1. 中心對稱映射：index 9 = ratio 0.0（零動作），無 Prepend-Zero
2. 動態加速度邊界：依據當前速度計算合法加速度範圍，確保 v_next ∈ [-v_max, +v_max]
3. 允許倒車：速度域 [-1.0, +1.0] m/s
4. 正規化狀態輸出：ā_t, ω̄_t ∈ [-1, 1] 供觀測函數讀取
"""

from __future__ import annotations

import math
import torch
from typing import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass


class DiscreteDifferentialDriveAction(ActionTerm):
    """離散差速驅動動作 — MultiDiscrete([19, 19]) + 動態加速度邊界

    完整流程：
    1. NN 輸出 [accel_idx, omega_idx]，各 ∈ [0, 18]
    2. 索引 → 意圖比例 ratio ∈ [-1, 1]（以 center=9 為零點）
    3. 計算動態合法加速度邊界（確保 v_next 不超出 ±v_max）
    4. ratio × 邊界 → 實際物理加速度 (m/s²) 與角速度 (rad/s)
    5. 速度積分：v_next = clamp(v + a × Δt, -v_max, +v_max)
    6. apply_actions：將速度寫入模擬器
    """

    cfg: DiscreteDifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DiscreteDifferentialDriveActionCfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]

        # 控制時間間隔：decimation × sim_dt
        self._dt = env.step_dt

        N = env.num_envs

        # 動作緩衝區（v3: [N, 2] = [accel_idx, omega_idx]）
        self._raw_actions = torch.zeros(N, 2, device=self.device)
        # processed_actions: [v_next, ω] — 供 apply_actions 使用
        self._processed_actions = torch.zeros(N, 2, device=self.device)
        # applied_accelerations: [actual_accel, actual_omega] — 供觀測函數讀取
        self._applied_accelerations = torch.zeros(N, 2, device=self.device)
        # commanded_accelerations: [issued_accel, issued_omega] — 最近送入致動器
        # queue 的命令。83D action history 使用這個值，與車端 policy_node 對齊。
        self._commanded_accelerations = torch.zeros(N, 2, device=self.device)
        # v3d: actuator tracking error [err_v_norm, err_w_norm] — 指令(pre-DR) − 實際(post-DR)
        #   = 致動延遲/馬達響應的「沒跟上」量。actuator DR 關閉或 delay=0 時恆為 0。
        #   供 obs action_error 模式讀取（顯式延遲簽名，訓練端建模延遲，論文 §34）。
        self._actuator_tracking_error = torch.zeros(N, 2, device=self.device)

        # Post-step diagnostics need the command that produced a terminal
        # transition. ManagerBasedRLEnv auto-resets done environments inside
        # env.step(), which clears the normal action state before play/eval can
        # inspect it. These snapshots are therefore updated by process_actions
        # and deliberately survive reset until the next command overwrites them.
        # They are read-only diagnostics and never feed back into control.
        self._last_pre_delay_command = torch.zeros(N, 2, device=self.device)
        self._last_post_delay_command = torch.zeros(N, 2, device=self.device)
        # Deployment-only output scaling is deliberately downstream of the
        # decoder, issued-command history, and actuator queue. This snapshot is
        # the command actually written to the simulator and survives auto-reset
        # so post-step evaluators can inspect terminal transitions.
        self._last_deployment_command = torch.zeros(N, 2, device=self.device)

        # 速度狀態
        self._current_velocity = torch.zeros(N, device=self.device)
        # 角速度狀態（用於 α slew clamp）
        self._current_omega = torch.zeros(N, device=self.device)
        # 上一筆已發出的角速度命令。啟用 actuator lag 後，它與實際角速度不同；
        # action-side slew 必須沿命令序列計算，才能對齊車端 decoder。
        self._commanded_omega = torch.zeros(N, device=self.device)

        # 正規化狀態輸出（供 s_t^ego 的 ā_t, ω̄_t）
        self._a_bar = torch.zeros(N, device=self.device)
        self._omega_bar = torch.zeros(N, device=self.device)

        # 預計算常數
        self._center = cfg.num_bins // 2    # 9

        # 啟動時列印配置
        if not hasattr(DiscreteDifferentialDriveAction, '_config_printed'):
            DiscreteDifferentialDriveAction._config_printed = True
            print(f"[ACTION] DiscreteDifferentialDrive: MultiDiscrete([{cfg.num_bins}, {cfg.num_bins}])")
            print(f"  v_max=+{cfg.max_linear_velocity:.2f}/-{cfg.max_linear_velocity*cfg.reverse_velocity_scale:.2f} m/s "
                  f"(rev_scale={cfg.reverse_velocity_scale:.2f}), "
                  f"a_max={cfg.max_linear_accel} m/s², "
                  f"ω_max={cfg.max_angular_vel:.4f} rad/s, α_max={cfg.max_angular_accel:.2f} rad/s², "
                  f"dt={self._dt:.3f}s")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """2: [accel_idx, omega_idx]，各 ∈ [0, num_bins-1]"""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """[num_envs, 2]: [next_velocity (m/s), angular_velocity (rad/s)]"""
        return self._processed_actions

    @property
    def applied_accelerations(self) -> torch.Tensor:
        """[num_envs, 2]: [actual_linear_accel (m/s²), actual_angular_vel (rad/s)]

        供觀測函數讀取裁切後的物理指令。
        """
        return self._applied_accelerations

    @property
    def commanded_accelerations(self) -> torch.Tensor:
        """[num_envs, 2]: issued [linear_accel, angular_velocity] commands.

        These are post-decode/post-slew but pre-delay commands. The last two
        entries form the pending-action queue exposed by the 83D observation.
        """
        return self._commanded_accelerations

    @property
    def actuator_tracking_error(self) -> torch.Tensor:
        """[num_envs, 2]: [err_v_norm, err_w_norm] — 致動延遲 tracking error。

        = (actuator DR 前的意圖速度指令) − (DR 後實際寫入 sim 的速度)，正規化到 ~[-1,1]。
        actuator DR 關閉或 delay=0 時恆為 0。供 obs action_error 模式作為顯式延遲簽名。
        """
        return self._actuator_tracking_error

    @property
    def last_pre_delay_command(self) -> torch.Tensor:
        """Last decoded ``[v, omega]`` command before actuator dynamics.

        Unlike ``processed_actions``, this transition snapshot is not erased
        by an auto-reset. It exists solely for post-step diagnostics.
        """
        return self._last_pre_delay_command

    @property
    def last_post_delay_command(self) -> torch.Tensor:
        """Last ``[v, omega]`` command after delay/scale/lag processing."""
        return self._last_post_delay_command

    @property
    def last_deployment_command(self) -> torch.Tensor:
        """Last ``[v, omega]`` command after deployment output scaling."""
        return self._last_deployment_command

    @property
    def a_bar(self) -> torch.Tensor:
        """[num_envs]: 正規化加速度 ā_t ∈ [-1, 1]"""
        return self._a_bar

    @property
    def omega_bar(self) -> torch.Tensor:
        """[num_envs]: 正規化角速度 ω̄_t ∈ [-1, 1]"""
        return self._omega_bar

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def process_actions(self, actions: torch.Tensor):
        """將 MultiDiscrete 索引解碼為物理指令。

        Args:
            actions: [num_envs, 2] float — NN 輸出的 [accel_idx, omega_idx]
        """
        # Keep the policy-issued indices. Actuator delay is applied after decode
        # to the physical (v, omega) command, matching the real cmd_vel pipeline.
        self._raw_actions[:] = actions

        # ── 第一步：直接取兩個獨立索引 ──
        accel_idx = actions[:, 0].round().long().clamp(0, self.cfg.num_bins - 1)  # [N] 0~18
        omega_idx = actions[:, 1].round().long().clamp(0, self.cfg.num_bins - 1)  # [N] 0~18

        # ── 第二步：索引 → 意圖比例 ratio ∈ [-1, 1] ──
        #   idx=0  → (-9)/9 = -1.0   idx=9  → 0/9 = 0.0   idx=18 → 9/9 = +1.0
        center = self._center
        ratio_linear  = (accel_idx.float() - center) / center   # [N]
        ratio_angular = (omega_idx.float() - center) / center   # [N]

        # ── 第三步：計算動態合法加速度邊界 ──
        #   v_next = v + a × Δt，要求 v_next ∈ [-v_max, +v_max]
        #   ⇒ a ≤ (+v_max - v) / Δt   ...上界
        #   ⇒ a ≥ (-v_max - v) / Δt   ...下界
        #   再與車體物理極限 ±a_max 取交集
        v = self._current_velocity
        dt = self._dt
        a_max = self.cfg.max_linear_accel
        v_max_pos = self.cfg.max_linear_velocity                            # 正向上限
        v_max_neg = self.cfg.max_linear_velocity * self.cfg.reverse_velocity_scale  # 反向上限（可縮放）

        allowable_accel_max = torch.min(
            torch.full_like(v, +a_max),
            (+v_max_pos - v) / dt,
        )  # [N] 正向加速上界

        allowable_accel_min = torch.max(
            torch.full_like(v, -a_max),
            (-v_max_neg - v) / dt,
        )  # [N] 反向制動下界（負值，受 reverse_velocity_scale 限制）

        # ── 第四步：ratio × 動態邊界 → 實際線加速度 ──
        #   ratio ≥ 0：accel = ratio × allowable_max（正向加速）
        #   ratio <  0：accel = -ratio × allowable_min（allowable_min 為負，乘積為負 → 制動）
        actual_linear_accel = torch.where(
            ratio_linear >= 0,
            ratio_linear * allowable_accel_max,
            -ratio_linear * allowable_accel_min,
        )  # [N] m/s²

        # 安全保底
        actual_linear_accel = torch.clamp(
            actual_linear_accel,
            min=allowable_accel_min,
            max=allowable_accel_max,
        )

        # ── 第五步：角速度映射 + α slew clamp ──
        #   target_ω = ratio × ω_max
        #   actual_ω = clamp(target_ω - ω_prev, ±α_max·dt) + ω_prev
        # 防止 policy 瞬間翻轉 ±ω_max 造成舞龍舞獅
        target_angular_vel = ratio_angular * self.cfg.max_angular_vel  # [N] rad/s
        max_dw = self.cfg.max_angular_accel * dt                       # 每步最大 Δω
        actual_angular_vel = self._commanded_omega + torch.clamp(
            target_angular_vel - self._commanded_omega,
            -max_dw, max_dw,
        )
        actual_angular_vel = actual_angular_vel.clamp(
            -self.cfg.max_angular_vel, self.cfg.max_angular_vel,
        )

        # ── 第六步：速度積分（正反向上限非對稱）──
        next_velocity = (v + actual_linear_accel * dt).clamp(-v_max_neg, +v_max_pos)

        # v3d: 擷取 actuator DR 之前的「意圖速度指令」(policy 想要的)，
        #      之後與 DR 後實際寫入 sim 的速度相減 → actuator tracking error。
        _v_intended = next_velocity.clone()
        _w_intended = actual_angular_vel.clone()

        # Preserve the issued-command history before actuator delay. For delay
        # d<=2 this is exactly the pending-action queue required by the augmented
        # state s_tilde=(s_t,a_{t-1},a_{t-2}).
        self._commanded_accelerations[:, 0] = actual_linear_accel
        self._commanded_accelerations[:, 1] = actual_angular_vel
        self._commanded_omega[:] = actual_angular_vel

        target_vel = torch.stack([next_velocity, actual_angular_vel], dim=1)
        self._last_pre_delay_command[:] = target_vel

        # Actuator DR is downstream of policy decoding, like the real cmd_vel
        # path: decoded command -> delay -> velocity scaling -> motor lag.
        if self.cfg.enable_actuator_dr:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.domain_randomization.actuator_dr import (
                apply_actuator_dynamics,
            )
            lag_alpha = self.cfg.actuator_motor_lag_by_channel
            if lag_alpha is None:
                lag_alpha = self.cfg.actuator_motor_lag
            target_vel = apply_actuator_dynamics(
                self._env,
                target_vel,
                self.cfg.actuator_delay_range,
                self.cfg.actuator_velocity_scale,
                lag_alpha,
            )

        next_velocity = target_vel[:, 0].clamp(-v_max_neg, +v_max_pos)
        actual_angular_vel = target_vel[:, 1].clamp(
            -self.cfg.max_angular_vel, self.cfg.max_angular_vel
        )
        self._last_post_delay_command[:, 0] = next_velocity
        self._last_post_delay_command[:, 1] = actual_angular_vel
        applied_linear_accel = (next_velocity - v) / dt

        # ── 第七步：正規化 → ā_t, ω̄_t ──
        self._a_bar[:] = applied_linear_accel / a_max
        self._omega_bar[:] = ratio_angular  # 本身就是 [-1, 1]

        # v3d: actuator tracking error = (意圖 pre-DR) − (實際 post-DR)，正規化到 ~[-1,1]
        #   actuator DR 關閉時 next_velocity / actual_angular_vel 未被改 → err = 0。
        self._actuator_tracking_error[:, 0] = (
            (_v_intended - next_velocity) / self.cfg.max_linear_velocity
        ).clamp(-1.0, 1.0)
        self._actuator_tracking_error[:, 1] = (
            (_w_intended - actual_angular_vel) / self.cfg.max_angular_vel
        ).clamp(-1.0, 1.0)

        # ── 儲存 ──
        self._applied_accelerations[:, 0] = applied_linear_accel
        self._applied_accelerations[:, 1] = actual_angular_vel
        self._processed_actions[:, 0] = next_velocity
        self._processed_actions[:, 1] = actual_angular_vel
        self._current_velocity[:] = next_velocity
        self._current_omega[:] = actual_angular_vel

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置速度狀態。"""
        if env_ids is None:
            ids = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().clone().to(device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            ids = env_ids
        else:
            ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._current_velocity[ids] = 0.0
        self._current_omega[ids] = 0.0
        self._commanded_omega[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._applied_accelerations[ids] = 0.0
        self._commanded_accelerations[ids] = 0.0
        self._actuator_tracking_error[ids] = 0.0
        self._a_bar[ids] = 0.0
        self._omega_bar[ids] = 0.0

    def apply_actions(self):
        """將速度寫入模擬器。"""
        robot_quat_w = self._asset.data.root_quat_w  # [N, 4]
        num_envs = self._env.num_envs

        # Match the real policy-node safety multiplier: keep decoder state and
        # issued-action history unscaled, and scale only the command sent to the
        # vehicle. Both linear and angular channels use the same speed_rate.
        deployment_scale = float(self.cfg.deployment_speed_scale)
        deployment_command = self._processed_actions * deployment_scale
        self._last_deployment_command[:] = deployment_command

        # 差速車：只有前進方向速度（body-frame x 軸）
        local_velocity = torch.zeros(num_envs, 3, device=self.device)
        local_velocity[:, 0] = deployment_command[:, 0]

        # body-frame → world-frame
        global_linear_velocity = math_utils.quat_apply(robot_quat_w, local_velocity)

        # 6D root velocity: [vx, vy, vz, wx, wy, wz]
        root_velocity = torch.zeros(num_envs, 6, device=self.device)
        root_velocity[:, 0:3] = global_linear_velocity
        root_velocity[:, 5] = deployment_command[:, 1]  # yaw angular velocity

        self._asset.write_root_velocity_to_sim(root_velocity)

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.vel_goal_visualizer = VisualizationMarkers(marker_cfg)
            # 藍色 vel_current 箭頭已永久移除（僅保留綠色 vel_goal 速度目標箭頭）。
            self.vel_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "vel_goal_visualizer"):
                self.vel_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "vel_goal_visualizer"):
            return
        if not self._asset.is_initialized:
            return

        base_pos_w = self._asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_goal_xy = self._current_velocity.unsqueeze(1)
        vel_goal_scale, vel_goal_quat = self._resolve_velocity_to_arrow(vel_goal_xy)

        self.vel_goal_visualizer.visualize(base_pos_w, vel_goal_quat, vel_goal_scale)

    def _resolve_velocity_to_arrow(self, xy_velocity: torch.Tensor):
        if hasattr(self, "vel_goal_visualizer"):
            default_scale = self.vel_goal_visualizer.cfg.markers["arrow"].scale
        else:
            default_scale = (0.5, 0.5, 0.5)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)

        if xy_velocity.shape[1] == 1:
            vel_magnitude = torch.abs(xy_velocity[:, 0])
            heading_angle = torch.where(
                xy_velocity[:, 0] >= 0,
                torch.zeros_like(xy_velocity[:, 0]),
                torch.full_like(xy_velocity[:, 0], math.pi),
            )
        else:
            vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
            heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])

        arrow_scale[:, 0] *= vel_magnitude * 3.0

        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self._asset.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@configclass
class DiscreteDifferentialDriveActionCfg(ActionTermCfg):
    """離散差速驅動配置 — MultiDiscrete([19, 19]) + 動態加速度邊界

    Attributes:
        num_bins:             每軸離散格數（19 → MultiDiscrete([19, 19])）
        max_linear_velocity:  線速度極限 (m/s)
        max_linear_accel:     線加速度極限 (m/s²)
        max_angular_vel:      角速度極限 (rad/s)
        max_angular_accel:    角加速度極限 (rad/s²) — 對應 cmd filter slew，防止舞龍舞獅
        reverse_velocity_scale: 反向速度上限縮放 (1.0=對稱，0.2=反向上限為 max×0.2)
    """
    class_type: type = DiscreteDifferentialDriveAction
    asset_name: str = "robot"

    num_bins: int = 19                               # 19×19 = 361 離散動作
    max_linear_velocity: float = 1.0                 # m/s
    max_linear_accel: float = 0.5                    # m/s²
    max_angular_vel: float = 0.25 * math.pi          # rad/s ≈ 0.7854
    max_angular_accel: float = 3.0                   # rad/s² — 對應 cmd_max_accel_angular
    reverse_velocity_scale: float = 1.0              # 反向速度縮放，1.0=對稱，0.2=強制前進偏好

    debug_vis: bool = True

    # --- Actuator domain randomization (off by default) ---
    enable_actuator_dr: bool = False
    actuator_delay_range: tuple[int, int] = (0, 2)         # [lo, hi] action delay steps (per-env, per-episode)
    actuator_velocity_scale: tuple[float, float] = (0.9, 1.1)  # per-episode velocity scale per (v, ω)
    actuator_motor_lag: float = 0.3                         # 1st-order low-pass α (0=no response, 1=instant)
    actuator_motor_lag_by_channel: tuple[float, float] | None = None  # optional (alpha_v, alpha_omega)
    # Deployment safety multiplier applied only when writing (v, omega) to sim.
    # Decoder integration, d1 queue, and 83D issued-action history stay unscaled.
    deployment_speed_scale: float = 1.0
