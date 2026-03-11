"""離散差速驅動動作 (Discrete Differential Drive Action)

動作空間：Discrete(361) = 19 (線加速度) × 19 (角速度)。

設計特點：
1. 中心對稱映射：index 9 = ratio 0.0（零動作），無 Prepend-Zero
2. 動態加速度邊界：依據當前速度計算合法加速度範圍，確保 v_next ∈ [-v_max, +v_max]
3. 允許倒車：速度域 [-1.0, +1.0] m/s
4. 正規化狀態輸出：ā_t, ω̄_t ∈ [-1, 1] 供觀測函數讀取

NN 輸出單一離散索引 action_index ∈ [0, 360]，經 divmod 拆解為：
  accel_idx = action_index // 19   ∈ [0, 18]
  omega_idx = action_index %  19   ∈ [0, 18]

再經中心對稱映射 + 動態邊界 → 實際物理指令。
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
    """離散差速驅動動作 — 19×19 = 361 離散索引 + 動態加速度邊界

    完整流程：
    1. NN 輸出 action_index ∈ [0, 360]
    2. divmod 拆解為 accel_idx (0~18) 與 omega_idx (0~18)
    3. 索引 → 意圖比例 ratio ∈ [-1, 1]（以 center=9 為零點）
    4. 計算動態合法加速度邊界（確保 v_next 不超出 ±v_max）
    5. ratio × 邊界 → 實際物理加速度 (m/s²) 與角速度 (rad/s)
    6. 速度積分：v_next = clamp(v + a × Δt, -v_max, +v_max)
    7. apply_actions：將速度寫入模擬器
    """

    cfg: DiscreteDifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DiscreteDifferentialDriveActionCfg, env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]

        # 控制時間間隔：decimation × sim_dt
        self._dt = env.step_dt

        N = env.num_envs

        # 動作緩衝區
        self._raw_actions = torch.zeros(N, 1, device=self.device)
        # processed_actions: [v_next, ω] — 供 apply_actions 使用
        self._processed_actions = torch.zeros(N, 2, device=self.device)
        # applied_accelerations: [actual_accel, actual_omega] — 供觀測函數讀取
        self._applied_accelerations = torch.zeros(N, 2, device=self.device)

        # 速度狀態
        self._current_velocity = torch.zeros(N, device=self.device)

        # 正規化狀態輸出（供 s_t^ego 的 ā_t, ω̄_t）
        self._a_bar = torch.zeros(N, device=self.device)
        self._omega_bar = torch.zeros(N, device=self.device)

        # 預計算常數
        self._center = cfg.num_bins // 2    # 9
        self._total_actions = cfg.num_bins ** 2  # 361

        # 啟動時列印配置
        if not hasattr(DiscreteDifferentialDriveAction, '_config_printed'):
            DiscreteDifferentialDriveAction._config_printed = True
            print(f"[ACTION] DiscreteDifferentialDrive: {cfg.num_bins}×{cfg.num_bins} = {self._total_actions} actions")
            print(f"  v_max={cfg.max_linear_velocity} m/s, a_max={cfg.max_linear_accel} m/s², "
                  f"ω_max={cfg.max_angular_vel:.4f} rad/s, dt={self._dt:.3f}s")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """1: 單一離散索引 (0 ~ 360)"""
        return 1

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
        """將離散索引解碼為物理指令。

        Args:
            actions: [num_envs, 1] float — NN 輸出的離散索引（浮點數）
        """
        self._raw_actions[:] = actions

        # ── 第一步：索引拆解 ──
        # action_index = accel_idx × 19 + omega_idx
        action_index = actions[:, 0].round().long().clamp(0, self._total_actions - 1)
        accel_idx = action_index // self.cfg.num_bins   # [N] 0~18
        omega_idx = action_index %  self.cfg.num_bins   # [N] 0~18

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
        v_max = self.cfg.max_linear_velocity

        allowable_accel_max = torch.min(
            torch.full_like(v, +a_max),
            (+v_max - v) / dt,
        )  # [N] 正向加速上界

        allowable_accel_min = torch.max(
            torch.full_like(v, -a_max),
            (-v_max - v) / dt,
        )  # [N] 反向制動下界（負值）

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

        # ── 第五步：角速度直接映射 ──
        #   actual_ω = ratio × ω_max
        actual_angular_vel = ratio_angular * self.cfg.max_angular_vel  # [N] rad/s

        # ── 第六步：速度積分 ──
        next_velocity = (v + actual_linear_accel * dt).clamp(-v_max, +v_max)

        # ── 第七步：正規化 → ā_t, ω̄_t ──
        self._a_bar[:] = actual_linear_accel / a_max
        self._omega_bar[:] = ratio_angular  # 本身就是 [-1, 1]

        # ── 儲存 ──
        self._applied_accelerations[:, 0] = actual_linear_accel
        self._applied_accelerations[:, 1] = actual_angular_vel
        self._processed_actions[:, 0] = next_velocity
        self._processed_actions[:, 1] = actual_angular_vel
        self._current_velocity[:] = next_velocity

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
        self._processed_actions[ids] = 0.0
        self._applied_accelerations[ids] = 0.0
        self._a_bar[ids] = 0.0
        self._omega_bar[ids] = 0.0

    def apply_actions(self):
        """將速度寫入模擬器。"""
        robot_quat_w = self._asset.data.root_quat_w  # [N, 4]
        num_envs = self._env.num_envs

        # 差速車：只有前進方向速度（body-frame x 軸）
        local_velocity = torch.zeros(num_envs, 3, device=self.device)
        local_velocity[:, 0] = self._current_velocity

        # body-frame → world-frame
        global_linear_velocity = math_utils.quat_apply(robot_quat_w, local_velocity)

        # 6D root velocity: [vx, vy, vz, wx, wy, wz]
        root_velocity = torch.zeros(num_envs, 6, device=self.device)
        root_velocity[:, 0:3] = global_linear_velocity
        root_velocity[:, 5] = self._processed_actions[:, 1]  # yaw angular velocity

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

                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.vel_current_visualizer = VisualizationMarkers(marker_cfg)
            self.vel_goal_visualizer.set_visibility(True)
            self.vel_current_visualizer.set_visibility(True)
        else:
            if hasattr(self, "vel_goal_visualizer"):
                self.vel_goal_visualizer.set_visibility(False)
                self.vel_current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "vel_goal_visualizer"):
            return
        if not self._asset.is_initialized:
            return

        base_pos_w = self._asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_goal_xy = self._current_velocity.unsqueeze(1)
        vel_goal_scale, vel_goal_quat = self._resolve_velocity_to_arrow(vel_goal_xy)

        robot_vel_w = self._asset.data.root_lin_vel_w[:, :2]
        vel_current_scale, vel_current_quat = self._resolve_velocity_to_arrow(robot_vel_w)

        self.vel_goal_visualizer.visualize(base_pos_w, vel_goal_quat, vel_goal_scale)
        self.vel_current_visualizer.visualize(base_pos_w, vel_current_quat, vel_current_scale)

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
    """離散差速驅動配置 — 19×19 中心對稱 + 動態加速度邊界

    Attributes:
        num_bins:             每軸離散格數（19 → 動作空間 361）
        max_linear_velocity:  線速度極限 (m/s)
        max_linear_accel:     線加速度極限 (m/s²)
        max_angular_vel:      角速度極限 (rad/s)
    """
    class_type: type = DiscreteDifferentialDriveAction
    asset_name: str = "robot"

    num_bins: int = 19                               # 19×19 = 361 離散動作
    max_linear_velocity: float = 1.0                 # m/s
    max_linear_accel: float = 0.5                    # m/s²
    max_angular_vel: float = 0.25 * math.pi          # rad/s ≈ 0.7854

    debug_vis: bool = True
