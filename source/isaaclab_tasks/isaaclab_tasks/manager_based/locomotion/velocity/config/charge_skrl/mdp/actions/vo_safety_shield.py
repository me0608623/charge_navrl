"""VO (Velocity Obstacle) Safety Shield — M1.2 NavRL-style predictive shield.

繼承 ShieldedDiscreteDifferentialDriveAction，把 distance-based shield 換成
velocity-aware 的預測式 shield。

核心想法（NavRL paper Section III-D 啟發）:
  1. 對每個 obstacle 計算 closest-point-of-approach (CPA) 與 time-to-CPA (TTC)
  2. 若 TTC 在 horizon 內 AND CPA 距離 < safety_radius → 預測碰撞
  3. 對被預測碰撞的 envs:
     a) 線性降速 (severity 高 → 速度更低)
     b) 角速度設為 evasion turn (遠離 obstacle 方向)
     c) hard mode 若 TTC 極短 → 強制 v=0

差別 vs distance shield:
  - distance shield 只看「現在最近的 ray」
  - VO shield 看「未來 horizon 秒內會不會碰到」
  - 對快速橫向移動的 obstacle 特別有效

設計選擇:
  - 簡化 heuristic (非真 LP)，每 obstacle ~10 ops，64 envs × 50 obs = 32k ops/step
  - 連續 (linear_v, angular_v) 投影，不破壞 PPO action distribution
  - 從 v21 best_agent warm start 友善（不改 obs 不改 architecture）

Modes:
  "none": 不做 shield (baseline)
  "soft": severity 線性降速 + 角速度 evasion
  "hard": severity > 0.7 強制 v=0
"""

from __future__ import annotations

import math
import torch
from typing import Sequence

from isaaclab.assets import Articulation
from isaaclab.utils import configclass

from .safety_shield import (
    ShieldedDiscreteDifferentialDriveAction,
    ShieldedDiscreteDifferentialDriveActionCfg,
)


class VOShieldedDiscreteDifferentialDriveAction(ShieldedDiscreteDifferentialDriveAction):
    """VO (Velocity Obstacle) shield — 預測式碰撞避免。

    覆寫 _apply_shield() 改用 obstacle velocity 預測未來碰撞。
    繼承 base shield 的所有 stats logging 與 inheritance pipeline。
    """

    cfg: VOShieldedDiscreteDifferentialDriveActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._vo_threat_count = 0  # 累計被 VO 判定為 threat 的 envs
        self._vo_evade_count = 0   # 累計實際 angular evasion 觸發次數
        # M1.3 debug: 周期性 print 統計
        self._vo_diag_step = 0
        self._vo_severity_sum = 0.0
        self._vo_severity_n = 0
        self._vo_diag_print_every = 100  # 每 100 個 _apply_shield 呼叫 print 一次
        if cfg.shield_mode != "none":
            print(
                f"[VO_SHIELD] horizon={cfg.vo_horizon:.2f}s "
                f"safety_radius={cfg.vo_safety_radius:.2f}m "
                f"max_obs={cfg.vo_max_obstacles} "
                f"evade_gain={cfg.vo_evade_gain:.2f}",
                flush=True,
            )

    def _apply_shield(self):
        """VO predictive shield — 取代 distance-based logic."""
        if self.cfg.shield_mode == "none":
            return

        device = self.device
        N = self._processed_actions.shape[0]

        # ====================================================================
        # Step 1: 取 robot state (world frame)
        # ====================================================================
        robot: Articulation = self._env.scene[self.cfg.asset_name]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [N, 2]
        robot_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)  # [N, 2]

        # 取 yaw (用 quaternion 算 atan2)
        quat = robot.data.root_quat_w  # [N, 4] (w,x,y,z)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # [N]

        # ====================================================================
        # Step 2: 取 obstacle states (position + velocity, world frame)
        # ====================================================================
        if not hasattr(self._env, "_obstacle_velocities"):
            # Fallback: 沒 obstacle velocity 資料 → 退回 base distance shield
            return super()._apply_shield()

        obs_velocities = self._env._obstacle_velocities  # [N, num_obs, 2]
        num_obs = min(obs_velocities.shape[1], self.cfg.vo_max_obstacles)

        # Build obstacle position tensor (從 scene entities)
        BIG_DIST = 1e6
        obs_positions = torch.full((N, num_obs, 2), BIG_DIST, device=device)
        obs_visible = torch.zeros(N, num_obs, dtype=torch.bool, device=device)

        for i in range(num_obs):
            name = f"obstacle_{i}"
            if name not in self._env.scene.keys():
                continue
            entity = self._env.scene[name]
            pos_w = entity.data.root_pos_w
            visible = pos_w[:, 2] > 0.0  # Z > 0 = visible (Z=-10 是 hidden)
            pos_xy = torch.nan_to_num(pos_w[:, :2], nan=BIG_DIST)
            obs_positions[:, i, :] = torch.where(
                visible.unsqueeze(-1), pos_xy, torch.full_like(pos_xy, BIG_DIST)
            )
            obs_visible[:, i] = visible

        # ====================================================================
        # Step 3: 計算相對 state (world frame)
        # ====================================================================
        rel_pos = obs_positions - robot_pos.unsqueeze(1)  # [N, num_obs, 2]
        rel_vel = obs_velocities[:, :num_obs, :] - robot_vel.unsqueeze(1)  # [N, num_obs, 2]

        # 距離與相對速度大小平方
        dist_sq = (rel_pos * rel_pos).sum(dim=-1)  # [N, num_obs]
        dist = dist_sq.sqrt().clamp(min=1e-3)
        rel_vel_sq = (rel_vel * rel_vel).sum(dim=-1).clamp(min=1e-4)  # avoid div by 0

        # ====================================================================
        # Step 4: 計算 Closest Point of Approach (CPA)
        # ====================================================================
        # t_cpa = -(rel_pos · rel_vel) / |rel_vel|²
        # 若 t_cpa > 0: 兩物還在接近 (對方仍將靠近最近點)
        # 若 t_cpa < 0: 兩物已過最近點，正在遠離
        dot = (rel_pos * rel_vel).sum(dim=-1)  # [N, num_obs]
        t_cpa = -dot / rel_vel_sq  # [N, num_obs]

        # CPA position (relative)
        cpa_rel_pos = rel_pos + rel_vel * t_cpa.unsqueeze(-1)  # [N, num_obs, 2]
        cpa_dist_sq = (cpa_rel_pos * cpa_rel_pos).sum(dim=-1)  # [N, num_obs]

        # ====================================================================
        # Step 5: Threat 判定
        # ====================================================================
        safety_radius_sq = self.cfg.vo_safety_radius ** 2

        # Threat conditions:
        #   (a) obstacle visible
        #   (b) CPA in future (t_cpa > 0) and within horizon (t_cpa < horizon)
        #   (c) CPA distance < safety_radius (will collide)
        #   (d) currently NOT already in collision (dist > safety_radius/2)
        is_threat = (
            obs_visible
            & (t_cpa > 0.0)
            & (t_cpa < self.cfg.vo_horizon)
            & (cpa_dist_sq < safety_radius_sq)
            & (dist_sq > 0.01)  # 避免處理已碰撞的 obstacle
        )

        # ====================================================================
        # Step 6: 對 threat 的 envs 做 shielding
        # ====================================================================
        # 找每個 env 的 worst threat (最早碰撞，即最小 t_cpa)
        t_cpa_threats = torch.where(is_threat, t_cpa, torch.full_like(t_cpa, 1e6))
        min_ttc, worst_obs_idx = t_cpa_threats.min(dim=1)  # [N], [N]
        has_threat = min_ttc < 1e6  # [N]

        forward_mask = self._processed_actions[:, 0] > 0.0
        needs_shield = has_threat & forward_mask

        self._shield_total_steps += N
        self._vo_diag_step += 1

        # === M1.3 DIAG ===
        # 計算 visible threats (in horizon AND in safety radius), 不管是否 forward
        visible_threats_per_env = is_threat.sum(dim=1)  # [N]
        # 統計：當前 step 有多少 envs 至少有 1 個 visible threat?
        envs_with_visible_threat = (visible_threats_per_env > 0).sum().item()
        # 多少 envs needs shield (含 forward 限制)?
        envs_needs_shield = needs_shield.sum().item() if needs_shield.any() else 0

        # 周期性 print
        if self._vo_diag_step % self._vo_diag_print_every == 0:
            avg_sev = self._vo_severity_sum / max(self._vo_severity_n, 1)
            print(
                f"[VO_DIAG] step#{self._vo_diag_step} | "
                f"visible_threats: {envs_with_visible_threat}/{N} envs "
                f"({100*envs_with_visible_threat/N:.0f}%) | "
                f"shield_fired: {envs_needs_shield}/{N} envs | "
                f"avg_severity: {avg_sev:.3f} | "
                f"cumulative threats: {self._vo_threat_count}, evades: {self._vo_evade_count}",
                flush=True,
            )

        if not needs_shield.any():
            return

        n_threatened = needs_shield.sum().item()
        self._vo_threat_count += n_threatened
        self._shield_intervention_count += n_threatened

        # ====================================================================
        # Step 7: 計算 evasion (linear velocity reduction + angular turn)
        # ====================================================================
        env_idx = torch.nonzero(needs_shield, as_tuple=False).squeeze(-1)  # [N_threat]
        worst_idx_t = worst_obs_idx[env_idx]  # [N_threat]

        # Severity: 1.0 = TTC=0 (imminent), 0.0 = TTC=horizon (just at limit)
        # M1.3 Path A: 改用 sqrt 讓 severity 早期更激進
        # 例如 TTC=1.4, horizon=2.0 → linear=0.30, sqrt=√0.30=0.55
        ttc_threat = min_ttc[env_idx]  # [N_threat]
        severity_linear = (1.0 - ttc_threat / self.cfg.vo_horizon).clamp(0.0, 1.0)
        severity = severity_linear.sqrt()  # 開根號 → 凸曲線早期更陡
        # M1.3 DIAG: track severity stats
        self._vo_severity_sum += severity.sum().item()
        self._vo_severity_n += severity.shape[0]

        # ----- 7a. Linear velocity reduction (M1.3 Path A: 二次降速 + hard fallback) -----
        # Hard fallback: TTC < 0.5s 強制 v=0 (regardless of mode)
        hard_stop = ttc_threat < 0.5
        if self.cfg.shield_mode == "soft":
            # Soft mode: v_scale = (1-severity)^2, severity=0.55 → v_scale=0.20
            v_scale = (1.0 - severity).clamp(min=0.0) ** 2
        else:  # "hard"
            # Hard mode: severity > 0.7 強制停止，否則 (1-severity)^2
            v_scale = torch.where(
                severity > 0.7,
                torch.zeros_like(severity),
                (1.0 - severity).clamp(min=0.0) ** 2,
            )
        # 套用 hard_stop fallback (覆蓋上面的 v_scale)
        v_scale = torch.where(hard_stop, torch.zeros_like(v_scale), v_scale)

        old_v = self._processed_actions[env_idx, 0].clone()
        self._processed_actions[env_idx, 0] = old_v * v_scale
        self._current_velocity[env_idx] = self._processed_actions[env_idx, 0]

        actually_changed = (old_v - self._processed_actions[env_idx, 0]).abs() > 0.01
        self._shield_override_count += actually_changed.sum().item()

        # ----- 7b. Angular evasion (turn away from worst threat) -----
        # 取 worst threat 的相對位置 (world frame)
        worst_rel_pos_x = rel_pos[env_idx, worst_idx_t, 0]
        worst_rel_pos_y = rel_pos[env_idx, worst_idx_t, 1]

        # Angle to obstacle (world frame)
        obs_angle_world = torch.atan2(worst_rel_pos_y, worst_rel_pos_x)
        # Robot heading
        robot_yaw_threat = robot_yaw[env_idx]
        # Angle to obstacle in body frame: positive = obstacle on left
        rel_angle = obs_angle_world - robot_yaw_threat
        # Normalize to [-pi, pi]
        rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))

        # Evasion angular velocity:
        #   obstacle on left (rel_angle > 0)  → turn right (negative omega)
        #   obstacle on right (rel_angle < 0) → turn left (positive omega)
        #   only evade if obstacle is in front (|rel_angle| < pi/2)
        in_front = rel_angle.abs() < (math.pi / 2.0)
        evade_omega = -torch.sign(rel_angle) * self.cfg.max_angular_vel * severity * self.cfg.vo_evade_gain
        # Override angular velocity only when threat is in front
        new_omega = torch.where(in_front, evade_omega, self._processed_actions[env_idx, 1])
        self._processed_actions[env_idx, 1] = new_omega.clamp(
            -self.cfg.max_angular_vel, self.cfg.max_angular_vel
        )

        self._vo_evade_count += in_front.sum().item()

    @property
    def shield_stats(self) -> dict:
        """擴展 base shield stats，加入 VO 專屬指標."""
        base_stats = super().shield_stats
        total = max(self._shield_total_steps, 1)
        base_stats.update({
            "vo_threat_rate": self._vo_threat_count / total,
            "vo_evade_rate": self._vo_evade_count / total,
        })
        return base_stats

    def reset_shield_stats(self):
        super().reset_shield_stats()
        self._vo_threat_count = 0
        self._vo_evade_count = 0


@configclass
class VOShieldedDiscreteDifferentialDriveActionCfg(ShieldedDiscreteDifferentialDriveActionCfg):
    """VO Shield 配置 — 預測式碰撞避免."""
    class_type: type = VOShieldedDiscreteDifferentialDriveAction

    # VO-specific parameters
    vo_horizon: float = 1.0
    """預測時間 (秒)。NavRL paper 沒明說，advisor 建議 1.0s 對齊 robot stopping distance."""

    vo_safety_radius: float = 0.45
    """碰撞安全半徑 (m)。body_radius (0.35) + buffer (0.10)."""

    vo_max_obstacles: int = 50
    """最多檢查的 obstacle 數 (跟 LiDAR ray 與 obstacle entity 上限一致)."""

    vo_evade_gain: float = 1.0
    """Angular evasion gain。1.0 = severity=1 時用 max_angular_vel 全速 turn."""
