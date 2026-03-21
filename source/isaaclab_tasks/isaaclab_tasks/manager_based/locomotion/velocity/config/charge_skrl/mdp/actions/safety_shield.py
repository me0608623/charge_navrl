"""Safety Shield Action — NavRL04 消融實驗

繼承 DiscreteDifferentialDriveAction，在 apply_actions() 前
根據 LiDAR 距離裁切前進速度。

Modes:
  "none": 不做任何裁切（baseline 行為）
  "soft": d_safe < shield_d_safe 時線性降速
  "hard": d_safe < shield_d_danger 時強制 v=0，只保留轉向
"""

from __future__ import annotations

import math
import torch
from typing import Sequence

from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

from .discrete_differential_drive import (
    DiscreteDifferentialDriveAction,
    DiscreteDifferentialDriveActionCfg,
)


class ShieldedDiscreteDifferentialDriveAction(DiscreteDifferentialDriveAction):
    """離散差速驅動 + LiDAR safety shield。

    Override apply_actions() 在寫入模擬器前裁切前進速度。
    Shield 只影響正向速度 (v > 0)，倒車不受限。
    """

    cfg: ShieldedDiscreteDifferentialDriveActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._shield_intervention_count = 0
        self._shield_override_count = 0
        self._shield_total_steps = 0

        if cfg.shield_mode != "none":
            print(
                f"[SHIELD] mode={cfg.shield_mode} "
                f"d_danger={cfg.shield_d_danger:.2f} "
                f"d_safe={cfg.shield_d_safe:.2f} "
                f"body_radius={cfg.body_radius:.2f}",
                flush=True,
            )

    def apply_actions(self):
        """Override: 在寫入模擬器前注入速度裁切。"""
        if self.cfg.shield_mode != "none":
            self._apply_shield()
        super().apply_actions()

    def _apply_shield(self):
        """根據 LiDAR bottom-10 距離裁切前進速度。"""
        sensor: RayCaster = self._env.scene.sensors[self.cfg.sensor_name]

        # 計算 d_safe (同 _get_lidar_safety_stats)
        sensor_pos_2d = sensor.data.pos_w[:, :2]
        hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
        distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
        distances_2d = torch.nan_to_num(
            distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance,
        )
        k = min(10, distances_2d.shape[1])
        bottom_k = torch.topk(distances_2d, k=k, dim=1, largest=False).values
        d_safe = (bottom_k.mean(dim=1) - self.cfg.body_radius).clamp(min=1e-4)  # [N]

        v_current = self._processed_actions[:, 0]  # next_velocity
        forward_mask = v_current > 0  # 只裁切前進

        N = v_current.shape[0]
        self._shield_total_steps += N

        if self.cfg.shield_mode == "soft":
            # 線性降速: d_safe 從 shield_d_safe 到 shield_d_danger 之間，速度從 100% 降到 0%
            needs_shield = (d_safe < self.cfg.shield_d_safe) & forward_mask
            if needs_shield.any():
                scale = ((d_safe - self.cfg.shield_d_danger) /
                         (self.cfg.shield_d_safe - self.cfg.shield_d_danger + 1e-6)).clamp(0.0, 1.0)
                old_v = self._processed_actions[needs_shield, 0].clone()
                self._processed_actions[needs_shield, 0] *= scale[needs_shield]
                self._current_velocity[needs_shield] = self._processed_actions[needs_shield, 0]
                actually_changed = (old_v - self._processed_actions[needs_shield, 0]).abs() > 0.01
                self._shield_intervention_count += needs_shield.sum().item()
                self._shield_override_count += actually_changed.sum().item()

        elif self.cfg.shield_mode == "hard":
            # 硬煞車: d_safe < d_danger → v=0
            emergency = (d_safe < self.cfg.shield_d_danger) & forward_mask
            if emergency.any():
                self._processed_actions[emergency, 0] = 0.0
                self._current_velocity[emergency] = 0.0
                self._shield_intervention_count += emergency.sum().item()
                self._shield_override_count += emergency.sum().item()

            # 軟降速: d_danger < d_safe < d_safe_threshold
            soft_zone = (d_safe >= self.cfg.shield_d_danger) & (d_safe < self.cfg.shield_d_safe) & forward_mask
            if soft_zone.any():
                scale = ((d_safe - self.cfg.shield_d_danger) /
                         (self.cfg.shield_d_safe - self.cfg.shield_d_danger + 1e-6)).clamp(0.0, 1.0)
                self._processed_actions[soft_zone, 0] *= scale[soft_zone]
                self._current_velocity[soft_zone] = self._processed_actions[soft_zone, 0]
                self._shield_intervention_count += soft_zone.sum().item()

    @property
    def shield_stats(self) -> dict:
        """Shield 統計量，供 diagnostics 讀取。"""
        total = max(self._shield_total_steps, 1)
        return {
            "shield_intervention_rate": self._shield_intervention_count / total,
            "shield_override_rate": self._shield_override_count / total,
            "shield_total_steps": self._shield_total_steps,
        }

    def reset_shield_stats(self):
        """重置統計計數器。"""
        self._shield_intervention_count = 0
        self._shield_override_count = 0
        self._shield_total_steps = 0


@configclass
class ShieldedDiscreteDifferentialDriveActionCfg(DiscreteDifferentialDriveActionCfg):
    """帶 safety shield 的離散差速驅動配置。"""
    class_type: type = ShieldedDiscreteDifferentialDriveAction
    shield_mode: str = "none"
    shield_d_danger: float = 0.55
    shield_d_safe: float = 1.2
    sensor_name: str = "lidar"
    body_radius: float = 0.35
