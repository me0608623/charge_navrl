"""Curriculum-Aware Dual-Rate Normalizer (CADN)

Drop-in 替代 SKRL RunningStandardScaler，用於 observation preprocessing。
Per-branch 雙速率 EMA，在課程切換時快速適應分布漂移。

Usage:
    # 由 train_charge_ac.py 在 --use_cadn 時自動注入
    from cadn_preprocessor import CurriculumAwareDualRateNormalizer
    cadn = CurriculumAwareDualRateNormalizer(size=139, device="cuda:0")
    agent._state_preprocessor = cadn
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# Single-branch dual-rate EMA normalizer
# ═══════════════════════════════════════════════════════════════════════════

class _DualRateEMA(nn.Module):
    """Single-branch dual-rate EMA with drift detection."""

    def __init__(
        self,
        size: int,
        alpha_fast: float = 0.01,
        alpha_slow: float = 0.0005,
        tau_lo: float = 0.2,
        tau_hi: float = 0.5,
        clip_threshold: float = 5.0,
        epsilon: float = 1e-8,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.size = size
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.clip_threshold = clip_threshold
        self.epsilon = epsilon

        # Fast channel
        self.register_buffer("mu_f", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("var_f", torch.ones(size, dtype=torch.float64, device=device))
        # Slow channel
        self.register_buffer("mu_s", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("var_s", torch.ones(size, dtype=torch.float64, device=device))
        # Init flag
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        batch_mean = x.double().mean(dim=0)
        batch_var = x.double().var(dim=0, correction=0).clamp(min=self.epsilon)

        if not self._initialized.item():
            self.mu_f.copy_(batch_mean)
            self.var_f.copy_(batch_var)
            self.mu_s.copy_(batch_mean)
            self.var_s.copy_(batch_var)
            self._initialized.fill_(True)
            return

        # Fast channel
        self.mu_f.lerp_(batch_mean, self.alpha_fast)
        diff_f = batch_mean - self.mu_f
        self.var_f.lerp_(diff_f * diff_f + batch_var, self.alpha_fast)

        # Slow channel
        self.mu_s.lerp_(batch_mean, self.alpha_slow)
        diff_s = batch_mean - self.mu_s
        self.var_s.lerp_(diff_s * diff_s + batch_var, self.alpha_slow)

    def _active_stats(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (mu, var, blend_weight). blend_weight ∈ [0,1], 1=fast."""
        var_s_safe = self.var_s.clamp(min=self.epsilon)
        delta = (self.mu_f - self.mu_s).norm() / (
            math.sqrt(self.size) * var_s_safe.sqrt().norm().clamp(min=self.epsilon)
        )
        w = float(torch.clamp((delta - self.tau_lo) / (self.tau_hi - self.tau_lo + 1e-12), 0.0, 1.0).item())
        mu = w * self.mu_f + (1.0 - w) * self.mu_s
        var = w * self.var_f + (1.0 - w) * self.var_s
        return mu, var.clamp(min=self.epsilon), w

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mu, var, _ = self._active_stats()
        return torch.clamp(
            (x - mu.float()) / (var.float().sqrt() + self.epsilon),
            -self.clip_threshold,
            self.clip_threshold,
        )

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mu, var, _ = self._active_stats()
        return torch.clamp(x, -self.clip_threshold, self.clip_threshold) * var.float().sqrt() + mu.float()

    def get_drift(self) -> float:
        """Return drift magnitude (for diagnostics)."""
        var_s_safe = self.var_s.clamp(min=self.epsilon)
        delta = (self.mu_f - self.mu_s).norm() / (
            math.sqrt(self.size) * var_s_safe.sqrt().norm().clamp(min=self.epsilon)
        )
        return float(delta.item())

    def get_blend_weight(self) -> float:
        _, _, w = self._active_stats()
        return w


# ═══════════════════════════════════════════════════════════════════════════
# Main CADN class — drop-in for RunningStandardScaler
# ═══════════════════════════════════════════════════════════════════════════

class CurriculumAwareDualRateNormalizer(nn.Module):
    """Per-branch dual-rate EMA normalizer for 139D VLP16 observations.

    Observation layout: ego(4) + goal(2) + LiDAR(72) + obstacles(60) + time(1) = 139D

    Interface-compatible with skrl.resources.preprocessors.torch.RunningStandardScaler:
        forward(x, train=False, inverse=False, no_grad=True) -> Tensor
    """

    # Fixed obs layout (VLP16)
    _EGO_END = 4
    _GOAL_END = 6
    _LIDAR_END = 78
    _OBS_END = 138
    _TIME_END = 139

    def __init__(
        self,
        size: Union[int, Tuple[int], "gymnasium.Space"] = 139,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,  # Absorb any extra kwargs from Runner
    ):
        super().__init__()
        # Resolve size from gymnasium Space if needed
        if hasattr(size, "shape"):
            size = size.shape[-1] if hasattr(size.shape, "__len__") else size.shape
        elif isinstance(size, (tuple, list)):
            size = size[-1]
        self._obs_dim = int(size)
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        if self._obs_dim == 139:
            # Per-branch: State(7D), LiDAR(72D), Obstacle(60D)
            self.branch_state = _DualRateEMA(
                size=7, alpha_fast=0.008, alpha_slow=0.0003,
                tau_lo=0.25, tau_hi=0.55, clip_threshold=clip_threshold,
                epsilon=epsilon, device=device,
            )
            self.branch_lidar = _DualRateEMA(
                size=72, alpha_fast=0.005, alpha_slow=0.0002,
                tau_lo=0.3, tau_hi=0.6, clip_threshold=clip_threshold,
                epsilon=epsilon, device=device,
            )
            self.branch_obstacle = _DualRateEMA(
                size=60, alpha_fast=0.01, alpha_slow=0.0005,
                tau_lo=0.2, tau_hi=0.5, clip_threshold=clip_threshold,
                epsilon=1e-6, device=device,  # Larger epsilon for zero-padded obstacles
            )
            self._per_branch = True
        else:
            # Fallback: single branch (e.g., size=1 for value preprocessor)
            self.branch_single = _DualRateEMA(
                size=self._obs_dim, alpha_fast=0.01, alpha_slow=0.0005,
                tau_lo=0.2, tau_hi=0.5, clip_threshold=clip_threshold,
                epsilon=epsilon, device=device,
            )
            self._per_branch = False

    def forward(
        self,
        x: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
    ) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)

    def _compute(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False
    ) -> torch.Tensor:
        if not self._per_branch:
            # Single branch fallback
            if train:
                self.branch_single._update(x)
            return self.branch_single.denormalize(x) if inverse else self.branch_single.normalize(x)

        # --- Per-branch processing ---
        # Split: ego(0:4) + goal(4:6) + lidar(6:78) + obs(78:138) + time(138:139)
        ego = x[:, :self._EGO_END]
        goal = x[:, self._EGO_END:self._GOAL_END]
        lidar = x[:, self._GOAL_END:self._LIDAR_END]
        obstacles = x[:, self._LIDAR_END:self._OBS_END]
        time_feat = x[:, self._OBS_END:self._TIME_END]

        # Concatenate state branch: ego(4) + goal(2) + time(1) = 7D
        state_cat = torch.cat([ego, goal, time_feat], dim=-1)

        if train:
            self.branch_state._update(state_cat)
            self.branch_lidar._update(lidar)
            self.branch_obstacle._update(obstacles)

        if inverse:
            state_out = self.branch_state.denormalize(state_cat)
            lidar_out = self.branch_lidar.denormalize(lidar)
            obs_out = self.branch_obstacle.denormalize(obstacles)
        else:
            state_out = self.branch_state.normalize(state_cat)
            lidar_out = self.branch_lidar.normalize(lidar)
            obs_out = self.branch_obstacle.normalize(obstacles)

        # Reassemble original layout: ego(4) + goal(2) + lidar(72) + obs(60) + time(1)
        ego_out = state_out[:, :self._EGO_END]
        goal_out = state_out[:, self._EGO_END:self._GOAL_END]
        time_out = state_out[:, self._GOAL_END:]  # 1D
        return torch.cat([ego_out, goal_out, lidar_out, obs_out, time_out], dim=-1)

    def get_diagnostics(self) -> dict:
        """Return CADN diagnostic metrics for WandB logging."""
        if not self._per_branch:
            return {
                "cadn/drift_single": self.branch_single.get_drift(),
                "cadn/blend_weight_single": self.branch_single.get_blend_weight(),
            }
        return {
            "cadn/drift_state": self.branch_state.get_drift(),
            "cadn/drift_lidar": self.branch_lidar.get_drift(),
            "cadn/drift_obstacle": self.branch_obstacle.get_drift(),
            "cadn/blend_weight_state": self.branch_state.get_blend_weight(),
            "cadn/blend_weight_lidar": self.branch_lidar.get_blend_weight(),
            "cadn/blend_weight_obstacle": self.branch_obstacle.get_blend_weight(),
        }
