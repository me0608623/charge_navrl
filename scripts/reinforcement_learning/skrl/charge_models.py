# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-Branch CNN Feature Extractor for Charge Navigation (SKRL).

Architecture overview:
  Branch 1: LiDAR Conv1d   — exploits spatial locality of adjacent rays
  Branch 2: Obstacle encoder — permutation-invariant Top-K aggregation
  Branch 3: Robot state MLP  — low-dimensional proprioception

Fusion: cat([32, 32, 16]) = 80-dim embedding  (~15K params vs ~750K MLP)

Designed for 3-frame stacked observations.
Supports two per-frame layouts via (frame_dim, obs_start) parameters:

  frame_dim=111, obs_start=81 (Phase 0 CNN — no angular_vel, no ORCA):
  | Index     | Name                 | Dim |
  |-----------|----------------------|-----|
  | 0:72      | lidar_scan           |  72 |
  | 72:81     | state (9)            |   9 |
  | 81:111    | topk_obstacles (5x6) |  30 |

  frame_dim=120, obs_start=82 (Phase 0 full / Competitive):
  | Index     | Name                 | Dim |
  |-----------|----------------------|-----|
  | 0:72      | lidar_scan           |  72 |
  | 72:82     | state_core (10)      |  10 |
  | 82:112    | topk_obstacles (5x6) |  30 |
  | 112:120   | orca_features        |   8 |
  STATE_DIM = 10 + 8 = 18 (state_core + orca folded into state branch)
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin, MultiCategoricalMixin


# ============================================================================
# Core Feature Extractor
# ============================================================================

class ActorFeatureExtractor(nn.Module):
    """Multi-branch feature extractor with structural inductive biases.

    Splits a flat stacked observation into three branches:
      - LiDAR   [B, num_stack, 72]  -> Conv1d   -> [B, 32]
      - Obstacles [B, 5, 18]        -> MLP+MaxPool -> [B, 32]
      - State    [B, state*stack]   -> MLP       -> [B, 16]

    Output: [B, 80]

    Args:
        frame_dim: Per-frame observation dimensionality (111 or 120).
        num_stack: Number of stacked frames (default 3).
        obs_start: Per-frame index where topk_obstacles begin.
            If None, defaults to ``frame_dim - NUM_OBSTACLES * OBS_FEATURES``
            (backward compatible: 111-30=81).
    """

    LIDAR_DIM = 72
    NUM_OBSTACLES = 5
    OBS_FEATURES = 6  # per obstacle per frame

    def __init__(self, frame_dim: int = 111, num_stack: int = 3, obs_start: int | None = None):
        super().__init__()
        self.frame_dim = frame_dim
        self.num_stack = num_stack
        self.total_dim = frame_dim * num_stack

        # Derive layout from parameters
        obs_dim = self.NUM_OBSTACLES * self.OBS_FEATURES  # 30
        if obs_start is None:
            obs_start = frame_dim - obs_dim  # backward compat: 111-30=81
        self.obs_start = obs_start
        obs_end = obs_start + obs_dim

        # STATE_DIM = (state_core between lidar and obstacles)
        #           + (tail after obstacles, e.g. ORCA features)
        self.STATE_DIM = (obs_start - self.LIDAR_DIM) + (frame_dim - obs_end)
        # 111-dim: (81-72)+(111-111) = 9
        # 120-dim: (82-72)+(120-112) = 10+8 = 18

        # Pre-compute gather indices (registered as buffers for device tracking)
        self._build_index_buffers()

        # --- Branch 1: LiDAR Conv1d ---
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(num_stack, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.lidar_pool = nn.AdaptiveMaxPool1d(1)
        self.lidar_proj = nn.Linear(32, 32)
        self.lidar_ln = nn.LayerNorm(32)

        # --- Branch 2: Permutation-invariant obstacle encoder ---
        self.obs_mlp = nn.Sequential(
            nn.Linear(self.OBS_FEATURES * num_stack, 32),  # 18 -> 32
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.obs_ln = nn.LayerNorm(32)

        # --- Branch 3: Robot state MLP ---
        self.state_mlp = nn.Sequential(
            nn.Linear(self.STATE_DIM * num_stack, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.state_ln = nn.LayerNorm(16)

    def _build_index_buffers(self):
        """Pre-compute gather index tensors for slicing the flat observation."""
        obs_dim = self.NUM_OBSTACLES * self.OBS_FEATURES  # 30
        obs_end = self.obs_start + obs_dim

        # LiDAR: frames x lidar_dim contiguous
        lidar_idx = []
        for f in range(self.num_stack):
            base = f * self.frame_dim
            lidar_idx.extend(range(base, base + self.LIDAR_DIM))
        self.register_buffer("lidar_idx", torch.tensor(lidar_idx, dtype=torch.long))

        # State: (state_core before obstacles) + (tail after obstacles, e.g. ORCA)
        state_idx = []
        for f in range(self.num_stack):
            base = f * self.frame_dim
            # state_core: [base + LIDAR_DIM, base + obs_start)
            state_idx.extend(range(base + self.LIDAR_DIM, base + self.obs_start))
            # tail (e.g. ORCA): [base + obs_end, base + frame_dim)
            state_idx.extend(range(base + obs_end, base + self.frame_dim))
        self.register_buffer("state_idx", torch.tensor(state_idx, dtype=torch.long))

        # Obstacles: rearranged so same obstacle's temporal history is contiguous
        obs_idx = []
        for obs_i in range(self.NUM_OBSTACLES):
            for f in range(self.num_stack):
                start = f * self.frame_dim + self.obs_start + obs_i * self.OBS_FEATURES
                obs_idx.extend(range(start, start + self.OBS_FEATURES))
        self.register_buffer("obs_idx", torch.tensor(obs_idx, dtype=torch.long))

    @property
    def output_dim(self) -> int:
        """Feature embedding dimensionality: 32 + 32 + 16 = 80."""
        return 80

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-branch features from a flat stacked observation.

        Args:
            x: [B, 333] flat stacked observation

        Returns:
            [B, 80] fused feature embedding
        """
        # --- Branch 1: LiDAR Conv1d ---
        lidar = x[:, self.lidar_idx]  # [B, 216]
        lidar = lidar.view(-1, self.num_stack, self.LIDAR_DIM)  # [B, 3, 72]
        lidar = self.lidar_conv(lidar)  # [B, 32, 18]
        lidar = self.lidar_pool(lidar).squeeze(-1)  # [B, 32]
        lidar = self.lidar_ln(self.lidar_proj(lidar))  # [B, 32]

        # --- Branch 2: Obstacles (permutation invariant) ---
        obs = x[:, self.obs_idx]  # [B, 90]
        obs = obs.view(-1, self.NUM_OBSTACLES, self.OBS_FEATURES * self.num_stack)  # [B, 5, 18]
        obs = self.obs_mlp(obs)  # [B, 5, 32] — shared MLP broadcasts over dim=1
        obs = obs.max(dim=1).values  # [B, 32] — permutation invariant aggregation
        obs = self.obs_ln(obs)  # [B, 32]

        # --- Branch 3: State ---
        state = x[:, self.state_idx]  # [B, 27]
        state = self.state_mlp(state)  # [B, 16]
        state = self.state_ln(state)  # [B, 16]

        return torch.cat([lidar, obs, state], dim=-1)  # [B, 80]


# ============================================================================
# SKRL Policy (Actor)
# ============================================================================

class ChargePolicy(GaussianMixin, Model):
    """SKRL Gaussian policy using multi-branch feature extractor.

    Input:  stacked observation (from AAC wrapper)
    Output: 2-dim action [forward_velocity, angular_velocity]

    Architecture:
      ActorFeatureExtractor -> 80-dim -> Linear(80,64) -> ReLU -> Linear(64,2)
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        # Custom architecture params
        frame_dim: int = 111,
        num_stack: int = 3,
        obs_start: int | None = None,
        # GaussianMixin params (explicit — SKRL does not accept **kwargs)
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        reduction: str = "sum",
        role: str = "",
        **kwargs,  # absorb any extra YAML keys silently
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)

        self.extractor = ActorFeatureExtractor(frame_dim=frame_dim, num_stack=num_stack, obs_start=obs_start)
        self.head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), initial_log_std))

    def compute(self, inputs, role=""):
        """Compute action mean and log_std.

        Args:
            inputs: dict with "states" key -> [B, 333] stacked observation
            role: model role (unused)

        Returns:
            Tuple of (mean_actions, log_std_parameter, empty_dict)
        """
        features = self.extractor(inputs["states"])
        mean = self.head(features)
        return mean, self.log_std_parameter, {}


# ============================================================================
# SKRL Discrete Policy (Actor) — MultiDiscrete([21, 21])
# ============================================================================

class ChargeDiscretePolicy(MultiCategoricalMixin, Model):
    """SKRL MultiCategorical policy for discrete action space.

    Input:  stacked observation (from AAC wrapper)
    Output: 42-dim logits (21 accel bins + 21 omega bins)

    Architecture:
      ActorFeatureExtractor -> 80-dim -> Linear(80,64) -> ReLU -> Linear(64,42)
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        # Custom architecture params
        frame_dim: int = 111,
        num_stack: int = 3,
        obs_start: int | None = None,
        # MultiCategoricalMixin params
        unnormalized_log_prob: bool = True,
        reduction: str = "sum",
        role: str = "",
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction, role)

        self.extractor = ActorFeatureExtractor(frame_dim=frame_dim, num_stack=num_stack, obs_start=obs_start)
        self.head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 64),  # 80 -> 64
            nn.ReLU(),
            nn.Linear(64, self.num_actions),            # 64 -> 42 (21+21)
        )

    def compute(self, inputs, role=""):
        """Compute action logits.

        Args:
            inputs: dict with "states" key -> [B, 333] stacked observation
            role: model role (unused)

        Returns:
            Tuple of (logits [B, 42], empty_dict)
        """
        features = self.extractor(inputs["states"])
        logits = self.head(features)
        # Guard against NaN logits — Categorical with NaN triggers CUDA assert
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)
        return logits, {}


# ============================================================================
# SKRL Value Function (Critic)
# ============================================================================

class ChargeValue(DeterministicMixin, Model):
    """SKRL deterministic value function for AAC.

    Input:  stacked_policy_obs + privileged_info
    Output: scalar value

    Architecture:
      First policy_dim dims -> ActorFeatureExtractor -> 80-dim
      Last privileged_dim   -> Linear -> ReLU -> LayerNorm -> 32-dim
      Concat(80, 32) = 112 -> MLP -> 1
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        # Custom architecture params
        frame_dim: int = 111,
        num_stack: int = 3,
        obs_start: int | None = None,
        privileged_dim: int = 50,
        # DeterministicMixin params (explicit — SKRL does not accept **kwargs)
        clip_actions: bool = False,
        role: str = "",
        **kwargs,  # absorb any extra YAML keys silently
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions, role)

        self.policy_dim = frame_dim * num_stack
        self.privileged_dim = privileged_dim

        # Shared feature extractor (same architecture as actor)
        self.extractor = ActorFeatureExtractor(frame_dim=frame_dim, num_stack=num_stack, obs_start=obs_start)

        # Privileged information encoder
        self.priv_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )

        # Value head: 80 (extractor) + 32 (privileged) = 112
        self.value_head = nn.Sequential(
            nn.Linear(self.extractor.output_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role=""):
        """Compute state value.

        Args:
            inputs: dict with "states" key -> [B, 383] shared_states
            role: model role (unused)

        Returns:
            Tuple of (value, empty_dict)
        """
        states = inputs["states"]

        # Split: first 333 dims = stacked policy obs, last 50 = privileged
        policy_obs = states[:, :self.policy_dim]
        priv_obs = states[:, self.policy_dim:]

        features = self.extractor(policy_obs)  # [B, 80]
        priv_features = self.priv_encoder(priv_obs)  # [B, 32]

        combined = torch.cat([features, priv_features], dim=-1)  # [B, 112]
        value = self.value_head(combined)  # [B, 1]

        return value, {}


__all__ = [
    "ActorFeatureExtractor",
    "ChargePolicy",
    "ChargeDiscretePolicy",
    "ChargeValue",
]
