"""
VLP-16 v2 Network Models for Charge Navigation (SKRL).

v2 觀測空間 (139D Policy / 139D Critic) 三分支特徵萃取網路。
單幀 observation，無 frame stacking。

Architecture:
  Branch 1: LiDAR Conv1d — 72D (72 bins × 1 frame) → 64D
  Branch 2: Obstacle MLP — 60D (Top-10 × 6D) → 32D
  Branch 3: State MLP — 7D (ego 4 + goal 2 + time 1) → 32D
  Fusion: cat([64, 32, 32]) = 128D embedding

觀測空間佈局 (139D — concat by ObsManager):
  [0]      ego: normalized_linear_acceleration    1D
  [1]      ego: normalized_linear_velocity        1D
  [2]      ego: normalized_angular_velocity       1D
  [3]      ego: robot_radius                      1D
  [4:6]    goal: waypoint (x, y) in robot frame   2D
  [6:78]   static: LiDAR 72 bins                  72D
  [78:138] obs: Top-10 obstacles × 6D             60D
  [138]    time: remaining ratio                   1D

動作空間: Discrete(361) = 19×19 中心對稱 + 動態加速度邊界
  NN 輸出單一離散索引 action_index ∈ [0, 360]
  經 divmod 拆為 accel_idx (0~18) × omega_idx (0~18)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from gymnasium.spaces import Discrete

from skrl.models.torch import Model, DeterministicMixin, CategoricalMixin


# ============================================================================
# Observation layout constants (v2 — 139D)
# ============================================================================

EGO_START = 0
EGO_END = 4           # accel(1) + vel(1) + omega(1) + radius(1)
EGO_DIM = 4

GOAL_START = 4
GOAL_END = 6           # (x, y) in robot frame
GOAL_DIM = 2

LIDAR_START = 6
LIDAR_END = 78         # 72 bins × 1 frame
LIDAR_DIM = 72
LIDAR_BINS = 72

OBS_START = 78
OBS_END = 138          # Top-10 obstacles × 6D
OBS_DIM = 60
OBS_TOP_K = 10
OBS_PER_OBJ = 6       # (x, y, vx, vy, r, m)

TIME_START = 138
TIME_END = 139
TIME_DIM = 1

STATE_DIM = EGO_DIM + GOAL_DIM + TIME_DIM  # 7D (non-LiDAR, non-obstacle)

POLICY_DIM = 139
CRITIC_DIM = 139       # v2: symmetric critic (no privileged info yet)

# Action space constants — 19×19 = 361 flat discrete
NUM_BINS = 19
TOTAL_ACTIONS = NUM_BINS ** 2  # 361


# ============================================================================
# v2 Feature Extractor (3-branch)
# ============================================================================

class VLP16FeatureExtractor(nn.Module):
    """v2 三分支特徵提取器。

    Branch 1: LiDAR Conv1d
        [B, 72] → reshape [B, 1, 72] → Conv1d → MaxPool → 64D

    Branch 2: Obstacle MLP
        [B, 60] (Top-10 × 6D) → MLP → MaxPool over objects → 32D

    Branch 3: State MLP
        [B, 7] (ego 4 + goal 2 + time 1) → MLP → 32D

    Output: [B, 128]
    """

    def __init__(self):
        super().__init__()

        # --- Branch 1: LiDAR Conv1d (1 frame × 72 bins) ---
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),     # [B, 32, 72]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 36]
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 18]
            nn.ReLU(),
        )
        self.lidar_pool = nn.AdaptiveMaxPool1d(1)            # [B, 64, 1]
        self.lidar_proj = nn.Linear(64, 64)
        self.lidar_ln = nn.LayerNorm(64)

        # --- Branch 2: Obstacle MLP (Top-10 × 6D → per-object → MaxPool) ---
        # 排列不變：每個 obstacle 獨立通過同一 MLP，再做 MaxPool
        self.obs_mlp = nn.Sequential(
            nn.Linear(OBS_PER_OBJ, 32),   # 6 → 32
            nn.ReLU(),
            nn.Linear(32, 32),            # 32 → 32
            nn.ReLU(),
        )
        self.obs_ln = nn.LayerNorm(32)

        # --- Branch 3: State MLP (ego + goal + time = 7D) ---
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 32),   # 7 → 32
            nn.ReLU(),
            nn.Linear(32, 32),          # 32 → 32
            nn.ReLU(),
        )
        self.state_ln = nn.LayerNorm(32)

    @property
    def output_dim(self) -> int:
        """Feature embedding dimensionality: 64 + 32 + 32 = 128."""
        return 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from 139D flat observation.

        Args:
            x: [B, 139] flat observation

        Returns:
            [B, 128] fused feature embedding
        """
        # --- Branch 1: LiDAR ---
        lidar = x[:, LIDAR_START:LIDAR_END]                  # [B, 72]
        lidar = lidar.unsqueeze(1)                             # [B, 1, 72]
        lidar = self.lidar_conv(lidar)                         # [B, 64, 18]
        lidar = self.lidar_pool(lidar).squeeze(-1)             # [B, 64]
        lidar = self.lidar_ln(self.lidar_proj(lidar))          # [B, 64]

        # --- Branch 2: Obstacle (permutation-invariant MaxPool) ---
        obs_flat = x[:, OBS_START:OBS_END]                     # [B, 60]
        obs_reshaped = obs_flat.view(-1, OBS_TOP_K, OBS_PER_OBJ)  # [B, 10, 6]
        obs_per_obj = self.obs_mlp(obs_reshaped)               # [B, 10, 32]
        obs_pooled = obs_per_obj.max(dim=1).values             # [B, 32]
        obs_feat = self.obs_ln(obs_pooled)                     # [B, 32]

        # --- Branch 3: State (ego + goal + time) ---
        ego = x[:, EGO_START:EGO_END]                         # [B, 4]
        goal = x[:, GOAL_START:GOAL_END]                       # [B, 2]
        time = x[:, TIME_START:TIME_END]                       # [B, 1]
        state = torch.cat([ego, goal, time], dim=-1)           # [B, 7]
        state = self.state_mlp(state)                          # [B, 32]
        state = self.state_ln(state)                           # [B, 32]

        return torch.cat([lidar, obs_feat, state], dim=-1)     # [B, 128]


# ============================================================================
# SKRL Discrete Policy (Actor) — Discrete(361) = 19×19
# ============================================================================

class VLP16DiscretePolicy(CategoricalMixin, Model):
    """v2 Categorical policy for flat discrete action space.

    Input:  139D flat observation
    Output: 361-dim logits (19 accel × 19 omega, flat index)

    Architecture:
      VLP16FeatureExtractor → 128D → Linear(128,128) → ReLU → Linear(128,361)
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        unnormalized_log_prob: bool = True,
        role: str = "",
        **kwargs,
    ):
        discrete_action_space = Discrete(TOTAL_ACTIONS)
        Model.__init__(self, observation_space, discrete_action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob, role)

        self.extractor = VLP16FeatureExtractor()
        self.head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 128),   # 128 → 128
            nn.ReLU(),
            nn.Linear(128, TOTAL_ACTIONS),                # 128 → 361
        )

    def compute(self, inputs, role=""):
        states = inputs["states"]
        features = self.extractor(states)
        logits = self.head(features)
        if torch.isnan(logits).any():
            if not hasattr(self, '_nan_logit_warned'):
                self._nan_logit_warned = True
                print(f"[NaN POLICY] Logits contain NaN! input_nan={torch.isnan(states).any().item()}, "
                      f"features_nan={torch.isnan(features).any().item()}")
                for name, p in self.named_parameters():
                    if torch.isnan(p).any():
                        print(f"  PARAM NaN: {name}")
                        break
            logits = torch.nan_to_num(logits, nan=0.0)
        return logits, {}


# ============================================================================
# SKRL Value Function (Critic)
# ============================================================================

class VLP16Value(DeterministicMixin, Model):
    """v2 deterministic value function.

    Input:  139D (same as policy — no privileged info in v2)
    Output: scalar value

    Architecture:
      VLP16FeatureExtractor → 128D → MLP → 1
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        clip_actions: bool = False,
        role: str = "",
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions, role)

        self.extractor = VLP16FeatureExtractor()

        # Value head: 128D → MLP → 1
        self.value_head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 64),   # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Critic 輸出層初始化
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.constant_(self.value_head[-1].bias, -8.0)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        features = self.extractor(states)              # [B, 128]
        value = self.value_head(features)               # [B, 1]

        if torch.isnan(value).any():
            if not hasattr(self, '_nan_value_warned'):
                self._nan_value_warned = True
                print(f"[NaN CRITIC] Value output contains NaN!")
                print(f"  input_nan={torch.isnan(states).any().item()}, "
                      f"features_nan={torch.isnan(features).any().item()}")
                for name, p in self.named_parameters():
                    if torch.isnan(p).any():
                        print(f"  PARAM NaN: {name}")
                        break

        return value, {}


__all__ = [
    "VLP16FeatureExtractor",
    "VLP16DiscretePolicy",
    "VLP16Value",
    "POLICY_DIM",
    "CRITIC_DIM",
    "NUM_BINS",
    "TOTAL_ACTIONS",
]
