"""
VLP-16 v3 Network Models for Charge Navigation (SKRL).

v3 觀測空間 (139D Policy / 139D Critic) 三分支特徵萃取網路。
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

動作空間: MultiDiscrete([19, 19]) — 線加速度 × 角速度，獨立採樣
  NN 輸出 38 個 logits（19+19），兩組獨立 Categorical
  v3 變更：從 Discrete(361) 降維至 MultiDiscrete，解決維度詛咒
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import MultiDiscrete

from skrl.models.torch import Model, DeterministicMixin, MultiCategoricalMixin


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

# Action space constants — MultiDiscrete([19, 19])
NUM_BINS = 19
TOTAL_LOGITS = NUM_BINS * 2  # 38 (19 + 19, NOT 19² = 361)


# ============================================================================
# v2 Feature Extractor (3-branch)
# ============================================================================

class AttentionObstacleEncoder(nn.Module):
    """v24: HEIGHT-style cross-attention obstacle encoder.

    替換原本 `ObsMLP + MaxPool`。
    - Query = robot ego state (7D: ego4 + goal2 + time1)
    - Key/Value = TopK obstacles (K=10, 6D each)
    - MaxPool 丟失 pairwise interaction，在 dense dynamic scene 造成 ping-pong；
      cross-attention 讓 policy 能「以自己為中心」分配 attention 權重到威脅最大的 obstacle。

    保持 permutation invariance：attention 對 key 順序不變。

    Output: [B, 32]
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.obs_proj = nn.Linear(OBS_PER_OBJ, d_model)       # 6 → 32
        self.ego_proj = nn.Linear(STATE_DIM, d_model)         # 7 → 32
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(32)

    def forward(self, obstacles_6d: torch.Tensor, ego_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obstacles_6d: [B, K, 6] with last channel = mask (1=valid, 0=invalid)
            ego_state:    [B, 7]    (ego4 + goal2 + time1)
        Returns:
            [B, 32]
        """
        mask_valid = obstacles_6d[:, :, -1] > 0.5             # [B, K] bool
        # key_padding_mask: True 表示 ignore
        key_padding_mask = ~mask_valid                         # [B, K]

        # Edge case：envs with zero valid obstacles → MHA 會輸出 NaN。
        # 解法：強制保留 slot 0 為 valid（其內容是 zero-padded，attention 會輸出接近零的向量）
        all_invalid = key_padding_mask.all(dim=1)              # [B]
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False

        kv = self.obs_proj(obstacles_6d)                       # [B, K, d_model]
        q = self.ego_proj(ego_state).unsqueeze(1)              # [B, 1, d_model]

        attn_out, _ = self.mha(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )                                                      # [B, 1, d_model]
        feat = self.out(attn_out.squeeze(1))                   # [B, 32]

        # 對原本 all-invalid 的 env，強制輸出零（attention 給的也接近零，但保險）
        if all_invalid.any():
            feat = feat.masked_fill(all_invalid.unsqueeze(1), 0.0)

        return self.ln(feat)


class HeterogeneousAttentionEncoder(nn.Module):
    """v25: HEIGHT-style heterogeneous cross-attention encoder.

    針對 v24 失敗原因：LiDAR 分支用 AdaptiveMaxPool(1) 把 18 個 spatial tokens
    壓成 channel-wise scalar，丟失「哪個角度」資訊 → policy 無法定位 static walls
    → v24 訓練時 collision_static_ratio=0.61 主導碰撞，SR 卡在 62%。

    v25 修正：
      - LiDAR Conv1d 保留 18 個 spatial tokens（不做 final pool），當成 static nodes
      - 加 angular positional encoding (sin/cos)：告訴 model 每個 token 對應哪個角度
      - Dynamic obstacles 仍是 Top-K 10 個 nodes
      - Learnable type embedding 區分 static / dynamic（兩類語意分佈差很大）
      - 統一 concat → cross-attention，Q=ego，KV=[static_tokens, dynamic_tokens]

    這是 HEIGHT 原版做法（Liu et al. 2024, arXiv:2411.12150），不是自創。
    參考 open-source: Shuijing725/CrowdNav_Prediction_AttnGraph
    """

    def __init__(
        self,
        lidar_bins: int = LIDAR_BINS,
        lidar_conv_channels: int = 64,
        dynamic_k: int = OBS_TOP_K,
        dynamic_dim: int = OBS_PER_OBJ,
        ego_dim: int = STATE_DIM,
        d_model: int = 96,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # --- LiDAR → spatial tokens (NO final MaxPool → keep L=18 tokens) ---
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),             # [B, 32, 72]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 36]
            nn.ReLU(),
            nn.Conv1d(64, lidar_conv_channels, kernel_size=3, stride=2, padding=1),  # [B, 64, 18]
            nn.ReLU(),
        )
        self.num_static_tokens = lidar_bins // 4  # 72 → 18 after two stride=2
        self.static_proj = nn.Linear(lidar_conv_channels, d_model)

        # --- Angular positional encoding for static (LiDAR sector → angle) ---
        # Each of 18 tokens covers 20° of the 360° sweep
        angles = torch.linspace(0.0, 2.0 * torch.pi, self.num_static_tokens + 1)[:-1]
        pos_enc_sincos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # [L, 2]
        self.register_buffer("static_pos_sincos", pos_enc_sincos)
        self.static_pos_proj = nn.Linear(2, d_model)

        # --- Dynamic obstacle projection ---
        self.dynamic_proj = nn.Linear(dynamic_dim, d_model)
        self.dynamic_k = dynamic_k

        # --- Type embedding: 0=static, 1=dynamic ---
        self.type_emb = nn.Embedding(2, d_model)

        # --- Ego query projection ---
        self.ego_proj = nn.Linear(ego_dim, d_model)

        # --- Cross-attention ---
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        lidar_raw: torch.Tensor,
        dynamic: torch.Tensor,
        ego: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lidar_raw: [B, 72] raw LiDAR bins
            dynamic:   [B, K, 6] dynamic obstacles (last channel = mask)
            ego:       [B, 7] ego + goal + time
        Returns:
            [B, d_model]
        """
        B = lidar_raw.shape[0]
        device = lidar_raw.device

        # --- Static tokens from LiDAR Conv1d (NO final pool) ---
        lidar_in = lidar_raw.unsqueeze(1)                          # [B, 1, 72]
        lidar_feat = self.lidar_conv(lidar_in)                     # [B, 64, 18]
        lidar_feat = lidar_feat.transpose(1, 2).contiguous()       # [B, 18, 64]
        static_tok = self.static_proj(lidar_feat)                  # [B, 18, d_model]
        # Angular positional encoding (broadcast over batch)
        pos = self.static_pos_proj(self.static_pos_sincos)         # [18, d_model]
        static_tok = static_tok + pos.unsqueeze(0)
        # Type embedding (static=0)
        type_static = self.type_emb(torch.zeros(1, dtype=torch.long, device=device))  # [1, d]
        static_tok = static_tok + type_static                      # broadcast

        # --- Dynamic tokens ---
        mask_valid = dynamic[:, :, -1] > 0.5                       # [B, K]
        dyn_tok = self.dynamic_proj(dynamic)                       # [B, K, d_model]
        type_dyn = self.type_emb(torch.ones(1, dtype=torch.long, device=device))
        dyn_tok = dyn_tok + type_dyn

        # --- Concat KV: [static (18) + dynamic (K)] ---
        kv = torch.cat([static_tok, dyn_tok], dim=1)               # [B, 18+K, d_model]

        # Key padding mask: static always unmasked, dynamic from mask
        static_pad = torch.zeros(
            B, self.num_static_tokens, dtype=torch.bool, device=device
        )
        dyn_pad = ~mask_valid                                      # [B, K]
        key_padding_mask = torch.cat([static_pad, dyn_pad], dim=1) # [B, 18+K]
        # Static tokens always present → MHA never all-masked, no NaN guard needed

        # --- Ego query ---
        q = self.ego_proj(ego).unsqueeze(1)                        # [B, 1, d_model]

        attn_out, _ = self.mha(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )                                                          # [B, 1, d_model]
        feat = self.ln_out(attn_out.squeeze(1))                    # [B, d_model]
        return feat


class VLP16HeterogeneousExtractor(nn.Module):
    """v25: 2-branch feature extractor using HeterogeneousAttentionEncoder.

    Architecture:
      Branch 1: HeterogeneousAttentionEncoder (LiDAR tokens + dynamic + ego query) → 96D
      Branch 2: State MLP (ego + goal + time = 7D) → 32D
      Output:   concat → 128D (與 v2/v24 extractor 同 output_dim，head 層不變)
    """

    def __init__(self, d_model: int = 96, n_heads: int = 4):
        super().__init__()
        self.encoder = HeterogeneousAttentionEncoder(
            d_model=d_model, n_heads=n_heads
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.state_ln = nn.LayerNorm(32)
        self._output_dim = d_model + 32  # 96 + 32 = 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from 139D flat observation."""
        lidar = x[:, LIDAR_START:LIDAR_END]                        # [B, 72]
        obs_flat = x[:, OBS_START:OBS_END]                         # [B, 60]
        dynamic = obs_flat.view(-1, OBS_TOP_K, OBS_PER_OBJ)        # [B, 10, 6]

        ego = x[:, EGO_START:EGO_END]                              # [B, 4]
        goal = x[:, GOAL_START:GOAL_END]                           # [B, 2]
        time = x[:, TIME_START:TIME_END]                           # [B, 1]
        state_7d = torch.cat([ego, goal, time], dim=-1)            # [B, 7]

        attn_feat = self.encoder(lidar, dynamic, state_7d)         # [B, 96]
        state_feat = self.state_ln(self.state_mlp(state_7d))       # [B, 32]
        return torch.cat([attn_feat, state_feat], dim=-1)          # [B, 128]


class VLP16FeatureExtractor(nn.Module):
    """v2 三分支特徵提取器（支援 maxpool / attention obstacle encoder）。

    Branch 1: LiDAR Conv1d
        [B, 72] → reshape [B, 1, 72] → Conv1d → MaxPool → 64D

    Branch 2: Obstacle Encoder（依 obstacle_encoder 選擇）
        - "maxpool" (v2 default): [B, 10, 6] → MLP → MaxPool → 32D
        - "attention" (v24): [B, 10, 6] + ego → CrossAttention → 32D

    Branch 3: State MLP
        [B, 7] (ego 4 + goal 2 + time 1) → MLP → 32D

    Output: [B, 128]
    """

    def __init__(self, obstacle_encoder: str = "maxpool"):
        super().__init__()
        if obstacle_encoder not in ("maxpool", "attention"):
            raise ValueError(f"obstacle_encoder must be 'maxpool' or 'attention', got {obstacle_encoder}")
        self.obstacle_encoder_mode = obstacle_encoder

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

        # --- Branch 2: Obstacle Encoder (maxpool 或 attention) ---
        if obstacle_encoder == "attention":
            self.attention_encoder = AttentionObstacleEncoder(d_model=32, n_heads=4)
            self.obs_mlp = None
            self.obs_ln = None
        else:
            # 排列不變：每個 obstacle 獨立通過同一 MLP，再做 MaxPool
            self.obs_mlp = nn.Sequential(
                nn.Linear(OBS_PER_OBJ, 32),   # 6 → 32
                nn.ReLU(),
                nn.Linear(32, 32),            # 32 → 32
                nn.ReLU(),
            )
            self.obs_ln = nn.LayerNorm(32)
            self.attention_encoder = None

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

        # --- Branch 3 state 先組出來（attention 需要用） ---
        ego = x[:, EGO_START:EGO_END]                         # [B, 4]
        goal = x[:, GOAL_START:GOAL_END]                       # [B, 2]
        time = x[:, TIME_START:TIME_END]                       # [B, 1]
        state_7d = torch.cat([ego, goal, time], dim=-1)        # [B, 7]

        # --- Branch 2: Obstacle encoder ---
        obs_flat = x[:, OBS_START:OBS_END]                     # [B, 60]
        obs_reshaped = obs_flat.view(-1, OBS_TOP_K, OBS_PER_OBJ)  # [B, 10, 6]
        if self.obstacle_encoder_mode == "attention":
            obs_feat = self.attention_encoder(obs_reshaped, state_7d)  # [B, 32]
        else:
            obs_per_obj = self.obs_mlp(obs_reshaped)           # [B, 10, 32]
            obs_pooled = obs_per_obj.max(dim=1).values         # [B, 32]
            obs_feat = self.obs_ln(obs_pooled)                 # [B, 32]

        # --- Branch 3: State (ego + goal + time) ---
        state = self.state_mlp(state_7d)                       # [B, 32]
        state = self.state_ln(state)                           # [B, 32]

        return torch.cat([lidar, obs_feat, state], dim=-1)     # [B, 128]


# ============================================================================
# SKRL MultiDiscrete Policy (Actor) — MultiDiscrete([19, 19])
# ============================================================================

class VLP16DiscretePolicy(MultiCategoricalMixin, Model):
    """v3 MultiCategorical policy — two independent categorical heads.

    Input:  139D flat observation
    Output: 38-dim logits (19 accel + 19 omega, independent)

    v3 變更：Discrete(361) → MultiDiscrete([19, 19])
      - 解決維度詛咒：38 logits vs 361 logits
      - 兩組獨立 Categorical 分佈，各自採樣
      - 最大熵 = ln(19) + ln(19) ≈ 5.89 ≈ ln(361)（不變）

    Architecture:
      VLP16FeatureExtractor → 128D → Linear(128,128) → ReLU → Linear(128,38)
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        unnormalized_log_prob: bool = True,
        role: str = "",
        obstacle_encoder: str = "maxpool",
        **kwargs,
    ):
        multi_discrete_space = MultiDiscrete(np.array([NUM_BINS, NUM_BINS]))
        Model.__init__(self, observation_space, multi_discrete_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction="sum", role=role)

        self.extractor = VLP16FeatureExtractor(obstacle_encoder=obstacle_encoder)
        self.head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 128),   # 128 → 128
            nn.ReLU(),
            nn.Linear(128, TOTAL_LOGITS),                 # 128 → 38 (19+19)
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
        obstacle_encoder: str = "maxpool",
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions, role)

        self.extractor = VLP16FeatureExtractor(obstacle_encoder=obstacle_encoder)

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


class VLP16AttentionPolicy(VLP16DiscretePolicy):
    """v24: HEIGHT-style attention obstacle encoder policy."""

    def __init__(self, *args, **kwargs):
        kwargs["obstacle_encoder"] = "attention"
        super().__init__(*args, **kwargs)


class VLP16AttentionValue(VLP16Value):
    """v24: HEIGHT-style attention obstacle encoder value."""

    def __init__(self, *args, **kwargs):
        kwargs["obstacle_encoder"] = "attention"
        super().__init__(*args, **kwargs)


# ============================================================================
# v25: Heterogeneous Attention Policy / Value
# ============================================================================

class VLP16HeterogeneousPolicy(MultiCategoricalMixin, Model):
    """v25: Full HEIGHT heterogeneous attention policy.

    LiDAR static tokens (18) + dynamic obstacle tokens (10) unified cross-attention,
    修正 v24 的 static branch MaxPool-to-1 瓶頸（collision_static_ratio=0.61 主因）。
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        unnormalized_log_prob: bool = True,
        role: str = "",
        d_model: int = 96,
        n_heads: int = 4,
        **kwargs,
    ):
        multi_discrete_space = MultiDiscrete(np.array([NUM_BINS, NUM_BINS]))
        Model.__init__(self, observation_space, multi_discrete_space, device)
        MultiCategoricalMixin.__init__(
            self, unnormalized_log_prob, reduction="sum", role=role
        )

        self.extractor = VLP16HeterogeneousExtractor(d_model=d_model, n_heads=n_heads)
        self.head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, TOTAL_LOGITS),
        )

    def compute(self, inputs, role=""):
        states = inputs["states"]
        features = self.extractor(states)
        logits = self.head(features)
        if torch.isnan(logits).any():
            if not hasattr(self, "_nan_logit_warned"):
                self._nan_logit_warned = True
                print(
                    f"[NaN POLICY v25] Logits contain NaN! "
                    f"input_nan={torch.isnan(states).any().item()}, "
                    f"features_nan={torch.isnan(features).any().item()}"
                )
            logits = torch.nan_to_num(logits, nan=0.0)
        return logits, {}


class VLP16HeterogeneousValue(DeterministicMixin, Model):
    """v25: Deterministic value function with heterogeneous attention extractor."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Optional[torch.device] = None,
        clip_actions: bool = False,
        role: str = "",
        d_model: int = 96,
        n_heads: int = 4,
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions, role)

        self.extractor = VLP16HeterogeneousExtractor(d_model=d_model, n_heads=n_heads)
        self.value_head = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.constant_(self.value_head[-1].bias, -8.0)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        features = self.extractor(states)
        value = self.value_head(features)
        if torch.isnan(value).any():
            if not hasattr(self, "_nan_value_warned"):
                self._nan_value_warned = True
                print("[NaN CRITIC v25] Value contains NaN!")
        return value, {}


__all__ = [
    "VLP16FeatureExtractor",
    "AttentionObstacleEncoder",
    "HeterogeneousAttentionEncoder",
    "VLP16HeterogeneousExtractor",
    "VLP16DiscretePolicy",
    "VLP16Value",
    "VLP16AttentionPolicy",
    "VLP16AttentionValue",
    "VLP16HeterogeneousPolicy",
    "VLP16HeterogeneousValue",
    "POLICY_DIM",
    "CRITIC_DIM",
    "NUM_BINS",
    "TOTAL_LOGITS",
]
