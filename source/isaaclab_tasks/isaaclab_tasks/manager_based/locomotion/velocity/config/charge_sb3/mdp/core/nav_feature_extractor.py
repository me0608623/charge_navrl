"""Navigation Feature Extractor — 三分支結構化觀測處理

針對 112 維扁平觀測向量的自訂特徵萃取器，將語義不同的觀測子段
分別送入專用分支處理，再融合為統一的 80 維狀態表示。

觀測向量佈局 (112 維):
    [0:72)   → LiDAR scan (72 dim)
    [72:82)  → Kinematics + Goal (10 dim)
    [82:112) → Top-K obstacles, 5×6 (30 dim)

三分支架構:
    Branch A: 1D CNN   — LiDAR 72D → 32D  (空間特徵萃取)
    Branch B: SharedMLP — Obstacles 5×6 → 32D  (排列不變性)
    Branch C: Linear   — Kinematics 10D → 16D  (本體狀態)

融合輸出: concat(32 + 32 + 16) = 80 維

設計依據:
    - Branch A 使用 1D CNN 是因為 LiDAR 射線具有局部空間相關性
      (相鄰射線指向相鄰方向)，卷積核能捕捉「角落」「走廊」等幾何模式。
    - Branch B 使用 Shared MLP + Max Pooling 保證排列不變性 (Permutation
      Invariance)。障礙物順序不應影響輸出——PointNet 啟發的設計。
    - Branch C 是低維本體狀態，一層線性映射即可。

SB3 整合:
    在 policy_kwargs 中指定:
        policy_kwargs = dict(
            features_extractor_class=NavFeatureExtractor,
            features_extractor_kwargs=dict(
                lidar_dim=72,
                kinematics_dim=10,
                obstacle_dim=30,
            ),
        )
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavFeatureExtractor(BaseFeaturesExtractor):
    """三分支導航特徵萃取器 (1D-CNN + SharedMLP + Linear)

    將 112 維扁平觀測拆分為三個語義子段，分別處理後融合為 80 維。

    Architecture:
        ┌──────────────────────────────────────────────────────────────┐
        │               Input: [batch, 112]                           │
        │                                                              │
        │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
        │  │ [0:72) LiDAR│  │[72:82) Kine │  │ [82:112) Obstacles   │ │
        │  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘ │
        │         │                │                     │             │
        │    Branch A         Branch C             Branch B           │
        │    1D CNN           Linear(10→16)        SharedMLP          │
        │    72 → 32          10 → 16              5×6 → 5×32 → 32   │
        │         │                │                     │             │
        │         └────────┬───────┴─────────────────────┘             │
        │                  │                                           │
        │            torch.cat(dim=1)                                  │
        │            [batch, 80]                                       │
        └──────────────────────────────────────────────────────────────┘

    Args:
        observation_space: Gymnasium observation space (Box with shape (112,))
        lidar_dim: LiDAR 射線數 (default: 72)
        kinematics_dim: 本體運動學維度 (default: 10)
        obstacle_dim: Top-K 障礙物總維度 (default: 30, i.e. 5×6)
        lidar_embed_dim: Branch A 輸出維度 (default: 32)
        obstacle_embed_dim: Branch B 輸出維度 (default: 32)
        kinematics_embed_dim: Branch C 輸出維度 (default: 16)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lidar_dim: int = 72,
        kinematics_dim: int = 10,
        obstacle_dim: int = 30,
        num_obstacles: int = 5,
        per_obstacle_dim: int = 6,
        lidar_embed_dim: int = 32,
        obstacle_embed_dim: int = 32,
        kinematics_embed_dim: int = 16,
    ):
        # 融合後的總輸出維度
        features_dim = lidar_embed_dim + obstacle_embed_dim + kinematics_embed_dim
        super().__init__(observation_space, features_dim=features_dim)

        # ─── 記錄切片索引 ───
        self.lidar_dim = lidar_dim
        self.kinematics_dim = kinematics_dim
        self.obstacle_dim = obstacle_dim
        self.num_obstacles = num_obstacles
        self.per_obstacle_dim = per_obstacle_dim

        # 驗證維度一致性
        total_expected = lidar_dim + kinematics_dim + obstacle_dim
        obs_dim = observation_space.shape[0]
        assert obs_dim == total_expected, (
            f"Observation dim mismatch: space has {obs_dim}, "
            f"but lidar({lidar_dim}) + kinematics({kinematics_dim}) "
            f"+ obstacles({obstacle_dim}) = {total_expected}"
        )
        assert obstacle_dim == num_obstacles * per_obstacle_dim, (
            f"Obstacle dim mismatch: {obstacle_dim} != "
            f"{num_obstacles} × {per_obstacle_dim}"
        )

        # ═══════════════════════════════════════════════════════════════
        # Branch A: 靜態 LiDAR (1D CNN)
        # ═══════════════════════════════════════════════════════════════
        # Input:  [batch, 1, 72]
        # Conv1:  kernel=5, stride=2 → floor((72 - 5) / 2) + 1 = 34
        #         Output: [batch, 16, 34]
        # Conv2:  kernel=3, stride=2 → floor((34 - 3) / 2) + 1 = 16
        #         Output: [batch, 32, 16]
        # Pool:   AdaptiveAvgPool1d(4) → [batch, 32, 4]
        # Flatten: [batch, 128]
        # Linear:  128 → 32
        # Output: [batch, 32]
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=4),
            nn.Flatten(),  # [batch, 32 * 4] = [batch, 128]
            nn.Linear(32 * 4, lidar_embed_dim),
            nn.ReLU(),
        )

        # ═══════════════════════════════════════════════════════════════
        # Branch B: 動態障礙物 (Shared MLP + Max Pooling)
        # ═══════════════════════════════════════════════════════════════
        # Input:    [batch, 5, 6]
        # Linear1:  6 → 32  → [batch, 5, 32]   (per-obstacle, shared weights)
        # ReLU
        # Linear2:  32 → 32 → [batch, 5, 32]   (per-obstacle, shared weights)
        # ReLU
        # Max pool: dim=1   → [batch, 32]       (permutation invariant)
        #
        # 關鍵設計: nn.Linear 作用在最後一維，自然對 5 個障礙物共享權重。
        # 不使用 Flatten，保留 [batch, 5, 32] 結構直到 max pooling。
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(per_obstacle_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obstacle_embed_dim),
            nn.ReLU(),
        )

        # ═══════════════════════════════════════════════════════════════
        # Branch C: 本體運動學 (Linear)
        # ═══════════════════════════════════════════════════════════════
        # Input:  [batch, 10]
        # Linear: 10 → 16
        # Output: [batch, 16]
        self.kinematics_fc = nn.Sequential(
            nn.Linear(kinematics_dim, kinematics_embed_dim),
            nn.ReLU(),
        )

        # ─── 初始化權重 ───
        self._init_weights()

    def _init_weights(self) -> None:
        """正交初始化 (Orthogonal Init)，與 SB3 預設風格一致。"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """前向傳播：拆分 → 三分支處理 → 融合

        Args:
            observations: [batch, 112] 扁平觀測向量

        Returns:
            [batch, 80] 融合後的狀態表示

        Shape trace:
            observations           : [B, 112]
            ├── lidar_raw          : [B, 72]
            │   └── reshape        : [B, 1, 72]   (add channel dim)
            │       └── lidar_cnn  : [B, 32]      (Branch A output)
            ├── kinematics_raw     : [B, 10]
            │   └── kinematics_fc  : [B, 16]      (Branch C output)
            └── obstacles_raw      : [B, 30]
                └── reshape        : [B, 5, 6]    (num_obstacles, per_dim)
                    └── shared_mlp : [B, 5, 32]   (per-obstacle features)
                        └── max    : [B, 32]      (Branch B output)
            ──────────────────────────────────
            cat([32, 32, 16], dim=1) → [B, 80]
        """
        # ─── Step 1: 切片 (Slice) ───
        lidar_raw = observations[:, :self.lidar_dim]
        # [B, 72]

        kin_start = self.lidar_dim
        kin_end = kin_start + self.kinematics_dim
        kinematics_raw = observations[:, kin_start:kin_end]
        # [B, 10]

        obstacles_raw = observations[:, kin_end:]
        # [B, 30]

        # ─── Step 2: Branch A — LiDAR 1D CNN ───
        lidar_input = lidar_raw.unsqueeze(1)
        # [B, 72] → [B, 1, 72]  (add channel dimension for Conv1d)

        lidar_features = self.lidar_cnn(lidar_input)
        # [B, 1, 72] → Conv1d(1→16, k=5, s=2) → [B, 16, 34]
        #            → Conv1d(16→32, k=3, s=2) → [B, 32, 16]
        #            → AdaptiveAvgPool1d(4)     → [B, 32, 4]
        #            → Flatten                  → [B, 128]
        #            → Linear(128, 32) + ReLU   → [B, 32]

        # ─── Step 3: Branch B — Shared MLP + Max Pooling ───
        obstacles_input = obstacles_raw.view(-1, self.num_obstacles, self.per_obstacle_dim)
        # [B, 30] → [B, 5, 6]

        obstacle_per_features = self.obstacle_mlp(obstacles_input)
        # [B, 5, 6] → Linear(6, 32) + ReLU → [B, 5, 32]
        #           → Linear(32, 32) + ReLU → [B, 5, 32]
        # nn.Linear broadcasts over dim=1, so weights are shared across 5 obstacles.
        # Shape [B, 5, 32] is preserved — NO flatten before max pooling.

        obstacle_features, _ = torch.max(obstacle_per_features, dim=1)
        # [B, 5, 32] → max over dim=1 → [B, 32]
        # Permutation Invariant: 無論 5 個障礙物的排列順序如何，
        # max pooling 的結果相同。

        # ─── Step 4: Branch C — Kinematics Linear ───
        kinematics_features = self.kinematics_fc(kinematics_raw)
        # [B, 10] → Linear(10, 16) + ReLU → [B, 16]

        # ─── Step 5: Fusion — Concatenate ───
        fused = torch.cat([lidar_features, obstacle_features, kinematics_features], dim=1)
        # cat([B, 32], [B, 32], [B, 16], dim=1) → [B, 80]

        return fused
