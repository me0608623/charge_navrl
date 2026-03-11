"""
雙分支混合感知特徵萃取網路 (Hybrid Sensor Fusion Architecture)
=================================================================

架構概覽：

    ┌──────────────────┐   ┌──────────────────┐   ┌───────────────┐
    │  分支一: MOT 模擬  │   │ 分支二: LiDAR CNN │   │ 分支三: 本體感覺│
    │ [N, obj, 7]      │   │ [N, 1, 72]       │   │ [N, state_dim]│
    │                  │   │ (已前處理)         │   │               │
    │ Top-K (K=5)      │   │                  │   │  直接保留      │
    │ Sim2Real 雜訊    │   │ Conv1d ×3        │   │               │
    │ Flatten → MLP    │   │ Flatten → FC     │   │               │
    │     ↓ 64 維      │   │     ↓ 64 維      │   │    ↓ 6 維     │
    └────────┬─────────┘   └────────┬─────────┘   └───────┬───────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
                            ┌───────▼───────┐
                            │  Concat 134 維 │
                            │  Actor MLP     │
                            │   ↓ action_dim │
                            └───────────────┘

所有運算皆為 batch-vectorized，第一維度恆為 num_envs。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 分支一：動態障礙物 MOT 模擬 (Object-Level Branch)
# ============================================================================

class MOTBranch(nn.Module):
    """動態障礙物 Top-K 選取 + Sim-to-Real 雜訊 + MLP 萃取

    輸入: [num_envs, num_objects, 7]
        7 維特徵 = (Δx, Δy, 相對朝向角, 類型, 半徑, vx, vy)

    處理流程:
        1. 計算每個物件與自車的距離 (由 Δx, Δy 求歐幾里得距離)
        2. Top-K 選取最近的 K 個物件；不足 K 個時補零
        3. (訓練時) 注入高斯雜訊 + 5% 機率模擬追蹤掉幀
        4. Flatten → MLP → 輸出特徵向量

    輸出: [num_envs, out_dim]
    """

    def __init__(
        self,
        feat_dim: int = 7,
        top_k: int = 5,
        out_dim: int = 64,
        noise_std: float = 0.02,
        dropout_prob: float = 0.05,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

        # Flatten 後的維度 = top_k * feat_dim
        flat_dim = top_k * feat_dim

        # 兩層 MLP：flat_dim → 128 → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ELU(),
            nn.Linear(128, out_dim),
            nn.ELU(),
        )

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """
        Args:
            objects: [num_envs, num_objects, 7]
                     若場景無物件，允許 num_objects=0 的空 Tensor。

        Returns:
            [num_envs, out_dim] 障礙物特徵向量
        """
        num_envs = objects.shape[0]
        num_objects = objects.shape[1]
        device = objects.device
        K = self.top_k

        # --- 處理空場景（num_objects == 0）→ 全零輸入 ---
        if num_objects == 0:
            flat = torch.zeros(num_envs, K * self.feat_dim, device=device)
            return self.mlp(flat)

        # --- Step 1: 計算每個物件與自車的歐幾里得距離 ---
        # Δx = objects[:, :, 0], Δy = objects[:, :, 1]
        dx = objects[:, :, 0]  # [num_envs, num_objects]
        dy = objects[:, :, 1]  # [num_envs, num_objects]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # [num_envs, num_objects]

        # --- Step 2: Top-K 選取 ---
        if num_objects >= K:
            # 取距離最小的 K 個（largest=False）
            _, topk_idx = torch.topk(dist, k=K, dim=1, largest=False)
            # [num_envs, K]

            # Batch gather：根據索引收集完整 7 維特徵
            topk_idx_exp = topk_idx.unsqueeze(2).expand(-1, -1, self.feat_dim)
            # [num_envs, K, 7]
            topk_feats = torch.gather(objects, dim=1, index=topk_idx_exp)
            # [num_envs, K, 7]
        else:
            # 物件不足 K 個：先排序、再補零 (Padding)
            _, sort_idx = torch.sort(dist, dim=1)  # 由近到遠排序
            sort_idx_exp = sort_idx.unsqueeze(2).expand(-1, -1, self.feat_dim)
            sorted_feats = torch.gather(objects, dim=1, index=sort_idx_exp)
            # [num_envs, num_objects, 7]

            # 補零至 K 個
            pad_size = K - num_objects
            padding = torch.zeros(
                num_envs, pad_size, self.feat_dim, device=device
            )
            topk_feats = torch.cat([sorted_feats, padding], dim=1)
            # [num_envs, K, 7]

        # --- Step 3: Sim-to-Real 雜訊注入（僅訓練時） ---
        if self.training:
            # (a) 高斯雜訊：模擬感測器量測誤差
            noise = torch.randn_like(topk_feats) * self.noise_std
            topk_feats = topk_feats + noise

            # (b) 隨機掉幀：5% 機率將整個物件特徵歸零
            #     mask shape = [num_envs, K, 1]，廣播到 7 維
            drop_mask = (
                torch.rand(num_envs, K, 1, device=device) > self.dropout_prob
            ).float()
            topk_feats = topk_feats * drop_mask

        # --- Step 4: Flatten + MLP ---
        flat = topk_feats.reshape(num_envs, -1)  # [num_envs, K * 7 = 35]
        return self.mlp(flat)  # [num_envs, out_dim]


# ============================================================================
# 分支二：LiDAR 幾何特徵 1D CNN
# ============================================================================

class LiDARBranch(nn.Module):
    """LiDAR 1D CNN 特徵萃取

    輸入: [num_envs, 1, 72]
        環境已完成 Min Pooling / Clipping / Normalization 等前處理，
        本模組直接接收 72-beam 距離向量。

    處理流程:
        1. Conv1d ×3 層提取空間幾何特徵
        2. Flatten → 全連接層輸出 64 維

    網路結構:
        [N, 1, 72]
        → Conv1d(1→32,  k=5, s=2) → ELU → [N, 32, 34]
        → Conv1d(32→64, k=3, s=2) → ELU → [N, 64, 16]
        → Conv1d(64→64, k=3, s=2) → ELU → [N, 64, 7]
        → Flatten → [N, 448]
        → Linear(448→64) → ELU → [N, 64]

    輸出: [num_envs, out_dim]
    """

    def __init__(
        self,
        num_beams: int = 72,
        out_dim: int = 64,
    ):
        super().__init__()
        self.num_beams = num_beams

        # --- 1D CNN ---
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.ELU(),
        )

        # 動態計算 CNN 輸出維度（避免手動算錯）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_beams)
            cnn_out_size = self.cnn(dummy).numel()  # 1 × channels × length

        # 全連接層：CNN flatten → out_dim
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, out_dim),
            nn.ELU(),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lidar: [num_envs, 1, 72] 已前處理的 LiDAR 距離向量

        Returns:
            [num_envs, out_dim] LiDAR 特徵向量
        """
        N = lidar.shape[0]

        # 通過 CNN：[N, 1, 72] → [N, 64, 7]
        x = self.cnn(lidar)

        # Flatten：[N, 64, 7] → [N, 448]
        x = x.reshape(N, -1)

        # 全連接：[N, 448] → [N, out_dim]
        return self.fc(x)


# ============================================================================
# 完整雙分支混合感知網路
# ============================================================================

class HybridSensorFusionNet(nn.Module):
    """雙分支混合感知特徵萃取 + Actor MLP

    三種輸入：
        1. 動態障礙物: [num_envs, num_objects, 7] → MOT 分支 → 64 維
        2. LiDAR 特徵:  [num_envs, 1, 72]         → CNN 分支 → 64 維（已前處理）
        3. 本體感覺:    [num_envs, state_dim]      → 直接保留  → state_dim 維

    融合後：
        concat(64 + 64 + state_dim) → Actor MLP → action_dim
    """

    def __init__(
        self,
        # --- MOT 分支參數 ---
        mot_feat_dim: int = 7,
        mot_top_k: int = 5,
        mot_out_dim: int = 64,
        mot_noise_std: float = 0.02,
        mot_dropout_prob: float = 0.05,
        # --- LiDAR 分支參數 ---
        lidar_num_beams: int = 72,
        lidar_out_dim: int = 64,
        # --- 本體感覺參數 ---
        state_dim: int = 6,
        # --- Actor MLP 參數 ---
        action_dim: int = 2,
        actor_hidden: list[int] | None = None,
    ):
        super().__init__()

        # 儲存維度資訊
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fusion_dim = mot_out_dim + lidar_out_dim + state_dim

        # --- 分支一：MOT ---
        self.mot_branch = MOTBranch(
            feat_dim=mot_feat_dim,
            top_k=mot_top_k,
            out_dim=mot_out_dim,
            noise_std=mot_noise_std,
            dropout_prob=mot_dropout_prob,
        )

        # --- 分支二：LiDAR CNN ---
        self.lidar_branch = LiDARBranch(
            num_beams=lidar_num_beams,
            out_dim=lidar_out_dim,
        )

        # --- 分支三：本體感覺（直接保留，無需額外網路層）---

        # --- Actor MLP ---
        if actor_hidden is None:
            actor_hidden = [256, 128]

        layers = []
        in_dim = self.fusion_dim
        for h in actor_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # 動作輸出限制在 [-1, 1]

        self.actor_mlp = nn.Sequential(*layers)

    def forward(
        self,
        objects: torch.Tensor,  # [num_envs, num_objects, 7]
        lidar: torch.Tensor,    # [num_envs, 1, 72]
        state: torch.Tensor,    # [num_envs, state_dim]
    ) -> torch.Tensor:
        """
        Args:
            objects: 動態障礙物特徵 [num_envs, num_objects, 7]
            lidar:   已前處理 LiDAR  [num_envs, 1, 72]
            state:   本體感覺狀態    [num_envs, state_dim]

        Returns:
            actions: [num_envs, action_dim] 動作輸出（範圍 [-1, 1]）
        """
        # 分支一：MOT 障礙物特徵
        mot_feat = self.mot_branch(objects)    # [num_envs, 64]

        # 分支二：LiDAR CNN 特徵
        lidar_feat = self.lidar_branch(lidar)  # [num_envs, 64]

        # 分支三：本體感覺（直接使用）
        proprio_feat = state                    # [num_envs, state_dim]

        # --- 特徵融合（Concatenation）---
        fused = torch.cat([mot_feat, lidar_feat, proprio_feat], dim=1)
        # [num_envs, 64 + 64 + state_dim = 134]

        # --- Actor MLP → 動作輸出 ---
        return self.actor_mlp(fused)  # [num_envs, action_dim]


# ============================================================================
# 測試區塊
# ============================================================================

def _test_shape_flow():
    """驗證 forward pass 的 shape 流轉。

    設定 num_envs = 4096，確認：
    1. 各分支輸出維度正確
    2. 融合後維度正確
    3. 最終動作輸出維度正確
    4. 不會出現維度不匹配或 OOM
    5. 空場景（num_objects=0）也能正常運作
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"測試裝置: {device}")
    print("=" * 60)

    # --- 建構模型 ---
    model = HybridSensorFusionNet(
        mot_feat_dim=7,
        mot_top_k=5,
        mot_out_dim=64,
        lidar_num_beams=72,
        lidar_out_dim=64,
        state_dim=6,
        action_dim=2,
        actor_hidden=[256, 128],
    ).to(device)

    # 列印模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    print()

    # --- 測試 1: 正常場景 (num_objects=10) ---
    print("--- 測試 1: 正常場景 (num_envs=4096, num_objects=10) ---")
    N = 4096
    objects = torch.randn(N, 10, 7, device=device)
    lidar = torch.rand(N, 1, 72, device=device)  # 已前處理的 72-beam
    state = torch.randn(N, 6, device=device)

    model.train()  # 訓練模式（啟用 Sim2Real 雜訊）
    actions = model(objects, lidar, state)
    print(f"  障礙物輸入:     {list(objects.shape)}")
    print(f"  LiDAR 輸入:     {list(lidar.shape)}")
    print(f"  本體感覺輸入:   {list(state.shape)}")
    print(f"  動作輸出:       {list(actions.shape)}")
    print(f"  動作值域:       [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    assert actions.shape == (N, 2), f"期望 ({N}, 2), 實際 {actions.shape}"
    print("  ✓ 通過")
    print()

    # --- 測試 2: 物件不足 K 個 (num_objects=3 < K=5) ---
    print("--- 測試 2: 物件不足 (num_envs=4096, num_objects=3) ---")
    objects_few = torch.randn(N, 3, 7, device=device)
    actions2 = model(objects_few, lidar, state)
    print(f"  障礙物輸入:     {list(objects_few.shape)} (不足 K=5，自動補零)")
    print(f"  動作輸出:       {list(actions2.shape)}")
    assert actions2.shape == (N, 2)
    print("  ✓ 通過")
    print()

    # --- 測試 3: 空場景 (num_objects=0) ---
    print("--- 測試 3: 空場景 (num_envs=4096, num_objects=0) ---")
    objects_empty = torch.zeros(N, 0, 7, device=device)
    actions3 = model(objects_empty, lidar, state)
    print(f"  障礙物輸入:     {list(objects_empty.shape)} (空場景，全零特徵)")
    print(f"  動作輸出:       {list(actions3.shape)}")
    assert actions3.shape == (N, 2)
    print("  ✓ 通過")
    print()

    # --- 測試 4: 推理模式（Sim2Real 雜訊關閉）---
    print("--- 測試 4: 推理模式 (eval) ---")
    model.eval()
    with torch.no_grad():
        actions4 = model(objects, lidar, state)
    print(f"  動作輸出:       {list(actions4.shape)}")
    print(f"  動作值域:       [{actions4.min().item():.4f}, {actions4.max().item():.4f}]")
    assert actions4.shape == (N, 2)
    print("  ✓ 通過")
    print()

    # --- 測試 5: 各分支獨立驗證 ---
    print("--- 測試 5: 各分支獨立 shape 驗證 ---")
    model.eval()
    with torch.no_grad():
        mot_out = model.mot_branch(objects)
        lidar_out = model.lidar_branch(lidar)
    print(f"  MOT 分支輸出:   {list(mot_out.shape)} (期望 [4096, 64])")
    print(f"  LiDAR 分支輸出: {list(lidar_out.shape)} (期望 [4096, 64])")
    assert mot_out.shape == (N, 64)
    assert lidar_out.shape == (N, 64)
    print("  ✓ 通過")
    print()

    # --- 記憶體使用 ---
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU 峰值記憶體使用: {mem_mb:.1f} MB")
    print()
    print("=" * 60)
    print("所有測試通過！Shape 流轉正確，無 OOM 或維度不匹配。")


if __name__ == "__main__":
    _test_shape_flow()
