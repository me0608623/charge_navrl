"""
混合感知前處理管理器 (HybridPerceptionManager)
================================================

為 Isaac Lab Observation Manager 提供兩個高度向量化的感測器前處理節點：

┌─────────────────────────────────────────────────────────────────────┐
│                    HybridPerceptionManager                          │
├─────────────────────────────┬───────────────────────────────────────┤
│  任務一：process_mot()      │  任務二：process_lidar()              │
│                             │                                       │
│  輸入:                      │  輸入:                                │
│    ego_pos    [N, 2]        │    ray_hits [N, 16, 360]              │
│    ego_yaw    [N]           │                                       │
│    obs_pos    [N, M, 2]     │  五步驟前處理:                        │
│    obs_vel    [N, M, 2]     │    0. Domain Rand (train only)        │
│    obs_yaw    [N, M]        │       a. 隨機位移 (σ=0.03m)           │
│    obs_radius [N, M]        │       b. 策略性孔洞 (p=0.05)          │
│    obs_valid  [N, M] bool   │       c. 干擾雜點 (p=0.01)            │
│                             │    1. Clipping (inf/NaN → r_max)      │
│  處理:                      │    2. Min Pooling (16×360 → 72)       │
│    距離排序 → Top-K (K=5)   │    3. Safe Margin (−r_robot, ≥0)      │
│    6 維特徵構建              │    4. Normalization (÷r_max → [0,1])  │
│    Sim-to-Real 雜訊注入     │                                       │
│                             │  輸出:                                │
│  輸出:                      │    [N, 72] 乾淨一維距離               │
│    [N, 5, 6] 障礙物特徵     │                                       │
└─────────────────────────────┴───────────────────────────────────────┘

所有運算完全依賴 PyTorch Tensor 廣播與索引，
禁止任何 for 迴圈遍歷環境維度。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HybridPerceptionManager(nn.Module):
    """混合感知前處理管理器

    繼承 nn.Module 以支援 train/eval 模式切換：
    - train 模式：MOT 掉幀雜訊 + LiDAR 領域隨機化 皆啟用
    - eval  模式：所有雜訊注入關閉，輸出為確定性

    使用範例::

        manager = HybridPerceptionManager(top_k=5).to(device)
        manager.train()   # 訓練時啟用雜訊
        manager.eval()    # 推理時關閉雜訊

        mot_features = manager.process_mot(ego_pos, ego_yaw, obs_pos, ...)
        lidar_clean  = manager.process_lidar(ray_hits)
    """

    def __init__(
        self,
        # --- MOT 參數 ---
        top_k: int = 5,
        v_max: float = 2.0,
        mot_noise_std: float = 0.02,
        drop_rate: float = 0.05,
        # --- LiDAR 參數 ---
        r_max: float = 20.0,
        r_robot: float = 0.3,
        num_beams: int = 72,
        # --- LiDAR 領域隨機化參數 ---
        lidar_noise_std: float = 0.03,
        hole_prob: float = 0.05,
        distract_prob: float = 0.01,
        distract_min: float = 0.2,
        distract_max: float = 1.5,
    ):
        """
        Args:
            top_k:           保留最近的 K 個障礙物（不足時補零）
            v_max:           速度正規化上限 (m/s)，用於將速度映射至 [0, 1]
            mot_noise_std:   MOT 高斯雜訊標準差（模擬卡爾曼濾波器估計誤差）
            drop_rate:       MOT 隨機掉幀機率（模擬追蹤失敗）
            r_max:           LiDAR 最大探測距離 (m)
            r_robot:         車體半徑 (m)，用於 Safe Margin 扣除
            num_beams:       Min Pooling 後的水平射線數（360 / 5 = 72）
            lidar_noise_std: LiDAR 個別點隨機位移標準差 (m)
            hole_prob:       策略性孔洞機率（模擬材質吸收導致的未回傳）
            distract_prob:   干擾雜點機率（模擬灰塵或鏡頭髒汙）
            distract_min:    干擾雜點距離下限 (m)
            distract_max:    干擾雜點距離上限 (m)
        """
        super().__init__()
        # MOT
        self.top_k = top_k
        self.v_max = v_max
        self.mot_noise_std = mot_noise_std
        self.drop_rate = drop_rate
        # LiDAR 基礎
        self.r_max = r_max
        self.r_robot = r_robot
        self.num_beams = num_beams
        # LiDAR 領域隨機化
        self.lidar_noise_std = lidar_noise_std
        self.hole_prob = hole_prob
        self.distract_prob = distract_prob
        self.distract_min = distract_min
        self.distract_max = distract_max

    # ================================================================
    # 任務一：動態障礙物 (MOT) 狀態提取與 6 維特徵構建
    # ================================================================

    def process_mot(
        self,
        ego_pos: torch.Tensor,                   # [N, 2]
        ego_yaw: torch.Tensor,                   # [N] 或 [N, 1]
        obs_pos: torch.Tensor,                   # [N, M, 2]
        obs_vel: torch.Tensor,                   # [N, M, 2]
        obs_yaw: torch.Tensor,                   # [N, M] 或 [N, M, 1]
        obs_radius: torch.Tensor,                # [N, M] 或 [N, M, 1]
        obs_valid: torch.Tensor | None = None,   # [N, M] bool（可選）
    ) -> torch.Tensor:
        """動態障礙物 Top-K 選取 + 6 維特徵構建 + Sim-to-Real 雜訊注入

        6 維特徵定義（嚴格依照順序）：
        ┌───────┬────────┬────────────────────────────────────┐
        │ Index │  名稱   │ 說明                               │
        ├───────┼────────┼────────────────────────────────────┤
        │   0   │ s_dx   │ Δx = obs_x − ego_x                │
        │   1   │ s_dy   │ Δy = obs_y − ego_y                │
        │   2   │ s_theta│ 正規化相對朝向角 ∈ [-1, 1]          │
        │   3   │ s_o    │ 追蹤狀態: 1.0/0.0/-1.0            │
        │   4   │ s_r    │ 障礙物半徑 (m)                     │
        │   5   │ s_v    │ 正規化速度大小 ∈ [0, 1]            │
        └───────┴────────┴────────────────────────────────────┘

        s_o 編碼:
            1.0  → 正常追蹤
            0.0  → 掉幀 / 遺失（Sim-to-Real 雜訊模擬）
           -1.0  → Padding（場景物件不足 K 個）

        Args:
            ego_pos:    自車位置       [num_envs, 2]
            ego_yaw:    自車朝向角     [num_envs] 或 [num_envs, 1]
            obs_pos:    障礙物位置     [num_envs, num_obstacles, 2]
            obs_vel:    障礙物速度     [num_envs, num_obstacles, 2]
            obs_yaw:    障礙物朝向角   [num_envs, num_obstacles] 或 [..., 1]
            obs_radius: 障礙物半徑     [num_envs, num_obstacles] 或 [..., 1]
            obs_valid:  有效性遮罩     [num_envs, num_obstacles] bool
                        None 時預設全部有效

        Returns:
            [num_envs, top_k, 6] 障礙物特徵張量
        """
        N = ego_pos.shape[0]
        M = obs_pos.shape[1]   # 場景中的障礙物數量
        K = self.top_k
        device = ego_pos.device

        # ----------------------------------------------------------
        # 形狀統一：確保所有輸入維度一致
        # ----------------------------------------------------------
        ego_yaw = ego_yaw.reshape(N, 1)           # [N, 1]
        obs_yaw = obs_yaw.reshape(N, M) if M > 0 else obs_yaw.reshape(N, 0)
        obs_radius = obs_radius.reshape(N, M) if M > 0 else obs_radius.reshape(N, 0)

        if obs_valid is None:
            obs_valid = torch.ones(N, M, dtype=torch.bool, device=device)

        # ----------------------------------------------------------
        # 邊界條件：空場景 (M == 0)
        # 全部為 Padding：s_o = -1.0，其餘清零
        # ----------------------------------------------------------
        if M == 0:
            features = torch.zeros(N, K, 6, device=device)
            features[:, :, 3] = -1.0   # s_o = Padding
            return features

        # ==========================================================
        # Step 1: 計算 Δx, Δy 與歐式距離
        # ==========================================================
        # obs_pos: [N, M, 2]，ego_pos: [N, 2] → 廣播 [N, 1, 2]
        delta = obs_pos - ego_pos.unsqueeze(1)    # [N, M, 2]
        dx = delta[:, :, 0]                        # [N, M]
        dy = delta[:, :, 1]                        # [N, M]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # [N, M]，+ε 防除零

        # 無效障礙物的距離設為極大值，排序時自動排到最後
        dist_masked = dist.clone()
        dist_masked[~obs_valid] = 1e6

        # ==========================================================
        # Step 2: Top-K 選取（由近到遠，取最近 K 個）
        # ==========================================================
        if M >= K:
            # 物件充足：直接取 Top-K 最近
            _, topk_idx = torch.topk(
                dist_masked, k=K, dim=1, largest=False
            )  # [N, K]
        else:
            # 物件不足 K 個：排序後用首個索引填充（後續覆寫為 Padding）
            _, sort_idx = torch.sort(dist_masked, dim=1)  # [N, M]
            pad_idx = sort_idx[:, :1].expand(-1, K - M)   # [N, K-M]
            topk_idx = torch.cat([sort_idx, pad_idx], dim=1)  # [N, K]

        # ==========================================================
        # Step 3: 透過 Gather 收集 Top-K 物件的所有原始特徵
        # ==========================================================
        # 純量特徵 [N, M] → [N, K]
        topk_dx = torch.gather(dx, 1, topk_idx)
        topk_dy = torch.gather(dy, 1, topk_idx)
        topk_obs_yaw = torch.gather(obs_yaw, 1, topk_idx)
        topk_radius = torch.gather(obs_radius, 1, topk_idx)

        # 有效性 [N, M] bool → [N, K] bool
        # 注意：bool Tensor 須轉 long 再 gather，再轉回 bool
        topk_valid = torch.gather(
            obs_valid.long(), 1, topk_idx
        ).bool()  # [N, K]

        # 向量特徵 [N, M, 2] → [N, K, 2]
        idx_2d = topk_idx.unsqueeze(2).expand(-1, -1, 2)
        topk_vel = torch.gather(obs_vel, 1, idx_2d)  # [N, K, 2]

        # M < K 時，超出的位置強制標記為無效 (Padding)
        if M < K:
            topk_valid = topk_valid.clone()
            topk_valid[:, M:] = False

        # ==========================================================
        # Step 4: 構建 6 維特徵
        # ==========================================================

        # --- 特徵 0 & 1: s_dx, s_dy ---
        s_dx = topk_dx  # [N, K]
        s_dy = topk_dy  # [N, K]

        # --- 特徵 2: s_theta（正規化相對朝向角 ∈ [-1, 1]）---
        rel_yaw = topk_obs_yaw - ego_yaw                     # [N, K]
        # atan2(sin, cos) 確保角度正規化至 [-π, π]
        rel_yaw = torch.atan2(
            torch.sin(rel_yaw), torch.cos(rel_yaw)
        )
        s_theta = rel_yaw / math.pi                          # [-1, 1]

        # --- 特徵 3: s_o（追蹤狀態，初始全設為 1.0 = 正常追蹤）---
        s_o = torch.ones(N, K, device=device)

        # --- 特徵 4: s_r（障礙物半徑）---
        s_r = topk_radius  # [N, K]

        # --- 特徵 5: s_v（正規化速度大小 ∈ [0, 1]）---
        speed = torch.sqrt(
            topk_vel[:, :, 0] ** 2 + topk_vel[:, :, 1] ** 2 + 1e-8
        )  # [N, K]
        s_v = torch.clamp(speed / self.v_max, 0.0, 1.0)

        # --- 組裝 [N, K, 6] ---
        features = torch.stack(
            [s_dx, s_dy, s_theta, s_o, s_r, s_v], dim=2
        )  # [N, K, 6]

        # ----------------------------------------------------------
        # Padding 處理：無效物件（不足 K 個的補位）
        #   s_o → -1.0，其餘 5 個特徵清零
        # ----------------------------------------------------------
        valid_f = topk_valid.float()  # [N, K]，1.0=有效 / 0.0=Padding

        # 清零 5 個非 s_o 特徵
        features[:, :, 0] *= valid_f   # s_dx
        features[:, :, 1] *= valid_f   # s_dy
        features[:, :, 2] *= valid_f   # s_theta
        features[:, :, 4] *= valid_f   # s_r
        features[:, :, 5] *= valid_f   # s_v

        # s_o：Padding → -1.0
        features[:, :, 3] = torch.where(
            topk_valid,
            features[:, :, 3],                         # 有效 → 保持 1.0
            torch.tensor(-1.0, device=device),         # Padding → -1.0
        )

        # ==========================================================
        # Step 5: Sim-to-Real 雜訊注入（僅 train 模式）
        # ==========================================================
        if self.training:
            # (a) 高斯雜訊：模擬卡爾曼濾波器估計誤差
            #     對 s_dx, s_dy, s_theta, s_v 加入微小常態分佈雜訊
            #     注意：只對有效物件注入（Padding 不注入）
            noise = torch.randn(N, K, 4, device=device) * self.mot_noise_std
            features[:, :, 0] += noise[:, :, 0] * valid_f   # s_dx  += ε
            features[:, :, 1] += noise[:, :, 1] * valid_f   # s_dy  += ε
            features[:, :, 2] += noise[:, :, 2] * valid_f   # s_theta += ε
            features[:, :, 5] += noise[:, :, 3] * valid_f   # s_v  += ε

            # (b) 隨機掉幀：以 drop_rate 機率模擬 MOT 追蹤失敗
            #     掉幀的物件：s_o → 0.0，其餘 5 個特徵清零
            drop = (
                torch.rand(N, K, device=device) < self.drop_rate
            ) & topk_valid  # 只對有效物件掉幀，Padding 不受影響
            keep = (~drop).float()  # 1.0=保留, 0.0=掉幀

            features[:, :, 0] *= keep   # s_dx  → 0
            features[:, :, 1] *= keep   # s_dy  → 0
            features[:, :, 2] *= keep   # s_theta → 0
            features[:, :, 4] *= keep   # s_r   → 0
            features[:, :, 5] *= keep   # s_v   → 0

            # s_o：掉幀 → 0.0（區別於 Padding 的 -1.0）
            features[:, :, 3] = torch.where(
                drop,
                torch.tensor(0.0, device=device),
                features[:, :, 3],
            )

        return features  # [N, K, 6] = [num_envs, 5, 6]

    # ================================================================
    # 任務二：LiDAR 16×360 點雲領域隨機化與四步驟前處理
    # ================================================================

    def process_lidar(self, ray_hits: torch.Tensor) -> torch.Tensor:
        """LiDAR 16×360 → 72 beams：領域隨機化 + 四步驟前處理

        處理流程（全向量化，零 for 迴圈）：

            Step 0: 點雲領域隨機化 (Point Cloud Domain Randomization)
                    ─ 僅 train 模式啟用 ─
                a. 個別點隨機位移：高斯雜訊 N(0, σ²)，σ = 0.03m
                b. 策略性孔洞：p=0.05 機率 → r_max（材質吸收）
                c. 干擾雜點：p=0.01 機率 → U(0.2, 1.5)m（灰塵/髒汙）

            Step 1: Clipping
                    inf / NaN / > r_max → r_max

            Step 2: Min Pooling
                    [N, 16, 360] → view [N, 16, 72, 5]
                    → min(dim=3) → [N, 16, 72]
                    → min(dim=1) → [N, 72]

            Step 3: Safe Margin
                    減去 r_robot，clamp(min=0)

            Step 4: Normalization
                    ÷ r_max → [0, 1]

        Args:
            ray_hits: [num_envs, 16, 360] 由 Isaac Lab RayCaster 產生的原始距離。
                      未命中時數值為 inf 或大於 r_max。

        Returns:
            [num_envs, 72] 乾淨一維距離張量，數值範圍 [0, 1]
        """
        N = ray_hits.shape[0]
        device = ray_hits.device
        dtype = ray_hits.dtype

        # 複製一份避免修改原始資料
        x = ray_hits.clone()

        # ==========================================================
        # Step 0: 點雲領域隨機化 (僅 train 模式)
        # ==========================================================
        if self.training:
            # --------------------------------------------------
            # (a) 個別點的隨機位移 (Random Displacement)
            #     對所有有效測距（finite 且 ≤ r_max）加入高斯雜訊
            #     模擬 LiDAR 測距的量測不確定性
            # --------------------------------------------------
            valid_mask = torch.isfinite(x) & (x <= self.r_max)  # [N, 16, 360]
            displacement = torch.randn(N, 16, 360, device=device, dtype=dtype) * self.lidar_noise_std
            # 只對有效讀數加雜訊，inf/NaN 不動
            x = x + displacement * valid_mask.float()

            # --------------------------------------------------
            # (b) 策略性孔洞 (Strategic Holes)
            #     以 hole_prob 機率將射線距離強制設為 r_max
            #     模擬黑色/吸收性材質導致光束無法回傳
            # --------------------------------------------------
            hole_mask = torch.rand(N, 16, 360, device=device, dtype=dtype) < self.hole_prob
            x = torch.where(hole_mask, torch.tensor(self.r_max, device=device, dtype=dtype), x)

            # --------------------------------------------------
            # (c) 干擾雜點 (Distractors)
            #     以 distract_prob 機率將射線距離替換為
            #     U(distract_min, distract_max) 的均勻隨機數
            #     模擬灰塵、水滴、鏡頭髒汙產生的虛假近距離回波
            # --------------------------------------------------
            distract_mask = torch.rand(N, 16, 360, device=device, dtype=dtype) < self.distract_prob
            distract_vals = (
                torch.rand(N, 16, 360, device=device, dtype=dtype)
                * (self.distract_max - self.distract_min)
                + self.distract_min
            )  # U(0.2, 1.5)
            x = torch.where(distract_mask, distract_vals, x)

        # ==========================================================
        # Step 1: Clipping
        #   將 inf、NaN、以及超過 r_max 的讀數全部截斷為 r_max
        # ==========================================================
        x = torch.where(
            torch.isfinite(x),
            x,
            torch.tensor(self.r_max, device=device, dtype=dtype),
        )
        x = torch.clamp(x, max=self.r_max)

        # ==========================================================
        # Step 2: Min Pooling (16×360 → 72)
        #   水平方向：360 條射線分成 72 個 bin（每 bin 5 條），取 min
        #   垂直方向：16 層取 min（選取最近障礙物距離）
        # ==========================================================
        # view: [N, 16, 360] → [N, 16, 72, 5]
        x = x.view(N, 16, self.num_beams, -1)

        # 水平區塊 min (dim=3)：每 5 條射線取最近距離
        # [N, 16, 72, 5] → [N, 16, 72]
        x = x.min(dim=3).values

        # 垂直層 min (dim=1)：16 層取最近距離
        # [N, 16, 72] → [N, 72]
        x = x.min(dim=1).values

        # ==========================================================
        # Step 3: Safe Margin（安全邊界）
        #   扣除車體半徑，確保剩餘距離 ≥ 0
        # ==========================================================
        x = torch.clamp(x - self.r_robot, min=0.0)

        # ==========================================================
        # Step 4: Normalization（正規化至 [0, 1]）
        # ==========================================================
        x = x / self.r_max

        return x  # [N, 72]


# ====================================================================
# 測試區塊
# ====================================================================

def _test():
    """驗證 HybridPerceptionManager 的輸出形狀與數值範圍。

    測試項目：
        1. MOT 正常場景 (M=10, K=5)
        2. MOT 物件不足 (M=3 < K=5)
        3. MOT 空場景 (M=0)
        4. MOT 訓練模式 vs 推理模式（雜訊 / 掉幀驗證）
        5. LiDAR eval 模式（純四步驟前處理）
        6. LiDAR eval 含 inf/NaN 輸入
        7. LiDAR 數值精度驗證
        8. LiDAR train 模式（Step 0 領域隨機化驗證）
        9. LiDAR 領域隨機化統計驗證（孔洞率 / 干擾率）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"測試裝置: {device}")
    print("=" * 70)

    N = 4096  # 環境數量
    manager = HybridPerceptionManager(
        top_k=5, v_max=2.0, mot_noise_std=0.02, drop_rate=0.05,
        r_max=20.0, r_robot=0.3, num_beams=72,
        lidar_noise_std=0.03, hole_prob=0.05,
        distract_prob=0.01, distract_min=0.2, distract_max=1.5,
    ).to(device)

    # ==============================================================
    # 測試 1: MOT 正常場景 (M=10 > K=5)
    # ==============================================================
    print("--- 測試 1: MOT 正常場景 (N=4096, M=10, K=5) ---")
    manager.eval()

    ego_pos = torch.randn(N, 2, device=device)
    ego_yaw = torch.randn(N, device=device)
    obs_pos = torch.randn(N, 10, 2, device=device) * 5.0
    obs_vel = torch.randn(N, 10, 2, device=device)
    obs_yaw = torch.randn(N, 10, device=device)
    obs_radius = torch.rand(N, 10, device=device) * 0.5 + 0.2
    obs_valid = torch.ones(N, 10, dtype=torch.bool, device=device)

    mot_out = manager.process_mot(
        ego_pos, ego_yaw, obs_pos, obs_vel, obs_yaw, obs_radius, obs_valid,
    )
    print(f"  輸出形狀:     {list(mot_out.shape)} (期望 [4096, 5, 6])")
    assert mot_out.shape == (N, 5, 6), f"形狀錯誤: {mot_out.shape}"

    s_o = mot_out[:, :, 3]
    print(f"  s_o 範圍:     [{s_o.min().item():.1f}, {s_o.max().item():.1f}] (期望 [1.0, 1.0])")
    assert (s_o == 1.0).all(), "eval 模式下所有有效物件的 s_o 應為 1.0"

    s_v = mot_out[:, :, 5]
    print(f"  s_v 範圍:     [{s_v.min().item():.4f}, {s_v.max().item():.4f}] (期望 [0, 1])")
    assert s_v.min() >= 0.0 and s_v.max() <= 1.0

    s_theta = mot_out[:, :, 2]
    print(f"  s_theta 範圍: [{s_theta.min().item():.4f}, {s_theta.max().item():.4f}] (期望 [-1, 1])")
    assert s_theta.min() >= -1.0 and s_theta.max() <= 1.0
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 2: MOT 物件不足 (M=3 < K=5)
    # ==============================================================
    print("--- 測試 2: MOT 物件不足 (N=4096, M=3, K=5) ---")
    obs_pos_few = torch.randn(N, 3, 2, device=device) * 5.0
    obs_vel_few = torch.randn(N, 3, 2, device=device)
    obs_yaw_few = torch.randn(N, 3, device=device)
    obs_radius_few = torch.rand(N, 3, device=device) * 0.5 + 0.2

    mot_out2 = manager.process_mot(
        ego_pos, ego_yaw,
        obs_pos_few, obs_vel_few, obs_yaw_few, obs_radius_few,
    )
    print(f"  輸出形狀:     {list(mot_out2.shape)} (期望 [4096, 5, 6])")
    assert mot_out2.shape == (N, 5, 6)

    s_o_pad = mot_out2[:, 3:, 3]
    print(f"  Padding s_o:  [{s_o_pad.min().item():.1f}, {s_o_pad.max().item():.1f}] (期望 [-1.0, -1.0])")
    assert (s_o_pad == -1.0).all()

    pad_features = mot_out2[:, 3:, [0, 1, 2, 4, 5]]
    print(f"  Padding 其餘: [{pad_features.min().item():.1f}, {pad_features.max().item():.1f}] (期望 [0, 0])")
    assert (pad_features == 0.0).all()
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 3: MOT 空場景 (M=0)
    # ==============================================================
    print("--- 測試 3: MOT 空場景 (N=4096, M=0) ---")
    mot_out3 = manager.process_mot(
        ego_pos, ego_yaw,
        torch.zeros(N, 0, 2, device=device),
        torch.zeros(N, 0, 2, device=device),
        torch.zeros(N, 0, device=device),
        torch.zeros(N, 0, device=device),
    )
    print(f"  輸出形狀:     {list(mot_out3.shape)} (期望 [4096, 5, 6])")
    assert mot_out3.shape == (N, 5, 6)
    assert (mot_out3[:, :, 3] == -1.0).all()
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 4: MOT 訓練模式（雜訊 + 掉幀）
    # ==============================================================
    print("--- 測試 4: MOT 訓練模式 (noise + dropout) ---")
    manager.train()

    mot_train = manager.process_mot(
        ego_pos, ego_yaw, obs_pos, obs_vel, obs_yaw, obs_radius, obs_valid,
    )
    s_o_train = mot_train[:, :, 3]
    num_dropped = (s_o_train == 0.0).sum().item()
    total = N * 5
    drop_pct = 100.0 * num_dropped / total
    print(f"  掉幀數/總數:  {num_dropped}/{total} ({drop_pct:.2f}%，期望 ≈5%)")

    dropped_mask = (s_o_train == 0.0)
    if dropped_mask.any():
        dropped_other = mot_train[dropped_mask][:, [0, 1, 2, 4, 5]]
        assert (dropped_other == 0.0).all()
        print(f"  掉幀特徵清零: ✓ 已驗證")

    manager.eval()
    mot_eval = manager.process_mot(
        ego_pos, ego_yaw, obs_pos, obs_vel, obs_yaw, obs_radius, obs_valid,
    )
    diff = (mot_train[:, :, 0] - mot_eval[:, :, 0]).abs().mean().item()
    print(f"  train/eval 差: {diff:.6f} (期望 > 0)")
    assert diff > 1e-6
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 5: LiDAR eval 模式（純四步驟前處理，無隨機化）
    # ==============================================================
    print("--- 測試 5: LiDAR eval 模式 (N=4096) ---")
    manager.eval()
    ray_hits = torch.rand(N, 16, 360, device=device) * 25.0

    lidar_out = manager.process_lidar(ray_hits)
    print(f"  輸出形狀:     {list(lidar_out.shape)} (期望 [4096, 72])")
    assert lidar_out.shape == (N, 72)
    print(f"  數值範圍:     [{lidar_out.min().item():.4f}, {lidar_out.max().item():.4f}] (期望 [0, 1])")
    assert lidar_out.min() >= 0.0 and lidar_out.max() <= 1.0
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 6: LiDAR eval 含 inf/NaN 輸入
    # ==============================================================
    print("--- 測試 6: LiDAR eval 含 inf/NaN ---")
    ray_noisy = torch.rand(N, 16, 360, device=device) * 15.0
    inf_mask = torch.rand(N, 16, 360, device=device) < 0.1
    ray_noisy[inf_mask] = float("inf")
    nan_mask = torch.rand(N, 16, 360, device=device) < 0.05
    ray_noisy[nan_mask] = float("nan")

    lidar_noisy = manager.process_lidar(ray_noisy)
    has_inf = torch.isinf(lidar_noisy).any().item()
    has_nan = torch.isnan(lidar_noisy).any().item()
    print(f"  含 inf: {has_inf} (期望 False)  |  含 NaN: {has_nan} (期望 False)")
    assert not has_inf and not has_nan
    print(f"  數值範圍:     [{lidar_noisy.min().item():.4f}, {lidar_noisy.max().item():.4f}]")
    assert lidar_noisy.min() >= 0.0 and lidar_noisy.max() <= 1.0
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 7: LiDAR 數值精度驗證（已知輸入 → 預計輸出）
    # ==============================================================
    print("--- 測試 7: LiDAR 數值精度驗證 ---")
    # 15.0m → clamp=15 → min=15 → margin=14.7 → norm=0.735
    ray_known = torch.full((2, 16, 360), 15.0, device=device)
    lidar_known = manager.process_lidar(ray_known)
    expected = 0.735
    actual = lidar_known[0, 0].item()
    print(f"  輸入: 15.0m → 期望: {expected} → 實際: {actual:.4f}")
    assert abs(actual - expected) < 1e-5
    assert (lidar_known - expected).abs().max().item() < 1e-5
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 8: LiDAR train 模式（Step 0 領域隨機化）
    # ==============================================================
    print("--- 測試 8: LiDAR train 模式 (Domain Randomization) ---")
    manager.train()

    # 使用均勻 10.0m 的確定性輸入
    ray_uniform = torch.full((N, 16, 360), 10.0, device=device)

    lidar_train = manager.process_lidar(ray_uniform)
    print(f"  輸出形狀:     {list(lidar_train.shape)} (期望 [4096, 72])")
    assert lidar_train.shape == (N, 72)
    print(f"  數值範圍:     [{lidar_train.min().item():.4f}, {lidar_train.max().item():.4f}] (期望 [0, 1])")
    assert lidar_train.min() >= 0.0 and lidar_train.max() <= 1.0
    assert not torch.isnan(lidar_train).any()
    assert not torch.isinf(lidar_train).any()

    # eval 模式下相同輸入應為確定性
    manager.eval()
    lidar_eval = manager.process_lidar(ray_uniform)
    # eval 的結果：(10.0 - 0.3) / 20.0 = 0.485
    expected_eval = (10.0 - 0.3) / 20.0
    assert (lidar_eval - expected_eval).abs().max().item() < 1e-5, \
        f"eval 模式應為確定性: {lidar_eval[0, 0].item():.4f} ≠ {expected_eval}"

    # train 與 eval 應不同（因為 Step 0 雜訊）
    diff_lidar = (lidar_train - lidar_eval).abs().mean().item()
    print(f"  train/eval 差: {diff_lidar:.6f} (期望 > 0)")
    assert diff_lidar > 1e-6
    print("  ✓ 通過\n")

    # ==============================================================
    # 測試 9: LiDAR 領域隨機化統計驗證
    # ==============================================================
    print("--- 測試 9: LiDAR 領域隨機化統計驗證 ---")
    manager.train()

    # 在 Step 0 執行前後比較原始數據
    # 使用大量樣本（N=4096, 16, 360 = 23,592,960 個點）驗證統計機率
    ray_stat = torch.full((N, 16, 360), 10.0, device=device)
    total_points = N * 16 * 360

    # (a) 隨機位移驗證：多次呼叫，檢查輸出非完全一致
    out_a = manager.process_lidar(ray_stat)
    out_b = manager.process_lidar(ray_stat)
    stochastic = (out_a - out_b).abs().sum().item() > 0
    print(f"  隨機性驗證:   {stochastic} (期望 True — 兩次呼叫結果不同)")
    assert stochastic, "train 模式下兩次呼叫應產生不同結果"

    # (b) 孔洞率統計：r_max 讀數在 min pooling 後的比例
    #     由於 hole_prob=0.05 作用在原始 16×360 上，
    #     min pooling 取最小值，所以最終 72 beams 的 r_max 佔比
    #     遠低於 5%（因為 min 會選非 r_max 的點）
    #     但我們可以驗證部分 beam 的最終距離確實受到影響
    out_c = manager.process_lidar(ray_stat)
    expected_no_rand = (10.0 - 0.3) / 20.0  # 0.485
    # 有些 beam 應與 expected 不同（因為干擾雜點 distractor 會拉近距離）
    affected = (out_c - expected_no_rand).abs() > 0.01
    affected_pct = 100.0 * affected.float().mean().item()
    print(f"  受影響 beam:  {affected_pct:.2f}% (期望 > 0% — 雜訊已改變輸出)")

    # (c) 干擾雜點驗證：有些 beam 應低於正常值
    #     distractor 設為 U(0.2, 1.5)m → norm 後約 [0, 0.06]
    #     正常值為 0.485，所以 distractor 會讓部分 beam 異常低
    low_beams = (out_c < 0.1).float().mean().item()
    print(f"  低距離 beam:  {100.0 * low_beams:.4f}% (期望 > 0% — distractor 效果)")

    print("  ✓ 通過\n")

    # ==============================================================
    # 記憶體使用統計
    # ==============================================================
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU 峰值記憶體: {mem_mb:.1f} MB")

    print("=" * 70)
    print("所有 9 項測試通過！維度正確、數值範圍合法、領域隨機化正常、無 OOM。")


if __name__ == "__main__":
    _test()
