# Charge-SKRL：基於離散 PPO 的 Isaac Lab 安全導航

> 一個基於離散動作空間 Proximal Policy Optimization、3D LiDAR 感知與潛能型獎勵塑形的差速驅動機器人 Sim-to-Real 導航框架。

[![English](https://img.shields.io/badge/lang-English-blue)](README_en.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-red)](README_zh_TW.md)

---

## 目錄

1. [概述](#1-概述)
2. [系統架構](#2-系統架構)
3. [觀測空間](#3-觀測空間)
4. [動作空間](#4-動作空間)
5. [網路架構](#5-網路架構)
6. [獎勵設計](#6-獎勵設計)
7. [終止條件](#7-終止條件)
8. [訓練環境](#8-訓練環境)
9. [域隨機化](#9-域隨機化)
10. [PPO 超參數](#10-ppo-超參數)
11. [快速開始](#11-快速開始)
12. [專案結構](#12-專案結構)
13. [參考文獻](#13-參考文獻)

---

## 1. 概述

本專案訓練差速驅動移動機器人（**Charge**）在包含靜態與動態障礙物的混亂環境中安全導航至目標位置。系統基於 NVIDIA Isaac Lab 建構，使用 SKRL 強化學習函式庫。

### 關鍵設計選擇

| 面向 | 選擇 | 理由 |
|------|------|------|
| 動作空間 | Discrete(361) | 消除連續動作熵崩潰；實現中心對稱探索 |
| 感知 | VLP-16 LiDAR → 72-bin 2D 投影 | Sim-to-real 可轉移；對 3D 幾何穩健 |
| 獎勵塑形 | 潛能型 (PBRS) | 策略不變性；總獎勵有界 |
| 安全信號 | 線性近障礙物懲罰 + 二元碰撞 | 碰撞前連續梯度 + 接觸時硬懲罰 |
| 訓練範式 | 混合並行課程 | 50% 空曠 / 30% 靜態 / 20% 動態障礙環境 |

### 物理參數

| 參數 | 值 | 單位 |
|------|-----|------|
| 機器人機身半徑 | 0.35 | m |
| 最大線速度 | 1.0 | m/s |
| 最大線加速度 | 0.5 | m/s² |
| 最大角速度 | 0.25π ≈ 0.785 | rad/s |
| 控制頻率 | 5 | Hz |
| 模擬 dt | 0.01 | s |
| Decimation | 20 | steps |
| 環境 dt | 0.2 | s |

---

## 2. 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab 環境                            │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  場景     │  │  VLP-16  │  │ 接觸     │  │   目標     │ │
│  │ (機器人, │  │  LiDAR   │  │ 感測器   │  │   命令     │ │
│  │  牆壁,   │  │ 16×360   │  │          │  │ (5 目標)   │ │
│  │  障礙物) │  │ = 5760   │  │          │  │            │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │
│       │             │             │               │        │
│  ┌────▼─────────────▼─────────────▼───────────────▼──────┐ │
│  │              觀測管理器 (139D)                          │ │
│  │  ego(4) + goal(2) + LiDAR(72) + obstacles(60) + t(1) │ │
│  └───────────────────────┬───────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────────┐ │
│  │              獎勵管理器 (8 項)                          │ │
│  │  r = Σ func_i(s) × weight_i × dt                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           域隨機化 (每次重置)                           │ │
│  │  物理 DR · 感測器 DR · 干擾 DR                          │ │
│  └───────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ obs (139D)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    SKRL PPO Agent                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            VLP16FeatureExtractor (128D)                │  │
│  │                                                        │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐  │  │
│  │  │ LiDAR Conv1d│ │ Obstacle MLP │ │   State MLP    │  │  │
│  │  │ 72 → 64D    │ │ 60 → 32D     │ │   7 → 32D     │  │  │
│  │  │ (MaxPool)   │ │ (MaxPool)    │ │                │  │  │
│  │  └──────┬──────┘ └──────┬───────┘ └───────┬────────┘  │  │
│  │         └───────────────┴─────────────────┘            │  │
│  │                     concat = 128D                      │  │
│  └─────────────────────────┬──────────────────────────────┘  │
│                            │                                  │
│         ┌──────────────────┴──────────────────┐               │
│         ▼                                     ▼               │
│  ┌─────────────┐                      ┌─────────────┐        │
│  │   Actor     │                      │   Critic    │        │
│  │ 128→128→361 │                      │ 128→64→32→1 │        │
│  │ (Categorical│                      │ (Scalar V)  │        │
│  │  Logits)    │                      │             │        │
│  └──────┬──────┘                      └─────────────┘        │
│         │                                                     │
│    action ∈ {0, ..., 360}                                     │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│          離散差速驅動控制器                                   │
│                                                              │
│  action_index → (accel_idx, omega_idx) → (a, ω)            │
│  v_{t+1} = clamp(v_t + a·dt, -v_max, +v_max)               │
│  將機體坐標系速度應用於模擬                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 觀測空間

策略和價值網路接收相同的 **139 維** 觀測向量（對稱 Actor-Critic，無特權資訊）。

### 3.1 佈局

| 索引 | 群組 | 維度 | 說明 | 範圍 |
|------|------|------|------|------|
| `[0]` | Ego | 1 | 正規化線加速度：`ā = a / a_max` | [-1, 1] |
| `[1]` | Ego | 1 | 正規化線速度：`v̄ = v / v_max` | [-1, 1] |
| `[2]` | Ego | 1 | 正規化角速度：`ω̄ = ω_z / ω_max` | [-1, 1] |
| `[3]` | Ego | 1 | 機器人機身半徑（常數） | 0.35 |
| `[4:6]` | Goal | 2 | 機體坐標系目標位置 `(x, y)` | m |
| `[6:78]` | LiDAR | 72 | 2D 最小距離 bins，正規化 | [0, 1] |
| `[78:138]` | Obstacles | 60 | Top-10 障礙物 × 6D 機體坐標系 | mixed |
| `[138]` | Time | 1 | 剩餘時間比例：`1 - t/T_max` | [0, 1] |

### 3.2 LiDAR 處理管線

VLP-16 感測器發射 5,760 條射線（16 垂直通道 × 360 水平取樣）。這些被壓縮為 72 維向量：

```
ray_hits_w [N, 5760, 3]          (原始 3D 擊中點)
  │
  ├── 2D 投影：d_ij = ||hit_xy - sensor_xy||₂
  │
  ├── 裁剪：d > r_max → r_max, NaN/Inf → r_max
  │
  ├── 點雲損壞 (Sim-to-Real DR)：
  │     • 高斯位移：σ = 0.02m
  │     • 射線丟失：0.5% → d = r_max
  │     • 幽靈點：0.2% → d ~ U(0.2, 2.0)m
  │
  ├── 重塑：[N, 5760] → [N, 16ch, 72bins, 5rays/bin]
  │
  ├── 最小池化：min over rays_per_bin → min over channels
  │     → [N, 72]   (每 bin 最小距離)
  │
  ├── 機器人半徑減法：d' = max(d - r_body, 0)
  │
  └── 正規化：d̄ = d' / r_max ∈ [0, 1]
```

### 3.3 障礙物觀測

每個障礙物以機體坐標系 6D 向量表示：

```
o_i = [x̂, ŷ, v̂_x, v̂_y, r, m]
```

| 欄位 | 定義 | 正規化 |
|------|------|--------|
| `x̂, ŷ` | 機體坐標系相對位置 | `/ 8.0m`, clamp [-1, 1] |
| `v̂_x, v̂_y` | 機體坐標系相對速度 | `/ 1.5 m/s`, clamp [-2, 2] |
| `r` | 物體半徑 | 公尺（原始） |
| `m` | 可見性遮罩 | 1.0 = 可見, 0.0 = 遮擋/填充 |

**機體坐標系旋轉**：
```
[x̂]   [cos(ψ)   sin(ψ)] [Δx]
[ŷ] = [-sin(ψ)  cos(ψ)] [Δy]
```

**Top-K 選擇**：依距離排序，保留最近 10 個。無效欄位以 `torch.where` 歸零（非乘法，避免 IEEE 754 NaN 傳播）。

**牆壁 LOS 遮擋**：對所有迷宮牆壁進行 AABB 線段相交測試。被遮擋的障礙物設為 `m = 0`。

---

## 4. 動作空間

### 4.1 Discrete(361) = 19 × 19 中心對稱網格

神經網路輸出單一整數 `a ∈ {0, 1, ..., 360}`。

**解碼**：
```
accel_idx = a // 19     ∈ {0, ..., 18}
omega_idx = a % 19      ∈ {0, ..., 18}

ratio_a = (accel_idx - 9) / 9     ∈ [-1.0, +1.0]
ratio_ω = (omega_idx - 9) / 9     ∈ [-1.0, +1.0]
```

索引 9 對應零（無加速度，無轉向）。此**中心對稱**設計確保動作流形的均勻覆蓋。

### 4.2 動態加速度邊界

確保積分後速度保持在物理限制內：

```
a_upper = min(+a_max, (+v_max - v_t) / dt)
a_lower = max(-a_max, (-v_max - v_t) / dt)
```

實際加速度：
```
        ┌ ratio_a × a_upper    if ratio_a ≥ 0
a_t =   │
        └ -ratio_a × a_lower   if ratio_a < 0
```

### 4.3 速度積分

```
v_{t+1} = clamp(v_t + a_t · dt,  -v_max, +v_max)
ω_t     = ratio_ω · ω_max
```

結果 `(v_{t+1}, ω_t)` 通過機器人的偏航四元數從機體坐標系轉換到世界坐標系，作為根部速度應用。

---

## 5. 網路架構

### 5.1 特徵提取器（3 分支，128D 輸出）

#### LiDAR 分支（72D → 64D）
```
Input [B, 72] → reshape [B, 1, 72]
  → Conv1d(1→32, kernel=5, pad=2) → ReLU
  → Conv1d(32→64, kernel=5, stride=2, pad=2) → ReLU
  → Conv1d(64→64, kernel=3, stride=2, pad=1) → ReLU
  → AdaptiveMaxPool1d(1)
  → Linear(64→64) → LayerNorm(64)
Output [B, 64]
```

#### 障礙物分支（60D → 32D，排列不變）
```
Input [B, 60] → reshape [B, 10, 6]
  → Linear(6→32) → ReLU → Linear(32→32) → ReLU   (每物體共享)
  → MaxPool over 10 objects (dim=1)
  → LayerNorm(32)
Output [B, 32]
```

每物體 MLP 在所有 10 個物體間共享。MaxPool 聚合確保表示對障礙物順序**排列不變**——這是關鍵特性，因為物體身份是任意的。

#### 狀態分支（7D → 32D）
```
Input [B, 7]   (ego 4D + goal 2D + time 1D)
  → Linear(7→32) → ReLU → Linear(32→32) → ReLU → LayerNorm(32)
Output [B, 32]
```

#### 融合
```
features = cat([LiDAR(64), Obstacles(32), State(32)]) = 128D
```

### 5.2 策略頭（Actor）

```
features [B, 128]
  → Linear(128→128) → ReLU
  → Linear(128→361)
Output: 361 unnormalized logits → Categorical distribution
```

### 5.3 價值頭（Critic）

```
features [B, 128]
  → Linear(128→64) → ReLU
  → Linear(64→32) → ReLU
  → Linear(32→1)
Output: scalar V(s)
```

初始化：輸出層使用 `orthogonal_(gain=0.01)`，bias = `-8.0`（悲觀初始價值估計，防止早期高估）。

---

## 6. 獎勵設計

### 6.1 公式

IsaacLab 的獎勵管理器計算每個獎勵項：

```
r_i(t) = func_i(s_t) × weight_i × dt
```

其中 `dt = 0.2s`。

### 6.2 獎勵項

#### (1) 到達目標 — `reaching_goal`

```
         ┌ 1.0    if d(robot, goal) - r_body < threshold
func =   │
         └ 0.0    otherwise

weight = +250.0
r = +50.0 per trigger   (250 × 0.2)
```
**類別**：稀疏終止獎勵。機器人機身邊緣進入目標區域時觸發一次。閾值 = 0.35m。

#### (2) 潛能進度 — `potential_progress`

```
Φ(s) = -d(robot, goal)
func = Φ(s_{t+1}) - Φ(s_t) = d_t - d_{t+1}
func = clamp(func, -1.0, +1.0)

weight = +60.0
r = 12.0 × Δd   (60 × 0.2 × Δd)
```
**類別**：潛能型獎勵塑形 (Ng et al., 1999)。望遠鏡性質保證總塑形獎勵等於 `12 × (d_0 - d_final)`，與路徑無關。這保持了最佳策略不變性。

#### (3) 近障礙物懲罰 — `near_obstacle_penalty`

```
d_min = min(LiDAR 2D distances)

func = max(0, α · (d_safe - d_min))
     = max(0, 4.0 · (1.2 - d_min))

weight = -1.0
r = -0.2 × max(0, 4.0 · (1.2 - d_min))
```

**類別**：連續安全信號。在碰撞發生前提供梯度資訊。

| d_min | func | r per step |
|-------|------|------------|
| ≥ 1.2m | 0 | 0 |
| 1.0m | 0.8 | -0.16 |
| 0.7m | 2.0 | -0.40 |
| 0.5m | 2.8 | -0.56 |

#### (4) 碰撞終止 — `collision_terminal`

```
func = 1.0   if d_min ≤ 0.7m   (= r_body + collision_buffer)
     = 0.0   otherwise

weight = -80.0
r = -16.0 per trigger   (-80 × 0.2)
```
**類別**：二元終止懲罰。在 episode 終止的同一步觸發。

#### (5) 時間懲罰 — `time_penalty`

```
func = 1.0   (每步常數)

weight = -0.3
r = -0.06 per step
```
**類別**：反閒晃。完整 episode（225 步）：累計 = -13.5。

#### (6) 速度過低 — `velocity_too_low`

```
func = 1.0   if ||v_xy||₂ < 0.05 m/s
     = 0.0   otherwise

weight = -0.3
r = -0.06 per step when stationary
```
**類別**：反凍結。防止退化到「什麼都不做 = 安全」策略。

#### (7) 加速度懲罰 — `acceleration_penalty`

```
func = |applied_a|²    (from action term, post-clipping)
func = clamp(func, 0, 10)

weight = -0.15
r = -0.03 × a²
```
**類別**：平滑性。a_max=0.5 時最大懲罰：r = -0.0075/step（可忽略）。

#### (8) 角速度懲罰 — `angular_velocity_penalty`

```
func = |ω_z|²
func = clamp(func, 0, 10)

weight = -0.15
r = -0.03 × ω²
```
**類別**：平滑性。鼓勵直線運動，懲罰不必要的振盪。

### 6.3 獎勵尺度分析

| 場景 | 主導項 | 總 r/step |
|------|--------|-----------|
| 正常導航（0.5 m/s，路徑暢通） | progress ≈ +1.2, time = -0.06 | **≈ +1.1** |
| 近障礙物（d_min = 0.8m） | progress ≈ +1.2, near_obs ≈ -0.32 | **≈ +0.8** |
| 到達目標 | goal = +50.0 | **≈ +51** |
| 碰撞 | collision = -16.0 | **≈ -16.4** |
| 靜止 | time = -0.06, vel_low = -0.06 | **≈ -0.12** |

**設計原則**：獎勵層級為 `goal(+50) >> progress(+2.4/step max) >> safety(-16) >> anti-loaf(-0.12)`。一次碰撞需要 16.0 / 1.1 ≈ **15 步正常導航**才能恢復。

---

## 7. 終止條件

| 條件 | 類型 | 觸發 | 備註 |
|------|------|------|------|
| `time_out` | 截斷 | Episode > 45s（225 步） | Bootstrapped（GAE 繼續） |
| `goal_reached` | 終止 | `d(robot, goal) - r_body < 0.35m` | Episode 成功 |
| `collision` | 終止 | LiDAR d_min ≤ 0.7m | `r_body + 0.35m` 緩衝 |
| `robot_tipped_over` | 終止 | Z 軸與向上向量點積 < 0.5 | > 60° 傾斜 |
| `physics_explosion` | 終止 | `||v|| > 10 m/s` 或 `|ω| > 20 rad/s` 或 NaN | 模擬不穩定保護 |

---

## 8. 訓練環境

### 8.1 場地

- **房間大小**：16m × 16m（邊界牆在 ±8m）
- **內部牆壁**：6 個迷宮區段（長方體，運動學）
- **並行環境**：1024–6144（可配置）
- **環境間距**：18m（防止跨環境干擾）

### 8.2 混合並行課程

每個環境獨立分配難度級別：

| 難度 | 比例 | 障礙物 | 障礙物運動 |
|------|------|--------|-----------|
| 空曠 | 50% | 0 | — |
| 靜態 | 30% | 5 | 靜止 |
| 動態 | 20% | 8 | 0.3–1.2 m/s 隨機漫步 |

此混合並行方法提供自然的課程：Agent 同時體驗簡單和困難場景，防止災難性遺忘同時保持訓練效率。

### 8.3 障礙物生成

- **與機器人安全距離**：最小 1.5m
- **與目標安全距離**：最小 1.0m
- **障礙物間距**：最小 1.0m
- **放置方法**：基於象限的拒絕採樣（最多 50 次嘗試後回退）
- **類型**：長方體（0.4–0.7m）和圓柱體（r = 0.2–0.35m），隨機混合

### 8.4 目標命令

- **多目標**：每 episode 5 個連續目標
- **距離範圍**：[3, 8]m 從機器人
- **角度範圍**：完整 360°
- **拒絕採樣**：邊界檢查（7.5m）、牆壁鄰近（0.5m）、障礙物淨空（1.0m）

---

## 9. 域隨機化

域隨機化在每次重置時應用於 episode，以改善 sim-to-real 轉移。

### 9.1 物理隨機化

| 參數 | 範圍 | 物理意義 |
|------|------|---------|
| 機器人質量 | ±15% | 負載變化 |
| 地面摩擦 | ±30% | 表面材質（磁磚、地毯、混凝土） |
| 質心偏移 XY | ±5cm | 不對稱負載放置 |
| 質心偏移 Z | ±2.5cm | 垂直負載變化 |

### 9.2 初始狀態隨機化

| 參數 | 範圍 |
|------|------|
| 線速度 X | ±0.5 m/s |
| 線速度 Y | ±0.15 m/s |
| 角速度 Z | ±0.5 rad/s |

### 9.3 外部干擾

| 類型 | 參數 | 機制 |
|------|------|------|
| 隨機推力 | 10–30N, 15% 環境, 隨機方向 | `instantaneous_wrench_composer`（單步衝量） |
| 持續風力 | 0–5N, 每 episode 隨機方向 | `permanent_wrench_composer`（持續力） |

### 9.4 感測器噪聲（在 LiDAR 處理中）

| 類型 | 參數 | VLP-16 規格參考 |
|------|------|----------------|
| 高斯位移 | σ = 0.02m | ±3cm 典型精度 |
| 射線丟失 | 0.5% 機率 | 鏡面反射、暗色表面 |
| 幽靈點 | 0.2%, d ∈ [0.2, 2.0]m | 多路徑反射 |
| 輸出噪聲 | 均勻 ±0.02（正規化） | ADC 量化 |

---

## 10. PPO 超參數

| 參數 | 值 | 備註 |
|------|-----|------|
| 演算法 | PPO (Clip) | SKRL 實現 |
| Rollout 長度 | 128 步 | 每次更新 |
| 學習 epochs | 8 | 每次 PPO 更新 |
| Mini-batches | 8 | 每 epoch |
| 折扣因子 (γ) | 0.98 | 從 0.99 降低以限制價值幅度 |
| GAE lambda (λ) | 0.95 | |
| 學習率 | 1×10⁻⁴ | KL-Adaptive 調度器 (kl_threshold = 0.016) |
| Ratio clip (ε) | 0.2 | 標準 PPO clip |
| Value clip | 0.2 | `clip_predicted_values = True` |
| 熵係數 | 0.01 | H_max = ln(361) ≈ 5.89 |
| Value loss scale | 0.5 | 防止 critic 梯度主導 |
| 梯度範數 clip | 1.0 | |
| 狀態預處理器 | RunningStandardScaler | Policy 和 value 輸入 |
| Time limit bootstrap | True | 防止超時偏差 |
| Seed | 42 | |

**批次大小計算**：
- 每次更新：`128 steps × N_envs` 樣本
- 4096 環境：`524,288 samples / 8 mini-batches = 65,536` 每 mini-batch
- 每次 PPO 更新總梯度更新：`8 epochs × 8 mini-batches = 64`

---

## 11. 快速開始

### 前置需求

- NVIDIA Isaac Lab (Isaac Sim 4.5+)
- SKRL ≥ 1.3
- Python 3.11, PyTorch 2.x, CUDA 12+
- Weights & Biases 帳號（可選）

### 訓練

```bash
conda activate env_isaaclab

# 標準訓練（4096 並行環境）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 4096 \
  --headless

# GPU 記憶體優化
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 6144 \
  --headless
```

### 評估

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 4 \
  --checkpoint logs/skrl/Isaac-Navigation-Charge-VLP16-AC/<timestamp>/checkpoints/best_agent.pt
```

### GPU 記憶體指南

| num_envs | VRAM（約） | 建議 GPU |
|----------|-----------|---------|
| 1024 | ~10 GB | RTX 3080 |
| 2048 | ~14 GB | RTX 4080 |
| 4096 | ~22 GB | RTX 4090 / A6000 |
| 6144 | ~28 GB | RTX 5090 |

---

## 12. 專案結構

```
charge_skrl/
├── scripts/reinforcement_learning/skrl/
│   ├── train_charge_ac.py          # 訓練入口
│   ├── vlp16_models.py             # VLP16 3 分支網路
│   ├── wandb_trainer.py            # WandB 訓練器
│   ├── aac_wrapper.py              # 非對稱 AC wrapper
│   ├── training_debug_logger.py    # Debug 診斷
│   ├── console_summary.py          # Console 指標記錄器
│   └── play_charge.py              # 評估腳本
│
├── charge_skrl/
│   ├── __init__.py                 # Gymnasium 任務註冊
│   ├── goal_command.py             # 單目標命令
│   ├── multi_goal_command.py       # 多目標順序命令
│   │
│   ├── cfg/
│   │   ├── charge_cfg.py           # 機器人 USD/URDF 配置
│   │   ├── charge_env_cfg.py       # 基礎環境配置
│   │   └── charge_env_cfg_vlp16.py # VLP-16 環境配置
│   │
│   ├── agents/
│   │   └── skrl_ppo_cfg_vlp16.yaml # PPO 超參數
│   │
│   ├── mdp/
│   │   ├── actions/
│   │   │   └── discrete_differential_drive.py   # Discrete(361) 動作
│   │   ├── observations/
│   │   │   ├── functions.py         # Ego 狀態觀測
│   │   │   └── obs_functions.py     # LiDAR & 障礙物觀測
│   │   ├── rewards/
│   │   │   ├── potential_based_rewards.py  # PBRS, 懲罰
│   │   │   ├── goal_rewards.py      # 到達目標獎勵
│   │   │   └── safety_rewards.py    # 碰撞懲罰
│   │   ├── terminations/
│   │   │   ├── goal.py              # 到達目標終止
│   │   │   └── robot_state.py       # 碰撞、傾倒、爆炸
│   │   └── events/
│   │       ├── reset.py             # 安全初始狀態採樣
│   │       ├── obstacles.py         # 障礙物生成
│   │       └── mixed_parallel.py    # 難度課程
│   │
│   ├── domain_randomization/
│   │   ├── dr_events.py             # DR 編排
│   │   ├── physics_dr.py            # 質量、摩擦、質心
│   │   ├── sensor_dr.py             # LiDAR 噪聲
│   │   ├── disturbance_dr.py        # 風、推力
│   │   └── actuator_dr.py           # 馬達響應變化
│   │
│   └── charge_URDF/urdf/charge.urdf # 機器人模型
```

---

## 13. 參考文獻

1. **Ng, A. Y., Harada, D., & Russell, S. (1999)**. Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017)**. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

3. **Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021)**. RMA: Rapid Motor Adaptation for Legged Robots. *RSS*.

4. **Xu, H., et al. (2025)**. NavRL: Learning Safe Flight in Dynamic Environments. *IEEE RA-L*.

5. **NVIDIA Isaac Lab**. https://github.com/isaac-sim/IsaacLab

6. **SKRL**. https://github.com/Toni-SM/skrl

---

*Generated for the Charge-SKRL project. Branch: `charge-skrl-vlp16`.*
