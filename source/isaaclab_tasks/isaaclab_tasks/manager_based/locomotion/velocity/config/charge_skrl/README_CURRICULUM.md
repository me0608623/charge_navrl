# Charge-SKRL NavRL Curriculum — 8 階段課程學習導航訓練

> **基於 Isaac Lab + SKRL PPO 的移動機器人自主導航，
> 採用 NavRL-style dense rewards 與 8 階段線性課程學習策略**

---

## 專案概述

本專案在 **20×20m** 封閉場景中訓練 Charge 差速驅動機器人（SKRL PPO），
透過 8 階段線性課程逐步增加障礙物密度與牆壁數量，
同時保持**獎勵函數在所有階段完全不變**（只改環境參數）。

### 核心特色

- **8 階段線性課程**：Stage 1 純導航 → Stage 8 終極挑戰，每階段 +1 static / +1 dynamic / -1 goal
- **NavRL-style dense rewards**：velocity-to-goal + safety log-distance + safe progress（固定權重）
- **Per-env 隨機牆壁**：8 個牆壁槽位，每個 env 獨立隨機放置（rejection sampling）
- **VLP-16 LiDAR 72-bin** + Top-10 obstacles 6D body-frame 觀測 = **139D**
- **Discrete 19×19 動作空間**：動態加速度邊界，允許倒車
- **防止災難性遺忘**：升級需連續 5 次通過 + 自動降級機制

---

## 快速開始

### 訓練

```bash
# NavRL dense rewards + 8 階段課程（推薦）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_NavRL.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
    --num_envs 256 \
    --headless

# v9 expected-value rewards + 相同課程
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_NavRL.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum \
    --num_envs 256 \
    --headless
```

### 從 checkpoint 繼續

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_NavRL.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
    --checkpoint logs/charge_vlp16/checkpoints/best_agent.pt \
    --num_envs 256 \
    --headless
```

### WandB

訓練自動連接 WandB（headless 模式下 project = `charge_skrl_navrl`）。
設定環境變數 `WANDB_MODE=disabled` 可關閉。

---

## 架構總覽

```
Scene (20×20m, env_spacing=22m)
├── Robot (Charge)
│   ├── Differential drive — v ∈ [-1.0, +1.0] m/s, ω ∈ [-0.25π, +0.25π] rad/s
│   └── Body radius = 0.35m
├── Boundary Walls (4)
│   └── 1.0m thickness, 21m length (封閉房間)
├── Internal Walls (8 slots)
│   └── Per-env randomized — lengths 2.5~5.0m, 1.0m thick, 1.5m tall
├── Obstacles (10 entities)
│   ├── 5 cuboids (0.4~0.7m)
│   ├── 5 cylinders (r=0.2~0.35m)
│   └── Mixed parallel: empty / static / dynamic ratio per curriculum
└── Goals (1~8)
    └── Multi-goal sequential command, distance 2.0~13.0m
```

---

## 觀測空間 (139D)

| 索引 | 模組 | 維度 | 說明 |
|------|------|------|------|
| `[0:4]` | Ego State | 4 | `[ā_t, v̄_t, ω̄_t, r_a]` — 正規化加速度、速度、角速度、機器人半徑(0.35) |
| `[4:6]` | Goal Command | 2 | `[Δx, Δy]` — 目標在 robot body frame 的相對位置 |
| `[6:78]` | LiDAR (VLP-16) | 72 | 72-bin 距離掃描，r_max=20m，含噪聲 (σ=0.02m, hole=0.5%, distractor=0.2%) |
| `[78:138]` | Obstacles | 60 | Top-10 × 6D `[x, y, vx, vy, r, m]` — body frame，LOS 遮擋 mask |
| `[138]` | Time | 1 | `T_rem` — 剩餘時間比例 ∈ [0, 1] |

**說明**：
- Policy 和 Critic 使用相同 139D 觀測（對稱架構，無 privileged information）
- 障礙物觀測：`topk_obstacles_6d()` 提取最近 10 個障礙物，轉換至 body frame
- LOS 遮擋：若牆壁遮擋視線，`m=0`（不可見）；超出 8m 或不存在的槽位 → `[0,0,0,0,0,0]`

---

## 動作空間

**Discrete(361)** = 19 × 19 展平索引

| 分量 | 類型 | 範圍 | 分箱 |
|------|------|------|------|
| 線速度 (v) | Discrete | -1.0 ~ +1.0 m/s | 19 bins |
| 角速度 (ω) | Discrete | -0.25π ~ +0.25π rad/s | 19 bins |

**物理限制**

| 參數 | 值 |
|------|-----|
| v_max | 1.0 m/s |
| a_max | 0.5 m/s² |
| ω_max | 0.25π rad/s (≈ 0.785 rad/s) |
| dt (policy step) | 0.2s (sim.dt=0.01s × decimation=20) |

**動態加速度邊界**：

```
a_min = max(-a_max, (-v_max - v_curr) / dt)
a_max = min(+a_max, (v_max - v_curr) / dt)
```

- 中心索引 (9, 9) = 零速度
- 允許倒車：v ∈ [-v_max, +v_max]

---

## 獎勵設計 (NavRL Dense)

所有階段使用完全相同的獎勵函數與權重（課程只改環境參數）。

| # | 獎勵項 | 權重 | 範圍 | 說明 |
|---|--------|------|------|------|
| 1 | `reaching_goal` | +500 | terminal | 距離目標 < 0.35m 時觸發 |
| 2 | `velocity_to_goal` | +15 | [-1, 1] | 朝向目標速度投影，近障礙物時衰減 |
| 3 | `safe_progress` | +20 | [-1, 1] | PBRS 距離縮減 × 安全 gate（danger zone 內翻負 ×-0.5） |
| 4 | `safety_log_distance` | +3 | [-3, 3] | log(d_safe) × 速度耦合（靜止 ×0.1） |
| 5 | `collision_terminal` | -100 | terminal | LiDAR 最近距離 < 0.45m |
| 6a | `acceleration_penalty` | -0.05 | per-step | deadzone=0.1，馬達保護 |
| 6b | `angular_velocity_penalty` | -0.05 | per-step | 靠近障礙物時允許急轉（max_reduction=0.7） |

**行為排序**：安全前進(+6.8) >> 原地不動(≈0) >> 危險前進(-1.2) >> 危險撤退(-0.8) >> 碰撞(-100)

---

## 課程階段表 (8 階段)

由 `_build_stages()` 線性生成，規則：每階 -1 goal, +1 static, +1 dynamic。

| Stage | Goals | Static | Dynamic | Walls (min-max) | Episode | γ | Empty% | 升級 SR | 升級 CR | 升級 TO |
|-------|-------|--------|---------|-----------------|---------|-------|--------|---------|---------|---------|
| **1** | 8 | 0 | 0 | 0-2 | 45s | 0.990 | 100% | > 72% | — | < 30% |
| **2** | 7 | 1 | 1 | 0-3 | 51s | 0.991 | 88% | > 65% | < 40% | < 30% |
| **3** | 6 | 2 | 2 | 1-4 | 52s | 0.993 | 76% | > 65% | < 40% | < 30% |
| **4** | 5 | 3 | 3 | 1-5 | 58s | 0.994 | 63% | > 65% | < 40% | < 30% |
| **5** | 4 | 4 | 4 | 2-6 | 64s | 0.995 | 51% | > 65% | < 40% | < 30% |
| **6** | 3 | 5 | 5 | 3-7 | 71s | 0.997 | 39% | > 65% | < 40% | < 30% |
| **7** | 2 | 6 | 6 | 3-8 | 77s | 0.998 | 27% | > 65% | < 40% | < 30% |
| **8** | 1 | 7 | 7 | 4-8 | 90s | 0.998 | 15% | — | — | — |

**升降級機制**

| 機制 | 條件 | 備註 |
|------|------|------|
| 升級 | SR + CR + TO 全部通過，連續 **5 次** | `UPGRADE_PASS_REQUIRED = 5` |
| 降級 | SR < 15% **或** CR > 70% **或** TO > 65% | 一次即降，Stage 1 不降 |
| 窗口 | max(2000, num_envs × 5) episodes | 最低停留 max(5000, num_envs × 8) episodes |

---

## 牆壁系統

### WALL_SLOT_SPECS (8 個槽位)

| Slot | 長度 | 寬度 | 高度 |
|------|------|------|------|
| 0 | 4.0m | 1.0m | 1.5m |
| 1 | 3.0m | 1.0m | 1.5m |
| 2 | 5.0m | 1.0m | 1.5m |
| 3 | 3.5m | 1.0m | 1.5m |
| 4 | 4.5m | 1.0m | 1.5m |
| 5 | 2.5m | 1.0m | 1.5m |
| 6 | 4.0m | 1.0m | 1.5m |
| 7 | 3.0m | 1.0m | 1.5m |

### Per-env 隨機化機制

1. **Startup**：所有牆壁槽位初始化隱藏於 Z=-10.0
2. **Reset**：每個 env 獨立隨機放置 `min_walls` ~ `max_walls` 面牆（由課程控制）
3. **Rejection sampling**（最多 30 次嘗試）：
   - 牆壁 AABB 不超出場地邊界（boundary=8.5m）
   - 牆壁間最小間距 2.0m
   - 與機器人最小距離 1.5m
4. **Per-env 數據**：`_maze_wall_centers [N, 8, 2]`、`_maze_wall_sizes [N, 8, 2]`、`_maze_wall_mask [N, 8]`
5. **Boundary walls**（全局共享）：20×20m, 1.0m 厚度

### LOS 遮擋

- `check_los_perenv()`：AABB slab method 檢查每條視線是否被牆壁阻擋
- 複雜度：O(N × F × W) = 256 × 10 × 12 ≈ 30K — GPU 上微不足道
- 被遮擋的障礙物在觀測中 mask=0

---

## 障礙物系統

### 實體配置 (10 個)

- **5 cuboids**：邊長 0.4m ~ 0.7m，高度 0.9m ~ 1.4m
- **5 cylinders**：半徑 0.2m ~ 0.35m，高度 0.8m ~ 1.5m
- 初始隱藏於 Z=-10.0（Isaac Sim kinematic body）

### Mixed Parallel 環境分布

課程控制 `empty_ratio` / `static_ratio` / `dynamic_ratio`，每個 env 在 reset 時分配類型：
- **Empty**：無障礙物，純導航練習
- **Static**：靜態障礙物，學習避障
- **Dynamic**：動態障礙物，最高難度

### 動態障礙物移動

| 參數 | 值 |
|------|-----|
| 更新間隔 | 0.2s |
| 速度範圍 | 0.3 ~ 1.2 m/s |
| 目標到達門檻 | 0.5m |
| 速度重採樣 | 每 10 步 |
| 活動邊界 | 9.0m |

---

## 終止條件

| 條件 | 函數 | 參數 | 觸發類型 |
|------|------|------|----------|
| `goal_reached` | `goal_reached()` | threshold=0.35m | 成功（non-timeout） |
| `collision` | `collision_occurred()` | LiDAR threshold=0.45m | 碰撞（non-timeout） |
| `wall_collision` | `wall_collision_termination()` | AABB threshold=0.45m | 碰撞（互補安全網） |
| `robot_tipped_over` | `robot_tipped_over()` | max_angle=45° | 翻倒 |
| `physics_explosion` | `physics_explosion()` | max_vel=100 m/s | NaN/物理不穩定 |
| `time_out` | `time_out()` | episode_length_s / 0.2 steps | 超時 |

- **Stage 1**：45s = 225 steps
- **Stage 8**：90s = 450 steps

---

## 訓練基礎設施

### SKRL PPO 超參數

| 參數 | 值 |
|------|-----|
| Rollout Steps | 128 |
| Learning Epochs | 6 |
| Mini Batches | 8 |
| Discount Factor (γ) | 動態：0.990 → 0.998（課程同步） |
| Learning Rate | 1e-4 (KL Adaptive: 3e-5 ~ 1.5e-4) |
| Entropy Scale | 0.01 |
| Gradient Clip | 1.0 |
| Ratio Clip | 0.2 |

### 3-Branch Feature Extractor (NavRL-style)

```
Input: 139D observation
  │
  ├─ LiDAR (72D) ──→ Conv2d(4, [5,3]) → Conv2d(16, [5,3], s=[2,1])
  │                   → Conv2d(16, [5,3], s=[2,2]) → Flatten → FC(128)
  │                   → LayerNorm(128) ──→ 128D
  │
  ├─ Obstacles (60D) → Linear(128) → LeakyReLU → LayerNorm(128)
  │                   → Linear(64)  → LeakyReLU → LayerNorm(64) ──→ 64D
  │
  └─ State (7D: ego+goal+time) ──→ pass-through ──→ 7D
                                                        │
                    Concat ◄───────────────────── 199D ──┘
                      │
                 Shared MLP [256, 256] → LeakyReLU → LayerNorm
                      │
                   256D fused
                    ╱    ╲
         Actor Head      Critic Head
     (256 → action)    (256 → 1 value)
```

### WandB 追蹤指標

訓練期間自動記錄：
- `stage` / `success_rate` / `collision_rate` / `timeout_rate`
- `dynamic_sr` — 動態環境成功率
- `sr_gap` / `cr_gap` / `to_gap` — 與升級門檻的差距
- `gamma` / `episode_length_s` — 當前課程參數
- PPO 標準指標：policy_loss, value_loss, entropy, kl_divergence

---

## 檔案結構

```
charge_skrl/
├── __init__.py                        # Gymnasium 環境註冊
├── README_CURRICULUM.md               # 本文檔
├── charge.usd                         # Robot URDF/USD model
├── goal_command.py                    # 單目標命令
├── multi_goal_command.py              # 多目標序列命令
│
├── cfg/
│   ├── charge_cfg.py                  # Robot articulation 配置
│   ├── charge_env_cfg.py              # 基礎環境配置
│   ├── charge_env_cfg_vlp16.py        # VLP-16 觀測/動作/終止配置
│   └── charge_env_cfg_vlp16_curriculum.py  # 課程環境 + NavRL rewards
│
├── curriculum/
│   ├── goal_obstacle_curriculum.py    # v12 — 8 階段線性課程邏輯
│   └── mixed_curriculum.py            # 混合訓練調度器
│
├── mdp/
│   ├── actions/                       # 離散差速驅動動作
│   ├── observations/                  # LiDAR binning + topk_obstacles_6d
│   ├── rewards/                       # NavRL dense + safety rewards
│   ├── terminations/                  # 目標到達 / 碰撞 / 超時
│   ├── events/                        # 障礙物隨機化 + 重置
│   └── wall_layout.py                 # 牆壁幾何 + per-env LOS 查詢
│
├── agents/                            # SKRL PPO yaml 配置
├── domain_randomization/              # 物理/感測器 DR (實驗性)
├── models/                            # 感測器融合模型 (實驗性)
├── wrappers/                          # CBF action wrapper (實驗性)
└── test/                              # 驗證測試
```

**訓練腳本**（`scripts/reinforcement_learning/skrl/`）：

| 檔案 | 說明 |
|------|------|
| `train_charge_NavRL.py` | NavRL 課程訓練主入口 |
| `navrl_models.py` | 3-branch CNN feature extractor |
| `aac_wrapper.py` | AAC (Actor-Actor-Critic) wrapper |
| `wandb_trainer.py` | WandB + curriculum 同步 trainer |
| `console_summary.py` | 終端機即時指標 + 升級差距顯示 |

---

## 已知問題與修復

### 1. NaN Poisoning (已修復)

**問題**：Scene entities 在 reset/collision 後可能出現 NaN 位置；IEEE 754 中 `NaN × 0 = NaN`，
導致無效槽位的零化乘法失敗，NaN 進入觀測後永久污染 `RunningStandardScaler` 統計。

**修復**：使用 `torch.where()` 替代乘法零化，並在 `topk_obstacles_6d()` 入口以 `nan_to_num()` 清洗輸入。

### 2. Temporal Mismatch (已修復)

**問題**：`shared_states` 從 `infos` 取得時對應的是 t+1 而非 t，導致觀測-獎勵不一致。

**修復**：在正確的時間點取樣 shared_states，確保 obs_t 對應 reward_t。

### 3. 16×16 Wall Bug (已修復)

**問題**：舊版 16×16m 場景的牆壁座標未更新至 20×20m，導致牆壁超出場景邊界。

**修復**：新增 `BOUNDARY_WALLS_20x20` 和 `MAZE_WALLS_20x20` 獨立定義，1.0m 厚度。

---

## 參考文獻

1. **NavRL** — Xu et al., Learning Safe Flight in Dynamic Environments, IEEE RA-L 2025
2. **Curriculum Learning** — Bengio et al., ICML 2009
3. **PPO** — Schulman et al., arXiv 2017
4. **SKRL** — https://github.com/Toni-SM/skrl
5. **Isaac Lab** — https://github.com/isaac-sim/IsaacLab
