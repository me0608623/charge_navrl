# Charge-SKRL Navigation — 完整技術規格書

> 版本: 2026-03-24 | Branch: `abl`
> Reward: navrl_ground_v4 | Curriculum: goal_first_v2
> Commit: 見 `git rev-parse HEAD`

---

## 1. 訓練設定與實驗可重現性

### 1.1 演算法

| 項目 | 值 |
|------|-----|
| 演算法 | PPO (Proximal Policy Optimization) |
| 實作來源 | SKRL (v1.4+), PyTorch backend |
| Repo | `git@github.com:Toni-SM/skrl.git` |
| 自訂整合 | `wandb_trainer.py` (繼承 SequentialTrainer) |

### 1.2 PPO 超參數（精確值）

| 參數 | 值 | 備註 |
|------|-----|------|
| rollouts | 128 | 每次 PPO update 收集的 transitions |
| learning_epochs | 6 | 每輪 rollout 的梯度更新次數 |
| mini_batches | 16 | 每 epoch 的 batch 分割數 |
| discount_factor (γ) | 0.990 | Phase 1 初始值；課程動態調整 0.990→0.998 |
| GAE lambda (λ) | 0.95 | |
| learning_rate (初始) | 1e-4 | LinearLR scheduler |
| learning_rate (最終) | 3e-5 | = 初始 × 0.3 |
| LR scheduler | LinearLR | factor: 1.0→0.3, total_iters=7324 |
| gradient_clip (max_grad_norm) | 1.0 | |
| ratio_clip (ε) | 0.2 | PPO surrogate clipping |
| value_clip | 0.2 | |
| clip_predicted_values | False | |
| entropy_loss_scale | 0.01 | |
| value_loss_scale | 1.0 | |
| KL threshold | 0.0 | 停用 |
| rewards_shaper_scale | 1.0 | |
| time_limit_bootstrap | True | timeout 時 bootstrap value |
| state_preprocessor | RunningStandardScaler | 觀測正規化 |
| value_preprocessor | RunningStandardScaler | 價值正規化 |
| random_timesteps | 0 | |
| learning_starts | 0 | |
| memory | RandomMemory (unbounded) | |

### 1.3 訓練步數

| 項目 | 值 |
|------|-----|
| total_timesteps | 234,375 |
| PPO updates | 234,375 / 128 = 1,831 |
| total gradient steps | 1,831 × 4 = 7,324 |
| 預設 num_envs | 6144（可 CLI 覆蓋） |
| total env steps | 234,375 × 6144 = 1,440,000,000 |

### 1.4 Seed

| 項目 | 值 |
|------|-----|
| 預設 seed | 42 |
| CLI 覆蓋 | `--seed N`（-1 = random） |
| 設定位置 | `env_cfg.seed` + `agent_cfg["seed"]` |

### 1.5 硬體

| 項目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| CPU | Intel Core Ultra 7 265K |
| RAM | 48 GB |
| OS | Ubuntu 24.04.3 LTS |
| CUDA | 13.0 |
| Driver | 580.126.09 |
| Isaac Sim | 5.1.0-rc.19 |
| PhysX | 107.3.26 |

### 1.6 Checkpointing

| 項目 | 值 |
|------|-----|
| 保存間隔 | SKRL 預設（每 N updates） |
| 保存內容 | Actor weights, Critic weights, optimizer state, RunningStandardScaler (mean/std/count) |
| Phase 2 載入 | `--checkpoint <path> --phase 2` |

---

## 2. 感測器與輸入觀測

### 2.1 LiDAR (Velodyne VLP-16)

| 參數 | 值 |
|------|-----|
| 類型 | MultiMeshRayCasterCfg |
| 垂直通道 | 16 |
| 垂直 FoV | -15° ~ +15° |
| 水平 FoV | -180° ~ +180° (全 360°) |
| 水平解析度 | 1.0° / ray |
| 總 ray 數 | 16 × 360 = 5,760 |
| 最大距離 | 20.0 m |
| 更新頻率 | 0.2 s (= control dt) |
| 安裝偏移 | (0, 0, 0.5) m (base_link 上方) |
| Ray alignment | yaw |
| 追蹤目標 | Ground (static), Wall_.* (track_transforms=True), Obstacle_.* (track_transforms=True) |

### 2.2 LiDAR 噪聲 (Domain Randomization)

| 參數 | 值 | 說明 |
|------|-----|------|
| displacement_std | 0.02 m | 每條 ray 的高斯距離雜訊 (±2cm 1σ) |
| hole_rate | 0.005 | ray 遺失機率 (0.5%) |
| distractor_rate | 0.002 | 幽靈點機率 (0.2%) |
| distractor_range | (0.2, 2.0) m | 幽靈點距離範圍 |

### 2.3 接觸感測器

| 參數 | 值 |
|------|-----|
| 安裝位置 | Robot/base_link |
| 更新頻率 | 每 sim step (0.01s) |
| 過濾目標 | Obstacle_.* |

### 2.4 觀測結構 (139D)

| 索引 | 欄位 | 維度 | 範圍 | 函數 |
|------|------|------|------|------|
| [0] | 正規化線加速度 ā | 1 | [-1, 1] | normalized_linear_acceleration |
| [1] | 正規化線速度 v̄ | 1 | [-1, 1] | normalized_linear_velocity |
| [2] | 角速度 ω_z | 1 | ~[-1.5, 1.5] rad/s | base_angular_velocity_z |
| [3] | 機器人半徑 | 1 | 0.35 (常數) | robot_radius_obs |
| [4:6] | 目標相對位置 (dx, dy) | 2 | body frame, m | goal_position_in_robot_frame |
| [6:78] | LiDAR 72 bins | 72 | [0, 1] 正規化 | lidar_vlp16_to_2d_bins |
| [78:138] | Top-10 障礙物 × 6D | 60 | (見下) | topk_obstacles_6d |
| [138] | 剩餘時間比例 | 1 | [0, 1] | time_remaining_ratio |

**障礙物 6D 結構**: `[dx, dy, vx, vy, radius, mask]`
- dx, dy: body frame 相對位置 (m)
- vx, vy: body frame 相對速度 (m/s)
- radius: 障礙物半徑 (m)
- mask: 1.0=可見, 0.0=遮擋/超距

**LiDAR bins**: 5,760 rays → min-pool to 72 bins (80 rays/bin) → 距離/r_max 正規化

### 2.5 觀測正規化

| 項目 | 值 |
|------|-----|
| 方法 | RunningStandardScaler |
| 公式 | obs_norm = (obs - running_mean) / (running_std + eps) |
| 更新 | 每步更新 running stats |
| 推論 | 凍結 stats |
| 已知風險 | NaN 輸入會永久汙染 stats → 已加 NaN 保護 |

---

## 3. 特徵抽取與網路架構

### 3.1 Feature Extractor (VLP16FeatureExtractor)

Actor 和 Critic **各自有獨立的 feature extractor**（models_separate=True）。

#### Branch 1: LiDAR Conv1d → 64D
```
Input: [B, 1, 72]
  Conv1d(1, 32, k=5, p=2) + ReLU → [B, 32, 72]
  Conv1d(32, 64, k=5, s=2, p=2) + ReLU → [B, 64, 36]
  Conv1d(64, 64, k=3, s=2, p=1) + ReLU → [B, 64, 18]
  AdaptiveMaxPool1d(1) → [B, 64, 1]
  Linear(64, 64) + LayerNorm(64) → [B, 64]
```

#### Branch 2: Obstacle MLP (Permutation-Invariant) → 32D
```
Input: [B, 10, 6]
  Per-object: Linear(6, 32) + ReLU + Linear(32, 32) + ReLU → [B, 10, 32]
  MaxPool(dim=1) → [B, 32]
  LayerNorm(32) → [B, 32]
```

#### Branch 3: State MLP (ego+goal+time) → 32D
```
Input: [B, 7]
  Linear(7, 32) + ReLU → [B, 32]
  Linear(32, 32) + ReLU → [B, 32]
  LayerNorm(32) → [B, 32]
```

#### Fusion
```
cat([64, 32, 32]) → [B, 128]
```

### 3.2 Actor Head (VLP16DiscretePolicy)

```
[B, 128] → Linear(128, 128) + ReLU → Linear(128, 38) → [B, 38] logits
```
- 38 logits = 19 (linear accel) + 19 (angular vel)
- Mixin: MultiCategoricalMixin (unnormalized_log_prob=True, reduction="sum")
- Weight init: Kaiming uniform (PyTorch 預設)

### 3.3 Critic Head (VLP16Value)

```
[B, 128] → Linear(128, 64) + ReLU → Linear(64, 32) + ReLU → Linear(32, 1)
```
- 最後一層: orthogonal_(gain=0.01), bias=constant_(-8.0)
- Mixin: DeterministicMixin

### 3.4 正規化

| 層 | 位置 |
|----|------|
| LayerNorm(64) | LiDAR branch 輸出 |
| LayerNorm(32) | Obstacle branch 輸出 |
| LayerNorm(32) | State branch 輸出 |
| 無 BatchNorm | |
| 無 Dropout | |

---

## 4. 動作空間與控制鏈

### 4.1 動作空間

| 參數 | 值 |
|------|-----|
| 類型 | MultiDiscrete([19, 19]) |
| 維度 | 2 (accel_idx, omega_idx) |
| 索引範圍 | [0, 18]，center=9 為零動作 |
| 映射 | idx → ratio = (idx - 9) / 9 ∈ [-1, 1] |

### 4.2 物理限制

| 參數 | 值 | 單位 |
|------|-----|------|
| v_max | 1.0 | m/s |
| a_max | 0.5 | m/s² |
| ω_max | 0.25π ≈ 0.785 | rad/s |
| 控制 dt | 0.2 | s |
| 速度域 | [-1.0, +1.0] | m/s（允許倒車） |

### 4.3 動態加速度邊界

```
a_min = max(-a_max, (-v_max - v_current) / dt)
a_max_actual = min(a_max, (v_max - v_current) / dt)

if ratio ≥ 0: accel = ratio × a_max_actual
if ratio < 0: accel = -ratio × a_min

v_next = clamp(v_current + accel × dt, -v_max, +v_max)
```

### 4.4 控制鏈

```
NN output [B, 38 logits]
  → Categorical sampling → [B, 2] indices
  → Index → ratio mapping
  → Dynamic accel bounds
  → Velocity integration
  → write_root_velocity_to_sim (differential drive)
```

### 4.5 Safety Shield (NavRL04, 可選)

| 參數 | 值 |
|------|-----|
| 類型 | Action post-processing (override apply_actions) |
| soft mode | d_safe < 1.2m 線性降速, < 0.55m 停止 |
| hard mode | d_safe < 0.55m 強制 v=0 |
| 影響 | 只裁切正向速度 (v > 0)，倒車不限 |
| 訓練/推論 | 訓練時即介入（agent 學到 shielded 行為） |

---

## 5. 環境、場景與 Curriculum

### 5.1 場景

| 參數 | Base (16×16m) | Curriculum (20×20m) |
|------|---------------|---------------------|
| 房間大小 | ±8m | ±10m |
| 邊界牆厚度 | 0.2m | 1.0m |
| 邊界牆高度 | 1.5m | 1.5m |
| 內部牆 | 6 (static AssetBaseCfg) | 8 slots (RigidObjectCfg, per-env 隨機) |
| 障礙物 | 10 (cuboid/cylinder) | 10 (同) |
| env_spacing | 18.0m | 22.0m |

### 5.2 模擬參數

| 參數 | 值 |
|------|-----|
| sim.dt | 0.01 s |
| decimation | 20 |
| control dt | 0.2 s (5 Hz) |
| episode_length | 45s~90s (課程動態) |
| max_steps/episode | 225~450 |
| gravity | 啟用 |
| ground friction | static=1.0, dynamic=1.0 |

### 5.3 障礙物

| 類型 | 數量 | 尺寸範圍 |
|------|------|---------|
| Cuboid | 5 | 0.4~0.7m × 0.4~0.7m × 0.9~1.4m |
| Cylinder | 5 | r=0.2~0.35m, h=0.8~1.5m |

動態障礙物行為: 隨機目標點移動, 速度 [0.3, 1.2] m/s, 每 10 步重新採樣速度, 8m 活動半徑

### 5.4 Domain Randomization

| 參數 | 範圍 |
|------|------|
| 質量縮放 | [0.85, 1.15] |
| 摩擦縮放 | [0.7, 1.3] |
| 質心偏移 | [-0.05, +0.05] m |
| 外力干擾 | [0, 3.0] N |
| LiDAR 距離雜訊 | ±0.02m (1σ) |
| LiDAR ray 遺失 | 0.5% |
| LiDAR 幽靈點 | 0.2% |

### 5.5 Reset 隨機化

| 參數 | Curriculum 20×20 |
|------|------------------|
| X | [-7.0, 7.0] m |
| Y | [-7.0, 7.0] m |
| Yaw | [-π, π] rad |
| 初速 vx | [-0.5, 0.5] m/s |
| 初速 vy | [-0.15, 0.15] m/s |
| 初速 ωz | [-0.5, 0.5] rad/s |

### 5.6 Curriculum — goal_first_v2 (12 階段)

| Stage | 名稱 | Goals | Static | Dynamic | Walls | γ | Episode | 升級 SR |
|-------|------|-------|--------|---------|-------|---|---------|---------|
| 1 | goal_open | 7 | 0 | 0 | 0-1 | 0.990 | 45s | >85% |
| 2 | goal_walls | 6 | 0 | 0 | 1-2 | 0.990 | 50s | >85% |
| 3 | static_light | 5 | 2 | 0 | 2-3 | 0.992 | 55s | >80% |
| 4 | static_medium | 4 | 4 | 0 | 3-4 | 0.993 | 60s | >80% |
| 5 | static_dense | 3 | 6 | 0 | 3-5 | 0.994 | 65s | >78% |
| 6 | dynamic_intro | 3 | 5 | 2 | 4-5 | 0.995 | 72s | >75% |
| 7 | dynamic_medium | 2 | 6 | 4 | 5-6 | 0.996 | 78s | >72% |
| 8 | crowded | 1 | 7 | 6 | 6-7 | 0.997 | 85s | >70% |
| 9 | dense_9 | 1 | 8 | 8 | 7-8 | 0.998 | 90s | >68% |
| 10 | dense_10 | 1 | 9 | 9 | 7-8 | 0.998 | 90s | >65% |
| 11 | dense_11 | 1 | 10 | 10 | 8-8 | 0.998 | 90s | >60% |
| 12 | ultimate | 1 | 10 | 10 | 8-8 | 0.998 | 90s | 100% |

升級條件: SR > 門檻 AND CR < 門檻 AND TO < 門檻，需連續 5 次通過。
降級條件: SR < 降級門檻（一次即降）。

**Stage-Dependent Reward Weights:**

| Stage | reaching_goal | goal_vel | goal_prog | static_safety | dynamic_safety |
|-------|:---:|:---:|:---:|:---:|:---:|
| 1-2 | 500 | 5.0 | 6.0 | 0.2 | 0.0 |
| 3 | 500 | 4.5 | 5.5 | 0.4 | 0.0 |
| 4 | 500 | 4.0 | 5.0 | 0.8 | 0.0 |
| 5 | 500 | 3.5 | 4.5 | 1.2 | 0.0 |
| 6-8 | **1000** | 2.0-3.0 | 3.0-4.0 | 1.0-1.4 | 0.4-1.2 |
| 9-12 | **1500** | 2.0 | 3.0 | 1.6-2.0 | 1.6-2.0 |

---

## 6. Reward 設計 — NavRL-Ground v4

> R_t = Σ f_i(s,a) × w_i × dt, dt=0.2s
> 完整數學公式見 `docs/REWARD_DESIGN_V4.md`

### 6.1 獎勵項（基礎權重，由課程動態覆蓋）

| # | 項目 | Weight | Per-step | 數學定義 | 類型 |
|---|------|:---:|:---:|---------|------|
| 1 | reaching_goal | **500** | 100.0 | 1{dist-r_body < 0.35m} | Terminal |
| 2 | goal_velocity | 2.0* | 0.40 | clamp(v_toward/v_max, -1, 1) | Per-step |
| 3 | goal_progress | 3.0* | 0.60 | clamp(d_{t-1} - d_t, -0.25, 0.25) | Per-step |
| 4 | static_safety | 2.0* | 0.40 | mean(ln(c_j)) + front_block | Per-step |
| 5 | dynamic_safety | 2.0* | 0.40 | log_clearance - closing_risk | Per-step |
| 6 | smoothness | -0.1 | 0.02 | Δa² + Δω² | Per-step |
| 7 | collision_ground | -50 | 10.0 | 1{LiDAR_min ≤ 0.45m} | Terminal |
| 8 | ~~alive~~ | **0** | — | 已移除：r_vel + γ折扣已提供時間壓力 | — |

*由 goal_first_v2 課程動態覆蓋（見 5.6 節）

**設計核心:** 無 alive reward — NavRL 用 r_vel 鼓勵高速朝目標，γ<1 產生時間壓力。
**gate 設定:** use_soft_gate=False, use_soft_scale=False（v4 無安全門控/縮放）

### 6.2 Terminal 條件

| 終止條件 | 門檻 | Reward | Bootstrap |
|---------|------|--------|-----------|
| goal_reached | dist < 0.35m | **+100** | No |
| collision | LiDAR_min ≤ 0.45m | **-10** | No |
| wall_collision | AABB < 0.45m | 無 | No |
| physics_explosion | vel > 10 或 NaN | 無 | No |
| timeout | episode_length_s | 無 | **Yes** |

---

## 7. 終止條件

| 條件 | 門檻 | 類型 | Bootstrap |
|------|------|------|-----------|
| time_out | episode_length_s (45~90s) | Truncation | Yes |
| goal_reached | dist < 0.35m | Success | No |
| collision_occurred | LiDAR 2D min ≤ 0.45m | Failure | No |
| wall_collision | AABB dist < 0.45m | Failure (safety net) | No |
| robot_tipped_over | Z-axis < 0.5 | Failure | No |
| physics_explosion | vel > 10 m/s 或 NaN | Failure | No |

---

## 8. WandB 指標完整清單

> 對應 WandB dashboard 分組。weight=0 的 reward term 在 v4 中不會顯示。
> 記錄頻率: 每 128 timesteps（rollout flush 時）一次。

### 8.1 `perf` (5)

來源: `WandBSequentialTrainer._get_task_metrics()` → `wandb.log()`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `perf/success_rate` | 全局成功率（到達目標/總 episode） | 0.3→0.85 |
| `perf/collision_rate` | 全局碰撞率 | 0.5→0.05 |
| `perf/timeout_rate` | 全局超時率 | 0.3→0.10 |
| `perf/total_episodes` | 累計 episode 數 | 持續增長 |
| `perf/episode_length` | 平均 episode 長度（步數） | 30→50 |

### 8.2 `behavior` (13)

來源: `AblationMetricsLogger.get_and_reset()` → `wandb.log()`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `behavior/stuck_events` | 卡住事件數（連續 ≥5 步 v<0.02） | <1000 |
| `behavior/freeze_ratio` | 靜止步數佔比 (v<0.02) | <0.10 |
| `behavior/oscillation` | v_toward 符號翻轉頻率 | <0.3 |
| `behavior/retreat_ratio` | 近障礙時後退佔比 | 0.1→0.3 |
| `behavior/progress_near_obstacle` | 近障礙時每步進度 (m) | >0 |
| `behavior/goal_velocity_near_obstacle` | 近障礙時朝目標速度 (m/s) | >0 |
| `behavior/obstacle_distance_avg` | 全局平均安全距離 (m) | 1.0→2.0 |
| `behavior/obstacle_distance_min` | rollout 最小安全距離 (m) | >0.45 |
| `behavior/danger_zone_ratio` | 危險區 (d_safe<0.8m) 步數佔比 | <0.15 |
| `behavior/speed_near_obstacle` | 近障礙時平均速度 (m/s) | 0.1→0.4 |
| `behavior/front_clearance` | 前方淨空距離 (m) | 1.0→2.0 |
| `behavior/avg_episode_length` | 平均 episode 步數 | 30→50 |
| `behavior/collision_episode_length` | 碰撞 episode 平均步數 | <avg |

### 8.3 `train` (6)

來源: `WandBSequentialTrainer` + `ModuleEntropyMonitor`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `train/fps` | 每秒 timesteps | 2.5→3.5 |
| `train/elapsed_time` | 訓練經過時間 (秒) | 持續增長 |
| `train/timestep` | 當前全局 timestep | 0→234,375 |
| `train/module_entropy` | log10(actor_grad / critic_grad) | -1→+1（≈0=平衡） |
| `train/actor_grad_norm` | Actor 梯度 L2 norm | 0.2→0.5 |
| `train/critic_grad_norm` | Critic 梯度 L2 norm | 0.3→0.6 |

### 8.4 `Loss` (3)

來源: SKRL PPO agent tracking_data

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Loss / Policy loss` | PPO clip surrogate loss | -0.05→-0.005 |
| `Loss / Value loss` | MSE(V_pred, returns) | 下降趨勢 |
| `Loss / Entropy loss` | -entropy_coeff × mean(entropy) | -0.04→-0.06 |

### 8.5 `Termination` (6)

來源: TerminationManager → `Episode_Termination/{name}` → `Termination / {name}`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Termination / goal_reached` | 成功到達目標的比例 | 0.3→0.85 |
| `Termination / collision` | LiDAR 碰撞終止比例 | 0.5→0.05 |
| `Termination / wall_collision` | 牆壁碰撞終止比例 | ≈0 |
| `Termination / time_out` | 超時終止比例 | 0.3→0.10 |
| `Termination / physics_explosion` | 物理引擎爆炸（>0=bug） | ≈0 |
| `Termination / robot_tipped_over` | 機器人翻倒 | ≈0 |

### 8.6 `Curriculum` (23)

來源: `goal_obstacle_curriculum()` → `infos["log"]` → `Curriculum / {leaf}`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Curriculum / stage` | 當前階段 (1-12) | 1→8+ |
| `Curriculum / success_rate` | 滑動窗口 SR | 隨 stage 變化 |
| `Curriculum / collision_rate` | 滑動窗口 CR | <0.35 |
| `Curriculum / timeout_rate` | 滑動窗口 TO | <0.25 |
| `Curriculum / dynamic_sr` | 僅動態場景 SR（Stage 6+） | 0→0.6 |
| `Curriculum / sr_gap` | SR - 升級門檻（>0=達標） | 升級時≈0 |
| `Curriculum / cr_gap` | CR 門檻 - 當前 CR | >0=達標 |
| `Curriculum / to_gap` | TO 門檻 - 當前 TO | >0=達標 |
| `Curriculum / upgrade_pass_count` | 連續達標次數 (/5) | 0→5 循環 |
| `Curriculum / upgrade_sr_target` | 當前 stage 升級 SR 門檻 | 0.60→0.85 |
| `Curriculum / upgrade_cr_target` | 當前 stage 升級 CR 門檻 | 0.35→0.45 |
| `Curriculum / upgrade_to_target` | 當前 stage 升級 TO 門檻 | 0.20→0.35 |
| `Curriculum / window_fill` | 窗口填充率 | 0→1.0 |
| `Curriculum / approx_rollout_cycles` | 階段內 PPO updates | 持續增長 |
| `Curriculum / num_goals` | 當前目標數 | 7→1 |
| `Curriculum / num_obstacles_static` | 靜態障礙物數 | 0→10 |
| `Curriculum / num_obstacles_dynamic` | 動態障礙物數 | 0→10 |
| `Curriculum / min_walls` | 牆壁下限 | 0→8 |
| `Curriculum / max_walls` | 牆壁上限 | 1→8 |
| `Curriculum / gamma` | 折扣因子 | 0.990→0.998 |
| `Curriculum / episode_length_s` | Episode 長度 (秒) | 45→90 |
| `Curriculum / num_episodes` | 累計 episode 數 | 持續增長 |
| `Curriculum / stage_episodes` | 當前 stage episode 數 | 升級後歸零 |

### 8.7 `Reward` (v3: 15, v4: ~13)

來源: RewardManager → `Episode_Reward/{name}` → `Reward / {name}`
值 = episode_sum / max_episode_length_s（每秒平均 reward）
v4 過濾: weight=0 的 term 不送 WandB（val==0.0 跳過）

**Per-term reward（v4 活躍項）:**

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Reward / reaching_goal` | 到達目標獎勵（稀疏） | 0.1→0.5 |
| `Reward / goal_velocity` | 朝目標速度獎勵 | 0.05→0.20 |
| `Reward / goal_progress` | 距離縮減 PBRS | 0.01→0.05 |
| `Reward / static_safety` | LiDAR 靜態安全 | 0.05→0.15 |
| `Reward / dynamic_safety` | 動態障礙物安全（Stage 6+） | 0→0.10 |
| `Reward / smoothness` | 控制平滑懲罰（負值） | -0.01→-0.005 |
| `Reward / collision_ground` | 碰撞懲罰（負值） | -0.15→-0.02 |

**Aggregate reward:**

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Reward / Total reward (max)` | 最高單 episode reward | >80 |
| `Reward / Total reward (mean)` | 平均 episode reward | 15→40 |
| `Reward / Total reward (min)` | 最低單 episode reward | >-15 |
| `Reward / Instantaneous reward (max)` | 單步最高 reward | ~20 (觸發 goal) |
| `Reward / Instantaneous reward (mean)` | 每步平均 reward | 0.3→0.8 |
| `Reward / Instantaneous reward (min)` | 單步最低 reward | ~-10 (碰撞) |

**v3 額外顯示（v4 已過濾, weight=0）:**
alive, acceleration_penalty, angular_velocity_penalty, collision_terminal, near_obstacle_penalty, potential_progress, time_penalty, velocity_too_low

### 8.8 `eval` (6)

來源: `WandBSequentialTrainer._log_per_env_type_metrics()`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `eval/success_rate_empty` | 空曠場景 SR | 0.8→0.95 |
| `eval/collision_rate_empty` | 空曠場景 CR | <0.05 |
| `eval/timeout_rate_empty` | 空曠場景 TO | <0.10 |
| `eval/success_rate_static` | 靜態障礙場景 SR | 0.3→0.70 |
| `eval/collision_rate_static` | 靜態障礙場景 CR | 0.5→0.15 |
| `eval/timeout_rate_static` | 靜態障礙場景 TO | 0.2→0.15 |

> dynamic 場景指標（Stage 6+ 才出現）: `eval/{success,collision,timeout}_rate_dynamic`

### 8.9 `robot` (9)

來源: `TrainingDebugLogger.get_and_reset()`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `robot/speed_mean` | 平均 2D 速度 (m/s) | 0.2→0.6 |
| `robot/speed_max` | 最大 2D 速度 (m/s) | ~1.0 |
| `robot/angular_vel_mean` | 平均 \|ω_z\| (rad/s) | 0.2→0.4 |
| `robot/angular_vel_max` | 最大 \|ω_z\| (rad/s) | ~0.785 |
| `robot/progress_per_step` | 每步接近目標 (m) | 0.02→0.10 |
| `robot/min_obstacle_dist_mean` | 平均最近障礙距離 (m) | 1.0→2.0 |
| `robot/min_obstacle_dist_min` | 最小障礙距離 (m) | >0.45 |
| `robot/lidar_min_raw` | LiDAR 最小距離平均 (m) | 0.3→0.5 |
| `robot/lidar_no_hit_ratio` | 無回波 ray 比例 | 0.2→0.3 |

### 8.10 `action` (4)

來源: `TrainingDebugLogger.get_and_reset()`

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `action/raw_mean` | 策略輸出原始值平均 | ≈0 |
| `action/raw_std` | 策略輸出標準差 | 5→8 |
| `action/accel_scatter` | 線加速度 vs 角加速度散佈圖 | — |
| `action/speed_scatter` | 線速度 vs 角速度散佈圖 | — |

### 8.11 `Policy` (1) / `Learning` (1)

來源: SKRL PPO agent tracking_data

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Policy / Standard deviation` | 策略標準差 | N/A（離散動作=nan） |
| `Learning / Learning rate` | 當前 LR | 1e-4→3e-5 |

### 8.12 `Episode` (3)

來源: SKRL SequentialTrainer

| WandB Key | 物理意義 | 期望值 |
|-----------|---------|:---:|
| `Episode / Total timesteps (max)` | 最長 episode 步數 | ≤max_steps |
| `Episode / Total timesteps (mean)` | 平均 episode 步數 | 30→50 |
| `Episode / Total timesteps (min)` | 最短 episode 步數 | ≥1 |

### 8.13 `System` (21)

來源: WandB 自動收集（GPU/CPU/Memory/Network）

| WandB Key 範例 | 物理意義 |
|-----------|---------|
| `system/gpu.0.gpu` | GPU 使用率 (%) |
| `system/gpu.0.memory` | GPU 記憶體使用率 (%) |
| `system/gpu.0.temp` | GPU 溫度 (°C) |
| `system/gpu.0.powerWatts` | GPU 功耗 (W) |
| `system/cpu` | CPU 使用率 (%) |
| `system/memory` | 系統記憶體使用率 (%) |
| `system/disk.*` | 磁碟 I/O |
| `system/network.*` | 網路 I/O |
| `system/proc.memory.*` | 程序記憶體 |

> 共 ~21 項，由 WandB agent 自動採集，不需手動設定。

---

## 附錄 A: 關鍵檔案路徑

```
source/isaaclab_tasks/.../charge_skrl/
├── __init__.py                              # Task registration
├── cfg/
│   ├── charge_cfg.py                        # Robot USD config
│   ├── charge_env_cfg_vlp16.py              # Base VLP16 config
│   └── charge_env_cfg_vlp16_curriculum.py   # Curriculum + NavRL + Ground configs
├── mdp/
│   ├── rewards/
│   │   ├── navrl_rewards.py                 # NavRL dense rewards (current)
│   │   ├── navrl_ground_rewards.py          # NavRL-Ground v1 rewards (new)
│   │   ├── potential_based_rewards.py       # PBRS + collision
│   │   ├── smoothness_rewards.py            # Deadzone accel + context-aware ω
│   │   └── gap_rewards.py                   # Gap-seeking rewards
│   ├── observations/obs_functions.py        # 139D observation pipeline
│   ├── actions/
│   │   ├── discrete_differential_drive.py   # MultiDiscrete([19,19]) action
│   │   └── safety_shield.py                # Safety shield action
│   ├── terminations/
│   │   ├── goal.py                         # Goal reached + timeout
│   │   └── robot_state.py                  # Tipped/flying/explosion/wall collision
│   ├── events/
│   │   ├── mixed_parallel.py               # Obstacle randomization
│   │   ├── walls.py                        # Per-env wall randomization
│   │   └── reset.py                        # Safe position reset
│   └── wall_layout.py                      # Wall geometry + GPU queries
├── curriculum/goal_obstacle_curriculum.py   # 8-stage curriculum
├── goal_command.py                         # Multi-goal command
├── multi_goal_command.py                   # Multi-goal extension
├── domain_randomization/dr_events.py       # Physics/sensor DR
└── agents/skrl_ppo_cfg_vlp16.yaml          # PPO hyperparameters

scripts/reinforcement_learning/skrl/
├── train_charge_ac.py                      # Main training entry
├── wandb_trainer.py                        # WandB integration
├── vlp16_models.py                         # 3-branch network
├── console_summary.py                      # Terminal summary
└── diagnostics/ablation_metrics.py         # 15 behavioral metrics
```

## 附錄 B: Git 資訊

```bash
git remote: git@github.com:me0608623/charge_skrl.git
git branch: abl
git log --oneline -5  # 查看最近 commits
```
