# Charge-NavRL — 差速移動機器人安全導航訓練框架

> **基於 Isaac Lab + SKRL 的 RNN 導航策略訓練：3D LiDAR 感知 × 課程學習 × Sim-to-Real 域隨機化**

本 repo（`charge_navrl`）收錄 Charge 差速底盤機器人的完整訓練程式碼、網路架構、課程設計、
獎勵設計，以及對齊真實 **Velodyne VLP-16** 的雜訊模型與域隨機化。目標是在模擬中訓練出可直接
部署到真機（rover_rl 車端）的 RNN 避障導航策略。

> ⚠️ **訓練好的模型權重（`logs/**/*.pt`）不納入 git**（`.gitignore` 排除 `logs/`、`runs/`、`outputs/`），
> 需另外經由 WandB artifact 或 rsync 傳遞。本 repo 只保證「程式碼 + 設定」可完整重現。

---

## 目錄

1. [快速開始](#快速開始)
2. [訓練架構總覽](#訓練架構總覽)
3. [網路架構](#網路架構)
4. [觀測與動作空間](#觀測與動作空間)
5. [VLP-16 LiDAR 雜訊模型（TLNI）](#vlp-16-lidar-雜訊模型tlni)
6. [域隨機化（Domain Randomization）](#域隨機化domain-randomization)
7. [課程學習（SA1–SA8）](#課程學習sa1sa8)
8. [獎勵設計](#獎勵設計)
9. [版本演進（v2 / v3 / v3c / v3f）](#版本演進)
10. [關鍵檔案地圖](#關鍵檔案地圖)

---

## 快速開始

```bash
# 環境
conda activate env_isaaclab          # Python 3.11 + Isaac Sim 5.1 + CUDA 13.0

# 從頭訓練 SA1 v3（乾淨 baseline）
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa1_v3 --headless \
  --run_name sa1_v3_ne1024_s42

# 接續訓練（v3f-react，從 SA4 ckpt 銜接 SA5 反應訓練）
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa5_v3f --headless \
  --checkpoint logs/rnn_car/sa4_v3f_ne1024_s42/checkpoint_420000.pt \
  --timesteps 18000 --run_name sa5_v3f_react_ne1024_s42

# 回放 / 評估（GUI 啟動器）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/play_launcher.py
```

> 所有實驗皆以 `--experiment_config <name>` 切換 variant，設定檔位於
> `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/*.yaml`。

---

## 訓練架構總覽

| 項目 | 設定 | 來源 |
|------|------|------|
| **框架** | Isaac Lab（manager-based env）+ SKRL 風格自訂 trainer | `train/train_rnn_car_wdclip.py` |
| **演算法** | A2C（預設，單 epoch full-batch）/ PPO clipping（`--use_ppo`） | 同上 |
| **優勢估計** | GAE，`λ = 0.95`，`δ = r + γ·V(s′) − V(s)` | `--gae_lambda 0.95` |
| **折扣因子** | `γ` 依 stage 設定（v3 約 0.984，課程後期 → 0.99x） | curriculum phase config |
| **雙 optimizer** | RL head（`lr=2e-4`）與 RNN/aux（`lr=5e-4`）分開更新 | `train_rnn_car_wdclip.py` |
| **梯度裁剪** | WD 風格 actor/critic 分離 clip（actor 8.0 / critic 30.0），主 clip 1.0 | `--wd_actor_update_clip / --wd_critic_update_clip` |
| **損失夾緊** | `policy_loss_clamp=20.0`、`vf_term_clamp=8.0`、`vf_coeff=0.025` | 同上 |
| **物理步長** | `dt = 0.2 s`（`decimation=20 × sim.dt=0.01`），Episode 45 s | `cfg/charge_env_cfg_vlp16.py:757-760` |
| **並行環境** | 典型 `num_envs=1024`（安全上限 4096，超過易觸 PhysX OOM） | — |

### Loss 結構（雙 loss）

```
總更新分兩條獨立路徑（WD module-connected 設計）：

  RL loss   = policy_loss(A2C/PPO surrogate) + vf_coeff · value_loss
              └─ 只更新 policy_head + value_head（輸入為 detach 後的 RNN 特徵）

  Aux loss  = Σ log(clamp(|pred − target|, 0.01)) · w_dim
              └─ 只更新 RNN cell + predict_head
              └─ target = 最近 2 個障礙的 body-frame (x, y, dist) + timestep（7D 特權幾何）
```

`.detach()` 邊界讓 RL 梯度不流入 RNN；RNN 只由 auxiliary 預測任務訓練。

---

## 網路架構

檔案：`scripts/reinforcement_learning/skrl/models/modular_rnn_models.py`

```
obs (79D / 83D)
   │
   ├─ LiDAR [6:78] (72D) ──► Conv1d(1→32,k5) ► Conv1d(32→64,k5,s2) ► Conv1d(64→64,k3,s2)
   │                          ► Flatten ► Linear(1152→64) ► LayerNorm           ─┐
   │                          (Conv1d 採 circular padding，對應 360° 環狀掃描)      │ concat
   ├─ State [ego4 + goal2 + time1 (+act_hist4)] ──► MLP(→32) ► LayerNorm ────────┘ → 96D
   │
   └────────────► PreprocessRNN：FC(96→48) ► RNN(48→30, relu) ► concat ► FC(→12)
                     │                                                    │
                     ├─► predict_head: Linear(12→7)  ← auxiliary 目標（無末端 ReLU）
                     └─► 12D preprocess feature（detach 給 RL head）
                                   │
            obs(79/83D) ⊕ preprocess(12D) = 91/95D
                     ├─► PolicyHead  MLP[256,256,256,512] ► 38 logits = 19(accel) + 19(ω)
                     └─► ValueHead   MLP[256,256,256,512,512] ► 1（可選 asymmetric privileged 分支）
```

| 元件 | 設定 |
|------|------|
| RNN 類型 | Vanilla RNN（`nonlinearity='relu'`），可切 GRU；`hidden_dim=30`（對齊 WarpDrive memory_dim） |
| Feature extractor | LiDAR Conv1d 分支（64D）+ State MLP 分支（32D）= 96D |
| Preprocess 維度 | 12D（`--preprocess_dim 12`），FC 前置 48D |
| Policy head | MultiDiscrete，38 logits（19×19，中心對稱 idx 9 = 零動作） |
| Value head | 預設 symmetric；asymmetric critic 已實作（priv 分支預設權重歸零、未啟用） |
| Auxiliary | predict_head 預測 7D 障礙幾何；末端 ReLU 已移除（修正 dead-ReLU 致輸出恆 0 的 bug） |

---

## 觀測與動作空間

### 觀測（Policy 實際使用 79D / 83D）

```
EGO       [0:4]   accel(1) + linear vel(1) + omega(1) + body radius(1)
GOAL      [4:6]   下一 waypoint (x, y)，body-frame
LIDAR     [6:78]  72 bins（5° 角解析度），距離歸一化至 [0,1]
TIME      [78:79] episode 剩餘時間比例
ACT_HIST  [79:83] （v3c/v3d/v3g）過去 2 步動作 (a_norm, ω_norm) × 2
```

| 版本 | 維度 | 說明 |
|------|------|------|
| v3 / v3b / v3f | **79D** | ego + goal + lidar + time（無動作歷史） |
| v3c / v3d / v3g | **83D** | + action history（4D） |

> 環境實際輸出 139D（含 obstacle buffer 60D），但訓練只取前 79/83D，刻意避開
> obstacles-60D 的維度詛咒（該欄位在部署時不可得，詳見設計筆記）。

### 動作（MultiDiscrete([19, 19])）

檔案：`.../charge_skrl/mdp/actions/discrete_differential_drive.py`

| 物理量 | 數值 |
|--------|------|
| 線速度 `v` | ∈ [−1.0, +1.0] m/s（允許倒車，`reverse_velocity_scale` 可縮放） |
| 線加速度 `a_max` | 0.5 m/s² |
| 角速度 `ω_max` | 0.25π ≈ 0.785 rad/s |
| 角加速度 `α_max` | 3.0 rad/s²（限制 Δω，抗「舞龍舞獅」極限環） |
| 步長 `dt` | 0.2 s |

索引 → 比例：`ratio = (idx − 9) / 9 ∈ [−1, +1]`，並以**動態加速度邊界**保證
`v_next = clamp(v + a·dt, −v_max, +v_max)`。

---

## VLP-16 LiDAR 雜訊模型（TLNI）

> **TLNI = Three-Layer Noise Injection**，把雜訊分成「射線級 → bin 級 → 每-episode DR 採樣」三層，
> 以盡量逼近真實 Velodyne VLP-16 的退化行為。核心實作在 `obs_functions.py`，預設值在
> `experiment_config.py`，各 stage 覆寫在 `configs/wd_sa*_v3*.yaml`。

### 硬體與 2D 投影

| 項目 | 數值 | 來源 |
|------|------|------|
| 感測器 | Velodyne VLP-16（16 通道） | `cfg/charge_env_cfg_vlp16.py` |
| 水平 FOV / 解析度 | ±180° / 1°（360 rays） | 同上 |
| 垂直 FOV | −15° ~ +15° | 同上 |
| 最大距離 `r_max` | 20.0 m | `obs_functions.py:53` |
| 安裝高度 | z = 1.43 m | cfg |
| 投影輸出 | 16ch × 360 rays → **72 bins**（5°，先 horizontal-min 再 vertical-min） | `obs_functions.py:85-91` |
| 車體半徑 `r_robot` | 0.3 m（距離扣除後 clamp≥0） | `obs_functions.py:54` |
| **最小偵測距離 `r_min`** | **0.25 m** | `obs_functions.py`（v3 呼叫值） |
| Z 軸過濾 `z_filter` | 0.5 m（移除 z=−10 鬼影命中） | obs_functions |
| 歸一化 | `(d − r_robot) / r_max` → clamp [0, 1] | `obs_functions.py:91` |

#### `r_min = 0.25 m` 的物理推導

```
實測 VLP-16 偵測行人中心的最近距離：
  ├─ 光學中心 → 行人中心（表面）：0.2 m   ← 實測（2026-06-08）
  ├─ VLP-16 物理半徑（∅103 mm）：0.0515 m
  └─ 合計 0.2515 m → 向上取整保守設為 0.25 m
```

### Layer 1 — 射線級雜訊（min-pool 之前）

| 雜訊 | 模型 / 預設 | 說明 |
|------|------------|------|
| 距離相依高斯 | `σ(r) = 0.00036 · r`（SA4 校準值） | 從 VLP-16 實測標定，距離越遠越大 |
| 軟目標固定 σ | 基礎 0.0，SA5 採 [0.001, 0.010] | 人/衣物等低反射率目標；以 RSS 合併 `σ_total = √(σ_r² + σ_soft²)` |
| 射線丟失（holes） | 基礎 0.20（真機實測 ~21%） | 隨機射線 → `r_max` |
| 幽靈點（distractor） | 0.002，距離 U(0.2, 2.0) m | 多路徑反射 / 感測缺陷 |
| 系統性距離偏差 | `bias(d) = k·d + b`，`k ∈ [−0.005, 0.005] m/m`，`b ∈ [−0.012, 0.012] m` | ToF 時鐘偏移 + 常數偏移，每 episode 採樣 |
| 每-通道偏差（per-ring） | 16 通道固定 offset ±22.7 mm（白牆校準實測） | VLP-16 每線安裝偏差，DR 再 ×[0.5, 1.5] |

VLP-16 16 通道實測 per-ring bias（m）：
```
[-0.0041, +0.0019, -0.0031, -0.0093, -0.0071, +0.0083, +0.0140, +0.0027,
 +0.0157, +0.0167, +0.0123, +0.0068, -0.0113, -0.0227, -0.0137, -0.0196]
```

### Layer 2 — bin 級雜訊（min-pool 之後）

| 雜訊 | 預設 | 說明 |
|------|------|------|
| 觀測高斯雜訊 | σ = 0.005 | ObsTerm 後對 72D 向量加 noise |
| Block dropout | 0.0（預設關） | 連續扇形遮擋（3–8 bins，15°–40°），用於硬目標模擬 |

### Layer 3 — 每-episode DR 採樣（reset 時重抽）

每個 env 在 `episode_length_buf == 0` 時，從區間 uniform 重新採樣，使每條 episode 的退化程度不同：

| 參數 | SA3 範例範圍 | SA5 範例範圍 |
|------|-------------|-------------|
| `displacement_std_per_meter_dr` | [0.00020, 0.00045] | [0.00015, 0.00030] |
| `displacement_std_soft_dr` | [0.002, 0.015] | [0.001, 0.010] |
| `hole_rate_dr` | [0.08, 0.20] | [0.05, 0.15] |

> 由低難度（SA1）到高難度（SA5+）逐步加重雜訊；裝置間變異全靠 Layer 3 的 DR 區間覆蓋。

---

## 域隨機化（Domain Randomization）

每 episode reset 重新採樣，提升 Sim-to-Real 韌性。預設值見 `configs/wd_sa1_v3.yaml`。

| 類別 | 參數 | 範圍 | 物理意義 |
|------|------|------|---------|
| **物理** | 質量縮放 | [0.85, 1.15] | ±15% 載重（電池 / 貨物） |
| | 地面摩擦 | [0.7, 1.3] | ±30% 地面材質差異 |
| | 質心偏移 | ±0.05 m（XY） | 組裝誤差 |
| **擾動** | 風力 | [0.0, 3.0] N | 環境側風 |
| | 推力 | [5.0, 20.0] N，每步 10% 機率 | 行人碰撞 / 隨機衝擊 |
| **致動器** | 動作延遲 | [0, 1] step（0–200 ms） | 通訊 + 處理延遲 |
| | 速度縮放 | [0.97, 1.03] | ±3% 馬達效率差異 |
| | 馬達滯後 | 0.5（一階低通係數） | 致動慣性 |
| **觀測延遲** | `obs_delay_steps` | **[0, 0]（已關閉）** | 見下方注意事項 |

### ⚠️ 致動延遲 vs 觀測延遲（關鍵踩雷）

真機實測：ω 通道致動死時間 **~200 ms**（互相關，兩趟一致）。由於
`control_dt = 200 ms`，`delay / control_dt ≈ 1.0` 落在相位裕度最差區 → 真機「舞龍舞獅」極限環的根因。

- ✅ **保留**：`actuator_delay`（致動端延遲）— 這才是真機極限環的對症建模。
- ❌ **關閉**：`obs_delay_steps`（觀測端延遲）— 曾誤把藥方植成觀測延遲，導致 policy 對「過時觀測」
  過度反應而**滿舵失控**（2026-06-20 定案修正，空曠場景也飽和即為病態訊號）。

部署端必須 `CHARGE_USE_ACT_HIST=0`（79D 路線），且回放使用 **deterministic**（非隨機探索）以避免
entropy 污染抽動指標。

---

## 課程學習（SA1–SA8）

**SA = Single-Agent**（相對 WarpDrive 原版 MARL，本線為單機訓練）。
逐步加大 goal 距離、障礙密度、牆面數與碰撞懲罰。

| Stage | 名稱 | Goals | Static/Dynamic | Walls | 碰撞懲罰 | Episode |
|-------|------|-------|----------------|-------|---------|---------|
| SA1 | nav_bootstrap | 多 | 低 | 0–1 | −5 | 短 |
| SA2 | nav_static | 多 | 低 | 0–1 | −8 | 短 |
| SA3 | walls_crossing | 中 | 中 | 1–2 | −12 | 中 |
| SA4 | spatial_plan | 少 | 中 | 1–2 | −15 | 中 |
| SA5 | endurance | 中 | 中 | 2–3 | −15 | 長 |
| SA6 | dense_avoid | 少 | 高 | 2–3 | −25 | 長 |
| SA7 | high_pressure | 少 | 高 | 2–3 | −50 | 很長 |
| SA8 | final | 1 | 高 | 2–3 | −100 | 最長 |

> 上表為相對趨勢；各 stage 精確的 goals/障礙數/γ/penalty/episode 長度以
> `curriculum/phases/wd_single_agent_v3.py` 內 `STAGES[i]` 定義為準。

### 升降階機制（`goal_obstacle_curriculum.py`）

- **升階**：連續 5 個檢查窗同時 `SR > upgrade_sr`、`CR ≤ upgrade_max_cr`、`TO ≤ upgrade_max_to`
  （動態 stage 另需 `Dynamic SR > upgrade_min_dyn_sr`）。
- **降階**：`SR < downgrade_sr` 且（`CR ≥ downgrade_min_cr` 或 `TO ≥ downgrade_min_to`）。

> SR / CR / TO 一律以「已完成 episodes」正規化，不報 raw %。

---

## 獎勵設計

檔案：`scripts/reinforcement_learning/skrl/rnn_car_wdclean/rewards.py`（`compute_wd_charge_reward`）

| 項目 | 型式 | v3 預設 | 說明 |
|------|------|---------|------|
| `goal_reward` | 稀疏（到達時） | +40.0 | 主導訊號 |
| `penalty_hit` | 稀疏（碰撞時） | −5 → −100 | 依 stage 遞增（牆 / 障礙分開計） |
| `cost_operate` | 每步 | 0.03 | 動作成本，鼓勵高效 |
| `penalty_timeout` | 稀疏 | 0（SA6+ 才引入 ≈ `−penalty_hit/2`） | 逾時懲罰 |
| `penalty_smoothness` | 每步 | 0.005 | `−0.005·\|Δratio_ang\|`，抗單幀抽動 |
| `penalty_speed_near_obs` | 每步 | 0.8（僅 v3f-react SA5） | clearance-gated 減速：`−(w/fps)·p²·v_forward` |

> **重要教訓**：`penalty_smoothness` 雖抗單幀抖動，但過度懲罰 flip 會誘發「低頻 weave（sin 波）」，
> 因此 v3e/v3f 在課程後期將其調回 0；分析抽動須用 per-env p95，全 env 平均會掩蓋單幀現象。

---

## 版本演進

| 版本 | obs | 重點改動 |
|------|-----|---------|
| **v2** | — | 早期迭代（已作為 baseline 凍結） |
| **v3** | 79D | `r_min 0.9→0.25`（VLP-16 實測）+ 全 stage `penalty_smoothness=0.005` + 新 curriculum；**不能 resume v1/v2 ckpt** |
| **v3b** | 79D | 加 `obs_delay [0,1]` + 加倍 entropy（後證實 obs_delay 為病因，已撤） |
| **v3c** | 83D | action stacking（+4D obs）+ 顯式阻尼抗抽動；部署端用 `*_v3c.yaml` |
| **v3d** | 83D | action-history dropout 弱化 copy-shortcut（後發現 action_error 自我參照=內建振盪器） |
| **v3e** | 83D | `penalty_smoothness→0` + entropy floor，移除 sin 波主因 |
| **v3f** | 79D | 回到純 LiDAR baseline（移除 action history），SA4 加狹窄通道 |
| **v3f-react** | 79D | SA5 反應訓練補丁：`penalty_speed_near_obs` 對症「動態障礙晚反應撞擊」，由 SA4_v3f ckpt 銜接 |
| **v3g** | 83D | 實驗性變體 |

---

## 關鍵檔案地圖

```
scripts/reinforcement_learning/skrl/
├── train/
│   ├── train_rnn_car_wdclip.py        ← 主訓練入口（A2C/PPO + 雙 optimizer + WD clip）
│   └── train_launcher.py              ← 訓練 GUI 啟動器
├── models/
│   └── modular_rnn_models.py          ← RNN + Conv1d extractor + policy/value/predict head
├── play_eval/
│   ├── play_launcher.py               ← 回放/評估 GUI 啟動器
│   ├── play_rnn_car.py                ← 回放主程式（含 --jitter_eval）
│   └── rvo2_safety_filter.py          ← RVO2 安全濾波（部署側對照）
├── rnn_car_modular/
│   ├── experiment_config.py           ← LEGO 式實驗組合 + 雜訊預設
│   ├── configs/wd_sa*_v3*.yaml        ← 各 stage / variant 設定
│   └── rewards/wd_sparse.py
├── rnn_car_wdclean/rewards.py         ← compute_wd_charge_reward
└── tools/compare_noise_impact.py      ← 雜訊模型比較工具

source/isaaclab_tasks/.../charge_skrl/
├── cfg/charge_env_cfg_vlp16{,_curriculum}.py   ← VLP-16 感測器 + DR events
├── mdp/observations/obs_functions.py           ← TLNI 雜訊注入 + 72-bin 投影
├── mdp/actions/discrete_differential_drive.py  ← MultiDiscrete([19,19]) + 動力學
└── curriculum/
    ├── goal_obstacle_curriculum.py             ← 升降階邏輯
    └── phases/wd_single_agent_v3{,f_react,...}.py  ← SA1–SA8 stage 定義
```

---

> 文件以 2026-06-23 程式碼快照為準；精確的 per-stage 數值請以對應原始檔案為最終依據。
