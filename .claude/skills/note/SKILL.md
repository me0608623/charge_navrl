---
name: note
description: 將 Isaac Lab 訓練進度同步至 Obsidian — 建立訓練筆記（含獎勵設計、網路架構、課程學習）、流程畫布、並加入 RL 監控資料表
user-invocable: true
argument-hint: "[run_name 或留空自動偵測最新 run]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
  - Skill
---

# /note — Isaac Lab 訓練進度 → Obsidian

你是一名專精於 Isaac Lab 強化學習與 Obsidian 知識管理的自動化專家。

**使用方式**：`/note [run_name]`
- 若不帶參數，自動偵測 `logs/skrl/` 下最新修改的 run。
- `run_name` 為 run 資料夾名稱，例如 `rw_groundv4_goalfirstv2_full_seed1`。

---

## 執行步驟

### Step 1：偵測目標 Run

若 `$ARGUMENTS` 為空，自動偵測最新 run：

```bash
find /home/aa/IsaacLab/logs/skrl -mindepth 2 -maxdepth 2 -type d \
  -exec stat --format="%Y %n" {} \; | sort -rn | head -1 | awk '{print $2}'
```

確認後取得：
- `RUN_PATH`：完整路徑
- `RUN_NAME`：資料夾名稱（對應筆記命名）
- `TASK_NAME`：父資料夾名稱（task）

---

### Step 2：擷取所有訓練資料

#### 2-A：training_params YAML（`{RUN_PATH}/training_params_{RUN_NAME}.yaml`）

讀取以下所有區段：
- **Section 1 Basic**：`task`、`phase`、`seed`、`device`
- **Section 2 Env**：`num_envs`、`decimation`、`sim_dt`、`env_dt`、`episode_length_s`
- **Section 3 Action**：`max_linear_velocity`、`max_linear_accel`、`max_angular_vel`、`num_bins`
- **Section 4 PPO**：`rollouts`、`learning_epochs`、`mini_batches`、`discount_factor`、`lambda_gae`、`learning_rate`、`entropy_loss_scale`、`grad_norm_clip`、`ratio_clip`、`state_preprocessor`
- **Section 6 Rewards**：完整 `rewards:` dict，包含每個 term 的 `function`、`weight`、`category`、`meaning`、`params`
- **Section 8 Env Distribution**：`empty_ratio`、`static_ratio`、`dynamic_ratio`
- **Section 9 Model**：`policy.class`、`value.class`、`separate`
- **消融 CLI 參數**：`v_gate_mode`、`progress_gate_mode`、`use_gap_reward`、`gap_reward_type`、`gap_reward_weight`、`use_safety_shield`、`shield_mode`（若存在）

#### 2-B：debug_metrics.csv 最後 20 行

```bash
tail -20 "{RUN_PATH}/debug_metrics.csv"
```

從最後一行（最新時步）提取：

| 欄位 | CSV 欄名 |
|------|---------|
| 課程階段 | `Curriculum / stage` |
| 成功率 SR | `Curriculum / success_rate` |
| 碰撞率 CR | `Curriculum / collision_rate` |
| Dynamic SR | `Curriculum / dynamic_sr` |
| 升級 SR 門檻 | `Curriculum / upgrade_sr_target` |
| 升級 CR 門檻 | `Curriculum / upgrade_cr_target` |
| 通過計數 | `Curriculum / upgrade_pass_count` |
| 升級 TO 門檻 | `Curriculum / upgrade_to_target` |
| 總 episode 數 | `Curriculum / num_episodes` |
| 各獎勵 term 值 | `Reward / {term_name}` 所有欄 |
| 總訓練時步 | `train/timestep` |
| FPS | `train/fps` |
| 已訓練時間(s) | `train/elapsed_time` |
| Freeze ratio | `behavior/freeze_ratio` |
| Oscillation | `behavior/oscillation` |
| 平均障礙距離 | `behavior/obstacle_distance_avg` |
| Shield rate | `ablation/shield_rate`（若存在） |
| eval SR empty | `eval/success_rate_empty` |
| eval SR static | `eval/success_rate_static` |

計算 SR 趨勢（最後 10 行 SR 均值 vs 前 10 行 SR 均值 → ↑上升/↓下降/→持平）。

---

### Step 3：建立 Obsidian 訓練筆記

呼叫 `obsidian-cli` skill，在 vault 的 `RL訓練` 資料夾建立筆記 `RL訓練/{RUN_NAME}.md`。

使用 `obsidian-markdown` skill 格式化，內容包含以下所有區段（繁體中文，專業語氣）：

```markdown
---
title: {RUN_NAME}
tags:
  - rl-training
  - isaac-lab
  - charge-nav
task: {TASK_NAME}
seed: {seed}
phase: {current_phase}
sr: {success_rate}
cr: {collision_rate}
timestep: {total_timestep}
updated: {YYYY-MM-DD HH:MM:SS}
status: 訓練中 / 已完成
---

# {RUN_NAME}

> [!info] 訓練快照
> **更新時間**：{timestamp}
> **Task**：`{TASK_NAME}`
> **Seed**：{seed} ｜ **Envs**：{num_envs} ｜ **時步**：{total_timestep:,}

## 消融設定

| 參數 | 值 |
|------|----|
| v_gate_mode | {v_gate_mode 或 baseline} |
| progress_gate_mode | {progress_gate_mode 或 baseline} |
| gap_reward | {use_gap_reward} ({gap_reward_type} w={gap_reward_weight}) |
| safety_shield | {use_safety_shield} ({shield_mode}) |

---

## 課程學習狀態

| 指標 | 當前值 | 門檻 |
|------|--------|------|
| **課程階段** | Phase {stage} / 8 | — |
| **成功率 SR** | {SR:.1%} | ≥ {sr_target:.0%} |
| **碰撞率 CR** | {CR:.1%} | ≤ {cr_target:.0%} |
| **Dynamic SR** | {dynamic_sr:.1%} | — |
| **升級計數** | {pass_count} / 5 連續達標 | — |
| **SR 趨勢** | {↑上升 / ↓下降 / →持平} | — |
| **總 Episodes** | {num_episodes:,} | — |

> [!{tip 若SR上升 / warning 若SR下降 / note 若持平}] SR 趨勢分析
> 最近 20 個 checkpoint：{SR_trend_description}

### 課程版本設計（{curriculum_version}）

<!-- 根據 training_params 的 curriculum_version 填入對應的 goal_first_v* 或 baseline_v* 設計 -->
<!-- 從 goal_obstacle_curriculum.py CURRICULUM_CONFIGS 讀取 -->

#### 升降級邏輯

| Phase | 名稱 | Goals | Static | Dynamic | Walls | γ | episode_s | 升級 SR | 升級 CR | 最低更新數 |
|-------|------|-------|--------|---------|-------|---|-----------|---------|---------|-----------|
<!-- 依 curriculum_version 填入完整 8 個 stage 的參數 -->
<!-- 當前所在 phase 用 **粗體** 標示 -->

#### Stage-dependent 獎勵權重（若有）

| Phase | goal_velocity | goal_progress | static_safety | dynamic_safety |
|-------|--------------|---------------|---------------|----------------|
<!-- 填入各 phase 的 reward_weights（若 curriculum 版本有定義） -->

---

## 獎勵設計

### 環境參數
- **env_dt**：{env_dt} s（decimation={decimation} × sim_dt={sim_dt}）
- **reward 計算**：`reward = func() × weight × dt`（WandB 值 = episode_sum / episode_length_s）

### 獎勵項目總表

| 名稱 | 函數 | weight | 類別 | 說明 |
|------|------|--------|------|------|
<!-- 從 training_params Section 6 rewards 逐一填入所有 term -->
<!-- 格式: | {term_name} | `{function}` | {weight} | {category} | {meaning} | -->

> [!note] 有效獎勵（weight ≠ 0）
> 列出所有 weight ≠ 0 的 term，並依類別分組：
> - **成功**：{reaching_goal weight}
> - **前進**：{goal_progress, goal_velocity weights}
> - **安全**：{static_safety, dynamic_safety, collision_ground weights}
> - **平滑**：{smoothness, acceleration_penalty, angular_velocity_penalty weights}

### 當前各項獎勵值（最新 checkpoint）

| 名稱 | 當前值 | 預期趨勢 |
|------|--------|---------|
<!-- 從 debug_metrics.csv Reward / {term_name} 欄填入最新值 -->
<!-- 預期趨勢來自 training_params 的 expected_trend 欄位 -->

### 終止條件

| 條件 | 函數 | time_out |
|------|------|---------|
<!-- 從 training_params Section 7 terminations 填入 -->

---

## 神經網路架構

### 觀測空間（139D Policy / 139D Critic — 對稱設計，無 privileged info）

| 區段 | 維度 | 內容 |
|------|------|------|
| ego | 4D | normalized_accel(1) + normalized_vel(1) + normalized_omega(1) + robot_radius(1) |
| goal | 2D | waypoint (x, y) in robot frame |
| LiDAR | 72D | 72-bin 掃描，body frame，單幀（無 frame stacking） |
| obstacles | 60D | Top-10 × 6D：(x, y, vx, vy, r, m)，body frame，LOS occlusion |
| time | 1D | remaining episode ratio |

### 特徵提取器（VLP16FeatureExtractor — 三分支）

```
Branch 1: LiDAR Conv1d
  [B, 72] → unsqueeze → [B, 1, 72]
  → Conv1d(1→32, k=5, p=2)  → ReLU  → [B, 32, 72]
  → Conv1d(32→64, k=5, s=2, p=2) → ReLU → [B, 64, 36]
  → Conv1d(64→64, k=3, s=2, p=1) → ReLU → [B, 64, 18]
  → AdaptiveMaxPool1d(1) → [B, 64]
  → Linear(64→64) → LayerNorm(64) → [B, 64]

Branch 2: Obstacle MLP（排列不變，MaxPool over objects）
  [B, 60] → reshape → [B, 10, 6]
  → Linear(6→32) → ReLU → Linear(32→32) → ReLU → [B, 10, 32]
  → max(dim=1) → [B, 32]
  → LayerNorm(32) → [B, 32]

Branch 3: State MLP（ego + goal + time = 7D）
  [B, 7] → Linear(7→32) → ReLU → Linear(32→32) → ReLU → [B, 32]
  → LayerNorm(32) → [B, 32]

Fusion: cat([Branch1, Branch2, Branch3]) = [B, 128]
```

### Policy（VLP16DiscretePolicy）

```
動作空間：MultiDiscrete([19, 19])
  = 19 bins 線加速度 × 19 bins 角速度，獨立 Categorical 採樣
  共 38 logits（vs 舊版 Discrete(361) = 361 logits）
  最大熵 = ln(19)+ln(19) ≈ 5.89 ≈ ln(361)（等效）

架構：
  VLP16FeatureExtractor → [B, 128]
  → Linear(128→128) → ReLU
  → Linear(128→38)  → [accel_logits(19), omega_logits(19)]
  → 兩組獨立 Categorical → MultiCategoricalMixin (reduction="sum")
```

### Critic（VLP16Value）

```
輸入：139D（與 policy 相同，對稱設計）
架構：
  VLP16FeatureExtractor → [B, 128]
  → Linear(128→128) → ReLU
  → Linear(128→1)   → scalar value
  → DeterministicMixin
```

### PPO 超參數

| 參數 | 值 |
|------|----|
| rollouts | {rollouts} |
| learning_epochs | {learning_epochs} |
| mini_batches | {mini_batches} |
| batch_size | {rollouts} × {num_envs} = {batch_size:,} |
| discount_factor γ | {discount_factor} |
| GAE λ | {lambda_gae} |
| learning_rate | {learning_rate} |
| entropy_loss_scale | {entropy_loss_scale} |
| grad_norm_clip | {grad_norm_clip} |
| ratio_clip | {ratio_clip} |
| state_preprocessor | {state_preprocessor} |
| LR scheduler | LinearLR ({start_factor}→{end_factor}, {total_iters} iters) |

---

## 行為診斷

| 指標 | 值 | 說明 |
|------|----|----|
| Freeze ratio | {freeze_ratio:.2%} | speed<0.02 步數佔比 |
| Oscillation | {oscillation:.3f} | v_toward 符號翻轉頻率 |
| 平均障礙距離 | {obstacle_distance_avg:.3f} m | |
| Retreat ratio | {retreat_ratio:.2%} | 近障礙時 v_toward<0 佔比 |
| Shield 介入率 | {shield_rate:.2%} | （Family 4 專用） |
| eval SR (empty) | {eval_sr_empty:.1%} | 空場景評估 |
| eval SR (static) | {eval_sr_static:.1%} | 靜態障礙評估 |

---

## 訓練效能

- **FPS**：{fps:,.0f}
- **已訓練時間**：{elapsed_h:.1f} 小時
- **總時步**：{total_timestep:,}（目標 {target_timesteps:,}）
- **進度**：{progress:.1%}

## 檢查點

- 最佳模型：`{RUN_PATH}/best_model/`
- 最新 checkpoint：`{RUN_PATH}/checkpoints/`

## 相關連結

- [[RL訓練監控]] — 整體資料表
- [[{TASK_NAME}]] — Task 說明（若存在）
```

---

### Step 4：建立訓練流程畫布

呼叫 `json-canvas` skill，在 vault 建立或更新 `RL訓練/{RUN_NAME}_canvas.canvas`。

畫布佈局（x/y 單位：像素）：

**節點群**：
1. **訓練設定節點**（左側 x=-300, y=200, 200×320）
   - 顯示消融參數 + PPO 關鍵超參 + 獎勵 weight ≠ 0 的 term

2. **Phase 1~8 節點**（主排 y=0, x = phase_idx × 250, 200×160）
   - 當前階段：橙色 `#EC9A29`
   - 已通過：綠色 `#4CAF50`
   - 未到達：灰色 `#9E9E9E`
   - 節點文字：`**Phase {n} — {stage_name}**\nGoals:{goals} S:{static} D:{dynamic}\nWalls:{min_walls}~{max_walls} γ:{gamma}\nSR門檻: ≥{upgrade_sr:.0%}`

3. **當前指標節點**（右側 x=2200, y=0, 240×200）
   - 顯示 SR/CR/Dynamic SR/timestep/趨勢

4. **獎勵設計節點**（左側 x=-300, y=-200, 200×280）
   - 列出各 reward term 的 weight（只列 ≠ 0 的）

5. **網路架構節點**（右側 x=2200, y=250, 240×200）
   - 三分支架構摘要：LiDAR Conv1d(64D) + ObsMLP(32D) + StateMLP(32D) → 128D → Policy/Value

**邊**：
- Phase N → Phase N+1：label = `SR≥{sr_target:.0%}, CR≤{cr_target:.0%}`
- 訓練設定 → Phase 1（無 label）
- Phase 8 → 當前指標（無 label）

---

### Step 5：加入 RL 訓練監控資料表

呼叫 `obsidian-bases` skill，在 vault 的 `RL訓練監控.base` 中操作（若不存在則建立）。

資料表欄位：
```
run_name (text), task (text), curriculum_version (text),
phase (number), sr (number), cr (number), dynamic_sr (number),
timestep (number), seed (number), num_envs (number),
v_gate_mode (text), progress_gate_mode (text),
use_gap_reward (checkbox), gap_type (text), gap_weight (number),
use_safety_shield (checkbox), shield_mode (text),
reaching_goal_w (number), goal_progress_w (number), goal_velocity_w (number),
static_safety_w (number), dynamic_safety_w (number),
freeze_ratio (number), oscillation (number),
fps (number), elapsed_h (number),
status (text), updated (date), note_link (text)
```

新增或更新（依 `run_name` 去重）此次 run 的記錄。

---

### Step 6：輸出摘要

```
✅ Obsidian 筆記已同步完成

📝 筆記路徑  ：RL訓練/{RUN_NAME}.md
🗺  畫布路徑  ：RL訓練/{RUN_NAME}_canvas.canvas
📊 資料表    ：RL訓練監控.base

─── 訓練快照 ──────────────────────
Run      ：{RUN_NAME}
Task     ：{TASK_NAME}
Curriculum：{curriculum_version}
Phase    ：{stage} / 8 — {stage_name}
SR       ：{sr:.1%}  （門檻 {sr_target:.0%}，趨勢 {trend}）
CR       ：{cr:.1%}  （門檻 {cr_target:.0%}）
Dynamic SR：{dynamic_sr:.1%}
時步     ：{timestep:,} / {target_timesteps:,} ({progress:.1%})
已訓練   ：{elapsed_h:.1f} 小時 @ {fps:,.0f} FPS
────────────────────────────────────
有效獎勵 term（weight≠0）：
  {term_name}: {weight} ...
網路架構 ：VLP16FeatureExtractor(3-branch) → 128D → Policy(38 logits) / Value
────────────────────────────────────
```

---

## 資料來源對照

| 資訊類型 | 來源檔案 |
|---------|---------|
| 獎勵設計 & 權重 | `training_params_{RUN_NAME}.yaml` Section 6 |
| 課程學習設計 | `training_params_{RUN_NAME}.yaml` + `curriculum/goal_obstacle_curriculum.py` |
| 網路架構 | `training_params_{RUN_NAME}.yaml` Section 9 + `scripts/.../vlp16_models.py` docstring |
| PPO 超參 | `training_params_{RUN_NAME}.yaml` Section 4 |
| 即時指標 | `debug_metrics.csv` 最後 20 行 |

---

## 錯誤處理

- 若 `debug_metrics.csv` 不存在 → 僅使用 YAML 建立基礎筆記，指標欄位標示 `N/A（尚未開始）`
- 若 Obsidian 未開啟 → 詢問 vault 路徑，改為直接寫入檔案系統
- 若找不到任何 run → 列出 `logs/skrl/` 下所有可用 run 供選擇
- 若 curriculum_version 不在 CURRICULUM_CONFIGS 中 → 以 YAML 內的原始 stage 資料填充，標注「自訂版本」
