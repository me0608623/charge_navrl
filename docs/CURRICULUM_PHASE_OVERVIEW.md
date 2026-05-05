# Curriculum Phase Overview — 各版本場景設計對照表

> 自動整理自 `goal_obstacle_curriculum.py` 的 `CURRICULUM_CONFIGS`
> 訓練時用 `--curriculum_version <name>` 切換

## 當前訓練使用: `warp_drive_single_agent_v1`（固定 SA1）

---

## 系統架構說明

```
┌─────────────────────────────────────────────────────────────────────────┐
│ charge_env_cfg_vlp16_curriculum.py                                       │
│                                                                         │
│  定義「場景骨架」:                                                        │
│    - 物理場景: 20×20m, 4 外牆, 8 internal wall slots, 100 obstacle slots │
│    - 觀測空間: 139D (ego+goal+LiDAR+obstacles+time)                     │
│    - 動作空間: MultiDiscrete [19,19]                                     │
│    - NavRL 獎勵結構 (但 train_rnn_car_wdclip.py 用自己的 WD sparse)       │
│    - 指向 curriculum function                                            │
└──────────────┬──────────────────────────────────────────────────────────┘
               │ runtime 動態修改 event/command 參數
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ goal_obstacle_curriculum.py                                              │
│                                                                         │
│  定義「各 Phase 場景參數」:                                                │
│    - 每個 curriculum_version 有 N 個 stage                               │
│    - 每個 stage 指定: goals, static_obs, dynamic_obs, walls, γ, etc.    │
│    - 升降階邏輯: SR/CR/TO 門檻                                            │
│    - 課程函式每次 env.reset 後被呼叫，動態改寫 events/commands             │
└──────────────┬──────────────────────────────────────────────────────────┘
               │ train_rnn_car_wdclip.py 額外覆蓋
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ train_rnn_car_wdclip.py 覆蓋層                                           │
│                                                                         │
│  1. Reward: 完全繞過 cfg → 用 compute_wd_charge_reward() WD sparse       │
│  2. 障礙物: _obstacle_policy_active=True → learned policy 取代 scripted  │
│  3. 數量: max_active_obstacles=10 (只 activate 前 10 slot)              │
│  4. LiDAR: --lidar_no_noise → 清除 distractor/displacement             │
│  5. Curriculum: --curriculum_version → 覆蓋 cfg 預設的 baseline_v1       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## `warp_drive_single_agent_v1` (你目前用的)

設計意圖: Single-agent 條件下，對齊 WD 「成功事件頻率」。
Phase 1 就有 dynamic + wall，用多 goals + 短距離補回低頻率。

| Phase | Name | Goals | Static | Dynamic (min~max) | Walls (min~max) | γ | Episode | Goal距離 | 碰撞懲罰 | Action Cost | Entropy(L/A) | Obs Speed |
|-------|------|-------|--------|-------------------|-----------------|------|---------|---------|----------|-------------|--------------|-----------|
| SA1 | goal_wall_dyn | 10 | 0 | 1~2 | 0~1 | 0.990 | 60s | 2~6m | -5 | 0.03 | 0.10/0.375 | 0.8 |
| SA2 | goal_wall_2dyn | 8 | 0 | 1~2 | 0~1 | 0.991 | 60s | 2~7m | -5 | 0 | 0.20/0.375 | 0.8 |
| SA3 | dyn_intro | 6 | 0 | 2~3 | 1~2 | 0.993 | 60s | 2~8m | -8 | 0 | 0.30/0.375 | 0.85 |
| SA4 | nav_avoid | 4 | 0 | 3~4 | 1~2 | 0.994 | 75s | 2~9m | -12 | 0 | 0.30/0.375 | 0.85 |
| SA5 | medium | 2 | 0 | 4~6 | 1~2 | 0.995 | 90s | 2~10m | -15 | 0 | 0.30/0.375 | 0.85 |

升階條件:
- SA1→SA2: SR≥72%, TO≤30%, 50+ updates
- SA2→SA3: SR≥65%, 65+ updates
- SA3→SA4: SR≥65%, dyn_SR≥35%, 80+ updates
- SA4→SA5: SR≥65%, dyn_SR≥35%, 95+ updates
- SA5: SR≥60%, dyn_SR≥30%, 110+ updates (最終階段)

---

## `warp_drive_goal_first` (Goal → Static → Dynamic 漸進)

| Phase | Name | Goals | Static | Dynamic | Walls | γ | Episode | 碰撞懲罰 | Obs Speed | obs_agent |
|-------|------|-------|--------|---------|-------|------|---------|----------|-----------|-----------|
| GF1 | goal_open | 8 | 0 | 0 | 0 | 0.990 | 45s | -5 | 0.8 | OFF |
| GF2 | goal_walls | 6 | 0 | 0 | 0~1 | 0.990 | 50s | -5 | 0.8 | OFF |
| GF3 | static_intro | 4 | 2 | 0 | 1 | 0.992 | 55s | -8 | 0.8 | OFF |
| GF4 | static_dense | 2 | 4 | 0 | 1 | 0.993 | 60s | -12 | 0.85 | OFF |
| GF5 | dynamic_intro | 1 | 4 | 2 | 1 | 0.994 | 70s | -12 | 0.85 | ON |
| GF6 | dynamic_medium | 1 | 4 | 6 | 1 | 0.996 | 90s | -40 | 0.85 | ON |
| GF7 | dense | 1 | 4 | 10 | 1~2 | 0.998 | 210s | -85 | 1.15 | ON |
| GF8 | stable | 1 | 4 | 6 | 1~2 | 0.998 | 210s | -200 | 1.15 | ON |

---

## `warp_drive_v1` (完整照 WD 原始設計)

核心: 從 Phase 1 就有 3 dynamic，無 static（WD 沒有 static 概念）。

| Phase | Name | Goals | Static | Dynamic | Walls | γ | Episode | 碰撞懲罰 | Obs Speed | Virtual Spots |
|-------|------|-------|--------|---------|-------|------|---------|----------|-----------|---------------|
| WD1 | nav_3obs_6spot | 8 | 0 | 3 | 0~1 | 0.990 | 60s | -5 | 0.8 | 5 |
| WD2 | 3obs_3spot | 8 | 0 | 3 | 0~1 | 0.991 | 60s | -8 | 0.85 | 2 |
| WD3 | single_3spot | 1 | 0 | 2 | 0~1 | 0.993 | 90s | -12 | 0.85 | 2 |
| WD4 | wall_3spot | 1 | 0 | 2 | 0~1 | 0.994 | 60s | -12 | 0.85 | 2 |
| WD5 | long_2spot | 1 | 0 | 2 | 0~1 | 0.995 | 100s | -12 | 0.85 | 1 |
| WD6 | 10obs_3spot | 1 | 0 | 10 | 0~1 | 0.997 | 210s | -15 | 0.85 | 2 |
| WD7 | main_2spot | 1 | 0 | 6 | 0~2 | 0.998 | 210s | -85 | 1.15 | 1 |
| WD8 | stable_2spot | 1 | 0 | 6 | 0~2 | 0.998 | 210s | -200 | 1.15 | 1 |

---

## `baseline_v1` (原始 v12 線性 8 階段)

最簡單: 每階 -1G, +1S, +1D

| Phase | Name | Goals | Static | Dynamic | Walls | γ | Episode |
|-------|------|-------|--------|---------|-------|------|---------|
| 1 | 純導航 | 8 | 0 | 0 | 0~2 | 0.990 | 45s |
| 2 | 1S+1D | 7 | 1 | 1 | 0~3 | 0.991 | 51s |
| 3 | 2S+2D | 6 | 2 | 2 | 1~4 | 0.993 | 56s |
| 4 | 3S+3D | 5 | 3 | 3 | 1~5 | 0.994 | 61s |
| 5 | 4S+4D | 4 | 4 | 4 | 2~6 | 0.996 | 67s |
| 6 | 5S+5D | 3 | 5 | 5 | 2~7 | 0.997 | 72s |
| 7 | 6S+6D | 2 | 6 | 6 | 3~8 | 0.998 | 78s |
| 8 | 終極挑戰 | 1 | 7 | 7 | 4~8 | 0.998 | 90s |

---

## `_make_stage()` 參數說明

| 參數 | 說明 | 影響什麼 |
|------|------|---------|
| `num_goals` | 同時存在的目標數 | 更多 = 更容易觸發 goal_reached 事件 |
| `n_static` | 靜態障礙物數量 | 不會動，需繞路 |
| `n_dynamic` | 動態障礙物上限 | 由 learned obstacle policy 控制移動 |
| `min_obstacles_dynamic` | 動態障礙物下限 | per-env random 在 [min, max] 之間 |
| `min_walls` / `max_walls` | 內部牆壁數量範圍 | per-env random count + random position |
| `gamma` | 折扣因子 | 越高 → agent 越重視遠期回報 |
| `episode_s` | Episode 最大秒數 | 越長 → 更多探索機會但也更難收到 reward |
| `goal_distance` | 目標距離範圍 (m) | (min, max) — 近距離 = 高成功率 |
| `spot_penalty_hit` | 碰撞懲罰 | 負值，Phase 升高 → 懲罰加重 |
| `spot_reward_get_goal` | 到達目標獎勵 | 固定 +40 |
| `spot_cost_operate` | 動作成本/步 | Phase 1: 0.03 → Phase 2+: 0 |
| `ent_coeff_linear` | Linear head entropy 係數 | 低=保守, 高=探索 |
| `ent_coeff_angular` | Angular head entropy 係數 | 通常固定 0.375 |
| `obstacle_speed_rate` | 障礙物最大速度倍率 | Phase 7+: 1.15（超過 charge 速度） |
| `obs_size_rand` | 障礙物半徑隨機量 | 增加場景多樣性 |
| `scene_bound_rand` | 場景邊界隨機量 | 增加場景多樣性 |
| `upgrade_sr` | 升階所需 Success Rate | 必須達到此值才能晉級 |
| `upgrade_max_cr` | 升階最大 Collision Rate | 碰撞率必須低於此值 |
| `upgrade_max_to` | 升階最大 Timeout Rate | 超時率必須低於此值 |
| `upgrade_min_dyn_sr` | 升階最小 Dynamic SR | 有 dynamic 時的成功率門檻 |
| `min_stage_updates` | 最少更新次數 | 防止升太快 |
| `downgrade_sr` | 降階 SR 門檻 | SR 低於此值 → 退回上一階 |

---

## 物理場景（所有 version 共用）

定義在 `charge_env_cfg_vlp16_curriculum.py` → `MySceneCfgVLP16_20x20`:

| 項目 | 值 |
|------|-----|
| 場地 | 20×20m (±10m) |
| 外牆 | 4 面, 1.0m 厚, 3.0m 高 |
| 內牆 slots | 8 個 (隱藏在 Z=-10, 由 curriculum 決定幾個可見) |
| 障礙物 slots | 100 個 (但 train_rnn_car_wdclip.py 只用前 10 個) |
| 障礙物外觀 | 10 種模板循環 (cuboid/cylinder, 0.2~0.35m radius, 1.6~1.8m height) |
| LiDAR | VLP16, z=1.6m, 72 bins, 360° |
| Robot | charge.usd (body_radius=0.35m) |
| 物理 | v_max=1.0 m/s, a_max=0.5 m/s², ω_max=0.25π rad/s, dt=0.2s |
