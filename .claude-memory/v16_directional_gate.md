---
name: v16 Directional Gate Design
description: v16 方向性 gate 設計 — 解決 v15 Stage 8 retreat-dominant 問題的核心改動，含問題根因、公式、參數、檔案變更
type: project
---

## 問題根因 (v15 Stage 8)

v15 的 gate 使用 omnidirectional `d_safe`（全場最近 10 條 LiDAR ray 的均值），在 Stage 8（17 obstacles）場景中 d_safe 幾乎永遠 < gate_dmin (0.6m)，導致：

```
gate ≈ gate_beta = 0.15 → goal_velocity 永遠只剩 15%
→ collision penalty (-0.211) 壓倒 goal signal (0.028)
→ retreat 成為最優策略 (63.8% retreat / 22.7% forward / 13.5% stop)
→ timeout 27.9% → SR 停滯在 71% (target 80%) 且持續退步
```

關鍵診斷數據：
- `ctx_near_obs_count = 16384 (ALL)` — 100% timesteps 都在「近障礙區」
- `front_clearance = 1.62m` — 但前方其實常常暢通
- KL=0.30（所有 stage 最高）, clip_fraction=0.77, critic dead neurons=19.4%

## v16 設計：方向性 Gate

**核心改動**：gate 的 d_safe 輸入從 omnidirectional 改為 goal-direction cone。

### 新增函數

1. `_get_d_goal_direction(env, sensor_cfg, robot_cfg, ...)` — 計算 goal 方向 ±cone 的 LiDAR clearance
   - 將 goal direction 轉到 robot body frame → 找對應 LiDAR bin
   - 取 ±cone_half_bins (default ±6 = ±30°) 範圍的 bins
   - cone 內取 bottom-k (default 3) ray 的均值 - body_radius

2. `_compute_effective_d_safe(...)` — 混合 directional + omnidirectional
   - `d_effective = (1 - omni_blend) * d_goal_dir + omni_blend * d_omnidirectional`
   - `directional=False` 時退化為原行為

### 修改的函數

- `goal_velocity_reward()`: 新增 `directional_gate`, `cone_half_bins`, `cone_bottom_k`, `omni_blend` 參數
- `goal_progress_reward()`: 新增 `directional_scale` + 同上參數

### CLI 參數

```
--directional_gate              # 啟用方向性 gate (default: off = v15 行為)
--gate_cone_half_bins 6         # ±30° cone (5°/bin)
--gate_cone_bottom_k 3          # cone 內取最近 3 條 ray
--gate_omni_blend 0.2           # 混合 20% omnidirectional (保留全局安全意識)
```

### 檔案變更

| 檔案 | 變更 |
|------|------|
| `mdp/rewards/navrl_ground_rewards.py` | +`_get_d_goal_direction()`, +`_compute_effective_d_safe()`, 修改兩個 reward fn |
| `train_charge_ac.py` | +4 CLI args, 兩處 override (通用 + reward_mode v1) |
| `charge_env_cfg_vlp16_curriculum.py` | 未改（靠 CLI 啟用） |

### 備份

v15 原版在 `.backup/v15_gate/`

### 預期效果

Stage 8: d_goal_dir (前方) ≈ 1.2~1.6m → gate ≈ 0.55~0.75 (vs 原 0.15)
→ goal_velocity 有效信號從 0.028 提升到 ~0.120，能與 collision penalty 抗衡

### 啟動指令

```bash
# v16: directional gate
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name v16_dir-gate_s1_ne512 --directional_gate

# v16b: + 提高 floor
... --directional_gate --goal_vel_gate_beta 0.25
```

**Why:** v15 的 omnidirectional gate 在密集障礙場景永遠壓制 goal signal，導致 retreat-dominant policy。方向性 gate 讓側面/後方障礙不影響 goal attraction。

**How to apply:** 任何涉及 gate/reward balance 的改動都要考慮：gate 是否在目標場景中永遠啟動？如果是，floor 值和方向性就很關鍵。

### 文獻依據

- **Dynamic Warning Zone** (Springer 2023) — per-obstacle 方向性 warning zone
- **SMPO** (arXiv 2025) — learned safety critic 調制 reward（概念最接近 gate_beta）
- **Zhang et al. (2020)** — 過度 obstacle penalty → overly cautious → mission failure（直接印證 v15 問題）
- **SARL** (ICRA 2019) — attention over obstacles，中期可考慮替代 MaxPool
