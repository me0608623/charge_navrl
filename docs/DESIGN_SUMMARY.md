# Charge-SKRL NavRL-Ground v3 設計總覽

> 最後更新: 2026-03-24 | Branch: `abl`

---

## 一、系統架構

```
Task: Isaac-Navigation-Charge-VLP16-Curriculum-NavRL
環境: Isaac Lab + Isaac Sim 5.1 + SKRL PPO
機器人: Charge 二輪差速驅動
場景: 20×20m, 4 邊界牆(1m厚) + 8 內部牆 slots + 20 障礙物
控制: 5Hz (dt=0.2s, decimation=20, sim_dt=0.01s)
```

---

## 二、觀測空間 (139D)

```
[0:4]     ego: ā(1) + v̄(1) + ω(1) + radius(1)           4D
[4:6]     goal: (dx, dy) body frame                        2D
[6:78]    LiDAR: 72 bins × 5° = 360°                     72D
[78:138]  obstacles: Top-10 × 6D (dx,dy,vx,vy,r,mask)    60D
[138]     time: remaining ratio                            1D
                                                   合計: 139D
```

### LiDAR 壓縮流程
```
VLP-16: 16 channels × 360 rays = 5760 條 3D ray
  → 投影 2D (捨棄 Z)
  → 噪聲注入 (±0.02m, 0.5% 遺失, 0.2% 幽靈)
  → reshape [N, 16, 72, 5]
  → min-pool (水平 5 rays/bin → 取最近)
  → min-pool (垂直 16 層 → 取最近)
  → 減車體半徑 0.3m, clamp ≥ 0
  → 正規化 ÷ 20m → [0, 1]
  → 輸出 [N, 72]
```

---

## 三、動作空間

```
MultiDiscrete([19, 19])
  idx 0: 線加速度 (0~18, center=9=零)
  idx 1: 角速度   (0~18, center=9=零)

物理限制:
  v_max  = 1.0 m/s     (允許倒車 [-1, +1])
  a_max  = 0.5 m/s²
  ω_max  = 0.25π rad/s
  dt     = 0.2 s
  max displacement/step = 0.2 m
```

---

## 四、網路架構

```
139D 觀測
  ├── [6:78]  LiDAR 72D  → Conv1d(1→32→64→64) + MaxPool + Linear → 64D
  ├── [78:138] Obstacles 60D → reshape [10,6] → MLP(6→32→32) + MaxPool → 32D
  └── [0:4]+[4:6]+[138] State 7D → MLP(7→32→32) → 32D
                                                    cat → 128D
                    ┌────────────────┴────────────────┐
              Actor Head                         Critic Head
         128→128→38 logits                  128→64→32→1 value
         (2×Categorical)              (orthogonal init, bias=-8.0)

Actor/Critic 各自獨立 feature extractor (不共享)
總參數量 ≈ 109K
```

---

## 五、獎勵設計 (NavRL-Ground v3)

### 設計哲學

對齊 NavRL 原論文: **目標方向信號永遠 100%，安全靠加法負項引導繞行。**
地面車調整: 保留 PBRS progress、front_block 懲罰、碰撞懲罰。
密度自適應: reward weights 由 curriculum stage 動態控制。

### 8 項 Reward

```
r_total = w_alive   × 1.0               每步存活
        + w_vel     × goal_velocity      dot(goal_dir, vel), 無 gate
        + w_prog    × goal_progress      d_prev - d_curr, 無 scale
        + w_ss      × static_safety      72-bin log clearance + front_block
        + w_ds      × dynamic_safety     per-obstacle log + closing risk
        + w_smooth  × smoothness         dv² + dw²
        + w_goal    × reaching_goal      到達目標 (終端)
        + w_coll    × collision          碰撞 (終端)

RewardManager 自動乘 dt=0.2
```

### 各項公式

| 項目 | 公式 | 範圍 |
|------|------|------|
| alive | 1.0 (無條件) | [1] |
| goal_velocity | clamp(dot(vel, goal_dir)/v_max, -1, 1) × (dist>0.1) | [-1, 1] |
| goal_progress | clamp(d_prev - d_curr, -0.25, 0.25) | [-0.25, 0.25] |
| static_safety | mean(ln(clearance_72bins)) + front_block | [-6, 3] |
| dynamic_safety | mean_vis(ln(clearance_i)) - mean_vis(risk_i) | [-6, 3] |
| smoothness | dv² + dw² | [0, 10] |
| reaching_goal | 1.0 if dist < 0.35m | {0, 1} |
| collision | 1.0 if LiDAR_min ≤ 0.45m | {0, 1} |

### front_block 公式
```
前方 ±30° 扇區 (12 bins), 取最近 5 條平均 = d_front
r_front = -max(0, 1.2 - d_front) × forward_speed
```

### closing_risk 公式 (dynamic_safety)
```
closing_rate = -dot(normalize(rel_pos), rel_vel)   # >0 = 接近中
risk = max(0, closing_rate) × exp(-clearance / 2.0)
```

### 固定權重 (不隨 stage 變)

| 項目 | weight |
|------|--------|
| alive | 0.2 |
| smoothness | -0.1 |
| reaching_goal | +100 |
| collision | -50 |
| time_penalty | 0 (停用) |

### Stage-dependent 權重 (由 curriculum 動態控制)

| Stage | vel | prog | ss | ds | 理由 |
|-------|-----|------|-----|-----|------|
| 1-2 (無障礙) | **5.0** | **6.0** | **0.2** | 0 | 專注 goal-reaching |
| 3-5 (static) | 4.5→3.5 | 5.5→4.5 | 0.4→1.2 | 0 | 漸增靜態避障 |
| 6-8 (dynamic) | 3.0→2.0 | 4.0→3.0 | 1.0→1.4 | 0.4→1.2 | 加入動態 |
| 9-12 (開放式) | 2.0 | 3.0 | 1.6→2.0 | 1.6→2.0 | 最大密度 |

---

## 六、避障機制

**不靠壓低前進動機，靠方向選擇:**

```
r_vel 梯度 → 指向目標
r_ss 梯度 → 指向遠離障礙的方向
合力 → 繞行方向

直衝: vel=+0.4 + ss=-0.6 = -0.2 (虧)
繞行: vel=+0.3 + ss=+0.1 = +0.4 (賺)  ← 最佳策略
不動: vel= 0   + ss=+0.1 = +0.1 (比繞行差)
後退: vel=-0.3 + ss=+0.1 = -0.2 (虧)
```

---

## 七、課程學習 (goal_first_v2, 12 Stage)

### 設計原則
1. 先導航後避障 (Stage 1-2 無障礙物)
2. static 先於 dynamic (Stage 3-5 無 dynamic)
3. 開放式 (Stage 9-12 超過 10 個障礙物, 最多 20 個)
4. reward weights 跟障礙物密度自動掛鉤

### Stage 表

| Stage | 名稱 | Goals | Static | Dynamic | Walls | SR門檻 |
|-------|------|-------|--------|---------|-------|--------|
| 1 | goal_open | 7 | 0 | 0 | 0-1 | >85% |
| 2 | goal_walls | 6 | 0 | 0 | 1-2 | >85% |
| 3 | static_light | 5 | 2 | 0 | 2-3 | >80% |
| 4 | static_medium | 4 | 4 | 0 | 3-4 | >80% |
| 5 | static_dense | 3 | 6 | 0 | 3-5 | >78% |
| 6 | dynamic_intro | 3 | 5 | 2 | 4-5 | >75% |
| 7 | dynamic_medium | 2 | 6 | 4 | 5-6 | >72% |
| 8 | crowded | 1 | 7 | 6 | 6-7 | >70% |
| 9 | dense_9 | 1 | **8** | **8** | 7-8 | >68% |
| 10 | dense_10 | 1 | **9** | **9** | 7-8 | >65% |
| 11 | dense_11 | 1 | **10** | **10** | 8-8 | >60% |
| 12 | ultimate | 1 | 10 | 10 | 8-8 | 100% |

### 升降級機制
- 升級: SR > 門檻 AND CR < 門檻 AND TO < 門檻，**連續 5 次通過**
- 降級: SR < 15% OR CR > 70% OR TO > 65%，**一次即降**
- 升級時清空 outcome window
- 每 Stage 有最低停留 rollout 數

---

## 八、場景配置

```
場景: 20×20m (±10m)
邊界牆: 4 面, 厚度 1.0m, 高度 1.5m
內部牆: 8 slots (RigidObjectCfg, per-env 隨機位置/方向)
障礙物: 20 個 (10 cuboid + 10 cylinder, 初始 Z=-10 隱藏)
         由 curriculum 控制啟用數量 (0→20)

動態障礙物行為:
  速度: [0.3, 1.2] m/s
  移動: 隨機目標點, 每 10 步重新採樣速度
  邊界: 8m 活動半徑, 9m 反彈
```

---

## 九、PPO 超參數

| 參數 | 值 |
|------|-----|
| rollouts | 128 |
| learning_epochs | 6 |
| mini_batches | 16 |
| 梯度更新/rollout | **96** |
| γ | 0.990→0.998 (stage-dependent) |
| GAE λ | 0.95 |
| LR | 1e-4 → 3e-5 (LinearLR) |
| LR total_iters | 10986 |
| gradient_clip | 1.0 |
| ratio_clip | 0.2 |
| entropy_coeff | 0.01 |
| value_loss_coeff | 1.0 |
| state_preprocessor | RunningStandardScaler |
| value_preprocessor | RunningStandardScaler |
| total_timesteps | 234,375 |
| checkpoint_interval | 5% (11,718 步, 20 個 checkpoint) |

---

## 十、終止條件

| 條件 | 門檻 | 類型 | Bootstrap |
|------|------|------|-----------|
| goal_reached | dist < 0.35m | Success | No |
| collision (LiDAR) | min ≤ 0.45m | Failure | No |
| wall_collision | AABB ≤ 0.45m | Failure | No |
| time_out | 45~90s | Truncation | **Yes** |
| physics_explosion | vel > 10 m/s | Failure | No |

---

## 十一、Domain Randomization

| 參數 | 範圍 |
|------|------|
| 質量縮放 | [0.85, 1.15] |
| 摩擦縮放 | [0.7, 1.3] |
| 質心偏移 | [-0.05, 0.05] m |
| 外力干擾 | [0, 3.0] N |
| LiDAR 雜訊 | ±0.02m |
| LiDAR 遺失 | 0.5% |
| LiDAR 幽靈 | 0.2% |
| Reset 位置 | X,Y ∈ [-7, 7]m, Yaw ∈ [-π, π] |

---

## 十二、診斷指標 (15 項)

| WandB Key | 定義 |
|-----------|------|
| ablation/stuck_count | 連續 ≥5 步 speed<0.02 的次數 |
| ablation/freeze_ratio | speed<0.02 步數佔比 |
| ablation/oscillation_score | v_toward 符號翻轉頻率 |
| ablation/retreat_ratio | 近障礙時 v_toward<0 的比例 |
| ablation/progress_near_obs | 近障礙時 progress 平均 |
| ablation/d_safe_mean | 全局平均安全距離 |
| ablation/d_safe_min | rollout 中最小安全距離 |
| ablation/danger_ratio | d_safe<0.8 步數佔比 |
| ablation/speed_near_obs | 近障礙時平均速度 |
| ablation/shield_rate | shield 介入比例 |
| ablation/front_clearance | 前方扇區平均距離 |
| ablation/avg_ep_length | episode 平均長度 |
| ablation/collision_ep_len | 碰撞 episode 平均長度 |
| train/module_entropy | log10(actor_grad/critic_grad) |

---

## 十三、版本演進

| 版本 | 核心改動 | 問題 |
|------|---------|------|
| **baseline (NavRL Dense)** | v_gate 硬關閉 + safe_progress 翻負 | 90% timeout (煞車→後退→徘迴) |
| **Ground v1** | soft_gate(0.2) + soft_scale(0.3) | 90% timeout (gate 仍在壓低) |
| **Ground v2** | 無 gate + alive=0.2 + 無 scale | 97% timeout (static_safety 佔 92%) |
| **Ground v3** | v2 + 密度自適應權重 + 20 obstacles + 12 Stage | 目前訓練中 |

### v3 vs v2 差異

| 面向 | v2 | v3 |
|------|-----|-----|
| ss/ds weight | 固定 2.0/2.0 | **密度自適應** 0.2→2.0 |
| vel/prog weight | 固定 2.0/3.0 | **前高後低** 5.0→2.0 / 6.0→3.0 |
| MAX_OBSTACLES | 10 | **20** |
| Stage 數 | 8 (封閉) | **12 (開放式)** |
| Stage 1 ss 佔比 | 61% | **~10%** |

---

## 十四、訓練指令

```bash
# v3 + goal_first_v2 + closing_risk
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v3 \
  --curriculum_version goal_first_v2 \
  --dynamic_safety_mode closing_risk \
  --run_name 'rw_groundv3_goalfirstv2_full_seed1' \
  --seed 1 --num_envs 2048 --headless
```

### 可用 CLI 參數

| 參數 | 用途 | 預設 |
|------|------|------|
| --reward_mode | current/v1/v2/v3 | current |
| --curriculum_version | baseline_v1/goal_first_v1/goal_first_v2 | baseline_v1 |
| --dynamic_safety_mode | log_distance/closing_risk | log_distance |
| --w_alive | alive 權重 | 0.2 |
| --goal_vel_use_soft_gate | 恢復 gate | False |
| --run_name | 實驗命名 | timestamp |
| --seed | 隨機種子 | 42 |

---

## 十五、檔案結構

```
source/isaaclab_tasks/.../charge_skrl/
├── __init__.py                              # 3 個 task 註冊
├── cfg/
│   ├── charge_cfg.py                        # Robot USD (自動搜尋路徑)
│   ├── charge_env_cfg_vlp16.py              # Base: MAX_OBSTACLES=20
│   └── charge_env_cfg_vlp16_curriculum.py   # Curriculum + NavRL + Ground v1/v2/v3
├── mdp/
│   ├── rewards/
│   │   ├── navrl_rewards.py                 # 舊版 NavRL (v_gate)
│   │   ├── navrl_ground_rewards.py          # Ground v1-v3 (無 gate + alive)
│   │   ├── gap_rewards.py                   # Gap-seeking
│   │   └── potential_based_rewards.py       # PBRS + collision
│   ├── actions/
│   │   ├── discrete_differential_drive.py   # MultiDiscrete([19,19])
│   │   └── safety_shield.py                # Optional shield
│   ├── observations/obs_functions.py        # 139D pipeline
│   ├── wall_layout.py                       # Per-env wall geometry
│   └── events/walls.py                      # Wall randomization
├── curriculum/goal_obstacle_curriculum.py   # baseline_v1/goal_first_v1/v2
└── agents/skrl_ppo_cfg_vlp16.yaml           # PPO 超參數

scripts/reinforcement_learning/skrl/
├── train_charge_ac.py                      # 主訓練入口 + CLI
├── wandb_trainer.py                        # WandB 整合
├── vlp16_models.py                         # 3-branch 網路
├── console_summary.py                      # 終端摘要
└── diagnostics/ablation_metrics.py         # 15 指標
```
