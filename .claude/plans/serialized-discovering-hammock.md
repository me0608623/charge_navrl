# v8: 修正 (ss+ds)/goal 比例失衡

## Context

v7 Stage 6 的 (ss+ds)/goal = 112%，安全 reward 超過核心 reward。
導致 agent 速度慢 (0.34 m/s) 且目標前躊躇。
v6 Stage 4 的比例只有 5%，速度 0.67 m/s 且果斷。

## 根因

Bootstrap 和 open-ended 的 safety weight 太高：
- B5: ss=0.8, ds=0.3
- B6: ss=1.0, ds=0.5
- OE: ss=min(1.2,...), ds=min(1.0,...)

而 goal_velocity 隨 stage 遞減 (5.0→2.5)，兩者交叉後 safety 主導。

## 修改：降低 safety weight，提高 goal_velocity 下限

### Bootstrap stages

| Stage | ss (v7) | ss (v8) | ds (v7) | ds (v8) | vel (v7) | vel (v8) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B1 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 5.0 |
| B2 | 0.2 | 0.2 | 0.0 | 0.0 | 5.0 | 5.0 |
| B3 | 0.4 | 0.3 | 0.0 | 0.0 | 4.8 | 4.8 |
| B4 | 0.6 | 0.4 | 0.0 | 0.0 | 4.5 | 4.5 |
| B5 | 0.8 | **0.5** | 0.3 | **0.2** | 4.0 | **4.5** |
| B6 | 1.0 | **0.5** | 0.5 | **0.3** | 3.5 | **4.0** |

### Open-ended reward weights

```python
# v7 (現在)
"goal_velocity": max(2.5, 3.5 - 0.08 * level),
"goal_progress": max(3.0, 4.0 - 0.08 * level),
"static_safety": min(1.2, 1.0 + 0.05 * level),
"dynamic_safety": min(1.0, 0.5 + 0.05 * level),

# v8 (新)
"goal_velocity": max(3.5, 4.0 - 0.05 * level),   # 下限提高 2.5→3.5
"goal_progress": max(3.5, 4.0 - 0.05 * level),    # 下限提高 3.0→3.5
"static_safety": min(0.5, 0.3 + 0.02 * level),    # 上限降低 1.2→0.5
"dynamic_safety": min(0.4, 0.2 + 0.02 * level),   # 上限降低 1.0→0.4
```

### 預期效果

Stage 6 的 (ss+ds) weight: 1.5 → 0.8
goal_vel weight: 3.5 → 4.0

預估期望值比例:
```
ss+ds 期望值 ≈ 0.5 × func_ss + 0.3 × func_ds ≈ 0.5×1.5 + 0.3×0.8 ≈ 0.99
→ 乘以 dt=0.2 → per-sec ≈ 0.20
goal 期望值 ≈ 0.93 (不變，取決於 SR)
(ss+ds)/goal ≈ 20-30% (從 112% 降到合理範圍)
```

## 修改檔案

1. `curriculum/goal_obstacle_curriculum.py`
   - B3-B6 的 `reward_weights` 中 ss/ds 降低
   - B5-B6 的 goal_velocity 提高
   - `_open_ended_reward_weights()` 函數修改

2. `cfg/charge_env_cfg_vlp16_curriculum.py`
   - 新增 `RewardsCfgVLP16NavRLGroundV8` 繼承 V7

3. `train_charge_ac.py`
   - 新增 `navrl_ground_v8` 選項

## 訓練指令

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v8 \
  --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk \
  --no_walls --use_cadn \
  --run_name rw_groundv8_openendedv1__seed1_nowalls \
  --seed 1 --num_envs 6144 --headless
```

## 驗證

- Stage 6 的 (ss+ds)/goal < 30%
- 速度 > 0.5 m/s
- SR 維持 > 80%
- CR 維持 < 5%
