# NavRL-Ground v2 消融實驗設計

> Baseline: `rw_groundv2_closingrisk_full_seed1`
> Task: `Isaac-Navigation-Charge-VLP16-Curriculum-NavRL`
> Reward mode: `navrl_ground_v2` + `closing_risk` + `goal_first_v1`

---

## 實驗目的

驗證 NavRL-Ground v2 每個設計決策的必要性。
每組只改一個因素，其他全部保持 baseline。

---

## Family 1: alive reward 權重

**假說**: alive=0.2 是否為最佳平衡點？太高 → 不到達目標；太低 → 失去存活動機。

| run_name | alive weight | 預期 |
|----------|-------------|------|
| `abl_rw_v2_alive0_full_seed1` | 0 | SR↑ 但可能更冒險 |
| `abl_rw_v2_alive02_full_seed1` | 0.2 **(baseline)** | — |
| `abl_rw_v2_alive05_full_seed1` | 0.5 | TO↑ (活著太舒服) |
| `abl_rw_v2_alive10_full_seed1` | 1.0 | TO↑↑ (NavRL 原值，但 v_max=1 不適用) |

```bash
# alive=0 (無存活獎勵)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version goal_first_v1 \
  --dynamic_safety_mode closing_risk \
  --w_alive 0 \
  --run_name 'abl_rw_v2_alive0_full_seed1' --seed 1 --num_envs 512 --headless

# alive=0.5
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version goal_first_v1 \
  --dynamic_safety_mode closing_risk \
  --w_alive 0.5 \
  --run_name 'abl_rw_v2_alive05_full_seed1' --seed 1 --num_envs 512 --headless
```

**觀察指標**: `nav/timeout_rate`, `nav/success_rate`, `ablation/freeze_ratio`

---

## Family 2: goal_velocity gate

**假說**: 無 gate（v2 設計）是否真的比有 gate（v1 設計）好？

| run_name | gate 設定 | 預期 |
|----------|----------|------|
| `abl_rw_v2_nogate_full_seed1` | use_soft_gate=False **(baseline)** | — |
| `abl_rw_v2_gate02_full_seed1` | use_soft_gate=True, beta=0.2 | TO↑ freeze↑ (v1 復現) |
| `abl_rw_v2_gate05_full_seed1` | use_soft_gate=True, beta=0.5 | 介於 v1/v2 之間 |

```bash
# 恢復 soft_gate (beta=0.2, 模擬 v1 行為)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version goal_first_v1 \
  --dynamic_safety_mode closing_risk \
  --goal_vel_use_soft_gate --goal_vel_gate_beta 0.2 \
  --run_name 'abl_rw_v2_gate02_full_seed1' --seed 1 --num_envs 512 --headless
```

**觀察指標**: `ablation/retreat_ratio`, `ablation/stuck_count`, `nav/timeout_rate`

---

## Family 3: goal_progress 貢獻

**假說**: 地面車是否需要 PBRS progress？NavRL 原論文沒有。

| run_name | progress weight | 預期 |
|----------|----------------|------|
| `abl_rw_v2_prog0_full_seed1` | 0 (純 r_vel 驅動) | 遠距離目標 TO↑ |
| `abl_rw_v2_prog3_full_seed1` | 3.0 **(baseline)** | — |
| `abl_rw_v2_prog6_full_seed1` | 6.0 (加強 progress) | SR↑ 但可能衝障礙 |

```bash
# progress=0 (純 NavRL 風格，只靠 r_vel)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version goal_first_v1 \
  --dynamic_safety_mode closing_risk \
  --w_prog 0 \
  --run_name 'abl_rw_v2_prog0_full_seed1' --seed 1 --num_envs 512 --headless
```

**觀察指標**: `nav/success_rate` (遠距離目標), `ablation/progress_near_obs`

---

## Family 4: static_safety 設計

**假說**: 72-bin log clearance + front_block 是否都必要？

| run_name | 改動 | 預期 |
|----------|------|------|
| `abl_rw_v2_ss_full_full_seed1` | baseline (72-bin + front_block) | — |
| `abl_rw_v2_ss_nofrontblock_full_seed1` | a_front_block=0 (無前向懲罰) | CR↑ (正面撞障礙) |
| `abl_rw_v2_ss_bottomk_full_seed1` | 改回 bottom-K scalar (舊版) | 失去方向性 |

**觀察指標**: `nav/collision_rate`, `ablation/front_clearance`, `ablation/d_safe_min`

---

## Family 5: dynamic_safety mode

**假說**: closing_risk 是否比純 log_distance 好？

| run_name | mode | 預期 |
|----------|------|------|
| `abl_rw_v2_dslog_full_seed1` | log_distance (無 closing risk) | 動態場景 CR↑ |
| `abl_rw_v2_dsclosing_full_seed1` | closing_risk **(baseline)** | — |

```bash
# 純 log_distance (無 closing risk)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version goal_first_v1 \
  --dynamic_safety_mode log_distance \
  --run_name 'abl_rw_v2_dslog_full_seed1' --seed 1 --num_envs 512 --headless
```

**觀察指標**: Stage 5+ 的 `nav/collision_rate`, `eval/collision_rate_dynamic`

---

## Family 6: reaching_goal 終端獎勵

**假說**: goal=100 是否合適？NavRL 原論文是 0。

| run_name | goal weight | 預期 |
|----------|------------|------|
| `abl_rw_v2_goal0_full_seed1` | 0 (NavRL 原論文) | 學習更慢，SR 早期低 |
| `abl_rw_v2_goal100_full_seed1` | 100 **(baseline)** | — |
| `abl_rw_v2_goal300_full_seed1` | 300 | SR↑ 但可能衝障礙物拿分 |

**觀察指標**: 學習曲線前期斜率, `nav/success_rate`

---

## Family 7: collision 懲罰

**假說**: collision=-50 是否足夠？NavRL 實際上是 0。

| run_name | collision weight | 預期 |
|----------|-----------------|------|
| `abl_rw_v2_coll0_full_seed1` | 0 (NavRL 原論文) | CR↑ (完全靠 r_ss/r_ds 避障) |
| `abl_rw_v2_coll50_full_seed1` | -50 **(baseline)** | — |
| `abl_rw_v2_coll100_full_seed1` | -100 | CR↓ 但可能過度保守 |

**觀察指標**: `nav/collision_rate`, `nav/timeout_rate`

---

## Family 8: smoothness 權重

**假說**: smoothness=-0.1 是否讓差速車轉彎代價過高？

| run_name | smooth weight | 預期 |
|----------|--------------|------|
| `abl_rw_v2_smooth0_full_seed1` | 0 (無平滑懲罰) | 控制抖動但可能更靈活 |
| `abl_rw_v2_smooth01_full_seed1` | -0.1 **(baseline)** | — |
| `abl_rw_v2_smooth03_full_seed1` | -0.3 (重罰抖動) | 轉彎困難，TO↑ |

**觀察指標**: `ablation/oscillation_score`, 轉彎靈活度

---

## Family 9: 課程學習版本

**假說**: goal_first_v1（先導航再避障）是否比 baseline_v1（同時加 static+dynamic）好？

| run_name | curriculum | 預期 |
|----------|-----------|------|
| `abl_curr_baselinev1_full_seed1` | baseline_v1 (Stage 2 就有 dynamic) | 早期學習更混亂 |
| `abl_curr_goalfirstv1_full_seed1` | goal_first_v1 **(baseline)** | — |

```bash
# baseline curriculum
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v2 --curriculum_version baseline_v1 \
  --dynamic_safety_mode closing_risk \
  --run_name 'abl_curr_baselinev1_full_seed1' --seed 1 --num_envs 512 --headless
```

**觀察指標**: Stage 1-2 的學習速度, 最終 Stage 到達數

---

## Family 10: 權重比例整體

**假說**: vel:ss:ds 的比例是否為最佳？

| run_name | vel:prog:ss:ds | 設計理念 |
|----------|---------------|---------|
| `abl_rw_v2_ratio111_full_seed1` | 1:1:1:1 (NavRL 原始) | 最忠實 NavRL |
| `abl_rw_v2_ratio2322_full_seed1` | 2:3:2:2 **(baseline)** | — |
| `abl_rw_v2_ratio4211_full_seed1` | 4:2:1:1 (強目標驅動) | 目標驅動最大化 |

**觀察指標**: SR/CR/TO 三角平衡

---

## 實驗優先序

按影響度排序，建議先跑：

| 優先 | Family | 原因 |
|------|--------|------|
| **1** | Family 2 (gate) | 驗證 v2 核心設計假說 |
| **2** | Family 1 (alive) | 驗證 alive weight 的最佳值 |
| **3** | Family 3 (progress) | 驗證地面車是否需要 PBRS |
| **4** | Family 5 (dynamic mode) | 驗證 closing_risk 的價值 |
| **5** | Family 9 (curriculum) | 驗證課程設計 |
| 6 | Family 6 (goal) | 終端獎勵影響 |
| 7 | Family 7 (collision) | 碰撞懲罰影響 |
| 8 | Family 4 (static) | 靜態安全設計 |
| 9 | Family 8 (smoothness) | 平滑懲罰影響 |
| 10 | Family 10 (ratio) | 整體權重比例 |

---

## 所需 CLI 擴展

目前缺少的 CLI 參數（需新增才能跑 Family 1, 4）：

| 參數 | 用途 | 預設值 |
|------|------|--------|
| `--w_alive` | alive reward weight | 0.2 |
| `--goal_vel_use_soft_gate` | 是否啟用 soft gate | (v2=False) |
| `--ss_front_block_weight` | front_block 係數 | 1.0 |

其他 Family 的參數已存在：`--w_vel`, `--w_prog`, `--w_ss`, `--w_ds`, `--w_smooth`, `--w_goal`, `--w_collision`, `--dynamic_safety_mode`, `--curriculum_version`

---

## 結果判讀標準

| 指標 | 好 | 可接受 | 差 |
|------|-----|--------|-----|
| SR (Stage 1) | >85% | 50-85% | <50% |
| CR (全局) | <15% | 15-30% | >30% |
| TO (全局) | <30% | 30-50% | >50% |
| freeze_ratio | <5% | 5-15% | >15% |
| retreat_ratio | <20% | 20-40% | >40% |

---

## 多 Seed 驗證

每個最終方案需跑 3 seeds：

```bash
--seed 1 --run_name '..._seed1'
--seed 42 --run_name '..._seed2'
--seed 123 --run_name '..._seed3'
```

WandB 用 `Group by: variant` 查看平均 ± 標準差。
