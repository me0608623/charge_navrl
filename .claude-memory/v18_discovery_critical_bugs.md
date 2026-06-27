---
name: v18_discovery_critical_bugs
description: 2026-04-07 v18b/v19 訓練實驗發現的兩個 SHOWSTOPPER bugs (kinematic 障礙物無法 contact + LiDAR 內建 distractor 噪聲) 與修復記錄
type: project
---

# 2026-04-07 重大發現：兩個 SHOWSTOPPER bugs

## 起因

v17/v18b/v19 訓練都在 metric 上看起來不錯（CR < 1%, SR > 80%），但 GUI play 時可以看見 **agent 直接撞上障礙物還卡著**。透過 diagnostic 資料追蹤，發現兩個系統性 bugs。

## Bug A: 障礙物 kinematic + contact threshold 0.1N → CR 嚴重低估

**證據鏈：**

1. `charge_env_cfg_vlp16.py:272` 障礙物宣告為 `kinematic_enabled=True`
2. 障礙物用 `move_obstacles_vectorized` → `write_root_pose_to_sim` teleport 移動
3. PhysX kinematic body 不會與動態 robot 產生足夠 contact force
4. `safety_rewards.py:500` 碰撞判定要求 `force_magnitude > 0.1 N`
5. 結果：teleport 進入 robot 時 force ≈ 0 → 無 collision termination

**症狀：** CR=0% 但 robot 與 obstacle 物理重疊，agent 卡在障礙物上 oscillate 動作 (REV=83% + 隨機方向)

**修復：** 加入 `obstacle_collision_geometric` 純幾何碰撞函數：
- `mdp/terminations/robot_state.py` 新函數
- `cfg/charge_env_cfg_vlp16.py` TerminationsCfgVLP16 加 `obstacle_collision = DoneTerm(...)`
- 直接計算 ‖robot_pos - obs_pos‖ < (body_radius + obs_radius)
- 自動排除隱藏障礙物 (z ≤ 0)

## Bug B: LiDAR 內建 distractor → lidar.min 永遠 ≈ 0

**證據鏈：**

1. `cfg/charge_env_cfg_vlp16.py:410-414` LiDAR ObsTerm 寫死 `distractor_rate=0.002`
2. `lidar_vlp16_to_2d_bins` 內建 displacement_std=0.02 + hole_rate=0.005 + 偽近距離 distractor
3. VLP-16 = 5760 rays × 0.002 = 平均每幀 11.5 個假近距離讀數 (0.2-2.0m)
4. 經過 `(dist - r_robot) / r_max` normalization 後，min 值永遠 ≈ 0
5. ObsTerm 上還有額外 `noise=Unoise(±0.02)`

**症狀：** lidar.min 永遠回報 0.000-0.026，無論真實障礙物距離多少。Policy 學會「不能信 LiDAR 近距離」→ 與 Bug A 結合 → 「忽略近距離 + 慢慢推進 + 卡在障礙物上」

**致命影響：** `--no_domain_randomization` flag **沒有關掉這些 LiDAR 內建噪聲**（因為它們寫死在 obs function params 裡，不是 DR event）。所以 v18b/v19 號稱「對照組」其實還在用 hand-tuned 合成噪聲。

**修復：**
- `train_charge_ac.py` 新增 `--lidar_no_noise` flag
- 處理邏輯：遍歷所有 obs group 找 lidar* term，設 displacement/hole/distractor=0 + term.noise=None
- `obs_functions.py` 加 `r_min` 與 `z_filter` 參數
- 預設 `r_min=0.9`（真實 VLP-16 / RPLidar 盲區）+ `z_filter=0.5`（過濾 z 異常命中）

## 真實 LiDAR 校準關鍵數值

- **真實 LiDAR min_range = 0.9 m**（用戶實測）
  - 比此距離近的物體 → 真實 lidar 看不到（no return）
  - sim 必須匹配，否則 sim-to-real 失敗
  - 修復：lidar function `r_min=0.9` 將 < 0.9m 的命中視為 r_max

## 對所有歷史訓練的影響

| Run | 受影響 | 嚴重度 |
|-----|:---:|:---:|
| v17 (full DR) | ✓ A + B | 高 - CR 嚴重低估 |
| v18b (no_DR + 6m) | ✓ A + B | 高 - 對照組失效 |
| v19 (no_DR + 13m) | ✓ A + B | 高 - 對照組失效 |
| 所有 best_agent.pt | ✓ | 訓練時 obs 是假的 |

**結論：** 所有 v17 之前的訓練都不能信。需要在修復後重訓 v20 baseline。

## 之前曾經存在的 LiDAR collision 函數

`mdp/temp/terminations/collision.py:76` 有舊的 `lidar_collision()` 函數：
- 用 LiDAR 2D 距離 < threshold (預設 0.5m) 判定碰撞
- 已被搬到 `temp/` 資料夾不再使用
- 換成 contact sensor 是因為「contact sensor 比較準」← 但實際上 kinematic obstacles 完全推翻這個假設
- 未來可以考慮恢復這個函數作為 Bug A 的另一種解法

## 修復後的對照組訓練指令

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v8 --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk --no_walls \
  --no_domain_randomization \
  --lidar_no_noise \
  --run_name rw_groundv8_..._true_control_v20 \
  --seed 1 --num_envs 6144 --headless --directional_gate
```

加 `--no_domain_randomization` + `--lidar_no_noise` 才是真正無噪聲 baseline。

**How to apply:** 
- 新 session 看到 user 提到 v17/v18/v19/CR/tipping 時，要先告訴 user 這些訓練有 Bug A/B 影響
- v20+ 才是修復後的乾淨訓練
- user 的研究設計：sim 真實 LiDAR 校準噪聲 vs 無噪聲對照組 → 這兩個 bug 修復是前提條件
