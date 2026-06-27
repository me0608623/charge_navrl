---
name: project-v2-roadmap
description: 未來訓練方向：SA1~SA7 為 baseline，從 SA1_v2 重新開始整合 Asymmetric Critic + VLP16 噪聲 + Sim-to-Real DR
metadata: 
  node_type: memory
  type: project
  originSessionId: 6c7a8fa6-156d-4c20-b4f6-050d27e4d932
---

## SA1~SA7 = Baseline（舊架構）

SA1~SA7 系列訓練使用的是 **symmetric critic + 無 LiDAR 噪聲 + 無域隨機化** 的基礎架構。
這些 runs 作為 baseline 對照，不會再延續訓練。

## v2 訓練方向：從 SA1_v2 重新開始

未來所有訓練從 SA1_v2 開始，整合以下三大改善：

### 1. Asymmetric Critic 架構
- Critic 使用 96D 特權 obs（障礙物 50D + 牆壁 40D + goal queue 6D）
- Policy（Actor）維持原本 LiDAR-only obs，部署時不受影響
- `--critic_profile asymmetric`
- 設計文件：[[2026-05-26_Privileged-Information-設計文件]]

### 2. VLP16 雜訊模型（TLNI 三層）
- Layer 1 Per-Ray: displacement_std, hole_rate, distractor_rate, distance_bias, per_ring_bias
- Layer 2 Per-Bin: obs_noise_std, block_dropout
- Layer 3 Per-Episode DR: displacement_std_dr, hole_rate_dr
- `--lidar_no_noise false` + 各層參數
- 設計文件：[[2026-05-26_Sim-to-Real-4項實作筆記]]

### 3. Sim-to-Real Gap 域隨機化
- Physics DR: mass_dr, friction_dr, com_offset
- External disturbance: wind_force, push_force, push_ratio
- Observation latency DR: obs_delay_steps
- Heading stability reward: heading_stability_weight
- RGDR 環境加權（可選）
- DORAEMON 自動 DR 擴張（可選）
- `--no_domain_randomization false`
- 設計文件：[[2026-05-26_Sim-to-Real-Gap-文獻綜述-2023-2026]]

### 附加：RNN Capacity + Velocity Aux（Step 2+3）
- hidden_dim 30 → 64, fc_dim 48 → 64, wd_middle_dim → 48
- predict_dim 7 → 13（+top-3 obstacle body-frame velocity）
- YAML: `wd_sa6_rnn64_vel.yaml`
- 設計文件：[[2026-05-26_動態避障三步驟強化設計]]

**Why:** SA1~SA7 的 SR 在 SA6+ 卡在 ~60%，瓶頸是 (1) RNN 30D 容量不足以追蹤多動態障礙物 (2) symmetric critic 無法提供精準 value gradient (3) 無噪聲/DR 導致 sim-to-real gap 過大。v2 架構一次整合這三項，從頭訓練建立新 baseline。

**How to apply:** 所有新訓練實驗應基於 v2 架構，YAML config 應包含 asymmetric critic + TLNI + DR 設定。舊 SA1~SA7 的 checkpoint 不相容（hidden_dim 改變），不用嘗試續訓。

## v2 不需要新 gym task（2026-05-27 設計決策）

經查 code 確認：v2 三大改善**都不需要新建 `--task`**，沿用現有 `Isaac-Navigation-Charge-VLP16-Curriculum-WD`（20×20）/ `-WD-TCorridor`（T 走廊），差異全封裝在 experiment_config YAML（`wd_sa*_v2.yaml`）+ CLI flag：
- **Asymmetric critic**：`scripts/.../utils/privileged_obs.py` 的 `extract_privileged_obs` 從 scene 實體直接抽 96D（障礙 50 + 牆 40 + goal 6），與 task 無關。env cfg 的 `CriticCfg`（139D symmetric 殼）沒被 WD pipeline 使用，可忽略。
- **TLNI 雜訊**：LiDAR obs term `wd_like_sweep_72` 已有 noise 參數，靠 `--lidar_no_noise false` + YAML 開。
- **DR**：`charge_env_cfg_vlp16_curriculum.py` 已有 `domain_randomization` EventTerm，靠 `--no_domain_randomization false` 開。
- CLI flag 全已接好：`--critic_profile asymmetric`、`--hidden_dim 64`、`--predict_dim 13`。

v2 LiDAR 雜訊是**漸進 curriculum**（已驗證，非 bug）：SA1_v2 `lidar_no_noise: true`（clean bootstrap，刻意），SA2_v2 起 `lidar_no_noise: false` + 完整 TLNI 三層參數（SA2 為 50% 校準低強度，SA3+ 遞增）。SA1 clean 是 sim-to-real 標準做法（clean bootstrap → progressive DR），不需改。詳見 [[2026-05-27_v2-task-是否需要新建-設計決策]]。
