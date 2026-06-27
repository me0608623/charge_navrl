---
name: deploy_v3c_rover_rl
description: rover_rl 部署 v3c (SA3_v3c) — 83D obs + action stacking + α slew + r_min 0.25 的完整對接規格與啟動方式
metadata: 
  node_type: memory
  type: project
  originSessionId: 7b272734-c0b9-48e1-bb8b-79c55f6ca122
---

rover_rl (車端部署 repo, origin: me0608623/rover_rl) 從 SA6 (79/139D) 升級支援 v3c (SA3_v3c, 83D)。2026-06-12 完成並合併 (origin/main `8b0dd38`)。SA6 路徑完全保留、行為 byte 不變；v3c 靠「換模型 + 換 yaml」啟用。

**模型**：`sa3_v3c_240000.ts`（也匯出了 `sa2_v3c_90000.ts`，同架構）。raw_obs=83 / hidden=64 / middle=48 / predict=13 / RNN。`export_policy.py` 已自動偵測 raw_obs=83（加了分支 + `LidarStateExtractor(include_act_hist=raw==83)`）。訓練端 `modular_rnn_models.py` 早已內建 83D（`RAW_OBS_DIM=83`、`STATE_DIM=11`）。

**83D obs 佈局**：`[0:4]ego [4:6]goal [6:78]lidar(72) [78]time [79:83]action_stack`。網路雙分支：state_mlp=11D(ego4+goal2+time1+act_hist4)、lidar_conv=72D。

**四項 v3c 改動 vs SA6**：
1. obs 79/139 → 83（`obs_builder.build_obs_raw(action_history=)` 新增 83 分支）
2. action stacking 4D — policy_node 維護 deque、`act_stack_*` ROS param、`_push_act_hist` clip **[-2,2]**（對齊訓練 `discrete_applied_action_history`，**不是 [-1,1]**）。來源是 policy 自己上 2 步 decode 的 (actual_accel, cmd_w)，非硬體量測。
3. LiDAR r_min 0.9 → 0.25（純 yaml `lidar_preprocessor_params_v3c.yaml`）
4. 角速度 ω 上限 2.0→0.25π + α slew 3.0 rad/s²（`action_decoder.max_angular_accel`，0=舊行為；decode 傳 `current_angular_vel`）

**三個 ω 別混**（曾搞混）：obs[2]正規化=**1.5**（訓練 ObsTerm `base_angular_velocity_z max_angular_velocity=1.5`，charge_env_cfg_vlp16.py:389，已查證正確）；動作 ω 上限=**0.25π≈0.785**；act_stack 正規化=**π/15≈0.209**。動作 dim0=線加速度(積分→v_max=1.0)、dim1=角速度(直接映射→0.25π)。

**啟動**：`deploy_with_bev.launch.py params_file:=...policy_params_v3c.yaml preprocessor_params_file:=...lidar_preprocessor_params_v3c.yaml`。speed_rate 建議 1.0（slew/action_stack 僅 rate=1 精確對齊）。`episode_horizon_s=60.0` 暫沿用、待驗證（只影響 obs[78]）。

詳見 repo 內 `V3C_DEPLOY.md` + `CLAUDE.md` v3c 段。相關：[[feedback_parallel_claude_sessions_git]]
