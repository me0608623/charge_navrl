---
name: deploy_v3e_rover_rl
description: rover_rl 部署 v3e (sa1/sa2_v3e) — 83D 同 v3c 架構，新增 v3e config + deploy_rl_shell 選單支援；act_hist 語義差異待驗
metadata: 
  node_type: memory
  type: project
  originSessionId: b146bb33-145c-41da-ace5-02bfe75255cb
---

2026-06-17 把 v3e 模型部署到 aa@192.168.3.14（hostname ubuntu，home /home/aa）的 rover_rl。

**模型**（export_policy.py 自動偵測 raw_obs=83，與 v3c 同架構）：
- `sa2_v3e_240000.ts` — 動態場景最終（resume/checkpoint_240000.pt），預設
- `sa1_v3e_60000.ts` — 靜態（sa1_v3e_ne1024_s42_resume/checkpoint_60000.pt，=SA2 起點）
- 架構：raw=83 / used=83 / hidden=64 / preprocess=12 / fc=64 / middle=48 / predict=13 / RNN，logits=38。state_mlp 11D。
- 已在車端 torch 2.10.0+cpu 驗證 jit.load + forward OK（logits (1,38)）。

**新增 config**（車端 `~/rover_rl/src/rover_rl_bringup/config/`，也在 PC repo 內版控）：
- `policy_params_v3e.yaml`、`lidar_preprocessor_params_v3e.yaml`
- 與 v3c 逐字相同，**唯一差異 act_max_angular_velocity = 1.2**（修正 v3c 0.785 bug，見 [[finding_v3c_omega_max_decode_bug]]；底盤 profile_omega_max=1.2）。r_min 0.25、speed_rate 0.5、act_stack 同 v3c。

**deploy_rl_shell.sh**：把原本只認 `*v3c*` 的選單泛化成 variant 偵測（v3c/v3e → 自動帶 `*_<variant>.yaml`）。車端版本含 VO 安全層 prompt（PC-A HEAD 沒有），故是「下載車端檔→patch→上傳」保留 VO 段，非直接覆蓋（[[feedback_parallel_claude_sessions_git]]）。原檔備份 `deploy_rl_shell.sh.bak_v3e_*`。

**act_hist 語義已做成「隨模型走」（2026-06-17 補完）**：
- **export_policy.py** 產生 sidecar `<model>.obs_spec.json`+`.obs_spec.md`（act_hist_mode 由 experiment_config 推：v3d/v3e→action_error、其餘→raw；可 `--act-hist-mode` 覆寫）。已為 sa1/sa2_v3e（action_error）+ sa2/sa3_v3c（raw）產生。
- **deploy_rl_shell** 選完 checkpoint 會 cat obs_spec.md 印「觀測維度語義列表」。
- **policy_node** 新增 `_resolve_act_hist_mode`：載入模型時讀同名 obs_spec.json 設 `_act_hist_mode`（param `act_hist_mode=auto`）。`_act_hist_flat` 分模式：raw=[a_{t-1},ω_{t-1},a_{t-2},ω_{t-2}]；action_error=[a_{t-1},ω_{t-1},err_v,err_w]。
- **err 計算**（用戶選 measured 預設、可切 zero）：err = (上拍 policy-frame 指令 − odom 實測×inv)/v_max，clamp[-1,1]；v_max=1.0、ω_max=1.2（對齊訓練 `actuator_tracking_error` 正規化）。1 步延遲 = odom 實測本身是上拍指令的實現結果。param `act_hist_err_source` 可 measured/zero 現場切。
- v3c 行為 byte 不變（sidecar=raw → 走原 concat 分支）；79D 模型 use_act_stack=False 不受影響。
- ⚠️ 仍須架空驗證：measured err 是 odom 回授，[[finding_action_error_selfref_oscillation]] 警告其振盪風險；不對勁就 `ros2 param set /rover_rl_policy act_hist_err_source zero`。
- symlink-install → 改 policy_node.py / yaml 不用 rebuild，下次 launch 生效。

**啟動**：`deploy_rl_shell` → 選 sa2_v3e_240000.ts（印 [v3e ω_max=1.2] + 觀測語義表）→ 自動帶 v3e config，policy_node 自動 action_error 模式。
