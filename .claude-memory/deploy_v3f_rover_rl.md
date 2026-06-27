---
name: deploy_v3f_rover_rl
description: rover_rl v3f 部署 (sa4_v3f)：79D obs（act_hist 整段移除）；新增 deploy_rl_shell v3f 分支 + *_v3f.yaml；校準同 v3e
metadata: 
  node_type: memory
  type: project
  originSessionId: 2901dbf6-f93d-429f-95ac-68029cd8f4fe
---

aa@192.168.3.14:~/rover_rl 部署 sa4_v3f（2026-06-22）。

**v3f vs v3e/v3c 關鍵差別**：v3f 是 **79D obs**（act_hist 整段移除，訓練 `CHARGE_USE_ACT_HIST=0`），
v3e/v3c 都是 83D。架構其餘相同：RNN hidden=64 / fc=64 / middle=48 / preprocess=12 / predict=13 / logits=38。

**車端 79D 自動相容**：`policy_node.py` 用 `use_act_stack = (bundle.raw_obs_dim == 83)`，
偵測 79D 會自動關閉 action stacking / act_hist，故 v3f config 裡的 act_stack_* / act_hist_* 參數不作用（保留無害）。
`raw_obs_dim` 來自 .ts 的 `_meta_dims` buffer（順序 [raw, used, hidden, preprocess, total_logits]），不是直接屬性。

**轉換流程（必須在 PC-A IsaacLab 跑，遠端無 modular_rnn_models.py）**：
```
cd ~/IsaacLab/rover_rl
PYTHONPATH=src/rover_rl_inference python -m rover_rl_inference.export_policy \
  --checkpoint ~/IsaacLab/logs/rnn_car/sa4_v3f_ne1024_s42/checkpoint_240000.pt \
  --output /tmp/.../sa4_v3f_240000.ts
# 產出 .ts + .obs_spec.json + .obs_spec.md（export 自動偵測 raw_obs_dim=79 identity slice）
# 再 scp 三檔到 aa@192.168.3.14:~/rover_rl/models/
```

**本次新建/改動的車端檔**（皆有 .bak）：
- `models/sa4_v3f_240000.{ts,obs_spec.json,obs_spec.md}`
- `src/rover_rl_bringup/config/policy_params_v3f.yaml`（複製 v3e + model_path→sa4_v3f + v3f banner；r_min=0.25 / 動作 ω_max=1.2 / obs ω 正規化=1.5 同 v3e）
- `src/rover_rl_bringup/config/lidar_preprocessor_params_v3f.yaml`（與 v3e 逐字相同，r_min=0.25）
- `deploy_rl_shell.sh` 加 `*v3f*` 兩處 case（選單 tag + VARIANT 偵測）→ 選到自動帶 *_v3f.yaml + 印 obs_spec.md

**啟動**：`deploy_rl_shell` → 選單選 `sa4_v3f_240000.ts`（#5）→ 自動帶 v3f config。
與 [[deploy_v3c_rover_rl]] / [[deploy_v3e_rover_rl]] 同管線。ω_max 三者勿混見 [[finding_v3c_omega_max_decode_bug]]。
SA4 完訓診斷見 [[project_sa4_v3f_completion_diagnosis]]。
