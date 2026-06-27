---
name: finding_v3c_omega_max_decode_bug
description: v3c 車端部署 ω_max decode 不一致 — 訓練 1.2 但車端誤設 0.785 (0.25π)，policy 角速度意圖被縮成 65%
metadata: 
  node_type: memory
  type: project
  originSessionId: 152f93e4-d03b-4af1-9051-68ff6f8bbce9
---

v3c 模型（sa1/sa2/sa3_v3c，含匯出的 .ts）的角速度 decode 在訓練端與車端不一致，疑似 sim-to-real bug。

**訓練端實際 ω_max = 1.2 rad/s**（完整證據鏈）：
- WD task `Isaac-Navigation-Charge-VLP16-Curriculum-WD` → env cfg `charge_env_cfg_wd_sparse.py:ChargeNavigationEnvCfgVLP16CurriculumWD`，繼承 `ChargeNavigationEnvCfgVLP16Curriculum`（`max_angular_vel=1.2`，`cfg/charge_env_cfg_vlp16_curriculum.py:948`，2026-06-02 由 2.0→1.2 對齊馬達）。WD 子類、v3 curriculum stages、`train_rnn_car_wdclip.py` 都**未覆寫**。
- `play_eval/play_rnn_car.py:2251` decode 讀 `cfg.max_angular_vel` = 1.2。
- 注意 `DiscreteDifferentialDriveActionCfg` 的 **class default 是 0.25π≈0.785**，但被 env cfg 覆寫成 1.2 — 不要被 class default 騙（我和車端都犯過此錯）。

**車端誤設 ω_max = 0.785（0.25π），兩處同一誤值**：
- `rover_rl/src/rover_rl_bringup/config/policy_params_v3c.yaml:55` → `act_max_angular_velocity: 0.7853981634`（policy_node 真正 decode 用）。
- `rover2_ws/.../campusrover_navigation/campusrover_rl_policy/scripts/charge_skrl_adapter.py:72` → `ACTION_OMEGA_MAX = 0.25 * math.pi`。
- 兩處註解都誤稱「v3c 訓練值」；`rover_rl/CLAUDE.md` 與 `rover_rl/models/sa*_v3c_obs_spec.md` 也都誤寫 0.25π。

**後果**：policy 輸出 ratio r_ω∈[-1,1]，ω=r_ω·ω_max。車端 0.785/1.2≈0.654 → 同指令只給 65% 角速度，系統性少轉約 35%（轉彎偏鈍）。底盤 `profile_omega_max=1.2`（`driver_chgh.yaml:24`），故改回 1.2 既對上訓練值又在底盤內。

**三個 ω 尺度勿混**（只有第 2 個是 bug）：(1) obs[2] 正規化分母 1.5（`base_angular_velocity_z`，`charge_env_cfg_vlp16.py:389`）；(2) 動作 decode ω_max：訓練 1.2 vs 車端誤設 0.785 ← bug；(3) act_hist 正規化分母 π/15≈0.209。

**車端其他演算法角速度上限**（對照）：TEB 0.3 < DWA 0.5 < RL v3c 0.785 < path_following 1.0 ≈ 底盤 1.2 < 舊RL(SA6) 2.0。0.785 不離譜（夾在 DWA 與 path_following 間），但沒對上 RL 訓練值。

**修法**：車端把 `policy_params_v3c.yaml:55` 與 `charge_skrl_adapter.py:72` 的 0.785 改回 1.2 並修註解。取捨：1.2 忠實還原訓練但轉彎比操作員習慣的 DWA(0.5)更猛。

**✅ 2026-06-17 修復（rover_rl 路徑）**：車端 `~/rover_rl/src/rover_rl_bringup/config/policy_params_v3c.yaml:55` 已改 0.7853981634 → **1.2**（PC-A 本機 repo 早已是 1.2，車端這次補上；`speed_rate=0.5` 車端值保留不動）。deploy_rl_shell 選單 v3c tag 同步改顯示 ω_max=1.2。v3e yaml 一開始就設 1.2。
⚠️ **另一處 `rover2_ws/.../charge_skrl_adapter.py:72`（ACTION_OMEGA_MAX=0.25π）尚未修** — 那是舊 campusrover_rl_policy adapter 路徑，rover_rl 部署（deploy_rl_shell→policy_node）不走它；若改用該 adapter 才需一起改。

完整分析寫在 Obsidian：`isaaclab_v2/環境配置與觀測/v3e-觀測與獎勵改動全紀錄 — v2 到 v3e 演進.md` §2.0.6。相關 [[finding_cmd_delay_limit_cycle]]、[[deploy_v3e_rover_rl]]。
