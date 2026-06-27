---
name: project_v3d_launch
description: v3d 抗 sin 波抽動三件套 — action_error obs 編碼 + dropout + curriculum floor，2026-06-12 從 SA1 從頭訓
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

v3d 起因：v3c SA2/SA3 播放 ckpt，目標在正前方仍走 sin 波；`jitter/per_env_ratio_flip_rate_p95` 全程卡 50-56% RED。診斷為三因子共謀：(a) act_hist raw 編碼讓 policy 學成「copy 上一步 action」shortcut；(b) SA3 curriculum 把 ent_ang sync 到 0.01 → angular 分布過尖鎖死 limit cycle；(c) penalty_smoothness=0.005 太弱。

**三件套修法（2026-06-12 從 SA1_v3d 從頭訓，run=sa1_v3d_ne1024_s42）：**
1. **act_hist action_error 編碼**（env var `CHARGE_ACT_HIST_MODE=action_error`，train/play 必須一致）。4D 尾巴從 `[a_{t-1},ω_{t-1},a_{t-2},ω_{t-2}]` 改成 `[a_{t-1},ω_{t-1},err_lin,err_ang]`。obs 仍 83D，模型不用 reshape。
2. **act_hist dropout 0.2**（train-only mask，`act_hist_dropout` config 欄位 → `LidarStateExtractor`）。
3. **curriculum `warp_drive_single_agent_v3d`**：penalty_smoothness 0.005→0.015 + entropy_angular floor 0.02（全 stage）。是 v3 的 deep-copy patch，v3 未被污染。

**關鍵 obs-design 洞見（用戶 + GPT 討論後確認）**：policy 要看「上一指令」還是「實際車速」？答案是兩者，但 **ego[1:3] 已含實際 robot root velocity/omega（= odom 等價，physics 量測）**，act_hist 已含上一指令（applied_accelerations）。所以 GPT 建議的 6 維裡有 4 維已存在，**唯一缺的是 action_error**（指令 vs 實際的落差）。err 在 action term 算（actuator DR 前的意圖 − DR 後實際寫入 sim），不在 obs builder 減，避免和 obs_delay（感測延遲）混在一起。err≈0 時無法被 copy → 天生斷 shortcut；err≠0 = 馬達沒跟上 = 顯式致動延遲簽名（論文 §34 訓練端建模延遲）。

**⚠️ 2026-06-15 推翻：三件套「成功」是假象。** 長 goal play 仍 sin 波。決定性對照（sa3_v2 baseline）證明 **flip rate 是反指標**（高 flip=快抖乾淨、低 flip=慢 weave sin 波），下方「deterministic p95_flip 0.127=乾淨」結論看反了。真正根因是 **penalty_smoothness（罰 Δratio=罰 flip→逼出慢 weave）+ act_hist**。修法=移除這些非加懲罰。詳見 [[finding_action_error_selfref_oscillation]]。

**（以下為 2026-06-13 當時的誤判驗收，保留供對照）三件套「成功」，sin 波解決 ✅**
- **關鍵教訓**：訓練端 p95_flip（採樣 policy）全程 ~40-50% RED，但那是 entropy 假象（SA1 用加倍 ent_ang=0.04 撐高探索）。真實驗收必須用 **deterministic（argmax）rollout**，見 [[finding_p95flip_entropy_confound]]。
- deterministic eval p95_flip：iter100 **0.000** / iter300 **1.0%** / 完訓(iter900-equiv) **6.7%** —— 全部 << 0.15「視覺乾淨」門檻。部署 policy 完全不左右抖。
- 主任務飽和 SR ~96% / CR ~4%。
- **訓練曾在 iter ~647 CUDA wedged stall**（卡在 `_grad_l2_norm` 的 `.item()` sync）→ kill + 從 checkpoint_180000.pt resume 跑完最後 300 iters（run=sa1_v3d_ne1024_s42_resume，最終 ckpt checkpoint_90000.pt）。見 [[feedback_stall_autorecover]]。
- **SA2_v3d 動態場景驗證成功（2026-06-14）**：從 sa1_v3d_ne1024_s42_resume/checkpoint_90000.pt 續訓（run=sa2_v3d_ne1024_s42, 1400 iters, stage 2 動態障礙+static, LiDAR noise ON 50%）。iter-200 ckpt 含動態障礙 deterministic eval：**p95_flip 7.9%（<< 0.15 乾淨）、SR 97.7% / CR 2.2%**。→ **三件套在靜態(6.7%)+動態(7.9%)都成功，v3c 在 SA2 爆 sin 波的問題徹底解決。** 訓練端 p95_flip 全程 ~50% RED 是 entropy 假象。
- 後續：SA2 完訓 → SA3+ 更密集動態，沿用三件套 + curriculum v3d。

**SA2_v3d 第二次 stall + SA3 自動鏈（2026-06-15）**：sa2_v3d_ne1024_s42 在 iter 1200（ckpt 360000，目標 420000）再次 CUDA wedge stall（同樣卡 `_grad_l2_norm` train_rnn_car_wdclip.py:685，log 凍結 ~10h、GPU 0%、py-spy 三採同行）。用戶失聯 → 依 [[feedback_stall_autorecover]] 授權自動 SIGKILL + 從 360000 resume，run=**sa2_v3d_ne1024_s42_resume**（剩 200 iter / 60000 ts → 完訓 ckpt = `sa2_v3d_ne1024_s42_resume/checkpoint_60000.pt`）。resume 後 iter[1/200] SR 93.1%/CR 4.0% 健康。
- **SA3 自動啟動鏈**：detached watcher `/tmp/chain_sa2_to_sa3_v3d.sh`（log `/tmp/chain_sa2_to_sa3.log`）→ 偵測 SA2 resume "Training complete" + ckpt 60000 存在 → 跑最終動態 deterministic eval（stage 2）→ 自動啟動 **SA3_v3d**（`/tmp/launch_sa3_v3d.sh`，run=sa3_v3d_ne1024_s42）。watcher 內建 stall(>25min+GPU0%)/crash 自動 re-resume。
- **wd_sa3_v3d.yaml ckpt 路徑已改**：原指向 `sa2_v3d_ne1024_s42/checkpoint_420000.pt`（因 stall 永不存在）→ 改 `sa2_v3d_ne1024_s42_resume/checkpoint_60000.pt`。SA3 = stage 3（牆壁穿越+密集動態），v3d 對全 stage floor ent_ang 0.02 + smoothness 0.015，正是修 v3c 在 SA3 爆 sin 波的元兇。
- 監控 cron 換成 81c0e871（v3d 訓練鏈監控，每 15 分）。
- **鏈執行成功（2026-06-15 15:10）**：SA2 resume 完訓「Training complete: 60,000 steps」iter200 SR 97.1%/CR 2.9%；chain watcher 自動跑最終動態 deterministic eval → **p95_flip mean 0.100 / p95 0.127 / max 0.140 < 0.15 視覺乾淨 ✅**（|omega|_std 0.445）；自動啟動 **SA3_v3d**（run=sa3_v3d_ne1024_s42, ckpt sa2_..._resume/checkpoint_60000.pt, 1400 iter）。SA3 iter1 stage3 SR 61.2%/CR 33.2%（harder stage cold-start，預期）；**ent sync (0.005, 0.02)，ent_ang=0.02 floored 生效 ✅ — 正是 v3c 在 SA3 sync 到 0.01 鎖死 limit cycle 的修復**。

新增工具：`play_rnn_car.py --jitter_eval --deterministic`（量測部署 policy 真實角度翻轉率，去 entropy 噪聲）。

（原驗收標準 iter 300 p95_flip < 30% 是針對採樣指標，被 entropy 污染不適用，已由 deterministic eval 取代。）

改動檔案：`discrete_differential_drive.py`(_actuator_tracking_error)、`obs_functions.py`(action_error mode)、`modular_rnn_models.py`(act_hist_dropout)、`experiment_config.py`、`train_rnn_car_wdclip.py`、新 `wd_single_agent_v3d.py` + `wd_sa1_v3d.yaml`。SA3_v3c 已 kill。相關：[[finding_cmd_delay_limit_cycle]] [[project_v3_launch]] [[feedback_per_env_sampling]]
