---
name: project-sa1-sim2real-v1-20260728
description: 2026-07-28 從零重訓的 SA1 sim-to-real 主線；同時納入窄縫、長走廊、實測 VLP-16 雜訊與 0/200/400ms 致動延遲
metadata:
  node_type: memory
  type: project
  status: active
  date: 2026-07-28
---

# SA1 sim-to-real v1：目前權威訓練主線

## 決策

W1-c10 的實車 TorchScript `w1c10_k8_e2e_1280.ts` 出現舞龍舞獅、突然大腳
速度／轉向與撞牆，禁止再當部署候選。新主線不從 W1、SA7、pilot checkpoint
續訓，而是從 SA1 隨機初始化，讓 policy 從第一步就共同學習：

- 原生 SA1 場景；
- 1.2--1.4 m 窄縫；
- 4 m × 10 m 長走廊與動態障礙；
- 實測 VLP-16 LiDAR 雜訊；
- 致動命令延遲 U{0,1,2} steps = 0/200/400 ms。

觀測延遲 `obs_delay_steps=(0,0)` 維持關閉。延遲加在解碼後的 `(v, ω)` 命令，
不是延遲 observation，也不是延遲離散 action index。

## 正式配方

- Config：`e2e_sa1_k8_obb_sim2real_v1`
- 1024 env、seed 42、rollout length 128、PPO LR 2e-4。
- Reset 指派：native 78%、narrow 12%、long corridor 10%。
- Narrow：淨寬 1.2--1.4 m、barrier length 10 m、yaw ±4°。
- Corridor：4 m × 10 m；density 25/35/20/15/5：
  `3S1D/4S2D/4S3D/5S3D/5S5D`。
- Motion：`env_stratified`；`random_2d=wander`；gate-aligned share 0。
- VLP-16 `full` measured noise：sigma 0.008672 m、dropout 0.194859、
  mixed pixel 0.002515、per-ring bias on。
- Actuator DR：delay U{0,1,2}；velocity scale `(1.0,1.0)`；
  motor lag alpha 1.0（未量測的量保持中性）。
- Policy observation：83D；K=8 E2E LiDAR frame stack；兩步 issued-action
  history 4D，正規化尺度 `(0.5,1.2)`。

注意：78/12/10 是每次 reset 的指派比例，不等於 PPO env-step 梯度比例。pilot
中因窄縫與走廊 episode 較長，瞬時 active mix 曾約為 42/32/26。這不是指派 bug，
但正式解讀 retention 時必須納入。

## RNN 決策

目前主線不是 recurrent policy。歷史命名仍含 `rnn_car`，checkpoint 也可能含
RNN/aux 模組，但 policy head 的實際輸入為 `83 + 96 = 179D`，不含 hidden state；
RNN/aux LR 全為 0。部署不應維護虛假的 RNN state。

先使用 K=8 疊幀加 action history。只有成熟 policy 的成對實驗證明時間資訊不足，
才新增 GRU ablation；不可在本次正式 run 中途改架構。

## 已修接線與測試

- Trainer 原先漏呼叫 `_apply_actuator_dr_config`，導致 train 無延遲、play 有延遲；
  已修並以 trainer/play 對稱性測試守住。
- 補齊 narrow/corridor scene mix，修正 scheduler active=2、capacity=10。
- Narrow barrier 由 9 m 對齊 20 m SA1 房間為 10 m。
- 修正 83D history 的 legacy normalization saturation。
- 測試：347 passed + 27 subtests；corridor events 75 passed + 995 subtests。
- 256-env wiring smoke：場景、雜訊、延遲與 audit 全通過。

## Pilot-30

- Run：`sa1_sim2real_v1_ne1024_s42_pilot30_20260728`
- W&B：`4p4dfa1z`
- 30/30 iterations，3,840 steps，974 s。
- Iter 30：SR 80.6%、CR 7.6%、TO 12.1%；last-5 mean 80.1/8.7/11.1%。
- 500 corridor assignments：density exact、family 1/3 each、crossing 94/94、
  side 34/34、wander 375/375、spacing/speed violations 0。
- 無 traceback、RuntimeError、NaN、OOM 或 overflow dump。
- Pilot 只證明學習與接線健康；其中六顆 checkpoint 全部禁止部署。

## 正式 R1（RUNNING）

- Run：`sa1_sim2real_v1_ne1024_s42_r1`
- systemd user unit：`sa1-sim2real-v1-r1.service`
- W&B：`gxtcgj3y`
- 啟動：2026-07-28 20:21:55 CST。
- Budget：`timesteps=270000`；完整 rollout 實際為 2109 iterations /
  269,952 steps；每 50 iter 存檔。
- Fresh model + optimizer；未載入 W1、SA7 或 pilot checkpoint；無平行 eval。
- 首批 runtime audit：corridor 100/1024、narrow 125/1024，兩者可解性 100%；
  density exact；crossing 13/13；side 8/8；wander 75/75。
- 累計 n=500：family 1/3 each、density exact、crossing 87/87、side 43/43、
  wander 375/375、`s_spacing/s_speed_delta=0/0`。
- `loginctl enable-linger aa` 已於 2026-07-28 啟用並確認 `Linger=yes`；
  登出不再終止 user systemd 與本 run。

## 完訓後 gates

Training complete 不等於可部署。成熟 checkpoint 必須依序通過：

1. 固定 delay `d=0`、`d=1`、`d=2` 分開報告，不得只報混合平均；
2. native、窄縫、四種走廊模式的 deterministic gates；
3. action stability：滿舵率、角速度翻轉、slew gap、突發大腳命令；
4. Python 與 TorchScript action/decoder parity；
5. 車端 staged acceptance：架高輪／低速空曠／牆邊／窄縫／走廊。

Claude 已由使用者回報完成車端 361 組 decoder parity、83D history、TF guard
與安全鏈；本訓練 repo 尚未獨立重跑車端測試，所以記為「使用者回報」，不是本端
驗證證據。

## 權威文件

- `docs/sa1_sim2real_v1_pilot30_result_20260728.md`
- `docs/freeze/sa1_sim2real_v1_pilot30_source_20260728.md`
- `docs/freeze/sa1_sim2real_v1_full_r1_20260728.md`
- Obsidian：
  `isaaclab/訓練報告/2026-07-28_SA1_sim2real_v1_正式訓練.md`

