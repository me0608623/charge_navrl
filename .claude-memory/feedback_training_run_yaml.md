---
name: training_run_yaml_metadata
description: 每次啟動或監控訓練時，必須在 checkpoint 目錄產生 run_metadata.yaml 記錄完整訓練設計
type: feedback
---

每次啟動訓練或開始監控訓練時，必須在 `logs/rnn_car/{run_name}/run_metadata.yaml` 產生一份完整的訓練元資料檔案。

**Why:** 使用者要求所有訓練都要有可追溯的完整記錄，避免事後遺忘實驗意義與參數。checkpoint 目錄只有 .pt 檔案不夠，需要 YAML 記錄「為什麼跑這次訓練」。

**How to apply:**
- 觸發時機：啟動訓練 (`nohup` / `bash`) 或首次監控一個新 run 時
- 檔案路徑：`logs/rnn_car/{run_name}/run_metadata.yaml`（與 checkpoint_*.pt 同目錄）
- 若目錄內已有 `run_metadata.yaml`，不要覆蓋，除非使用者要求更新

**YAML 必須包含的欄位：**

```yaml
run_name: "..."
purpose: "此次訓練的目的與意義（中文）"
date: "YYYY-MM-DD"
script: "train_rnn_car.py / train_rnn_car_wdclip.py / ..."
status: "running / completed / killed / hung"

# 環境與硬體
environment:
  num_envs: 512
  rollout_length: 300
  timesteps: 450000
  iterations: 1500  # = timesteps // rollout_length
  seed: 1
  headless: true
  gpu: "RTX 5090"

# RL 演算法
algorithm:
  type: "A2C"  # or PPO
  ppo_epochs: 1
  lr: 0.0002
  gamma: 0.984
  vf_coeff: 0.025
  max_grad_norm: 0.5

# 神經網路架構
network:
  encoder_mode: "raw_fc_rnn"  # or 3branch_vlp16
  obs_dim: 79  # raw_fc_rnn 的輸入維度
  rnn_type: "GRU"
  rnn_hidden: 128
  output_dim: 12
  policy_head: "Categorical 361 (19x19)"
  value_head: "linear"

# Auxiliary 訓練
auxiliary:
  mode: "tbptt"
  seq_len: 15
  seq_batch_size: 256
  rnn_lr: 0.0005
  aux_lr_predict_head: 0
  aux_lr_fc_middle: 0
  aux_lr_fc_front: 0
  aux_lr_extractor: 0  # or N/A for raw_fc_rnn
  target: "7D next-step positions of nearest 2 obstacles"

# 動作空間
action_space:
  type: "MultiDiscrete([19, 19])"
  linear_accel_range: "[-0.5, 0.5] m/s²"
  angular_vel_range: "[-0.25π, 0.25π] rad/s"
  v_max: 1.0
  dt: 0.2

# 獎勵設計
reward:
  type: "WD sparse reward"
  description: "Warp Drive style sparse reward with goal/collision/time terminals"

# Curriculum
curriculum:
  version: "warp_drive_single_agent_v1"
  initial_stage: 1
  fixed_stage: true

# 預期效果
expected_outcome: "（中文描述預期目標、成功標準、與其他 run 的比較）"

# 實際結果（訓練完成後填寫）
actual_outcome:
  final_iter: null
  SR: null
  CR: null
  VE: null
  Return: null
  verdict: null  # "成功 / 失敗 / 可參考"
```
