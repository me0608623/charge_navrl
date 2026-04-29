#!/bin/bash
# ============================================================================
# WD Phase 1 Pilot — TBPTT aux training @ 5Hz (Pilot Tier: 1,800 iter)
# ============================================================================
#
# 目的：以 WD train_rnn_car Phase 1 的比例做一個 10% pilot，驗證
#       TBPTT + WD-style sparse reward + 7D aux target 在 Isaac Lab 5Hz 下是否可行。
#
# 環境準備：
#   conda activate env_isaaclab
#   # 目前最穩的啟動方式是直接用 conda env 的 python
#
# Batch 對齊：
#   WD Phase 1: 50K batch (208 envs × 240 steps @ 4Hz)
#   本腳本:     50,400 batch (168 envs × 300 steps @ 5Hz)
#   300 steps = 60 seconds @ 5Hz（對應 WD 的 60s episode）
#
# ─── WD Phase 1 Budget 三層對照 ───
#   Sanity:     300 outer iter →  90K timesteps (run_wd_p1_tbptt_env_scaling_auto.sh)
#   Pilot:    1,800 outer iter → 540K timesteps (本腳本, ~10% WD budget)
#   WD-local: 18,000 trainer iter = train_num_env(180) × env_train_times(100)
#             其中 spot updates ~12,000, spot agent-steps ~600M
#
# 公式：
#   WD Phase 1 full budget = train_num_env(180) × env_train_times(100) = 18,000 trainer iter
#   WD spot agent-steps = 12,000 spot updates × 50K batch ≈ 600M
#   本腳本 pilot:  1,800 updates × 50.4K batch ≈ 90.7M agent-steps (≈ WD 的 10%)
#   timesteps = 540,000 → 540000 / 300 = 1,800 iterations
# ─────────────────────────────────
#
# 不自動升級 stage：--fixed_stage 鎖定 Stage 1
# ============================================================================
set -e

PYTHON=/home/aa/miniconda3/envs/env_isaaclab/bin/python

PYTHONUNBUFFERED=1 $PYTHON scripts/reinforcement_learning/skrl/train_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --curriculum_version warp_drive_v1 \
  --initial_stage 1 \
  --fixed_stage \
  --num_envs 168 \
  --rollout_length 300 \
  --timesteps 540000 \
  --mini_batches 12 \
  --aux_mode tbptt \
  --aux_seq_len 15 \
  --aux_seq_batch_size 256 \
  --use_a2c \
  --ppo_epochs 1 \
  --lr 0.0002 \
  --rnn_lr 0.0005 \
  --aux_lr 0.0 \
  --vf_coeff 0.025 \
  --gamma 0.984 \
  --seed 1 \
  --headless \
  --log_interval 5 \
  --save_interval 200 \
  --run_name wd_p1_tbptt_pilot_5hz
