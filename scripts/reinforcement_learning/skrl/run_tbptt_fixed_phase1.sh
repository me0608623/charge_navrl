#!/bin/bash
# Run B: TBPTT + WD sparse reward + fixed WarpDrive-style Phase 1.
# This isolates TBPTT from IsaacLab auto-promotion curriculum effects.
set -e

PYTHONUNBUFFERED=1 /home/aa/miniconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/train_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 512 --headless --seed 1 \
  --timesteps 500000 --rollout_length 256 \
  --use_a2c --ppo_epochs 1 \
  --lr 0.0002 --rnn_lr 0.0005 --aux_lr 0.0 \
  --gamma 0.984 --vf_coeff 0.025 \
  --curriculum_version warp_drive_single_agent_v1 \
  --initial_stage 1 --fixed_stage \
  --aux_mode tbptt --aux_seq_len 15 --aux_seq_batch_size 256 \
  --log_interval 5 --save_interval 200 \
  --run_name runB_tbptt_wd_sparse_fixed_p1_s1
