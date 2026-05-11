#!/bin/bash
# 4-way TBPTT ablation: step vs tbptt × normal vs zero_feature
# Sequential — only 1 Isaac Sim instance at a time
set -e

COMMON="--task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 2048 --headless --seed 1 \
  --timesteps 500000 --rollout_length 64 \
  --use_a2c --ppo_epochs 1 \
  --lr 0.0002 --rnn_lr 0.0005 --aux_lr 0.0 \
  --gamma 0.984 --vf_coeff 0.025 \
  --curriculum_version warp_drive_single_agent_v1 \
  --log_interval 5 --save_interval 200"

echo "========== [1/4] step + normal =========="
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \
  $COMMON --aux_mode step \
  --run_name abl_step_normal_s1

echo "========== [2/4] tbptt + normal =========="
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \
  $COMMON --aux_mode tbptt --aux_seq_len 15 --aux_seq_batch_size 256 \
  --run_name abl_tbptt_normal_s1

echo "========== [3/4] step + zero_feature =========="
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \
  $COMMON --aux_mode step --zero_preprocess_feature_for_rl \
  --run_name abl_step_zero_s1

echo "========== [4/4] tbptt + zero_feature =========="
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \
  $COMMON --aux_mode tbptt --aux_seq_len 15 --aux_seq_batch_size 256 \
  --zero_preprocess_feature_for_rl \
  --run_name abl_tbptt_zero_s1

echo "========== All 4 runs complete =========="
