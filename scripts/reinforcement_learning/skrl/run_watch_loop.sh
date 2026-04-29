#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-logs/ablation_4way.log}"
OUT_FILE="${2:-logs/run1_watch.log}"
INTERVAL="${3:-60}"

cd /home/aa/IsaacLab

while true; do
  printf "\n[%s]\n" "$(date '+%F %T')" >> "${OUT_FILE}"
  /home/aa/miniconda3/envs/env_isaaclab/bin/python \
    scripts/reinforcement_learning/skrl/analyze_wandb_run.py \
    monitor-log --log-file "${LOG_FILE}" --tail 120 >> "${OUT_FILE}" 2>&1
  sleep "${INTERVAL}"
done
