#!/usr/bin/env bash
# 每 10 分鐘產生一份訓練詳細監督報告。
#
# 與 cron_training_supervisor.sh 併存、互不修改：
#   cron_training_supervisor.sh  → :00 :10 :20 …  簡短存活/ALERT + auto-advance tick
#   cron_detail_report.sh        → :05 :15 :25 …  完整指標/分場景/趨勢/判定
# 兩者錯開 5 分鐘，等效於每 5 分鐘有一次檢查。
#
# 本腳本只回報，不會停止任何訓練。

set -uo pipefail

# cron 的 PATH 極簡，nvidia-smi / pgrep 都找不到，必須自己補。
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO="/home/aa/IsaacLab"
PYTHON="/home/aa/miniconda3/envs/env_isaaclab/bin/python"
SCRIPT="$REPO/scripts/reinforcement_learning/skrl/monitoring/detail_report.py"
LOCK_FILE="/tmp/isaaclab_detail_report.lock"

# 前一輪還沒跑完就直接跳過，不要疊加。
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

exec "$PYTHON" "$SCRIPT" "$@"
