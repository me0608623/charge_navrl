#!/usr/bin/env bash
set -uo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO="/home/aa/IsaacLab"
STATE_DIR="$REPO/logs/training_supervisor"
REPORT="$STATE_DIR/cron_10min.log"
EXPECTED_RUN_FILE="$STATE_DIR/expected_run.txt"
STATUS_FILE="$STATE_DIR/status.txt"
LOCK_FILE="/tmp/isaaclab_training_supervisor.lock"
AUTO_SUPERVISOR="$REPO/scripts/reinforcement_learning/skrl/monitoring/auto_advance_supervisor.py"
AUTO_STATE="$STATE_DIR/auto_advance_state.json"
PYTHON="/home/aa/miniconda3/envs/env_isaaclab/bin/python"

resolve_training_log() {
  local run_name="$1"
  local console_log="$REPO/logs/rnn_car/${run_name}.console.log"
  local legacy_log="/tmp/${run_name}.log"

  if [[ -f "$console_log" ]]; then
    printf '%s\n' "$console_log"
  elif [[ -f "$legacy_log" ]]; then
    printf '%s\n' "$legacy_log"
  else
    printf '%s\n' "$console_log"
  fi
}

mkdir -p "$STATE_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

timestamp="$(date '+%F %T %Z')"
expected_run=""
status="UNSPECIFIED"
[[ -f "$EXPECTED_RUN_FILE" ]] && expected_run="$(tr -d '\r\n' < "$EXPECTED_RUN_FILE")"
[[ -f "$STATUS_FILE" ]] && status="$(tr '\n' ' ' < "$STATUS_FILE" | sed 's/[[:space:]]\+/ /g; s/[[:space:]]$//')"

{
  printf '\n=== %s ===\n' "$timestamp"
  printf 'expected_run=%s status=%s\n' "${expected_run:-<none>}" "$status"

  mapfile -t train_pids < <(
    pgrep -f '^/[^ ]*/python(3([.][0-9]+)?)? ([-][^ ]+ )*[^ ]*train_rnn_car_wdclip[.]py( |$)' || true
  )
  if ((${#train_pids[@]} == 0)); then
    if [[ -n "$expected_run" ]]; then
      printf 'ALERT expected training process is not running\n'
    else
      printf 'IDLE no training process expected\n'
    fi
  else
    for pid in "${train_pids[@]}"; do
      cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
      run_name="$(sed -n 's/.*--run_name[= ]\([^ ]*\).*/\1/p' <<< "$cmd")"
      printf 'PROCESS '
      ps -p "$pid" -o pid=,stat=,etime=,%cpu=,%mem= 2>/dev/null || true
      printf 'run_name=%s\n' "${run_name:-<unknown>}"

      if [[ -n "$expected_run" && "$run_name" != "$expected_run" ]]; then
        printf 'ALERT active run does not match expected_run\n'
      fi

      log_file="$(resolve_training_log "$run_name")"
      if [[ -n "$run_name" && -f "$log_file" ]]; then
        progress="$(rg -a '\[[[:space:]]*[0-9]+/[0-9]+\].*fps=' "$log_file" | tail -n 1 || true)"
        printf 'LATEST %s\n' "${progress:-<no rollout line>}"
        anomaly="$(tail -n 500 "$log_file" | rg -ai 'traceback|cuda out of memory|\bOOM\b|runtimeerror|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' | tail -n 3 || true)"
        if [[ -n "$anomaly" ]]; then
          printf 'ALERT anomaly in recent log:\n%s\n' "$anomaly"
        else
          printf 'health=no recent NaN/OOM/traceback\n'
        fi
      else
        printf 'WARN training log not found at %s\n' "$log_file"
      fi

      if [[ -n "$run_name" && -d "$REPO/logs/rnn_car/$run_name" ]]; then
        latest_ckpt="$(find "$REPO/logs/rnn_car/$run_name" -maxdepth 1 -name 'checkpoint_*.pt' -printf '%T@ %f\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
        printf 'checkpoint=%s\n' "${latest_ckpt:-<none>}"
      fi
    done
  fi

  gpu_rows="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$gpu_rows" ]]; then
    printf 'GPU processes:\n%s\n' "$gpu_rows"
  else
    printf 'GPU processes=<none or unavailable>\n'
  fi

  if [[ -f "$AUTO_STATE" ]]; then
    printf 'AUTO_ADVANCE tick:\n'
    "$PYTHON" "$AUTO_SUPERVISOR" tick || printf 'ALERT auto-advance tick failed rc=%s\n' "$?"
  else
    printf 'AUTO_ADVANCE not initialized\n'
  fi
} >> "$REPORT" 2>&1
