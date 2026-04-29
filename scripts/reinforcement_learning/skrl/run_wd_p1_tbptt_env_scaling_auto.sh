#!/bin/bash
# ============================================================================
# WD Phase 1 — num_envs Scaling Test (Sanity-Check Tier)
# ============================================================================
#
# Runs 4 experiments sequentially: num_envs = 168 / 224 / 256 / 320
# Each run: 300 iterations (sanity-check tier, ~1.7% of WD Phase 1 full budget)
# Estimated total: ~9-10 hours
#
# ─── WD Phase 1 Budget 三層對照 ───
#   Sanity:     300 outer iter → 90K timesteps  (本腳本，僅驗 GPU/VRAM/無 crash)
#   Pilot:    1,800 outer iter → 540K timesteps (run_wd_p1_tbptt_pilot.sh, ~10%)
#   WD-local: 18,000 trainer iter = train_num_env(180) × env_train_times(100)
#             其中 spot updates ~12,000, spot agent-steps ~600M
# ─────────────────────────────────
#
# Monitors GPU/VRAM every 30s, captures errors, produces per-run summaries
# and a final comparison report.
#
# Usage:
#   conda activate env_isaaclab
#   bash scripts/reinforcement_learning/skrl/run_wd_p1_tbptt_env_scaling_auto.sh
# ============================================================================
set -euo pipefail

PYTHON=/home/aa/miniconda3/envs/env_isaaclab/bin/python
SCRIPT=scripts/reinforcement_learning/skrl/train_rnn_car.py
LOGDIR=logs/env_scaling
mkdir -p "$LOGDIR"

# --- Common args (everything except --num_envs and --run_name) ---
COMMON_ARGS=(
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL
  --curriculum_version warp_drive_v1
  --initial_stage 1
  --fixed_stage
  --aux_mode tbptt
  --aux_seq_len 15
  --aux_seq_batch_size 256
  --use_a2c
  --ppo_epochs 1
  --lr 0.0002
  --rnn_lr 0.0005
  --aux_lr 0.0
  --vf_coeff 0.025
  --gamma 0.984
  --rollout_length 300
  --headless
  --seed 1
  --timesteps 90000
  --log_interval 5
  --save_interval 100
)

# --- Experiment configs: (num_envs run_name) ---
EXPERIMENTS=(
  "168 wd_p1_tbptt_env168"
  "224 wd_p1_tbptt_env224"
  "256 wd_p1_tbptt_env256"
  "320 wd_p1_tbptt_env320"
)

# --- Error patterns to scan ---
ERROR_PATTERNS="Traceback|CUDA out of memory|foundLostPairsCapacity|PxGpuDynamicsMemoryConfig|segmentation fault|NaN|hydra.errors|command not found"

# ============================================================================
# Helper: GPU monitor (background, writes CSV every 30s)
# ============================================================================
start_gpu_monitor() {
  local csv_file="$1"
  echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_draw_w" > "$csv_file"
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits 2>/dev/null >> "$csv_file"
    sleep 30
  done
}

# ============================================================================
# Helper: Generate per-run summary
# ============================================================================
generate_summary() {
  local run_name="$1"
  local num_envs="$2"
  local log_file="$LOGDIR/${run_name}.log"
  local gpu_file="$LOGDIR/${run_name}.gpu.csv"
  local summary_file="$LOGDIR/${run_name}.summary.md"
  local exit_code="$3"
  local duration="$4"

  # --- GPU stats ---
  local gpu_stats=""
  if [[ -f "$gpu_file" ]] && [[ $(wc -l < "$gpu_file") -gt 1 ]]; then
    gpu_stats=$($PYTHON -c "
import csv, sys
rows = []
with open('$gpu_file') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            rows.append({
                'util': float(r['gpu_util_pct'].strip()),
                'mem': float(r['mem_used_mib'].strip()),
                'mem_total': float(r['mem_total_mib'].strip()),
                'power': float(r['power_draw_w'].strip()),
            })
        except (ValueError, KeyError):
            pass
if not rows:
    print('No GPU data')
    sys.exit(0)
utils = [r['util'] for r in rows]
mems = [r['mem'] for r in rows]
powers = [r['power'] for r in rows]
mem_total = rows[0]['mem_total']
print(f'GPU util: mean={sum(utils)/len(utils):.1f}% peak={max(utils):.0f}%')
print(f'VRAM used: mean={sum(mems)/len(mems):.0f} MiB peak={max(mems):.0f} MiB / {mem_total:.0f} MiB')
print(f'Power: mean={sum(powers)/len(powers):.0f}W peak={max(powers):.0f}W')
print(f'Samples: {len(rows)}')
" 2>/dev/null || echo "GPU parse error")
  else
    gpu_stats="No GPU data collected"
  fi

  # --- Training stats: extract last CHARGE iteration line ---
  local last_charge_line=""
  local last_aux_line=""
  local first_charge_line=""
  if [[ -f "$log_file" ]]; then
    first_charge_line=$(grep -m1 'CHARGE' "$log_file" 2>/dev/null || echo "")
    last_charge_line=$(grep 'CHARGE' "$log_file" 2>/dev/null | tail -1 || echo "")
    last_aux_line=$(grep 'AUX(' "$log_file" 2>/dev/null | tail -1 || echo "")
  fi

  # --- Error scan ---
  local errors=""
  if [[ -f "$log_file" ]]; then
    errors=$(grep -cE "$ERROR_PATTERNS" "$log_file" 2>/dev/null || echo "0")
    local error_samples=""
    if [[ "$errors" != "0" ]]; then
      error_samples=$(grep -E "$ERROR_PATTERNS" "$log_file" 2>/dev/null | head -5)
    fi
  fi

  # --- Write summary ---
  cat > "$summary_file" << SUMEOF
# Summary: $run_name

## Config
- num_envs: $num_envs
- batch/update: $((num_envs * 300))
- rollout: 300 steps (60s @ 5Hz)
- timesteps: 90,000 -> 300 iterations (sanity-check tier, NOT WD-full budget)
- exit_code: $exit_code
- duration: ${duration}s

## GPU/VRAM
$gpu_stats

## Training
### First iteration
\`\`\`
$first_charge_line
\`\`\`

### Last iteration
\`\`\`
$last_charge_line
$last_aux_line
\`\`\`

## Errors/Warnings
- Error pattern matches: $errors
SUMEOF

  if [[ "$errors" != "0" ]] && [[ -n "$error_samples" ]]; then
    cat >> "$summary_file" << ERREOF
- Samples:
\`\`\`
$error_samples
\`\`\`
ERREOF
  fi

  echo "[SCALING] Summary written: $summary_file"
}

# ============================================================================
# Main loop
# ============================================================================
echo "============================================"
echo "WD Phase 1 num_envs Scaling Test (Sanity Tier: 300 iter)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo "============================================"

RESULTS=()

for exp in "${EXPERIMENTS[@]}"; do
  read -r NUM_ENVS RUN_NAME <<< "$exp"
  BATCH=$((NUM_ENVS * 300))

  echo ""
  echo "========== [$RUN_NAME] num_envs=$NUM_ENVS batch=$BATCH =========="
  echo "[SCALING] Starting at $(date '+%H:%M:%S')"

  LOG_FILE="$LOGDIR/${RUN_NAME}.log"
  GPU_FILE="$LOGDIR/${RUN_NAME}.gpu.csv"

  # Start GPU monitor in background
  start_gpu_monitor "$GPU_FILE" &
  GPU_PID=$!

  # Start training
  START_TIME=$(date +%s)
  set +e
  PYTHONUNBUFFERED=1 $PYTHON $SCRIPT \
    "${COMMON_ARGS[@]}" \
    --num_envs "$NUM_ENVS" \
    --run_name "$RUN_NAME" \
    > "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  set -e
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  # Stop GPU monitor
  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true

  echo "[SCALING] Finished $RUN_NAME: exit=$EXIT_CODE duration=${DURATION}s"

  # Generate summary
  generate_summary "$RUN_NAME" "$NUM_ENVS" "$EXIT_CODE" "$DURATION"

  RESULTS+=("$RUN_NAME:$NUM_ENVS:$EXIT_CODE:$DURATION")

  # Brief pause between runs to let GPU cool / release memory
  sleep 10
done

# ============================================================================
# Final report
# ============================================================================
echo ""
echo "========== Generating final report =========="

$PYTHON -c "
import csv, sys, os, re

logdir = '$LOGDIR'
experiments = [
    ('wd_p1_tbptt_env168', 168),
    ('wd_p1_tbptt_env224', 224),
    ('wd_p1_tbptt_env256', 256),
    ('wd_p1_tbptt_env320', 320),
]

report = []
report.append('# WD Phase 1 num_envs Scaling — Final Report (Sanity Tier: 300 iter)\n')
report.append(f'Generated: $(date \"+%Y-%m-%d %H:%M:%S\")\n')

# Per-run data
run_data = []
for run_name, num_envs in experiments:
    batch = num_envs * 300
    d = {'run_name': run_name, 'num_envs': num_envs, 'batch': batch}

    # GPU stats
    gpu_file = os.path.join(logdir, f'{run_name}.gpu.csv')
    if os.path.exists(gpu_file):
        rows = []
        with open(gpu_file) as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append({
                        'util': float(r['gpu_util_pct'].strip()),
                        'mem': float(r['mem_used_mib'].strip()),
                        'power': float(r['power_draw_w'].strip()),
                    })
                except (ValueError, KeyError):
                    pass
        if rows:
            utils = [r['util'] for r in rows]
            mems = [r['mem'] for r in rows]
            d['gpu_mean'] = f'{sum(utils)/len(utils):.1f}'
            d['gpu_peak'] = f'{max(utils):.0f}'
            d['vram_mean'] = f'{sum(mems)/len(mems):.0f}'
            d['vram_peak'] = f'{max(mems):.0f}'
        else:
            d['gpu_mean'] = d['gpu_peak'] = d['vram_mean'] = d['vram_peak'] = 'N/A'
    else:
        d['gpu_mean'] = d['gpu_peak'] = d['vram_mean'] = d['vram_peak'] = 'N/A'

    # Log stats
    log_file = os.path.join(logdir, f'{run_name}.log')
    d['last_iter'] = 'N/A'
    d['last_sr'] = 'N/A'
    d['last_cr'] = 'N/A'
    d['last_ent'] = 'N/A'
    d['last_ve'] = 'N/A'
    d['last_aux'] = 'N/A'
    d['last_rnn_delta'] = 'N/A'
    d['errors'] = 0
    d['exit_code'] = 'N/A'
    d['physx_warn'] = 'No'
    d['crash'] = 'No'

    if os.path.exists(log_file):
        with open(log_file) as f:
            content = f.read()

        # Error count
        error_pats = r'Traceback|CUDA out of memory|foundLostPairsCapacity|PxGpuDynamicsMemoryConfig|segmentation fault|NaN|hydra\.errors|command not found'
        d['errors'] = len(re.findall(error_pats, content))
        if 'foundLostPairsCapacity' in content or 'PxGpuDynamicsMemoryConfig' in content:
            d['physx_warn'] = 'Yes'
        if 'CUDA out of memory' in content or 'segmentation fault' in content.lower():
            d['crash'] = 'Yes'

        # Last CHARGE line
        charge_lines = re.findall(r'\[(\d+)/(\d+)\] CHARGE.*?SR=([\d.]+)%.*?CR=([\d.]+)%.*?TO=([\d.]+)%.*?ent=([\d.]+)', content)
        if charge_lines:
            last = charge_lines[-1]
            d['last_iter'] = f'{last[0]}/{last[1]}'
            d['last_sr'] = f'{last[2]}%'
            d['last_cr'] = f'{last[3]}%'
            d['last_ent'] = last[5]

        # Last AUX line
        aux_lines = re.findall(r'AUX\(.*?\): loss=([\d.eE+-]+).*?rnn_delta=([\d.eE+-]+).*?VE=([\d.eE+-]+)', content)
        if aux_lines:
            last_aux = aux_lines[-1]
            d['last_aux'] = last_aux[0]
            d['last_rnn_delta'] = last_aux[1]
            d['last_ve'] = last_aux[2]

    # Summary file for exit code
    summary_file = os.path.join(logdir, f'{run_name}.summary.md')
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            sc = f.read()
        m = re.search(r'exit_code:\s*(\d+)', sc)
        if m:
            d['exit_code'] = m.group(1)
        m2 = re.search(r'duration:\s*(\d+)s', sc)
        if m2:
            d['duration_min'] = f'{int(m2.group(1))/60:.1f}'
        else:
            d['duration_min'] = 'N/A'

    run_data.append(d)

# Build report
report.append('## 1. Configuration\n')
report.append('| run_name | num_envs | batch/update |')
report.append('|----------|----------|-------------|')
for d in run_data:
    report.append(f'| {d[\"run_name\"]} | {d[\"num_envs\"]} | {d[\"batch\"]:,} |')
report.append('')

report.append('## 2. Hardware Summary\n')
report.append('| run_name | GPU mean% | GPU peak% | VRAM mean | VRAM peak | PhysX warn | Crash/OOM | Duration |')
report.append('|----------|-----------|-----------|-----------|-----------|------------|-----------|----------|')
for d in run_data:
    report.append(f'| {d[\"run_name\"]} | {d[\"gpu_mean\"]} | {d[\"gpu_peak\"]} | {d[\"vram_mean\"]} MiB | {d[\"vram_peak\"]} MiB | {d[\"physx_warn\"]} | {d[\"crash\"]} | {d.get(\"duration_min\",\"N/A\")} min |')
report.append('')

report.append('## 3. Training Summary\n')
report.append('| run_name | last iter | SR | CR | entropy | VE | aux loss | rnn_delta | exit |')
report.append('|----------|-----------|-----|-----|---------|-----|----------|-----------|------|')
for d in run_data:
    report.append(f'| {d[\"run_name\"]} | {d[\"last_iter\"]} | {d[\"last_sr\"]} | {d[\"last_cr\"]} | {d[\"last_ent\"]} | {d[\"last_ve\"]} | {d[\"last_aux\"]} | {d[\"last_rnn_delta\"]} | {d[\"exit_code\"]} |')
report.append('')

report.append('## 4. Conclusion\n')
report.append('(Auto-generated placeholder — review manually)\n')

# Find best SR
srs = []
for d in run_data:
    try:
        srs.append((d['run_name'], float(d['last_sr'].replace('%',''))))
    except (ValueError, AttributeError):
        srs.append((d['run_name'], -1))
srs.sort(key=lambda x: -x[1])
if srs and srs[0][1] > 0:
    report.append(f'- Best SR: **{srs[0][0]}** ({srs[0][1]:.1f}%)')
report.append('')

# Write
report_path = os.path.join(logdir, 'final_report.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(report))
print(f'Final report: {report_path}')
" 2>&1

echo ""
echo "============================================"
echo "All experiments complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $LOGDIR/"
echo "============================================"
