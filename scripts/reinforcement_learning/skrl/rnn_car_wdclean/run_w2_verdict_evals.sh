#!/usr/bin/env bash
# 07-27 二次裁決指定的評測佇列（全部零訓練，GPU 循序）：
#   1) narrow teacher 的 Gate5 direct 指標（teacher 是否自己也繞行）
#   2) W2-c10 起始距離 1.5/2.0/3.0m Gate5 sweep（close→far curriculum 可行性）
#   3) c10 mixed_iid seed 818 補跑（使 c10→c20 為固定 seeds 515/616/818 比較）
#   4) s1 seed 複驗的 c2..c20 Gate5 screening
set -uo pipefail

REPO="/home/aa/IsaacLab"
PY="/home/aa/miniconda3/envs/env_isaaclab/bin/python"
WD="$REPO/scripts/reinforcement_learning/skrl/rnn_car_wdclean"
TEACHER="$REPO/logs/rnn_car/sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
C10="$REPO/logs/rnn_car/sa6_k8_obb_corridor_w2_wander_lrdecay_s42/checkpoint_1280.pt"
S1_RUN="$REPO/logs/rnn_car/sa6_k8_obb_corridor_w2_wander_lrdecay_s1"

echo "[VERDICT-EVAL] 1/4 teacher Gate5 (3 seeds)"
PYTHONUNBUFFERED=1 "$PY" "$WD/run_narrow_path_suite.py" "$TEACHER" \
    --output-dir "$REPO/logs/gates/w2/teacher_narrow/narrow_path" --seeds 404,505,606 \
    || echo "[VERDICT-EVAL] teacher gate exit $?"

echo "[VERDICT-EVAL] 2/4 c10 start-distance sweep"
for dist in 1.5 2.0 3.0; do
    log="$REPO/logs/gates/w2/c10/narrow_dist_sweep/dist_${dist}.log"
    mkdir -p "$(dirname "$log")"
    if grep -aq "NARROW-GAP-METRICS" "$log" 2>/dev/null; then
        echo "[VERDICT-EVAL] dist=$dist already done"; continue
    fi
    echo "[VERDICT-EVAL] dist=${dist}m seed=404"
    PYTHONUNBUFFERED=1 "$REPO/isaaclab.sh" -p \
        "$REPO/scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py" \
        --checkpoint "$C10" \
        --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
        --curriculum_version warp_drive_e2e_final20_v1 \
        --deterministic --num_envs 64 --steps 1200 --headless \
        --num_goals_override 1 --no_goal_movement \
        --stage 5 --arena_size 10 --num_static_obs 0 --num_dynamic_obs 0 \
        --obs_near_goal_count 0 \
        --narrow_gap_eval --narrow_gap_width 1.2 --narrow_gap_yaw_limit_deg 10 \
        --narrow_gap_start_distance "$dist" --seed 404 \
        > "$log" 2>&1 || echo "[VERDICT-EVAL] dist=$dist exit $?"
    grep -a "NARROW-GAP-METRICS" "$log" | tail -1
done

echo "[VERDICT-EVAL] 3/4 c10 mixed_iid s818 top-up"
if [ ! -f "$REPO/logs/gates/w2/c10/full_mixed_s818/corridor_motion_suite.json" ]; then
    PYTHONUNBUFFERED=1 "$PY" "$WD/run_corridor_motion_suite.py" "$C10" \
        --output-dir "$REPO/logs/gates/w2/c10/full_mixed_s818" \
        --profile deployment_wander_v1 --modes mixed_iid --seeds 818 \
        || echo "[VERDICT-EVAL] c10 mixed s818 exit $?"
else
    echo "[VERDICT-EVAL] c10 mixed s818 already done"
fi

echo "[VERDICT-EVAL] 4/4 s1 Gate5 screening"
W2_RUN_DIR="$S1_RUN" W2_GATE_PREFIX="w2_s1" bash "$WD/run_w2_gate5_screen.sh"

echo "[VERDICT-EVAL] all done"
