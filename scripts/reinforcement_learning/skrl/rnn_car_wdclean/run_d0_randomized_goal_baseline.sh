#!/usr/bin/env bash
# N1 前置量測：D0 在「隨機化目標」窄縫 replay 上的零射基準。
#
# 為什麼需要：N1 把窄縫目標從固定的「缺口正對面 3.0m」改成隨機
# （距離 2–4m、橫偏 ±1.5m）。既有的 D0 / W1 / W2 成績全部是在共線目標下量的，
# 與 N1 不可比。沒有這個基準，就無法判定 N1 的通過條件
# （crossing ≥95% / direct ≥95%，以及 random_2d/mixed_iid 不得比 D0 惡化 >2pp）。
#
# 與 teacher 驗證的唯一差別：不帶 --narrow_scripted_teacher，
# 也就是由 D0 policy 自己開，量它在新分佈下真實的起點成績。
set -uo pipefail

REPO="/home/aa/IsaacLab"
D0="$REPO/logs/rnn_car/sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
OUT="$REPO/logs/gates/n1/d0_randomized_goal_baseline"
SEEDS="${D0_BASELINE_SEEDS:-404 505 606}"
GOAL_DIST_RANGE="${N1_GOAL_DIST_RANGE:-2.0 4.0}"
GOAL_DY_RANGE="${N1_GOAL_DY_RANGE:--1.5 1.5}"
mkdir -p "$OUT"

# 基準分佈必須與 N1 config 相符，否則量到的不是 N1 的起點。
PYTHONPATH="$REPO/scripts/reinforcement_learning/skrl" \
"/home/aa/miniconda3/envs/env_isaaclab/bin/python" - "$GOAL_DIST_RANGE" "$GOAL_DY_RANGE" <<'PY' || exit 1
import sys
from rnn_car_modular.configs.registry import get_experiment_config
cfg = get_experiment_config("e2e_sa6_n1_direct_imitation_from_d0")
want_d = tuple(float(v) for v in sys.argv[1].split())
want_o = tuple(float(v) for v in sys.argv[2].split())
got_d = tuple(cfg.narrow_passage_goal_distance_range or ())
got_o = tuple(cfg.narrow_passage_goal_lateral_offset_range or ())
if got_d != want_d or got_o != want_o:
    sys.exit(f"基準分佈與 N1 config 不符：{want_d}/{want_o} vs {got_d}/{got_o}")
print(f"[D0-BASELINE] 目標分佈對齊 config: dist={got_d} m, dy={got_o} m")
PY

for seed in $SEEDS; do
    log="$OUT/d0_randgoal_s${seed}.log"
    if grep -aq "NARROW-REPLAY-METRICS" "$log" 2>/dev/null; then
        echo "[D0-BASELINE] seed=$seed already done"
    else
        echo "[D0-BASELINE] seed=$seed"
        PYTHONUNBUFFERED=1 "$REPO/isaaclab.sh" -p \
            "$REPO/scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py" \
            --checkpoint "$D0" \
            --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
            --curriculum_version warp_drive_e2e_final20_v1 \
            --deterministic --num_envs 64 --steps 1200 --headless \
            --num_goals_override 1 --no_goal_movement \
            --stage 6 --arena_size 14 \
            --num_static_obs 0 --num_dynamic_obs 0 --obs_near_goal_count 0 \
            --narrow_replay_eval \
            --narrow_replay_width_range 1.2 1.4 \
            --narrow_replay_yaw_limit_deg 4.0 \
            --narrow_replay_goal_distance_range $GOAL_DIST_RANGE \
            --narrow_replay_goal_lateral_offset_range $GOAL_DY_RANGE \
            --seed "$seed" \
            > "$log" 2>&1 || echo "[D0-BASELINE] seed=$seed exit $?"
    fi
    grep -ao "crossing_rate=[0-9.]* direct_crossed=[0-9]* direct_crossing_rate=[0-9.]*" "$log" | tail -1
    grep -a "成功率 (到達目標)\|碰撞率 (總計)" "$log" | tail -2
done

echo "[D0-BASELINE] done — 這是 N1 通過條件的比較基準"
