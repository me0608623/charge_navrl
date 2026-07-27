#!/usr/bin/env bash
# N1 步驟 2：scripted 直穿 teacher 在「正式隨機窄縫 replay」上的三 seed 驗證。
#
# 與固定 Gate5 的差別：gap 中心 y∈[-1,1]、barrier x∈[-0.5,0.5]、左右方向各半、
# 初始位置與 yaw 隨機。通過門檻（Codex 07-27 裁決）：
#   direct_crossing_rate >= 0.99
#   collision_rate       <= 0.01
#
# ★ 目標分佈必須與 N1 訓練一致。N1 config 已改為隨機化目標
#   （距離 2.0–4.0 m、橫偏 ±1.5 m），不再是歷史的「缺口正對面 3.0 m」。
#   用共線目標驗證會漏測 teacher 的兩段瞄準（過牆前瞄縫、過牆後轉向目標），
#   而那正是偏軸目標下唯一會被考驗的邏輯。
#   覆寫用 N1_GOAL_DIST_RANGE / N1_GOAL_DY_RANGE（需與 config 同步）。
#
# checkpoint 只用來滿足 play 的載入路徑，動作全程由 scripted teacher 接管。
set -uo pipefail

REPO="/home/aa/IsaacLab"
CKPT="${TEACHER_CHECK_CKPT:-$REPO/logs/rnn_car/sa6_k8_obb_corridor_w2_wander_lrdecay_s42/checkpoint_1280.pt}"
OUT="$REPO/logs/gates/n1/teacher_replay_check"
SEEDS="${TEACHER_CHECK_SEEDS:-404 505 606}"
# 預設對齊 e2e_sa6_n1_direct_imitation_from_d0 的 narrow_passage_goal_*_range
GOAL_DIST_RANGE="${N1_GOAL_DIST_RANGE:-2.0 4.0}"
GOAL_DY_RANGE="${N1_GOAL_DY_RANGE:--1.5 1.5}"
mkdir -p "$OUT"

# 驗證分佈必須與 N1 config 相符，否則驗的不是要訓練的東西。
PYTHONPATH="$REPO/scripts/reinforcement_learning/skrl" \
"/home/aa/miniconda3/envs/env_isaaclab/bin/python" - "$GOAL_DIST_RANGE" "$GOAL_DY_RANGE" <<'PY' || exit 1
import sys
from rnn_car_modular.configs.registry import get_experiment_config
cfg = get_experiment_config("e2e_sa6_n1_direct_imitation_from_d0")
want_d = tuple(float(v) for v in sys.argv[1].split())
want_o = tuple(float(v) for v in sys.argv[2].split())
got_d = cfg.narrow_passage_goal_distance_range
got_o = cfg.narrow_passage_goal_lateral_offset_range
if tuple(got_d or ()) != want_d or tuple(got_o or ()) != want_o:
    sys.exit(
        f"驗證分佈與 N1 config 不符：script dist={want_d} dy={want_o} "
        f"vs config dist={got_d} dy={got_o}"
    )
print(f"[N1-TEACHER-CHECK] 目標分佈對齊 config: dist={got_d} m, dy={got_o} m")
PY

for seed in $SEEDS; do
    log="$OUT/replay_s${seed}.log"
    if grep -aq "NARROW-REPLAY-METRICS" "$log" 2>/dev/null; then
        echo "[N1-TEACHER-CHECK] seed=$seed already done"
    else
        echo "[N1-TEACHER-CHECK] seed=$seed"
        PYTHONUNBUFFERED=1 "$REPO/isaaclab.sh" -p \
            "$REPO/scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py" \
            --checkpoint "$CKPT" \
            --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
            --curriculum_version warp_drive_e2e_final20_v1 \
            --deterministic --num_envs 64 --steps 1200 --headless \
            --num_goals_override 1 --no_goal_movement \
            --stage 6 --arena_size 14 \
            --num_static_obs 0 --num_dynamic_obs 0 --obs_near_goal_count 0 \
            --narrow_replay_eval --narrow_scripted_teacher \
            --narrow_replay_width_range 1.2 1.4 \
            --narrow_replay_yaw_limit_deg 4.0 \
            --narrow_replay_goal_distance_range $GOAL_DIST_RANGE \
            --narrow_replay_goal_lateral_offset_range $GOAL_DY_RANGE \
            --seed "$seed" \
            > "$log" 2>&1 || echo "[N1-TEACHER-CHECK] seed=$seed exit $?"
    fi
    grep -a "NARROW-REPLAY-METRICS" "$log" | tail -1
    grep -a "成功率 (到達目標)\|碰撞率 (總計)" "$log" | tail -2
done

echo "[N1-TEACHER-CHECK] done — 門檻 direct>=0.99 / collision<=0.01"
