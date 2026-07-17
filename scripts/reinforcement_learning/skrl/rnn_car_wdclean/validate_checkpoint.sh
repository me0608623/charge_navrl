#!/bin/bash
# validate_checkpoint.sh — e2e checkpoint 晉級四閘驗收(用戶 2026-07-15 協定)
# 用法: bash validate_checkpoint.sh <ckpt.pt> [stage=1] [wandb_id=gtefxx15]
#   跑 gate#2(3-seed held-out det) + gate#3(可解的路徑擋路場 det+dump) + gate#4(SA(N+1) preview),
#   再呼叫 validate_gates.py 拉 gate#1(wandb 健康) + 逐 gate 判 PASS/FAIL。
# 注意: play 自動從 ckpt 偵測 e2e 架構(frame_stack/end_to_end);e2e 不加 --feat_norm。
set -euo pipefail
CKPT="$1"; STAGE="${2:-1}"; WID="${3:-gtefxx15}"
REPO=/home/aa/IsaacLab
PY=/home/aa/miniconda3/envs/env_isaaclab/bin/python
GATES_PY="$REPO/scripts/reinforcement_learning/skrl/rnn_car_wdclean/validate_gates.py"
PLAY="$REPO/scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
CURR=warp_drive_e2e_final20_v1
RUN_NAME="$(basename "$(dirname "$CKPT")")"
OUT="/tmp/validate_${RUN_NAME}_$(basename "$CKPT" .pt)_s${STAGE}"
mkdir -p "$OUT"
cd "$REPO"; source /home/aa/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
[ -f "$CKPT" ] || { echo "找不到 $CKPT"; exit 2; }

# 逐階段場景(arena/靜/動 + 下一階段 preview 場)
case "$STAGE" in
  1) A=20; S=2;  D=0; A2=18; S2=4;  D2=0; PST=2;;
  2) A=18; S=4;  D=0; A2=17; S2=6;  D2=1; PST=3;;
  3) A=17; S=6;  D=1; A2=16; S2=8;  D2=2; PST=4;;
  4) A=16; S=8;  D=2; A2=15; S2=10; D2=3; PST=5;;
  5) A=15; S=10; D=3; A2=14; S2=12; D2=4; PST=6;;
  6) A=14; S=12; D=4; A2=13; S2=13; D2=5; PST=7;;
  7) A=13; S=13; D=5; A2=12; S2=14; D2=6; PST=8;;
  8) A=12; S=14; D=6; A2=0;  S2=0;  D2=0; PST=0;;
  *) echo "不支援 Stage $STAGE（只接受 1..8）"; exit 2;;
esac

DUMP=""   # 只有 gate#3 設;其餘空
_play() {  # _play <out.log> <scene args...>
  local log="$1"; shift
  echo "  ▶ $(basename "$log") ..."
  PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 CHARGE_LIDAR_DUMP="$DUMP" \
    ./isaaclab.sh -p "$PLAY" \
    --checkpoint "$CKPT" --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
    --curriculum_version "$CURR" --deterministic --num_envs 64 --steps 1200 --headless \
    --num_goals_override 1 --no_goal_movement "$@" > "$log" 2>&1
}

echo "=== 驗收 $CKPT (Stage $STAGE) → $OUT ==="

# Gate #2: 3 held-out seeds (SA 場, seed≠訓練42)
for SD in 101 202 303; do
  _play "$OUT/det_s${SD}.log" \
    --stage "$STAGE" --arena_size "$A" --num_static_obs "$S" --num_dynamic_obs "$D" \
    --obs_near_goal_count 0 --seed "$SD"
done

# Gate #3: 擋路場（單障礙放在 robot→goal 線上 1.8m，goal 在其後）+ dump
# 不使用 obs_near_goal：它可能把障礙放進 goal 可達區，製造不可解場景。
DUMP="$OUT/blk_dump.npz"
_play "$OUT/blk.log" \
  --stage "$STAGE" --arena_size "$A" --num_static_obs "$S" --num_dynamic_obs "$D" \
  --goal_distance_min 5 --goal_distance_max 9 --obs_near_goal_count 0 \
  --path_blocker_count 1 --path_blocker_distance 1.8 --seed 42
DUMP=""

# Gate #4: SA(N+1) preview (不訓練,直接丟下一階段場)
PREVIEW_ARG=""
if [ "$PST" != "0" ]; then
  _play "$OUT/preview.log" \
    --stage "$PST" --arena_size "$A2" --num_static_obs "$S2" --num_dynamic_obs "$D2" \
    --obs_near_goal_count 0 --seed 42
  PREVIEW_ARG="--preview_log $OUT/preview.log"
fi

# 分析 + 逐 gate 判定
echo
$PY "$GATES_PY" --stage "$STAGE" --wandb_id "$WID" --ckpt "$CKPT" \
  --det_glob "$OUT/det_s*.log" --blk_log "$OUT/blk.log" --blk_dump "$OUT/blk_dump.npz" $PREVIEW_ARG
