#!/bin/bash
# validate_checkpoint.sh — e2e checkpoint 晉級四閘驗收(用戶 2026-07-15 協定)
# 用法: bash validate_checkpoint.sh <ckpt.pt> [stage=1] [wandb_id=gtefxx15]
# GUI 可視化: GUI=1 CAMERA=top NUM_ENVS=4 bash validate_checkpoint.sh ...
#   跑 gate#2(3-seed held-out det) + gate#4(SA(N+1) preview)；SA5 起另跑
#   gate#3(可解的路徑擋路場 det+dump)與 OBB gate#5(1.2m 部署窄縫)，
#   並另跑 1.0m 壓力診斷（不影響晉級），
#   再呼叫 validate_gates.py 拉 gate#1(wandb 健康) + 逐 gate 判 PASS/FAIL。
# 注意: play 自動從 ckpt 偵測 e2e 架構(frame_stack/end_to_end);e2e 不加 --feat_norm。
set -euo pipefail
CKPT="$1"; STAGE="${2:-1}"; WID="${3:-gtefxx15}"
REPO=/home/aa/IsaacLab
CONDA_ENV=/home/aa/miniconda3/envs/env_isaaclab
PY="$CONDA_ENV/bin/python"
GATES_PY="$REPO/scripts/reinforcement_learning/skrl/rnn_car_wdclean/validate_gates.py"
PLAY="$REPO/scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
CURR=warp_drive_e2e_final20_v1
RUN_NAME="$(basename "$(dirname "$CKPT")")"
NUM_ENVS="${NUM_ENVS:-64}"
OUT_SUFFIX=""
RENDER_ARGS=(--headless)
if [ "${GUI:-0}" = "1" ]; then
  RENDER_ARGS=(--camera "${CAMERA:-top}")
  OUT_SUFFIX="_gui${NUM_ENVS}"
fi
OUT="/tmp/validate_${RUN_NAME}_$(basename "$CKPT" .pt)_s${STAGE}${OUT_SUFFIX}"
mkdir -p "$OUT"
cd "$REPO"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV=env_isaaclab
export PATH="$CONDA_ENV/bin:$PATH"
if ! env -u PYTHONPATH "$PY" -c 'import yaml' >/dev/null 2>&1; then
  echo "env_isaaclab 缺少可獨立載入的 PyYAML: $PY" >&2
  echo "請安裝至該環境，不可依賴外部 ROS PYTHONPATH。" >&2
  exit 78
fi
[ -f "$CKPT" ] || { echo "找不到 $CKPT"; exit 2; }
LOCK_KEY="$(printf '%s\n' "$CKPT|$STAGE" | sha256sum | cut -d' ' -f1)"
LOCK_FILE="/tmp/isaaclab_validate_${LOCK_KEY}.lock"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  echo "同一 checkpoint/stage 的驗收已在執行: $CKPT stage=$STAGE" >&2
  exit 75
fi
USE_OBB="$($PY - "$CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = ck.get("args", {})
a = a if isinstance(a, dict) else vars(a)
print("1" if a.get("use_obb_collision", False) else "0")
PY
)"

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
RUN_ADVANCED_GATES=0
if [ "$STAGE" -ge 5 ]; then
  RUN_ADVANCED_GATES=1
fi

DUMP=""   # 只有 gate#3 設;其餘空
_play() {  # _play <out.log> <scene args...>
  local log="$1"; shift
  echo "  ▶ $(basename "$log") ..."
  PYTHONUNBUFFERED=1 CHARGE_LIDAR_DUMP="$DUMP" \
    ./isaaclab.sh -p "$PLAY" \
    --checkpoint "$CKPT" --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
    --curriculum_version "$CURR" --deterministic --num_envs "$NUM_ENVS" --steps 1200 "${RENDER_ARGS[@]}" \
    --num_goals_override 1 --no_goal_movement "$@" > "$log" 2>&1
  if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:' "$log"; then
    echo "  ✗ $(basename "$log") simulator traceback; see $log" >&2
    return 1
  fi
  if [ -n "$DUMP" ] && [ ! -s "$DUMP" ]; then
    echo "  ✗ $(basename "$log") did not produce required dump $DUMP" >&2
    return 1
  fi
}

echo "=== 驗收 $CKPT (Stage $STAGE) → $OUT ==="

# Gate #2: 3 held-out seeds (SA 場, seed≠訓練42)
for SD in 101 202 303; do
  _play "$OUT/det_s${SD}.log" \
    --stage "$STAGE" --arena_size "$A" --num_static_obs "$S" --num_dynamic_obs "$D" \
    --obs_near_goal_count 0 --seed "$SD" \
    --solvability_audit_output "$OUT/solvability_s${SD}.json"
done

# Gate #3: 受控可解擋路場。兩側長牆內表面間距 4.0m；blocker 每回合
# 在 x∈[1.2,3.8], y∈[-1,+1] 重採樣，最窄側仍有 0.65m 淨空，
# 起點/goal 邊緣淨空至少 0.5m。預設 50% 靜止、50% 以 0.3m/s
# 隨機 2D heading 巡邏並在安全邊界反射；沒有舊版第三面壓力牆。
BLK_ARGS=()
if [ "$RUN_ADVANCED_GATES" = "1" ]; then
  DUMP="$OUT/blk_dump.npz"
  _play "$OUT/blk.log" \
    --stage "$STAGE" --arena_size "$A" --num_static_obs "$S" --num_dynamic_obs "$D" \
    --goal_distance_min 5 --goal_distance_max 9 --obs_near_goal_count 0 \
    --controlled_blocker_eval --seed 42
  DUMP=""
  BLK_ARGS=(--blk_log "$OUT/blk.log" --blk_dump "$OUT/blk_dump.npz")
else
  echo "  ↷ Gate3 deferred until SA5 (current SA${STAGE})"
fi

# Gate #4: SA(N+1) preview (不訓練,直接丟下一階段場)
PREVIEW_ARG=""
if [ "$PST" != "0" ]; then
  _play "$OUT/preview.log" \
    --stage "$PST" --arena_size "$A2" --num_static_obs "$S2" --num_dynamic_obs "$D2" \
    --obs_near_goal_count 0 --seed 42
  PREVIEW_ARG="--preview_log $OUT/preview.log"
fi

# Gate #5: OBB-lineage checkpoints must pass the deployment-representative
# 1.2 m opening. The 1.0 m opening is a stress diagnostic and cannot fail
# advancement by itself. The 0.85 m case remains a geometry test, not a policy gate.
NARROW_ARGS=()
if [ "$USE_OBB" = "1" ] && [ "$RUN_ADVANCED_GATES" = "1" ]; then
  _play "$OUT/narrow_deploy_1p2.log" \
    --stage "$STAGE" --arena_size 10 --num_static_obs 0 --num_dynamic_obs 0 \
    --obs_near_goal_count 0 --narrow_gap_eval --narrow_gap_width 1.2 \
    --narrow_gap_yaw_limit_deg 10 --seed 404
  _play "$OUT/narrow_stress_1p0.log" \
    --stage "$STAGE" --arena_size 10 --num_static_obs 0 --num_dynamic_obs 0 \
    --obs_near_goal_count 0 --narrow_gap_eval --narrow_gap_width 1.0 \
    --narrow_gap_yaw_limit_deg 10 --seed 405
  NARROW_ARGS=(
    --narrow_deploy_log "$OUT/narrow_deploy_1p2.log"
    --narrow_stress_log "$OUT/narrow_stress_1p0.log"
  )
elif [ "$USE_OBB" = "1" ]; then
  echo "  ↷ Gate5 deferred until SA5 (current SA${STAGE})"
fi

# 分析 + 逐 gate 判定
echo
$PY "$GATES_PY" --stage "$STAGE" --wandb_id "$WID" --ckpt "$CKPT" \
  --det_glob "$OUT/det_s*.log" "${BLK_ARGS[@]}" $PREVIEW_ARG "${NARROW_ARGS[@]}"
