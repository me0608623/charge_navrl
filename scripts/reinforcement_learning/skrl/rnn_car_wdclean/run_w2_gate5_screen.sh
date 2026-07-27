#!/usr/bin/env bash
# W2 per-checkpoint Gate5 screening（Codex 07-27 協定步驟 3）
#
# 每顆 checkpoint (c2..c20) 跑 1.2m 窄縫 Gate5，direct crossing 為硬閘。
# 預設單 seed 404 篩選（與 W1-c10 基準同條件）；候選 ckpt 再手動加跑
# --seeds 404,505,606 三 seed 確認。c10/c20 另跑走廊四模式 + Gate2（不在本腳本）。
set -euo pipefail

REPO="/home/aa/IsaacLab"
RUN_DIR="${W2_RUN_DIR:-$REPO/logs/rnn_car/sa6_k8_obb_corridor_w2_wander_lrdecay_s42}"
GATE_PREFIX="${W2_GATE_PREFIX:-w2}"   # 輸出到 logs/gates/$GATE_PREFIX/c{N}/narrow_path
SUITE="$REPO/scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_narrow_path_suite.py"
PY="/home/aa/miniconda3/envs/env_isaaclab/bin/python"
SEEDS="${W2_GATE5_SEEDS:-404}"

# save_interval=2 × rollout 128 → checkpoint_256..2560 = c2..c20
for iters in 2 4 6 8 10 12 14 16 18 20; do
    steps=$((iters * 128))
    ckpt="$RUN_DIR/checkpoint_${steps}.pt"
    out="$REPO/logs/gates/$GATE_PREFIX/c${iters}/narrow_path"
    if [ ! -f "$ckpt" ]; then
        echo "[W2-GATE5] SKIP c${iters}: missing $ckpt"
        continue
    fi
    if [ -f "$out/narrow_path_suite.json" ]; then
        echo "[W2-GATE5] SKIP c${iters}: already done"
        continue
    fi
    echo "[W2-GATE5] c${iters} (checkpoint_${steps}) seeds=$SEEDS"
    # suite 以非零退出表示 gate FAIL — 記錄但不中斷 screening
    PYTHONUNBUFFERED=1 "$PY" "$SUITE" "$ckpt" --output-dir "$out" --seeds "$SEEDS" \
        || echo "[W2-GATE5] c${iters}: gate FAIL (exit $?)"
done

echo "[W2-GATE5] screening done. Summary:"
for iters in 2 4 6 8 10 12 14 16 18 20; do
    json="$REPO/logs/gates/$GATE_PREFIX/c${iters}/narrow_path/narrow_path_suite.json"
    [ -f "$json" ] && "$PY" -c "
import json
d = json.load(open('$json'))
a = d['aggregate']
print(f\"c${iters}: SR={a['sr']:.3f} CR={a['cr']:.3f} crossing={a['crossing_rate']:.3f} direct={a['direct_crossing_rate']:.3f} n={a['episodes']}\")"
done
