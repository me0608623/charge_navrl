#!/usr/bin/env bash
# N1 步驟 5+6：啟動直穿模仿臂，並在 seed42 通過後跑 seed1 複驗。
#
# ⚠️ 前置條件（腳本會硬性檢查，未滿足直接拒跑）：
#   1. scripted teacher 已在「正式隨機窄縫 replay」通過三 seed 驗證
#      （direct >= 0.99、collision <= 0.01）—— run_narrow_replay_teacher_check.sh
#   2. λ 已由 shadow rollout 校準 —— run_n1_shadow_calibration.sh，
#      且 config 的 _LAMBDA_IS_CALIBRATED 已設為 True
#
# 用法：
#   bash run_n1_arm.sh            # seed 42
#   N1_SEED=1 bash run_n1_arm.sh  # seed 1 複驗（seed42 通過後才跑）
set -uo pipefail

REPO="/home/aa/IsaacLab"
PY="/home/aa/miniconda3/envs/env_isaaclab/bin/python"
SEED="${N1_SEED:-42}"
RUN_NAME="sa6_k8_obb_corridor_n1_direct_imitation_s${SEED}"
cd "$REPO"

# --- 前置檢查 0：這條臂是否已被裁決封鎖 ---
if ! PYTHONPATH="$REPO/scripts/reinforcement_learning/skrl" "$PY" -c "
from rnn_car_modular.configs import e2e_sa6_n1_direct_imitation_from_d0 as m
import sys
if getattr(m, '_ARM_IS_BLOCKED', False):
    sys.exit(f'N1 已被封鎖：{m._BLOCKED_REASON}')
print('[N1] 臂未被封鎖')
"; then
    echo "[N1] 拒絕啟動：前提已被 2026-07-27 裁決推翻。" >&2
    echo "     封死的 Gate5a 上 D0 已達 direct 1.000（n=2304、零碰撞、橫偏中位 0.052m），" >&2
    echo "     此臂會在 SA6/14m 教一個模型本來就做得完美的動作。" >&2
    echo "     依裁決，直穿模仿延到 SA8/12m 情境，且僅在該處 Gate5a 仍失敗時才啟動。" >&2
    exit 1
fi

# --- 前置檢查 1：λ 必須已校準 ---
if ! PYTHONPATH="$REPO/scripts/reinforcement_learning/skrl" "$PY" -c "
from rnn_car_modular.configs import e2e_sa6_n1_direct_imitation_from_d0 as m
import sys
if not m._LAMBDA_IS_CALIBRATED:
    sys.exit('λ 尚未校準：先跑 run_n1_shadow_calibration.sh，把量到的 '
             'lambda_for_half_ppo 填進 _LAMBDA 並設 _LAMBDA_IS_CALIBRATED=True')
print(f'[N1] λ = {m._LAMBDA} (calibrated)')
"; then
    echo "[N1] 拒絕啟動：λ 未校準（裁決禁止憑空猜權重）" >&2
    exit 1
fi

# --- 前置檢查 2：teacher 必須已在隨機 replay 上通過 ---
CHECK_DIR="$REPO/logs/gates/n1/teacher_replay_check"
if ! PYTHONPATH="$REPO/scripts/reinforcement_learning/skrl" "$PY" - "$CHECK_DIR" <<'PY'
import glob, re, sys
from pathlib import Path

logs = sorted(glob.glob(str(Path(sys.argv[1]) / "replay_s*.log")))
if len(logs) < 3:
    sys.exit(f"隨機 replay teacher 驗證不足三個 seed（找到 {len(logs)}）")
worst_direct, worst_collision = 1.0, 0.0
for path in logs:
    text = Path(path).read_text(errors="ignore")
    m = re.search(r"direct_crossing_rate=([0-9.]+)", text)
    c = re.search(r"碰撞率 \(總計\):\s+\d+ \(([0-9.]+)%\)", text)
    if not m or not c:
        sys.exit(f"無法解析 {path} 的 direct/collision")
    worst_direct = min(worst_direct, float(m.group(1)))
    worst_collision = max(worst_collision, float(c.group(1)) / 100.0)
print(f"[N1] teacher 隨機 replay: worst direct={worst_direct:.4f} "
      f"worst collision={worst_collision:.4f}")
if worst_direct < 0.99:
    sys.exit(f"teacher direct {worst_direct:.4f} < 0.99，不得蒸餾")
if worst_collision > 0.01:
    sys.exit(f"teacher collision {worst_collision:.4f} > 0.01，不得蒸餾")
PY
then
    echo "[N1] 拒絕啟動：scripted teacher 未通過隨機 replay 驗證" >&2
    exit 1
fi

echo "[N1] 啟動 $RUN_NAME (seed=$SEED)"
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
    --experiment_config e2e_sa6_n1_direct_imitation_from_d0 \
    --headless --seed "$SEED" --run_name "$RUN_NAME" \
    > "$REPO/logs/rnn_car/n1_launch_s${SEED}.out" 2>&1 \
    || echo "[N1] 訓練 exit $?"

echo "[N1] 訓練結束。接著跑 per-checkpoint Gate5 screening："
echo "  W2_RUN_DIR=$REPO/logs/rnn_car/$RUN_NAME W2_GATE_PREFIX=n1_s${SEED} \\"
echo "    bash $REPO/scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_w2_gate5_screen.sh"
