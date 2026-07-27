#!/usr/bin/env bash
# N1 步驟 3+4：shadow rollout 量 teacher CE 與 PPO actor 梯度 → 校準 λ。
#
# --narrow_imitation_shadow 讓 CE 的係數固定為 0：照常算 teacher 標籤、CE 與
# 兩邊梯度範數並記錄，但完全不影響參數更新。跑兩個 iteration 就夠取樣。
#
# 產出 wandb 指標 narrow_imitation/*：
#   ce / ce_linear / ce_angular      — 學生與直穿 teacher 的分歧程度
#   agreement_linear / _angular      — 動作 argmax 一致率
#   grad_ppo_actor / grad_ce_unweighted
#   grad_ratio_at_lambda1            — λ=1 時 CE 梯度 ÷ PPO actor 梯度
#   lambda_for_half_ppo              — 讓 CE 梯度 = PPO 的 50% 所需 λ（要的數字）
set -uo pipefail

REPO="/home/aa/IsaacLab"
PY="/home/aa/miniconda3/envs/env_isaaclab/bin/python"
cd "$REPO"

RUN_NAME="${N1_SHADOW_RUN_NAME:-n1_shadow_lambda_calib_s42}"
OUT="$REPO/logs/rnn_car/${RUN_NAME}"
mkdir -p "$OUT"
METRICS="$OUT/supervisor_metrics.jsonl"
LAUNCH_LOG="$OUT/launch.out"

# The trainer appends JSONL. Archive an earlier calibration so a rerun cannot
# silently average stale rows from a previous process.
if [ -f "$METRICS" ]; then
    mv "$METRICS" "${METRICS}.bak.$(date +%Y%m%d_%H%M%S)"
fi

echo "[N1-SHADOW] 開始 shadow rollout（不更新參數，只量梯度）"
PYTHONUNBUFFERED=1 "$PY" \
    scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
    --experiment_config e2e_sa6_n1_direct_imitation_from_d0 \
    --headless \
    --narrow_imitation_shadow \
    --narrow_imitation_weight 0 \
    --disable_aux_training \
    --timesteps 256 \
    --save_interval 1000 \
    --run_name "$RUN_NAME" \
    > "$LAUNCH_LOG" 2>&1 \
    || echo "[N1-SHADOW] exit $?"

echo "[N1-SHADOW] 量測結果："
if [ -f "$METRICS" ]; then
    "$PY" - "$METRICS" <<'PY'
import json, sys
rows = []
for line in open(sys.argv[1]):
    d = json.loads(line)
    ni = d.get("narrow_imitation")
    if ni:
        rows.append((d, ni))
if not rows:
    print("  (no narrow_imitation stats — 窄縫 replay 幀可能為 0，檢查 fraction)")
    sys.exit(1)
iterations = [d["iteration"] for d, _ in rows]
if len(rows) != 2 or iterations != [1, 2]:
    print(
        "  [FAIL] calibration 必須恰有乾淨的 iteration 1/2；"
        f"實際為 {iterations}"
    )
    sys.exit(1)
for d, _ in rows:
    no_update = (
        d.get("narrow_imitation_shadow") is True
        and d.get("ppo_update_count") == 0.0
        and d.get("actor_param_delta_norm") == 0.0
        and d.get("critic_param_delta_norm") == 0.0
    )
    if not no_update:
        print(
            "  [FAIL] shadow 更新了參數或缺少 no-update 證據："
            f"iter={d['iteration']} ppo_updates={d.get('ppo_update_count')} "
            f"actor_delta={d.get('actor_param_delta_norm')} "
            f"critic_delta={d.get('critic_param_delta_norm')}"
        )
        sys.exit(1)

graded = [
    (d["iteration"], ni)
    for d, ni in rows
    if "lambda_for_half_ppo" in ni
]
for d, ni in rows:
    it = d["iteration"]
    print(
        f"  iter {it}: CE={ni['ce']:.4f} "
        f"(lin {ni['ce_linear']:.4f} / ang {ni['ce_angular']:.4f}) "
        f"agree lin={ni['agreement_linear']:.3f} ang={ni['agreement_angular']:.3f} "
        f"frames={ni['active_frames']}"
    )
    if "lambda_for_half_ppo" in ni:
        print(
            f"           grad PPO={ni['grad_ppo_actor']:.4e} "
            f"CE={ni['grad_ce_unweighted']:.4e} "
            f"ratio@λ=1={ni['grad_ratio_at_lambda1']:.4f} "
            f"→ λ_for_half_ppo={ni['lambda_for_half_ppo']:.5f}"
        )
if len(graded) != 2:
    print("\n[N1-SHADOW] ⚠️ 兩輪都必須有梯度統計")
    sys.exit(1)

# Pool both iterations so λ × Σ||g_CE|| = 0.5 × Σ||g_PPO||.
ppo_total = sum(ni["grad_ppo_actor"] for _, ni in graded)
ce_total = sum(ni["grad_ce_unweighted"] for _, ni in graded)
lam = 0.5 * ppo_total / ce_total
print(f"\n[N1-SHADOW] 建議 λ = {lam:.5f}（2 iter pooled gradient norm）")
print("[N1-SHADOW] no-update hard check: PASS")
print("[N1-SHADOW] 填入 configs/e2e_sa6_n1_direct_imitation_from_d0.py 的 _LAMBDA")
print("            並把 _LAMBDA_IS_CALIBRATED 設為 True")
PY
else
    echo "  (missing $METRICS)"
fi
