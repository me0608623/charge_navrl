---
name: Policy oscillation stuck behavior
description: SA4 checkpoint 390000 在障礙物 1.1m + 目標在後方時出現 v_body ±0.078 交替震盪，導致 agent 原地不動 100+ 步
type: project
---

SA4 checkpoint_390000.pt 在 T corridor + stage 4 配置下，回合 10 出現明顯的 policy oscillation 卡住行為。

**現象：**
- Agent 位置 `(0.77~0.82, 6.09~6.14)` 完全不動，持續 100+ 步
- `v_body_pref` 在 `+0.078` 和 `-0.078` 之間每步交替
- 實際速度 0.01~0.08 m/s（幾乎為零）
- ORCA 完全沒介入（`orca=N`）— 不是 safety filter 的問題
- 最近障礙物距離 1.09~1.15m（穩定）
- 目標在左後方 `goal_body=(-1.5, +2.0)` 約 2.5m

**根因：** Policy 對「障礙物在 ~1.1m + 目標在背後需轉身」的情境學到了猶豫震盪行為，而非果斷轉身。這是訓練品質問題。

**Why:** 訓練時的 reward shaping 可能沒有足夠的「轉身面向目標」激勵，或者近距離障礙物的 penalty 讓 policy 不敢大動作轉身。

**How to apply:**
- 已在 RVO2 filter 加 anti-oscillation 層（v_body 符號翻轉 ≥6/10 步時強制方向）
- 已降低 recovery 閾值（`stuck_pref_threshold` 0.3→0.05）
- 未來訓練應考慮加強「面向目標轉身」的 reward，或在 near-obstacle 情境下降低轉身 penalty
