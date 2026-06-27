---
name: Shield 不能補救 under-trained policy
description: 設計 shield/safety mechanism 的順序原則
type: feedback
---

# Shield 不能 substitute reward shaping

**Why:** 2026-04-08 v21 VO Shield 實驗證實：inference-time shield 只能對 well-trained baseline policy 做 last-mile safety net，無法救援沒學會避障的 policy。實測 v21 加 VO shield 從 CR 49% → 41.9% (-7pp)，遠低於 NavRL paper 的 -68% drop，因為 v21 policy 的 ds effective weight 只有 0.31% of goal_velocity，從未真正學會 closing_risk avoidance。

**How to apply:**
- 看到 RL system "agent 不會避動態障礙物" 時，**先檢查 reward effective weight ratio** 而不是直接想架構級解法
- 不要把 paper 的 SOTA shield 數字直接套到 baseline 不同的 system
- shield/cbf/safety projection 的順序：(1) 確保 policy reward gradient 對齊目標 → (2) 訓練到 baseline 達標 → (3) 加 shield 做 last-mile
- 反向順序 (先加 shield 救爛 policy) 失敗率極高
- 簡化 VO heuristic (single-worst-threat) 在 dense scene 必然 ping-pong → evade A 撞到 B
- 真正 VO 需要 multi-obstacle LP，不能用 simplified heuristic 替代
