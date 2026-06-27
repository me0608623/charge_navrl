---
name: feedback_let_runs_finish
description: 不要提前建議 kill 訓練；讓 run 跑完整 budget，除非真的壞掉（NaN/OOM/process 死）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

監控訓練時**不要因為中途指標看起來不好就建議 kill**，讓 run 跑完整 budget（例：SA1_v3d 要跑滿 900 iter，即使 SR 已 90% 看似收斂）。

**Why**：用戶多次糾正「繼續訓」是對的 —— SA3 監控早期 CR>35%、SA2 #124 ent 跌、SA1_v3d 訓練 p95_flip RED，我都曾建議 kill，事後證明都是 transient / 假象，繼續訓才對。提前 kill 會丟失完整收斂曲線 + 最終 ckpt，且常常是誤判（如 entropy confound）。明確指示：「SA1_v3d 至少要跑完」(2026-06-12)。

**How to apply**：
1. 只在**真壞掉**才建議停：NaN/Inf、PhysX OOM crash、process 死、SR 持續崩跌且不恢復。
2. 中途指標 YELLOW/RED 但主任務（SR/CR）健康 → 報告觀察、**不建議 kill**，繼續監控。
3. 看似收斂（SR 高）也讓它跑完 budget，拿完整曲線 + 完訓 ckpt 再判定。
4. 抽動/震盪類 RED 先用 deterministic eval 排除假象再說（見 [[finding_p95flip_entropy_confound]]）。

相關：[[finding_p95flip_entropy_confound]] [[feedback_entropy_intervention_framework]] [[project_v3d_launch]]
