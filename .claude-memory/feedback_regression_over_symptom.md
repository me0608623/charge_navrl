---
name: feedback_regression_over_symptom
description: 遇到回歸性 bug（某版本後才出現）先做回歸分析「改了什麼」，不要用新 reward/懲罰去壓症狀
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

當某問題是「某版本之後才出現」（回歸性 bug），**先做回歸分析找出「改了什麼造成的」並修那個根因，不要用加 reward/懲罰/獎勵的方式去壓症狀**。

**Why**：2026-06-15 sin 波殘留，我第一反應是提議加「第 4 件套」reward 懲罰（heading-aligned |ω| penalty / heading_stability）。用戶糾正：「加獎勵不見得是好方法，我們要去想是改了什麼造成 sin 波，原本沒有這些角速度懲罰獎勵也不會出現此問題」。用新機制壓另一個機制造成的副作用 → 越疊越複雜、治標不治本。正解是回歸 bisect：v2（乾淨）→ v3/v3b/v3c 哪個改動引入。結果根因是 v3d 的 action_error obs 自我參照編碼（見 [[finding_action_error_selfref_oscillation]]），該移除不是該加懲罰。

**How to apply**：
1. 問「哪個版本開始壞的？上一個乾淨版本是什麼？」→ 找乾淨 baseline（如還在的舊 ckpt）。
2. `git log -S` bisect 出引入改動的 commit，列出所有候選改動。
3. 用乾淨 ckpt 重現/不重現做對照（cheap，不必重訓）。
4. 修/還原那個根因改動，而非疊加新 reward 去抵銷。
5. 加 reward 懲罰只在「確實是缺少某 reward 訊號」時用，不是拿來補別的改動的副作用。

相關：[[finding_action_error_selfref_oscillation]] [[feedback_reward_analysis_rigor]] [[finding_cmd_delay_limit_cycle]]
