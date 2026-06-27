---
name: feedback_normalize_episode_metrics
description: 監控報告的 SR/CR/TO 必須用已完成 episodes 為分母正規化，不能直接報告原始比例
type: feedback
---

監控報告中 SR/CR/TO 必須正規化為已完成 episodes 的比例，不能直接報告 raw %。

**Why:** Raw SR+CR+TO < 100%，因為包含仍在進行中的 episodes。直接報告會低估真實 success/collision rate，造成誤判。

**How to apply:** 
- completed_rate = SR_raw + CR_raw + TO_raw
- SR_norm = SR_raw / completed_rate
- CR_norm = CR_raw / completed_rate  
- TO_norm = TO_raw / completed_rate
- 報告時標示 completed_rate 和 normalized 值
