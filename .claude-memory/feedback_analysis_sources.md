---
name: feedback_analysis_sources
description: 訓練分析應使用四個資料來源交叉驗證，不可只靠單一來源
type: feedback
---

訓練分析必須使用四個資料來源交叉驗證：

1. **debug_metrics.csv** — 本地 rollout 級別指標（主要）
2. **Console log** — 即時 PPODiag / Curriculum / SpawnOpt 輸出
3. **Checkpoint** — 離線 gradient 分解、logit 分佈、weight 分析
4. **WandB API** (`/wandblook`) — step-level 歷史、趨勢圖、跨 run 比較

**Why:** CSV 只有 rollout 級別快照，可能漏掉 step-level 異常。WandB 有完整歷史但需要 API 呼叫。兩者交叉驗證能避免單一來源的盲點（如 logging bug 導致 CSV 數據無效的情況）。

**How to apply:** `/scan` 診斷時同時讀 CSV 和呼叫 `/wandblook`，用 WandB 數據驗證 CSV 結論。若兩者矛盾，優先信任 WandB（因為它記錄的是 agent.track_data 的原始值）。
