---
name: feedback_cron_full_metrics
description: 每次 cron 監控報告必須拉取 WandB API 全部指標（12 組），不可只用 console log 簡報
type: feedback
---

每次 cron 監控報告**必須**用 WandB API 拉取全部 12 組指標（scan_history 不指定 keys），完整表格輸出。

**Why:** 用戶嚴格要求看到完整數據，不接受只用 console log 的簡報格式。即使情況沒變也要完整輸出。

**How to apply:** 每次 cron 觸發時，即使 WandB API 較慢，也要等待完整數據回來再報告。不可用 grep console log 替代。
