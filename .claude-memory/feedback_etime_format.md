---
name: feedback_etime_format
description: ps etime 格式是 HH:MM:SS，不是 DD:HH:MM，避免再次誤判訓練時長
type: feedback
---

`ps -o etime` 的輸出格式是 `HH:MM:SS`（不足 1 小時顯示 `MM:SS`，超過 1 天顯示 `D-HH:MM:SS`）。

**Why:** 曾經把 `01:35:30` 誤讀為「1 天 35 小時 30 分鐘」，導致整個監控報告的時間計算全部錯誤（把 1.5 小時說成 96 小時、把 1.25 min/iter 說成 45 min/iter、把 21 小時說成 32 天）。

**How to apply:** 任何用 `ps -o etime` 監控訓練進程時，記住格式規則：
- `MM:SS` = 分:秒
- `HH:MM:SS` = 時:分:秒
- `D-HH:MM:SS` = 天-時:分:秒（注意天數後面是 `-` 不是 `:`）
