---
name: scan_timestamp
description: /scan 報告必須顯示系統當下時間（台北時區 UTC+8），用 date 指令取得
type: feedback
---

/scan 報告標題的時間戳必須用系統當下時間，不能猜測。

**Why:** 用戶需要精確知道每次掃描的時間點，方便對照訓練進度。

**How to apply:** 每次 /scan 開頭先跑 `date '+%Y-%m-%d %H:%M'` 取得當前時間，寫進報告標題。
