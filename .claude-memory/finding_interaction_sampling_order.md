---
name: finding-interaction-sampling-order
description: 互動型態必須先抽、family再服從；random_2d不得參與配對否則被wander覆寫成假陽性
metadata:
  type: project
---

⭐⭐ **場景互動的取樣順序決定能否達標**（2026-07-27 走廊行人互動）

**錯誤順序**：先抽 family → 問「這組做得出什麼互動」→ 從可行選項抽。
後果：分佈被 family 的偶然組合決定，湊不出承諾的 60/20/20，抽到做不出的只能默默降級。
> 降級比缺功能更糟：它讓錯誤的數字看起來是對的。

**正確順序**：**互動是契約，family 為實現它而指派**。
crossing = lateral+longitudinal；side_by_side = longitudinal×2；其餘 slot 補回整批平衡。
這樣 `realized` 恆為 True，不需要降級路徑。

**random_2d 絕不可參與配對** —— wander 之後會重抽 heading，把擺好的互動幾何整個蓋掉。
舊版實測 76.9% 並排配對含 random_2d、52.7% 跨 family；若 audit 量「安裝前」張量會全部通過 → **假陽性**。
**audit 必須讀安裝後的 scheduler 狀態，並跨步複驗。**

**crossing 要量事件不要量相位**：逐幀要求「朝交點走」是錯的 ——
`x_past_crossing` 4665（已越過=完成）、`x_zero_speed` 1179（端點暫停）主導失敗，
而真正該防的 `x_wrong_family` 是 **0**。改為保存固定交點+起始側別，側別翻轉即記穿越；
**分母用已結算而非已安裝**，否則未結束回合會壓低完成率。

**並排速差必須實作為 0**：契約門檻 ≤0.05 m/s，但有速差就必然失步。
同速 + 關閉該 pair 隨機暫停（per-slot `patrol_no_pause`）後：
隊形維持 73.2% → **98.5%**、間距違規 785 → **0**。
⚠️ 測試若只斷言「≤0.05」，隨機加 ±0.05 的舊版也全綠 → **要斷言完全相等**。
