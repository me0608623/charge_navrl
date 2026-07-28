---
name: feedback-no-append-plus-replace-same-file
description: 同檔案混用 cat>> 追加與 replace 修改會造成重複定義，舊版靜默生效且測試全綠
metadata:
  type: feedback
---

同一個檔案**不要混用「`cat >>` 追加新實作」與「`str.replace(old,new,1)` 修改」**。
追加落在檔尾、replace 落在第一個出現處 → 新舊兩份並存 → Python 只認最後一份（舊版）。

**Why**：2026-07-27 在 `corridor_density.py` 實測踩到 —— 掃出 **9 個重複的 top-level 名稱**。
`apply_interaction_geometry` 的「並排同速」改在第一份，runtime 跑最後一份仍隨機加 ±0.05 m/s 速差。
**所有單元測試全綠**，因為測試 import 到的也是同一份錯誤版本。
連續兩個 GPU smoke（v7、v8）的並排結果因此作廢。

**How to apply**：
1. 改既有函式就整份取代，不要追加新版
2. `str.replace(..., 1)` 找不到目標時**靜默無效** → 一律先 `assert old in text`
   （另一次事故：import 沒加成功，`ast.parse` 也過，只有跑 sim 才炸 `NameError`）
3. 加 AST 守門測試禁止 top-level 名稱重複，**必須涵蓋 `AnnAssign`**
   （只看 `Assign` 會漏掉 `X: tuple[float,float] = (...)` 這種帶型別註解的常數，實際漏過一份）

見 [[finding_ledger_desync_early_return]]
