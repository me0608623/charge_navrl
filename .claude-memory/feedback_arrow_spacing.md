---
name: Arrow spacing before numbers
description: 箭頭後面接數字時必須空一格，避免渲染遮擋
type: feedback
---

箭頭後面接數字時，必須在箭頭後加一個空白，例如 `→ 32` 而不是 `→32`。

**Why:** `→32` 在終端渲染時箭頭會遮擋後面的數字，用戶看不清楚。

**How to apply:** 任何 ASCII/Unicode 箭頭（`→`, `←`, `↑`, `↓`, `⟶` 等）後面緊接數字、字母、括號時，一律補一個空格。
