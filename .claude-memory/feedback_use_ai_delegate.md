---
name: Use ai-delegate for non-coding tasks
description: User reminded that research/notes/log-analysis must be delegated to Gemini/Codex via ai-delegate to save Opus tokens — I had been doing all work in main session
type: feedback
---

必須使用 `ai-delegate` 把非寫碼任務丟給 Gemini/Codex，不要全部留在主 Claude session。

**Why:** 用戶在 2026-04-08 發現我做 v21/v22 實驗時，所有 research advisor 多輪討論、Obsidian 筆記寫入、/scan CSV 分析全都在主 session 裡 burn Opus tokens，沒按 `~/.claude/rules/common/delegation.md` 規則路由。1M Opus tokens 被浪費在原本應該免費 Gemini 處理的工作。

**How to apply:**
- **必須 delegate**:
  - Research / paper 查證 (NavRL, HEIGHT, ORCA 細節) → `ai-delegate research -q -- "..."`
  - Obsidian 筆記寫入 (`/home/aa/Documents/Obsidian Vault/`) → `ai-delegate note -q -- "..."`
  - /scan CSV / WandB metrics 摘要分析 → `ai-delegate log-analysis -q -f <csv> -- "..."`
  - 文件更新 (SYNC_TO_5070.md 等) → `ai-delegate doc -q -- "..."`
  - Git commit / push → `ai-delegate git -q -d <dir> -- "..."`
- **必須留在 Claude**:
  - 寫/改 code (Edit/Write 工具)
  - 多檔案 debug (Read → Grep → Edit chain)
  - reward formula / arch 推理
  - 訓練監控決策 (停止/重啟/調參)
- **判斷準則**: 「這任務需要用 Claude Code tool (Edit/Grep/MCP) 嗎？」 否 → delegate
- 若 delegate 失敗 (provider error)，回退自己做
- 不要跟用戶說在 routing — 自然呈現結果即可
