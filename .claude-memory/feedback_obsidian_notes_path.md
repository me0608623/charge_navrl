---
name: obsidian-vault
description: 用戶要求所有筆記都要寫到 Obsidian Vault 路徑
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

# 筆記必須寫到 Obsidian Vault

**Why:** 2026-04-08 用戶明確指示「以後我說寫筆記都要寫到 /home/aa/Documents/Obsidian Vault」。Claude Code memory 系統的筆記用戶看不到 (在 .claude/ 隱藏目錄)，Obsidian 是用戶實際使用的知識管理工具。

**How to apply:**
- 用戶說「筆記」/「寫起來」/「記下來」時，**必須**寫到 `/home/aa/Documents/Obsidian Vault/`
- **v2 工作（SAx_v2、asymmetric critic、TLNI、sim-to-real DR）→ 寫到 `isaaclab_v2/`**（2026-05-27 起，用戶明確指示「未來 sax_v2 都寫到 isaaclab_v2」）。該 vault 已有結構：CLAUDE.md(交接) + README_v2索引總覽 + 子目錄（架構與演算法研究/神經網路架構/環境配置與觀測/訓練任務(task)/課程學習/訓練獎勵設計/訓練報告/訓練歷程與工具）。寫完要更新 README + CLAUDE 的連結索引。
- **舊 baseline / SA1~SA7 / 一般 IsaacLab 筆記 → 寫到 `isaaclab/`**（同 vault，`[[wikilink]]` 可跨資料夾互連）
- 訓練報告（SA 各階段）放 `訓練報告/`（一 stage 一檔 + 總覽）
- **同時也存到 Claude memory** (`/home/aa/.claude/projects/-home-aa-IsaacLab/memory/`)，這樣未來 Claude session 仍能 recall
- Obsidian 用 Markdown 格式，可用 wiki link `[[檔案名]]` 互相連結
- 已存在的 RL訓練 檔案命名規範: `rw_<reward>_<curriculum>__<flags>.md` 或主題式如 `v18_xxx.md`
