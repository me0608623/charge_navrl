---
name: Claude-Mem Global Integration
description: IsaacLab 項目連接到全域 Claude-Mem 記憶系統
type: reference
---

# Claude-Mem 全域整合

## 配置日期
2026-04-13

## 位置

### 全域記憶（Cross-Project）
```
~/.claude-mem/
├── claude-mem.db          ← SQLite 數據庫（所有會話、工具調用）
├── chroma/                ← 向量嵌入（語義搜索）
├── settings.json          ← Claude-Mem 配置
└── INTEGRATION.md         ← 詳細集成指南
```

### 項目記憶（IsaacLab 特定）
```
~/.claude/projects/-home-aa-IsaacLab/memory/
├── MEMORY.md                           ← 索引和快速鏈接
├── feedback_*.md                       ← 方法反饋（3+ 條）
├── project_*.md                        ← 項目狀態和計劃
├── v2X_*.md                            ← 實驗發現
└── warp_drive_*.md                     ← Warp Drive 子項目
```

## 自動功能

### Claude-Mem 在每個會話中：
1. **捕捉工具調用**: 所有 Bash、Edit、Read 等操作記錄
2. **語義索引**: 將對話和結果向量化
3. **自動注入**: 新會話時注入相關全域記憶
4. **模式識別**: 識別重複模式和最佳實踐

### 項目記憶保留用於：
1. **快速檢索**: MEMORY.md 索引（<200 行）
2. **決策記錄**: 為什麼做某個選擇
3. **反饋規則**: "不要這樣做，因為..."
4. **當前狀態**: 訓練進度、實驗階段

## 工作流程

### 啟動會話時
```
Session Start
    ↓
[Claude-Mem] 自動注入全域相關記憶
    ↓
[Claude Code] 加載項目 MEMORY.md（快速索引）
    ↓
完整上下文準備就緒
```

### 對話中
```
用戶輸入
    ↓
工具調用 (Read/Edit/Bash...)
    ↓
[Claude-Mem] Hook 捕捉工具+結果
    ↓
Worker 壓縮觀察（後台）
    ↓
Vector DB 索引（用於下次搜索）
```

### 編輯項目記憶
```bash
# 編輯項目級別的記憶（IsaacLab 特定）
vi ~/.claude/projects/-home-aa-IsaacLab/memory/feedback_new.md

# 內容應符合 frontmatter 格式
# （見 MEMORY.md 規範）
```

## 分層決策

| 應存在於... | 內容類型 | 示例 |
|-----------|--------|------|
| **全域 (Claude-Mem)** | 跨項目的通用模式 | "不要在構建時使用 conda run"、"Haiku 適合輕量代理" |
| **項目 (memory/)** | IsaacLab 特定 | "v21 shield 實驗失敗"、"LiDAR r_min = 0.9m" |
| **MEMORY.md 索引** | 最頻繁訪問 (≤150 chars) | "訓練文件在 scripts/skrl/train_charge_ac.py" |

## 驗證集成

### 檢查 Claude-Mem 狀態
```bash
# 檢查進程是否運行
ps aux | grep worker-service

# 檢查 SQLite 數據庫
ls -lh ~/.claude-mem/claude-mem.db

# 檢查向量存儲
ls -lh ~/.claude-mem/chroma/

# 快速健康檢查
curl -s http://localhost:37777/health || echo "Worker 未運行（會自動啟動）"
```

### 檢查項目記憶
```bash
# 檢查 MEMORY.md 索引
head -20 ~/.claude/projects/-home-aa-IsaacLab/memory/MEMORY.md

# 列出所有項目記憶
ls -1 ~/.claude/projects/-home-aa-IsaacLab/memory/*.md | wc -l
```

## 便利命令

```bash
# 搜索全域記憶（Claude-Mem）
curl -s "http://localhost:37777/search?q=isaac+navigation" | jq .

# 查看最近會話
curl -s http://localhost:37777/memories?limit=20 | jq .

# 獲取統計
curl -s http://localhost:37777/stats | jq .

# 備份全域數據庫
cp ~/.claude-mem/claude-mem.db ~/.claude-mem/backups/claude-mem.$(date +%s).db
```

## 配置調整

如需修改 Claude-Mem 行為，編輯：
```json
~/.claude-mem/settings.json
{
  "autoCompress": true,          // 自動壓縮長對話
  "compressionThreshold": 10,    // 壓縮触發的消息數
  "retentionDays": 90,          // 自動刪除舊記憶
  "features": {
    "semanticSearch": true,     // 啟用向量搜索
    "projectContextInjection": true  // 注入項目上下文
  }
}
```

## 故障排除

### Claude-Mem Worker 未啟動
```bash
# 手動啟動（通常 Claude Code 會自動做）
cd ~/.claude-mem-src
npm run worker

# 檢查日誌
tail -f ~/.claude-mem/worker.log
```

### 記憶搜索不精確
```bash
# 重建向量索引
rm -rf ~/.claude-mem/chroma/
# 下個會話時自動重新構建
```

### 數據庫文件損壞
```bash
# 備份並重置
cp ~/.claude-mem/claude-mem.db ~/.claude-mem/claude-mem.db.corrupted
rm ~/.claude-mem/claude-mem.db
# 系統會自動重新創建
```

## 下一步

1. ✅ Claude-Mem 已安裝到 ~/.claude-mem-src
2. ✅ 插件已複製到 ~/.claude/plugins/marketplaces/thedotmack/
3. ✅ 配置已創建 (~/.claude-mem/settings.json)
4. ⏭️ **重啟 Claude Code** 以激活 Claude-Mem
5. ⏭️ Claude-Mem 會在新會話中自動開始記憶和學習

## 相關文檔

- 全局集成指南: `~/.claude-mem/INTEGRATION.md`
- 項目記憶索引: `./MEMORY.md`
- Claude Code 規則: `~/.claude/rules/`
