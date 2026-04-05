# wandblook — W&B RL Training Analyzer

自動連接 Weights & Biases API，抓取 RL 訓練 runs 的 history / summary / config，
執行 PPO 穩定度、任務表現、趨勢與消融指標分析，產出結構化診斷報告。

## 功能摘要

- **單一 / 多 run 分析**：指定 run path 或自動抓最新 N 個
- **欄位別名映射**：自動識別不同命名慣例（`kl` / `train/kl` / `ppo/kl` / `approx_kl`）
- **穩健 history 讀取**：使用 `scan_history()` 串流讀取，避免記憶體爆炸
- **11 條 heuristic 診斷規則**：KL、clip、entropy、advantage、gradient、success、collision/timeout、ablation
- **趨勢分析**：最近 20% vs 前 80% 比較，自動判定上升/下降/持平
- **多 run 橫向比較表**
- **多格式輸出**：終端摘要 + Markdown + JSON + CSV

## 依賴套件

```
wandb >= 0.15.0
```

（已包含在 `env_isaaclab` conda 環境中）

## 環境變數

| 變數 | 必要性 | 說明 |
|------|--------|------|
| `WANDB_API_KEY` | 選用 | W&B API key（若已 `wandb login` 則不需要） |
| `WANDB_PROJECT` | 選用 | 預設 project（腳本預設 `charge_skrl`） |

設定方式：
```bash
# 方法 1: 環境變數
export WANDB_API_KEY=<your-key>

# 方法 2: wandb CLI 登入（存到 ~/.netrc）
wandb login
```

## 基本使用方式

### 作為 Claude Code Skill

```
/wandblook                                          # 最新 1 個 run
/wandblook me0608623-none/charge_skrl/runs/abc123   # 指定 run
/wandblook --latest 5                               # 最新 5 個比較
/wandblook --filter "abl1_*" --since-days 3         # 3 天內 abl1 系列
/wandblook --latest 3 --metrics kl,entropy --csv    # 自訂指標 + CSV
/wandblook --state running                          # 只看執行中
```

### 作為獨立 Python 腳本

```bash
cd /home/aa/IsaacLab

# 最新 1 個 run
python .claude/skills/wandblook/wandblook.py

# 指定 run
python .claude/skills/wandblook/wandblook.py me0608623-none/charge_skrl/runs/ldvz34m4

# 最新 5 個 + JSON + CSV
python .claude/skills/wandblook/wandblook.py --latest 5 --json --csv

# 過濾 + 時間範圍
python .claude/skills/wandblook/wandblook.py --filter "abl2_*" --since-days 7

# 自訂輸出目錄
python .claude/skills/wandblook/wandblook.py --latest 3 --output-dir reports/

# 只看 crashed runs
python .claude/skills/wandblook/wandblook.py --latest 10 --state crashed --quiet --csv
```

## 輸出內容

### 終端摘要
- Run 名稱、ID、狀態
- 各類 metrics 最新值
- 診斷結論（含層級標記）

### Markdown 報告（`/tmp/wandblook_report.md`）
1. **Run Metadata** — 名稱、ID、狀態、URL、tags
2. **Config Snapshot** — 關鍵超參數
3. **Key Metrics** — latest、mean、std、recent 20%、trend、min、max
4. **Trend Analysis** — 上升/下降/持平分類
5. **Diagnostic Conclusion** — confirmed / suspicious / info 三級
6. **Suggested Next Checks** — 可行動建議
7. **Multi-Run Comparison** — 橫向比較表（多 run 時）

### JSON（`--json`）
結構化數據，含 meta、config、latest values、diagnosis、trends。

### CSV（`--csv`）
一 row 一 run，columns = 所有偵測到的 metrics，方便 Excel / pandas 後處理。

## 診斷規則

所有門檻集中在 `DiagnosticThresholds` dataclass（程式碼 Section 1），標記為 `[HEURISTIC]`。

| # | 規則名 | 門檻 | 觸發條件 |
|---|--------|------|----------|
| 1 | `kl_high` | KL > 0.05 | PPO update 幅度過大 |
| 2 | `kl_rising` | recent > earlier × 1.5 | KL 趨勢上升 |
| 3 | `clip_high` | clip_fraction > 0.3 | 策略更新頻繁被 clip |
| 4 | `ratio_extreme` | ratio_max > 3.0 | IS ratio 過大 |
| 5 | `entropy_collapsed` | entropy < 0.1 | 策略幾乎確定性 |
| 6 | `entropy_premature_collapse` | recent < earlier × 0.5 且 success 未升 | 過早收斂 |
| 7 | `adv_weak` | adv_std < 0.01 | Actor 可用訊號極弱 |
| 8 | `ev_negative` | explained_variance < 0 | Critic 差於 baseline |
| 9 | `grad_explode` | grad_norm > 100 | Gradient 爆炸 |
| 10 | `success_stagnation` | Δ < 0.02 且 < 50% | 表現停滯 |
| 11 | `conservative_policy` | collision↓ + timeout↑ | 過於保守 |

## 限制事項

- **需要網路**：必須連線到 W&B API
- **History 上限**：預設最多讀取 2000 rows（可用 `--max-history` 調整）
- **門檻為 heuristic**：診斷規則基於經驗值，非絕對標準
- **不畫圖**：目前僅文字分析，未來可擴充趨勢圖
- **單一 project**：一次只分析一個 W&B project

## 後續擴充方向

- [ ] 自動畫趨勢圖（matplotlib / plotly）
- [ ] 比較不同 `reward_mode` / `curriculum_version` 的效果
- [ ] 自動抓最新 crashed/failed run 並分析原因
- [ ] 整合 Obsidian 筆記輸出
- [ ] 支援 W&B Artifacts / Model Registry 查詢
- [ ] 支援 `--watch` 模式持續監控 running run

## 檔案結構

```
.claude/skills/wandblook/
├── SKILL.md        # Claude Code skill 定義
├── wandblook.py    # 主程式（可獨立執行）
└── README.md       # 本文件
```
