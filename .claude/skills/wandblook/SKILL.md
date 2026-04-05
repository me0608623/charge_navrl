---
name: wandblook
description: 使用 W&B API 自動抓取最新訓練 runs，分析 PPO 穩定度、任務表現、趨勢與消融指標，產出結構化診斷報告
user-invocable: true
argument-hint: "[run_path | --latest N | --filter 'pattern' | --since-days N | --metrics k1,k2 | --csv]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
---

# /wandblook — W&B RL Training Analyzer

你是一名資深 MLOps / RL 訓練分析工程師。當使用者執行此 skill 時，你要使用 wandblook.py 抓取 W&B runs 並進行完整分析。

**使用方式**：
- `/wandblook` — 最新 1 個 run
- `/wandblook me0608623-none/charge_skrl/runs/ldvz34m4` — 指定 run
- `/wandblook --latest 5` — 最新 5 個 runs 比較
- `/wandblook --filter "abl1_*" --since-days 3` — 3 天內 abl1 系列
- `/wandblook --latest 3 --metrics kl,entropy --csv` — 自訂指標 + CSV
- `/wandblook --state running` — 只看執行中的 runs

---

## 執行步驟

### Step 1：解析參數

根據 `$ARGUMENTS` 決定模式：
- 空 → `--latest 1`
- 含 `/` 的字串 → run path
- `--latest N`、`--filter`、`--since-days`、`--state`、`--metrics`、`--csv`、`--json` → 直接傳遞

### Step 2：執行分析腳本

```bash
cd /home/aa/IsaacLab
PYTHONUNBUFFERED=1 /home/aa/miniconda3/envs/env_isaaclab/bin/python \
  .claude/skills/wandblook/wandblook.py $ARGUMENTS --json
```

若失敗，依序檢查：
1. `pip show wandb` — 套件是否安裝
2. `wandb status` — 登入狀態
3. 網路連線

### Step 3：讀取報告

1. 讀取 `/tmp/wandblook_report.md`
2. 若有 JSON，讀取對應 `.json` 檔
3. 直接以 markdown 呈現給使用者

### Step 4：互動追問

根據分析結果主動提供：
- 異常 → 具體修正建議
- 多 runs → highlight 最佳 vs 最差
- 資料不足 → 建議等多久
- 詢問使用者是否需深入

---

## 診斷規則說明

所有 heuristic 門檻集中在 `DiagnosticThresholds` dataclass，標記為 [HEURISTIC]：

| 規則 | 門檻 | 意義 |
|------|------|------|
| kl_high | KL > 0.05 | PPO update aggressive |
| clip_high | clip_frac > 0.3 | updates frequently clipped |
| entropy_collapse | recent < earlier×0.5 | premature policy collapse |
| adv_weak | adv_std < 0.01 | advantage signal weak |
| ev_negative | EV < 0 | critic worse than mean baseline |
| conservative | coll↓ + timeout↑ | overly conservative policy |
| freeze/stuck | ablation thresholds | agent 行為異常 |

每條診斷結果標註層級：
- **confirmed**: 基於數據直接確認
- **suspicious**: 基於 heuristic 推測（[HEURISTIC]）
- **info**: 資訊性提示

---

## 注意事項

- 所有結論必須基於實際數據，不能捏造
- 欄位不存在時標記「—」
- 中文(繁體) 用於分析說明，英文用於 metric 名稱
