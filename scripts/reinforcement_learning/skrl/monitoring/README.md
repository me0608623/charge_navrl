# Monitoring 工具

訓練過程即時監控腳本。

| 檔案 | 功能 |
|------|------|
| `cron_training_supervisor.sh` | **:00 :10 :20 …** 存活檢查 + ALERT + auto-advance tick，寫 `logs/training_supervisor/cron_10min.log` |
| `cron_detail_report.sh` | **:05 :15 :25 …** 呼叫 `detail_report.py`，錯開 5 分鐘 |
| `detail_report.py` | 完整詳細監督報告：指標表、分場景 SR/CR/TO、趨勢、ETA、錯誤掃描、GREEN/YELLOW/RED |
| `test_detail_report.py` | `detail_report.py` 的 59 個單元測試 |
| `auto_advance_supervisor.py` | 階段自動推進判定（目前停在 `HALTED_ALERT`，終態） |
| `monitor_wd6fix.py` | wd6fix run 的即時監控 — 讀取 WandB CSV + output.log |

## 兩支 cron 的分工

```
:00  cron_training_supervisor.sh   進程在不在、run_name 對不對、有沒有 NaN/OOM
:05  cron_detail_report.sh         這輪跑得好不好、哪個場景在退步、還要多久
:10  cron_training_supervisor.sh
:15  cron_detail_report.sh
```

錯開 5 分鐘等效於每 5 分鐘有一次檢查。兩支互不修改對方的檔案；
`cron_training_supervisor.sh` 管 `expected_run.txt` / `status.txt`，
`detail_report.py` 只管自己的 `detail_report_state.json`。

## detail_report.py

### 輸出

| 路徑 | 內容 |
|------|------|
| `logs/training_supervisor/detail_10min.log` | 逐次附加，完整歷史 |
| `logs/training_supervisor/detail_latest.txt` | 只有最新一份，方便 `cat` |
| `logs/training_supervisor/detail_report_state.json` | 上次觀測（run/時間/iteration/log 大小），用來算速率與偵測 stall |

### 常用指令

```bash
# 看最新一份
cat logs/training_supervisor/detail_latest.txt

# 手動跑一次（只印不寫檔，不影響 cron 的狀態檔）
./scripts/reinforcement_learning/skrl/monitoring/cron_detail_report.sh --stdout

# 指定 run
python scripts/reinforcement_learning/skrl/monitoring/detail_report.py --run <run_dir_name> --stdout
```

### 判定門檻

全部集中在 `Thresholds` dataclass，判定式內不得寫死數字（有測試把關）：

| 檢查 | YELLOW | RED |
|------|--------|-----|
| KL | > 0.010 | > 0.020 |
| clip_fraction | > 0.20 | > 0.30 |
| entropy 下降（每 50 iter） | ≥ 0.25 | ≥ 0.50 |
| CR 回彈（每 50 iter） | ≥ +0.02 | ≥ +0.05 |
| fps 近期中位數 | 低於基準 70% | — |
| 場景 episode 數 | < 30 | — |
| 核心欄位 NaN/None/**缺漏** | — | 一律 RED |
| 進程消失且 iteration < target | — | RED |
| iteration **且** console log 同時凍結 | — | RED（stall） |

### 三個刻意的設計選擇

1. **只回報，永不介入。** 模組內不含任何終止進程的手段，由
   `test_module_never_kills_a_process` 以來源契約保證。要不要 kill 是人的決定。
2. **欄位不存在 ≠ 檢查通過。** 核心欄位缺漏直接 RED。曾經發生過「查不到欄位」
   被讀成「檢查通過」的誤判，這條規則是為了讓那類錯誤不可能安靜發生。
3. **趨勢回報實際跨度。** 歷史不足 50 iter 時印出真實跨度並換算，不假裝跨滿。
4. **fps 掉只給 YELLOW。** 單卡共享，別人的 job 會偽裝成 stall；訊息本身就要求
   先看 `nvidia-smi`。stall 必須 iteration 與 log 同時凍結才成立。

### 測試

```bash
cd scripts/reinforcement_learning/skrl/monitoring
/home/aa/miniconda3/envs/env_isaaclab/bin/python -m pytest test_detail_report.py -q
```
