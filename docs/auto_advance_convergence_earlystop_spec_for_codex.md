# Spec：收斂偵測自動早停 → 接既有 auto-advance 管線（給 Codex 實作）

作者：監督端（唯讀交叉驗證）｜日期：2026-07-23｜狀態：已實作並上線

## 實作狀態（2026-07-23）

- `monitoring/auto_advance_supervisor.py`：持久化狀態機、收斂判定、合作式停止、Gate 啟動與 PASS 後晉級。
- `monitoring/cron_training_supervisor.sh`：每 10 分鐘在健康報告後執行一次狀態機 tick。
- `train/train_rnn_car_wdclip.py`：每 iteration 寫 `supervisor_metrics.jsonl`；收到
  `supervisor_stop.request` 後在 rollout 邊界強制存檔、正常 `wandb.finish()`、釋放 simulator。
- `rnn_car_wdclean/validate_checkpoint.sh`：同 checkpoint/stage 的驗收使用專用 `flock`，拒絕重複執行。
- SA5–SA8 均有預先凍結的 `e2e_saN_k8_obb.py`，晉級器不在 runtime 動態改訓練配方。

實測：SA5 iter 2 收到停止請求後生成 `checkpoint_256.pt`，log 出現
`Training stopped cooperatively`，W&B 正常同步並退出；再由該 checkpoint 保留 optimizer
恢復 managed SA5。收斂器的穩定、暖機、SR 漂移、VF 漂移、Gate fail-closed 與下一 stage
命令已有 8 個單元測試覆蓋。

## 背景與範圍

現況（實測）：
- Codex **已具備**「det測試（Gate1/2/4，SA5 起+Gate3/5）→ 全 PASS → 建下一 SA config → 從 ckpt 啟訓」的自動晉級**後端**（SA4→SA5 這次就是自動完成的）。
- **缺的只是前端**：訓練是**跑滿 `--timesteps` budget** 才停，不是**偵測到收斂就早停**。用戶要的是「若已收斂則自動啟動完訓 gate 測試並自動啟動下一 stage」——即**收斂偵測 → 自動停訓 → 觸發既有後端**。

本 spec 只補這一段前端；後端（det測試→gate→晉級）沿用既有邏輯，不重寫。

## 目標流程

```
訓練中(每 M iter 抽指標)
  → 收斂偵測器判定「已收斂」
    → 保存當前 checkpoint、乾淨停訓(SIGTERM，非 kill -9)
    → 觸發既有 det測試：validate_checkpoint.sh <ckpt> <stage> <wandb_id>
      → validate_gates 綜合裁決
        → 全 PASS → 既有「建下一 SA config + 啟訓」
        → 任一硬 Gate FAIL → 停下 + 告警，**不自動晉級**(等人工)
```

## 收斂判準（沿用用戶 SA3 iter-650 的實測判準）

同時滿足、且**連續 K 個監控窗**（建議 K=5、每窗 10 iter）才算收斂：

| 指標 | 收斂條件 | 理由 |
|------|---------|------|
| SR | 連 K 窗維持高位且窗間變異 < 1pp（例 ≥ 該階段 det_sr 門檻且平台化）| 表現飽和 |
| vf | 連 K 窗相對變化 < 15%（末段持平，非仍單調下降）| critic 收斂 |
| ent | 連 K 窗在 ±5% 帶內（非單調衰減、非爆升）| 探索穩定 |
| kl / clip_frac | kl 中位 < 0.01 且 clip_frac < 0.10 | 策略更新趨緩 |
| enc_g | 中位停止快速下降（可高檔平台，只要不發散）| 表徵大致穩 |

**保底**：仍保留 `--timesteps` 為硬上限——即使沒偵測到收斂，跑滿 budget 一樣觸發後端（現行行為），避免收斂判準太嚴導致永不停。

**下限**：設 `--min_iters_before_earlystop`（建議 該階段 budget 的 20%，如 SA 的 ~420 iter）——避免暖啟初期假平台誤判早停（參 SA4 轉換期 vf/enc_g 高但 SR 已高的假象）。

## ★ 不可省的安全護欄（最重要）

1. **只在「適用 Gate 全 PASS」才自動晉級**；任一**硬** Gate FAIL 就**停下 + 告警，絕不自動推進下一 stage**。
   - SA1–SA4：適用 = Gate1/2/4（Gate3/5 延遲，不擋）。
   - **SA5 起：適用 = Gate1/2/3/4/5**（Gate3 擋路 + Gate5 0.85m 窄縫轉硬門檻）。→ 自動晉級器**必須讀 `advanced_gates = stage >= 5`**，SA5+ 若 Gate3/5 FAIL 一定停下等人工（這正是防「不會鑽 0.85m 縫卻硬推進」）。
2. **停訓用 SIGTERM 讓 wandb/ckpt 乾淨收尾**，不可 kill -9（避免 wandb crashed + ckpt 半寫）。
3. **啟 det測試前先確認沒有同 ckpt 的 det測試在跑**（避免重複啟動污染 `/tmp/validate_*` 同一輸出目錄——監督端 07-23 曾因未先確認而重複啟動、截斷 det_s101.log，靠 regen 補救；自動化務必加此檢查：`pgrep -f "validate_checkpoint.sh .*<ckpt>"` 為空才啟）。
4. **det測試輸出目錄唯一化 or 加鎖**（每次晉級用獨立子目錄或 flock），根絕並行覆蓋。

## 落點建議

- 掛在既有 `scripts/reinforcement_learning/skrl/monitoring/cron_training_supervisor.sh`（目前只被動監控），或獨立 `auto_advance_supervisor.sh`。
- 讀 `/tmp/<run_name>.log` 的 `^[iter/total] ... SR= CR= vf= ent= enc_g= kl clip` 行做收斂判定（該 log 已有這些欄）。
- 狀態機用檔案持久化（`logs/training_supervisor/status.txt` 已有雛形）：`TRAINING → CONVERGED → GATING → (PASS)ADVANCING / (FAIL)HALTED_ALERT`。

## 驗收

- 人工在一支已明顯收斂的 run 上啟用，確認：偵測到收斂 → 乾淨停 → det測試 → PASS → 自動起下一 stage；**且蓄意讓某硬 Gate 不過時，確認它停在 HALTED_ALERT 不推進**。
- 監督端（我）會在自動化上線後，對每次自動晉級**唯讀重跑 validate_gates 交叉驗證**、diff 新 config 抓夾帶。

## 關聯
- 授權協定：Gate3/5 延遲 SA1-4、SA5 起硬門檻（用戶 07-22 授權）。
- 收斂判準來源：用戶 SA3 iter-650 裁決（SR/vf/ent/kl/clip/enc_g 平台化）。
