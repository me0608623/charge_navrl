---
name: cron 監控結束後自動執行 /train_summary
description: 訓練完成或中止時，cron 監控必須自動執行 /train_summary 產生完整報告
type: feedback
---

Cron 監控偵測到訓練結束（PID 死亡）時，不論是正常完成或異常中止，都必須自動執行 `/train_summary` 指令產生完整訓練報告。

**Why:** 用戶希望每次訓練結束都有一份標準化報告存入 Obsidian vault (`/home/aa/Documents/Obsidian Vault/訓練報告/`)，避免手動補寫。

**How to apply:**
1. Cron 監控報告偵測到 PID 不存在時
2. 先完成最終監控報告（更新 run_metadata.yaml 的 actual_outcome）
3. 然後自動執行 `/train_summary {wandb_run_id}` 產生 Obsidian 報告
4. 適用於所有訓練 run，不限 SA1/SA2 或特定架構
