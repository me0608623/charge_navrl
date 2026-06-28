---
name: feedback_stall_autorecover
description: 訓練 stall（hung）且用戶長時間未回覆時，授權自動 kill + 從最近 ckpt resume 跑完，不需再等
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

訓練若**真的 stall/hung**（process wedged：log 長時間不更新 + GPU util 0% + iter 不前進，例如 CUDA context 卡死在 `.item()`/`.nonzero()` sync），**一律自動處理，不需問、不需等用戶回覆**：kill 卡死的 process → 從最近的 checkpoint（stall 當下最新的 ckpt）resume → 跑完剩下的 budget。

**Why**：2026-06-13 SA1_v3d iter ~647 CUDA wedged，我守「等用戶決定才 kill」空等 6+ 小時浪費 GPU。2026-06-28 SA2_v3f_vaux iter 990 又 CUDA hang（py-spy 5/5 卡 `sample_aux_sequences:943` `.nonzero()`、GPU 0%、log 凍 25 分），我確診後仍停下問 A/B/C；用戶當場再次明確指示：**「若再發生 stall 則自動 resume」**——以後確診 stall 直接 kill+resume，不要再問。

**How to apply**：
1. 先確診是**真 stall**（非 entropy 假象/慢/別人 GPU job 競爭）：log mtime 停滯 >10-15 分 + GPU util 連續 0% + py-spy 連續多次卡同一行 + 無他人 compute job（nvidia-smi 查）。三訊號齊全才算。參考 [[finding_p95flip_entropy_confound]]、[[finding_shared_gpu_contention_fake_stall]]。
2. 確認 stall → **立即自動 kill（wedged 需 SIGKILL -9 python PID），不必等用戶**。
3. kill 後確認 GPU 釋放（記憶體回 baseline、我方 compute proc 消失）。
4. 從 stall 當下最新 ckpt **自動 resume**：`--checkpoint <該 stage 最新>.pt`，run_name 沿用原 run，env var 與原訓練一致（v3f 須 `CHARGE_USE_ACT_HIST=0`、`--feat_norm --aux_epochs 8`）。
   - ⚠️ **resume 計數器語義（2026-06-28 讀 train_rnn_car_wdclip.py L3006-3064 確認）**：`--checkpoint` 只還原權重+normalizer+optimizer，**不還原 iteration/curriculum 計數器**（L4537 存的 iteration 沒被讀回）→ resume 會計數器歸零、重跑完整 budget。權重續用故非從零，但 iter 顯示重來。這是已知代價，**仍照 resume**（用戶要的是別枯等、把 budget 跑完）。
5. 報告已自動處置 + resume 進度，PushNotification 通知，繼續監控 loop。

這條是 [[feedback_let_runs_finish]] 的例外：平常中途 RED 不 kill，但**確診真 stall** 時主動 kill+resume，不枯等、不問。注意 loop prompt 內的「異常即停別resume」是指 metric 發散（rel_err突升/SR崩/NaN）那種，**不適用於 hang**；hang 從健康 ckpt resume 是正解。

相關：[[feedback_let_runs_finish]] [[finding_p95flip_entropy_confound]] [[project_v3d_launch]]
