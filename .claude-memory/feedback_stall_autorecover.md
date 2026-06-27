---
name: feedback_stall_autorecover
description: 訓練 stall（hung）且用戶長時間未回覆時，授權自動 kill + 從最近 ckpt resume 跑完，不需再等
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

訓練若**真的 stall/hung**（process wedged：log 長時間不更新 + GPU util 0% + iter 不前進，例如 CUDA context 卡死在 `.item()` sync），而**用戶一直沒有回覆**決策時，**授權自動處理**：kill 卡死的 process → 從最近的 checkpoint（stall 當下最新的 ckpt）resume → 跑完剩下的 budget。不必無限等用戶回 A/B/C。

**Why**：2026-06-13 SA1_v3d 在 iter ~647 CUDA wedged，我守「等用戶決定才 kill」，結果空等 6+ 小時、GPU 整段閒置浪費。用戶明確指示：以後遇到 stall 又沒人回，直接 kill+resume 跑完。

**How to apply**：
1. 先確診是**真 stall**（非 entropy 假象/慢）：log mtime 停滯 >15-20 分 + GPU util 連續 0% + py-spy 卡同一行 → 才算 stall。參考 [[finding_p95flip_entropy_confound]] 別把假象當死。
2. 確認 stall 後，若用戶在合理時間內（例如 1-2 輪巡檢 ~30 分）未回覆 → 自動 kill（wedged 需 SIGKILL）。
3. kill 後確認 GPU 釋放/恢復（記憶體回 baseline、util 正常）。
4. 從 stall 當下最新 ckpt resume：`--checkpoint <latest>.pt --timesteps <剩餘 iters×rollout> --run_name <orig>_resume`，env var 要與原訓練一致（如 `CHARGE_ACT_HIST_MODE`）。跑完剩餘 budget。
5. 報告已自動處置 + resume 進度。

這條是 [[feedback_let_runs_finish]] 的例外：平常中途 RED 不 kill，但**確診 stall + 用戶失聯**時要主動 kill+resume，不要枯等。

相關：[[feedback_let_runs_finish]] [[finding_p95flip_entropy_confound]] [[project_v3d_launch]]
