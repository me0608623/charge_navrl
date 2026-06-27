---
name: project_goal_curriculum_plan
description: SA2+ curriculum 計劃：減少 goal 數量、增加最遠距離、goal 隨機移動，解決回合太短問題
type: project
---

SA1 回合長度只有 ~14.5 步（2.9 秒），原因是 num_goals=10 在場景中密度太高，agent 幾乎不需要導航就能碰到 goal。

用戶計劃在未來 stage 調整：

1. **減少 goal 數量** — 降低 num_goals，讓 agent 需要實際導航才能到達
2. **增加 goal 最遠距離** — 讓 goal 可以出現在場景更遠的位置，延長所需導航時間
3. **Goal 隨機移動** — 讓 goal 會移動，增加追蹤難度

**Why:** 回合太短導致 (a) aux valid_seq 不足（14.5 步 ≈ seq_len=15，幾乎無法切出有效序列）(b) 無法考驗 agent 的長程規劃能力 (c) RNN memory 沒有被充分利用

**How to apply:** SA2+ 的 curriculum phase config 需要同步調整這三項參數。設計新 stage 時要確保回合長度回到 30-60 步以上，讓 aux 有足夠 valid_seq 且 RNN 能發揮記憶優勢。
