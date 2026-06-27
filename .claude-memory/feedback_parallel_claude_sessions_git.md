---
name: feedback_parallel_claude_sessions_git
description: 同一 repo 有兩個 Claude session 平行動 git 會互踩 — 動前先 fetch 對齊、push 到分支不直接 main
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7b272734-c0b9-48e1-bb8b-79c55f6ca122
---

2026-06-12：PC 端與車端 (192.168.3.14) 兩個 Claude session 同時在 rover_rl repo 各自實作 v3c，差點重複 commit、各做一份 obs_builder/policy_node（API 不同），且車端版有 bug（action history clip 用 [-1,1]，訓練端是 [-2,2]）、缺 action_decoder slew。

**Why**：兩端從同一 base 平行開發、未先對齊，導致重工 + 分叉 + 衝突。git 在「兩份同功能實作」textual 不衝突時會**靜默 auto-merge 成 Frankenstein**（兩個 deque、兩個 obs build call），不可信。

**How to apply**：
- 動 repo 前一律先 `git fetch` 看 ahead/behind，落後就先 pull 對齊再改。
- 跨機傳工作用**分支**（`git push origin HEAD:xxx-wip`），不要硬推已分叉的 main（會被拒）。
- 合併兩份平行實作時，**不要信 auto-merge** 的同功能檔——挑一邊整檔為 base，再手動補另一邊缺的（如取車端工程設計 + PC 的 action_decoder slew）。
- 分工：PC 端負責訓練/匯出/合併，車端負責部署實測；別讓兩 session 同時動同一 repo 的 git。
- 純落後（0 ahead / N behind）= 乾淨 fast-forward，`git pull --ff-only` 即可，不需 merge。

相關：[[deploy_v3c_rover_rl]]
