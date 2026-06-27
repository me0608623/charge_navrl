---
name: feedback_no_topk_obs
description: 不要用 60D TopK/MOT obstacle obs — WD 只用 LiDAR angle bins，不用分離的障礙物特徵
type: feedback
---

不要在 obs 裡放 TopK obstacle 60D 特徵。

**Why:** WD 原始設計是用 Sweep_angle（36 bins 距離）= 等價於我們的 LiDAR 72 bins。WD 沒有分離的 MOT/TopK obstacle 觀測。RNN policy 的 `POLICY_OBS_INDICES` 已經跳過 60D（只用 79D = ego4 + goal2 + LiDAR72 + time1）。60D 是浪費計算。

**How to apply:** charge obs 應為 79D（不是 139D）。移除 `_compute_topk_obstacles()`。env 的 obs_dim 改為 79。
