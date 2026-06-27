---
name: feedback_obs_index_layout
description: 79D obs layout 正確索引：[0]=acceleration, [1]=velocity, [2]=omega, [3]=radius, [4:6]=goal, [6:78]=lidar, [78]=time
type: feedback
---

79D observation layout（正確索引）：
- obs[0] = linear_acceleration（normalized）
- obs[1] = linear_velocity（normalized by v_max=1.0）
- obs[2] = angular_velocity（normalized by omega_max=1.5）
- obs[3] = collision radius
- obs[4:6] = goal position (2D, body frame)
- obs[6:78] = LiDAR 72 rays（normalized by max_range）
- obs[78] = time_remaining

**Why:** play 診斷誤讀 obs[0] 為 speed 導致錯誤結論「agent 速度只有 0.149」。實際 obs[0] 是 acceleration，speed 在 obs[1]。WandB 確認 speed_x_mean = 0.612 m/s。

**How to apply:** 讀取 obs tensor 時必須對照 cfg 中的 ObsTerm 定義順序，不能假設 index 0 = speed。
