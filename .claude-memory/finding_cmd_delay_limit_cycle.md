---
name: finding_cmd_delay_limit_cycle
description: 實車 goal-following 舞龍舞獅 = 致動延遲極限環；推論端補償會惡化；須訓練端建模延遲
metadata: 
  node_type: memory
  type: project
  originSessionId: 68b75281-1394-4620-b219-a466e0575982
---

2026-06-08 實車部署（CampusRover, VLP-16, 5Hz）發現 goal-following 車頭左右擺動（舞龍舞獅）的根因：

- **致動死時間 ~200ms**（ω 通道互相關量測，兩趟一致；LagEstimator 150ms r=0.97）。v 通道互相關不可信（等速直行訊號太平、誤報 600ms）。
- `delay / control_dt ≈ 1.0`（200ms/200ms）→ 控制理論最差相位裕度區 → 延遲驅動極限環（0.2~0.43Hz）。震盪在 policy 原始輸出 `rl_w` 就存在，非濾波/底盤造成。
- **無效止血**：cmd 濾波（震盪頻率低於截止、且在 policy 下游）；speed_rate 降速只 −20% 幅度、頻率不變。
- **反例（重要）**：推論端 obs 延遲補償（predict pose forward 0.2s）**反而更糟**（act_w RMS +35%，頻率 0.22→0.43Hz）。因 damping 項本身也過 200ms 死時間 → 變成「透過延遲的微分回饋」、過增益。
- **根治 = 訓練端**：(1) action delay buffer ~1步 + DR delay∈[0.1,0.3]s；(2) anti-jitter smoothness penalty（罰相鄰幀動作差）；(3) per-env p95 抽動監控（全 env 平均會掩蓋單 env 抽動）。

**Why**：這是 sim-to-real 被忽略的關鍵 gap，且推論端線性補償的失敗反向證明訓練端建模的必要性。
**How to apply**：v3 起 `obs_delay_steps=[0,1]` + `penalty_smoothness` 已部分採納，SA1_v3b 驗證方向正確（見 [[project_v3_launch]]）。重訓時順帶修 LiDAR 高度 1.6→1.43m、ω_max 2.0→1.2。完整論文化見 Obsidian `論文/34_實機延遲不匹配與抗抽動.md`。關聯 [[finding_policy_oscillation_stuck]]、[[feedback_per_env_sampling]]。

**⛔ 更正（2026-06-20）**：上面「How to apply」把藥方寫成 `obs_delay_steps=[0,1]` 是**類型誤植**。本筆記正文要的藥方是延遲「動作→馬達」（action delay buffer），但 `obs_delay_steps` 延遲的是「policy 的觀測」——**馬達延遲 ≠ 觀測延遲**。正確做法是用 `actuator_delay_range`（馬達延遲，保留）而非 `obs_delay_steps`（觀測延遲，要移除）。觀測延遲反而在 sim 把 policy 訓成 bang-bang 滿舵。完整釐清見 [[finding_obsdelay_misimplemented_as_motordelay]]。
