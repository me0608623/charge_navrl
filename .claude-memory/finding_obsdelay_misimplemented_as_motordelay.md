---
name: finding_obsdelay_misimplemented_as_motordelay
description: v3「延遲建模」把筆記要的『動作/馬達延遲』誤植成『觀測延遲 obs_delay_steps』；兩者不同，馬達延遲是治實車極限環的藥方要保留，觀測延遲是 sim 害 policy 滿舵塌縮的兇手要移除
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**核心釐清（2026-06-20，對齊 [[finding_cmd_delay_limit_cycle]] 與 [[finding_bangbang_degenerate_penalty0]]）**：config 有兩個獨立延遲開關，v3 把藥方寫錯類型。

**兩種延遲不同東西**：
- **馬達/致動延遲（動作端）**：指令 →[延遲~200ms]→ 馬達 → 實際車速。config = `enable_actuator_dr` + `actuator_delay_range` + `actuator_motor_lag`。**這是 [[finding_cmd_delay_limit_cycle]] 實車舞龍舞獅的物理根因，也是它開的藥方（訓練端建模動作延遲）。要保留。**
- **觀測/感測延遲（感測端）**：真實狀態 →[延遲]→ policy 看到的觀測（ring buffer 延遲整個 79D，含本體速度）。config = `obs_delay_steps`。**這是 sim 裡 policy 學成 deterministic bang-bang 滿舵的兇手。要移除。**

**誤植經過**：finding_cmd_delay_limit_cycle 的「How to apply」把藥方寫成「v3 起 obs_delay_steps=[0,1] 已部分採納」——但筆記正文要的是「action delay buffer ~1步」（動作延遲），結果實作加成了 `obs_delay_steps`（觀測延遲）。**類型錯了**：要延遲的是「動作→馬達」，卻延遲了「policy 的觀測」。觀測延遲多延遲了 policy 自己的速度回授 → 過補償 → 滿舵塌縮。

**隔離矩陣證明（皆 stage3/79D/從 sa2_v3f/210000/iter100 deterministic 空曠 eval）**：
| 實驗 | 馬達延遲(藥方) | 觀測延遲(誤植) | 空曠飽和 |
|------|:---:|:---:|:---:|
| nodelay | OFF | ON | 100%（關藥方也壞→不是馬達延遲在 sim 害 policy） |
| noobsdelay | **ON(保留)** | **OFF** | **0%(乾淨)** |

**正解 = 保留馬達延遲(actuator_dr，筆記藥方) + 移除觀測延遲(obs_delay_steps，v3 誤植)** = noobsdelay config。與筆記完全一致，只修掉類型誤植。

**補充判斷**：真實 LiDAR 管線延遲 ~30ms，相對 dt=200ms 僅 ~0.15 step，`obs_delay_steps=[0,1]`（0 或整步 200ms）反而**高估**真實感測延遲。真正有害的延遲在致動通道（200ms，已由 actuator_dr 建模）。故 obs_delay 整個關掉（或最多只延遲 LiDAR slice [6:78]、本體速度永遠即時），不該延遲整個觀測。

**✅ 修好版 SA3 baseline 完訓（2026-06-22 00:33）**：obs_delay 修掉後從 sa2_v3f/210000 接 stage3 訓練完成（中途 1 次 CUDA wedge stall 自動 kill+resume，最終 capped 600 iters 湊滿原 1400 預算）。最終 checkpoint = `logs/rnn_car/sa3_v3f_noobsdelay_cont_ne1024_s42/checkpoint_180000.pt`（效果 iter 1400，SR ~89%、CR ~11%、sw 全程低、deterministic 無 sin 波）。這是接 SA4 的乾淨起點（SA4 用已修好 obs_delay[0,0] 的 config）。

相關：[[finding_cmd_delay_limit_cycle]] [[finding_bangbang_degenerate_penalty0]] [[finding_action_error_selfref_oscillation]]
