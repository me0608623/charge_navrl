---
name: finding-sa7-corridor-ceiling-gate-results
description: SA7-r3四顆候選正式gate全FAIL;窄縫100%但走廊卡住;訓練SR最低的c90走廊最強
metadata:
  type: project
---

★★★ **SA7 (`sa7_wander_w1c10_ne1024_s42_r3`, W&B hvf16ewq) 未達 SA8 晉級標準**
（2026-07-28 過夜自動驗收，四顆候選正式長走廊 gate 全數 FAIL，**SA8 未建立未啟動**）

**正式 gate 矩陣**（門檻各模式 SR≥90%/CR≤10%/TO≤5%，seeds 515/616/717、1200 steps）：

| 候選 | lateral | longitudinal | random_2d | mixed_iid | 最差 margin |
|---|---|---|---|---|---|
| **c90** | **89.8%** | 77.3% | **69.8%** | **78.5%** | **−20.2pp**（最差=random_2d；四顆中最佳但僅領先 1.7pp）|
| c190 | 87.0% | 68.1% | 70.7% | 73.6% | −21.9pp |
| c160 | 87.5% | 67.8% | 70.4% | 73.5% | −22.2pp |
| c140 | 88.0% | 67.3% | 72.3% | 73.5% | −22.7pp |

**TO 全 0.0%、structural 全 True —— 失敗純粹來自碰撞。**

**三項關鍵發現**：

1. **窄縫能力完整保住**：Gate5a（sealed/stage7/arena13）六顆候選全部 SR/crossing/direct **100%**、CR **0%**。
   逐 env 密度混合訓練**沒有洗掉**窄縫能力 → 68/10/12/10 replay 比例設計有效。**走廊是唯一瓶頸。**

2. **訓練指標無法代理走廊能力**：訓練 SR 最低的 c90（89.2%）走廊最強；訓練最佳的 c210（92.8%/CR 6.9%）
   走廊 sentinel 只排第 6。**訓練 SR 是全場景混合平均，會掩蓋走廊子場景。**

3. **c90（iter 90）優於後續 210 個 iteration 的任何 checkpoint**。後段訓練把訓練 SR 從 89.2% 推到 92.8%，
   卻是在改善其他場景而**犧牲走廊**。→ 若要續攻走廊，從 c90 出發比從 final 出發合理。

**最弱模式因候選而異**：c140/c160/c190 是 longitudinal（−21.9～−22.7pp），c90 是 random_2d（−20.2pp）。四模式全部未達標，沒有單一瓶頸模式。
**注意**：只有 4 顆跑過正式四模式 gate，**不能宣稱 c90 優於全部 30 顆**。

**流程教訓**：sentinel 只跑 `random_2d,mixed_iid` 造成排序失真 —— longitudinal（最弱）從未進入篩選，
且 c90 在 sentinel 排第 4 卻是四模式綜合的 Pareto 最佳。**sentinel 的模式覆蓋範圍會決定候選順序。**
（sentinel 對其**涵蓋**的模式估計可靠：random_2d 誤差僅 1.5–1.7pp。）

證據：`docs/sa7_to_sa8_overnight_20260727.md`、`logs/gates/sa7_r3/c{N}/`。
訓練本身健康：300/300、零 traceback/NaN/OOM/dump、corridor audit 六項全達標。
