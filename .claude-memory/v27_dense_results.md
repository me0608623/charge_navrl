---
name: v27_dense_results
description: v27 dense curriculum 實驗結果 — HEIGHT TopK=10 在 22 total obs 的 capacity wall
type: project
---

v27 (navrl_hybrid_dense) 從 v26 best_model warm start，推高密度到 12s+10d。

**v27 密度 vs SR 完整軌跡：**
| total obs | nS | nD | SR | dynSR | 階段 |
|---|---|---|---|---|---|
| 12 | 6 | 6 | 77.2% | 62.5% | D0 |
| 16 | 8 | 8 | 72.8% | 56.3% | D1 |
| 18 | 10 | 8 | 66.9% | 47.8% | D2 |
| **22** | **12** | **10** | **61.6%** | **40.6%** | **D3 (卡住)** |

**Capacity wall 原因：**
- TopK=10 的 attention 只能覆蓋 22 個障礙物中的 10 個
- 剩餘 12 個只靠 LiDAR 18 tokens 間接感知
- dynSR 系統性低於 SR ~20pp，且 gap 隨密度擴大
- D3 dynSR 40.6% < 升級門檻 0.45，plateau 14+ 小時無改善

**v27 已驗證：**
- HEIGHT + hybrid curriculum 可推到 22 total obstacles (歷史新高)
- v24 天花板 62.5% (12 obs) vs v27 61.6% (22 obs) — 密度增 83% 且 SR 幾乎相同

**下一步方向（待討論）：**
1. 增大 TopK 10→20 — 讓 attention 覆蓋更多障礙物
2. 移除 dynSR 門檻 — 讓 policy 直接面對更高密度（暴力推）
3. 靜態/動態分開 TopK — 各自 TopK=10

**Why:** HEIGHT 架構在 12-16 total obs 表現優異，但 >18 obs 時 TopK=10 的覆蓋率不足。

**How to apply:** 下一個實驗 (v28) 需要架構改動而非單純調門檻。
