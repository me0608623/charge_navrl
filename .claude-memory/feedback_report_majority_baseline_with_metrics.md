---
name: 報分類指標必須同時報多數類基準
description: 2026-08-17 teacher learnability probe 的 angular within-1-bin 94.5% 其實低於「永遠猜直走」的 95.9% 基準，等於零資訊；沒有基準的準確率無法解讀
type: feedback
date: 2026-08-17
---

teacher learnability probe 原本報四個指標，其中兩個沒填 baseline。補算後：

| 指標 | 報告值 | 多數類基準 | **淨值** | 可用性 |
|---|---:|---:|---:|---|
| joint 精確 | 53.7% | 32.3%（恆猜 `(9,9)`）| **+21.4 pp** | ✅ |
| angular within-1-bin | 94.5% | **95.9%**（恆猜 angular 9）| **−1.4 pp** | ❌ **作廢** |
| linear 方向 | 84.4% | 45.7% | **+38.7 pp** | ✅ 最強 |
| angular 方向 | 77.8% | 72.6% | **+5.2 pp** | ⚠️ 很弱 |

成因：**教練 72.6% 的標籤就是「直走」**（bin 9）。任何「差一格以內」的指標都會
被這個不平衡撐高。原筆記把 94.5% 標成「高」——它其實低於常數預測。

**Why:** 沒有基準的準確率**無法解讀**。而且修正後結論改變了：不是籠統的
「特權 LiDAR 學不全」，而是**只有油門方向明確可學（+38.7 pp）、轉向方向幾乎
不可學（+5.2 pp）**。這與 [[teacher設計驗收]] §4.1 的互資訊分析獨立一致
（不同方法、不同資料、同一結論）——論文可作互相驗證。

**How to apply:**
- 報任何分類/一致率指標時，**同一張表**放多數類基準與淨值；只報原始百分比等於沒報
- 標籤不平衡時特別小心「within-N」「top-K」這類寬容指標
- 教練最有價值的資訊（往哪繞）恰好是學生最學不到的部分 —— 這是比「有落差」
  更具體的說法，寫論文用這個
- ⚠️ 尚未結案：linear 標籤集中在 bin 9(45.7%) 與 bin 18(~21%)，合計 66%，
  而這兩格在 `v=v_max` 時解碼成同一動作。要區分需重新 dump 並存原始
  `current_velocity`/`current_omega`（既存 dataset 的 `policy_input` 已被
  RunningStandardScaler 白化，物理值無法反推）
- 探針 `scripts/.../rnn_car_wdclean/probes/teacher_label_baseline_probe.py`
  （commit `e6827dd19dc`）
- Obsidian `09_走廊攻堅_特權蒸餾_SA6轉場.md`、`10_專有名詞與原理速查.md`、
  `方法論/teacher設計驗收.md` §4.4.4 皆已更正
