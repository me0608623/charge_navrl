---
name: teacher C_goal 的除零防護變成主導行為的參數，R=10 可修
description: 2026-08-17 max(D,1) 原意只是防分母趨零，卻造成 goal 距離 1.5-3m 的 clearance 危險帶；換成固定 R=10 消除危險帶且不靠降速換取
type: finding
date: 2026-08-17
status: measured-offline-synthetic
---

走廊 teacher 的目標項是

```
C_goal = ||g − p_H|| / max(||g − p_0||, 1.0)
```

那個 `max(⋯, 1.0)` 在筆記裡記錄的理由是**「避免逼近目標時分母趨零而爆炸」**
—— 它是一個**除零防護**，不是行為設計。

但它讓「前進 1 m 的價值」與距離**成反比**（`3.0/max(D,1)`），於是造成
[[finding_corridor_teacher_clearance_u_curve]] 那個 U 形危險帶。

## 實測（離線、純 torch、無 GPU）

`teacher_goal_denominator_probe.py`，**先完全重現** 2026-07-29 的 sweep
（誤差 `0.00000000`，校準障礙 x=2.5 m）才信新臂。把分母換成固定常數 R：

| 障礙側偏 0.3 m | 現行 max(D,1) | R=2 | R=4 | **R=10** |
|---|---:|---:|---:|---:|
| 危險帶(D=1.5–3m) clearance | 0.104 m | 0.104 | 0.104 | **0.382 m** |
| 放棄裕度（最大） | 0.325 m | 0.325 | 0.325 | **0.047 m** |
| 貼硬底線 ±1cm 的格子 | 3/72 | 3/72 | 1/72 | **0/72** |

**R=2 / R=4 無效**（還讓遠場更激進）。**R=10 完全消除危險帶且遠場幾乎不變**
（自我一致性檢查：現行規則在 D≥10 本來就等於 R=10，兩者在 D=10 給出相同值）。

## 進度守門（關鍵：確認不是靠變膽小換的）

| 危險帶 | 現行 | R=10 |
|---|---:|---:|
| 選中速度 | 0.90 m/s | **0.90 m/s（不變）** |
| 2 秒縮短的目標距離 | 0.98–1.63 m | 0.63–1.25 m |
| 停車格子數 | 0/72 | 0/72 |

→ **同速度繞更開**，不是變慢。代價每 2 s 少推進 0.3–0.5 m，換 +0.28 m clearance。

**Why:** 一個工程防護意外變成整個任務**最常經過區間**的主導行為參數。
這是「權重與門檻數值全部無來源」那個缺口造成的最具體後果，論文可直接引用為
「參數需要有來源」的實例。

**How to apply:**
- 要重新啟用走廊蒸餾**之前**先修 teacher；否則是把「近場貼障礙」教給 student
- R=10 只是「把遠場的保守程度套用到所有距離」，**不是掃出來的最佳值**
- 保留：合成單靜態障礙、單一初速、`reverse_velocity_scale` 未掃、
  動態暫停分支未測；量的是 cost 函數傾向不是實跑分佈
- 探針 `scripts/.../rnn_car_wdclean/probes/teacher_goal_denominator_probe.py`
  （commit `e6827dd19dc`），結果 `logs/probes/teacher_goal_denominator_probe.json`
- ⚠️ 7/29 那支 `clearance_probe.py` 因為只放 scratchpad **已遺失**，本次靠筆記
  文字重建。**離線探針一律存進 repo**，不可只放 scratchpad
- Obsidian `方法論/teacher設計驗收.md` §4.4.1
