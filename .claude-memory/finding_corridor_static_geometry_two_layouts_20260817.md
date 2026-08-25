---
name: 走廊靜態幾何只有 2 種佈局，且 policy 對鏡像敏感
description: 2026-08-17 實測走廊靜態障礙是固定格位+±0.10m抖動+左右鏡像，只有 2 種骨架；抖動從不改變可通行性；policy 在兩鏡像上表現顯著不同，且差異方式隨行人運動型態反轉
type: finding
date: 2026-08-17
status: measured-single-seed
---

起因：**使用者 GUI 目視**發現走廊 reset 時靜態障礙每次都在同樣位置。
讀碼推論是「固定格位+抖動」，但那是推論，所以做了實測。

## 取樣器實測（20,000 draws，純 torch 無 GPU）

`corridor_geometry_diversity_probe.py`，呼叫生產 `sample_conflict_free_layout`。

**SA4 stage-4 實際密度是 `3S+2D`、`free_width 4.4 m`**（由真實 screen cell 的
`configured_static_obstacles=3` 確認 —— 我第一版探針誤用 4S，數字全錯過一次）。

- `static_x = 1.25` 固定、`static_y = (-3.0, -0.6, 0.6, 3.0)` 固定列、
  `static_xy_jitter = 0.10`、`dynamic_y = (-1.8, 1.8)` 兩條固定車道
- 靜態骨架**只有 2 種**：`-+-` 50.30% / `+-+` 49.70`%`，就是同一格位的左右鏡像
- 每 slot 的 `|x|` 只有一個叢集；靜態中心可達面積 **0.24 / 44.00 m² = 0.55%**
- **可通行性翻轉次數 = 0**：被擋側恆 `0.50–0.70 m`（機器人需 `0.80 m`），
  開放側恆 `2.9–3.2 m`。±0.10 m 抖動**從不改變路線拓樸**
- longitudinal 的動態起點**完全寫死**（`x=∓0.4, y=±1.8`，範圍 0）

## 閉環分組實測（SA4-R3 c6400、seed818、d1、64env）

儀器 `_long_corridor_static_layout_id`（`active*100 + x 正負號 bitmask`）＋
`play_rnn_car` 的 `static_layout_outcomes`。兩格整體都**精確重現既有 screen**
（lateral n=2215 SR 81.13% CR 18.87%；longitudinal n=1753 SR 78.21% CR 21.73%），
證明儀器未擾動 rollout。

| 場景 | layout | eps | SR | 障礙 CR | 牆 CR |
|---|---|---:|---:|---:|---:|
| lateral | 302 `-+-` | 1128 | 82.54% | 17.38% | **0.09%** |
| lateral | 305 `+-+` | 1087 | 79.67% | 16.65% | **3.68%** |
| longitudinal | 302 | 838 | 79.36% | 18.14% | **2.39%** |
| longitudinal | 305 | 915 | 77.16% | **22.73%** | **0.11%** |

- lateral：牆 CR 差 `3.59 pp`（約 1 次 vs 40 次），`z≈6.2` 顯著；
  障礙 CR 差 `−0.73 pp`，`z≈0.46` **不**顯著
- longitudinal：**撞牆的鏡像反轉**（302 高）；且**障礙 CR 也差** `+4.59 pp`，`z≈2.4`

**Why:** 三件事因此改變：
1. 「走廊訓練曝光不足」不只是數量問題 —— **題目只有一種拓樸**（含鏡像）。
   R3 screen 的 2,215 個 lateral 回合只覆蓋 2 種靜態骨架。
2. 堆輪次的預期報酬很低：等於把同兩張考卷再練幾千次。
3. policy 對鏡像有顯著反應，**但反應方式隨行人運動型態反轉** ——
   這既不符合「學到通用避障」，也不符合「單純偏好往某側轉」。

**How to apply:**
- 引用走廊 CR 時**不要**用 episode 數暗示幾何多樣性；那 2,000+ 回合幾何多樣性≈1
- 先做「增加靜態佈局隨機性」再談堆輪次；要讓抖動有意義需 ≥ ±0.35 m 才會翻轉可通行性
- ⚠️ 鏡像效應目前**單 seed、單 checkpoint**，鏡像與 seed 的交互未分離，
  寫論文前需補 seeds 515/616
- 探針/儀器/測試已 commit `82438e82b62`；`play_rnn_car.py` 消費端**尚未提交**
  （該檔另有 1,688 行平行 session 未提交工作）
- 資料：`logs/gates/sa4_layout_split/r1_20260817/`、
  `logs/probes/corridor_geometry_diversity_probe.json`
- 相關：[[finding_sa7_corridor_ceiling_gate_results]]、
  [[project_sa4_d9_replication_r3_pilot_result_20260817]]
