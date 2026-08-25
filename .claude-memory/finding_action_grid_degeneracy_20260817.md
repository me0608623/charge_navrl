---
name: 19x19 動作網格有 42-70% 是重複指令，同時影響 teacher 與 policy
description: 2026-08-17 用真實訓練參數實測，361 個 bin 在巡航時只有 110 個不同 (v,omega)；角速度軸因 slew < omega_max 永久浪費 8 格；index entropy/KL 因此高估行為多樣性
type: finding
date: 2026-08-17
status: mechanism-verified-consequence-unmeasured
---

用**真實訓練設定**（`max_angular_vel=1.2`、`max_angular_accel=3.0`、`dt=0.2`、
`v_max=1.0`、`a_max=0.5`）實測 `decode_discrete_drive_action_grid`：

| 入速 | 不同 v | 不同 ω | 不同 (v,ω) | 重複率 |
|---|---:|---:|---:|---:|
| 0.0–0.9 m/s | 19 | **11** | 209 / 361 | **42.1%** |
| 1.0 m/s（巡航） | **10** | **11** | 110 / 361 | **69.5%** |

兩個成因：

1. **角速度軸永久浪費 8 格。** slew 上限 `α_max·dt = 0.6 rad/s` 小於
   `ω_max = 1.2 rad/s`，目標 `|ω| > 0.6` 一步到不了，全被夾到邊界
   （bin 0–4 全 `−0.6`、bin 14–18 全 `+0.6`）。**在每個速度下都成立。**
2. **線速度軸在 `v = v_max` 剩 10 格**：正向加速被 `(v_max−v)/dt` 夾成 0，
   bin 9–18 全部等於「維持 1.0 m/s」。

佐證：GUI 實跑量到 `|omega|p90 = 0.600 rad/s`，正好貼在 slew 上限。

**Why:** 這**不只影響 teacher，也影響 policy** —— policy 輸出頭是
`MultiDiscrete([19,19])`，走**同一個解碼器**。因此：
- 361 個 logit 中 42–70% 對應重複指令 → **回報的 entropy 高估行為多樣性**
- 探索有一部分花在互相等價的動作上
- **索引空間的 KL ≠ 行為空間的 KL**，會牽動 entropy 介入判準與 KL retention 解讀
- 對 teacher 蒸餾：同狀態下與選中動作**物理完全相同**的可行格中位數 **10 個**
  （最多 50），所以「精確 bin 一致率」不等於「行為一致率」

**How to apply:**
- 報 entropy / KL 時標明是**索引空間**；不可直接當行為多樣性
- 修法有三種，**都會讓舊 checkpoint 行為改變、成績不可比**：
  (a) 讓 bin 只覆蓋可達範圍（語意變成「轉速增量」，動作空間定義改變）
  (b) 降 `ω_max` 1.2→0.6（最大轉速砍半）
  (c) 升 `α_max` 3.0→6.0（⚠️ 該值是為對齊實車 cmd filter 防振盪而設，動它等於放寬已知防護）
- **尚未量**「索引 entropy vs 行為 entropy 差多少」——機制已驗證，後果未量化
- ⚠️ 退化**不是** teacher 53.7% 天花板的成因：僅 0.31% 標籤落在崩塌的角速度邊界格
  （教練幾乎不打大舵）；linear 軸未結案，見 [[feedback_report_majority_baseline_with_metrics]]
- 探針：`scripts/.../rnn_car_wdclean/probes/teacher_argmin_stability_probe.py`，
  結果 `logs/probes/teacher_argmin_stability_probe.json`（commit `e6827dd19dc`）
