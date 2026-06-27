---
name: sa1-sa7-baseline-results
description: SA1~SA7 全階段 baseline 訓練最終結果表（canonical run + WandB ID + SR/CR + checkpoint chain）
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

SA1~SA7 baseline 系列（A2C + extractor_rnn + wd_7d_geometry aux）最終結果。場景 SA1-SA4 為 20×20m，SA5-SA7 轉 T corridor。指標為各 run 末 10% 平均（WandB `me0608623-none/charge_skrl`）。

| Stage | canonical run | WandB ID | SR | CR | 場景/壓力 |
|-------|--------------|----------|-----|-----|----------|
| SA1 | sa1_a2c_aux_extractor_ne1024_s42 | yxo3tdz5 | 98.0% | 1.8% | 20×20, 無障礙 |
| SA2 | (探索期, 無 canonical) | 9zaegv8o 等 | best 98.2% | — | 確立 rule_based obstacle + WD clip + normalize_return |
| SA3 | sa3_a2c_aux_extractor_ne1024_s42_lowent | wnkpznf1 | 91.2% | 8.8% | +static+walls, VE 0.85 最高 |
| SA4 | sa4_a2c_aux_extractor_ne1024_s42 (baseline) | ij4ee1rj | 84.2% | 15.7% | +dynamic5+移動goal, 全系列最低 |
| SA4 | sa4_a2c_aux_partial_norm (選為主線) | bvnct7n3 | 78.8% | 21.5% | adv_norm partial, entropy 較健康 |
| SA5 TC | sa5_tc_g1_p30_ne1024_s42 | fqeu5i94 | 89.6% | 10.5% | T corridor 轉場, goals=1, penalty -30 |
| SA6 TC | sa6_tc_dense_ne1024_s42 | k1iayrfo | 90.5% | 8.4% | dynamic6+timeout penalty, CR 最低 |
| SA7 TC | sa7_tc_dense_ne1024_s42 (cont) | c5bhtwl4+xjoqhync | 89.4% | 10.0% | dynamic8+occlusion20%+penalty-50, 高壓 |

Checkpoint chain（主線, 各 checkpoint_270000 接續，SA6→SA7 用 checkpoint_330000）：
SA1(yxo3tdz5) → SA3(wnkpznf1) → SA4 baseline(ij4ee1rj) → SA4 partial_norm(bvnct7n3) → SA5 TC(fqeu5i94) → SA6 TC(k1iayrfo) → SA7 TC(xjoqhync, 終點 checkpoint_300000)。

關鍵事實：
- SA7 cont 在 2026-05-27 因 unattended-upgrade 自動更新 NVIDIA 驅動（580.126→580.159）造成 nvidia-smi mismatch，手動重開機修復後於 cont iter 1060/2000（53%）中止；成效已飽和 ~89%。已關閉 unattended-upgrades。
- TO 全系列皆 ~0%。SA5 起轉 T corridor 後 SR 穩定 89-90%，即使壓力遞增（penalty -5→-50, dynamic 0→8, 加遮擋）。
- SA7 SR 呈週期性波動（谷底 ~84% ↔ 峰值 ~90%），每次 20 iter 內自我恢復，非劣化。

報告位置：`/home/aa/Documents/Obsidian Vault/訓練報告/`（一 stage 一檔 + `SA1_SA7_全階段訓練總覽.md`）。
這些是舊架構 baseline，後續 v2 方向見 [[project-v2-roadmap]]。
