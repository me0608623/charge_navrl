---
name: feedback_decode_each_stage_rnn_health
description: 用戶要求 reluFix lineage 每階段(SA1→SA5)都「一定要跑 aux 解碼確認 RNN 健康正常」,不可跳過;解碼是必要健康驗證關卡,非可選
metadata:
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**用戶指示（2026-06-24）**：reluFix lineage 每個階段（SA1→SA5）完訓/換棒前**一定要跑 aux 解碼確認 RNN 健康正常**，解碼是**必跑的健康驗證關卡，不可跳過**。

**Why**：reluFix 修的是 RNN velocity-aux 從未訓練的根因 bug（dead-ReLU）。整條 lineage 的核心是「RNN/aux 是否健康運作」，所以每階段都必須用解碼確認，不能只看 SR/CR 訓練曲線就放行。曾因誤讀「停止訓練 解碼 後接SA2」把解碼當成要取消（實際是「停訓→解碼確認→接下一棒」），用戶糾正。

**How to apply**：
- 每階段換棒/完訓時，跑 `play_rnn_car.py --aux_debug`（範本見 /tmp/aux_decode_sa1*.sh：stage N / 對應 curriculum / --deterministic --num_envs 32 --steps 250 --headless --lidar_no_noise --lidar_r_min 0.25）解碼。
- 健康判讀（兩層）：(1)**結構健康**=predict_head 輸出 signed 非0（非 dead-ReLU 的恆 0）+ 三模組 grad 非0 + preprocess_loss 是正常正值；(2)**預測學會**=速度相對誤差 <100%（稀疏障礙 SA1 註定 100% 只猜均值，但仍要跑確認結構健康；密集障礙 SA2-5 才考驗是否真學會追蹤）。
- 即使 rel err 100% 也要跑（確認結構健康、無回歸）。
- 凍結 checkpoint 解碼可與下一棒訓練並行（GPU 餘量足時），但留意 contention/stall。

相關：[[finding_predict_head_dead_relu]] [[project_reluFix_retrain_log]] [[feedback_detailed_report_each_cycle]]
