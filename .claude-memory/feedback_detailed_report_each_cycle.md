---
name: feedback_detailed_report_each_cycle
description: "用戶要求每次監控/巡檢都要「詳細回報」(非一行簡短)並「寫到記憶」;reluFix 重訓期間每個 cycle 給完整指標+aux 狀態+解讀,並把快照寫進進度記錄檔"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**用戶指示（2026-06-23）**：每次監控報告都要**詳細回報 + 寫到記憶**，不要一行簡短。

**Why**：reluFix 整條重訓是重大實驗（修了 RNN 從未訓練的根因 bug），用戶要追蹤完整軌跡 + 持久記錄，不只看當下一行。

**How to apply**：
- 每個監控 cycle 給**完整詳細報告**：SR/CR/TO 軌跡（非單點）+ ent/sw + fps/GPU + ★aux 三模組 grad（predict_head/rnn/extractor 是否非0）+ aux/preprocess_loss 軌跡（是否下降=預測變準）+ 有 checkpoint 時的解碼相對誤差 + 文字解讀（在學什麼/正不正常/該擔心什麼）。
- 每個 cycle 把詳細快照**寫進進度記錄檔** [[project_reluFix_retrain_log]]（append 時間戳 + 指標一行表），保持時間序記錄。
- 覆蓋先前「每輪簡短除非異常」的指示（[[feedback_monitor_interval]] 仍 15 分一次，只是報告改詳細）。

相關：[[finding_predict_head_dead_relu]] [[project_reluFix_retrain_log]]
