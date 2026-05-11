# Utils — 訓練輔助工具

支援訓練流程的 wrapper、normalizer、logger 等公用模組。

| 檔案 | 功能 |
|------|------|
| `aac_wrapper.py` | Asymmetric Actor-Critic Wrapper。擴展 IsaacLab wrapper，正確處理 shared_states (critic obs) 的傳遞 |
| `cadn_preprocessor.py` | Curriculum-Aware Dual-Rate Normalizer (CADN)。替代 SKRL RunningStandardScaler，per-branch 雙速率 EMA 在課程切換時快速適應分布漂移 |
| `charge_env_overrides.py` | CLI 參數 → env_cfg 覆寫。將 --v_gate_mode, --progress_gate_mode 等 CLI flag 轉為環境配置 |
| `console_summary.py` | 終端摘要 Logger。訓練過程中輸出結構化中文摘要，方便複製給 AI debug |
| `wandb_trainer.py` | WandB-enabled SKRL Trainer。擴展 SequentialTrainer，整合 WandB logging |
| `training_debug_logger.py` | 訓練 Debug Metrics Logger。記錄詳細的中間變數（grad norm, loss components 等）供除錯用 |
| `training_params_logger.py` | 訓練參數自動記錄器。啟動時生成完整參數檔（獎勵項目、權重、物理意義、預期走勢） |
| `wd_aux_targets.py` | WD-style 7D Privileged Geometry Target + Module Loss。移植自 Warp Drive 的 auxiliary target 計算與 log-clamped loss |
