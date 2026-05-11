# RNN Car Experiments Ledger

> 追蹤所有 train_rnn_car_wdclip 相關的 code 改動與實驗。

---

## code_20260508_01 — Experiment Config Profiles (Phase 1)

**日期**: 2026-05-08
**類型**: code infrastructure
**者**: Claude Code (PC-A)

### 變更

新增 LEGO-style experiment config 系統，用 `--experiment_config <name>` 取代大量 CLI 參數。

### 修改檔案

- `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py` — 新增 --experiment_config / --print_experiment_config
- 新增 `scripts/reinforcement_learning/skrl/rnn_car_modular/experiment_config.py`
- 新增 `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/` (registry + 4 built-in configs)

### Built-in Configs

| Config | Algorithm | Aux | Entropy (lin/ang) |
|---|---|---|---|
| wd_sa2_a2c_aux | A2C-WD | WD 7D geometry | 0.05 / 0.10 |
| wd_sa2_a2c_aux_lowent | A2C-WD | WD 7D geometry | 0.01 / 0.02 |
| wd_sa2_a2c_noaux | A2C-WD | none | 0.05 / 0.10 |
| wd_sa2_ppo_noaux | PPO clip | none | 0.05 / 0.10 |

### 驗證

- py_compile: 8 檔全通
- Unit tests: 5/5 pass
- Backward compatible: 不加 --experiment_config 時行為不變

---

## smoke_20260508_01 — Experiment Config Smoke Test

**日期**: 2026-05-08
**類型**: smoke test
**者**: Hermes (PC-A)

### 測試

```
--experiment_config wd_sa2_a2c_aux_lowent --headless --num_envs 4 --timesteps 300 --rollout_length 30
```

### 結果: 全 PASS

- 38 欄位正確套用 (initial_stage=2, fixed_stage=True, use_a2c=True, aux=tbptt, ent=0.01/0.02, lidar_no_noise=True)
- CLI override 正確優先 (num_envs=4 覆蓋 config 的 1024)
- WandB 上傳成功，run name 正確
- 10 iterations 完整跑完，無 crash
- Boolean flag mapping 正確

### 結論

Phase 1 config 系統驗證通過，可進入正式實驗。

---
