---
name: Warp Drive Train Car
description: train_rnn_car.py 完整追蹤 — 入口/呼叫鏈/car_mode/env/model/obs/action/reward/雙 loss
type: reference
---

## 入口
- 檔案: `new_warp_drive/train_rnn_car.py`
- 執行: `cd new_warp_drive && python train_rnn_car.py`
- 入口: `Spot_Env().train()` (Line 797-799)

## 核心特徵: car_mode=True (Line 96)
- CUDA kernel 影響: 遮蔽 y 軸 obs (step.cu:526)、y 軸速度懲罰 (:1084)、目標判定 (:1122)
- `results_dir_ = "car_spot_pretrain"` (Line 25)

## 呼叫鏈
```
Spot_Env.train() → for PHASE: Calculate_PHASE_Parameter → for env:
  cuda_init → env_init(spot/goal 交替 3:1) → creat_RL_ENV:
    Spot_3d(spot_3dmodule) → EnvWrapper(cuda) → Trainer(config)
      spot: CustomModuleConnected (module_model)
      goal: CustomModelConnected (fully_connected)
  → load_weight(WandB artifact) → trainer.train() → clear_reg
```

## 模型配置
- spot_model: [256, 32] FC
- module_network: [64, 'rnn', 32, 'rl']
- policy_network: [256, 256, 256, 512]
- critic_network: [256, 256, 256, 512, 512]
- memory_dim: 30, rnn_layers: 1, rnn_seq_len: 15, rnn_type: RNN

## 訓練參數
- algorithm: A2C | spot_lr: 0.0002 | rnn_lr: 0.0005
- spot_action_levels: 19 | rl_fps: 4
- spot_reward_get_goal: 40.0 | spot_penalty_hit: -5 to -100 (phase-dependent)
- train_num_env: 180 | env_train_times: 100

## 雙 Loss (custom_trainer.py:943-1003)
1. PreProcess Module Loss: 先執行，獨立 optimizer，訓練 RNN + preprocess
2. A2C RL Loss: 後執行，獨立 optimizer，訓練 policy + critic
梯度隔離: module_connected.py:505 `concat_input = rl_in_.detach()`
