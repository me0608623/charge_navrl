---
name: Warp Drive Train RNN
description: RNN 模組化訓練分析 — 嵌入 policy 的 auxiliary 訓練，非獨立流程
type: reference
---

## 結論
RNN 不是獨立訓練流程，而是 Spot policy 內 CustomModuleConnected 的子模組。
有獨立 lr 和 auxiliary loss，但在同一 trainer.train() 中同步執行。

## 模組化架構 (module_connected.py)
```
Obs → preprocess_front FC(→64) → RNN(64→30, concat→94) → middle FC(94→32)
  → [training only] back FC(32→feture_dim) → PreProcess_Module_Loss (auxiliary)
  → [GRADIENT DETACH] → cat(obs_raw, preprocess.detach())
  → policy FC [256,256,256,512] → action heads
  → critic FC [256,256,256,512,512] → value head
```

## 梯度流
- PreProcess_Module_Loss → backward → 更新 preprocess_front + RNN + middle + back
- A2C Loss → backward → 更新 policy_network + critic_network + policy_head + vf_head
- detach 點: module_connected.py:505 `concat_input = rl_in_.detach()`
- 因此: A2C 不訓練 RNN，只有 preprocess loss 訓練 RNN

## RNN 記憶管理
- memory_banch: 每 rnn_seq_len=15 步存一次 h_t
- 訓練時從 memory_banch 重建序列 → permute → reshape → 傳入 model
- 依據: custom_trainer.py:960-963, module_connected.py:139-140

## 配置
- rnn_type: RNN (可選 LSTM/GRU, Line 345)
- memory_dim: 30 | n_layers: 1 | rnn_seq_len: 15 | concat_rnn: True
- rnn_model_lr: 0.0005 (獨立於 RL lr 0.0002)
