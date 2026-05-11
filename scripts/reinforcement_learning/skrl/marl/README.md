# MARL (Multi-Agent RL) 模組

多 agent 強化學習訓練，對齊 Warp Drive 架構的 Isaac Lab 實作。

| 檔案 | 功能 |
|------|------|
| `train_marl_rnn.py` | MARL RNN 訓練入口 — N 台車 parameter sharing + vanilla RNN + auxiliary loss，雙 optimizer (RL head vs aux module) 分離設計 |
| `virtual_spot_utils.py` | 虛擬 Spot 工具 — 用 kinematic rigid body 模擬多台 policy-controlled car，支援 parameter sharing 與分析式 LiDAR 觀測合成 |
