---
name: Warp Drive Project Overview
description: new_warp_drive 專案總覽 — 北科大碩士論文 Spot 行為規劃 RL，WarpDrive 框架，A2C + 模組化 RNN
type: project
---

## 專案來源
- 路徑: `/home/aa/IsaacLab/new_warp_drive/`（獨立 git repo: `github.com:jqu314159/new_warp_drive`）
- 論文: 「強化學習應用於機器狗之行為規劃策略」北科大 AI 碩士，汪品儀，2025

## 核心目標
用 RL 訓練 Boston Dynamics Spot 的行為規劃: goal-reaching + 動態避障 + 姿態控制 + 上下樓梯

## 技術棧
- **框架**: Salesforce WarpDrive (GPU-accelerated Multi-Agent Multi-Env RL)
- **演算法**: A2C (Advantage Actor-Critic)
- **環境**: 自建 2D Spot_3d 環境 + CUDA step kernel
- **模型**: 模組化 CustomModuleConnected — preprocess FC → RNN → gradient detach → RL head
- **訓練**: Multi-Phase curriculum (7+ phases) + Multi-Agent (Spot/Goal/Obstacle 交替)

## 關鍵創新
1. 模組化 NN: 分離動態障礙預測與策略決策，gradient detach 隔離
2. RNN auxiliary loss: PreProcess_Module_Loss 預測障礙物特徵，獨立 optimizer
3. Multi-Agent adversarial: Goal Agent 為移動目標（GAN 類比），Obs Agent 為動態障礙
4. car_mode: 限制 y 軸速度，模擬車輛運動模型

## 與 Isaac Lab 的關係
- 目前 Isaac Lab charge_skrl 專案不含 RNN/模組化/auxiliary loss
- 遷移目標: 借鏡 RNN 記憶 + 模組化架構到 Isaac Lab 的 vlp16_models.py
- 三套融合方案: 保守(RNN only) / 平衡(modular+aux loss) / 積極(multi-agent)
