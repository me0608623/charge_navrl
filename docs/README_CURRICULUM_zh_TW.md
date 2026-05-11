# Charge-SKRL 課程學習

> **基於 SKRL PPO 的移動機器人四階段課程學習導航訓練**

[![English](https://img.shields.io/badge/lang-English-blue)](README_CURRICULUM.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-red)](README_CURRICULUM_zh_TW.md)

---

## 專案概述

本專案實現了一個**四階段課程學習框架**，用於在 Isaac Lab 模擬環境中訓練差速驅動移動機器人（Charge）在動態障礙物環境中執行自主導航任務，採用 SKRL PPO 演算法。

### 核心特色

- **四階段漸進訓練**：從密集探索到終極挑戰，逐步提升難度
- **Goal-Obstacle 聯動**：目標數量、距離與障礙物密度同步調整
- **動態獎勵權重**：碰撞懲罰隨階段加重，避免獎勵衝突
- **防止災難性遺忘**：支援混合課程調度器
- **VLP-16 LiDAR 感知**：72 點雷射掃描 + Top-10 物體觀測

---

## 四階段課程設計

### 第一階段：密集探索（Dense Exploration）

**目標**：學會「走到目標」

| 參數 | 值 |
|------|-----|
| 目標數量 | 8 個 |
| 目標距離 | 2.0 - 5.0 m |
| 靜態障礙物 | 0 |
| 動態障礙物 | 0 |
| 碰撞懲罰 | -30 |
| 升級條件 | SR > 72% |

**設計理念**：大量近距離目標提供頻繁的成功獎勵；無障礙物環境專注學習基本導航；輕碰撞懲罰不干擾方向感學習。

---

### 第二階段：稀疏導航（Sparse Navigation）

**目標**：學會「遠距離穩定導航」

| 參數 | 值 |
|------|-----|
| 目標數量 | 3 個 |
| 目標距離 | 4.0 - 8.0 m |
| 靜態障礙物 | 2 |
| 動態障礙物 | 1 |
| 碰撞懲罰 | -50 |
| 升級條件 | SR > 72% AND CR < 30% |

**設計理念**：較少目標且距離較遠，防止僥倖成功；漸進引入障礙物；保持導航為主，碰撞懲罰適中。

---

### 第三階段：安全避障（Safe Obstacle Avoidance）

**目標**：在已有導航能力上，學會「避開障礙物」

| 參數 | 值 |
|------|-----|
| 目標數量 | 2 個 |
| 目標距離 | 3.0 - 8.0 m |
| 靜態障礙物 | 5 |
| 動態障礙物 | 3 |
| 碰撞懲罰 | -200 |
| 升級條件 | SR > 80% AND CR < 20% |

**設計理念**：大幅提高碰撞懲罰讓 Agent「感受痛」；增加障礙物密度專注避障學習；Agent 已具備導航能力，不會混淆。

---

### 第四階段：終極挑戰（Ultimate Challenge）

**目標**：在密集動態障礙環境下精確導航

| 參數 | 值 |
|------|-----|
| 目標數量 | 1 個 |
| 目標距離 | 3.0 - 8.0 m |
| 靜態障礙物 | 5 |
| 動態障礙物 | 8 |
| 碰撞懲罰 | -300 |
| 降級條件 | SR < 20% OR CR > 50% |

**設計理念**：最終部署條件，單目標配合大量動態障礙物；最嚴格碰撞懲罰強制安全行為；允許降級防止卡住。

---

## 訓練配置

### 觀測空間（139 維）

| 模組 | 維度 | 說明 |
|------|------|------|
| Ego State | 4 | [v_x, v_y, cos(θ), sin(θ)] |
| Goal Command | 2 | 到目標的 [Δx, Δy] |
| LiDAR (VLP-16) | 72 | 72-beam 距離掃描 |
| Obstacles | 60 | Top-10 物體 × 6D (x,y,vx,vy,r,m) |
| Time | 1 | Episode 進度 |

### 動作空間

- **MultiDiscrete([19, 19])**：線速度 × 角速度
- 線速度：19 檔（-1.0 ~ +1.0 m/s）
- 角速度：19 檔（-0.25π ~ +0.25π rad/s）

### PPO 超參數

| 參數 | 值 |
|------|-----|
| Rollouts | 128 |
| Learning Epochs | 6 |
| Mini Batches | 8 |
| Discount Factor | 0.995 |
| Learning Rate | 1e-4 (KL Adaptive: 3e-5 ~ 1.5e-4) |
| Entropy Scale | 0.01 |
| Gradient Clip | 1.0 |
| Ratio Clip | 0.2 |

---

## 快速開始

### 訓練命令

```bash
# 課程學習訓練（自動從 Phase 1 開始）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Charge-Navigation-Curriculum-v0 \
    --num_envs 256 \
    --headless
```

### 從檢查點繼續

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Charge-Navigation-Curriculum-v0 \
    --checkpoint logs/charge_vlp16/checkpoints/best_agent.pt \
    --num_envs 256 \
    --headless
```

---

## 課程轉換邏輯

### 升級條件

```python
# Phase 1 → Phase 2
if SR > 0.72:
    upgrade()

# Phase 2 → Phase 3
if SR > 0.72 AND CR < 0.30:
    upgrade()

# Phase 3 → Phase 4
if SR > 0.80 AND CR < 0.20:
    upgrade()
```

### 降級條件

```python
# 任何階段（除 Phase 1）
if SR < 0.25 OR CR > threshold:
    downgrade()
```

---

## 設計原則

### 1. 避免獎勵衝突

- Phase 1/2：只教導航，碰撞懲罰輕（-30, -50）
- Phase 3/4：大幅提高碰撞懲罰（-200, -300）

### 2. 防止過度擬合

- Phase 1 升級門檻 72%（非 100%），避免卡在簡單場景
- 混合訓練確保各階段能力不退化

### 3. 動態獎勵調整

```python
# 課程自動調整獎勵權重
_apply_stage(env, stage)
rm.set_term_cfg("collision_terminal", weight=stage_cfg["collision_terminal_weight"])
```

---

## 效能目標

| 階段 | 成功率目標 | 碰撞率限制 |
|------|-----------|-----------|
| Phase 1 | > 72% | 不限 |
| Phase 2 | > 72% | < 30% |
| Phase 3 | > 80% | < 20% |
| Phase 4 | > 50% | < 20% |

---

## 檔案結構

```
charge_skrl/
├── cfg/
│   ├── charge_env_cfg_vlp16.py           # VLP16 基礎環境配置
│   └── charge_env_cfg_vlp16_curriculum.py # 課程環境配置
├── curriculum/
│   ├── goal_obstacle_curriculum.py        # Goal-Obstacle 聯動課程
│   └── mixed_curriculum.py                # 混合訓練調度器
├── agents/
│   └── skrl_ppo_cfg_vlp16.yaml            # PPO 超參數配置
├── mdp/
│   ├── actions/                           # 離散差速驅動動作
│   ├── observations/                      # LiDAR + 物體觀測
│   ├── rewards/                           # Potential-based 獎勵
│   ├── terminations/                      # 終止條件
│   └── events/                            # 障礙物隨機化
└── README_CURRICULUM.md
```

---

## 參考文獻

1. **Curriculum Learning** - Bengio et al., ICML 2009
2. **PPO** - Schulman et al., arXiv 2017
3. **SKRL** - https://github.com/Toni-SM/skrl

---

## 授權

本專案基於 Isaac Lab 框架開發，遵循其授權條款。
