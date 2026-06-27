---
name: Dense Dynamic Avoidance 文獻研究結論 (2026-04-08)
description: NavRL VO Shield, M1.4 reward boost 失敗教訓, 與 SOTA 改善方向 (HEIGHT/Bounded Rationality/Trajectory Prediction)
type: project
---

# Dense Dynamic Obstacle Avoidance — 研究結論摘要

## 背景
v21 best_agent (Stage 8 SR=89.7%) 在 19s+15d 高動態壓力測試 CR=49%。
目標 CR < 25%。

## 已測試方案 + 失敗原因

| 方案 | 結果 | 失敗原因 |
|------|------|---------|
| Distance shield (M1.1) | CR 49→46% (-3pp) | 不看 obstacle velocity |
| Simplified VO (single-worst-threat, M1.2) | CR 49→44% (-5pp) | dense scene ping-pong: evade A 撞 B |
| Sqrt severity + (1-s)² (M1.3 Path A) | CR 49→46% (反而變差) | 強化 evasion 不解 ping-pong 問題 |
| **M1.4 ds reward boost** | **未測完，6144 envs PhysX OOM** | hardware 問題，重啟 4096 envs |

## NavRL Paper 真相 (advisor 第三輪研究)

從 NavRL GitHub C++ source (`safeAction.cpp` + `solver.h`) 反推：

1. **NavRL 用的是 ORCA**（Reciprocal Velocity Obstacles），不是簡化 VO
2. **責任因子 = 1.0**（不是 reciprocal 0.5）— obstacle 視為被動
3. **Half-plane intersection**（凸集合保證），不是 cone intersection
4. **Seidel 2D LP** 自寫，~2ms/step on GPU (batched)
5. **Training 時 NavRL 不開 shield，只 inference time** — `env.py` 完全沒 shield code
6. **NavRL 的 horizon `τ` = 2.0s**（從 yaml config 反推）
7. **paper Table II**: dynamic CR 2.70 → 0.85 (-68%)
   - 但這是 well-trained policy + ORCA shield，不是純 ORCA
   - charge baseline 不可比（charge policy 弱於 NavRL）
8. **diff-drive 不能直接用 NavRL VO**：UAV 是全向，charge 是 unicycle
   - 需 NH-ORCA 或 unicycle projection 後處理

## 我們的 M1.2 simplified VO 為什麼失敗 (數學分析)

1. **Single-worst-threat → 退化成單一 half-plane**: ORCA 凸集 guarantee 失效
2. **(1-severity)² 與 VO 幾何無關**: 純時間函式，不告訴你「往哪轉」
3. **Angular evasion 沒考慮 cross-traffic**: head-on case 退化成只剎車
4. **sqrt(severity) 放大 TTC 估計雜訊**: PPO 學「躲 shield」而非避障

## SOTA 文獻 Top-5 改善方向 (2024-2025)

完整 paper 引用見 `/home/aa/IsaacLab/.claude/plans/research_2026-04-08_dense_dynamic_obstacle_innovations.md`

### 🥇 P0: HEIGHT-style Heterogeneous Graph Transformer + Cross-Attention
- Paper: Liu et al., HEIGHT (T-ASE 2026, arXiv:2411.12150)
- 替換 obstacle branch 的 `MLP+MaxPool` 為 `Multi-Head Attention + cross-attention(robot_query, obstacles_kv)`
- **直接攻擊 charge 的最弱點**: MaxPool 丟失 pairwise interaction
- 預期 CR 49% → 35-42% (信心高)
- 成本: 5-15 GPU hr, ~150 行 code, **warm start 可行**

### 🥈 P1: Bounded Rationality Adversarial Curriculum
- Paper: arXiv:2503.11467 (CoRL 2024)
- 動態 obstacle 變成 SAC adversary，QRE temperature 自動退火
- 直接修 OOD bug (train cap 0.40 vs eval 0.44)
- 預期 CR 49% → 38-45%
- 成本: 15-30 GPU hr, 雙 RL training

### 🥉 P2: N-step Trajectory Prediction Augmentation
- Papers: ICRA 2023 (arXiv:2203.01821) + Future-Oriented Nav 2025 (arXiv:2505.00237)
- obs 增加 N-step constant velocity prediction (139→179D)
- 預期 CR 49% → 42-46% (中)
- 成本: 1-3 GPU hr，**最低成本**
- 與 P0 正交可疊加

### 不入選: Pure VO Shield, Pure LSTM, TTC features
- VO Shield: M1.2 失敗陰影 + 簡化版必然 ping-pong
- LSTM: POPGym 證明速度已給的話邊際效益小
- TTC features: 不是 SOTA 主流路線

## 推薦執行順序
```
Week 1 (P0): HEIGHT-style attention encoder (warm start, 8h)
Week 2 (P1): Bounded Rationality Adversary
Week 3 (P2): Trajectory prediction obs augmentation
```

## 對 M1.4 reactive boost 的最終定位
M1.4 ds reward boost 的價值是 **diagnostic experiment**:
- 若 v22 (4096 envs) CR 49% → 35%: 證明 reward 訊號弱**就是**主要 bottleneck
- 若 v22 CR 49% → 40-45%: 證明 architecture 才是真 bottleneck → 走 P0 HEIGHT
- 若 v22 CR 49% → 50%: reward boost saturated → 必須改架構

不論結果，v22 都會告訴我們下一步該走哪條。
