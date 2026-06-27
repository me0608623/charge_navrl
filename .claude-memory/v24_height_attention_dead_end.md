---
name: v24 HEIGHT Attention 半套實作失敗分析
description: v24 只把 attention 加在 dynamic obstacle 分支，LiDAR 仍 Conv1d+MaxPool(1)。Stage 7 SR 卡 62.5% / CR 37.5% 9.5h 無突破。碰撞 61% 來自靜態 → 動態 attention 無法救靜態。
type: project
---

## 最終訓練狀態（v24_height_attention_s1_ne4096, 中止於 ~9h55m）

| 指標 | 值 |
|---|---|
| Stage | 7/8（6 static + 6 dynamic）|
| SR | 0.625 |
| CR | 0.375 |
| dynamic_sr | 0.461 |
| collision_static_ratio | **0.61** ← 主導 |
| collision_dynamic_ratio | 0.39 |
| lidar_min_mean | 0.41m（門檻 0.45m） |
| freeze_ratio | 0.019（正常）|
| oscillation | 0.071（正常）|
| retreat_ratio | 0.066（正常）|
| speed_avg | 0.45 |
| speed_near_obstacle | 0.46（不保守）|
| path_efficiency | 2.55 |
| KL (stage7 頭→尾) | 0.27 → **0.47** 發散 |
| policy_loss | ≈-0.001（advantage collapse）|

## 根因（rl-research-advisor 確認）

### 1. 架構瓶頸：LiDAR branch `AdaptiveMaxPool(1)`

v24 feature extractor 結構：
```
LiDAR 72 → Conv1d stack → [B, 64, 18] → AdaptiveMaxPool(1) → [B, 64, 1] → 64D
```
把 18 個 spatial tokens pool 成 channel-wise scalar → **丟失所有角度位置資訊**。
policy 只知道「某方向有東西多近」的 channel 統計量，不知道是哪個角度。

v24 的 attention 只作用在**動態 obstacle 分支**（60D Top-K → 32D），靜態側一點都沒修。
→ 撞靜態 61% vs 撞動態 39%，attention 救不到。

### 2. Advantage collapse 症狀（下游）

reaching_goal reward 0.80 主導 return，collision_terminal 只 -0.045 (5.6%)
→ 多數 transition advantage ≈0
→ policy_loss ≈0, 但少數 outlier 推梯度
→ KL 爆表（0.47），訓練 thrashing 不收斂

## v24 vs v25 的 learning

1. **HEIGHT attention 不是改動態就好** — 必須 static/dynamic 都做 heterogeneous attention
2. **MaxPool 在 LiDAR 上是致命的** — 空間任務一定要保留 spatial tokens
3. **Reward 失衡會掩蓋架構改善** — 但必須先隔離單一變因驗證
4. **v24 warm-start 從 v21 沒意義** — 架構改動後 weight 不相容

## 不該做的事（教訓）

- ❌ 只改一半的 heterogeneous attention（必須 static + dynamic 都改）
- ❌ LiDAR 用 AdaptiveMaxPool(1)（空間資訊被毀）
- ❌ 在架構有問題時先調 PPO hyperparam（KL 降 LR 是壓症狀）
- ❌ 同時改架構 + reward（無法 attribute）

## 下一步 → v25
完整 HEIGHT 實作，見 [v25_heterogeneous_attention.md](v25_heterogeneous_attention.md)
