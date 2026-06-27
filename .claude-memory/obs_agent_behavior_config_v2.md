---
name: Obs Agent BehaviorConfig v2 設計規格
description: 8 種 rule-based obstacle behavior 設計、BehaviorScheduler 調度、6-stage curriculum、RNN aux 關聯
type: project
---

Rule-based Obstacle Behavior Agent v2 — deterministic 非 NN 的動態障礙物控制器。

**Why:** 比 random walk 更有效率地訓練 RNN 時序表徵，行為完全確定性避免 non-stationarity。

**How to apply:**
- Obstacle 不再是簡單的「靜態 vs 動態」，而是 8 種 BehaviorConfig 控制的行為
- Play 時需用 BehaviorScheduler 驅動障礙物，不是 scripted interval events
- --scripted_obstacles flag 概念需升級

## 8 種 Behavior

1. **Static**: v=0, 提供 stationary LiDAR baseline
2. **Patrol**: 2-4 waypoints 週期巡邏, v=0.3-0.6 m/s, 50% ping-pong
3. **Random Walk**: 非週期, 每 5-25 步改方向, 3 步平滑轉向, v=0.2-0.7
4. **Horizontal Crossing**: 橫向穿越
5. **Path Crossing**: 路徑交叉
6. **Near Miss**: 預計算軌跡擦過 robot（不碰撞）
7. **Corridor Crossing**: 走廊穿越
8. **Occlusion**: 多物體遮擋群組

## 核心架構

| 檔案 | 功能 |
|------|------|
| `obstacle_agent/behavior_config.py` | BehaviorConfig + SafetyConstraints dataclass |
| `mdp/events/rule_behaviors.py` | 8 種 behavior step/spawn GPU 向量化 |
| `mdp/events/behavior_scheduler.py` | BehaviorScheduler: per-env 分配、調度、metrics |
| `curriculum/phases/rule_based_v1.py` | 6-stage curriculum config |

## 6-Stage Curriculum

| Stage | Behaviors | Obs | Speed | Goals |
|-------|-----------|-----|-------|-------|
| RB1 | 100% static | 3 | 0 | 10 |
| RB2 | 40%S+40%P+20%RW | 5 | 0.2~0.4 | 8 |
| RB3 | +15%HC+15%PC | 7 | 0.3~0.7 | 6 |
| RB4 | 均勻 7 種 | 8 | 0.4~0.9 | 4 |
| RB5 | 全 8 種含 occ | 10 | 0.5~1.0 | 3 |
| RB6 | 全 8 種 (NM 20%) | 10 | 0.5~1.0 | 2 |

## Scheduler 調度機制

- `behavior_type` tensor [E, N]: 0=inactive, 1-8 各 behavior
- Per-behavior-type batched tensor ops（非 per-obstacle loop）
- WandB metrics: behavior 分佈 / 速度 / collision attribution
- `collision_near_miss` 應始終 = 0（否則 precomputed trajectory 有 bug）

## RNN Aux 關聯

- Static → aux loss floor (vx=vy=0)
- Patrol → aux near_d 週期性下降
- Random Walk → aux loss spike at direction change
- Occlusion → RNN 必須靠 memory 維持被遮擋物體的位置預測
