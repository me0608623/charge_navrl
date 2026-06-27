---
name: v21 VO Shield 失敗教訓
description: 為什麼 inference-time VO shield 救不了 under-trained policy，以及 simplified VO heuristic 在 dense scene 的 ping-pong 問題
type: project
---

# v21 VO Shield 實驗失敗紀錄 (2026-04-08)

## 背景
v21 best_agent (agent_58590) 在 19s+15d 高動態場景 SR=51% / CR=49%。Advisor Round 3 建議用 VO Shield (NavRL-style) inference-time 救援，預期 CR → 7-12%。

## M1.1-M1.3 實驗結果

| 配置 | SR | CR | speed |
|------|----|----|-------|
| 無 shield | 51% | 49% | 0.70 |
| Distance shield (M1.1) | 54.3% | 45.6% | 0.67 |
| VO h=1.0 g=1.0 (M1.3 v1) | 55.9% | 43.9% | 0.68 |
| VO h=2.0 g=2.0 linear severity | 57.9% | 41.9% | 0.68 |
| **VO + sqrt severity + (1-s)² + hard fallback (Path A)** | **54.2%** | **45.7%** ⬆️ | - |

**Path A 反而變差了** — 讓 shield 更激進反而 CR 上升 3.8pp。

## VO Shield diag 數據

```
[VO_DIAG] step#1800 | visible_threats: 5/64 envs (8%) | avg_severity: 0.598
                      cumulative threats: 494, evades: 147
```

- visible threats: 每 step 5-9% envs 有 visible threat
- avg severity (sqrt 版): 0.55-0.60
- 33% threats 觸發 angular evasion (其餘 obstacle 在 robot 後方)

## 三個 Root Cause

### 1. Simplified VO heuristic 在 dense scene 會「ping-pong」
- 我寫的 VO 只看 worst threat (最小 TTC obstacle)
- Evasion 方向**沒考慮其他 obstacles**
- Robot turn away from A → 撞上 B
- 更激進的 evasion = 更糟的 ping-pong
- **NavRL 用真正的 LP solver 同時考慮所有 obstacles 的可行域交集**

### 2. v21 policy 本身就爛 (ds effective weight 0.31% of goal_velocity)
- closing_risk WAS enabled (line 731 of train_charge_ac.py)
- 但 effective weight 太弱:
  - RewTerm weight=2.0 × OE base_ds=0.32 = 0.64
  - vs goal_velocity ~166 (v05 monkey patch ×3)
  - ratio = 0.39%
- **v21 policy 從未真正學會避動態障礙物**
- Stage 8 SR=89% 是「reactive LiDAR + collision penalty」撐起來的
- 不是真的 closing_risk learning

### 3. NavRL paper 的 SOTA 數字假設 well-trained policy
- Paper Table II: NavRL **無 shield** dynamic CR = 2.70 → 有 shield 0.85
- NavRL 的 policy 本身在 dynamic 場景已經 SR > 95%
- v21 完全不在這個 baseline level
- Advisor Round 3 的 "VO shield CR 7-12%" 預測過於樂觀，沒考慮到 v21 policy 的 baseline

## 教訓：對 RL system 的反模式

### ❌ 反模式：用 inference-time shield 補 under-trained policy
- 「政策不會避障 → 加 shield 救援」是錯的
- shield 設計的角色是 last-mile safety net
- 必須先有 baseline-decent policy 才能加 shield
- shield 無法 substitute reward shaping

### ❌ 反模式：簡化 VO heuristic
- 真 NavRL VO 是 LP optimization
- 我寫的 single-worst-threat heuristic 在 dense scene 必然 ping-pong
- 要實作得對，要 multi-obstacle simultaneous projection
- 否則退化成 reactive distance shield

### ✅ 正確順序
1. 先讓 policy 在 reward 層面學會避障 (M1.4: 提高 ds effective weight)
2. 等 baseline policy SR > 70% on dense dynamic
3. 再加 shield 做 last-mile

## 對 advisor 報告的教訓

Advisor Round 1: 漏看 train_charge_ac.py:731 的 mode override → 錯誤判斷 closing_risk 沒啟用

Advisor Round 2: 計算 effective weight 比例 (0.31%) 是對的，但建議 22x boost 太激進，會讓 policy freeze

Advisor Round 3: 提出 VO shield 方向是對的，但**對 simplified VO heuristic 過度樂觀**，且**忽略了 v21 policy 的 baseline 水準遠低於 NavRL**

**通用教訓**: RL system 分析必須先 ground-truth check policy 的 baseline 能力，不要直接 extrapolate paper 的數字到不同 baseline 的 system。

## 下一步：M1.4 計畫

修 reward 重訓 policy:
- B1: OE base_ds 0.4 → 1.5 (3.75x，不是 advisor 原版 6.5x)
- B2: risk_sigma 2.0 → 1.2 (中庸)
- B3: b_risk 1.0 → 2.0 (中庸)
- B4: --reward_speed_v05 也補 ds ×3 (修原版 monkey-patch 漏洞)
- B5: 保留 distance shield 為 safety net
- B6: warm start v21 best_agent

預期 8-12 GPU hr → CR 49% → 25-30%, SR 51% → 65-70%

如果 M1.4 達到 CR ≈ 25%，再加 VO shield (路線 c+) 可能再降到 15-20%。
