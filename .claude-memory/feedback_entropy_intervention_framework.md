---
name: Entropy Intervention Framework
description: SA4+ entropy 鋸齒谷底監控三維度 + 介入方案（ent_coeff ×1.25~1.5 優先）
type: feedback
---

## 鋸齒谷底歷史

| 谷底 | Iter | SR 最低 | CR 最高 | LE 最低 | AE 最低 | 恢復 SR | 恢復 LE | 恢復 AE |
|------|------|---------|---------|---------|---------|---------|---------|---------|
| 第一次 | ~90 | 81.9% | 18.3% | ~2.86 | — | 84.6% | 3.52 | — |
| 第二次 | 260 | 76.8% | 24.5% | 1.598 | 1.523 | 84.4% | 1.80 | 1.74 |

## 三維度監控（第三次谷底）

### 維度 1：谷底是否更深？
**立即介入門檻**（不再等）：
- SR < 75%
- 或 CR > 25%

### 維度 2：恢復高點是否變弱？
目前恢復到 SR=84.4%。如果下一波恢復後：
- SR 只到 82%
- CR 仍 17~18%
→ 每次反彈都變弱 = 訓練正在退化

### 維度 3：entropy 反彈高點是否下降？
目前反彈到 LE=1.798, AE=1.743。如果之後反彈高點：
- LE < 1.65
- AE < 1.60
→ 探索能力逐步萎縮

## 舊有 7 條觸發條件（仍適用，≥ 2 條觸發）

1. SR < 83%
2. CR > 17%
3. LE 反彈後 < 1.45
4. AE 反彈後 < 1.55
5. collision_length < 32 steps
6. action_cost < 0.035
7. dynamic CR > 90%

## 介入方案優先序

### 方案 A：提高 ent_coeff（最優先，最對症）
- 目的：降低 entropy 鋸齒振幅，不讓谷底過深
- **倍率：×1.25 或 ×1.5（不要直接 ×2）**
  - 0.005 → 0.00625 或 0.0075
  - 0.01 → 0.0125 或 0.015
- 理由：現在 policy 還能學（KL 正常、VE 正常、entropy 會反彈），×2 太激進可能回到 high-entropy 學不動
- 適合：entropy 均衡點太低 + 探索不足 + policy 過早 deterministic

### 方案 B：partial advantage normalization（長期方案，留給 SA5）
- `A = (A - mean) / sqrt(std + eps)`
- 若 std≈12：放大倍率從 12x 壓到 ~3.5x
- module_entropy 預計從 ~1.0 降到 ~0.5

### 方案 C：advantage scale floor / capped normalization
- `A = (A - mean) / min(std, cap)` (cap=3)
- std=12 時只除以 3，保留 ~4x 放大

### 方案 D：降低 actor update clip（不優先）
- wd_actor_update_clip 8 → 4
- **不優先的理由：KL=0.007 正常、VE=0.729 正常、action_cost 恢復、entropy 會反彈 — 不是 update 爆掉的問題**

**Why:** 移除 advantage std normalization 後 actor gradient 放大 ~12x，entropy 鋸齒越來越深，需控制振幅。
**How to apply:** 每次監控報告檢查三維度 + 7 條觸發條件。第三次谷底若觸及門檻（SR<75% 或 CR>25%），立即用方案 A（×1.25 先試）。
