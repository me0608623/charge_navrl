---
name: SA2 cont2 monitoring protocol
description: 15-min monitoring checklist for wd_sa2_cont2_wdclip_rb — 5 categories with GREEN/YELLOW/RED thresholds
type: feedback
---

15-minute monitoring protocol for SA2 cont2 (wdclip + rule_based).

**Why:** Actor collapse is the core risk; WD gradient clipping is the new intervention being tested. Need to catch SR/CR/gΔ/entropy degradation early (usually iter 30-300).

**How to apply:** Every 15 min, check 5 categories. Report format below. Watch TRENDS not single points.

## Regime sanity (every check)
- obstacle_mode = rule_based, no OBS summary
- Stage 2 / SA2_goal_wall_2dyn, fixed_stage
- wd_update_clip enabled, actor cap=8, critic cap=30
- ent_coeff_linear=0.05, angular=0.10

## Actor health (priority 1)
GREEN: SR>80%, CR<25%, gΔ>0.8, gV>0.20, h<65°, entropy slow decline
YELLOW: SR 75-80%, CR 25-30%, gΔ 0.5-0.8, entropy dropping fast, h 65-75°
RED/STOP: SR<75% ×2, CR>30% ×2, gΔ<0.5, entropy<4.3 + SR/CR worsening, h>75° + gV/gΔ↓

## Critic health
GREEN: VE>0.5, vf_loss stable/declining
YELLOW: VE 0.3-0.5, vf_loss rising
RED: VE<0.3, VE<0 or near -1, vf_loss exploding

## RNN/Aux stability
GREEN: rnn_delta<0.1, n1d=0, n2d=0
RED: n1d/n2d>0, rnn_delta>1.0, rnn_feature_mean>10

## WD clip / rl_trust (new core)
- actor grad_norm>8 but post_clip≈8 → clipping working, normal
- actor clip_fraction high + SR/CR worsening → cap too loose, consider cap=4 or lr=1e-4
- actor clip_fraction≈0 but still collapse → not gradient issue, check reward/entropy/action dist
- rl_trust policy shift large → lower lr or actor cap
- rl_trust small but SR dropping → rule_based regime mismatch or checkpoint transfer issue

## Immediate STOP triggers
- n1d/n2d > 0
- rnn_feature_mean > 10
- VE < 0
- SR < 70%
- CR > 35%
- entropy < 4.0 + SR/CR worsening
- gΔ <= 0
