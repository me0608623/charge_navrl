---
name: feedback_monitor_wandb_full
description: 訓練監控必須每次都從 WandB 拉取所有分組指標，不能只看 console log
type: feedback
---

訓練監控每 15 分鐘必須從 WandB API 拉取**所有**分組指標，不能只看 console log 或精簡子集。

**Why:** 之前多次監控為了簡潔而省略部分指標組，導致漏看重要趨勢。用戶明確要求：每次監控都要拉完整指標。

**How to apply:** 每次 cron 監控都要用 `wandb.Api().run().summary` + `.history()` 拉取以下**全部**分組，不得省略任何一組：

1. `rl/*` — return_mean, success_rate, collision_rate, timeout_rate, value_loss, total_loss, entropy_linear, entropy_angular
2. `rl_critic/*` — variance_explained, value_pred_mean, value_pred_std, value_target_mean, value_target_std
3. `rl_adv/*` — mean, std, raw_std
4. `aux/*` — loss_per_step, preprocess_loss, rnn_grad, rnn_delta, ph_delta, fm_delta, valid_samples
5. `wd_update_actor/*` — grad_norm, clip_fraction, param_delta, update_ratio
6. `wd_update_critic/*` — grad_norm, clip_fraction, param_delta, update_ratio
7. `wd_update/*` — module_entropy, merged_grad_norm
8. `expect_value/*` — V_total, p_goal, p_collision, ep_lengths
9. `reward/term/*` — 所有 env reward terms（weight=0 的被動診斷指標也要列）

每組都要在報告中列出，即使值是 `---`（未上報）也要標示。趨勢分析用 Q1 vs Q4 quartile 比較。
