---
name: finding_p95flip_entropy_confound
description: 訓練 p95_flip 被高 entropy 探索污染，抽動驗收必須用 deterministic rollout，不能單看訓練指標
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

訓練時的 `jitter/per_env_ratio_flip_rate_p95` 是用**會抽樣**的 policy 算的，會被 entropy 高低污染 → 不能單獨當 sin 波/抽動指標。

**Why**：SA1 故意用加倍 entropy coeff（0.04/0.02）撐高探索，ent 釘在 ~3.2，動作分布寬，**抽樣本身就讓角度 ratio 符號亂跳**。SA1_v3d 訓練 p95_flip 連續 7 輪 40-51% RED，但同期主任務 SR 82-86% 健康（矛盾）。關鍵反證：ent 升↔flip 升同步（3.16→3.26 時 flip 47→51%），證明兩者是同一回事（探索），不是 policy 學壞。實測 iter-100 ckpt **deterministic（argmax）rollout p95_flip = 0.000**，部署完全不抖 → 訓練 RED 是 100% entropy 假象。

**How to apply**：
1. 監控若 p95_flip RED 但 SR/CR 健康 → 先懷疑 entropy confound，**別急著 kill**。
2. 抽動/sin 波驗收一律用 deterministic rollout（argmax），這才是部署實際行為。
3. 工具：`play_rnn_car.py --jitter_eval --deterministic`（2026-06-12 新增），算 per-env 角度 ratio 符號翻轉率 + |ω| std，done 歸零。範本在 `/tmp/jitter_eval_sa1v3d.sh`。
4. deterministic 判讀基準：p95_flip < 0.15 視覺乾淨 / 0.15-0.30 輕微擺動 / >0.30 仍 sin 波。
5. 記得 `CHARGE_ACT_HIST_MODE=action_error` 要與訓練一致。

相關：[[project_v3d_launch]] [[feedback_per_env_sampling]] [[finding_cmd_delay_limit_cycle]]
