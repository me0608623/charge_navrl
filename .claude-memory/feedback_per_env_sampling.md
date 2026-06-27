---
name: feedback-per-env-sampling
description: 評估 policy 行為一致性必須隨機抽單 env 觀察，不能只看全 env 平均（平均會掩蓋抽動）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

評估 policy 行為（特別是抽動、震盪、unstable 等單幀現象）時必須**隨機抽 8~16 個 env 個別記錄**，不能只看全 envs 平均後的 std。

**Why:** 2026-06-08 用戶指出：我用 `omega_std_over_steps=0.022` 當「policy 平滑」的證據是錯的。這個 metric 是 1024 envs × steps 的整體 std，即使每個 env 都偶爾抽動，**抽動的 phase 不同步 → 平均後彼此抵消**，看起來很小。實際部署 SA_v1 機器人就會直走中突然抽動。個別 env 可能有 ω_std=0.2+ rad/s 但全 env 平均到 0.022。

**How to apply:**

1. **訓練監控**：除了既有 `ablation/omega_std` 之外，新增 per-env p50/p95：
   ```python
   sample_ids = torch.randperm(num_envs)[:16]
   for env_id in sample_ids:
       traj_std = applied_omega[:, env_id].std()
       traj_flip = ((ratio[1:, env_id] - ratio[:-1, env_id]).abs() > 0.5).float().mean()
   metrics["jitter/per_env_omega_std_p50"] = ...
   metrics["jitter/per_env_omega_std_p95"] = ...   # ⭐ 抽動真相
   metrics["jitter/per_env_ratio_flip_rate_p95"] = ...
   ```

2. **判斷 policy 健康**：
   - 全 env 平均 std 小 ≠ policy 平滑
   - **p95 > 0.15 rad/s 就要警戒**抽動嚴重
   - p50 仍會偏向平均，是不夠的

3. **部署前驗證**：argmax determinstic action 模擬一遍 → 看 deterministic_ω_std_p95，這才是真實部署行為。

4. **適用範圍**: 任何「一致性」、「穩定性」、「抽動」、「flip」相關現象 — 都要 per-env 抽樣，不只 ω。也適用 v_x、heading、reward variance 等。

**Related:** [[finding_policy_oscillation_stuck]] — SA4 ckpt 在特定 obs 配置下震盪，正是 per-env 才看得到的現象
