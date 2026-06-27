---
name: project-v3-launch
description: v3 從 SA1_v3 從頭訓練，整合 r_min=0.25 (VLP-16 實測) + penalty_smoothness=0.005 (anti-jitter) + 新 curriculum + per-env p95 監控
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

v3 是 2026-06-08 啟動的新訓練鏈，與 v2 不相容。從 SA1_v3 從頭訓練，不能 resume v1/v2 ckpt。

**Why:** v2 末段（SA3_v2 完訓後）部署實機發現 policy 直走中會突然抽動（左/右轉幅度不小）。深入分析：
1. 1024 envs 平均的 `omega_std=0.022` 看似平滑，但 per-env p95 可能很高（不同 env 抽動相位不同 → 平均互相抵消）
2. entropy_angular=1.5（log19=2.94 的 51%）— policy 分布太寬，argmax 對 logits 微差敏感
3. r_min=0.9 高估 LiDAR 盲區（真實 VLP-16 表面到人物中心可看到 0.2m）

決定一次性修正三點，從頭重訓。

**How to apply:**

### 改動清單（已完成 2026-06-08）

| 變更 | 檔案 |
|------|------|
| LiDAR `r_min` 0.9→0.25 | `charge_env_cfg_vlp16.py:414,476` (×2) |
| `r_min` 註解更新 | `obs_functions.py:176` |
| Reward `penalty_smoothness` 參數 | `rnn_car_wdclean/rewards.py:compute_wd_charge_reward` (signature + breakdown) |
| Module facade 加 smoothness | `rnn_car_modular/rewards/wd_sparse.py` (init/update_params/compute) |
| 主訓練 prev_actions buffer + 同步 | `train_rnn_car_wdclip.py` (~L2870, L3032, L3127) |
| Curriculum v3 | `curriculum/phases/wd_single_agent_v3.py` (copy v1 + penalty_smoothness=0.005 全 stage) |
| Phase registry | `curriculum/phases/__init__.py` 加 `warp_drive_single_agent_v3` |
| Experiment yaml | `configs/wd_sa1_v3.yaml` (curriculum_version=v3, no checkpoint) |
| Per-env p95 監控 | `train_rnn_car_wdclip.py` MetricsCollector (jitter_n_sample=16) |
| WandB metric | `jitter/per_env_omega_std_p95`, `jitter/per_env_ratio_flip_rate_p95` |
| CLAUDE.md | 加 v3 改動段落 |

### r_min=0.25 由來

- VLP-16 物理直徑 ∅103mm → 半徑 0.0515m
- 用戶實測：LiDAR 表面到人物中心最近 0.2m
- sim r_min 是 LiDAR optical center 起算 → 0.2 + 0.0515 ≈ 0.25m
- ⚠️ 不要寫成 0.20，會讓 sim 比實機多看到一圈 0.05m，部署反應變晚

### penalty_smoothness=0.005 量級

- |Δratio_ang| ∈ [0, 2]，每幀最壞 -0.01
- 一 episode (300 步) 最壞 -3.0 ≈ 20% collision penalty
- 實際平均 ~-0.3 ~ -0.5 / episode，僅作 inductive bias 不破壞主目標

### 啟動指令

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa1_v3 --headless \
  --run_name sa1_v3_ne1024_s42
```

### 監控判讀

| 指標 | GREEN | YELLOW | RED |
|------|-------|--------|-----|
| `jitter/per_env_omega_std_p95` | < 0.05 | 0.05-0.15 | > 0.15 |
| `jitter/per_env_ratio_flip_rate_p95` | < 0.05 | 0.05-0.10 | > 0.10 |
| `phase_parameter/penalty_smoothness` | = 0.005 (常數) | - | 異常變 0 |

**Related:** [[feedback_per_env_sampling]], [[project_lidar_rmin_change]], [[feedback_obs_index_layout]]
