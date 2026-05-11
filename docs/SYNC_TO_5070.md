# Sync To 5070 — v21

## Runtime Repo Path
/home/aa/IsaacLab

## Branch Name
exp/v21-reward-speed-v05

## Commit Hash
6d69226a299ee5a6631d1e7b27934a5b10c8ffdd

## Included Files
- `scripts/reinforcement_learning/skrl/train_charge_ac.py` — 新增 `--reward_speed_v05` flag 與 handler，monkey-patch `_open_ended_reward_weights`

## Excluded Files
- `.director-mode/changelog.jsonl` — Director mode 本地 state
- `SYNC_TO_5070.md` — 此文件本身（會在 push 時重新生成）
- `.backup/` — 本地備份目錄 (v15_gate)

## v21 實驗結果摘要

使用 `best_agent.pt` (agent_58590, Level 5 checkpoint) 評估：

| 環境 | SR | CR | TO | speed |
|------|-----|-----|-----|-------|
| Stage 8 (8s+3d) | **89.7%** | 10.3% | 0% | **0.74 m/s** |
| Level 5 (19s+7d) | **78.0%** | 21.9% | 0.1% | **0.72 m/s** |

對比 v20 baseline:
- speed: 0.06 → 0.74 (**12x 提升**)
- Stage 8 SR: 79% → 89.7% (+10.7pp)
- 升 6 個 Open Ended levels (Level 1→6)
- KL converged to 0.98 (peak 2.39)

## Exact Training Command
```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v8 --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk --no_walls \
  --no_domain_randomization --lidar_no_noise --reward_speed_v05 \
  --run_name rw_groundv8_openendedv1__seed1_nowalls_true_control_speed05_v21_5070 \
  --seed 1 --num_envs 6144 --headless --directional_gate
```

## Notes for 5070
- 必須帶 `--reward_speed_v05` flag 才能套用 reward 修改（goal_velocity ×3, static_safety ×0.375, time_penalty -0.6）
- 必須帶 `--no_domain_randomization --lidar_no_noise` 維持對照組「無噪音」性質
- USD 已在 repo 內 (`assets/usd/charge/charge.usd`)
- 預計 6h 內可達 SR ~85%, speed ~0.74 m/s
- 預計 10h 內升到 Open Ended Level 5+
- 預計總訓練時間 ~24h 達 234375 steps

## Push Result
- Remote: charge_skrl (git@github.com:me0608623/charge_skrl.git)
- Branch: exp/v21-reward-speed-v05
- Status: success
- HEAD: 6d69226a299ee5a6631d1e7b27934a5b10c8ffdd
