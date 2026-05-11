# 5090 Repro Run from 5070 Branch

## Repo
- Path: `/home/aa/IsaacLab`
- Branch: `repro/5070-v4b-goalfirstv2b-cadn`
- Commit: `7a9be094f253b2f9ff49629ba6120eb92840cc67`
- Git status: Clean (0 tracked changes)
- Note: `cadn_preprocessor.py` symlink → `cadn.py` (repro branch 的 CADN 模組名稱不同)

## Command
```bash
PYTHONUNBUFFERED=1 /home/aa/miniconda3/envs/env_isaaclab/bin/python \
  /home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v4_b \
  --curriculum_version goal_first_v2_b \
  --dynamic_safety_mode closing_risk \
  --use_cadn \
  --run_name rw_groundv4_goalfirstv2_wall3_gap_reward_b_5090_repro \
  --seed 1 --num_envs 2048 --headless
```

## Execution
- PID: 2484327
- Start: 2026-03-25 19:43 (local)
- GPU: RTX 5090, 6.2GB VRAM
- Speed: ~3.1 it/s
- Status: **Running**

## Paths
- Log: `/home/aa/IsaacLab/logs/repro_5090.log`
- Run dir: `/home/aa/IsaacLab/logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC/rw_groundv4_goalfirstv2_wall3_gap_reward_b_5090_repro/`
- WandB: project `charge_skrl`, run `rw_groundv4_goalfirstv2_wall3_gap_reward_b_5090_repro`

## Issues Encountered
1. `isaaclab.sh` 找不到 Python → 改用直接 `python` 路徑
2. `ModuleNotFoundError: cadn_preprocessor` → 建 symlink `cadn_preprocessor.py -> cadn.py`
3. v6 (PID 2417844, 13.6GB) 同時在跑，共用 GPU

## Concurrent Processes
- v6 open_ended (PID 2417844): 13.6GB — abl branch
- repro (PID 2484327): 6.2GB — repro branch
- Total GPU: ~20GB / 32.6GB

## Environment Version Risk
| Item | 5090 | 5070 (expected) | Match? |
|------|------|-----------------|:---:|
| PyTorch | 2.7.0+cu128 | 未確認 | ⚠️ |
| CUDA | 13.0 | 未確認 | ⚠️ |
| Isaac Sim | 5.1.0.0 | 未確認 | ⚠️ |
| SKRL | 1.4.3 | 未確認 | ⚠️ |
| GPU | RTX 5090 | RTX 5070 Ti | ❌ 不同 |
| num_envs | 2048 | 2048 | ✅ |
| seed | 1 | 1 | ✅ |
| branch/commit | 7a9be094 | 7a9be094 | ✅ |
