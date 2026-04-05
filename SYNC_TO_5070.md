# Sync To 5070

## Runtime Repo Path
```
/home/aa/IsaacLab
```

## Branch Name
```
charge_skrl/abl
```

## Commit Hash
```
d453150fa4
```

## Commit Message
```
feat: OE 障礙物擴展到 100 + 診斷指標 + console 速度顯示

Key changes:
- _open_ended_params(): total=min(100, 11+level*3), dyn_ratio 27%→40%
- 升級門檻固定 SR>80%, CR<15%, TO<20%
- difficulty_level 加入 WandB
- ablation_metrics: +5 指標 (collision_type, path_efficiency, local_density, dynamic_encounter, speed_avg)
- console_summary: OE-L{N} 顯示 + 門檻對比 + Agent 速度區塊
- goal_command: 高密度障礙自動縮小 goal safe distance
- play_charge_ac_curriculum: diagnostic/deterministic/cadn_online flags
```

## Included Files
| 檔案 | 改動摘要 |
|------|----------|
| `goal_obstacle_curriculum.py` | OE 障礙物公式擴展到 100 + 固定門檻 + difficulty_level WandB |
| `ablation_metrics.py` | 新增 5 診斷指標 (collision_type, path_efficiency, local_density, dynamic_encounter, speed_avg) |
| `console_summary.py` | OE-L{N} 顯示 + 門檻對比 + Agent 速度區塊 |
| `play_charge_ac_curriculum.py` | diagnostic/deterministic/cadn_online flags |
| `goal_command.py` | 高密度障礙自動縮小 goal safe distance |

## Excluded Files
| 模式 | 原因 |
|------|------|
| `.claude/` | Claude Code 設定 |
| `logs/`, `wandb/` | 訓練輸出 |
| `*.pt` | checkpoint |
| `__pycache__/` | Python 快取 |
| `TRAINING_SNAPSHOT_*.md` | 本機快照 |

## Exact Training Command (5070)

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v8 \
  --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk \
  --no_walls --use_cadn \
  --run_name rw_groundv8_openendedv1__seed1_nowalls_5070 \
  --seed 1 --num_envs 6144 --headless
```

## Notes for 5070

### 環境確認
- Python: 3.11 (env_isaaclab)
- Isaac Sim: 5.1
- CUDA: 13.0
- USD 已在 repo 內 (`assets/usd/charge/charge.usd`)
- `wandb login` 需在 5070 單獨設定

### 同步指令
```bash
cd /home/aa/IsaacLab
git fetch charge_skrl
git checkout charge_skrl/abl
```

## Push Result
- **Remote**: `charge_skrl`
- **Branch**: `abl`
- **Status**: ✅ 已同步 (HEAD = d453150fa4)
