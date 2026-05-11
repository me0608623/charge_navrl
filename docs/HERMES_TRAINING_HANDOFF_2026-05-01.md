---
title: Hermes Training Handoff 2026-05-01
date: 2026-05-01
project: IsaacLab RNN Car / WD-style Single-Agent Navigation
status: handoff
---

# Hermes Training Handoff 2026-05-01

This document is the portable handoff for continuing the IsaacLab RNN car training/debugging work with Hermes agent or another coding agent.

## 1. Where This Session's Records Live

The full Codex chat transcript is managed by the Codex/OpenAI session backend and is not stored as a normal project file in `/home/aa/IsaacLab`.

Local transferable records are:

| Type | Location | Notes |
|---|---|---|
| Codex compressed memory context | `/home/aa/.codex/AGENTS.md` | Contains compact observation IDs and summaries, not the full raw transcript. Latest snapshot includes IDs `750-806` for Apr 30 training/debug work. |
| Hermes agent guide | `/home/aa/.hermes/hermes-agent/AGENTS.md` | Describes Hermes session DB/tool architecture, not this IsaacLab training session. |
| Training logs | `/home/aa/IsaacLab/logs/` | Console logs, run metadata, reports. |
| RNN car run logs | `/home/aa/IsaacLab/logs/rnn_car/` | Most important current training logs and metadata. |
| Structured experiment ledger | `/home/aa/IsaacLab/docs/RNN_CAR_EXPERIMENTS_LEDGER.md` | Fixed-schema experiment records for Hermes / Claude / Codex synchronization. |
| Obsidian notes | `/home/aa/Documents/Obsidian Vault/消融前/` and `/home/aa/Documents/Obsidian Vault/isaaclab/train rnn car/` | Human-readable training notes, experiment conclusions, RL concepts. |
| WandB metrics | Project `charge_skrl` under user/entity used by existing scripts | Needs WandB API or browser access. Run IDs below. |
| Code state | `/home/aa/IsaacLab` git worktree | Contains modified training scripts and diagnostics. |

Important limitation: if Hermes needs the literal full chat transcript, export it from the Codex UI if available. Otherwise this handoff plus logs/notes is the practical transfer artifact.

## 2. Environment Rules

Use this Python, not a generic system Python:

```bash
/home/aa/miniconda3/envs/env_isaaclab/bin/python
```

The machine normally needs `env_isaaclab` active; the most stable launch path has been the direct interpreter above.

Do not assume `./isaaclab.sh` is the canonical way for this user's current setup.

Training and play commands should generally be launched from:

```bash
cd /home/aa/IsaacLab
```

## 3. Current Git / Code State

Recent `git status --short` showed:

```text
 M .claude/scheduled_tasks.lock
 M .director-mode/changelog.jsonl
 M CLAUDE.md
 M scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py
?? .director-mode/changelog.20260420_144341.jsonl
?? cleanup_math.py
?? convert_math.py
?? convert_math_v2.py
?? convert_math_v3.py
?? convert_math_v4.py
?? fix_latex.py
?? reconstruct_math.py
?? scripts/pedestrian/
?? scripts/reinforcement_learning/skrl/train_marl_rnn.py
?? scripts/reinforcement_learning/skrl/train_rnn_car_ppo_mb.py
?? scripts/reinforcement_learning/skrl/virtual_spot_utils.py
?? source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/direct_marl/
?? write_fixed.py
```

Do not blindly reset or checkout these files. Some are user/Claude-generated work.

Most relevant modified file:

```text
scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py
```

Recent additions in this file:

- `--normalize_return`
- `--value_init_bias`
- `--wd_update_clip`
- `--aux_grad_clip`
- `--no_resume_optimizer`
- `--action_table_sample_size`
- optimizer-state resume
- goal reset diagnostics
- action speed/accel WandB table
- RNN feature distribution diagnostics
- value target/prediction diagnostics

## 3.5 Original WD Project vs Current IsaacLab Code Map

Hermes must distinguish three codebases:

| Role | Path | What to use it for |
|---|---|---|
| Primary WD reference for this project | `/home/aa/Documents/train rnn car origin/train_rnn_car.py` | This is the compact local "origin" snapshot that should be treated as the main WD `train_rnn_car` reference for phase budget, reward/config style, RNN module setup, and trainer behavior. |
| Primary WD algorithms reference | `/home/aa/Documents/train rnn car origin/warp_drive/training/algorithms/a2c_ken.py` and `/home/aa/Documents/train rnn car origin/warp_drive/training/algorithms/a2c_entropy.py` | Use these to compare WD A2C loss clamp, entropy/loss balance, and update-limit behavior. |
| Primary WD env/model reference | `/home/aa/Documents/train rnn car origin/custom_envs/` | Use for WD environment reward, phase parameters, module-connected network, and custom trainer details. |
| Expanded WD repo copy | `/home/aa/IsaacLab/new_warp_drive/` | Larger repo with history-like notes and many WD scripts. Good for cross-checking if origin snapshot is unclear. |
| Alternate expanded WD repo copy | `/home/aa/new_warp_drive/` | Another copy; use only as secondary reference if needed. |
| Current IsaacLab main experiment trainer | `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py` | Current active experimental trainer with WD-style clipping, normalize-return, diagnostics, optimizer resume, action scatter, and RNN feature monitoring. This is the file used in Plan D / Plan D resume. |
| Current IsaacLab baseline trainer | `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train_rnn_car.py` | Earlier/mainline RNN car trainer. Some Obsidian notes still reference this. Use for baseline comparison, but current debug work has moved to `train_rnn_car_wdclip.py`. |
| PPO mini-batch experiment | `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train_rnn_car_ppo_mb.py` | Experimental PPO/mini-batch branch. Do not treat as current mainline unless explicitly testing PPO. |
| Play / BEV diagnostic | `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/play_rnn_car.py` | Playback, BEV, LiDAR diagnostic, checkpoint visual behavior validation. |
| WD aux target port | `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/wd_aux_targets.py` | IsaacLab implementation of WD-style 7D privileged geometry aux targets. Recently changed toward predicted surface geometry. |

Important reading order for WD comparison:

1. `/home/aa/Documents/train rnn car origin/train_rnn_car.py`
2. `/home/aa/Documents/train rnn car origin/custom_envs/custom_trainer.py`
3. `/home/aa/Documents/train rnn car origin/custom_envs/module_connected.py`
4. `/home/aa/Documents/train rnn car origin/warp_drive/training/algorithms/a2c_ken.py`
5. `/home/aa/Documents/train rnn car origin/warp_drive/training/algorithms/a2c_entropy.py`
6. `/home/aa/IsaacLab/scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py`

Do not confuse `/home/aa/Documents/train rnn car origin/train_rnn_car.py` with `/home/aa/Documents/train rnn car/scripts/reinforcement_learning/skrl/train_rnn_car.py`. The first one is the WD origin reference; the second appears to be an intermediate copied IsaacLab-style workspace.

## 4. Current Training Lineage

### Key Runs

| Run | WandB | Summary |
|---|---|---|
| `wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429` | `kpjfl0xt` | Plan D 300-update run. Critic fixed: `VE≈0.52`, but actor stalled: SR around `57.6%`, heading error around `90°`. |
| `wd_sa_v1_rawfc_freeze_env512_p1_planD_resume_to_u900_diag_0430` | `y62stzbf` | Resume from Plan D to cumulative 900 updates. Aborted at resume update 75/600 due RNN feature drift explosion and critic collapse. |

### Important Local Logs / Reports

```text
/home/aa/IsaacLab/logs/rnn_car/planD_resume_to_u900_diag_0430.log
/home/aa/Documents/Obsidian Vault/消融前/plan_d_300_report.md
/home/aa/Documents/Obsidian Vault/消融前/plan_d_resume_900_report.md
/home/aa/IsaacLab/logs/rnn_car/plan_b_c_report.md
```

### Last Aborted Run

Run:

```text
wd_sa_v1_rawfc_freeze_env512_p1_planD_resume_to_u900_diag_0430
```

WandB:

```text
y62stzbf
```

Checkpoint source:

```text
logs/rnn_car/wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429/checkpoint_90000.pt
```

Command family:

```bash
PYTHONUNBUFFERED=1 /home/aa/miniconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --curriculum_version warp_drive_single_agent_v1 \
  --initial_stage 1 --fixed_stage \
  --charge_encoder_mode raw_fc_rnn \
  --aux_mode tbptt --aux_seq_len 15 --aux_seq_batch_size 256 \
  --use_a2c --ppo_epochs 1 \
  --num_envs 512 \
  --rollout_length 300 --timesteps 180000 \
  --lr 0.0002 --rnn_lr 0.0005 \
  --normalize_return --value_init_bias 0.0 --vf_coeff 0.5 \
  --wd_update_clip --aux_grad_clip 0.5 \
  --aux_lr_predict_head 0 --aux_lr_fc_middle 0 --aux_lr_fc_front 0 --aux_lr_extractor 0 \
  --checkpoint logs/rnn_car/wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429/checkpoint_90000.pt \
  --headless --seed 1 \
  --log_interval 5 --save_interval 100 \
  --action_table_sample_size 2048 \
  --run_name wd_sa_v1_rawfc_freeze_env512_p1_planD_resume_to_u900_diag_0430
```

The run loaded optimizer state successfully after patches:

```text
Loaded optimizer state: charge_opt_rl
Loaded optimizer state: charge_opt_aux
Loaded optimizer state: obs_optimizer
```

## 5. Confirmed Technical Conclusions

### Observation / Goal State

Policy observation uses:

```text
POLICY_OBS_INDICES = [0..77] + [138]
```

This includes:

```text
linear_acceleration 1D
linear_velocity     1D
angular_velocity    1D
radius              1D
goal_position       2D
lidar_static        72D
time_remaining      1D
```

The goal observation is robot-body-frame `(x, y)`, not world-frame. Goal distance is not explicitly given, but is implicit:

```text
goal_distance = sqrt(goal_x^2 + goal_y^2)
goal_angle = atan2(goal_y, goal_x)
```

`heading_error` is diagnostic only. It is computed from world goal direction minus robot yaw. It should not be used alone to judge success because obstacle avoidance may require temporary heading offsets.

Better interpretation:

| heading_error | velocity_to_goal | Interpretation |
|---|---|---|
| low | high | ideal, moving toward goal |
| high | high | may be reasonable obstacle-avoidance / side movement |
| high | low | likely not goal-directed |
| low | high collision | too greedy / unsafe |

### LiDAR Fix

A previous ray origin bug was fixed. The correct per-ray origin is `_ray_starts_w`, not `sensor.data.pos_w`.

Validated with play diagnostic:

- ray origin z around `1.6m`
- ground rays rejected by z-filter
- no false `~5.6m` ground echo domination
- walls and pedestrian-height obstacles visible

### Critic Fix

The critic problem was fixed by the Plan D stack:

```text
--normalize_return
--value_init_bias 0.0
--vf_coeff 0.5
--wd_update_clip
--aux_grad_clip 0.5
```

Why:

- `--normalize_return` puts value target at roughly mean 0 / std 1.
- `--value_init_bias 0.0` fixes the previous value head bias `-8.0`, which was incompatible with normalized returns.
- `vf_coeff=0.025` was too weak under normalized value targets.
- `vf_coeff=0.5` made critic learning real: Plan D reached `VE≈0.52`.

### Actor Still Not Learning Goal-Directed Behavior

Even when critic is healthy:

```text
heading_error ≈ 90°
velocity_to_goal ≈ 0.088
entropy ≈ 5.876, near max for 19x19 action space
SR around 56-58%
CR around 40-43%
```

This means actor is updating but not forming a strong goal-directed action bias.

Action mean diagnostics showed `speed_x`, `accel_x`, `omega` near 0. This can mean either:

- policy barely moves, or
- positive/negative actions cancel in the mean.

Use the W&B action scatter table instead of means:

```text
charge/action_speed_accel_table
x = v_x
y = a_x
color = velocity_to_goal or terminal_code
```

### RNN Feature Drift Is Critical

Plan D resume catastrophically diverged because RNN features exploded:

```text
rnn_feature_mean: 0.16 -> 5718
rnn_feature_delta: 0.01 -> 25168
VE: 0.55 -> -1.0
critic_clip_fraction: 0 -> 1.0
```

`rnn_grad_norm` did not exceed the old abort threshold, so it was not sufficient. The true guard should be `rnn_feature_mean/std/delta`.

Do not continue long training with:

```text
rnn_lr=5e-4
aux_lr_predict_head=0
aux_lr_fc_middle=0
aux_lr_fc_front=0
aux_lr_extractor=0
```

This updates only RNN while all downstream mapping is frozen, allowing the latent representation to drift away from what actor/critic expect.

## 6. Current Best Diagnosis

The current stack has separated failures:

1. Sensor/observation pipeline is largely correct after LiDAR ray-origin fix.
2. Critic learning is now correct with normalized returns, value bias 0, and `vf_coeff=0.5`.
3. Actor is not learning strong goal-directed behavior.
4. Continuing aux/RNN training with frozen downstream heads causes feature drift and catastrophic critic collapse.

Therefore the next experiment should not be "just run longer with the same aux settings."

## 7. Recommended Next Experiments

### Experiment A: Pure RL Resume, RNN Frozen

Purpose: determine whether actor can improve when 12D RNN feature stops drifting.

Use Plan D 300-update checkpoint as source. Disable aux/RNN updates. If no explicit CLI exists to disable aux, add one cleanly:

```text
--disable_aux_training
```

Expected interpretation:

- If SR/velocity_to_goal improves: RNN drift was blocking actor.
- If actor remains stuck: problem is reward/action/entropy/curriculum, not RNN drift.

### Experiment B: Lower Entropy / Entropy Decay

Reason: entropy remains near max (`~5.876` vs `ln(361)=5.889`), so actor may be staying too random.

Test lowering per-head entropy coefficients or implementing decay. Do not change reward and entropy at the same time.

### Experiment C: Goal-Action Correlation Diagnostics

Add or analyze:

```text
corr(goal_y, omega)
corr(goal_x, speed_x)
corr(heading_error_signed, omega)
velocity_to_goal histogram by action
advantage mean by action bin
```

Need to prove whether actor actions respond to body-frame goal vector.

### Experiment D: Reward Shaping Minimal Test

Only after A-C:

```text
velocity_to_goal
safe_progress
small action smoothness / anti-zigzag term
```

But the current priority is diagnosis, not immediately adding reward.

## 8. Important Files to Read First

Hermes should start by reading:

```text
HERMES_TRAINING_HANDOFF_2026-05-01.md
/home/aa/IsaacLab/docs/RNN_CAR_EXPERIMENTS_LEDGER.md
/home/aa/Documents/Obsidian Vault/消融前/plan_d_300_report.md
/home/aa/Documents/Obsidian Vault/消融前/plan_d_resume_900_report.md
/home/aa/Documents/Obsidian Vault/消融前/03_WandB_指標手冊_train_rnn_car_wdclip.md
/home/aa/Documents/Obsidian Vault/消融前/31_Experiment_normalize_return兩輪報告.md
/home/aa/IsaacLab/logs/rnn_car/plan_b_c_report.md
/home/aa/IsaacLab/logs/rnn_car/planD_resume_to_u900_diag_0430.log
scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py
scripts/reinforcement_learning/skrl/wd_aux_targets.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/observations/functions.py
```

## 9. Prompt To Give Hermes

Copy this prompt into Hermes:

```text
You are continuing IsaacLab RNN car WD-style single-agent navigation training work.

Start in /home/aa/IsaacLab.

First read:
- HERMES_TRAINING_HANDOFF_2026-05-01.md
- docs/RNN_CAR_EXPERIMENTS_LEDGER.md
- /home/aa/Documents/Obsidian Vault/消融前/plan_d_300_report.md
- /home/aa/Documents/Obsidian Vault/消融前/plan_d_resume_900_report.md
- /home/aa/Documents/Obsidian Vault/消融前/03_WandB_指標手冊_train_rnn_car_wdclip.md
- /home/aa/IsaacLab/logs/rnn_car/planD_resume_to_u900_diag_0430.log
- scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py

Current conclusion:
- LiDAR and goal observation are correct.
- Critic is fixed by normalize_return + value_init_bias=0 + vf_coeff=0.5.
- Actor still does not learn goal-directed behavior.
- Continuing aux/RNN training with frozen heads causes RNN feature drift and critic collapse.

Do not resume the failed y62stzbf run.
Do not run rnn_lr=5e-4 with frozen aux heads for long training.

Next priority:
1. Add or verify a clean way to freeze/disable aux/RNN updates during resume.
2. Resume from Plan D 300-update checkpoint with pure RL only.
3. Monitor SR, CR, VE, entropy, velocity_to_goal, heading_error, action scatter, actor_grad/delta, and RNN feature mean/std/delta.
4. If actor still stalls, analyze entropy/action-space/reward shaping. Do not change multiple variables at once.

Use:
/home/aa/miniconda3/envs/env_isaaclab/bin/python

Never use destructive git commands. The worktree is dirty and contains user/Claude-generated work.
```

## 10. Immediate Safety Rules

- Do not use collapsed checkpoints from `y62stzbf`.
- Prefer the Plan D source checkpoint:

```text
logs/rnn_car/wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429/checkpoint_90000.pt
```

- If resuming with optimizer state and changing optimizer hyperparameters, think carefully. Optimizer state may override or interact with new LR/groups.
- If only model weights should be loaded, use:

```text
--no_resume_optimizer
```

- Add abort guards based on:

```text
aux/rnn_feature_mean
aux/rnn_feature_std
aux/rnn_feature_delta_norm
rl/variance_explained
wd_update/critic_clip_fraction
entropy
SR/CR
```

## 11. Current Open Questions

1. Does actor learn if RNN features are frozen after Plan D?
2. Is entropy coefficient too high for 361 discrete actions?
3. Does action space need reduction or action masking?
4. Is sparse WD reward enough in the single-agent goal density setup, or is minimal `velocity_to_goal` shaping required?
5. Are successful episodes caused by real goal-directed motion or by short-distance/many-goal chance contacts?
6. Should aux target train end-to-end, or should aux be pretraining-only and then frozen during RL?
