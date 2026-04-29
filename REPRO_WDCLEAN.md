# WDClean Repro Notes

This note is for reproducing the current Charge WDClean / WDClip training setup on another machine.

## Key Rule

Do not copy individual files by hand if the target machine already has the same IsaacLab repo layout.
Use Git to move the whole code state:

```bash
cd /home/aa/IsaacLab
git fetch
git checkout <branch-or-commit>
git pull
```

The training depends on files across `scripts/`, `source/isaaclab_tasks/`, `source/isaaclab/`, and `assets/`.
Manual copying is likely to miss an environment, task, observation, termination, or asset dependency.

## Required Repo Paths

The following areas are part of the reproducible training state:

```text
scripts/reinforcement_learning/skrl/rnn_car_wdclean/
scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py
scripts/reinforcement_learning/skrl/modular_rnn_models.py
scripts/reinforcement_learning/skrl/wd_aux_targets.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/
source/isaaclab/isaaclab/scene/interactive_scene.py
assets/usd/charge/charge.usd
```

`charge_skrl/direct_marl/` is optional for WDClean / WDClip ManagerBased training. The task registry skips it when it is not present.

## Commit Checklist

Before pushing, review the worktree explicitly. Do not use a blind `git add .` unless the status has been checked.

```bash
git status --short
git diff --name-only
```

Recommended WDClean reproducibility commit contents. Use explicit paths; do not stage the whole `charge_skrl/` directory if `direct_marl/` is not part of this commit.

```bash
git add REPRO_WDCLEAN.md
git add scripts/reinforcement_learning/skrl/rnn_car_wdclean/
git add scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py
git add scripts/reinforcement_learning/skrl/wd_aux_targets.py
git add source/isaaclab/isaaclab/scene/interactive_scene.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/__init__.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/__init__.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_env_cfg_vlp16.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_env_cfg_vlp16_curriculum.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/goal_obstacle_curriculum.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/__init__.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/mixed_parallel.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/obstacles.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/walls.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/observations/__init__.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/observations/obs_functions.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/terminations/robot_state.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/wall_layout.py
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/multi_goal_command.py
git diff --cached --name-only
```

Already tracked and required:

```text
assets/usd/charge/charge.usd
scripts/reinforcement_learning/skrl/modular_rnn_models.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_cfg.py
```

Do not stage local run artifacts:

```text
logs/
wandb/
outputs/
*.pt
__pycache__/
.claude/scheduled_tasks.lock
.director-mode/
```

The repo `.gitignore` ignores log/wandb/output contents. USD files are generally ignored, but `assets/usd/charge/charge.usd` is already tracked and is intentionally part of this setup.

Optional files that are not required for WDClean:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/direct_marl/
scripts/reinforcement_learning/skrl/run_*.sh
scripts/reinforcement_learning/skrl/analyze_wandb_run.py
scripts/reinforcement_learning/skrl/train_marl_rnn.py
scripts/reinforcement_learning/skrl/train_rnn_car_ppo_mb.py
scripts/reinforcement_learning/skrl/virtual_spot_utils.py
scripts/pedestrian/
```

If DirectMARL is needed later, commit `direct_marl/` in a separate feature commit and test it independently. It is intentionally optional for this WDClean reproduction path.

## Current Dirty-Tree Caveat

The PID `3960465` reference run was launched from a dirty worktree. For rigorous future runs, record the commit hash and whether the tree was dirty at launch time:

```bash
git rev-parse HEAD
git status --short
```

If a formal paper or comparison run is started, prefer launching from a clean commit and putting the exact command in this file or the run metadata.

## USD Asset

The Charge robot USD is expected to resolve from the repo:

```text
assets/usd/charge/charge.usd
```

No `CHARGE_USD_PATH` is required when the repo asset exists. If a machine uses a custom USD, set:

```bash
export CHARGE_USD_PATH=/absolute/path/to/charge.usd
```

## Environment

Expected local environment:

```text
Python: 3.11
Conda env: env_isaaclab
Isaac Sim: 5.1
PyTorch: 2.7.0+cu128
CUDA compiled: 12.8
```

Recommended shell setup:

```bash
source /home/aa/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=charge_skrl
```

Run `wandb login` first if WandB logging is needed.

## Smoke Test

Run this after `git pull`:

```bash
cd /home/aa/IsaacLab

PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/train.py --help

PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/train.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 16 --headless --seed 1 --use_a2c \
  --timesteps 128 --rollout_length 32 \
  --log_interval 1 --save_interval 999999 \
  --run_name smoke_wdclean_s1
```

The smoke test should create the env, run rollout/update, compute WD reward, train the obstacle pseudo-agent, run TBPTT aux loss, and exit without import errors.

## Current Reference PID

Reference run captured from PID `3960465` on 2026-04-29:

```bash
python scripts/reinforcement_learning/skrl/train_rnn_car_wdclip.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 512 --headless --seed 1 --use_a2c \
  --charge_encoder_mode raw_fc_rnn \
  --curriculum_version warp_drive_single_agent_v1 \
  --rollout_length 300 --timesteps 90000 \
  --fixed_stage --initial_stage 1 \
  --max_grad_norm 0.5 --wd_update_clip --aux_grad_clip 0.5 \
  --lidar_no_noise --normalize_return \
  --value_init_bias 0.0 --vf_coeff 0.5 \
  --run_name wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429
```

Runtime facts:

```text
cwd: /home/aa/IsaacLab
python: /home/aa/miniconda3/envs/env_isaaclab/bin/python3.11
task: Isaac-Navigation-Charge-VLP16-Curriculum-NavRL
curriculum: warp_drive_single_agent_v1
stage: fixed Stage 1, SA1_goal_wall_dyn
encoder: raw_fc_rnn
num_envs: 512
rollout_length: 300
timesteps: 90000
normalize_return: true
value_init_bias: 0.0
vf_coeff: 0.5
wandb run id: kpjfl0xt
local log: logs/rnn_car/wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429/train.log
```

Equivalent WDClean entrypoint:

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/train.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 512 --headless --seed 1 --use_a2c \
  --charge_encoder_mode raw_fc_rnn \
  --curriculum_version warp_drive_single_agent_v1 \
  --rollout_length 300 --timesteps 90000 \
  --fixed_stage --initial_stage 1 \
  --max_grad_norm 0.5 --wd_update_clip --aux_grad_clip 0.5 \
  --lidar_no_noise --normalize_return \
  --value_init_bias 0.0 --vf_coeff 0.5 \
  --run_name wdclean_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429
```

## Play / Replay

The replay entrypoint is also included on this branch:

```text
scripts/reinforcement_learning/skrl/play_rnn_car.py
scripts/reinforcement_learning/skrl/charge_env_overrides.py
```

`play_rnn_car.py` depends on the same model and aux-target files used by training:

```text
scripts/reinforcement_learning/skrl/modular_rnn_models.py
scripts/reinforcement_learning/skrl/wd_aux_targets.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/
```

Basic replay command:

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 1 \
  --checkpoint logs/rnn_car/<run_name>/checkpoint_<steps>.pt \
  --stage 1 \
  --curriculum_version warp_drive_single_agent_v1 \
  --camera top \
  --deterministic
```

Useful debugging options:

```text
--aux_debug --aux_debug_interval 25
--bev_vis --bev_frame body
--bev_frame world
--use_vo_shield
--scripted_obstacles
```

If replaying the PID `3960465` family of checkpoints, use the same task and curriculum version:

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_rnn_car.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 1 \
  --checkpoint logs/rnn_car/wd_sa_v1_rawfc_freeze_env512_p1_normret_vbias0_vfc05_u300_0429/checkpoint_<steps>.pt \
  --stage 1 \
  --curriculum_version warp_drive_single_agent_v1 \
  --camera top \
  --deterministic \
  --aux_debug \
  --bev_vis
```

## Notes

Old-vs-new smoke comparison has matched logged reward/loss/aux/obstacle metrics for the tested short run. Do not claim long-run bit-exactness across machines; Isaac Sim / PhysX / GPU scheduling should be treated as statistically reproducible, not bit-exact.
