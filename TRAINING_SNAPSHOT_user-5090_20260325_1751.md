# Training Snapshot Report

> **Host:** user-5090 | **Date:** 2026-03-25 17:51 | **Branch:** abl | **Commit:** d03c17c4bb

---

## 1. Machine Identity

| Item | Value | Source |
|------|-------|--------|
| Hostname | `user-5090` | `hostname` |
| OS | Ubuntu 24.04.3 LTS (Noble Numbat) | `/etc/os-release` |
| Kernel | `6.17.0-19-generic x86_64` | `uname -a` |
| CPU | Intel Core Ultra 7 265K | `lscpu` |
| Cores | 12 | `nproc` |
| RAM | 47 GiB | `free -h` |
| GPU | NVIDIA GeForce RTX 5090 (32607 MiB) | `nvidia-smi` |
| Driver | 580.126.09 | `nvidia-smi` |
| CUDA | 13.0 | `nvidia-smi` |

---

## 2. Repository Identity

| Item | Value |
|------|-------|
| Repo path | `/home/aa/IsaacLab` |
| Remote (primary) | `charge_skrl` → `git@github.com:me0608623/charge_skrl.git` |
| Remote (upstream) | `origin` → `git@github.com:isaac-sim/IsaacLab.git` |
| Remote (fork) | `myfork` → `git@github.com:me0608623/IsaacLab.git` |
| Branch | `abl` |
| HEAD commit | `d03c17c4bbc111c2403384b3e7d0633fb5b9ff21` |

---

## 3. Git Status

**Working tree:** Clean (no staged or unstaged changes)

**Stash:** 2 entries on `main` branch (not on `abl`)
- `stash@{0}`: WIP — Add NavRL-style reward functions
- `stash@{1}`: Temp stash before push

**Recent 10 commits (abl):**

| Hash | Message |
|------|---------|
| d03c17c4bb | fix: play 腳本 --no_curriculum 模式更新 max_obstacles |
| d180962978 | feat: 兩段式 Curriculum — Bootstrap (B1-B6) + Open-ended |
| 093b67fde8 | chore: 移除 charge/paper/ 大型 PDF |
| 82fe160abd | chore: 全代碼同步 |
| efe38d35b3 | fix: 重生安全距離 2m → 1m |
| 03a2ab2df6 | fix: agent/goal 重生位置離障礙物和牆壁至少 2m |
| 5c1eaf0a49 | feat: --no_walls CLI flag |
| 1bc853fdd7 | fix: DebugLogger accel_table guard + 啟動摘要 |
| e69ca51385 | docs: REWARD_DESIGN_V4 新增 v5 附錄 |
| dad65b97df | fix: v5 縮小改動範圍 — 只改 Stage 3-6 ss |

**Editable installs:**

| Package | Version | Location |
|---------|---------|----------|
| isaaclab | 0.52.1 | `/home/aa/IsaacLab/source/isaaclab` |
| isaaclab_assets | 0.2.4 | `/home/aa/IsaacLab/source/isaaclab_assets` |
| isaaclab_tasks | 0.11.10 | `/home/aa/IsaacLab/source/isaaclab_tasks` |
| isaaclab_rl | 0.4.7 | `/home/aa/IsaacLab/source/isaaclab_rl` |

---

## 4. Training Entry Points

**Primary entry:** `scripts/reinforcement_learning/skrl/train_charge_ac.py`

**Launcher:** `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py`

**Latest training command (v6 open-ended):**
```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v6 \
  --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk \
  --no_walls \
  --run_name rw_groundv6_openendedv1__seed1_nowalls \
  --seed 1 --num_envs 6144 --headless
```

**Task registry:** `Isaac-Navigation-Charge-VLP16-Curriculum-NavRL` →
- Env: `ChargeNavigationEnvCfgVLP16CurriculumNavRLGroundV3`
- Scene: `MySceneCfgVLP16_20x20` (20×20m, 20 obstacles, 8 wall slots)

**Key CLI parameters (train_charge_ac.py lines 43-141):**

| Arg | Default | Description |
|-----|---------|-------------|
| `--reward_mode` | current | current/v1/v2/v3/v4/v5/v6 |
| `--curriculum_version` | None | baseline_v1/goal_first_v1/v2/v3/open_ended_v1 |
| `--dynamic_safety_mode` | log_distance | log_distance/closing_risk |
| `--num_envs` | None (cfg default 1024) | Typically 6144 via CLI |
| `--seed` | None (yaml default 42) | |
| `--timesteps` | None (yaml 234375) | |
| `--no_walls` | False | Remove all internal walls |
| `--use_cadn` | False | CADN preprocessor |
| `--use_safety_shield` | False | Safety shield |
| `--goal_vel_use_soft_gate` | False | Re-enable velocity soft gate |
| `--w_goal` | 500.0 | Goal terminal weight |
| `--w_collision` | -100.0 | Collision weight |

---

## 5. Effective PPO Hyperparameters

**Source:** `agents/skrl_ppo_cfg_vlp16.yaml`

| Parameter | Value | CLI Override? |
|-----------|-------|:---:|
| Algorithm | PPO (SKRL) | No |
| rollouts | 128 | No |
| learning_epochs | 6 | No |
| mini_batches | 16 | No |
| discount_factor (γ) | 0.990 (initial, curriculum adjusts) | No |
| GAE lambda | 0.95 | No |
| learning_rate | 1e-4 | No |
| LR scheduler | LinearLR (1.0→0.3, total_iters=10986) | No |
| grad_norm_clip | 1.0 | No |
| ratio_clip (ε) | 0.2 | No |
| value_clip | 0.2 | No |
| clip_predicted_values | False | No |
| entropy_loss_scale | 0.01 | No |
| value_loss_scale | 1.0 | No |
| kl_threshold | 0.0 (disabled) | No |
| rewards_shaper_scale | 1.0 | No |
| time_limit_bootstrap | True | No |
| state_preprocessor | RunningStandardScaler | No |
| value_preprocessor | RunningStandardScaler | No |
| timesteps | 234,375 | `--timesteps` |
| checkpoint_interval | 11,718 | No |
| seed | 42 (yaml), 1 (CLI override) | `--seed` |
| num_envs | 1024 (cfg), 6144 (CLI override) | `--num_envs` |
| batch_size | rollouts × num_envs = 128 × 6144 = 786,432 | Derived |
| mini_batch_size | 786,432 / 16 = 49,152 | Derived |
| PPO updates | 234,375 / 128 = 1,831 | Derived |
| total gradient steps | 1,831 × 6 = 10,986 | Derived |
| total env steps | 234,375 × 6144 = 1,440,000,000 | Derived |

---

## 6. Effective Reward Design

**Current:** `navrl_ground_v6` → `RewardsCfgVLP16NavRLGroundV6` (inherits V3)

**Source:** `cfg/charge_env_cfg_vlp16_curriculum.py`

**RewardManager computation:** `R_t = Σ f_i(s,a) × w_i × dt`, dt=0.2s

| # | Term | Function | Weight | Per-step (f=1) | Type | Key Params | Stage Override? |
|---|------|----------|:---:|:---:|------|------------|:---:|
| 1 | reaching_goal | reaching_goal | **+500** | 100.0 | Terminal | threshold=0.35, body_radius=0.35 | No |
| 2 | goal_velocity | goal_velocity_reward | +2.0 | 0.40 | Per-step | use_soft_gate=False, bottom_k=36, v_max=1.0, min_goal_dist=0.1 | **Yes** (curriculum) |
| 3 | goal_progress | goal_progress_reward | +3.0 | 0.60 | Per-step | use_soft_scale=False, progress_clip=0.25 | **Yes** (curriculum) |
| 4 | static_safety | static_safety_reward | +2.0 | 0.40 | Per-step | **a_global=1.0, a_front_block=0.4**, front_angle=30°, front_k=5, warn_dist=1.2 | **Yes** (curriculum) |
| 5 | dynamic_safety | dynamic_safety_reward | +2.0 | 0.40 | Per-step | max_obstacles=20, mode=closing_risk, σ=2.0 | **Yes** (curriculum) |
| 6 | smoothness | control_smoothness_penalty | -0.1 | 0.02 | Per-step | v_coeff=1.0, w_coeff=1.0 | No |
| 7 | collision_ground | collision_terminal_penalty | **-50** | 10.0 | Terminal | threshold=0.45 (LiDAR) | No |
| 8 | alive | alive_reward | **0.0** | — | Disabled | — | No |
| 9 | time_penalty | per_step_time_penalty | **0.0** | — | Disabled | — | No |

**V6 vs V3 diff:** `a_front_block` 1.0→0.4 (弱化前向堵塞)

**V6 vs V5 diff:** `collision_ground` -100→-50 (不加倍)

---

## 7. Effective Curriculum Design

**Current:** `open_ended_v1` — 兩段式 (Bootstrap + Open-ended)

**Source:** `curriculum/goal_obstacle_curriculum.py`

### Bootstrap Stages (B1-B6)

| Stage | Name | Goals | Static | Dynamic | Walls | γ | Episode | Upgrade SR | Reward: vel/prog/ss/ds |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B1 | goal_open | 6 | 0 | 0 | 0 | 0.990 | 45s | >85% | 5.0/6.0/0.0/0.0 |
| B2 | sparse_static | 5 | 2 | 0 | 0 | 0.992 | 50s | >82% | 5.0/5.8/0.2/0.0 |
| B3 | static_light | 4 | 4 | 0 | 0 | 0.993 | 55s | >80% | 4.8/5.5/0.4/0.0 |
| B4 | static_medium | 3 | 6 | 0 | 0 | 0.994 | 60s | >78% | 4.5/5.0/0.6/0.0 |
| B5 | dynamic_intro | 3 | 6 | 2 | 0 | 0.995 | 68s | >75% | 4.0/4.5/0.8/0.3 |
| B6 | dynamic_bridge | 2 | 8 | 3 | 0 | 0.996 | 75s | >72% | 3.5/4.0/1.0/0.5 |

All: `reaching_goal=500, collision=-50, smoothness=-0.1, alive=0, walls=0`
Upgrade: SR + CR<35% + TO<25%, consecutive 5 passes, clear window on promote.

### Open-ended (after B6)

| Level | Static | Dynamic | Goals | Episode | γ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 8 | 3 | 2 | 75s | 0.9960 |
| 1 | 9 | 3 | 2 | 77s | 0.9962 |
| 2 | 10 | 3 | 1 | 79s | 0.9964 |
| 6 | 14 | 5 | 1 | 87s | 0.9972 |
| 12 | 20 | 5 | 1 | 95s | 0.9980 |
| 13+ | 20 | 5 | 1 | 95s | 0.9980 |

Beyond L12: speed_scale ↑ (max 1.4×), spawn compactness ↑ (max 1.3×), goal distance ↑ (max 1.25×)

**Upgrade targets:** SR > max(0.58, 0.72-0.01×level), CR < min(0.28, 0.18+0.01×level), TO < 0.30
**Downgrade:** SR < target-0.12 OR CR > target+0.12, consecutive 2 windows
**Window:** max(4000, num_envs×6) = 36,864 for 6144 envs
**Cooldown:** max(2000, num_envs×4) = 24,576 episodes

---

## 8. Scene / Environment Configuration

**Source:** `cfg/charge_env_cfg_vlp16_curriculum.py`, `MySceneCfgVLP16_20x20`

| Parameter | Value | Source |
|-----------|-------|--------|
| Room size | 20×20m (±10m) | line 86 |
| Boundary wall thickness | 1.0m | line 87 |
| Boundary wall height | 1.5m | line 88 |
| Internal wall slots | 8 (RigidObjectCfg, per-env random) | lines 136-147 |
| Obstacle entities | 20 (hidden at Z=-10, curriculum-controlled) | lines 163-216 |
| Obstacle types | 10 cuboid + 10 cylinder | lines 163-216 |
| env_spacing | 22.0m | line 448 |
| sim.dt | 0.01s | inherited |
| decimation | 20 | inherited |
| control dt | 0.2s | derived |
| episode_length_s | 45s (initial, curriculum adjusts 45→95s) | line 456 |
| max_episode_steps | 225-475 (derived from episode_length_s) | derived |

**Terminations:**

| Condition | Threshold | Type |
|-----------|-----------|------|
| time_out | episode_length_s | Truncation (bootstrap=True) |
| goal_reached | dist < 0.35m (body-subtracted) | Success |
| collision (LiDAR) | d_min ≤ 0.45m | Failure |
| wall_collision (AABB) | dist < 0.45m | Failure |
| robot_tipped_over | z-axis < threshold | Failure |
| physics_explosion | vel > threshold or NaN | Failure |

**Spawn safety:** SPAWN_SAFE_DIST = 1.0m (walls + obstacles), 50 retry max

---

## 9. Observation / Sensor Configuration

### LiDAR (VLP-16 simulated)

| Parameter | Value |
|-----------|-------|
| Type | MultiMeshRayCasterCfg |
| Channels | 16 |
| Vertical FoV | -15° ~ +15° |
| Horizontal FoV | -180° ~ +180° (360°) |
| Horizontal res | 1.0°/ray |
| Total rays | 16 × 360 = 5,760 |
| Max distance | 20.0m |
| Update period | 0.2s (= control dt) |
| Mount offset | (0, 0, 0.5)m above base_link |
| Compression | Min-pool 5760 rays → 72 bins, normalize to [0,1] |

### Observation Vector (139D, symmetric actor=critic)

| Index | Field | Dim | Function |
|-------|-------|:---:|----------|
| [0] | normalized linear acceleration | 1 | normalized_linear_acceleration |
| [1] | normalized linear velocity | 1 | normalized_linear_velocity |
| [2] | angular velocity ω_z | 1 | base_angular_velocity_z |
| [3] | robot radius (constant 0.35) | 1 | robot_radius_obs |
| [4:6] | goal position (body frame) | 2 | goal_position_in_robot_frame |
| [6:78] | LiDAR 72 bins | 72 | lidar_vlp16_to_2d_bins |
| [78:138] | Top-10 obstacles × 6D | 60 | topk_obstacles_6d |
| [138] | time remaining ratio | 1 | time_remaining_ratio |

**Obstacle 6D:** [dx, dy, vx, vy, radius, mask] in body frame, LOS occlusion

**Normalization:** RunningStandardScaler (obs_norm = (obs - μ) / (σ + ε))

---

## 10. Network Architecture

**Source:** `scripts/reinforcement_learning/skrl/vlp16_models.py`

### Feature Extractor (128D output)

| Branch | Input | Architecture | Output |
|--------|:---:|-------------|:---:|
| LiDAR Conv1d | [B,72] | Conv1d(1→32,k5) → ReLU → Conv1d(32→64,k5,s2) → ReLU → Conv1d(64→64,k3,s2) → ReLU → AdaptiveMaxPool1d(1) → Linear(64,64) → LayerNorm | 64D |
| Obstacle MLP | [B,60]→[B,10,6] | Linear(6→32) → ReLU → Linear(32→32) → ReLU → MaxPool(dim=1) → LayerNorm | 32D |
| State MLP | [B,7] | Linear(7→32) → ReLU → Linear(32→32) → ReLU → LayerNorm | 32D |

Fusion: cat([64, 32, 32]) = **128D**

### Actor (VLP16DiscretePolicy)

```
139D → FeatureExtractor → 128D → Linear(128,128) → ReLU → Linear(128,38)
```
- 38 logits = 19 (linear accel) + 19 (angular vel)
- Mixin: MultiCategoricalMixin (unnormalized_log_prob=True, reduction="sum")
- **Separate** feature extractor from critic (models_separate=True)

### Critic (VLP16Value)

```
139D → FeatureExtractor → 128D → Linear(128,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)
```
- Init: orthogonal_(last_layer.weight, gain=0.01), constant_(bias, **-8.0**)
- Mixin: DeterministicMixin

---

## 11. Action Space / Control Pipeline

| Parameter | Value | Source |
|-----------|-------|--------|
| Action type | MultiDiscrete([19, 19]) | DiscreteDifferentialDriveActionCfg |
| Dimensions | 2 (accel_idx, omega_idx) | |
| Index range | [0, 18], center=9 = zero action | |
| v_max | 1.0 m/s | cfg line 314 |
| a_max | 0.5 m/s² | cfg line 315 |
| ω_max | 0.25π ≈ 0.785 rad/s | cfg line 316 |
| Velocity domain | [-1.0, +1.0] m/s (reverse allowed) | |
| Dynamic accel bounds | a_min = max(-a_max, (-v_max-v)/dt), a_max_actual = min(a_max, (v_max-v)/dt) | |
| Safety shield | **Not enabled by default** (`--use_safety_shield` required) | |
| Gap reward | **Not enabled by default** (`--use_gap_reward` required) | |

---

## 12. Domain Randomization

**Source:** `domain_randomization/`

| Category | Parameter | Value |
|----------|-----------|-------|
| Physics | mass_scale | (0.85, 1.15) |
| Physics | friction_scale | (0.7, 1.3) |
| Physics | COM offset | ±0.05m XY, ±0.025m Z |
| Sensor | LiDAR dropout | 5% |
| Sensor | distance noise | ±0.03m (Gaussian) |
| Sensor | channel bias | ±0.05m |
| Sensor | ghost rate | 0.2% |
| Sensor | ghost range | (0.2, 2.0)m |
| Obs noise | angular_velocity | Uniform(-0.05, 0.05) |
| Obs noise | LiDAR bins | Uniform(-0.02, 0.02) |
| Disturbance | push force | (5, 20)N, 10% of envs |
| Disturbance | wind force | (0, 3)N |
| Actuator | action delay | 0-2 steps (disabled by default) |
| Actuator | velocity scale | (0.9, 1.1) (disabled by default) |
| Reset | init vx | (-0.5, 0.5) m/s |
| Reset | init vy | (-0.15, 0.15) m/s |
| Reset | init ωz | (-0.5, 0.5) rad/s |

---

## 13. Runtime Environment

| Package | Version |
|---------|---------|
| Python | 3.11.14 |
| conda env | env_isaaclab |
| PyTorch | 2.7.0+cu128 |
| CUDA (runtime) | 13.0 |
| NVIDIA Driver | 580.126.09 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 0.52.1 (editable) |
| SKRL | 1.4.3 |
| WandB | 0.23.1 |

**Environment variables:**
- `WANDB_PROJECT` = (unset)
- `CHARGE_USD_PATH` = (unset, uses repo default `assets/usd/charge/charge.usd`)
- `CUDA_VISIBLE_DEVICES` = (unset, uses GPU 0)

---

## 14. Recent Runs / Checkpoints

| Run Name | Reward | Curriculum | Envs | Seed | --no_walls | CSV Rows | Latest Ckpt |
|----------|--------|------------|:---:|:---:|:---:|:---:|-------------|
| rw_groundv6_openendedv1__seed1_nowalls | v6 | open_ended_v1 | 6144 | 1 | Yes | 143 | agent_11718.pt |
| rw_groundv5_goalfirstv3_full_seed1 | v5 | goal_first_v3 | 6144 | 1 | Yes | 12 (restarted) | agent_164052.pt |
| rw_groundv4_goalfirstv2_full_seed1 | v4 | goal_first_v2 | 6144 | 1 | No | 494 | agent_58590.pt |
| rw_groundv3_goalfirstv2_full_seed1 | v3 | goal_first_v2 | 6144 | 1 | No | 413 | agent_46872.pt |
| rw_groundv2_closingrisk_full_seed1_v2fix | v2 | goal_first_v1 | 6144 | 1 | No | 1551 | agent_187488.pt |
| rw_groundv1_closingrisk_full_seed1 | v1 | goal_first_v1 | **512** | 1 | No | 369 | agent_46874.pt |

---

## 15. Code-vs-Document Consistency Check

| Item | Code | docs/REWARD_DESIGN_V4.md | Match? |
|------|------|--------------------------|:---:|
| reaching_goal weight | 500 (V3/V6) | 500 | ✅ |
| collision_ground weight | -50 (V6) | -50 | ✅ |
| alive weight | 0.0 (V3/V6) | 0.0 | ✅ |
| a_front_block | 0.4 (V6) | Not yet documented for V6 | ⚠️ |
| open_ended_v1 | Code exists | Not in REWARD_DESIGN_V4.md | ⚠️ |
| SPAWN_SAFE_DIST | 1.0m (code) | CLAUDE.md says 1m | ✅ |
| max_obstacles | 20 (curriculum scene) | CLAUDE.md says 10 | ❌ |

---

## 16. High-Risk Difference Candidates

When comparing PC-A vs PC-B, check these in priority order:

| # | Risk Factor | Status | Evidence |
|---|-------------|:---:|---------|
| 1 | **Branch / commit** | ✅ Confirmed | `abl` @ `d03c17c4bb` |
| 2 | **Uncommitted changes** | ✅ Clean | `git status` shows no M/A |
| 3 | **reward_mode CLI** | Check | v6 is latest; v1-v5 also available |
| 4 | **curriculum_version CLI** | Check | open_ended_v1 is latest |
| 5 | **num_envs** | Check | 6144 via CLI (default 1024) |
| 6 | **seed** | Check | 1 via CLI (default 42) |
| 7 | **--no_walls** | Check | v6 uses it; v4 did not |
| 8 | **--use_cadn** | Check | v5 used it; v6 does not |
| 9 | **GPU model** | ✅ Confirmed | RTX 5090 |
| 10 | **CUDA / Driver** | ✅ Confirmed | CUDA 13.0 / 580.126.09 |
| 11 | **PyTorch** | ✅ Confirmed | 2.7.0+cu128 |
| 12 | **Isaac Sim** | ✅ Confirmed | 5.1.0.0 |
| 13 | **SKRL** | ✅ Confirmed | 1.4.3 |
| 14 | **Isaac Lab** | ✅ Confirmed | 0.52.1 (editable) |
| 15 | **Python** | ✅ Confirmed | 3.11.14 |
| 16 | **Safety shield** | ✅ Disabled | Not in any recent command |
| 17 | **LR scheduler total_iters** | Check | 10986 in YAML (= total gradient steps) |
| 18 | **checkpoint resume** | Check | Most runs start from scratch |
| 19 | **WANDB_PROJECT** | ⚠️ Unset | May differ on PC-B |
| 20 | **USD asset path** | ✅ Repo default | `assets/usd/charge/charge.usd` |

---

## 17. Appendix: Raw Commands and Evidence

### A. System info commands
```
hostname → user-5090
uname -a → Linux user-5090 6.17.0-19-generic ... x86_64
python --version → Python 3.11.14
nvidia-smi → RTX 5090, 580.126.09, CUDA 13.0, 32607 MiB
```

### B. Git commands
```
git remote -v → charge_skrl git@github.com:me0608623/charge_skrl.git
git branch --show-current → abl
git rev-parse HEAD → d03c17c4bbc111c2403384b3e7d0633fb5b9ff21
git status --short → (clean, only untracked .claude/ dirs)
git stash list → 2 entries on main
```

### C. Package versions
```
pip show torch → 2.7.0+cu128
pip show skrl → 1.4.3
pip show wandb → 0.23.1
pip show isaacsim → 5.1.0.0
pip show isaaclab → 0.52.1
```

### D. Key file paths
```
Train script: scripts/reinforcement_learning/skrl/train_charge_ac.py
PPO YAML:     .../charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml
Reward cfg:   .../charge_skrl/cfg/charge_env_cfg_vlp16_curriculum.py
Curriculum:   .../charge_skrl/curriculum/goal_obstacle_curriculum.py
Network:      scripts/reinforcement_learning/skrl/vlp16_models.py
Scene cfg:    .../charge_skrl/cfg/charge_env_cfg_vlp16.py
Actions:      .../charge_skrl/mdp/actions/discrete_differential_drive.py
Reset:        .../charge_skrl/mdp/events/reset.py
Obstacles:    .../charge_skrl/mdp/events/mixed_parallel.py
DR:           .../charge_skrl/domain_randomization/dr_events.py
Rewards:      .../charge_skrl/mdp/rewards/navrl_ground_rewards.py
```
