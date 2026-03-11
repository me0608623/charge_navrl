# Charge-SKRL: Safe Navigation via Discrete PPO in Isaac Lab

> A sim-to-real navigation framework for differential-drive robots using Proximal Policy Optimization with a discrete action space, 3D LiDAR perception, and potential-based reward shaping.

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Observation Space](#3-observation-space)
4. [Action Space](#4-action-space)
5. [Network Architecture](#5-network-architecture)
6. [Reward Design](#6-reward-design)
7. [Termination Conditions](#7-termination-conditions)
8. [Training Environment](#8-training-environment)
9. [Domain Randomization](#9-domain-randomization)
10. [PPO Hyperparameters](#10-ppo-hyperparameters)
11. [Quick Start](#11-quick-start)
12. [Project Structure](#12-project-structure)
13. [References](#13-references)

---

## 1. Overview

This project trains a differential-drive mobile robot (**Charge**) to navigate safely to goal positions in cluttered environments with both static and dynamic obstacles. The system is built on NVIDIA Isaac Lab and uses the SKRL reinforcement learning library.

### Key Design Choices

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Action space | Discrete(361) | Eliminates continuous-action entropy collapse; enables center-symmetric exploration |
| Perception | VLP-16 LiDAR → 72-bin 2D projection | Sim-to-real transferable; robust to 3D geometry |
| Reward shaping | Potential-based (PBRS) | Policy-invariant; bounded total reward |
| Safety signal | Linear near-obstacle penalty + binary collision | Continuous gradient before collision + hard penalty at contact |
| Training paradigm | Mixed parallel curriculum | 50% empty / 30% static / 20% dynamic obstacle environments |

### Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Robot body radius | 0.35 | m |
| Max linear velocity | 1.0 | m/s |
| Max linear acceleration | 0.5 | m/s² |
| Max angular velocity | 0.25π ≈ 0.785 | rad/s |
| Control frequency | 5 | Hz |
| Simulation dt | 0.01 | s |
| Decimation | 20 | steps |
| Environment dt | 0.2 | s |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab Environment                     │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Scene    │  │  VLP-16  │  │ Contact  │  │   Goal     │ │
│  │ (Robot,  │  │  LiDAR   │  │ Sensor   │  │  Command   │ │
│  │  Walls,  │  │ 16×360   │  │          │  │ (5 goals)  │ │
│  │  Obs.)   │  │ = 5760   │  │          │  │            │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │
│       │             │             │               │        │
│  ┌────▼─────────────▼─────────────▼───────────────▼──────┐ │
│  │              Observation Manager (139D)                │ │
│  │  ego(4) + goal(2) + LiDAR(72) + obstacles(60) + t(1) │ │
│  └───────────────────────┬───────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────────┐ │
│  │              Reward Manager (8 terms)                  │ │
│  │  r = Σ func_i(s) × weight_i × dt                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           Domain Randomization (per reset)            │ │
│  │  Physics DR · Sensor DR · Disturbance DR              │ │
│  └───────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ obs (139D)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    SKRL PPO Agent                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            VLP16FeatureExtractor (128D)                │  │
│  │                                                        │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐  │  │
│  │  │ LiDAR Conv1d│ │ Obstacle MLP │ │   State MLP    │  │  │
│  │  │ 72 → 64D    │ │ 60 → 32D     │ │   7 → 32D     │  │  │
│  │  │ (MaxPool)   │ │ (MaxPool)    │ │                │  │  │
│  │  └──────┬──────┘ └──────┬───────┘ └───────┬────────┘  │  │
│  │         └───────────────┴─────────────────┘            │  │
│  │                     concat = 128D                      │  │
│  └─────────────────────────┬──────────────────────────────┘  │
│                            │                                  │
│         ┌──────────────────┴──────────────────┐               │
│         ▼                                     ▼               │
│  ┌─────────────┐                      ┌─────────────┐        │
│  │   Actor     │                      │   Critic    │        │
│  │ 128→128→361 │                      │ 128→64→32→1 │        │
│  │ (Categorical│                      │ (Scalar V)  │        │
│  │  Logits)    │                      │             │        │
│  └──────┬──────┘                      └─────────────┘        │
│         │                                                     │
│    action ∈ {0, ..., 360}                                     │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│          Discrete Differential Drive Controller              │
│                                                              │
│  action_index → (accel_idx, omega_idx) → (a, ω)            │
│  v_{t+1} = clamp(v_t + a·dt, -v_max, +v_max)               │
│  Apply body-frame velocity to simulation                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Observation Space

The policy and value networks receive an identical **139-dimensional** observation vector (symmetric Actor-Critic, no privileged information).

### 3.1 Layout

| Index | Group | Dim | Description | Range |
|-------|-------|-----|-------------|-------|
| `[0]` | Ego | 1 | Normalized linear acceleration: `ā = a / a_max` | [-1, 1] |
| `[1]` | Ego | 1 | Normalized linear velocity: `v̄ = v / v_max` | [-1, 1] |
| `[2]` | Ego | 1 | Normalized angular velocity: `ω̄ = ω_z / ω_max` | [-1, 1] |
| `[3]` | Ego | 1 | Robot body radius (constant) | 0.35 |
| `[4:6]` | Goal | 2 | Goal position in robot body frame `(x, y)` | m |
| `[6:78]` | LiDAR | 72 | 2D min-distance bins, normalized | [0, 1] |
| `[78:138]` | Obstacles | 60 | Top-10 obstacles × 6D in body frame | mixed |
| `[138]` | Time | 1 | Remaining time ratio: `1 - t/T_max` | [0, 1] |

### 3.2 LiDAR Processing Pipeline

The VLP-16 sensor emits 5,760 rays (16 vertical channels × 360 horizontal samples). These are compressed into a 72-dimensional vector:

```
ray_hits_w [N, 5760, 3]          (raw 3D hit points)
  │
  ├── 2D projection: d_ij = ||hit_xy - sensor_xy||₂
  │
  ├── Clipping: d > r_max → r_max, NaN/Inf → r_max
  │
  ├── Point Cloud Corruption (Sim-to-Real DR):
  │     • Gaussian displacement: σ = 0.02m
  │     • Ray dropout: 0.5% → d = r_max
  │     • Ghost points: 0.2% → d ~ U(0.2, 2.0)m
  │
  ├── Reshape: [N, 5760] → [N, 16ch, 72bins, 5rays/bin]
  │
  ├── Min-pool: min over rays_per_bin → min over channels
  │     → [N, 72]   (per-bin minimum distance)
  │
  ├── Robot radius subtraction: d' = max(d - r_body, 0)
  │
  └── Normalize: d̄ = d' / r_max ∈ [0, 1]
```

### 3.3 Obstacle Observations

Each obstacle is represented by a 6D vector in the robot's body frame:

```
o_i = [x̂, ŷ, v̂_x, v̂_y, r, m]
```

| Field | Definition | Normalization |
|-------|-----------|---------------|
| `x̂, ŷ` | Relative position in body frame | `/ 8.0m`, clamp [-1, 1] |
| `v̂_x, v̂_y` | Relative velocity in body frame | `/ 1.5 m/s`, clamp [-2, 2] |
| `r` | Object radius | meters (raw) |
| `m` | Visibility mask | 1.0 = visible, 0.0 = occluded/padded |

**Body-frame rotation**:
```
[x̂]   [cos(ψ)   sin(ψ)] [Δx]
[ŷ] = [-sin(ψ)  cos(ψ)] [Δy]
```

**Top-K selection**: Sorted by distance, nearest 10 kept. Invalid slots zeroed with `torch.where` (not multiplication, to avoid IEEE 754 NaN propagation).

**Wall LOS occlusion**: AABB segment intersection against all maze walls. Occluded obstacles receive `m = 0`.

---

## 4. Action Space

### 4.1 Discrete(361) = 19 × 19 Center-Symmetric Grid

The neural network outputs a single integer `a ∈ {0, 1, ..., 360}`.

**Decoding**:
```
accel_idx = a // 19     ∈ {0, ..., 18}
omega_idx = a % 19      ∈ {0, ..., 18}

ratio_a = (accel_idx - 9) / 9     ∈ [-1.0, +1.0]
ratio_ω = (omega_idx - 9) / 9     ∈ [-1.0, +1.0]
```

Index 9 maps to zero (no acceleration, no turning). This **center-symmetric** design ensures uniform coverage of the action manifold.

### 4.2 Dynamic Acceleration Bounds

To guarantee velocity remains within physical limits after integration:

```
a_upper = min(+a_max, (+v_max - v_t) / dt)
a_lower = max(-a_max, (-v_max - v_t) / dt)
```

The actual acceleration is:
```
        ┌ ratio_a × a_upper    if ratio_a ≥ 0
a_t =   │
        └ -ratio_a × a_lower   if ratio_a < 0
```

### 4.3 Velocity Integration

```
v_{t+1} = clamp(v_t + a_t · dt,  -v_max, +v_max)
ω_t     = ratio_ω · ω_max
```

The resulting `(v_{t+1}, ω_t)` is transformed from body frame to world frame via the robot's yaw quaternion and applied as root velocity.

---

## 5. Network Architecture

### 5.1 Feature Extractor (3-Branch, 128D output)

#### LiDAR Branch (72D → 64D)
```
Input [B, 72] → reshape [B, 1, 72]
  → Conv1d(1→32, kernel=5, pad=2) → ReLU
  → Conv1d(32→64, kernel=5, stride=2, pad=2) → ReLU
  → Conv1d(64→64, kernel=3, stride=2, pad=1) → ReLU
  → AdaptiveMaxPool1d(1)
  → Linear(64→64) → LayerNorm(64)
Output [B, 64]
```

#### Obstacle Branch (60D → 32D, permutation-invariant)
```
Input [B, 60] → reshape [B, 10, 6]
  → Linear(6→32) → ReLU → Linear(32→32) → ReLU   (shared per object)
  → MaxPool over 10 objects (dim=1)
  → LayerNorm(32)
Output [B, 32]
```

The per-object MLP is shared across all 10 objects. MaxPool aggregation ensures the representation is **permutation-invariant** with respect to obstacle ordering — a critical property since object identity is arbitrary.

#### State Branch (7D → 32D)
```
Input [B, 7]   (ego 4D + goal 2D + time 1D)
  → Linear(7→32) → ReLU → Linear(32→32) → ReLU → LayerNorm(32)
Output [B, 32]
```

#### Fusion
```
features = cat([LiDAR(64), Obstacles(32), State(32)]) = 128D
```

### 5.2 Policy Head (Actor)

```
features [B, 128]
  → Linear(128→128) → ReLU
  → Linear(128→361)
Output: 361 unnormalized logits → Categorical distribution
```

### 5.3 Value Head (Critic)

```
features [B, 128]
  → Linear(128→64) → ReLU
  → Linear(64→32) → ReLU
  → Linear(32→1)
Output: scalar V(s)
```

Initialization: output layer uses `orthogonal_(gain=0.01)` with bias = `-8.0` (pessimistic initial value estimate to prevent early overestimation).

---

## 6. Reward Design

### 6.1 Formulation

IsaacLab's Reward Manager computes each reward term as:

```
r_i(t) = func_i(s_t) × weight_i × dt
```

where `dt = 0.2s`.

### 6.2 Reward Terms

#### (1) Goal Reaching — `reaching_goal`

```
         ┌ 1.0    if d(robot, goal) - r_body < threshold
func =   │
         └ 0.0    otherwise

weight = +250.0
r = +50.0 per trigger   (250 × 0.2)
```
**Category**: Sparse terminal reward. Fires once when robot body edge enters goal region. Threshold = 0.35m.

#### (2) Potential Progress — `potential_progress`

```
Φ(s) = -d(robot, goal)
func = Φ(s_{t+1}) - Φ(s_t) = d_t - d_{t+1}
func = clamp(func, -1.0, +1.0)

weight = +60.0
r = 12.0 × Δd   (60 × 0.2 × Δd)
```
**Category**: Potential-Based Reward Shaping (Ng et al., 1999). The telescoping property guarantees that total shaping reward equals `12 × (d_0 - d_final)`, independent of path taken. This preserves optimal policy invariance.

#### (3) Near-Obstacle Penalty — `near_obstacle_penalty`

```
d_min = min(LiDAR 2D distances)

func = max(0, α · (d_safe - d_min))
     = max(0, 4.0 · (1.2 - d_min))

weight = -1.0
r = -0.2 × max(0, 4.0 · (1.2 - d_min))
```

**Category**: Continuous safety signal. Provides gradient information before collision occurs.

| d_min | func | r per step |
|-------|------|------------|
| ≥ 1.2m | 0 | 0 |
| 1.0m | 0.8 | -0.16 |
| 0.7m | 2.0 | -0.40 |
| 0.5m | 2.8 | -0.56 |

#### (4) Collision Terminal — `collision_terminal`

```
func = 1.0   if d_min ≤ 0.7m   (= r_body + collision_buffer)
     = 0.0   otherwise

weight = -80.0
r = -16.0 per trigger   (-80 × 0.2)
```
**Category**: Binary terminal penalty. Fires on the same step as episode termination.

#### (5) Time Penalty — `time_penalty`

```
func = 1.0   (constant every step)

weight = -0.3
r = -0.06 per step
```
**Category**: Anti-loafing. Full episode (225 steps): cumulative = -13.5.

#### (6) Velocity Too Low — `velocity_too_low`

```
func = 1.0   if ||v_xy||₂ < 0.05 m/s
     = 0.0   otherwise

weight = -0.3
r = -0.06 per step when stationary
```
**Category**: Anti-freezing. Prevents the degenerate "do nothing = safe" policy.

#### (7) Acceleration Penalty — `acceleration_penalty`

```
func = |applied_a|²    (from action term, post-clipping)
func = clamp(func, 0, 10)

weight = -0.15
r = -0.03 × a²
```
**Category**: Smoothness. Max penalty at a_max=0.5: r = -0.0075/step (negligible).

#### (8) Angular Velocity Penalty — `angular_velocity_penalty`

```
func = |ω_z|²
func = clamp(func, 0, 10)

weight = -0.15
r = -0.03 × ω²
```
**Category**: Smoothness. Encourages straight-line motion, penalizes unnecessary oscillation.

### 6.3 Reward Scale Analysis

| Scenario | Dominant Terms | Total r/step |
|----------|---------------|--------------|
| Normal navigation (0.5 m/s, clear path) | progress ≈ +1.2, time = -0.06 | **≈ +1.1** |
| Near obstacle (d_min = 0.8m) | progress ≈ +1.2, near_obs ≈ -0.32 | **≈ +0.8** |
| Goal reached | goal = +50.0 | **≈ +51** |
| Collision | collision = -16.0 | **≈ -16.4** |
| Stationary | time = -0.06, vel_low = -0.06 | **≈ -0.12** |

**Design principle**: The reward hierarchy is `goal(+50) >> progress(+2.4/step max) >> safety(-16) >> anti-loaf(-0.12)`. One collision costs 16.0 / 1.1 ≈ **15 steps of normal navigation** to recover.

---

## 7. Termination Conditions

| Condition | Type | Trigger | Notes |
|-----------|------|---------|-------|
| `time_out` | Truncation | Episode > 45s (225 steps) | Bootstrapped (GAE continues) |
| `goal_reached` | Terminal | `d(robot, goal) - r_body < 0.35m` | Episode success |
| `collision` | Terminal | LiDAR d_min ≤ 0.7m | `r_body + 0.35m` buffer |
| `robot_tipped_over` | Terminal | Z-axis dot product with up < 0.5 | > 60° tilt |
| `physics_explosion` | Terminal | `||v|| > 10 m/s` or `|ω| > 20 rad/s` or NaN | Simulation instability guard |

---

## 8. Training Environment

### 8.1 Arena

- **Room size**: 16m × 16m (boundary walls at ±8m)
- **Internal walls**: 6 maze segments (cuboid, kinematic)
- **Parallel environments**: 1024–6144 (configurable)
- **Environment spacing**: 18m (prevents cross-env interference)

### 8.2 Mixed Parallel Curriculum

Each environment is independently assigned a difficulty level:

| Difficulty | Proportion | Obstacles | Obstacle Motion |
|------------|-----------|-----------|-----------------|
| Empty | 50% | 0 | — |
| Static | 30% | 5 | Stationary |
| Dynamic | 20% | 8 | 0.3–1.2 m/s random walk |

This mixed-parallel approach provides a natural curriculum: the agent simultaneously experiences easy and hard scenarios, preventing catastrophic forgetting while maintaining training efficiency.

### 8.3 Obstacle Spawning

- **Safe distance from robot**: 1.5m minimum
- **Safe distance from goal**: 1.0m minimum
- **Inter-obstacle spacing**: 1.0m minimum
- **Placement method**: Quadrant-based rejection sampling (max 50 attempts with fallback)
- **Types**: Cuboids (0.4–0.7m) and cylinders (r = 0.2–0.35m), randomly mixed

### 8.4 Goal Command

- **Multi-goal**: 5 sequential goals per episode
- **Distance range**: [3, 8]m from robot
- **Angle range**: full 360°
- **Rejection sampling**: boundary check (7.5m), wall proximity (0.5m), obstacle clearance (1.0m)

---

## 9. Domain Randomization

Domain randomization is applied per-episode at reset to improve sim-to-real transfer.

### 9.1 Physics Randomization

| Parameter | Range | Physical Meaning |
|-----------|-------|-----------------|
| Robot mass | ±15% | Payload variation |
| Ground friction | ±30% | Surface material (tile, carpet, concrete) |
| CoM offset XY | ±5cm | Asymmetric load placement |
| CoM offset Z | ±2.5cm | Vertical load variation |

### 9.2 Initial State Randomization

| Parameter | Range |
|-----------|-------|
| Linear velocity X | ±0.5 m/s |
| Linear velocity Y | ±0.15 m/s |
| Angular velocity Z | ±0.5 rad/s |

### 9.3 External Disturbances

| Type | Parameters | Mechanism |
|------|-----------|-----------|
| Random push | 10–30N, 15% of envs, random direction | `instantaneous_wrench_composer` (single-step impulse) |
| Continuous wind | 0–5N, random direction per episode | `permanent_wrench_composer` (persistent force) |

### 9.4 Sensor Noise (in LiDAR processing)

| Type | Parameters | VLP-16 Spec Reference |
|------|-----------|----------------------|
| Gaussian displacement | σ = 0.02m | ±3cm typical accuracy |
| Ray dropout | 0.5% probability | Specular reflection, dark surfaces |
| Ghost points | 0.2%, d ∈ [0.2, 2.0]m | Multi-path reflections |
| Output noise | Uniform ±0.02 (normalized) | ADC quantization |

---

## 10. PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | PPO (Clip) | SKRL implementation |
| Rollout length | 128 steps | Per update |
| Learning epochs | 8 | Per PPO update |
| Mini-batches | 8 | Per epoch |
| Discount factor (γ) | 0.98 | Reduced from 0.99 to limit value magnitude |
| GAE lambda (λ) | 0.95 | |
| Learning rate | 1×10⁻⁴ | KL-Adaptive scheduler (kl_threshold = 0.016) |
| Ratio clip (ε) | 0.2 | Standard PPO clip |
| Value clip | 0.2 | `clip_predicted_values = True` |
| Entropy coefficient | 0.01 | H_max = ln(361) ≈ 5.89 |
| Value loss scale | 0.5 | Prevents critic gradient dominance |
| Gradient norm clip | 1.0 | |
| State preprocessor | RunningStandardScaler | Both policy and value inputs |
| Time limit bootstrap | True | Prevents timeout bias |
| Seed | 42 | |

**Batch size calculation**:
- Per update: `128 steps × N_envs` samples
- With 4096 envs: `524,288 samples / 8 mini-batches = 65,536` per mini-batch
- Total gradient updates per PPO update: `8 epochs × 8 mini-batches = 64`

---

## 11. Quick Start

### Prerequisites

- NVIDIA Isaac Lab (Isaac Sim 4.5+)
- SKRL ≥ 1.3
- Python 3.11, PyTorch 2.x, CUDA 12+
- Weights & Biases account (optional)

### Training

```bash
conda activate env_isaaclab

# Standard training (4096 parallel envs)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 4096 \
  --headless

# With GPU memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 6144 \
  --headless
```

### Evaluation

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge.py \
  --task Isaac-Navigation-Charge-VLP16 \
  --num_envs 4 \
  --checkpoint logs/skrl/Isaac-Navigation-Charge-VLP16-AC/<timestamp>/checkpoints/best_agent.pt
```

### GPU Memory Guide

| num_envs | VRAM (approx.) | Recommended GPU |
|----------|----------------|-----------------|
| 1024 | ~10 GB | RTX 3080 |
| 2048 | ~14 GB | RTX 4080 |
| 4096 | ~22 GB | RTX 4090 / A6000 |
| 6144 | ~28 GB | RTX 5090 |

---

## 12. Project Structure

```
charge_skrl/
├── scripts/reinforcement_learning/skrl/
│   ├── train_charge_ac.py          # Training entry point
│   ├── vlp16_models.py             # VLP16 3-branch network
│   ├── wandb_trainer.py            # WandB-enabled trainer
│   ├── aac_wrapper.py              # Asymmetric AC wrapper
│   ├── training_debug_logger.py    # Debug diagnostics
│   ├── console_summary.py          # Console metrics logger
│   └── play_charge.py              # Evaluation script
│
├── charge_skrl/
│   ├── __init__.py                 # Gymnasium task registration
│   ├── goal_command.py             # Single goal command
│   ├── multi_goal_command.py       # Multi-goal sequential command
│   │
│   ├── cfg/
│   │   ├── charge_cfg.py           # Robot USD/URDF configuration
│   │   ├── charge_env_cfg.py       # Base environment config
│   │   └── charge_env_cfg_vlp16.py # VLP-16 environment config
│   │
│   ├── agents/
│   │   └── skrl_ppo_cfg_vlp16.yaml # PPO hyperparameters
│   │
│   ├── mdp/
│   │   ├── actions/
│   │   │   └── discrete_differential_drive.py   # Discrete(361) action
│   │   ├── observations/
│   │   │   ├── functions.py         # Ego state observations
│   │   │   └── obs_functions.py     # LiDAR & obstacle observations
│   │   ├── rewards/
│   │   │   ├── potential_based_rewards.py  # PBRS, penalties
│   │   │   ├── goal_rewards.py      # Goal reaching reward
│   │   │   └── safety_rewards.py    # Collision penalties
│   │   ├── terminations/
│   │   │   ├── goal.py              # Goal reached termination
│   │   │   └── robot_state.py       # Collision, tip-over, explosion
│   │   └── events/
│   │       ├── reset.py             # Safe initial state sampling
│   │       ├── obstacles.py         # Obstacle spawning
│   │       └── mixed_parallel.py    # Difficulty curriculum
│   │
│   ├── domain_randomization/
│   │   ├── dr_events.py             # DR orchestration
│   │   ├── physics_dr.py            # Mass, friction, CoM
│   │   ├── sensor_dr.py             # LiDAR noise
│   │   ├── disturbance_dr.py        # Wind, push forces
│   │   └── actuator_dr.py           # Motor response variation
│   │
│   └── charge_URDF/urdf/charge.urdf # Robot model
```

---

## 13. References

1. **Ng, A. Y., Harada, D., & Russell, S. (1999)**. Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017)**. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

3. **Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021)**. RMA: Rapid Motor Adaptation for Legged Robots. *RSS*.

4. **Xu, H., et al. (2025)**. NavRL: Learning Safe Flight in Dynamic Environments. *IEEE RA-L*.

5. **NVIDIA Isaac Lab**. https://github.com/isaac-sim/IsaacLab

6. **SKRL**. https://github.com/Toni-SM/skrl

---

*Generated for the Charge-SKRL project. Branch: `charge-skrl-vlp16`.*
