# Charge-SKRL Curriculum Learning

> **4-Phase Curriculum Learning for Mobile Robot Navigation with SKRL PPO**

[![English](https://img.shields.io/badge/lang-English-blue)](README_CURRICULUM.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-red)](README_CURRICULUM_zh_TW.md)

---

## Overview

This project implements a **4-phase curriculum learning framework** for training a differential-drive mobile robot (Charge) to perform autonomous navigation in dynamic obstacle environments using Isaac Lab simulation and SKRL PPO.

### Key Features

- **Progressive 4-Phase Training**: From dense exploration to ultimate challenge
- **Goal-Obstacle Coupling**: Target count, distance, and obstacle density adjusted in sync
- **Dynamic Reward Weighting**: Collision penalties increase with phase progression
- **Catastrophic Forgetting Prevention**: Mixed curriculum scheduler support
- **VLP-16 LiDAR Perception**: 72-beam laser scan + Top-10 object observations

---

## 4-Phase Curriculum Design

### Phase 1: Dense Exploration

**Goal**: Learn to "reach the target"

| Parameter | Value |
|-----------|-------|
| Goals | 8 |
| Goal Distance | 2.0 - 5.0 m |
| Static Obstacles | 0 |
| Dynamic Obstacles | 0 |
| Collision Penalty | -30 |
| Upgrade Condition | SR > 72% |

**Rationale**: Frequent success rewards from many close targets; no obstacles to distract from navigation learning; light collision penalty doesn't interfere with direction sensing.

---

### Phase 2: Sparse Navigation

**Goal**: Learn "long-range stable navigation"

| Parameter | Value |
|-----------|-------|
| Goals | 3 |
| Goal Distance | 4.0 - 8.0 m |
| Static Obstacles | 2 |
| Dynamic Obstacles | 1 |
| Collision Penalty | -50 |
| Upgrade Condition | SR > 72% AND CR < 30% |

**Rationale**: Fewer targets at longer distances prevent luck-based success; gradual obstacle introduction; navigation focus maintained with moderate collision penalty.

---

### Phase 3: Safe Obstacle Avoidance

**Goal**: Learn to "avoid obstacles" with existing navigation capability

| Parameter | Value |
|-----------|-------|
| Goals | 2 |
| Goal Distance | 3.0 - 8.0 m |
| Static Obstacles | 5 |
| Dynamic Obstacles | 3 |
| Collision Penalty | -200 |
| Upgrade Condition | SR > 80% AND CR < 20% |

**Rationale**: Significantly increased collision penalty lets agent "feel the pain"; higher obstacle density focuses on avoidance; agent already knows navigation, won't be confused.

---

### Phase 4: Ultimate Challenge

**Goal**: Precise navigation in dense dynamic obstacle environments

| Parameter | Value |
|-----------|-------|
| Goals | 1 |
| Goal Distance | 3.0 - 8.0 m |
| Static Obstacles | 5 |
| Dynamic Obstacles | 8 |
| Collision Penalty | -300 |
| Downgrade Condition | SR < 20% OR CR > 50% |

**Rationale**: Final deployment condition with single target and many dynamic obstacles; strictest collision penalty enforces safe behavior; downgrade allowed prevents getting stuck.

---

## Training Configuration

### Observation Space (139D)

| Module | Dims | Description |
|--------|------|-------------|
| Ego State | 4 | [v_x, v_y, cos(θ), sin(θ)] |
| Goal Command | 2 | [Δx, Δy] to goal |
| LiDAR (VLP-16) | 72 | 72-beam distance scan |
| Obstacles | 60 | Top-10 objects × 6D (x,y,vx,vy,r,m) |
| Time | 1 | Episode progress |

### Action Space

- **MultiDiscrete([19, 19])**: Linear velocity × Angular velocity
- Linear velocity: 19 levels (-1.0 ~ +1.0 m/s)
- Angular velocity: 19 levels (-0.25π ~ +0.25π rad/s)

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rollouts | 128 |
| Learning Epochs | 6 |
| Mini Batches | 8 |
| Discount Factor | 0.995 |
| Learning Rate | 1e-4 (KL Adaptive: 3e-5 ~ 1.5e-4) |
| Entropy Scale | 0.01 |
| Gradient Clip | 1.0 |
| Ratio Clip | 0.2 |

---

## Quick Start

### Training Command

```bash
# Curriculum learning training (starts from Phase 1 automatically)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Charge-Navigation-Curriculum-v0 \
    --num_envs 256 \
    --headless
```

### Resume from Checkpoint

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Charge-Navigation-Curriculum-v0 \
    --checkpoint logs/charge_vlp16/checkpoints/best_agent.pt \
    --num_envs 256 \
    --headless
```

---

## Curriculum Transition Logic

### Upgrade Conditions

```python
# Phase 1 → Phase 2
if SR > 0.72:
    upgrade()

# Phase 2 → Phase 3
if SR > 0.72 AND CR < 0.30:
    upgrade()

# Phase 3 → Phase 4
if SR > 0.80 AND CR < 0.20:
    upgrade()
```

### Downgrade Conditions

```python
# Any Phase (except Phase 1)
if SR < 0.25 OR CR > threshold:
    downgrade()
```

---

## Design Principles

### 1. Avoid Reward Conflict

- Phase 1/2: Teach navigation only, light collision penalty (-30, -50)
- Phase 3/4: Significantly increase collision penalty (-200, -300)

### 2. Prevent Overfitting

- Phase 1 upgrade threshold at 72% (not 100%) to avoid getting stuck in easy scenarios
- Mixed training ensures capabilities don't degrade across phases

### 3. Dynamic Reward Adjustment

```python
# Curriculum automatically adjusts reward weights
_apply_stage(env, stage)
rm.set_term_cfg("collision_terminal", weight=stage_cfg["collision_terminal_weight"])
```

---

## Performance Targets

| Phase | Success Rate Target | Collision Rate Limit |
|-------|---------------------|----------------------|
| Phase 1 | > 72% | Unlimited |
| Phase 2 | > 72% | < 30% |
| Phase 3 | > 80% | < 20% |
| Phase 4 | > 50% | < 20% |

---

## File Structure

```
charge_skrl/
├── cfg/
│   ├── charge_env_cfg_vlp16.py           # VLP16 base environment config
│   └── charge_env_cfg_vlp16_curriculum.py # Curriculum environment config
├── curriculum/
│   ├── goal_obstacle_curriculum.py        # Goal-Obstacle coupled curriculum
│   └── mixed_curriculum.py                # Mixed training scheduler
├── agents/
│   └── skrl_ppo_cfg_vlp16.yaml            # PPO hyperparameters
├── mdp/
│   ├── actions/                           # Discrete differential drive
│   ├── observations/                      # LiDAR + object observations
│   ├── rewards/                           # Potential-based rewards
│   ├── terminations/                      # Termination conditions
│   └── events/                            # Obstacle randomization
└── README_CURRICULUM.md
```

---

## References

1. **Curriculum Learning** - Bengio et al., ICML 2009
2. **PPO** - Schulman et al., arXiv 2017
3. **SKRL** - https://github.com/Toni-SM/skrl

---

## License

This project is built on the Isaac Lab framework and follows its license terms.
