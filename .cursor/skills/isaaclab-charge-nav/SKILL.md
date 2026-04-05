---
name: isaaclab-charge-nav
description: Manages IsaacLab Charge navigation RL training tasks including environment configuration, reward engineering, PPO training, phase progression, and modular MDP components. Use when working with charge navigation environments, modifying training phases, adjusting reward functions, or debugging RL training issues in the IsaacLab framework.
---

# IsaacLab Charge Navigation

## Quick Reference

**Project Path**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge`

**Key Files**:
- `cfg/charge_env_cfg.py` - Environment configurations (Phase 1/2/3)
- `agents/rsl_rl_ppo_cfg*.py` - PPO training configurations
- `mdp/` - Modular MDP components (actions, observations, rewards, terminations, events)

**Training Phases**:
- Phase 1 (`Isaac-Navigation-Charge-v0`): Basic navigation, no obstacles
- Phase 2 (`Isaac-Navigation-Charge-v1`): Static obstacle avoidance
- Phase 3 (`Isaac-Navigation-Charge-v3`): Goal distance curriculum learning

---

## When to Use This Skill

Use when:
- User asks about charge navigation, IsaacLab, or this specific RL task
- Modifying environment configurations or reward functions
- Debugging training issues (snake behavior, convergence problems)
- Implementing phase transitions or curriculum learning
- Working with MDP components (observations, rewards, actions)
- Adjusting PPO hyperparameters or training setup

---

## Project Architecture

### Directory Structure

```
charge/
├── cfg/                    # Environment configs
│   ├── charge_env_cfg.py     # Phase 1 config
│   ├── charge_env_cfg_v2.py  # Phase 2 config (obstacles)
│   ├── charge_env_cfg_v3.py  # Phase 3 config (curriculum)
│   └── charge_cfg.py         # Robot physical model
├── agents/                  # RL algorithm configs
│   ├── rsl_rl_ppo_cfg.py    # Base PPO config
│   ├── rsl_rl_ppo_cfg_v2.py # Phase 2 PPO
│   └── rsl_rl_ppo_cfg_v3.py # Phase 3 PPO (resume from v2)
├── mdp/                    # Modular MDP components
│   ├── actions/              # Action space (differential_drive)
│   ├── observations/         # Observation space (lidar, goal, velocity)
│   ├── rewards/             # Reward functions (goal, safety, motion)
│   ├── terminations/        # Termination conditions
│   ├── events/             # Event handling (reset, curriculum)
│   └── core/              # State management, types
├── goal_command.py          # Goal position generator
└── test/                  # Test scripts
```

### Environment Registration

Environments are registered in `__init__.py`:
- `Isaac-Navigation-Charge-v0` (training)
- `Isaac-Navigation-Charge-Play-v0` (playback)

---

## MDP Component Architecture

### Import Pattern

Always use modular imports from mdp subdirectories:

```python
# Actions
from ..mdp.actions import DifferentialDriveActionCfg

# Observations
from ..mdp.observations import (
    lidar_scan_2d_sweep,
    base_velocity_xy,
    goal_position_in_robot_frame,
    goal_distance,
)

# Rewards
from ..mdp.rewards import (
    progress_to_goal,
    reaching_goal,
    progressive_collision_penalty,
)

# Terminations
from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    collision_occurred,
)

# Events
from ..mdp.events import reset_obstacles, reset_root_state_fixed_per_env
```

**NEVER** import from old `charge_mdp.py` - it's deprecated.

---

## Training Commands

### Phase 1 Training (No obstacles)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 128 \
    --headless
```

### Phase 2 Training (Static obstacles)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v1 \
    --num_envs 128 \
    --headless
```

### Phase 3 Training (Curriculum, resume from Phase 2)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v3 \
    --num_envs 128 \
    --resume \
    --load_run "/path/to/phase2/run"
```

**IMPORTANT**: `--resume` flag is required in command line, even if set in config.

### Playback

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Navigation-Charge-Play-v3 \
    --num_envs 3
```

### Testing

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10 \
    --headless
```

---

## Common Tasks

### Modifying Reward Weights

Location: `cfg/charge_env_cfg.py` → `RewardsCfg` class

**Key Principle**: Maintain goal:avoidance ratio ≥ 25:1 to prevent snake behavior

```python
@configclass
class RewardsCfg:
    # Core rewards (goal-directed)
    reaching_goal = RewTerm(
        func=reaching_goal,
        weight=500.0,  # Decisive reward
    )
    
    distance_to_goal = RewTerm(
        func=progress_to_goal,
        weight=5.0,  # Guidance reward
    )
    
    # Safety rewards (keep low weight)
    progressive_collision = RewTerm(
        func=progressive_collision_penalty,
        weight=-1.5,  # Moderate penalty
    )
    
    safe_navigation = RewTerm(
        func=safe_navigation_bonus,
        weight=0.2,  # Keep low
    )
```

**Reward Ratio Rules**:
- Phase 1: 50:1 (no obstacles needed)
- Phase 2: 25:1 ~ 50:1 (balance safety and efficiency)
- Phase 3: 50:1+ (focus on pure goal seeking)

### Adjusting PPO Hyperparameters

Location: `agents/rsl_rl_ppo_cfg.py`

**Common Adjustments**:

```python
@configclass
class ChargeNavigationPPORunnerCfg:
    # Phase 3: Tighten for stability
    clip_param = 0.1        # Down from 0.2 (lock good policy)
    entropy_coef = 0.001      # Down from 0.01 (reduce exploration)
    
    # Learning rate
    learning_rate = 1.0e-4    # Standard
    # learning_rate = 5.0e-5    # Slower, more stable
```

**When to adjust**:
- **clip_param**: If policy oscillates or degrades in late training
- **entropy_coef**: Stop exploration when success rate plateaus
- **learning_rate**: If convergence is too fast (unstable) or too slow

### Adding New Observations

1. Implement function in `mdp/observations/functions.py`:

```python
def new_observation(env: EnvType) -> torch.Tensor:
    """Calculate new observation"""
    # Your implementation
    return obs_tensor
```

2. Export in `mdp/observations/__init__.py`:

```python
from .functions import new_observation
__all__ = ["new_observation", ...]
```

3. Add to config:

```python
@configclass
class ObservationsCfg:
    policy = ObsGroup(
        new_observation=ObsTerm(func=new_observation),
        # ... other observations
    )
```

**Critical**: Observation dimensions must be consistent across phases for weight transfer.

### Phase Transitions

**Phase 2 → Phase 3** (Weight Transfer):

1. Observation space must be identical (or padded to same dimensions)
2. Action space must be identical
3. Network architecture must be identical

In Phase 3 config:
```python
@configclass
class ChargeNavigationPPORunnerCfgPhase3(ChargeNavigationPPORunnerCfg):
    experiment_name = "charge_navigation_phase3"
    resume = True  # Required
    load_checkpoint = "model_.*\\.pt$"  # Load latest model
```

Command must include `--resume` and `--load_run`.

---

## Debugging Common Issues

### Snake Behavior (Oscillating Motion)

**Symptom**: Agent oscillates left/right when approaching obstacles

**Root Cause**: Reward conflict - goal:avoidance ratio too low

**Check**:
```python
# In cfg/charge_env_cfg.py
goal_weight = 5.0
avoidance_weight = 0.4
ratio = 5.0 / 0.4  # = 12.5:1 (TOO LOW!)
```

**Solution**: Increase goal weight or decrease avoidance weight:
```python
goal_weight = 5.0  # Increase or keep
avoidance_weight = 0.2  # Decrease
ratio = 5.0 / 0.2  # = 25:1 (GOOD)
```

### Import Errors

**Error**: `ImportError: cannot import name 'charge_mdp'`

**Cause**: Using deprecated import path

**Fix**:
```python
# ❌ OLD (deprecated)
from ..charge_mdp import progress_to_goal

# ✅ NEW (modular)
from ..mdp.rewards import progress_to_goal
```

### Observation Dimension Mismatch

**Error**: `RuntimeError: shape mismatch` or PPO std=0 error

**Cause**: Observation dimensions changed between phases

**Check**:
```python
# Count observation dimensions
obs_dim = env.observation_space.shape[0]
print(f"Observation dim: {obs_dim}")
```

**Fix**: Use fixed dimensions with padding:
```python
MAX_OBSTACLES = 10  # Fixed max
# Pad to MAX_OBSTACLES even if fewer obstacles exist
```

### Resume Not Working

**Error**: Training starts from scratch instead of loading checkpoint

**Check**:
1. Command line includes `--resume`?
2. `--load_run` path is correct?
3. `load_checkpoint` pattern matches files?

**Fix**:
```bash
# Must include --resume in command line
./isaaclab.sh -p scripts/.../train.py \
    --task Isaac-Navigation-Charge-v3 \
    --resume \
    --load_run "/path/to/phase2/run"
```

---

## Key Concepts

### Differential Drive Action

**Input**: `[linear_speed, angular_speed]` range [-1, 1]

**Mapping**:
- `linear_speed`: -1.0 → -1.5 m/s (backward), +1.0 → +1.5 m/s (forward)
- `angular_speed`: -1.0 → -1.5 rad/s (left), +1.0 → +1.5 rad/s (right)

**Features**:
- Dynamic acceleration limits based on goal distance
- Velocity smoothing
- Debug visualization (arrows for target/actual velocity)

### LiDAR Observation

**Type**: Multi-layer horizontal sweep

**Dimensions**: 120 total (3 layers × 40 rays/layer)

**Layers**:
- Bottom: -10° vertical (ground clearance)
- Middle: 0° vertical (horizontal plane)
- Top: +10° vertical (obstacle height)

**Range**: 0-10 meters

### Reward Gating

**Purpose**: Prevent reward for invalid states (e.g., tipped robot)

**Implementation**:
```python
def gated_reward(env: EnvType) -> torch.Tensor:
    is_upright = check_upright(env)
    base_reward = calculate_reward(env)
    
    return torch.where(
        is_upright,
        base_reward,           # Upright: give reward
        torch.zeros_like(base_reward)  # Tipped: no reward
    )
```

### Curriculum Learning (Phase 3)

**Mechanism**: Auto-adjust goal distance based on success rate

**Logic**:
```python
if success_rate > 80%:
    goal_distance += 0.5  # Increase difficulty
elif success_rate < 50%:
    goal_distance -= 0.5  # Decrease difficulty
# Otherwise maintain
```

**Range**: 2m to 15m

**Update frequency**: Every 100 episodes

---

## Reward Engineering Guidelines

### Design Principles

1. **Dense + Sparse**: Combine guidance rewards with goal rewards
2. **Weight Balance**: Avoid conflicting rewards (snake behavior)
3. **Gating**: Disable rewards for invalid states
4. **Shape Matters**: Use appropriate functions (linear, quadratic, inverse)

### Common Reward Shapes

**Linear**: Good for simple, uniform effects
```python
reward = distance_delta * weight
```

**Quadratic**: Amplifies effects for extreme values
```python
penalty = (distance / max_distance) ** 2 * weight
```

**Inverse**: Steep penalty near threshold
```python
penalty = 1.0 + (threshold / distance - 1.0)
penalty = clamp(penalty, 1.0, 5.0)
```

### Directional Penalty (Advanced)

**Purpose**: Only penalize moving TOWARD obstacle, not sideways

```python
# Calculate velocity toward obstacle
vel_toward_obstacle = dot(robot_velocity, obstacle_direction)

# Apply multiplier based on speed ratio
speed_ratio = abs(vel_toward_obstacle) / max_speed
multiplier = 1.0 + 2.0 * (speed_ratio ** 2)  # [1.0, 3.0]

penalty = base_penalty * multiplier
```

---

## Testing and Validation

### Run Tests

```bash
# Import test
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/test_import_simple.py

# Modularization test
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/modularization_verification_test.py

# Training test (quick)
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10
```

### Validation Checklist

Before modifying code:
- [ ] Understand the training phase and its purpose
- [ ] Check reward weight ratios (goal:avoidance ≥ 25:1)
- [ ] Verify observation dimensions are consistent
- [ ] Test import paths use modular structure
- [ ] Run quick test before full training

After modifications:
- [ ] Run test script
- [ ] Monitor training metrics (TensorBoard)
- [ ] Check for expected behavior improvement

---

## Additional Resources

- [Project architecture details](reference.md)
- [Reward function reference](reward-reference.md)
- [Observation functions guide](observation-guide.md)
- [Training troubleshooting guide](debugging-guide.md)
