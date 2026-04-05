# IsaacLab Charge Navigation - Architecture Reference

## Project Overview

**Location**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge`

**Framework**: Isaac Lab (based on NVIDIA Isaac Sim)
**Task**: Point-to-point navigation with obstacle avoidance for differential-drive robot
**Algorithm**: PPO (Proximal Policy Optimization) with RSL-RL implementation

## Core Components

### 1. Environment Configuration (`cfg/`)

| File | Purpose | Phase |
|-------|-----------|--------|
| `charge_env_cfg.py` | Base environment configuration | Phase 1 |
| `charge_env_cfg_v2.py` | Static obstacle environment | Phase 2 |
| `charge_env_cfg_v3.py` | Goal distance curriculum | Phase 3 |
| `charge_cfg.py` | Robot physical model (URDF) | All |
| `charge_env.py` | Custom environment class with observation checks | All |

**Configuration Classes**:

```python
@configclass
class ChargeNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Main configuration class"""
    
    # Scene
    scene: MySceneCfg = MySceneCfg()
    
    # Simulation settings
    decimation: int = 4              # Physics steps per decision
    episode_length_s: float = 20.0     # Episode duration
    
    # Action space
    action: DifferentialDriveActionCfg = DifferentialDriveActionCfg()
    
    # Observation space
    observations: ObservationsCfg = ObservationsCfg()
    
    # Reward functions
    rewards: RewardsCfg = RewardsCfg()
    
    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Events
    events: EventCfg = EventCfg()
```

### 2. Agent Configuration (`agents/`)

| File | Purpose |
|------|---------|
| `rsl_rl_ppo_cfg.py` | Base PPO configuration |
| `rsl_rl_ppo_cfg_v2.py` | Phase 2 PPO config |
| `rsl_rl_ppo_cfg_v3.py` | Phase 3 PPO config (resume from v2) |

**Key PPO Hyperparameters**:

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `learning_rate` | 1.0e-4 | Step size for policy update |
| `clip_param` | 0.1-0.2 | Policy clipping range |
| `entropy_coef` | 0.001-0.01 | Exploration coefficient |
| `gamma` | 0.99 | Discount factor |
| `horizon_length` | 24 | Steps per PPO update |

### 3. MDP Components (`mdp/`)

Modular structure - each component in separate subdirectory:

```
mdp/
├── actions/              # Action space
│   ├── differential_drive.py
├── observations/         # Observation functions
│   ├── functions.py      # Main observation implementations
│   └── utils.py         # Observation utilities
├── rewards/             # Reward functions
│   ├── goal_rewards.py     # Goal-related rewards
│   ├── safety_rewards.py   # Safety/obstacle rewards
│   └── motion_rewards.py   # Movement rewards
├── terminations/        # Termination conditions
│   ├── goal.py           # Goal reached
│   ├── collision.py      # Collision detection
│   ├── robot_state.py    # Robot state checks
│   └── timeout.py        # Episode timeout
├── events/              # Event handlers
│   ├── curriculum.py     # Curriculum learning
│   ├── obstacles.py      # Obstacle management
│   └── reset.py          # Reset logic
└── core/               # Core utilities
    ├── state.py          # State management
    └── types.py          # Type definitions
```

## Environment Constants

**Location**: `cfg/charge_env_cfg.py` (lines 97-124)

```python
# Robot parameters
ROBOT_BODY_RADIUS = 0.5              # Robot body radius (meters)

# Goal parameters
GOAL_REACH_THRESHOLD = ROBOT_BODY_RADIUS  # Goal distance threshold (meters)
COLLISION_THRESHOLD = ROBOT_BODY_RADIUS   # Collision distance threshold (meters)

# Safety distances
SAFE_DISTANCE = 1.5                   # Safe distance (meters)
DANGER_DISTANCE = 0.8                  # Danger distance (meters)
COLLISION_DISTANCE = 0.4               # Collision distance (meters)

# Obstacle reset parameters
MIN_ROBOT_DISTANCE = 1.5              # Min robot-obstacle distance
MIN_GOAL_DISTANCE = 1.0               # Min goal-obstacle distance
MIN_OBSTACLE_SPACING = 1.0           # Min obstacle-obstacle spacing

# Observation dimensions
MAX_OBSTACLES = 10                   # Max obstacles (for fixed obs dims)
```

**Usage**: These constants are used throughout the codebase. Change here, not hardcoded values.

## Observation Space

**Total Dimensions**: 131 (Phase 1/2) to 134 (Phase 2.5)

| Observation | Dims | Description |
|------------|-------|-------------|
| **LiDAR scan** | 120 | 3 layers × 40 rays/laser (distance readings) |
| **Base velocity** | 3 | [vx, vy, vz] in world frame |
| **Goal position (robot frame)** | 2 | [dx, dy] relative to robot |
| **Goal distance** | 1 | Euclidean distance to goal |
| **Time remaining** | 1 | (1 - elapsed/episode_length) |
| **Alive flag** | 1 | 1=alive, 0=terminated |
| **Obstacle state** | 3 | [count, avg_size, min_distance] |

### LiDAR Configuration

**Sensor**: Multi-layer horizontal sweep LiDAR

**Layers**: 3 vertical layers
- Bottom: -10° elevation (ground clearance)
- Middle: 0° elevation (horizontal plane)
- Top: +10° elevation (obstacle height)

**Rays**: 40 rays/layer = 120 total rays

**Range**: 0-10 meters

**Pattern**: Horizontal FOV: 360° (±180°), Resolution: 9°

## Action Space

**Type**: Differential Drive

**Dimensions**: 2
```python
action = [linear_speed, angular_speed]
         # [-1, 1]         [-1, 1]
         # forward/backward    left/right turn
```

**Mapping to physical values**:
- Linear: [-1, 1] → [-1.5, 1.5] m/s
- Angular: [-1, 1] → [-1.5, 1.5] rad/s

**Features**:
1. Dynamic acceleration limits (based on goal distance)
2. Velocity smoothing (prevents sudden changes)
3. Coordinate transformation (robot → world frame)
4. Debug visualization (arrows for target/actual velocity)

## Reward System

### Reward Hierarchy

```python
# Primary (Goal-Directed)
├── reaching_goal: +500.0 (decisive, sparse)
├── distance_to_goal: +5.0 (guidance, dense)
└── velocity_toward_goal: +3.0 (efficiency)

# Secondary (Safety)
├── progressive_collision: -1.0 to -2.5 (gradient penalty)
├── safe_navigation: +0.1 to +0.4 (comfort bonus)
└── speed_control: +0.1 to +0.4 (speed regulation)

# Tertiary (Motion)
├── move_reward: +0.2 (encourages movement)
└── alignment_reward: +2.5 (orientation incentive)

# Penalties
├── tipped_over: -200.0 (extreme behavior)
└── wall_collision: -1.0 (boundary penalty)
```

### Weight Balance Rules

**Critical**: Maintain goal:avoidance ratio ≥ 25:1 to prevent snake behavior

| Phase | Goal Weight | Avoidance Weight | Ratio | Behavior |
|-------|-------------|------------------|--------|----------|
| Phase 1 | 5.0 | 0.1 | 50:1 | No snake (no obstacles) |
| Phase 2 | 5.0 | 0.2 | 25:1 | Good balance |
| Phase 2.5 | 3.0 | 0.4 | 7.5:1 | ❌ Snake behavior |
| Phase 3 | 5.0 | 0.0 | ∞:1 | Pure goal seeking |

## Termination Conditions

| Condition | Trigger | Handling |
|-----------|---------|----------|
| `goal_reached` | Distance < 0.5m | Success (reward +500) |
| `collision_occurred` | LiDAR < 0.3m | Failure (episode end) |
| `robot_tipped_over` | Z-tilt > 45° | Failure (episode end) |
| `robot_flying` | Height > 0.5m | Failure (episode end) |
| `timeout` | Time > 20s | Failure (no reward) |

## Event System

### Reset Events

```python
# Obstacle reset (on episode start)
reset_obstacles(env, env_ids)
    - Randomize obstacle positions
    - Ensure minimum spacing
    - Avoid goal/robot overlap

# Root state reset (per environment)
reset_root_state_fixed_per_env(env, env_ids)
    - Set robot position to fixed location
    - Reset velocity to zero
    - Randomize robot orientation
```

### Curriculum Events (Phase 3)

```python
# Goal distance curriculum (adaptive difficulty)
update_goal_distance_curriculum(env, env_ids)
    - Monitor recent success rate (100 episodes)
    - If success_rate > 80%: Increase distance
    - If success_rate < 50%: Decrease distance
    - Else: Maintain current level
    
    # Distance range: 2m to 15m
    # Update frequency: Every 100 episodes
```

## Training Phases

### Phase 1: Basic Navigation

**Environment ID**: `Isaac-Navigation-Charge-v0`

**Characteristics**:
- No obstacles
- Goal distance: 2-8m
- Focus: Differential drive control
- Challenge: Straight-line navigation

**Success Metrics**:
- Learn to move forward
- Align with goal direction
- Basic obstacle avoidance (walls only)

### Phase 2: Static Obstacle Avoidance

**Environment ID**: `Isaac-Navigation-Charge-v1`

**Characteristics**:
- 5-10 static obstacles
- Obstacle sizes: 0.2-0.5m radius
- Goal distance: 2-8m
- Challenge: Navigate to goal while avoiding obstacles

**Key Issue**: Snake behavior (oscillation near obstacles)
- **Root Cause**: Reward conflict (goal:avoidance ratio dropped from 50:1 to 7.5:1)
- **Solution**: Restore goal:avoidance ≥ 25:1 ratio

**Success Metrics**:
- Goal reach rate > 75%
- Collision rate < 20%
- Path efficiency < 1.2

### Phase 3: Goal Distance Curriculum

**Environment ID**: `Isaac-Navigation-Charge-v3`

**Characteristics**:
- No obstacles (focus on pure goal seeking)
- Goal distance: 2-15m (adaptive)
- Challenge: Long-distance navigation

**Goal**: Build robust "ballistic navigation" capability

**Curriculum Logic**:
```python
if success_rate > 80%:
    goal_distance = min(goal_distance + 0.5, 15.0)  # Upgrade
elif success_rate < 50%:
    goal_distance = max(goal_distance - 0.5, 2.0)  # Downgrade
else:
    pass  # Maintain
```

**Success Metrics**:
- Reach goal from 15m at > 90% success rate
- Strong goal-seeking behavior
- Prepare for Phase 4 (dynamic obstacles)

## Import Patterns

### Modular MDP Imports

**Correct** (use for new code):
```python
from ..mdp.actions import DifferentialDriveActionCfg
from ..mdp.observations import lidar_scan_2d_sweep, base_velocity_xy
from ..mdp.rewards import progress_to_goal, reaching_goal
from ..mdp.terminations import goal_reached, robot_tipped_over
from ..mdp.events import reset_obstacles, reset_root_state_fixed_per_env
```

**Incorrect** (deprecated):
```python
from ..charge_mdp import progress_to_goal  # ❌ Don't use
```

### Scene Entity Access

```python
# Get robot asset
robot = env.scene["robot"]

# Get sensor data
lidar_data = env.scene.sensors["lidar"].data.ray_hits_w

# Get command data
goal_pos = env.command_manager.get_command("goal_command")

# Get robot state
pos_w = robot.data.root_pos_w
vel_w = robot.data.root_lin_vel_w
quat_w = robot.data.root_quat_w
```

## Coordinate Systems

### Robot Frame (Local)

- **Origin**: Robot center (base_link)
- **X-axis**: Forward direction
- **Y-axis**: Left side
- **Z-axis**: Up direction

### World Frame (Global)

- **Origin**: (0, 0, 0) at environment origin
- **X-axis**: East direction
- **Y-axis**: North direction
- **Z-axis**: Up direction

### Coordinate Transformations

```python
import isaaclab.utils.math as math_utils

# World → Robot
robot_vec = math_utils.subtract_frame_transforms(
    robot_pos_w, robot_quat_w,
    world_pos_w, torch.zeros_like(robot_quat_w)
)

# Robot → World
world_vec = math_utils.quat_apply(robot_quat_w, robot_vec)
```

## Performance Optimization

### GPU Utilization

- **Batch processing**: All operations use vectorized tensors
- **Shape**: [num_envs, ...] for parallel environments
- **Device**: Use `env.device` for GPU/CPU

### Memory Management

```python
# Avoid unnecessary tensor copies
# ❌ Bad
result = value.clone()
# ✅ Good
result = value

# Use in-place operations when possible
tensor.add_(other_tensor)  # In-place addition
```

### Simulation Efficiency

```python
# Decimation: Control physics/decision ratio
decimation = 4  # 4 physics steps per decision

# Episode length: Balance exploration vs. training time
episode_length_s = 20.0  # 20 seconds per episode
```

## Code Quality Standards

### Type Hints

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def my_function(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Always use type hints for clarity"""
    pass
```

### Documentation

```python
def my_function(env: ManagerBasedRLEnv, param: float) -> torch.Tensor:
    """Function summary
    
    Detailed description of what the function does,
    including algorithm, parameters, and return values.
    
    Args:
        env: Environment instance
        param: Parameter description
        
    Returns:
        shape [num_envs]: Return value description
        
    Design Philosophy:
        Explanation of why this function exists
    """
    pass
```

### Error Handling

```python
# Always handle NaN/Inf
value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)

# Always clamp to valid range
value = torch.clamp(value, min=0.0, max=10.0)

# Use reward term checker
reward = _check_reward_term("function_name", reward, env, raise_on_error=True)
```

## Testing Strategy

### Unit Testing

```python
# Test individual functions
def test_observation_dim():
    env = make("Isaac-Navigation-Charge-v0")
    obs, _ = env.reset()
    assert obs.shape[0] == 131, f"Expected 131, got {obs.shape[0]}"
```

### Integration Testing

```bash
# Run full environment test
cd test
python modularization_verification_test.py

# Run training test (quick validation)
python training_test.py --task Isaac-Navigation-Charge-v0 --num_envs 4 --max_iterations 10
```

## Common Patterns

### Configuration Pattern

```python
@configclass
class MyConfig(BaseConfig):
    """Configuration class with type hints"""
    
    # Always use SceneEntityCfg for scene references
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")
    
    # Use clear, descriptive names
    my_parameter: float = 1.0  # With unit comment
```

### Reward Term Pattern

```python
@configclass
class RewardsCfg:
    """Always use RewTerm for rewards"""
    
    my_reward = RewTerm(
        func=my_reward_function,
        weight=1.0,  # Always specify weight
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # ... other parameters
        }
    )
```

### Observation Term Pattern

```python
@configclass
class ObservationsCfg:
    """Always use ObsTerm for observations"""
    
    my_observation = ObsTerm(
        func=my_observation_function,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
```

## Debugging Tools

### Visualization

```python
# Enable debug visualization in differential drive
# This shows velocity arrows in Isaac Sim
visualization_markers = [
    GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/visual/arrow/target"),
    BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/visual/arrow/actual"),
]
```

### Logging

```python
# Print debug information
print(f"Goal distance: {goal_dist.mean().item():.2f}m")
print(f"Reward breakdown: {reward_dict}")

# Check tensor shapes
print(f"Observation shape: {obs.shape}")
```

### TensorBoard

```bash
# Monitor training
tensorboard --logdir logs/rsl_rl/charge_navigation_phase3/

# Key metrics:
# - Train/mean_reward
# - Train/success_rate
# - Train/collision_rate
# - Policy/entropy (exploration)
# - Loss/value_function
```

## File Modification Checklist

Before modifying any file:

- [ ] Understand the current implementation
- [ ] Identify all dependencies
- [ ] Check impact on other phases
- [ ] Verify observation dimensions (if changing)
- [ ] Run tests after modification

After modification:

- [ ] Run modularization verification test
- [ ] Check for import errors
- [ ] Monitor training metrics
- [ ] Update documentation if needed

## Version Control

### Commit Message Format

```
<type>(<scope>): <subject>

<body>
Detailed description of changes

- Change 1
- Change 2

</body>

<footer>
Resolves #issue
```

### Branch Naming

- `feature/<feature-name>`: New features
- `fix/<issue-description>`: Bug fixes
- `refactor/<component>`: Code restructuring
- `phase<phase-number>`: Phase-specific changes

## Troubleshooting Quick Reference

### Common Errors

| Error | Cause | Solution |
|--------|--------|----------|
| `ImportError: cannot import 'charge_mdp'` | Using old import path | Use `from ..mdp.rewards import ...` |
| `RuntimeWarning: invalid value encountered in divide` | Division by zero | Add epsilon: `+ 1e-6` |
| `RuntimeError: shape mismatch` | Observation dimension changed | Ensure consistent dimensions or use padding |
| `PPO std >= 0` | Reward terms have zero variance | Check observation normalization, use `_check_reward_term` |

### Performance Issues

| Symptom | Cause | Fix |
|----------|--------|------|
| Snake behavior | Reward conflict (goal:avoidance ratio too low) | Increase goal weight or decrease avoidance weight |
| Slow learning | Entropy too high or learning rate too low | Adjust `entropy_coef` or `learning_rate` |
| Policy degradation | Clipping too loose | Reduce `clip_param` |
| Overfitting | Not enough exploration | Increase `entropy_coef` |
