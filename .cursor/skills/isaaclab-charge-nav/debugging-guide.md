# Debugging Guide

Complete troubleshooting guide for IsaacLab Charge Navigation training.

## Quick Diagnostic Checklist

When training fails or behaves unexpectedly, check these items:

```python
✅ Environment imports correct?
✅ Observation dimensions match?
✅ Reward terms return valid values?
✅ PPO hyperparameters reasonable?
✅ Episode length appropriate?
✅ Terminal conditions working?
✅ Isaac Sim version compatible?
```

---

## Common Training Issues

### 1. PPO StdDev = 0 Error

**Error Message**:
```
RuntimeError: The standard deviation of the action distribution is zero.
This can happen when the action distribution is too deterministic.
```

**Root Cause**: Observation has zero variance (all same value)

**Diagnosis Steps**:
```python
# 1. Check observation values
obs, _ = env.reset()
print(f"Obs range: [{obs.min()}, {obs.max()}]")
print(f"Obs std: {obs.std(dim=0)}")

# 2. Check each observation component
print(f"LiDAR mean: {obs[:, :120].mean()}, std: {obs[:, :120].std(dim=0)}")
print(f"Goal position: {obs[:, 120:122]}")
```

**Common Causes**:

| Cause | Symptom | Fix |
|--------|----------|-----|
| LiDAR returns all same distance | obs[0:120].std() ≈ 0 | Check sensor configuration, add randomness |
| Goal position always [0, 0] | Check goal_command.py | Fix randomization |
| Observation not normalized | Large variance, gradient issues | Add normalization to [0, 1] |

**Solutions**:

```python
# Solution 1: Add noise to LiDAR
@configclass
class RewardsCfg:
    # Add noise term to encourage exploration
    # (if appropriate for your task)
    pass

# Solution 2: Check reward term checker
# In mdp/rewards/utils.py
def _check_reward_term(name, reward, env, raise_on_error=True):
    # Always check for NaN/Inf
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    if raise_on_error and not torch.all(torch.isfinite(reward)):
        raise ValueError(f"Reward {name} contains NaN/Inf: {reward}")
    
    return reward

# Solution 3: Verify observation normalization
# In observation functions, ensure output is in [0, 1]
normalized = torch.clamp(raw_value / max_range, 0.0, 1.0)
```

---

### 2. Snake Behavior (Oscillation)

**Symptoms**:
- Agent oscillates left/right near obstacles
- Path shows sinusoidal pattern
- Progress slows down significantly

**Root Cause Analysis**:

```python
# Calculate goal:avoidance ratio
goal_weight = 5.0  # distance_to_goal
avoidance_weight = 0.4  # progressive_collision
ratio = goal_weight / avoidance_weight  # = 12.5:1

if ratio < 25:
    print(f"❌ Snake behavior expected: ratio too low ({ratio:.2f}:1)")
else:
    print(f"✅ Ratio healthy: {ratio:.2f}:1")
```

**Phase 2 vs Phase 2.5 Comparison**:

| Config | Goal Weight | Avoidance Weight | Ratio | Behavior |
|--------|-------------|------------------|--------|
| Phase 2 | 5.0 | 0.1 | 50:1 | ✅ No snake |
| Phase 2.5 (v2.5.1) | 3.0 | 0.4 | 7.5:1 | ❌ Snake |

**Fix Strategy 1: Increase Goal Weight**

```python
# In cfg/charge_env_cfg.py
distance_to_goal = RewTerm(
    func=progress_to_goal,
    weight=7.0,  # Increased from 3.0
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

**Fix Strategy 2: Decrease Avoidance Weight**

```python
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-1.0,  # Decreased from -2.5
    params={
        "sensor_cfg": SceneEntityCfg("lidar"),
        "safe_distance": 1.5,
        "danger_distance": 0.8,
        "collision_distance": 0.6,
    }
)
```

**Fix Strategy 3: Use Reward Gating**

```python
def velocity_toward_goal_gated(env, asset_cfg, sensor_cfg):
    """Only reward velocity when not too close to obstacles"""
    
    # Get base reward
    base_reward = velocity_toward_goal(env, asset_cfg)
    
    # Get obstacle distance
    lidar_data = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
    min_dist = torch.min(torch.norm(lidar_data[:, :, :2] - sensor_pos_2d, dim=-1), dim=1)[0]
    
    # Gate: Reduce reward when close to obstacle
    gated_weight = torch.where(
        min_dist > 2.0,  # Far from obstacle
        1.0,             # Full reward
        0.3              # Reduced reward
    )
    
    return base_reward * gated_weight
```

---

### 3. Agent Freezes (Stops Moving)

**Symptoms**:
- Agent velocity becomes zero
- Stands still while far from goal
- Success rate drops to 0%

**Root Cause**: Fear dominates motivation

**Diagnosis**:

```python
# Check collision penalty vs. movement rewards
collision_penalty_weight = -2.5
move_reward_weight = 0.2

if abs(collision_penalty) > abs(move_reward) * 10:
    print("❌ Collision fear too high - agent freezes")
```

**Solutions**:

```python
# Solution 1: Add move reward
move_reward = RewTerm(
    func=lambda env: torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1),
    weight=0.5,  # Encourages movement
)

# Solution 2: Reduce collision penalty
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-1.0,  # Reduced from -2.5
)

# Solution 3: Enable directional penalty (allows safe movement)
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-1.5,
    params={
        "use_directional_penalty": True,  # Only penalize moving TOWARD obstacle
    }
)
```

---

### 4. Agent Rams Obstacles

**Symptoms**:
- High collision rate (> 50%)
- Agent drives directly into obstacles
- No attempt to avoid

**Root Cause**: Collision penalty too weak OR directional penalty disabled

**Diagnosis**:

```python
# Check if directional penalty is enabled
# In reward function
print(f"Directional penalty: {use_directional_penalty}")

# Monitor collision rate
collision_rate = env episode_statistics["collision_occurred"].mean()
print(f"Collision rate: {collision_rate:.2%}")
```

**Solutions**:

```python
# Solution 1: Increase collision penalty
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-2.5,  # Increased
)

# Solution 2: Use nonlinear penalty
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-1.5,
    params={
        "use_nonlinear_penalty": True,  # Makes danger zone "very uncomfortable"
    }
)

# Solution 3: Add safe navigation bonus
safe_navigation = RewTerm(
    func=safe_navigation_bonus,
    weight=0.4,  # Bonus for staying away
)
```

---

### 5. Observation Dimension Mismatch

**Error Message**:
```
RuntimeError: observation space has wrong dimension
Expected: 131, Got: 134
```

**Root Cause**: Phase transition with changed observation space

**Diagnosis**:

```python
# Check current observation dimensions
env = gym.make("Isaac-Navigation-Charge-v2")
print(f"Obs space: {env.observation_space.shape}")

# Count dimensions manually
def count_obs_dims(env):
    obs, _ = env.reset()
    total = 0
    for obs_term in env.observations_cfg.policy:
        if hasattr(obs_term, 'func'):
            func_result = obs_term.func(env)
            if isinstance(func_result, torch.Tensor):
                total += func_result.shape[0] if len(func_result.shape) == 1 else func_result.shape[1]
    return total

print(f"Expected dims: {count_obs_dims(env)}")
```

**Fix Strategy 1: Maintain Fixed Dimensions**

```python
# In all phases, use MAX_OBSTACLES = 10
# Pad obstacle observations to fixed size
if num_obstacles < 10:
    obstacle_obs = torch.cat([
        actual_obstacle_data,
        torch.zeros(num_envs, 10 - num_obstacles, device=device)
    ], dim=1)
```

**Fix Strategy 2: Resume from Compatible Checkpoint**

```bash
# Only resume if obs dims match
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v2 \
    --num_envs 128 \
    --resume \
    --load_run "logs/.../phase2"  # Load only if compatible
```

---

### 6. Reward Contains NaN/Inf

**Symptoms**:
- Loss becomes NaN or Inf
- Training crashes
- Mean reward becomes meaningless

**Root Cause**: Reward calculation has mathematical error

**Diagnosis**:

```python
# Enable strict checking
# In training script or env class
for step in range(100):
    obs, rew, done, info = env.step(actions)
    
    # Check for invalid rewards
    if not torch.all(torch.isfinite(rew)):
        print(f"Step {step}: Invalid reward detected!")
        print(f"  Value: {rew}")
        print(f"  Shape: {rew.shape}")
        print(f"  Min: {rew.min()}, Max: {rew.max()}")
        break
```

**Common Issues**:

| Issue | Cause | Fix |
|--------|--------|-----|
| Division by zero | min_dist = 0 in velocity calculation | Add epsilon: `+ 1e-6` |
| Log of negative | `reward = log(distance)` with distance ≤ 0 | Use `log(distance + 1e-6)` |
| Exp overflow | `reward = exp(large_value)` | Clamp: `exp(min(value, 10))` |
| Index out of bounds | Array access with invalid index | Validate indices: `indices = torch.clamp(indices, 0, max)` |

**Solution Pattern**:

```python
def safe_division(numerator, denominator, epsilon=1e-6):
    """Safe division that handles zero"""
    return numerator / (denominator + epsilon)

def safe_log(value):
    """Safe log that handles non-positive"""
    return torch.log(torch.clamp(value, min=1e-6))

def safe_exp(value, max_val=10.0):
    """Safe exp that prevents overflow"""
    return torch.exp(torch.clamp(value, min=0.0, max=max_val))
```

---

### 7. Training Not Converging

**Symptoms**:
- Mean reward oscillates or plateaus
- Success rate doesn't improve
- Policy entropy remains high

**Root Cause**: Learning rate, exploration, or hyperparameters mismatched

**Diagnosis**:

```python
# Check learning rate vs. entropy
lr = agent.cfg.learning_rate
entropy = agent.cfg.entropy_coef

print(f"Learning rate: {lr}")
print(f"Entropy coef: {entropy}")
print(f"Expected exploration decay: {entropy * exp(-500)}")  # 500 steps

# Check if exploration is decaying
import tensorboard
# Look at Policy/entropy plot over training
# Should decrease from ~1.0 to <0.1
```

**Solutions**:

```python
# Solution 1: Reduce entropy coef
entropy_coef = 0.001  # From 0.01
# This reduces exploration, focuses on exploitation

# Solution 2: Reduce learning rate
learning_rate = 5.0e-5  # From 1.0e-4
# Slower learning = more stable training

# Solution 3: Tighten clipping
clip_param = 0.1  # From 0.2
# Locks in good policy, prevents degradation
```

---

### 8. Episode Terminates Early

**Symptoms**:
- Average episode length < 5 seconds
- Agent terminates unexpectedly
- Low success rate

**Root Cause**: Terminal conditions too strict

**Diagnosis**:

```python
# Check termination reasons
termination_reasons = {}
for step in range(1000):
    obs, rew, done, info = env.step(actions)
    if done.any():
        reason = detect_termination_reason(env, info)
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

print("Termination statistics:")
for reason, count in termination_reasons.items():
    print(f"  {reason}: {count} episodes")
```

**Common Issues**:

| Issue | Parameter | Fix |
|--------|---------|-----|
| Timeout too short | episode_length_s = 20.0 | Increase to 30.0s |
| Collision threshold too loose | collision_distance = 0.3 | Reduce to 0.2m |
| Tipped angle too strict | tilt_threshold = 45° | Increase to 60° |
| Goal too far | max_goal_distance = 15.0 | Reduce to 8.0m |

---

### 9. Simulation Physics Instability

**Symptoms**:
- Robot flies or falls through ground
- Unrealistic velocities
- Training crashes with physics errors

**Root Cause**: Physics material or mass settings incorrect

**Diagnosis**:

```python
# Check robot properties
robot = env.scene["robot"]
print(f"Robot mass: {robot.root_physx_props.mass}")
print(f"Robot velocity: {robot.data.root_lin_vel_w}")

# Check ground friction
ground = env.scene["ground"]
print(f"Ground friction: {ground.physx_material.static_friction}")
```

**Solutions**:

```python
# In cfg/charge_cfg.py
robot = CHARGE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    # Add mass damping if needed
    # rigid_body_props=sim_utils.RigidBodyPropertiesCfg(
    #     mass=50.0,  # Heavier robot
    # ),
)
```

---

## Performance Debugging

### GPU Memory Issues

**Symptoms**:
- CUDA out of memory error
- Training starts but crashes
- Sluggish performance

**Diagnosis**:

```python
# Check memory usage
import torch
print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Check batch size
num_envs = 128
print(f"Environments: {num_envs}")
```

**Solutions**:

```python
# Solution 1: Reduce environments
num_envs = 64  # From 128

# Solution 2: Reduce horizon length
horizon_length = 16  # From 24

# Solution 3: Reduce observation dimensions
# (if possible)
```

### Training Speed Optimization

**Monitor**:

```bash
# Check steps per second
tensorboard --logdir logs/rsl_rl/charge_phase3/
# Look at: Train/steps_per_second metric
# Target: > 1000 steps/sec
```

**Optimization Strategies**:

1. **Increase decimation** (fewer physics steps per decision)
```python
decimation = 8  # From 4
# Better GPU utilization
```

2. **Vectorize operations** (avoid loops)
```python
# ❌ Bad: Sequential processing
for i in range(num_envs):
    reward[i] = calculate(env, i)

# ✅ Good: Vectorized processing
rewards = calculate_vectorized(env)  # Single operation
```

3. **Reduce observation complexity**
```python
# Use efficient distance calculation
# ❌ Bad: Loop through all rays
for ray in range(360):
    distances[ray] = calculate()

# ✅ Good: Vectorized ray operations
distances = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)
```

---

## Environment-Specific Debugging

### Phase 1 Debugging

**Issue**: Agent doesn't move

```python
# Check if action is being applied
env.action_manager.print_actions()

# Check if robot velocity updates
print(f"Robot velocity: {env.scene['robot'].data.root_lin_vel_w}")

# Check differential drive parameters
print(f"Max linear vel: {env.action_manager.action_terms[0].cfg.max_linear_velocity}")
print(f"Max angular vel: {env.action_manager.action_terms[0].cfg.max_angular_velocity}")
```

### Phase 2 Debugging

**Issue**: Snake behavior

```python
# Monitor individual reward terms
reward_breakdown = env.unwrapped._reward_buffer
print(f"Distance to goal: {reward_breakdown['distance_to_goal'].mean()}")
print(f"Collision penalty: {reward_breakdown['progressive_collision'].mean()}")
print(f"Safe navigation: {reward_breakdown['safe_navigation_bonus'].mean()}")

# Calculate ratio
goal_to_collision = abs(reward_breakdown['distance_to_goal'].mean()) / abs(reward_breakdown['progressive_collision'].mean())
print(f"Goal:Collision ratio: {goal_to_collision:.2f}:1")
```

### Phase 3 Debugging

**Issue**: Curriculum not adjusting

```python
# Check curriculum state
env.unwrapped._curriculum_level
env.unwrapped._curriculum_stats

# Manually trigger update
env.command_manager.get_command("goal_command")._update_curriculum(env_ids=torch.arange(env.num_envs))
```

---

## Isaac Sim Integration

### Headless vs GUI Mode

**Headless** (faster):
```bash
./isaaclab.sh -p scripts/.../train.py \
    --task Isaac-Navigation-Charge-v3 \
    --headless  # No GUI window
```

**GUI** (for debugging):
```bash
./isaaclab.sh -p scripts/.../train.py \
    --task Isaac-Navigation-Charge-v3 \
    --headless False  # Show GUI
```

### Recording Videos

```python
# In environment class
from isaaclab.utils.video import save_video

# Record episode
save_video(env, "episode_000.mp4", camera_prim_paths=["/World/camera"])

# Or use built-in Isaac Sim recording
# In launch script
enable_cameras: true
```

---

## Data Analysis

### Training Metrics Extraction

```python
# Load training logs
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(path="logs/rsl_rl/charge_phase3/events.out.tfevents.")

# Get specific metrics
reward_events = ea.ReloadScalars('Train/mean_reward')
success_events = ea.ReloadScalars('Train/success_rate')
collision_events = ea.ReloadScalars('Train/collision_rate')

# Plot or analyze
import matplotlib.pyplot as plt
plt.plot(reward_events.Scalars('Train/mean_reward'))
plt.title("Training Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.show()
```

### Episode Analysis

```python
# Analyze individual episodes
for episode in range(num_episodes):
    # Get episode data
    obs, rew, done, info = env.get_episode_data(episode)
    
    # Calculate metrics
    episode_length = env.episode_length_buf
    total_reward = rew.sum()
    
    print(f"Episode {episode}: Length={episode_length}, Reward={total_reward:.2f}")
```

---

## Advanced Debugging Techniques

### Gradient Inspection

```python
# Check policy gradients
policy = env.unwrapped._actor_critic.actor
for name, param in policy.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.4f}")
```

### Observation Visualization

```python
# Visualize what agent sees
import matplotlib.pyplot as plt

# Get LiDAR observation
obs, _ = env.reset()
lidar = obs[:120].reshape(72, -1)  # Convert to polar

# Plot
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.plot(angles, lidar)
ax.set_title("LiDAR Observation")
plt.show()
```

### Action Visualization

```python
# Track action distribution over time
action_history = []

for step in range(1000):
    actions = env.action_manager.action
    action_history.append(actions.clone())

# Convert to numpy and analyze
actions_np = torch.stack(action_history).cpu().numpy()

# Plot histogram
plt.hist(actions_np[:, 0], bins=50, alpha=0.5, label='Linear')
plt.hist(actions_np[:, 1], bins=50, alpha=0.5, label='Angular')
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Action Distribution')
plt.show()
```

---

## System-Level Debugging

### Isaac Sim Version Check

```bash
# Verify Isaac Sim version
python -c "from omni.isaac.kit import simulation_app; print(simulation_app.get_version())"

# Check Isaac Lab compatibility
./isaaclab.sh --version
```

### CUDA/PyTorch Check

```python
import torch
import isaaclab.sim as sim_utils

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Best Practices

### Before Debugging

1. **Save current state**: Commit working code
2. **Isolate the problem**: Test one change at a time
3. **Keep a log**: Document what you tried and results
4. **Reproduce consistently**: Ensure issue is not intermittent

### During Debugging

1. **Use small test runs**: `--num_envs 4 --max_iterations 100`
2. **Monitor all metrics**: Not just reward
3. **Check intermediate values**: Print debugging info
4. **Compare with baseline**: Keep working version as reference

### After Fixing

1. **Run full training**: Validate fix at scale
2. **Monitor TensorBoard**: Ensure no regression
3. **Check success rate**: Verify improvement
4. **Document the fix**: Update README/debug guide

---

## Quick Reference

### Common Commands

```bash
# Run quick test
python test/training_test.py --task Isaac-Navigation-Charge-v3 --num_envs 4 --max_iterations 50

# Monitor training
tensorboard --logdir logs/rsl_rl/charge_phase3/ --host localhost --port 6006

# Play with checkpoint
./isaaclab.sh -p scripts/.../play.py \
    --task Isaac-Navigation-Charge-Play-v3 \
    --num_envs 3 \
    --load_run logs/rsl_rl/charge_phase3/<run_name>
```

### File Locations

| Component | Path |
|-----------|-------|
| Config | `cfg/charge_env_cfg.py` |
| Rewards | `mdp/rewards/*.py` |
| Observations | `mdp/observations/*.py` |
| Training logs | `logs/rsl_rl/charge_phase*/` |
| Checkpoints | `logs/rsl_rl/charge_phase*/model_*.pt` |

---

## Getting Help

### Internal Resources

- Check existing documentation in `readme/` directory
- Review Phase design docs
- Look at CHANGELOG.md for recent fixes

### External Resources

- Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab
- PPO implementation: RSL-RL library
- PyTorch documentation: https://pytorch.org/docs

### Reporting Issues

When documenting bugs, include:

1. Environment ID and version
2. Isaac Sim and Isaac Lab versions
3. Minimal reproducible example
4. Expected vs. actual behavior
5. System specifications (GPU, OS, etc.)

---

## Summary Flowchart

```
Training Issue?
    ↓
Yes → Quick Checklist
    ↓
Observation Issue?
    ↓ Yes → Fix normalization/NaNs
    ↓ No
Reward Issue?
    ↓ Yes → Check weights, fix ratio
    ↓ No
PPO Issue?
    ↓ Yes → Adjust hyperparameters
    ↓ No
Physics Issue?
    ↓ Yes → Check robot properties
    ↓ No
Converging?
    ↓ No → Reduce LR, tighten clip
    ↓ Yes
Fixed?
    ↓ Yes → Validate with full training
    ↓ No → Collect more data, repeat diagnosis
```
