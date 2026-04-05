# Reward Function Reference

Complete guide to all reward functions in Charge Navigation.

## Reward Hierarchy

```
Primary Rewards (Goal-Directed)
├── reaching_goal (+500.0)      - Decisive, sparse
├── distance_to_goal (+5.0)     - Guidance, dense
└── velocity_toward_goal (+3.0)    - Efficiency, dense

Secondary Rewards (Safety)
├── progressive_collision (-1.0 ~ -2.5)    - Gradient penalty
├── safe_navigation (+0.1 ~ +0.4)          - Comfort bonus
└── speed_control (+0.1 ~ +0.4)             - Speed regulation

Tertiary Rewards (Motion)
├── move_reward (+0.2)          - Encourage movement
└── alignment_reward (+2.5)         - Orientation incentive

Penalties
├── tipped_over (-200.0)                 - Extreme behavior
└── wall_collision (-1.0)               - Boundary penalty
```

## Goal-Directed Rewards

### 1. Reaching Goal

**Function**: `reaching_goal()`

**File**: `mdp/rewards/goal_rewards.py:302-348`

**Purpose**: Decisive reward for successful goal arrival

```python
rewards = (distance < threshold).float()
# Returns [0, 1]: 1=reached, 0=not reached
```

**Parameters**:
- `threshold`: Distance threshold (default: 0.5m)
- `body_radius`: Robot body radius (default: 0.0m)

**Usage in config**:
```python
reaching_goal = RewTerm(
    func=reaching_goal,
    weight=500.0,  # High weight = main objective
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "threshold": 0.5,
        "body_radius": 0.5,
    }
)
```

**Design Notes**:
- This is the **most important reward** - main training objective
- Sparse: Only non-zero at episode end
- Creates strong gradient toward goal

---

### 2. Distance to Goal

**Function**: `progress_to_goal()`

**File**: `mdp/rewards/goal_rewards.py:222-299`

**Purpose**: Reward progress toward goal (decreasing distance)

```python
reward = prev_distance - current_distance
# Returns: positive=closer, negative=farther
```

**Key Implementation Details**:
- Uses `_prev_goal_dist` buffer to track previous distance
- Reset protection: Returns 0 on episode start (step 0)
- Range: [-10, 10] (clamped)

**Why this reward?**
- Prevents "circling" behavior (to farm velocity reward)
- Measures **effective progress** (not just speed)
- Complements velocity reward for better learning

---

### 3. Velocity Toward Goal

**Function**: `velocity_toward_goal()`

**File**: `mdp/rewards/goal_rewards.py:25-104`

**Purpose**: Reward speed component directed toward goal

```python
# Calculate projection of robot velocity onto goal direction
velocity_projection = dot(robot_vel, goal_direction)
reward = clamp(velocity_projection, min=0.0)  # Only reward forward motion
```

**Parameters**:
- `min_dist`: Minimum distance to avoid collision (default: 1.0m)

**Implementation**:
1. Calculate goal direction (unit vector)
2. Project robot velocity onto this direction
3. Gate: Only reward if distance > min_dist
4. Return range [0, max_linear_velocity]

**Design Philosophy**:
- Encourages **active movement** toward goal
- More direct than distance reward (measures actual speed)
- Prevents "ramming" goal (colliding at high speed)

---

### 4. Velocity Toward Goal (Smooth)

**Function**: `velocity_toward_goal_smooth()`

**File**: `mdp/rewards/goal_rewards.py:107-219`

**Purpose**: Adaptive velocity reward - slow down when approaching goal

**Parameters**:
- `slow_distance`: Start slowing down (default: 1.0m)
- `stop_distance`: Stop rewarding (default: 0.28m)
- `max_reward_speed`: Target speed at distance (default: 1.5 m/s)
- `min_reward_speed`: Target speed near goal (default: 0.3 m/s)

**Reward Calculation**:
```
Distance > 1.0m:    Target = 1.5 m/s (fast approach)
0.28m < D < 1.0m: Target = interpolate(0.3, 1.5) (smooth slowdown)
Distance < 0.28m:      Target = 0.0 m/s (stop - avoid collision)
```

**Actual Reward**: Exponential decay based on speed error

```python
speed_error = abs(velocity_projection - target_speed)
reward = exp(-speed_error / scale)
# reward ∈ [0, 1]: 1.0=perfect match, 0.0=far off
```

**Design Philosophy**:
- Encourages automatic deceleration near goal
- Prevents "overshooting" goal
- Uses exponential decay for smooth transitions

---

### 5. Heading to Goal

**Function**: `heading_to_goal()`

**File**: `mdp/rewards/goal_rewards.py:418-466`

**Purpose**: Reward proper orientation toward goal

```python
# Calculate cosine of angle between robot forward and goal direction
heading_reward = dot(forward_vec, goal_direction)
reward = clamp(heading_reward, min=0.0)  # Only reward facing goal
# Returns: 1.0=facing goal, 0.0=side-on, -1.0=facing away
```

**Design Philosophy**:
- More important when **close to goal** (turn before moving)
- Less important when **far from goal** (speed is priority)
- Works with distance-weighted version

---

### 6. Heading to Goal (Distance-Weighted)

**Function**: `heading_to_goal_distance_weighted()`

**File**: `mdp/rewards/goal_rewards.py:469-517`

**Purpose**: Dynamically weight heading reward based on distance to goal

**Parameters**:
- `close_distance`: Max weight distance (default: 2.0m)
- `far_distance`: Min weight distance (default: 5.0m)

**Weight Calculation**:
```python
if distance < 2.0m:
    weight = 1.0      # Full weight when close
elif distance > 5.0m:
    weight = 0.2      # Low weight when far
else:
    weight = interpolate(1.0, 0.2)  # Linear fade
```

**Design Philosophy**:
- Heading matters more **near goal** (need to align before moving)
- Heading matters less **far from goal** (speed is more important)
- Creates smooth transition between modes

---

### 7. Alignment Reward

**Function**: `alignment_reward()`

**File**: `mdp/rewards/goal_rewards.py:520-614`

**Purpose**: Incentivize proper heading, penalize misaligned high speed

**Parameters**:
- `misalignment_penalty_scale`: Penalty coefficient (default: 0.5)
- `high_speed_threshold`: Speed threshold (default: 1.0 m/s)

**Calculation**:
```python
# Base reward: Facing goal
alignment_base = clamp(dot(forward, goal_direction), min=0.0)

# Penalty condition: Misaligned (>72°) AND moving fast (>1.0 m/s)
if misaligned > 0.3 and speed_x > 1.0:
    penalty = misalignment * (speed_x - 1.0) * scale
else:
    penalty = 0

reward = alignment_base - penalty
```

**Design Philosophy**:
- Prevents "rushing" at wrong heading
- Encourages: **align first, then accelerate**
- Important for differential drive (can't strafe)
- Based on ROS "proportional controller" concept

---

### 8. Reverse Toward Goal

**Function**: `reverse_toward_goal_distance_weighted()`

**File**: `mdp/rewards/goal_rewards.py:673-742`

**Purpose**: Encourage backing up when facing away from goal

**Use Case**: Allows "3-point turn" navigation strategy

**Parameters**:
- `min_dist`: Minimum distance (default: 0.28m)
- `close_distance`: Max weight distance (default: 2.0m)
- `far_distance`: Min weight distance (default: 5.0m)

**Reward Calculation**:
```python
# Check if facing away (dot < 0)
is_facing_away = dot(forward, goal_direction) < 0.0

# Calculate backward speed toward goal
reverse_speed = clamp(-velocity_projection, min=0.0)

# Apply only when facing away
reward = reverse_speed * distance_weight * is_facing_away
```

**Design Philosophy**:
- Enables complex navigation: backup → turn → drive forward
- Distance-weighted: More important when closer to goal
- Helps avoid tight spaces

---

### 9. Approaching Goal Bonus

**Function**: `approaching_goal_bonus()`

**File**: `mdp/rewards/goal_rewards.py:351-415`

**Purpose**: Bonus for getting close to goal (even if not reached)

**Parameters**:
- `close_threshold`: Start giving bonus (default: 2.0m)
- `decay_scale`: Decay steepness (default: 0.5)

**Reward Calculation**:
```python
# Exponential decay based on distance
scale = close_threshold * decay_scale  # 2.0 * 0.5 = 1.0

reward = exp(-goal_distance / scale)
# Distance 2.0m → exp(-2.0) ≈ 0.14
# Distance 1.0m → exp(-1.0) ≈ 0.37
# Distance 0.5m → exp(-0.5) ≈ 0.61
# Distance 0.0m → exp(0) = 1.0
```

**Design Philosophy**:
- Combats "fear" - encourages approaching goal
- Smooth exponential decay (not sudden)
- Balances with progressive_collision (can approach, just not crash)
- Helps agent overcome obstacle aversion

---

## Safety Rewards

### 10. Progressive Collision Penalty

**Function**: `progressive_collision_penalty()`

**File**: `mdp/rewards/safety_rewards.py:113-259`

**Purpose**: Gradient penalty for approaching obstacles (directional + nonlinear)

**Parameters**:
- `safe_distance`: No penalty (default: 1.5m)
- `danger_distance`: Start medium penalty (default: 0.8m)
- `collision_distance`: Max penalty (default: 0.6m)
- `use_directional_penalty`: Enable directional penalty (default: True)
- `use_nonlinear_penalty`: Enable nonlinear (default: True)

**Three Zones**:

**Zone 1: Safe** (distance ≥ 1.5m)
```
Penalty: 0.0
```

**Zone 2: Warning** (0.8m ≤ distance < 1.5m)
```
Penalty: 0.5 × linear_progress
# Where linear_progress ∈ [0.5, 1.0)
```

**Zone 3: Danger** (0.6m ≤ distance < 0.8m)
```
Penalty: 0.5² to 1.0²
# Quadratic: 0.25 to 1.0
```

**Zone 4: Collision** (distance < 0.6m)
```
Penalty: 1.0 + (collision_distance / distance - 1.0)
Clamped: [1.0, 5.0]
# Inverse: spikes as distance → 0
```

**Directional Multiplier** (only in Zone 3):
```python
speed_ratio = abs(velocity_toward_obstacle) / max_speed
multiplier = 1.0 + 2.0 × (speed_ratio²)  # [1.0, 3.0]

penalty = base_penalty × multiplier
```

**Example Scenarios**:

| Scenario | Distance | Speed | Base Penalty | Multiplier | Final Penalty |
|-----------|----------|-------|-------------|---------------|
| Safe | 2.0m | 1.5 m/s | 0.0 | 1.0 | 0.0 |
| Danger | 0.5m | 1.5 m/s toward | 1.0²=1.0 | 3.0 | -3.0 |
| Danger | 0.5m | 0.2 m/s sideways | 1.0²=1.0 | 1.0 | -1.0 |
| Danger | 0.4m | 1.5 m/s toward | 1.0+4.0=5.0 | 3.0 | -5.0 |

**Design Philosophy**:
- Makes "danger zone" **psychologically uncomfortable**
- Directional: Only penalizes **approaching at speed**
- Nonlinear: Amplifies effect in collision zone
- Allows sideways movement (lower penalty)

---

### 11. Safe Navigation Bonus

**Function**: `safe_navigation_bonus()`

**File**: `mdp/rewards/safety_rewards.py:262-302`

**Purpose**: Reward keeping safe distance from obstacles

**Parameters**:
- `comfort_distance`: Full bonus distance (default: 2.5m)

**Reward Calculation**:
```python
reward = clamp(min_distance / comfort_distance, 0.0, 1.0)
# Distance 2.5m → 1.0 (full bonus)
# Distance 1.25m → 0.5 (half bonus)
# Distance 0.5m → 0.2 (small bonus)
# Distance 0.0m → 0.0 (no bonus)
```

**Design Philosophy**:
- Encourages **active avoidance** (not passive)
- More comfortable to "go around" than "edge-past"
- Balances with progressive_collision
- Helps overcome "fear" of obstacles

---

### 12. Speed Control Near Obstacles

**Function**: `speed_control_near_obstacles()`

**File**: `mdp/rewards/safety_rewards.py:305-388`

**Purpose**: Encourage slowing down when near obstacles

**Parameters**:
- `warning_distance`: Start regulating (default: 2.0m)
- `max_safe_speed`: Maximum speed near obstacles (default: 0.5 m/s)

**Reward Calculation**:
```python
# Only applies when distance < 2.0m
if close_to_obstacle:
    speed_excess = robot_speed - max_safe_speed
    distance_weight = 1.0 - (distance / 2.0)  # [0, 1]
    
    reward = -speed_excess * distance_weight
```

**Example**:
```
Distance = 1.0m, Speed = 1.0 m/s:
  speed_excess = 0.5
  distance_weight = 0.5
  reward = -0.5 × 0.5 = -0.25 (penalty)

Distance = 1.0m, Speed = 0.3 m/s:
  speed_excess = -0.2 (negative, no penalty)
  distance_weight = 0.5
  reward = -(-0.2) × 0.5 = +0.1 (reward)
```

**Design Philosophy**:
- Distance-weighted: Slower penalty when closer
- Allows moderate speed when far
- Promotes **gradual deceleration**
- Works with progressive_collision

---

## Motion Rewards

### 13. Move Reward

**Function**: `move_reward()`

**File**: `mdp/rewards/motion_rewards.py`

**Purpose**: Simple encouragement to keep moving (any direction)

**Reward Calculation**:
```python
reward = k × (speed_x + b) × weight
# Default: 1.0 × (speed_x + 0.0) × 0.2
```

**Example**:
```
speed_x = 0.0 m/s → reward = 0.0 (no movement)
speed_x = 0.5 m/s → reward = +0.1 (slow movement)
speed_x = 1.0 m/s → reward = +0.2 (fast movement)
speed_x = 1.5 m/s → reward = +0.3 (max speed)
```

**Design Philosophy**:
- Prevents "freezing" (stopping due to fear)
- Simple and interpretable
- Direction-agnostic (backward also counts)
- Works with other rewards to encourage forward motion

---

## Extreme Behavior Penalties

### 14. Tipped Over Penalty

**Function**: `robot_tipped_over()`

**File**: `mdp/terminations/robot_state.py` (also used as penalty)

**Purpose**: Terminate and heavily penalize robot flipping

**Trigger**: Z-axis tilt > 45° from upright

**Usage**:
```python
# As termination condition
tipped_over = DoneTerm(
    func=robot_tipped_over,
    params={"asset_cfg": SceneEntityCfg("robot")},
)

# As reward penalty
tipped_over_penalty = RewTerm(
    func=robot_tipped_over,
    weight=-200.0,  # Extreme penalty
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

**Design Philosophy**:
- Robot must be upright to receive movement rewards
- Creates "gate" (tipped = all rewards = 0)
- -200.0 penalty discourages this behavior
- Triggers episode termination

---

### 15. Wall Collision Penalty

**Function**: `wall_collision_penalty()`

**File**: `mdp/rewards/safety_rewards.py:494-560`

**Purpose**: Penalty for getting too close to environment boundaries

**Parameters**:
- `boundary`: Wall distance (default: 5.0m)
- `robot_radius`: Robot radius (default: 0.28m)
- `safe_distance`: Start penalty (default: 1.0m)

**Reward Calculation**:
```python
# Distance to nearest wall
min_wall_dist = min(dist_east, dist_west, dist_north, dist_south) - robot_radius

# Linear penalty: closer = worse
if distance < safe_distance:
    penalty = 1.0 - (distance / safe_distance)
else:
    penalty = 0.0

# Clamp to [0, 1]
penalty = clamp(penalty, 0.0, 1.0)
```

**Design Philosophy**:
- Encourages staying away from walls
- Linear increase as robot approaches wall
- Maximum penalty at distance ≤ 0 (touching wall)
- Helps prevent "corner trapping"

---

## Weight Balance Rules

### Critical Ratios

**Rule**: `goal_weight : avoidance_weight ≥ 25 : 1`

| Phase | Goal | Avoidance | Ratio | Behavior |
|--------|-------|------------|--------|----------|
| Phase 1 | 5.0 | 0.1 | 50:1 | ✅ No snake |
| Phase 2 | 5.0 | 0.2 | 25:1 | ✅ Balanced |
| Phase 2.5 | 3.0 | 0.4 | 7.5:1 | ❌ Snake |
| Phase 3 | 5.0 | 0.0 | ∞:1 | ✅ Pure goal |

### Adjustment Guide

**If agent snakes (oscillates)**:
```python
# Option 1: Increase goal weight
distance_to_goal.weight = 5.0  # From 3.0

# Option 2: Decrease avoidance weight
progressive_collision.weight = -1.0  # From -2.5
safe_navigation.weight = 0.2  # From 0.4

# Option 3: Both
```

**If agent is too conservative (stops often)**:
```python
# Increase movement incentive
move_reward.weight = 0.3  # From 0.2
velocity_toward_goal.weight = 4.0  # From 3.0
```

**If agent ignores obstacles (too many collisions)**:
```python
# Increase safety penalties
progressive_collision.weight = -2.0  # From -1.0
speed_control_near_obstacles.weight = 0.4  # From 0.2
```

---

## Reward Gating Patterns

### 1. Upright Gate

**Purpose**: Disable movement rewards if robot is tipped

```python
def gated_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    is_upright = check_upright(env)
    base_reward = calculate_reward(env)
    
    return where(
        is_upright,
        base_reward,
        zeros_like(base_reward)
    )
```

### 2. Distance Gate

**Purpose**: Prevent rewarding velocity when too close to goal/obstacle

```python
def velocity_toward_goal(env, min_dist=1.0) -> torch.Tensor:
    velocity_projection = calculate_projection(...)
    
    # Gate: Don't reward if too close (to avoid ramming)
    gated = where(
        goal_distance > min_dist,
        velocity_projection,
        zeros_like(velocity_projection)
    )
    
    return gated
```

### 3. Direction Gate

**Purpose**: Enable/disable rewards based on moving direction

```python
# Example: Only reward forward motion
reward = clamp(velocity_projection, min=0.0)
# Backward motion → 0 reward
```

---

## Reward Shape Selection Guide

### When to Use Each Shape

| Shape | Use Case | Example |
|-------|-----------|----------|
| **Constant** | Binary events | Reaching goal (0 or 1) |
| **Linear** | Uniform influence | Distance penalty, wall collision |
| **Quadratic** | Amplify effect | Progressive collision danger zone |
| **Inverse** | Sharp increase near threshold | Collision zone inverse penalty |
| **Exponential** | Smooth transition | Velocity smooth, approaching bonus |
| **Cosine** | Angular alignment | Heading to goal |
| **Piecewise** | Different rules per region | Distance-weighted heading |

### Common Mistakes

**❌ Don't**: Mix unrelated reward types
```python
# Bad: Combining alignment and collision
reward = alignment_reward * collision_penalty
# Confusing signal for agent
```

**✅ Do**: Separate into distinct reward terms
```python
# Good: Clear reward terms
alignment = RewTerm(func=alignment_reward, weight=2.5)
collision = RewTerm(func=progressive_collision, weight=-1.5)
```

**❌ Don't**: Use conflicting goals
```python
# Bad: Encourage fast but also slow
fast_reward = RewTerm(func=speed, weight=5.0)
slow_reward = RewTerm(func=speed_control, weight=-3.0)
# Confusing: What should agent do?
```

**✅ Do**: Use complementary rewards
```python
# Good: Speed control, not "don't be fast"
speed_control = RewTerm(
    func=speed_control_near_obstacles,
    weight=0.4,
    params={"max_safe_speed": 0.5}  # Encourage this speed
)
```

---

## Debugging Reward Issues

### Symptom: Agent oscillates (snakes)

**Diagnosis**:
1. Check reward weights
2. Look for conflicting signals
3. Calculate goal:avoidance ratio

**Investigation**:
```python
# Print reward breakdown
reward_dict = {
    "goal": distance_to_goal * 5.0,
    "avoidance": progressive_collision * (-1.5),
    "total": total_reward
}
print(reward_dict)
```

**Solution**:
- Increase goal weight (e.g., 5.0 → 7.0)
- Decrease avoidance weight (e.g., -1.5 → -1.0)
- Aim for ratio ≥ 25:1

---

### Symptom: Agent freezes (stops moving)

**Diagnosis**:
1. Check if fear is dominating (avoidance too high)
2. Check if movement rewards exist
3. Check for excessive collision penalties

**Solution**:
```python
# Add move reward
move_reward = RewTerm(
    func=move_reward,
    weight=0.3,
)

# Or reduce collision penalty
progressive_collision = RewTerm(
    func=progressive_collision_penalty,
    weight=-0.8,  # From -1.5
)
```

---

### Symptom: Reward is always zero

**Diagnosis**:
1. Check observation normalization (NaN/Inf issues)
2. Check reward function returns
3. Check gating conditions

**Fix**:
```python
# Always handle NaN/Inf
value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)

# Always clamp to valid range
value = torch.clamp(value, min=-10.0, max=10.0)

# Always use reward term checker
reward = _check_reward_term("function_name", reward, env, raise_on_error=True)
```

---

## Customizing Rewards

### Adding New Reward

1. Implement function in `mdp/rewards/`:

```python
def my_custom_reward(env: ManagerBasedRLEnv, param: float = 1.0) -> torch.Tensor:
    """My custom reward"""
    
    # Calculate reward
    reward = calculate_something(env)
    
    # Safety checks
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, -10.0, 10.0)
    reward = _check_reward_term("my_custom_reward", reward, env, raise_on_error=True)
    
    return reward
```

2. Export in `mdp/rewards/__init__.py`:

```python
from .functions import my_custom_reward
__all__ = ["my_custom_reward", ...]
```

3. Add to config:

```python
@configclass
class RewardsCfg:
    my_custom = RewTerm(
        func=my_custom_reward,
        weight=1.0,
        params={"param": 1.0}
    )
```

### Modifying Existing Reward

1. Locate reward in `cfg/charge_env_cfg.py`
2. Adjust weight or parameters
3. Test with training run
4. Monitor TensorBoard metrics

**Example modification**:
```python
# Original
distance_to_goal = RewTerm(
    func=progress_to_goal,
    weight=5.0,
)

# Modified (increase importance)
distance_to_goal = RewTerm(
    func=progress_to_goal,
    weight=7.0,  # Increased from 5.0
)
```

---

## Performance Monitoring

### Key Metrics

```python
# In TensorBoard
tensorboard --logdir logs/rsl_rl/charge_navigation_phase3/

# Monitor:
# - Train/mean_reward
# - Train/episodic_reward (per episode)
# - Individual reward terms (if logged)
```

### Success Rate Targets

| Phase | Target Success Rate | Target Collision Rate |
|--------|-------------------|----------------------|
| Phase 1 | > 90% | N/A (no obstacles) |
| Phase 2 | > 75% | < 20% |
| Phase 3 | > 90% | N/A (no obstacles) |

### Path Efficiency

```python
# Ideal: straight line
ideal_distance = initial_distance

# Actual: path taken
actual_path_length = sum(step_distances)

# Efficiency
efficiency = ideal_distance / actual_path_length
# Target: < 1.2 (20% longer than optimal)
```

---

## Reference

For more details on implementation:
- See `mdp/rewards/goal_rewards.py` for goal-directed rewards
- See `mdp/rewards/safety_rewards.py` for safety rewards
- See `mdp/rewards/motion_rewards.py` for motion rewards
- See `mdp/rewards/utils.py` for utility functions
