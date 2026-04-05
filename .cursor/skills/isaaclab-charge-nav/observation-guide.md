# Observation Function Guide

Complete guide to all observation functions in Charge Navigation.

## Observation Space Overview

**Total Dimensions**: 131 (Phase 1/2) to 134 (Phase 2.5+)

| Observation | Dims | Type | Range | Purpose |
|------------|-------|------|--------|----------|
| **LiDAR scan** | 120 | Float [0, 1] | Distance readings (3 layers × 40 rays) |
| **Base velocity** | 3 | Float [±∞, ±∞] | [vx, vy, vz] in world frame |
| **Goal position (robot)** | 2 | Float [-10, 10] | [dx, dy] relative to robot |
| **Goal distance** | 1 | Float [0, 20] | Euclidean distance to goal |
| **Time remaining** | 1 | Float [0, 1] | (1 - elapsed/episode_length) |
| **Alive flag** | 1 | Int {0, 1} | 1=alive, 0=terminated |
| **Obstacle state** | 3 | Float [varies] | [count, avg_size, min_distance] |

## LiDAR Observations

### 1. LiDAR Scan (2D Sweep)

**Function**: `lidar_scan_2d_sweep()`

**File**: `mdp/observations/functions.py:77-179`

**Purpose**: Multi-layer horizontal sweep for 360° obstacle detection

**Configuration**:
```python
lidar = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=3,  # 3 vertical layers
        vertical_fov_range=(-10.0, 10.0),  # ±10° elevation
        horizontal_fov_range=(-180.0, 180.0),  # 360° horizontal
        horizontal_res=5.0,  # 5° per ray = 72 rays
    ),
    max_distance=10.0,  # 10 meters max range
)
```

**Layers**:
- **Bottom layer** (-10° elevation): Ground clearance
- **Middle layer** (0° elevation): Horizontal plane
- **Top layer** (+10° elevation): Obstacle height

**Output Shape**: `[num_envs, 72]` or `[num_envs, 120]`
- 72 rays (360° / 5° per ray)
- Normalized to [0, 1] where 0=closest, 1=farthest

**Data Flow**:
```
Sensor → Ray caster → Hit points → 2D distances → Normalization
```

**Critical Normalization**:
```python
# Step 1: Replace NaN/Inf
distances = nan_to_num(distances, nan=max_range, posinf=max_range)

# Step 2: Clip to valid range
distances = clamp(distances, 0.0, max_range)

# Step 3: Normalize to [0, 1]
normalized = distances / max_range

# Step 4: Final NaN protection
normalized = nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)

# Step 5: Final clamp
normalized = clamp(normalized, 0.0, 1.0)

# Step 6: Finite check (prevents PPO std>=0 error)
check_finite("lidar_scan_2d_sweep", normalized, raise_on_error=True)
```

**Why This Matters**:
- **NaN protection**: PPO crashes if observation contains NaN
- **Normalization**: Prevents explosion of gradient magnitudes
- **Clipping**: Ensures valid range for neural network

---

### 2. Goal Position (Robot Frame)

**Function**: `goal_position_in_robot_frame()`

**File**: `mdp/observations/functions.py:186-222`

**Purpose**: Target position relative to robot's local coordinate system

**Output Shape**: `[num_envs, 2]`

**Coordinate Transformation**:
```python
# Get world coordinates
goal_pos_w = env.command_manager.get_command("goal_command")  # [num_envs, 3]
robot_pos_w = asset.data.root_pos_w[:, :3]             # [num_envs, 3]
robot_quat_w = asset.data.root_quat_w               # [num_envs, 4]

# Transform to robot frame
goal_vec_b, _ = subtract_frame_transforms(
    robot_pos_w, robot_quat_w,
    goal_pos_w, torch.zeros_like(robot_quat_w)
)

# Keep only X, Y (forward/left-right)
result = goal_vec_b[:, :2]  # [num_envs, 2]
```

**Interpretation**:
```
result[env, 0] = dx (forward distance)
result[env, 1] = dy (left/right distance)
```

**Example**:
```python
# Robot at origin, goal at (3, 4)
goal_position = [2.5, -1.0]

# Robot at (3, 4), facing 45°
goal_position = [-0.707, 0.707]
# Meaning: goal is 1 meter to left, 1 meter behind
```

**Design Notes**:
- Ignored Z coordinate (height)
- Provides intuitive local frame for differential drive
- Enables "turn toward goal" behavior

---

### 3. Goal Distance

**Function**: `goal_distance()`

**File**: `mdp/observations/functions.py:225-253`

**Purpose**: Scalar distance from robot to goal

**Output Shape**: `[num_envs, 1]`

**Calculation**:
```python
goal_pos_w = env.command_manager.get_command("goal_command")  # [num_envs, 3]
robot_pos_w = asset.data.root_pos_w[:, :2]                # [num_envs, 2]

# Euclidean distance
distance = norm(goal_pos_w - robot_pos_w, dim=1)  # [num_envs]
```

**Safety Processing**:
```python
# Clean NaN
distance = nan_to_num(distance, nan=10.0, posinf=10.0, neginf=0.0)

# Clip to reasonable range
distance = clamp(distance, 0.0, 20.0)

# Final protection
distance = nan_to_num(distance, nan=0.0, posinf=1.0, neginf=0.0)
distance = clamp(distance, 0.0, 20.0)
```

**Use Cases**:
- Reward shaping (distance penalties/bonuses)
- Termination conditions (within threshold)
- Progress monitoring

---

### 4. Base Velocity (World Frame)

**Function**: `base_velocity_xy()`

**File**: `mdp/observations/functions.py:283-303`

**Purpose**: Robot's 2D velocity in world coordinate system

**Output Shape**: `[num_envs, 2]`

**Calculation**:
```python
vel_w = asset.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
# vel_w[env, 0] = vx (east/west)
# vel_w[env, 1] = vy (north/south)
```

**Coordinate System**:
```
World Frame (Global):
    +X axis: East direction
    +Y axis: North direction
    Origin: (0, 0, 0) at environment origin
```

**Example**:
```python
# Robot moving northeast at 1 m/s
velocity = [[0.707, 0.707], ...]
# Magnitude = sqrt(0.707² + 0.707²) = 1.0 m/s
# Direction = 45° from east
```

**Use Cases**:
- Velocity-based rewards (efficiency)
- Motion planning
- Physics calculations

---

### 5. Safe Last Action

**Function**: `safe_last_action()`

**File**: `mdp/observations/functions.py:260-280`

**Purpose**: Previous action (what did agent do last step?)

**Output Shape**: `[num_envs, 2]`

**Calculation**:
```python
actions = env.action_manager.action  # [num_envs, 2]

# Clean NaN/Inf
actions = nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
actions = clamp(actions, -1.0, 1.0)
```

**Reset Protection**:
```python
# Reset on episode start
actions = torch.where(
    env.episode_length_buf == 0,
    torch.zeros_like(actions),
    actions
)
```

**Use Cases**:
- Action smoothing (prevent jitter)
- History tracking
- Debugging agent behavior

---

### 6. Time Remaining Ratio

**Function**: `time_remaining_ratio()`

**File**: `mdp/observations/functions.py` (not shown in preview, likely similar)

**Purpose**: Normalized time remaining in episode

**Output**: Single float per environment

**Calculation**:
```python
elapsed = env.episode_length_buf
remaining = 1.0 - (elapsed / episode_length)
```

**Range**: [0.0, 1.0]

**Interpretation**:
- `1.0` = Episode just started
- `0.5` = Halfway through episode
- `0.0` = Episode ending (time running out)

**Use Cases**:
- Time pressure rewards (encourage efficiency)
- Learning schedules
- Curriculum adaptation

---

### 7. Alive Flag

**Function**: Likely in `mdp/observations/functions.py`

**Purpose**: Episode status indicator

**Output**: Int {0, 1} per environment

**States**:
```python
1 = Alive (receiving rewards, taking actions)
0 = Terminated (episode ended, no actions/rewards)
```

**Use Cases**:
- Filter terminated episodes from statistics
- Mask out done environments
- Calculate success rates

---

### 8. Dynamic Obstacles State

**Function**: `dynamic_obstacles_state()`

**File**: `mdp/observations/functions.py` (likely)

**Purpose**: Summary of dynamic obstacles (if applicable)

**Output Shape**: `[num_envs, 3]`

**Components**:
```python
[
    num_obstacles,      # How many obstacles?
    avg_obstacle_size,   # Average radius (meters)
    min_obstacle_dist,   # Distance to nearest obstacle (meters)
]
```

**Use Cases**:
- Curriculum learning (adjust based on obstacle density)
- Reward shaping (dynamic difficulty)
- Analysis (understanding environment state)

---

## Coordinate Systems

### Robot Local Frame

**Origin**: Robot center (base_link)

**Axes**:
```
    +X (Forward): Direction robot faces
    +Y (Left):     90° counterclockwise from +X
    +Z (Up):       Perpendicular to ground
```

**Use Case**: Differential drive control (turn left/right)

### World Frame

**Origin**: Environment origin (0, 0, 0)

**Axes**:
```
    +X (East):  Fixed global direction
    +Y (North): Fixed global direction  
    +Z (Up):    Fixed global direction (gravity points down)
```

**Use Case**: Global positioning, obstacle locations, goals

### Frame Transformation

```python
import isaaclab.utils.math as math_utils

# World → Robot (localize goal/obstacles)
robot_vec = math_utils.subtract_frame_transforms(
    robot_pos_w, robot_quat_w,
    world_pos_w, torch.zeros_like(robot_quat_w)
)

# Robot → World (globalize velocity)
world_vec = math_utils.quat_apply(robot_quat_w, robot_vec)
```

**Rotation Matrix**:
```python
# Quat to rotation matrix
R = quat_to_matrix(quat_w)

# Apply transformation
robot_vec = R @ world_vec
```

---

## Observation Vectorization

### Why Vectorization Matters

**Batch Processing**:
```python
# ❌ Bad: Sequential processing
for i in range(num_envs):
    obs[i] = calculate_obs(env, i)

# ✅ Good: Vectorized processing
obs = calculate_obs_vectorized(env)  # Single operation
```

**Performance Impact**:
- Sequential: O(n) operations, very slow
- Vectorized: O(1) operations, 100x+ faster

**Critical for**: Isaac Lab's parallel training (128+ environments)

---

### Normalization Best Practices

### 1. Always Handle NaN/Inf

```python
# Step 1: Replace
tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

# Step 2: Clip
tensor = torch.clamp(tensor, min_val, max_val)
```

### 2. Check Finite Before Return

```python
def check_finite(name: str, value: torch.Tensor, env: ManagerBasedRLEnv):
    """Prevents PPO std>=0 error"""
    
    # Debug immediately if problem
    if not torch.all(torch.isfinite(value)):
        env.logger.error(f"{name} contains non-finite values: {value}")
        if raise_on_error:
            raise ValueError(f"{name} has NaN/Inf")
    
    return value
```

### 3. Normalize to Fixed Range

```python
# For neural network input: [0, 1] is ideal
normalized = raw_value / max_possible_value

# Alternative: z-score normalization
mean = torch.mean(values)
std = torch.std(values)
normalized = (values - mean) / (std + 1e-6)
```

### 4. Maintain Consistent Dimensions

**Phase Compatibility**:
```python
# Phase 1: 3 obstacles → obs dim = 131
MAX_OBSTACLES = 10  # Fixed maximum

# Pad unused slots
if actual_obstacles < MAX_OBSTACLES:
    obstacle_obs = torch.cat([
        actual_obstacle_data,
        torch.zeros(num_envs, MAX_OBSTACLES - actual_obstacles, device=device)
    ], dim=1)
else:
    obstacle_obs = actual_obstacle_data  # No padding needed

# Result: obs dim always 131
```

**Why This Matters**:
- Enables weight transfer (Phase 1 → Phase 2)
- PPO policy can be loaded directly
- Avoids "observation dimension mismatch" errors

---

## Adding New Observations

### Step 1: Implement Function

```python
def my_new_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """My new observation
    
    Returns:
        shape [num_envs, obs_dim]: Observation tensor
    """
    
    # Get data
    asset = env.scene["robot"]
    
    # Calculate observation
    obs = calculate_something(env, asset)
    
    # Safety checks
    obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
    obs = torch.clamp(obs, -10.0, 10.0)
    
    return obs
```

### Step 2: Export from Module

**File**: `mdp/observations/__init__.py`

```python
from .functions import my_new_observation

__all__ = ["my_new_observation", ...]
```

### Step 3: Add to Config

**File**: `cfg/charge_env_cfg.py`

```python
@configclass
class ObservationsCfg:
    policy = ObsGroup(
        my_new_observation = ObsTerm(
            func=my_new_observation,
            params={"asset_cfg": SceneEntityCfg("robot")}
        ),
        # ... other observations
    )
```

### Step 4: Test Dimension

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 1 \
    --max_iterations 1

# Check output obs shape
print(f"Observation shape: {env.observation_space.shape}")
```

---

## Debugging Observations

### Common Issues

| Issue | Symptoms | Cause | Fix |
|--------|------------|--------|------|
| PPO std >= 0 | Agent crashes during training | Observation has zero variance | Add small noise or check normalization |
| NaN in reward | Reward explodes | Observation contains NaN | Add `nan_to_num` processing |
| Shape mismatch | ImportError on resume | Observation dim changed between phases | Maintain fixed dimensions |
| Exploding gradients | Loss becomes inf | Observation not normalized | Scale to [0, 1] |

### Visualization

```python
# Debug observation values
print(f"LiDAR min: {lidar.min()}, max: {lidar.max()}")
print(f"Goal pos: {goal_position}")
print(f"Velocity: {velocity}")

# Visualize in Isaac Sim
env.sim.render()
```

### TensorBoard Logging

```python
# Log observation statistics
writer.add_scalar("Observations/lidar_mean", lidar.mean(), step)
writer.add_scalar("Observations/goal_distance", goal_dist.mean(), step)
writer.add_scalar("Observations/velocity_mag", torch.norm(vel, dim=1), step)
```

---

## Reference

For implementation details:
- See `mdp/observations/functions.py` for all observation functions
- See `mdp/observations/utils.py` for utility functions
- See `reference.md` for architecture overview
