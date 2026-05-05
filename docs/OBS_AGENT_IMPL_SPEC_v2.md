Now I have enough context about the current system. Let me produce the comprehensive design spec.
I now have sufficient understanding of the current obstacle system. Let me produce the complete design specification.
Now I have a comprehensive understanding of the current system. Let me write the design specification.
Good. I now have all the context needed. Let me write the complete design specification document.

---

# Rule-Based Obstacle Behavior Agent — 可實作設計規格 v2

## 1. Hard Safety Constraints（全域 + 各 behavior 覆寫）

### 1.1 全域 Safety Constraints

| 約束名 | 值 | 物理理由 |
|--------|-----|---------|
| `min_spawn_distance_from_robot` | 2.0 m | robot(0.35) + obs(0.30) + buffer(1.35)；防止 spawn 即碰撞，且留 1s 反應時間 (v_obs_max~0.85) |
| `min_spawn_distance_from_goal` | 1.5 m | 不允許 goal 被 obstacle 實體擋住 |
| `min_wall_clearance` | 0.6 m | obs_radius(0.30) + 安全帶(0.30)；AABB proximity check |
| `min_obstacle_obstacle_distance` | 0.8 m | 2 * obs_radius(0.30) + gap(0.20)；允許 robot 勉強通過 |
| `max_blocking_on_goal_line` | 0 steps (禁止) | 預設不允許 obstacle 停留在 robot-goal 直線上超過 10 frames |
| `reset_timeout` | 750 steps (150s) | 遠超 episode timeout(90s)，作為 bug catcher |

### 1.2 Per-Behavior Constraint Overrides

| Behavior | spawn_from_robot | spawn_from_goal | wall_clearance | obs_obs_dist | blocking_on_goal_line | 特殊約束 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
| static | 1.5 m | 1.5 m | 0.6 m | 0.8 m | 25 steps (5s) | -- |
| patrol | 2.0 m | 1.5 m | 0.8 m | 0.8 m | 15 steps (3s) | waypoint 必須全部在 wall_clearance 內 |
| random_walk | 2.0 m | 1.5 m | 0.6 m | 0.8 m | 15 steps (3s) | -- |
| horizontal_crossing | 3.0 m | 2.0 m | 0.8 m | 1.0 m | 0 (允許穿越) | 穿越時間 < 3.0s |
| path_crossing | 3.5 m | 2.0 m | 0.8 m | 1.0 m | 0 (穿越) | TTC_to_robot_path >= 1.5s |
| near_miss | 2.5 m | 2.0 m | 0.6 m | 1.0 m | 0 (穿越) | surface_clearance >= 0.10m |
| corridor_crossing | 2.5 m | 1.5 m | 0.4 m (需 corridor) | 1.2 m | 0 (穿越) | 必須在 detected corridor gap 內 |
| multi_obstacle_occlusion | 2.5 m | 2.0 m | 0.6 m | 0.5 m (front/back pair) | 15 steps | min_visible_frames=8 (1.6s) |

---

## 2. Behavior 定義

### 2.1 Static Behavior（修正版）

**目的**: 提供 stationary LiDAR baseline，讓 RNN 學習區分 persistent geometry（牆壁 + 靜態障礙物）與 dynamic temporal changes（移動障礙物）。

**設計原則**:
- Static obstacle 在 LiDAR 中產生**不隨時間變化的回波** (constant bin value across frames)
- 與牆壁 LiDAR 回波的差異：位置離散化 (5-degree bin 中只佔 1~2 bins)，而牆壁佔多連續 bins
- RNN 不需要為 static obstacle 分配時序追蹤資源
- WD aux target: 對 static obstacle，(vx, vy) = (0, 0)，有效降低 auxiliary prediction 的 baseline error

**參數**:
```
position: uniform_sample(boundary) with collision check
velocity: (0, 0) — 永不移動
radius: uniform(0.20, 0.35)  — 與 dynamic 相同分佈，防止 policy 用 size 區分
lifetime: entire episode
```

### 2.2 Patrol Behavior

**目的**: 可預測路徑，讓 policy 學會「等待通過」或「timing 插入」。

**參數**:
```
waypoints: 2~4 points, convex hull area < 9.0 m^2
speed: uniform(0.3, 0.6) m/s — 低速巡邏
pause_at_waypoint: uniform(0, 5) steps (0~1s)
patrol_mode: "loop" | "ping_pong"  (50%/50%)
direction: uniform random starting waypoint index
```

### 2.3 Random Walk Behavior

**目的**: 不可預測運動，訓練 policy 的 reactive avoidance 能力。

**參數**:
```
speed: uniform(0.2, 0.7) m/s
direction_change_interval: uniform(5, 25) steps (1~5s)
direction_change_angle: uniform(-pi/2, pi/2) — 最大 90 degree turn
smooth_turn_steps: 3 steps (0.6s) — 線性插值轉向，非瞬間轉向
boundary_bounce: reflect at wall_clearance
```

### 2.4 Horizontal Crossing Behavior

**目的**: 模擬行人橫穿機器人前方路徑。

**參數**:
```
crossing_speed: uniform(0.5, 1.0) m/s
crossing_offset: uniform(2.0, 6.0) m ahead of robot (along robot heading)
crossing_direction: left_to_right | right_to_left (50%/50%)
crossing_width: 6.0~12.0 m — 足夠穿越 robot 的路徑寬度
spawn_side_offset: 3.0~6.0 m perpendicular from robot-goal line
repeat_interval: uniform(30, 60) steps (6~12s) after completing crossing
```

### 2.5 Path Crossing Behavior

**目的**: obstacle 沿 robot-to-goal 路徑的**交叉方向**穿過，更接近真實場景中的 "交叉口" 情境。

**參數**:
```
crossing_speed: uniform(0.4, 0.9) m/s
crossing_angle: uniform(30, 150) degrees relative to robot-goal line (不完全平行或正交)
TTC_to_path: uniform(1.5, 4.0) s — 障礙物從可見到穿過 robot 路徑的時間
spawn_distance_from_path: 2.0~5.0 m (perpendicular)
min_distance_ahead: 2.5 m — 必須在 robot 前方
```

### 2.6 Near-Miss Behavior（修正版）

**目的**: 訓練 policy 在 tight clearance 場景的精確操控，同時確保 obstacle 軌跡不依賴 robot 的迴避動作。

**設計原則**:
- Obstacle 的軌跡是 episode 開始時完全**預先計算** (pre-computed trajectory)
- 計算 clearance 時基於 robot_initial_heading_extrapolation（假設 robot 不轉向不減速的直線位置）
- 即使 robot 完全不動/不閃，obstacle 也不會 center-hit robot（surface clearance >= 0.10m from initial projection）
- 這防止 robot 學到「obstacle 會自己讓路」— obstacle 不知道 robot 在哪，是獨立軌跡

**Clearance Bucket 分佈**:
```
clearance_distribution:
  tight:  0.10~0.20 m surface clearance — 30% 比例
  medium: 0.20~0.40 m surface clearance — 50% 比例
  loose:  0.40~0.60 m surface clearance — 20% 比例
```

**TTC Range**:
```
ttc_visible_to_closest: uniform(2.0, 5.0) s
  — 從 obstacle 進入 LiDAR range 到最近接觸點的時間
  — 對應 approach_speed: clearance_dist / ttc * correction_factor
```

**Logging**:
```
log_fields:
  - actual_surface_clearance: float (m) — 實際最小 surface 距離
  - robot_did_decelerate: bool — robot speed decreased > 0.2 m/s in last 10 steps before closest point
  - robot_did_turn: bool — heading changed > 15 degrees in last 10 steps
  - robot_did_wait: bool — robot speed < 0.05 m/s for >= 3 consecutive steps before closest point
  - planned_clearance_bucket: str — "tight" | "medium" | "loose"
```

**軌跡預計算 Pseudo-code**:
```
def compute_near_miss_trajectory(robot_pos, robot_heading, clearance_target, ttc):
    # 1. 沿 robot heading 前方 approach_dist 處定義 closest_point
    #    approach_dist = robot.v_max * ttc (robot 不動的情況下 obstacle 到達的位置)
    closest_point = robot_pos + robot_heading_vec * (robot_speed * ttc)
    
    # 2. 計算 closest_point 的 perpendicular offset = clearance_target + robot_r + obs_r
    offset_distance = clearance_target + ROBOT_RADIUS + obs_radius
    perpendicular = rotate_90(robot_heading_vec) * random_sign()
    actual_closest = closest_point + perpendicular * offset_distance
    
    # 3. 從 LiDAR range 邊界反推 spawn position
    obs_speed = offset_distance / (ttc * 0.3)  # 調整使 obstacle 在 ttc 時到達 closest
    spawn_pos = actual_closest - obs_velocity_dir * obs_speed * ttc
    
    # 4. 產出 linear trajectory: pos(t) = spawn_pos + obs_vel * t
    return spawn_pos, obs_velocity, obs_speed
```

### 2.7 Corridor Crossing Behavior

**目的**: 在牆壁 gap (corridor) 中製造交通，訓練 policy 判斷「何時進入狹窄通道」。

**前提**: 需要 wall_layout 提供 detected corridor gaps (width >= 1.5m)。

**參數**:
```
crossing_speed: uniform(0.3, 0.7) m/s — corridor 內移動較慢
corridor_detection:
  min_gap_width: 1.5 m (robot 0.35*2 + obs 0.30*2 + buffer 0.20)
  max_gap_width: 4.0 m
crossing_direction: along corridor major axis
spawn_offset_from_gap: 1.0~3.0 m (在 corridor 的一端 spawn)
repeat: True — 反覆穿越 (ping-pong)
```

### 2.8 Multi-Obstacle Occlusion Behavior（修正版）

**目的**: 訓練 RNN 的 temporal memory — 被遮擋 obstacle 的位置預測。

**設計原則**:
- 由 2~3 個 obstacle 組成一個 **occlusion group** (1 front blocker + 1~2 back targets)
- Back target 在被遮擋前必須至少可見 `min_visible_frames` = 8 frames (1.6s)
- 遮擋後再次出現的位置必須符合 constant velocity extrapolation
- 遮擋判定: front_obstacle 在 robot → back_obstacle 的 LOS 上，且 front angular width > back angular width

**Angular Width 計算**:
```
angular_width(robot_pos, obs_pos, obs_radius):
    distance = ||obs_pos - robot_pos||
    return 2 * arctan(obs_radius / distance)

is_occluded(robot_pos, front_pos, front_r, back_pos, back_r):
    # 1. front 在 robot→back 的 LOS 上
    angle_to_back = atan2(back_pos.y - robot_pos.y, back_pos.x - robot_pos.x)
    angle_to_front = atan2(front_pos.y - robot_pos.y, front_pos.x - robot_pos.x)
    angle_diff = abs(wrap_angle(angle_to_back - angle_to_front))
    
    # 2. front 的角寬度覆蓋 back
    front_angular_width = angular_width(robot_pos, front_pos, front_r)
    back_angular_width = angular_width(robot_pos, back_pos, back_r)
    
    return (angle_diff < front_angular_width / 2) and (front_angular_width > back_angular_width)
```

**連續性約束**:
```
# 當 back obstacle 從 occluded 恢復 visible 時：
expected_pos = last_visible_pos + last_visible_vel * hidden_duration
actual_pos = expected_pos + noise(sigma=0.1m)  # 小量漂移模擬真實不確定性
# 不允許: actual_pos 與 expected_pos 差距 > 0.5m
```

**Occlusion 排程**:
```
occlusion_schedule:
  visible_phase: min 8 frames (1.6s), max 25 frames (5.0s) — back target 可見
  occluded_phase: 5~20 frames (1.0~4.0s) — back target 被遮擋
  reveal_phase: back target 重新可見 (front 移開 or robot 移動)
```

---

## 3. BehaviorConfig Dataclass 草案

```python
@dataclass
class SafetyConstraints:
    """全域安全約束"""
    min_spawn_distance_from_robot: float = 2.0   # m
    min_spawn_distance_from_goal: float = 1.5    # m
    min_wall_clearance: float = 0.6              # m
    min_obstacle_obstacle_distance: float = 0.8  # m
    max_blocking_on_goal_line: int = 0           # steps, 0 = use behavior default
    reset_timeout: int = 750                     # steps
    min_visible_frames_before_occlusion: int = 8 # only for occlusion behavior


@dataclass
class StaticConfig:
    """靜態障礙物 — LiDAR baseline provider"""
    radius_range: tuple[float, float] = (0.20, 0.35)
    # override
    min_spawn_from_robot: float = 1.5
    max_blocking_on_goal_line: int = 25  # 5s — 允許短暫出現在 goal line


@dataclass
class PatrolConfig:
    """巡邏行為"""
    num_waypoints: tuple[int, int] = (2, 4)
    convex_hull_area_max: float = 9.0        # m^2
    speed_range: tuple[float, float] = (0.3, 0.6)  # m/s
    pause_steps_range: tuple[int, int] = (0, 5)
    mode_distribution: dict = field(default_factory=lambda: {"loop": 0.5, "ping_pong": 0.5})


@dataclass
class RandomWalkConfig:
    """隨機漫步"""
    speed_range: tuple[float, float] = (0.2, 0.7)    # m/s
    direction_change_interval: tuple[int, int] = (5, 25)  # steps
    max_turn_angle: float = 1.5708  # pi/2
    smooth_turn_steps: int = 3


@dataclass
class HorizontalCrossingConfig:
    """水平橫穿"""
    speed_range: tuple[float, float] = (0.5, 1.0)    # m/s
    offset_ahead_range: tuple[float, float] = (2.0, 6.0)  # m ahead of robot
    crossing_width_range: tuple[float, float] = (6.0, 12.0)  # m
    spawn_side_offset: tuple[float, float] = (3.0, 6.0)      # m perpendicular
    repeat_interval_steps: tuple[int, int] = (30, 60)
    min_spawn_from_robot: float = 3.0


@dataclass
class PathCrossingConfig:
    """路徑交叉"""
    speed_range: tuple[float, float] = (0.4, 0.9)
    crossing_angle_range: tuple[float, float] = (0.5236, 2.6180)  # 30~150 degrees
    ttc_to_path_range: tuple[float, float] = (1.5, 4.0)  # seconds
    spawn_perp_distance: tuple[float, float] = (2.0, 5.0)  # m
    min_distance_ahead: float = 2.5  # m
    min_spawn_from_robot: float = 3.5


@dataclass
class NearMissConfig:
    """近距離通過"""
    clearance_buckets: dict = field(default_factory=lambda: {
        "tight":  {"range": (0.10, 0.20), "weight": 0.30},
        "medium": {"range": (0.20, 0.40), "weight": 0.50},
        "loose":  {"range": (0.40, 0.60), "weight": 0.20},
    })
    ttc_visible_to_closest: tuple[float, float] = (2.0, 5.0)  # seconds
    speed_range: tuple[float, float] = (0.4, 1.0)  # m/s
    precomputed_trajectory: bool = True  # 軌跡不依賴 robot 動作
    min_spawn_from_robot: float = 2.5


@dataclass
class CorridorCrossingConfig:
    """走廊穿越"""
    speed_range: tuple[float, float] = (0.3, 0.7)
    min_gap_width: float = 1.5  # m — corridor 最小寬度
    max_gap_width: float = 4.0  # m
    spawn_offset_from_gap: tuple[float, float] = (1.0, 3.0)  # m
    repeat: bool = True  # ping-pong
    min_spawn_from_robot: float = 2.5


@dataclass
class OcclusionGroupConfig:
    """遮擋群組"""
    group_size: tuple[int, int] = (2, 3)  # 1 front + 1~2 back
    front_speed_range: tuple[float, float] = (0.2, 0.5)  # 慢速 blocker
    back_speed_range: tuple[float, float] = (0.3, 0.8)
    visible_frames_range: tuple[int, int] = (8, 25)  # 1.6s~5.0s
    occluded_frames_range: tuple[int, int] = (5, 20)  # 1.0s~4.0s
    position_noise_on_reveal: float = 0.1  # m — 再次出現時的位置偏差
    max_position_drift: float = 0.5        # m — 不允許超過此偏差
    min_obs_obs_distance: float = 0.5      # m — front/back pair 較近
    min_spawn_from_robot: float = 2.5


@dataclass
class BehaviorConfig:
    """完整 behavior 配置（8 behaviors + safety + stage mix）"""
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    static: StaticConfig = field(default_factory=StaticConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    random_walk: RandomWalkConfig = field(default_factory=RandomWalkConfig)
    horizontal_crossing: HorizontalCrossingConfig = field(default_factory=HorizontalCrossingConfig)
    path_crossing: PathCrossingConfig = field(default_factory=PathCrossingConfig)
    near_miss: NearMissConfig = field(default_factory=NearMissConfig)
    corridor_crossing: CorridorCrossingConfig = field(default_factory=CorridorCrossingConfig)
    occlusion: OcclusionGroupConfig = field(default_factory=OcclusionGroupConfig)
```

---

## 4. BehaviorScheduler Pseudo-code

```python
class BehaviorScheduler:
    """向量化 obstacle behavior 排程器
    
    負責：
    1. 根據 stage config 的 behavior_mix 為每個 env 的每個 obstacle 分配 behavior
    2. 每步向量化 step() 移動所有 obstacle
    3. Episode reset 時重新分配
    4. 產出 WandB 診斷指標
    """

    def __init__(self, stage_config: dict, num_envs: int, max_obstacles: int, device: str):
        """
        Args:
            stage_config: 來自 curriculum 的 stage dict (含 behavior_mix, speed params)
            num_envs: 並行環境數
            max_obstacles: 每個 env 最大 obstacle slots
            device: "cuda:0"
        """
        self.num_envs = num_envs
        self.max_obstacles = max_obstacles
        self.device = device
        self.cfg = BehaviorConfig()  # 讀取 stage_config 覆寫 defaults
        
        # === 核心狀態 Tensors ===
        # behavior_type: [num_envs, max_obstacles] — enum index (0=inactive, 1=static, ..., 8=occlusion)
        self.behavior_type = torch.zeros(num_envs, max_obstacles, dtype=torch.long, device=device)
        # positions: [num_envs, max_obstacles, 2] — XY
        self.positions = torch.zeros(num_envs, max_obstacles, 2, device=device)
        # velocities: [num_envs, max_obstacles, 2] — vx, vy
        self.velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)
        # phase_timer: [num_envs, max_obstacles] — 各 behavior 的 phase 內步數計數
        self.phase_timer = torch.zeros(num_envs, max_obstacles, dtype=torch.long, device=device)
        
        # === Behavior-specific state ===
        # patrol_waypoints: [num_envs, max_obstacles, max_waypoints=4, 2]
        self.patrol_waypoints = torch.zeros(num_envs, max_obstacles, 4, 2, device=device)
        self.patrol_current_wp = torch.zeros(num_envs, max_obstacles, dtype=torch.long, device=device)
        
        # near_miss_trajectories: [num_envs, max_obstacles, 2] — precomputed velocity
        self.near_miss_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
        
        # occlusion state
        self.occlusion_visible = torch.ones(num_envs, max_obstacles, dtype=torch.bool, device=device)
        self.occlusion_partner = torch.full((num_envs, max_obstacles), -1, dtype=torch.long, device=device)
        self.last_visible_pos = torch.zeros(num_envs, max_obstacles, 2, device=device)
        self.last_visible_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
        self.hidden_steps = torch.zeros(num_envs, max_obstacles, dtype=torch.long, device=device)
        
        # === Metrics accumulators ===
        self.metrics = BehaviorMetrics(num_envs, max_obstacles, device)


    def assign_behaviors(self, env_ids: Tensor, robot_pos: Tensor, goal_pos: Tensor,
                         wall_data: Tensor, behavior_mix: dict) -> None:
        """Per-env per-obstacle 分配行為
        
        Args:
            env_ids: [N] 需要分配的 env IDs (reset 時呼叫)
            robot_pos: [N, 2] 機器人 XY
            goal_pos: [N, 2] 目標 XY
            wall_data: [N, W, 4] per-env wall AABB data
            behavior_mix: dict {"static": 0.2, "patrol": 0.15, ...}
        """
        N = len(env_ids)
        num_active = behavior_mix.get("total_obstacles", self.max_obstacles)
        
        # 1. 計算每種 behavior 分配多少 obstacle
        #    e.g. behavior_mix = {"static": 0.2, "patrol": 0.15, "random_walk": 0.20, 
        #                         "horizontal_crossing": 0.10, "path_crossing": 0.10,
        #                         "near_miss": 0.10, "corridor_crossing": 0.05, "occlusion": 0.10}
        counts = allocate_counts(behavior_mix, num_active)
        # counts = {"static": 2, "patrol": 2, "random_walk": 2, ...}
        
        # 2. Sequential assignment (因為 occlusion group 需要連續 slots)
        slot_idx = 0
        for btype, count in counts.items():
            if btype == "occlusion":
                # Occlusion 佔 2~3 slots per group
                num_groups = count  # count 指 groups 數量
                for g in range(num_groups):
                    group_size = random_int(2, 3)
                    self._assign_occlusion_group(env_ids, slot_idx, group_size, robot_pos, goal_pos, wall_data)
                    slot_idx += group_size
            elif btype == "corridor_crossing":
                # 需要偵測 corridor gap
                corridors = detect_corridor_gaps(wall_data, min_width=1.5)
                for c in range(count):
                    if corridors is not empty:
                        self._assign_corridor(env_ids, slot_idx, corridors, robot_pos)
                    else:
                        # fallback to random_walk if no corridor
                        self._assign_random_walk(env_ids, slot_idx)
                    slot_idx += 1
            else:
                for c in range(count):
                    self._assign_single(env_ids, slot_idx, btype, robot_pos, goal_pos, wall_data)
                    slot_idx += 1
        
        # 3. 剩餘 slots 設為 inactive (Z = -10)
        for s in range(slot_idx, self.max_obstacles):
            self.behavior_type[env_ids, s] = 0  # inactive


    def step(self, env, dt: float = 0.2) -> None:
        """向量化移動所有 obstacle（每 env step 呼叫一次）
        
        核心：根據 behavior_type 用 masked tensor ops 移動，避免 Python loop over envs。
        """
        # 1. Static: 不動 (skip)
        
        # 2. Patrol: 向當前 waypoint 移動，到達則切換下一個
        patrol_mask = (self.behavior_type == 2)  # enum: patrol=2
        if patrol_mask.any():
            self._step_patrol(patrol_mask, dt)
        
        # 3. Random Walk: direction change timer check + smooth turn
        rw_mask = (self.behavior_type == 3)
        if rw_mask.any():
            self._step_random_walk(rw_mask, dt)
        
        # 4. Horizontal Crossing: linear motion along crossing direction
        hc_mask = (self.behavior_type == 4)
        if hc_mask.any():
            self._step_linear(hc_mask, dt)
        
        # 5. Path Crossing: linear motion
        pc_mask = (self.behavior_type == 5)
        if pc_mask.any():
            self._step_linear(pc_mask, dt)
        
        # 6. Near Miss: precomputed linear trajectory
        nm_mask = (self.behavior_type == 6)
        if nm_mask.any():
            self._step_near_miss(nm_mask, dt)
        
        # 7. Corridor Crossing: linear with boundary reflection
        cc_mask = (self.behavior_type == 7)
        if cc_mask.any():
            self._step_corridor(cc_mask, dt)
        
        # 8. Occlusion: move normally, update visibility state
        occ_mask = (self.behavior_type == 8)
        if occ_mask.any():
            self._step_occlusion(occ_mask, dt, env)
        
        # 9. Global: boundary clamp + wall proximity bounce
        self._enforce_boundaries()
        
        # 10. Write positions to sim
        self._write_to_sim(env)
        
        # 11. Increment phase_timer
        self.phase_timer += 1
        
        # 12. Update metrics
        self.metrics.update(self, env)


    def reset(self, env_ids: Tensor, env) -> None:
        """Episode reset 時重新分配 behaviors
        
        1. 清空 env_ids 的所有 behavior state
        2. 讀取當前 stage 的 behavior_mix
        3. 呼叫 assign_behaviors()
        """
        # Clear state
        self.behavior_type[env_ids] = 0
        self.positions[env_ids] = 0.0
        self.velocities[env_ids] = 0.0
        self.phase_timer[env_ids] = 0
        self.occlusion_visible[env_ids] = True
        self.hidden_steps[env_ids] = 0
        
        # Get current positions
        robot_pos = env.scene["robot"].data.root_pos_w[env_ids, :2]
        goal_pos = env.command_manager.get_command("goal_command")[env_ids, :2]
        wall_data = get_combined_wall_data(env)[env_ids]
        
        # Get behavior mix from current curriculum stage
        behavior_mix = self._get_current_behavior_mix()
        
        # Assign
        self.assign_behaviors(env_ids, robot_pos, goal_pos, wall_data, behavior_mix)


    def get_metrics(self) -> dict:
        """回傳 WandB 用指標 dict
        
        Returns:
            dict with keys like "behavior/static_count", "behavior/near_miss_avg_clearance", etc.
        """
        return self.metrics.compute_and_reset()


    # === Private helper methods ===
    
    def _step_patrol(self, mask: Tensor, dt: float):
        """Patrol: 向 current waypoint 移動
        
        active_envs, active_obs = mask.nonzero() — 找出所有 patrol obstacles
        target = patrol_waypoints[active_envs, active_obs, current_wp_idx]
        direction = normalize(target - positions[active_envs, active_obs])
        positions += direction * speed * dt
        
        # 到達 waypoint check
        dist_to_target = norm(target - positions)
        reached = dist_to_target < 0.3
        patrol_current_wp[reached] = (patrol_current_wp[reached] + 1) % num_waypoints
        """
        pass  # placeholder for pseudo-code

    def _step_random_walk(self, mask: Tensor, dt: float):
        """Random Walk: timer-based direction change + smooth turn
        
        # Check timer: if phase_timer[mask] % direction_change_interval == 0:
        #   sample new direction delta (within max_turn_angle)
        #   start smooth_turn countdown
        
        # During smooth turn: linearly interpolate heading
        # After turn: constant velocity motion
        
        positions[mask] += velocities[mask] * dt
        """
        pass

    def _step_near_miss(self, mask: Tensor, dt: float):
        """Near Miss: precomputed trajectory, no robot dependency
        
        positions[mask] += near_miss_vel[mask] * dt
        
        # 軌跡完成 check (obstacle 超出 LiDAR range):
        # if distance_from_spawn > trajectory_length:
        #   reset to spawn position (loop) or deactivate
        """
        pass

    def _step_occlusion(self, mask: Tensor, dt: float, env):
        """Occlusion group step:
        
        1. Move all obstacles in group normally
        2. For back obstacles: check if front obstacle still occluding (angular width check)
        3. Track visibility transitions:
           - visible → occluded: record last_visible_pos, last_visible_vel
           - occluded → visible: verify position matches extrapolation
        4. Update hidden_steps counter
        5. If hidden_steps > max_occluded_frames: force reveal (move front obstacle aside)
        """
        # Move all normally
        self.positions[mask] += self.velocities[mask] * dt
        
        # Compute occlusion state
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [num_envs, 2]
        
        for env_idx in mask_env_indices:
            for obs_idx in back_obstacle_indices:
                front_idx = self.occlusion_partner[env_idx, obs_idx]
                was_visible = self.occlusion_visible[env_idx, obs_idx]
                is_now_occluded = check_occlusion(
                    robot_pos[env_idx],
                    self.positions[env_idx, front_idx],
                    self.positions[env_idx, obs_idx],
                    obs_radius
                )
                
                if was_visible and is_now_occluded:
                    # Transition: visible → occluded
                    self.last_visible_pos[env_idx, obs_idx] = self.positions[env_idx, obs_idx]
                    self.last_visible_vel[env_idx, obs_idx] = self.velocities[env_idx, obs_idx]
                    self.hidden_steps[env_idx, obs_idx] = 0
                    # Verify min_visible_frames constraint was met
                    assert self.phase_timer[env_idx, obs_idx] >= MIN_VISIBLE_FRAMES
                
                elif not was_visible and not is_now_occluded:
                    # Transition: occluded → visible (reveal)
                    # Enforce continuity: actual position must match extrapolation
                    expected = self.last_visible_pos[env_idx, obs_idx] + \
                               self.last_visible_vel[env_idx, obs_idx] * self.hidden_steps[env_idx, obs_idx] * dt
                    actual = self.positions[env_idx, obs_idx]
                    assert (actual - expected).norm() < MAX_POSITION_DRIFT
                
                self.occlusion_visible[env_idx, obs_idx] = not is_now_occluded
                if is_now_occluded:
                    self.hidden_steps[env_idx, obs_idx] += 1

    def _assign_occlusion_group(self, env_ids, slot_start, group_size, robot_pos, goal_pos, wall_data):
        """分配 occlusion group
        
        1. slot_start = front blocker, slot_start+1..+n = back targets
        2. front blocker: 較大 radius (0.30~0.35), 慢速 (0.2~0.5)
        3. back target: 在 front 正後方 (相對 robot 視角), distance > front + buffer
        4. Initial state: back target 可見 (front 不在 LOS 上)
        5. front 的 trajectory 會在 visible_frames 後移到 LOS 上 (開始遮擋)
        """
        # front_pos: sample valid position (in front of robot, visible)
        # back_pos: behind front (relative to robot), initially offset so NOT occluded
        # front_vel: designed to move INTO occlusion position after min_visible_frames
        pass
```

---

## 5. Per-Env Behavior Assignment Pseudo-code

```python
def allocate_counts(behavior_mix: dict, total_slots: int) -> dict:
    """根據 behavior_mix 比例分配 obstacle slot 數量
    
    特殊處理:
    - occlusion: 每個 "count" 代表一個 group，實際佔 2~3 slots
    - corridor: 如果場景無 corridor gap，fallback to random_walk
    - 剩餘 slots 四捨五入分配
    
    Example:
        behavior_mix = {
            "static": 0.20,
            "patrol": 0.15,
            "random_walk": 0.20,
            "horizontal_crossing": 0.10,
            "path_crossing": 0.10,
            "near_miss": 0.10,
            "corridor_crossing": 0.05,
            "occlusion": 0.10,  # 比例指 group 數量
        }
        total_slots = 10
        
        # Step 1: 先分配 occlusion groups (因為佔多 slots)
        occlusion_groups = round(0.10 * total_slots / avg_group_size(2.5)) = 1 group
        occlusion_slots_used = 2~3
        remaining_slots = 10 - 3 = 7
        
        # Step 2: 按比例分配剩餘 slots
        remaining_mix = normalize(mix - occlusion)
        counts = {type: round(ratio * remaining_slots) for type, ratio in remaining_mix.items()}
        
        # Step 3: 校正 — 確保總數 == total_slots
        adjust_largest_bucket_to_match_total()
        
        return {"static": 2, "patrol": 1, "random_walk": 2, 
                "horizontal_crossing": 1, "path_crossing": 1, 
                "near_miss": 0, "corridor_crossing": 0, "occlusion": 1(group)}
    """
    pass


def detect_corridor_gaps(wall_data: Tensor, min_width: float = 1.5) -> list:
    """偵測 per-env 的 corridor gaps
    
    演算法:
    1. 對所有 wall pairs，找出 parallel walls (angle diff < 10 degrees)
    2. 計算 perpendicular distance between parallel walls
    3. 如果 distance in [min_width, max_gap_width]: 這是一個 corridor
    4. 也檢查 wall endpoint gaps:
       - wall end 到另一牆壁的垂直距離 in [min_width, max_gap_width]
       - 加上 boundary walls 形成的 corridors
    
    Returns:
        list of CorridorGap(center_xy, direction_vec, width, length)
    """
    pass
```

---

## 6. WandB Logging Pseudo-code

```python
@dataclass
class BehaviorMetrics:
    """Behavior 系統診斷指標收集器"""
    
    def __init__(self, num_envs, max_obstacles, device):
        # Per-behavior counters
        self.behavior_counts = torch.zeros(9, device=device)  # 0=inactive, 1~8 behaviors
        
        # Near-miss specific
        self.near_miss_clearances = []        # 實際 surface clearance 紀錄
        self.near_miss_robot_decelerated = 0  # robot 主動減速次數
        self.near_miss_robot_turned = 0       # robot 主動轉向次數
        self.near_miss_robot_waited = 0       # robot 停等次數
        self.near_miss_total = 0
        
        # Occlusion specific
        self.occlusion_hidden_durations = []  # 遮擋持續時間紀錄
        self.occlusion_prediction_errors = [] # 重現時 pos drift (m)
        self.occlusion_groups_active = 0
        
        # Collision by behavior type
        self.collisions_by_type = torch.zeros(9, device=device)
        
        # Blocking violation counter
        self.blocking_violations = 0
    
    def update(self, scheduler, env):
        """每步更新指標
        
        1. Count active behaviors by type
        2. For near_miss: check if closest point reached this step
           - If yes: record actual_clearance, robot actions
        3. For occlusion: track hidden duration
        4. Check blocking violations (obstacle on goal line too long)
        """
        # 1. Behavior counts
        for btype in range(1, 9):
            self.behavior_counts[btype] = (scheduler.behavior_type == btype).sum()
        
        # 2. Near-miss clearance tracking
        nm_mask = (scheduler.behavior_type == 6)
        if nm_mask.any():
            robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
            obs_pos = scheduler.positions
            # Compute surface distance = center_dist - robot_r - obs_r
            center_dist = (obs_pos[nm_mask] - robot_pos.unsqueeze(1).expand_as(obs_pos)[nm_mask]).norm(dim=-1)
            surface_dist = center_dist - ROBOT_RADIUS - OBS_RADIUS
            # Record minimum per near-miss event
            # (only record when obstacle is at closest approach)
        
        # 3. Blocking check
        # project obstacle onto robot-goal line, check if distance < threshold
        # if blocking_steps > max_allowed: increment violation counter
    
    def compute_and_reset(self) -> dict:
        """計算 episode-level 指標並重置 accumulators
        
        Returns:
            {
                "behavior/static_count_mean": float,
                "behavior/patrol_count_mean": float,
                "behavior/random_walk_count_mean": float,
                "behavior/horizontal_crossing_count_mean": float,
                "behavior/path_crossing_count_mean": float,
                "behavior/near_miss_count_mean": float,
                "behavior/corridor_crossing_count_mean": float,
                "behavior/occlusion_count_mean": float,
                
                "behavior/near_miss_avg_clearance": float,  # m
                "behavior/near_miss_tight_ratio": float,    # tight bucket 比例
                "behavior/near_miss_robot_decel_rate": float,
                "behavior/near_miss_robot_turn_rate": float,
                "behavior/near_miss_robot_wait_rate": float,
                
                "behavior/occlusion_avg_hidden_duration": float,  # steps
                "behavior/occlusion_avg_prediction_error": float, # m
                "behavior/occlusion_groups_per_env": float,
                
                "behavior/collision_by_static": float,
                "behavior/collision_by_patrol": float,
                "behavior/collision_by_random_walk": float,
                "behavior/collision_by_crossing": float,  # h+p combined
                "behavior/collision_by_near_miss": float,
                "behavior/collision_by_corridor": float,
                "behavior/collision_by_occlusion": float,
                
                "behavior/blocking_violations": int,
            }
        """
        result = {}
        # ... compute averages ...
        # Reset accumulators
        self.near_miss_clearances.clear()
        self.occlusion_hidden_durations.clear()
        self.occlusion_prediction_errors.clear()
        self.collisions_by_type.zero_()
        self.blocking_violations = 0
        return result
```

---

## 7. Curriculum Config 草案（6 Stage）

```python
BEHAVIOR_CURRICULUM = {
    "version": "rule_based_obs_v1",
    "description": "Rule-based obstacle behavior curriculum for RNN aux training",
    
    "stages": [
        # === Stage 1: 基礎導航 + 靜態 baseline ===
        {
            "name": "static_only",
            "total_obstacles": 3,
            "behavior_mix": {
                "static": 1.0,
            },
            "speed_overrides": {},  # 全靜態
            "safety_overrides": {
                "min_spawn_distance_from_robot": 2.5,  # 寬鬆 spawn
            },
            "upgrade_conditions": {
                "min_sr": 0.70,
                "max_cr": 0.15,
                "min_updates": 50,
            },
            "gamma": 0.990,
            "episode_length_s": 45,
        },
        
        # === Stage 2: 引入可預測動態 ===
        {
            "name": "predictable_dynamic",
            "total_obstacles": 5,
            "behavior_mix": {
                "static": 0.40,       # 2 static
                "patrol": 0.40,       # 2 patrol (可預測)
                "random_walk": 0.20,  # 1 random walk (微量不可預測)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.2, 0.4)},       # 慢速巡邏
                "random_walk": {"speed_range": (0.15, 0.4)}, # 慢速隨機
            },
            "safety_overrides": {},
            "upgrade_conditions": {
                "min_sr": 0.65,
                "max_cr": 0.20,
                "min_updates": 65,
            },
            "gamma": 0.992,
            "episode_length_s": 51,
        },
        
        # === Stage 3: 引入 crossing + 速度提升 ===
        {
            "name": "crossing_intro",
            "total_obstacles": 7,
            "behavior_mix": {
                "static": 0.28,               # 2 static
                "patrol": 0.28,               # 2 patrol
                "random_walk": 0.14,          # 1 random walk
                "horizontal_crossing": 0.15,  # 1 crossing
                "path_crossing": 0.15,        # 1 path crossing
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.3, 0.6)},
                "random_walk": {"speed_range": (0.2, 0.6)},
                "horizontal_crossing": {"speed_range": (0.4, 0.7)},
                "path_crossing": {"speed_range": (0.3, 0.7)},
            },
            "safety_overrides": {},
            "upgrade_conditions": {
                "min_sr": 0.60,
                "max_cr": 0.25,
                "min_updates": 80,
            },
            "gamma": 0.994,
            "episode_length_s": 56,
        },
        
        # === Stage 4: 引入 near-miss + corridor ===
        {
            "name": "tight_clearance",
            "total_obstacles": 8,
            "behavior_mix": {
                "static": 0.125,               # 1
                "patrol": 0.25,                # 2
                "random_walk": 0.125,          # 1
                "horizontal_crossing": 0.125,  # 1
                "path_crossing": 0.125,        # 1
                "near_miss": 0.125,            # 1
                "corridor_crossing": 0.125,    # 1
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.3, 0.7)},
                "random_walk": {"speed_range": (0.3, 0.7)},
                "horizontal_crossing": {"speed_range": (0.5, 0.9)},
                "path_crossing": {"speed_range": (0.4, 0.8)},
                "near_miss": {"speed_range": (0.4, 0.8)},
                "corridor_crossing": {"speed_range": (0.3, 0.6)},
            },
            "safety_overrides": {
                "near_miss": {
                    "clearance_distribution_override": {
                        "tight": 0.15,   # Stage 4: less tight
                        "medium": 0.55,
                        "loose": 0.30,
                    }
                }
            },
            "upgrade_conditions": {
                "min_sr": 0.55,
                "max_cr": 0.30,
                "min_updates": 100,
            },
            "gamma": 0.995,
            "episode_length_s": 65,
        },
        
        # === Stage 5: 引入 occlusion + 全速 ===
        {
            "name": "occlusion_and_full_speed",
            "total_obstacles": 10,
            "behavior_mix": {
                "static": 0.10,                # 1
                "patrol": 0.15,                # 1~2
                "random_walk": 0.15,           # 1~2
                "horizontal_crossing": 0.10,   # 1
                "path_crossing": 0.10,         # 1
                "near_miss": 0.15,             # 1~2
                "corridor_crossing": 0.10,     # 1
                "occlusion": 0.15,             # 1 group (2~3 slots)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.4, 0.8)},
                "random_walk": {"speed_range": (0.3, 0.8)},
                "horizontal_crossing": {"speed_range": (0.5, 1.0)},
                "path_crossing": {"speed_range": (0.4, 0.9)},
                "near_miss": {"speed_range": (0.5, 1.0)},
                "corridor_crossing": {"speed_range": (0.3, 0.7)},
            },
            "safety_overrides": {
                "near_miss": {
                    "clearance_distribution_override": {
                        "tight": 0.25,
                        "medium": 0.50,
                        "loose": 0.25,
                    }
                }
            },
            "upgrade_conditions": {
                "min_sr": 0.50,
                "max_cr": 0.35,
                "min_updates": 120,
            },
            "gamma": 0.997,
            "episode_length_s": 75,
        },
        
        # === Stage 6: 壓力測試 — 全 behavior + tight clearance + 高速 ===
        {
            "name": "full_pressure",
            "total_obstacles": 10,
            "behavior_mix": {
                "static": 0.10,                # 1
                "patrol": 0.10,                # 1
                "random_walk": 0.15,           # 1~2
                "horizontal_crossing": 0.10,   # 1
                "path_crossing": 0.10,         # 1
                "near_miss": 0.20,             # 2 — 壓力測試重點
                "corridor_crossing": 0.10,     # 1
                "occlusion": 0.15,             # 1 group (2~3 slots)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.5, 0.85)},
                "random_walk": {"speed_range": (0.4, 0.85)},
                "horizontal_crossing": {"speed_range": (0.6, 1.0)},
                "path_crossing": {"speed_range": (0.5, 1.0)},
                "near_miss": {"speed_range": (0.6, 1.0)},
                "corridor_crossing": {"speed_range": (0.4, 0.8)},
            },
            "safety_overrides": {
                "min_obstacle_obstacle_distance": 0.7,  # 稍密集
                "near_miss": {
                    "clearance_distribution_override": {
                        "tight": 0.30,   # Full target: 30% tight
                        "medium": 0.50,
                        "loose": 0.20,
                    }
                }
            },
            "upgrade_conditions": None,  # Terminal stage
            "gamma": 0.998,
            "episode_length_s": 90,
        },
    ],
    
    # === 全域 curriculum 參數 ===
    "global": {
        "downgrade_sr": 0.15,           # SR < 15% 觸發降階
        "downgrade_min_cr": 0.70,       # CR > 70% 觸發降階
        "clear_window_on_promote": True,
        "stage_metric_window": 100,     # 滑動窗口大小 (updates)
    }
}
```

---

## 8. Contribution 語氣修正

以下是本設計的定位（用 investigate/evaluate/compare 語氣）：

### 8.1 Research Questions

1. **Investigate** whether structured obstacle behaviors (vs. uniform random walk) improve RNN auxiliary prediction accuracy and downstream collision avoidance performance.

2. **Evaluate** the effect of pre-computed near-miss trajectories on policy's proactive deceleration rate vs. baseline (reactive-only avoidance).

3. **Compare** occlusion-aware training (with min_visible_frames constraint) against unconstrained random occlusion in terms of RNN hidden state prediction fidelity.

4. **Propose** a curriculum schedule that introduces behavior complexity incrementally, measuring whether each stage transition preserves success rate within 5pp of pre-transition level.

### 8.2 Validation Plan (Draft)

| Metric | Baseline (random walk) | Target (rule-based) | Measurement |
|--------|:---:|:---:|------|
| WD Aux MSE (2-nearest) | TBD | < 0.8x baseline | WandB `aux/mse_loss` |
| Near-miss proactive decel rate | TBD | > 40% | `behavior/near_miss_robot_decel_rate` |
| Occlusion prediction error | TBD | < 0.3m | `behavior/occlusion_avg_prediction_error` |
| Stage 6 SR | TBD | > 50% | Standard eval |
| Stage 6 CR | TBD | < 30% | Standard eval |

### 8.3 Risk Assessment

| 風險 | 嚴重度 | 緩解策略 |
|------|--------|---------|
| Near-miss 的 precomputed trajectory 被 robot 快速移動打破假設 | 中 | Clearance 計算加入 robot_max_speed * ttc 的保守 buffer |
| Occlusion group 佔 2~3 slots，導致其他 behavior slot 不足 | 低 | Stage 5/6 才啟用，且限制最多 1 group per env |
| Corridor detection 在隨機 wall layout 中找不到 corridor | 中 | Fallback to random_walk；或特定 stage 的 wall layout 保證至少 1 corridor |
| 向量化困難：不同 behavior 的 step 邏輯差異大 | 中 | 用 behavior_type mask 分組 dispatch；每組內 tensor ops 向量化 |
| Phase timer overflow (長 episode) | 低 | 用 modulo 或 clamp，各 behavior 有自己的 period |

---

## 9. 實作優先序建議

| Priority | Component | 預估工時 | 依賴 |
|:---:|------|:---:|------|
| P0 | `BehaviorConfig` dataclass + `SafetyConstraints` | 2h | 無 |
| P0 | `BehaviorScheduler.__init__` + `reset` + `assign_behaviors` (static + random_walk only) | 4h | P0 dataclass |
| P1 | `step()` for static + random_walk + patrol | 3h | P0 scheduler |
| P1 | Collision check integration (reuse existing `check_wall_proximity_perenv`) | 2h | P0 |
| P2 | Horizontal/Path crossing step logic | 3h | P1 |
| P2 | Near-miss precomputed trajectory + logging | 4h | P1 |
| P3 | Corridor detection + corridor crossing | 3h | P2 |
| P3 | Occlusion group assignment + visibility tracking | 5h | P2 |
| P4 | WandB metrics integration | 2h | P2 |
| P4 | Curriculum config integration (6 stages) | 2h | all above |

Total estimate: ~30h (分 4 weeks，P0+P1 first week)。

---

## 10. 與現有系統的整合點

| 現有模組 | 整合方式 |
|---------|---------|
| `events/obstacles.py::reset_obstacles` | 被 `BehaviorScheduler.reset()` 取代 |
| `events/obstacles.py::move_obstacles_goal_directed` | 被 `BehaviorScheduler.step()` 取代 |
| `events/mixed_parallel.py::randomize_obstacles_by_difficulty` | 修改為呼叫 scheduler.assign_behaviors()，保留 empty/static/dynamic ratio logic |
| `wall_layout.py::check_wall_proximity_perenv` | 直接 reuse，不修改 |
| `wall_layout.py::check_los_perenv` | Occlusion behavior 使用，不修改 |
| `observations/obs_functions.py::topk_obstacles_body_frame` | 不修改 — scheduler 只負責 position/velocity，obs 函數讀取 sim state |
| `curriculum/goal_obstacle_curriculum.py` | 新增 `behavior_mix` field per stage；現有 `num_obstacles_static/dynamic` 改用 scheduler 計算 |
| `train_charge_ac.py` (或 `train_rnn_car.py`) | 新增 `--behavior_mode rule_based` CLI flag，默認 `random_walk`(backward compatible) |

---

以上是完整的 Rule-Based Obstacle Behavior Agent 設計規格。所有數值基於以下物理參數推導：
- robot_radius = 0.35m, obs_radius = 0.20~0.35m
- dt = 0.2s, v_max_robot = 1.0 m/s
- LiDAR: 72 bins, 5 degree/bin, 360 degree, r_max = 20m
- Scene: 20x20m with 1.0m thick boundary walls (effective interior: 18x18m)
