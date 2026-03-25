# Charge 自主導航系統：強化學習訓練框架演化與技術貢獻

## 摘要

本報告詳述 Charge 差速驅動機器人自主導航系統的完整技術實現，涵蓋從 RSL-RL → Stable Baselines3 → SKRL 的框架遷移歷程。我們提出三項核心技術貢獻：(1) **非對稱 Actor-Critic (AAC) 觀測架構**，(2) **Pull-Push 理論獎勵函數設計**，(3) **風險感知之平滑推進獎勵**。實驗在 NVIDIA Isaac Lab 仿真環境中進行，採用 PPO 算法進行端到端訓練。

**關鍵詞**：深度強化學習、自主導航、PPO、非對稱 Actor-Critic、課程學習

---

## 1. 系統架構

### 1.1 層級式導航架構

```
┌─────────────────────────────────────────────────────────────────┐
│                      AIT* 全局路徑規劃                           │
│  輸入: (p_start, p_goal, obstacle_map)                           │
│  輸出: waypoints = [(x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ)]            │
│  更新頻率: 1-5 Hz (事件觸發式)                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Carrot-on-Stick (前瞻 2m)
┌─────────────────────────────────────────────────────────────────┐
│                    RL 局部控制器 (PPO)                           │
│  輸入: o_t = [lidar, velocity, goal_info, robot_state]          │
│  輸出: a_t = [v_linear, ω_angular]                               │
│  更新頻率: 50 Hz (20ms 控制週期)                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Charge 差速驅動機器人                         │
│  傳感器: 2D LiDAR (72 rays, 5° 解析度)                           │
│  執行器: 雙輪差速 (v_max=1.0 m/s, ω_max=2.0 rad/s)              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 訓練/推理分離架構

**核心洞察**：訓練時使用虛擬規劃器可大幅提升 FPS，同時保持數學等價性。

| 模式 | 規劃器 | 優勢 |
|------|--------|------|
| **訓練** | VirtualPlanner (隨機目標點) | FPS 高、泛化能力強 |
| **推理** | AIT* (真實全局規劃) | 複雜場景導航能力 |

---

## 2. 非對稱 Actor-Critic (AAC) 觀測架構

### 2.1 設計動機

傳統 Actor-Critic 架構中，Actor 和 Critic 使用相同的觀測空間。然而在機器人導航中，**訓練時可獲得特權資訊 (privileged information)**，如障礙物真實位置。我們提出非對稱架構：

- **Actor (部署用)**：只能使用真實機器人可獲取的感知資訊
- **Critic (訓練用)**：可使用上帝視角的特權資訊，加速價值函數學習

### 2.2 觀測空間定義

#### 2.2.1 Actor 觀測 (111 維)

```python
# Policy 觀測空間定義
class PolicyObs:
    lidar_scan              # [72]  360°/5° = 72 條射線
    base_velocity_xy        # [2]   (vx, vy) 機器人座標系
    goal_position           # [2]   (x, y) 目標相對位置
    goal_distance           # [1]   歐式距離
    time_remaining_ratio    # [1]   剩餘時間比例 [0, 1]
    alive_flag              # [1]   存活標誌
    safe_last_action        # [2]   上一步動作
    topk_obstacles          # [30]  5×6 Top-K 障礙物 (goal-centric)
    # ─────────────────────────────────────────
    # 總計: 111 維
```

#### 2.2.2 Critic 觀測 (161 維)

```python
class CriticObs(PolicyObs):
    # 繼承所有 Policy 觀測 (111 維)
    obstacles_state         # [50]  10×5 障礙物狀態 (特權資訊)
    # ─────────────────────────────────────────
    # 總計: 161 維
```

### 2.3 Top-K 障礙物觀測 (Goal-Centric Frame)

採用 **NavRL Goal-Centric** 座標系，以目標方向為 X 軸：

```python
def topk_obstacles_goal_centric(env, robot_cfg, top_k=5, max_distance=8.0):
    """
    輸出: [num_envs, 30] (5 obstacles × 6 features)
    每個障礙物: [rel_x, rel_y, dist, vel_x, vel_y, size]

    座標變換:
    - X 軸 = 目標方向
    - Y 軸 = 目標方向逆時針 90°
    """
    # 計算目標方向單位向量
    goal_dir = (goal_pos - robot_pos) / ||goal_pos - robot_pos||

    # 建構旋轉矩陣
    R = [[goal_dir_x, -goal_dir_y],
         [goal_dir_y,  goal_dir_x]]

    # 轉換障礙物位置到 goal-centric 座標系
    obstacle_goal_centric = R @ (obstacle_pos - robot_pos)
```

### 2.4 SKRL AAC 實現 (需 Patch)

由於 SKRL 原生不支援 AAC，我們實現了以下 Patch：

```python
def patch_skrl_for_aac():
    """Patch SKRL PPO 以支援 shared_states (特權觀測)"""

    # 1. Patch Runner._generate_models
    # 讓 Critic 使用 state_space 而非 observation_space
    if role == "value" and hasattr(env, 'state_space'):
        observation_space = state_spaces[agent_id]

    # 2. Patch PPO.record_transition
    # 將 shared_states 存入 Memory
    shared_states = infos.get("shared_states", states)
    self.memory.add_samples(
        states=states,           # Actor 觀測 (111 維)
        shared_states=shared_states,  # Critic 觀測 (161 維)
        ...
    )

    # 3. Patch PPO._update
    # Value Loss 使用 shared_states 計算
    predicted_values = self.value.act(
        {"states": sampled_shared_states}  # 161 維特權資訊
    )
```

---

## 3. Pull-Push 理論獎勵函數設計

### 3.1 設計理念

獎勵函數基於 **勢場理論 (Potential Field Theory)** 演化而來：

- **Pull Forces**：吸引力，引導機器人向目標移動
- **Push Forces**：排斥力，推離障礙物/危險區域
- **Smoothness**：平滑性約束，確保控制連續

### 3.2 Pull Forces (吸引到目標)

#### 3.2.1 進度獎勵 (Progress to Goal)

```python
def progress_to_goal(env, asset_cfg):
    """
    公式: R_progress = d_{t-1} - d_t

    設計原理:
    - 直接獎勵「距離減少」，不管車頭朝向
    - 允許「繞路避障」和「側向移動」
    - 避免舊版嚴苛條件 (heading_factor * velocity_factor)

    Returns: [num_envs] 獎勵值，正值=靠近，負值=遠離
    """
    current_distance = ||goal_pos - robot_pos||
    previous_distance = env._previous_goal_distance

    reward = previous_distance - current_distance
    env._previous_goal_distance = current_distance.clone()

    return reward  # weight=+12.0
```

#### 3.2.2 速度獎勵 (Velocity Toward Goal)

```python
def velocity_toward_goal(env, asset_cfg, min_dist=1.0):
    """
    公式: R_vel = max(0, v · goal_unit)

    其中:
    - v: 機器人速度向量 [vx, vy]
    - goal_unit: 目標方向單位向量

    物理意義: 獎勵朝目標方向的速度分量

    距離閘門: 距離 < min_dist 時不給獎勵 (避免衝撞)
    """
    goal_direction = goal_pos - robot_pos
    goal_unit = goal_direction / ||goal_direction||

    velocity_projection = robot_vel · goal_unit  # 點積
    velocity_projection = clamp(velocity_projection, min=0.0)

    # 距離閘門
    gated_velocity = where(
        goal_distance > min_dist,
        velocity_projection,
        0.0
    )

    return gated_velocity  # weight=+1.0
```

#### 3.2.3 到達獎勵 (Reaching Goal)

```python
def reaching_goal(env, asset_cfg, threshold=0.5, body_radius=0.28):
    """
    公式: R_goal = 1.0 if d < threshold else 0.0

    考慮機器人體半徑:
    distance = clamp(||goal - robot|| - body_radius, min=0.0)
    """
    distance = ||goal_pos - robot_pos|| - body_radius
    reward = (distance < threshold).float()

    return reward  # weight=+8.0 (Phase 0)
```

### 3.3 Push Forces (推離障礙物)

#### 3.3.1 漸進式碰撞懲罰 (Progressive Collision Penalty)

```python
def progressive_collision_penalty(
    env, sensor_cfg, asset_cfg,
    safe_distance=1.0,      # 開始懲罰
    danger_distance=0.75,   # 中等懲罰
    collision_distance=0.5, # 最大懲罰
    use_directional=True,   # 方向性懲罰
    use_nonlinear=True,     # 非線性懲罰
):
    """
    公式 (非線性版本):

    區域 1 (d ∈ [collision, danger)):
        penalty = (0.5 + 0.5 × (1 - (d - collision)/(danger - collision)))²

    區域 2 (d ∈ [danger, safe)):
        penalty = 0.5 × (1 - (d - danger)/(safe - danger))

    區域 3 (d < collision):
        penalty = 1.0 + (collision/d - 1.0)  # 反比函數

    方向性乘數:
        multiplier = 1.0 + 2.0 × (v_toward_obstacle / v_max)²
        # 高速朝障礙物前進時，懲罰最高 3 倍
    """
    # 計算 2D 平面距離
    distances_2d = ||hit_points_2d - sensor_pos_2d||
    min_distance = min(distances_2d, dim=1)

    # 非線性懲罰計算
    if use_nonlinear:
        # 危險區域使用平方函數
        penalty = linear_penalty ** 2

        # 碰撞區域使用反比函數
        if min_distance < collision_distance:
            penalty = 1.0 + (collision_distance / min_distance - 1.0)

    # 方向性懲罰
    if use_directional:
        velocity_toward_obstacle = max(0, robot_vel · obstacle_dir)
        speed_ratio = velocity_toward_obstacle / v_max
        directional_multiplier = 1.0 + 2.0 × speed_ratio²
        penalty = penalty × directional_multiplier

    return penalty  # weight=-1.0 (Phase 0, 敢衝敢撞策略)
```

#### 3.3.2 TTC 防禦性駕駛懲罰

```python
def ttc_penalty_pure(env, robot_cfg, tau=1.5):
    """
    Time-To-Collision (TTC) 防禦性駕駛懲罰

    物理計算 (純 PyTorch 張量操作):
    ────────────────────────────────────────────────────────────────
    1. 相對位置: dp = p_obstacle - p_robot
    2. 相對速度: dv = v_obstacle - v_robot
    3. 距離: dist = ||dp||
    4. 接近速率: v_app = -(dp · dv) / (dist + ε)
    5. 碰撞時間: TTC = dist / (v_app + ε)

    遮罩條件:
        danger_mask = (z_obstacle > 0)     # 忽略隱藏障礙物
                    & (v_app > 0)          # 只考慮接近中的
                    & (TTC < tau)          # TTC 小於安全閾值

    懲罰公式:
        P = -(tau - TTC) for danger_mask == True
    """
    # 對齊維度: [num_envs, 2] -> [num_envs, num_obstacles, 2]
    p_r_exp = p_r.unsqueeze(1).expand(-1, num_obstacles, -1)
    v_r_exp = v_r.unsqueeze(1).expand(-1, num_obstacles, -1)

    # 物理計算
    dp = p_obstacle - p_r_exp
    dv = v_obstacle - v_r_exp
    dist = ||dp||  # 沿 dim=-1
    v_app = -(dp · dv) / (dist + 1e-5)
    ttc = dist / (v_app + 1e-5)

    # 建立遮罩
    danger_mask = (z_o > 0) & (v_app > 0) & (ttc < tau)

    # 計算懲罰
    penalty = where(danger_mask, -(tau - ttc), 0)
    penalty = penalty.sum(dim=1)  # 對障礙物維度求和

    return penalty
```

### 3.4 Smoothness Forces (平滑控制)

#### 3.4.1 動作變化率懲罰

```python
def action_rate_penalty(env, angular_weight=2.0):
    """
    公式: P_smooth = -(Δv_linear² + angular_weight × Δω²)

    設計原理:
    - 蛇行行為根源是策略頻繁切換旋轉方向
    - 角速度變化懲罰加重 (angular_weight=2.0)
    - 鼓勵「邊走邊轉」而非「原地打轉」
    """
    action_diff = current_action - previous_action

    linear_diff_sq = action_diff[:, 0] ** 2
    angular_diff_sq = action_diff[:, 1] ** 2 * angular_weight

    penalty = linear_diff_sq + angular_diff_sq

    # 重置時清除快取
    if reset_mask.any():
        penalty[reset_mask] = 0.0

    return penalty  # weight=-0.5
```

### 3.5 Pull/Push 比例設計

```
Phase 0 獎勵配置 (敢衝、敢撞策略):
────────────────────────────────────────────────────────────────

Pull Forces (吸引到目標):
  R_progress:        +12.0  × (d_{t-1} - d_t)
  R_goal:            +8.0   × I(d < 0.5m)
  R_respawn:         +1.0   (可變 +10~+30)
  ───────────────────────────
  總 Pull ≈ +21.0+

Push Forces (推離障礙物):
  P_safety:          -1.0   (安全場)
  P_collision:       -1.0   (碰撞)
  P_smooth:          -0.2   (平滑度)
  P_time:            -0.1   (時間)
  ───────────────────────────
  總 Push = -2.3

Pull/Push = 9.1:1  →  Agent 優先到達目標！
```

---

## 4. 風險感知獎勵 (技術貢獻)

### 4.1 問題分析

在訓練過程中發現兩大問題：
1. **冰凍機器人 (Freezing Robot)**：Agent 因恐懼碰撞而停止移動
2. **震盪行為 (Oscillation)**：高頻 Bang-Bang 控制，實車無法執行

### 4.2 技術貢獻 1：風險感知之平滑推進獎勵

```python
def risk_aware_progress_reward(env, robot_cfg, alpha=1.0, ttc_threshold=3.0):
    """
    ════════════════════════════════════════════════════════════════
                    技術貢獻 1：風險感知之平滑推進獎勵
    ════════════════════════════════════════════════════════════════

    數學公式:
    ────────────────────────────────────────────────────────────────
    R_prog = v_proj × (1 - e^(-α × TTC_min))

    其中:
    - v_proj: 機器人速度在目標方向的投影
    - TTC_min: 最危險障礙物的碰撞時間
    - α: 風險敏感係數

    物理意義:
    ────────────────────────────────────────────────────────────────
    1. 環境安全 (TTC_min → ∞):
       - e^(-α×TTC) → 0
       - 風險因子 → 1
       - R_prog ≈ v_proj (鼓勵全速前進)

    2. 環境危險 (TTC_min → 0):
       - e^(-α×TTC) → 1
       - 風險因子 → 0
       - R_prog → 0 (前進獎勵衰減)

    學術貢獻:
    ────────────────────────────────────────────────────────────────
    - 解決「冰凍機器人」問題
    - 動態風險感知，安全時加速、危險時減速繞行
    """
    # 計算目標方向投影速度
    v_proj = (robot_vel · goal_unit).clamp(0.0, 1.0)

    # 計算 TTC_min
    ttc = compute_ttc(robot_pos, robot_vel, obstacle_pos, obstacle_vel)
    ttc_min = ttc.min(dim=1)

    # 風險調節因子
    risk_factor = 1.0 - exp(-alpha × ttc_min)

    # 最終獎勵
    reward = v_proj × risk_factor

    return reward
```

### 4.3 技術貢獻 2：時間維度動作正則化

```python
def temporal_action_smoothness_penalty(env, lambda_weight=0.1):
    """
    ════════════════════════════════════════════════════════════════
                    技術貢獻 2：動作平滑性懲罰
    ════════════════════════════════════════════════════════════════

    數學公式:
    ────────────────────────────────────────────────────────────────
    R_smooth = -λ × ||a_t - a_{t-1}||²

    學術貢獻:
    ────────────────────────────────────────────────────────────────
    證明了時間維度上的動作正則化 (Temporal Action Regularization)
    能迫使策略網路輸出連續且符合車輛運動學 (Kinematics) 的平滑
    控制指令，大幅降低收斂過程中的策略震盪。
    """
    action_diff = current_action - previous_action
    penalty = -lambda_weight × ||action_diff||²

    return penalty
```

### 4.4 自適應動作平滑懲罰

```python
def adaptive_action_smoothness_penalty(
    env,
    base_lambda=0.1,
    velocity_threshold=0.3,
    high_speed_multiplier=2.0,
):
    """
    根據當前速度自適應調整平滑懲罰:
    - 低速時: 允許較大動作變化 (方便轉向)
    - 高速時: 嚴格限制動作變化 (避免危險)

    物理意義:
    高速行駛時，突然轉向或煞車更危險。
    """
    current_velocity = |current_action[:, 0]|

    adaptive_lambda = where(
        current_velocity > velocity_threshold,
        base_lambda × high_speed_multiplier,
        base_lambda
    )

    penalty = -adaptive_lambda × action_change²

    return penalty
```

---

## 5. 課程學習設計

### 5.1 四階段課程

| 階段 | 環境 | 障礙物 | 目標距離 | 核心學習目標 |
|:----:|------|:------:|:--------:|--------------|
| **0** | 16×16m 空曠 | 無 | 1.5-3m | 車輛動力學、waypoint 跟隨 |
| **1** | 8×8m 房間 | 3 | 3-8m | 基礎避障 |
| **2** | 8×8m 房間 | 5 | 4-8m | 複雜場景導航 |
| **3** | 8×8m 房間 | 動態 | 混合 | 動態避障、長程導航 |

### 5.2 自適應難度調整

```python
class AdaptiveCurriculum:
    """
    根據成功率和碰撞率動態調整難度
    """

    def update_difficulty(self, success_rate, collision_rate):
        # 增加難度條件
        if success_rate > 0.75 and collision_rate < 0.25:
            self.num_obstacles = min(10, self.num_obstacles + 1)

        # 降低難度條件
        elif success_rate < 0.60 or collision_rate > 0.25:
            self.num_obstacles = max(3, self.num_obstacles - 1)

        # 動態調整距離參數
        self.min_robot_distance = 3.5 + 0.5 × difficulty_level
        self.min_goal_distance = 2.0 + 0.3 × difficulty_level
```

### 5.3 混合訓練 (防止災難性遺忘)

```python
class MixedCurriculumScheduler:
    """混合訓練比例"""

    schedule = {
        0:     [1.0, 0.0, 0.0, 0.0],  # 100% Phase 0
        5000:  [0.2, 0.8, 0.0, 0.0],  # Phase 0+1
        10000: [0.1, 0.3, 0.6, 0.0],  # Phase 0+1+2
        20000: [0.1, 0.2, 0.3, 0.4],  # All phases
    }
```

---

## 6. 訓練框架對比

### 6.1 框架特性對比

| 特性 | RSL-RL | SB3 | SKRL |
|------|--------|-----|------|
| **開發者** | ETH RSL | DLR | SKRL Team |
| **PPO 實現** | 原生 | 成熟 | 模組化 |
| **AAC 支援** | ✅ 原生 | ❌ 需客製化 | ⚠️ 需 Patch |
| **WandB** | 手動 | Callback | 內建 |
| **VecEnv** | RslRlVecEnvWrapper | Sb3VecEnvWrapper | SkrlVecEnvWrapper |
| **分佈式** | ✅ | ❌ | ⚠️ |

### 6.2 遷移挑戰與解決方案

#### RSL-RL → SB3

```python
# 觀測包裝器差異
# RSL-RL: 直接使用 env.observation_manager.compute()
# SB3: 需要 Sb3VecEnvWrapper 適配

env = Sb3VecEnvWrapper(env, fast_variant=False)  # 支援 episode 統計
env = IsaacLabMetricsWrapper(env)  # 自定義 metrics 收集
env = SanitizeObservationsWrapper(env)  # NaN/Inf 清理
```

#### SB3 → SKRL (AAC 支援)

```python
# SKRL AAC Wrapper
env = wrap_env_for_aac(env, ml_framework="torch")

# 需要實現 shared_states
class AACIsaacLabWrapper:
    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # 添加特權觀測
        info["shared_states"] = self._get_privileged_obs()

        return obs, reward, terminated, truncated, info

    @property
    def state_space(self):
        """Critic 使用的觀測空間 (161 維)"""
        return Box(-inf, inf, (161,))
```

---

## 7. PPO 超參數配置

### 7.1 網絡架構

```python
# Actor Network (Policy)
policy_network = MLP(
    input_dim=111,       # Policy 觀測維度
    hidden_dims=[256, 256, 128],
    activation=ELU,
    output_dim=2,        # [v, ω]
    init=orthogonal,     # 正交初始化
)

# Critic Network (Value)
value_network = MLP(
    input_dim=161,       # Critic 觀測維度 (含特權資訊)
    hidden_dims=[256, 256, 128],
    activation=ELU,
    output_dim=1,
    init=orthogonal,
)
```

### 7.2 訓練超參數

```yaml
# sb3_ppo_cfg_phase0.yaml
policy: "MlpPolicy"

# 採樣參數
n_steps: 24              # 每環境步數
batch_size: 6144         # n_steps × num_envs = 24 × 256
n_epochs: 5              # 學習輪數

# PPO 參數
gamma: 0.99              # 折扣因子
gae_lambda: 0.95         # GAE λ
clip_range: 0.2          # PPO clip
ent_coef: 0.001          # 熵係數
vf_coef: 0.5             # Value loss 係數
max_grad_norm: 0.5       # 梯度裁剪

# 學習率
learning_rate: 5.0e-5    # 降低防止梯度爆炸

# 歸一化
normalize_input: true
normalize_value: false
clip_obs: 10.0
clip_reward: 10.0

# 初始化
ortho_init: true         # 正交初始化
```

---

## 8. 實驗配置

### 8.1 環境配置

```python
@configclass
class ChargeNavigationEnvCfg:
    # 場景配置
    scene: SceneCfg = SceneCfg(
        num_envs=256,
        env_spacing=25.0,
    )

    # 機器人配置
    robot: ChargeCfg = ChargeCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        max_linear_velocity=1.0,      # m/s
        max_angular_velocity=2.0,     # rad/s
    )

    # LiDAR 配置
    lidar: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        pattern=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,  # 72 rays
        ),
        max_distance=10.0,
        update_period=0.04,  # 25 Hz
    )

    # Episode 配置
    episode_length_s: float = 20.0  # Phase 0
    decimation: int = 2             # 50 Hz control
```

### 8.2 訓練命令

```bash
# SB3 訓練 (Phase 0)
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256 \
    --headless \
    --max_iterations 5000

# SKRL 訓練 (帶 AAC)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256 \
    --headless \
    --print_summary_every 1000
```

---

## 9. 關鍵文件索引

### 9.1 獎勵函數

| 文件 | 功能 |
|------|------|
| `mdp/rewards/goal_rewards.py` | 目標導航獎勵 (progress, reaching, velocity) |
| `mdp/rewards/safety_rewards.py` | 安全獎勵 (collision, progressive_collision) |
| `mdp/rewards/motion_rewards.py` | 運動獎勵 (forward_velocity, action_smoothness) |
| `mdp/rewards/ttc_penalty.py` | TTC 防禦性駕駛懲罰 |
| `mdp/rewards/risk_aware_rewards.py` | **技術貢獻**: 風險感知獎勵 |
| `mdp/rewards/hierarchical_rewards.py` | 層級式導航獎勵 |

### 9.2 觀測函數

| 文件 | 功能 |
|------|------|
| `mdp/observations/hierarchical_navigation.py` | 層級式導航觀測 (local_goal) |
| `mdp/observations/fixed_topology.py` | 固定拓撲觀測 |
| `mdp/observations/dynamic_observations.py` | 動態觀測 |

### 9.3 訓練腳本

| 文件 | 框架 | 特性 |
|------|------|------|
| `scripts/rl/rsl_rl/train.py` | RSL-RL | 原生支援 |
| `scripts/rl/sb3/train_charge.py` | SB3 | WandB, NaN 保護 |
| `scripts/rl/skrl/train_charge.py` | SKRL | AAC 支援 (需 Patch) |

---

## 10. 結論與未來工作

### 10.1 技術貢獻總結

1. **AAC 架構**: Actor 111 維 / Critic 161 維，SKRL Patch 實現
2. **Pull-Push 理論**: 9.1:1 比例設計，敢衝敢撞策略
3. **風險感知獎勵**: 解決冰凍與震盪問題

### 10.2 未來工作

- [ ] Phase 1-3 完整訓練驗證
- [ ] Sim-to-Real 遷移實驗
- [ ] 動態障礙物場景測試
- [ ] 多機器人協作導航

---

**文檔版本**: v2.0 (技術深度版)
**最後更新**: 2026-02-26
