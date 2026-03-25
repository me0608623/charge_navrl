# AIT* + RL (PPO) 層級式導航架構

## 系統架構概覽

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        層級式導航系統                                       │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐        ┌─────────────────┐       ┌───────────────┐  │
│  │   AIT* 全域     │   路徑   │   Local Goal    │  目標  │    RL PPO     │  │
│  │   規劃器        │ ───────► │   提取器        │ ─────► │    控制器      │  │
│  │   (大腦)        │  1 Hz   │   (介面層)      │ 10 Hz │   (小腦)       │  │
│  └─────────────────┘        └─────────────────┘       └───────────────┘  │
│         │                            │                        │           │
│         ▼                            ▼                        ▼           │
│  [已知地圖]                   [局部目標點]              [動作輸出]        │
│  + 當前位置                    + 相對座標              v, ω              │
│  + 目標位置                                              │               │
│                                ┌────────────────────────┘               │
│                                ▼                                        │
│                        ┌───────────────┐                               │
│                        │   機器人      │                               │
│                        │   (Charge)    │                               │
│                        └───────────────┘                               │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 層級 1：AIT* 全域規劃器 (Global Planner)

### 頻率
- **1 Hz ~ 5 Hz**（低頻，不需要每幀都算）
- 只在以下情況重新規劃：
  - 目標位置改變
  - 機器人偏離路徑太遠
  - 發現新的靜態障礙物

### 輸入
| 輸入 | 格式 | 說明 |
|------|------|------|
| 地圖 | GridMap | 已知環境的靜態地圖 |
| 起點位置 | (x, y) | 機器人當前世界坐標 |
| 終點位置 | (x, y) | 目標世界坐標 |

### 處理
```python
# AIT* 規劃算法
path = aitstar_plan(
    start=robot_position,
    goal=goal_position,
    map=static_map,
    robot_radius=0.3,
)

# 路徑平滑
smoothed_path = path_smoother.smooth(path)
```

### 輸出
```python
# 全局路徑點列
P_global = [
    (x_1, y_1),  # 起點
    (x_2, y_2),
    ...
    (x_n, y_n),  # 終點
]

# 格式: Tensor[num_waypoints, 2]
```

---

## 層級 2：局部目標提取器 (Interface - 關鍵！)

### 頻率
- **10 Hz ~ 20 Hz**（中頻）
- 比規劃快，比控制慢

### 輸入
| 輸入 | 格式 | 說明 |
|------|------|------|
| 全局路徑 | P_global | AIT* 輸出的路徑點 |
| 機器人位置 | (x, y) | 當前世界坐標 |
| 機器人朝向 | θ | Yaw 角（弧度） |
| Lookahead | L | 前瞻距離（米） |

### 處理步驟

#### 步驟 1：選擇局部目標 (Carrot-on-Stick)
```python
def select_local_goal(path, robot_pos, lookahead_distance=2.0):
    """
    從全局路徑中選擇距離機器人前方 L 米的點作為局部目標
    """
    # 找到路徑上離機器人最近的點
    nearest_idx = find_nearest_point(path, robot_pos)

    # 從最近點開始，累積距離直到超過 lookahead_distance
    accumulated_dist = 0
    for i in range(nearest_idx, len(path)):
        if i > nearest_idx:
            accumulated_dist += distance(path[i], path[i-1])
        if accumulated_dist >= lookahead_distance:
            return path[i]  # 返回局部目標

    # 如果路徑快走完了，返回終點
    return path[-1]
```

#### 步驟 2：坐標轉換 (World Frame → Robot Frame)
```python
def world_to_robot_frame(target_world, robot_pos, robot_yaw):
    """
    將目標點從世界坐標轉換到機器人坐標系

    世界坐標系: 固定不動，原點在 (0, 0)
    機器人坐標系: 隨機器人移動，原點在機器人中心，X軸指向前方
    """
    # 平移
    dx = target_world[0] - robot_pos[0]
    dy = target_world[1] - robot_pos[1]

    # 旋轉（逆時針轉回去）
    cos_yaw = cos(robot_yaw)
    sin_yaw = sin(robot_yaw)

    target_robot_x = dx * cos_yaw + dy * sin_yaw
    target_robot_y = -dx * sin_yaw + dy * cos_yaw

    return (target_robot_x, target_robot_y)
```

#### 步驟 3：計算導航特征
```python
def compute_navigation_features(target_robot):
    """
    計算給 RL 的導航信息
    """
    # 極坐標表示
    distance = sqrt(target_robot[0]**2 + target_robot[1]**2)
    heading = atan2(target_robot[1], target_robot[0])  # 相對於車頭的角度

    # 前向/側向分解
    forward = target_robot[0]  # 正值 = 前方，負值 = 後方
    lateral = target_robot[1]  # 正值 = 左側，負值 = 右側

    return {
        'distance': distance,
        'heading': heading,
        'forward': forward,
        'lateral': lateral,
    }
```

### 輸出
```python
# 導航觀測（給 RL）
nav_obs = {
    'target_distance': 2.5,      # 目標距離（米）
    'target_heading': 0.3,       # 目標方向偏差（弧度）
    'target_forward': 2.4,       # 目標前方分量
    'target_lateral': 0.7,       # 目標側方分量
}
```

---

## 層級 3：RL PPO 控制器 (Local Controller)

### 頻率
- **50 Hz ~ 100 Hz**（高頻，控制頻率）
- 每個環境步驟都執行

### 輸入：觀測空間 (Observation Space)

```python
# 完整觀測向量
observation = concat([
    # === 感知層 ===
    lidar_scan,           # [72] LiDAR 掃描（360° 或 180°）
                          # 範圍: [0, 1]，0 = 很近，1 = 很遠

    # === 導航層 ===
    target_distance,      # [1]  目標距離（米），歸一化
    target_heading,       # [1]  目標方向偏差（弧度），範圍 [-π, π]
    current_velocity_x,   # [1]  當前線速度（米/秒）
    current_velocity_z,   # [1]  當前角速度（弧度/秒）

    # === 歷史層（可選）===
    prev_lidar_scan,      # [72] 上一幀 LiDAR
    prev_action,          # [2]  上一幀動作
])
```

**總維度**: ~150-200 維

### 處理：PPO 神經網路

```python
# 網路架構
class PPOPolicy(nn.Module):
    def __init__(self):
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(256, 2)  # 輸出動作均值
        self.actor_log_std = nn.Parameter(torch.zeros(2))  # 動作標準差
        self.critic = nn.Linear(256, 1)  # 價值函數

    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
```

### 輸出：動作空間 (Action Space)

```python
# 連續動作
action = [
    linear_velocity,   # [0, 1.0]  前進速度（米/秒）
    angular_velocity,  # [-1.0, 1.0]  轉向速度（弧度/秒）
]
```

---

## 獎勵函數設計

### 完整獎勵公式

```python
R_total = R_reach + R_progress + R_tracking - P_collision - P_unstable
```

### 各項詳細

| 獎勵項 | 公式 | 權重 | 說明 |
|--------|------|------|------|
| R_reach | `+10` if `d < threshold` | 10.0 | 抵達局部目標 |
| R_progress | `d_{t-1} - d_t` | 1.0 | 距離變化（靠近=正，遠離=負）|
| R_tracking | `-abs(heading_error)` | 0.5 | 方向對齊懲罰 |
| P_collision | `-20` if collision | -20.0 | 碰撞懲罰 |
| P_unstable | `-||a_t - a_{t-1}||²` | -0.1 | 動作平滑懲罰 |

### Python 實現

```python
def compute_reward(env, robot_cfg, sensor_cfg):
    # 獲取狀態
    robot = env.scene[robot_cfg]
    sensor = env.scene[sensors[sensor_cfg]]

    # 1. 抵達獎勵
    target_dist = get_distance_to_target(env)
    r_reach = torch.where(target_dist < 0.3, 10.0, 0.0)

    # 2. 前進獎勵
    prev_dist = env._prev_target_dist
    r_progress = prev_dist - target_dist
    env._prev_target_dist = target_dist.clone()

    # 3. 方向對齊獎勵
    heading_error = get_heading_error(env)
    r_tracking = -torch.abs(heading_error)

    # 4. 碰撞懲罰
    min_lidar = get_min_lidar_distance(sensor)
    p_collision = torch.where(min_lidar < 0.3, -20.0, 0.0)

    # 5. 平滑懲罰
    action_change = env.current_action - env.prev_action
    p_unstable = -torch.norm(action_change, dim=-1) ** 2

    # 總獎勵
    return r_reach + r_progress + r_tracking + p_collision + p_unstable
```

---

## 訓練流程

### 階段 1：基礎訓練（無障礙物）
- 目標：學會跟隨局部目標
- 環境：空曠場地
- AIT*：直線路徑

### 階段 2：靜態障礙物
- 目標：學會避障 + 跟隨路徑
- 環境：3-5 個靜態障礙物
- AIT*：規劃繞過障礙物的路徑

### 階段 3：動態障礙物
- 目標：應對移動障礙物
- 環境：加入動態障礙物
- AIT*：定期重新規劃

### 階段 4：真實場景
- 目標：泛化到真實環境
- 環境：Domain Randomization
- AIT*：完整集成

---

## 實現檔案結構

```
charge_sb3/
├── mdp/
│   ├── path_planner/
│   │   ├── aitstar_adapter.py       # AIT* 規劃器
│   │   ├── local_goal_extractor.py  # 局部目標提取 ⭐
│   │   ├── frenet_transform.py      # 坐標轉換
│   │   └── path_visualizer.py       # 視覺化
│   ├── observations/
│   │   └── navigation_observations.py  # 導航觀測 ⭐
│   └── rewards/
│       └── navigation_rewards.py      # 導航獎勵 ⭐
└── cfg/
    └── hierarchical_env_cfg.py        # 分層環境配置 ⭐
```

---

## 關鍵實現細節

### 1. Lookahead Distance 自適應
```python
# 根據速度調整前瞻距離
lookahead = base_lookahead * (1 + current_velocity / max_velocity)
```

### 2. 局部目標更新條件
```python
# 當滿足以下條件時更新局部目標
if (distance_to_current_goal < 0.5 or  # 已接近
    time_since_last_update > 0.1):     # 超時
    update_local_goal()
```

### 3. 緊急停車
```python
# LiDAR 檢測到前方有障礙物且距離很近時
if min_front_lidar < 0.2:
    action = [0.0, 0.0]  # 緊急停車
```

---

## 測試與驗證

### 單元測試
1. AIT* 規劃器測試
2. 坐標轉換測試
3. 局部目標提取測試

### 整合測試
1. 無障礙物直線跟隨
2. 繞過單個障礙物
3. 狹窄通道通行

### 性能指標
- 成功率 (Success Rate)
- 平均路徑長度
- 碰撞次數
- 平均速度
