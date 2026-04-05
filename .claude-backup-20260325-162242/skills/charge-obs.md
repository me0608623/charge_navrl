---
name: charge-obs
description: 處理 Charge 專案的觀測空間相關任務，包含觀測函數設計、觀測維度管理、LiDAR 處理等。在需要修改或調試觀測空間時使用。
---

# Charge 專案 - 觀測空間專用 Skill

## 👁️ 觀測空間架構

### 觀測函數位置
```
charge_sb3/mdp/observations/
├── functions.py              # 基礎觀測函數
├── utils.py                  # 觀測工具函數
├── fixed_topology.py         # 固定拓撲觀測系統（122維）
└── hierarchical_navigation.py # 層級式導航觀測
```

### 統一導入
```python
# 所有觀測函數從 mdp/observations/__init__.py 導出
from ..mdp.observations import (
    # 基礎觀測
    goal_position_in_robot_frame,
    goal_distance,
    base_velocity_xy,
    base_angular_velocity_z,
    # LiDAR
    lidar_scan,
    lidar_scan_2d_sweep,
    # 其他
    time_remaining_ratio,
    alive_flag,
    heading_error_to_goal,
    safe_last_action,
)
```

---

## 📏 觀測維度速查

### 標準觀測空間（Phase 1-3）

| 觀測項 | 維度 | 描述 |
|--------|:----:|------|
| LiDAR Scan | 131 | 360度掃描（含無效值標記）|
| Goal Position (robot frame) | 2 | [x, y] 相對位置 |
| Goal Distance | 1 | 歐式距離 |
| Base Velocity (xy) | 2 | [vx, vy] 線速度 |
| Angular Velocity (z) | 1 | ωz 角速度 |
| Time Remaining Ratio | 1 | 剩餘時間比例 |
| Alive Flag | 1 | 存活標誌 |
| Heading Error | 1 | 朝向目標誤差角 |
| **總計** | **140** | |

### Phase 0 最小觀測空間

| 觀測項 | 維度 | 描述 |
|--------|:----:|------|
| Goal Position | 2 | 相對目標位置 |
| Goal Distance | 1 | 到目標距離 |
| Base Velocity | 2 | 底盤速度 |
| Angular Velocity | 1 | 角速度 |
| Time Ratio | 1 | 時間比例 |
| Alive Flag | 1 | 存活標誌 |
| Actions (safe clipped) | 2 | 上一步動作 |
| **總計** | **10** | 無 LiDAR，專注運動控制 |

### 固定拓撲觀測系統（122維）

```python
# 用於處理可變數量的障礙物
# 來源：mdp/observations/fixed_topology.py

TOTAL_OBS_DIM = 122
├── LIDAR_DIM = 24                    # 下採樣 LiDAR
├── STATIC_SLOTS_DIM = 60             # 靜態障礙物（6個 × 10維）
├── DYNAMIC_SLOTS_DIM = 30            # 動態障礙物（3個 × 10維）
├── NAV_DIM = 5                       # 導航命令
└── PROPRIOCEPTION_DIM = 3            # 本體感覺
```

---

## 🔧 標準修改流程

### 1. 定義新觀測函數
```python
# 在 mdp/observations/functions.py 中
import torch
from isaaclab.envs import ManagerBasedEnv

def your_custom_obs(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    """
    你的自定義觀測函數

    Args:
        env: 環境實例
        dt: 時間步長

    Returns:
        torch.Tensor: 觀測值，形狀 [num_envs, obs_dim]
    """
    # 獲取需要的數據
    robot_state = env.robot.data.root_state_w  # [num_envs, 13]

    # 計算觀測
    obs = torch.zeros(env.num_envs, your_dim, device=env.device)

    # 你的邏輯...
    obs = your_calculation(robot_state)

    # 重要：檢查有限值
    assert torch.all(torch.isfinite(obs)), "Observation contains NaN/Inf!"

    return obs
```

### 2. 導出觀測函數
```python
# 在 mdp/observations/__init__.py 中
from .functions import your_custom_obs

__all__ = [..., "your_custom_obs"]
```

### 3. 在環境配置中使用
```python
# 在 cfg/charge_env_cfg_*.py 中
from ..mdp.observations import your_custom_obs

@configclass
class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "your_obs": ObsTerm(
                func=your_custom_obs,
                params={},  # 可選參數
            ),
        },
    )
```

---

## 📡 LiDAR 觀測詳解

### LiDAR 配置
```python
# 在場景配置中
ray_caster = MultiMeshRayCasterCfg(
    prim_path="/envs/.*/Robot/LidarSensor",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    attach_yaw_only=True,  # 只跟隨偏航角
    pattern=patterns.PatternCfg(
        # 360度掃描，131條射線
        type="uniform",
        num_rows=1,
        num_cols=131,
        horizontal_fov=360.0,
    ),
    max_range=10.0,  # 最大偵測距離 10m
    debug_vis=False,  # 設為 True 可視化射線
)
```

### LiDAR 觀測處理
```python
# 原始輸出：[num_envs, 131]
# 無效值：max_range + 1.0 (表示射線未擊中任何物體)

def lidar_scan(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    """標準 LiDAR 觀測處理"""
    # 獲取原始距離
    lidar_dist = env.sensors["lidar"].data.output["dist"][..., 0]

    # 歸一化到 [0, 1]
    max_range = env.cfg.sensors.lidar.pattern.max_range
    lidar_norm = lidar_dist / max_range

    # 處理無效值（設為 1.0）
    lidar_norm = torch.nan_to_num(lidar_norm, nan=1.0, posinf=1.0, neginf=0.0)

    return lidar_norm
```

### LiDAR 下採樣（減少維度）
```python
def lidar_downsample(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    """下採樣 LiDAR 到 24 維"""
    lidar_full = lidar_scan(env, dt)  # [num_envs, 131]

    # 均勻下採樣：每隔 5 個取 1 個
    indices = torch.arange(0, 131, 5, device=lidar_full.device)
    lidar_down = lidar_full[:, indices]

    return lidar_down  # [num_envs, 26] ≈ 24
```

---

## 🎯 觀測設計原則

### 原則 1：歸一化到 [0, 1]
```python
# ❌ 錯誤：原始未歸一化
def bad_obs(env):
    return env.robot.data.root_state_w[:, :3]  # 可能是 [-100, 100]

# ✅ 正確：歸一化
def good_obs(env):
    raw = env.robot.data.root_state_w[:, :3]
    # 假設環境範圍 [-10, 10]
    normalized = (raw + 10.0) / 20.0
    return torch.clamp(normalized, 0.0, 1.0)
```

### 原則 2：相對座標優先
```python
# ❌ 錯誤：絕對座標
def bad_goal_obs(env):
    return env.goal_buffer[:, :3]  # 世界座標，不變

# ✅ 正確：相對座標
def good_goal_obs(env):
    # 轉換到機器人座標系
    robot_pos = env.robot.data.root_state_w[:, :3]
    goal_pos = env.goal_buffer[:, :3]

    # 計算相對位置
    rel_pos = goal_pos - robot_pos

    # 旋轉到機器人朝向
    heading = env.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    x = rel_pos[:, 0] * cos_h + rel_pos[:, 1] * sin_h
    y = -rel_pos[:, 0] * sin_h + rel_pos[:, 1] * cos_h

    return torch.stack([x, y], dim=1)
```

### 原則 3：處理無效值
```python
def safe_observation(env) -> torch.Tensor:
    """安全的觀測函數，處理 NaN/Inf"""
    raw = calculate_raw_observation(env)

    # 替換無效值
    safe = torch.nan_to_num(
        raw,
        nan=0.0,    # NaN → 0
        posinf=1.0, # +Inf → 1
        neginf=0.0  # -Inf → 0
    )

    return safe
```

---

## 🔍 觀測調試技巧

### 檢查觀測統計
```python
def print_obs_stats(obs, name="Observation"):
    """打印觀測統計信息"""
    print(f"\n=== {name} ===")
    print(f"Shape: {obs.shape}")
    print(f"Range: [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"Mean: {obs.mean():.4f}")
    print(f"Std: {obs.std():.4f}")
    print(f"NaN count: {torch.isnan(obs).sum()}")
    print(f"Inf count: {torch.isinf(obs).sum()}")

# 使用
obs, _ = env.reset()
print_obs_stats(obs[:, :131], "LiDAR")
print_obs_stats(obs[:, 131:133], "Goal Position")
```

### 可視化觀測
```python
import matplotlib.pyplot as plt

def visualize_lidar(env, env_idx=0):
    """可視化 LiDAR 掃描"""
    lidar = env.sensors["lidar"].data.output["dist"][env_idx, :, 0]

    angles = torch.linspace(0, 2*math.pi, len(lidar))
    x = lidar * torch.cos(angles)
    y = lidar * torch.sin(angles)

    plt.figure()
    plt.plot([0, x[0]], [0, y[0]], 'r-')  # 機器人朝向
    plt.scatter(x, y, c=angles, cmap='hsv')
    plt.axis('equal')
    plt.title(f"LiDAR Scan (env {env_idx})")
    plt.show()
```

### 觀測相關性分析
```python
def check_observation_redundancy(env, num_samples=1000):
    """檢查觀測之間的相關性"""
    obs_samples = []
    for _ in range(num_samples):
        obs, _ = env.reset()
        obs_samples.append(obs)

    obs_stack = torch.cat(obs_samples, dim=0)  # [num_samples*num_envs, obs_dim]

    # 計算相關矩陣
    corr_matrix = torch.corrcoef(obs_stack.T)

    # 找出高相關性對（>0.95）
    high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
    print(f"Highly correlated observation pairs: {high_corr.sum() // 2}")
```

---

## 📊 觀測空間配置示例

### Phase 0 最小配置
```python
@configclass
class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "goal_position": ObsTerm(func=goal_position_in_robot_frame),
            "goal_distance": ObsTerm(func=goal_distance),
            "base_velocity": ObsTerm(func=base_velocity_xy),
            "angular_velocity": ObsTerm(func=base_angular_velocity_z),
            "time_ratio": ObsTerm(func=time_remaining_ratio),
            "alive": ObsTerm(func=alive_flag),
            "actions": ObsTerm(func=safe_last_action),
        },
    )
    # 總維度：2 + 1 + 2 + 1 + 1 + 1 + 2 = 10
```

### Phase 3 標準配置
```python
@configclass
class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "lidar": ObsTerm(func=lidar_scan_2d_sweep),
            "goal_position": ObsTerm(func=goal_position_in_robot_frame),
            "goal_distance": ObsTerm(func=goal_distance),
            "base_velocity": ObsTerm(func=base_velocity_xy),
            "angular_velocity": ObsTerm(func=base_angular_velocity_z),
            "time_ratio": ObsTerm(func=time_remaining_ratio),
            "alive": ObsTerm(func=alive_flag),
            "heading_error": ObsTerm(func=heading_error_to_goal),
            "actions": ObsTerm(func=safe_last_action),
        },
    )
    # 總維度：131 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 2 = 142
```

---

## ⚠️ 常見觀測問題

### 問題 1：觀測維度不匹配
```
ValueError: Observation shape mismatch. Expected X, got Y.
```
**診斷**：
```python
# 檢查配置與實際輸出
from isaaclab.envs import ManagerBasedRLEnv

env = gym.make("Isaac-Navigation-Charge-SB3-v0")
obs, _ = env.reset()

print(f"Expected: {env.unwrapped.observation_manager.num_obs}")
print(f"Actual: {obs.shape[-1]}")
```

### 問題 2：PPO std >= 0（觀測方差為零）
**解決**：確保觀測有足夠變化
```python
# 添加隨機性
obs_with_noise = obs + torch.randn_like(obs) * 0.01

# 或檢查傳感器配置
# RayCaster 可能有 pattern 問題
```

### 問題 3：NaN/Inf 傳播
**解決**：使用 `check_finite` 包裝
```python
from ..mdp.observations import check_finite

@check_finite  # 自動檢查並替換 NaN/Inf
def your_obs(env, dt):
    return raw_observation(env)
```
