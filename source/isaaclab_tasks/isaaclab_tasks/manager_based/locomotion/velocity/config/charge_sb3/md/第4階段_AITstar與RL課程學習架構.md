# AIT* + RL 層級式導航課程學習架構

本架構實現了 AIT*（全局規劃器）與 RL（局部策略）的層級式導航系統，通過課程學習實現由易到難的漸進式訓練。

---

## 📋 概述

本架構結合了 AIT* 全域規劃器和 RL 局部策略，通過 4 階段課程學習逐步提高難度，最終實現複雜環境下的自主導航。

---

## 🏗️ 架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    課程學習管理器                              │
│              (CurriculumManager - 4 Stages)                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Stage 1 │  │ Stage 2 │  │ Stage 3 │  │ Stage 4 │         │
│  │ 無障礙  │→ │ 靜態障礙│→ │ 複雜地形│→ │ 動態環境│         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AIT* 全局規劃器                              │
│                   (AITStarPathPlanner)                          │
│  • 生成從起點到終點的最優路徑                                   │
│  • 動態更新啟發式場                                            │
│  • 提供引導信號給 RL                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RL 局部策略                                 │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │   狀態空間      │  │   獎勵函數      │                     │
│  │ • 相對路徑位移  │  │ • 進步獎勵      │                     │
│  │ • 路徑切線方向  │  │ • CTE 懲罰      │                     │
│  │ • 路徑進度      │  │ • 方向一致性    │                     │
│  │ • 啟發式場      │  │ • 碰撞懲罰      │                     │
│  └─────────────────┘  └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 課程階段設計

### Stage 1: Goal-Oriented Following (無障礙物)

**配置**:
```python
CurriculumStageConfig(
    name="Goal-Oriented Following",
    obstacle_density=0.0,
    goal_distance_range=(5.0, 15.0),
    path_complexity_target=1.0,
    success_threshold=0.85,
    sensor_noise=0.0,
)
```

**學習目標**: 基礎運動控制（加速、轉向）

**獎勵權重**:
- progress: 1.0
- cross_track: 0.2
- alignment: 0.1

### Stage 2: Static Obstacle Navigation (靜態障礙)

**配置**:
```python
CurriculumStageConfig(
    name="Static Obstacle Navigation",
    obstacle_density=0.15,
    obstacle_types=["cuboid", "cylinder"],
    goal_distance_range=(8.0, 18.0),
    success_threshold=0.80,
    sensor_noise=0.02,
)
```

**學習目標**: 處理感知偏差，結合 LiDAR 與路徑

**獎勵權重**:
- progress: 1.0
- cross_track: 0.5
- alignment: 0.3

### Stage 3: Complex Topology (複雜地形)

**配置**:
```python
CurriculumStageConfig(
    name="Complex Topology",
    obstacle_density=0.35,
    obstacle_types=["cuboid", "cylinder", "wall", "narrow_passage"],
    goal_distance_range=(10.0, 20.0),
    success_threshold=0.75,
    sensor_noise=0.05,
)
```

**學習目標**: 狹窄空間精準控制

**獎勵權重**:
- progress: 1.2
- cross_track: 0.8
- alignment: 0.4

### Stage 4: Dynamic & Noisy Environment (動態環境)

**配置**:
```python
CurriculumStageConfig(
    name="Dynamic & Noisy Environment",
    obstacle_density=0.40,
    has_dynamic_obstacles=True,
    goal_distance_range=(12.0, 25.0),
    success_threshold=0.70,
    sensor_noise=0.10,
)
```

**學習目標**: 即時反應路徑變動

**獎勵權重**:
- progress: 1.5
- cross_track: 1.0
- alignment: 0.5

---

## 🎯 狀態空間設計

### 相對路徑觀測 (Relative Path Observations)

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.observations import (
    relative_path_displacement,
    path_tangent_direction,
    path_progress,
    cross_track_error_obs,
    combined_path_observations,
)

# 組合觀測 (推薦)
obs = combined_path_observations(
    env=env,
    path_points=aitstar_path,
    goal_pos=goal,
    include_heuristic=False,  # 可選啟發式場
)

# 觀測維度:
# - [2] 相對路徑點位移 (機器人坐標系)
# - [2] 路徑切線方向 (機器人坐標系)
# - [1] 路徑進度 (0-1)
# - [1] Cross-track error (米)
# 總計: 6 維 (不含啟發式場)
```

### AIT* 啟發式場 (Heuristic Field)

```python
heuristic_field = aitstar_heuristic_field(
    env=env,
    path_points=aitstar_path,
    goal_pos=goal,
    grid_size=20,  # 20x20 網格
    map_range=10.0,
)
# 輸出: [num_envs, 400] 展平的啟發式場
```

---

## 🏆 獎勵函數設計

### 綜合獎勵公式

```
R = w1 × progress - w2 × cte - w3 × collision + w4 × alignment + w5 × velocity - w6 × time
```

### 使用方式

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.rewards import (
    aitstar_guided_reward,
    curriculum_adjusted_reward,
)

# 方式 1: 固定權重
reward, progress = aitstar_guided_reward(
    env=env,
    path_points=aitstar_path,
    robot_cfg="robot",
    weights=AITStarRewardWeights(
        progress=1.0,
        cross_track=0.5,
        alignment=0.3,
    ),
    previous_progress=last_progress,
)

# 方式 2: 根據課程階段自動調整（推薦）
reward, progress = curriculum_adjusted_reward(
    env=env,
    path_points=aitstar_path,
    robot_cfg="robot",
    stage=curriculum.current_stage,
    previous_progress=last_progress,
)
```

---

## 💻 使用範例

### 完整訓練循環

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.path_planner import (
    create_curriculum_manager,
)

# 1. 創建課程管理器
curriculum = create_curriculum_manager(
    initial_stage=1,
    auto_progress=True,
)

# 2. 訓練循環
for episode in range(num_episodes):
    # 生成目標
    goal_pos = curriculum.generate_goal_for_stage(env)

    # AIT* 規劃路徑
    robot_pos = env.scene["robot"].data.root_pos_w[0, :2]
    aitstar_path = curriculum.plan_path_with_aitstar(robot_pos, goal_pos, env)

    # 執行回合
    last_progress = None
    for step in range(max_steps):
        # 計算獎勵
        reward, last_progress = curriculum_adjusted_reward(
            env=env,
            path_points=aitstar_path,
            stage=curriculum.current_stage,
            previous_progress=last_progress,
        )

        # RL 採樣動作
        action = rl_policy(obs)

        # 執行
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break

    # 更新課程指標
    curriculum.update_metrics(
        reward=total_reward,
        success=is_success,
        path_length=actual_length,
        time_to_goal=steps,
        collided=collision_detected,
    )

    # 檢查是否晉升
    if curriculum.current_stage > previous_stage:
        print(f"🎓 晉升到 Stage {curriculum.current_stage}!")
```

---

## 📁 文件結構

```
mdp/
├── path_planner/
│   ├── aitstar_adapter.py       # AIT* Isaac Lab 適配器
│   ├── curriculum_manager.py     # 課程學習管理器 ⭐
│   ├── environment_map.py        # 環境地圖
│   └── path_smoother.py          # 路徑平滑器
│
├── observations/
│   └── path_relative.py          # 相對路徑觀測 ⭐
│
└── rewards/
    ├── aitstar_guided.py         # AIT* 引導獎勵 ⭐
    └── path_following.py         # 路徑跟隨獎勵
```

---

## 🔗 導航接口

```python
# 課程管理器
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3 import (
    CurriculumManager,
    create_curriculum_manager,
)

# AIT* 規劃器
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3 import (
    AITStarPathPlanner,
    create_aitstar_planner,
)

# 相對路徑觀測
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.observations import (
    relative_path_displacement,
    path_tangent_direction,
    path_progress,
    combined_path_observations,
)

# AIT* 引導獎勵
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.rewards import (
    AITStarRewardWeights,
    aitstar_guided_reward,
    curriculum_adjusted_reward,
)
```

---

## 🚀 下一步

1. 創建 Phase 4 環境配置 (`cfg/charge_env_cfg_v4.py`)
2. 整合課程管理器到訓練腳本
3. 實現引導採樣 (Guided Sampling)
4. 添加視覺化工具（顯示 AIT* 路徑 + RL 軌跡）

---

**最後更新**: 2026-02-05
