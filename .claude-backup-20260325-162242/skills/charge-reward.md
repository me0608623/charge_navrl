---
name: charge-reward
description: 處理 Charge 專案的獎勵函數相關任務，包含目標獎勵、安全獎勵、運動獎勵的設計與調整。在需要修改或調試獎勵函數時使用。
---

# Charge 專案 - 獎勵函數專用 Skill

## 🎁 獎勵函數架構

### 獎勵函數位置
```
charge_sb3/mdp/rewards/
├── goal_rewards.py       # 目標相關獎勵
├── safety_rewards.py     # 安全相關獎勵
├── motion_rewards.py     # 運動相關獎勵
└── utils.py              # 獎勵工具函數
```

### 導入位置
```python
# 所有獎勵函數都從 mdp/rewards/__init__.py 統一導出
from ..mdp.rewards import (
    # 目標獎勵
    velocity_toward_goal,
    reaching_goal,
    progress_to_goal,
    # 安全獎勵
    collision_penalty,
    obstacle_avoidance_reward,
    # 運動獎勵
    forward_velocity_reward,
    action_rate_penalty,
)
```

---

## 📋 常用獎勵函數速查

### 目標獎勵 (goal_rewards.py)

| 函數 | 描述 | 典型權重 |
|------|------|:--------:|
| `velocity_toward_goal` | 速度在目標方向的投影 | +1.0 ~ +3.0 |
| `reaching_goal` | 抵達目標（距離 < 0.3m）| +500.0 |
| `progress_to_goal` | 距離減少量 (d_{t-1} - d_t) | +2.0 ~ +5.0 |
| `heading_to_goal` | 朝向目標的獎勵 | +0.5 |
| `approaching_goal_bonus` | 接近目標額外獎勵 | +1.0 |

### 安全獎勵 (safety_rewards.py)

| 函數 | 描述 | 典型權重 |
|------|------|:--------:|
| `collision_penalty` | 碰撞懲罰 | -1.0 ~ -5.0 |
| `wall_collision_penalty` | 牆壁碰撞懲罰 | -10.0 |
| `obstacle_avoidance_reward` | 避障獎勵 | +0.1 ~ +0.5 |
| `progressive_collision_penalty` | 漸進式碰撞懲罰 | -0.5 |
| `safe_navigation_bonus` | 安全導航獎勵 | +0.2 |

### 運動獎勵 (motion_rewards.py)

| 函數 | 描述 | 典型權重 |
|------|------|:--------:|
| `forward_velocity_reward` | 前進速度獎勵 | +0.5 ~ +1.0 |
| `forward_motion_reward` | 向前移動獎勵 | +0.3 |
| `move_reward` | 移動獎勵（非零速度）| +0.1 |
| `action_rate_penalty` | 動作變化懲罰（防抖）| -0.01 ~ -0.1 |
| `time_out_penalty` | 超時懲罰 | -0.05 |

---

## 🔧 標準修改流程

### 1. 定義新獎勵函數
```python
# 在 mdp/rewards/your_rewards.py 中
import torch
from isaaclab.envs import ManagerBasedEnv

def your_custom_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """
    你的自定義獎勵函數

    Args:
        env: 環境實例

    Returns:
        torch.Tensor: 獎勵值，形狀 [num_envs]
    """
    # 獲取需要的數據
    robot_state = env.robot.data.root_state_w  # [num_envs, 13]
    goal_pos = env.goal_buffer  # [num_envs, 3]

    # 計算獎勵
    reward = torch.zeros(env.num_envs, device=env.device)

    # 你的邏輯...
    reward = your_calculation(robot_state, goal_pos)

    return reward
```

### 2. 導出獎勵函數
```python
# 在 mdp/rewards/__init__.py 中
from .your_rewards import your_custom_reward

__all__ = [..., "your_custom_reward"]
```

### 3. 在環境配置中使用
```python
# 在 cfg/charge_env_cfg_*.py 中
from ..mdp.rewards import your_custom_reward

@configclass
class RewardsCfg:
    your_reward_term = RewTerm(
        func=your_custom_reward,
        weight=1.0,
        params={}  # 可選參數
    )
```

---

## ⚖️ 獎勵權重設計原則

### 權重比例建議

**目標 vs 避障比例**：
```
理想比例 = 目標導向獎勵 : 避障懲罰 ≈ 25:1 以上

# 示例：
velocity_toward_goal (weight=5.0)
collision_penalty (weight=-0.2)
比例 = 5.0 / 0.2 = 25:1 ✅ 健康的比例
```

**終極獎勵（Sparse Reward）**：
```python
# 抵達目標應該給予極大的獎勵
reaching_goal = RewTerm(
    func=reaching_goal,
    weight=500.0,  # 遠大於其他獎勵
)
```

### 獎勵歸一化
```python
# 確保所有獎勵在同一數量級
# 建議範圍：[-10, +10] per step

# 使用 ExpectedValueTracker 追蹤
from ..mdp.rewards import ExpectedValueTracker

tracker = ExpectedValueTracker()
tracker.update("your_reward", reward_values)
# 定期打印檢查
```

---

## 🐛 常見問題排查

### 問題 1：Agent 蛇行（Oscillation）
**症狀**：Agent 左右擺動，無法直行

**診斷**：
```python
# 檢查目標:避障比例
goal_weight = 5.0
avoid_weight = 0.4
ratio = goal_weight / avoid_weight  # = 12.5:1 ❌ 太低

# 解決：增加目標權重或降低避障權重
goal_weight = 10.0  # 提高到 10
# ratio = 10.0 / 0.4 = 25:1 ✅
```

### 問題 2：Agent 翻車騙分（Tipping Hack）
**症狀**：Agent 故意翻車滑行到目標

**解決**：使用姿態門控
```python
# 在運動獎勵中加入姿態檢查
def forward_velocity_reward_gated(env: ManagerBasedEnv) -> torch.Tensor:
    reward = forward_velocity_reward(env)

    # 檢查是否正立（z 軸向上）
    up_vector = env.robot.data.root_state_w[:, 6:9]  # [num_envs, 3]
    is_upright = up_vector[:, 2] > 0.5  # z 分量 > 0.5

    # 翻車則取消獎勵
    reward = reward * is_upright.float()

    return reward
```

### 問題 3：Agent 原地打轉（Spinning）
**症狀**：Agent 原地旋轉不前進

**解決**：添加前進獎勵
```python
class RewardsCfg:
    # 獎勵向前的速度分量
    forward_velocity = RewTerm(
        func=forward_velocity_reward,
        weight=0.5,
    )
    # 懲罰角速度過大（轉太圈）
    action_rate_penalty = RewTerm(
        func=action_rate_penalty,
        weight=-0.05,
    )
```

---

## 📊 獎勵函數調試技巧

### 使用 Reward Checker
```python
# 在 mdp/rewards/utils.py 中有內建的檢查工具
from ..mdp.rewards import _check_reward_term

# 在訓練腳本中啟用
export REWARD_CHECK=1  # 環境變量開啟檢查
```

### TensorBoard 監控
```python
# 確保獎勵被正確記錄
with torch.no_grad():
    for name, reward_term in env.reward_manager terms:
        writer.add_scalar(f"rewards/{name}", reward_term.mean(), step)
```

---

## 🎚️ Phase 專用獎勵配置

### Phase 0（基礎運動學）
```python
class RewardsCfg:
    # 專注於移動學習
    progress = RewTerm(func=progress_to_goal, weight=20.0)
    goal = RewTerm(func=reaching_goal, weight=10.0)
    collision = RewTerm(func=collision_penalty, weight=-10.0)
    smooth = RewTerm(func=action_rate_penalty, weight=-0.5)
```

### Phase 3（長程導航）
```python
class RewardsCfg:
    # 專注於目標導向
    reaching_goal = RewTerm(func=reaching_goal, weight=500.0)
    velocity_to_goal = RewTerm(func=velocity_toward_goal, weight=3.0)
    distance_progress = RewTerm(func=progress_to_goal, weight=5.0)
    tipped_over = RewTerm(func=tipped_over_penalty, weight=-200.0)
```
