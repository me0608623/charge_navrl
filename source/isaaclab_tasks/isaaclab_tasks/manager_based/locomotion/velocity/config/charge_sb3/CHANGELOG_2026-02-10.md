# 修改记录 - 2026-02-10

## 📅 今日工作总结

### 🎯 主要目标
修复 Phase 0 训练失败问题，优化奖励函数设计，改善可视化调试。

---

## 🔧 代码修改清单

### 1. **奖励函数优化** (`mdp/rewards/goal_rewards.py`)

#### 1.1 `progress_to_goal` 改为向量点积实现

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp/rewards/goal_rewards.py:230-311`

**修改内容**:
- 从距离变化 (`prev_dist - curr_dist`) 改为向量点积 (`v · d`)
- 加入朝向判断：只有"面向目标"且"朝目标移动"才给奖励
- 奖励公式：`reward = heading_factor × velocity_factor`

**核心代码**:
```python
# 计算朝向余弦（机器前方向量与目标方向）
heading_cosine = torch.sum(forward_w * goal_direction, dim=1)
heading_factor = torch.clamp(heading_cosine, min=0.0, max=1.0)

# 计算速度在目标方向的投影
velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
velocity_factor = torch.clamp(velocity_projection, min=0.0)

# 组合奖励
reward = heading_factor * velocity_factor
```

**效果对比**:

| 场景 | 旧实现（距离变化） | 新实现（向量点积×朝向） |
|------|------------------|----------------------|
| 正对目标前进 1.5 m/s | +0.5 ✓ | **+1.5** ✅ (最高奖励) |
| 倒退朝向目标 1.5 m/s | +0.5 ✓ | **0.0** ❌ (不奖励作弊) |
| 侧向滑行 1.5 m/s | 0.0 | **0.0** ❌ (无奖励) |
| 背离目标移动 1.5 m/s | -0.5 | **0.0** ❌ (无奖励) |

---

#### 1.2 加强 NaN 处理

**修改的函数**:
- `progress_to_goal` (line 264-267)
- `goal_position_in_robot_frame` (观测函数, line 210-220)
- `goal_distance` (观测函数, line 265-267)
- `base_velocity_xy` (观测函数, line 342-346)
- `base_angular_velocity_z` (观测函数, line 383-384)

**通用处理模式**:
```python
# 1. 在输入端立即清理
data = torch.nan_to_num(data, nan=0.0, posinf=MAX, neginf=-MIN)

# 2. 对于四元数，需要归一化
quat = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-6)

# 3. 计算后再次清理
result = torch.nan_to_num(result, nan=0.0, posinf=MAX, neginf=0.0)
result = torch.clamp(result, MIN, MAX)
```

---

### 2. **可视化改进** (`goal_command.py`)

#### 2.1 颜色与大小调整

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/goal_command.py:103-127`

**修改前**:
```python
marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)  # 小箭头
marker_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.0, 0.0)  # 红色
max_envs_to_viz = min(4, self.num_envs)  # 只可视化 4 个环境
```

**修改后**:
```python
marker_cfg.markers["arrow"].scale = (1.0, 1.0, 1.0)  # 放大 2 倍
marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # 绿色
# 为所有环境创建可视化器（批量模式）
```

#### 2.2 批量可视化优化

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/goal_command.py:103-127, 468-485`

**修改前**: 为每个环境创建独立 VisualMarkers 实例（前 4 个环境）

**修改后**: 使用单个 VisualMarkers 实例批量绘制所有环境
```python
# 单个可视化器，批量处理
self.goal_visualizer = VisualizationMarkers(marker_cfg)

# 批量更新所有环境
def _update_goal_markers(self):
    marker_pos = self.goal_pos_w.clone()  # [num_envs, 3]
    marker_pos[:, 2] = 0.1  # 抬高一点
    marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
    marker_quat[:, 0] = 1.0
    self.goal_visualizer.visualize(marker_pos, marker_quat)
```

**效果**:
- ✅ 所有环境都能看到绿色 goal 箭头
- ✅ 性能更好（单个 USD prim 批量更新）
- ✅ 箭头更大，更容易观察

---

### 3. **配置调整** (`cfg/charge_env_cfg_phase0.py`)

#### 3.1 启用 `velocity_toward_goal` 奖励

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/cfg/charge_env_cfg_phase0.py:333-339`

**修改前**:
```python
# R_velocity: 朝向目標速度獎勵（移除，避免衝刺行為）
# velocity_toward_goal = RewTerm(...)  # ❌ 被注释掉
```

**修改后**:
```python
# R_velocity: 朝向目標速度獎勵（重新啟用，提供直接引導）
velocity_toward_goal = RewTerm(
    func=velocity_toward_goal,
    params={"asset_cfg": SceneEntityCfg("robot"), "min_dist": 0.5},
    weight=3.0,  # 給予適當權重，強化"朝目標移動"的行為
)
```

**原因**: 这是训练失败的核心原因 - 机器人缺少"朝目标移动"的直接引导信号

---

#### 3.2 降低任务难度

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/cfg/charge_env_cfg_phase0.py:426-434`

**修改前**:
```python
goal_reached = DoneTerm(
    func=goal_reached_dynamic,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "threshold": GOAL_REACH_THRESHOLD,
        "min_goals": 5,  # 需完成 5 个 Goal 才能终止
    },
)
```

**修改后**:
```python
# 使用标准终止条件（Phase 0 简化）
goal_reached = DoneTerm(
    func=goal_reached,  # 不是 goal_reached_dynamic
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "threshold": GOAL_REACH_THRESHOLD,
    },
)
# 事件配置中也调整了
min_goals: 5 → 1  # 只要到達 1 個目標就算成功
```

**原因**: 原任务太难（需要连续完成 5 个目标），导致机器人无法学习

---

### 4. **观测函数 NaN 处理** (`mdp/observations/functions.py`)

#### 4.1 `goal_position_in_robot_frame`

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp/observations/functions.py:207-237`

**添加的清理代码**:
```python
# 清理 _local_goal_world 的 NaN/Inf
goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

# 清理 goal_command 的 NaN/Inf
goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

# 清理机器人状态的 NaN/Inf
robot_pos_w = torch.nan_to_num(robot_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)
robot_quat_w = torch.nan_to_num(robot_quat_w, nan=0.0, posinf=1.0, neginf=-1.0)

# 归一化四元数
robot_quat_w = robot_quat_w / (torch.norm(robot_quat_w, dim=1, keepdim=True) + 1e-6)
```

---

#### 4.2 `base_velocity_xy`

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp/observations/functions.py:327-358`

**添加的清理代码**:
```python
# 在输入端清理 NaN/Inf
vel_w = torch.nan_to_num(vel_w, nan=0.0, posinf=10.0, neginf=-10.0)
quat_w = torch.nan_to_num(quat_w, nan=0.0, posinf=1.0, neginf=-1.0)
# 归一化四元数
quat_w = quat_w / (torch.norm(quat_w, dim=1, keepdim=True) + 1e-6)
```

---

#### 4.3 `base_angular_velocity_z`

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp/observations/functions.py:363-393`

**添加的清理代码**:
```python
# 在输入端清理 NaN/Inf
angular_vel_b = torch.nan_to_num(angular_vel_b, nan=0.0, posinf=5.0, neginf=-5.0)
```

---

### 5. **路径规划 NaN 处理** (`mdp/events/aitstar_integration.py`)

**文件**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp/events/aitstar_integration.py:203-232`

**添加的清理代码**:
```python
# 获取机器人和目标位置
robot_pos_world = robot.data.root_pos_w[env_ids, :2]
goal_pos_world = env.command_manager.get_command("goal_command")[env_ids, :2]
env_origins = env.scene.env_origins[env_ids, :2]

# 🔥 重要：清理所有输入数据的 NaN/Inf
robot_pos_world = torch.nan_to_num(robot_pos_world, nan=0.0, posinf=100.0, neginf=-100.0)
goal_pos_world = torch.nan_to_num(goal_pos_world, nan=0.0, posinf=100.0, neginf=-100.0)
env_origins = torch.nan_to_num(env_origins, nan=0.0, posinf=100.0, neginf=-100.0)

robot_pos_local = robot_pos_world - env_origins
goal_pos_local = goal_pos_world - env_origins
```

---

## 🐛 问题诊断与解决

### **训练失败分析**

**症状**:
- ep_len_mean: 528 → 432 → 324 (大幅缩短)
- ep_rew_mean: 99.8 → 93.3 → 78.8 (持续下降)
- Success: 0.0% (一直为 0)
- explained_variance: 0.879 (Critic 清醒，Actor 找不到方法)

**根本原因**:
1. ❌ `velocity_toward_goal` 被注释掉 - 缺少"朝目标移动"的直接引导
2. ❌ `min_goals = 5` - 任务太难（需要连续完成 5 个目标）
3. ❌ `progress_to_goal` 使用距离变化 - 侧向滑行被误判为有进度

**解决方案**:
1. ✅ 启用 `velocity_toward_goal` (weight=3.0)
2. ✅ 降低 `min_goals: 5 → 1`
3. ✅ 改进 `progress_to_goal` 为向量点积 × 朝向判断

---

## 📊 修改前后对比

| 指标 | 修改前 | 修改后 | 改进 |
|------|--------|--------|------|
| **velocity_toward_goal** | ❌ 禁用 | ✅ weight=3.0 | 提供直接引导 |
| **progress_to_goal** | 距离变化 | 向量点积×朝向 | 避免作弊行为 |
| **min_goals** | 5 | 1 | 降低任务难度 |
| **goal 可视化** | 红色,小,4个环境 | 绿色,大,全部环境 | 改善调试体验 |
| **NaN 处理** | 部分 | 全面（9个函数） | 提高训练稳定性 |

---

## 🚀 下一步行动

### **立即行动**:
1. 停止当前训练（已经走下坡路）
2. 使用修复后的配置重新开始训练
3. 监控前 10 个 rollout 的指标

### **预期效果**:
- ✅ ep_len_mean 应该**增加**（机器人活得更久）
- ✅ ep_rew_mean 应该**稳定或增加**
- ✅ **Success > 0%**（这是最关键的信号）

### **监控指标**:
```bash
# 查看训练日志
tensorboard --logdir runs/Phase0_2026-02-10_20-01-20

# 关键指标
- ep_len_mean: 应该 > 500（活得更久）
- ep_rew_mean: 应该 > 100（得分增加）
- rollouts/success_rate: 应该 > 0%（有成功案例）
```

### **进阶调整**（当 Success > 30% 后）:
```python
min_goals: 1 → 3 → 5           # 逐步增加目标数
velocity_toward_goal weight: 3.0 → 2.0 → 1.0  # 逐步降低引导
```

---

## 📝 技术规格总结

### **观测空间**: 131 维
- lidar_scan: 72 维（360°/5° = 72 条射线）
- speed: 2 维（本体坐标系线速度）
- goal_position: 2 维（相对位置）
- goal_distance: 1 维（欧式距离）
- time_remaining: 1 维（剩余时间比例）
- alive: 1 维（存活标志）
- actions: 2 维（上一步动作）
- obstacles: 50 维（padding，Phase 0 无障碍物）

### **LiDAR 规格**:
- 扫描范围: 360° 全向环绕
- 角度分辨率: 5°（72 条射线）
- 最大距离: 10 米
- 更新频率: 25 Hz（decimation=4）
- 投影方式: 2D 平面距离（丢弃 Z 轴）

### **控制频率**:
- 物理模拟: 100 Hz (dt=0.01s)
- 控制决策: 25 Hz (每 4 个物理步决策一次)
- 推理预算: 40 毫秒/决策

### **动作空间**: 2 维
- (linear_velocity, angular_velocity)
- 范围: [-1, 1] → 缩放到实际控制范围

---

## 📚 相关文档

- 项目路径: `/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/`
- 配置文件: `cfg/charge_env_cfg_phase0.py`
- 奖励函数: `mdp/rewards/goal_rewards.py`
- 观测函数: `mdp/observations/functions.py`
- Goal 可视化: `goal_command.py`

---

## ⚠️ 注意事项

1. **不要混合使用**:
   - 旧模型 + 新配置 ❌
   - 新模型 + 旧配置 ❌
   - ✅ 使用新配置从头训练

2. **调试建议**:
   ```bash
   # GUI 模式观察机器人行为
   python scripts/reinforcement_learning/sb3/train_charge.py \
       --num_envs 2 \
       --headless False
   ```

3. **成功指标**:
   - Phase 0 毕业标准: Success > 95%
   - 预计训练步数: 2M-5M steps
   - 建议环境数: 256-512

---

**修改时间**: 2026-02-10
**修改人**: Claude (Sonnet 4.5)
**项目**: Charge Navigation - Phase 0 Training Fix
