# Charge Navigation 项目进度记录

> 更新时间: 2026-02-10
> 目的: 记录项目架构、已完成工作和当前状态，供后续 Claude Code agent 参考

---

## 项目架构概述

### 训练/推理分离架构 🆕

**核心思想**: 训练时使用虚拟规划器（高 FPS），推理时使用真实 AIT*（全局规划）

```
┌─────────────────────────────────────────────────────────────┐
│                        训练阶段 (Training)                     │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │  Virtual Planner│ ──▶│  RL Agent       │                  │
│  │  (随机目标点)     │    │  (学习导航能力)   │                  │
│  └─────────────────┘    └─────────────────┘                  │
│  目的：高效训练，泛化能力强，FPS 高                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       推理阶段 (Inference)                     │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │  AIT* Planner   │ ──▶│  RL Agent       │                  │
│  │  (真实全局规划)   │    │  (执行导航)      │                  │
│  └─────────────────┘    └─────────────────┘                  │
│  目的：利用全局规划实现复杂导航                                 │
└─────────────────────────────────────────────────────────────┘
```

### 为什么训练时不使用 AIT*？

| 原因 | 说明 |
|------|------|
| **效率问题** | AIT* 需要算力，会拖慢训练 FPS。RL 需要几百万步才能收敛 |
| **数学等价** | 在空旷环境下，AIT* 规划的路径本质上就是「一条线」 |
| **解耦能力** | RL 只需学会「给定目标，走过去」，不关心目标来源 |
| **避免过拟合** | 防止 RL 依赖 AIT* 的特定路径特征 |

### 不同 Phase 的虚拟规划策略

| Phase | 真实 AIT* 行为 | 虚拟规划策略 |
|-------|----------------|-------------|
| **Phase 0** | 直线 | 在前方扇形区域（±30°）随机生成点 |
| **Phase 1** | 避开牆壁 | 在房间内部随机生成点 |
| **Phase 2** | 绕过障碍物 | 在障碍物旁边或后面生成目标 |
| **Phase 3** | 穿过窄门 | 把目标点设在窄门另一端 |

---

## 分层导航策略

```
┌─────────────────────────────────────────────────────────────┐
│                      全局路径规划 (AIT*)                      │
│  输入: 起点、终点、障碍物地图                                  │
│  输出: 全局路径航点列表 (waypoints)                           │
│  作用: 提供宏观导航方向，绕过大型障碍物                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    局部路径跟随 (RL Agent)                    │
│  输入: LiDAR观测、当前waypoint、机器人状态                     │
│  输出: (线速度, 角速度) 控制指令                               │
│  作用: 局部避障、精确控制、跟随waypoint                        │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策**:
- **使用 AIT* 而非 A***: 用户明确要求使用 AIT* 算法进行全局路径规划（推理时）
- **训练用虚拟规划器**: 训练时使用随机目标点，提高 FPS 和泛化能力 🆕
- **Carrot-on-stick**: RL agent 不直接冲向终点，而是跟随规划的 waypoint
- **分层训练**: 先在简单环境学习基础能力，再逐步增加复杂度

---

## Phase 设计

| Phase | 环境特点 | 虚拟规划策略 | 训练目标 | 状态 |
|-------|----------|-------------|----------|------|
| **Phase 0** | 16x16m 空旷房间，只有四面边界墙 | 前方扇形区域随机点 | 车辆动力学、waypoint跟随 | ✅ 可训练 |
| **Phase 1** | 添加内部障碍物 | 房间内部随机点 | 局部避障 + 路径跟随 | 待开发 |
| **Phase 2** | 更复杂障碍物配置 | 障碍物旁/后面生成 | 高级导航能力 | 待开发 |
| **Phase 3** | 动态/随机障碍物 | 窄门另一端目标 | 适应性和鲁棒性 | 待开发 |

---

## 新增功能

### 7. 虚拟规划器实现 🆕
**文件**: `mdp/path_planner/virtual_planner.py`
**功能**: 训练时使用随机目标点代替 AIT*，提高训练效率
**配置**:
```python
# charge_env_cfg_phase0.py
plan_aitstar_path = EventTerm(
    params={
        "use_virtual_planner": True,  # 启用虚拟规划器
        ...
    }
)
```

**Phase 0 策略**:
- 在朝向目标的方向上随机选择距离（2m ~ 5m）
- 添加 ±30° 角度偏移，模拟路径跟随的小误差
- 返回随机航点供 RL 跟随

---

## 已完成的修复

### 1. ImportError 修复
**文件**: `wall_geometry_sampler.py:654`
**问题**: `__all__` 中列出了不存在的函数 `sample_wall_geometry_from_usd`
**修复**: 从 `__all__` 列表中移除该函数名

### 2. AIT* 数值 Bug 修复
**文件**: `globle_planner/src/aitstar_path_planner/aitstar_path_planner/aitstar.py:422`
**问题**: `prev_cost = float('inf')` 导致 `inf - inf = nan`，触发 RuntimeWarning
**修复**:
```python
# 修复前:
if self.solution_cost < float('inf'):
    improvement = (prev_cost - self.solution_cost) / max(prev_cost, 1e-6)

# 修复后:
if self.solution_cost < float('inf') and prev_cost < float('inf'):
    improvement = (prev_cost - self.solution_cost) / max(prev_cost, 1e-6)
```

### 3. 可视化 Z 高度对齐
**文件**: `mdp/path_planner/path_visualizer.py:132`
**问题**: 绿色路径在 z=0.3m，红色箭头在 z=0.0m，视觉上看起来不对齐
**修复**: 将路径 Z 坐标从 0.3m 改为 0.1m，与箭头高度更接近

### 4. LiDAR 调试可视化关闭
**文件**: `charge_env_cfg_phase{0,1,2,3}.py`
**问题**: 地面上有大量红色射线（LiDAR debug_vis）
**修复**: 将 `debug_vis` 设为 `False`，减少视觉干扰

### 5. 墙壁网格点颜色更改
**文件**: `wall_geometry_sampler.py:672`
**问题**: 墙壁网格点是红色，与 LiDAR 射线混淆
**修复**: 从红色 `(1.0, 0.0, 0.0)` 改为紫色 `(0.5, 0.0, 1.0)`

### 6. 坐标对比调试输出
**文件**: `mdp/events/aitstar_integration.py:318-357`
**添加**: 详细的 AIT* 规划器输出与红色箭头坐标对比
```
🎯 紅色箭頭（目標位置 GoalCommand）:
   世界坐標: [x, y]
   局部坐標: [x, y]
📍 AIT* 路徑終點（局部坐標）:
   path[-1]: [x, y]
✓ 座標驗證:
   start 誤差: 0.0000m  ✓ 正確
   goal 誤差:  0.0000m   ✓ 正確
```

---

## 关键配置参数

### Phase 0 AIT* 集成配置
**文件**: `cfg/charge_env_cfg_phase0.py:546-559`

```python
plan_aitstar_path = EventTerm(
    func=plan_aitstar_and_update_local_goal,
    mode="reset",
    params={
        "lookahead_distance": 2.0,      # Carrot-on-stick 前瞻距離
        "map_size": (16.0, 16.0),        # 地圖大小
        "robot_radius": 0.3,             # 機器人半徑
        "visualize_path": True,          # 啟用 AIT* 路徑可視化
        "sample_walls_from_usd": True,   # 從配置添加牆壁幾何
        "run_verification": True,        # 運行驗收測試
        "phase": "phase0",               # Phase 0：只有四面邊界牆
        "use_astar": False,              # 使用 AIT* 算法（重要！）
    },
)
```

**重要**: `use_astar: False` - 用户明确要求使用 AIT*，不要改成 A*

---

## 观测空间设计 (131维)

当前所有 Phase 统一使用 131 维观测空间：

| 模块 | 维度 | 说明 |
|------|------|------|
| base_velocity | 3 | [vx, vy, ω] 机器人速度 |
| goal_heading | 3 | [cos(θ), sin(θ), θ_norm] 朝向目标的航向角 |
| goal_position | 3 | [x, y, z] 目标相对位置 |
| lidar_2d | 120 | 120束 LiDAR 距离测量 |
| tracked_obj_position | 2 | [x, y] 追踪的waypoint位置（当前未使用） |

---

## 训练命令

### Phase 0 训练
```bash
python scripts/reinforcement_learning/sb3/train_charge.py --task Isaac-Navigation-Charge-Phase0
```

### 其他 Phase
```bash
python scripts/reinforcement_learning/sb3/train_charge.py --task Isaac-Navigation-Charge-Phase1
```

---

## 待办事项

### 高优先级
- [ ] Phase 0 训练验证 - 确认 RL agent 能学会跟随 waypoint
- [ ] Phase 1 障碍物配置设计
- [ ] 验证 Phase 1 中 AIT* 能生成复杂路径

### 中优先级
- [ ] 奖励函数调优
- [ ] 课程学习策略设计
- [ ] 评估指标定义

---

## 重要提醒

### 算法选择
- **必须使用 AIT***: 用户明确要求"我就要AIT*"，不要私自改成 A* 或其他算法
- **用途**: AIT* 用于全局路径规划，RL 用于局部控制

### Phase 0 的 AIT* 路径
- **正常现象**: Phase 0 中 AIT* 生成 `[2, 2]` 路径（只有起点和终点的直线）
- **原因**: 没有内部障碍物，最短路径就是直线
- **目的**: 让 RL 学习基础的 waypoint 跟随能力

### 文件修改原则
- 修改前先阅读文件，理解现有逻辑
- 避免过度工程化
- 保留中文注释，便于理解

---

## 关键文件清单

### 配置文件
- `cfg/charge_env_cfg_phase0.py` - Phase 0 环境配置
- `cfg/charge_env_cfg_phase1.py` - Phase 1 环境配置
- `cfg/charge_env_cfg_phase2.py` - Phase 2 环境配置
- `cfg/charge_env_cfg_phase3.py` - Phase 3 环境配置

### MDP 模块
- `mdp/events/aitstar_integration.py` - AIT* 集成逻辑
- `mdp/path_planner/path_visualizer.py` - 路径可视化
- `mdp/path_planner/wall_geometry_sampler.py` - 墙壁几何采样

### 全局规划器
- `globle_planner/src/aitstar_path_planner/aitstar_path_planner/aitstar.py` - AIT* 核心算法

### 可视化
- `goal_command.py` - 红色箭头目标标记
- `scripts/demos/test_aitstar_visualization.py` - AIT* 可视化测试

---

## 联系和反馈

如有问题或需要继续开发，请参考本文档了解当前状态和设计意图。

---

## 2026-02-10 更新記錄 🆕

### 核心策略變更：「敢衝、敢撞」訓練策略

**設計理念**：優先讓 Agent 學會導航，碰撞懲罰很低，鼓勵探索

```
碰撞 → 死亡機制（立即終止重置）+ 輕微懲罰
結果：Agent 願意冒險嘗試到達目標
```

### 1. Episode 內動態 Goal 重置功能 🆕

**新增文件**:
- `mdp/events/dynamic_goal.py` - 動態 Goal 重置事件
- `mdp/rewards/dynamic_goal_rewards.py` - 動態 Goal 獎勵函數
- `mdp/terminations/dynamic_goal.py` - 動態 Goal 終止條件

**核心功能**:
```
Agent 抵達 Goal → Goal 重置到新位置 → Episode 繼續
                         ↓
                    累積獎勵（遞增機制）
```

**配置示例** (Phase 0):
```python
dynamic_goal_init = EventTerm(
    func=initialize_dynamic_goal_respawn,
    mode="startup",
    params={"min_goals": 3, "max_goals": 5},
)

dynamic_goal_respawn = EventTerm(
    func=respawn_goal_on_reach,
    mode="interval",
    interval_range_s=(1.0, 1.0),
    params={
        "threshold": 0.5,
        "reward_on_respawn": 10.0,
        "enable_progressive_reward": True,   # 🆕 遞增獎勵
        "progressive_multiplier": 0.5,       # 🆕 每多 1 個 Goal +50%
    },
)
```

### 2. 虛擬規劃器目標追蹤修復 🆕

**問題**: VirtualPlanner 生成隨機 waypoint，但用戶希望 Agent 追蹤紅色箭頭（goal_command）

**修復** (`mdp/events/aitstar_integration.py`):
```python
# 虛擬規劃器模式下，直接使用 goal_command（紅色箭頭）作為局部目標
if use_virtual_planner:
    local_goal = goal  # 使用目標位置，而非隨機 waypoint
else:
    local_goal = extract_local_goal(path)  # AIT* 模式提取航點
```

### 3. 碰撞懲罰權重大幅降低

**策略**: Pull >> Push，讓 Agent 敢於嘗試

| Phase | 碰撞懲罰 | 之前 | 變動 |
|-------|---------|------|------|
| **Phase 0** | `safety_field` | -10.0 | **-1.0** |
| **Phase 0** | `collision_penalty` | -30.0 | **-1.0** |
| **Phase 3** | `dynamic_collision` | -10.0 | **-1.0** |

**結果**: Pull/Push = 10.5:1，Agent 優先考慮到達目標

### 4. LiDAR 碰撞終止條件統一

**新增函數** (`mdp/terminations/collision.py`):
- `lidar_collision(env, sensor_cfg, threshold=0.5)` - LiDAR 距離檢測
- `physx_contact_collision(env, sensor_cfg, force_threshold=0.1)` - PhysX 物理接觸檢測

**各 Phase 終止條件**:
| Phase | LiDAR 碰撞終止 | threshold | 狀態 |
|-------|---------------|-----------|------|
| **0** | ✅ `collision` (LiDAR) | 0.5m | 完整 |
| **1** | ✅ `lidar_collision` | 0.5m | 完整 |
| **2** | ✅ `lidar_collision` | 0.5m | 完整 |
| **3** | ✅ `dynamic_collision` | 0.5m | 完整 |

### 5. 遞增獎勵機制 🆕

**目的**: 鼓勵 Agent 學會「活得越久，連續到達更多 Goal」

```
第 1 個 Goal: 10.0 × 1.0 = 10.0
第 2 個 Goal: 10.0 × 1.5 = 15.0 (+50%)
第 3 個 Goal: 10.0 × 2.0 = 20.0 (+100%)
第 4 個 Goal: 10.0 × 2.5 = 25.0 (+150%)
第 5 個 Goal: 10.0 × 3.0 = 30.0 (+200%)
```

**公式**: `R = base_reward × (1 + 0.5 × (count - 1))`

### 6. Phase 0 獎勵總結 (v3)

```
Pull Forces (吸引到目標):
  R_progress: +12.0
  R_goal: +8.0
  R_respawn: +1.0 (可變 +10~+30)
  R_dynamic_goal: +2.0
  ───────────────────────────
  總 Pull ≈ +23.0+

Push Forces (推離障礙物):
  P_safety: -1.0
  P_collision: -1.0
  P_smooth: -0.2
  P_time: -0.1
  ───────────────────────────
  總 Push = -2.2

Pull/Push = 10.5:1  →  Agent 優先到達目標，不怕碰撞！
```

### 7. 關鍵配置修改清單

| 文件 | 修改內容 |
|------|----------|
| `cfg/charge_env_cfg_phase0.py` | 碰撞懲罰 -30.0→-1.0，新增動態Goal重置，遞增獎勵 |
| `cfg/charge_env_cfg_phase1.py` | 新增 LiDAR 碰撞終止條件 `lidar_collision` |
| `cfg/charge_env_cfg_phase2.py` | 新增 LiDAR 碰撞終止條件 `lidar_collision` |
| `cfg/charge_env_cfg_phase3.py` | 碰撞懲罰 -10.0→-1.0，修復 `sensor_cfg` 參數 |
| `mdp/events/dynamic_goal.py` | 動態 Goal 重置事件實現（3個函數） |
| `mdp/rewards/dynamic_goal_rewards.py` | 動態 Goal 獎勵函數（3個函數） |
| `mdp/terminations/dynamic_goal.py` | 動態 Goal 終止條件（2個函數） |
| `mdp/terminations/collision.py` | 新增 `lidar_collision`, `physx_contact_collision` |
| `mdp/events/aitstar_integration.py` | 虛擬規劃器直接使用 `goal_command` |
| `mdp/observations/functions.py` | 更新註釋說明訓練時追蹤紅色箭頭 |

### 8. 待辦事項更新

- [x] Phase 0 碰撞懲罰降低 ✅
- [x] 所有 Phase 添加 LiDAR 碰撞終止條件 ✅
- [x] Episode 內動態 Goal 重置功能 ✅
- [x] 虛擬規劃器追蹤紅色箭頭 ✅
- [ ] Phase 0 訓練驗證
- [ ] Phase 1-3 障礙物配置優化

### 9. 重要提醒 (下次 Claude Code 參考)

#### 訓練策略：「敢衝、敢撞」
- **碰撞懲罰很低**（-1.0），讓 Agent 願意冒險
- **死亡機制保留**: 碰撞後立即終止，但懲罰輕微
- **Pull/Push = 10.5:1**: 吸引力遠大於排斥力

#### VirtualPlanner 行為
- 訓練時使用虛擬規劃器（不調用 AIT*）
- Agent 追蹤 **紅色箭頭**（goal_command），不是隨機 waypoint
- 推理時才使用 AIT* 進行全局路徑規劃

#### 遞增獎勵
- 每多完成 1 個 Goal，獎勵增加 50%
- 鼓勵 Agent 在一個 episode 中完成更多 Goal

#### 終止條件
- 所有 Phase 都有 LiDAR 碰撞終止條件（threshold=0.5m）
- 碰撞後 episode 立即終止（死亡機制）
- 終止條件與獎勵懲罰分開設計


---

## 2026-02-10 晚間訓練調試與 NaN 問題修復 🔧

### 問題背景

訓練過程中發現以下問題：
1. **觀測包裝器順序錯誤** - `SanitizeObservationsWrapper` 位置不正確
2. **獎勵函數產生 NaN** - `dynamic_goal_rewards.py` 缺少 nan 處理
3. **策略網絡權重變成 NaN** - 訓練到 786k 步後崩潰
4. **任務難度過高** - 目標距離 3-8m、需完成 5 個 Goal → Success 0%

### 修改 1：觀測清理與包裝器優化

**文件**: `scripts/reinforcement_learning/sb3/train_charge.py`

```python
# 新增 SanitizeObservationsWrapper（在 IsaacLabMetricsWrapper 之後）
env = SanitizeObservationsWrapper(env)

# 添加 NanProtectionCallback（監控策略網絡權重）
nan_protection = NanProtectionCallback(verbose=1)
callbacks.append(nan_protection)
```

**功能**:
- 自動清理觀測中的 nan/inf
- 清理獎勵中的 nan/inf（每步檢查）
- 每 1000 步檢查網絡權重
- 發現 nan 時立即停止訓練

### 修改 2：獎勵函數 NaN 修復

**文件**: `mdp/rewards/dynamic_goal_rewards.py`

| 函數 | 修改內容 |
|------|----------|
| `dynamic_goal_progress_reward` | 添加 `torch.nan_to_num` 清理 |
| `dynamic_goal_respawn_reward` | 添加 `torch.nan_to_num` 清理 |
| `dynamic_goal_bonus_reward` | 添加 `torch.nan_to_num` 清理 |

**文件**: `mdp/rewards/perception_rewards.py`

| 函數 | 修改內容 |
|------|----------|
| `lidar_clearance_reward` | 除法添加 `+ 1e-6` 保護 |

### 修改 3：Phase 0 任務難度大幅降低

| 參數 | 修改前 | 修改後 | 原因 |
|------|--------|--------|------|
| **目標距離** | 3-8 米 | **1.5-3 米** | 太遠，Agent 學不會 |
| **終止條件** | `goal_reached_dynamic` | `goal_reached` | 簡化邏輯 |
| **min_goals** | 5 個 | **移除**（使用標準終止）| 太難，從未成功 |
| **遞增獎勵** | 啟用 | **禁用** | 只有 1 個 Goal 不需要 |

**文件**: `cfg/charge_env_cfg_phase0.py`

### 修改 4：PPO 超參數優化

**文件**: `agents/sb3_ppo_cfg_phase0.yaml`

| 參數 | 修改前 | 修改後 | 原因 |
|------|--------|--------|------|
| `learning_rate` | 1e-4 | **5e-5** | 防止梯度爆炸 |
| `max_grad_norm` | 1.0 | **0.5** | 更激進的梯度裁剪 |
| `ortho_init` | False | **True** | 穩定初始化 |

### 修改 5：VecNormalize 獎勵裁剪

**文件**: `scripts/reinforcement_learning/sb3/train_charge.py`

```python
clip_reward=10.0  # 原來是 np.inf（無限制）
```

### 修改 6：Goal Command 可視化優化

**文件**: `goal_command.py`

| 問題 | 解決方案 |
|------|----------|
| 為每個環境創建獨立可視化器（2048 個！） | 使用單個批量可視化器 |
| headless 模式不必要渲染 | 添加 `disable_debug_vis()` 函數 |

### 修改 7：PPO Clip 確認

**檢查結果**: ✅ `clip_range: 0.2` 已正確配置

```
# sb3_ppo_cfg_phase0.yaml
clip_range: !!float 0.2  # ✅ 標準值
```

### 訓練命令

```bash
# 基礎訓練（headless 模式）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 2048 \
    --headless

# GUI 模式（調試用）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256 \
    --video
```

### Phase 0 最終配置（簡化版）

**終止條件**: 抵達目標即終止（1 個 Goal）

**獎勵結構**:
```
成功 Episode 總獎勷 ≈ 20-30 分
├── progress_to_goal (+12): 累計朝向目標移動
├── reaching_goal (+8): 抵達目標一次性獎勵
├── time_out_penalty (-1): 時間懲罰
├── action_smoothness (-0.2): 平滑度
├── safety_field (-1): 障礙物接近
└── collision_penalty (-1): 碰撞
```

**Pull/Push = 30:1** → 強烈鼓勵探索

### 待解決問題

- [ ] 驗證 Phase 0 訓練是否成功（Success 率 > 30%）
- [ ] 確認「神風特攻隊」行為是否出現
- [ ] 如果行為異常，調整 collision_penalty 權重
- [ ] Phase 1-3 需要調整獎勵權重（建議 -10 或更高）

---

**上次更新**: 2026-02-10 早上（Day 1 修改）
**本次更新**: 2026-02-10 晚間（Day 1 晚間調試與修復）
