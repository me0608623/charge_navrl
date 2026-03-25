# Phase 2.5 設計筆記：自適應課程學習框架

## 📋 目錄
1. [概述](#概述)
2. [自適應課程學習框架](#自適應課程學習框架)
3. [難度調整機制](#難度調整機制)
4. [統計追蹤](#統計追蹤)
5. [動態參數調整](#動態參數調整)
6. [配置說明](#配置說明)
7. [使用指南](#使用指南)

---

## 概述

**Phase 2.5 設計目標：**
- 實現自適應課程學習框架，根據訓練表現動態調整環境難度
- 根據導航成功率與碰撞率自動調整障礙物數量和距離參數
- 平衡訓練效率和學習效果，避免過難或過易的環境
- **修復 Phase 2 的訓練問題：獎勵失衡、起始難度過高、局部最優策略**

**文件位置：**
- 配置文件：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/charge_env_cfg_v2_5.py`
- MDP 函數：`charge_mdp.py`（新增自適應課程學習函數）
- PPO 配置：`agents/rsl_rl_ppo_cfg_v2_5.py`
- 環境 ID：`Isaac-Navigation-Charge-v2.5`（訓練）、`Isaac-Navigation-Charge-Play-v2.5`（測試）

### Phase 2 訓練問題分析

**發現的問題：**
1. ❌ 成功率在 10k 步後崩潰（從 0.4 降至接近 0）
2. ❌ 碰撞率持續高（collision 達到 0.4-0.5）
3. ❌ 安全導航獎勵暴跌（safe_navigation 從 0.01 降至 0）
4. ❌ 速度控制持續違規（speed_control 持續為負）

**根本原因：**
1. **獎勵機制嚴重失衡**：進度獎勵過高（5.0），安全獎勵過低（0.1）
2. **起始難度過高**：5 個障礙物對初學者太難
3. **機器人學到局部最優策略**：高速衝刺導致碰撞

**Phase 2.5 修復策略：**
1. **重新平衡獎勵機制**：降低進度獎勵，提高安全獎勵和碰撞懲罰
2. **降低起始難度**：從 3 個障礙物開始（而不是 5 個）
3. **更保守的課程學習**：更嚴格的難度調整閾值，確保穩定性

---

## 自適應課程學習框架

### 1. 核心概念

**自適應課程學習（Adaptive Curriculum Learning）** 是一種動態調整訓練難度的方法：
- 根據智能體的當前表現（成功率和碰撞率）自動調整環境難度
- 當智能體表現良好時增加難度，表現不佳時降低難度
- 確保訓練始終在適當的難度範圍內進行

### 2. 難度等級

| 難度等級 | 障礙物數量 | 說明 |
|---------|-----------|------|
| 3 | 3 個 | 最小難度（初始設置） |
| 4 | 4 個 |  |
| 5 | 5 個 | Phase 2 的難度 |
| 6 | 6 個 |  |
| 7 | 7 個 |  |
| 8 | 8 個 |  |
| 9 | 9 個 |  |
| 10 | 10 個 | 最大難度 |

### 3. 難度調整邏輯

**增加難度的條件（更保守）：**
- 成功率 ≥ **75%**（從 80% 降低）
- 碰撞率 < **25%**（從 20% 放寬）
- 當前難度 < 最大難度（10）

**降低難度的條件（更積極）：**
- 成功率 < **60%**（從 50% 提高） **或** 碰撞率 > **25%**（從 40% 降低）
- 當前難度 > 最小難度（3）

**保持當前難度：**
- 其他情況

**調整理由：**
- 根據 Phase 2 訓練問題，成功率在 10k 步後崩潰，說明難度增加太快
- 更保守的難度增加策略，確保訓練穩定性
- 更積極的難度降低策略，避免訓練停滯

---

## 難度調整機制

### 1. 統計追蹤

**滑動窗口統計：**
- 窗口大小：100 個 episodes
- 計算滑動平均成功率和碰撞率
- 使用最近 10 個 episodes 的平均值進行決策

**統計來源：**
- 成功率：從終止管理器的 `goal_reached` 終止條件獲取
- 碰撞率：從終止管理器的 `collision` 或 `collision_contact` 終止條件獲取

### 2. 參數調整策略

**根據難度等級動態調整距離參數：**

| 參數 | 基礎值（難度 3） | 最大值（難度 10） | 調整公式 |
|------|----------------|-----------------|---------|
| `min_robot_distance` | 3.5 米 | 2.45 米 | `base × (1.0 - factor × 0.3)` |
| `min_goal_distance` | 2.0 米 | 1.4 米 | `base × (1.0 - factor × 0.3)` |
| `min_obstacle_spacing` | 2.0 米 | 1.4 米 | `base × (1.0 - factor × 0.3)` |

其中 `factor = (difficulty - 3) / (10 - 3)`，範圍 [0, 1]

**設計理由：**
- 難度越高，距離參數越小（更困難）
- 但不會過度縮小（最多減少 30%），確保基本的安全性

---

## 統計追蹤

### 1. 統計獲取方法

**方法 1：從終止管理器獲取**
```python
# 從 _last_episode_dones 中獲取
goal_reached = termination_manager._last_episode_dones[:, goal_reached_idx]
success_rate = goal_reached.float().mean().item()
```

**方法 2：從 extras 中獲取**
```python
# 從 env.extras["log"] 中獲取 Episode_Termination 統計
for key, value in env.extras["log"].items():
    if "Episode_Termination" in key and "goal_reached" in key:
        success_rate = float(value)
```

### 2. 終止條件名稱

- **成功率：** `goal_reached` 終止條件
- **碰撞率：** `collision` 或 `collision_contact` 終止條件（任一觸發都算碰撞）

### 3. 更新時機

**課程學習項更新：**
- 在每個 step 結束時檢查 `reset_buf`
- 如果有環境重置（episode 結束），則更新統計並調整難度
- 更新後的參數存儲在 `env._adaptive_curriculum_params` 中

**參數應用：**
- 在 `reset_obstacles_with_adaptive_params` 事件中應用更新後的參數
- 根據當前難度等級調整障礙物數量和距離參數

---

## 動態參數調整

### 1. 障礙物數量調整

**實現方式：**
- 場景中創建 MAX_OBSTACLES（10）個障礙物
- 根據當前難度等級，只啟用前 N 個障礙物（N = 當前難度）
- 未使用的障礙物移動到場景外（Z = -10 米）

**優點：**
- 避免動態創建/刪除障礙物的複雜性
- 保持場景結構穩定
- 支持快速難度調整

### 2. 距離參數調整

**動態調整的參數：**
- `min_robot_distance`：機器人與障礙物最小距離
- `min_goal_distance`：目標與障礙物最小距離
- `min_obstacle_spacing`：障礙物之間最小間距

**調整公式：**
```python
difficulty_factor = (current_difficulty - min_difficulty) / (max_difficulty - min_difficulty)
adjusted_distance = base_distance * (1.0 - difficulty_factor * 0.3)
```

### 3. 觀測配置調整

**動態觀測：**
- `dynamic_obstacles_state` 函數優先使用 `env._num_obstacles`
- 觀測維度固定為 `MAX_OBSTACLES × 5`（支持權重遷移）
- 不存在的障礙物用 `[0, 0, 0, 0, -1]` 填充（size = -1 標記不存在）

---

## 配置說明

### 1. 場景配置

```python
class MySceneCfgV2_5(MySceneCfgV2):
    """場景配置（Phase 2.5）
    
    創建 MAX_OBSTACLES（10）個障礙物，但初始只啟用 3 個。
    後續會根據課程學習動態調整啟用的數量。
    """
```

### 2. 課程學習配置

```python
class CurriculumsCfgV2_5:
    """自適應課程學習配置（根據 Phase 2 問題調整）"""
    
    adaptive_difficulty = CurrTerm(
        func=charge_mdp.adaptive_curriculum_update,
        params={
            "target_success_rate": 0.65,  # 目標成功率：65%（從 70% 降低，更現實）
            "max_collision_rate": 0.25,  # 最大可接受碰撞率：25%（從 30% 降低，更嚴格）
            "difficulty_increase_threshold": 0.75,  # 難度增加閾值：75%（從 80% 降低，更保守）
            "difficulty_decrease_threshold": 0.60,  # 難度降低閾值：60%（從 50% 提高，更積極）
        },
    )
```

**調整說明：**
- **目標成功率**：70% → 65%（更現實的目標，避免過度激勵）
- **最大碰撞率**：30% → 25%（更嚴格的安全要求）
- **難度增加閾值**：80% → 75%（更保守的難度提升，避免過快增加）
- **難度降低閾值**：50% → 60%（更積極的難度降低，避免訓練停滯）

### 3. 事件配置

```python
class EventCfgV2_5(EventCfgV2):
    """事件配置（Phase 2.5）"""
    
    reset_obstacles = EventTerm(
        func=charge_mdp.reset_obstacles_with_adaptive_params,
        mode="reset",
        params={
            "boundary": 8.0,  # 場景邊界：8.0 米
            # 其他參數會在運行時根據課程學習結果動態設置
        },
    )
```

---

## 使用指南

### 1. 訓練命令

**從 Phase 2 checkpoint 繼續訓練：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v2.5 \
    --num_envs 128 \
    --resume \
    --load_run "/home/aa/IsaacLab/logs/rsl_rl/charge_navigation_phase2/charge_phase2"
```

**命令參數說明：**
- `--task Isaac-Navigation-Charge-v2.5`：指定 Phase 2.5 環境（自適應課程學習）
- `--num_envs 128`：並行環境數量（可根據 GPU 記憶體調整）
- `--resume`：**必須添加**，否則即使配置文件設定 `resume=True` 也會被覆蓋為 `False`
- `--load_run`：指定 Phase 2 checkpoint 的路徑（絕對路徑，支持跨目錄載入）
- **不指定 `--checkpoint`**：系統會自動使用最新的 checkpoint（匹配 `model_.*\.pt$` 模式）

**重要注意事項：**
1. **必須添加 `--resume` 參數**：即使配置文件中 `resume = True`，命令行參數會覆蓋配置
2. **不指定 `--checkpoint`**：讓系統自動選擇最新的 checkpoint（例如 `model_15998.pt`）
3. **如果指定 `--checkpoint`**：必須確保該文件存在，否則會報錯
4. **訓練日誌位置**：`logs/rsl_rl/charge_navigation_phase2_5/`（新的實驗名稱，不會覆蓋 Phase 2）

### 2. 監控訓練

**課程學習日誌：**
- 當難度調整時，會打印日誌：
  ```
  [Adaptive Curriculum] 增加難度: 3 -> 4 (成功率: 85.2%, 碰撞率: 15.3%)
  [Adaptive Curriculum] 降低難度: 5 -> 4 (成功率: 45.1%, 碰撞率: 42.3%)
  ```

**統計追蹤：**
- 成功率歷史：`_adaptive_curriculum_stats["success_rate_history"]`
- 碰撞率歷史：`_adaptive_curriculum_stats["collision_rate_history"]`
- 當前難度：`_adaptive_curriculum_stats["current_difficulty"]`

### 3. 參數調整建議

**如果訓練太容易（難度增加太快）：**
- 提高 `difficulty_increase_threshold`（例如：0.8 → 0.85）
- 降低 `target_success_rate`（例如：0.7 → 0.65）

**如果訓練太困難（難度降低太快）：**
- 降低 `difficulty_decrease_threshold`（例如：0.5 → 0.45）
- 提高 `max_collision_rate`（例如：0.3 → 0.35）

**如果難度調整太頻繁：**
- 增加 `window_size`（例如：100 → 200）
- 使用更長的滑動平均（例如：最近 20 個 episodes 而不是 10 個）

---

## 設計優勢

### 1. 自動化難度調整
- 無需手動調整難度參數
- 根據實際訓練表現自動優化

### 2. 平衡訓練效率
- 避免過難導致訓練停滯
- 避免過易導致學習不充分

### 3. 平滑難度過渡
- 使用滑動窗口統計，避免單次波動影響
- 逐步調整難度，不會突然變化

### 4. 靈活配置
- 可調整閾值和目標值
- 支持不同的難度調整策略

---

## 技術實現細節

### 1. 全局狀態管理

```python
_adaptive_curriculum_stats = {
    "success_rate_history": [],  # 成功率歷史
    "collision_rate_history": [],  # 碰撞率歷史
    "window_size": 100,  # 滑動窗口大小
    "current_difficulty": 3,  # 當前難度等級
    "min_difficulty": 3,  # 最小難度
    "max_difficulty": 10,  # 最大難度
}
```

### 2. 參數存儲

```python
# 課程學習參數存儲在環境中
env._adaptive_curriculum_params = {
    "num_obstacles": 5,
    "min_robot_distance": 3.15,
    "min_goal_distance": 1.8,
    "min_obstacle_spacing": 1.8,
    "success_rate": 0.75,
    "collision_rate": 0.25,
    "difficulty": 5,
}
```

### 3. 障礙物管理

- 創建 10 個障礙物（MAX_OBSTACLES）
- 根據難度等級只啟用前 N 個
- 未使用的障礙物移動到場景外（Z = -10 米）

---

## 獎勵機制重新平衡

### Phase 2 獎勵問題

**Phase 2 的獎勵配置導致機器人學到「高速衝刺」策略：**
- `progressive_collision`: weight=-1.0（懲罰太低）
- `distance_to_goal`: weight=5.0（進度獎勵太高）
- `reaching_goal`: weight=500（目標獎勵太高）
- `safe_navigation`: weight=0.1（安全獎勵太低）
- `speed_control_near_obstacles`: weight=0.1（速度控制太低）

### Phase 2.5 獎勵調整

| 獎勵項 | Phase 2 | Phase 2.5 | 調整理由 |
|--------|---------|-----------|---------|
| `progressive_collision` | -1.0 | **-2.5** | 提高碰撞懲罰，強化安全意識 |
| `distance_to_goal` | 5.0 | **2.5** | 降低進度獎勵，避免過度激勵 |
| `reaching_goal` | 500 | **300** | 降低目標獎勵，平衡安全與效率 |
| `safe_navigation` | 0.1 | **0.8** | 大幅提高安全獎勵，強化安全行為 |
| `speed_control_near_obstacles` | 0.1 | **0.5** | 大幅提高速度控制，強化提前減速 |
| `max_safe_speed` | 1.0 m/s | **0.8 m/s** | 降低最大安全速度，強化提前減速策略 |

**調整效果預期：**
- 機器人更重視安全，避免高速衝刺
- 平衡安全與效率，不會過度追求速度
- 強化提前減速和迴避策略

---

## 未來改進方向

1. **更細粒度的難度調整**
   - 不僅調整障礙物數量，還可以調整障礙物大小、移動速度等

2. **多維度統計**
   - 不僅考慮成功率和碰撞率，還可以考慮平均 episode 長度、平均獎勵等

3. **自適應閾值**
   - 根據訓練階段動態調整閾值（早期更寬鬆，後期更嚴格）

4. **難度預測**
   - 使用機器學習模型預測最優難度，而不是基於規則

---

---

## Phase 2 訓練問題修復總結

### 1. 獎勵機制重新平衡 ✅

**問題：** 進度獎勵過高（5.0），安全獎勵過低（0.1），導致機器人學到「高速衝刺」策略

**修復：**
- 降低進度獎勵：5.0 → 2.5
- 降低目標獎勵：500 → 300
- 提高碰撞懲罰：-1.0 → -2.5
- 提高安全獎勵：0.1 → 0.8
- 提高速度控制：0.1 → 0.5

### 2. 起始難度降低 ✅

**問題：** 5 個障礙物對初學者太難，成功率在 10k 步後崩潰

**修復：**
- 初始難度：3 個障礙物（而不是 5 個）
- 更保守的難度增加策略（75% 成功率才增加）
- 更積極的難度降低策略（60% 成功率就降低）

### 3. 課程學習閾值調整 ✅

**問題：** 難度調整策略不夠保守，導致訓練不穩定

**修復：**
- 目標成功率：70% → 65%（更現實）
- 最大碰撞率：30% → 25%（更嚴格）
- 難度增加閾值：80% → 75%（更保守）
- 難度降低閾值：50% → 60%（更積極）

### 預期效果

1. **訓練穩定性提升**：成功率不會在 10k 步後崩潰
2. **碰撞率降低**：機器人更重視安全，避免高速衝刺
3. **安全導航獎勵提升**：safe_navigation 不會降至 0
4. **速度控制改善**：speed_control 不會持續為負值

---

**最後更新：** 2026-01-19
**版本：** Phase 2.5（自適應課程學習 + 獎勵重新平衡）
