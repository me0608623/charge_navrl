# 動態重規劃與 Frontier 探索

本文檔詳細說明層級式導航系統的動態重規劃機制和未知環境探索策略。

---

## 📋 目錄

1. [動態重規劃](#動態重規劃)
2. [域隨機化](#域隨機化)
3. [Frontier 探索](#frontier-探索)
4. [完整系統架構](#完整系統架構)
5. [使用示例](#使用示例)
6. [關鍵參數調優](#關鍵參數調優)
7. [故障排除](#故障排除)

---

## 動態重規劃

### 為什麼需要動態重規劃？

AIT* 全域規劃器基於**靜態地圖**規劃路徑，但實際環境中：

- **地圖可能不完整**：訓練時地圖可能沒有包含所有障礙物
- **障礙物可能移動**：動態障礙物會改變環境
- **機器人可能偏離**：RL 為了避障可能偏離原路徑太遠

### 重規劃觸發條件

```python
class ReplanTrigger:
    # 條件 1: 機器人卡住
    # 速度 < 0.1 m/s 持續 2 秒 → 觸發重規劃

    # 條件 2: 偏離路徑太遠
    # 路徑距離 > 1.5m → 觸發重規劃

    # 條件 3: 沒有進步
    # 進步速度 < 0.2 m/s 持續 3 秒 → 觸發重規劃
```

### 重規劃流程

```
┌─────────────────────────────────────────────────────────────┐
│                   重規劃決策流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RL 執行 ──► 監測狀態                                      │
│             │                                              │
│             ▼                                              │
│      ┌────────────────┐                                     │
│      │ 檢查觸發條件    │                                     │
│      └────────────────┘                                     │
│             │                                              │
│             ├─► 卡住？─────────────┐                       │
│             ├─► 偏離？───────────┤                       │
│             └─► 無進步？─────────┤                       │
│                                │                           │
│                    YES           ▼                           │
│              ┌──────────────────────┐                        │
│              │ 觸發 AIT* 重規劃    │                        │
│              └──────────────────────┘                        │
│                                │                           │
│                                ▼                           │
│                    ┌──────────────────────┐                    │
│                    │  更新局部目標       │                    │
│                    └──────────────────────┘                    │
│                                │                           │
│                                └─► RL 繼續執行              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### RL 的兩種應對策略

#### 策略 A：靠 RL 硬解（小範圍）

```
情況：LiDAR 檢測到前方有小障礙物

RL 反應：
1. 減速
2. 輕微轉向避開
3. 繞過後回到原路徑

條件：偏離 < 0.5m，不觸發重規劃
```

#### 策略 B：叫 AIT* 重算（大範圍）

```
情況：路被完全堵死

RL 反應：
1. 嘗試避障失敗
2. 速度降為 0（卡住）
3. 等待 2 秒

系統反應：
1. 檢測到卡住
2. 觸發 AIT* 重規劃
3. AIT* 算出新路徑（可能繞遠路）
4. RL 開始跟隨新路徑
```

---

## 域隨機化

### 為什麼需要域隨機化？

**問題**：在 Isaac Sim 訓練的 RL Agent，部署到真實機器人後表現很差。

**原因**：模擬環境與真實環境之間的「差距」

| 類別 | 模擬環境 | 真實環境 | 差距 |
|------|---------|---------|------|
| 地面摩擦 | 固定 0.5 | 磁磚/地毯/泥地 | ⚠️ |
| 感測器 | 完美數據 | 有雜訊、會故障 | ⚠️ |
| 機器人質量 | 固定 10kg | 載重變化 | ⚠️ |
| 障礙物位置 | 固定 | 隨機變化 | ⚠️ |

### 域隨機化配置

```python
class DomainRandomizationConfig:
    # 物理隨機化
    friction_range = (0.3, 0.9)      # 模擬不同地面
    restitution_range = (0.1, 0.5)   # 模擬不同彈性
    mass_range = (0.8, 1.2)           # 模擬載重變化

    # 感測器隨機化
    lidar_noise_std = 0.02            # 高斯雜訊
    lidar_dropout_rate = 0.05          # 5% 射線丟失
    lidar_max_dropout = 5             # 最多丟 5 條射線

    # 動力學隨機化
    torque_range = (0.9, 1.1)         # 扭矩常數
```

### 隨機化效果

**未隨機化訓練**：
- Agent 只學會了「這張地圖」的策略
- 換地圖就死

**隨機化訓練**：
- Agent 學會「通用規則」：
  - 「看到東西就閃」
  - 「看到紅蘿蔔就追」
  - 「不管地上滑不滑，都要保持平衡」

### 訓練建議

```python
# 階段 1: 無隨機化（基線）
train(env_id="Isaac-Navigation-Charge-v0", randomization=None)

# 階段 2: 輕度隨機化
train(env_id="Isaac-Navigation-Charge-v0",
       randomization={"lidar_noise": 0.01})

# 階段 3: 中度隨機化
train(env_id="Isaac-Navigation-Charge-v0",
       randomization={"lidar_noise": 0.02, "friction": True})

# 階段 4: 高度隨機化（地獄模式）
train(env_id="Isaac-Navigation-Charge-v0",
       randomization="full")
```

---

## Frontier 探索

### 什麼是 Frontier？

```
┌────────────────────────────────────────────────────────────┐
│                         未知地圖                              │
│  ████████████████████████████████████████████████████████  │
│                                                            │
│   已探索區域                Frontier             未知區域      │
│   (Free Space)             (邊界線)            (Black)        │
│                                                            │
│   ████████████████████████████████████████████████████████  │
│                     ↑                                    ↑        │
│                   當前位置                            下個目標  │
└────────────────────────────────────────────────────────────┘
```

**Frontier 定義**：已探索區域與未知區域的交界線

### Frontier 探索算法

```
算法：Frontier-Based Exploration

輸入：
- 當前地圖 (exploration_map)
- 機人位置 (robot_pos)
- LiDAR 數據 (lidar_scan)

流程：
1. 更新探索地圖
   └─ 根據 LiDAR 數據標記已探索區域

2. 尋找 Frontiers
   └─ 檢測邊界線（未知且鄰近已探索）

3. 選擇目標 Frontier
   ├─ 計算每個 Frontier 的優先級
   └─ 選擇最高優先級的 Frontier

4. 設定為 AIT* 目標
   └─ AIT* 規劃到 Frontier 的路徑

5. RL 執行
   └─ RL 跟隨路徑到達 Frontier

6. 重複 1-5
   └─ 直到探索完成
```

### Frontier 優先級計算

```python
# 優先級 = 1 / (距離 + 探索收益 + 路徑成本)

distance = norm(frontier.center - robot.pos)
exploration_bonus = frontier.size  # 大 Frontier 優先級高
path_cost = aitstar_plan_cost(robot.pos, frontier.center)

priority = exploration_bonus / (distance + 0.1) - path_cost * 0.1
```

### 探索完成判斷

```python
def is_exploration_complete():
    # 條件 1: 沒有更多 Frontiers
    if len(frontiers) == 0:
        return True

    # 條件 2: 已探索比例 > 95%
    explored_ratio = explored_cells / total_cells
    if explored_ratio > 0.95:
        return True

    return False
```

---

## 完整系統架構

```
┌─────────────────────────────────────────────────────────────────────┐
│              層級式導航完整架構 (含動態重規劃 + Frontier 探索)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  全局規劃層 (Global Planner)                   │  │
│  │                                                              │  │
│  │  ┌─────────────┐   ┌───────────────┐   ┌─────────────────┐    │  │
│  │  │ AIT* 靜態   │   │ Frontier      │   │ 定期重規劃      │    │  │
│  │  │ 規劃 (初始化)│   │ 探索 (未知)   │   │ (觸發式)        │    │  │
│  │  └─────────────┘   └───────────────┘   └─────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               局部目標層 (Local Goal Extractor)                │  │
│  │                                                              │  │
│  │   Carrot-on-stick: lookahead = 2m (可根據速度自適應)         │  │
│  │                                                              │  │
│  │   輸出: [前向距離, 側向距離, 距離, 角度]                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  RL 層 (PPO 控制器)                            │  │
│  │                                                              │  │
│  │   觀測: [LiDAR(72) + 導航特征(6) + 速度(2) + 動作(2)] = 82      │  │
│  │                                                              │  │
│  │   輸出: [線速度, 角速度]                                        │  │
│  │                                                              │  │
│  │   獎勵: R_reach + R_progress + R_tracking - P_collision - P_smooth │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  域隨機化層 (Domain Randomization)              │  │
│  │                                                              │  │
│  │   物理：摩擦、彈性、質量                                          │  │
│  │   感測：雜訊、丟失                                                │  │
│  │   動力學：扭矩、阻尼                                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 使用示例

### 啟動訓練（帶動態重規劃）

```bash
# 基礎訓練（無動態重規劃）
python scripts/reinforcement_learning/sb3/train_hierarchical.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --num_envs 128 \
    --max_iterations 10000

# 啟用 AIT* 視覺化
python scripts/reinforcement_learning/sb3/train_hierarchical.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --enable_viz \
    --replan_interval 2.0

# 啟用域隨機化
python scripts/reinforcement_learning/sb3/train_hierarchical.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --enable_dr
```

### 程式碼集成

```python
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.hierarchical import (
    HierarchicalNavigationManager,
    ReplanTrigger,
    DomainRandomizationConfig,
)

# 創建管理器
nav_manager = HierarchicalNavigationManager(
    env=env,
    lookahead_distance=2.0,
    replan_interval=2.0,
    enable_visualization=True,
)

# 在訓練循環中更新
for step in range(num_steps):
    # 執行環境步驟
    obs, reward, done, truncated, info = env.step(action)

    # 更新導航系統（包含重規劃邏輯）
    debug_info = nav_manager.update(dt=0.02)
```

---

## 關鍵參數調優

### 前瞻距離 (Lookahead Distance)

| 環境類型 | 推薦值 | 說明 |
|---------|-------|------|
| 開闊空間 | 3.0-5.0m | 可以看遠一點，提高效率 |
| 有障礙物 | 1.5-2.0m | 需要更靈活地反應 |
| 窄通道 | 1.0-1.5m | 精確跟隨要求高 |

### 重規劃間隔

| 間隔 | 適用場景 | 計算成本 |
|-----|---------|---------|
| 0.5s | 高度動態環境 | 高 |
| 2.0s | 一般動態環境 | 中 |
| 5.0s | 低動態環境 | 低 |

### 重規劃觸發閾值

| 參數 | 推薦值 | 說明 |
|-----|-------|------|
| stuck_threshold | 0.1 m/s | 速度低於此視為卡住 |
| stuck_duration | 2.0 s | 持續這麼久才真的觸發 |
| off_path_threshold | 1.5 m | 偏離路徑超過此值觸發 |

---

## 故障排除

### Q: AIT* 重規劃太頻繁，訓練速度慢？

**A**: 調整 `replan_interval` 和觸發閾值：
- 增加 `replan_interval`
- 提高 `off_path_threshold`
- 降低 `stuck_threshold`

### Q: RL 總是卡住，無法自主探索？

**A**: 檢查：
1. LiDAR 觀測範圍是否足夠
2. 獎勵函數是否給予足夠的探索獎勵
3. 動作空間是否受限

### Q: 域隨機化後訓練不收斂？

**A**: 使用漸進式隨機化：
- 階段 1：無隨機化
- 階段 2：小幅度隨機化
- 階段 3：逐步增加到目標隨機化水平

---

**最後更新**: 2026-02-05
