# 層級式導航環境設置指南

## 概述

層級式導航環境整合了 AIT* 全局路徑規劃器和 RL (PPO) 局部控制器：

```
AIT* (Global) → Local Goal → RL (Local) → Action
   1-5 Hz           10 Hz          50 Hz      機器人
```

## 環境註冊

### 新增的環境 ID

- `Isaac-Navigation-Charge-Hierarchical-v0` - 無障礙物，帶 AIT* 視覺化
- `Isaac-Navigation-Charge-Hierarchical-v1` - Phase 1 (3 個障礙物)，帶 AIT* 視覺化

## 使用方式

### 1. 測試 AIT* 視覺化

```bash
# 測試層級式環境
./isaaclab.sh -p scripts/demos/test_aitstar_visualization.py --test hierarchical

# 測試所有視覺化
./isaaclab.sh -p scripts/demos/test_aitstar_visualization.py --test all
```

### 2. 使用 SB3 訓練

```bash
# 基礎訓練（使用層級式環境）
python scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Hierarchical-v0 \
    --num_envs 128 \
    --headless
```

### 3. 在 Python 代碼中使用

```python
import gymnasium as gym

# 創建層級式導航環境
env = gym.make(
    "Isaac-Navigation-Charge-Hierarchical-v0",
    num_envs=1,
)

# 重置環境
obs, info = env.reset()

# 執行步驟
for i in range(100):
    actions = ...  # 你的策略
    obs, reward, terminated, truncated, info = env.step(actions)

# AIT* 路徑會自動視覺化（綠色線條和點）
```

## 視覺化說明

- **綠色小球** - AIT* 規劃的路徑點
- **綠色線條** - 連接路徑點的線段
- **綠色大球** - 起點標記
- **紅色大球** - 終點標記

路徑每 50 步（約 2 秒）更新一次。

## 故障排除

### Q: AIT* 路徑沒有顯示？

A: 檢查以下項目：
1. 確認不是 headless 模式
2. 確認 AIT* 模組已正確導入
3. 查看控制台是否有 "[警告] 無法初始化 AIT* 規劃器" 的消息

### Q: ImportError: No module named 'aitstar_path_planner'

A: 需要確保 AIT* 模組路徑正確：
```bash
ls source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/globle_planner/src/aitstar_path_planner/
```

### Q: KeyboardInterrupt cleanup 錯誤

A: 這是 Python gc 的已知問題，不影響實際使用。已簡化 cleanup_pbar 函數來最小化影響。

## 架構說明

### HierarchicalChargeNavigationEnv

繼承自 `ChargeNavigationEnv`，添加了：
- `_aitstar_planner` - AIT* 路徑規劃器
- `_path_visualizer` - 路徑視覺化器
- `_current_paths` - 存儲當前路徑

### 關鍵方法

- `_update_path_visualization()` - 每 50 步更新路徑
- `plan_path()` - AIT* 規劃新路徑
- `visualize()` - 視覺化路徑（綠色線條和點）

## 下一步

1. 實現局部目標提取器 (LocalGoalExtractor)
2. 添加動態重規劃觸發器 (ReplanTrigger)
3. 整合域隨機化 (Domain Randomization)
