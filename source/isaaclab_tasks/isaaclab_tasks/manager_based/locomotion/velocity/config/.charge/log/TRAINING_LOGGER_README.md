# 訓練記錄器使用說明

## 概述

`training_logger.py` 是一個用於記錄每次訓練的超參數配置和訓練結果的工具，會自動將數據導出到 CSV 文件中。這方便您追蹤不同超參數配置下的訓練效果，進行超參數調優。

## 功能特點

- ✅ 自動提取超參數：從配置文件中自動讀取所有關鍵超參數
- ✅ 記錄訓練結果：記錄獎勵、成功率、損失等關鍵指標
- ✅ CSV 導出：數據自動追加到 CSV 文件，方便後續分析
- ✅ 易於使用：簡單的 API，幾行代碼即可完成記錄

## 文件位置

- **記錄器文件**：`log/training_logger.py`
- **CSV 輸出位置**：`log/charge_navigation_training_log.csv`
- **示例文件**：`log/training_logger_example.py`
- **說明文檔**：`log/TRAINING_LOGGER_README.md`

## 快速開始

### 1. 基本使用

```python
from log.training_logger import TrainingLogger
# 或者使用相對導入
from .log.training_logger import TrainingLogger

# 創建記錄器
logger = TrainingLogger()

# 訓練結束後記錄結果
logger.log_training_results(
    final_mean_reward=700.12,
    final_mean_episode_length=444.31,
    success_rate=0.25,  # 25% 成功率
    timeout_rate=0.75,  # 75% 超時率
    final_value_loss=6.3465,
    final_surrogate_loss=0.0162,
    final_entropy_loss=3.1015,
    final_action_noise_std=1.17,
    total_timesteps=3686400,
    total_training_time_seconds=1970,
    steps_per_second=2186,
    episode_length_s=45.0,  # 手動指定（因為在 __post_init__ 中設置）
    num_envs=32,  # 手動指定環境數量
    notes="第一次訓練 - 調整後的超參數",
)
```

### 2. 使用便捷函數

```python
from log.training_logger import quick_log
# 或者使用相對導入
from .log.training_logger import quick_log

quick_log(
    final_mean_reward=700.12,
    final_mean_episode_length=444.31,
    success_rate=0.25,
    timeout_rate=0.75,
    notes="快速記錄",
)
```

### 3. 從訓練輸出字典記錄

```python
logger = TrainingLogger()

training_output = {
    "mean_reward": 700.12,
    "mean_episode_length": 444.31,
    "success_rate": 0.25,
    "timeout_rate": 0.75,
    "value_loss": 6.3465,
    # ... 其他指標
}

logger.log_from_training_output(
    training_output=training_output,
    log_dir="/path/to/logs",
    notes="從訓練輸出記錄",
)
```

## 記錄的數據

### 超參數（自動提取）

- **環境超參數**：episode_length_s, num_envs
- **獎勵超參數**：各獎勵項的權重和閾值
- **PPO 算法超參數**：learning_rate, gamma, entropy_coef 等
- **網絡結構超參數**：actor/critic 隱藏層維度、激活函數等

### 訓練結果（手動提供）

- **性能指標**：平均獎勵、平均回合長度、成功率、超時率等
- **損失指標**：價值函數損失、代理損失、熵損失等
- **獎勵項統計**：各獎勵項的平均值
- **其他信息**：訓練時間、總時間步數、備註等

## CSV 文件結構

CSV 文件包含以下列：

1. **時間戳和實驗名稱**
2. **環境超參數**（episode_length_s, num_envs）
3. **獎勵超參數**（各獎勵項的權重和閾值）
4. **終止條件超參數**
5. **PPO 算法超參數**（learning_rate, gamma, entropy_coef 等）
6. **網絡結構超參數**
7. **訓練結果指標**（reward, success_rate, timeout_rate 等）
8. **訓練損失指標**
9. **性能指標**（timesteps, training_time, fps）
10. **獎勵項統計**
11. **其他信息**（notes, log_dir）

## 在訓練腳本中集成

### 方法 1：訓練結束後記錄

```python
# 在訓練腳本中
from training_logger import TrainingLogger

logger = TrainingLogger()

# ... 訓練代碼 ...
# runner.learn(num_learning_iterations=agent_cfg.max_iterations)

# 訓練結束後記錄
logger.log_training_results(
    final_mean_reward=final_reward,  # 從訓練統計中獲取
    final_mean_episode_length=mean_ep_len,
    success_rate=success_count / total_episodes,
    timeout_rate=timeout_count / total_episodes,
    # ... 其他參數
    log_dir=log_dir,  # 訓練日誌目錄
    notes="訓練完成",
)
```

### 方法 2：從終端輸出解析

訓練結束後，從終端輸出中提取關鍵指標，然後記錄：

```python
# 從終端輸出中提取的數據
logger.log_training_results(
    final_mean_reward=700.12,  # 從 "平均獎勵: 700.12" 提取
    final_mean_episode_length=444.31,  # 從 "平均回合長度: 444.31" 提取
    success_rate=0.25,  # 從 "回合終止/目標達成: 0.2500" 提取
    timeout_rate=0.75,  # 從 "回合終止/超時: 0.7500" 提取
    # ...
)
```

## 注意事項

1. **episode_length_s 和 num_envs**：這兩個參數在 `__post_init__` 中設置，無法自動提取，需要手動指定。

2. **CSV 文件位置**：默認保存在 `log/charge_navigation_training_log.csv`，可以通過 `TrainingLogger(csv_path="custom/path.csv")` 自定義。

3. **數據追加**：每次調用 `log_training_results()` 都會追加新的一行，不會覆蓋現有數據。

4. **可選參數**：大部分參數都是可選的，可以只記錄關鍵指標。

## 查看和分析數據

記錄完成後，可以使用以下工具查看和分析 CSV 數據：

```python
import pandas as pd

# 讀取 CSV
df = pd.read_csv("log/charge_navigation_training_log.csv")

# 查看所有記錄
print(df)

# 按成功率排序
df_sorted = df.sort_values("success_rate", ascending=False)
print(df_sorted[["timestamp", "success_rate", "final_mean_reward", "notes"]])

# 分析超參數對性能的影響
print(df.groupby("ppo_entropy_coef")["success_rate"].mean())
```

## 示例

完整的使用示例請參考 `training_logger_example.py` 文件。

## 問題排查

如果遇到導入錯誤，請確保：
1. 文件在正確的目錄下（`log/` 子目錄）
2. 使用相對導入：`from log.training_logger import TrainingLogger` 或 `from .log.training_logger import TrainingLogger`
3. 或者使用絕對導入：`from isaaclab_tasks.manager_based.locomotion.velocity.config.charge.log.training_logger import TrainingLogger`
