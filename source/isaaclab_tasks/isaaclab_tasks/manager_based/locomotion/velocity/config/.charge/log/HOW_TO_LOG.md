# 如何記錄訓練結果

## 問題
訓練完成後，結果沒有自動記錄到 `charge_navigation_training_log.csv` 文件中。

## 原因
訓練腳本 (`train.py`) 沒有自動集成記錄器，需要手動記錄訓練結果。

## 解決方案

### 方法 1：使用快速記錄腳本（推薦）

1. **編輯 `quick_log.py` 文件**，填入您的訓練結果：

```python
# 必需參數
final_mean_reward = 700.12  # 從訓練輸出中獲取
final_mean_episode_length = 444.31  # 從訓練輸出中獲取
success_rate = 0.25  # 目標達成率
timeout_rate = 0.75  # 超時率

# 可選參數
notes = "第一次訓練 - 調整後的超參數"
```

2. **運行腳本**：

```bash
cd /home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/log
python quick_log.py
```

### 方法 2：使用 Python 直接記錄

在 Python 中運行：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from log.training_logger import quick_log

quick_log(
    final_mean_reward=700.12,
    final_mean_episode_length=444.31,
    success_rate=0.25,
    timeout_rate=0.75,
    notes="訓練完成",
)
```

### 方法 3：從訓練日誌中提取數據

如果您有 TensorBoard 日誌或其他訓練日誌，可以：

1. **查看訓練輸出**：訓練結束時會顯示最後一次迭代的統計信息
2. **查看 TensorBoard 日誌**：在 `logs/rsl_rl/{experiment_name}/{run_name}/summaries/` 目錄下
3. **提取關鍵指標**：
   - `Mean reward`: 最終平均獎勵
   - `Mean episode length`: 最終平均回合長度
   - 終止條件統計：計算成功率、超時率等

## 如何獲取訓練結果數據

### 從訓練輸出中獲取

訓練結束時，RSL-RL 會輸出類似以下的信息：

```
Learning iteration 150/150

Mean reward: 700.12
Mean episode length: 444.31
Value function loss: 6.3465
Surrogate loss: 0.0162
Entropy loss: 3.1015
Mean action noise std: 1.17
```

### 從終止條件統計中計算

您需要查看訓練日誌中的終止條件統計：
- **成功率** = 到達目標的環境數 / 總環境數
- **超時率** = 超時的環境數 / 總環境數
- **碰撞率** = 碰撞的環境數 / 總環境數

### 從 TensorBoard 日誌中提取

如果使用 TensorBoard 記錄，可以：

1. 打開 TensorBoard：
```bash
tensorboard --logdir logs/rsl_rl/{experiment_name}/{run_name}/summaries
```

2. 查看最後一次迭代的統計數據

## 記錄的數據字段

### 必需字段
- `final_mean_reward`: 最終平均獎勵
- `final_mean_episode_length`: 最終平均回合長度
- `success_rate`: 目標達成率（0.0-1.0）
- `timeout_rate`: 超時率（0.0-1.0）

### 可選字段
- `collision_rate`: 碰撞率
- `final_value_loss`: 最終價值函數損失
- `final_surrogate_loss`: 最終代理損失
- `final_entropy_loss`: 最終熵損失
- `final_action_noise_std`: 最終動作噪聲標準差
- `total_timesteps`: 總時間步數
- `total_training_time_seconds`: 總訓練時間（秒）
- `steps_per_second`: 每秒步數
- `notes`: 備註信息

## 注意事項

1. **超參數會自動提取**：記錄器會自動從配置文件中提取所有超參數
2. **CSV 文件會自動創建**：如果文件不存在，會自動創建並寫入表頭
3. **數據會追加**：每次記錄都會追加到 CSV 文件末尾，不會覆蓋之前的數據

## 未來改進

如果您希望訓練腳本自動記錄結果，可以：

1. 修改 `train.py`，在訓練結束後自動調用記錄器
2. 從 RSL-RL runner 中提取統計數據
3. 計算終止條件統計（成功率、超時率等）

需要我幫您修改訓練腳本以自動記錄嗎？
