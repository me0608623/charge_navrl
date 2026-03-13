# Events 事件模組

## 功能定位
管理環境的狀態轉移：episode 重置、障礙物配置、動態物體移動。
Isaac Lab 的事件系統有三種模式：`startup`（初始化）、`reset`（重置時）、`interval`（定時）。

## 物理意義
模擬真實世界的不確定性：
- 機器人每次啟動位置不同 → 安全隨機重置
- 環境中障礙物數量/位置/動靜態比例不同 → 混合難度配置
- 動態障礙物（行人/其他車輛）在移動 → 定時更新位置

## 事件配置 (VLP16)

| 事件 | 模式 | 函數 | 說明 |
|------|------|------|------|
| `reset_base` | reset | `reset_root_state_random_safe` | 安全位置重置 |
| `domain_randomization` | reset | `apply_domain_randomization` | 域隨機化 |
| `randomize_obstacles_startup` | startup | `randomize_obstacles_by_difficulty` | 初始障礙物 |
| `randomize_obstacles` | reset | `randomize_obstacles_by_difficulty` | 每 episode 重配 |
| `move_dynamic_obstacles` | interval(0.2s) | `move_obstacles_vectorized` | 動態移動 |

## 檔案說明

### `state.py`（原 core/state.py）
障礙物全局狀態管理（模組間共享數據）：
- `set_obstacle_metadata(num, sizes)`: 場景初始化時設定障礙物數量和尺寸
- `get_obstacle_num/sizes/metadata`: 查詢障礙物資訊
- `reset_obstacle_targets`: 初始化動態障礙物的移動方向
- `get/update_obstacle_start_positions`: 記錄移動起始點
- `get/update_obstacle_directions`: 記錄移動方向（弧度角）
- `move_obstacles_toward_target`: 來回移動邏輯（Phase 2 用）

### `obstacles.py`
障礙物重置與移動的核心實現：
- `reset_obstacles`: 隨機放置障礙物（避開機器人和目標，保持間距）
  - 參數：`speed_range`, `min_robot_distance=1.5m`, `min_goal_distance=1.0m`
  - 使用牆壁 proximity 檢查避免放在牆內
- `move_obstacles_vectorized`: GPU 向量化的動態障礙物移動
  - 邊界反彈、速度重採樣、到達目標後隨機新目標
  - 參數：`move_dt=0.2`, `speed_min/max`, `area_limit=6.0`

### `mixed_parallel.py`
混合難度環境策略（取代傳統分階段課程學習）：
- `randomize_obstacles_by_difficulty`: 在同一批次中訓練不同難度
  - 50% empty（無障礙物）+ 30% static + 20% dynamic
  - 參數：`empty_ratio`, `static_ratio`, `dynamic_ratio`
  - `active_obstacle_ratio=0.25`: 只啟用 25% 障礙物為動態

### `reset.py`
安全重置機制：
- `reset_root_state_random_safe`: 隨機位置重置 + 牆壁/障礙物碰撞檢查
  - 嘗試最多 50 次採樣，確保初始位置安全
  - 使用 `check_wall_proximity_batch` GPU 並行檢查
  - 參數：`pose_range`, `velocity_range`
- `hide_unused_obstacles`: 隱藏多餘障礙物到 Z=-10
- `reset_root_state_fixed_per_env`: 固定每個 env 的初始位置（debug 用）
