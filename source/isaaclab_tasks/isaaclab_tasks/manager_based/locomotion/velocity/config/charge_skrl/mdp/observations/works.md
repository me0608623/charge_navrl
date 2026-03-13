# Observations 觀測模組

## 功能定位
從模擬器提取機器人可觀測的狀態信息，組裝成 RL agent 的輸入向量。

## v2 觀測設計 (139D)

單幀 observation，無 frame stacking。
ego + goal + static geometry + object-level obstacles + time。

### Policy & Critic (139D) — 對稱設計
| 觀測項 | 維度 | 索引 | 來源 | 物理含義 |
|--------|------|------|------|----------|
| **ego state** | | | | |
| `linear_acceleration` | 1 | [0] | IMU | 前進加速度 ā_t [-1,1] |
| `linear_velocity` | 1 | [1] | IMU | 前進速度 v̄_t [-1,1] |
| `angular_velocity` | 1 | [2] | IMU | 角速度 ω̄_t [-1,1] |
| `robot_radius` | 1 | [3] | 常數 | 碰撞半徑 r_a (0.3m) |
| **goal state** | | | | |
| `goal_position` | 2 | [4:6] | 導航 | waypoint (x,y) in robot frame |
| **static state** | | | | |
| `lidar_static` | 72 | [6:78] | LiDAR | VLP-16 72 bins 歸一化距離 |
| **obs state** | | | | |
| `obstacle_obs` | 60 | [78:138] | MOT | Top-10 obstacles × 6D (body-frame) |
| **time state** | | | | |
| `time_remaining` | 1 | [138] | 系統 | (T_max - t) / T_max |

### obs state 每物件特徵 (6D)
```
o_i = [x_i^robot, y_i^robot, vx_i^robot, vy_i^robot, r_i, m_i]
```
- (x, y): 障礙物在 robot body frame 下的相對位置 (歸一化 by max_distance)
- (vx, vy): 障礙物在 robot body frame 下的相對速度 (歸一化 by v_max)
- r_i: 障礙物半徑 (歸一化 by max_distance)
- m_i: 有效位元 (1=可見, 0=填充/遮擋)

### 設計原則
- **ego**: 差速驅動機器人只需 1D 速度（前進 v_x），不需 v_y
- **goal**: current waypoint 在 robot frame 下，由 goal_command 或 AIT* 提供
- **static**: LiDAR 72 bins 已含牆壁+靜態障礙物距離（幾何距離場）
- **obs**: Top-10 物件級觀測，提供速度/半徑/遮擋等 LiDAR 無法表達的語義
- **time**: 剩餘時間讓 policy 學習在超時前加速

### static vs obs 的區別
- **static** (LiDAR): 72 bins 距離場，反映所有可見表面幾何，不區分靜態/動態
- **obs** (Top-K): 物件級，提供每個障礙物的獨立速度向量、碰撞半徑、遮擋位元
- 兩者互補：LiDAR 覆蓋全景幾何，obs 提供物件語義（誰在動、多快、多大）

### 未來擴充（暫不啟用）
- frame stacking: 多幀歷史觀測（恢復時間推理能力）
- privileged critic: 特權資訊（完整障礙物世界座標）

## v1 觀測設計 (297D/347D) — 已保留但不使用

舊版使用 3-frame stacking + obstacle encoder + 多種本體感覺：
- Policy: 297D = LiDAR(216) + TopK(50) + State(31)
- Critic: 347D = Policy + 50D obstacles_state

## 檔案說明

### `functions.py`
基礎觀測函數（Isaac Lab 標準格式）：
- **v2 使用**: `normalized_linear_acceleration`, `normalized_linear_velocity`,
  `base_angular_velocity_z`, `robot_radius_obs`, `goal_position_in_robot_frame`,
  `time_remaining_ratio`
- **v1 保留**: `goal_distance`, `base_velocity_xy`, `dynamic_obstacles_state`, etc.

### `obs_functions.py`
VLP16 專用觀測管線：
- **v2 使用**: `lidar_vlp16_to_2d_bins` — 16通道 LiDAR → 72 bins (含 DR 噪聲)
- **v2 使用**: `topk_obstacles_6d` — Top-10 障礙物 × 6D (body-frame, LOS 遮擋)
- **v1 保留**: `topk_obstacles_body_frame` (7D 版本), `robot_heading_normalized`, etc.

### `utils.py`
觀測工具函數：
- `check_finite`: 檢測觀測中的 NaN/Inf（防止 scaler 中毒）
