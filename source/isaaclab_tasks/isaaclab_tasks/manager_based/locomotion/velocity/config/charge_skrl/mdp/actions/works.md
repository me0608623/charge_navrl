# Actions 動作模組

## 功能定位
將 RL agent 輸出的離散動作索引轉換為物理差動驅動指令。

## 物理意義
Charge 機器人使用**差動驅動 (Differential Drive)** 運動學：
- 兩輪獨立驅動，通過左右輪速差實現轉向
- 等效控制量：線加速度 `a` (m/s^2) + 角速度 `omega` (rad/s)
- 離散化為 MultiDiscrete([21, 21]) = 441 種組合

## 檔案說明

### `discrete_differential_drive.py`
離散差動驅動動作類，將離散索引映射為連續物理量。

**參數：**
- `num_accel_bins=21`: 加速度離散化級數
- `num_omega_bins=21`: 角速度離散化級數
- `accel_range=(-0.1, 0.2)`: 加速度範圍 [m/s^2]（非對稱，前進優先）
- `omega_range=(-12deg, +12deg)`: 角速度範圍
- `max_linear_velocity=1.0`: 速度飽和上限 [m/s]

**動作映射：**
```
action_index ∈ {0, 1, ..., 440}
→ (accel_idx, omega_idx) = divmod(action_index, 21)
→ accel = -0.1 + accel_idx × 0.015  [m/s^2]
→ omega = -0.209 + omega_idx × 0.0209  [rad/s]
```

**速度積分：**
```
v(t+1) = clamp(v(t) + a × dt, 0, max_linear_velocity)
theta(t+1) = theta(t) + omega × dt
```
