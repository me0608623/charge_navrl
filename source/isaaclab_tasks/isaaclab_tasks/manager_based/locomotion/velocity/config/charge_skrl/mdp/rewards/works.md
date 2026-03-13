# Rewards 獎勵模組

## 功能定位
計算每步的獎勵信號，引導 agent 學會「快速到達目標 + 安全避障 + 平穩行駛」。

## 設計原則：Potential-Based Reward Shaping (PBRS)
核心使用 Ng et al. (1999) 的勢函數法，保證策略不變性：
```
R(t) = Φ(s_{t+1}) - Φ(s_t) = d(t) - d(t+1)
```
- 靠近目標 → 正獎勵
- 遠離目標 → 負獎勵
- 整個 episode 的 PBRS 總和 = d_初始 - d_最終（telescoping sum）

## 獎勵結構 (VLP16)

| 獎勵項 | 權重 | 類型 | 物理意義 |
|--------|------|------|----------|
| `reaching_goal` | +250 | Terminal | 到達目標獎金（一次性） |
| `collision_terminal` | -25 | Terminal | 碰撞懲罰（一次性） |
| `potential_progress` | +100 | Dense | PBRS 距離縮減（每步） |
| `time_penalty` | -0.5 | Dense | 時間懲罰（鼓勵效率） |
| `acceleration_penalty` | -0.25 | Dense | 加速度平方懲罰（平滑度） |
| `angular_velocity_penalty` | -0.25 | Dense | 角速度平方懲罰（減少抖動） |
| `velocity_too_low` | -0.5 | Dense | 靜止懲罰（防止凍結行為） |

## 預期回報分析
```
G(成功到達) ≈ +250×dt + 100×Δd - 0.5×T×dt ≈ +50~+60
G(碰撞死亡) ≈ -25×dt + 小progress ≈ -6~-10
G(原地不動) ≈ -0.5×T×dt - 0.5×T×dt ≈ -16（最差）
```
→ agent 學到的排序：成功 >> 碰撞 >> 不動

## 檔案說明

### `potential_based_rewards.py`（核心）
PBRS 獎勵函數集：
- `potential_progress_reward`: Φ(s')−Φ(s) 距離進展，帶 episode reset 偵測
- `smooth_collision_penalty`: 警告區 [0.3m,0.8m] 的二次懲罰
- `collision_terminal_penalty`: LiDAR < threshold 的二元碰撞判定
- `per_step_time_penalty`: 常數 1.0，由 weight 控制大小
- `discrete_acceleration_squared_penalty`: 直接讀 applied_accelerations 的平方
- `angular_velocity_squared_penalty`: |ω_z|² 懲罰不必要轉彎
- `velocity_too_low_penalty`: 速度 < 0.05 m/s 時返回 1.0（Fix 3）

### `goal_rewards.py`
目標達成獎勵：
- `reaching_goal`: 距離 < threshold 時一次性獎勵

### `safety_rewards.py`
安全相關判定：
- `collision_occurred`: LiDAR 最近距離 < threshold（用於終止條件）
- `collision_contact_occurred`: PhysX 接觸感測器判定

### `utils.py`
獎勵診斷工具：
- `_print_diagnostics`: 印出獎勵統計（debug 用）
- `_check_reward_term`: 檢查獎勵值是否異常
