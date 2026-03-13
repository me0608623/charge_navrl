# Terminations 終止條件模組

## 功能定位
判斷 episode 是否應該結束。終止時環境自動重置。

## 物理意義
模擬真實部署中的「任務完成」或「不可恢復故障」：
- **到達目標**: 任務成功完成
- **碰撞**: 機器人撞到障礙物/牆壁，可能造成硬體損壞
- **翻倒**: 機器人傾斜過大，不可自行恢復
- **物理爆炸**: 模擬器數值不穩定，需要重置避免數據污染
- **超時**: 在規定時間內未完成任務

## 終止條件 (VLP16)

| 條件 | 函數 | 閾值 | time_out |
|------|------|------|----------|
| 超時 | `time_out` | 45 秒 (225 steps) | True |
| 到達目標 | `goal_reached` | 距離 < 0.5m | False |
| 碰撞 | `collision_occurred` | LiDAR min < 1.0m | False |
| 翻倒 | `robot_tipped_over` | 傾斜 > 閾值 | False |
| 物理爆炸 | `physics_explosion` | 速度 > 10 m/s | False |

**`time_out=True` 的意義**: 超時不是失敗，PPO 會做 value bootstrap（不截斷 value 估計）。

## 檔案說明

### `goal.py`
- `goal_reached`: 比較機器人與目標距離，考慮 body_radius 修正

### `robot_state.py`
- `robot_tipped_over`: 檢查 Z 軸方向向量偏離垂直的程度
- `robot_flying`: 檢查機器人是否離地過高
- `physics_explosion`: **Fix 2** — 偵測 |v| > 10 m/s 或 |ω| > 20 rad/s
  - 防止物理引擎穿牆/碰撞後產生超大速度
  - 觸發後立即終止，避免 NaN 污染 RunningStandardScaler
