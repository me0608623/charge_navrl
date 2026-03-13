# MDP 模組總覽

## 功能定位
MDP (Markov Decision Process) 模組實現強化學習環境的完整決策循環：
**觀測 → 動作 → 獎勵 → 終止 → 事件（重置）**

## 物理意義
將機器人導航問題建模為 MDP：
- **狀態 (State)**: 機器人位姿、速度、LiDAR 感知、障礙物位置
- **動作 (Action)**: 離散化的加速度 + 角速度指令
- **轉移 (Transition)**: 物理引擎模擬一步 (dt=0.2s)
- **獎勵 (Reward)**: 引導 agent 學會避障導航的信號

## 子模組

| 資料夾 | 功能 | 對應 MDP 元素 |
|--------|------|---------------|
| `actions/` | 將離散動作索引轉換為差動驅動指令 | A (動作空間) |
| `observations/` | 從模擬器提取觀測向量 | O (觀測空間) |
| `rewards/` | 計算每步獎勵信號 | R (獎勵函數) |
| `terminations/` | 判斷 episode 是否結束 | D (終止條件) |
| `events/` | 環境重置、障礙物管理 | T (狀態轉移) |
| `wall_layout.py` | 迷宮牆壁幾何定義 + GPU proximity 工具 | 環境幾何 |

## 訓練流程

```
每個 Episode:
  1. events/reset.py     → 隨機安全位置重置 agent
  2. events/obstacles.py → 隨機配置障礙物 (static/dynamic)
  Loop (max 225 steps = 45s):
    3. observations/     → 組合 79D policy obs / 79D critic obs (v2)
    4. SKRL PPO          → 選擇離散動作 (0~440)
    5. actions/           → 轉換為 (accel, omega) 物理指令
    6. Isaac Sim          → 模擬 20 substeps (decimation=20)
    7. rewards/           → 計算 PBRS progress + collision + time penalty
    8. terminations/      → 檢查碰撞/到達/翻倒/爆炸
```

## 觀測維度 (VLP16 v2)
- **Policy**: 79D = ego(4) + goal(2) + LiDAR(72) + time(1)
- **Critic**: 79D（與 Policy 相同，暫不加特權資訊）

## 動作空間
- **MultiDiscrete([21, 21])** = 441 組合
- 線加速度: [-0.1, +0.2] m/s^2
- 角速度: [-12, +12] deg/s
