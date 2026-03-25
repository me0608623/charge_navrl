# Charge 機器人導航訓練架構文檔

## 文檔信息

- **項目名稱**: Charge 機器人 Hierarchical 導航系統
- **版本**: v1.0
- **最後更新**: 2026-02-06
- **目錄**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/`

---

## 1. 項目概述

### 1.1 研究目標

本項目旨在實現一個基於強化學的層級式導航系統，使 Charge 差速驅動機器人能夠在複雜的室內環境中自主導航。

### 1.2 核心技術

- **仿真環境**: NVIDIA Isaac Lab
- **強化學算法**: Stable Baselines3 (PPO)
- **全局路徑規劃**: AIT* (Anytime Trajectory Search)
- **局部控制**: Deep Reinforcement Learning
- **Sim-to-Real**: Domain Randomization
- **實驗管理**: Weights & Biases (WandB)

### 1.3 機器人規格

| 屬性 | 值 |
|------|-----|
| 機器人型號 | Charge 差速驅動機器人 |
| 驅動方式 | 雙輪差速 (Differential Drive) |
| 感知傳感器 | 2D LiDAR (360°, 72 條射線, 5° 解析度) |
| IMU | 陀螺儀 + 加速度計 |
| 最大速度 | 前進 1.0 m/s |
| 最大角速度 | 旋轉 1.0 rad/s |
| 機器人半徑 | 0.3 m |

---

## 2. 系統架構

### 2.1 整體架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                     Charge 導航系統架構                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   全局規劃    │    │   局部控制    │    │   機器人    │    │
│  │              │    │              │    │              │    │
│  │     AIT*      │───▶│      RL      │───▶│   Charge     │    │
│  │  Global Path  │    │  Policy π    │    │   Robot      │    │
│  │    Planner    │    │              │    │              │    │
│  │              │    │              │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                   Isaac Lab 仿真環境                         │  │
│  │  - 物理引擎 (PhysX 5)                                        │  │
│  │  - 渲染引擎 (RTX Renderer)                                   │  │
│  │  - 並行環境 (256+ instances)                                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 層級式導航架構

#### 2.2.1 全局層 (Global Layer)

**功能**: 生成從起始點到目標點的全局路徑

**實現**:
- AIT* 算法 (Anytime Trajectory Search)
- 輸入: 地圖 (16x16m)、起始點、目標點、障礙物位置
- 輸出: 路徑點序列 `[(x0, y0), (x1, y1), ...]`
- 更新頻率: 1-5 Hz (當機器人偏離路徑或卡住時)

#### 2.2.2 局部層 (Local Layer)

**功能**: 追隨全局路徑的局部目標，執行底層控制

**實現**:
- Deep RL Policy (PPO)
- 輸入:
  - 局部目標 (相對機器人座標)
  - LiDAR 點雲 (72 維)
  - IMU 數據
  - 當前速度
- 輸出: 線速度 `v`、角速度 `ω`
- 更新頻率: 50 Hz (20 ms)

#### 2.2.3 Carrot-on-Stick 策略

```
全局路徑: Start ─── P1 ─── P2 ─── P3 ─── Goal
                           │
                    Carrot-on-stick (2m 前瞻)
                           │
                  局部目標 (Local Goal)
                           │
                     RL Policy 追隨
```

---

## 3. 課練課程設計 (Curriculum Design)

### 3.1 課程階段

本系統採用漸進式課程學習，分為 4 個階段：

| 階段 | 名稱 | 目標 | 難度 |
|------|------|------|------|
| **Phase 0** | 車輛動力學校準 | 學會基本運動控制 | ⭐ |
| **Phase 1** | 內牆結構導航 | 在簡單室內環境導航 | ⭐⭐ |
| **Phase 2** | 走廊與窄通道 | 穿越狹窄空間 | ⭐⭐⭐ |
| **Phase 3** | 複雜地形 | 處理動態障礙物 | ⭐⭐⭐⭐ |

### 3.2 Phase 0 詳細設計

#### 3.2.1 環境配置

```python
環境: 16x16m 房間，四面牆壁
機器人重生: 中心 10x10m 區域
目標距離: 3-8m
障礙物: 無
Episode 長度: 20 秒 (約 500 步 @ 25 Hz)
```

#### 3.2.2 觀測空間 (131 維，統一所有 Phase)

| 維度 | 大小 | 說明 |
|------|------|------|
| `lidar_scan` | 72 | 360° LiDAR (5° 解析度) |
| `speed` | 2 | 線速度 (vx, vy) |
| `goal_position` | 2 | 目標相對位置 (x, y) |
| `goal_distance` | 1 | 到目標歐式距離 |
| `time_remaining` | 1 | 剩餘時間比例 |
| `alive` | 1 | 存活標誌 |
| `actions` | 2 | 上一步動作 (v, ω) |
| `obstacles` | 50 | 障礙物狀態 (10×5，Phase 0 全為 0) |

**總計**: 72 + 2 + 2 + 1 + 1 + 1 + 2 + 50 = **131 維**

#### 3.2.3 動作空間

```python
動作: [v, ω]  # 線速度、角速度
範圍: v ∈ [-1.0, 1.0] m/s
      ω ∈ [-1.0, 1.0] rad/s
```

---

## 4. 獎勵設計 (Pull-Push Theory)

### 4.1 設計理念

基於**拉-推理論** (Pull-Push Theory) 的獎勵函數設計：

- **Pull Forces**: 吸引機器人向目標移動
- **Push Forces**: 推開機器人遠離障礙物/危險區域
- **Smoothness**: 獎勵平滑、自然的運動

### 4.2 獎勵項配置

| 獎勵項 | 權重 | 說明 |
|--------|------|------|
| **Pull Forces** | | |
| `progress_to_goal` | +20.0 | 進度獎勵 (d_{t-1} - d_t) |
| `reaching_goal` | +15.0 | 抵達目標獎勵 (d < 0.5m) |
| `velocity_toward_goal` | +1.0 | 朝向目標速度 |
| **Push Forces** | | |
| `safety_field` | -2.0 | 安場懲罰 (0.25m < d < 0.6m) |
| `collision_penalty` | -20.0 | 碰撞懲罰 (d < 0.25m) |
| **Smoothness** | | |
| `action_smoothness_linear` | -0.1 | 平滑懲罰 (||Δv||, 線性項) |
| `time_out_penalty` | -0.05 | 時間懲罰 |

### 4.3 終止條件

| 條件 | 閾值 | 說明 |
|------|------|------|
| `goal_reached` | d < 0.3m | 抵達目標 |
| `robot_tipped_over` | 傾角 > 60° | 機器人翻倒 |
| `wall_collision` | 接觸牆壁 | 牆壁碰撞 |

### 4.4 安場懲罰公式

```python
# 安全距離懲罰 (Safety Field Penalty)
P_safety(d) = {
    -1.0,                    if d < 0.25m (碰撞閾值)
    -(0.6 - d) / (0.6 - 0.25), if 0.25m ≤ d < 0.6m (線性衰減)
    0,                       if d ≥ 0.6m (安全)
}
```

---

## 5. 域隨機化 (Domain Randomization)

### 5.1 目標

提高 Sim-to-Real 遷移效果，使策略在真實機器人上也能良好運作。

### 5.2 隨機化項目

| 隨機化項 | 範圍 | 頻率 |
|----------|------|------|
| **初始速度隨機** | ±0.5 m/s (線), ±0.5 rad/s (角) | 每次 reset |
| **LiDAR 噪聲** | ±0.1 m | 每個 step |
| **質量隨機** | ±10% | 每次 reset |
| **摩擦力隨機** | ±20% | 每次 reset |
| **外部擾動** | 5-20N, 5% 概率 | 每 1 step |

### 5.3 實現

```python
# 文件位置
charge_sb3/domain_randomization/dr_manager.py      # 管理器
charge_sb3/domain_randomization/dr_events.py      # 事件處理器
```

---

## 6. Stable Baselines3 配置

### 6.1 PPO 超參數

```yaml
# 文件: agents/sb3_ppo_cfg_phase0.yaml

policy: "MlpPolicy"
learning_rate: 0.0001
n_steps: 24
batch_size: 4096          # n_steps × num_envs
n_epochs: 5
n_minibatches: 4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
ent_coef: 0.001
vf_coef: 0.5
max_grad_norm: 0.5

# 經驗回放
normalize_input: false
normalize_value: false
```

### 6.2 神經網路架構

```python
Actor Network (Policy):
  - 輸入: 131 維觀測
  - 隱藏層: [256, 256, 128]
  - 激活函數: ELU
  - 輸出: 均值 + 標準差

Critic Network (Value):
  - 輸入: 131 維觀測
  - 隱藏層: [256, 256, 128]
  - 激活函數: ELU
  - 輸出: 狀標值 (標量)
```

---

## 7. 訓練流程

### 7.1 訓練命令

```bash
# Phase 0 訓練 (5M 步)
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256 \
    --headless \
    --agent sb3_cfg_entry_point \
    --max_iterations 20000000

# 評估 (可選)
python scripts/enjoy.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 1 \
    --checkpoint logs/sb3/Isaac-Navigation-Charge-Phase0/.../model.zip
```

### 7.2 訓練配置文件位置

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
├── charge_sb3/
│   ├── cfg/
│   │   ├── charge_env_cfg_phase0.py    # Phase 0 環境配置
│   │   ├── charge_env_cfg_phase1.py    # Phase 1 環境配置
│   │   ├── charge_env_cfg_phase2.py    # Phase 2 環境配置
│   │   └── charge_env_cfg_phase3.py    # Phase 3 環境配置
│   ├── agents/
│   │   ├── sb3_ppo_cfg_phase0.yaml     # Phase 0 SB3 配置
│   │   ├── sb3_ppo_cfg_phase1.yaml     # Phase 1 SB3 配置
│   │   ├── sb3_ppo_cfg_phase2.yaml     # Phase 2 SB3 配置
│   │   └── sb3_ppo_cfg_phase3.yaml     # Phase 3 SB3 配置
│   ├── mdp/                             # MDP 函數 (觀測、獎勵、終止)
│   │   ├── observations/
│   │   ├── rewards/
│   │   ├── terminations/
│   │   └── events/
│   ├── domain_randomization/           # 域隨機化
│   │   ├── dr_manager.py
│   │   └── dr_events.py
│   └── __init__.py                      # Gym 註冊
└── scripts/
    └── reinforcement_learning/sb3/
        └── train_charge.py               # 訓練腳本
```

---

## 8. WandB 集成

### 8.1 配置

```python
# 項目名稱: charge_sb3
# 自動記錄: 僅在 --headless 模式下記錄
# TensorBoard 同步: sync_tensorboard=True
```

### 8.2 記錄指標

| 類別 | 指標 | 說明 |
|------|------|------|
| **訓練指標** | `rollouts/ep_rew_mean` | 平均 reward |
| | `rollouts/ep_len_mean` | 平均 episode 長度 |
| | `train/policy_loss` | 策略損失 |
| | `train/value_loss` | 價值損失 |
| | `train/learning_rate` | 學習率 |
| **環境統計** | `Episode_Reward/progress_to_goal` | 進度獎勵 |
| | `Episode_Reward/reaching_goal` | 到達目標 |
| | `Episode_Reward/safety_field` | 安場懲罰 |
| | `Episode_Termination/goal_reached` | 成功次數 |
| | `Episode_Termination/wall_collision` | 碰撞次數 |
| **計算指標** | `success_rate` | 成功率 (%) |

### 8.3 實現方式

```python
# 文件: scripts/reinforcement_learning/sb3/train_charge.py

# 1. 設置環境變量
os.environ["WANDB_PROJECT"] = "charge_sb3"

# 2. 初始化 WandB
wandb.init(
    project="charge_sb3",
    name=f"Phase0_{timestamp}",
    sync_tensorboard=True,  # 同步 TensorBoard
)

# 3. 自定義回調記錄環境統計
class WandBCallback(BaseCallback):
    def _on_step(self):
        # 每 10,000 步記錄一次
        metrics = self._extract_env_metrics()
        wandb.log(metrics, step=self.num_timesteps)
```

---

## 9. 畢業標準

### 9.1 Phase 0 畢業條件

- [ ] 成功率 > 95%
- [ ] 平均 episode reward > 100
- [ ] 無震盪 (動作平滑)
- [ ] 路徑效率 > 0.9

### 9.2 評估方法

```python
# 成功率 = goal_reached / (goal_reached + timeout + collision)
# 路徑效率 = 直線距離 / 實際路徑長度
```

---

## 10. 故障排除

### 10.1 常見問題

#### 問題 1: Gym 註冊錯誤

```
ValueError: Could not find configuration for the environment: 'Isaac-Navigation-Charge-Phase0'
```

**解決**: 確保 `isaaclab_tasks` 已正確導入

#### 問題 2: WandB 無法連接

```
WARNING: Failed to initialize WandB: ...
```

**解決**:
- 檢查 API Key: `wandb login`
- 確認網絡連接
- GUI 模式下 WandB 會自動禁用

#### 問題 3: 觀測維度不匹配

```
RuntimeError: Observation space mismatch
```

**解決**: 確保所有 Phase 使用統一的 131 維觀測空間

---

## 11. 參考文獻

### 11.1 核心論文

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
2. **Domain Randomization**: Tobin et al., "Domain Randomization for Robotics", 2017
3. **Hierarchical RL**: Sutton et al., "Between MDPs", 1999

### 11.2 相關項目

- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- AIT*: https://www.jairs.org/index.php/jairs/article/view/4325

---

## 12. 更新日誌

| 日期 | 版本 | 更新內容 |
|------|------|----------|
| 2026-02-06 | v1.0 | 初始版本，包含 Phase 0-3 完整設計 |

---

**文檔維護**: Claude Opus 4.5
**最後審核**: 2026-02-06
