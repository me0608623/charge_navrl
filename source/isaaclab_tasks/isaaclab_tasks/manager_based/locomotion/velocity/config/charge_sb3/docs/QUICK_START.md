# Charge 導航訓練 - 快速參考指南

## 📋 目錄

1. [環境設置](#環境設置)
2. [訓練命令](#訓練命令)
3. [配置文件說明](#配置文件說明)
4. [常見問題](#常見問題)

---

## 🚀 環境設置

### 安裝依賴

```bash
# 進入 IsaacLab 目錄
cd /home/aa/IsaacLab

# 激活 Python 環境
conda activate env_isaaclab

# 安裝 Isaac Lab
python -m pip install -e .
```

---

## 🎯 訓練命令

### Phase 0 訓練 (基礎運動控制)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256 \
    --headless \
    --agent sb3_cfg_entry_point \
    --max_iterations 20000000
```

### 參數說明

| 參數 | 說明 | 默認值 |
|------|------|--------|
| `--task` | 環境 ID | - |
| `--num_envs` | 並行環境數量 | 256 |
| `--headless` | 無頭模式 (GUI 模式禁用 WandB) | False |
| `--agent` | Agent 配置入口點 | sb3_cfg_entry_point |
| `--max_iterations` | 最大訓練迭代次數 | - |

### 訓練階段選擇

```bash
# Phase 0: 車輛動力學校準
--task Isaac-Navigation-Charge-Phase0

# Phase 1: 內牆結構導航
--task Isaac-Navigation-Charge-Phase1

# Phase 2: 走廊與窄通道
--task Isaac-Navigation-Charge-Phase2

# Phase 3: 複雜地形
--task Isaac-Navigation-Charge-Phase3
```

---

## 📁 配置文件說明

### 核心配置位置

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/

cfg/
├── charge_env_cfg_phase0.py    # Phase 0 環境配置
├── charge_env_cfg_phase1.py    # Phase 1 環境配置
├── charge_env_cfg_phase2.py    # Phase 2 環境配置
├── charge_env_cfg_phase3.py    # Phase 3 環境配置

agents/
├── sb3_ppo_cfg_phase0.yaml     # Phase 0 SB3 PPO 超參數
├── sb3_ppo_cfg_phase1.yaml     # Phase 1 SB3 PPO 超參數
├── sb3_ppo_cfg_phase2.yaml     # Phase 2 SB3 PPO 超參數
└── sb3_ppo_cfg_phase3.yaml     # Phase 3 SB3 PPO 超參數
```

### 環境配置項 (`charge_env_cfg_phase0.py`)

```python
@configclass
class ChargeNavigationEnvCfgPhase0:
    # 環境場景
    scene: MySceneCfgPhase0 = MySceneCfgPhase0(
        num_envs=256,
        env_spacing=18.0,  # 大於房間尺寸 (16m)
    )

    # 觀測空間 (131 維)
    observations: ObservationsCfgPhase0 = ObservationsCfgPhase0()

    # 獎勵函數
    rewards: RewardsCfgPhase0 = RewardsCfgPhase0()

    # 終止條件
    terminations: TerminationsCfgPhase0 = TerminationsCfgPhase0()

    # 命令管理
    commands: CommandsCfgPhase0 = CommandsCfgPhase0()

    # 事件管理 (域隨機化)
    events: EventCfgPhase0 = EventCfgPhase0()
```

### SB3 PPO 配置項 (`sb3_ppo_cfg_phase0.yaml`)

```yaml
policy: MlpPolicy
learning_rate: 0.0001
n_steps: 24
batch_size: 4096
n_epochs: 5
n_minibatches: 4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.001

# 正規化
normalize_input: false
normalize_value: false
```

---

## 🏆 獎勢設計 (Pull-Push Theory)

### Pull Forces (吸引到目標)

| 獎勵項 | 權重 | 公式 |
|--------|------|------|
| `progress_to_goal` | 20.0 | d_{t-1} - d_t |
| `reaching_goal` | 15.0 | +15 if d < 0.5m |
| `velocity_toward_goal` | 1.0 | v · (goal_dir) |

### Push Forces (推離障礙)

| 獎勵項 | 權重 | 公式 |
|--------|------|------|
| `safety_field` | -2.0 | 見下方「安場懲罰」 |
| `collision_penalty` | -20.0 | -20 if d < 0.25m |

### Smoothness (平滑度)

| 獎勵項 | 權重 | 公式 |
|--------|------|------|
| `action_smoothness` | -0.1 | -0.1 × ||a_t - a_{t-1}|| |
| `time_out_penalty` | -0.05 | -0.05 × (1 - t/T) |

### 安場懲罰 (Safety Field Penalty)

```
距離 d         懲罰值
─────────────────────────
0.00m         -1.0
0.25m         -1.0
0.30m         -0.86
0.40m         -0.57
0.50m         -0.29
0.60m          0.0
> 0.60m         0.0
```

---

## 🤖 域隨機化配置

### 啟用的隨機化項

```python
# 文件: charge_sb3/domain_randomization/dr_manager.py

@dataclass
class DomainRandConfig:
    enable: bool = True

    # 初始速度隨機
    init_velocity_range: tuple = (-0.5, 0.5)      # m/s
    init_angular_velocity_range: tuple = (-0.5, 0.5)  # rad/s

    # LiDAR 噪聲
    lidar_noise_range: tuple = (-0.1, 0.1)  # ±10cm

    # 物理參數隨機
    mass_range: tuple = (0.9, 1.1)      # 質量 ±10%
    friction_range: tuple = (0.8, 1.2)  # 摩擦力 ±20%

    # 外部擾動
    external_force_prob: float = 0.05     # 5% 概率
    external_force_range: tuple = (5.0, 20.0)  # 5-20N
```

---

## 📊 觀測空間 (131 維統一)

| 維度名 | 大小 | 範圍/類型 |
|--------|------|-----------|
| `lidar_scan` | 72 | [0, 10m] (距離) |
| `speed` | 2 | [-1, 1] m/s (歸一化) |
| `goal_position` | 2 | [-8, 8] m (相對位置) |
| `goal_distance` | 1 | [0, 12] m |
| `time_remaining` | 1 | [0, 1] |
| `alive` | 1 | {0, 1} |
| `actions` | 2 | [-1, 1] (上一步動作) |
| `obstacles` | 50 | Phase 0 全為 0 |

---

## 🔧 修改指南

### 添加新的獎勵項

1. 編輯 `mdp/rewards/my_reward.py`:
```python
def my_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    # 實現你的獎勵邏輯
    return reward  # [num_envs] 形狀
```

2. 在 `charge_env_cfg_phaseX.py` 中註冊:
```python
from ..mdp.rewards import my_reward

my_reward = RewTerm(
    func=my_reward,
    weight=1.0,
)
```

### 修改獎勵權重

```python
# 在 RewardsCfgPhaseX 中修改權重
progress_to_goal = RewTerm(
    func=progress_to_goal,
    weight=20.0,  # ← 修改這裡
)
```

### 調整觀測空間

```python
# 在 ObservationsCfgPhaseX.PolicyCfg 中添加/移除觀測項
my_observation = ObsTerm(
    func=my_observation_func,
    params={"sensor_cfg": SceneEntityCfg("my_sensor")},
)
```

---

## 📈 WandB 監控

### 查看訓練進度

```bash
# 登入 WandB 專目
wandb login

# 訪練時會自動記錄到 charge_sb3 項目
# 訪練結束後查看:
https://wandb.ai/your-username/charge_sb3
```

### 重要指標

- `rollouts/ep_rew_mean`: 平均 reward (越高越好)
- `rollouts/ep_len_mean`: episode 長度
- `success_rate`: 成功率 (目標 > 95%)
- `Episode_Termination/goal_reached`: 成功次數
- `Episode_Termination/wall_collision`: 碰撞次數

---

## 🐛 常見問題

### Q1: 訓練時出現 `AttributeError: 'Articulation' object has no attribute 'apply_forces'`

**A**: 已修復。使用 `permanent_wrench_composer.set_forces_and_torques()` 替代。

### Q2: WandB 無法記錄指標

**A**: 確保:
1. 使用 `--headless` 模式
2. 已執行 `wandb login`
3. 網絡連接正常

### Q3: 觀測維度不匹配

**A**: 檢查所有 Phase 使用統一的 131 維觀測。

### Q4: 訓練速度慢

**A**:
- 增加 `--num_envs` (如 512)
- 關閉可視化 (`--headless`)
- 減少 `--video` 記錄頻率

### Q5: 內存不足

**A**:
- 減少 `--num_envs`
- 降低 batch_size
- 減小模型規模

---

## 📚 相關文檔

- [完整架構文檔](docs/CHARGE_TRAINING_ARCHITECTURE.md)
- [Phase 0 規格書](docs/PHASE0_SPEC.md) (待創建)
- [AIT* 整合指南](docs/AITSTAR_INTEGRATION.md) (待創建)

---

**最後更新**: 2026-02-06
**維護者**: Claude Opus 4.5
