# CBF + SB3 PPO 整合設計文檔

## 環境設置

**重要**: 測試 IsaacLab Python 環境時，需使用 conda 環境 `env_isaaclab`

```bash
conda activate env_isaaclab
```

## CBF 安全濾鏡架構設計

### 1. 核心概念

在 SB3 PPO 中，CBF (Control Barrier Function) 作為一個**安全濾網**插入在 Policy Network 輸出與 Environment 之間：

```
Observation → Policy Network → Action Distribution → Sampled Action (u_rl)
                                                           ↓
                                                     CBF Safety Filter
                                                           ↓
                                                     Safe Action (u_safe)
                                                           ↓
                                                      Environment
```

### 2. 實作架構

使用 Gym `ActionWrapper` 來攔截 PPO 輸出的動作，經過 CBF 修正後再傳給環境。

**核心問題 - On-Policy PPO**:
- PPO 假設訓練數據由當前策略產生
- 當 CBF 修改動作 (u_safe ≠ u_rl)，PPO 會誤以為 u_safe 是自己想出來的
- **解決方案**: 加入 CBF 修正懲罰 (penalty) 到 reward 中

### 3. 檔案結構

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/
├── wrappers/
│   ├── __init__.py
│   └── cbf_action_wrapper.py    # CBF ActionWrapper for SB3
├── mdp/
│   └── rewards/
│       └── cbf_penalty.py        # CBF 修正懲罰
└── hierarchical/
    ├── safety_shield.py          # 現有的 CBF 實作
    └── types.py                  # 類型定義
```

### 4. 使用方式

```python
from stable_baselines3 import PPO
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.wrappers import CBFActionWrapper

# 1. 建立 Isaac Sim 環境
env = gym.make("Isaac-charge-SB3-v0")

# 2. 套上 CBF 安全濾鏡
safe_env = CBFActionWrapper(env, penalty_coeff=0.5)

# 3. 轉換為 SB3 VecEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
safe_env = Sb3VecEnvWrapper(safe_env)

# 4. 建立 PPO 模型
model = PPO("MlpPolicy", safe_env, verbose=1)

# 5. 開始訓練
model.learn(total_timesteps=100000)
```

### 5. 獎勵機制設計

#### CBF 修正懲罰 (Correction Penalty)

當 CBF 介入修改動作時，給予 PPO 負獎勵：

```python
penalty = penalty_coeff * ||u_rl - u_safe||^2
reward -= penalty
```

**參數說明**:
- `penalty_coeff`: 懲罰係數 (建議 0.5 ~ 2.0)
- `||u_rl - u_safe||`: 修正量 (L2 norm)

#### 記錄資訊

在 `info` 中記錄 CBF 觸發情況：
- `cbf_triggered`: CBF 是否介入 (bool)
- `cbf_correction`: 修正量大小 (float)
- `cbf_reason`: CBF 介入原因 (str)

### 6. AIT* 整合

AIT* 作為 Global Planner：
- `env.reset()`: 呼叫 AIT* 規劃全域路徑
- `env.step()`: 計算局部目標 (local goal)，放入 observation
- 路徑受阻時觸發 AIT* repair/rewire

### 7. 課程學習 (Curriculum Learning)

建議分階段調整 CBF 參數：

| 階段 | CBF 係數 | penalty_coeff | 說明 |
|------|----------|---------------|------|
| 初期 | 寬鬆 | 低 (0.1) | 讓 PPO 多經歷危險狀態 |
| 中期 | 適中 | 中 (0.5) | 平衡學習與安全 |
| 後期 | 嚴格 | 高 (1.0) | 強化安全行為 |

### 8. 監控指標

在 TensorBoard 追蹤：
- `train/cbf_trigger_rate`: CBF 觸發頻率
- `train/cbf_correction_mean`: 平均修正量
- `train/cbf_correction_max`: 最大修正量

理想情況：隨訓練進行，CBF 觸發率逐漸下降。

---

**版本**: 1.0
**建立日期**: 2026-02-04
**相關文件**: CHARGE_TRAINING_DESIGN.md
