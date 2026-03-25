# Charge 專案 - AI 快速參考指南

> 📌 **每次新 Session 首先閱讀此文！**

---

## 🎯 專案核心信息

### 專案是什麼？
**Sim-to-Real 自主導航機器人** - 使用深度強化學習訓練 Charge 機器人進行無地圖導航。

### 機器人類型
- **差速驅動** (Differential Drive)
- **2D LiDAR** 感知
- **端到端學習**：LiDAR + State → 神經網路 → Velocity Command

### 專案路徑
```
/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3
```

---

## 📊 訓練階段 (當前進度)

| Phase | 描述 | 障礙物 | 狀態 |
|:-----:|------|:------:|:----:|
| **0** | 基礎運動學 | 無 | 基礎訓練 |
| **1** | 靜態避障 | 3個 | ✅ 完成 |
| **2** | 複雜靜態 | 5個 | ⚠️ 蛇行問題 |
| **3** | 長程導航 | 無（2-15m自適應）| 🔄 **進行中** |
| **4** | 動態環境 | 動態 | 待開始 |

**當前重點**：Phase 3 - 建立強大的「導航意志」，消除恐懼

---

## 🗂️ 關鍵文件速查

```
charge_sb3/
├── __init__.py                    ⭐ 所有環境註冊（入口！）
│
├── cfg/
│   ├── charge_cfg.py              🤖 機器人物理配置
│   ├── charge_env_cfg.py          Phase 1 配置
│   ├── charge_env_cfg_v2.py       Phase 2 配置
│   ├── charge_env_cfg_v3.py       Phase 3 配置（自適應課程）
│   ├── charge_env_cfg_phase0.py   Phase 0 配置（基礎運動學）
│   └── charge_env.py              HierarchicalChargeNavigationEnv
│
├── agents/
│   ├── sb3_ppo_cfg.yaml           SB3 PPO 參數
│   ├── sb3_ppo_cfg_v2.yaml
│   └── sb3_ppo_cfg_v3.yaml
│
└── mdp/
    ├── observations/              👁️ 觀測函數
    ├── rewards/                   🎁 獎勵函數
    ├── terminations/              🛑 終止條件
    ├── events/                    📅 事件/課程學習
    └── path_planner/              🗺️ AIT* 路徑規劃
```

---

## 🎮 註冊的環境 ID

### Phase 0（基礎運動學）
```
Isaac-Navigation-Charge-Phase0           # 訓練
Isaac-Navigation-Charge-Phase0-Play      # 測試
```

### Phase 1-3（RSL-RL）
```
Isaac-Navigation-Charge-v0/v1/v2/v3      # 訓練
Isaac-Navigation-Charge-Play-v0/v1/v2/v3 # 測試
```

### Stable-Baselines3
```
Isaac-Navigation-Charge-SB3-v0/v1/v2/v3       # 訓練
Isaac-Navigation-Charge-SB3-Play-v0/v1/v2/v3  # 測試
```

### 層級式導航（AIT*）
```
Isaac-Navigation-Charge-Hierarchical-v0/v1
```

---

## 🔧 修改配置標準流程

### 1️⃣ 修改觀測 (Observations)

```python
# 1. 在 mdp/observations/functions.py 定義函數
def your_new_obs(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    """你的觀測函數"""
    return some_value

# 2. 在 mdp/observations/__init__.py 導出
__all__ = [..., "your_new_obs"]

# 3. 在 cfg/charge_env_cfg_*.py 中使用
from ..mdp.observations import your_new_obs

class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "your_obs_name": ObsTerm(func=your_new_obs),
        },
    )
```

### 2️⃣ 修改獎勵 (Rewards)

```python
# 1. 在 mdp/rewards/your_rewards.py 定義函數
def your_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """你的獎勵函數，返回 [num_envs] 張量"""
    return reward_values

# 2. 在 mdp/rewards/__init__.py 導出
# 3. 在 cfg/charge_env_cfg_*.py 中使用
class RewardsCfg:
    your_reward_term = RewTerm(
        func=your_reward,
        weight=1.0,  # 獎勵權重
        params={},    # 可選參數
    )
```

### 3️⃣ 修改終止條件 (Terminations)

```python
# 流程同上，使用 DoneTerm
class TerminationsCfg:
    your_termination = DoneTerm(func=your_term_func)
```

### 4️⃣ 創建新環境版本

```python
# 1. 複製 cfg/charge_env_cfg_v3.py → cfg/charge_env_cfg_v4.py
# 2. 修改類名：ChargeNavigationEnvCfgV4
# 3. 在 cfg/__init__.py 導出
# 4. 在主 __init__.py 註冊
gym.register(
    id="Isaac-Navigation-Charge-SB3-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v4:ChargeNavigationEnvCfgV4",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```

---

## 🎁 常用獎勵函數速查

| 函數 | 位置 | 用途 |
|------|------|------|
| `velocity_toward_goal` | goal_rewards.py | 速度在目標方向投影 |
| `reaching_goal` | goal_rewards.py | 抵達目標獎勵 |
| `collision_penalty` | safety_rewards.py | 碰撞懲罰 |
| `forward_velocity_reward` | motion_rewards.py | 前進速度獎勵 |
| `action_rate_penalty` | motion_rewards.py | 防止抖動 |
| `progress_along_path` | goal_rewards.py | 路徑跟隨（AIT*） |

---

## 📝 環境配置結構模板

```python
@configclass
class ChargeNavigationEnvCfgVX:
    """Phase X 環境配置"""

    # 場景配置
    scene: MySceneCfg = MySceneCfg(num_envs=4096)

    # 觀測配置
    observations: ObservationsCfg = ObservationsCfg()

    # 動作配置
    actions: ActionsCfg = ActionsCfg()

    # 獎勵配置
    rewards: RewardsCfg = RewardsCfg()

    # 終止配置
    terminations: TerminationsCfg = TerminationsCfg()

    # 事件配置（課程學習）
    events: EventCfg = EventCfg()

    # 命令配置（目標生成）
    commands: CommandsCfg = CommandsCfg()
```

---

## 🚀 訓練命令

```bash
# SB3 訓練
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --num_envs 256 --headless

# 測試模型
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Navigation-Charge-SB3-Play-v0 \
    --load_path path/to/model.zip
```

---

## ⚠️ 已知問題

| 問題 | 解決方案 |
|------|----------|
| **蛇行** | 降低碰撞懲罰，增加目標導向獎勵 |
| **翻車騙分** | 使用姿態門控 |
| **PPO std>=0** | 使用 ChargeNavigationEnv 自定義環境類 |
| **恐懼障礙** | Phase 3 先移除障礙，建立導航意志 |

---

## 💡 開發原則

1. **先讀再改**：修改前先看現有實現
2. **模組化**：新函數放在對應 mdp 子目錄
3. **導出記得**：修改後更新 `__init__.py`
4. **向後兼容**：新增用 v4/v5，覆蓋要小心
5. **測試先行**：小規模測試再大訓練

---

## 📚 更多信息

完整手冊：`AI_SESSION_MANUAL.md`
設計文檔：`md/` 目錄
