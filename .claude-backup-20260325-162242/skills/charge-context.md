---
name: charge-context
description: 載入 IsaacLab Charge 機器人導航專案的完整上下文，包含專案概述、檔案結構、環境ID、開發流程等。在處理 charge_sb3 相關任務時使用。
---

# Charge 專案上下文 (Charge Project Context)

## 🎯 專案概述

**專案類型**: Sim-to-Real 自主導航機器人
**機器人**: Charge（差速驅動 + 2D LiDAR）
**演算法**: Stable-Baselines3 PPO / RSL-RL PPO
**當前階段**: Phase 3 - 長程導航（自適應課程學習 2-15m）

**專案路徑**:
```
/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3
```

---

## 📁 關鍵檔案位置

| 用途 | 檔案路徑 |
|------|---------|
| 環境註冊 | `charge_sb3/__init__.py` |
| 機器人配置 | `charge_sb3/cfg/charge_cfg.py` |
| Phase 0 配置 | `charge_sb3/cfg/charge_env_cfg_phase0.py` |
| Phase 1 配置 | `charge_sb3/cfg/charge_env_cfg.py` |
| Phase 2 配置 | `charge_sb3/cfg/charge_env_cfg_v2.py` |
| Phase 3 配置 | `charge_sb3/cfg/charge_env_cfg_v3.py` |
| SB3 PPO 參數 | `charge_sb3/agents/sb3_ppo_cfg*.yaml` |
| 觀測函數 | `charge_sb3/mdp/observations/functions.py` |
| 獎勵函數 | `charge_sb3/mdp/rewards/*.py` |
| 終止條件 | `charge_sb3/mdp/terminations/*.py` |
| 課程學習 | `charge_sb3/mdp/events/curriculum.py` |
| 層級式環境 | `charge_sb3/cfg/charge_env.py` |

---

## 🎮 已註冊環境 ID

### Phase 0（基礎運動學）
```
Isaac-Navigation-Charge-Phase0           # 訓練
Isaac-Navigation-Charge-Phase0-Play      # 測試
```

### Phase 1-3（RSL-RL）
```
Isaac-Navigation-Charge-v0/v1/v2/v3           # 訓練
Isaac-Navigation-Charge-Play-v0/v1/v2/v3      # 測試
```

### Stable-Baselines3
```
Isaac-Navigation-Charge-SB3-v0/v1/v2/v3           # 訓練
Isaac-Navigation-Charge-SB3-Play-v0/v1/v2/v3      # 測試
```

### 層級式導航（AIT*）
```
Isaac-Navigation-Charge-Hierarchical-v0/v1
```

---

## 🔧 標準開發流程

### 修改觀測 (Observations)
```python
# 1. 在 mdp/observations/functions.py 定義函數
def your_new_obs(env: ManagerBasedEnv, dt: float) -> torch.Tensor:
    return some_value

# 2. 在 mdp/observations/__init__.py 導出
__all__ = [..., "your_new_obs"]

# 3. 在 cfg/charge_env_cfg_*.py 使用
class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "your_obs": ObsTerm(func=your_new_obs),
        },
    )
```

### 修改獎勵 (Rewards)
```python
# 1. 在 mdp/rewards/your_file.py 定義函數
def your_reward(env: ManagerBasedEnv) -> torch.Tensor:
    return reward_values

# 2. 在 mdp/rewards/__init__.py 導出
# 3. 在 cfg/charge_env_cfg_*.py 使用
class RewardsCfg:
    your_reward = RewTerm(func=your_reward, weight=1.0)
```

### 修改終止條件 (Terminations)
```python
# 流程同上，使用 DoneTerm
class TerminationsCfg:
    your_term = DoneTerm(func=your_term_func)
```

### 創建新環境版本
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

## 🚀 訓練指令

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

## ⚠️ 已知問題與解決方案

| 問題 | 解決方案 |
|------|----------|
| 蛇行行為（Snake Behavior） | 降低碰撞懲罰權重，增加目標導向獎勵 |
| 翻車騙分（Tipping Hack） | 使用姿態門控（Posture Gating） |
| PPO std>=0 錯誤 | 使用 `ChargeNavigationEnv` 自定義環境類 |
| 恐懼障礙物（Fear of Obstacles） | Phase 3 先移除障礙，建立導航意志 |

---

## 📚 參考文檔

- **完整手冊**: `charge_sb3/AI_SESSION_MANUAL.md`
- **快速參考（中文）**: `charge_sb3/AI_QUICK_REFERENCE.md`
- **快速參考（英文）**: `charge_sb3/AI_QUICK_REFERENCE_EN.md`
- **設計文檔**: `charge_sb3/md/`

---

## 使用說明

當 AI 載入此 skill 後，應該：
1. 理解這是一個 DRL 導航專案
2. 知道關鍵檔案的位置
3. 熟悉標準的修改流程
4. 了解當前訓練階段和已知問題
