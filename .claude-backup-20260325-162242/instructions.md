# Claude Code 開發指令 - IsaacLab Charge 專案

## 🚨 首次啟動必讀 (First-Time Setup)

**請在開始任何工作前，先執行以下指令：**

```
請閱讀 /home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/AI_QUICK_REFERENCE.md
了解這個專案的背景和結構
```

---

## 專案概述

這是一個 **Sim-to-Real 自主導航機器人** 專案，使用深度強化學習（DRL）訓練 Charge 機器人進行無地圖導航。

### 當前訓練階段
- **Phase 3 (進行中)**: 長程導航，自適應課程學習（2-15m 目標距離）

### 專案路徑
```
/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3
```

---

## 快速文件定位

| 需求 | 文件路徑 |
|------|---------|
| **環境註冊** | `charge_sb3/__init__.py` |
| **機器人配置** | `charge_sb3/cfg/charge_cfg.py` |
| **Phase 0 配置** | `charge_sb3/cfg/charge_env_cfg_phase0.py` |
| **Phase 1 配置** | `charge_sb3/cfg/charge_env_cfg.py` |
| **Phase 2 配置** | `charge_sb3/cfg/charge_env_cfg_v2.py` |
| **Phase 3 配置** | `charge_sb3/cfg/charge_env_cfg_v3.py` |
| **SB3 PPO 參數** | `charge_sb3/agents/sb3_ppo_cfg*.yaml` |
| **觀測函數** | `charge_sb3/mdp/observations/functions.py` |
| **獎勵函數** | `charge_sb3/mdp/rewards/*.py` |
| **終止條件** | `charge_sb3/mdp/terminations/*.py` |
| **課程學習** | `charge_sb3/mdp/events/curriculum.py` |

---

## 註冊的環境 ID

### Phase 0（基礎運動學）
- `Isaac-Navigation-Charge-Phase0` (訓練)
- `Isaac-Navigation-Charge-Phase0-Play` (測試)

### Phase 1-3（RSL-RL）
- `Isaac-Navigation-Charge-v0/v1/v2/v3` (訓練)
- `Isaac-Navigation-Charge-Play-v0/v1/v2/v3` (測試)

### Stable-Baselines3
- `Isaac-Navigation-Charge-SB3-v0/v1/v2/v3` (訓練)
- `Isaac-Navigation-Charge-SB3-Play-v0/v1/v2/v3` (測試)

### 層級式導航（AIT*）
- `Isaac-Navigation-Charge-Hierarchical-v0/v1`

---

## 標準開發流程

### 修改觀測
1. 在 `mdp/observations/functions.py` 定義函數
2. 在 `mdp/observations/__init__.py` 導出
3. 在 `cfg/charge_env_cfg_*.py` 的 `ObservationsCfg` 中添加

### 修改獎勵
1. 在 `mdp/rewards/` 中定義函數
2. 在 `mdp/rewards/__init__.py` 導出
3. 在 `cfg/charge_env_cfg_*.py` 的 `RewardsCfg` 中添加

### 創建新環境版本
1. 複製現有配置文件（如 `charge_env_cfg_v3.py`）
2. 修改類名（如 `ChargeNavigationEnvCfgV4`）
3. 在 `cfg/__init__.py` 導出
4. 在主 `__init__.py` 註冊新環境

---

## 訓練命令

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

## 已知問題

| 問題 | 解決方案 |
|------|----------|
| 蛇行行為 | 降低碰撞懲罰，增加目標導向獎勵 |
| 翻車騙分 | 使用姿態門控 |
| PPO std>=0 | 使用 `ChargeNavigationEnv` 自定義環境類 |
| 恐懼障礙 | Phase 3 先移除障礙物 |

---

## 參考文檔

- **完整手冊**: `charge_sb3/AI_SESSION_MANUAL.md`
- **快速參考**: `charge_sb3/AI_QUICK_REFERENCE.md`
- **快速參考 (英文)**: `charge_sb3/AI_QUICK_REFERENCE_EN.md`
- **設計文檔**: `charge_sb3/md/`
