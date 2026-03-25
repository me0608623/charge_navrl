# Stable Baselines 3 PPO - Charge 導航快速參考卡

## 🚀 快速開始

### 一行命令訓練

```bash
# Phase 1（3 個障礙物）- 基礎訓練
./train_sb3.sh v1

# Phase 2（5 個障礙物）- 進階訓練
./train_sb3.sh v2

# Phase 3（目標距離課程）- 最終訓練
./train_sb3.sh v3
```

---

## 📋 環境 ID 一覽

### 訓練環境

| 環境 ID | 階段 | 障礙物數量 | 配置文件 |
|---------|------|-----------|----------|
| `Isaac-Navigation-Charge-SB3-v0` | v0 | 0 | `sb3_ppo_cfg.yaml` |
| `Isaac-Navigation-Charge-SB3-v1` | v1 | 3 | `sb3_ppo_cfg.yaml` |
| `Isaac-Navigation-Charge-SB3-v2` | v2 | 5 | `sb3_ppo_cfg_v2.yaml` |
| `Isaac-Navigation-Charge-SB3-v3` | v3 | 0（可變距離） | `sb3_ppo_cfg_v3.yaml` |

### 測試環境

| 環境 ID | 用途 |
|---------|------|
| `Isaac-Navigation-Charge-SB3-Play-v0/v1/v2/v3` | 測試/演示 |

---

## ⚙️ 常用訓練命令

### 基礎命令

```bash
# 使用 IsaacLab 腳本
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --num_envs 128 \
    --headless \
    --agent sb3_cfg_entry_point

# 使用便捷腳本
./train_sb3.sh v1
```

### 自定義環境數量

```bash
# 64 環境（GPU 內存有限時）
./train_sb3.sh v1 --num-envs 64

# 256 環境（GPU 內存充足時）
./train_sb3.sh v1 --num-envs 256
```

### 使用不同種子

```bash
# 種子 42（默認）
./train_sb3.sh v1 --seed 42

# 種子 123
./train_sb3.sh v1 --seed 123
```

### 啟用 GUI（調試模式）

```bash
# 顯示模擬窗口
./train_sb3.sh v1 --no-headless
```

### 啟用錄像

```bash
# 錄製訓練過程
./train_sb3.sh v1 --video
```

### 從檢查點恢復

```bash
# 從最新檢查點恢復
./train_sb3.sh v1 --resume

# 從指定檢查點恢復
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --checkpoint logs/sb3/Isaac-Navigation-Charge-SB3-v1/.../model.zip \
    --agent sb3_cfg_entry_point
```

---

## 📊 TensorBoard 監控

```bash
# 啟動 TensorBoard
tensorboard --logdir logs/sb3/Isaac-Navigation-Charge-SB3-v1/

# 瀏覽器打開：http://localhost:6006
```

### 關鍵指標

| 指標 | 說明 | 正常範圍 |
|------|------|----------|
| `ep_rew_mean` | 平均獎勵 | 逐漸上升 |
| `ep_len_mean` | 平均 episode 長度 | 逐漸下降或穩定 |
| `success_rate` | 成功率 | > 70% (v1), > 60% (v2), > 80% (v3) |
| `clip_fraction` | 被裁剪的比例 | 0.1 - 0.2 |
| `entropy_loss` | 熵損失 | 逐漸下降 |
| `policy_loss` | 策略損失 | 波動後穩定 |
| `value_loss` | 價值損失 | 逐漸下降 |

---

## 🔧 參數快速調整

### 當訓練不穩定時

```yaml
# 編輯配置文件：agents/sb3_ppo_cfg.yaml

# 降低學習率
learning_rate: 1e-4  →  5e-5

# 收緊 clipping
clip_range: 0.2  →  0.15

# 收緊梯度裁剪
max_grad_norm: 1.0  →  0.8
```

### 當收斂太慢時

```yaml
# 提高學習率
learning_rate: 1e-4  →  1.5e-4

# 增加探索
ent_coef: 0.001  →  0.005
```

### 當探索不足時

```yaml
# 增加熵係數
ent_coef: 0.001  →  0.005

# 放寬 clipping
clip_range: 0.2  →  0.3
```

---

## 📁 文件結構

```
charge_sb3/
├── train_sb3.sh                  # ✨ 便捷訓練腳本
├── README_SB3_PPO.md             # 詳細使用文檔
├── SB3_CONFIG_COMPARISON.md       # 配置對比表
├── QUICK_REFERENCE.md             # 本文件（快速參考）
├── __init__.py                   # 環境註冊
├── agents/
│   ├── sb3_ppo_cfg.yaml          # Phase 1 配置
│   ├── sb3_ppo_cfg_v2.yaml       # Phase 2 配置
│   ├── sb3_ppo_cfg_v3.yaml       # Phase 3 配置
│   ├── rsl_rl_ppo_cfg*.py        # RSL-RL 配置（保留）
│   └── __init__.py
├── cfg/
│   ├── charge_env_cfg.py         # Phase 1 環境
│   ├── charge_env_cfg_v2.py      # Phase 2 環境
│   ├── charge_env_cfg_v3.py      # Phase 3 環境
│   ├── charge_cfg.py             # 機器人配置
│   └── charge_env.py             # 自定義環境類
└── mdp/                          # MDP 模組
    ├── actions/                  # 動作
    ├── observations/             # 觀測
    ├── rewards/                  # 獎勵
    ├── terminations/             # 終止條件
    ├── events/                   # 事件
    └── core/                     # 核心函數
```

---

## 🎯 三階段訓練流程

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1 (v1)                                        │
│  ─────────────────────────────────                    │
│  障礙物：3 個靜態障礙物                               │
│  目標距離：1.5-3.0 米                                 │
│  訓練步數：50M                                       │
│  預期成功率：> 70%                                    │
│  配置：sb3_ppo_cfg.yaml                              │
│  命令：./train_sb3.sh v1                             │
└─────────────────────────────────────────────────────────┘
                         ↓
                         ↓ 完成後
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2 (v2)                                        │
│  ─────────────────────────────────                    │
│  障礙物：5 個靜態障礙物                               │
│  目標距離：1.5-4.0 米                                 │
│  訓練步數：30M                                       │
│  預期成功率：> 60%                                    │
│  配置：sb3_ppo_cfg_v2.yaml                           │
│  命令：./train_sb3.sh v2                             │
└─────────────────────────────────────────────────────────┘
                         ↓
                         ↓ 完成後
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 3 (v3)                                        │
│  ─────────────────────────────────                    │
│  障礙物：0（無障礙物）                                │
│  目標距離：2.0-15.0 米（可變）                         │
│  訓練步數：50M                                       │
│  預期成功率：> 80%                                    │
│  配置：sb3_ppo_cfg_v3.yaml                           │
│  命令：./train_sb3.sh v3                             │
└─────────────────────────────────────────────────────────┘
```

---

## 🐛 常見問題快速修復

### 問題：訓練啟動失敗

**錯誤**：`ModuleNotFoundError: No module named 'stable_baselines3'`

**修復**：
```bash
pip install stable-baselines3
```

---

### 問題：觀測包含 NaN/Inf

**錯誤**：`RuntimeError: Observation contains NaN/Inf`

**修復**：
- 檢查環境配置中的 `charge_env.py` 是否已導入
- 已經包含 `_sanitize_observation` 方法，自動清理異常值

---

### 問題：策略崩潰（獎驟劇烈下降）

**症狀**：`ep_rew_mean` 從 100 降到 -1000

**修復**：
```yaml
# 編輯配置文件，收緊參數
clip_range: 0.2  →  0.1
learning_rate: 1e-4  →  5e-5
max_grad_norm: 1.0  →  0.5
```

然後從最佳檢查點恢復：
```bash
./train_sb3.sh v1 --resume
```

---

### 問題：GPU 內存不足

**症狀**：`CUDA out of memory`

**修復**：
```bash
# 減少環境數量
./train_sb3.sh v1 --num-envs 64
```

---

### 問題：成功率不上升

**症狀**：`success_rate` 長期停留在 30%

**修復**：
```yaml
# 檢查獎勵權重（在環境配置文件中）
reaching_goal: 200  # 確保足夠大
progressive_collision: -0.5  # 確保不會壓垮目標獎勵
```

---

## 📚 詳細文檔

- **完整使用指南**：`README_SB3_PPO.md`
- **配置對比表**：`SB3_CONFIG_COMPARISON.md`
- **Stable Baselines 3 文檔**：https://stable-baselines3.readthedocs.io/
- **IsaacLab RL 指南**：https://isaac-sim.github.io/IsaacLab/

---

## 📞 獲取幫助

1. 查看 `README_SB3_PPO.md` 詳細文檔
2. 查看 `SB3_CONFIG_COMPARISON.md` 配置對比
3. 使用 TensorBoard 監控訓練指標
4. 在 GitHub Issues 中報告問題

---

## ✅ 快速檢查清單

在開始訓練前，檢查以下項目：

- [ ] 已安裝 `stable-baselines3`
- [ ] 已安裝 `tensorboard`（可選）
- [ ] GPU 可用（檢查 `nvidia-smi`）
- [ ] 磁盤空間充足（至少 10 GB）
- [ ] 選擇了正確的階段（v0/v1/v2/v3）
- [ ] 環境數量適合 GPU 內存（64/128/256）
- [ ] 設置了隨機種子（用於可重現性）

---

**祝您訓練成功！** 🚀
