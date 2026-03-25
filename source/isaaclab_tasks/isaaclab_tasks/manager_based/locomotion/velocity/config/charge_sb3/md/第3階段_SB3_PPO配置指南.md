# SB3 PPO 配置指南

本文件詳細說明使用 **Stable Baselines 3 (SB3)** 框架和 **PPO (Proximal Policy Optimization)** 算法訓練 Charge 機器人導航任務的完整配置。

---

## 📋 概述

本目錄包含使用 Stable Baselines 3 PPO 訓練 Charge 機器人導航任務的完整配置。

### 與 RSL-RL 的區別

| 特性 | RSL-RL | Stable Baselines 3 |
|------|--------|-------------------|
| **框架** | NVIDIA 專用的 RSL-RL | 開源 SB3（更通用） |
| **配置文件** | Python 類 (`.py`) | YAML (`.yaml`) |
| **環境註冊** | `Isaac-Navigation-Charge-v*` | `Isaac-Navigation-Charge-SB3-v*` |
| **訓練腳本** | `scripts/reinforcement_learning/rsl_rl/train.py` | `scripts/reinforcement_learning/sb3/train_charge.py` |
| **記錄** | WandB + TensorBoard | TensorBoard |

---

## 🚀 快速開始

### 環境 ID 列表

所有可用的 SB3 環境：

| 環境 ID | 階段 | 障礙物數量 | 用途 |
|---------|------|-----------|------|
| `Isaac-Navigation-Charge-SB3-v0` | Phase 3 | 0（無障礙物） | 基礎導航訓練 |
| `Isaac-Navigation-Charge-SB3-v1` | Phase 1 | 3 | 基礎障礙物避讓 |
| `Isaac-Navigation-Charge-SB3-v2` | Phase 2 | 5 | 中等複雜度避障 |
| `Isaac-Navigation-Charge-SB3-v3` | Phase 3 | 0（目標距離課程） | 進階導航訓練 |

**Play 環境**（用於測試/演示）：
- `Isaac-Navigation-Charge-SB3-Play-v0`
- `Isaac-Navigation-Charge-SB3-Play-v1`
- `Isaac-Navigation-Charge-SB3-Play-v2`
- `Isaac-Navigation-Charge-SB3-Play-v3`

---

### 基礎訓練命令

#### Phase 1（3 個障礙物）訓練

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --num_envs 128 \
    --headless \
    --agent sb3_cfg_entry_point
```

#### Phase 2（5 個障礙物）訓練

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v2 \
    --num_envs 128 \
    --headless \
    --agent sb3_cfg_entry_point
```

#### Phase 3（目標距離課程）訓練

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v3 \
    --num_envs 128 \
    --headless \
    --agent sb3_cfg_entry_point
```

---

### 訓練參數說明

#### 常用參數

| 參數 | 說明 | 默認值 | 建議值 |
|------|------|--------|--------|
| `--task` | 環境 ID | - | `Isaac-Navigation-Charge-SB3-v1` |
| `--num_envs` | 並行環境數量 | 128 | 128（GPU充足）/ 64（GPU有限） |
| `--headless` | 無頭模式（無GUI） | False | True（訓練時） |
| `--agent` | Agent 配置入口點 | `sb3_cfg_entry_point` | `sb3_cfg_entry_point` |
| `--seed` | 隨機種子 | 42 | 42, 123, 456（不同實驗） |
| `--max_iterations` | 最大訓練迭代次數 | - | 從配置文件讀取 |
| `--checkpoint` | 從檢查點恢復訓練 | None | 路徑到 `.zip` 文件 |
| `--log_interval` | 日誌記錄間隔 | 100,000 | 50,000（更頻繁） |

#### 高級參數

| 參數 | 說明 |
|------|------|
| `--video` | 啟用錄像（訓練過程） |
| `--video_length` | 錄像長度（步數） |
| `--video_interval` | 錄像間隔（步數） |
| `--keep_all_info` | 保留所有訓練信息（較慢） |
| `--export_io_descriptors` | 導出 IO 描述符 |

---

## ⚙️ PPO 配置參數說明

### 核心參數

#### 基礎參數

```yaml
seed: 42                  # 隨機種子
n_timesteps: 5e7         # 總訓練步數（50M）
policy: 'MlpPolicy'       # 策略網絡類型
```

#### PPO 算法參數

```yaml
n_steps: 24                # 每次更新的步數（每個環境）
n_minibatches: 4          # Minibatch 數量
gae_lambda: 0.95          # GAE lambda 參數
gamma: 0.99               # 折扣因子
n_epochs: 5               # 每次更新的訓練 epoch 數
ent_coef: 0.001           # 熵係數（探索）
learning_rate: 1e-4       # 學習率
clip_range: 0.2           # PPO clipping 範圍
```

#### 策略網絡架構

```yaml
policy_kwargs:
  activation_fn: 'nn.ELU'      # 激活函數
  net_arch: [512, 256, 128]    # 隱藏層架構
  optimizer_kwargs:
    eps: 1e-8                  # Adam epsilon
  ortho_init: False            # 正交初始化
  vf_coef: 1.0                 # 價值函數損失權重
```

#### 梯度裁剪和歸一化

```yaml
max_grad_norm: 1.0         # 梯度裁剪範數
normalize_input: True      # 歸一化觀測
normalize_value: False     # 歸一化價值估計
clip_obs: 10.0             # 觀測裁剪閾值
```

---

## 🎯 訓練階段建議

### Phase 1：基礎導航（v1）

**目標**：學習基本的導航和避障能力

**環境配置**：
- 障礙物：3 個靜態障礙物
- 目標距離：1.5 - 3.0 米
- Episode 長度：45 秒

**建議訓練參數**：
```yaml
n_timesteps: 2e7         # 20M 步（基礎訓練）
learning_rate: 1e-4      # 標準學習率
ent_coef: 0.001          # 標準探索
clip_range: 0.2          # 標準 clipping
```

**預期結果**：
- 成功率 > 70%
- 平均獎勵逐漸上升
- Episode 長度逐漸下降（更快到達目標）

### Phase 2：中等複雜度（v2）

**目標**：處理更多障礙物，提高避障能力

**環境配置**：
- 障礙物：5 個靜態障礙物
- 目標距離：1.5 - 4.0 米
- Episode 長度：45 秒

**建議訓練參數**：
```yaml
n_timesteps: 3e7         # 30M 步（更多訓練）
learning_rate: 5e-5       # 降低學習率（更穩定）
ent_coef: 0.0005         # 減少探索
clip_range: 0.15         # 收緊 clipping
```

**預期結果**：
- 成功率 > 60%（障礙物更多）
- 更穩定的避障行為

### Phase 3：進階導航（v3）

**目標**：學習可變目標距離導航

**環境配置**：
- 障礙物：0（無障礙物）
- 目標距離：自適應（2-15 米）
- Episode 長度：45 秒

**建議訓練參數**：
```yaml
n_timesteps: 5e7         # 50M 步（完整訓練）
learning_rate: 5e-5       # 低學習率（更穩定）
ent_coef: 0.0005         # 減少探索
clip_range: 0.1          # 收緊 clipping（鎖住策略）
```

**預期結果**：
- 成功率 > 80%
- 能處理各種目標距離

---

## 📈 監控訓練

### TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir logs/sb3/Isaac-Navigation-Charge-SB3-v1/

# 在瀏覽器中打開：http://localhost:6006
```

### 關鍵指標

| 指標 | 說明 | 正常趨勢 |
|------|------|----------|
| `ep_rew_mean` | 平均 episode 獎勵 | 逐漸上升 |
| `ep_len_mean` | 平均 episode 長度 | 逐漸下降或穩定 |
| `success_rate` | 成功率 | 逐漸上升 |
| `loss/value_function` | 價值函數損失 | 逐漸下降 |
| `clip_fraction` | 被裁剪的比例 | < 0.2 |
| `entropy_loss` | 熵損失 | 逐漸下降 |
| `policy_loss` | 策略損失 | 波動後穩定 |

---

## 🔧 高級用法

### 從檢查點恢復訓練

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --num_envs 128 \
    --headless \
    --agent sb3_cfg_entry_point \
    --checkpoint logs/sb3/Isaac-Navigation-Charge-SB3-v1/2024-01-01_12-00-00/model.zip
```

### 使用不同種子進行多次實驗

```bash
# 實驗 1
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --seed 42 \
    --agent sb3_cfg_entry_point

# 實驗 2
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --seed 123 \
    --agent sb3_cfg_entry_point

# 實驗 3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --seed 456 \
    --agent sb3_cfg_entry_point
```

### 錄像訓練過程

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --num_envs 128 \
    --video \
    --video_length 200 \
    --video_interval 2000 \
    --agent sb3_cfg_entry_point
```

### 導出 IO 描述符（用於部署）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v1 \
    --export_io_descriptors \
    --agent sb3_cfg_entry_point
```

---

## 📚 參考資料

### Stable Baselines 3 文檔

- [官方文檔](https://stable-baselines3.readthedocs.io/)
- [PPO 算法說明](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [調試指南](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### IsaacLab 文檔

- [強化學習概覽](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [環境配置指南](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/intro_tutorials/index.html)

---

**最後更新**: 2026-02-05
