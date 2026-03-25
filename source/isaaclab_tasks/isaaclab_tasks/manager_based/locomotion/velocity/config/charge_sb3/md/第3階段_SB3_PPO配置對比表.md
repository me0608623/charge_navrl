# Stable Baselines 3 PPO 配置對比表

## 概述

本文檔提供了 Charge 導航任務的三個 Stable Baselines 3 PPO 配置的詳細對比。

---

## 配置對比表

| 參數 | sb3_ppo_cfg.yaml (v1) | sb3_ppo_cfg_v2.yaml (v2) | sb3_ppo_cfg_v3.yaml (v3) | 說明 |
|------|----------------------|------------------------|------------------------|------|
| **基礎配置** | | | | |
| 總訓練步數 | 50,000,000 | 30,000,000 | 50,000,000 | v2 步數較少但環境更複雜 |
| 隨機種子 | 42 | 42 | 42 | 所有配置相同種子 |
| 策略類型 | MlpPolicy | MlpPolicy | MlpPolicy | 多層感知機 |
| **PPO 參數** | | | | |
| n_steps | 24 | 24 | 24 | 每次更新的步數（每個環境） |
| n_minibatches | 4 | 4 | 4 | Minibatch 數量 |
| gae_lambda | 0.95 | 0.95 | 0.95 | GAE lambda 參數 |
| gamma | 0.99 | 0.99 | 0.99 | 折扣因子 |
| n_epochs | 5 | 6 | 8 | 每次更新的訓練 epoch 數（逐階段增加） |
| **探索和學習** | | | | |
| ent_coef | 0.001 | 0.0005 | 0.0001 | 熵係數（逐階段降低，減少探索） |
| learning_rate | 1e-4 | 5e-5 | 3e-5 | 學習率（逐階段降低，更穩定） |
| clip_range | 0.2 | 0.15 | 0.1 | PPO clipping 範圍（逐階段收緊） |
| **網絡架構** | | | | |
| activation_fn | nn.ELU | nn.ELU | nn.ELU | 激活函數（所有相同） |
| net_arch | [512, 256, 128] | [512, 256, 128] | [512, 256, 128] | 隱藏層架構（所有相同） |
| ortho_init | False | False | False | 正交初始化（所有相同） |
| vf_coef | 1.0 | 1.0 | 1.0 | 價值函數損失權重（所有相同） |
| **梯度和歸一化** | | | | |
| max_grad_norm | 1.0 | 0.8 | 0.5 | 梯度裁剪範數（逐階段收緊） |
| normalize_input | True | True | True | 歸一化觀測（所有相同） |
| normalize_value | False | True | True | 歸一化價值估計（v2/v3 啟用） |
| clip_obs | 10.0 | 10.0 | 10.0 | 觀測裁剪閾值（所有相同） |
| **環境特徵** | | | | |
| 障礙物數量 | 3 | 5 | 0 | v3 無障礙物但有目標距離課程 |
| 目標距離範圍 | 1.5-3.0m | 1.5-4.0m | 2.0-15.0m（可變） | v3 目標距離動態調整 |
| Episode 長度 | 45s | 45s | 45s | 所有相同 |

---

## 配置變化趨勢

### 1. 學習率遞減策略

```
v1 (Phase 1): 1e-4  →  v2 (Phase 2): 5e-5  →  v3 (Phase 3): 3e-5
    標準              更穩定                最穩定
```

**原因**：
- Phase 1：基礎訓練，標準學習率足夠
- Phase 2：環境更複雜，需要更穩定的訓練
- Phase 3：最複雜，需要最穩定的訓練

### 2. 探索係數遞減策略

```
v1 (Phase 1): 0.001  →  v2 (Phase 2): 0.0005  →  v3 (Phase 3): 0.0001
    標準探索          減少探索               最小探索
```

**原因**：
- Phase 1：需要較多探索來學習基本策略
- Phase 2：已學會基本策略，減少隨機性
- Phase 3：需要高度確定的策略

### 3. Clipping 範圍收緊策略

```
v1 (Phase 1): 0.2  →  v2 (Phase 2): 0.15  →  v3 (Phase 3): 0.1
    標準 clipping     收緊 clipping           激進收緊
```

**原因**：
- Phase 1：允許較大的策略更新
- Phase 2：更保守的更新，保護已學會的策略
- Phase 3：極度保守的更新，鎖住最終策略

### 4. 訓練 Epoch 數增加策略

```
v1 (Phase 1): 5 epochs  →  v2 (Phase 2): 6 epochs  →  v3 (Phase 3): 8 epochs
    標準              更多訓練              最多訓練
```

**原因**：
- Phase 1：標準訓練次數
- Phase 2：環境更複雜，需要更多訓練
- Phase 3：最複雜，需要最多訓練

### 5. 梯度裁剪收緊策略

```
v1 (Phase 1): 1.0  →  v2 (Phase 2): 0.8  →  v3 (Phase 3): 0.5
    標準裁剪          收緊裁剪              激進收緊
```

**原因**：
- Phase 1：標準梯度控制
- Phase 2：更激進的梯度控制
- Phase 3：最激進的梯度控制，確保穩定性

---

## 使用建議

### Phase 1 (v1) - sb3_ppo_cfg.yaml

**適用場景**：
- 基礎導航訓練
- 首次嘗試 Charge 導航任務
- 快速驗證環境配置

**訓練命令**：
```bash
./train_sb3.sh v1
```

**預期效果**：
- 成功率 > 70%
- 訓練時間：4-6 小時（50M 步）

### Phase 2 (v2) - sb3_ppo_cfg_v2.yaml

**適用場景**：
- 完成 Phase 1 後的進階訓練
- 需要處理更多障礙物
- 提高避障能力

**訓練命令**：
```bash
./train_sb3.sh v2
```

**預期效果**：
- 成功率 > 60%
- 訓練時間：3-5 小時（30M 步，但環境更複雜）

### Phase 3 (v3) - sb3_ppo_cfg_v3.yaml

**適用場景**：
- 完成 Phase 2 後的進階訓練
- 需要處理可變目標距離
- 最終生產環境訓練

**訓練命令**：
```bash
./train_sb3.sh v3
```

**預期效果**：
- 成功率 > 80%
- 訓練時間：6-8 小時（50M 步）

---

## 參數調整指南

### 當訓練不穩定時

**症狀**：獎勵劇烈波動，策略崩潰

**解決方案**（逐步嘗試）：
1. 降低學習率：`learning_rate: 5e-5` → `3e-5`
2. 收緊 clipping：`clip_range: 0.2` → `0.15`
3. 收緊梯度裁剪：`max_grad_norm: 1.0` → `0.8`
4. 增加訓練 epoch：`n_epochs: 5` → `8`

### 當收斂太慢時

**症狀**：獎勵長期不上升

**解決方案**（逐步嘗試）：
1. 提高學習率：`learning_rate: 1e-4` → `1.5e-4`
2. 增加探索：`ent_coef: 0.001` → `0.005`
3. 減少訓練 epoch：`n_epochs: 8` → `5`

### 當探索不足時

**症狀**：策略過早收斂到次優解

**解決方案**（逐步嘗試）：
1. 增加熵係數：`ent_coef: 0.001` → `0.005`
2. 放寬 clipping：`clip_range: 0.1` → `0.2`
3. 降低學習率：`learning_rate: 1e-4` → `5e-5`

---

## 監控指標參考值

### 正常範圍

| 指標 | Phase 1 (v1) | Phase 2 (v2) | Phase 3 (v3) |
|------|-------------|-------------|-------------|
| `clip_fraction` | 0.1 - 0.2 | 0.1 - 0.15 | 0.05 - 0.1 |
| `entropy_loss` | 0.5 - 0.8 | 0.3 - 0.5 | 0.1 - 0.3 |
| `policy_loss` | 波動後穩定 | 波動後穩定 | 逐漸穩定 |
| `value_loss` | 逐漸下降 | 逐漸下降 | 逐漸下降 |
| `ep_rew_mean` | 逐漸上升 | 波動中上升 | 波動中上升 |

### 警告信號

| 指標 | 警告值 | 可能原因 | 解決方案 |
|------|--------|----------|----------|
| `clip_fraction` | > 0.3 | 更新幅度太大 | 收緊 `clip_range` |
| `clip_fraction` | < 0.01 | 更新幅度太小 | 放寬 `clip_range` |
| `entropy_loss` | < 0.1 | 探索不足 | 增加 `ent_coef` |
| `policy_loss` | 劇烈波動 | 學習率太高 | 降低 `learning_rate` |
| `ep_rew_mean` | 長期下降 | 策略崩潰 | 從檢查點恢復 |

---

## 遷移學習建議

### Phase 1 → Phase 2 遷移

```bash
# 1. 先訓練 Phase 1
./train_sb3.sh v1

# 2. 找到 Phase 1 的最佳檢查點
# 例如：logs/sb3/Isaac-Navigation-Charge-SB3-v1/.../model.zip

# 3. 從 Phase 1 檢查點開始 Phase 2 訓練
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v2 \
    --checkpoint logs/sb3/Isaac-Navigation-Charge-SB3-v1/.../model.zip \
    --agent sb3_cfg_entry_point
```

### Phase 2 → Phase 3 遷移

```bash
# 1. 先訓練 Phase 2
./train_sb3.sh v2

# 2. 找到 Phase 2 的最佳檢查點

# 3. 從 Phase 2 檢查點開始 Phase 3 訓練
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v3 \
    --checkpoint logs/sb3/Isaac-Navigation-Charge-SB3-v2/.../model.zip \
    --agent sb3_cfg_entry_point
```

**注意**：遷移學習時，確保觀測空間和動作空間在不同階段之間保持一致。

---

## 常見問題 FAQ

### Q1: 為什麼 Phase 2 的訓練步數比 Phase 1 少？

**A**: 雖然 Phase 2 的步數較少（30M vs 50M），但環境更複雜（5 個障礙物 vs 3 個），實際訓練難度更高。如果收斂不理想，可以增加到 50M 步。

### Q2: 為什麼 Phase 3 無障礙物但還是最難？

**A**: Phase 3 的難度在於目標距離可變（2-15 米），策略需要適應不同距離的導航，這比固定距離更複雜。

### Q3: 能否跳過 Phase 1 直接訓練 Phase 2？

**A**: 可以，但不建議。Phase 1 提供了基礎的導航和避障能力，從零開始訓練 Phase 2 會更困難且收斂更慢。

### Q4: 如何知道應該使用哪個配置？

**A**:
- 首次嘗試 → v1 (Phase 1)
- 完成 Phase 1 → v2 (Phase 2)
- 完成 Phase 2 → v3 (Phase 3)
- 快速測試 → v0（無障礙物）

### Q5: 能否自定義配置？

**A**: 可以直接修改 YAML 配置文件，或基於現有配置創建新變體。建議保留原始配置作為備份。

---

## 參考資料

- [Stable Baselines 3 官方文檔](https://stable-baselines3.readthedocs.io/)
- [PPO 算法詳細說明](https://arxiv.org/abs/1707.06347)
- [IsaacLab RL 指南](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
