# Critic 表徵塌縮 + 動作飽和修復計畫

## Context

訓練 run `rw_groundv8_openendedv1__seed1_nowalls_levle_setv1` 在 6.6M episodes 後出現兩個深層問題：
- **eRank Critic = 1**（vs Actor = 33.1）— 經分析為**量測 bug**（scalar output layer 的 SVD 永遠 rank=1），但 critic 可塑性喪失仍是真實隱患
- **Action saturation = 84.4%** — policy 輸出極端動作、entropy 極低(-0.146)，導致動作粗糙
- SR 卡在 72.8%，無法突破 80% 升級門檻

用戶參考論文提出：**根據 Policy 更新量動態限制 Critic 最大更新幅度**，防止異常環境產生的極大 critic loss 破壞已學好的參數。

## 修改總覽

| # | 修改 | 檔案 | 影響 |
|---|------|------|------|
| 1 | 修正 eRank 量測 bug | `training_health.py` | 純診斷，不影響訓練 |
| 2 | Critic head 加 LayerNorm | `vlp16_models.py` | 模型結構變更，需新 run |
| 3 | 動態 Critic 梯度限制器 | 新檔 `critic_grad_limiter.py` + `wandb_trainer.py` | 訓練行為變更 |
| 4 | Action 分布診斷指標 | `ablation_metrics.py` | 純診斷，不影響訓練 |
| 5 | CLI 參數 + checkpoint 兼容 | `train_charge_ac.py` | 基礎設施 |

## 修改 1: 修正 eRank 量測 bug

**檔案**: `scripts/reinforcement_learning/skrl/diagnostics/training_health.py` (~L231-255)

**問題**: `_compute_erank()` 取最後一層 `nn.Linear` 的 weight 做 SVD。Critic 最後一層是 `Linear(32, 1)`，weight shape `[1, 32]`，SVD 永遠只有 1 個奇異值 → eRank = exp(0) = 1，**與網路健康無關**。

**修正**: 當最後一層 `out_features == 1` 時，改用倒數第二層（`Linear(64, 32)`，weight `[32, 64]`，eRank 最高可達 32）。

```python
def _compute_erank(model):
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return 0.0
    target = linear_layers[-1]
    if target.weight.shape[0] == 1 and len(linear_layers) >= 2:
        target = linear_layers[-2]  # penultimate layer
    W = target.weight.data
    # ... SVD unchanged
```

## 修改 2: Critic Head 加 LayerNorm

**檔案**: `scripts/reinforcement_learning/skrl/vlp16_models.py` (~L253-258)

**現況**: Feature extractor 三個 branch 都有 LayerNorm，但 critic value head 沒有：
```
128D → Linear(128,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)
```

**改為** (LayerNorm 在 ReLU 後，與 feature extractor 一致)：
```
128D → Linear(128,64) → ReLU → LayerNorm(64)
     → Linear(64,32)  → ReLU → LayerNorm(32)
     → Linear(32,1)   [bias=-8.0 不變]
```

**Checkpoint 兼容**: 舊 checkpoint 的 Sequential index 會錯位。在 `train_charge_ac.py` 加 `strict=False` 載入 wrapper（修改 5）。

## 修改 3: 動態 Critic 梯度限制器（核心）

**新檔案**: `scripts/reinforcement_learning/skrl/diagnostics/critic_grad_limiter.py`

**演算法** (論文啟發):
```
每個 minibatch（unscale_ 後、clip_grad_norm_ 前）:
  1. 計算 actor_grad_norm, critic_grad_norm
  2. 更新 EMA: ema_actor = α * actor_gn + (1-α) * ema_actor
  3. 若 warmup 完成 且 critic_gn > k * ema_actor:
     → 對 critic 所有 param.grad 乘以 scale = (k * ema_actor) / critic_gn
```

**參數**:
- `k = 2.0` — critic 梯度上限 = 2 倍 EMA(actor 梯度)
- `α = 0.01` — EMA 衰減率
- `warmup = 100` minibatches（≈ 1 rollout: 6 epochs × 16 batches = 96）

**為什麼用 EMA 而非 raw per-batch**: actor 梯度逐 batch 波動大，EMA 提供穩定參考基線。

**為什麼 k=2.0**: 允許 critic 學得比 actor 快（PPO 正常現象），但防止 10x~100x 的異常突增破壞參數。

**插入點**: `wandb_trainer.py` 的 `_hooked_unscale()`，在 module_entropy 量測之後、`clip_grad_norm_` 之前。

**WandB 指標**:
- `train/critic_grad_clip_rate` — 被限制的 minibatch 比例
- `train/critic_grad_limit_ema_actor` — EMA actor 梯度 baseline
- `train/critic_grad_clip_max_ratio` — 被限制時最大的 critic/actor 比值

## 修改 4: Action 分布診斷指標

**檔案**: `scripts/reinforcement_learning/skrl/diagnostics/ablation_metrics.py`

**新增指標** (在 `step()` 中用 actions tensor [N, 2] 計算):

| WandB Key | 說明 |
|-----------|------|
| `action/saturation_rate` | 兩軸都在邊界 (0 或 18) 的比例 |
| `action/center_fraction` | 在中心 (9,9) = 零動作的比例 |
| `action/linear_entropy` | 線速度軸的經驗分布熵 (H_max = ln19 ≈ 2.94) |
| `action/angular_entropy` | 角速度軸的經驗分布熵 |
| `action/linear_mean` | 線速度軸平均 index (中心=9) |
| `action/angular_mean` | 角速度軸平均 index |

**實現**: 用 `torch.bincount` 累積 19-bin histogram，`get_and_reset()` 時計算 entropy。

**為什麼先診斷不先介入**: 不確定 saturation 是「policy 正確收斂到最優極端動作」還是「梯度問題導致卡在局部最優」。有數據後才能判斷是否需要 entropy scheduling。

## 修改 5: CLI 參數 + Checkpoint 兼容

**檔案**: `scripts/reinforcement_learning/skrl/train_charge_ac.py`

**新 CLI 參數**:
```python
--critic_grad_limit_k  float  default=0.0  # 0=disabled, 2.0=recommended
```

**Checkpoint 載入 wrapper**: 偵測架構不匹配時用 `strict=False` 載入，feature extractor（key 相容）正常載入，critic head（key 錯位）從零開始。印 warning 告知。

## 執行順序

**Phase A: 純診斷修正（不影響訓練行為）**
1. 修正 `_compute_erank()` — penultimate layer
2. 新增 action 分布指標 — saturation/entropy/center

**Phase B: 模型結構改進**
3. Critic head 加 LayerNorm
4. Checkpoint 兼容 wrapper

**Phase C: 動態梯度限制**
5. 建立 `critic_grad_limiter.py`
6. 整合到 `wandb_trainer.py` hook
7. 新增 CLI 參數

**Phase D: 驗證**
8. 短跑測試 (`--num_envs 4 --timesteps 10`)
9. 確認 WandB 出現新指標
10. 清 `__pycache__`

## 修改的檔案清單

| 檔案 | 動作 |
|------|------|
| `diagnostics/training_health.py` | 修改 `_compute_erank()` |
| `diagnostics/ablation_metrics.py` | 新增 action 分布指標 |
| `vlp16_models.py` | VLP16Value critic head + LayerNorm |
| `diagnostics/critic_grad_limiter.py` | **新建** — CriticGradLimiter class |
| `wandb_trainer.py` | 整合 limiter 到 hook + flush |
| `train_charge_ac.py` | CLI 參數 + checkpoint wrapper |

## 驗證

```bash
# 短跑測試
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --reward_mode navrl_ground_v8 --curriculum_version open_ended_v1 \
  --dynamic_safety_mode closing_risk --no_walls --use_cadn \
  --critic_grad_limit_k 2.0 \
  --num_envs 4 --headless --seed 1 --timesteps 10

# 確認:
# 1. diag/erank_critic > 1 (penultimate layer)
# 2. action/saturation_rate 出現在 WandB
# 3. train/critic_grad_clip_rate 出現在 WandB
# 4. 模型正常初始化（LayerNorm 在 critic head）
```
