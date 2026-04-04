# v10: PPO KL 約束 + Reward Signal 修正

## Context

v6~v9b 的訓練全部在高難度 Stage 退化：
- KL 從 0.3 緩慢攀升到 1.4+（v9 爆到 18.4）
- SR 從 76% 峰值跌到 59%，Stage 8→7→6 降級
- adv_std 自然值 0.12-0.14，接近 floor=0.1 但未觸發
- fwd_ret_adv_gap ≈ 0（前進 vs 後退無 advantage 差異）
- timeout 40% 是主要失敗模式

根本原因：
1. **PPO 無 KL 約束** — `kl_threshold=0` 且 loss 無 KL penalty，policy drift 無限制
2. **Reward soft gate floor 壓縮 advantage** — danger zone 仍有 20-30% forward reward，安全/前進信號對比太弱

## 修改方案（3 個改動）

### Change 1: SKRL PPO 加入 KL Penalty（conda patch）

**檔案:** `skrl/agents/torch/ppo/ppo.py`（conda env 內）

**(a)** Line 145（`self._kl_threshold = ...` 之後）加入：
```python
self._kl_coeff = self.cfg.get("kl_coeff", 0.0)
```

**(b)** Line 500，替換：
```python
self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
```
為：
```python
# v10: differentiable KL penalty — 防止 policy drift
if self._kl_coeff > 0:
    kl_loss = self._kl_coeff * ((ratio - 1) - torch.log(ratio + 1e-8)).mean()
else:
    kl_loss = 0
self.scaler.scale(policy_loss + entropy_loss + value_loss + kl_loss).backward()
```

注意：line 481 的 `ratio = torch.exp(next_log_prob - sampled_log_prob)` 有梯度，可直接用。

### Change 2: 移除 Reward Soft Gate Floor

**檔案:** `charge_env_cfg_vlp16_curriculum.py`

`RewardsCfgVLP16NavRLGround` 中：
- Line 718: `"gate_beta": 0.2` → `"gate_beta": 0.0`
- Line 733: `"scale_gamma": 0.3` → `"scale_gamma": 0.0`

效果：danger zone（d_safe < dmin）不再給 forward reward，安全 reward 有清晰的 advantage 優勢。Gate 仍是平滑 ramp（dmin→dmax），無跳變。

### Change 3: YAML 加入 kl_coeff

**檔案:** `skrl_ppo_cfg_vlp16.yaml`

Line 101（`kl_threshold: 0.0` 之後）加入：
```yaml
kl_coeff: 0.3
```

同時更新 yaml 頂部版本註釋。

## 驗證標準

| 指標 | v9b | v10 目標 |
|------|-----|---------|
| KL (Stage 8+) | 0.3→1.4+ | < 0.3 穩定 |
| SR (Stage 8) | 76%→59% | > 70% 穩定不降 |
| fwd_ret_adv_gap | ≈0 | > 0.05 |
| Stage 軌跡 | 8→7→6 | 穩定 8 |
| adv_std | 0.12-0.14 | > 0.15 |

## 風險

- `kl_coeff=0.3` 太高：SR plateau 過早 → 降到 0.1
- `gate_beta=0` 導致過度停滯：不太可能（safety reward 已鼓勵繞行）→ 若觀察到可改 0.05

## 執行順序

1. Patch SKRL ppo.py（Change 1）
2. 改 reward config（Change 2）
3. 改 YAML（Change 3）
4. 啟動訓練，30 分鐘後第一次 /scan
