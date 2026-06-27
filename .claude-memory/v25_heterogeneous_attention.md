---
name: v25 Heterogeneous Attention — HEIGHT full implementation
description: v25 修復 v24 HEIGHT attention 不完整性。v24 只 attention 動態，靜態仍 Conv1d+MaxPool(1) 丟失角度資訊。v25 把 LiDAR 18 spatial tokens 當 static nodes 統一 cross-attention，加 angular pos + type emb。
type: project
---

## Why v25

v24 HEIGHT attention 跑 ~9.5h 停在 Stage 7/8 (6s+6d)，SR 62.5% / CR 37.5%。rl-research-advisor 分析結論：

1. **Static MaxPool 瓶頸**（架構問題）：v24 的 LiDAR 分支是 `Conv1d → AdaptiveMaxPool(1) → 64D`，把 18 個 spatial tokens 壓成 channel-wise scalar，**完全丟失「哪個角度」**。對應訓練實測 `collision_static_ratio=0.61`（六成碰撞是撞靜態）+ `lidar_min_mean=0.41m`（長期貼牆）。v24 的 attention 只作用在動態 60D 分支，靜態側一點都沒改善。

2. **Reward 失衡**（更底層，但先不動）：`reaching_goal=0.80` vs `collision_terminal=-0.045` (5.6%)，advantage collapse 導致 KL 0.27→0.47 發散、policy_loss≈0。但為了不 confound 架構驗證，**v25 第一輪只改架構，reward 保持不變**。如果 v25 無突破 75%，第二輪再動 reward。

## v25 Architecture（已在 vlp16_models.py 實作）

新增兩個類別：

### `HeterogeneousAttentionEncoder`

```
Input: lidar[B,72], dynamic[B,10,6], ego[B,7]

Static tokens:
  lidar → Conv1d(1→32→64→64) → [B, 64, 18]   ← NO final MaxPool
  transpose → [B, 18, 64]
  Linear(64→96) → static_tok
  + angular positional encoding (sin/cos of 18 angles, each 20°) via Linear(2→96)
  + type_emb(static=0)

Dynamic tokens:
  Linear(6→96) → dyn_tok
  + type_emb(dynamic=1)

KV = concat([static_tok, dyn_tok], dim=1) → [B, 28, 96]
key_padding_mask: static 全 False，dynamic 依 mask 欄位
(static 永遠至少 18 個 valid → 永遠不會 all-masked → 無 NaN 風險)

Query: ego Linear(7→96).unsqueeze(1) → [B, 1, 96]
MultiheadAttention(d=96, h=4) → [B, 1, 96]
LayerNorm → [B, 96]
```

### `VLP16HeterogeneousExtractor`

```
encoder → [B, 96]
state_mlp(7D ego+goal+time → 32) + LayerNorm → [B, 32]
concat → [B, 128]  ← 保持與 v24 相同 output_dim，head 層不變
```

### Policy/Value 類別

- `VLP16HeterogeneousPolicy` (MultiCategoricalMixin, head 128→128→38)
- `VLP16HeterogeneousValue` (Deterministic, 128→64→32→1, orthogonal init gain=0.01 bias=-8)

## Parameter count
encoder: ~70k params (vs v24 attention encoder ~6k)

## Sanity checks (已通過)
- forward shape [B,128] ✓
- dynamic obstacle permutation invariance (diff ~7e-7, numerical noise) ✓
- 無 NaN 路徑（static tokens 永遠至少一個 valid）✓

## CLI 使用

新增 `--model_variant heterogeneous` 選項。train_charge_ac.py 1015 行的 override block 改成同時支援 `attention` 和 `heterogeneous`。

**不使用 warm-start**：v25 的 extractor 結構完全不同（d_model 32→96, lidar tokens 從 pooled 改成 spatial），v24 checkpoint 不相容。從頭訓練。

## 訓練指令

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 4096 --headless --seed 1 \
  --model_variant heterogeneous \
  --run_name v25_hetero_attn_s1_ne4096 \
  > /tmp/v25_hetero.log 2>&1 < /dev/null & disown
```

## 預期 vs 退出條件

- **預期**：Stage 7 ceiling 從 62% → >70%, collision_static_ratio 0.61 → <0.4, dynamic_sr 維持或上升
- **失敗條件**：跑 10h 仍 <68% → 證明 reward 才是主因，進入第二輪（加重 collision penalty 2-3×）
- **Early stop**：KL 若持續 >0.5 → 可能架構改動引入不穩定，考慮降 LR

## 相關記錄

- [v24 失敗分析](v24_height_attention_dead_end.md)（待寫）
- [research_findings_dense_dynamic.md](research_findings_dense_dynamic.md)
