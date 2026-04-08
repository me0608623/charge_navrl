# v24: Dense Dynamic Obstacle 避障改善（專注主線）

## Context

先前試圖修復 v23 Stage 6+ 退化 → 發現對照組建構錯誤、機制假設錯誤。使用者決定**不再糾結對照組**，直接回到主線問題：

> **Charge policy 在 dense dynamic scenario (19 static + 15 dynamic) 的 CR 卡在 49%，目標 CR < 25%。**

這是 v16/v17/v21 長期奮戰的核心問題，對照組實驗之後再補。

## 已知事實（從 memory `research_findings_dense_dynamic.md`）

**已嘗試且失敗的方案：**

| 方案 | 結果 | 失敗原因 |
|------|------|---------|
| Distance shield (M1.1) | CR 49→46% | 不看 obstacle velocity |
| Simplified VO single-worst (M1.2) | CR 49→44% | dense scene ping-pong: 躲 A 撞 B |
| sqrt severity + (1-s)² (M1.3) | CR 49→46% | 強化 evasion 不解 ping-pong |
| M1.4 ds reward boost | OOM 未測完 | hardware (v22 6144 envs crash) |

**NavRL 真相（advisor 三輪研究後）：**
- NavRL 用的是 **ORCA** (reciprocal VO)，不是簡化 VO
- NavRL 只在 **inference time** 用 shield，training 不用
- Diff-drive (charge) 不能直接用 ORCA，需 NH-ORCA projection
- NavRL paper CR 2.70→0.85 是 well-trained policy + ORCA shield，不等於 shield 單獨貢獻

**M1.2 simplified VO 數學上為何必敗：**
1. Single-worst-threat 退化成單 half-plane，失去 ORCA 凸集 guarantee
2. `(1-severity)²` 純時間函式，不告訴 policy「往哪轉」
3. head-on case 只剎車不側閃
4. sqrt(severity) 放大 TTC 雜訊，PPO 學「躲 shield」而非避障

**SOTA 文獻 Top 3 方向：**

| 優先 | 方案 | 預期 CR | 成本 |
|---|---|---|---|
| 🥇 **P0** | HEIGHT-style Heterogeneous Graph Transformer + Cross-Attention | 49→35-42% | 5-15 GPU hr, ~150 LOC |
| 🥈 P1 | Bounded Rationality Adversarial Curriculum | 49→38-45% | 15-30 GPU hr |
| 🥉 P2 | N-step Trajectory Prediction Augmentation | 49→42-46% | 1-3 GPU hr (最低成本) |

## 推薦：先做 P0（HEIGHT-style Attention Encoder）

**為什麼 P0 優先：**
- 直接攻擊 charge 架構最弱點：目前 obstacle branch 是 `ObsMLP + MaxPool(TopK)`，MaxPool **丟失 pairwise interaction** — 這正是 dense dynamic ping-pong 的根因
- 可以從 v21 best_agent **warm start**，省訓練時間
- 與 P2 正交，之後可以疊加
- 不需要動 reward / shield / curriculum，減少變因

**HEIGHT 論文核心 idea：**
- Paper: Liu et al., HEIGHT (T-ASE 2026, arXiv:2411.12150)
- 把 obstacle set encoding 從 MLP+Pool 改成 **Multi-Head Cross-Attention**：
  - Query = robot ego state
  - Key/Value = TopK obstacles (含 mask)
- 讓 robot 學會「以自己為中心，attend 到威脅最大的 obstacle」而不是丟進 pool 平均
- 保留 permutation invariance（attention is permutation equivariant）

## 需要實作的變更（初步規劃）

### 1. 新增 `AttentionObstacleEncoder` 模組
**檔案**：`scripts/reinforcement_learning/skrl/vlp16_models.py`

現有架構（需讀檔確認）：
- `Conv1d(LiDAR 72→64D)` + `ObsMLP(TopK 10×6D→32D) + MaxPool` + `StateMLP(ego+goal+time→32D)` = 128D

新架構：
- `Conv1d(LiDAR 72→64D)` + **`CrossAttention(Q=ego→32D, KV=TopK obstacles)→32D`** + `StateMLP(ego+goal+time→32D)` = 128D

Cross-attention 實作：
```python
class AttentionObstacleEncoder(nn.Module):
    def __init__(self, obs_dim=6, ego_dim=7, d_model=32, n_heads=4):
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, 32)
    def forward(self, obstacles, ego, mask):
        # obstacles: (B, K, 6), ego: (B, 7), mask: (B, K)
        kv = self.obs_proj(obstacles)          # (B, K, d_model)
        q = self.ego_proj(ego).unsqueeze(1)    # (B, 1, d_model)
        # key_padding_mask: True 表示 ignore
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=~mask)
        return self.out(attn_out.squeeze(1))
```

### 2. Warm start 從 v21 best_agent
- Load `logs/skrl/.../v21_best_agent.pt`
- **只覆蓋新的 obstacle branch 權重**（隨機初始化），其餘層保留
- 需要實作 partial state_dict load

### 3. 訓練指令
```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
  --num_envs 4096 --headless --seed 1 \
  --checkpoint logs/skrl/.../v21_best_agent.pt \
  --model_variant attention_obstacle \
  --run_name v24_height_attention_warmstart_s1_ne4096 \
  > /tmp/v24_attn.log 2>&1 < /dev/null & disown
```
（`--model_variant` flag 需新增到 argparse）

## 需要讀的檔案（執行階段再讀）

- `scripts/reinforcement_learning/skrl/vlp16_models.py` — 現有 3-branch network 實作位置
- `scripts/reinforcement_learning/skrl/train_charge_ac.py` — 找 model 建立的點，argparse 新增位置
- `scripts/reinforcement_learning/skrl/aac_wrapper.py` — 確認 obs 解碼（ego/obstacles/mask 的 index）
- `source/isaaclab_tasks/.../charge_skrl/mdp/observations/obs_functions.py` — 確認 TopK obstacles 的 6D 欄位順序與 mask 位置

## Verification

1. **Unit test** 新 encoder：餵隨機 (B=4, K=10, 6) + mask，確認輸出 shape (B, 32) 且對 obstacle 排列順序不變
2. **Sanity run** 64 envs × 2k steps，確認 loss 有下降、無 NaN
3. **正式 4096 envs**：
   - 30 min scan：Stage 1-2 SR 爬升正常
   - 1h scan：未退步於 v21 warm start 初始表現
   - 到 Stage 8 後做 eval：19s+15d 壓力測試，CR 目標 < 40%（vs v21 的 49%）
4. **消融**：若成功，再跑一個「attention + random init（無 warm start）」確認改善不只是 warm start 效應

## 風險與後備

- **風險 1**：attention 在 PPO + 4096 envs 下 forward 變慢，FPS 降 20-30%
  - 後備：減 n_heads 從 4 → 2，或 d_model 32 → 16
- **風險 2**：warm start 時新 attention layer 隨機，初期會破壞 v21 policy
  - 後備：前 10k steps freeze 其他層，只訓 attention；之後 unfreeze
- **風險 3**：改動後 obs layout 有變，scaler 需重建
  - 後備：用 fresh scaler 而非 resume v21 scaler

## 不做的事

- ❌ 不建 v23 對照組（用戶決定之後再說）
- ❌ 不碰 reward gate / shield / curriculum（Stage 6+ 退化留待架構改善後再看）
- ❌ 不做 P1（Bounded Rationality）或 P2（Trajectory Prediction），等 P0 結果再決定
