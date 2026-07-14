---
date: 2026-07-13
type: finding
status: high-confidence-hypothesis-not-ablated
related_obsidian: "/home/aa/Documents/Obsidian Vault/bug/2026-07-13_sa3_deploy_dense_二次崩塌_actor_critic訓練規則.md"
wandb:
  true_sa3: tzq22v0w
  diagnostic_dyn0: fflbtxgz
---

# Finding: SA3 deploy_dense 二次崩塌 = value-led；主嫌是訓練規則非網路結構

## 使用者意圖（請記住）

- 目標線：**dense 部署重訓**（deploy_dense），先 ① 穩銜接 SA3，再 ② 早避靜態。
- ② flag 已確認：`--penalty_speed_near_obs 0.2`（clearance-gated；先不要 teardrop）。
- **底座不穩時禁止加 ②**；不要用加稅來查崩因。
- 用戶問「為什麼一直崩」→ 要分清多種崩型，不要混為一談。
- 用戶問「是不是 actor/critic 設計問題」→ **網路結構大致 OK；可疑的是分數怎麼算 / 步子怎麼走**。
- 證據層級必須誠實：**高機率假說，未單變因坐實**，不能講成 100% 定罪。

## Runs

| Run | id | 設定 | 結果 |
|-----|-----|------|------|
| 診斷 | `fflbtxgz` | 6S+0D, near_goal 0~2, ent 0.04/0.08 | 熵死亡螺旋（ent→~0.05, SR~0.5） |
| 真 SA3 | `tzq22v0w` | 6S+1D deploy_dense | 高原 ~0.80 → step 8100 二次深崩 → 地板 ~0.30–0.34 |

Config: `wd_sa3_deploy_dense` / curriculum `v3e_deploy_dense` / resume SA2 ckpt_30000 / 79D LiDAR-only

## ① 穩銜接 — 正式判定：不通過

- 中段高原 step ~2700–7800 SR~0.80（可長達 ~17 點）**是真的**，但不能提早結案。
- step 8100 起第二次崩：vl/VE/return 先壞 → CR 升 → SR→~0.31；ent 仍高、KL 多數低。
- 通過門檻應為：高原夠長 + 無二次深崩 + **至少一顆 disk ckpt** + VE 未失效。
- 教訓：不要「穩 10 點就宣佈磐石」。

## 崩塌指紋（tzq）

```
return mean 17→1, VE 0.54→0.16, vl 先升, CR→0.5+, SR→0.3
ent 總仍 ~1.0–1.5；ent_linear→~0；ent_angular 仍高
retmin≈-12（penalty_hit=-12 生效）
```

類型：**I. Value-led 慢漂**（≠ II 熵死 fflb，≠ III KL 炸 07-11）

## 程式事實（主嫌）

`train_rnn_car_wdclip.py` + `wd_sa3_deploy_dense.yaml`：

- `normalize_return: true` → critic 目標每 batch 標準化
- `adv_norm_mode: mean_only` → A 只減 mean **不除 std**（偏離 SKRL/rsl_rl 預設 full）
- A2C、`mini_batches: 1`、弱 KL 煞車
- 進 SA3：curriculum `lr→5e-4`、`max_grad→1`、Adam 動量清零、`penalty_hit→-12`

**PolicyHead/ValueHead 架構、GAE 公式：正常。**  
**Asymmetric critic：不是錯。**

主流（SKRL PPO / rsl_rl）：`A = (A-mean)/(std+eps)` + mini-batch PPO + 常有 KL 控制。

## 建議下一步（未執行，等用戶）

1. 停 tzq（0.3x 無 ckpt 價值低）— 已建議
2. 單變因重開：**`adv_norm_mode=full`** 或 **LR 鎖 2e-4**（二選一）
3. 提早 `save_interval`（9k–15k）
4. 坐實後再 ② `--penalty_speed_near_obs 0.2`

## 完整長文

Obsidian vault：

`/home/aa/Documents/Obsidian Vault/bug/2026-07-13_sa3_deploy_dense_二次崩塌_actor_critic訓練規則.md`
