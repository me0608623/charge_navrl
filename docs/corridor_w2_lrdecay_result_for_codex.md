# W2 LR 衰減臂結果（供 Codex 裁決）— 2026-07-27

依 07-27 裁決執行：①補 D0 三 seed Gate5 → ②W2（lr_decay=0.05, 20 iters, save every 2）→
③每顆 Gate5 + c10/c20 四模式與 Gate2。步驟 ④ 條件成立，seed 1 複驗已啟動。

## 步驟 1 — D0 起點確認：窄縫在 D0 就已失去

D0 三 seed Gate5（404/505/606, n=2927）：**SR 10.4% / CR 89.6% / crossing 10.4% / direct 0.000**。
→ W1-c10 的 36.3% crossing 是 12% 窄縫 replay「部分收復」而非新退化；direct 自 D0 起即 0。
report: `logs/gates/w2/d0/narrow_path/narrow_path_suite.json`

## 步驟 2 — W2 訓練（run `sa6_k8_obb_corridor_w2_wander_lrdecay_s42`, wandb `jc7u5tqm`）

- lock test 鎖單變因：W2 vs W1 允許差異僅 timesteps(20it)/save_interval(2)/lr_decay(0.05)+metadata
  （`test_sa6_sa8_replay_configs.py` 53 passed；12% 窄縫 replay、teacher KL 0.30、10% 走廊全鎖定繼承）
- **真實 LR 已驗證**：新增 wandb `train/lr_rl_actual` + supervisor `lr_rl_actual` 直讀
  `param_group["lr"]`；實測 2.0e-4 → 1.0e-5 完美線性（iter1 1.9e-4 … iter20 1.0e-5），
  trainer 於載入 Adam state 後重蓋 `initial_lr`，無「看似降 LR 實際沒降」問題
- 訓練面：KL 0.0064→0.0004 隨 LR 同縮；entropy 4.15→4.22 持平；vf 穩 ~5 不上升；SR ~95%；TO 全程 0

## 步驟 3a — Gate5 screening（每顆 ckpt, seed 404, n≈700–900/顆）

| ckpt | SR | CR | crossing | direct |
|---|---|---|---|---|
| D0 | 10.4% | 89.6% | 10.4% | 0 |
| W1-c10 (對照) | 36.3% | 63.2% | 36.3% | 0 |
| c2 | 26.9% | 73.1% | 26.9% | 0 |
| c4 | 87.9% | 12.1% | 87.9% | 0 |
| **c6–c20 (8顆)** | **99.1–100%** | 0–0.9% | **99.1–100%** | **0** |

**iter10 崩塌治好了**：高原 c6→c20 不衰退（W1 同區間從峰值回落）。
**但 direct 全程 0**：pre_y_p95 卡 3.6–3.8m（門檻 1.0m）＝每次都先繞側 ~3.7m 才過縫，
D0/W1/W2 皆同 → LR decay 治漂移、不治繞行模式。直穿是行為模式問題，需另一種介入。

## 步驟 3b — c10/c20 完整閘

Joint（`logs/gates/w2/c{10,20}/gate2/joint_report.json`）：

| | Gate2 (3 seed held-out) | 擋路走廊 | Gate5 |
|---|---|---|---|
| c10 | **PASS** SR 91.9/CR 8.1/TO 0 | **PASS** SR 92.9/CR 7.1 | crossing 99.1 / direct 0 → FAIL |
| c20 | **PASS** SR 91.5/CR 8.5/TO 0 | **PASS** SR 94.1/CR 5.9 | crossing 100 / direct 0 → FAIL |

四模式（deployment_wander_v1, seeds 515/616/717；CR 越低越好）。
**D0 基準採正式乾淨基準（`deployment_wander_v1/d0_v5_affected`，07-27 裁決校正）**，
早前引用的 `d0/`（random_2d 21.9）為舊版量測、不再使用：

| CR | D0（正式基準） | W2-c10 | W2-c20 |
|---|---|---|---|
| lateral | 7.41 ✓ | 7.02 ✓ | 7.33 ✓ |
| longitudinal | 2.23 ✓ | 2.09 ✓ | 2.51 ✓ |
| random_2d | 29.85 ✗ | 28.88 ✗ | 31.33 ✗ |
| mixed_iid | 17.11 ✗ | 17.58 ✗ | **18.5** ✗ (3rd seed=818*) |

**正確判讀（07-27 裁決校正）**：c10 大致守住 D0 水位，c20 開始退化 —
**不是** W2 大幅改善 random_2d。W2 證明的是：LR 衰減能保住已學能力、避免
iter10 後策略走壞，但不會憑空創造新行為（LR 只控制每步走多遠，不決定行為方向）。

*c20 mixed_iid seed 717 觸發評測場景生成 fail-fast（`sample_conflict_free_layout` 20 次重抽
仍有 1 env 重疊，拒裝場景而 crash）— 決定性，重跑同 seed 無意義，以 seed 818 替代
（s515 19.1 / s616 17.5 / s818 18.9，三 seed 聚合 18.5%）。
**固定 seeds 正式定量（07-27 補跑 c10 s818 後，兩點共用 515/616/818）**：
c10 mixed CR **16.53%**（s515 16.56 / s616 17.85 / s818 15.18, n=3829）→
c20 **18.48%**（19.09 / 17.49 / 18.85, n=3955）＝ **+1.95pp 臂內漂移**；
對照官方 D0 17.11%：c10 略優於 D0、c20 略差於 D0。

## 判定與行動（依 07-27 Codex 二次裁決更新）

**W2 結論的正確 scope**：LR 衰減保住已學能力（Gate5 高原 c6–c20、Gate2/lateral/
longitudinal 全程守住），但 c10 只是守住 D0、c20 開始退化，且不創造新行為。

**破口不只一個 — 目前三個**：
1. Gate5 direct = 0%
2. random_2d CR ≈ 29%（D0 起就是這水位）
3. mixed_iid CR ≈ 18%

## ★ seed 複驗結果：s42 的 Gate5 高原**沒有複現**

`sa6_k8_obb_corridor_w2_wander_lrdecay_s1`（wandb `7hjrhkt1`，同 config 僅 seed 42→1；
途中抓到並修復 `--seed` 被 env `cfg.seed=42` 全域重 seed 洗掉的 bug，
`train_rnn_car_wdclip.py:2989` — 首次 s1 run `dj6uzpog` 因此作廢重跑。歷史
deterministic gate 的 seed 有效，受影響的是過去宣稱「不同訓練 seed」的 run）。

Gate5 crossing（seed 404，每顆 n≈450–700）：

| ckpt | c2 | c4 | c6 | c8 | c10 | c12 | c14 | c16 | c18 | c20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **s42** | 26.9 | 87.9 | **100** | **100** | **99.1** | **99.8** | **100** | **100** | **100** | **100** |
| **s1** | 49.5 | 36.7 | 55.2 | 70.5 | 74.4 | **94.3** | 79.9 | 61.9 | 68.7 | 72.2 |

（direct crossing 兩 seed 全程 0.000。）

**判讀**：s1 從未站上高原 — 在 c12 觸頂 94.3% 後於 LR 已降至 7e-5–1e-5 的後段
反覆震盪回落到 62–72%。因此 **「LR 衰減使最佳區間延長」不是可複現的臂性質，
而是 s42 的單 seed 現象**；先前基於 s42 的樂觀結論必須撤回。

**訓練面完全看不出這個震盪**（s1 全程 SR 94–96% / CR 4–5% / entropy 4.15–4.19 /
KL 隨 LR 縮至 0.0003 / vf 無爆）→ 訓練指標無法作為窄縫能力的代理，
per-checkpoint Gate5 硬閘是必要的。

### ★ seed 變異只集中在窄縫，一般導航跨 seed 穩定

補跑 s1 的 c10 完整閘（原本只有 Gate5 screening，缺與 s42 的對照）後：

| c10 指標 | s1 | s42 | 差 |
|---|---|---|---|
| Gate2 held-out | SR 93.1 / CR 6.8 **PASS** | 91.9 / 8.1 **PASS** | s1 略優 |
| 擋路走廊 | SR 93.2 / CR 6.8 **PASS** | 92.9 / 7.1 **PASS** | 0.3 pp |
| lateral CR | 6.5 ✓ | 7.0 ✓ | 0.5 pp |
| longitudinal CR | 2.1 ✓ | 2.1 ✓ | 0.0 pp |
| mixed_iid CR | 16.4 ✗ | 16.5 ✗ | 0.1 pp |
| random_2d CR | 30.4 ✗ | 28.9 ✗ | 1.5 pp |
| **Gate5 crossing** | **74.4** | **99.1** | **24.7 pp** |

一般導航的 seed 間差異全部 ≤1.5 pp，唯獨窄縫差近 25 pp。

### ★★ 裁決點 3 定案：臂內漂移不可複現 → 是 seed 變異，D0 anchor 不升級

補齊 s1 的 c20 完整閘後，兩顆 seed 的 c10→c20 臂內漂移逐科對照
（每科三 seed 515/616/818 聚合，n≈3.2k–4.2k）：

| 模式 | s1 c10→c20 | s42 c10→c20 | 方向 |
|---|---|---|---|
| lateral | 6.52 → 5.39（**−1.13**）| 7.02 → 7.33（+0.31）| ✗ 相反 |
| longitudinal | 2.14 → 3.44（+1.30）| 2.09 → 2.51（+0.42）| ✓ 同向 |
| **random_2d** | **30.38 → 28.94（−1.44）** | **28.88 → 31.33（+2.45）** | **✗ 相反** |
| **mixed_iid** | **16.44 → 15.69（−0.75）** | **16.53 → 18.48（+1.95）** | **✗ 相反** |
| 擋路走廊 | 6.8 → 5.8（−1.00）| 7.1 → 5.9（−1.20）| ✓ 同向 |

五科中三科方向相反，且**裁決點指名的兩科（random_2d、mixed_iid）都相反** —— s42
的「變差 ~2 pp」在 s1 上都變成「變好」。升級條件要求「複驗顯示 LR floor 後單調
漂移**且**行為退化超出 seed 變異」，兩項皆不成立。

**→ D0 KL anchor 不升級。** 同時 W2 的負面判讀（「c20 開始退化」）也是單 seed
現象，W2 的最終結論收斂為：**LR 衰減對此配方沒有可複現的影響，正反皆無。**

（方法論註記：擋路走廊是唯一兩顆 seed 同向且幅度接近的項目〔−1.0 / −1.2 pp〕，
說明這個對照方法本身能區分訊號與雜訊，不是所有項目都被判成雜訊。）

**直穿問題 protocol 執行結果（07-27，兩步皆已量測）**：

1. **Teacher 假說證實**：narrow teacher（`sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/
   checkpoint_1280.pt`，KL 0.30）三 seed Gate5（404/505/606, n=2094）＝
   **crossing 95.9% / direct 0.000 / pre_y_p95 3.86m** — teacher 自己就是繞行者，
   W2 的 teacher KL 一直在穩定蒸餾「先偏 3.9m 再回來」。
   report: `logs/gates/w2/teacher_narrow/narrow_path/narrow_path_suite.json`
2. **距離 sweep 全滅**（W2-c10, seed 404, play 新增 `--narrow_gap_start_distance`）：
   1.5m → crossing 100% / direct 0 / pre_y_p50 3.79m；
   2.0m → 100% / 0 / 3.88m；3.0m → 99.1% / 0 / 3.71m（與 screening 吻合）。
   離縫 1.5m 起步照樣先擺到 ±3.8m — 繞行是深植行為模式，與距離無關。
   → **分支 3（close→far curriculum）出局**。
3. **分支 4 已建成並通過驗證**：scripted/privileged 直穿 teacher
   （`rnn_car_wdclean/scripted_narrow_teacher.py`，7 個單元測試先 RED 後 GREEN；
   play 旗標 `--narrow_scripted_teacher`）。三 seed（404/505/606, **n=7104**）：

   | 指標 | scripted teacher | RL 血緣（含 SA5 teacher） | 直穿門檻 |
   |---|---|---|---|
   | direct_crossing | **1.000 / 1.000 / 1.000** | 0.000 | ≥0.99 |
   | pre_cross \|y\| p95 | **0.0096 m** | 3.71–3.93 m | ≤1.0 |
   | path_length_ratio p50 | **0.852** | 2.10–3.64 | ≤1.35 |
   | first_cross_time p50 | **4.2 s** | 14.6–29.6 s | ≤10 |
   | backtrack p95 | **0.000 m** | 1.15–4.57 m | ≤0.5 |

   （path ratio <1 因分母為全程 6 m 直線、穿越判定在牆面即成立。）
   結論：**窄縫場景物理上完全可直穿，繞行是學到的行為而非場景限制**。
   實作為兩段瞄準（過牆前瞄喉道後 0.6 m、過牆後瞄 goal）＋ heading P 控制，
   期望 (v,ω) 以與 privileged corridor teacher 相同的 decode 反查最近可達 bin。
   **未經新裁決不啟動任何蒸餾/訓練臂。**

## ★ 用戶發現（07-27）：窄縫 replay 的目標永遠與缺口共線

查證 `narrow_passage_bridge.py`：

```python
goal_x = barrier_x + direction * start_goal_distance   # 3.0，固定不隨機
goal[:, 1] = origins[:, 1] + gap_center                # y 完全等於缺口中心
start_y   = gap_center + uniform(±0.19)                # 起點有橫偏，目標沒有
```

且 `maintain_narrow_passage_goal()` 每步把目標**重新釘回**該點（連 SA5 的移動目標
機制都無法移動它）。對照六個已隨機化的維度（缺口 y ±1.0 / 牆 x ±0.5 / 方向各半 /
起始橫偏 ±0.19 / yaw ±4° / 寬 1.2–1.4 m），**目標距離與橫向偏移是唯二完全沒有
隨機化的**。全 repo grep 確認無其他覆寫點。

**三個後果**：

1. **任務退化成直線跟隨**：目標在缺口正後方 → 「朝目標走」≡「朝缺口走」。
   policy 永遠不需要學會「先對準缺口穿過去、出去之後再轉向目標」——
   這兩件事在訓練分佈裡是同一件事。
2. **部署現實性缺口**：真實門/縫很少落在人與目的地的連線上；目標偏軸時直接
   朝目標開會撞牆。此能力既未訓練也未評測（Gate5 固定場同樣共線）。
3. **影響 N1**：scripted teacher 的兩段瞄準（過牆前瞄縫、過牆後瞄目標）在單元
   測試中有驗證偏軸情形，但**在實際訓練分佈中該切換永遠不觸發**（兩瞄準點共線）
   → N1 學到的會是「沿直線開」而非「找縫」。

**不能解釋現有繞行行為**：目標共線時「朝目標走」本來就是直穿，policy 仍繞路，
故繞行另有原因。此為獨立的分佈缺口。

**2026-07-27 裁決：採 B。** 在 N1 啟動前先補目標隨機化：
牆後距離 `U[2.0,4.0]m`、相對缺口橫向偏移 `U[-1.5,+1.5]m`。D0/W1/W2
維持歷史固定 `3.0m / 0m`，只有 N1 config 明確覆寫，因此不會追溯改寫既有結果。
scripted teacher 的偏軸極值閉迴路測試已補；正式 N1 前仍須跑新目標分布三 seed
teacher gate（direct≥99%、CR≤1%），再重新做 shadow λ 校準。

三 seed teacher gate 已完成：404/505/606 合計 6,706 回合，
`direct=6,706/6,706`、碰撞 0，first-cross p95 皆 4.2s，pre-cross lateral
p95 0.158/0.160/0.160m。teacher 前置閘通過。no-update shadow 的兩個有效
iteration 給 lambda50 0.03332/0.10024，pooled gradient norm 得 0.06669，
N1 已凍結為 `lambda=0.067`。修正前會執行 PPO step 的舊 row 明確排除；
N1 尚未啟訓。

## ★★★ 2026-07-27 晚：D0 新分佈基準量到推翻性矛盾

依 config notes 要求量 D0 在隨機化目標窄縫 replay 的零射基準，結果與既有認知衝突：

| | D0 @ 隨機窄縫 replay 場 | D0 @ 固定 Gate5 場 |
|---|---|---|
| seeds / episodes | 404/505/606, **n=6738** | 404/505/606, n=2927 |
| crossing | **99.97%** | 10.4% |
| **direct crossing** | **99.93%** | **0.000** |
| collision | **0.0%** | 89.6% |

已排除：非 teacher 代開（命令無 `--narrow_scripted_teacher`）、checkpoint 正確、
牆確實存在（LiDAR 最近 0.21–0.26 m，是貼牆穿縫而非穿模或繞過）。

**若成立，含意極大**：
1. 「全血緣 direct = 0」**只在固定 Gate5 場成立**，非模型在訓練分佈上的行為
2. 「SA5 teacher 自己也繞路」同樣是固定 Gate5 場的量測
3. **N1 要解決的問題可能不存在於訓練分佈中**
4. W1/W2 的 Gate5 高原與震盪，量的可能是「對一個 OOD 評測場的適應」

**兩場景已知差異**：stage 5 vs 6、arena 10 vs 14 m、牆來自內牆 slot（旋轉 90°）vs
專用 bridge 資產、單段牆長 3.9 vs 9.0 m、缺口寬 1.2 固定 vs 1.2–1.4 m、目標共線 vs 隨機。

### ★★★ 隔離實驗完成：根因 = `arena_size` 錯配

在同一 replay 場內逐步逼近 Gate5 條件，一次一變因（牆在 10/14 m 下皆封死，無污染）：

| 案例 | 變動 | n | crossing | direct | 碰撞 |
|---|---|---|---|---|---|
| A | 目標改共線（3.0 m / 0）| 2304 | 100% | **100%** | 0% |
| B | A + 縫寬固定 1.2 m | 2304 | 100% | **100%** | 0% |
| C | B + stage 5 + arena 10 | 242* | 31.4% | — | **68.6%** |
| E | B + stage 5 + arena 14 | 832* | 100% | — | **0%** |
| **F** | B + **stage 6 + arena 10** | **850** | **41.2%** | **0.000** | **58.8%** |

\* C/E 因外部 pkill 中斷為部分樣本；F 完整。

**2×2 交叉**：

| | arena 10 | arena 14 |
|---|---|---|
| stage 5 | 撞牆 68.6% | 0% 撞牆 |
| **stage 6** | **direct 0.000** | **direct 100%** |

→ **`stage` 完全無關；`arena_size` 是唯一決定因素。**
F 用訓練時的 stage 6、只把場地縮到 10 m，direct 從 100% 精確掉到 0.000，
與固定 Gate5 場的 0.000 一致。

**幾何機制**（用戶 GUI 目視確認）：`NarrowGapSpec.arena_half_extent` 寫死 5.0，
單段牆長 = 4.5 − 0.6 = 3.9 m，牆的 y 範圍 [0.6, 4.5]。
10 m 場外牆內側在 ±4.5 → 牆**剛好封死**，牆端與外牆形成**死角**；
14 m 場外牆內側在 ±6.5 → 牆端離外牆有 **2 m 空隙**。
用戶目視 10 m 場：「繞行往旁邊圍牆前進，撞到圍牆卡在牆端與外牆之間的死角」。

**污染案例作廢**：「固定 Gate5 + stage 6 + arena 14」曾得 direct 100%，但該組合牆未封死
（可繞過），測的不是穿縫能力。結論改依 replay 場的乾淨對照。

### 被推翻的結論

| 原本 | 修正後 |
|---|---|
| 全血緣 direct = 0 | 只在 **10 m 評測場**成立 |
| SA5 teacher 自己也繞路 | 同樣是 10 m 場的量測 |
| 距離 curriculum 無效 | 全部在 10 m 場量的 |
| 繞行是行為病理 | **不成立**，訓練場地下 direct ~100% |
| N1 要教模型直穿 | **前提不存在** |

### 仍未解、且比 Gate5 分數更重要

同一模型為何場地縮小就改變路徑選擇？14 m 不繞行、10 m 繞行撞死。
假設是 LiDAR 觀測到外圍牆變近觸發避障把它推離中線 —— **未驗證**。
若真實環境走廊比訓練場地窄，可能出現同樣行為 → **部署風險，需單獨查**。

### 後續建議

1. **N1 暫停** —— 前提不成立
2. **重估 Gate5 驗收協定** —— 10 m 是刻意壓力測試還是歷史錯配？
   所有用它做過的晉級判定都受影響
3. 查「場地縮小 → 繞行」是否為真實部署風險

**處置：N1 不啟動。**

## 待裁決（更新後）

1. **是否蒸餾 scripted 直穿 teacher**（取代/並存現行 SA5 narrow teacher KL 0.30）？
   若是，需要指定：權重、作用範圍（僅窄縫 replay 的 12% 還是全域）、
   以及是否先做 rollout override 的 A/B 而非直接 KL。
2. **s1 複驗推翻 s42 高原後，W2 血緣如何定位**：是否還值得保留 LR 衰減設定，
   或視為與 W1 同級（c10 守住 D0、之後漂移），改把資源全押在直穿蒸餾臂。
3. **D0 KL anchor 是否升級**：s1 顯示 LR floor 後仍震盪回落（62–94% 區間）、
   但 s42 無此現象；這算「replicated monotonic drift」還是「seed 變異」？
4. **窄縫目標共線（已裁決）**：採 B；N1 使用距離/橫偏隨機目標，歷史 config
   保持固定目標。
（「把 spawn 放正」已確認非有效實驗：訓練窄縫本來就正對缺口、goal 在另一端、牆不可繞。）
