# random_2d 方向決策請求：機制稽核全部完成，pause 假設出局，wander 基準出爐

**日期**：2026-07-26（台北）
**產出者**：監督端
**性質**：**決策請求。** 你核准的實驗序列（pause A/B → counterfactual → teacher shadow/override）已全部執行完畢，外加用戶指定的 wander 口徑實作與基準。以下是完整數據與待你裁決的方向問題。
**完整細節**：`docs/corridor_d1_sweep_result_for_codex.md` §10–11。本文件只做決策所需的濃縮。

---

## 1. 你核准的實驗序列：結果總覽

全部在 D0（`sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt`）× random_2d × seeds 515/616/717 執行。無訓練、無 reward/網路/取樣改動、真錯誤 0。

### 1.1 Pause A/B —— 你的門檻三條全不過，且方向相反

| | default (0-5) | zero (0,0) | Δ |
|---|---|---|---|
| CR | 33.79% | **39.18%** | **+5.39pp（三 seed 全惡化：+8.52/+5.63/+1.93）** |
| SR | 63.69% | 56.22% | −7.47pp |
| TO | 2.53% | 4.61% | +2.08pp |

**pause blindness 不是主因**——移除暫停反而更差。依你的規則：未達門檻 → 未測 c10 pause。
平台可信度：default 臂 CR 33.79% 與歷史 D0 逐位相符；zero 臂 JSON 實測 `pause_steps_range=[0,0]`。

### 1.2 Phase audit —— 危險不在暫停，在反向前夕

| phase | 幀佔比 | 碰撞佔比 | enrichment |
|---|---|---|---|
| paused | 16.55% | 12.23% | 0.739（**相對安全**） |
| post_switch_1s | 30.29% | 20.33% | 0.671 |
| **pre_waypoint_1s** | 14.79% | 30.74% | **2.079（唯一 >1）** |
| steady | 38.37% | 36.69% | 0.956 |

幾何假象已排除（碰撞幀 robot 距離跨 phase 一致 0.70–0.72m）。zero 臂重現同形態。
**機制修正**：等速外推的失效點是「即將反向」而非「靜止隱形」。zero 更差的原因自洽：移除 dwell 使高危 pre_waypoint 幀 +20.2%。

### 1.3 Counterfactual —— 單步反事實下碰撞難逆轉

碰撞當下僅 3.4–6.6% 有安全替代動作；最佳提前量 1.2s 也只有 10.6% 可逃。self-check 全 0。
⚠️ 單步口徑，不可推論「場景不可解」。

### 1.4 Teacher —— 無蒸餾資格，且不可作為可解性證據

| | shadow | **override（實駕，與 shadow 分開跑）** |
|---|---|---|
| feasible | 74.4–75.3% | — |
| SR / CR / TO | — | **16.97% / 50.18% / 32.85%** |

門檻 SR≥90/CR≤10/TO≤5 三項全不過 → **依你的規則，蒸餾出局**。
teacher 比 policy 差 46.7pp（SR），其 25% infeasible 反映自身弱點（2s horizon、禁倒車、單步貪婪、brake 27%/stop 8% → TO 33%），**不是環境可解性下界**。監督端先前「可能不可解」的傾向已撤回。

---

## 2. 用戶決策改變了問題本身：wander 口徑

**用戶裁示（07-26 晚，原話大意）**：「目前部署不會發生突然轉身。我需要的是訓練障礙物在 4m 寬走廊上亂走，不會固定距離後折返，是真的不斷亂走。」

這直接挑戰 patrol 口徑的部署效度：random_2d 的 ~1.5m 乒乓折返（每 3.3s 一次，恰落在 1.5s future-occupancy horizon 內）是測試設計產物，非部署分佈。§1 整條調查鏈也收斂到同一點——**所有失效機制都掛在「固定點急折返」上**。

### 2.1 已實作（預設不變，歷史 gate 完好）

- `random_2d_kinematics ∈ {patrol(預設), wander}`，plumbing：geometry → replay → **event.params** → play/suite CLI → JSON 記錄
- wander = 有界隨機遊走：每 1–5s 隨機偏轉 ≤90°、平滑 0.6s、**永不停頓、無折返點**、全走廊活動（~3.1×9.1m）、牆面鏡面反彈
- 順帶修復既有 bug：`step_random_walk` 宣稱的邊界反彈從未實作（無界行為逐位不變，測試釘死）
- **實作過程抓到並修復 wiring bug**：首版開關未進 event.params，auto-reset 後被預設覆寫回 patrol，產出假 wander 數據；由 paused 幀佔比 16%（應為 0）+ CR≈patrol 交叉抓到。已修 + 回歸測試鎖死。教訓：300 步煙測驗不出 auto-reset 路徑。
- 測試 **189 passed**（含 2000 步逐步圍堵斷言、event.params 回歸鎖）

### 2.2 D0 × wander 三 seed 基準（修復後，paused=0 全程驗證）

| seed | SR | CR | TO |
|---|---|---|---|
| 515 | 79.36% | 20.64% | 0.00% |
| 616 | 76.22% | 23.78% | 0.00% |
| 717 | 78.67% | 21.33% | 0.00% |
| **合計 (ep=4203)** | **78.06%** | **21.94%** | **0.00%** |

**vs patrol 同 ckpt 同 seeds：CR −11.85pp / SR +14.37pp / TO −2.53pp。零訓練即改善近 12pp。**
但距 CR≤10% 門檻仍 ~12pp（D0 未在 wander 場景訓練過的裸考）。

---

## 3. 請你裁決的問題

### Q1：random_2d 的 gate 口徑是否改為 wander？

- **支持**：用戶明示部署分佈；patrol 的失效機制（反向前夕 enrichment 2.08、45% 預測跨反向點）是測試設計產物；wander 下 TO 歸零、三 seed 全距僅 3.1pp。
- **反對/保留**：換口徑=換考題，D0/D1 全部歷史數字失去直接可比性（可保留 patrol 為 legacy regression，如 mixed/mixed_iid 的先例）；wander 的「難度」尚無多 checkpoint 曲線。
- **附帶**：mixed 內的 random_2d 成分是否同步換？（目前 mixed 仍用 patrol 腳）

### Q2：是否啟動 wander 訓練臂？

- 目標：21.94% → ≤10%。訓練端注入已就緒（config 加 `random_2d_kinematics="wander"` 一個參數，走既有 event.params 路徑）。
- 起點候選：D0 checkpoint_3840（主基準）。
- D1 的教訓在此適用：加權集中取樣兩邊都輸；wander 臂建議維持 1:1:1 env-stratified 不動配比，單變因只換 kinematics。
- **監督端不啟動任何訓練，等你的配方。**

### Q3：patrol 口徑的遺留問題還追不追？

若 Q1 採 wander，則「patrol random_2d 為何學不動」降為學術問題。未解清單：曝光無反應機制、c10 後全面劣化成因（需對照臂）、`random+random` vs pure gate 的 2σ 差異、mixed 平衡發牌偏差處置（`mixed_iid` 已備妥未預設）。建議全部凍結，除非 Q1 決定留 patrol。

### Q4：future-occupancy reward 是否需要動？

pause A/B 已排除「靜止盲區」為主因，但「反向前夕外推錯誤」機制在 wander 下**理論上減弱而非消失**（wander 仍每 1–5s 改向，只是幅度 ≤90° 非 180°）。建議：**先不動 reward**（本輪禁令延續），等 wander 訓練臂的 gate 結果說話；若 wander 訓練後仍卡，再考慮 horizon 縮短 A/B（1.5s→0.8s，eval-only 先行）。

---

## 4. 監督端立場

- 傾向 **Q1 採 wander 為部署口徑、patrol 降為 legacy**（用戶部署判斷 + 整條機制調查都指向 patrol 口徑的人工性）
- 傾向 **Q2 啟動單變因 wander 臂**（1:1:1 配比、D0 起點、30 iter 同預算，沿用四閘 + 21.15% 類停損邏輯換算 wander 基準）
- Q3 凍結、Q4 先不動
- **決策權在你。** 工作區 9 M + 2 新檔（phase audit / pause 旋鈕 / mixed_iid / wander）已過你上輪四項 review 修正 + 189 tests，未 commit，等你對本文件的裁決一併處置。

---

## 5. W1 執行結果（2026-07-27 凌晨，過夜協定完整走完）

**run**：`sa6_k8_obb_corridor_w1_wander_s42`（wandb `tiaqc2dr`），30/30 iter，零 NaN/OOM/Traceback。
**凍結基準（D0 × deployment_wander_v1，物理乾淨場景）**：lateral 7.41 / longitudinal 2.23 / random_2d 29.84 / mixed_iid 17.08（v4+v5，12/12 結構閘過）。

### Checkpoint 對照表（CR%；c5/c15/c25 為 s616 sentinel，餘為三 seed）

| ckpt | lateral | longitudinal | random_2d | mixed_iid | Gate2 |
|---|---|---|---|---|---|
| D0 基準 | 7.41 | 2.23 | 29.84 | 17.08 | — |
| c5ˢ | 7.48 | — | 26.48 | 15.88 | — |
| **c10** | **5.53** | **1.41** | **27.91** | **15.49** | **7.00 PASS** |
| c15ˢ | 8.24 | — | 31.55 | 16.54 | — |
| c20 | 9.45 | 4.20 | 32.10 | 21.16 | — |
| c25ˢ | 12.15 | — | 34.73 | 25.93 | — |
| c30 | 7.27 | 5.96 | 32.02 | 19.21 | 8.49 PASS |

全程結構閘（ds=0 / not_ready=0 / audited=153600）12 顆次全過；動-動互穿依裁決僅記錄。

### 結論

1. **無 JOINT PASS**（random_2d/mixed_iid 距 10% 門檻仍 17.9/5.5pp）。依協定不延長訓練，只報告。
2. **最佳 checkpoint = c10**：唯一一顆四模式**全部優於 D0 基準**（史上第一次，D1 從未做到）+ Gate2 PASS。
3. **c10 峰值後回落，與 D1 形態一致**：兩個完全不同的配方（加權取樣 / 換運動學）從同一 D0 warm-start 都在 ~10 iter 觸頂後退化。**指向配方無關的共同機制**（warm-start 後過擬合窗口？）——這可能是比配方選擇更根本的問題。
4. wander 方向有效但不足：c10 的 random_2d 改善 −1.93pp，方向正確、量級不夠。30 iter 單變因不能填 20pp 缺口。

### 交決策

- **W1-c10 是否取代 D0 為主基準**（四模式全優 + Gate2 過，但需你的 CI 檢驗——D1 時代 c10 曾因 CI 全跨 0 被拒）
- 下一步方向：追「iter 10 過擬合窗口」機制（比再堆配方更根本）？或接受 c10 增量、繼續短臂迭代？
