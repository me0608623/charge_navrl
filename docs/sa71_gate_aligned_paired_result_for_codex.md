# SA7.1 gate-aligned replay — 成對對照結果（給 Codex 裁決）

**日期**：2026-07-28（因果範圍已依裁決收斂，見 §0）

## 0. 結論的精確範圍 ⚠️ 先讀這節

**已證明**：

> 「SA7.1 完整配方」在 c25、c30 兩顆 checkpoint 上都比 Control 差，
> 且差距在後期擴大（四模式 Δ 總和 −5.34 → −13.34）。

**尚未證明**（不得宣稱）：

> ✗ 傷害來自「加入 Gate 題型」
> ✗ 傷害來自「native 減少 10%」

**理由**：這兩者在 SA7.1 裡是**同時被改動的**。走廊從 10% 擴到 20%（半數是 Gate 題型），
native 由 68% 降到 58% —— 一次實驗無法分離兩個同時變動的因子。要分離需要第三臂
（例如 native 58% 但那 10% 仍用真實混合，不放 Gate 題型）。

**訓練 SR 兩臂之間也不可比較** —— 兩臂的訓練場景難度組合不同（Control native 68%；
Treatment native 58% + Gate 題型 10%），分母不同的比率相減不對應任何有意義的量。
本報告中的訓練 SR 僅作**健康度監控**。

以下 §3.1 對 random_2d 的討論是**假說層級的解讀**，不是已證明的因果。

---

## 1. 實驗設計

依 2026-07-28 Codex 裁決，crossing 幾何是 source-level 改動（固定交點 →
每回合隨機交點），dataclass config lock 抓不到，因此 SA7.1 不能與歷史 SA7
直接比較，必須跑**成對對照**。

| 臂 | config | 走廊配方 |
|---|---|---|
| **Control** | `e2e_sa7_wander_from_w1c10` | native 68% / 走廊 10% 真實混合 |
| **Treatment** | `e2e_sa71_gate_aligned_from_w1c10` | native 58% / 走廊 10% 混合 + **10% Gate 題型** |

**共用**（唯一變因就是上面那一項）：

* 暖啟動 checkpoint：W1-c10 `sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt`
* seed 42、1024 env、30 iterations、每 5 iter 存檔
* 窄縫 12% / 前階段 replay 10%（兩臂皆不動）
* reward / network / LR / gate 門檻：完全不動
* source hash：`docs/freeze/sa71_paired_source_freeze.md`（含新版隨機 crossing）

**成對性驗證**：兩臂 iter 1 的訓練 SR 為 64.1% vs 63.7%（差 0.4pp）、
CR 12.1% vs 11.9%（差 0.2pp）。起點重合，因果歸屬乾淨。

### Treatment 機制驗證（1024 env 實測）

```
[SCENE-MIX] native=0.580 sa5_general=0.100 narrow=0.120 long_corridor=0.200
gate_aligned=lateral 26, longitudinal 26, random_2d 26, mixed_iid 25   mix_envs=92
```

Gate 四模式輪派差 ≤1，Gate:mix = 52.8%（目標 50%），count-mix 密度分佈
`{3S1D 0.25, 4S2D 0.35, 4S3D 0.20, 5S3D 0.15, 5S5D 0.05}` 與凍結表逐項相等。

---

## 2. 結果：c30（訓練最久的 checkpoint）

評測 profile `deployment_wander_v1`，四模式 × seeds 515/616/717，64 env × 1200 步。

| mode | Control SR | Treatment SR | ΔSR | Control Gate | Treatment Gate |
|---|---|---|---|---|---|
| lateral | 91.97% | **85.05%** | **−6.91** | **PASS** | **FAIL** ← 掉閘 |
| longitudinal | 90.38% | **95.12%** | **+4.74** | PASS | PASS |
| mixed_iid | 82.43% | **76.05%** | **−6.38** | FAIL | FAIL |
| random_2d | 69.18% | **64.40%** | **−4.78** | FAIL | FAIL |
| **Δ 總和** | | | **−13.34 pp** | **2 過 2 敗** | **1 過 3 敗** |

樣本數：Control n=14940、Treatment n=16643（四模式各 3 seed）。
兩臂皆 `ALL=FAIL JOINT=FAIL`。

**判定尺規**：Control 單臂 seed 全距 lateral 2.3pp、longitudinal 3.1pp。
上述四個 Δ 中有三個超過 4.7pp，皆在雜訊帶之外。

---

## 3. 三個結論

### 3.1 「補 Gate 曝光就會過」的假說**未獲支持**（假說層級，非因果證明）

random_2d 原本是**最有理由期待受益**的模式：曝光審計（n=60001）顯示訓練分佈裡
random_2d **從不參與配對**（0% 的 pair 含 random 家族），而 gate-aligned 的
四分之一正是純 random_2d 題型。

實測 **−4.78 pp**（3 seed，n=4845）—— 沒有受益。

**能說的**：整包 SA7.1 配方沒有讓 random_2d 變好。
**不能說的**：「因為曝光不是瓶頸」。SA7.1 同時砍了 native，兩個因子混在一起，
本實驗無法歸因。「能力缺口 vs 曝光不足」目前是**待驗證的假說**，不是結論。

補充事實（見 §3.4）：正式 Gate 裡根本沒有配對互動，所以 pair 相關的推論
在 Gate 上不適用。

### 3.2 四模式 Δ 總和為負且隨訓練擴大

| | Δ @ c25 | Δ @ c30 | 位移 |
|---|---|---|---|
| lateral | −3.75 | −6.91 | −3.16 |
| longitudinal | +6.96 | +4.74 | −2.22 |
| mixed_iid | −4.67 | −6.38 | −1.71 |
| random_2d | −3.88 | −4.78 | −0.90 |
| **總和** | **−5.34** | **−13.34** | **−8.00** |

四模式位移全為負。**這證明的是「整包配方越訓越差」，不是「Gate 題型有害」。**

### 3.3 在 min 判準下，這種重分配對晉級不利

唯一提升的 longitudinal 本來就已 PASS（90.4 → 95.1，對晉級零貢獻），而下降
的三項中有一項把 PASS 打成 FAIL。晉級取最弱項，「強項更強、弱項更弱」對
min 判準不利。（同樣不歸因到單一因子。）

### 3.4 附帶發現：正式 Gate 沒有配對互動

`crossing` / `side_by_side` 只存在於 count-mix 路徑
（`_sample_mixed_density_layout` / `_install_mixed_density_obstacles`）。
兩個 gate runner 完全不帶 `count_mix` 與 `interaction`。

| | 有配對互動比例 |
|---|---|
| 訓練場景 | 24.56% |
| **正式 Gate** | **0%** |

→ **碰撞分析不得套用 crossing / side_by_side 分類**（Gate 上不存在該標籤）。
random_2d Gate 是兩個各自獨立的二維隨機行人，應以相對運動描述。

---

## 4. 附帶結論：Control 臂本身也在退步

Control = 原 SA7 配方再訓 30 iterations，四模式 CR 全數高於 W1-c10 起點：

| mode | W1-c10 CR | Control c30 CR | Δ |
|---|---|---|---|
| lateral | 5.53% | 8.03% | +2.50 |
| longitudinal | 1.41% | 9.62% | +8.21 |
| mixed_iid | 15.49% | 17.57% | +2.08 |
| random_2d | 27.91% | 30.82% | +2.91 |

四項同向（雜訊假設下機率約 1/16）。與「SA7 r3 跑滿 300 iterations 產不出優於
起點的 checkpoint」一致 —— 30 iterations 只是把同一現象壓縮呈現。

注意：此為跨 profile 比較（W1-c10 基準量測環境不同），僅作跡象，不作定案。
定案的是 §2 的同 profile 成對數字。

訓練 SR 曲線（**僅供健康度監控，不可作判準** —— 兩臂分母是不同的場景組合）：

| iter | Control SR/CR | Treatment SR/CR |
|---|---|---|
| 1 | 64.1 / 12.1 | 63.7 / 11.9 |
| 10 | 86.4 / 13.7 | 85.2 / 14.8 |
| 20 | 86.4 / 13.5 | 84.5 / 15.4 |
| 30 | 85.5 / 14.4 | 82.8 / 18.0 |

兩臂後段皆回落，是共有現象，非 gate-aligned 造成。

---

## 5. 過程中修掉的一個真 bug（0S0D 帳本脫節）

64-env smoke 的首發審計印出 `density_mix_realized={'0S0D': 4, ...}` ——
一個不存在的密度組合。根因：Gate 題型走 legacy install 不經過 count-mix
取樣器，`_long_corridor_density_counts` 留在 `(0,0)`。

這與先前「56 個非走廊 env 被算成 0S0D」是**同型缺陷的第二次發作**：帳本欄位
預設值 0，而 0 剛好是合法值域內的數字，讀取端無法區分「沒寫入」與「寫入了 0」。

修復三處：① Gate env 寫入真實 `(4,2)`；② count-mix 審計母體改用 `mix_selected`
（Gate 是固定密度，混進去會灌爆 4S2D，使「取樣器是否重現凍結表」失去意義）；
③ 審計行加印 `gate_aligned=` 逐模式計數。新增 `test_gate_envs_record_their_
real_density_not_zero`。

測試現況：events 75 + config lock 64 = **139 項全綠**。

---

## 6. 執行計畫（2026-07-28 裁決）

正式三-seed Gate 掃描**已於 12:56 停止**（未跑完的 `ctrl_c2560` 部分結果已刪除）。
依序執行：

1. **c5 / c10 / c15 / c20 兩臂單-seed 四模式 screening**（32 次 rollout）——
   便宜地確認早期是否存在更好的點，避免只看 c25/c30 就下結論。
2. **升級規則（2026-07-28 修正）**：
   - **任一模式 < 85% → 該 checkpoint 直接淘汰**
   - **四模式全部 ≥ 85% → 才升級正式三-seed Gate**

   修正理由：正式 Gate 要**四科全過**，只要有一科明顯落後，這顆就不可能晉級，
   沒有必要花昂貴的三-seed 去確認。原本寫成「都低於 85% 才淘汰」是**取最寬鬆
   的條件**，與 min 判準相反 —— 判定用最弱項，篩選也必須用最弱項。
3. ~~補跑 ctrl/treat c25、c30 的 Gate2~~ —— **2026-07-28 13:30 暫緩**。

   理由：Gate2 只能檢查一般導航能力有沒有退化，**不能解釋 random_2d 為何撞車**。
   在研究問題已收斂到「方向頻繁改變的動態障礙」之後，它不是關鍵路徑。
   （執行面：以預先建立 `DONE` 標記的方式跳過，未修改執行中的腳本。）
4. **碰撞時相分析（現為主線實驗）**

### 研究問題已轉移

同一條走廊、同樣 4 靜態 + 2 動態，只換行人的運動型態：

| 行人運動 | 成功率（treat c20，單 seed） |
|---|---|
| 直線前後走（longitudinal） | **97.3%** |
| 連續隨機漫步、每 1–5 s 轉向最多 90°（random_2d / **wander**） | **69.4%** |

→ 問題已縮小為：**policy 不擅長處理「運動方向經常改變」的動態障礙。**

⚠️ **只能排除「走廊幾何本身太難」**（同一幾何能跑到 97.3%）。
**不能排除模型容量不足** —— 先前報告寫「排除模型容量」是**過度宣稱**，已更正。

仍在候選的機制（互不排斥，需逐一驗證）：

* observation 沒提供足夠的**運動歷史**
* RNN 無法可靠推測轉向
* future-occupancy 用**等速外推**，**遇轉向必然預測錯**
  （正式 Gate 的 random_2d 走 wander：**從不停止**、恆 0.30–0.60 m/s，
   每 1–5 s 轉向一次最多 90°。「行人停住→風險算 0」假說在 Gate 上永不觸發，已刪除）
* **行人轉向 150 °/s vs 機器人 ω_max 45 °/s（快 3.3 倍）** ——
  即使完美預測，角速度上限也做不出對等規避；只剩煞車（2.0 s / 1.0 m）或提早繞開
* reward 沒有正確懲罰這些風險
* 模型容量不足

**設計要求：random_2d 與 longitudinal 對照比較**（同幾何、同密度，唯一差異是運動型態），
至少拆：
   - 接近中 / 最近交會點 / 已遠離
   - TTC、最近距離、車速與轉向
   - 撞靜態 or 動態
   - random_2d：兩個獨立動態的**相對運動**
   - mixed_iid：依 lateral / longitudinal / random_2d family 組合拆分
   - **不得套用 crossing / side_by_side 分類**（Gate 上不存在，見 §3.4）

**SA8 / N1 / λ=0.067 / `_LAMBDA_IS_CALIBRATED` 全部維持 HOLD。**

### 執行紀律：不得修改執行中程式會 import 的檔案

Phase A/B 期間**禁止修改它們的 import 鏈上任何檔案**。碰撞分析程式一律先在
scratch 副本開發，**等 GPU 評測全部結束再合併**。

實據：`play_rnn_car.py:4309` 會 `from corridor_motion_phase_audit import ...`，
而 suite **每次 rollout 都重新啟動一個 play 進程** —— 就地改檔會被下一次 rollout
讀到，造成同一批評測前後跑的不是同一份程式，且不會有任何錯誤訊息。

scratch 位置：`<scratchpad>/collision_dev/`
合併前守恆基準：`corridor_motion_phase_audit.py` md5 = `27091a1ea8249ab25c2ae92763cf3ace`

### 未做但需記錄的實驗缺口（不阻擋診斷）

要把傷害歸因到單一因子，需要**第三臂**：native 58%、走廊 20% 但**全部是真實混合**
（不放 Gate 題型）。與 Treatment 比可分離「Gate 題型」的效果，與 Control 比可分離
「native 減少 10%」的效果。

**第三臂只在需要釐清因果、設計下一個配方時再跑；目前不用阻擋碰撞診斷。**
§0 的限制在它跑完之前持續有效。

---

## 7. 資料位置

* 評測 JSON：`logs/gates/sa71/eval/corridor/{ctrl,treat}_c3840/`
* 訓練 log：`logs/gates/sa71/paired/{control,treatment}.log`
* checkpoint：`logs/rnn_car/sa71paired_{ctrl_sa7recipe,treat_gatealigned}_s42/`
* source freeze：`docs/freeze/sa71_paired_source_freeze.md`
* config：`scripts/.../rnn_car_modular/configs/e2e_sa71_gate_aligned_from_w1c10.py`
