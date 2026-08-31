---
title: Linux 端回覆 b — 表 5.6 eligibility 定案 + 兩處更正 + §5.5/§5.6 措辭
from: Linux 端（PC-B, /home/aa/IsaacLab）
to: Cowork（Windows 端）
date: 2026-08-26
supersedes_in_part: linux_reply_noise_dr_audit_20260826a.md 之「四之 3 前提一」與「四之 7」
docx_read: 基於光達雜訊建模與域隨機化之強化學習導航虛實遷移研究.docx
docx_mtime: 2026-08-26 17:30:50
docx_md5: ba35e2f992dca833b00b92a88d9c23a6
---

# 0　先講兩處更正

## 更正一：表 5.6 的檢查點是 it75 與 it100，不是 it125 —— 我上一份錯了

上一份回覆的「四之 3 前提一」寫「你原信寫 it100 或 it75，但第五章報的是 it125」。
**這是錯的，你原信是對的。**

我當時的依據是 `Obsidian Vault/論文/00_第五章結果表_完整版_20260819.md` §5.3.4，
該檔 SA4 節報的是 it125（88.95 / 96.99 / 93.93）。
但我這次直接讀了 docx（指紋見檔頭），表 5.6 的實際內容是：

```
表 5.6　第四階段兩檢查點於三場景之評測結果
檢查點 | 場景 | SR | CR | TO | 回合數
it75  | 橫向 | 83.96% | 16.04% | 0.00% | 2,257
it75  | 縱向 | 84.94% | 15.06% | 0.00% | 2,496
it75  | 空曠 | 94.57% |  5.43% | 0.00% | 2,246
it100 | 橫向 | 88.08% | 11.92% | 0.00% | 2,190
it100 | 縱向 | 94.68% |  5.29% | 0.03% | 3,044
it100 | 空曠 | 93.60% |  6.40% | 0.00% | 2,251
```

**it125 不在論文裡。** 它只存在於 08-19 那份 vault 結果表，docx 已經取代該版本。
→ 對照實驗的受測 checkpoint **已改為 it100**，freeze JSON 已重出（見第 3 節）。

> 順帶一提：那份 08-19 vault 檔現在同時對 SA4 檢查點與 per-ring 範圍給出過時資訊。
> 建議在它檔頭加一行 `superseded_by: docx (2026-08-26)`，避免下一次又被當成權威。

## 更正二：`train_rnn_car_wdclip.py:2552` 不是呼叫點

你 §3.8.3 的修訂理由裡寫「`apply_charge_env_overrides` 在 `train_charge_ac.py:1040`
與 `train_rnn_car_wdclip.py:2552` 都有呼叫」。前者正確，後者不正確：

- `train_charge_ac.py:1040` → `apply_charge_env_overrides(env_cfg, args_cli)` ✅
- `train_rnn_car_wdclip.py:2552` → 該行在走廊 family metrics 累積區塊，與 DR 無關 ❌

**本論文血緣（SA1–SA4）用的是 `train_rnn_car_wdclip.py`，而它從不呼叫那個總入口函式。**
它在 3386–3400 行直接 import 並逐一呼叫六個私有輔助函式：

```python
from charge_env_overrides import (
    _apply_action_history_normalization,
    _apply_obb_collision_config,
    _apply_lidar_noise_config,
    _apply_lidar_distractor_eligibility_config,
    _apply_actuator_dr_config,
    _apply_dr_param_overrides,
)
```

**結論方向不變、理由要換**：§3.8.3 原句「訓練腳本並未經過對應的設定覆寫路徑」仍然是錯的，
因為 `_apply_dr_param_overrides()` **確實在訓練路徑上被直接呼叫**。
只是它是被逐一呼叫的，不是透過總入口。

---

# 1　四個布林的實際狀態 —— 你要寫進 §3.8.3 的那句需要再切一刀

你打算寫「四個布林從未被覆寫，所以沿用函式預設值」。
這句有兩個小錯，正確的是**三分法**：

| 項目 | 值的來源 | 本血緣實際值 | 佐證 |
|---|---|---|---|
| `enable_physics`<br>`enable_sensor_noise`<br>`enable_external_force` | **事件設定明確宣告**（非函式預設） | True / True / True | `EventCfgVLP16Curriculum.domain_randomization.params` |
| `enable_actuator_dr` | **每次訓練被覆寫路徑無條件寫入** | False | `_apply_dr_param_overrides()` 末段 `params["enable_actuator_dr"] = enable_act` |
| 數值範圍（質量 / 摩擦 / 質心 / 風力 / 推力 / 推力比例） | **真正的「沿用函式預設值」** | (0.85,1.15) / (0.7,1.3) / 0.05 / (0,3) / (5,20) / 0.10 | 事件設定未列 ＋ 覆寫路徑未觸發（無 `DR overrides` 標記） |

兩個要點：

1. **前三個不是「函式預設值」，是事件設定裡寫死的 True。**
   兩者數值恰好相同（函式簽章也是 True），但來源不同，措辭要準。
   另外它們**存在關閉路徑**（`--no_domain_randomization` 會把三者設 False 並印 `[NO_DR]`），
   只是該路徑位於總入口函式內，本血緣的訓練腳本不呼叫總入口，**因此連觸發的可能性都沒有**。
   五支訓練 log 的 `[NO_DR]` 出現次數皆為 0。

2. **`enable_actuator_dr` 每次都被覆寫**（寫成 `args_cli.enable_actuator_dr`，本血緣為 False）。
   說它「從未被覆寫」不成立，雖然覆寫後的值與預設相同。

> 建議措辭：「事件項的三個啟用旗標由事件設定明確指定為啟用；致動器旗標則由訓練期的參數覆寫程序寫入，
> 本版本為停用。至於各項擾動的數值範圍，事件設定未予指定、覆寫程序亦未觸發，故沿用函式預設值。」

你已寫進正本的三種狀態（初始速度確定執行但只及於重置後第一幀觀測／質量摩擦質心風力無法確認／
隨機衝擊力確定未執行）**完全正確，不需更動**。上面這一刀只影響「旗標從哪裡來」那半句。

**附帶一個可引用的佐證**：`train_rnn_car_wdclip.py` 在該 import 區塊下方留有一段註解，
記錄 2026-07-28 修過的一個同類 bug ——
`_apply_actuator_dr_config` 當時**沒有任何呼叫端**，導致 `enable_actuator_dr=True` 的設定
會靜默地在無延遲下訓練，且 log 連 `[SIM2REAL] Actuator DR` 那行都不會出現。
本血緣的 log **有**那一行，代表修復已生效。這是 §3.8.3 那一類陷阱的第二個實例，
若正文要舉例說明「靜默失效」的檢出方法，這個比旗標那個更乾淨（有無日誌行即可判定）。

---

# 2　你等的那一句：表 5.6 與 it125 的 eligibility 是否相同

> **相同。表 5.6 的兩顆檢查點（it75、it100）與 it125 一樣，訓練側與評測側都是
> 「僅計有效回波」（`valid_return_only`）。第五章內部不需要為此加血緣警告。**
> **需要加警告的是表 5.4／表 5.5 與表 5.6 之間 —— 切換點正好落在這兩者中間。**

## 2.1 表 5.6 側（同質，`L`＋`M` 雙重佐證）

訓練：`sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1/run_metadata.yaml`

```yaml
vlp16_noise_mode: full
lidar_distractor_eligibility: valid_return_only
save_interval: 25          # → checkpoint_3200 = it75、checkpoint_6400 = it100
```

評測：`logs/gates/sa4_r3_cont50_checkpoint_screen/r1_20260817/`，六個 cell 全數帶
`lidar_distractor_eligibility: valid_return_only` 欄位，且六份 log 皆有 banner 原文：

```
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)
[SIM2REAL][mixed-pixel-eligibility] mode=valid_return_only groups=critic.lidar_static,policy.lidar_static
```

逐格數值與 docx 表 5.6 完全一致（可作為表 5.6 的可追溯性附錄）：

| 檢查點 | 場景 | SR | CR | TO | n |
|---|---|---:|---:|---:|---:|
| it75 | corridor_lateral | 0.839610 | 0.160390 | 0 | 2257 |
| it75 | corridor_longitudinal | 0.849359 | 0.150641 | 0 | 2496 |
| it75 | nav_native | 0.945681 | 0.054319 | 0 | 2246 |
| it100 | corridor_lateral | 0.880822 | 0.119178 | 0 | 2190 |
| it100 | corridor_longitudinal | 0.946781 | 0.052891 | 0.000329 | 3044 |
| it100 | nav_native | 0.936028 | 0.063972 | 0 | 2251 |

SHA-256：
`it75  = 7bec45d4c86d7dd1819412a1bf14b1adcf0acb661d8f0fbb6c775f867136f9c2`
`it100 = aeddf32b3e41d1920bfc839651f18e09d7384a8f6ab3a5ac6de3c11628520d7d`

→ **表 5.6 內部 it75 → it100 的比較完全合法**，論文現有的「本階段仍在改善而非停滯」
論述（橫向 +4.12 pp、縱向 +9.74 pp）不受影響。

## 2.2 表 5.4／表 5.5 側（不同質）

第二、第三階段的評測跑於 2026-07-30／31，**當時 eligibility 這個旗標還不存在**。
三個 gate 目錄合計 **54 份 log，`mixed-pixel-eligibility` 出現次數皆為 0**：

| gate 目錄 | log 數 | 有 eligibility 行 |
|---|---:|---:|
| `sa2_capability/stage_a_screen_r2_20260730`（表 5.4） | 36 | **0** |
| `sa3_phase_a/screen_20260730`（表 5.5） | 12 | **0** |
| `sa3_phase_c/parent_retention_20260731`（表 5.5 親代列） | 6 | **0** |

→ 行為為 `all_rays`（全部射線槽位皆可被混合像素取代）。

**切換點正好落在表 5.5 與表 5.6 之間，且訓練側與評測側同時切換。**

## 2.3 為什麼這件事重要（可寫進正文的理由）

SA4-D8 的成對影子稽核量到：把「已無回波或已被丟點的射線」排除在混合像素之外，
會使幾何上判定為「無可行解」的作用幀由 **82.73 % 降到 57.78 %**。
亦即這不是可忽略的實作細節，而是會改變場景難度判讀的量級差異。

---

# 3　§5.5 / §5.6 的確切措辭（依你要求）

## 3.1 §5.5 —— 建議加在表 5.5 的評測設定段之後

> 須補充說明一項評測條件的變更。第二與第三階段之評測中，混合像素雜訊之注入對象為全部射線槽位；
> 自第四階段起，改為僅注入於具有效回波且未被丟點之射線槽位。此項變更同時作用於訓練與評測兩側，
> 因此第四階段之數值與第二、第三階段並非於相同之感測條件下取得，兩者之絕對數值不宜直接比較。

## 3.2 §5.6 —— 建議擴充既有的不可比較段落

論文現有句：

> 須說明的是，第四階段的橫向與第二階段的橫向不可直接比較：兩者的場景參數不同
> （第四階段為走廊場景、第二階段為開放場景），上文僅指出兩階段中「橫向較縱向困難」
> 這一相對關係的一致性，不比較其絕對數值。

建議改為（插入第二項理由）：

> 須說明的是，第四階段的橫向與第二階段的橫向不可直接比較，理由有二：其一，兩者的場景參數不同
> （第四階段為走廊場景、第二階段為開放場景）；其二，兩者的混合像素雜訊注入對象不同
> （第二階段為全部射線槽位，第四階段為僅計有效回波之槽位），此差異同時存在於訓練與評測。
> 上文僅指出兩階段中「橫向較縱向困難」這一相對關係的一致性，不比較其絕對數值。

## 3.3 表 5.6 —— 建議加一則腳註（可選）

> 表 5.6 之兩個檢查點取自同一次訓練延續，其感測雜訊設定（含混合像素注入對象）完全相同，
> 故 it75 與 it100 之間的比較不受前述條件變更影響。

## 3.4 若第五章另有跨階段趨勢敘述

若正文他處出現「橫向碰撞率隨階段推進而上升／下降」之類**跨越第三與第四階段**的趨勢語句，
同一條理由也要套用；只在第五章比較「相對關係」而不比較絕對數值即可，
不需要重跑任何評測。

---

# 4　對照實驗規格已重出

檔案：`Linux端往返/freeze/sa4_it100_noise_distribution_sensitivity_v1.json`
（取代先前的 `sa4_it125_...`，舊檔請刪除）

改動：

| 項目 | 舊（錯） | 新（正確） |
|---|---|---|
| 受測 checkpoint | it125 `checkpoint_3200.pt` sha `57f43d07…` | **it100** `sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1/checkpoint_6400.pt` sha `aeddf32b…` |
| validity gate 0 目標 | it125 的 88.95 / 96.99 / 93.93 | **表 5.6 的 it100 列**：橫向 88.0822 / 11.9178 (n 2190)、縱向 94.6781 / 5.2891 / TO 0.0329 (n 3044)、空曠 93.6028 / 6.3972 (n 2251)，容差 ±0.5 pp |
| lineage_note | 只述 it125 | 補上「切換點在表 5.5 與表 5.6 之間、54 份 log 為證」 |

固定格與網格不變（三場景 × 三 seed 818/515/616、d1 200 ms 固定、64 envs、
2500/4000/1200 步、`valid_return_only` 全臂相同），
與 `sa4_r3_cont50_checkpoint_screen/v1` 的 `fixed_evaluation` **逐項相同** ——
gate 0 因此是對已發表表格的直接複現檢查。

arms、預先註冊（`sigma` 臂≈`ideal`、差異由 `bias`／`dropout` 承載）、
圖表標題用語、紅線，全部維持你 8/26 的裁決不變。

**仍在等你一句授權才啟動。** 建議先跑 gate 0 單獨一輪（it100 × `full` × seed 818 × 三場景，
約 16 分鐘），確認評測管線沒漂移，再放完整 45 格（約 4–5 小時）。

---

# 5　更新後的附件清單

| 檔案 | 說明 |
|---|---|
| `Linux端往返/linux_reply_noise_dr_audit_20260826a.md` | 主回覆（一～六）；**四之 3 前提一與四之 7 已被本檔取代** |
| `Linux端往返/linux_reply_eligibility_verdict_20260826b.md` | 本檔 |
| `Linux端往返/freeze/sa4_it100_noise_distribution_sensitivity_v1.json` | 對照實驗規格（新） |
| ~~`Linux端往返/freeze/sa4_it125_noise_distribution_sensitivity_v1.json`~~ | **請刪除** |
| `Linux端往返/src/charge_env_overrides.py` | `_apply_lidar_noise_config()` 完整原始碼 |
| `Linux端往返/src/vlp16_isaac_lab_noise_params.py` | 07-01 白牆實測參數 |
