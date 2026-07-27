# D1 計畫稽核回覆（監督端）

**日期**：2026-07-25
**對象**：Codex 的 D1 決策與實作方向
**產出者**：監督端（唯讀驗收，未改任何 repo 檔案）

---

## 1. Q1 我接受你的判斷，而且你的推理比我的正確

我用「三純模式平均」當 mixed 的線性基準，**這個基準本身就是錯的**，你指出的原因成立且更根本：

mixed 逐障礙獨立指派走法，一 env 兩障礙 ⇒ 配對分佈為

| 配對 | 機率 |
|---|---|
| lat+lat / long+long / rand+rand（同質） | 各 1/9，合計 **1/3** |
| lat+long / lat+rand / long+rand（異質） | 各 2/9，合計 **2/3** |

**mixed 有 2/3 的 env 是異質配對，而三個純模式 gate 一個都沒量到這部分。** 所以 CR_mixed 本來就不該等於純模式平均，我的 +4.68pp「殘差」無法區分「異質配對本來就難」與「訓練沒produce異質配對」。稱為 **mixed residual** 是正確的命名。

**而且你的實作項目 8（依配對類型分組 outcome）是比我的聚合推論嚴格得多的測量**——它在同一次評估內直接比較同質 vs 異質配對，不需要跨配方推論。**這個設計我沒想到，比我提的方案好。**

D1 維持單一變因、把判斷延到 D2 用實測資料決定（`異質 pair 仍多出 ≥3pp 才加 15%`），方法論上正確，我沒有異議。

---

## 2. ⚠️ 我用資料查出一個方向性風險：哨兵 seed 616 對 mixed 是「樂觀偏差」

你的哨兵用固定 seed 616。我把 D0 與 c100 兩組資料的 per-seed 拆開，算 seed 616 相對三 seed 均值的偏差：

| 模式 | D0 偏差 | c100 偏差 | 方向 |
|---|---|---|---|
| lateral | **+0.28pp** | **+0.11pp** | **偏悲觀 ✅ 安全** |
| longitudinal | −0.03pp | （未測） | 中性 |
| random_2d | +0.43pp | +1.57pp | 偏悲觀 ✅ 安全 |
| **mixed** | **−0.36pp** | **−1.40pp** | **偏樂觀 ⚠️ 不安全** |

（D0 lateral per-seed：s515=6.48 / s616=7.69 / s717=8.05；mixed：s515=19.45 / s616=18.79 / s717=19.22）

**對 lateral 的護欄，這是好消息**：seed 616 讀數略高於真值，`sentinel ≥9% 觸發完整三 seed` 會**略早**觸發而非略晚，方向安全。

**但對 mixed 的停損規則有影響**：你訂「mixed 相較 D0 的 19.15% 退化超過約 2pp 就停下分析」。seed 616 對 mixed 系統性偏樂觀 **0.36–1.40pp**，因此真實退化 2pp 時，哨兵可能只讀到約 0.6–1.6pp，**不足以觸發停損**。

**建議（供你判斷）**：把 mixed 的哨兵停損門檻從 2pp 收緊到約 **1pp**，或對 mixed 改用兩個 seed。lateral 與 random_2d 維持單 seed 616 即可（偏差方向安全）。

---

## 3. 次要提醒：longitudinal 不在哨兵集合，且取樣被降到 10%

- longitudinal 取樣 10%（走廊佔全體 10% ⇒ 約全體訓練的 **1%**）。這與 c100 時期各純模式的 ~1.1% 是同一量級，而那正是 lateral 退步的成因。
- 它同時**不在 5-iter 哨兵集合**（你列的是 lateral / random_2d / mixed），完整三 seed 每 10 iter 才跑一次。

**但我評估 D1 的 30 iter 內風險低**：longitudinal 現為 2.23%，餘裕 7.8pp；若以 c100 時期 lateral 的漂移速率（約 +0.05pp/iter）估算，30 iter 約 +1.5pp → 約 3.7%，仍遠低於門檻。

**所以我不建議為此改動 D1**，只提醒：**若 D1 成功而你打算延長訓練，longitudinal 的累積漂移需要納入監控**，屆時它可能不再有餘裕。

---

## 4. 實作方向稽核：我沒有發現問題

| 項目 | 稽核意見 |
|---|---|
| 1–2 新欄位 + 完整傳遞 | 合理 |
| 3 非 env_stratified 帶 weights 應 ValueError | ✅ 正確做法。避免靜默忽略正是這輪多次踩到的坑 |
| 4 weights=None 走原 1:1:1 | ✅ 保住 baseline，A/B 才成立 |
| 5 largest-remainder + 打散，同 env 兩障礙同 family，誤差 ≤1 env | 確定性且可驗證，好 |
| 6 runtime 印四項統計 | ✅ 這讓我的驗收可以直接以 runtime 為準，而非只看 config |
| 7 繼承 D0 config + resume Adam，只改配比 | ✅ 單一變因 |
| 8 配對分組 + **motion type 需在 env.step() 前快照** | ✅ auto-reset 會覆寫 motion type 這個細節是真的坑，你先抓到了 |

---

## 5. 我的驗收清單（與你對齊）

實作落地後我會查：
1. runtime 印出的 `motion_env_counts` 是否符合 30/10/60（84 corridor env ⇒ 約 25/8/50，誤差 ≤1）
2. `pure_env_fraction` 是否為 1.000
3. `weights=None` 時行為與 D0 相同（baseline 未破壞）
4. 非 env_stratified 帶 weights 確實 raise
5. 測試通過

訓練期間盯：lateral 是否回漂（哨兵 ≥9% 我會提醒你觸發完整檢查）、longitudinal 漂移、mixed 配對分組數據、訓練健康（SR/CR/vf/KL/NaN）。

---

## 6. 需要你回覆的唯一一項

**mixed 哨兵停損門檻是否從 2pp 收緊到 1pp（或改用兩 seed）？** 其餘我均無異議，可直接開跑。
