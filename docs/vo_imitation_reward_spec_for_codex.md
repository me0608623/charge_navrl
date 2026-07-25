# Spec v2：訓練期特權 DWA/MPC teacher（bounded-regret，離散動作）改善局部繞行

作者：監督端（唯讀交叉驗證）｜日期：2026-07-23｜狀態：**暫緩；先完成目前 OBB+K8+future 訓練與 gates，本輪不實作**

> [!note] v2 變更（依 Codex 兩路一致審查）
> v1 的「VO 模仿獎勵 + holonomic→unicycle 投影」被否決（投影後動作可能重回碰撞集、單位混用會錯罰正確轉向、VO 不看牆）。**改為在差速車原生 19×19 離散動作空間直接評分的訓練期特權 DWA/MPC teacher（bounded-regret reward）。** 且新增**實驗前置閘門**（先重驗問題是否仍存在，再動 reward）。

---

## 0. ★ 前置閘門（未過不得加 reward、不得重訓）
1. **重驗問題**：對**目前 SA5 收斂 checkpoint** 重跑 near-wall crossing 8 方位 det-eval，確認 `first_turn_correct` 仍偏低。舊「4/12」是**先前 checkpoint、僅 12 次**，未在現行 OBB+K8+走廊 curriculum 重驗，證據鏈不完整。**若已修好 → 本案取消。**
2. **離線 teacher audit（不訓練）**：左右鏡像平衡、牆側不被選中、無解率、停止率、teacher 自己的首轉正確率。
3. **teacher-action probe**：對 K8 frozen obs 驗證學生能否預測 teacher 的**轉向符號＋速度級別**（非只 crossing sign 96%）。K8 對完整 teacher action 的可預測性未證。
4. 三閘通過 → 從**同一 accepted checkpoint 做短期 A/B**（**非**立即 SA1 重訓）。

---

## 1. 目的
改善 e2e K8 policy 的局部動態繞行（疑似選錯邊/herding）。做法 = 訓練期用**上帝視角**建一個 **wall-aware、future-aware、unicycle-reachable 的離散動作 teacher**，以 **bounded-regret 獎勵**教 policy 逼近 teacher 的最優可執行動作；**部署維持純 LiDAR 反應式、零 MOT**。

---

## 2. 背景與問題（已由 Codex 對照 `clean_progress.py` 校正）
- **部署 VO/MPPI 失敗**（用戶實測「連走都不會」）：**部分**歸因 MOT 雜訊放大 VO → freezing robot（Trautman & Krause 2010）；**但完美狀態下 VO 在密集/無可行速度/非完整運動學時仍會凍結**——不能全歸 MOT。故 VO 不放部署、RL 保留（零 MOT）。
- **現有 reward 事實**（校正 v1 誤述）：
  - `progress`（朝 goal）與 `future-occupancy`（等速外推 1.5s、**已 action-conditioned**：用實際 (v,ω) 圓弧算未來最小間距，`clean_progress.py:220`）**相加即已是單一 scalar 聯合目標**。
  - future-occupancy **已能隱含分辨左右**（不同動作間距不同）；v1 說它「不分左右」**錯誤**。
  - RL 最大化長期回報、非沿 reward 空間梯度；**不能只憑 4/12 斷定「力平衡」**。
- **真正待補的缺口（精確版）**：現有 reward **沒有直接提供「在所有可執行動作中，哪個同時最安全、最朝 goal、又不撞牆」的反事實排名（counterfactual ranking）**。future-occupancy 只罰動態未來碰撞，未把「靜態/牆 + goal progress + 可達性」合成單一動作排名 → 可能選「動態上安全但牆側錯」的方向。

---

## 3. ★ 主張裁決（Codex 兩路一致，記錄供追溯）
| # | 主張 | 裁決 | 要點 |
|---|------|------|------|
| C1 | 部署 VO 凍結是 MOT 問題 | **部分對** | MOT 放大會凍結；但完美狀態＋密集/無解/非完整/過保守也會凍，不能全歸 MOT |
| C2 | 非合作行人用 α=1.0/全責 | **大致對** | 全責較合理；但行人非等速/會反應時 α=1.0 仍只是近似非唯一解 |
| C3 | 缺單一聯合「安全+goal」項 | **部分錯** | 相加已是聯合目標；真正缺的是**可執行動作間的反事實排名**（含牆），非「單一項」 |
| C4 | VO v\* 天生選對邊 | **部分對** | 對稱幾何非唯一解會翻邊、**不看牆**、不含差速可達性、非長期最優 |
| C5 | 特權 teacher 蒸餾 LiDAR | **有條件成立** | 避開部署 MOT；但學生須能辨識 teacher 決策（轉向符號+速度級別，未證） |
| C6 | 全向 VO 無法給差速車 | **完全正確** | 事後投影可能重入碰撞集；**規格最大問題** → 改離散動作 teacher |

---

## 4. 具體規格：訓練期特權 DWA/MPC teacher（bounded-regret）
**不算全向 VO、不做投影。** 直接在車真正的離散動作空間找 teacher。

### 4.1 候選動作
- 對 19×19 個 `(v, ω)` 候選（= 既有 MultiDiscrete action table），**納入加速度上限與 actuator 動態**，只保留該步物理可達者。

### 4.2 每候選展開與成本
- 用既有 unicycle 圓弧（`clean_progress.py:216–229`）展開 **1.5–2.0s**。
- 聯合成本：
$$J(a)=w_\text{dyn}C_\text{dyn}(a)+w_\text{stat}C_\text{static\&wall}(a)+w_\text{goal}\,d_\text{goal}^{\text{horizon}}(a)+w_\text{turn}\,\text{heading}(a)+w_\text{smooth}\Delta a+w_\text{stall}C_\text{stall}(a)$$
  - `C_dyn`：動態障礙未來佔用（上帝視角等速外推，同 future-occupancy）。
  - `C_static&wall`：靜態障礙 + 牆 + **OBB swept collision**（複用既有 OBB）。
  - `d_goal^horizon`：展開終點到 goal 距離。
  - `heading`、`Δa`、`C_stall`：轉向誤差、平滑、停滯懲罰。
- 允許「**停下讓行**」為合法候選（`v=0` 在集合內）。

### 4.3 bounded-regret 獎勵
$$a^*=\arg\min_a J(a),\qquad
r_\text{teacher}=-w\cdot\text{clip}\!\Big(\frac{J(a_\text{exec})-J(a^*)}{\text{scale}},\,0,\,1\Big)$$
- `a_exec`：policy 該步實際執行動作（**時序需明確**：是當前 action 或前一 action 的結果，制動器延遲下要對齊）。
- **bounded [0,1]**：解決 v1 的單位混用/權重不可比（future-occ 已 clip[0,1]，此項亦然）。
- 對稱幾何 `J(a)` 左右相等時 → regret 對兩邊皆低 → **soft、不強迫亂選邊**。

### 4.4 gating / 落點
- 只在近端（`near=3.0m`）有動態障礙時算，receding horizon 每步重算。
- 加在 `clean_progress.py` 與 future-occupancy 並存；CLI `--use_teacher_regret` + `teacher_regret_weight`，**預設關（baseline 不變）**。

---

## 5. 護欄與風險
1. teacher 自己在密集場景可能選零速 → student 繼承凍結 → **離線 audit teacher 停止率/無解率**（前置閘門 §0.2）。
2. 非 potential-based → 可能帶偏最優 → 軟權重、A/B 驗、掃權重。
3. `C_static&wall` 必含牆，否則重演「動態安全但牆側錯」。
4. `scale` 校準：用 teacher regret 的合理量級（離線 audit 得分佈）設定，避免壓過主 reward。
5. 時序對齊（`a_exec` 定義）＋障礙速度座標系（避免自車旋轉偽動態）需明確。

---

## 6. 驗收（A/B，從同一 accepted checkpoint，read-only det-eval）
- control（`--use_teacher_regret` off）vs treatment（on），其餘逐字同。
- 8 方位 crossing det-eval：`first_turn_correct`（主指標）、最小牆/人淨空、lateral drift、CR、freeze、繞路時間、八方位一致性、車自旋假移動率。
- 通過：首轉正確率顯著↑ 且 CR/freeze/牆淨空不惡化。

## 7. 關聯
- Obsidian `rnn/2026-07-23_VO凍結與MOT依賴_future-occupancy教練優化討論`（含 v1→v2 更正）。
- memory `finding_carside_crossing_herding`、`finding_future_occupancy_reward_fixes_crossing`。

---

## 8. 2026-07-23 決策：先完成目前訓練

### 8.1 本輪處置
- 不修改目前 SA5 的 reward、網路、optimizer、curriculum 或自動晉級流程。
- 不在目前 lineage 中加入 teacher reward、behavior-cloning loss 或額外 privileged observation。
- 先讓 OBB + K8 + action-history + future-occupancy 配方完成訓練與 deterministic gates。
- 只有目前 accepted checkpoint 仍能重現「首轉選錯／near-wall herding」時，才恢復本案。

### 8.2 恢復本案時的證據順序
1. **先證明問題仍存在**：用現行 accepted checkpoint 跑 near-wall crossing 與 8 方位 crossing；舊 `4/12` 不可直接沿用。
2. **先證明 teacher 正確**：teacher-only 閉環跑同一套 gates；安全採硬限制，無解或左右成本近似時 abstain，不強迫單一答案。
3. **再證明學生看得懂**：K8 frozen-observation probe 預測 teacher 的轉向符號、速度級別與停讓決策；crossing sign 96.15% 只是必要條件。
4. **最後才做短期 A/B**：相同 accepted checkpoint、seed、步數；control 維持現配方，treatment 才加入 teacher supervision。
5. **短期 A/B 通過後才考慮 SA1 重訓**。

### 8.3 「最佳」的限定
Teacher 不提供全域最佳路徑，只提供：

```text
在本步動力學可達候選集合中，
依指定 1.5–2.0 秒模型、障礙預測與安全/goal 排序，
成本最低的局部控制命令。
```

每 0.2 秒只執行第一步並重算（receding horizon）。若要宣稱全域最佳，需要地圖、完整長期障礙預測與全域規劃器，超出本案範圍。

### 8.4 Teacher 正確性護欄
- 先淘汰 swept OBB 會碰動態障礙、靜態障礙或牆的候選；goal 收益不得抵銷碰撞。
- 納入目前速度、加速度限制；未來 actuator bridge 啟用後再納入實際延遲模型。
- 左右鏡像輸入應得到鏡像控制；若第一、第二候選成本差 `ΔJ` 太小，給 soft target 或不教。
- 所有候選都不安全時記錄 no-feasible/abstain，不生成虛假的 expert label。
- Teacher 本身未通過 CR、freeze、timeout、first-turn 與 path-recovery gates，不得進學生訓練。

### 8.5 學生可學性護欄
- 單靠 bounded-regret reward 不是最直接的模仿方式；若恢復本案，優先比較：
  - `PPO + bounded-regret reward`
  - teacher demonstrations 預訓後再 PPO
  - `PPO +` 衰減式 teacher cross-entropy
- Probe 至少量測：轉向符號準確率、速度 `±1 bin` 準確率、左右鏡像偏差、teacher-action conditional entropy。
- Teacher 只能在 policy 實際可觀測資訊足以分辨的狀態給強 supervision；否則 privileged label 會在觀測別名下互相衝突，學生可能平均成停車。

### 8.6 文獻定位與不可過度宣稱處
- DWA 支持「在動力學可達速度中展開短期軌跡並評分」。
- VO 支持「依移動障礙速度排除未來碰撞速度」。
- Planner imitation、Reinforced Imitation、Learning by Cheating 支持「模擬 expert／privileged teacher → sensor-only student」。
- 沒有單一論文直接驗證本專案的完整組合：
  `K8 VLP16 + 19x19 unicycle rollout + dynamic occupancy + swept OBB/wall + privileged teacher + PPO`。
- 因此目前判定是「元件有先例、組合有可行度、整體仍須逐閘證偽」，不是已被論文保證有效。

### 8.7 計算量備忘
未優化上限約為：

```text
1024 env × 361 candidate × 8 horizon samples × 20 obstacles
≈ 5,900 萬次距離比較 / policy step
```

這是 GPU tensor 幾何，不是 361 倍 Isaac Sim，但仍可能降低 FPS。若恢復本案，先做離線 benchmark，再依序採：
- 只對近端動態障礙 active env 計算。
- 只保留近端障礙。
- 對加速度限制後得到相同實際速度的候選去重。
- 分塊計算，避免一次配置大 tensor。
- 先用縮減候選集做 teacher audit，證明值得後再測完整 361。
