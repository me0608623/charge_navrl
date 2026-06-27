---
name: finding_bangbang_degenerate_penalty0
description: stage3 滿舵塌縮(deterministic sin波/舞龍舞獅)兇手定案=obs_delay_steps[0,1]觀測延遲(感測端非致動端)；單變因隔離排除 actuator延遲/penalty/ent/特權資訊；用 lidar regime 判病態
metadata:
  node_type: memory
  type: project
  originSessionId: 531aefcd-2bb8-4fac-84f7-301302631a09
---

**⛔⛔ 重大更正（2026-06-20 晚，系統性比較後）：本檔原本說「penalty_smoothness=0 是兇手」是錯的。** 證據：(1) **乾淨的 sa3_v2(stage3) 也是 penalty_smoothness=0** → penalty=0 不可能是兇手。(2) 修法 penalty 0.003（SA3_v3g）試金石**失敗**（空曠飽和仍 100%）。(3) penalty 罰 stochastic rollout 的 |Δω|（訓練時本來就低 sw 0.02），碰不到 deterministic argmax 才出現的滿舵 → 必然無效。

**✅ 正確結論（系統性比較：乾淨 sa3_v2 vs 退化 SA3_v3f，都 stage3、都 79D 無 act_hist）**：
- ❌ **排除 penalty_smoothness**（乾淨的也是 0）。
- ❌ **排除特權資訊/asymmetric critic**（用戶提的；乾淨 sa3_v2 也用 asymmetric critic、predict_dim 13）。
- ❌ **排除 reward 設計**（cost_operate 0.03 / cost_turn_rate 0.5 兩邊相同；action_reward 的 (1-nomal_turn)² 其實獎勵直行但太弱 ~0.003/step）。
- ❌ **排除 ent_angular**（只影響塌縮快慢非根因）。
- ⭐ **真兇縮到「v3 為 sim-to-real 加的東西」**：乾淨 sa3_v2 vs 退化 SA3_v3f 唯一差異 = **ω_max 2.0→1.2（真馬達上限,不能改）、actuator 延遲 OFF→ON、r_min 0.9→0.25、obs_delay [0,0]→[0,1]**。sa2_v3f/210000(ω_max1.2,stage2)乾淨 → 是 **stage3 難度 × v3改動 的交互**。**用戶最初的延遲假設回到嫌疑名單（actuator 延遲在乾淨 v2 OFF、退化 v3 ON）。**

**✅✅ 兇手定案（2026-06-20 夜，單變因隔離完成）：`obs_delay_steps [0,1]`（觀測延遲）是 sin 波/舞龍舞獅滿舵塌縮的兇手。** 用戶最初「底盤延遲」直覺**對了，但在感測端（observation delay）不是致動端**。

**🖥️ GUI 目視確認（2026-06-20）**：用戶用 play_launcher 跑 noobsdelay/checkpoint_30000.pt（stage3、空曠、deterministic、障礙物=0）→ **確認無 sin 波、直行不扭**。比飽和率指標更直接的視覺證據。修法（全 18 個 v3 config `obs_delay_steps [0,1]→[0,0]`，actuator 保留）已坐實。詳見 [[finding_obsdelay_misimplemented_as_motordelay]]。

**單變因隔離矩陣（三實驗皆 stage3、79D 無 act_hist、從 sa2_v3f/210000、iter100=checkpoint_30000 deterministic 空曠 eval）**：
| 實驗 | actuator_dr | obs_delay | iter100 空曠飽和 |
|------|:-----------:|:---------:|:---------------:|
| SA3_v3f (原) | ON | **[0,1]** | ~100% 退化 |
| SA3_v3f_nodelay | OFF | [0,1] | 100% → **actuator 無罪** |
| **SA3_v3f_noobsdelay** | ON | **[0,0]** | **0/12=0%** → ⭐ **obs_delay 坐實** |

noobsdelay/30000 全 63 樣本只 1 個飽和（空曠 0/12，target ω min 全 0.55-0.72 用小修正）。唯一把飽和 100%→0% 的變因 = obs_delay。**obs_delay 既必要又充分**（nodelay 只留 obs_delay 仍 100%；noobsdelay 只關 obs_delay 即 0%）；actuator 延遲既非必要也非充分。

**機制**：obs_delay [0,1] = 觀測隨機延遲 0 或 1 步(0.2s) 的回授迴路延遲（非平穩）。RNN policy 對延遲/過時狀態學會激進過補償 → deterministic argmax 塌縮成 bang-bang 滿舵 = 慢頻大弧 weave。屬 [[finding_cmd_delay_limit_cycle]] 的 loop-delay limit cycle，但在**感測端**。

**待解（部署兩難）**：obs_delay 是為 sim-to-real 建模真實感測延遲，不能直接拿掉（實車有延遲）。修法選項（未定，待用戶決策，勿盲調）：(1)拿掉 obs_delay 接受 sim-real gap；(2)保留但把延遲量塞進 obs 讓 policy 知道（delay-aware）；(3)降延遲幅度/變異；(4)用 clean policy 再 fine-tune；(5)直接罰飽和/bang-bang。

（以下為更早的 nodelay 隔離紀錄，actuator 已排除）`wd_sa3_v3f_nodelay`：唯一變因 enable_actuator_dr true→false。判定結果 iter100 空曠飽和 100% → actuator 延遲**無罪**。run=sa3_v3f_nodelay_ne1024_s42。

**症狀本質（不變的事實）**：deterministic argmax（部署用這個！）下 target ω 死貼 max(1.07-1.20)、完全不用小修正(<0.8)、sw(|Δω|)低=慢頻大弧 weave=舞龍舞獅。stochastic 採樣也 98% 飽和(分布整個壓 max,非雙峰)。乾淨 ckpt(sa2_v3f/210000)用小修正 0.37-0.68。退化是 stage3 訓練累積(210000乾淨→SA3 滿舵)。

---
（以下為更正前的初步診斷，penalty 結論已推翻，但 lidar regime 判別法仍有效）

**🎯 根因定案（2026-06-20，SA2_v3f/SA3_v3f deterministic eval + lidar regime 分析）：v3e/v3f 殘留 sin 波的兇手是 `penalty_smoothness=0`，不是 ent_angular。**

**機制**：`penalty_smoothness` 罰 |Δω|（角速度逐幀變化）。設 0 時，**bang-bang ±最大角速度（左右抵銷=平均直行）與「平順小修正直行」拿到完全相同 reward** → policy 沒理由不 bang-bang → 訓練久了塌縮到這個「免費」degenerate 解（target ω 死貼 ±1.20 滿舵）。bang-bang = 每幀最大 Δω，所以 penalty_smoothness 直接命中它。

**證據鏈**：
- SA2_v3f（ent_ang 0.05, penalty 0）：deterministic 隔離 eval 飽和率 iter700 ~0% → iter1000 100%，**漸進塌縮**（~700 iter）。
- SA3_v3f（ent_ang 0.02, penalty 0, 從 sa2_v3f/210000 接 stage3）：**iter100 就 100% 飽和**，更快。→ **降 ent_angular 反而更快塌縮 → ent_angular 不是兇手**（只影響快慢：熵低塌縮更快）。共同點是 penalty=0。
- 訓練 sw 一直低（0.02）但 deterministic eval 100% 飽和 → 矛盾的解：訓練 stochastic rollout（in-distribution 有障礙）平順，但 policy 已塌縮成 bang-bang，deterministic/隔離才現形。

**★ lidar regime 判別法（用戶洞察，2026-06-20）：看 target ω 飽和必須同時看 lidar raw 最近障礙距離。**
- **空曠（lidar最近 > 1.2m，附近無障礙）也滿舵 = 真病態 bang-bang weave = sin 波。**
- 近障礙（lidar ≤ 0.5m）滿舵 = 可能合理避障，不一定病態。
- 決定性對照：SA3_v3f/30000 空曠飽和 **7/7=100%**；sa2_v3f/210000@s3 空曠飽和 **0/9=0%** → 同場景同 stage，late ckpt 空曠也滿舵=病態，clean ckpt 不會。三重確認（stochastic 也飽和 36/36、無牆也飽和 41/41、空曠也飽和）排除 argmax/牆/metric 問題。
- grep 範本：`grep "目標[0-9.]+~.*雷射最近=[0-9.]+"` 配對抽 (target ω min, lidar最近)，awk 篩 lidar>1.2 算飽和率。

**修正先前誤判**：[[finding_action_error_selfref_oscillation]] 原本「penalty_smoothness 害 sin 波」是 v3d（同時有 act_hist + 編碼 bug）的混淆 —— 那個 sin 波更可能是 act_hist 害的。**penalty=0 的代價（bang-bang 塌縮）當時被忽略。** sin 波其實有 U 型 penalty 關係：penalty 太高(0.015)→ hold-direction 慢弧 weave；penalty=0 → bang-bang 塌縮。甜蜜點 = 小 penalty（~0.003）。

**修法（2026-06-20 執行）**：建 curriculum `warp_drive_single_agent_v3g`（仿 v3e 但 `PENALTY_SMOOTHNESS=0.003`，ent_ang floor 0.02 不變），從 sa2_v3f/checkpoint_210000.pt 重訓 SA3（run sa3_v3g）。0.003 ≪ 0.015，讓 bang-bang 有代價但不逼 hold-weave。

**play 端 bug（順手修，2026-06-20）**：play_rnn_car.py 原本無 `--enable_actuator_dr` 等 flag → play 一直無致動延遲（訓練卻有）。已補 flag。但延遲不一致非 sin 波主因（開延遲 330000 仍飽和）。

**乾淨基準**：sa2_v3f/checkpoint_210000.pt（iter700，空曠飽和 5%）。eval 用 stage 2 或 stage 3 皆有效（210000 在兩 stage 都乾淨）。

相關：[[finding_action_error_selfref_oscillation]] [[finding_cmd_delay_limit_cycle]] [[project_v3d_launch]]
