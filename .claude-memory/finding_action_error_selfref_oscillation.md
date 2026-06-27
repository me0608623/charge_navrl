---
name: finding_action_error_selfref_oscillation
description: v3d action_error obs 是自我參照的延遲殘差，是長 goal sin 波殘留的根因；p95_flip 測不到低頻 weave
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**⚠️ 重大更正（2026-06-15，sa3_v2 baseline 對照後）：flip rate 是「反指標」，smoothness penalty 才是 sin 波主因。**

決定性對照：`sa3_v2_ne1024_s42_v2/checkpoint_90000.pt`（v2 stage3, 79D 無 act_hist, penalty_smoothness=0）用戶確認「乾淨」，同場景 play 結果 **SR 95% 前進、28 步到達，但 flip p95=0.362（高！）**。v3d sin 波卻 flip p95=0.127（低）。→ **flip rate 與視覺 sin 波相反**：
- **高 flip（sa3_v2 0.36）= 每步快抖 → 平均抵銷成直線 → 乾淨 ✅**
- **低 flip（v3d 0.13）= 持住同方向幾步才換 → slew 累積成大弧 → 慢頻大 weave = sin 波 ❌**

**真正根因 = penalty_smoothness（v3 起新增）**：它罰 |Δratio|（角速度改變）= 直接罰 flip → policy 學會「不翻、持住」→ 低 flip → slew 累積成慢 sin 波。v2 penalty_smoothness=0 → 高 flip 快抖 → 乾淨。act_hist/action_error（給上一步指令鼓勵持住）為加強因。**「三件套 deterministic p95_flip 0.127=乾淨」是假成功，0.127 低 flip 正是 sin 波簽名（之前把指標看反）。** 修法 = 移除 penalty_smoothness（主）+ act_hist（次），非加懲罰。sin 波正確指標應為「低頻 heading 振盪週期/path 曲率」，不是 flip rate。

**SA1_v3e 驗證結果（2026-06-15，方案 A：penalty_smoothness=0、保留 act_hist action_error，from scratch）**：
- SA1_v3e 完訓飽和 SR ~96%。checkpoint_60000.pt(iter200) deterministic play（同 sa3_v2/v3d 長 goal 場景，r_min0.2/ω_max1.2/action_error）：
- **✅ 明確勝利：bang-bang 滿舵消失** —— 目標|ω| **avg 0.5 << max 1.2**（v3d 是 avg=max=1.2 滿舵）。移除 smoothness 後 policy 自然學「小角速度修正」非「滿舵硬轉」→ 直接砍掉 v3d sin 波核心機制。SR 88%、前進 0.79m/s、28-30 步到達。
- **⚠️ flip p95=0.068 極低（比 v3d 0.127 還低）**：「flip 反指標」理論是用 v3d vs sa3_v2（都高 omega）校準的，v3e 改用**低 omega**破壞校準 → 純數字判不準是否還有「小振幅慢 weave」。**低 flip + 小 omega = 大致直行小修正（可能乾淨）vs 低 flip + 滿舵 = sin 波（v3d）。判別關鍵是 omega 幅度 × 是否持續，不是 flip 單獨。**
- **結論：方案 A 方向正確（滿舵消失），但「是否完全乾淨」必須 GUI BEV play 親眼看路徑**（headless 只有數字判不出小振幅慢 weave）。若仍有小 S → v3f 拿掉 action_error（速度落差）。
- 教訓：jitter 數值指標（flip/sw/|ω|_std）都會被 entropy + omega 幅度 regime 污染，**最終驗收只能靠視覺路徑**。

**SA1_v3e 完訓最終 eval（2026-06-16，checkpoint_60000.pt resume = iter900）：方案 A 成功 ✅**。
- ⭐ **關鍵決定性指標 = slew gap（差 = |目標 ω − 實際 ω|，play 診斷的第三個數字）**：v3d 差 **0.5~1.8（大）** = policy 打滿舵 ±1.2 但 slew 跟不上 → 相位落差 → 極限環 → sin 波；v3e 差 **0.04~0.6（小）** = policy 打 slew 跟得上的溫和 omega → 無相位落差 → **無極限環**。這比 flip/omega 更直接判別 sin 波（極限環 = 致動延遲落差大）。
- v3e 完訓 ckpt：目標|ω| avg 0.45<<max 1.2（不滿舵）、flip p95 0.098、SR 92%、29-30 步直達。三條獨立證據（不滿舵 + slew 差小 + 高效直達）都指向乾淨。
- **結論：penalty_smoothness 確認是 sin 波單獨主因，action_error 無罪（保留仍乾淨）。** 移除 smoothness 後 policy 不再被逼持住滿舵 → 改用 slew 跟得上的小修正 → 不累積成 weave。
- 唯一保留：headless 數字無法 100% 排除肉眼極小擺動，待 GUI BEV play 最終確認。但 slew 差小是硬的反極限環證據。
- 後續：v3e 收工 → SA2_v3e 接動態；action_error 不必拿掉（v3f 不需要）。最終 ckpt = `sa1_v3e_ne1024_s42_resume/checkpoint_60000.pt`。

**🏁 決定性判決（2026-06-19，SA1_v3f 完訓 in-distribution eval）：act_hist/action_error 確認是動態 sin 波元兇，延遲 DR 無罪。**
- `sa1_v3f_ne1024_s42/checkpoint_210000.pt`（iter 700 成熟平台 SR 96.4%，79D 無 act_hist、penalty_smoothness=0、**保留延遲 DR**）跑無障礙隔離 eval（0靜0動+牆1, goal 5-6, deterministic, `CHARGE_USE_ACT_HIST=0`）。三軸全過：
  - **|ω| 地板（min）0.05~0.33**（大量 0.25-0.33）= 用細微小角修正，**和 sa3_v2(0.33-0.46) 同級甚至更低**。對比 v3f **iter 200 半成品** 地板 0.57-0.83（偏高）→ 證實 iter 200 偏高是「**訓練不足 artifact 不是延遲 DR**」。
  - **slew 差 max：154 回合中 116(75%)=0.600**（恰一步 slew，ω_max1.2/α3.0/dt0.2→單步Δω0.6，actuator 永遠1步內追上=無相位落差=無極限環）；34(22%)=1.2、3=1.8 是瞬間滿舵轉向（sa3_v2 乾淨基準同樣偶有 1.8）。對比 sin 波(v3d/v3f-iter200)=持續大差。
  - **fwd 96-100%**（不後退 weave）。
- **結論：移除 act_hist（v3f→79D）+ penalty_smoothness=0、保留延遲 DR → policy 乾淨。延遲 DR 不獨立致 sin 波 —— RNN 從 ego 實際車速序列把延遲隱式吸收。** 用戶 session 初始直覺（速度落差/觀測維度有問題）+「先回歸找改了什麼別加懲罰」方向完全正確。
- **乾淨完訓 ckpt = `sa1_v3f_ne1024_s42/checkpoint_210000.pt`**（可作部署版）。下一步 SA2_v3f 動態驗證（wd_sa2_v3f.yaml，從此 ckpt 接 stage 2，CHARGE_USE_ACT_HIST=0）。
- 註：eval 撞牆/撞障礙率高(35%)是 goal5-6m+牆+deterministic 的場景/距離 artifact（撞邊界牆），與 sin 波無關。
- play_rnn_car.py 修了 7D-state 裁切 bug：`CHARGE_USE_ACT_HIST=0` 時 env 已 79D，不可再裁 act_hist（否則 79→75=雙重移除→`size mismatch net.0 [256,91]vs[256,87]`）。修法見 play_rnn_car.py ~1986 行 `_env_act_hist_removed`。
- ⚠️ SA1_v3f 訓練中 CUDA wedge stall 一次（iter 720，`_grad_l2_norm:685`+GPU0%）→ kill+從 checkpoint_210000 resume，warp 載入 10 iter 內 SR 回 96.6%（resume 計數器歸零但權重 warm）。

**⚠️ slew 差判別法更正（2026-06-17，sa3_v2 對照 play）**：不能只看 slew 差的 **max**。sa3_v2（乾淨 baseline）play 25 回合，slew 差 max 照樣摸到 1.0~1.8（瞬間轉向對準 goal 時 actuator 跟不上，乾淨/sin 波都會出現）。真正的乾淨簽名是三條同時成立：(1) **fwd% 96~100%（不後退 weave）**，(2) **|ω| 不飽和**（min 降到 0.33~0.46，會用小角速度修正，非 avg=max=1.2 滿舵 bang-bang），(3) **高效直達（~23-37 步）**。sin 波（v3d）反之：|ω| 持續飽和 1.2 + 低 fwd% + 持續大差（極限環）。→ **判別關鍵 = 持續飽和×低fwd%×持續大差，不是瞬間 max 差**。sa3_v2 三條全中 = 乾淨確認。這也是 v3f 完訓後的驗收標準。

**⚠️⚠️ 重大更正（2026-06-17，SA2_v3e 動態 eval）：penalty_smoothness=0 不足，sin 波有兩個獨立元兇，act_hist/action_error 是動態的元兇。**
- SA2_v3e（penalty_smoothness=0、**有** act_hist action_error、動態 stage2）完訓後 **無障礙隔離 eval**（0靜0動+牆1，純直線遠 goal，障礙最近=無）：**每回合目標|ω| avg=max=1.20 滿舵 + slew 差 0.32-1.80（大）= sin 波極限環**。和 SA1_v3e 靜態的乾淨（avg0.45/差0.04-0.6）完全相反。
- **三方對照揭真相**：sa3_v2（**無**act_hist, penalty=0, 動態）=乾淨；SA1_v3e（有act_hist, penalty=0, **靜態**）=乾淨；SA2_v3e（有act_hist, penalty=0, **動態**）=sin波。→ **差別就在 act_hist/action_error + delays**。靜態場景不觸發 act_hist 自激振盪，一進動態就爆。
- **修正前的「penalty_smoothness 是唯一/單獨主因」錯誤**：sin 波有 2 個獨立元兇 ——(1) penalty_smoothness（靜態也會），(2) **act_hist/action_error 自我參照（動態才現形）**。SA1_v3e 靜態驗證乾淨是假象（靜態遮住了 (2)）。
- **用戶 session 初始的直覺（action_error/速度落差有問題）正確**，被 SA1 靜態假象誤導以為只是 smoothness。
- **下一步 v3f**：拿掉 act_hist/action_error（→ raw 或移除 act_hist 回 79D，≈ sa3_v2 乾淨 baseline），penalty_smoothness 維持 0，從頭重訓，動態 eval 用無障礙隔離（0靜0動）看 slew 差確認。

---
（以下為更正前的初步診斷，部分仍成立但 p95_flip 解讀已反轉）

**2026-06-15 發現：v3d「三件套」沒有真正解決 sin 波**。用戶把 goal 距離拉長 play `sa2_v3d_ne1024_s42_resume/checkpoint_60000.pt`，sin 波重現。診斷出兩層根因：

**1. bang-bang 滿舵 + slew 延遲 = 極限環（control theory）**
- 診斷數據：`目標角速度 avg=max=1.20=ω_max` → policy 每步都選最極端 angular bin（ratio=±1），從不用接近 0 的 bin。
- ω_max=1.2, α_max=3.0, dt=0.2 → 每步最大 Δω=0.6 → 全翻轉需 4 步 → weave 週期 ≈ 8 步 ×0.2s = 1.6s = 視覺 sin 波。
- **p95_flip 盲區**：它測「每幀符號翻轉率」（高頻抖），週期 8 步的慢 weave 每幀 Δ 小 → flip≈3/28≈0.13（對上 deterministic eval 的 0.127「乾淨」假象）。flip_rate 與 episode 長度無關 → 拉長 goal → weave 次數變多視覺更糟，但指標不變 → **p95_flip 測不到低頻 sin 波**。smoothness penalty 罰的也是 Δratio（變化），同樣漏掉。

**2. action_error obs 自我參照 = 內建振盪器（用戶洞察，根因）**
- `_actuator_tracking_error[:,1] = (_w_intended − actual_angular_vel)/ω_max`（discrete_differential_drive.py:243-250）。`_w_intended`=slew 後，`actual`=經 actuator DR(motor_lag=0.5 一階延遲+velocity_scale) 後。註解明寫「DR 關閉時 err=0」。
- **err 是「延遲殘差」不是「追蹤失敗」**：馬達會到達指令，只是晚 1-2 步（用戶：「可以達到但 Delay」）。當 error 餵 policy 語義就錯。
- **err 是隨機 DR 噪聲**：來自 actuator_velocity_scale[0.97,1.03]+motor_lag 隨機 DR，train/play 分布不一致 → 部署失效。
- **最致命：obs 自我參照**。act_hist 尾巴 `[a_{t-1},ω_{t-1},err_lin,err_ang]` 在 obs dims 79-83，餵進 state branch。policy 輸出 → 下一步自己 input。RNN 易學成「上step +ω 且有 lag → 這step −ω 補償」= 字面上的延遲回授振盪器，無外部環境也自激 sin 波。這就是用戶說的「觀測維度的問題」。

**回歸時間線**：sin 波嫌疑全在 v3 系列「往角速度迴路加延遲」的改動，不是 reward：
- 56c8d5961e(v3): ω_max 2.0→1.2 + actuator delay DR + smoothness
- 4a28cab735(v3b): obs_delay [0,1]
- da28775bbb(v3c): act_hist 4D（原 sin 波觸發）→ v3d 改 action_error 編碼仍沒解
- **v2（sa1_v2_ne1024_s42 等 ckpt 還在，79D 無 act_hist/無 delay/ω_max=2.0）= 乾淨對照組**。

**正確修法方向（root，非加 reward 懲罰）**：移除/重設計 act_hist obs — 拿掉 action_error（延遲當 error 的錯誤編碼），可能整個 act_hist 拿掉，讓 RNN 從 ego 速度序列（真 odom，非自我參照）學延遲補償。obs 83D→79D = 從頭重訓。先用 v2 ckpt 長 goal play 做回歸確認再動 code。

相關：[[finding_cmd_delay_limit_cycle]] [[finding_p95flip_entropy_confound]] [[project_v3d_launch]] [[feedback_regression_over_symptom]]
