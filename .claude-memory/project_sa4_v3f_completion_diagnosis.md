---
name: project_sa4_v3f_completion_diagnosis
description: SA4_v3f 跑完後待辦——做碰撞 breakdown 診斷(SR 卡 ~82-83% 是否=SA4 難度天花板 vs 可改善);用戶 2026-06-22 指示「跑完做診斷」
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**待辦(2026-06-22,用戶指示「跑完做診斷」)**：SA4_v3f(run sa4_v3f_ne1024_s42,stage4,從 SA3 修好版 cont/checkpoint_180000 接,1400 iters,最終 checkpoint_420000)跑完後，**先做碰撞 breakdown 診斷再決定要不要為多 1-2pt SR 動 curriculum/reward**。

**背景**：SA4_v3f SR 高原 ~82-83%、CR ~18%、TO ~0%、sw 0.006 無 sin 波。對照之前 SA4 baseline(記憶 [[sa1_sa7_baseline_results]])：baseline ij4ee1rj 84.2%(全系列最低)、主線 partial_norm bvnct7n3 78.8%。**所以 82-83% 正常、贏主線、可能就是 SA4 難度天花板**(SA4 難在 goals 2 長導航 + dynamic 5 + 移動 goal + penalty -15，一次跳太多)。v3f 額外解了 sin 波 = 淨贏。

**診斷步驟(治本前提，勿盲調 reward——見 [[feedback_regression_over_symptom]] [[feedback_reward_analysis_rigor]])**：
1. 跑 SA4 場景 deterministic play(CHARGE_USE_ACT_HIST=0，play_rnn_car.py，stage 4，含 dynamic 5 + 移動 goal + walls)，印碰撞當下 LiDAR/障礙最近距 + 速度。
2. 拆 18% CR 死因：撞**動態** vs 撞**牆** vs **追移動 goal** 失誤 vs **stuck/retreat**(ablation metrics: stuck_count / retreat_ratio)。
3. 依診斷選槓桿(每個獨立實驗)：curriculum 跳太陡→插中間子階(先 goals 4 或先靜態 goal)；撞動態為主→避障 reward；只是訓練不夠→延長 timesteps(最省)；adv_norm(v3f 用 mean_only，baseline 用 full 拿 84.2%)。

**注意**：診斷前確認是否值得為 1-2pt SR 動手——v3f 已贏主線且無 sin 波。

---

**✅ 診斷完成（2026-06-22 21:49，deterministic play 2019 回合，stage4，CHARGE_USE_ACT_HIST=0，lidar_r_min 0.25）**：

最終 checkpoint_420000：deterministic **SR 88.5% / CR 11.5%（撞牆 0.6% + 撞障礙 10.9%）/ TO 0%**。注意 deterministic CR 11.5% ≪ training stochastic CR 17.3% → 部署用 deterministic 更乾淨。已贏 baseline ij4ee1rj 84.2% + 主線 78.8%，且無 sin 波（target ω 空曠用小修正 0.44-0.66，近障礙才滿舵=合理避障）= 淨贏。

**撞障礙 220 例死因拆解（單一死因，撞牆已解）**：
- 速度：**89% 行進中撞（>0.4 m/s，平均 0.667）**，stuck/retreat 僅 0.5%（1 例）→ ❌ 排除 stuck/retreat。
- LiDAR 當下：**97% 在 ≤0.5m 才到碰撞距（33% <0.3m 極近）**，僅 3% >0.5m → 不是「看到不避」是「**來不及避**」。
- 目標距：**100% 在 ≥2m 行進中段撞，0% 近 goal 纏鬥** → ❌ 排除追移動 goal 失誤。
- 步數：70% 中段導航（9-40步），7% 剛 spawn，23% 後段。
- 障礙 mix：static 0.375 / patrol 0.625（**動態為主**）+ near_goal=1。

**死因定案 = 行進中對動態(patrol)障礙的「晚偵測/反應距離不足碰撞」**：全速 0.67 m/s + dt=0.2s + ω_max=1.2 + LiDAR 單幀（看不到障礙速度方向）→ 障礙進碰撞走廊到偵測只剩 ≤0.5m，剩餘距離轉不開。屬 SA4 難度天花板（patrol 動態 × 反應 horizon），**非 reward 失衡**。

**改善槓桿（每個獨立實驗，勿盲加碰撞 penalty——見 [[feedback_regression_over_symptom]]）**：
- **A.（最對症/最省）clearance-gated 預測性減速**：動態障礙進中距(0.5-1.0m)主動降速換反應時間，直接命中「89% 全速撞」。可用既有 safety shield soft（d_safe<1.2 線性降速）或 reward 端 clearance-gated speed penalty。
- B. adv_norm mean_only→full（baseline 用 full 拿 84.2%）。
- C. 把 aux 預測的 topk obstacle 速度餵進 RL policy（補單幀 LiDAR 看不到速度方向；動架構大改）。
- D. 延長 timesteps（70% 撞中段，但 SR 已高原，邊際效益低）。

**關鍵決策點（待用戶）**：A 的代價是減速 = trade SR vs 通行效率。v3f 已贏所有 baseline 且無 sin 波，**先問是否值得為避障再犧牲速度**再動手。
