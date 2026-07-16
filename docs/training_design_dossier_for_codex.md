# Charge-Car RL 訓練設計 Dossier — 給 Codex 審查

> **目的**：使用者懷疑「網路架構 / 資訊流 / 梯度更新」有結構性問題（別人論文 reward 簡單卻能學會避障，我們複雜卻學不會）。請 Codex 據此審查以下彙整，找出設計缺陷，再協助重設計 reward 並從 **SA1 從頭重訓**。
> **背景結論（已用大樣本 det eval 坐實）**：現行政策在真實部署密度（12×12 / 20 靜動態 = 0.139 obs/m²）SR≈68%、撞障礙 26%（動態 15.6% > 靜態 11.7%）；政策**不會在 1.5–2.0m 主動朝安全方向繞開**，安全方向一致率僅 ~58%（≈亂猜），reward-shaping（r_arc / gap / TTC / 多幀 / velocity-aux）多次嘗試都無法撼動避障行為。
> 所有 file:line 為 `/home/aa/IsaacLab` 相對路徑；已由程式碼實查（非臆測）。

---

## TL;DR — 最可疑的 4 個根本問題（依重要性）

| # | 問題 | 位置 | 為何致命 |
|---|------|------|----------|
| **P1** | **RNN/extractor 收不到 RL 梯度**，只由 aux optimizer 訓練 | rollout `no_grad` `train:3763`；RL 讀快取 `flat_ri` `train:4549`；aux-only opt `train:3129-3139` | actor 的時序記憶(12D)品質 100% 靠 aux；aux 一塌，actor 只剩反應式，RL 無法修正。標準 PPO 是 RL 端到端穿 encoder，這裡切斷了。 |
| **P2** | **reward 純稀疏、goal 經濟學壓倒避障**：+40 goal / −15 撞，無任何 pre-contact 避障梯度 | reward 表 `wd_single_agent_v3.py:401-407`；impl `rewards.py:76-201` | 撞的代價 < 1/3 goal，goal 會 respawn，timeout 免費(SA4=0) → EV(衝+偶爾撞) > EV(謹慎繞)。避障沒有任何 shaping 梯度可學。 |
| **P3** | **梯度更新無 trust region**：KL early-stop 是死碼、a2c 單次 full-batch、`max_grad_norm` 被繞過、mean_only advantage 保留原始量級 | KL 死碼 `train:4558/4686` 但無 break；clip 繞過 `train:4634-4641`；adv `train:4443-4445` | 每個 rollout 一次無界大步更新 → 已知崩潰模式(trust-region breakdown)。clip ε=0.2 是唯一護欄，擋不住大 LR full-batch 單步。 |
| **P4** | **aux target 可能在 runtime 退化成常數**（速度=0 → next≡t0；無動態→常數 10.0）+ predict_head 僅 `Linear(12→13)` 低容量 | target `wd_aux_targets.py:350-370, 439-447`；head `modular_rnn_models.py:281-287` | 若 aux target 退化，P1 的唯一資訊管道跟著失效 → RNN 塌成常數（歷史 finding 已見）。 |

**綜合診斷假說**：避障學不會，**不是 reward 不夠花俏，而是 (a) 承載障礙動態的唯一管道（RNN→12D feat）被 RL 梯度切斷、且靠會塌的 aux 撐（P1+P4）；(b) reward 根本沒給 pre-contact 避障梯度（P2）；(c) 優化不穩(P3)。** 建議重設計方向見末節。

---

## Part 0 — 路徑與執行

| 元件 | 路徑 |
|------|------|
| 訓練入口 | `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py` (~4700 行) |
| 模型定義 | `scripts/reinforcement_learning/skrl/models/modular_rnn_models.py` |
| 實驗 config | `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa4_deploy_dense.yaml`（+ `_distr.yaml`）|
| reward 實作 | `source/isaaclab_tasks/.../charge_skrl/mdp/rewards/rewards.py`（`WDSparseReward`/`compute_wd_charge_reward`）|
| reward 參數(逐階段) | `.../charge_skrl/curriculum/phases/wd_single_agent_v3.py`（SA1-8 STAGES）|
| curriculum(密度/場景) | `.../curriculum/phases/wd_single_agent_v3e_deploy_dense.py`（只覆寫 scene/behavior/密度/LR，**不動 reward**）|
| obs 建構 | `.../charge_skrl/mdp/observations/{obs_functions.py, functions.py}` |
| aux target | `scripts/reinforcement_learning/skrl/utils/wd_aux_targets.py` |
| privileged critic | `scripts/reinforcement_learning/skrl/utils/privileged_obs.py` |

執行：`PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 ./isaaclab.sh -p <train> --experiment_config wd_sa1_v3f_vaux --headless ...`

物理：v_max=1.0, a_max=0.5, ω_max≈1.2, dt=0.2s, body_radius=0.35, r_min(LiDAR 盲區)=0.5。動作=MultiDiscrete([19,19]) accel×ω。

---

## Part 1 — 網路架構

**⚠️ config 覆寫了程式碼 docstring/預設**（docstring 說 hidden=30/vanilla-RNN，實際見下表；照 docstring 讀會理解錯）。

| 參數 | wd_sa1_v3f_vaux | wd_sa4_deploy_dense |
|---|---|---|
| hidden_dim | 64 | 64 |
| rnn_type | **GRU** | **RNN(relu)** |
| predict_dim(aux) | 7(pos-only) | 13(7+3×2 vel) |
| critic | asymmetric(priv 50D) | asymmetric(priv 50D) |

資料流：`obs 79D → extractor 96D →(feat_norm)→ RNN(64) →concat fc(128)→ fc_middle → preprocess_feat 12D`；`rl_input = obs79 ⊕ feat12 = 91D → policy/value heads`。

- **Extractor** `LidarStateExtractor` `modular_rnn_models.py:73-186`：
  - LiDAR 分支：`Conv1d(1→32,k5,circular)→(32→64,k5,s2)→(64→64,k3,s2)` +ReLU → flatten1152 → `Linear→64` + LayerNorm。
  - State 分支(7D=ego4+goal2+time1)：`Linear(7→32)→(32→32)` +LayerNorm → 32D。
  - concat = **96D**。
- **RNN** `PreprocessRNN:193-365`：`fc_front Linear(96→64)+ReLU → GRU/RNN(64→64,1層) → concat(rnn64,fc64)=128 → fc_middle(128→48→12) → preprocess_feat 12D`。hidden `[1,E,64]`，`update()` **detach**（`:386`）、done 歸零（`:388-391`）。
- **Policy head** `:398-441`：`91→256→256→256→512→38`，末層無激活；38 logits split 成 **兩個獨立 Categorical(19)**（accel, ω），log_prob/entropy 各自加總。
- **Value head(asymmetric)** `:444-499`：actor 91D→`rl_proj(91→128)`；privileged 50D→`priv_proj(50→128, zero-init)`；merge→trunk `256→...→1`。
- **Aux predict_head** `:281-287`：**單一 `Linear(12→predict_dim)`，無末端激活**（先前 dead-ReLU bug 已於 2026-06-23 修掉，現乾淨）。sa4 predict_dim=13。

**🚩 架構旗標**
1. **P1 梯度邊界**（見 TL;DR）：兩個 disjoint optimizer——`charge_opt_rl`(只 policy+value)、`charge_opt_aux`(rnn/predict/fc_middle/fc_front/extractor, lr=5e-4)。rollout 下 `rnn_feat` 在 `no_grad` 算好快取，RL backward 到不了 RNN/extractor。`rnn_rl_grad=False`(預設)。
2. **12D bottleneck + Linear(12→13) 低容量 aux head**：13D 帶符號速度目標由 12D 瓶頸單層線性回歸，容量偏低（配合 P1 易 under-fit）。
3. **GRU→RNN 中途換 cell（高價值待查）**：sa1_vaux 用 GRU、sa4_dense 用 vanilla relu-RNN，且 sa4 從 SA3 ckpt_30000 resume。**請 Codex 對實際 checkpoint chain 驗證 SA3 祖先是否也是 relu-RNN**；若 SA3 是 GRU，GRU→RNN resume 會 shape 相容但 recurrent 權重語意錯配/靜默丟失（GRU 有 3× gate 參數）。
4. privileged `priv_proj` zero-init：critic 起始忽略 50D 障礙通道須自學；低 LR/低 entropy resume 時可能一直沒學會用（風險非 bug）。

---

## Part 2 — 資訊流 / 觀測

**79D obs layout（已對照程式碼確認）**：`[0]=accel [1]=speed [2]=ω [3]=radius(常數0.35) [4:6]=goal(body-frame,原始公尺clamp±10) [6:78]=LiDAR(72) [78]=time`。act_hist(CHARGE_USE_ACT_HIST=0)與 LV-DOT 動態通道(預設關)皆不在 79D 內。

- **LiDAR `wd_like_sweep_72`** `obs_functions.py:360-710`：normalize = `clamp(range − r_robot(0.35), 0, r_max(20))/20`。**單次減 body_radius，無雙扣 bug**（先前 r_arc 端的雙扣已修）。r_min=0.5 盲區、z_filter=0.5。deploy 用 `lidar_no_noise:true` + 系統偏差(k·d+b + 16-ring)。🚩待查：`distractor_rate=0.002` ghost 在 env cfg 硬編（`charge_env_cfg_vlp16.py:430`），確認 `lidar_no_noise` 真的有歸零它。
- **Actor vs Critic**：actor 只有 91D(LiDAR+ego+goal+RNNfeat)；**critic 多吃 50D privileged**＝10 障礙×5D`[rel_x,rel_y,rel_dir,speed,size]`(body-frame, `privileged_obs.py:24,153-170`)。**障礙 ground-truth 只餵 critic，不餵 actor。**
- 🚩 **P1 的資訊流後果**：障礙運動要傳到 actor **只能經 RNN 的 12D feat**，而該 feat 只被 aux 塑形（無 RL 梯度）。aux 若編不出運動，actor 就是純反應式，critic 再準也救不了 actor。

**Aux target** `wd_aux_targets.py:290-481`：最近動態障礙的 body-frame 7D＝`[t0 now xy, next=pos+vel·0.2 xy, hist t−3 xy, center dist(權重0)]`；sa4 再加 3 障礙×(vx,vy)=13D。**target 來源＝oracle sim state**（直接讀 `scene["obstacle_i"].data.root_pos_w`，非 LiDAR）。
- 🚩 **P4 退化風險**：rule_based 下 `_obstacle_velocities` 從不寫入 → 舊碼 vel=0 → `next≡t0`、dynamic mask 全 False（歷史 `finding_zero_velocity_buffer_contamination`）。現有 finite-diff fallback（`:361-370`）；**請 Codex runtime 驗 dims[2,3](next)≠[0,1](t0)**，否則 target 退化成「當前位置」＝ trivial、RNN 幫不上。另：無動態障礙時 7 維全填常數 `_FAR_DEFAULT=10.0` → 誘導常數輸出塌縮。`_DYNAMIC_SPEED_THRESHOLD=0.01` 在 finite-diff 量化雜訊下可能誤判動/靜。
- **RNN hidden flow 乾淨**：store-then-reset 次序正確、done 歸零、aux 序列不跨 episode、`_lidar_hist`/障礙速度快取 done 重置。無跨場污染。唯一設計後果＝`update()` detach → RL 不 backprop RNN（＝P1）。

---

## Part 3 — Reward 設計（deploy_dense SA4；純稀疏）

reward profile = `wd_sparse`（`compute_wd_charge_reward` `rewards.py:76-201`）。deploy_dense curriculum **不覆寫 reward**，故 SA4 reward 直接來自 base v3 `SA4_spatial_plan`（`wd_single_agent_v3.py:401-407`）。

| Term | 權重(SA4) | 公式/閘 | file:line |
|---|---|---|---|
| goal_reward | **+40.0** | 到達 goal 的稀疏終端 | v3:404 / rewards.py:127 |
| wall_hit | **−15.0** | 撞牆 | v3:402 / rewards.py:131 |
| obs_hit | **−15.0** | 撞障礙(動+泛型；靜/動同價) | v3:402 / rewards.py:135 |
| other_death | −15.0 | 翻覆/爆炸(罕見) | rewards.py:139 |
| action_reward(cost_operate) | +0.03/fps | **零動作的正獎勵(idle bonus)**，非成本；每 alive 步累加 | v3:405 / rewards.py:143-151 |
| smoothness | −0.005 | −0.005·\|Δratio_ang\| | v3:404 / rewards.py:157 |
| timeout | **0(關)** | SA4 無 penalty_timeout | rewards.py:197 |
| speed_near_obs(teardrop) | **0(關)** | clearance-gated 減速稅 | rewards.py:179-194 |
| r_arc(swept-arc) | 預設關 | 見下 | swept_arc.py / train:3853-3875 |

**🚩 reward 經濟學（P2 根因）**
- **goal +40 = 撞 −15 的 2.67 倍**；撞比 1/3 goal 還便宜、goal respawn、SA4 timeout 免費 → **EV(衝+偶爾撞) > EV(謹慎繞)**。
- **SA4 完全沒有 dense 避障梯度**：唯一障礙訊號是接觸瞬間的 −15，**接觸前零梯度**教它減速/轉開（`rewards.py:167-176` 註解自承）。
- `cost_operate` idle bonus 進一步偏向不作為。
- **無 potential_progress / 距離 shaping**（純稀疏，無 PBRS）。
- **r_arc** 是唯一曾提供 pre-contact 方向性梯度的機制，但預設關、且經 A/B 證實無效（steer-not-slow：政策靠多轉滿足、不減速；且方向一致率沒提升），已決定移除。

---

## Part 4 — 梯度 / PPO 更新

config `algorithm:a2c` → `use_a2c=True`：**單 epoch、full-batch、無 minibatch**，但**仍套 PPO clipped surrogate**（"a2c" 指無 minibatch，非無 clip）。

損失（`train:4560-4598`）：
```
L = clamp_p(L_π) + c_v·clamp_v(L_V) − (α_lin·H_lin + α_ang·H_ang)
L_π = −E[min(ρA, clip(ρ,1±0.2)A)] ,  ρ=exp(logπ_new−logπ_old)
L_V = MSE(V, V_target)   (無 value clipping)
```
| 參數 | 值 | 備註 |
|---|---|---|
| epochs / minibatch | 1 / full-batch(300×1024≈307k) | use_a2c |
| clip ε | 0.2 | 唯一 trust-region 護欄 |
| vf_coeff | 0.5 | |
| ent_coeff lin/ang | 0.04 / 0.08 | |
| γ / λ | 0.984 / 0.97 | |
| normalize_return | false（改用 PopArt β0.99） | |
| adv_norm | **mean_only**（減均值、**不除 std**，保留原始~40量級） | |
| max_grad_norm | 0.5（**被 wd_update_clip 繞過、無效**） | |
| 實際 grad cap | actor L2≤**8.0** / critic≤**30.0**（分開，非 merged） | |
| loss clamp | policy_loss_clamp=20 / vf_term_clamp=8（vl 動態依\|pl\|縮放，非標準） | |
| target_kl | 0.0（**parse 了但全程未使用、無 break**） | |
| lr(RL head) | 2e-4，無 decay；aux/rnn 另一 optimizer lr 5e-4 | |
| rollout / envs / steps | 300 / 1024 / 270k | |

**🚩 梯度旗標（P3 根因）**
1. 🔴 **KL early-stop 是死碼**：`_approx_kl` 算了、log 了，但**無 break 用它**；`target_kl` 全程未被引用。單 full-batch 步也無從 early-stop。
2. 🔴 **full-batch 單步 + mean_only(原始量級) advantage + max_grad_norm 被繞過** → 每 rollout 一次無界大步；已知崩潰模式(trust-region breakdown，健康爬到~71% 才斷崖、可複現)。
3. 🟠 `max_grad_norm=0.5` 死的；有效護欄＝grad 8/30 + loss 20/8。審查勿假設 0.5 生效。
4. 🟠 非標準動態 value-loss clamp（`vl` 依 |pl| 縮放）耦合 critic/actor loss 量級。
5. 🟢 GAE truncation bootstrap(fix#1) 與 entropy 符號正確；PopArt denorm 尺度一致（`normalize_return=true` 那條才是尺度錯配 bug，此 run 已避開）。

---

## Part 5 — 先前調查發現（記憶彙整，避免 Codex 重推）

- `finding_reward_economics_determines_style`：避障風格由 reward 經濟學決定；gap/膨脹/多幀等前端 shaping 都不動 d_safe。
- `finding_critic_actor_scale_mismatch_normalize_return`：`normalize_return=true` = critic/actor 尺度錯配（此 run 已用 PopArt 避開）。
- `finding_ppo_crash_trust_region_breakdown`：連兩崩根因＝trust-region 破裂(非 RNN 容量)；三放大器＝a2c 無 KL early-stop + WD clip 繞過 merged clip + adv mean_only。→ **對應 P3**。
- `finding_predict_head_dead_relu`：aux predict_head 末端 ReLU→dead→輸出恆 0（**已修**）。
- `finding_velocity_aux_bottleneck`：probe 證障礙位置在 obs、過 extractor 仍在，但**進 RNN hidden 被洗成常數**；試遍 posonly/Huber/skip/GRU/WD凍結都沒讓 RNN 編碼位置 → **對應 P1+P4，velocity-aux 此路在當前架構未證可行**。
- `finding_ttc_velocity_channel_unused` / r_arc A/B：顯式速度稅與 action-conditioned r_arc 都無法讓政策提早繞（steer-not-slow / 速度欄未被用）。

---

## Part 6 — 請 Codex 優先驗證/裁決的問題

1. **P1 是否為刻意設計且該保留？** 標準 recurrent PPO 是 RL 端到端訓 encoder。此設計把 RNN 從 RL 切開、只靠 aux。請評估：(a) 是否讓 RNN 接受 RL 梯度（`--rnn_rl_grad` 已存在但關）；(b) 或移除 RNN、改用簡單 frame-stack + 端到端 RL（多數論文做法）。
2. **P4 runtime 驗證**：跑一小段，dump aux target dims[2,3] vs [0,1] 是否相異（finite-diff 有無 firing）、無動態時是否常數 10.0。若退化 → P1 管道實質失效。
3. **P3 修復**：加 KL early-stop 的實際 break、改 minibatch(≥4)、adv mean_std、或讓 max_grad_norm 真的生效。評估對崩潰的影響。
4. **GRU→RNN resume 語意**：驗 checkpoint chain cell type 一致性。
5. **reward 重設計**：在 P1-P4 修好前，單純加 dense 避障 reward 是否有意義？還是必須先修資訊流/梯度？請給「從 SA1 重訓」的 reward + 架構最小改動建議（對照論文常見簡單設計）。

---

## Part 7 — 重設計起點（供討論，非定案）

使用者目標：從 SA1 重訓、reward function 先設好、對齊「論文簡單 reward 卻能 work」。候選方向：
- **資訊流**：讓 encoder 端到端吃 RL 梯度（開 `rnn_rl_grad` 或改 frame-stack + 純 RL），別依賴會塌的 aux 當唯一管道。
- **reward**：加一個乾淨的 pre-contact dense 避障項（多數論文＝到最近障礙距離的排斥/clearance potential，PBRS 形式避免改變最優策略），並調 goal/撞比例（goal 40 vs 撞 15 太偏；提高撞代價或加 timeout 壓力）。
- **優化**：minibatch PPO + KL early-stop + 標準 merged grad clip，回到穩定 trust region。
- **真實密度對齊**：curriculum 密度 ramp 終點對齊 12×12/20（0.139/m²），別停在 SA4 的 ~0.04/m²。

**部署真實密度基準（Control@180000, 12×12/14靜+6動）**：SR 67.8% / 撞障礙 25.9%(靜11.7/動15.6) / 撞牆 6.2%。動態為主凶。

---

## 走廊-橫越 curriculum injector — 驗收配方 (2026-07-16, K8-rebased)

**目的**：把部署的「近牆-橫越 herding」陷阱幾何放進 SA3-5 訓練場（12% env），讓標準 `collision −15` 直接給梯度教政策逃逸 —— 補 reward-based（future no-go）之外的場景槓桿。**K8 前提**：Codex probe 證 K4 看不清 crossing（88.75%<90%），故走廊只在 K8（lidar_frame_stack=8）stack 上訓才有意義。

**已完成 code**（branch wdclean-repro-20260429-pcB）：
- `mdp/events/corridor_crossing_geometry.py` — 純幾何 helper（走廊沿 world-Y，行人橫越走原生 X 軸）
- `mdp/events/corridor_crossing.py` — reset EventTerm `setup_corridor_crossing`（協調覆寫 2 牆 slot 0/1 轉 90°+robot+goal+1 顆 horizontal_crossing 行人；非走廊 slot 隱藏 Z=−10；fraction=0 baseline-safe）
- EventCfg 註冊在 `reset_base` 之後
- `e2e_final20_v1.py` + `goal_obstacle_curriculum._apply_stage` — `corridor_crossing_fraction` SA3-5=0.12 其餘 0.0

**待跑（GPU 空檔後，即 Codex K8 A/B 收工後）**：
1. sim smoke：`./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/smoke_corridor_crossing.py` — 斷言 2 走廊牆啟用 + 行人真的 +x 橫越（★驗 BehaviorScheduler.reset 排序沒蓋掉 slot 0）
2. K8 走廊 smoke 訓練（用 e2e_sa4_fs8_clean_antispin_control.py 為底，SA4 已帶 fraction 0.12）跑 ~200 iter 確認不崩
3. D 前測基準：`near_wall_crossing_eval.py --checkpoint <SA4 c89600> --near_wall_crossing_eval`（重現 spin_360≈7/8）
4. 完整 SA1→SA5 K8 重訓（走廊 SA3-5 active）→ 重跑 near_wall_crossing_eval 看 spin_360 是否降、first_turn_correct 是否升 + net_lateral_drift_m 是否降
**與 Codex K8+A+future 的關係**：兩者互補（scene-based vs reward-based）。若 K8+A+future D-eval 已治好 crossing，走廊可能不必訓；若只部分治好，走廊-on-K8 補足。
