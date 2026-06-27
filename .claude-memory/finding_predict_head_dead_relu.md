---
name: finding_predict_head_dead_relu
description: RNN velocity-aux 的 predict_head 末端有 nn.ReLU()→dead-ReLU 死亡螺旋→輸出恆0→aux 零梯度→RNN 從未學會預測障礙運動;v3f(可能更早)整條鏈的「運動預測」一直是死的;這是「晚反應撞動態障礙」真根因,非 reward 問題
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**根因定案（2026-06-23，用戶質疑「有 RNN 為何還晚反應」後逐層查出）**：`scripts/.../skrl/models/modular_rnn_models.py` 的 `predict_head` 末端有一個 `nn.ReLU()`（原註解「WD-style preprocess_info_back」照抄）。但 aux target 含**有正負**的 body-frame 障礙物位置(x/y)與速度(vbx/vby，障礙往後/左=負)。ReLU 強制輸出 ≥0 → 加上 **dead-ReLU 死亡螺旋**（pre-activation 全≤0→ReLU 全夾0→通過 dead ReLU 梯度=0→永遠救不回）→ **predict_head 對任何輸入輸出恆 0**。

**證據鏈（全部一致）**：
- checkpoint 比對 SA4(iter1399)→SA5(iter199)：policy_head Δ~1e-2(正常訓練) vs RNN/predict_head Δ~1e-5(凍結) vs extractor Δ=0(aux_lr_extractor=0 設計)。
- WandB：aux/predict_head_param_norm 全程 2.2243 不動、rnn_param_norm 6.6355 不動、aux grad≈0、aux/loss_per_step 負值(-0.09)且不改善。
- aux 解碼 eval（play_rnn_car.py 改 guard predict_dim>=7 + 擴充 print_aux_debug 解碼速度維 dims7-12）：predict_head 輸出**全 0.00**（連正的 d=2.54 都預測 0）、速度預測**相對誤差 100%**(MAE≈真實速度,n=768)。
- 「負 loss 不改善」真相：loss=log(|0−target|)=log(|target|)=常數，根本不是有意義的預測 loss。

**機制串連**：predict_head 死→aux 零梯度→不回傳給 fc_middle/rnn/fc_front→**RNN 的 12D preprocess_feat 從未被監督去編碼障礙運動**→RL policy 拿到的特徵沒有「障礙要往哪走」的資訊→只能反應當下單幀 LiDAR(72D,無速度)→**晚反應/全速撞動態障礙**（SA4 診斷的 89%全速撞、97%≤0.5m才偵測 = 這個的直接後果）。⚠️ obs 79D 不含 obstacle 60D，actor 看不到障礙位置/速度,動態預測**唯一**來源就是這個壞掉的 RNN aux。

**修法（2026-06-23 執行中驗證）**：移除 predict_head 末端 ReLU，保留 `nn.Sequential(nn.Linear(...))` 結構使 state_dict key 仍為 `predict_head.0.*`（相容舊 checkpoint）。預期：去 ReLU 後舊 Linear 權重直接輸出負 pre-activation→梯度恢復流動→aux 復活→RNN 重新學預測。驗證 run=sa5_v3f_relufix_verify(從 sa4 checkpoint_420000,timesteps 18000~60iter)，看 aux/predict_head_param_norm 是否脫離 2.2243 + loss 是否真收斂 + 重跑解碼相對誤差是否<100%。

**重大含意**：v3f（甚至更早 v2/v1，需查 git blame）的「velocity aux」**從未真正運作** → 之前所有「RNN 幫 policy 預測動態」的假設都不成立。SA4「晚反應」診斷的真根因是這個，不是 reward。**SA5 clearance-gated reward 是治標，dead-ReLU 才是治本**（見 [[project_sa5_v3f_react_design]] [[project_sa4_v3f_completion_diagnosis]]）。修好 aux 後 policy 才可能真正提前避障。

**✅ 修復後驗證數據（2026-06-23，兩階段）**：

階段一 — 從 SA4 ckpt 接、去 ReLU 短驗證（run sa5_v3f_relufix_verify, iter1→10）：predict_head_grad_norm 0.0→**0.20-0.26**、rnn_grad 0.0→**0.03-0.08**、param_norm 2.2243凍結→**2.2217在動**、preprocess_loss −1.5平→−0.07動。**aux 確認復活。**

階段二 — 整條從 SA1 從頭重訓（run sa1_v3f_reluFix_ne1024_s42, wandb 4frp0om5；**同時也解凍 extractor**:aux_lr_extractor 0→0.0005,因從零訓練本不該凍隨機 init,非實驗賭注。已改全 v3f 鏈 config）：iter~20 時 predict_head_grad **0.30-0.42**、rnn_grad **0.04-0.11**、**extractor_grad 0.08-0.32(解凍生效,死時亦為0)**,三模組全在學。preprocess_loss 從死時退化負常數(-1.5)變回**正的真實預測 loss(2.4-3.2)**。

⚠️ 但 iter~20 太早,preprocess_loss 還在 2.4-3.2 雜訊區**尚未明顯下降**;「RNN 真的學會預測」的最終驗收(loss 持續降 + aux 解碼相對誤差從 100% 往下掉)要等訓練數百 iter + checkpoint_30000(iter100) 才能跑解碼 eval。**目前只確認「aux 活著在學」,未確認「學得好」。** 解碼工具:play_rnn_car.py 已改(guard predict_dim>=7 + print_aux_debug 解碼速度維,累計相對誤差;見本檔證據鏈3)。

**🔬 SA1 iter100 解碼複驗（2026-06-23）**：dead-ReLU **結構修好確認** — predict_head 7D geometry 輸出從死時 exact 0.00 變成非0且有負值(近1 x=-0.03/y=-0.20/d=0.01)→ ReLU 不再夾死。**但速度預測相對誤差仍 100%**、且預測幾乎是常數(不隨輸入變)→ iter100+SA1稀疏障礙下 RNN 只學到「預測均值」沒學會「追蹤障礙」。**「頭活了」≠「學會預測」**：修 ReLU 是必要非充分。真正驗收 SA3-SA5 密集障礙下 rel err 是否 <100%；若仍卡 100% → vanilla RNN 容量不足,需 GRU/加 hidden/調 aux(下一層問題)。詳見 [[project_reluFix_retrain_log]]。

**🔬 SA1 iter300 解碼複驗（2026-06-23）**：**與 iter100 同結果 — 相對誤差仍 100%**(MAE 0.734≈真實均 0.734,n=1600)、預測仍近似常數(近1 x≈-0.02/y≈-0.30 不隨 step 變)。★重要校正：preprocess_loss 從 3.0→2.45 的下降**不代表學會追蹤**,只是「把常數均值擬合更準(variance 收斂)」——loss 降 ≠ rel err 降。SA1 稀疏障礙(dynamic 1-2)下 RNN 註定只學均值。**結論不變但更篤定:reluFix 結構必要性已證,但「RNN 是否真學會追蹤」的最終裁決完全押在 SA3-SA5 密集障礙的 rel err。** SA1 後續 iter600/完訓解碼大概率仍 100%,監控重心移到 SA3+。

**🔬🔬 SA3 iter500 解碼裁決（2026-06-25）— 密集障礙首測，命中「容量不足」分支**：SA3(dynamic 3-4 + head_on 0.20 正面來車 + 500 iter)解碼 **velocity rel err 仍 100%**(MAE 0.748≈真實均 0.747,n=2400+,predict_dim=13 含速度 dims 7-12 確有訓練)。位置 dims signed nonzero 但嚴重低估(近1 預測 x−0.00/y−0.31 vs 真實 x+0.76/y−1.43),速度 dims 功能性死(≈0)。**這正是本檔 iter300 條目預留的裁決點**：「若 SA3-SA5 密集障礙 rel err 仍 100% → vanilla RNN 容量不足,需 GRU/加 hidden/調 aux」——**SA3 命中此分支**。★重大修正:reluFix 監控期間「aux/preprocess_loss 深負=RNN 在學追蹤」的樂觀讀法**錯誤**,深負是位置 dims(權重主)變小誤差驅動,非速度追蹤(target-scale 混淆);**aux loss 負 ≠ tracking,解碼 rel err 才是黃金標準**。導航卻健康(det SR 83.8%/CR 16.2%)=policy 純 LiDAR 反應式避障,不靠預測。**結論:reluFix 結構必要性已證(頭活了),但「RNN 學會追蹤」在此架構/容量下未達成;晚反應動態的治本需架構升級(GRU/hidden↑/aux 重設計),非單純去 ReLU。** 詳見 [[project_reluFix_retrain_log]]。decode-gate 未通過,SA4 暫停等用戶決策。

**SA chain reluFix lineage 進度**:SA1→SA2→SA3(✅解碼 rel err 100%,容量不足裁決)→SA4(暫停)→SA5。每棒接前棒 checkpoint_270000(SA1)/各 stage final。修法檔案:modular_rnn_models.py(predict_head 去 ReLU)+ 5 個 v3f config(aux_lr_extractor 0→0.0005)。

相關：[[feedback_regression_over_symptom]]（先找根因別壓症狀，這次正是）[[design_obstacles_60d_unnecessary]]（actor 不看 obstacle 60D，動態靠 RNN）[[project_sa5_v3f_react_design]]（SA5 clearance reward 是治標,本 finding 是治本）
