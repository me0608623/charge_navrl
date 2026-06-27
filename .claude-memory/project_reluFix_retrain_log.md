---
name: project_reluFix_retrain_log
description: reluFix lineage 整條重訓(SA1→SA5)的時間序進度記錄;每監控 cycle append 詳細快照(SR/CR/aux grad/loss/解碼相對誤差);修了 predict_head dead-ReLU + 解凍 extractor 後從零重訓
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**reluFix lineage 重訓進度記錄**（修法見 [[finding_predict_head_dead_relu]]：predict_head 去 ReLU + extractor 解凍 aux_lr_extractor 0→0.0005；從零重訓 SA1→SA2→…→SA5）。用戶要求每 cycle 詳細回報並記錄（[[feedback_detailed_report_each_cycle]]）。

**run 索引**：
- SA1：run `sa1_v3f_reluFix_ne1024_s42`，wandb `4frp0om5`，PID 1647188，900 iter（270000 ts），curriculum v3e，從 scratch。
- SA2→SA5：待 SA1 完訓後依序接 checkpoint。

---

## 時間序快照

**2026-06-23 17:04 — SA1 iter ~53**
- 導航：SR 36→58→57→59→79→**82%**（iter1→50，從零快速爬升）；CR 23→39→43→41→20→**16%**（iter40 後學會避障掉下來）；TO 0-2.4%；ent 5.89→**4.22**（探索收斂）；sw 0.024→0.065（早期上升，stage1 簡單，續觀察）；fps ~4440（他人 job 競爭略低）；GPU 14.0GB。
- ★aux 活著：predict_head_grad **0.30-0.37**、rnn_grad **0.04-0.10**、extractor_grad **0.32→0.05**（皆非0；死時全0）。
- ⚠️aux/preprocess_loss **2.5-3.5 雜訊區尚未明顯下降**（iter53 太早，RNN 才開始學預測）。extractor_grad 從 0.32 降到 0.05（可能早期 conv 特徵快速收斂，續觀察是否過早停學）。
- checkpoint：尚無（首個 iter100=checkpoint_30000，屆時跑 aux 解碼量相對誤差 vs 死時100%）。
- 判定：導航學得好、aux 確認在學，但「預測學得準」尚未驗證（待 loss 降 + 解碼 <100%）。

**2026-06-23 17:16 — SA1 iter ~63**
- 導航：SR 82%@50→80%@60（高原小回落，正常）；CR 16→20%；ent 4.22→4.35；sw 0.063；fps 4349（他人 job 競爭續降但 OK）；GPU 14.0GB；etime 1:14。
- ★aux：grad 仍非0（predict_head 0.34→末0.17、rnn 0.07、extractor 0.12→0.05，皆在**縮小**）。
- ⚠️⚠️ aux/preprocess_loss **前半均2.94 後半均3.12 — 不僅沒降反略升**，63 iter 仍在雜訊區無改善。grad 在縮 + loss 不降 = 可能卡在平坦區/poor optimum。**關鍵觀察點**：iter100-200 若 loss 仍不降，疑 vanilla RNN(hidden64) 預測能力本身有限,或 aux 學習率/設定要調。目前仍算早,續盯。
- checkpoint：尚無（iter100）。

**2026-06-23 17:33 — SA1 iter ~77**
- 導航：SR 80→84%@70（續爬,健康高原）；CR ~16%；ent 4.25；sw 0.064；fps 4333（他人 job 競爭）；GPU 14.1GB；etime 1:30。
- ★aux：grad 全非0（predict_head 0.343→0.285 緩降、rnn 0.066→0.078 穩、extractor 0.126→0.060）。aux/preprocess_loss 三段 2.962/3.136/2.921 — **仍平、未降**(iter77)。
- ★場景確認：SA1_nav_bootstrap = goals10/static1/**dynamic 2(min1)**/walls0-1 → aux 預測**有 signal 但稀疏**(1-2動態,topk3 常 1-2 slot 空)。所以 SA1 loss 不降可能是「稀疏 signal 學得慢」而非「完全沒得學」。**真正的預測學習驗收在 SA3-SA5(動態多)**。
- 判定：導航健康;aux 活但 loss 未降,iter100 解碼給第一個硬數字;SA1 稀疏 signal 下 loss 慢屬可能,別過早下「RNN 學不會」結論。

**2026-06-23 17:51 — SA1 iter ~93**
- 導航：SR 80→84→85→**87%@90**（穩定爬升,健康）；CR 20→16→15→**14%**；ent 4.35→4.12；sw 0.063；fps 4318；GPU 14.2GB；etime 1:48。
- ★aux：grad 非0（predict_head 0.336→0.269、rnn 0.067→0.079、extractor 0.116→0.058）。preprocess_loss 三段 2.944/3.127/**2.855**（第3段微低,但仍 ~2.9 平,弱訊號別過讀）。
- checkpoint：即將生成（iter100,~9分）。下輪跑解碼。
- 判定：導航持續健康(SR87%);aux 活、loss 仍平(微弱下彎跡象);iter100 解碼是第一個硬數字。

**2026-06-23 18:08 — SA1 iter 100 ★第一個 aux 解碼硬數字**
- 導航：SR 86%@100/CR 14%/ent 3.95/sw 0.064/fps 4367/GPU 14GB/etime 2:05。健康。
- ★★解碼 eval(checkpoint_30000, stage1, n=704):
  - **速度預測相對誤差仍 100%**(MAE 0.832 ≈ 真實均 0.828)。
  - 但 7D geometry 預測**不再是 exact 0**:近1 預測 x=-0.03 y=-0.20 d=0.01(死時全 0.00)→ **ReLU 修復結構上生效:輸出非0且有負值(y=-0.20)**。
  - ⚠️ **但預測幾乎是常數**(不論真實 x=+1.59/+5.39/-1.76,預測都 ≈-0.03/-0.20/0.01)→ model 只學會「預測均值/bias」,**沒真的追蹤障礙** → 仍 100% rel err、對 policy 零資訊。
- **解讀(關鍵)**:dead-ReLU 已修(頭活了、有梯度、輸出 signed),但 iter100 + SA1 稀疏障礙下,RNN 只學到「預測常數均值」還沒學會「用輸入追蹤障礙」。這跟 preprocess_loss 平(預測常數=常數 loss)一致。**真正考驗:SA3-SA5 密集障礙下相對誤差會不會真的 <100%。若連那裡都卡 100% → vanilla RNN(hidden64) 預測能力不足,需架構改(GRU/加容量/調 aux)。**
- 下一步:續訓,SA1 完訓接 SA2;每階段重跑解碼追相對誤差趨勢。

**2026-06-25 02:32 — SA3 iter ~50/1400：aux 負值持續(非瞬態)**
- 導航：SR 72.7→73.7→**73.9%@50**（緩慢回升）；CR 28.2→27.1→**27.0%**（緩降,右方向但慢）；TO 0%；ent 2.4→3.07（探索升）；sw 0.013-0.014；fps 4365-4435；PID 2457352 etime 1:01h；log mtime 02:31 fresh 無 stall。
- ★★aux preprocess_loss 3seg **-0.210/-0.207/-0.193**(last5 ~-0.2,一個 outlier -0.58)→ **負值持續穿過 50 iter,非瞬態!** 強烈暗示 SA3 預測誤差真的<1(vs SA1/SA2 死平 +2.48)。grad:predict_head ~0.32、rnn ~0.04、extractor ~0.027(健康)。
- 判定:aux 突破跡象穩固(50 iter 持續負值);SR/CR 緩慢往好方向但慢。★仍需 SA3 完訓解碼 rel err 定案(可能 stage target scale 假象)。checkpoint(SA3)尚無(iter100)。

**2026-06-25 02:05 — ★★SA3 iter ~20/1400：aux loss 首次離開平台(暴跌負值!)**
- 導航：SR 69.0→72.7→**72.6%@20**（緩慢回升中,stage3+head_on 變難)；CR 27.8→28.2→**28.1%**（仍高,head_on 正面撞,policy 尚未學會避）；TO 0%；ent 2.9→2.4-2.7；sw 0.014；fps 4371-4419；PID 2457352 etime 34:59；log mtime 02:01 fresh 無 stall。
- ★★**aux preprocess_loss 3seg -0.203/-0.202/-0.250**(last5=[-0.85,0.04,-0.36,-0.12,-0.09])→ **從 SA1/SA2 死平 +2.48 暴跌到負值!** 負值=|pred-target|<1=預測誤差大幅縮小。grad:predict_head 0.38→0.30(高於 SA2)、rnn ~0.04、extractor ~0.025。
- ★解讀(謹慎):這是 reluFix 以來 aux loss **第一次離開 2.48 平台**。可能=head_on 等速直線運動高度可預測+密集障礙給 RNN 真 signal,RNN 開始學追蹤;但 iter20/n=27 太早,也可能=stage 轉換 target scale 變化的假象。★真裁決:SA3 完訓解碼 rel err 是否真 <100%(對比 SA1/SA2 都 100%)。
- checkpoint(SA3)尚無(iter100)。判定:SA3 健康、aux 出現突破跡象但需解碼驗證;CR 仍高待 policy 學避 head_on。

**2026-06-25 01:38 — ✅SA3 進訓練迴圈 + ★head_on 確認生效**
- SA3 PID 2457352 alive(etime 8:01)、GPU 子 2457367(11965MiB)、Loaded checkpoint_150000✓、新 runid **6zkvago1**(已寫檔)。
- ✅**★head_on 確認生效**:log `BehaviorScheduler created: 7 slots, mix={patrol:0.25,random_walk:0.2,static:0.15,horizontal_crossing:0.2,head_on:0.2}`,reset 正常 spawn(7 slots×envs),**無 head_on crash**(只有既有 PhysX foundLostPairsCapacity/Filter 警告,跟 head_on 無關、不影響訓練)。
- [1/1400] SR **69.0%** CR **27.8%**（★stage3 大幅變難:更多動態+牆+horizontal_crossing+head_on,SR 從 98% 暴跌、CR 暴增=正面來車先撞的預期 pattern）。ent 2.92。
- ★監控重點:SR 是否回升(policy 適應)+CR 是否從 27.8% 降(學會提早避 head_on)+aux preprocess_loss 是否 <SA2 的 2.48 開始降(密集+head_on 讓 RNN 開始學追蹤?)。SA3 跑到 +150000 steps(≈iter500)停+解碼(★rel err <100%?)+接 SA4。

**2026-06-25 01:30 — ✅SA2 解碼健康通過 → SA3 啟動(含 head_on)**
- ✅**SA2 健康確認解碼**(checkpoint_150000, stage2, n=1600):**rel err 仍 100%**(MAE 0.772≈真實均 0.771);但 predict_head **signed 非0**(近1 x=-0.05/y=-0.30、近2 x=-0.34/d=-0.34)=**結構健康、梯度有流 ✓**。stage2 障礙(動態3-4)仍不足以讓 vanilla RNN 學追蹤(只猜均值)→ 健康關卡過,放行 SA3。
- ⚠️ 接 SA3 時踩坑:(1)/tmp/sa3_v3f_startup.log 是 6/20 舊檔(誤讀成已在跑);(2)heredoc 建 launch script 失敗→改 nohup bash -lc 直接跑。最終正確啟動。
- **SA3 啟動**:run_name sa3_v3f_reluFix_ne1024_s42,**PID 2457352**(/tmp/sa1_v3f.pid),log=/tmp/sa3_v3f_startup.log,從 SA2 checkpoint_150000 接棒。config wd_sa3_v3f=stage3(靜態3+動態3-4+牆+horizontal_crossing+★head_on 0.20)+v3e+1400iter(同樣縮短到 +150000 steps≈iter500 停)。Isaac 啟動中,runid 待抓。
- ★SA3 監控重點:(a)head_on 有沒有正常 spawn(grep BehaviorScheduler/head_on)、無 crash (b)stage3 較難 SR 會從 98% 回落觀察回升 (c)★head_on 後 CR 是否先升再降(學會提早避) (d)★aux preprocess_loss/rel err 在更密+head_on 下是否終於 <100%(RNN 學追蹤真考驗)。

**2026-06-25 01:20 — ✅SA2 到 iter500 停 → 解碼中 → 待接 SA3**
- iter500 達標、checkpoint_150000 落地 → 依縮短預算計畫**停 SA2**(kill PID2213127+GPU子2213153,GPU 釋放 17MiB)。
- **SA2 reluFix 完訓摘要**(stage2,iter1→500=checkpoint_150000):SR 96.6%@1→飽和 **98.6-98.9%**、CR **1.0-1.4%**、TO 0%、sw 0.014-0.015。aux:三模組全活、**preprocess_loss 全程死平 ~2.48**(stage2 障礙仍沒讓 RNN 開始學追蹤)。
- 啟動 SA2 健康確認解碼:checkpoint_150000,stage2,DECODE_PID 2450637,log=/tmp/aux_decode_sa2.log。★依用戶規則[feedback_decode_each_stage_rnn_health]解碼確認 RNN 健康(rel err vs SA1 100%)後才接 SA3。
- 下步:讀解碼→確認健康→接 SA3(wd_sa3_v3f,從 checkpoint_150000,★含新 head_on 直線迎面)。

**2026-06-25 00:54 — SA2 iter ~480/1400（接近 iter500 停點)**
- 導航：SR **98.7%@460-480**（穩定平台）；CR 1.3→1.2→**1.2%**；TO 0%；ent ~2.7-2.98；sw 0.014-0.015；fps 3778-3792；PID 2213127 etime 11:01h；log mtime 00:51 fresh 無 stall。
- ★aux：preprocess_loss 仍 ~2.48 死平;grad 全非0。
- checkpoint(SA2)：checkpoint_120000 最新;停點 checkpoint_150000(iter500) 約再 ~27min(下輪很可能就到)。
- 判定：SA2 健康、SR 穩。下輪即觸發「停+解碼+接 SA3(含 head_on)」。

**2026-06-25 00:28 — SA2 iter ~460/1400（往 iter500 停點)**
- 導航：SR 98.7→98.6→**98.7%@460**（穩定）；CR 1.2→1.4→**1.3%**；TO 0%；ent ~2.72；sw 0.014-0.015；fps 3781-3809；PID 2213127 etime 10:35h；log mtime 00:26 fresh 無 stall。
- ★aux：preprocess_loss 仍 ~2.48 死平;grad 全非0。
- checkpoint(SA2)：checkpoint_120000 最新;停點 checkpoint_150000(iter500) 約再 ~55min。
- 判定：SA2 健康、SR 穩。未到停點,續訓。head_on 就緒待 SA3。

**2026-06-25 00:01 — SA2 iter ~440/1400（往 iter500 停點)**
- 導航：SR 98.9→98.6→**98.7%@440**（穩定）；CR 1.0→1.2→**1.2%**；TO 0%；ent ~2.6-2.72；sw 0.015；fps 3783-3796；PID 2213127 etime 10:09h；log mtime 00:00 fresh 無 stall。
- ★aux：preprocess_loss 仍 ~2.48 死平;grad 全非0。
- checkpoint(SA2)：checkpoint_120000 最新;停點 checkpoint_150000(iter500) 約再 ~1.4h。
- 判定：SA2 健康、SR 穩。未到停點,續訓。head_on 就緒待 SA3。

**2026-06-24 23:34 — SA2 iter ~420/1400（往 iter500 停點)**
- 導航：SR 98.5→98.7→**98.9%@420**（穩定,iter420 觸 98.9%/CR1.0%）；CR 1.4→1.2→**1.0%**；TO 0%；ent ~2.6-2.86；sw 0.014-0.015；fps 3705-3783；PID 2213127 etime 9:42h；log mtime 23:32 fresh 無 stall。
- ★aux：preprocess_loss 仍 ~2.48 死平（無趨勢，省略細節同前）。grad 全非0。
- checkpoint(SA2)：checkpoint_120000 最新;停點 checkpoint_150000(iter500) 約再 ~1.8h。
- 判定：SA2 健康、SR 穩。未到停點,續訓。head_on 已就緒待 SA3。

**2026-06-24 23:07 — SA2 iter ~400/1400（往 iter500 停點)**
- 導航：SR 98.7→98.7→**98.5%@400**（穩定）；CR 1.2→1.2→**1.4%**；TO 0%；ent ~2.7-2.86；sw 0.015；fps 3705-3796；PID 2213127 etime 9:15h；log mtime 23:06 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.268/0.259/0.257、rnn 0.050/0.042/0.043、extractor 0.029/0.023/0.022）。preprocess_loss 3seg **2.484/2.493/2.483**（★iter404 仍死平 2.48,SA2 確認沒讓 RNN 開始學追蹤）。
- checkpoint(SA2)：checkpoint_120000(iter400) 最新;停點 checkpoint_150000(iter500) 約再 ~2.2h。
- ⚙️**已完成 head_on 行為實作**(SA3-5 加直線迎面 0.20,見 [[project_head_on_behavior]]);SA3 launch 即生效。
- 判定：SA2 健康、SR 穩、aux 死平→SA2 解碼 rel err 大概率 100%,真考驗在 SA3+(更密+head_on)。未到停點,續訓。

**2026-06-24 22:35 — SA2 iter ~380/1400（往 iter500 停點)**
- 導航：SR **98.6-98.7%@360-380**（穩定）；CR 1.2-1.4%；TO 0%；ent ~2.8-2.97（震盪帶）；sw 0.014-0.015；fps 3770-3805；PID 2213127 etime 8:43h；log mtime 22:34 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.267/0.263/0.253、rnn 0.051/0.044/0.040、extractor 0.029/0.023/0.022）。preprocess_loss 3seg **2.480/2.494/2.485**,last5=[2.27,3.05,2.51,2.44,2.28] → ★**iter380 仍平 ~2.48,密集障礙到此仍沒讓 RNN 開始學追蹤(loss 沒降)**。
- checkpoint(SA2)：checkpoint_90000(iter300) 最新;停點 checkpoint_150000(iter500) 約再 ~2.7h。
- 判定：SA2 健康、SR 穩;aux 持續平台→SA2 解碼 rel err 大概率也 ~100%,真考驗在 SA3+(更密)。未到停點,續訓。

**2026-06-24 20:55 — ⚙️用戶決定：縮短 SA2 budget（不跑滿 1400）**
- 用戶問「怎麼跑那麼久」→ 解釋:SA2=1400 iter × ~82s/iter ≈ 32h(非變慢,純預算大;fps 3770 正常,1024envs×300rollout=307k steps/iter)。SA2 導航 iter10 就飽和(SR98.6%)、aux 平 2.48 無趨勢→剩 ~25h 多半冗餘(同 SA1)。
- ⚙️用戶選「**縮短 SA2 budget**」→ 新方案:SA2 續跑到 **iter ~500(checkpoint_150000,約再 4.5h)** 給 aux 多 ~200 iter 機會→**停+解碼確認 RNN 健康(rel err <100%?)+接 SA3**。不重啟(保留 optimizer state),只改停止點 1400→500。
- SA3 config 就緒:wd_sa3_v3f(stage3/v3e/也是 1400 iter,同樣會早停),handoff --checkpoint logs/rnn_car/sa2_v3f_reluFix_ne1024_s42/checkpoint_150000.pt。
- ⭐ 注意:SA3-5 障礙更密,才是 RNN 學追蹤(rel err <100%)的真考驗;SA3+ 解碼更關鍵。

**2026-06-24 20:40 — SA2 iter ~295/1400（~21%）**
- 導航：SR **98.6-98.8%@260-290**（穩定）；CR 1.2-1.4%；TO 0%；ent ~2.7-2.95（震盪帶）；sw 0.014-0.015；fps 3758-3779；PID 2213127 etime 6:47h；log mtime 20:39 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.263/0.269/0.256、rnn 0.050/0.049/0.037、extractor 0.030/0.025/0.021）。preprocess_loss 3seg **2.476/2.491/2.483**,last5=[2.73,2.41,2.26,2.32,2.38] → 平 ~2.48,無趨勢。
- checkpoint(SA2) checkpoint_60000。GPU 餘量足。
- 判定：SA2 健康、SR 穩。aux 平台 2.48。無異常,續訓。

**2026-06-24 20:14 — SA2 iter ~276/1400（~20%）**
- 導航：SR **98.6-98.8%@240-270**（穩定）；CR 1.1-1.4%；TO 0%；ent ~2.7-2.9（震盪帶）；sw 0.014-0.015；fps 3758-3775；PID 2213127 etime 6:21h；log mtime 20:11 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.265/0.270/0.258、rnn 0.051/0.049/0.038、extractor 0.030/0.025/0.021）。preprocess_loss 3seg **2.477/2.497/2.482**,last5=[2.36,2.30,2.44,2.31,2.58] → 平 ~2.49,無趨勢。
- checkpoint(SA2) checkpoint_60000。GPU 餘量足。
- 判定：SA2 健康(~20%)、SR 穩。aux 平台 2.49。無異常,續訓。

**2026-06-24 19:47 — SA2 iter ~256/1400（~18%）**
- 導航：SR 98.6→98.6→98.8→**98.6%@250**（穩定,iter240 觸 98.8%/CR1.1%）；CR 1.1-1.4%；TO 0%；ent ~2.7-2.9（震盪帶）；sw 0.014-0.015；fps 3748-3777；PID 2213127 etime 5:55h；log mtime 19:45 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.265/0.269/0.260、rnn 0.052/0.049/0.040、extractor 0.031/0.025/0.022）。preprocess_loss 3seg **2.465/2.490/2.506**,last5=[2.18,2.27,2.62,2.65,2.23] → 平 ~2.49,無趨勢。
- checkpoint(SA2) checkpoint_60000。GPU 餘量足。
- 判定：SA2 健康、SR 穩。aux 平台 2.49。無異常,續訓。

**2026-06-24 19:21 — SA2 iter ~237/1400（~17%）**
- 導航：SR **98.6-98.7%@200-230**（穩定）；CR 1.2-1.3%；TO 0%；ent ~2.7-2.85（震盪帶）；sw 0.014-0.015；fps 3733-3787；PID 2213127 etime 5:28h；log mtime 19:19 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.269/0.263/0.260、rnn 0.053/0.049/0.041、extractor 0.031/0.025/0.022）。preprocess_loss 3seg **2.467/2.499/2.489**,last5=[2.24,2.64,1.96,2.79,2.38] → 平 ~2.48,無趨勢。
- 僅本 SA2 + 他人小 job(588MiB),GPU 餘量足。checkpoint(SA2) checkpoint_60000。
- 判定：SA2 健康、SR 穩。aux 平台 2.48。無異常,續訓。⚙️用戶再確認:SA2 完訓後解碼確認 RNN 健康(★rel err 是否 <100%)再接 SA3。

**2026-06-24 18:54 — SA2 iter ~217/1400**
- 導航：SR **98.6-98.7%@180-210**（穩定）；CR 1.1-1.3%；TO 0%；ent ~2.5-2.82（震盪帶）；sw 0.014-0.015；fps 3694-3787；PID 2213127 etime 5:01h；log mtime 18:51 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.266/0.269/0.252、rnn 0.054/0.047/0.045、extractor 0.032/0.025/0.023[緩降,encoder 特徵穩定中,正常]）。preprocess_loss 3seg **2.452/2.511/2.492**,last5=[2.46,2.24,2.73,2.83,2.48] → 平 ~2.48,無趨勢。
- checkpoint(SA2)：**checkpoint_60000 已生成**（iter200）。GPU 又一他人小 job(2327281,588MiB),餘量足。
- 判定：SA2 健康、SR 穩。aux 平台 2.48。無異常,續訓。

**2026-06-24 18:28 — SA2 iter ~198/1400**
- 導航：SR **98.6-98.7%@160-190**（穩定）；CR 1.1-1.3%；TO 0%；ent ~2.3-2.95（震盪帶）；sw 0.014-0.015；fps 3694-3807；PID 2213127 etime 4:35h；log mtime 18:25 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.266/0.270/0.260、rnn 0.054/0.047/0.047、extractor 0.032/0.025/0.024）。preprocess_loss 3seg **2.458/2.516/2.474**,last5=[2.36,2.26,1.91,2.74,2.14] → 平 ~2.48,無趨勢。
- 僅 SA2 在卡(11961MiB)。checkpoint(SA2) checkpoint_30000。
- 判定：SA2 健康(~14%)、SR 穩。aux 平台 2.48。無異常,續訓。

**2026-06-24 18:02 — SA2 iter ~179/1400**
- 導航：SR **98.6-98.7%@150-180**（穩定）；CR 1.3→1.1→1.3→**1.1%**；TO 0%；ent ~2.3-2.95（震盪帶）；sw 0.014-0.015；fps 3694-3807；PID 2213127 etime 4:09h；log mtime 18:01 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.267/0.265/0.270、rnn 0.055/0.046/0.050、extractor 0.032/0.025/0.025）。preprocess_loss 3seg **2.483/2.473/2.502**,last5=[2.42,2.64,2.53,2.66,2.78] → 平 ~2.48,無趨勢。
- 僅 SA2 在卡(11961MiB)。checkpoint(SA2) checkpoint_30000。
- 判定：SA2 健康、SR 穩。aux 平台 2.48。無異常,續訓。

**2026-06-24 17:36 — SA2 iter ~160/1400**
- 導航：SR **98.6-98.7%@130-160**（穩定）；CR 1.3→1.3→1.3→**1.1%**；TO 0%；ent 2.82→2.63→**2.95**（緩升,同 SA1 角 floor 撐探索良性,SR 健康故 OK）；sw 0.014-0.015；fps 3782-3807；PID 2213127 etime 3:43h；log mtime 17:34 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.264/0.264/0.271、rnn 0.055/0.046/0.052、extractor 0.033/0.026/0.026）。preprocess_loss 3seg **2.477/2.487/2.479**（★完全平 ~2.48,160 iter 無下降趨勢）。
- 僅 SA2 在卡(11961MiB)。checkpoint(SA2) checkpoint_30000。
- 判定：SA2 健康、SR 穩。aux loss 平在 2.48(密集障礙到 iter160 仍沒讓 RNN 開始學追蹤,續觀察;完訓解碼是硬驗收)。

**2026-06-24 17:10 — SA2 iter ~140/1400**
- 導航：SR **98.6-98.7%@110-140**（穩定）；CR **1.2-1.3%**；TO 0%；ent ~2.4-2.8（震盪帶,iter130 觸 2.82）；sw 0.014-0.015；fps 3740-3804；PID 2213127 etime 3:17h；log mtime 17:09 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.268/0.262/0.270、rnn 0.056/0.046/0.049、extractor 0.033/0.027/0.025）。preprocess_loss 3seg **2.486/2.464/2.503**,last5=[2.43,2.76,2.75,2.25,2.48] → 仍 ~2.5 雜訊平台,無趨勢。
- 僅 SA2 在卡(11961MiB)。checkpoint(SA2) checkpoint_30000。
- 判定：SA2 健康、SR 穩。aux 平台。無異常,續訓。

**2026-06-24 16:44 — SA2 iter ~121/1400**
- 導航：SR **98.6-98.7%@90-120**（穩定）；CR **1.2-1.3%**；TO 0%；ent ~2.4-2.8（震盪帶）；sw 0.014-0.015；fps 3740-3790；PID 2213127 etime 2:51h；log mtime 16:43 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.269/0.267/0.265、rnn 0.056/0.050/0.047、extractor 0.033/0.029/0.025）。preprocess_loss 3seg **2.472/2.464/2.495**,last5=[2.32,1.98,1.87,2.29,2.60] → 仍 ~2.5 雜訊平台,無趨勢。
- GPU 他人小 job 已退,僅 SA2 在卡(11961MiB)。checkpoint(SA2) checkpoint_30000。
- 判定：SA2 健康、SR 穩。aux 平台。無異常,續訓。

**2026-06-24 16:18 — SA2 iter ~102/1400**
- 導航：SR **98.6-98.7%@70-100**（穩定）；CR **1.2-1.3%**；TO 0%；ent ~2.5-2.8（iter90 觸 2.83,震盪帶）；sw 0.014-0.015；fps 3683-3764；PID 2213127 etime 2:25h；log mtime 16:15 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.274/0.256/0.260、rnn 0.056/0.051/0.043、extractor 0.033/0.031/0.024）。preprocess_loss 3seg **2.440/2.473/2.535**,last5=[2.26,2.82,2.63,2.17,2.97] → 仍 ~2.5 雜訊平台,無趨勢。
- checkpoint(SA2)：**checkpoint_30000 已生成**（iter100）。GPU 他人小 job(588MiB)餘量足。
- 判定：SA2 健康、SR 穩。aux 平台。無異常,續訓。

**2026-06-24 15:51 — SA2 iter ~82/1400**
- 導航：SR **98.6-98.7%@50-80**（穩定）；CR **1.2-1.3%**；TO 0%；ent ~2.5；sw 0.015；fps 3683-3741；PID 2213127 etime 1:58h；log mtime 15:49 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.274/0.253/0.277、rnn 0.055/0.057/0.045、extractor 0.032/0.033/0.027）。preprocess_loss 3seg **2.413/2.546/2.437**,last5=[2.77,2.56,2.55,2.08,2.59] → 仍 ~2.4-2.5 雜訊平台,無趨勢。
- GPU 他人小 job(588MiB)餘量足。checkpoint(SA2)尚無(iter100 即將)。
- 判定：SA2 健康、SR 穩。aux 平台。無異常,續訓。

**2026-06-24 15:25 — SA2 iter ~63/1400**
- 導航：SR **98.6-98.7%@30-60**（穩定）；CR **1.2-1.3%**；TO 0%；ent ~2.4；sw 0.015；fps 3506-3715；PID 2213127 etime 1:32h；log mtime 15:23 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.271/0.265/0.264、rnn 0.053/0.057/0.055、extractor 0.032/0.034/0.031）。preprocess_loss 3seg **2.425/2.542/2.420**,last5=[2.55,2.36,1.84,2.09,2.38] → 仍 ~2.4 雜訊平台,無明確趨勢(守紀律不過早讀)。
- GPU 他人小 job 仍在(588MiB)餘量足。checkpoint(SA2)尚無(iter100)。
- 判定：SA2 健康、SR 穩。aux loss 平台。無異常,續訓。

**2026-06-24 14:58 — SA2 iter ~44/1400**
- 導航：SR 98.6→98.7→98.7→**98.6%@40**（穩 98.6-98.7%,適應 stage2 良好）；CR 1.4→1.3→1.2→**1.3%**；TO 0%；ent ~2.2；sw 0.014-0.015；fps 3506-3672；PID 2213127 etime 1:05h；log mtime 14:57 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.274/0.273/0.263、rnn 0.053/0.056/0.056、extractor 0.034/0.031/0.034）。preprocess_loss 3seg **2.469/2.386/2.593**,last5=[2.84,2.88,2.55,2.90,2.02] → ⚠️**修正:上輪 iter25「2.5→2.33 下彎」是雜訊,本輪回到 ~2.5 雜訊平台,無明確下降趨勢**。密集障礙是否讓 RNN 學追蹤仍待長期觀察(數百 iter loss 趨勢 + 完訓解碼 rel err)。
- GPU 他人小 job(2225579,588MiB)仍在,餘量足。checkpoint(SA2)尚無(iter100)。
- 判定：SA2 健康、SR 穩。aux loss 仍 ~2.5 平台(別過早讀單輪雜訊為趨勢)。

**2026-06-24 14:31 — SA2 iter ~25/1400（SR 已適應 stage2）**
- 導航：SR 96.6%@1→98.6%@10→**98.7%@20**（★stage2 適應極快,10 iter 內回到 98.7%,顯示 policy 從 SA1 良好泛化）；CR 1.3→1.4→**1.3%**；TO 0%；ent ~2.4；sw 0.014；fps 3661-3769；PID 2213127 etime 38:51；log mtime 14:30 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.266/0.280/0.264、rnn 0.047/0.059/0.059、extractor 0.034/0.030/0.032）。preprocess_loss 3seg **2.499/2.407/2.333**（★早期微降 2.5→2.33,stage2 密集障礙可能給 aux 更多 signal,但 n=25 太早,續觀察是否真降=RNN 開始學追蹤）。
- ⚠️GPU 多一個他人小 job(2225579, 588MiB),GPU 13GB 餘量足,留意。checkpoint(SA2 dir)尚無(首個 iter100)。
- 判定：SA2 健康、SR 適應佳。★關注 preprocess_loss 是否真降(密集障礙 RNN 學追蹤的第一個正向跡象)+SA2 完訓解碼 rel err 是否 <100%。

**2026-06-24 14:11 — ✅SA1 RNN 健康確認解碼通過 + SA2 進訓練迴圈**
- ✅**SA1 RNN 健康確認(checkpoint_30000, n=1600)**：predict_head 輸出 **signed 非0**(近1 x=-0.03/y=-0.26/d=-0.00、近2 x=-0.25/d=-0.29,皆非恆0、有負值)=**dead-ReLU 已修、結構健康、梯度有流、RNN/aux 正常運作 ✓**。rel err 100%(MAE 0.729≈真實均 0.727)=稀疏障礙下只猜均值(SA1 預期,非異常)。**健康關卡通過,可放行 SA2。**
- ✅**SA2 進訓練迴圈**：[1/1400] SR **96.6%**(stage2 較難,從 SA1 98% 小回落,預期會適應回升)、CR 1.3%、ent 2.31、fps 3769。PID 2213127、GPU 子 2213153(11901MiB)、runid bg7o5r0h。解碼 GPU 已清(僅 SA2 在卡)。
- 監控重點:SA2 SR 是否適應 stage2 回升;SA2 完訓再跑解碼確認 RNN 健康(★密集障礙 rel err 是否 <100% 是 RNN 學追蹤的真考驗)才接 SA3。

**2026-06-24 14:05 — ⚙️修正：解碼是「RNN 健康確認」必要關卡(不可省)**
- ⚙️用戶澄清前一指示被誤讀:「停止訓練 解碼 後接SA2」= 停訓→**解碼確認 RNN 健康**→接 SA2,解碼是**必跑的健康驗證**,非取消。★**新標準規則:每階段(SA1→SA5)都一定要跑 aux 解碼確認 RNN 健康正常,不可跳過**(即使 rel err 100%,解碼確認 predict_head 輸出 signed 非0=結構健康+梯度有流=RNN/aux 正常運作)。
- 補跑 SA1 最終健康確認解碼:_cont/checkpoint_30000(iter800-equiv),DECODE_PID 2217151,log=/tmp/aux_decode_sa1_final.log,與 SA2 並行(SA1 ckpt 凍結故獨立;GPU 5.7GB 餘量足)。
- SA2 已啟動並 Loaded checkpoint,新 runid **bg7o5r0h**(已寫 /tmp/sa1_v3f_runid.txt)。

**2026-06-24 14:00 — ✅SA1 收尾(用戶提早停)→ 接 SA2**
- ⚙️用戶「停止訓練 解碼 後接SA2」→ SA1 _cont 在 iter150/200(SR 98.6%,飽和)提早停,不跑解碼,直接接 SA2。kill PID 2151473+GPU 子程序 2151499(GPU 釋放 17MiB)。
- **SA1 reluFix 完訓摘要**：原 run iter1→790(卡死)→ _cont resume iter700→iter800-equiv(停)。SR 36%→飽和 **98.6-98.8%**、CR **1.2-1.3%**、TO 0%、sw 0.014-0.015。aux:三模組全活(grad 非0)、preprocess_loss 平台 **2.47-2.5**(均值擬合飽和)。★解碼 iter100/300/600 **三度都 100%**(SA1 稀疏障礙[dynamic 1-2]下 RNN 只學均值,定案)。dead-ReLU 結構修復必要性三度坐實;「RNN 是否學會追蹤」裁決移到 SA2-5 密集障礙。
- **SA2 啟動**：run_name `sa2_v3f_reluFix_ne1024_s42`,**PID 2213127**(/tmp/sa1_v3f.pid),log=/tmp/sa2_v3f_startup.log,腳本 /tmp/launch_sa2_v3f_relufix.sh。從 **logs/rnn_car/sa1_v3f_reluFix_ne1024_s42_cont/checkpoint_30000.pt**(SA1 iter800-equiv 飽和)接棒;config wd_sa2_v3f=stage2(靜態1+動態增)+v3e+420000ts(1400 iter)。新 runid 待抓。
- ★SA2 監控重點:SR 起點(stage2 較難會回落再學)+ ★aux 解碼相對誤差是否開始 <100%(密集障礙才是 RNN 學追蹤的真考驗;但用戶剛停解碼,SA2 解碼時機待與用戶確認,先監控 SR/CR/aux loss)。

**2026-06-24 13:53 — ⚙️用戶指示：取消 iter900 最後解碼**
- 用戶「停止跑最後解碼」→ SA1 完訓(iter200=checkpoint_60000)**直接接 SA2,不跑解碼**。理由:iter100/300/600 三度都 100%(稀疏障礙猜均值),第四次必同結論無新資訊。SA1 解碼結論定案=全程 100%(三點足夠)。當下無解碼程序在跑,無需中止。

**2026-06-24 13:28 — _cont iter ~139/200（七成）**
- 導航：SR 98.6→98.6→**98.7%@140**（飽和平穩）；CR 1.3→1.3→**1.2%**；TO 0%；ent ~2.2；sw 0.014-0.015；fps 3813-3828；GPU 11969MiB/28%；PID 2151473 etime 3:10h；log mtime 13:27 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.270/0.270/0.275、rnn 0.053/0.051/0.050、extractor 0.035/0.035/0.035）。preprocess_loss 3seg **2.510/2.558/2.528**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：checkpoint_30000 最新；完訓 iter200=checkpoint_60000=真 iter900。剩 ~60 iter ~1.4h。
- 判定：_cont 健康、SR 穩 98.7%。無異常,例行續訓至 iter200。

**2026-06-24 13:02 — _cont iter ~120/200（六成）**
- 導航：SR 98.8→98.6→**98.6%@120**（飽和平穩）；CR 1.0→1.3→**1.3%**；TO 0%；ent ~2.3-2.4；sw 0.014-0.015；fps 3813-3830；GPU 11969MiB/32%；PID 2151473 etime 2:44h；log mtime 13:00 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.264/0.274/0.280、rnn 0.054/0.050/0.054、extractor 0.036/0.035/0.035）。preprocess_loss 3seg **2.527/2.525/2.571**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：checkpoint_30000 最新；完訓 iter200=checkpoint_60000=真 iter900。剩 ~80 iter ~1.8h。
- 判定：_cont 健康、SR 穩 98.6%。無異常,例行續訓至 iter200。

**2026-06-24 12:36 — _cont iter ~101/200（過半）**
- 導航：SR 98.6→98.6→**98.8%@100**（飽和平穩,iter100 CR 創低 1.0%）；CR 1.3→1.3→**1.0%**；TO 0%；ent ~2.38；sw 0.014；fps 3814-3830；GPU 11969MiB/36%；PID 2151473 etime 2:18h；log mtime 12:34 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.264/0.274/0.269、rnn 0.054/0.050/0.053、extractor 0.035/0.035/0.036）。preprocess_loss 3seg **2.521/2.513/2.569**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：**checkpoint_30000 已生成**（_cont iter100=原 iter800-equiv）；完訓 iter200=checkpoint_60000=真 iter900。剩 ~100 iter ~2.2h。
- 判定：_cont 健康、SR 穩 98.8%。無異常,例行續訓至 iter200。

**2026-06-24 12:10 — _cont iter ~81/200**
- 導航：SR 98.7→98.7→98.7→**98.6%@80**（飽和平穩）；CR 1.3→1.2→**1.3%**；TO 0%；ent ~2.3；sw 0.014-0.015；fps 3814-3831；GPU 11969MiB/29%；PID 2151473 etime 1:52h；log mtime 12:09 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.265/0.272/0.268、rnn 0.056/0.049/0.051、extractor 0.036/0.034/0.036）。preprocess_loss 3seg **2.507/2.538/2.524**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：尚無（首個 iter100=checkpoint_30000,即將）。剩 ~120 iter ~2.7h。
- 判定：_cont 健康、SR 穩 98.6-98.7%。無異常,例行續訓至 iter200。

**2026-06-24 11:44 — _cont iter ~62/200**
- 導航：SR 98.8→98.6→98.7→**98.7%@60**（飽和平穩）；CR 1.1→1.3→**1.2%**；TO 0%；ent ~2.1；sw 0.014-0.015；fps 3820-3831；GPU 11969MiB/38%；PID 2151473 etime 1:26h；log mtime 11:44 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.260/0.270/0.273、rnn 0.058/0.052/0.047、extractor 0.038/0.034/0.033）。preprocess_loss 3seg **2.498/2.556/2.511**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：尚無（首個 iter100=checkpoint_30000）。剩 ~140 iter ~3h。
- 判定：_cont 健康、SR 穩 98.7%。無異常,例行續訓至 iter200。

**2026-06-24 11:18 — _cont iter ~42/200**
- 導航：SR 98.8→98.8→98.6→**98.6%@40**（飽和平穩,完全回到 stall 前水準）；CR 1.2→1.1→**1.3%**；TO 0%；ent ~2.2；sw 0.014-0.015；fps 3806-3829；GPU 11969MiB/28%；PID 2151473 etime 1:00h；log mtime 11:16 fresh 無 stall。
- ★aux：grad 全非0（predict_head 3seg 0.255/0.276/0.265、rnn 0.059/0.051/0.054、extractor 0.039/0.033/0.036）。preprocess_loss 3seg **2.474/2.550/2.547**（穩 ~2.5 平台）。
- checkpoint(_cont dir)：尚無（首個 iter100=checkpoint_30000）。剩 ~160 iter ~3.5h。
- 判定：_cont 完全健康,SR 回到 stall 前。無異常,例行續訓至 iter200。

**2026-06-24 10:51 — ✅_cont iter ~20/200（SR 已回升）**
- 導航：SR 96.5%@1→98.8%@10→**98.7%@20**（★冷啟回落 10 iter 內完全回升,確認 resume 無損）；CR 1.3→1.2→**1.3%**；TO 0%；ent ~2.3；sw 0.015；fps 3806-3829；GPU 11969MiB/31%；PID 2151473 alive etime 33:20。
- ★aux：grad 全非0（predict_head 3seg 0.248/0.261/0.279、rnn 0.062/0.056/0.056、extractor 0.039/0.039/0.036）。preprocess_loss 3seg **2.498/2.450/2.514**（與 stall 前同平台 ~2.49,aux 狀態完整接續）。
- checkpoint(_cont dir)：尚無（首個 iter100=checkpoint_30000,完訓 iter200=checkpoint_60000=真 iter900）。剩 ~180 iter ~4h。
- 判定：✅ stall 自動回復完全成功,SR 回升、aux 接續、無再卡。例行續訓至 iter200。

**2026-06-24 10:25 — ✅_cont 回復成功啟動**
- PID 2151473 alive(etime 7:20)、GPU 11967MiB/34%（訓練中）；**[INFO] Loaded checkpoint_210000 成功**；新 wandb run **duhv3gbh**（已寫 /tmp/sa1_v3f_runid.txt）。
- [1/200] SR **96.5%**（optimizer 冷啟+resume 小回落,iter160 起原本 98.6%,預期數 iter 內回升）；CR 1.3%；ent 2.28；sw 0.015；fps 3821。
- ✅ 自動回復確認成功,無再卡 _grad_l2_norm:685。剩 ~200 iter(~4.5h)到 iter900-equiv(=_cont dir checkpoint_60000)。
- run 索引更新：SA1 收尾 run = `sa1_v3f_reluFix_ne1024_s42_cont`(duhv3gbh,PID 2151473)；SA2 須接 logs/rnn_car/sa1_v3f_reluFix_ne1024_s42_cont/checkpoint_60000.pt。

**2026-06-24 10:15 — ⚠️SA1 stall 確診 + 自動回復**
- ⚠️**stall 三條件全中**（authorized auto-recover）：(1)GPU util 0%×3 連續 (2)log mtime 09:42→now 10:15 凍 **32min** (3)py-spy MainThread 卡在 **`_grad_l2_norm (train_rnn_car_wdclip.py:685)`**（正是 watcher 簽章；CPU 113% 空轉、GPU 無進度、_step 凍在 237900/iter790）。無他人 GPU job 競爭（僅本 PID）→ 真 hang 非 contention。
- 回復：kill -9 PID1647188（GPU 釋放→17MiB）→ 從 **checkpoint_210000(iter700)** 續跑剩餘 200 iter（--timesteps 60000，腳本確認 loop 不還原 iteration 計數器故傳「剩餘」量）。SA1 已飽和(SR98.6% since iter160)故 iter700→900 零品質損。
- **新 run**：run_name `sa1_v3f_reluFix_ne1024_s42_cont`，**NEW PID 2151473**（已寫 /tmp/sa1_v3f.pid），log=/tmp/sa1_v3f_cont_startup.log，run dir=logs/rnn_car/sa1_v3f_reluFix_ne1024_s42_cont/。⚠️ 因 loop 重新計數,**最終 checkpoint = 該 dir 的 checkpoint_60000（=真 iter900）**,非 checkpoint_270000 → SA2 handoff 須改用此路徑。新 runid 待啟動後抓。
- config no_resume_optimizer=true 故 optimizer 冷啟（飽和階段可忽略）。⚠️ 若 _cont 又卡 _grad_l2_norm:685 → 需查 line 685 根因(疑數值/同步問題),非單純重啟。

**2026-06-24 09:49 — SA1 iter ~793（八成八）**
- 導航：SR 98.7→98.7→**98.6%@790**（飽和平穩）；CR 1.2→1.3→**1.3%**；TO 0%；ent 2.21→2.19→**2.26**（震盪帶內）；sw 0.014-0.015；fps 3748-3798（util 瞬時 0% 但 iter 正常推進=rollout/update 間取樣,非 stall）；GPU 12.7GB；etime 17:46h。
- ★aux：grad 全非0（predict_head 3seg 0.288/0.269/0.268、rnn 0.069/0.060/0.050、extractor 0.058/0.043/0.035）。preprocess_loss 3seg **2.715/2.476/2.494**（穩 2.49 平台）。
- checkpoint：checkpoint_210000 最新；完訓 iter900 約再 ~2.3h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪。無異常,例行續訓。

**2026-06-24 09:23 — SA1 iter ~778（八成六）**
- 導航：SR **98.7%@750-770**（飽和平穩,連3點 98.7%）；CR **1.2%**；TO 0%；ent 2.34→2.31→**2.21**（震盪帶內回落）；sw 0.014-0.015；fps 3784-3798（util 29%）；GPU 12.7GB；etime 17:20h。
- ★aux：grad 全非0（predict_head 3seg 0.289/0.268/0.269、rnn 0.069/0.060/0.051、extractor 0.058/0.043/0.035）。preprocess_loss 3seg **2.720/2.473/2.492**（穩 2.49 平台）。
- checkpoint：checkpoint_210000 最新；完訓 iter900 約再 ~2.7h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪。無異常,例行續訓。

**2026-06-24 08:57 — SA1 iter ~759（八成四）**
- 導航：SR 98.6→98.7→**98.7%@760**（飽和平穩）；CR 1.3→1.2→**1.2%**；TO 0%；ent 2.20→2.34→**2.31**（震盪帶上緣）；sw 0.014-0.015；fps 3784-3805（util 35%）；GPU 12.7GB；etime 16:54h。
- ★aux：grad 全非0（predict_head 3seg 0.289/0.268/0.268、rnn 0.069/0.060/0.051、extractor 0.058/0.043/0.036）。preprocess_loss 3seg **2.725/2.477/2.483**（穩 2.48 平台）。
- checkpoint：checkpoint_210000 最新；完訓 iter900 約再 ~3h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪。無異常,例行續訓。

**2026-06-24 08:31 — SA1 iter ~739**
- 導航：SR 98.7→98.7→**98.6%@740**（飽和平穩）；CR 1.2→1.2→**1.3%**；TO 0%；ent 2.12→2.12→**2.20**（震盪帶內）；sw 0.015；fps 3783-3805（util 29%）；GPU 12.7GB；etime 16:28h。
- ★aux：grad 全非0（predict_head 3seg 0.290/0.268/0.270、rnn 0.069/0.061/0.052、extractor 0.059/0.044/0.036）。preprocess_loss 3seg **2.733/2.475/2.484**（穩 2.48 平台）。
- checkpoint：checkpoint_210000 最新；完訓 iter900 約再 ~3.3h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪。無異常,例行續訓。

**2026-06-24 08:05 — SA1 iter ~720（八成）**
- 導航：SR 98.7→98.6→**98.7%@720**（飽和平穩）；CR **1.2%**；TO 0%；ent 2.32→2.20→**2.12**（震盪帶內回落）；sw 0.014-0.015；fps 3788-3791（util 39%）；GPU 12.7GB；etime 16:02h。
- ★aux：grad 全非0（predict_head 3seg 0.290/0.270/0.265、rnn 0.070/0.062/0.051、extractor 0.059/0.044/0.036）。preprocess_loss 3seg **2.739/2.475/2.484**（穩 2.48 平台）。
- checkpoint：checkpoint_210000 最新；完訓 iter900 約再 ~3.7h（最後解碼點）。
- 判定：SA1 八成、飽和平穩、aux 平台、ent 震盪。無異常,例行續訓。

**2026-06-24 07:39 — SA1 iter ~701（七成八）**
- 導航：SR 98.6→98.6→**98.7%@700**（飽和平穩）；CR 1.3→1.3→**1.2%**；TO 0%；ent 2.15→2.32→**2.32**（震盪帶 ~2.0-2.3 上緣）；sw 0.014；fps 3786-3806（util 27%）；GPU 12.7GB；etime 15:36h。
- ★aux：grad 全非0（predict_head 3seg 0.290/0.270/0.267、rnn 0.070/0.062/0.052、extractor 0.060/0.044/0.037）。preprocess_loss 3seg **2.751/2.476/2.477**（穩 2.47 平台）。
- checkpoint：**checkpoint_210000 已生成**（iter700）；完訓 iter900 約再 ~4h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪上緣。無異常,例行續訓。

**2026-06-24 07:13 — SA1 iter ~682（七成半）**
- 導航：SR **98.6%@660-680**（飽和平穩）；CR 1.2→1.3→**1.3%**；TO 0%；ent 2.31→2.23→**2.15**（震盪帶內回落,~2.0-2.3）；sw 0.014；fps 3774-3806（util 35%）；GPU 12.7GB；etime 15:10h。
- ★aux：grad 全非0（predict_head 3seg 0.291/0.271/0.266、rnn 0.070/0.062/0.053、extractor 0.060/0.044/0.038）。preprocess_loss 3seg **2.761/2.481/2.469**（穩 2.47 平台）。
- checkpoint：checkpoint_180000 最新；完訓 iter900 約再 ~4.5h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪回落。無異常,例行續訓。

**2026-06-24 06:47 — SA1 iter ~663**
- 導航：SR 98.8→98.6→**98.6%@660**（飽和平穩）；CR 1.1→1.2→**1.2%**；TO 0%；ent 2.08→2.18→**2.31**（在 ~2.0-2.3 震盪帶內,SR 健康良性）；sw 0.014-0.015；fps 3774-3804（util 31%）；GPU 12.7GB；etime 14:44h。
- ★aux：grad 全非0（predict_head 3seg 0.291/0.271/0.268、rnn 0.070/0.062/0.054、extractor 0.060/0.044/0.039）。preprocess_loss 3seg **2.768/2.480/2.472**（穩 2.47 平台）。
- checkpoint：checkpoint_180000 最新；完訓 iter900 約再 ~5h（最後解碼點）。
- 判定：SA1 飽和平穩、aux 平台、ent 震盪帶內。無異常,例行續訓。

**2026-06-24 06:21 — SA1 iter ~643**
- 導航：SR 98.6→**99.0**→**98.8%@640**（觸 99.0% 新高,飽和平穩）；CR 1.3→1.0→**1.1%**；TO 0%；ent 2.05→2.09→**2.08**（穩 ~2.0）；sw 0.014-0.015；fps 3657-3801（util 27%）；GPU 12.7GB；etime 14:18h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.271/0.267、rnn 0.070/0.063/0.054、extractor 0.061/0.044/0.039）。preprocess_loss 3seg **2.784/2.479/2.466**（穩 2.47 平台）。
- checkpoint：checkpoint_180000 最新；完訓 iter900 約再 ~5.5h（最後解碼點）。
- 判定：SA1 飽和平穩（SR 觸新高 99.0%）、aux 平台、ent 穩。無異常,例行續訓。

**2026-06-24 05:55 — SA1 iter ~624（七成）**
- 導航：SR **98.6%@600-620**（飽和平穩）；CR 1.3→1.2→**1.3%**；TO 0%；ent 1.97→2.06→**2.05**（穩 ~2.0 平衡）；sw 0.015；fps 3777-3784（util 29%）；GPU 12.7GB；etime 13:52h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.272/0.268、rnn 0.070/0.062/0.056、extractor 0.061/0.043/0.040）。preprocess_loss 3seg **2.786/2.487/2.467**（穩 2.47 平台）。
- checkpoint：checkpoint_180000 最新；完訓 iter900(checkpoint_270000) 約再 ~6h（最後解碼點）。
- 判定：SA1 七成、飽和平穩、aux 平台、ent 穩 2.0。無異常,例行續訓。

**2026-06-24 05:28 — SA1 iter 600 ★★★第三個 aux 解碼硬數字（已讀）**
- 導航：SR **98.6%@600**（飽和）；CR 1.3%；ent 1.97；sw 0.015；GPU 12.7GB；etime 13:25h；train 健康。
- ★★★解碼結果(checkpoint_180000, stage1, n=1600)：**相對誤差仍 100%**(MAE 0.739 ≈ 真實均 0.739)。預測仍常數(近1 x≈-0.03/y≈-0.32、近2 x≈-0.32/d≈-0.40,不隨 step 變)。
- ★三點一線定論：**iter100 / iter300 / iter600 解碼全部 100%、全部猜常數均值**。SA1 稀疏障礙(dynamic 1-2)下 RNN 完全學不到追蹤,只能輸出均值——**已三度確認,SA1 全程註定 100%**。preprocess_loss 平在 2.47 = 常數均值擬合飽和,與此完全一致。
- ✅ 結構修復必要性三度坐實;❓「RNN 是否學會追蹤」裁決仍全押 SA3-SA5 密集障礙。完訓(iter900)再跑最後一次解碼確認 SA1 收尾,但預期仍 100%,不必過度期待。
- GPU 已清(decode 退,僅 train+他人小 job)。

**2026-06-24 04:53 — SA1 iter ~580**
- 導航：SR 98.6→98.7→**98.6%@580**（飽和平穩）；CR 1.2→1.3→**1.3%**；TO 0%；ent 2.01→2.01→**1.96**（穩 ~2.0）；sw 0.015；fps 3780-3795（util 33%）；GPU 12.7GB；etime 12:50h。
- ★aux：grad 全非0（predict_head 3seg 0.292/0.274/0.267、rnn 0.070/0.062/0.057、extractor 0.063/0.043/0.041）。preprocess_loss 3seg **2.808/2.487/2.465**（穩 2.47 平台）。
- checkpoint：checkpoint_150000 最新（checkpoint_180000 尚未落地,iter580 距 iter600 約 20 iter/~25min）；解碼點 iter600 下輪到。
- 判定：SA1 飽和平穩、aux 平台、ent 穩 2.0。無異常,例行續訓。

**2026-06-24 04:27 — SA1 iter ~560**
- 導航：SR 98.7→98.6→**98.6%@560**（飽和平穩）；CR 1.3→1.3→**1.2%**；TO 0%；ent 2.02→2.04→**2.01**（穩在 ~2.0 平衡）；sw 0.015；fps 3780-3786（util 28%）；GPU 12.7GB；etime 12:24h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.273/0.270、rnn 0.071/0.062/0.058、extractor 0.063/0.043/0.042）。preprocess_loss 3seg **2.824/2.475/2.473**（穩 2.47 平台）。
- checkpoint：checkpoint_150000 最新（checkpoint_180000 尚未落地,iter560 距 iter600 約 40 iter/~50min）；解碼點 iter600 下輪到。
- 判定：SA1 飽和平穩、aux 平台、ent 穩 2.0。無異常,例行續訓。

**2026-06-24 04:01 — SA1 iter ~540（六成）**
- 導航：SR 98.7→98.6→**98.7%@540**（飽和平穩）；CR 1.1→1.2→**1.3%**；TO 0%；ent 2.24→2.01→**2.02**（★從峰 2.51 回落並穩在 ~2.0,確認有界震盪平衡）；sw 0.014-0.015；fps 3781-3802（util 41%）；GPU 12.7GB；etime 11:58h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.274/0.267、rnn 0.071/0.063/0.059、extractor 0.064/0.043/0.042）。preprocess_loss 3seg **2.830/2.472/2.483**（穩 2.48 平台）。
- checkpoint：checkpoint_150000 最新；下個解碼點 iter600 約再 ~45min（下輪很可能就到）。
- 判定：SA1 飽和平穩、aux 平台、ent 回穩 ~2.0。無異常,例行續訓。

**2026-06-24 03:35 — SA1 iter ~521**
- 導航：SR **98.7%@510-520**（飽和平穩,CR 創低 1.1%）；CR 1.2→1.1→**1.2%**；TO 0%；ent 2.51→2.24→**2.15**（★峰值 2.51 後回落到 2.15 → 證實是「有界震盪平衡」非單調發散,比上輪「緩升」擔憂更安心）；sw 0.014-0.015；fps 3776-3802（util 31%）；GPU 12.7GB；etime 11:32h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.274/0.269、rnn 0.071/0.063/0.060、extractor 0.065/0.044/0.043）。preprocess_loss 3seg **2.851/2.462/2.479**（穩 2.48 平台）。
- checkpoint：checkpoint_150000 最新；下個解碼點 iter600 約再 ~1.2h。
- 判定：SA1 飽和平穩、aux 平台、ent 有界震盪(非發散)。無異常,例行續訓。

**2026-06-24 03:08 — SA1 iter ~501（過半多）**
- 導航：SR 98.7→98.6→**98.7%@500**（飽和平穩）；CR 1.2→1.3→**1.2%**；TO 0%；ent 2.27→2.51→**2.43**（★緩升中 ~2.4,角 ent floor 0.02 在飽和簡單階段持續加碼探索,SR 仍 98.7% 故良性;留意 iter900 可能 ~3+,但接 SA2 即重置難度）；sw 0.014-0.015；fps 3776-3792（util 39%）；GPU 12.7GB；etime 11:05h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.275/0.268、rnn 0.071/0.063/0.061、extractor 0.065/0.044/0.044）。preprocess_loss 3seg **2.867/2.462/2.473**（穩 2.47 平台）。
- checkpoint：**checkpoint_150000 已生成**（iter500）；下個解碼點 iter600 約再 ~1.7h。
- 判定：SA1 飽和平穩、aux 平台。ent 緩升屬飽和階段良性現象(SR 健康),非介入點。例行續訓。

**2026-06-24 02:42 — SA1 iter ~482**
- 導航：SR 98.6→98.7→**98.7%@480**（飽和平穩）；CR 1.4→1.2→**1.2%**；TO 0%；ent 1.93→2.02→**2.27**（在 ~2.0 平衡帶內波動,SR 健康故 OK）；sw 0.014-0.015；fps 3766-3791（util 46%）；GPU 12.7GB；etime 10:39h。
- ★aux：grad 全非0（predict_head 3seg 0.294/0.275/0.271、rnn 0.071/0.064/0.062、extractor 0.066/0.044/0.044）。preprocess_loss 3seg **2.885/2.465/2.474**（穩 2.47 平台,最低觸 1.86）。
- checkpoint：checkpoint_120000 最新；下個解碼點 iter600 約再 ~2h。
- 判定：SA1 飽和平穩、aux 平台、ent 平衡帶內波動。無異常,例行續訓。

**2026-06-24 02:16 — SA1 iter ~463**
- 導航：SR 98.7→98.6→**98.6%@460**（飽和平穩）；CR 1.2→1.3→**1.4%**；TO 0%；ent 1.97→1.93→**1.98**（平衡 ~1.95 穩）；sw 0.014-0.015；fps 3766-3792（util 27%）；GPU 12.7GB；etime 10:13h。
- ★aux：grad 全非0（predict_head 3seg 0.294/0.273/0.273、rnn 0.072/0.063/0.063、extractor 0.067/0.044/0.044）。preprocess_loss 3seg **2.894/2.467/2.487**（穩 2.48 平台）。
- checkpoint：checkpoint_120000 最新；下個解碼點 iter600 約再 ~2.5h。
- 判定：SA1 飽和平穩、aux 平台、ent 平衡。無異常,例行續訓。

**2026-06-24 01:50 — SA1 iter ~444（過半）**
- 導航：SR 98.5→98.6→**98.7%@440**（飽和平穩）；CR 1.4→1.3→**1.2%**；TO 0%；ent 2.00→2.02→**1.97**（平衡 ~2.0 穩定）；sw 0.014-0.015；fps 3779-3792（util 38%）；GPU 12.7GB；etime 9:47h。
- ★aux：grad 全非0（predict_head 3seg 0.293/0.275/0.275、rnn 0.072/0.064/0.063、extractor 0.068/0.045/0.044）。preprocess_loss 3seg **2.896/2.497/2.475**（穩 2.48 平台）。
- checkpoint：checkpoint_120000 最新；下個解碼點 iter600 約再 ~3h。
- 判定：SA1 過半(444/900)、飽和平穩、aux 平台、ent 平衡 ~2.0。無異常,例行續訓。

**2026-06-24 01:24 — SA1 iter ~424**
- 導航：SR 98.7→98.6→**98.5%@420**（飽和平穩,微幅波動正常）；CR 1.3→1.3→**1.4%**；TO 0%；ent 2.13→2.08→**2.00**（★ent 達平衡並微回,~2.0 非失控,角 floor 平衡點）；sw 0.014-0.015；fps 3780-3788（util 42%）；GPU 12.7GB；etime 9:21h。
- ★aux：grad 全非0（predict_head 3seg 0.294/0.277/0.275、rnn 0.072/0.065/0.061、extractor 0.069/0.045/0.043）。preprocess_loss 3seg **2.909/2.506/2.485**（穩 2.49 平台）。
- checkpoint：checkpoint_120000 最新；下個解碼點 iter600 約再 ~3.5h。
- 判定：SA1 飽和平穩、aux 平台、ent 平衡 ~2.0。無異常,例行續訓。

**2026-06-24 00:58 — SA1 iter ~405**
- 導航：SR 98.7→98.6→**98.7%@400**（飽和平穩）；CR 1.2→1.3→**1.3%**；TO 0%；ent 1.95→2.09→**2.13**（角 floor 撐探索續緩升,SR 仍 98.7% 故不擔心）；sw 0.014-0.015；fps 3772-3783（util 37%）；GPU 12.7GB；etime 8:55h。
- ★aux：grad 全非0（predict_head 3seg 0.294/0.279/0.275、rnn 0.072/0.066/0.061、extractor 0.070/0.046/0.042）。preprocess_loss 3seg **2.915/2.514/2.487**（穩 2.49 平台）。
- checkpoint：**checkpoint_120000 已生成**（iter400）；下個解碼點 iter600(checkpoint_180000) 約再 ~4h。
- 判定：SA1 飽和平穩、aux 平台、ent 緩升 2.13（SR 健康故 OK）。無異常,例行續訓。

**2026-06-24 00:32 — SA1 iter ~386**
- 導航：SR 98.5→98.7→**98.6%@380**（飽和平穩）；CR 1.3→1.2→**1.3%**；TO 0%；ent 1.87→1.97→**1.95**（★探索回升到平衡 ~1.95,角 floor 達穩態）；sw 0.014-0.015；fps 3752-3780（util 29%）；GPU 12.7GB；etime 8:29h。
- ★aux：grad 全非0（predict_head 3seg 0.296/0.282/0.272、rnn 0.073/0.066/0.060、extractor 0.071/0.046/0.042）。preprocess_loss 3seg **2.919/2.524/2.502**（穩 2.50 平台）。
- checkpoint：checkpoint_90000 最新；下個解碼點 iter600 約再 ~4.5h。
- 判定：SA1 飽和平穩、aux 平台、ent 達平衡 1.95。無異常,例行續訓。

**2026-06-24 00:06 — SA1 iter ~367**
- 導航：SR 98.6→98.5→**98.6%@360**（飽和平穩）；CR 1.4→1.3→**1.3%**；TO 0%；ent 1.77→1.87→**1.93**（角 floor 持續把探索往上撐,健康)；sw 0.014-0.015；fps 3752-3780（util 28%）；GPU 12.7GB；etime 8:03h。
- ★aux：grad 全非0（predict_head 3seg 0.297/0.283/0.269、rnn 0.073/0.066/0.061、extractor 0.072/0.045/0.043）。preprocess_loss 3seg **2.923/2.546/2.487**（穩 2.49 平台,均值擬合飽和）。
- checkpoint：checkpoint_90000 最新；下個解碼點 iter600 約再 ~5h。
- 判定：SA1 飽和平穩、aux 平台、ent 健康回升至 1.93。無異常,例行續訓。

**2026-06-23 23:40 — SA1 iter ~348**
- 導航：SR 98.6→98.7→**98.6%@340**（飽和平穩）；CR 1.3→1.3→**1.4%**；TO 0%；ent 1.66→1.66→**1.77**（角 floor 續撐探索回升）；sw 0.015；fps 3749-3792（util 47%）；GPU 12.7GB；etime 7:37h。
- ★aux：grad 全非0（predict_head 3seg 0.299/0.282/0.269、rnn 0.073/0.066/0.061、extractor 0.073/0.046/0.043）。preprocess_loss 3seg **2.934/2.570/2.467**（穩 2.47,均值擬合飽和平台）。
- checkpoint：checkpoint_90000 最新；下個解碼點 iter600 約再 ~5.5h。
- 判定：SA1 飽和平穩、aux loss 平台 2.47、ent 健康回升。無異常,例行續訓。

**2026-06-23 23:14 — SA1 iter ~328**
- 導航：SR **98.6%@320**（飽和平穩,連多輪）；CR 1.3→1.4→**1.3%**；TO 0%；ent 1.36→1.53→**1.66**（★角 floor 持續把探索撐回升,SA2 過渡前狀態佳）；sw 0.014-0.015；fps 3749-3792（他人 1 小 job,util 32%）；GPU 12.7GB；etime 7:11h。
- ★aux：grad 全非0（predict_head 3seg 0.301/0.283/0.268、rnn 0.073/0.065/0.063、extractor 0.075/0.046/0.044）。preprocess_loss 3seg **2.950/2.602/2.457**（穩 2.46 低檔,均值擬合飽和）,last5=[2.25,2.28,2.27,2.89,2.61]。
- checkpoint：checkpoint_90000 最新；下個解碼點 iter600(checkpoint_180000) 約再 ~6h。
- 判定：SA1 飽和平穩、aux 均值擬合飽和(loss 平在 2.46);ent 回升證明探索健康。無異常,例行續訓。

**2026-06-23 22:46 — SA1 iter 300 ★★第二個 aux 解碼硬數字（已讀）**
- 導航：SR **98.6%@300**（飽和）；CR 1.3%；TO 0%；ent 1.36；sw 0.014；fps 3776；GPU 12.7GB；etime 6:44h(train 健康,_step 92400=iter308)。
- ★★解碼結果(checkpoint_90000, stage1, n=1600)：**速度預測相對誤差仍 100%**(MAE 0.734 ≈ 真實均 0.734)。預測仍**近似常數**(近1 x≈-0.02/y≈-0.30、近2 x≈-0.35/d≈-0.40,不隨 step/真實值變)。
- ★關鍵判讀：**iter300 與 iter100 同結果(都 100%、都猜常數均值)**。preprocess_loss 從 3.0→2.45 的下降**不是「學會追蹤」,而是「把常數均值擬合得更準」(variance 收斂)**。RNN 在 SA1 稀疏障礙(dynamic 1-2)下只學到輸出固定均值。
- ⚠️ 這**符合預期**(SA1 signal 太稀疏)。真正驗收仍在 SA3-SA5 密集障礙：若那裡 rel err 真的 <100% → reluFix 讓 RNN 學會追蹤；若仍卡 100% → vanilla RNN(hidden64) 容量不足,需架構改(GRU/加 hidden/調 aux seq)。
- 後續解碼點：iter600(checkpoint_180000)、完訓(checkpoint_270000) 再各跑一次;但 SA1 大概率全程 100%(稀疏),重點放 SA3+。
- GPU 清理：decode 跑完已釋放(pgrep 殘影是自身 shell 指令字串誤判,實際 proc 已退)。

**2026-06-23 22:29 — SA1 iter ~296（解碼點前夕）**
- 導航：SR 98.6→98.7→**98.6%@290**（飽和平穩）；CR 1.2→1.2→**1.3%**；TO 0%；ent 1.24→1.16→**1.21**（角 floor 穩）；sw 0.014→0.015；fps 3783-3795（他人 job 退,速度穩定,util 30%）；GPU 12.7GB；etime 6:26h。
- ★aux：grad 全非0（predict_head 3seg 0.305/0.278/0.269、rnn 0.073/0.067/0.063、extractor 0.077/0.048/0.044）。preprocess_loss 3seg **2.967/2.646/2.479**（末段 2.48,穩低檔）,last5=[2.36,2.32,2.73,2.69,2.24]。
- checkpoint：仍 checkpoint_60000（_step 88800,iter296）；checkpoint_90000(iter300) 約再 ~5min 生成 → **改排短 wakeup 等 ckpt 落地再解碼,避免用舊 ckpt**。
- 判定：SA1 飽和平穩、aux 穩定。下輪即 iter300 首次趨勢解碼。

**2026-06-23 22:03 — SA1 iter ~276**
- 導航：SR 98.7→98.6→**98.6%@270**（飽和平穩）；CR 1.2→1.3→**1.2%**；TO 0%；ent 1.40→1.35→**1.24**（角 floor 穩）；sw 0.015→**0.014**；fps 3786-3792（他人 job 退,速度回升,util 35%）；GPU 12.7GB；etime 6:00h。
- ★aux：grad 全非0（predict_head 3seg 0.305/0.282/0.272、rnn 0.073/0.068/0.064、extractor 0.079/0.049/0.045）。preprocess_loss 3seg **2.982/2.671/2.478**（末段 2.48,穩低檔）,last5=[2.57,2.82,2.44,2.63,2.60]。
- checkpoint：checkpoint_60000 最新；解碼點 iter300(90000) 約再 ~30min（~75s/iter,他人 job 退後加速）。
- 判定：SA1 飽和平穩、aux loss 穩 2.48。無異常。★下輪巡檢很可能就是 iter300 解碼點（checkpoint_90000）— 屆時跑首次趨勢解碼對比 iter100 100%。

**2026-06-23 21:37 — SA1 iter ~257**
- 導航：SR 98.6→98.7→**98.6%@250**（飽和平穩）；CR 1.3→1.2→**1.3%**；TO 0%；ent 1.31→1.38→**1.40**（角 floor 穩 ~1.3-1.4）；sw 0.015；fps 3544-3660（他人 1 job 已退,util 29%）；GPU 12.7GB（降低,他人 job 少一個）；etime 5:34h。
- ★aux：grad 全非0（predict_head 3seg 0.309/0.278/0.280、rnn 0.073/0.069/0.065、extractor 0.081/0.049/0.045）。preprocess_loss 3seg **2.983/2.729/2.453**（末段 2.45,穩定低檔）,last5=[2.75,2.36,2.33,2.62,2.37]。
- checkpoint：checkpoint_60000 最新；解碼點 iter300(90000) 約再 ~60min。
- 判定：SA1 飽和平穩、aux loss 穩在 2.45 低檔。無異常。下個解碼點 iter300 估再 2 輪巡檢後到。

**2026-06-23 21:10 — SA1 iter ~238**
- 導航：SR **98.6%@230**（飽和平穩,連 4 輪 98.5-98.6）；CR **1.3%**；TO 0%；ent 1.38→**1.31**（角 floor 穩 ~1.3）；sw 0.015；fps 3293-3659（他人 job 競爭,util 35%）；GPU 14.2GB；etime 5:07h。
- ★aux：grad 全非0（predict_head 3seg 0.311/0.278/0.283、rnn 0.073/0.070/0.065、extractor 0.083/0.050/0.044）。preprocess_loss 3seg **2.999/2.775/2.458**（末段 2.46,續降）,last5=[2.64,2.08,2.15,2.26,2.57] → **下降延續,最低 2.08**。
- checkpoint：checkpoint_60000 最新；解碼點 iter300(90000) 約再 ~90min（~87s/iter）。
- 判定：SA1 飽和平穩、aux loss 穩定壓低（3.0→2.46）。無異常。下個解碼點 iter300 估再 2-3 輪巡檢後到。

**2026-06-23 20:44 — SA1 iter ~220**
- 導航：SR 98.5→98.6→**98.6%@220**（飽和穩定）；CR 1.4→1.2→**1.3%**；TO 0%；ent 1.35→1.32→**1.38**（角 floor 穩住 ~1.3-1.4）；sw 0.016→**0.015**；fps 3293-3659（他人 job 競爭,util 42%）；GPU 14.3GB；etime 4:41h。
- ★aux：grad 全非0（predict_head 3seg 0.315/0.271/0.288、rnn 0.072/0.072/0.065、extractor 0.084/0.052/0.045）。preprocess_loss 3seg **3.017/2.788/2.514**（末段壓到 2.51,續降）,last5=[1.93,2.75,2.23,2.11,2.52] → **下降延續,最低觸及 1.93**。
- checkpoint：checkpoint_60000 最新；解碼點 iter300(90000) 約再 ~50min。
- 判定：SA1 飽和健康（SR98.6%）；ent 角 floor 穩定撐住探索；aux loss 持續壓低（3.0→2.5）。下個動作 iter300 跑首次趨勢解碼（vs iter100 100%）。

**2026-06-23 20:17 — SA1 iter ~202**
- 導航：SR 98.4→98.5→**98.5%@200**（飽和高原穩定）；CR 1.5→1.4→**1.4%**；TO 0%；ent 1.00→1.26→**1.35**（★回升,角 ent floor 0.02 撐住探索,正是 SA2 過渡前要的）；sw 0.016；fps 3293-3626（他人 job 競爭,util 54%）；GPU 14.4GB；etime 4:14h。
- ★aux：grad 全非0（predict_head 3seg 0.318/0.269/0.286、rnn 0.071/0.073/0.067、extractor 0.086/0.054/0.046）。preprocess_loss 3seg **3.020/2.810/2.568**（末段 2.57,續壓低）,last5=[2.23,2.44,2.76,2.70,2.41] → **下降趨勢延續穩固**。
- checkpoint：**checkpoint_60000 已生成**（iter200）；解碼點 iter300(90000) 約再 ~1.3h。
- 判定：SA1 完全飽和（SR98.5%、CR1.4%）;ent 回升證明角探索 floor 生效（SA2 進密集障礙不會卡死探索）;aux loss 穩定下降。下個動作 iter300 跑首次趨勢解碼（vs iter100 的 100%）。

**2026-06-23 19:51 — SA1 iter ~184**
- 導航：SR 98.2→98.4→**98.4%@180**（高原飽和,SA1 簡單）；CR 1.6→1.4→**1.5%**；TO 0%；ent 1.60→1.04→**1.00**（續收斂,SA1 飽和正常）；sw 0.018→0.017→**0.016**；fps 3574-3652（他人 job 競爭,util 53%）；GPU 15.5GB（仍安全,他人 2 小 job 1.2GB）；etime 3:48h。
- ★aux：grad 全非0（predict_head 3seg 0.322/0.271/0.286、rnn 0.070/0.076/0.065、extractor 0.089/0.055/0.047）。preprocess_loss 3seg **3.040/2.807/2.637**，last5=[2.25,2.53,2.68,2.90,2.59] → **末段續壓到 2.6**（趨勢延續向下,中位數穩定降）。
- checkpoint：仍 checkpoint_30000；iter200(60000) 即將生成；解碼點 iter300(90000)。
- 判定：SA1 導航完全飽和（SR98.4%、CR1.5%、ent1.0）；aux loss 持續下降中（reluFix 生效徵兆穩固）。⚠️ 進度受 GPU 競爭拖慢 ~80s/iter，到 iter300 解碼點約再 2.5h、到完訓 iter900 約再 16h。iter300 速度預測解碼仍是硬驗收。

**2026-06-23 19:24 — SA1 iter ~165**
- 導航：SR 94.3→97.0→**98.2%@160**（持續爬升,SA1 bootstrap 近飽和）；CR 5.8→3.1→**1.6%**；TO 0%；ent 3.20→2.25→**1.60**（角探索快速收斂,SA1 簡單階段正常）；sw 0.030→0.020→**0.018**（很平滑）；fps 3642-3754（他人 job 競爭,util 47%）；GPU 14.2GB；etime 3:21h。
- ★aux：grad 全非0（predict_head 3seg 0.326/0.275/0.277、rnn 0.070/0.076/0.067、extractor 0.092/0.057/0.048）。preprocess_loss 3seg **3.055/2.839/2.733**，last5=[2.59,2.92,2.44,2.31,2.69] → **下降趨勢延續**（末段壓到 2.3-2.7,雜訊大但中位數續降）。
- checkpoint：仍 checkpoint_30000；save_interval=100 → 下個 iter200(60000)；解碼點 iter300(90000)。
- 判定：SA1 導航近飽和（SR98%、CR1.6%）；aux loss 穩定下降。⚠️ ent 已掉到 1.6（SA1 簡單故快收斂屬正常,但 SA2+ 進密集障礙時要看 ent 會不會被 floor 撐回探索）。iter300 解碼仍是速度預測硬驗收。

**2026-06-23 18:57 — SA1 iter ~146**
- 導航：SR 88.8→90.8→**94.3%@140**（明顯躍升,健康收斂）；CR 11.4→9.4→**5.8%**；TO 0%；ent 4.15→3.99→**3.20**（角探索開始收斂）；sw 0.059→0.048→**0.030**（更平滑）；fps 4355→3841（他人 job 競爭,util 33%）；GPU 14.3GB；etime 2:54h。
- ★aux：grad 全非0（predict_head 3seg 0.327/0.283/0.270、rnn 0.070/0.076/0.070、extractor 0.097/0.058/0.050）。preprocess_loss 3seg **3.041/2.890/2.779**，★**last5=[2.95,2.84,2.70,2.53,2.61]** — 末段明顯壓到 2.5-2.6（iter100 時還 ~2.9 平）→ **下降趨勢轉明確,非雜訊**。
- checkpoint：仍只 checkpoint_30000（iter100）；下個 iter200(60000)、iter300(90000=解碼點)。
- 判定：導航躍升 SR94%（reluFix lineage SA1 表現優於預期）；aux loss 下降趨勢轉明確（謹慎樂觀↑）；iter300 解碼仍是相對誤差硬驗收點。

**2026-06-23 18:43 — SA1 iter ~125**
- 導航：SR 86→88→**88.8%@120**（穩定高原,健康）；CR 14→11.4%；TO 0%；ent ~4.1；sw 0.059→0.064；fps 4355；GPU 14.2GB（他人 2 小 job 626+588MiB,無 OOM 風險）；etime 2:26h；PID 1647188 alive。
- ★aux：grad 全非0（predict_head 3seg 0.324/0.297/**0.270**、rnn 0.071/0.074/0.074 穩、extractor 0.104/0.058/**0.053**）。preprocess_loss 3seg **2.996/2.986/2.782**（last5 含 2.421，雜訊大但末段均 2.78 < 首段 3.0，與上輪 2.98/3.01/2.78 一致 → 緩降趨勢延續但很慢）。console AUX 行 VE=0.15-0.18、valid~16.5萬。
- 進度偏慢（上輪 iter122→本輪 125，~70s/iter，他人 job 競爭 GPU util 僅 30%）。非異常，續訓。
- 判定：導航健康；aux 緩降趨勢穩定延續（非單輪雜訊）但仍平緩；下個解碼硬點 iter300（~checkpoint_90000）看相對誤差是否真 <100%。

**2026-06-23 18:26 — SA1 iter ~122**
- 導航：SR 86→88→**88.8%@120**（穩定爬升,健康）；CR 14→11.4%；ent ~4.1；sw 0.059；fps 4355；GPU 14.2GB；etime 2:23。
- ★aux：grad 非0（predict_head 0.323→0.274、rnn 0.071→0.075、extractor 0.105→0.053）。preprocess_loss 三段 2.983/3.009/**2.785** — **第3段較前兩段明顯低,弱但漸明的下彎趨勢**(上輪是 2.94/3.13/2.86,這輪 2.98/3.01/2.78,末段續降)→ 預測可能開始學了。
- 判定：導航健康(SR88.8%);aux loss 出現較明顯下彎(3.0→2.78),謹慎樂觀但仍早;下個解碼點 iter300 看相對誤差有沒有真的 <100%。


---

## SA3_v3f reluFix（含 head_on）— 巡檢 2026-06-25 03:01 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 01:30:47）| wandb 6zkvago1 | 從 SA2 checkpoint_150000 續訓 | config wd_sa3_v3f = stage3 + v3e + head_on 0.20。

- **進度**：iter 70/1400（log mtime 02:57，距今 ~3.5min → 無 stall；fps≈4420）。尚無 checkpoint（目標 checkpoint_150000 ≈ iter500 停→解碼）。
- **導航**：SR 72.7% → **75.0%** → 74.2%（穩定回升，自 head_on 加入後的 ~74% 持續往上）；CR 28.2% → **26.0%** → 26.6%（同步下降 = 開始學會避正面來車 head_on）；TO 0%；sw 0.013（抗抽動健康）；ent 2.44 → 2.91（探索充足）。
- **★aux（關鍵）**：`aux/preprocess_loss` **持續負值**：3seg = −0.228 / −0.179 / **−0.263**、last5 = **−0.204**（穿 76 iter，非瞬態）→ |pred−target| < 1 = RNN 預測接近真值 = **學追蹤跡象延續**。console AUX 行甚至看到 loss=−0.5428 的瞬時更深負值。
- **grad norms**：predict_head 0.323（健康非 0，predict_head 在學）、rnn 0.034、extractor 0.022 — 三者皆正常更新。
- **GPU**：util 31% / mem 12.0GB（單 python pid 2457367），無別人 job、無 OOM 風險。
- **判定**：導航健康（SR 續升、CR 續降 = head_on 提早避讓在學）；aux 自 SA1/SA2 的 +2.48 plateau 已**明確翻負並維持**，這是 reluFix 後首次穩定負值 → 強烈 RNN 追蹤跡象。**待 iter500（checkpoint_150000）停 SA3 → 解碼 stage 3 看 ★rel err 是否 < 100% 定案**。
- **iter 速率實測**：02:32 iter50 → 03:01 iter70 ≈ 1.45 min/iter → 到 iter500 約還需 ~10h。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 03:33 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 02:02:57）| wandb 6zkvago1。

- **進度**：iter **100/1400**（log mtime 03:31，距今 ~1.5min → 無 stall；fps≈4419）。**首個 checkpoint_30000 已存**（確認命名 = iter×300，目標 checkpoint_150000 = iter500，每 100 iter 存一次）。
- **導航**：SR 74.9%（穩在 ~75%）；CR 26.2%（穩降）；TO 0%；sw 0.013；ent 2.60（探索仍足）。
- **★aux**：`aux/preprocess_loss` 3seg = −0.209 / −0.236 / **−0.239**、last5 = **−0.345**（n=104，**續降中**，console 瞬時 loss=−0.6756）→ 預測誤差持續縮小 = RNN 追蹤跡象**加強**。
- **grad**：predict_head 0.346、rnn 0.052、extractor 0.024 — 全健康更新。
- **GPU**：28% / 12.0GB，無別人 job、無 OOM。
- **判定**：穩定健康，aux 負值續深（last5 −0.345 比上輪 −0.204 更負）→ RNN 學追蹤訊號愈發明確。等 iter500 / checkpoint_150000 停機解碼定案。距 iter500 約還 ~10h。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 04:05 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 02:34:56）| wandb 6zkvago1。

- **進度**：iter **130/1400**（log mtime 04:02，距今 ~3min → 無 stall；fps≈4439）。checkpoint 仍 30000（下個 60000=iter200）。
- **導航**：SR 74.4%、CR 26.5%（在 ~74-75% / ~26% 區間平穩微震，正常）；TO 0%；sw 0.013；ent 2.82。
- **★aux**：preprocess_loss 3seg = −0.199 / −0.250 / −0.217、**last5 = −0.413**（n=131，持續負且 last5 續深，console 瞬時 −0.377）→ RNN 追蹤訊號穩定維持。
- **grad**：predict_head 0.251、rnn 0.041、extractor 0.032 — 健康。
- **GPU**：41% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康穩定，aux 持續負值。等 iter500 / checkpoint_150000 解碼定案（距今約 ~9h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 04:37 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 03:06:55）| wandb 6zkvago1。

- **進度**：iter **160/1400**（log mtime 04:36，距今 ~15s → 無 stall；fps≈4438）。checkpoint 仍 30000（下個 60000=iter200 即將）。
- **導航**：SR **75.0%**、CR **25.8%**（CR 首次跌破 26%，至今最佳）；TO 0%；sw 0.013；ent 2.60。
- **★aux**：preprocess_loss 3seg = −0.209 / −0.238 / **−0.283**（第3段最深，下彎續行）、last5 −0.199（console 瞬時 −0.74）→ RNN 追蹤訊號穩定。
- **grad**：predict_head 0.299、rnn 0.056、extractor 0.035 — 健康。
- **GPU**：37% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，CR 緩降至 25.8%（head_on 避讓在學）、aux 持續負值。等 iter500 / checkpoint_150000 解碼定案（距今約 ~8h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 05:09 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 03:38:54）| wandb 6zkvago1。

- **進度**：iter **180/1400**（log mtime 05:06，距今 ~2.5min → 無 stall；fps≈4425）。checkpoint 仍 30000（下個 60000=iter200 即將）。
- **導航**：SR 74.7%、CR 26.3%（在 ~75% / ~26% 平穩微震）；TO 0%；sw 0.013；ent 2.56；VE 0.705。
- **★aux**：preprocess_loss 3seg = −0.225 / −0.202 / **−0.373**（第3段明顯最深，下彎加速）、**last5 = −0.428**（console 瞬時 −0.71）→ RNN 追蹤訊號**持續加強**。
- **grad**：predict_head 0.282、rnn 0.047、extractor 0.029 — 健康。
- **GPU**：33% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，aux 第3段續深至 −0.37、last5 −0.43，RNN 學追蹤趨勢愈發明確。等 iter500 / checkpoint_150000 解碼定案（距今約 ~7.5h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 05:40 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 04:09:54）| wandb 6zkvago1。

- **進度**：iter **210/1400**（log mtime 05:36，距今 ~3.5min → 無 stall；fps≈4431）。**checkpoint_60000 已存**（iter200，下個 90000=iter300）。
- **導航**：SR **75.3%**、CR **25.4%**（雙雙刷新最佳：SR 更高、CR 更低）；TO 0%；sw 0.012；ent 2.73；VE 0.718。
- **★aux**：preprocess_loss 3seg = −0.225 / −0.244 / **−0.361**、last5 = **−0.404**（n=213，持續負、第3段最深）→ RNN 追蹤訊號穩定加強。
- **grad**：predict_head 0.306、rnn 0.057、extractor 0.032 — 健康。
- **GPU**：37% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，SR/CR 雙刷新（75.3%/25.4%，head_on 避讓在學）、aux 第3段 −0.36。等 iter500 / checkpoint_150000 解碼定案（距今約 ~7h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 06:12 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 04:41:54）| wandb 6zkvago1。

- **進度**：iter **240/1400**（log mtime 06:11，距今 ~45s → 無 stall；fps≈4455）。checkpoint 60000（下個 90000=iter300）。
- **導航**：SR 75.0%、CR 25.9%（穩在 ~75% / ~26%）；TO 0%；sw 0.013；ent 2.55；VE 0.704。
- **★aux**：preprocess_loss 3seg = −0.226 / −0.267 / **−0.354**、last5 = −0.342（n=241，持續負、第3段最深穩定 ~−0.35）→ RNN 追蹤訊號穩定維持。
- **grad**：predict_head 0.336、rnn 0.058、extractor 0.028 — 健康。
- **GPU**：35% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康穩定，aux 第3段續守 −0.35。等 iter500 / checkpoint_150000 解碼定案（距今約 ~6.5h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 06:43 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 05:12:54）| wandb 6zkvago1。

- **進度**：iter **260/1400**（log mtime 06:41，距今 ~1.5min → 無 stall；fps≈4447）。checkpoint 60000（下個 90000=iter300）。
- **導航**：SR 75.0%、CR 25.8%（穩在 ~75% / ~26% 高原）；TO 0%；sw 0.013；ent 2.49；VE 0.715。
- **★aux**：preprocess_loss 3seg = −0.232 / −0.285 / **−0.356**、last5 = −0.412（n=268，持續負、第3段穩守 ~−0.36）→ RNN 追蹤訊號穩定。
- **grad**：predict_head 0.304、rnn 0.044、extractor 0.023 — 健康。
- **GPU**：28% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康穩定，高原期。等 iter500 / checkpoint_150000 解碼定案（距今約 ~6h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 07:14 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 05:43:56）| wandb 6zkvago1。

- **進度**：iter **290/1400**（log mtime 07:11，距今 ~2min → 無 stall；fps≈4437）。checkpoint 60000（下個 90000=iter300 即將）。
- **導航**：SR **75.8%**、CR **24.7%**（雙雙刷新：SR 最高、CR 首次跌破 25%！）；TO 0%；sw 0.013；ent 2.65；VE 0.710。
- **★aux**：preprocess_loss 3seg = −0.225 / −0.303 / **−0.371**（單調下彎 −0.23→−0.30→−0.37）、last5 −0.295（console 瞬時 −0.42）→ RNN 追蹤訊號穩定加深。
- **grad**：predict_head 0.325、rnn 0.056、extractor 0.026 — 健康。
- **GPU**：35% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，**CR 首破 25%（24.7%）+ SR 75.8%**＝ head_on 避讓持續見效；aux 三段單調下彎。等 iter500 / checkpoint_150000 解碼定案（距今約 ~5h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 07:46 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 06:15:55）| wandb 6zkvago1。

- **進度**：iter **320/1400**（log mtime 07:42，距今 ~4min → 無 stall；fps≈4432）。**checkpoint_90000 已存**（iter300，下個 120000=iter400）。
- **導航**：SR 75.5%、CR 25.4%（穩在 ~75.5% / ~25%）；TO 0%；sw 0.013；ent 2.44；VE 0.722。
- **★aux**：preprocess_loss 3seg = −0.221 / −0.335 / **−0.343**（三段續守深負）、last5 = +0.051（近 5 iter 輕微正跳，瞬態雜訊，console −0.08~−0.42 區間振盪）→ 整體 RNN 追蹤訊號仍強，僅末端瞬時抖動。
- **grad**：predict_head 0.277、rnn 0.048、extractor 0.028 — 健康。
- **GPU**：39% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，導航 ~75.5% / ~25% 高原；aux 第3段深負（−0.34），last5 瞬時正跳屬雜訊不擔心。距 iter500 / checkpoint_150000 解碼僅剩 ~180 iter（約 ~4.5h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 08:18 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 06:47:55）| wandb 6zkvago1。

- **進度**：iter **350/1400**（log mtime 08:16，距今 ~1min → 無 stall；fps≈4465）。checkpoint 90000（下個 120000=iter400）。
- **導航**：SR **75.9%**、CR **24.9%**（iter340 瞬時達 SR 76.5% / CR 24.3%，續刷新）；TO 0%；sw 0.013；ent 2.44；VE 0.722。
- **★aux**：preprocess_loss 3seg = −0.218 / −0.350 / **−0.343**、last5 = **−0.489**（n=350，深負穩定，last5 回到深負 −0.49，上輪正跳確認為瞬態）→ RNN 追蹤訊號強。
- **grad**：predict_head 0.319、rnn 0.047、extractor 0.028 — 健康。
- **GPU**：30% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，SR 76% 帶 CR 跌至 ~24-25%（head_on 避讓持續見效）、aux last5 −0.49 確認上輪正跳是雜訊。距 iter500 / checkpoint_150000 解碼剩 ~150 iter（約 ~3.7h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 08:51 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 07:21:06）| wandb 6zkvago1。

- **進度**：iter **380/1400**（log mtime 08:50，距今 ~30s → 無 stall；fps≈4395）。checkpoint 90000（下個 120000=iter400 即將）。
- **導航**：SR **76.9%**、CR **24.0%**（iter370 瞬時 SR 77.1% / CR 23.5%，穩定爬升）；TO 0%；sw 0.013；ent 2.63；VE 0.723。
- **★aux**：preprocess_loss 3seg = −0.215 / −0.358 / **−0.373**（三段單調續深）、last5 = −0.339（n=379）→ RNN 追蹤訊號強且穩。
- **grad**：predict_head 0.316、rnn 0.066、extractor 0.029 — 健康。
- **GPU**：36% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，**SR 已達 77% / CR 24%**（head_on 避讓持續見效，SR 自 SA3 起點 72.7% 累升 ~4pp）、aux 三段續深 −0.37。距 iter500 / checkpoint_150000 解碼剩 ~120 iter（約 ~3h）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 09:23 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 07:52:55）| wandb 6zkvago1。

- **進度**：iter **400/1400**（log mtime 09:22，距今 ~45s → 無 stall；fps≈4433）。**checkpoint_120000 已存**（iter400，下個 150000=iter500=★解碼點）。
- **導航**：SR **77.4%**、CR **23.4%**（雙雙刷新最佳；iter390 76.4%/24.5%）；TO 0%；sw 0.013；ent 2.64；VE 0.722。
- **★aux**：preprocess_loss 3seg = −0.230 / −0.349 / **−0.372**、last5 = −0.219（n=407，三段深負穩定，console 瞬時 −0.59）→ RNN 追蹤訊號強。
- **grad**：predict_head 0.343、rnn 0.045、extractor 0.024 — 健康。
- **GPU**：48% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，**SR 77.4% / CR 23.4%**（自 SA3 起點 72.7%/28.2% 累進 ~5pp）、aux 三段續守 −0.37。距 iter500 / checkpoint_150000 ★解碼點剩 ~100 iter（約 ~2.5h，下下輪可能觸發停機+解碼）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 09:55 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 08:24:55）| wandb 6zkvago1。

- **進度**：iter **430/1400**（log mtime 09:53，距今 ~2min → 無 stall；fps≈4442）。checkpoint 120000（下個 150000=iter500=★解碼點，剩 ~70 iter）。
- **導航**：SR 76.7%、CR 24.1%（穩在 ~77% / ~24%）；TO 0%；sw 0.013；ent 2.64；VE 0.716。
- **★aux**：preprocess_loss 3seg = −0.231 / −0.367 / **−0.384**、last5 = −0.413（n=434，三段單調續深，至今最深 −0.38）→ RNN 追蹤訊號強。
- **grad**：predict_head 0.316、rnn 0.047、extractor 0.022 — 健康。
- **GPU**：36% / 12.0GB，無別人 job、無 OOM。
- **判定**：健康，導航 ~77% / ~24% 高原、aux 第3段至今最深 −0.38。距 iter500 / checkpoint_150000 ★解碼點剩 ~70 iter（約 ~1.7h，下一輪可能觸發停機+解碼）。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 10:27 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 08:56:56）| wandb 6zkvago1。

- **進度**：iter **460/1400**（log mtime 10:24，距今 ~2.5min → 無 stall；fps≈4314）。checkpoint 120000（下個 150000=iter500=★解碼點，剩 ~40 iter ≈ ~58min）。
- **導航**：SR **78.1%**、CR **22.8%**（雙雙刷新最佳；iter450 77.5%/23.3%）；TO 0%；sw 0.012；ent 2.65；VE 0.722。
- **★aux**：preprocess_loss 3seg = −0.245 / −0.374 / **−0.402**（第3段首破 −0.40，至今最深）、last5 = −0.225（console 瞬時 +0.12 為單點抖動）→ RNN 追蹤訊號強。
- **grad**：predict_head 0.344、rnn 0.031、extractor 0.019 — 健康。
- **GPU**：32% / 12.8GB；★多一個 pid 2595866=gnome-remote-desktop-daemon(588MiB,遠端桌面非訓練 job,benign)，無 OOM 風險。
- **判定**：健康，**SR 78.1% / CR 22.8%**（自 SA3 起點累進 ~5.4pp）、aux 第3段首破 −0.40。距 iter500 解碼點剩 ~40 iter，**下下輪預計觸發停機 → 解碼 stage3 → 接 SA4**。

---

## SA3_v3f reluFix — 巡檢 2026-06-25 10:59 CST

**Run**：sa3_v3f_reluFix_ne1024_s42 | PID 2457352（ALIVE，etime 09:28:55）| wandb 6zkvago1。

- **進度**：iter **490/1400**（log mtime 10:57，距今 ~1.5min → 無 stall；fps≈4426）。checkpoint 120000（★下個 150000=iter500 解碼點剩 ~10 iter ≈ 15min，下輪縮短至 17min 觸發停機）。
- **導航**：SR **78.6%**、CR **22.2%**（iter480 瞬時 78.8%/21.8%，續創新高）；TO 0%；sw 0.013；ent 2.83；VE 0.718。
- **★aux**：preprocess_loss 3seg = −0.248 / −0.353 / **−0.469**（第3段至今最深）、last5 = **−0.639**（至今最深，續降中）→ RNN 追蹤訊號強且加深。
- **grad**：predict_head 0.312、rnn 0.041、extractor 0.024 — 健康。
- **GPU**：33% / 12.8GB（含遠端桌面 daemon），無 OOM。
- **判定**：健康，**SR ~78.8% / CR ~22%**（自 SA3 起點累進 ~6pp）、aux 三段續深至 −0.47、last5 −0.64。★下輪（~17min）預計 checkpoint_150000 出現 → 停 SA3 → 解碼 stage3 → 接 SA4。

---

## SA3_v3f reluFix — ★iter500 停機 + 解碼觸發 2026-06-25 11:20 CST

**SA3 完訓快照**（停機前，iter500/1400，shortened budget 達標）：
- Run sa3_v3f_reluFix_ne1024_s42 | 跑 etime 09:48 | checkpoint_150000 已存（11:09）。
- **導航終值**：iter480 峰值 SR **78.8%** / CR **21.8%**；iter500 收 SR 76.5% / CR 24.1%（stochastic rollout 振盪，deterministic 解碼為準）。自 SA3 起點 72.7%/28.2% → 累進 SR +~6pp / CR −~6pp（head_on 正面避讓有學到）。
- **★aux 終值**：preprocess_loss 3seg = −0.250 / −0.351 / **−0.479**、last5 = −0.568（全程深負、單調續深，至今最深第3段 −0.48）。
- grad：predict_head 0.303 / rnn 0.056 / extractor 0.024（全程健康）。

**動作**：kill SA3 wrapper 2457352 + GPU child 2457367 → GPU 釋放（util 0%，僅剩 gnome-remote-desktop daemon 588MiB）→ 啟動 **stage-3 deterministic 解碼**（/tmp/aux_decode_sa3.sh, PID 2623196, checkpoint_150000, --stage 3 v3e, num_envs 32 steps 250, lidar r_min 0.25 no_noise, aux_debug）→ log /tmp/aux_decode_sa3.log。
**待讀**：★★rel err 是否 <100%（<100% = RNN 真學會追蹤障礙運動 = reluFix+head_on 成功，SA1/SA2 都 100%）+ 近1預測是否 signed nonzero。解碼完即接 SA4（wd_sa4_v3f，從 checkpoint_150000）。

---

## ★★ SA3_v3f reluFix — stage-3 解碼定案 2026-06-25 11:21 CST（關鍵負面結果）

**解碼**：checkpoint_150000（iter500）, stage3 v3e deterministic, 32 env × 250 步, n=2400+。

### ★★ velocity rel err = **100%**（MAE 0.748 ≈ 真實速度均 0.747）— 與 SA1/SA2 完全相同
→ **RNN 仍未學會追蹤障礙速度**（速度 dims 7-12 輸出 ≈ 0，純猜均值/零）。

### 拆解（誠實）
- **predict_dim=13**（非 7D）：weight=[1,1,1,1,0.7,0.7,0,**0.8,0.8,0.5,0.5,0.3,0.3**]，dims 7-12 = top-3 障礙 vx/vy = 速度目標**有在訓練**。
- **位置 dims（0-5）結構活著但偏弱**：近1 預測 (x −0.00, y −0.31) vs 真實 (x +0.76, y −1.43)；近2 預測 (x −0.32, d −0.38) vs 真實 (x −1.32, d −0.52)。→ **signed nonzero（reluFix 結構修復成功，非 dead-ReLU 卡 0），但量級嚴重低估、偏向原點小值**。
- **速度 dims（7-12）功能性死**：速度解碼 rel err 100% = 輸出 ≈ 0。
- **導航卻很好**：det SR **83.8%** / CR 16.2%（牆5.0%+障礙11.2%）/ TO 0% / 平均28.8步。→ policy 用 LiDAR **反應式**避障，**不靠預測**。

### ★ 修正先前判讀（重要）
- 先前每 cycle 報「aux/preprocess_loss 深負 = RNN 在學追蹤」**過度樂觀**。深負是**位置 dims（權重 1.0 為主）變小誤差**驅動的，**不是**速度追蹤。
- 這正是先前向用戶說明的 **target-scale 混淆**成真：aux 負值 ≠ 速度 tracking；解碼 rel err（用真實速度正規化）才是黃金標準，它**否決**了 tracking。
- **核心假設「reluFix 復活 velocity-aux → RNN 追蹤動態 → 修晚反應」在 SA3 未獲證實**：速度 rel err 仍 100%，head_on（可預測直線）+ 更密障礙 + 500 iter 都沒讓速度 dims 脫離 ≈0。

### 待決（decode-gate 未通過，**未自動接 SA4**）
標準規則 [feedback_decode_each_stage_rnn_health] 要求解碼確認 RNN 健康才接下一棒；velocity health 未通過 → 暫停等用戶決策（診斷 velocity head vs 照接 SA4 vs 重評 lineage）。
⚠️ 注意：此 lineage 無「已知 <100%」的 velocity 解碼參考，不排除 decode 指標/速度 target 本身有問題，須一併診斷。

---

## posonly 實驗 #1 監控 2026-06-25 12:39 CST

**run** sa3_v3f_posonly_ne1024_s42 | PID 2682491(ALIVE etime 33:37) | wandb kifjkvg1 | 從 SA3 checkpoint_150000 續訓, predict_dim 7(position-only).
- iter **20/400**（log 12:38 無 stall, fps 4435）。導航 SR 77.9%/CR 22.7%（≈SA3，RL policy 不變只換 aux 維度）。無 checkpoint（首個 iter100）。
- aux/preprocess_loss 3seg −0.37/−0.43/−0.43、last5 −0.48；predict_head_grad 0.37（重初始化 head 適應中）、rnn 0.034、extractor 0.011。
- ⚠️ aux 負≠位置學會（先前坑）+ 7D 權重和 4.2≠13D 5.1 不可直比 → **唯一定論=iter300 解碼位置 rel err / 預測vs真實收斂**。現僅「訓練健康」。
- GPU 39%/13GB 無別人 job。

---

## posonly 實驗 #1 監控 2026-06-25 13:32 CST

iter **70/400** | SR 79.5%/CR 21.3%（健康略優）| log 13:31 無 stall | GPU 33%.
- aux/preprocess_loss 3seg −0.42/−0.55/−0.64、last5 −0.77（持續下降，比 13D 更深）。
- ⚠️ **rnn_grad 沒升、停 ~0.034**（0.031/0.033/0.037）；extractor 0.011 平；predict_head 0.39。
- ★解讀（黃旗）：loss 降但 rnn_grad 平 = 符合「predict_head 收斂到最佳常數（loss 降）、RNN 特徵未被重塑」圖像 → 可能又是 SA3 常數陷阱。iter72 仍早不定論。
- 決策：iter100/checkpoint_30000 一出現即**提前解碼**（不等 iter300），搶看真追蹤 vs 常數。

---

## WD-faithful 實驗啟動 2026-06-25 19:35
- run sa3_v3f_wdfreeze_ne1024_s42、wandb 3wusjcwu、PID /tmp/posonly.pid、log /tmp/sa3_wdfreeze_startup.log。
- lr 確認:rnn_cell 0.0005 / predict_head 0 / fc_middle 0 / fc_front 0 / extractor 0(全凍只訓 RNN,復刻 WD train_rnn_car spot_preprocess_model_lr=0 + rnn_model_lr 5e-4)。
- iter7:pos_rel_err ~100.4%、pos_r2 -0.004~-0.011(早)、rnn_grad 0.044→0.022。太早。
- ★注意:移植版凍的 fc_middle/extractor 是 SA3 訓練值(非 WD 隨機),predict_head 是 reinit 隨機。若 r2 不動考慮 fc_middle 也重初始化貼 WD。盯 iter45。

---

## ★★ SA1 vaux 完訓 2026-06-27 07:20 CST（Cycle74 收尾）— velocity-aux 修復 RL 端徹底驗證成功 ★★

**run** sa1_v3f_vaux_ne1024_s42 | wandb 42ly520e | `Training complete: 270,000 steps in 70465s (1174.4min)` | PID DEAD、GPU 已釋放(17MiB/0%)、無 NaN/OOM。

**最終 checkpoint**: `logs/rnn_car/sa1_v3f_vaux_ne1024_s42/checkpoint_270000.pt`
- ✅ 含 `feat_normalizer`（torch.load 驗證確認）→ 部署路徑正確,play/車端載入即套用。

**74 cycle（~15.7hr）全程健康史（每 15 分鐘監控,零中斷零提前 kill）**:
- 導航 SR 全程 **93.6–95.0%**（iter600/iter480 達 95.0%）、CR 5.0–7.4%、TO 0%；最終 iter900 SR 94.7%/CR 5.4%。
- **aux/pos_rel_err 全程 19.8–34%**（後段 iter850 首破 20% 至 **19.8%**;全程最佳 pos_r2 **0.9065**）——**徹底擺脫舊 lineage ~100% 常數崩塌**。
- entropy 中段自發探索高原（峰 ~0.45,Cycle52🟡 警示經兩輪追蹤確認良性,ent_coeff 固定 0.02/0.04 未變）→ 後段已收斂回 0.25–0.33,SR 全程不受影響。
- 干擾事件:Cycle22–26 laksh 共用 GPU（fps 3940→2159,確證非 stall:log/episodes 推進+process Running),其結束後 fps 回 3940;依守則未動別人 job。

**修復配方（已在程式碼 + wd_sa1_v3f_vaux.yaml 固化,RL 端零異常驗證成功）**:
reshape bug 修復（aux 特徵序列 reshape(L,B)→reshape(B,L).permute,真 bug,無條件生效）+ feat_norm（extractor 輸出 per-dim 正規化進 RNN,★關鍵缺件）+ rnn_type GRU + aux_loss huber + aux_target_pos_scale 0.33 + predict_dim 7(position-only) + 可訓 readout（aux_lr_predict_head/fc_middle/fc_front 0.0005）+ aux_epochs 8。
→ **結論定案:RNN hidden 確實能編碼障礙位置（WD 能用 RNN 成功,我們也能）。** 見 [[finding_velocity_aux_bottleneck]]、Obsidian rnn/「velocity-aux 完整破案」。

**下一步**: SA2 — `wd_sa2_v3f_vaux`,resume `--checkpoint logs/rnn_car/sa1_v3f_vaux_ne1024_s42/checkpoint_270000.pt`（已對齊實際存出步數 270000）。⚠️ train/play 須 `CHARGE_USE_ACT_HIST=0`;probe 須 `--feat_norm`。
