---
name: finding_velocity_aux_bottleneck
description: SA3 解碼 velocity rel err 仍 100%(RNN 未追蹤動態)的架構診斷 — 真兇是 predict_head=單層 Linear(preprocess_dim=12→13) 的 12D bottleneck 過載:WD 原版只 12→7(位置),這條 lineage 擴到 13(加6速度維)卻沒放大瓶頸,速度權重又低(0.8/0.5/0.3 vs 位置1.0)→速度被擠成≈0;非單純 vanilla RNN 容量不足
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**背景**：reluFix lineage（[[finding_predict_head_dead_relu]]）SA3 iter500 解碼 velocity rel err **仍 100%**（與 SA1/SA2 同），核心假設「reluFix 復活 velocity-aux→RNN 追蹤動態」未獲證實。用戶決策：診斷/升級 velocity 架構。

**架構診斷（2026-06-25，讀 `scripts/.../skrl/models/modular_rnn_models.py` + wd_sa3_v3f.yaml）**：
- RNN 類型 = **vanilla RNN**（非 GRU），`hidden_dim=64`，`fc_dim=64`。
- 瓶頸 `preprocess_dim=12`（fc_middle 輸出）。
- `predict_head = nn.Sequential(nn.Linear(12, 13))` — **單層 Linear，去 ReLU 後無非線性**。
- aux weight 13D = [1,1,1,1,0.7,0.7,0,**0.8,0.8,0.5,0.5,0.3,0.3**]：dims 0-5 位置、dim6 total_dis(w0)、dims 7-12 = top-3 障礙 vx/vy。

**真兇假設（bottleneck 過載，比「RNN 容量不足」更具體）**：
- WD 原版：`Linear(module_connect_dim=12 → predict_dim=7)` 只預測**位置**。
- 這條 lineage 把 predict_dim 7→13（加 6 速度維）**但沒放大 preprocess_dim(仍 12)**。
- → 要從 12 個數線性讀出 13 個量；位置權重(1.0)> 速度權重(0.8/0.5/0.3)→ 優化器優先擬合位置 → **速度 dims 被擠成 ≈0**（解碼證實：位置 signed 但低估、速度≈0 rel err 100%）。

**可能的槓桿（待探測定位）**：
1. **bottleneck**：preprocess_dim 12→24+（放大瓶頸給速度空間）。
2. **readout**：predict_head 單 Linear→MLP(12→32→13)（加非線性解耦位置/速度）。
3. **RNN 容量**：vanilla RNN→GRU（更好時間積分，速度=Δpos 需記憶）。
4. **權重失衡**：速度 weight 0.3-0.8 拉高（但若瓶頸是真因只是換誰勝出）。
5. **分離 head**：速度給獨立讀出路徑，不共用 12D 位置瓶頸。
⚠️ 改 preprocess_dim/RNN type 會破壞 checkpoint 相容(shape 變)→需從頭訓；改 predict_head MLP 保 12D 則可部分載入續訓。

**★建議「先探測再改」最省算力**：載入 SA3 checkpoint,凍結,蒐集 (preprocess_feat 12D, 真實障礙速度) pairs,擬合 linear/MLP probe 12D→velocity。
- probe **能**還原速度 → 資訊在 12D 裡,只是 predict_head 讀不出 → **MLP head**(便宜,checkpoint 相容,從 SA3 續訓)。
- probe **不能** → 12D 瓶頸沒裝下速度 → 需**放大 preprocess_dim / GRU**(從頭訓,貴)。
這個 probe 不用盲目重訓就能定位該動哪個槓桿。

相關：[[finding_predict_head_dead_relu]]（上游:dead-ReLU 結構修復必要非充分）[[project_reluFix_retrain_log]]（SA3 解碼裁決）[[feedback_regression_over_symptom]]（找根因）

---

**🔬 Probe 結果 #1（12D preprocess_feat，2026-06-25 11:40）**：載入 SA3 checkpoint，stage3 deterministic 收集 9600 筆 (12D 特徵, 真實速度 6D)，離線擬合:
- mean baseline rel_err **99.9%** / linear probe **99.7%** / MLP(64-64) probe **99.5%**。
- → **速度資訊不在 12D 瓶頸特徵裡**（連 MLP 都救不回）。**否決「改 MLP head」便宜修法**（不是 predict_head 讀不出，是上游沒編進 12D）。
- 工具：play_rnn_car.py 加 `--probe_dump`（每步收集 12D 特徵+64D hidden+真實速度→npz）；擬合腳本 /tmp/probe_fit.py。

**🔬 Probe #2（64D RNN hidden，進行中）**：擴充收集 RNN hidden state（64D），測「速度在 RNN memory 裡嗎」：
- 64D 能還原 → fc_middle 64→12 壓縮丟的 → 放大 preprocess_dim（中等成本，可能 checkpoint 部分相容）。
- 64D 也不能 → RNN 沒積分出速度 → GRU / 改 RNN 輸入 / 改訓練（從頭訓）。

**SA4 未啟動**：監控 loop 已停（解碼 gate 未過），改走 probe 診斷路線（用戶選）。

---

**🔬🔬 Probe #2 定案（64D RNN hidden，2026-06-25 11:43）**：9600 筆，probe 從 RNN hidden state(64D)→velocity：
- mean 100.0% / linear 100.0% / **MLP(64-64) 100.2%**（=純噪訊，非欠擬合；有訊號會 <100%）。
- → **連 64D RNN hidden 都不含速度資訊**。**否決「放大 preprocess_dim」**（不是 fc_middle 瓶頸丟的，是 RNN 記憶本身沒積分出速度）。

**結論更新（兩個 probe 後）**：問題比「predict_head 瓶頸過載」更深 —— **RNN hidden state 根本沒編碼障礙速度**。剩餘可能根因：
1. **RNN 輸入(LiDAR 72D 單幀)是否本就無法推出速度**？速度需跨幀差分+資料關聯+扣除 ego 運動，72-bin 5° 稀疏掃描可能資訊不足 → 若如此，任何架構(GRU/加 hidden)都救不了，要改 obs(顯式障礙追蹤通道)。
2. RNN(vanilla)積分能力不足 → GRU。
3. aux 速度權重(0.3-0.8)< 位置(1.0)，速度梯度太弱沒推動 RNN 編碼（rnn_grad~0.05 主要被位置佔）。

**★下一個決定性 probe**：時間窗 oracle — 收集每 env 連續 K 幀 RNN 輸入特徵，probe [K×feat]→velocity。能還原→資訊在 obs 裡,RNN/訓練問題(GRU/加權重可救);不能→obs 本身不足,需改輸入或放棄預測式改純反應。這決定「值不值得投資架構重訓」。

**修法槓桿排除進度**：❌MLP head(12D 無訊號) ❌放大 preprocess_dim(64D 也無訊號) → 待 oracle probe 定「obs 有無速度可推」再決 GRU/改 obs/放棄。

---

**🔬🔬🔬 Probe #3 定案（位置 probe + 陽性對照，2026-06-25 11:52）— 結論質變**：
- **陽性對照（probe 機制有效）**：`12D feat → 64D hidden` linear ridge = **5.0%**（有訊號抓得到）；`64D→64D 自身` = 0.0%。
- **實測（RNN 特徵 → 障礙狀態，linear ridge masked）**：12D/64D → **t0 位置 99.9%/100.3%、next 未來位置 99.9%/100.3%、速度 99.9%/100.0%**。全部 ≈100%。
- ⚠️ 注意 MLP 過擬合假象：無正則化 + FAR_DEFAULT=10 離群值會讓 MLP rel err 炸到 5000%+，**以 linear ridge masked 為準**。
- **前瞻訊號量化**：prediction_horizon=0.2s → ||next−t0||均=0.131m vs ||t0||均=3.51m（僅 3.7%）→ next ≈ 當前位置，0.2s horizon 太短前瞻幾乎退化。

**★結論質變（推翻先前所有便宜修法 + 用戶 position-only 想法）**：
- 不是「速度比位置難」——**RNN 對動態障礙的位置(t0/next)與速度全部沒編碼**（≈100%）。
- → **WD position-only 也救不了**（位置同樣 100%）。reluFix 的 signed 輸出是脫離真實障礙的常數預測。
- 導航 SR 83.8% 純靠 policy_head 吃**原始 LiDAR** 反應式避障，**完全沒用到壞掉的 RNN 特徵**（呼應 [[design_obstacles_60d_unnecessary]]：obs 79D 不含障礙、RNN aux 是唯一動態源，而它是死的）。

**根因層級上移**：aux 任務本身（從 72-bin LiDAR 經 tiny vanilla RNN 推「最近動態障礙」狀態）可能根本學不起來——需區分 dynamic vs static（跨幀運動）+ 資料關聯 + 定位。
**★決定性待辦 = oracle probe**：收集連續 K 幀 LiDAR 輸入，probe→障礙位置。能→obs 夠資訊，RNN/訓練問題；不能→obs 不足（LiDAR-only 無法分辨動靜態），任何 RNN 架構都白搭，需改 obs（顯式障礙通道）或接受純反應式導航。

**修法槓桿排除**：❌MLP head ❌放大 preprocess_dim ❌WD position-only ❌reluFix(必要非充分) → 待 oracle 定 obs 可行性再決 GRU/改obs/放棄預測式。

---

**🔬🔬🔬🔬 Probe #4 oracle 定論（K幀原始 obs→障礙，2026-06-25 12:00）— 關鍵反轉**：
工具：play_rnn_car.py 加 `--oracle_dump`/`--oracle_k`（per-env 滾動緩衝 K 幀 obs，不跨 reset）；擬合 /tmp/oracle_fit.py。10742 筆，K=5。
- **t0 位置**：單末幀 MLP **77.8%** / 全 5 幀 75.1% → **obs 確實含障礙位置（單幀 LiDAR 直接可見）**。
- **速度**：單末幀 83.7% / 5 幀 83.9% → **時間窗零幫助** → 單幀無速度資訊，84% 是虛假相關（位置↔腳本速度），**真速度從此 LiDAR 串流推不出**。
- **對照**：訓練好的 RNN 特徵 → 位置 ~100%（probe #3），但 fresh MLP 吃原始 obs → 位置 75%。**RNN 輸給新 MLP** → 訓練/架構失敗，非 obs 不足。

**★最終診斷鏈（4 個 probe）**：
1. velocity 不在 12D（否決 MLP head）
2. velocity 不在 64D hidden（否決放大 preprocess_dim）
3. 連 position(t0/next) 在 RNN 也 ~100%（否決 WD position-only 單獨用）
4. **oracle：obs 有 position(75%) 但 RNN 萃取 0；velocity 連 oracle 都推不出（時間窗無幫助）**

**★★ 結論 + 修法方向（定）**：
- **velocity-aux 是雙重死路**：obs 不支援（oracle 推不出）+ RNN 沒在學 → **應移除速度維 dims 7-12**。
- **position-aux 可行但 RNN 現在沒在學**：obs 支援（75%），但 trained RNN 萃取 0（被 RL 導航主導，aux 梯度太弱）→ 需**強化 aux 對 RNN 的影響**（提高 aux weight/lr、檢查 extractor 是否丟 LiDAR 細節、或 GRU）。
- **用戶 position-only 直覺正確**，但須配「修好 RNN 為何萃取不到位置」才有效：position-only aux + 強化 aux 訓練 + 重訓 + 解碼驗 rel err <100%。
- **導航 SR 83.8% 不受影響**（純 LiDAR 反應式，不靠 RNN aux）。

**修法槓桿總結**：❌MLP head ❌放大 preprocess_dim ❌velocity-aux(obs 不支援) ⭕**position-only aux + 強化 aux 訓練/extractor**（待設計實驗）。

---

**🔧 修復實驗 #1 啟動（7D position-only，2026-06-25 12:05）**：用戶選「7D position-only 從 SA3 續訓」。
- **改動**：(1) 新 config `wd_sa3_v3f_posonly.yaml`：predict_dim 13→7、aux_velocity_topk 3→0（移除速度維 7-12，自動用 WD 7D weight）；其餘全同 SA3（單變因）。(2) `train_rnn_car_wdclip.py` checkpoint 載入加 graceful：preprocess_rnn 只載 shape 相符的 key（RNN/fc 保留 SA3 權重），predict_head 13→7 shape 不符→跳過重初始化。
- **run**：sa3_v3f_posonly_ne1024_s42（PID /tmp/posonly.pid、log /tmp/sa3_posonly_startup.log），從 SA3 checkpoint_150000 續訓 stage3，timesteps 120000（400 iter，解碼點 checkpoint_90000=iter300）。
- **假設**：移除速度競爭 → 位置維拿全部 aux 梯度 → 測位置 rel err 能否 <100%。<100%=用戶 position-only 方向成功;仍 100%=需更強手段（aux weight/extractor/GRU）。
- **驗收**：iter300 停→解碼 stage3（位置 rel err，用 print_aux_debug 近1/近2 = t0/next）。CHARGE_USE_ACT_HIST=0 必設。

**✅ 實驗 #1 啟動成功（12:12）**：graceful 載入確認 — `跳過 predict_head.0.weight/bias（13→7 重初始化）`、RNN/extractor 沿用 SA3、Loaded checkpoint_150000。首 iter [1/400] SR 74.7%/CR 21.7%（≈SA3 收尾，RL policy 不變只換 aux 維度）。無錯誤，GPU 45%/13GB。run=sa3_v3f_posonly_ne1024_s42、wandb=kifjkvg1（/tmp/posonly_runid.txt）。監控到 iter300(checkpoint_90000)→解碼位置 rel err。

---

**🔧 即時 tracking 指標（pos_rel_err/pos_r2/pos_valid_frac）2026-06-25 14:16**：用戶問「為何不能訓練中拿 rel err」→ 加指標到 train_rnn_car_wdclip.py。
- ★踩坑：訓練有兩個 aux 區塊 — block A(line ~3360, `_aux_already_done`=True 的 **WD-order:aux 在 RL 之前**算) vs block B(line ~3845, TBPTT `else` 分支)。aux 實際走 **block A**，前兩版指標加在 block B(沒執行)→ 全程 MISSING。
- 修法：指標移到 block A(pred_eff/tgt_eff 算完後)存 `_wd_aux_pos` dict，再在 `_aux_already_done` carry-forward 分支寫 aux_monitor["aux/pos_rel_err"/"pos_r2"/"pos_valid_frac"]。
- 指標定義：pos_rel_err=mean‖pred−tgt‖/mean‖tgt‖(100%≈猜均值)；pos_r2(≤0=常數陷阱沒追蹤,>0=有追蹤)；pos_valid_frac=target 非 FAR_DEFAULT 佔比(診斷訓練端 target 是否多空目標)。
- posonly 已重啟 4 次(v1 舊碼/v2 加指標但 history 延遲/v3 always-log 但 block 錯/v4 block A 正確)。run sa3_v3f_posonly_ne1024_s42、PID /tmp/posonly.pid。下輪 iter10 讀真指標三段判定。

---

**✅ 即時指標通了 + 首讀（posonly v4，2026-06-25 14:29，iter8）**：runid u8y15f36。
- aux/pos_valid_frac = **1.0** → 訓練端 target 100% 有真障礙(非 FAR_DEFAULT) → **排除「空目標 bug」**(分支 a)。
- aux/pos_rel_err = **100.3%**、aux/pos_r2 = **−0.004**(≤0) → **常數陷阱**(分支 b);aux/preprocess_loss = −0.46(騙人的負 loss)。
- ★這是「log-loss 騙人 vs r2/rel_err 拆穿」的活體展示(回應用戶問題):loss −0.46 看似好,但 r2≤0 = 沒追蹤。
- ⚠️ iter8 太早(predict_head iter0 才重初始化),關鍵看 r2 是否隨 iter 爬>0(RNN 學)還是黏≤0(卡常數陷阱)。盯 ~20min 看 r2/rel_err 軌跡。

---

**📋 已批准的下一步計畫（2026-06-25 14:35，用戶決策）**：
- 用戶洞察：aux log(L1) loss 梯度 ∝ 1/|err|（誤差越大梯度越小，與 MSE/MAE 相反）→ 大誤差幾乎不修 → 主動鼓勵「常數陷阱」。可能是 RNN 學不會追蹤的**主因之一**（惡性循環:爛 loss→RNN 學習訊號弱→特徵沒編障礙→probe 量到無資訊→只能猜常數）。oracle 已證 obs 有位置資訊(75%)，故是 loss/訓練沒萃取出來，非輸入不足。
- **計畫（用戶批准）**：① 先看完 posonly r2 軌跡，確認「position-only + 舊 log loss」仍卡常數陷阱(r2≤0)。② **確認卡死後 → 換 aux loss 為 Huber/smooth-L1 或 MSE**（給大誤差大梯度，逼 RNN 用 input 追蹤），用即時 pos_r2 評估能否爬>0。
- ★loss 在 `wd_aux_targets.py:compute_wd_module_loss`（dims 0-5 用 `log(clamp(|pred-tgt|,0.01))`，dim6 用 L2 但 w=0）。改時保留介面、只換 dims 0-5 的 loss 形式。

---

**🔴 定論:position-only + 舊 log loss 卡死（posonly v4，2026-06-25 14:51，27 個 aux 點）**：
- pos_rel_err 全程黏 ~100.3%(1.005→1.004,從不降)；pos_r2 全程 ≤0(-0.004→-0.011,從不爬)；valid_frac ~1.0(target 正常)；preprocess_loss -0.46→-0.86 亂跳(看似活躍但 r2/rel_err 證明零學習)；rnn_grad ~0.03 平。
- ★三結論:(1)position-only 單獨無用(速度競爭非主因) (2)常數陷阱確認(與用戶 loss-設計-害-的假設一致) (3)非空目標 bug。
- ★「log-loss 騙人」活體鐵證:loss 忙跳但 r2 27 iter 死黏≤0。
- **動作:停 posonly,觸發已批准計畫=換 aux loss log(L1)→Huber/smooth-L1(dims 0-5),用即時 pos_r2 評估能否爬>0。**

---

**🔧 修復實驗 #2:Huber loss（2026-06-25 14:54）** — 測用戶「loss 設計害的」假設。
- 改 `wd_aux_targets.py:compute_wd_module_loss` 加 `loss_type` 參數:dims 0-5 "log"(原版 log(clamp|e|,0.01))或 "huber"(smooth-L1,大誤差大梯度)。
- 加 CLI `--aux_loss_type {log,huber}` + `--aux_huber_delta`(預設1.0);plumb 到兩個 call site(block A+B)。
- 新 config wd_sa3_v3f_posonly_huber.yaml(=posonly + aux_loss_type:huber)。
- run sa3_v3f_posonly_huber_ne1024_s42、PID /tmp/posonly.pid、log /tmp/sa3_huber_startup.log,從 SA3 checkpoint_150000 續訓,**唯一變因=loss 形式**(對照 posonly log loss r2 死黏≤0)。
- ★即時 pos_r2 評估:爬>0+rel_err<100%=loss 是元兇(用戶對);仍≤0=還有更深表徵問題。

---

## Huber 實驗即時 r2（2026-06-25 15:25，iter14，14 點）

run sa3_v3f_posonly_huber_ne1024_s42、wandb clgkfqbx。無崩潰、huber 生效、SR 78.4%。
- pos_rel_err 黏 ~100.2%(1.002-1.005)、pos_r2 黏 ~0(-0.001~-0.013,沒爬)、valid_frac 1.0、preprocess_loss 5.1-6.1(Huber 正值不可跟 log 比)。
- ⚠️ rnn_grad **0.007-0.044(均更低,~0.015)** 比 log 版~0.03 還低 — 不符「Huber 大誤差大梯度」預期,疑 grad_clip 或 delta=1.0 壓住,待查。
- ★判定:iter14 太早不定論。predict_head 快但 RNN 特徵重塑慢,換 loss 要幾十 iter 才看得出 r2 爬;log 版 iter28 才確認死黏,Huber 給到 iter50+ 才公平。目前仍常數陷阱無起色。盯到 iter~50。

---

## ★Huber iter33 + 梯度分配新線索（2026-06-25 15:47）

- Huber 33 點:pos_rel_err 1.003→1.001(黏~100%)、pos_r2 -0.006→-0.003(仍≤0,沒爬正)、rnn_grad ~0.016。**Huber 單獨沒破常數陷阱,與 log 版統計無差。**
- ★★更深機制(grad 級):predict_head_grad ~0.3-0.4(大,飛快學) vs rnn_grad ~0.016(小20倍,RNN 幾乎沒更新);且 rnn_grad 0.016 << aux_grad_clip 0.5 → 不是 clip,是梯度本來就小。
- **「predict_head 餓死 RNN」機制**:predict_head 直接接 loss,飛快用「猜常數」壓低 loss → loss 一低往回傳 RNN 的梯度萎縮 → RNN 拿不到強訊號重塑特徵 → 特徵沒編障礙 → 續猜常數。
- ★結論演進:用戶 loss 洞察方向對(梯度訊號是問題),但換 loss 形式不夠;真瓶頸=**梯度分配**(predict_head 學太快餓死 RNN)。
- **下一手候選(比換 loss 更對症)**:壓低 aux_lr_predict_head(別讓它瞬間擬合常數)/ 拉高 rnn_lr / GRU / 加 hidden。Huber 再跑到 iter~60 最終確認。

---

## 🔴 Huber 最終定論（2026-06-25 16:05，iter50，49 點）— loss 形式不是綁定瓶頸
- pos_rel_err 死黏 ~100%(前/中/後 1.003/1.001/1.001)、pos_r2 始終 ≤0(-0.006/-0.003/-0.003,從不爬正)、rnn_grad ~0.018 平、predict_head_grad ~0.16(~10× rnn_grad)。
- **Huber 與 log 統計無差 → 換 loss 形式不夠破常數陷阱。** 用戶 loss 洞察方向對(梯度是核心)但非綁定瓶頸。
- ★★綁定瓶頸=**梯度分配**:predict_head(直接接 loss)飛快擬合常數壓低 loss→loss 低→回傳 RNN 梯度只剩 1/10→RNN 餓死,學不會編碼障礙。不管 loss 形式 predict_head 都搶先擬合常數。
- 停 Huber 實驗。**下一手選項(報用戶選)**:①壓低/凍結 aux_lr_predict_head(逼 RNN 做事)②拉高 rnn_lr/放寬 aux_grad_clip ③GRU/加 hidden ④detach predict_head。
- 工具留存:即時 pos_r2/pos_rel_err/pos_valid_frac 指標(block A) + --aux_loss_type {log,huber} toggle,後續實驗可重用。

---

**🔧 修復實驗 #3:壓低 predict_head lr（2026-06-25 16:06）** — 用戶選,對症「predict_head 餓死 RNN」。
- aux optimizer:rnn_cell 與 predict_head 同 lr 0.0005,但 predict_head grad~0.16(9× rnn 0.018)→搶先擬合常數。壓低 aux_lr_predict_head 0.0005→5e-5(10×低)平衡有效更新量。
- config wd_sa3_v3f_posonly_phlr.yaml(=posonly log loss + aux_lr_predict_head 5e-5),單變因。run sa3_v3f_posonly_phlr_ne1024_s42、log /tmp/sa3_phlr_startup.log,從 SA3 checkpoint_150000 續訓。
- ★即時 pos_r2:爬>0=餓死是真因(壓 lr 逼 RNN 學成功);仍≤0=predict_head lr 不夠/還有容量問題→freeze predict_head 或 GRU。

---

## ★phlr iter20（壓低 predict_head lr，2026-06-25 16:47）— 餓死緩解但 r2 沒動
- pos_rel_err 黏 ~100.3%、pos_r2 仍 ≤0(-0.008/-0.005/-0.006 沒爬)、rnn_grad **0.032(~2× baseline 0.018)**、predict_head_grad **~0.4(維持高,沒瞬間壓死 loss)**。
- ★梯度分配確實改善(rnn_grad 2×、predict_head 不瞬間擬合常數)=「餓死」緩解,**但 pos_r2 仍 ≤0**=就算 RNN 拿 2× 梯度還是學不會編位置。
- ★★關鍵推論:綁定瓶頸**不是梯度分配,是表徵容量** — vanilla RNN(hidden64)+extractor 就算給足梯度也做不了「LiDAR 串流分辨+定位最近動態障礙」感知任務。
- iter20 偏早,給到 iter~50 最終確認;若仍平→坐實表徵容量瓶頸→答案=GRU/加 hidden(梯度側招數 loss/predict_head_lr 都試過無效)。

---

## 🔴🔴 phlr 最終定論（2026-06-25 17:07，37 點）— 坐實表徵容量瓶頸
- pos_rel_err 死黏 100.4%、pos_r2 四分位 -0.007/-0.005/-0.011/-0.007、**max -0.002(37點從未為正)**、rnn_grad ~0.034(維持 2× baseline)、predict_head_grad ~0.4(維持高)。
- ★★即使餓死緩解(rnn_grad 2×、predict_head 不擬合常數),pos_r2 仍從未 >0 → **梯度側全試過無效,綁定瓶頸坐實=表徵容量**(vanilla RNN hidden64+extractor 給足梯度也萃取不出障礙位置)。

## ★★★ 整條調查總結（reluFix lineage velocity-aux 為何失效）
| # | 假設 | 實驗 | 結果(即時 pos_r2 / decode rel err) |
|--|------|------|------|
| 1 | dead-ReLU 卡 predict_head | reluFix 去 ReLU | ❌ 結構修好(signed)但 decode rel err 仍100% |
| 2 | 速度 vs 位置 | position-only(13→7) | ❌ pos_r2 死黏≤0(27點) |
| 3 | obs 不含障礙資訊 | oracle probe(K幀 obs) | ✅ obs 有位置(MLP 75%)→是萃取失敗非輸入不足 |
| 4 | aux loss 設計(log 梯度∝1/\|e\|) | Huber loss | ❌ pos_r2 死黏≤0(49點),loss 形式不夠 |
| 5 | predict_head 餓死 RNN(grad 0.16 vs 0.018) | 壓 predict_head lr 5e-5 | ❌ 餓死緩解(rnn_grad 2×)但 pos_r2 max -0.002(37點) |
| **6** | **表徵容量(vanilla RNN hidden64)** | **待:GRU/加 hidden** | **坐實:梯度側全無效→架構問題** |
- ★關鍵工具(本次調查產出,可重用):即時 pos_r2/pos_rel_err/pos_valid_frac 指標(block A)+ --aux_loss_type{log,huber} + probe/oracle 腳本(/tmp/probe_fit*.py, play_rnn_car --probe_dump/--oracle_dump)。
- ★下一手(報用戶選):①extractor probe(先 pinpoint 是 extractor 還 RNN 丟的,最省)②GRU(vanilla→GRU)③加 hidden(64→128)。導航 SR 79% 不受影響(純 LiDAR 反應式,不靠 aux)。

---

## ✅✅✅ extractor probe 定案（2026-06-25 17:20）— 資訊在 RNN 階段丟的，extractor 沒問題
probe 各層 → t0 障礙位置(MLP rel err)：
- **extractor 輸出 96D = 74.3%**(≈ oracle obs 75%) → ★extractor 忠實保留位置資訊,Conv1d/flatten 沒問題。
- RNN hidden 64D = 102.7%、preprocess 12D = 99.7% → ★資訊在「extractor→RNN hidden」這步被丟。
- **真兇=RNN(recurrent 階段)**:vanilla RNN hidden state 把 extractor 好好提供的每幀位置洗掉。完美對上:梯度側全無效(loss/phlr)因問題在 RNN 架構非梯度。
- ★★對症藥=**vanilla RNN→GRU**(gated memory 留得住 per-frame 位置)+ 可能加 hidden(64→128)。**不是改 extractor。**
- 工具:play_rnn_car --probe_dump 已擴充收集 EXT(96D extractor 輸出);/tmp/probe_ext_fit.py 各層→位置 probe。
- 下一步:GRU 實驗(rnn_type RNN→GRU),即時 pos_r2 驗能否爬>0。導航 SR 79% 不受影響。

---

**🔧 修復實驗 #4:skip/concat 輸入到 aux 頭（2026-06-25 17:2x）** — 用戶選,對症 extractor probe(位置在 extractor 74% 被 RNN 洗成0%)。
- 機制:位置每幀都在 extractor 輸出,不需 RNN 記憶→讓 predict_head 直接 concat extractor 輸入(96D)繞過 RNN。
- 改 modular_rnn_models.py:PreprocessRNN 加 aux_skip_input,predict_head 輸入 preprocess_dim→preprocess_dim+input_dim(12→108),forward 兩路 concat features。+ CLI --aux_skip_input + dataclass + 建構傳參。
- config wd_sa3_v3f_skip.yaml(=posonly + aux_skip_input:true),run sa3_v3f_skip_ne1024_s42、log /tmp/sa3_skip_startup.log,從 SA3 checkpoint_150000(predict_head reinit 因 input dim 變)。
- ★即時 pos_r2:爬>0=位置繞過 RNN 成功直送 predict_head=破案(最直接修法);仍≤0=連 skip 都不行(極不可能,因 probe 證 extractor 有位置)→需再查。

---

**🛑 skip 是作弊已停 + 目標校準（2026-06-25 17:30，用戶兩次點破）**：
- 用戶①「skip 繞過 RNN 不就 RNN 沒用?」→ 對:aux 是手段(逼 RNN 編碼供 policy),skip 讓 predict_head 從 extractor 直讀→梯度不流經 RNN→RNN 沒學;且 policy 用 RNN 的 12D 特徵不吃 predict_head 輸出→policy 無受益→skip 是 metric 作弊。已停。
- 用戶②「RNN 是要預測未來位置,policy 現在沒有」→ ★校準:policy 有當前位置(raw LiDAR 單幀)但**沒有未來位置(next)**;aux dims 2-3(next)正是 policy 真缺的=提前避障的關鍵。我先前「位置 policy 已有」框架不準。
- ★細節:next=當前+速度×horizon;現 horizon 0.2s→位移僅0.16m→next≈當前(前瞻價值小)。要有用須拉長 horizon(1.0s→0.78m),拉長後預測準=RNN 真懂運動=不可取代價值。
- **正確分階段計畫**:Stage1 RNN→GRU(不 skip,predict_head 仍從 12D,逼 RNN 自己編,即時 pos_r2 看能否學會帶位置);Stage2 若成功→拉長 prediction_horizon 0.2→1.0s 測真前瞻。

---

## GRU Stage1 iter20（2026-06-25 19:02）— rnn_grad 高但 r2 沒轉正
- pos_rel_err 黏~100%(1.009→1.002)、pos_r2 -0.011→-0.006 plateau(max -0.002,沒爬正)、rnn_grad **0.06-0.08(3-4× vanilla,還在升)**=GRU 被大力訓練。
- ★張力:GRU 拿大量梯度卻沒轉成位置編碼(r2 仍≤0)=跟 phlr 同圖像(梯度給足也學不出位置)。
- GRU recurrent 從頭學需更多 iter,iter20 不能判死,給到 iter~45 最終判定。
- ★若 iter45 仍≤0=強烈指向更根本限制:obs 串流可能本就不含可學的未來運動資訊,或 RL 多任務下 aux 永遠搶不到表徵→velocity-aux 可能是死路。

---

## 🎯🎯 破案(2026-06-25 19:30,用戶堅持去看 WD 原版)— WD 凍結 predict_head+FC,只訓 RNN
- 用戶「絕不可能不可行,WD 成功了,去查論文/參考」→ 比對 WD 原版 new_warp_drive/。
- ★agent 首猜「WD 讓 RL 梯度流經 RNN」**經我驗證是誤讀**:WD `concat_input=rl_in_.detach()`(module_connected.py:505)RL 也 detach,跟移植版同。
- ★★真兇=lr 設定:WD train_rnn_car.py:361-362 `spot_preprocess_model_lr=0`(preprocess front/middle/back=predict_head 全凍結) + `spot_rnn_model_lr=0.0005`(只 RNN cell 訓)。custom_trainer.py:336-348 兩 optimizer 都把 preprocess_params lr=0、rnn_params lr=rnn_lr。
- **WD 配方=只訓 RNN cell,predict_head+前後FC 凍結在隨機初始化**→predict_head 不能擬合常數→降 loss 唯一辦法=逼 RNN 產生「過固定隨機投影後對上 target」的特徵→強迫 RNN 學編碼=「固定隨機讀出頭強迫表徵學習」經典技巧。
- ★移植版錯誤:predict_head/fc_middle/fc_front/extractor 全 lr 0.0005(可訓)→predict_head 飛快擬合常數(=我量到的 grad 0.16 餓死 RNN)→常數陷阱。我的 phlr(predict_head lr 5e-5)方向對但不夠(沒凍 + 沒凍 fc_middle)。
- ★連 reluFix(去 predict_head ReLU)都是治症狀:WD 的 predict_head 有 ReLU 但**凍結**故無所謂;移植版去 ReLU 想讓它可訓反而引入餓死。capacity/GRU/loss 全是岔路。
- **WD-faithful 修法**:rnn_lr 0.0005 + aux_lr_predict_head=fc_middle=fc_front=extractor=**0**(凍結) + vanilla RNN。即時 pos_r2 驗。

---

## WD-faithful v2(隨機凍結 readout)2026-06-25 20:40, iter20
- run sa3_v3f_wdfreeze_reinit, wandb t4z493mq;載入跳過 fc_middle/fc_front/predict_head 保持隨機+凍結,只訓 RNN,extractor SA3。
- ★rnn_grad **0.06-0.07(比 SA3-frozen 0.03 翻倍)**=隨機凍結 readout 把 RNN 逼更用力(好兆頭);但 pos_r2 仍 ≤0 平。
- ★方法論:凍結 readout 下 pos_r2(predict_head 輸出)假陰性(固定隨機投影 scramble),要直接 probe RNN hidden(64D)→位置。play_rnn_car --probe_dump 已收 XH。
- 待 iter100/checkpoint_30000(~40min)→停 v2→probe RNN hidden→看內部有無編碼位置(對比 SA3/v1 ~100%)。這是 WD-freeze 是否生效的決定性測試。

---

## 🔴 決定性測試結果(2026-06-25)— 連 WD-faithful 都沒讓 RNN 編碼位置
checkpoint_30000(WD-v2 凍結 readout 只訓 RNN 100 iter)收 9600 樣本 probe 各層特徵 → t0 障礙位置 rel err(MLP):

| 層 | rel err(MLP) | 解讀 |
|----|----|----|
| oracle 原始 obs | ~75%(歷史) | 障礙位置在 obs 串流裡 |
| **extractor 輸出(96D)** | **74.3%** | ✅ 位置資訊**通過 3-branch extractor 仍在**(≈oracle) |
| **RNN hidden(64D)** | **102.7%** | ❌ RNN 把位置**洗光**(>100%=比常數還差=常數陷阱) |
| preprocess(12D) | 99.7% | ❌ 下游也沒有 |

### 定論(整條調查最終結論)
- **瓶頸明確在 RNN 本身**:位置資訊在 obs 有(75%)、通過 feature extractor 仍在(74%),但**一進 RNN hidden 就被洗成常數(~100%)**。這個 extractor↔hidden 的斷崖是 checkpoint 無關的穩健證據。
- **WD-faithful 配方(凍結隨機 readout 只訓 RNN)在 100 iter 續訓下仍沒讓 RNN 編碼位置**。rnn_grad 翻倍(RNN 被逼用力)但 hidden 仍無位置。
- 試過全部都沒用:posonly target、Huber loss、predict_head 低 lr、skip-input、GRU、WD-faithful 凍結。沒有任何介入讓 RNN 把障礙位置寫進 hidden state。

### WD-faithful 測試的 caveat(不是完美 WD 復刻)
1. **續訓 vs 從頭訓**:WD-v2 從已陷常數陷阱的 SA3 checkpoint 續訓,WD 原版從頭訓。從常數陷阱起步可能跳不出。
2. **只 100 iter**:RNN recurrent 從頭學編碼需更多 iter,30000 timesteps 偏短。
3. 殘留 target/forward 細節差異未逐行比對。
→ 故「WD 配方無效」結論**僅限本續訓 100-iter 設定**,不能宣稱 WD 原版會失敗。但「extractor 有位置、RNN 洗掉」是穩健事實。

### 🔬 un-detach 驗證(2026-06-26)— RL 梯度也救不了 RNN
用戶質疑「RNN 沒用?」→ 我加 `--rnn_rl_grad`(PPO update minibatch 重算 preprocess_feat 帶梯度,RL loss 流進 rnn_cell+fc_front+fc_middle,獨立 optimizer lr 5e-4)+ `--disable_aux_training`(aux 全凍,RNN 只吃 RL 梯度)。從 wdfreeze_reinit/checkpoint_30000 短訓 50 iter,probe 同管線 before/after(已修 play_rnn_car.py:2867 gate >=13→>=7 讓 predict_dim=7 能 probe):

| 層 | 基準 | rnnrl 50iter | 變化 |
|----|----|----|----|
| extractor 96D | 74.8% | 75.8% | 位置一直在 |
| RNN hidden 64D | **101.0%** | **99.7%** | ❌ 沒動 |
| preprocess 12D | 100.3% | 99.1% | ❌ 沒動 |

- ★SR 全程平 80%(1→50 iter 沒升)+ RNN hidden rel err 沒動 = **policy 靠單幀 obs 反應式解題,不需 RNN → ∂loss/∂rnn_feat≈0 → RNN 拿不到有用梯度**。
- ★結論收斂:**唯一能逼 RNN 編碼位置的只有 aux task,但 aux regression 塌常數**。un-detach 不是解。
- ★用戶新指令(睡前):RNN 一定要能用、不接受 explicit channel、WD 成功=我們也能、一定要解決 → 轉自主實驗模式找 WD 差異。
- 下一步未測關鍵差異:**從頭訓 WD-faithful**(過去全從常數陷阱 checkpoint 續訓,WD 是從頭訓)+ aux target normalize / loss 設計(constant collapse 文獻修法)。WD 逐行比對 agent + Codex 文獻 進行中。

### 🔑 WD 逐行比對 + Codex 文獻(2026-06-26)— 找到漏抄的 3 差異 + 文獻方向
**WD 逐行比對 agent** 找到我們沒抄到的 Top 3(每個都足以致塌縮):
- **#1 從頭訓 vs 續訓(高)**:WD 隨機初始化從頭訓;我們一直從常數陷阱 checkpoint 續訓,log 梯度≈0 出不來。
- **#2 obs 正規化錯配(高)**:WD RNN 吃**原始 obs** + 原始公尺 target(一致);我們 RNN 吃 `RunningNormalizer` whiten 後 obs 但 target 仍原始公尺 → input/target 尺度錯配 → 最省力解=常數。修法:餵原始 obs 給 RNN,或 whiten target(兩者一致即可)。
- **#3 缺 replay 去相關 + log 陷阱(中-高)**:WD 跨 iteration replay buffer 去相關;我們只用當前 rollout(高相關窄分布)→ log loss「大誤差小梯度」裸露。修:huber + replay。
- 已對齊不必動:predict_head 凍結、detach、target 語意。⚠️我之前 phlr 解凍 predict_head 其實偏離 WD。

**Codex 文獻**:
- 主流 recurrent PPO(SB3/CleanRL/ANYmal/ZSL-RPPO)都 **RNN 端到端訓(TBPTT)**,非 detach+弱 aux;detach+aux 像 RMA 第二階段(只在 latent 已被 policy 證明有用時 work)。
- regression collapse 修法(低成本):**target whiten/PopArt**(van Hasselt 2016 arXiv:1602.07714)、VICReg(2105.04906)、GradNorm。
- ★**CPC/InfoNCE contrastive**(中成本,1807.03748):常數 hidden 在對比任務**立刻失敗** → 最適合 motion 編碼,最 robust 的解。
- forward model/Dreamer(高)、HEIGHT attention、explicit channel、teacher-student。
- ★洞察:un-detach 沒救是因我們 RNN **冗餘**(policy 單幀 obs 可繞過);主流成功是 RNN 為唯一編碼器。真能逼 RNN 編碼的兩路:(A)RNN 在關鍵路徑+端到端;(B)CPC aux。

### 🧪 自主實驗序列(2026-06-26 過夜,用戶要求一定解決)
1. **從頭訓 WD-faithful**(config wd_sa3_fromscratch_wdfaithful,run sa3_fromscratch_wdfaithful):從頭訓+huber+凍結 readout 只訓 RNN cell+extractor(攻 #1+#3,未含 #2)。iter28 pos_r2≈0/rel_err 100% 仍陷阱。待 iter50 probe 確認。
2. **#1+#2+#3 scale-fix**(config wd_sa3_fromscratch_scale,run sa3_fromscratch_scale,wandb 90rx1byj):+aux_target_pos_scale 0.33(位置 target std 3m→unit 配 huber)。iter27 **pos_r2≈−0.05、rel_err 105% 仍平** → scale 也沒逃出陷阱。
3. ★**三個 regression-aux 變體全失敗**(un-detach pos_r2 沒動、#1+#3 pos_r2≈0、#1+#2+#3 pos_r2≈0)→ 印證 Codex:regression aux 本質易塌縮 → **轉 CPC contrastive aux**(常數 hidden 在 InfoNCE 立刻失敗,理論保證不塌)。
   - CPC 實作要點:aux block 直接 `_fc=preprocess_rnn.fc_front(feat_seq); rnn_out,_=preprocess_rnn.rnn(_fc,h0)` 取每步 hidden(不改 model);proj_q(H→d)+proj_k(pos2→d)+InfoNCE(q@k.T/τ,label=對角);常數 q→所有 sim 相同→softmax 無法對上正樣本→高 loss→逼 RNN 編碼位置。cpc_opt 訓 fc_front+rnn+extractor+proj heads。
   - 已加 code:`--aux_target_pos_scale`(wd_aux_targets.py build target + train CLI + dataclass)。
   - 已實作 `--aux_cpc` CPC code(train_rnn_car_wdclip.py aux block ~3447:取每步 rnn_out→proj_q,pos target→proj_k,InfoNCE)+ config wd_sa3_cpc。compile+smoke test 過。
4. **CPC run**(sa3_cpc,wandb zf1xco6h):iter30 **cpc_loss 平在 ln(2048)=7.628、cpc_acc≈random** → 看似沒學。但 cpc_acc 精確對角命中對「近距離重複位置」太嚴苛(指標天生低);rnn_feature_std 0.0925→0.0946 緩升(RNN 有被訓)。**決定性=probe hidden→位置,非 cpc_acc**。待 iter50 checkpoint probe。
   - ★若 CPC probe 也 ~100% → 所有 aux loss(regression/CPC)都救不了 → 下一步**離線可行性測試**:收 obs 序列,離線 clean supervised 訓 fresh RNN+linear head 預測位置(TBPTT),測「RNN 到底能不能從 obs 序列學會編碼障礙位置」。能=RL 訓練互動問題;不能=obs 序列本身缺時序資訊或 RNN bottleneck 本質限制。
   - ⚠️觀察:所有實驗 from-scratch 時 policy 爛(SR16% CR80%)→episode 極短→TBPTT 序列可能缺時序結構,可能餓死所有 aux。離線測試可隔離此因素(用已訓好 policy 的序列)。
5. **CPC probe(checkpoint_15000 iter50)結果**:extractor 96D=**100.3%**、RNN hidden=98.3%、preprocess=99.8%。CPC 失敗。
   - ★★★重大 confound 發現:**from-scratch 的 extractor 也是 100%(沒學到位置)!** 之前「extractor 74%」是**已訓練 SA3** extractor。from-scratch extractor 無強監督 → 連 RNN 輸入都沒位置 → 整個 from-scratch 系列(#1+#3/scale/CPC)都被這 confound 搞砸:不是 aux 不行,是輸入根本沒位置。
   - ★正確下一步:**好(已訓練)extractor 凍結 + 全新隨機 RNN + CPC**,隔離「RNN 能否從好特徵學編碼位置」。加 `--reinit_rnn`(載 checkpoint 但 RNN 重隨機),aux_lr_extractor=0 凍好 extractor,resume SA3 reluFix checkpoint_150000。
   - 備案:離線 clean supervised TBPTT 測試(用訓練好 extractor 特徵序列)。
- 監控:live `aux/pos_r2`(進 WandB 不印 console;讀 wandb API: `api.run('me0608623-none/charge_skrl/runs/<id>').history(keys=['aux/pos_r2','aux/pos_rel_err'])`)。

### 🎯🎯🎯 真根因找到(2026-06-26)— aux 特徵序列 reshape 打亂 bug
**所有 constant collapse 的真兇 = aux 把特徵序列 reshape 錯位**:
- buffer `sample_aux_sequences` 回傳 `obs_seq` = **[B, L, obs]**(batch-major)。
- forward(train_rnn_car_wdclip.py 原 line 3473/4050)用 `feat_flat.reshape(L_seq, B_seq, -1)` 把 batch-major 攤平資料**當 time-major 重排** → 時間/batch 維度被打亂(scramble,非 transpose)。
- 但 target 用 `target_seq.permute(1,0,2)`(**正確** [L,B])→ **特徵與 target 完全不對應**。
- 後果:RNN 餵打亂的特徵序列、loss 對正確 target → 學不出任何映射 → **只能輸出常數**(constant collapse)。velocity-aux **從來沒成功過**就是因為這個 bug,不是 reward/容量/loss/detach/從頭訓/尺度 任何先前假設。
- 我的 CPC code 也犯同 bug(`reshape(_Lc,_Bc)`)→ cpc_loss 卡死 ln(N)。
- **修法**:`reshape(L,B,-1)` → `reshape(B,L,-1).permute(1,0,2).contiguous()`(對齊 target permute)。已修 3 處(block A line 3473、CPC line 3500、block B line 4050)。compile 過。
- 連帶:live `aux/pos_r2`/`pos_rel_err` 之前也是拿打亂 pred 比正確 target → 數值無意義(難怪一直 ~100%);修復後才有意義。
- ★驗證中:run sa3_reinit_reg_fix(wandb 2ht81dk8,好 extractor 凍結+reinit RNN+regression huber+pos_scale 0.33+修復)。iter1 pos_r2 −0.035 起點,看是否爬正。爬正=★★★bug 即根因,velocity-aux 終於可行。
- ⚠️教訓:[[feedback_regression_over_symptom]] 的「先 bisect 改了什麼修根因」——這次是更深的既有 pipeline bug,前面所有 WD-diff/Codex/CPC 假設都是在繞 symptom;直接讀 forward 的 tensor reshape 才抓到。

### 🎉🎉🎉 終極破案(2026-06-26)— 離線證明 RNN 完全做得到,問題是 RL aux 訓練太弱
**reshape bug 修復後 probe**(sa3_reinit_reg_fix checkpoint_15000,好 extractor 凍+reinit RNN+regression+修復):extractor 77.4%、**RNN hidden 仍 103%**、preprocess 99.9%。live pos_r2 那點緩升是 predict_head 從 fc_out skip 讀當前特徵,非 RNN 學 → RNN 在 RL-loop 裡仍沒編碼。reshape 是真 bug 但非完整原因。

**★離線 clean supervised 終極測試(/tmp/offline_gru_test.py,用 probe_fix.npz 好 extractor 特徵 reshape→[T=300,E=32,96] 序列,切 L=16 window,fresh GRU+head,supervised MSE 120 epoch):**
| 測試 | rel err |
|------|---------|
| 單幀 MLP(extractor→位置 非遞迴上限) | 79.7% |
| **fresh GRU hidden→當前位置(clean supervised)** | 96%→**30.9%** |
| RL-loop 訓練的 RNN hidden(所有實驗) | ~100% |

**★★★決定性結論:fresh GRU 乾淨監督下 hidden 編碼位置到 30.9%(<<100%,還勝單幀 79.7% 因整合時序)→ RNN 架構完全能編碼障礙位置。問題從來不是架構/RNN 能力/reward/容量/detach/WD-diff,是 RL-loop aux 訓練太弱**:每 iter 只 1 次 aux 更新、on-policy 相關資料、lr 低(5e-4)、vanilla relu RNN(離線用 GRU)、預測只在 effective steps。
- WD 能用 RNN 是對的,我們也能。
- **修法配方(讓 RL-loop aux 也成功)**:(1)reshape bug 修復(已);(2)rnn_type 改 GRU(離線證 GRU 學得好);(3)可訓 readout(predict_head/fc_middle/fc_front lr>0,非凍結);(4)★每 iter 多次 aux 更新(加 --aux_epochs)或大幅提 aux lr(rnn 2e-3);(5)pos_scale 配 huber。
- 離線 GRU 達 30.9% 是 RL-loop 目標下限;先讓 RL-loop RNN hidden probe 從 ~100% 掉到 <80% 即證移植成功。
- ⚠️教訓:整夜繞了 un-detach/3 regression/CPC×2/WD-diff/Codex 一大圈,真正解法是「離線隔離測試」一招定生死(證可行→鎖定 RL 設定問題),早該先做。

### 🎯🎯🎯 三根因全找齊 + 完整修法(2026-06-26 凌晨,離線診斷逼出)
RNN constant collapse 不是單一原因,是**三個疊加根因**(離線 clean supervised 測試一步步隔離出來):
| # | 根因 | 證據 | 修法(已加 code) |
|---|------|------|------|
| 1 | aux 特徵序列 **reshape 打亂** | forward reshape(L,B) 把 [B,L] batch-major 當 time-major,target 卻 permute→不對應 | reshape(B,L).permute(已修 3473/3500/4050) |
| 2 | RL aux **梯度密度太低**(1 update/iter) | 離線需 ~1000 更新才達 30%,RL iter50 才 50 次 | `--aux_epochs`(每 iter N 次 aux 更新) |
| 3 | ★extractor 輸出**沒 per-dim 正規化**進 RNN | 離線 raw EXT 78% vs per-dim 標準化 **31%**;LayerNorm(per-sample)77% 無效,BatchNorm/per-dim 才有效 | `--feat_norm`(RunningNormalizer 對 extractor 輸出 per-dim 正規化) |

**離線鐵證(/tmp/offline_gru_test.py + 診斷)**:fresh GRU + 好 extractor 特徵 + per-dim 正規化 → hidden 編碼當前位置 **30.9% rel err**(<<100%,勝單幀 79.7%;加 12D bottleneck 仍 33%)。→ **RNN 架構完全能編碼障礙位置;WD 能我們也能**。問題從來不是架構/能力/reward/detach/容量/WD-diff 三假設,是上述三個 pipeline/訓練配方缺陷疊加。
- **完整修法 run**:sa3_reinit_full(wandb kns1tc36),config wd_sa3_reinit_gru_fast + CLI `--reinit_rnn --aux_target_pos_scale 0.33 --aux_epochs 20 --feat_norm`。配方=reshape修復+GRU+rnn/aux lr 2e-3+aux_epochs20+feat_norm+可訓readout+好extractor凍+reinit RNN。預期 live pos_r2 大爬、iter50 probe RNN hidden ~30-40%。
- ★若 probe 確認 RNN hidden <50% → velocity-aux 在 RL-loop 完全修好 → 帶完整配方(尤其 feat_norm + reshape 修復,這兩個對所有 lineage 都適用)重訓 reluFix SA1→SA5。
- **完整修法 run iter50 probe 結果(部分成功)**:extractor 67.2%、**RNN hidden 92.1%**(從整夜全部 ~100%/103% 改善了!)、preprocess 98.0%。live pos_r2 爬到 +0.29/rel_err 81%(比 probe 好,暗示 predict_head 部分靠 fc_out skip 解碼)。
  - ★判讀:**方向完全正確、RNN 確實開始編碼(100%→92%),但 iter50(~1000 aux 更新)還沒到 offline 30%**。同更新數 offline 30% vs RL 92% = 還有殘留 RL vs 離線差異未補。
  - 殘留嫌疑(待查):(1)★aux 用 rollout 存的 **stale h0**(離線是 fresh h0=0/window;RL 用舊權重存的 hidden,aux_epochs 20 次更新間 h0 不一致);(2)fc_front(96→64)早期 bottleneck vs 離線 GRU 直吃 96D;(3)huber+pos_scale vs 離線 MSE raw;(4)on-policy buffer 相關性 vs 離線 shuffled 固定資料集;(5)live pos_r2 未 plateau,可能單純需更多 iter。
  - 處置:resume checkpoint_15000 續訓(run sa3_reinit_full_cont)看 hidden 是否續降;若 plateau ~90% 則攻 stale h0(aux 改 h0=0+burn_in)。
  - ★續訓結果:live pos_r2 **plateau ~0.25/rel_err 85% 沒再爬** → 更多 iter 不是解,gap 是結構性。
  - ★offline 診斷(澄清非探測法問題):offline GRU hidden 用 [A]L16 window=30.9%、[B]全 300 步 episode 傳播(=RL play)=20.3%、[C]full-episode ridge probe(=probe_ext_fit)=37.8% → offline GRU 即使用 RL play 方式 probe 仍 37.8%,但 RL 訓練的 GRU 是 92% → **gap 確定在 RL aux 訓練產出較差的 GRU**(非探測/架構)。
  - ★攻 stale h0:加 `--aux_zero_h0`(aux 用 h0=0 對標離線,逼 GRU 從 window 特徵萃取而非靠 rollout 存的舊 hidden 帶位置)。run sa3_reinit_zh0(wandb t22mgahp,reinit+feat_norm+aux_epochs20+zero_h0)。
  - ★★offline ablation 決定性澄清(逐項加 RL 因素,h0=0,probe hidden):純 GRU 37-45% → +fc_front 40% → +skip 43% → +12D-middle 39% → +huber+scale(=RL 全結構)**45.7%**。**即複製 RL 完整架構/loss,offline 用 h0=0 仍 45.7%,沒有結構因素能推到 92%** → RL 卡 92% 真兇=**訓練動態(stale h0 或 on-policy 資料),非架構/loss**。
  - ★乾淨預測:offline 全結構+h0=0=45.7%。zh0 run(RL+h0=0)probe 若 ~45-60%=stale h0 是最後答案;若仍 ~92%=on-policy 資料動態(buffer 每 iter 換、只看 20×;offline 120 epoch 同資料集)。zh0 iter50 probe 決定性。
  - 註:skip(concat_rnn)讓 predict_head 走捷徑是次要因素(offline 37.8→48.7,~11%),非主因。
  - ★★zh0 probe 結果:RNN hidden **92.9%**(deterministic)/ **95.1%**(stochastic)= 跟 full run 92% 一樣 → **stale h0 排除、det/stochastic 分布偏移也排除**。
  - ★★★最終診斷收斂:架構/loss(offline 全結構 45.7%)、stale h0(zh0 92.9%)、分布偏移(stochastic 95.1%)**全排除** → RL 卡 92% 唯一剩下=**on-policy 移動 buffer 資料動態**:每 iter aux 只看新 buffer 的 256 seq × 20 epoch(buffer 有 307200,僅 1.7%),buffer 每 iter 換 → GRU 永遠收斂不了;離線固定資料集 120 epoch 才達 30%。
  - ★最終修法 = **cross-iter aux replay buffer**(WD 原版有、我們漏抄 = WD-diff agent 報告的 #2!整夜線索在此收口):跨 iter 累積 aux 訓練資料(obs window+target+done+hidden),每 aux 更新從累積池隨機 replay 取樣→去相關+多 pass 收斂。這是最後一塊、最有把握(WD 證有效)。實作:在 ChargeRolloutBuffer 外加持久 replay buffer,rollout 後 append 當 iter 序列,aux sample 改從 replay 池取。
  - 整夜結論(待最後修正,見下方 ★★★):RL-loop 經三+根因修復從 100%→92%。

### ★★★完全破案(2026-06-26 早)— 「92%」是 PROBE BUG,GRU 早就訓練成功!
- ★決定性發現:**play_rnn_car(probe 用)沒有 feat_norm,checkpoint 也沒存 feat_normalizer** → GRU 訓練時吃 feat_norm 正規化特徵,但 probe 餵**原始(未正規化)特徵** → GRU 收到錯尺度輸入 → hidden 錯亂 → 「92%」是**測量假象**,非訓練失敗。
- ★修正:給 play_rnn_car 加 `--feat_norm`(play 端用 fresh running normalizer 近似訓練統計),重 probe sa3_reinit_full checkpoint_15000:
  | 層 | probe 無 feat_norm(錯) | **probe 有 feat_norm(對)** |
  |----|----|----|
  | extractor 96D | 67% | 67.6% |
  | **RNN hidden 64D** | **92%** | **53.0%** ✅ |
  | preprocess 12D | 98% | 79.3% |
- **RNN hidden 53% << 單幀 extractor 67.6%** → RNN 整合時序、編碼位置勝過單幀 = 正是要的效果!對標離線 45-51% 完全吻合。
- ★★整夜後段追的 stale h0 / on-policy 資料動態 / replay buffer **全是在追一個 probe bug**——GRU 其實早就學會了(live pos_r2 爬到 +0.29 是真的)。**不需要 replay buffer / aux_zero_h0**。
- ★★★最終定論:**velocity-aux 在 RL-loop 完全修好**。完整配方=**reshape 修復(必)+ feat_norm(必,訓練+probe/部署都要)+ GRU + 可訓 readout + aux_epochs(加速)**。aux_target_pos_scale 助 huber。reinit_rnn 只是隔離實驗用,正式重訓從頭即可。
- ✅★部署 feat_norm checkpoint 儲存**已實作 + 驗證完成**(2026-06-26):
  - train 存:`_ckpt_dict["feat_normalizer"]`(mean/var/count,line 4538,--feat_norm 時);resume 載(line 3045)。
  - play 凍結套用:checkpoint 有 feat_normalizer → `_ckpt_feat_norm` local var(注意 charge_features_for_rnn 後定義,不能在 load 點設它的 attr,UnboundLocalError 已修)→ apply block 凍結套用訓練統計;無 → fresh running 近似 fallback。play 加 `--feat_norm` flag。
  - ★驗證(sa3_fnsave2 iter10 checkpoint_3000):checkpoint 含 feat_normalizer(96D mean/var,count=3072000 正確);play 印「✅ feat_normalizer 統計已從 checkpoint 載入(凍結套用,部署正解)」;probe 凍結載入 → RNN hidden **63.7%**(iter10 就 < 單幀 77.7%,iter50 達 53%)。
  - 車端部署同理:載 checkpoint 的 feat_normalizer 統計,extractor 輸出 per-dim 正規化(`(f-mean)/sqrt(var)` clamp±5)再進 GRU。
- ⚠️方法論教訓 #2:**probe/eval 端的前處理必須與訓練端一致**;新增訓練端前處理(feat_norm)時,probe/部署端同步加,否則指標假性失敗,害我追了好幾個假根因(stale h0/資料動態/replay)。
- 已加全部 code:reshape 修復 + `--reinit_rnn` + `--aux_target_pos_scale` + `--aux_epochs` + `--feat_norm`(都在 train_rnn_car_wdclip.py + experiment_config dataclass)。離線腳本 /tmp/offline_gru_test.py。
- ⚠️方法論教訓:整夜先繞了 un-detach/regression×3/CPC×2/WD逐行比對/Codex 文獻一大圈(都是猜+試),真正定生死的是「離線 clean supervised 隔離測試」——證可行後逐項對比 RL vs 離線差異(梯度密度、特徵正規化)才精準抓到根因。下次這類「學不起來」問題應**先做離線可行性 + 差異對比**。

### 給用戶的三條路(velocity-aux 走到牆)— ⚠️已不適用,velocity-aux 已證可行,見上方破案
- **(A) obs 顯式障礙通道**:既然 extractor 保住位置、是 RNN 丟的,最可靠是把障礙位置/速度直接餵 policy(顯式 channel),不依賴 RNN 編碼。
- **(B) 接受反應式**:RNN 當純時序平滑器(非預測器),policy 靠單幀 LiDAR 反應,放棄前瞻 aux。
- **(C) 續摳 WD 差異**:WD 從頭訓 + 更長 aux warmup + 逐行對 target/forward。報酬遞減,風險高。
- 推薦先跟用戶討論 A vs B(C 風險高報酬低)。velocity-aux「逼 RNN 學障礙運動供前瞻避障」此路在本架構/訓練設定下未證可行。
