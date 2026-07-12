# Dense-Mixed 部署導航：密度 ramp 重訓 + 部署 shield 補尾（方向 A）

- **日期**：2026-07-12
- **狀態**：設計定稿（待實作計畫）
- **作者**：me0608623 + Claude
- **背景記憶**：[[project_lvdot_encoder_experiment]]、[[project_deploy_shield_ladder]]、[[v27_dense_results]]、[[project_vdec2_canonical_curriculum]]、[[finding_bangbang_degenerate_penalty0]]

---

## 1. 問題陳述

**部署目標場景**：12×12m arena，**15 顆靜態障礙 + ≥4 顆動態障礙（人）**，機器人需近對角穿越（goal 距離 ~7–9m）。這是本專案最難的 dense-mixed 場景。

**現況**：現行部署候選 policy（`rmd`/`sa3_rmd_enc24`，curriculum `v3e_vdec`）只在 **3–5 靜 + 5 動** 的低密度訓練。實測在部署密度下崩潰：

| 場景 | 撞靜態 | 撞牆 | 到達(SR) |
|---|---|---|---|
| 訓練密度（11 靜/16×16/goal 5-9） | **0.0%** | 23.4% | 56.2% |
| 部署密度（15 靜/12×12/goal 7-9） | **74.3%** | 21.1% | **4.5%** |

同一 checkpoint，靜態碰撞 0% → 74%，差別全在**密度**（部署靜態密度 10.4/100㎡ ≈ 訓練 1.5/100㎡ 的 **7×**）。這是**密度容量牆**，不是訓練配置錯。

**幾何可行性**：障礙半徑 ≈ 0.3m、機器人半徑 0.35m。12×12 放 15 靜只佔機器人可行區 ~20% → **有縫可走，非飽和**。目標可訓練。

---

## 2. 核心設計原則

分開處理「可訓練」與「訓練解決不了」的兩部分：

```
靜態密度 → 課程逐階段加密,policy 學會穿密集靜物   (可訓練:單幀 LiDAR 空間問題)
動態密度 → 反應式 policy 打底 + 部署 shield 安全網 (預判已證死路:時間問題)
```

**為何動態不靠訓練預判**：本專案三大方法族（架構 encoder / reward Term A+TTC+decel+aux / 任務 crossing b3）系統性證明——在此動力學（v_max 1.0、單幀 72-ray LiDAR 已提供足夠空間資訊）下，**RL 揭示偏好始終不使用 LV-DOT 速度**。動態避障的「提早預判」在當前架構要不到，故改用部署端幾何安全層補尾。

---

## 3. 架構：兩塊

### 3.1 靜態塊——deployment-tuned 密度 ramp curriculum

**基底**：`vdec2`（`wd_single_agent_v3e_vdec2.py`）已把靜態做成單調 ramp（vs `v3e_vdec` 卡 3-5）。

**新 curriculum**：`warp_drive_single_agent_v3e_deploy_dense`（在 vdec2 上再 tune 到部署幾何）：

**非線性 ramp**（Grok 建議 2026-07-12：前段線性打底、後段密集切分，方便觀察容量牆 knee）：

| 階段 | 靜態 | 動態 | arena | 說明 |
|---|---|---|---|---|
| SA1 | 2 | 0 | 12×12 | nav bootstrap（線性段） |
| SA2 | 4 | 0 | 12×12 | 純靜態打底 |
| SA3 | 6 | 1 | 12×12 | 加少量動態 |
| SA4 | 8 | 2 | 12×12 | 線性段結束 |
| SA5 | 10 | 3 | 12×12 | **密集段起**（後段每階 +2） |
| SA6 | 12 | 3 | 12×12 | dense static |
| SA7 | 14 | 4 | 12×12 | high pressure |
| SA8 | **15** | **4** | 12×12 | **部署密度** |
| （掃描） | 17,19 | 4 | 12×12 | 訓完後 SA8 policy 容量牆掃描（§6，不訓） |

- **前段 SA1-4 線性**（2,4,6,8）打底基本穿縫；**後段 SA5-8 每階 +2**（10,12,14,15）密集觀察容量牆起點。
- **與 vdec2 差異**：arena 14×14→12×12；靜態尾端 13→15；後段非線性加密；動態 ramp 到部署密度（≥4）。
- **動態行為**：沿用 rmd 的 crossing/head_on 混合（練 reactive 反應到當前位置）。
- **LV-DOT channel**：K=5，位置欄**保留**（reactive 反應需要當前位置），速度欄雖證沒用但不動（避免改 obs 維度連鎖）。
- **config 硬約束**：`max_active_obstacles ≥ 24`（SA8 19 顆 + 邊際），否則 `train_rnn_car_wdclip.py` N_obs 硬 cap 靜默截斷。
- **goal 距離校準**：需驗證 12×12 + 15 靜下 goal 7-9m 可行放置（eval log 曾見 `Goal resample fallback` → 可能需放寬 near-goal 或縮 goal 距離）。

### 3.2 動態塊——部署 shield 三層階梯

部署端純幾何層（不進訓練，車端即時算，[[project_deploy_shield_ladder]]）：

```
① 減速閘 (slowdown gate):
   d_safe < d_slow → v_cmd *= linear_scale(d_safe)
   d_safe < d_stop → v_cmd = 0
   給 reactive policy 反應時間:「來不及閃」→「慢下等它過」

② α slew (動作平滑限速):
   |Δaction| per step ≤ slew_max
   壓 bang-bang 抽動(舞龍舞獅),實車動作平順

③ FGM 仲裁 / ORCA 速度避碰 (條件接管):
   3a FGM: shield 判極端貼身危險 → 純幾何 Follow-the-Gap 選最大縫覆蓋 policy
   3b ORCA(升級,Grok 建議): 用 LV-DOT 障礙速度(vx,vy)做幾何速度避碰
      → 「不學、直接幾何用速度」,補上 RL 拒學的預判(見 §5.5 H2 對照)
      → 車端已裝 pyrvo2 + 有移植插入點([[finding_car_lvdot_impl]]),近零成本
```

Shield 參數（`d_slow`、`d_stop`、`slew_max`、FGM/ORCA 觸發閾值）需在部署密度 eval 上校準。

**⚠️ freeze/deadlock 警訊**（Grok）：dense 場景減速閘易造 deadlock（大家都停）。逐層驗收必盯 `freeze_ratio`——若 shield 開了 `freeze_ratio` 暴增 → 減速閘過激或需 ORCA 取代純減速。

---

## 4. 訓練管線

```
SA1(2靜0動) → SA2(3靜0動) → … → SA8(15靜4動)
每階段:①訓到收斂 ②det eval SR≥90 才畢業 ③resume 進下一階
命名:sa{N}_deploy_dense_ne1024_s42  (依 run_name 規範)
固定:seed / vf_coeff / grad clip / rollout / lr / entropy floor
遞進:僅 stage 數字 + 密度
穩定化:高密度階段(SA6-8)防 trust-region 崩 → KL early-stop(target_kl 0.015)已有修法
```

**畢業硬門檻**（[[feedback_sr90_graduation_criterion]]）：**每階段** det eval SR≥90 才算畢業，切換前必跑 det eval 定位。

---

## 5. 驗收

**部署密度 det eval**（12×12 + 15 靜 + 4 動）：

| 指標 | 目標 | 理由 |
|---|---|---|
| 撞靜態 CR | **<5%** | 靜態可訓練,應壓得下 |
| 撞動態 CR（policy only） | 量測基線 | reactive 地板 |
| 撞動態 CR（+shield） | **<10–15%** | shield 接尾 |
| 總 SR（policy only） | 55–70%（估） | 容量牆現實 |
| 總 SR（policy + shield） | **80%+** | 堪用部署門檻 |
| `freeze_ratio`（+shield） | **不得暴增** | 減速閘 deadlock 警訊（Grok） |

**12×12 資訊論下限判準**（Grok 定量化）：SA8 15靜訓到收斂後 —
- **SR ≥ 70%** → 沒到硬限，繼續（shield + 微調可堪用）
- **SR < 40% 且怎麼訓都不升** → 接近資訊論硬限，須降場景難度或重新談部署幾何
- 40–70% → 灰區，靠 shield/ORCA 補到 80%+

**⚠️ 誠實天花板**：純 policy 在 15靜+4動 **不可能 SR 100%**（容量牆物理現實）。A 的價值 = 靜態碰撞壓近 0（可訓練）+ shield 接動態尾巴 → 合力「堪用部署」。若純 policy 卡 60%、shield 後也上不去 → 才考慮方向 B（架構升級），但**先做 A 量出真實天花板 + §5.5 診斷瓶頸來源**，不盲賭架構。

---

## 5.5 容量牆診斷：感知 vs 決策（Grok 2026-07-12）

若 §5 純 policy 撞容量牆（SR 卡住），**先拆瓶頸來源再決定方向 B 怎麼做**，不盲換架構。

**專案已有的部分答案**：v24（LiDAR MaxPool + 動態 attention）SR 62.5%、v25（完整 HEIGHT：LiDAR 拆 18 tokens cross-attention）、v27（HEIGHT dense 22 障礙 SR 61.6%）。→ **純換感知 encoder（attention）試過，仍 ~61% 撞牆**。但兩個 caveat 未拆：① v25/v27 用了 god-mode obstacles-60D 特權 channel（不可部署）→ **deployable/LiDAR-only 感知 encoder 還沒乾淨測**；② v27 容量牆可能是 HEIGHT TopK=10 截斷、非感知壓縮本質。

**三刀診斷協定**：

```
刀0 (零成本,先做): per-beam LiDAR 梯度歸因
   復用現成 CHARGE_GRAD_ATTR hook,改測高密度時 policy 對各 LiDAR beam 的 |∂/∂beam|
   梯度集中少數 beam → MaxPool 把多障礙混疊(感知壓縮) → 支持感知瓶頸假設

刀1 (感知): 固定 policy head,只換 LiDAR encoder
   MaxPool → CNN / Transformer(LiDAR-only,不用 60D 特權)
   同密度 SR 明顯升 → 感知瓶頸為主

刀2 (決策): 固定感知 encoder,只換 policy head
   RNN → Transformer / attention-over-obstacles
   SR 升 → 決策能力不足為主
```

Grok 預測：**感知瓶頸佔比較大**（單幀 VLP-16 dense 易遮擋混疊，MaxPool 壓多障礙成單一特徵）。刀0 幾乎零成本，Phase 0 就能先做一刀定調。

---

## 6. 成本與風險

| 項目 | 估計 |
|---|---|
| 訓練時間 | SA1→SA8 從頭，~2–4 天 |
| 風險① | 高密度階段(SA6-8) trust-region 崩 → KL early-stop 修法 |
| 風險② | SA8 15靜撞容量牆 SR 上不去 → §5.5 三刀診斷瓶頸 → 針對性轉 B |
| 風險③ | goal 放置飽和(12×12+15靜) → 校準 goal 距離/near-goal |
| 現成 | vdec2 骨架、shield 三層代碼、密度旋鈕、det eval、CHARGE_GRAD_ATTR hook、車端 pyrvo2 |
| 新建 | deploy_dense curriculum + config + shield/ORCA 校準 + 診斷刀0-2 |

---

## 7. 交付物

1. `curriculum/phases/wd_single_agent_v3e_deploy_dense.py`（新 curriculum，非線性 ramp）+ registry 註冊
2. `configs/wd_sa{1..8}_deploy_dense.yaml`（8 階段 config，max_active_obstacles≥24）
3. shield 三層參數校準（部署密度 eval），含 ORCA(3b) 對照
4. **ORCA baseline（H2 對照，Grok）**：pyrvo2 幾何速度避碰，量動態 CR vs reactive policy → 證「速度可用只是 RL 不學」
5. **容量牆診斷（§5.5）**：刀0 per-beam grad-attr（零成本，Phase 0）→ 撞牆才做刀1/刀2
6. det eval 報告（各階段畢業 + 部署密度驗收表 + 容量牆密度掃描曲線）
7. 更新記憶：run↔wandb id lineage、部署配方

---

## 8. 不做（YAGNI / scope 邊界）

- **不追「RL 學習式」動態預判**：velocity-aux / TTC 稅 / speed-encoder 已證死路，不再做。
  - ⚠️ 但 **ORCA 幾何式用速度不算違反**——它不「學」，直接幾何用 LV-DOT 速度，是部署層工具（§3.2-3b / §7-4 H2 對照）。這正是 Grok 點出的關鍵區分。
- **不改訓練 obs 維度**：LV-DOT 速度欄留著（避免連鎖改動），policy 不指望它，但 ORCA 部署層會用。
- **不預先做架構升級**（方向 B）：先 §5.5 三刀診斷瓶頸來源（刀0 零成本先做），撞牆才針對性換 encoder/head，非盲換。
- **不做結構化 layout**（方向 C）：除非確認真實場地有走道結構。
