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

| 階段 | 靜態 | 動態 | arena | 說明 |
|---|---|---|---|---|
| SA1 | 2 | 0 | 12×12 | nav bootstrap |
| SA2 | 3 | 0 | 12×12 | 純靜態打底 |
| SA3 | 5 | 1 | 12×12 | 加少量動態 |
| SA4 | 7 | 2 | 12×12 | |
| SA5 | 9 | 3 | 12×12 | |
| SA6 | 11 | 3 | 12×12 | dense static |
| SA7 | 13 | 4 | 12×12 | high pressure |
| SA8 | **15** | **4** | 12×12 | **部署密度** |

- **與 vdec2 差異**：arena 14×14→12×12；靜態尾端 13→15；動態 ramp 到部署密度（≥4）。
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

③ FGM 仲裁 (Follow-the-Gap 條件接管):
   shield 判極端貼身危險 → 純幾何 FGM 選最大縫方向覆蓋 policy
   最後防線,policy 撞前由幾何接管
```

Shield 參數（`d_slow`、`d_stop`、`slew_max`、FGM 觸發閾值）需在部署密度 eval 上校準。

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

**⚠️ 誠實天花板**：純 policy 在 15靜+4動 **不可能 SR 100%**（容量牆物理現實）。A 的價值 = 靜態碰撞壓近 0（可訓練）+ shield 接動態尾巴 → 合力「堪用部署」。若純 policy 卡 60%、shield 後也上不去 → 才考慮方向 B（架構升級），但**先做 A 量出真實天花板**，不盲賭架構。

---

## 6. 成本與風險

| 項目 | 估計 |
|---|---|
| 訓練時間 | SA1→SA8 從頭，~2–4 天 |
| 風險① | 高密度階段(SA6-8) trust-region 崩 → KL early-stop 修法 |
| 風險② | SA8 15靜撞容量牆 SR 上不去 → 量出天花板後轉 B |
| 風險③ | goal 放置飽和(12×12+15靜) → 校準 goal 距離/near-goal |
| 現成 | vdec2 骨架、shield 三層代碼、密度旋鈕、det eval pipeline |
| 新建 | deploy_dense curriculum + config + shield 參數校準 |

---

## 7. 交付物

1. `curriculum/phases/wd_single_agent_v3e_deploy_dense.py`（新 curriculum）+ registry 註冊
2. `configs/wd_sa{1..8}_deploy_dense.yaml`（8 階段 config，max_active_obstacles≥24）
3. shield 三層參數校準（部署密度 eval）
4. det eval 報告（各階段畢業 + 部署密度驗收表）
5. 更新記憶：run↔wandb id lineage、部署配方

---

## 8. 不做（YAGNI / scope 邊界）

- **不追動態預判**：速度預判已證死路，不再做 velocity-aux / TTC 稅 / encoder。
- **不改 obs 維度**：LV-DOT 速度欄留著（避免連鎖改動），只是不指望它。
- **不做架構升級**（方向 B）：留到 A 量出容量牆後再評估，非本 spec 範圍。
- **不做結構化 layout**（方向 C）：除非確認真實場地有走道結構。
