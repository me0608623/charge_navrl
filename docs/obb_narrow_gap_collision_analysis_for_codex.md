# OBB 窄縫碰撞分析 — 給 Codex 的查證＋計劃簡報

> 目的：用戶部署場景最窄必經縫 ≈ **0.85m**，圓形碰撞模型拒走。用戶已同意「可從頭訓練」。
> 用戶指示：**Codex 協助確認與計劃、先只寫 spec 不實作**。
> 本文由 Claude（監督方）撰寫，用戶對其中的 reward 歸因存疑 → **每條聲明附 file:line，請 Codex 逐條獨立驗證後再定計劃**。
> 日期：2026-07-20

---

## 0. TL;DR（Claude 已自我更正的結論）

- **靜態 0.85m 窄縫拒走的唯一根因 = LiDAR 各向同性碰撞 termination（threshold 0.45m），與 reward 無關。**
- Claude 最初主張「reward 需重設計（future_occupancy / penalty_speed_near_obs）」，**經查證為誤**（見 §2），已撤回。
- 修法收斂為：把碰撞 termination 由圓改為方向性長方形（OBB）。reward 核心不動。仍需從 SA1 重訓。

---

## 1. 幾何事實（可獨立算，無需信任 Claude）

實體車：寬 0.40m × 長 0.70m → 半寬 w=0.20、半長 l=0.35、半對角 d=√(0.20²+0.35²)=**0.403m**。

碰撞常數（`cfg/charge_env_cfg_vlp16.py:125-128`）：
- `ROBOT_BODY_RADIUS = 0.35`
- `COLLISION_BUFFER = 0.10`
- `COLLISION_THRESHOLD = 0.45`

圓模型最小可過縫 = 2 × 0.45 = **0.90m**。
必經縫 0.85m < 0.90m → **被各向同性門檻擋掉 0.05m**。
長方形車窄邊對準：只需側向淨空 ≈ w + buffer = 0.20 + 0.10 = 0.30m（單側），0.85m 縫兩側各 0.425m 遠超過 → **物理可過，sim 誤判**。

**圓模型 0.35 落在內切圓(0.20)與外接圓(0.403)之間** → 同時：
- 側向過保守（要求 0.35 淨空，實需 0.20）→ 拒走可過窄縫；
- 對角/旋轉不夠保守（要求 0.35，實需 0.403）→ 轉角可能刮到。

---

## 2. Reward 歸因查證（用戶存疑處，請 Codex 覆核）

K8 凍結配方 reward（`rnn_car_modular/configs/e2e_k8_future_frozen_base.py:32,92-109`）：
`reward_profile="clean_progress"`，含 `anti_spin_weight=0.15`、`future_occupancy_weight=0.10`、`penalty_speed_near_obs=-1.0`。

| reward 項 | 對「靜態 0.85m 縫直行通過」是否開火 | 查證 file:line |
|---|---|---|
| `future_occupancy` no-go | **否** — 只對移動障礙 (vel≥0.1m/s) | `rewards/clean_progress.py:264` `moving = obstacle_vel.norm(dim=-1) >= move_threshold_mps`；`:266` `valid = moving & near` |
| `penalty_speed_near_obs` | **否** — clean_progress 根本不讀它 | `grep -c penalty_speed_near_obs rewards/clean_progress.py rewards/factory.py` = **0 / 0**；只活在 `rewards/wd_sparse.py:113` |
| `anti_spin` | **否** — 需 `near_hazard & high_omega & low_progress`（原地打轉才罰）；直行過縫 ω≈0 不觸發 | `rewards/clean_progress.py:180-183` |
| clean_progress 核心（progress+goal+撞−15+timeout+smooth） | 撞−15 綁 termination；progress 各向獨立 | `rewards/clean_progress.py` |

**結論：無任何稠密 reward 項阻止靜態窄縫通過。** Claude 原「reward 主兇」說法撤回。
（請 Codex 確認：clean_progress 是否真未經由 factory/curriculum 側面注入 penalty_speed_near_obs；以及 future_occupancy 的 move_threshold 是否在 curriculum 被改為 0。）

---

## 3. 真兇：碰撞 termination（★2026-07-20 更正：是「幾何式」不是 LiDAR-min）

> **重要更正**：Claude 先前版本寫「路徑 A = lidar_min_distance / min(72 beam) < 0.45」是**錯的**。逐檔讀 `TerminationsCfgVLP16`（`charge_env_cfg_vlp16.py:652`）後確認：**實際 enable 的碰撞判定全是幾何式（用位置算），根本沒有 LiDAR-min termination**。line 615 那個 LiDAR+0.45 是 **RewTerm（reward penalty）不是 DoneTerm**。請 Codex 以本節為準。

Env：`Isaac-Navigation-Charge-VLP16-Curriculum-WD` → `charge_env_cfg_wd_sparse:ChargeNavigationEnvCfgVLP16CurriculumWD`，繼承 base → terminations 在 `cfg/charge_env_cfg_vlp16.py:652 TerminationsCfgVLP16`。

實際 enable 的碰撞 DoneTerm：

| DoneTerm | func | 機制（實測程式碼） | 對 0.85m 縫 | 改 OBB |
|---|---|---|---|---|
| `collision` | `collision_contact_occurred`（contact sensor） | 物理接觸力；註解自承 kinematic 障礙常 <0.1N **嚴重低估** → 近乎 inert | 幾乎不觸發 | 不用 |
| **`wall_collision`** | `wall_collision_termination`（`robot_state.py:169`） | **點對牆 AABB 距離 < 0.45**（robot 當一個點，牆膨脹 0.45 = 等價 0.45 半徑圓）。實證 `robot_state.py:212`：`delta=(pos−wall_c).abs()−wall_s/2; dist=norm(delta); 撞 if dist<0.45` | **牆縫真兇** | ✅ 要 |
| **`obstacle_collision`** | `obstacle_collision_geometric`（`robot_state.py:232`） | 中心距 < `collision_distance`（cfg 給 **0.9**）或 per-obs `_obstacle_radii`（`robot_state.py:306-315`） | 障礙縫culprit（見 §3.1 雷） | ✅ 部分 |
| `robot_tipped_over`/`physics_explosion`/`time_out`/`goal_reached` | — | 姿態/爆炸/超時/到達 | 無關 | 不用 |

### 3.1 ⚠️ 一個必須拆清的雷：`obstacle_collision` 的 0.9m 混了兩種概念
`obstacle_collision_geometric` 的 `collision_distance=0.9` 註解寫「真實 LiDAR 盲區（比 0.9m 近真車看不到→無法反應→算失敗）」。它**同時扮演兩角**：
- **①實體車身撞到障礙圓柱** → 該用「長方形車 到 圓(障礙) 最近距離 < r_obs+buffer」→ **方向性**。
- **②障礙進入 LiDAR 盲區失敗** → 盲區**本質各向同性（沒方向）** → **不該長方形化**。

→ 建議拆成兩個 term（見 §4 改法二）。**若直接把 0.9 這個 term 整個長方形化 = 概念錯**（把感測盲區當成車身形狀，側向盲區被誤縮）。

---

## 4. 碰撞修改方法（本次要交付實作的核心）

核心一句話：**讓碰撞判定知道「車是長方形」+「車頭朝哪」，用真正的長方形去比，取代現在把車當一個大圓/一個點。** 需要的資料（車位置 `root_pos_w`、車頭朝向 `root_quat_w`、牆 AABB、障礙中心/半徑）**runtime 全部就位**，不加任何新感測。

車體常數（新增）：半長 `l=0.35`、半寬 `w=0.20`、`buffer=0.10`（yaw 由 `root_quat_w` 取，`robot_state.py:68` 已示範）。

### 改法一：`wall_collision`（牆縫主兇）— 點→長方形（SAT）
- **現況**：`check_wall_proximity_perenv`（`wall_layout.py`）把 robot 當點、對牆 AABB 算 point-to-box 距離 < 0.45。
- **改成**：長方形(車) vs 牆 AABB 的 **SAT 分離軸重疊測試**。2D 只需檢查 4 條軸：牆的 x、y 軸 + 車體前向、左向（yaw 旋轉）。各軸投影，任一軸不重疊→沒撞；4 軸全重疊→撞。含 buffer。
- **效果**：車頭直行過縫時，側向軸用 `w=0.20` → 側牆 0.425m > 0.20+0.10=0.30 → **不判撞 → 過得去**。
- **改動點**：`wall_layout.py check_wall_proximity_perenv`（加 yaw 參數 + SAT）+ 呼叫端 `robot_state.py:169 wall_collision_termination` 傳 yaw。約 20–30 行、集中兩處。

### 改法二：`obstacle_collision`（障礙縫）— 拆兩件事
1. **`obstacle_body_collision`（改 OBB）**：把障礙中心轉進車體系（yaw），夾到長方形邊界求最近點，`dist < r_obs+buffer` → 撞。標準「長方形-到-圓最近點」約 10–15 行，用 `_obstacle_radii`。
2. **`sensing_blindzone_failure`（保留圓、校準 r_min）**：中心距 < 真實 LiDAR r_min（v3 已是 0.25–0.5，**非 0.9**，請 Codex 查證現值）→ 失敗。**維持各向同性**（盲區沒方向）。

### ρ(θ) 那條路不需要了
先前提的「每根 beam 比 ρ(θ)」是基於「碰撞走 LiDAR-min」的**錯誤假設**。實際碰撞是幾何式 → 用上面 SAT / 長方形-圓最近點即可，**termination 層不需要 ρ(θ)**（ρ(θ) 只有在改 LiDAR-based reward 軟訊號時才用，屬 Tier 2 可選、非碰撞判定）。

---

## 4b. Reward 幾乎不用改（三層）
- **Tier 0 不變**：`progress`/`goal`/`timeout`/`smoothness` — 基於車中心或動作，與形狀無關。
- **Tier 1 自動生效（不改 reward 程式碼）**：撞擊懲罰值不變，它綁 termination 觸發 → termination 改 OBB 後**自動**只在長方形真的撞到時才給。
- **Tier 2 可選精緻化**：`future_occupancy` 的 separation、`anti_spin` 的 `near_obs_dist_m` gate 目前用中心距（隱含圓）；門檻（1.0m / hazard_dist）≫ 形狀差(0.15m)，近似仍成立 → **低優先，先不動**。
- 建議：**先純改 termination 重訓，reward 不加新 shaping**（歷史 reward-economics 結論：gap/膨脹/多幀 shaping 對避障風格幾乎不動）。0.85m 直通縫車頭直行 0.40+buffer=0.60<0.85 已可過，不需「窄縫朝向對齊」shaping；只有「需邊走邊轉的 L 型死角」才可能要，事後補。

---

## 5. Sim2Real 需協調（重要，勿漏）

- 真車端 LiDAR 安全處置（shield / min-range 盲區）目前也是**各向同性**（車端 r_min、shield d_safe）。
- 若 sim 允許側向 0.30m 過縫、但車端 shield 仍在 0.45m 各向同性煞停 → **sim2real 行為不一致**，OBB 訓練的窄縫能力在車上被 shield 抵消。
- → OBB 若上，車端 shield/min-range 需同步改為方向性，或至少把窄縫走廊列為 shield 例外。**Codex 計劃須含車端對齊項。**

---

## 6. 重訓需求

- 現役 policy 內化了「避開 <0.9m 縫」（圓模型下訓成）。改 termination 邊界 = 改最優策略 = **必須從 SA1 重訓**（用戶已同意）。
- 這是**新血緣**（碰撞模型改了），與凍結 K8 lineage `b0e5f8a719f` **不再 checkpoint 可比**（可比性本來就因 OBB 而斷）。以凍結配方為基礎、只疊加 OBB delta（+ §7 決定的 DR/noise），命名另開 `sa{N}_e2e_k8_obb_*`。

---

## 7. ★用戶要求評估：同時加入「制動器延遲」+「full material 雜訊」

用戶問：這次 OBB 從頭重訓，能否**同時**併入 (a) 制動器延遲 (actuator delay DR)、(b) full material LiDAR 雜訊？Claude 評估如下，**Codex 定案**：

### (b) full material 雜訊 — ✅ 建議納入
- 旗標 `vlp16_noise_mode`（ideal/sigma/bias/dropout/full），既有配方即 **SA1=bias → SA2+=full**（VLP-16 實測雜訊，見 memory `finding_vlp16_measured_noise_integration`）。
- **低風險、部署必需、已是標準課程一環** → 建議 OBB 重訓從一開始就沿用既有 curriculum-gate（SA1 bias → SA2+ full），不視為額外實驗變因。

### (a) 制動器延遲 — ⚠️ 建議「不要」一開始就併入主線 OBB 重訓
理由三點：
1. **既有設計刻意隔離**：clean SA1→SA8 主線**刻意不含制動器延遲**，靠畢業後獨立「SA8-DR bridge」fine-tune 補（memory `project_sa8_dr_bridge_deploy`），為保因果乾淨 + checkpoint 可比 + 隔離其他變因。
2. **延遲×窄縫是最危險交互**：延遲讓車在窄縫反應遲鈍，正是舞龍舞獅/擺盪風險最高處；與全新碰撞模型（改變貼牆駕駛行為）同時上，交互難預測。
3. **多變因混一起無法歸因**：若 OBB+delay+noise 一起失敗，分不清是誰造成 → 違反單變因原則。

**建議路線**：OBB clean（含 full noise）→ 訓到畢業、**先驗證窄縫能力真的學到** → 再做獨立「OBB-DR bridge」短 fine-tune 加制動器延遲（`--enable_actuator_dr`, `actuator_delay_range=(0,2)`=0–400ms, `obs_delay_steps=(0,0)` 不開觀測延遲毒藥）。
**若用戶堅持一次到位**：至少把制動器延遲 **curriculum-gate 到後段（SA6–8）**，等窄縫技能先在前段成形；且驗收須含 delay 分布 deterministic 延遲矩陣。

---

## 8. 給 Codex 的行動請求（用戶已授權開始重訓）

1. **逐條覆核 §2、§3 的 file:line**（尤其：penalty_speed_near_obs 在 clean_progress **inert**、future_occupancy **moving-only**、真兇是 `wall_collision`/`obstacle_collision` **幾何式**非 LiDAR-min）。有反例請標出。
2. 查證 3 件事：(i) WD env **實際 enable 哪些 termination**；(ii) `obstacle_collision` runtime 用 **0.9 還是 per-obs `_obstacle_radii`**；(iii) 真實 LiDAR **r_min 現值**（盲區 term 用）。
3. 定案 buffer 取值、`obstacle_collision` 拆分（body OBB + 盲區圓）與否。
4. 實作 §4 改法一（wall SAT）+ 改法二（obstacle OBB / 盲區拆分）；reward 依 §4b 先不動。
5. 定案 §7：full noise 納入（curriculum-gate）、制動器延遲**建議走畢業後 bridge**（或後段 gate）。
6. **從 SA1 開始重訓**新血緣 `sa{N}_e2e_k8_obb_*`，四閘驗收（SR/CR/Δ|ω|/安全弧/TO）。
7. 不覆蓋現有 ckpt / 不影響 supervisor 的 GUI 未提交檔。

---

## 9. ★Codex 審查修正（2026-07-20，已由 Claude 逐條驗證 → 取代前述衝突處，為實作權威）

Codex 審查後確認主診斷正確（85cm 窄縫真兇=幾何碰撞把車當圓、非 LiDAR 非 reward；制動器延遲後處理也對），但抓出 spec 四個實作級錯誤。Claude 已逐檔驗證**四點全部成立**，以下為修正版，**優先於 §4/§5/§7 的衝突敘述**：

### 修正① 不改共用 `check_wall_proximity_perenv`（取代 §4 改法一的「改 wall_layout」）
- 該函式被多處呼叫：`goal_command.py:346/489`（目標採樣、出生淨空）、`mixed_parallel.py:446`（障礙擺放門檻0.8）、curriculum 牆反彈。直接改 OBB 會**改壞其他語意**。
- **正解**：**新增專用** `robot_obb_wall_collision`，**只由** `robot_state.py:169 wall_collision_termination` 呼叫。共用函式保持圓/點語意不動。

### 修正② termination 只做「真實 OBB 車身碰撞」，不新增圓形盲區 termination（取代 §4 改法二 point 2 + §3.1）
- 加圓形 `sensing_blindzone_failure` 會**把窄縫重新封死**：r_min=0.5m 是障礙**表面**距離，障礙半徑 0.3m → 中心門檻 0.8m；0.85m **表面**縫的中心距僅 1.45m，過縫時到各障礙中心 0.725m < 0.8m → 仍判失敗、無解。
- **正解**：termination **只有** OBB 車身碰撞。LiDAR 盲區留在 **observation / 診斷 / 部署 shield**，**不**當物理碰撞 termination。

### 修正③ 障礙 OBB 用 `_obstacle_phys_radii`（取代 §4 改法二「用 _obstacle_radii」）
- K8 的 `_obstacle_radii` 實際 = `_obstacle_phys_radii + edge`（膨脹碰撞距離，≈0.9 舊門檻語意），**不是物理半徑**。
- **正解**：OBB 實體碰撞用 `_obstacle_phys_radii`（`train_rnn_car_wdclip.py:2850`，phys≈0.3）。

### 修正④ full-material 不是 frozen K8 既有標準（取代 §7(b) 的「建議納入」）
- `e2e_k8_future_frozen_base.py:86-90` 明確 `lidar_no_noise=True / lidar_*_bias=False / no_domain_randomization=True / enable_actuator_dr=False / obs_delay_steps=(0,0)` → **frozen K8 是刻意乾淨**。
- full-material 的 human dropout / mixed-pixel 參數仍標 provisional（`charge_env_overrides.py:561`）。
- **正解**：full-material **不併入初始重訓**，等幾何過關後**獨立 A/B** 加入。

### 修正後的實驗順序（Codex 建議，Claude 認可）
1. **先確認 USD collision mesh 真實長寬**：目前 0.40×0.70m 尚未與實際 collision mesh 對證（用戶口述值，須核 USD）。
2. 只實作 **wall OBB（專用函式）+ obstacle OBB body collision（用 phys_radii）**。
3. **無雜訊 deterministic 窄縫單元測試**：驗 0.85m 可通過、真正碰撞仍會終止。
4. 幾何過關後，**再獨立加 full-material 做 A/B**。
5. 完整 **SA1→SA8 畢業後**，才做 **actuator-delay bridge**。

### USD collision footprint 實測與真車尺寸凍結（2026-07-20）

Codex 以 headless Isaac Sim 載入實際 runtime 資產
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/charge.usd`，展開 instance proxies，找到 13 個啟用 `PhysicsCollisionAPI` 的 prim。每個 primitive 由實際 Cube/Cylinder/Mesh 幾何點套用完整 local-to-world transform，再轉回 `base_link` 車體座標；不是用 render mesh，也不是沿用口述尺寸。

USD 實測 collision union（`base_link` frame）：

| 項目 | 實測值 |
|---|---:|
| longitudinal x range | `[-0.4580, +0.2020] m` |
| lateral y range | `[-0.3035, +0.3035] m` |
| collision footprint | `0.6600 × 0.6070 m` |
| OBB center offset from `base_link` | `(-0.1280, 0.0000) m` |
| USD union half length | `0.3300 m` |
| USD union half width | `0.3035 m` |

用戶最終以實體車完整外緣重測：**含左右輪寬度 0.60m、長度 0.70m**。這組是碰撞 footprint 的權威尺寸。USD `base_link` 主體碰撞盒為 0.66×0.40m，左右輪 collider 將完整寬度擴到 0.607m；與實車完整寬度只差 0.007m，模型基本對齊。感測器碰撞體沒有再擴大 XY union。

USD 主體 collider 與完整 union 都顯示幾何中心相對 `base_link` 位於 x=-0.128m，因此 OBB 實作仍須套用這個 body-frame center offset；若直接以 root 為中心，前後碰撞邊界會錯位。

凍結常數（新 OBB lineage）：

```text
ROBOT_OBB_HALF_LENGTH_M = 0.3500
ROBOT_OBB_HALF_WIDTH_M = 0.3000
ROBOT_OBB_CENTER_OFFSET_X_M = -0.1280
ROBOT_OBB_CENTER_OFFSET_Y_M = 0.0000
ROBOT_OBB_BUFFER_M = 0.1000
```

`buffer=0.10m` 保留既有 `COLLISION_BUFFER` 安全語意，避免在改形狀模型時同時放寬安全裕量。膨脹後語意 OBB 為 `0.9000 × 0.8000m`；正向置中的 0.85m 縫橫向總容差為 `0.8500-0.8000=0.0500m`，即每側 0.0250m。OBB center 相對旋轉 root 偏後 0.128m，偏航時會額外產生橫向掃掠；計入此 offset 後，置中通過的理論偏航界線約為 `±2.52°`（2°可通、3°碰撞）。窄縫測試除置中直行外，必須覆蓋 lateral offset 與 yaw，確認「正向可通、偏斜真撞」的實際邊界。PhysX contact sensor 不列為 oracle：現有 kinematic 牆/障礙由 teleport 更新，程式既有診斷已確認其接觸力會漏報；權威 termination 是解析 OBB 幾何。

### OBB 實作與驗證結果（2026-07-20）

- 新增純 torch SAT / OBB-circle 幾何：`mdp/terminations/obb_collision.py`。
- `wall_collision_termination(use_obb=True)` 只走專用 `robot_obb_wall_collision`，共用牆距離函式未改。
- `obstacle_collision_geometric(use_obb=True)` 使用 `_obstacle_phys_radii`；舊 `_obstacle_radii` 僅保留給 legacy circle 路徑。
- 新增顯式 `--use_obb_collision` lineage gate；預設關閉，舊 checkpoint 行為不變。新 SA1 配方為 `e2e_sa1_k8_obb`，checkpoint metadata 會記錄此旗標，play 端會自動繼承。
- 新 lineage 從 SA1 固定加入 4D applied-action history：`[aₜ₋₁, ωₜ₋₁, aₜ₋₂, ωₜ₋₂]`。policy observation 為 83D，K8 extractor 加入 96D 後 policy head 輸入為 179D；這讓畢業後 actuator-DR bridge 不必改 checkpoint input shape。
- 純 torch 測試：`12/12 passed`，涵蓋前/後/側障礙、center offset、0.85m 縫、lateral offset、2°/3° yaw 與 inactive wall mask。
- Isaac Sim 4-env smoke：直接 OBB 與 TerminationManager 均為 `[0,0,1,1]`（0°通過、2°通過、3°終止、偏移造成實體 footprint 穿透時終止）；legacy circle 為 `[1,1,1,1]`，證明舊模型會拒絕所有 0.85m 案例。
- 新增 policy-level Gate5：兩段牆從 arena 邊界延伸到中央，只留下 0.85m 開口，robot 必須由 x=-3m 穿至 x=+3m goal。Gate 同時檢查 goal SR、wall CR、實際跨越率，以及喉部 `|yaw|` p50/p90/p95 和落在 ±2.52° 內的比例；不以單純幾何 smoke 取代學習能力驗收。

### ⚠️ last-action observation 維度前置決策
若未來 bridge 要加 last-action observation，**必須在 SA1 之前決定輸入維度**；否則 SA8 後新增通道會使 checkpoint 第一層 shape 不相容。

### 現況與下一步（取代 §8「已授權開始重訓」）
- OBB 幾何、83D lineage、離線測試、4-env Isaac smoke 與 policy Gate5 均已完成。正式 SA1 已用 `e2e_sa1_k8_obb` fresh start；reward、K8、future occupancy、無雜訊與無 actuator DR 配方維持不變。
- 現行訓練程序目前**沒有在執行**（SA8 v2 於 iter490 停，ckpt_57600 為最佳安全點）。

---

## 附：相關記憶 / 筆記
- `.claude` memory: `finding_rect_robot_vs_circular_collision.md`
- Obsidian: `isaaclab/2026-07-20_長方形車體vs圓形碰撞模型.md`（含 §2 更正後需回填：reward 非主兇）
