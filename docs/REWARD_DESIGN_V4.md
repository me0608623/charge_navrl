# NavRL-Ground v4 獎勵設計文件

> navrl_ground_v4 + goal_first_v2 課程 — 2026-03-24

## 1. 總覽

### RewardManager 公式

Isaac Lab 的 `RewardManager` 每步計算：

$$
R_t = \sum_{i} f_i(s_t, a_t) \cdot w_i \cdot \Delta t
$$

- $f_i$ : 獎勵函數，回傳 `[N]` tensor（N = num_envs）
- $w_i$ : 權重（正=獎勵，負=懲罰）
- $\Delta t = 0.2\text{s}$（decimation=20, sim_dt=0.01s）

### PPO 折扣回報

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k \cdot R_{t+k}
$$

- $\gamma$ : 折扣因子（Stage-dependent, 0.990 → 0.998）
- $T$ : episode 最大步數（225 → 450，隨 Stage 增加）

---

## 2. 獎勵項數學定義

### 2.1 Goal Velocity（朝目標速度）

**數學:**
$$
\vec{d} = \frac{\vec{p}_\text{goal} - \vec{p}_\text{robot}}{\|\vec{p}_\text{goal} - \vec{p}_\text{robot}\|}
$$
$$
v_\text{toward} = \vec{v}_\text{robot} \cdot \vec{d}
$$
$$
f_\text{vel}(s_t) = \text{clamp}\left(\frac{v_\text{toward}}{v_\text{max}}, -1, 1\right) \cdot \mathbb{1}[d_\text{goal} > 0.1]
$$

**參數:** $v_\text{max} = 1.0$ m/s, `use_soft_gate=False`（v4 無安全門控）

**範圍:** $f_\text{vel} \in [-1, 1]$

**行為:**
| 速度 | $f_\text{vel}$ | 含義 |
|------|---------|------|
| 全速朝目標 (1.0 m/s) | +1.0 | 最大獎勵 |
| 半速朝目標 (0.5 m/s) | +0.5 | 線性比例 |
| 靜止 | 0.0 | 無獎勵 |
| 全速背離 (-1.0 m/s) | -1.0 | 最大懲罰 |

---

### 2.2 Goal Progress（距離縮減 PBRS）

**數學:**
$$
\text{progress}_t = \text{clamp}(d_{t-1} - d_t,\ -0.25,\ 0.25)
$$
$$
f_\text{prog}(s_t) = \text{progress}_t
$$

**參數:** `progress_clip=0.25`, `use_soft_scale=False`（v4 無安全縮放）

**範圍:** $f_\text{prog} \in [-0.25, 0.25]$

**物理限制:** 最大位移 $= v_\text{max} \times \Delta t = 1.0 \times 0.2 = 0.2$ m/step，因此 clip=0.25 留 25% 餘裕。

**特性:** PBRS 形式 — $\Phi(s) = -d_\text{goal}$，保證 policy invariance (Ng et al., 1999)。

---

### 2.3 Static Safety（靜態 LiDAR 安全）

**數學:**

(A) 全域 log clearance（72 bins）:
$$
c_j = \max(\text{bin}_j - r_\text{body},\ 10^{-3}), \quad j = 1, \ldots, 72
$$
$$
r_\text{global} = \text{clamp}\left(\frac{1}{72}\sum_{j=1}^{72}\ln(c_j),\ -6,\ 3\right)
$$

(B) 前向堵塞懲罰（±30° 扇區，最近 5 bins）:
$$
d_\text{front} = \text{mean}(\text{topK}_5(\text{bins}_{30..42}))
$$
$$
r_\text{front} = -\max(0,\ d_\text{warn} - d_\text{front}) \cdot v_\text{forward}^+
$$

**組合:**
$$
f_\text{ss}(s_t) = a_\text{global} \cdot r_\text{global} + a_\text{front} \cdot r_\text{front}
$$

**參數:** $r_\text{body} = 0.35$m, $d_\text{warn} = 1.2$m, $a_\text{global} = a_\text{front} = 1.0$

**典型值:**
| 場景 | $r_\text{global}$ | $r_\text{front}$ | $f_\text{ss}$ |
|------|------------|------------|-------|
| 空曠 (clearance ≈ 8m) | $\ln(7.65) \approx 2.0$ | 0 | **~2.0** |
| 中等 (clearance ≈ 1.5m) | $\ln(1.15) \approx 0.14$ | 0 to -0.3 | **~0.0** |
| 擁擠 (clearance ≈ 0.5m) | $\ln(0.15) \approx -1.9$ | -0.5 to -1.0 | **~-2.5** |

---

### 2.4 Dynamic Safety（動態障礙物安全）

**Mode: closing_risk**

$$
c_i = \|p_{\text{obs}_i} - p_\text{robot}\| - r_\text{body} - r_{\text{obs}_i}
$$
$$
\text{closing\_rate}_i = -\frac{(\vec{p}_{\text{obs}_i} - \vec{p}_\text{robot})}{\|...\|} \cdot (\vec{v}_{\text{obs}_i} - \vec{v}_\text{robot})
$$
$$
\text{risk}_i = \max(0,\ \text{closing\_rate}_i) \cdot \exp\left(-\frac{c_i}{\sigma}\right)
$$
$$
f_\text{ds}(s_t) = b_\text{log} \cdot \overline{\ln(c_i)}_\text{visible} - b_\text{risk} \cdot \overline{\text{risk}_i}_\text{visible}
$$

**參數:** $\sigma = 2.0$, $b_\text{log} = b_\text{risk} = 1.0$, max_obstacles = 20

**特性:**
- `closing_rate > 0` = 障礙物接近中 → risk 高
- `closing_rate ≤ 0` = 障礙物遠離 → risk = 0
- clearance 越小 + 接近越快 → 指數放大的風險懲罰

---

### 2.5 Alive（已移除，weight=0）

v3 原設計:
$$
f_\text{alive}(s_t) = 1.0 \quad \text{（無條件，每步都給）}
$$

**v4 移除原因:**

存活獎勵的本質是鼓勵「延長回合時間」。然而導航任務的目標是**盡快抵達終點**。
NavRL 的設計用 $r_\text{vel}$（速度獎勵）大力鼓勵高速朝目標前進，
配合 $\gamma = 0.99$ 的折扣壓力，agent 會自然想用最短時間完成任務。

$$
\text{alive 的矛盾:} \quad \underbrace{\text{活著}}_{\text{+reward}} \quad \text{vs} \quad \underbrace{\text{盡快到達}}_{\text{任務目標}}
$$

移除後，agent 的唯一「前進動力」來自:
1. $r_\text{vel}$: 朝目標越快 reward 越高
2. $r_\text{prog}$: 距離縮減 = 進度
3. $\gamma^t$ 折扣: 越早拿到 reward 越值錢
4. $r_\text{goal}$: 到達目標的大額 terminal reward

---

### 2.6 Control Smoothness（控制平滑）

$$
\Delta a_t = a_t^\text{linear} - a_{t-1}^\text{linear}
$$
$$
\Delta \omega_t = \omega_t - \omega_{t-1}
$$
$$
f_\text{smooth}(s_t) = c_v \cdot (\Delta a_t)^2 + c_\omega \cdot (\Delta \omega_t)^2
$$

**參數:** $c_v = c_\omega = 1.0$, weight = $-0.1$（負號 = 懲罰）

**範圍:** $f_\text{smooth} \in [0, 10]$（clamped）

---

### 2.7 Reaching Goal（到達目標 — 終端獎勵）

$$
f_\text{goal}(s_t) = \mathbb{1}[\|\vec{p}_\text{goal} - \vec{p}_\text{robot}\| - r_\text{body} < d_\text{threshold}]
$$

**參數:** $d_\text{threshold} = 0.35$m, $r_\text{body} = 0.35$m

**特性:**
- **稀疏**: 大部分步數 = 0，到達時 = 1（一次性）
- **觸發 → episode 結束**（termination condition 同時觸發）
- 實際收到的 reward = $w_\text{goal} \times 1.0 \times \Delta t$

---

### 2.8 Collision Ground（碰撞懲罰 — 終端懲罰）

$$
f_\text{collision}(s_t) = \mathbb{1}[\min_j(\text{lidar}_j) \leq d_\text{collision}]
$$

**參數:** $d_\text{collision} = 0.45$m（body_radius=0.35 + buffer=0.10）

**特性:**
- 觸發 → episode 結束
- 實際扣除 = $|w_\text{collision}| \times 1.0 \times \Delta t$

---

## 3. 權重表

### 3.1 基礎權重 (RewardsCfgVLP16NavRLGroundV3 — v4 使用)

| # | 獎勵項 | $w_i$ | $\|w_i \cdot \Delta t\|$ | 類型 | 觸發條件 |
|---|--------|-------|----------------|------|---------|
| 1 | reaching_goal | **500** | 100.0 | Terminal | $d_\text{goal} < 0.35$m |
| 2 | goal_velocity | 2.0 | 0.40 | Per-step | 每步 |
| 3 | goal_progress | 3.0 | 0.60 | Per-step | 每步 |
| 4 | static_safety | 2.0 | 0.40 | Per-step | 每步 |
| 5 | dynamic_safety | 2.0 | 0.40 | Per-step | 每步 |
| 6 | ~~alive~~ | **0.0** | 0.00 | ~~Per-step~~ | 已移除 |
| 7 | smoothness | -0.1 | 0.02 | Per-step | 每步 |
| 8 | collision_ground | -50 | 10.0 | Terminal | $d_\text{lidar} \leq 0.45$m |

### 3.2 Stage-Dependent 權重（goal_first_v2 課程覆蓋）

| Stage | 名稱 | reaching_goal | goal_vel | goal_prog | static_safety | dynamic_safety | γ | episode(s) |
|-------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | goal_open | 500 | **5.0** | **6.0** | 0.2 | 0.0 | 0.990 | 45 |
| 2 | goal_walls | 500 | **5.0** | **6.0** | 0.2 | 0.0 | 0.990 | 50 |
| 3 | static_light | 500 | 4.5 | 5.5 | 0.4 | 0.0 | 0.992 | 55 |
| 4 | static_medium | 500 | 4.0 | 5.0 | 0.8 | 0.0 | 0.993 | 60 |
| 5 | static_dense | 500 | 3.5 | 4.5 | 1.2 | 0.0 | 0.994 | 65 |
| 6 | dynamic_intro | **1000** | 3.0 | 4.0 | 1.0 | 0.4 | 0.995 | 72 |
| 7 | dynamic_medium | **1000** | 2.5 | 3.5 | 1.2 | 0.8 | 0.996 | 78 |
| 8 | crowded | **1000** | 2.0 | 3.0 | 1.4 | 1.2 | 0.997 | 85 |
| 9 | dense_9 | **1500** | 2.0 | 3.0 | 1.6 | 1.6 | 0.998 | 90 |
| 10 | dense_10 | **1500** | 2.0 | 3.0 | 1.8 | 1.8 | 0.998 | 90 |
| 11 | dense_11 | **1500** | 2.0 | 3.0 | 2.0 | 2.0 | 0.998 | 90 |
| 12 | ultimate | **1500** | 2.0 | 3.0 | 2.0 | 2.0 | 0.998 | 90 |

**設計原則:**
- `goal_vel/prog` 前期高（先學 goal-reaching）→ 後期低（讓出空間給安全項）
- `static/dynamic_safety` 隨障礙物密度遞增
- `reaching_goal` 隨 γ 和 episode 長度遞增（確保 terminal > 累計生存）

---

## 4. 期望值分析：為何 Agent 應該到達目標

### 4.1 問題：v3 的「目標前停滯」

在 v3 (reaching_goal=100, alive=0.2) 中，agent 學會在目標前 ~20cm 停下：

**停下不動每步獲得的 reward（「生存獎勵」）:**
$$
R_\text{survive} = w_\text{ss} \cdot f_\text{ss} \cdot \Delta t
$$

> 注意: v4 移除 alive (weight=0)。停下不動的唯一 reward 來自 static_safety。

**到達目標的一次性 reward（episode 結束後無未來 reward）:**
$$
R_\text{terminal} = w_\text{goal} \cdot 1.0 \cdot \Delta t
$$

**Value function 比較:**

$$
V(\text{stay}) = R_\text{survive} \cdot \frac{1 - \gamma^{T_\text{remain}}}{1 - \gamma}
$$

$$
V(\text{reach}) = R_\text{terminal}
$$

### 4.2 v3 各 Stage 的數值（修正前 — 錯誤激勵）

$$R_\text{survive} = (0.2 \times 1.0 + w_\text{ss} \times \bar{f}_\text{ss}) \times 0.2$$

| Stage | $w_\text{ss}$ | $\bar{f}_\text{ss}$ | $R_\text{survive}$/step | $T_\text{remain}$ | $\gamma$ | $V(\text{stay})$ | $V(\text{reach})$ | 結果 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.2 | 2.0 | 0.12 | 190 | 0.990 | 10.8 | 20 | **到達有利 ✅** |
| 4 | 0.8 | 2.0 | 0.36 | 265 | 0.993 | 43.3 | 20 | **停滯有利 ❌** |
| 8 | 1.4 | 1.0 | 0.36 | 390 | 0.997 | 86.4 | 20 | **停滯有利 ❌** |

### 4.3 v4 各 Stage 的數值（修正後 — 正確激勵）

$$R_\text{survive} = w_\text{ss} \times \bar{f}_\text{ss} \times 0.2 \quad \text{(alive=0, 無存活獎勵)}$$

| Stage | $w_\text{ss}$ | $\bar{f}_\text{ss}$ | $R_\text{survive}$/step | $V(\text{stay})$ | $w_\text{goal}$ | $V(\text{reach})$ | 比例 | 結果 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.2 | 2.0 | 0.08 | 7.2 | 500 | **100** | 14x | ✅ |
| 4 | 0.8 | 2.0 | 0.32 | 38.5 | 500 | **100** | 2.6x | ✅ |
| 5 | 1.2 | 1.5 | 0.36 | 49.9 | 500 | **100** | 2.0x | ✅ |
| 6 | 1.0 | 1.0 | 0.20 | 32.3 | 1000 | **200** | 6.2x | ✅ |
| 8 | 1.4 | 0.5 | 0.14 | 33.7 | 1000 | **200** | 5.9x | ✅ |
| 9 | 1.6 | 0.3 | 0.10 | 29.7 | 1500 | **300** | 10x | ✅ |
| 12 | 2.0 | 0.2 | 0.08 | 23.7 | 1500 | **300** | 13x | ✅ |

> **所有 Stage 都滿足 $V(\text{reach}) > V(\text{stay})$**

### 4.4 全速前進 vs 慢速前進的比較

Agent 不只要「到達」，還要「快速到達」。

**快速到達 (v=0.8 m/s, K=6 步走完 5m):**
$$
G_\text{fast} = \sum_{k=0}^{5} \gamma^k \cdot R_\text{approach}^{(k)} + \gamma^6 \cdot R_\text{terminal}
$$

Stage 4 ($\gamma=0.993$, $w_\text{vel}=4.0$, $w_\text{prog}=5.0$, alive=0):
$$
R_\text{approach} = (4.0 \times 0.8 + 5.0 \times 0.16 + 0.8 \times 2.0) \times 0.2 = 1.00 / \text{step}
$$
$$
G_\text{fast} = 6 \times 1.00 + 0.993^6 \times 100 = 6.00 + 95.9 = 101.9
$$

**慢速到達 (v=0.2 m/s, K=25 步走完 5m):**
$$
R_\text{slow} = (4.0 \times 0.2 + 5.0 \times 0.04 + 0.8 \times 2.0) \times 0.2 = 0.51 / \text{step}
$$
$$
G_\text{slow} = 25 \times 0.51 + 0.993^{25} \times 100 = 12.75 + 83.9 = 96.7
$$

**不到達（停在目標前）:**
$$
G_\text{stay} = 0.32 \times \frac{1 - 0.993^{265}}{0.007} = 38.5 \quad \text{(只剩 static\_safety)}
$$

$$
\boxed{G_\text{fast}(101.9) > G_\text{slow}(96.7) \gg G_\text{stay}(38.5)}
$$

**結論:** 移除 alive 後，停滯策略的回報只有到達策略的 38%。
$r_\text{vel}$ + $\gamma$ 折扣提供明確的「盡快到達」激勵。

---

## 5. 獎勵信號流圖

```
每步 t:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  goal_velocity: v_toward/v_max ∈ [-1,1]  ──┐               │
│                                             │               │
│  goal_progress: Δd clipped ∈ [-0.25,0.25] ─┤               │
│                                             ├──→ Σ ──→ R_t  │
│  static_safety: log(clearance) + front ────┤               │
│                                             │               │
│  dynamic_safety: log(c) - risk ────────────┤               │
│                                             │               │
│  ~~alive~~: weight=0 (已移除) ─────────────┤               │
│                                             │               │
│  smoothness: -(Δa² + Δω²) ────────────────┘               │
│                                                             │
│  [Terminal Events]                                          │
│  reaching_goal: +500×dt = +100 ──→ episode end             │
│  collision:     -50×dt  = -10  ──→ episode end             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 參考

- NavRL (Xu et al., 2025): reward decomposition 架構
- Ng et al. (1999): Potential-based reward shaping (PBRS) — policy invariance 證明
- `navrl_ground_rewards.py`: 8 項 reward 實作
- `goal_rewards.py`: reaching_goal 稀疏終端獎勵
- `charge_env_cfg_vlp16_curriculum.py`: RewardsCfgVLP16NavRLGroundV3 (v4), V5 (v5)
- `goal_obstacle_curriculum.py`: goal_first_v2 (v4), goal_first_v3 (v5)

---

## 附錄: NavRL-Ground v5 (navrl_ground_v5 + goal_first_v3)

> v4 訓練中期分析後的局部修正 — 2026-03-24

### 問題

v4 在 Stage 4 SR 卡在 ~74%，agent 走直線硬闖 (path efficiency 94%)。
Stage 4 的前進側 (goal_vel=4.0 + goal_prog=5.0) 遠強於安全側 (ss=0.8, ds=0.0)，
reward 結構偏向 aggressive progress，agent 不學繞行。

### v5 改動（最小可驗證版本）

**collision_ground**: -50 → **-100** (per-trigger: -10 → -20)

**goal_first_v3**: 只改 Stage 3-6 的 static_safety，其餘與 v2 相同

| Stage | ss (v2/v4) | ss (v3/v5) | 變化 |
|-------|:---:|:---:|:---:|
| 1-2 | 0.2 | 0.2 | 不動 |
| 3 | 0.4 | **0.8** | ×2 |
| 4 | 0.8 | **1.4** | ×1.75 |
| 5 | 1.2 | **1.8** | ×1.5 |
| 6 | 1.0 | **1.6** | ×1.6 |
| 7-12 | 原值 | 原值 | 不動 |

### 驗證目標

- CR 降低 (24% → <15%)
- path efficiency 降低 (94% → <90%，推論: 若 agent 開始繞行)
- SR 突破 80% 升到 Stage 5
- speed_mean 不大幅下降 (>0.4 m/s)
