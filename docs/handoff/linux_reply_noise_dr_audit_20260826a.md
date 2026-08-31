---
title: Linux 端回覆 — 光達雜訊 / 物理 DR / b_linear 定案 + 零重訓對照實驗
from: Linux 端（PC-B, /home/aa/IsaacLab）
to: Cowork（Windows 端）
date: 2026-08-26
replies_to: review/20_給Linux端_三項數值定案請求_20260826.md
稽核原則: 一律以執行期日誌為準；不由設定檔欄位名稱推論；無佐證標「無法確認」
---

# 回覆總表

| 請求 | 狀態 |
|---|---|
| 一、光達雜訊實際注入值 | ✅ 十格全填，另找到「20 mm」的精確出處 |
| 二、物理 DR 事件設定 | ✅ 有 runtime manager 表；4 項標「無法確認」並附原因 |
| 三、b_linear(d) | ✅ 一句話定案 |
| 四、零重訓對照實驗 | ✅ 可行、零改碼、約 4–5 小時 |

**證據等級**（每格都標，不同等級不可混寫成同一句）：

| 級 | 意義 |
|---|---|
| `L` | 執行期 console 原文 / Isaac Lab manager 自印表格 |
| `M` | 執行期寫出的中繼檔（`run_metadata.yaml`、`docs/freeze/*.json`） |
| `T` | 執行期張量追蹤，且合約要求與 policy 觀測**逐位元相同** |
| `K` | 程式常數（執行期只印開／關，**不印數值**） |
| `U` | **無法確認** |

> ⚠️ `T` 與 `K` 的差別是本次稽核的關鍵。
> 執行期日誌只印 `σ=ON`，**沒有印 8.672 這個數字**。
> 數字能升到 `T`，是因為 D7/D8 的追蹤合約強制它與 policy 觀測 bitwise 相同（見一之 4）。
> 這一點請寫進論文的可重現性敘述，不要寫成「日誌記錄了 8.672」。

---

# 一、光達雜訊的實際注入值

## 1. 執行期日誌原文（依要求貼原文，未摘要）

第一至第四階段主線，五支訓練 run 全部一致：

```
logs/sa1_sim2real_v1_ne1024_s42_r1.log:91
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)

logs/rnn_car/sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1.console.log:91
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)

logs/rnn_car/sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1.console.log:91
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)

logs/rnn_car/sa4_sim2real_v2_from_sa3r1_c100_ne1024_s42_p100_r1.console.log:74
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)

logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1.console.log:75-76   ← 第四階段 it125
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)
[SIM2REAL][mixed-pixel-eligibility] mode=valid_return_only groups=critic.lidar_static,policy.lidar_static
```

banner 尾巴 **`(fixed measured values, no DR)`** 這句可以直接引用 ——
它就是「實測固定值、無逐回合隨機化」的執行期自證。

`run_metadata.yaml`（`M` 級，執行期寫出）：

```yaml
fixed_recipe:
  vlp16_noise_mode: full
  lidar_distractor_eligibility: valid_return_only
  actuator_delay_range_steps: [0, 2]
```

## 2. 你要的十格表

`vlp16_noise_mode` 一旦有值，**細粒度 YAML 欄位全部被忽略**（preset 分支 early-return），
所以下表就是訓練期實際寫進 ObsTerm 的值。

| 參數 | 實際值 | 啟用？ | 級 |
|---|---|---|---|
| `displacement_std_soft`（σ_soft） | **0.008672 m ＝ 8.672 mm** | ✅ 啟用 | `T` |
| `distance_bias_k`（b_linear 斜率 k_b） | **0.0** | ❌ 未啟用 | `K` |
| `distance_bias_b`（b_linear 截距 b_0） | **0.0** | ❌ 未啟用 | `K` |
| `per_ring_bias` | **True**（16 值 ∈ [−0.476, +23.619] mm，均值 +12.945） | ✅ 啟用 | `T` |
| `per_ring_bias_scale` | **1.0** | ✅（等比例，未縮放） | `K` |
| `hole_rate`（p_hole） | **0.194859 ＝ 19.4859 %** | ✅ 啟用 | `T` |
| `block_dropout_prob`（L2 扇區遮擋） | **0.0** | ❌ 未啟用 | `K` |
| `human_dynamic_dropout` | **False** | ❌ 未啟用 | `K`+`L`（banner `human_dyn=off`） |
| 雜訊模式 | **`full`**（＝σ + bias + dropout，不含 human 材質層） | ✅ | `L`+`M` |
| L3 逐回合採樣 `*_dr_min` / `*_dr_max` | **全部 0.0**（preset 迴圈逐鍵清空） | ❌ 未啟用 | `K` |

**表外但同組必須一起報的兩個值**（否則 dropout 那列不完整）：

| 參數 | 實際值 | 級 |
|---|---|---|
| `distractor_rate`（混合像素／鬼影） | **0.002515 ＝ 0.2515 %** | `T` |
| `distractor_range`（取代值分布） | **U(0.2, 2.0) m** | `T` |

**幾何與歸一化**（§2.4.3 小表若要自足，建議一併列）：

| 項目 | 值 | 級 |
|---|---|---|
| `r_min` / `r_max` / `z_filter` | **0.5 m** / 20.0 m / 0.5 m | `K` / `T` / `K` |
| 射線數 / bin 數 / 每 bin 射線數 | **5760 / 72 / 80** | `T` |
| 聚合方式 | 每 bin 取**最小值**（`amin`） | `K` |
| 歸一化 | `clamp(d − 0.35, 0, 20) / 20`，再 clip 到 [0,1] | `T` |

> ⚠️ `r_min = 0.5 m`，不是 0.25 m。這條血緣走 LV-DOT 對齊碰撞邊界的設定，
> 訓練端**沒有**任何覆寫 `r_min` 的路徑（`charge_env_overrides.py` 無此鍵）。
> 你說論文只寫「硬截斷」沒給數值 —— 那就維持不給，或補 0.5 m，兩者都安全；
> 唯獨**不能寫 0.25**。

## 3. 最終送進策略的每條射線標準差 —— 一個數字

> **σ_total = σ_soft = 8.672 mm**（σ_hard ≡ 0，與距離無關）。
> 歸一化後 = 8.672 mm ÷ 20 m = **4.336 × 10⁻⁴**。

但請務必在同一段補一句，否則會被審稿抓「注入層與觀測層混用」：

```
σ_ray = 8.672 mm                      ← 注入層（本表第 1 列）
σ_bin ≈ 0.97 mm，另含偏移 ≈ −18 mm     ← 策略實際看到的
```

因為每個 5° bin 是 **80 條射線取最小值**。M 條 i.i.d. 高斯取 min：

```
E[min εᵢ] ≈ −σ · Φ⁻¹(1 / (M+1))     M = 80 → ≈ −2.1σ ≈ −18 mm
sd[min εᵢ] ≈ σ / √M                  ≈ 0.97 mm
```

符號：σ = 單條射線測距高斯標準差；M = 每 bin 射線數；Φ⁻¹ = 標準常態分位數函數。

（此式與論文 §1.4 C2 已有的 min-pool 推導一致，可直接互相引用。）

## 4. 為什麼 8.672 可以標 `T` 而不是 `K`

`docs/freeze/sa4_d7_lidar_residual_origin_v2.json` 的 `trace_rule` 原文：

```
"record tensors already realized by wd_like_sweep_72; never draw additional random
 numbers; select the trace whose normalized 72-bin sweep is bitwise identical to
 policy obs columns [6:78]"
```

同檔 `trace_spec`：

```json
{"exact_policy_trace_match": true,
 "expected_sigma_m": 0.008672,  "expected_hole_rate": 0.194859,
 "expected_distractor_rate": 0.002515, "expected_num_rays": 5760,
 "expected_num_bins": 72, "expected_r_max_m": 20.0, "expected_r_robot_m": 0.35,
 "require_per_ring_bias": true, "require_block_dropout_prob": 0.0,
 "require_human_dynamic_dropout": false}
```

亦即：這些數值不是「從 config 讀來的」，而是**與 policy 實際吃到的觀測逐位元比對過**。
`sa4_d8_noise_eligibility_sensitivity_v1.json` 有同一組 `noise_trace_spec`，互為佐證。

## 5. 你要的 `_apply_lidar_noise_config()` 原始碼

**完整檔已複製到** `Linux端往返/src/charge_env_overrides.py`（含全部相關函式）。

實際決定一切的是 preset 分支。`_apply_lidar_noise_config()` 開頭四行就轉走了：

```python
def _apply_lidar_noise_config(env_cfg, args_cli):
    mode = getattr(args_cli, "vlp16_noise_mode", None)
    if mode:
        _apply_vlp16_ablation_mode(env_cfg, mode)
        return                      # ← 細粒度 YAML 欄位全部不再讀取
    ...
```

`_apply_vlp16_ablation_mode()` 全文（`charge_env_overrides.py`，本回覆的權威依據）：

```python
def _apply_vlp16_ablation_mode(env_cfg, mode: str):
    mode = str(mode).lower()
    if mode not in _VLP16_NOISE_MODES:
        raise ValueError(...)
    # full_material = full + human noise on dynamic-obstacle rays
    on_sigma = mode in ("sigma", "full", "full_material")
    on_bias  = mode in ("bias",  "full", "full_material")
    on_drop  = mode in ("dropout", "full", "full_material")
    human    = mode == "full_material"
    for _gn, _tn, term in _find_lidar_obs_terms(env_cfg):
        p = term.params
        # 固定 σ 進 soft 槽；殺掉距離縮放路徑
        p["displacement_std_per_meter"] = 0.0
        p["displacement_std"]           = 0.0
        p["displacement_std_soft"]      = _VLP16_SIGMA_FIXED_M if on_sigma else 0.0
        # dropout 家族
        p["hole_rate"]       = _VLP16_HOLE_RATE       if on_drop else 0.0
        p["distractor_rate"] = _VLP16_DISTRACTOR_RATE if on_drop else 0.0
        # 系統偏差 = 實測 per-ring 陣列（已含 common-mode）；不另加全域 bias
        p["per_ring_bias"]    = on_bias
        p["distance_bias_k"]  = 0.0
        p["distance_bias_b"]  = 0.0
        # faithful/fixed → 清空每一個 per-episode DR 區間鍵
        for k in list(p.keys()):
            if k.endswith("_dr_min") or k.endswith("_dr_max"):
                p[k] = 0.0
        # 實測模型沒有 L2 blanket、沒有 block dropout
        p["block_dropout_prob"] = 0.0
        p["human_dynamic_dropout"] = human
        if human: ...
        if hasattr(term, "noise"):
            term.noise = None
    print(f"[SIM2REAL][VLP16-ablation] mode={mode} σ={...} bias={...} dropout={...} "
          f"human_dyn={...} (fixed measured values, no DR)")
```

常數定義（同檔）：

```python
_VLP16_SIGMA_FIXED_M   = 0.008672   # measured range σ (point-to-plane residual std)
_VLP16_HOLE_RATE       = 0.194859   # measured dropout (intensity < thr)
_VLP16_DISTRACTOR_RATE = 0.002515   # measured mixed-pixel / edge-outlier rate
```

注意上面第 12–14 行的註解已寫明**為什麼不加全域 bias**：per-ring 陣列已攜帶
common-mode（約 +1.3–1.5 cm），再加一次會重複計數。這句可直接翻進 §2.2.1。

## 6. ⭐「20 mm」的精確出處 —— 本次最重要的發現

表 2.3 的「σ 可達約 20 mm」不是模擬器的一般預設，也不是無主的數字。它是
**另一個舊觀測函式 `lidar_vlp16_to_2d_bins` 的 docstring 預設值**：

```
source/.../mdp/observations/obs_functions.py:18-21
    感測器噪聲 (域隨機化):
        - displacement_std=0.02: 距離高斯雜訊 ±2cm
        - hole_rate=0.005: 射線丟失 0.5%
        - distractor_rate=0.002: 幽靈點 0.2%
```

三件事同時成立：

1. **本論文血緣沒有使用這個函式。** 在 `charge_skrl/cfg/` 全目錄 grep
   `lidar_vlp16_to_2d_bins` **零命中**；實際使用的是 `wd_like_sweep_72`。
2. **這組預設被明確記錄為 bug。** `train/train_charge_ac.py:918`：

   ```
   # 修復 Bug B: lidar_vlp16_to_2d_bins 預設帶 displacement_std=0.02 + hole_rate=0.005
   # + distractor_rate=0.002 + Unoise(±0.02)，導致 lidar.min 永遠 ≈ 0（因為每幀都有
   # ~11 個假近距離 distractor），policy 學不到正確的近距離 obstacle 訊號。
   ```
3. **20 mm 在現行程式的任何設定下都達不到。** RSS 一般式為
   σ_total(r) = √((k_pm·r)² + σ_soft²)，而全 repo 中 `lidar_displacement_std_per_meter`
   的最大值是 `0.00060 /m`（`wd_sa6_v2.yaml`），在 r = 20 m 時距離項僅 12 mm，
   RSS = √(12² + 8.672²) ≈ **14.8 mm**。且 preset 已把 k_pm 強制歸零。

> **這一點請務必寫進你的 §2.4.3 小表註腳。**
> 它把「表 2.3 的數字錯了」升級成「表 2.3 引用了一個已知有害、且本工作未使用的舊預設」——
> 後者是可辯護的敘述，前者只是認錯。

## 7. 三種說法的裁決

| 出處 | 說法 | 裁決 |
|---|---|---|
| 表 2.3 註 | RSS 合成「可達約 20 mm，確保涵蓋真實雜訊」 | ❌ **刪除**。出處見一之 6 |
| §2.4.3 | 「退化為 σ_total = σ_soft」 | ✅ **正確，且是唯一成立的敘述** |
| 表 3.23 | 「固定 8.672 mm，啟用」 | ✅ **正確** |

你原本的處理方向（§2.4.3 建權威小表、刪表 2.3 那句涵蓋性宣稱）我認為正確，
且與 §2.4.3「逐回合擾動反而會偏離量測分布」的論證自洽 ——
因為 `no DR` 這件事現在有 banner 原文可引。

---

# 二、物理域隨機化的實際事件設定

## 1. 訓練期實際註冊的 EventTerm 完整清單（`L` 級，Isaac Lab 自印）

`logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1.console.log:116`，原文：

```
[INFO] Event Manager:  <EventManager> contains 3 active terms.
+-----------------------------------------+
|  Active Event Terms in Mode: 'startup'  |
+--------+--------------------------------+
| Index  | Name                           |
+--------+--------------------------------+
|   0    | init_walls                     |
|   1    | randomize_obstacles_startup    |
+--------+--------------------------------+
+--------------------------------------+
| Active Event Terms in Mode: 'reset'  |
+--------+-----------------------------+
|   0    | randomize_wall_positions    |
|   1    | randomize_obstacles         |
|   2    | reset_base                  |
|   3    | previous_stage_replay       |
|   4    | corridor_crossing           |
|   5    | long_corridor_replay        |
|   6    | narrow_passage_bridge       |
|   7    | domain_randomization        |
+--------+-----------------------------+
+----------------------------------------------------------------+
|             Active Event Terms in Mode: 'interval'             |
+-------+------------------------------+-------------------------+
| Index | Name                         | Interval time range (s) |
+-------+------------------------------+-------------------------+
|   0   | move_dynamic_obstacles       |        (0.2, 0.2)       |
|   1   | move_goal                    |        (0.2, 0.2)       |
|   2   | maintain_narrow_passage_goal |        (0.2, 0.2)       |
|   3   | maintain_long_corridor_goal  |        (0.2, 0.2)       |
+-------+------------------------------+-------------------------+
```

**這張表是 Isaac Lab 在建環境時自己列印的實際註冊清單**，
因此可以繞過「YAML 欄位存在但未被讀取」的整類問題 —— 沒列在上面的就是沒註冊。

物理類只有 `domain_randomization` 一項（reset 模式，index 7）。

## 2. 該 term 的 `func` / `mode` / `params` 實際值

註冊處（`cfg/charge_env_cfg_vlp16_curriculum.py:443`，
由 `EventCfgVLP16TCorridor` 繼承，wd_sparse 未覆寫）：

```python
domain_randomization = EventTerm(
    func=apply_domain_randomization,
    mode="reset",
    params={"enable_physics": True, "enable_sensor_noise": True,
            "enable_external_force": True},
)
```

**params 只有三個布林，所有數值一律走函式簽章的預設值**
（`domain_randomization/dr_events.py:apply_domain_randomization`）：

```python
init_lin_vel_range: float = 0.5          # x: ±0.5 m/s；y 另乘 0.3 → ±0.15 m/s
init_ang_vel_range: float = 0.5          # yaw ±0.5 rad/s
push_force_range: tuple = (5.0, 20.0)
push_env_ratio: float = 0.10
mass_scale: tuple = (0.85, 1.15)
friction_scale: tuple = (0.7, 1.3)
com_offset: float = 0.05
wind_force_range: tuple = (0.0, 3.0)
enable_actuator_dr: bool = False         # ← 未傳入 → 維持 False
```

**且訓練期沒有任何 YAML 覆寫過這些數值。** 佐證是**一行 banner 的缺席**：
`_apply_dr_param_overrides()` 只要有任一項被設定就會印

```
[SIM2REAL] DR overrides: mass=... friction=... com=... wind=... push=...@... actuator=...
```

五支訓練 log 全部**只有兩行 `[SIM2REAL]`**（見一之 1），沒有這一行。
→ `any_set = False` 且 `enable_actuator_dr = False`。

## 3. 逐項是否真的執行

| 子項 | 數值 | 是否執行 | 對軌跡有無影響 | 級 |
|---|---|---|---|---|
| 初始線速度 | x ±0.5 m/s、y ±0.15 m/s | ✅ **必定執行**（在 try 之外） | 見下 3.2 | `K`+`L` |
| 初始角速度 | yaw ±0.5 rad/s | ✅ **必定執行** | 見下 3.2 | `K`+`L` |
| 質量縮放 | (0.85, 1.15) | ⚠️ 呼叫了，但包在靜默 except 內 | **無**（見 3.1） | `U` |
| 地面摩擦 | (0.7, 1.3) | ⚠️ 同上 | **無** | `U` |
| 質心偏移 | ±0.05 m | ⚠️ 同上 | **無** | `U` |
| 持續風力 | (0.0, 3.0) N | ⚠️ 同上 | **無** | `U` |
| 隨機衝擊力 | (5.0, 20.0) N @ 10 % env | ❌ **從未執行** | 無 | `L` |
| 致動器 DR | — | ❌ `enable_actuator_dr = False` | — | `L` |

**衝擊力為何是「從未執行」而非「無法確認」**：`apply_random_push()` 只被
`domain_randomization_pre_step()` 呼叫，而該函式**沒有出現在上面 manager 表格的任何一個 mode**——
它從未被註冊成 EventTerm。缺席於自印表格是直接的執行期證據。

**四項為何是「無法確認」**：它們被包在

```python
# domain_randomization/dr_events.py
if enable_physics:
    try:
        from .physics_dr import randomize_robot_mass, randomize_ground_friction, randomize_com_offset
        randomize_robot_mass(env, env_ids, mass_scale=mass_scale)
        randomize_ground_friction(env, env_ids, friction_scale=friction_scale)
        randomize_com_offset(env, env_ids, com_offset_range=com_offset)
    except Exception:
        pass  # API 不可用時靜默跳過
```

成功或失敗都不留任何痕跡。這是 §3.8.3 那一類陷阱的變種 ——
**這次不是「旗標沒被讀」，是「函式被呼叫但整段可能被吞掉」**。

### 3.1 為什麼即使成功也不影響軌跡

`mdp/actions/discrete_differential_drive.py:367`：

```python
self._asset.write_root_velocity_to_sim(root_velocity)   # 6D 全寫
```

`apply_actions()` 由 ActionManager 在**每個 physics substep 之前**呼叫，
且 6 個分量全部被指定（vy = vz = wx = wy = 0；vx、wz 來自動作解碼）。
機器人的根速度是**被規定的**，不是被力積分出來的 —— 沒有馬達模型、沒有輪地接觸動力學。
因此質量、摩擦、質心、風力都無法改變軌跡。

> 這句可以直接寫進論文，而且它同時解釋了為什麼本工作的 sim-to-real 主戰場
> 是**感測層與延遲**，不是動力學 —— 是設計取捨，不是疏漏。

### 3.2 初始速度隨機化唯一的生效路徑

reset 時寫入的隨機初速，會在下一個 substep 被 `write_root_velocity_to_sim` 覆寫，
**所以它不會被積分**。但它有一條真實的生效路徑：

```python
# mdp/observations/functions.py:502-503
asset: Articulation = env.scene[asset_cfg.name]
vel_w = torch.nan_to_num(asset.data.root_lin_vel_w[:, :3], ...)
```

`normalized_linear_velocity()` 讀的是模擬器根速度，
所以 **reset 之後計算的那一幀觀測確實帶著隨機初速**。

→ 建議論文把它描述為「**觀測層的初始條件擾動**」，而非動力學隨機化。
這個區分要寫出來，否則「域隨機化」四個字會被審稿當成動力學 DR 來檢驗。

## 4. 表 2.5 與 §2.4.1 是並存還是取代 —— 你問的第三點

**都不是。正確關係是：表 2.5 前兩列是真的，其餘全部要重寫。**

| 出處 | 內容 | 裁決 |
|---|---|---|
| 表 2.5 初始線速度 | U(−0.5, 0.5) m/s | ✅ 值正確（y 軸另乘 0.3，若表要精確可註明 ±0.15） |
| 表 2.5 初始角速度 | U(−0.5, 0.5) rad/s | ✅ 值正確 |
| 表 2.5 外力擾動（10 % 環境） | — | ❌ **從未執行**（見 3） |
| §2.4.1 質量 ±15 % | [0.85, 1.15] | `U`＋結構上無效果 |
| §2.4.1 摩擦 [0.7, 1.3] | | `U`＋結構上無效果 |
| §2.4.1 質心 ±0.05 m | | `U`＋結構上無效果 |
| §2.4.1 側風 [0, 3] N | | `U`＋結構上無效果 |
| §2.4.1 衝擊力 [5, 20] N | 每控制步 10 % | ❌ **從未執行** |

§2.4.1 那四項數值**確實存在於程式的函式預設值中**（所以文字不是憑空捏造），
但它們沒有執行期佐證，而且即使執行也不影響軌跡。

**建議改寫方向**：把 §2.4.1 改成
「本工作的域隨機化集中於感測層（TLNI）與致動延遲；
因採用運動學速度指令介面（每 physics step 直接寫入根速度），
動力學參數隨機化在此架構下不產生效果，故未納入。
物理層僅保留 reset 時的初始速度擾動，作用於觀測初始條件。」

表 2.5 則加一欄「本版本啟用狀態」，三列分別填
✅ 啟用（觀測層）／✅ 啟用（觀測層）／❌ 未執行。

## 5. 關於「§3.8.3 已認定旗標只由評測腳本讀取、訓練期仍執行」

這條我**無法直接確認**，因為我這邊讀不到 §3.8.3 的原文，不知道它指的是哪一個旗標。

我能確認的是**相反方向的一件事**：本回覆涉及的 LiDAR 雜訊旗標
（`vlp16_noise_mode`、`lidar_distractor_eligibility`）**確實在訓練路徑上被讀取**——
`apply_ablation_overrides()` 同時被訓練入口與評測入口呼叫，
而訓練 log 有對應 banner（一之 1）就是證據。

→ 請把 §3.8.3 指涉的**旗標名稱**告訴我，我再逐一比對訓練路徑。
在那之前，這一項在本回覆中標 `U`。

---

# 三、b_linear(d) 到底有沒有啟用

## 1. 訓練期 `distance_bias_k` / `distance_bias_b` 的實際值

**兩者皆為 0.0**（`K` 級）。

`_apply_vlp16_ablation_mode()` 對每個 LiDAR ObsTerm 無條件執行：

```python
p["distance_bias_k"] = 0.0
p["distance_bias_b"] = 0.0
```

而 `wd_like_sweep_72()` 中距離偏差區塊的進入條件是

```python
has_k_dr = distance_bias_k_dr_min != 0.0 or distance_bias_k_dr_max != 0.0
has_b_dr = distance_bias_b_dr_min != 0.0 or distance_bias_b_dr_max != 0.0
has_scalar_bias = distance_bias_k != 0.0 or distance_bias_b != 0.0
if has_k_dr or has_b_dr or has_scalar_bias:
    ...   # ← 三者皆 False，整段不執行
```

三個條件同時為 False（DR 區間也被 preset 清空），**整段程式不執行**。

> 補充，避免你日後看到別處的數字混淆：程式裡確實存在一組
> `distance_bias_k = 0.021` / `distance_bias_b = -0.030`，
> 但它在 `_apply_lidar_noise_config()` 的 **legacy fallback 分支**，
> 需要 `lidar_no_noise=True` 或 `lidar_distance_bias=True` 且無 DR 區間才走得到，
> **本血緣走不到**。該分支的程式註解標它為「legacy linear fit（R²=0.58，已知殘差有結構）」。
> 這就是第三個 R² 的來源。你說論文只用 0.08 與 0.006、沒出現 0.58 —— 正確，維持即可。

## 2. (2.10) 式應保留還是刪除

**建議保留，但改標為「量測期評估過的候選模型，本版本未採用」**，理由有二：

1. §2.2.1 的論證是「擬合 R² = 0.006 → 拒絕距離相依模型」。
   要讓讀者看懂「拒絕了什麼」，被拒絕的那個式子必須在場。刪掉會讓 R² 判定失去對象。
2. 量測端檔案 `vlp16_noise/isaac_lab_noise_params.py` 也是這個結構 ——
   它同時保留 `LIDAR_BIAS_SLOPE = 0.002153` / `LIDAR_BIAS_INTERCEPT = 0.011045`
   與 `LIDAR_BIAS_TYPE = "fixed"`，用 TYPE 欄位表明「候選存在但未選用」。
   論文照同樣邏輯處理最一致。

若版面吃緊，次佳做法是把 (2.10) 移到附錄，正文只留一句「線性距離相依模型經擬合後拒絕（R² = 0.006）」。

## 3. `per_ring_bias` 與 `per_ring_bias_scale`

- `per_ring_bias = True`（`T` 級，D7 `require_per_ring_bias: true`）
- `per_ring_bias_scale = 1.0`（`K` 級；preset 不覆寫此鍵，沿用 ObsTerm 預設 1.0，即等比例不縮放）

實際注入的 16 元素陣列（`obs_functions.py`，逐值與 `vlp16_noise/isaac_lab_noise_params.py`
的 `LIDAR_PER_RING_BIAS` 相同）：

```
+0.004389  +0.009268  +0.018511  +0.023619  +0.004456  +0.013587  +0.014694  +0.001912
+0.014077  +0.023182  +0.019564  +0.019096  +0.019010  +0.016875  -0.000476  +0.005355   [m]
```

範圍 **[−0.476, +23.619] mm**、均值 **+12.945 mm**、標準差 7.527 mm。
論文寫的「−0.5 至 +23.6 毫米」與此一致 ✅。

ray → ring 映射為 `ring_id = ray_idx // H`（H = 5760/16 = 360，row-major），
程式註解記錄此處曾修過一個 bug（原為 `% 16`，會把每個 ring 的方位角打散到 16 個偏移上）。
若 §2.2 要談實作細節，這是一個值得寫的點。

## 4. 一句話定案

> **本版本實際注入的系統性偏差只有「逐環」一種：16 通道固定偏移，
> 範圍 [−0.476, +23.619] mm、均值 +12.945 mm、scale = 1.0；
> 線性距離相依偏差 b_linear(d) 的兩個係數在執行期被強制設為 0，(2.10) 式從未被使用。**

因此三處敘述的處理：

- §1.3／§1.4／§7.1「距離相依偏差未建模」→ ✅ **這句是對的，保留**，
  但**必須補一句**「系統性偏差改以 16 通道逐環實測偏移建模」，
  否則讀者會誤讀成「完全沒有 bias」——那是錯的。
- §2.2.1「不採距離相依模型、改以逐環偏差為主體」→ ✅ 與程式完全一致，保留。
- §3.8.3「σ(d) 刻意關閉，偏差由逐環偏差陣列承載」→ ✅ 一致，保留。

---

# 四、零重訓對照實驗

## 1. 跑得動嗎？ → 可以，且零改碼

`play_eval/play_rnn_car.py:797` 已有

```python
parser.add_argument("--vlp16_noise_mode", type=str, default=None,
                    help="... 要刻意乾淨評估請顯式傳 --vlp16_noise_mode ideal")
```

且第 3179–3185 行會在**未指定時**自動從 checkpoint 繼承訓練時的模式；
顯式傳參即覆寫。**評測腳本可以獨立指定雜訊模式，不動任何訓練設定。**

更好的是：**這個實驗的協定骨架已經存在且被採用過。**
`docs/freeze/sa4_d9_noise_closed_loop_ab_v1.json` 就是同一結構的閉環 A/B ——
固定 cell、雙臂只差一個雜訊性質、fresh process、runtime banner 必須符合所選臂、
明列 `interpretation_limits`、`inferential_claim: false`。
照抄該 schema 換掉操作變因即可，連免責措辭都是現成且已被接受過的。

## 2. 需要多久？

依 it125 三場景實測（`logs/gates/sa4_r3_it125_checkpoint_screen/r1_20260818/`，
log mtime 序列 14:46:59 → 14:54:58 → 14:57:50 推得）：

| 場景 | steps | 每 cell 約 |
|---|---:|---:|
| corridor_lateral | 2500 | 5.3 min |
| corridor_longitudinal | 4000 | 8.0 min |
| nav_native | 1200 | 2.9 min |
| **小計（1 seed × 1 臂）** | | **16.2 min** |

| 規模 | cells | 估時 |
|---|---:|---:|
| 2 臂（ideal / full）× 3 seed | 18 | **≈ 1.6 小時** |
| 5 臂（ideal / sigma / bias / dropout / full）× 3 seed | 45 | **≈ 4–5 小時** |

（含 fresh-process 啟動與指紋檢查；GPU 為 RTX 5090 單卡，期間不可並行訓練。）

## 3. 卡在哪裡？ → 沒有技術阻礙，但有兩個前提

**前提一：受測 checkpoint 用 it125，不要用 it100 或 it75。**
你原信寫「it100 或 it75」，但第五章 SA4 三場景表報的是 **it125**：

```
logs/gates/sa4_r3_it125_checkpoint_screen/r1_20260818/
  sa4_r3_it125_checkpoint_screen_summary.json → verdict.cells
    corridor_lateral       SR 0.8895348837  CR 0.1104651163  n 2236  TO 0
    corridor_longitudinal  SR 0.9698832207  CR 0.0301167793  n 3254  TO 0
    nav_native             SR 0.9393115942  CR 0.0602355072  n 2208  TO 0.000453
```

用 it125 才能讓下面的複現閘生效。

- checkpoint：`logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1/checkpoint_3200.pt`
- SHA-256：`57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d`
- 正式狀態：**SA4 = HOLD，`accepted_parent = null`**（正文必須明寫，見紅線）

**前提二：延遲固定 d1，不要同時掃三種延遲。**
你原信寫「同三種致動延遲」，但那會讓雜訊與延遲兩個因子同時變動，失去單變因性質。
九格改由「3 場景 × 3 evaluator seed」構成，延遲固定 200 ms。
若你要延遲維度，建議另開一組實驗，不要混在同一張表。

## 4. 定案的臂與固定格

依你 8/26 的裁決：

```yaml
arms:
  baseline: ideal        # 零改碼；每個單層臂與 baseline 恰好只差一個因子
  treatment: full
  probe: [sigma, bias, dropout]
  inferential_claim: false
```

| 維度 | 取值 |
|---|---|
| 場景 | `corridor_lateral` / `corridor_longitudinal` / `nav_native` |
| evaluator seed | 818 / 515 / 616（沿用 D9 replication 已用過的 seed 集） |
| 致動延遲 | d1 = 200 ms（1 step），**固定** |
| num_envs | 64 |
| rollout steps | 2500 / 4000 / 1200 |
| 最少完成 episodes | 1000 / 場景 |
| `lidar_distractor_eligibility` | `valid_return_only`（與 it125 訓練一致），**全臂相同** |
| action override / shield | none，全臂相同 |

**你原本的 A 臂（σ = 20 mm、p_hole = 0.5 %、無偏差）應退掉，理由比原先講的更強**：
一之 6 已追出那組數字是 `lidar_vlp16_to_2d_bins` 的 docstring 預設，
且被 `train_charge_ac.py:918` 記錄為 **Bug B**（每幀約 11 個假近距 distractor 把
`lidar.min` 壓到 ≈0）。拿一個**已知有害**的分布當對照臂，
差距會被歸因於 bug 而非雜訊建模。`ideal` 沒有這個問題。

## 5. 預先註冊（依你 8/26 提出的 min-pool 推論）

> **預測：`sigma` 臂相對 `ideal` 的差異接近零；可分辨的差異由 `bias` 與 `dropout` 承載。**
>
> 依據：每 bin 取 80 條射線的 min，σ_ray = 8.672 mm 到 bin 層級被壓到
> sd ≈ σ/√M ≈ 0.97 mm（同時引入 ≈ −18 mm 的系統性偏移）。
> 逐射線抖動幾乎不進入策略輸入；進得去的是偏移與缺值。

**兩種結果都可寫**：

- 成立 → 一條可推廣的發現：「在 min-pool 觀測管線下，逐射線高斯抖動幾乎不進入策略輸入，
  進入的是偏移與缺值」。這比「TLNI 有效」站得住，因為它是機制陳述而非效能宣稱。
- 不成立 → 代表 min-pool 的壓縮效果被低估，該現象本身需要解釋（例如 bin 內射線非同距、
  或 clip 邊界效應）。同樣是可寫的結果。

此段請在**跑之前**凍進 freeze JSON 的 `preregistration` 欄，
否則事後寫等於沒有預先註冊。

## 6. 圖表標題用語（依你指示，不放限制段）

> 本圖量測的是**同一顆已訓練策略在推論期對感測分布的敏感度**，以及 TLNI 三層各自的相對影響；
> **不是**「以 TLNI 訓練優於不以 TLNI 訓練」的比較 —— 後者需要重訓對照臂。

## 7. mixed-pixel eligibility 不同質會不會反噬本實驗？

**不會。反噬的是第五章的跨階段趨勢那條線。** 逐項證據：

**（a）it125 訓練側全鏈同質。** `run_metadata.yaml` 的 `lineage` 欄三段都記了：

```
SA3-R1 c100 ─→ sa4_r3_validreturn_from_sa3r1_c100   it50   valid_return_only
            ─→ sa4_r3_cont50_from_r3c6400           it100  valid_return_only
            ─→ sa4_r3_cont25_from_it100             it125  valid_return_only   ← 受測
```

切換點就是 SA4-R3 的第一支（run name 直接叫 `validreturn`，該分支本來就是為此而開）。

**（b）it125 評測側同樣是 `valid_return_only`。**
`logs/gates/sa4_r3_it125_checkpoint_screen/r1_20260818/*/*.log` 原文：

```
[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON human_dyn=off (fixed measured values, no DR)
[SIM2REAL][mixed-pixel-eligibility] mode=valid_return_only groups=critic.lidar_static,policy.lidar_static
```

→ **訓練與評測同質，兩臂又共用同一設定，本實驗內部效度完好。**

**（c）額外拿到一個免費的複現閘。**
`full` 臂在 seed 818 這一格與表 5.6 的 cell **設定完全相同**，因此它必須重現 3 之前提一的三組數字。
建議寫成 **validity gate 0：若 `full`@818 未重現表 5.6（容差 ±0.5 pp），
先停下來查環境／程式漂移，不要解讀對照結果。**

**（d）真正被反噬的是 §5.3.5 的貫穿趨勢。**
「lateral CR：SA2 2.63 → 6.73 ／ SA3 4.17 → 4.72 ／ SA4 **11.05**」——
前兩階段的 gate 評測是 `all_rays`（07-30 的 log **沒有** eligibility banner），
SA4 是 `valid_return_only`，**訓練側與評測側同時換了規則**。
D8 已量到這個差異會把幾何可行性判讀從 82.73 % 改成 57.78 %。

→ **建議**：第五章已註明「SA1 與 SA2+ 是不同血緣」，請**再補一條**
「SA4 起 mixed-pixel eligibility 由 all_rays 改為 valid_return_only，
故 SA4 的 CR 與 SA2/SA3 非同一量測條件」，或在該趨勢圖把 SA4 那點標為不同條件。

## 8. 紅線（沿用 D9 的 `interpretation_limits`）

✅ 可宣稱：同一權重在不同觀測分布下的表現差異＝**分布敏感度**；逐格判定（pooled 不取代逐格）。

❌ 不可宣稱：
- 「TLNI 讓真機導航變好」—— 沒有真機臂，屬跨域外推
- 「雜訊有害／有益」—— 用 `full` 訓練的策略在 `ideal` 下評測是 **OOD**，
  差異混合了「雜訊本身」與「訓練-評測分布不匹配」，不可分離
- 任何推論統計 —— `inferential_claim: false`，只用描述性規則

---

# 五、附帶：一項不影響論文、但事實表要修的落差

你比對後指出四項中三項沒沾到論文，正確 —— 我抓的是
`Obsidian Vault/論文/00_論文事實表.md`（Citation Gate），不是論文本體。
其中**一項確定要修**：

- 事實表 §1.3 寫 per-ring `β_r ∈ [−22.7, +16.7] mm`
- 實際注入與**論文寫的**都是 `[−0.5, +23.6] mm`

→ **論文是對的，事實表是錯的**，而且論文並未照事實表寫。
由於事實表自訂規則是「寫進論文的任何數字必須出現在本表」，
它現在比論文舊，下一個引用它的人會把錯數字寫回去。**建議只改事實表，論文不動。**

R² 標註那項依你的範圍處理：只確保 0.08 與 0.006 各自標明是**哪一個量**的擬合
（0.078 = σ vs 距離，來自 `LIDAR_NOISE_R2`；0.006 = bias vs 距離），0.58 不進論文。

---

# 六、建議的一行程式改動（下次訓練即生效，本次不動）

目前 banner 只印 `σ=ON bias=ON dropout=ON`，**不印數值**——
本回覆的 8.672 之所以能標 `T` 而非 `K`，完全靠 D7/D8 的 bitwise 追蹤合約撐著。
建議在 `_apply_vlp16_ablation_mode()` 的 print 補上實際數值：

```python
print(f"[SIM2REAL][VLP16-ablation] mode={mode} "
      f"sigma_m={_VLP16_SIGMA_FIXED_M if on_sigma else 0.0} "
      f"hole={_VLP16_HOLE_RATE if on_drop else 0.0} "
      f"distractor={_VLP16_DISTRACTOR_RATE if on_drop else 0.0} "
      f"per_ring_bias={on_bias} scale={...} human_dyn={human} (fixed measured values, no DR)")
```

之後任何一份 log 都能自證，不必再回頭翻凍結產物。
**本次對照實驗前不改**（避免動到評測路徑的原始碼指紋）。

---

# 附件

| 檔案 | 說明 |
|---|---|
| `Linux端往返/src/charge_env_overrides.py` | 你要的 `_apply_lidar_noise_config()` 完整原始碼所在檔 |
| `Linux端往返/src/vlp16_isaac_lab_noise_params.py` | 07-01 白牆實測參數（含 `LIDAR_BIAS_TYPE = "fixed"`） |
| `docs/freeze/sa4_d7_lidar_residual_origin_v2.json` | 追蹤合約（σ/hole/distractor 的 `T` 級來源） |
| `docs/freeze/sa4_d9_noise_closed_loop_ab_v1.json` | 本對照實驗的協定模板 |
| `docs/freeze/sa4_r3_it125_checkpoint_screen_v1.json` | 受測 checkpoint 與固定格定義 |
