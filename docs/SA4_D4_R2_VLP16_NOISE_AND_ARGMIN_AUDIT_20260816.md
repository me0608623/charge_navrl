# SA4-D4-r2：VLP-16 mixed-pixel 與 winner-ray 角度稽核

日期：2026-08-16  
狀態：固定 GPU 比較完成；未通過 SA4 Gate  
範圍：診斷證據，不啟動訓練、不啟動 SA5

## 1. 結論摘要

1. 現行 `0.2515% × 5,760 rays` 的注入**不等同於原始實測指標**。
2. `0.2515%` 的量測定義是白牆窄 ROI 中，既有回波被 MAD 判為離群值的平均比例；它不是全掃描每條 ray 生成近距假點的機率。
3. 現行 simulator 對每條 raw ray 獨立抽樣，命中後把距離改成 `U(0.2, 2.0) m`，再由 72-bin `amin` 取最小值。這會把很小的 per-ray 機率放大成大量近距 winner。
4. 本次 D4-r2 **不修改 noise preset**，只修正安全檢查器的 ray 角度，避免一次改兩個因素而失去可歸因性。
5. noise model 需另立 protocol；目前保存的 aggregate JSON 不足以推回 mixed-pixel 的距離偏差分布與 edge-conditioned 生成規則。
6. 固定比較中，actual-angle feasible fraction 為 `17.49%`，legacy center-angle 為 `17.89%`；paired net recovery 是 `-0.40 pp`（`-98` frames），未達預先固定的 clear-recovery 規則。
7. baseline 對 argmin 的 total CR 為 `16.95% → 16.64%`，但 obstacle CR 為 `16.45% → 16.44%`，實質不變；差異主要來自 wall CR `0.51% → 0.20%`。
8. 因此 5°中心角不是大量 no-feasible 的主因，c6400 仍不可作為 SA4 parent，SA5 未啟動。

## 2. 原始量測實際算了什麼

量測程式先做 intensity、height、distance 與方位角 ROI 過濾，再對 ROI 中既有點做 MAD outlier removal：

```text
每幀 mixed_pixel_rate_f
  = outliers_removed_f / in_roi_f

檔案內 mixed_pixel_rate
  = mean_f(mixed_pixel_rate_f)

六個距離的彙整值
  = mean_file(mixed_pixel_rate_file)
```

權威程式位置：

- `/home/aa/vlp16_measurement/measurement_node.py:248-290`
- `/home/aa/vlp16_measurement/measurement_node.py:371-389`
- `/home/aa/vlp16_measurement/analyze_gap.py:302-329`
- `/home/aa/vlp16_measurement/analyze_gap.py:567-608`

因此此指標的條件集合是：

```text
已有回波 ∩ intensity 過濾後 ∩ 高度/距離/±方位角 ROI 內
```

它沒有量到：

- no-return ray 變成近距假回波的機率；
- 全 360° 場景的 uniform phantom rate；
- 離群值應被改成哪個距離；
- 離群值是否只出現在物體邊界；
- 各 ray、ring、距離與材質間的條件分布。

`vlp16_noise/isaac_lab_noise_params.py:31-33` 保存的固定值為 `0.002515`。目前可找到的六個 guided white-wall JSON，其 unweighted mean 是 `0.002373665`，以 ROI 點數加權則是 `0.002839469`；兩者都不等於 `0.002515`。因此 2026-07-01 FIXED 檔的精確六檔輸入目前無法由現存 JSON 完整重建，不能把小數第六位視為已重新驗證。

## 3. Simulator 現在如何使用它

`charge_env_overrides.py` 把 `0.002515` 直接設為 `distractor_rate`：

- `scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py:626`
- `scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py:662-664`

觀測函式對每條 raw ray 做獨立 Bernoulli 抽樣，命中後把距離改成均勻近距值：

```text
Bᵢ ~ Bernoulli(0.002515)

若 Bᵢ = 1：rᵢ ← U(0.2, 2.0) m

72-bin[j] = min(rᵢ | ray i 落在 bin j)
```

實作位置：

- `source/.../observations/obs_functions.py:682-710`

5,760 rays 下的數量級：

```text
E[ghost rays/frame] = 5,760 × 0.002515 = 14.4864

每個 5° bin 約 80 rays
P(bin 至少一個 ghost) = 1 - (1 - 0.002515)^80 ≈ 18.2458%

E[可能含 ghost 的 bins/frame] ≈ 72 × 18.2458% = 13.137
```

由於 ghost 被限制在 0.2–2.0 m，對大多數較遠真實表面會贏得 `amin`。所以「0.2515% 很小」不代表 72-bin 輸出只受 0.2515% 影響。

## 4. 與真實 VLP-16 的 ray 數關係

VLP-16 的硬體輸出是旋轉掃描與 UDP return stream，不是固定 `16 × 360 = 5,760` 點的物理 frame。官方手冊描述每個 laser 約每秒發射 18,000 次，整機約 300,000 points/s；點數還會隨 rotation rate 與 return mode 改變。Isaac Lab 的 5,760 rays 是本專案的 `16 rings × 360 horizontal samples` 幾何離散化，不是硬體固定的一圈點數。

官方資料：<https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf>

若某比例真的是「每個有效 return 的獨立事件率」，可按 simulated return 數抽樣；但本次 `0.002515` 的條件不是全體 return，而且現行程式也會對原本無有效表面的 ray 生成近距值。因此目前不能稱為 faithful VLP-16 mixed-pixel model。

## 5. D4-r2 角度修正

舊 D4 把每個 72-bin 距離放在固定 5°區間中心。實際 72-bin 是由 5,760 raw rays 做 `amin`，winner 可位於 bin 中任何角度。

修正版新增：

- mode：`geometry_feasible_argmin`
- protocol：`sa4_d4_geometry_selector_argmin/v1`
- SHA-256：`aca15fbdc2a2a7545dbc3d4b6dfba86549e1fac9e39ae0e4c24549abe6facdcd`
- freeze：`docs/freeze/sa4_d4_geometry_selector_argmin_v1.json`

資料流：

```text
5,760 raw rays
→ 72-bin amin + winner raw-ray index/emitted-ray angle trace
→ 與 policy 當前 72-bin LiDAR 做 bitwise match
→ winner_actual_angle_rad（由 RayCaster direction 取得）
→ static point (r cos θ, r sin θ)
→ 19×19 geometry feasibility
→ optional action override
→ process_actions/decode
→ d1 queue
→ simulator
```

沒有 bitwise-matching trace、shape 不符、角度非有限值或超出 `[-π,π]` 時一律停止，不回退到 bin center。

為直接回答「安全候選是否恢復」，argmin arm 在每個 active frame 另算一份 legacy center-angle shadow。只有 argmin grid 能修改 action；帳本分為：

- both feasible；
- argmin only feasible；
- center only feasible；
- both infeasible。

四者必須精確加總為 active frames。

## 6. 固定重跑協定

條件完全沿用 D4：

```text
checkpoint  SA4-R2 c6400
SHA-256     c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197
stage       4
scenario    lateral corridor
seed        818
actuator    fixed d1 = 200 ms
budget      64 env × 2,500 steps per arm
arms        fresh baseline → geometry_feasible_argmin
noise       unchanged full preset
training    none
SA5         forbidden
```

預先固定的描述性「明顯恢復」規則：

```text
argmin-only - center-only ≥ 100 active frames
且 paired net feasible gain ≥ 5 percentage points
```

這不是統計顯著性檢定；單 checkpoint、單 evaluator seed 仍只算 diagnostic evidence。

## 7. 執行與 fail-closed 紀錄

2026-08-17 第一次正式啟動時，fresh baseline 完成，但 argmin arm 在 step 0 停止。原因是 trace 用 hit point 反推角度；no-hit ray 若被 mixed-pixel 注入成近距 winner，會產生 `atan2(inf, inf) = NaN`。

修正後改由 RayCaster 的 `_ray_directions_w` 取得 emitted-ray direction，再轉到 robot-yaw frame。這不改 policy observation、noise draw、binning 或 selector 門檻。失敗批次完整保留於：

`logs/gates/sa4_d4/argmin_angle_r2_20260816/`

其 `INCOMPLETE_RUNTIME_ERROR.md` 明確標為 `INCOMPLETE_INVALID_NO_COMPARISON`，不得與 retry 合併。

修正後驗證：

- 4 env × 100 step GPU smoke：400 records、2 episodes、`delay_errors=0`、`reconciliation=True`
- 全 `rnn_car_wdclean`：856 passed，27 subtests passed
- Python compile / `git diff --check`：通過
- 正式兩臂：無 NaN、OOM、traceback；source fingerprint 各臂前後一致
- protocol SHA-256：`aca15fbdc2a2a7545dbc3d4b6dfba86549e1fac9e39ae0e4c24549abe6facdcd`

有效輸出：

`logs/gates/sa4_d4/argmin_angle_r2_20260817/`

## 8. 固定比較結果

| Arm | episodes | SR | obstacle CR | wall CR | total CR | TO |
|---|---:|---:|---:|---:|---:|---:|
| fresh baseline | 1,976 | 83.05% | 16.45% | 0.51% | 16.95% | 0.00% |
| actual winner-angle | 1,989 | 83.36% | 16.44% | 0.20% | 16.64% | 0.00% |

事件計數為：

```text
baseline  success=1641  obstacle_collision=325  wall_collision=10
argmin    success=1658  obstacle_collision=327  wall_collision=4
```

total CR 的描述性差值為 `-0.31 pp`，但 obstacle CR 只差 `-0.007 pp`；不能把單 checkpoint、單 evaluator seed 的點估計宣稱為已證實改善。

同一批 `24,665` active frames 的 paired geometry 帳本：

| 分類 | frames | active share |
|---|---:|---:|
| both feasible | 3,846 | 15.59% |
| actual-angle only | 469 | 1.90% |
| center-angle only | 567 | 2.30% |
| both no-feasible | 19,783 | 80.21% |
| actual-angle feasible（合計） | 4,315 | 17.49% |
| center-angle feasible（合計） | 4,413 | 17.89% |

因此 paired net recovery = `469 - 567 = -98 frames = -0.40 pp`，`clear_candidate_recovery=False`。actual-angle 下 no-feasible 仍為 `20,350 / 24,665 = 82.51%`。

## 9. 裁決與下一步

1. 5°區間中心近似不是 D4 no-feasible 的主要來源；停止沿「再修角度」方向調參。
2. 這次沒有支持 c6400 成為 SA4 parent；SR 仍低於 90%，CR 仍高於 10%。SA4 維持 HOLD，SA5 不啟動。
3. 不應因本結果下調 0.10 m clearance 或縮短 2.4 s horizon；那會改變安全定義，且本輪沒有識別它們是主因。
4. 下一個獨立問題是 VLP-16 distractor/mixed-pixel 生成契約。必須另立 frozen protocol，不能把 noise 修正混寫成 D4-r2 的一部分。
5. 若能取得原始量測，先估計 eligible-return denominator、距離殘差、edge/ring/range/material 條件與 no-return transition；若資料不足，只能做 sensitivity ablation，不可宣稱 faithful sensor model。
