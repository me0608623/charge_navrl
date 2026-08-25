---
name: SA4 sim2real v2 pilot 收官與 HOLD
description: SA4 pilot、c50/c100 screen 與 D1-D7/D4-r2 診斷結果；SA4 持續 HOLD
type: project
date: 2026-07-31
updated: 2026-08-17
status: active-hold
---

# SA4 sim2real v2 pilot 收官與 HOLD

> [!important] 2026-08-17 後續裁決
> SA4-R2、D3 closed-loop 與 D4 geometry-feasible fixed comparison 均已完成且未通過
> 總體 Gate。D4 沒有觀察到改善；selector active frames 中 81.9% 依凍結模型已無
> jointly-feasible action，但這不是碰撞不可避免率。D5 baseline shadow 亦已完成：碰撞
> 最後有解 p50 2.4 s、linear warning 比 radial TTC 中位早 1.2 s，但 frozen static-feasible
> 全 frame 僅 34.53%。D6 已完成 all-active-wall sensitivity：已知牆面移除只救回
> 0.96% blocked frames，residual unmatched LiDAR points 移除則救回 97.65%。D7 再追出
> residual 的 59.95% 是 distractor winner、37.63% 是 1°→5°中心角表示／動態歸因，
> raw geometry unmatched 只有 1.76%。D4-r2 隨後直接比較 actual-angle 與 center shadow：
> feasible `17.49% vs 17.89%`，paired net `-0.40pp`，未恢復候選；obstacle CR
> `16.45% → 16.44%` 亦未改善。最新權威記憶見
> [finding_sa4_d4r2_argmin_result_20260817.md](finding_sa4_d4r2_argmin_result_20260817.md)。
> 本檔後面的 pilot/D1/D2 數據仍有效；角度分支已結案，下一步只限獨立的 noise contract。

## 權威裁決

- SA4 pilot `100/100` 正常完訓，無 NaN/OOM/traceback。
- 不延長 SA4 c100 相同 recipe。
- 不啟動 SA5。
- c50/c100 都只保留為 diagnostic checkpoint，沒有 SA4 accepted parent。
- auto-advance 維持 `HALTED_ALERT`。

完整 paper-ready 資料卡：

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/18_SA4_sim2real_v2_Pilot與CheckpointScreen_20260731.md`

## 血緣

- SA3 Phase A：12/12 PASS，選定 SA3 c100。
- SA3 Phase B：4/4 PASS，但 native_crossing margin 僅 +0.000226。
- SA3 Phase C：6/6 PASS，matched retention PASS。
- SA4 parent：
  `sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1/checkpoint_12800.pt`
- parent SHA-256：
  `e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88`
- SA4 run：
  `sa4_sim2real_v2_from_sa3r1_c100_ne1024_s42_p100_r1`

## Pilot 訓練證據

- 100 iterations，46.8 min，291,989 completed episodes。
- episode-weighted corridor CR：first 25 `54.746%` → last 25 `23.459%`。
- native CR：`10.127%` → `9.568%`。
- narrow 維持地板。
- 這只證明 SA4 分布「學得動」，不是 capability PASS。

## 固定 checkpoint screen r2

輸出：

`logs/gates/sa4_checkpoint_screen/screen_r2_20260731/sa4_checkpoint_screen_summary.json`

完整性：

- 8/8 completed，invalid 0；
- seed818，delay d1=200 ms，stage4 geometry；
- source fingerprint stable；
- r1 因 longitudinal 低於 1,000 episodes 作廢並保留，不得與 r2 混用。

| probe | c50 CR | c100 CR | c100-c50 |
|---|---:|---:|---:|
| nav_native | 6.054% | 7.332% | +1.278 pp |
| native_crossing | 7.565% | 9.149% | +1.584 pp |
| corridor_lateral | 19.178% | 14.105% | -5.073 pp |
| corridor_longitudinal | 27.543% | 29.320% | +1.777 pp |

關鍵區分：

- native retention PASS 只代表 SR/CR 退化未超過 2 pp。
- c100 四個絕對 hard gate 全 FAIL。
- corridor 等權平均有改善，但 worst-direction CR
  `27.543% → 29.320%`，未達預先要求的至少 0.5 pp 改善。
- `resume_candidate=null`、`training_extension_authorized=false`。
- lateral 改善在簡化二項近似下明確；longitudinal +1.777 pp 本身不足以宣稱顯著，
  所以不能把單一 seed 結果寫成普遍因果。

## Action-level 訊號

c50 → c100 longitudinal：

- mean speed：0.199 → 0.303 m/s；
- stop：24.318% → 19.957%；
- reverse：37.042% → 26.016%；
- extreme turn：2.123% → 3.712%；
- obstacle CR：27.097% → 28.830%。

與「較少停車/倒車讓行、較積極通行」一致，但只是相關性，不是根因定案。

## Supervisor

正常完訓後 stale `expected_run.txt` 已在 SA1/SA3/SA4 重複造成假 ALERT。
2026-07-31 已實作 fail-closed `COMPLETE → IDLE`：

- iteration 必須等於 target；
- matching final checkpoint 必須存在；
- console completion marker 必須同 steps；
- strict anomaly scan 必須為零；
- 既有 flock 內寫 status/ledger，最後才原子清空 expected_run；
- 不授權 capability PASS，也不觸發下一 stage。

驗證：187 relevant tests passed、shell syntax PASS、SA4 真實 artifacts 可形成
completion evidence、production IDLE path PASS。

## SA4-D1 三 seed paired replication（完成）

權威輸出：

`logs/gates/sa4_d1/seed_pool_r4_20260731/sa4_d1_pool_summary.json`

完整性：

- 8/8 新格有效，既有 seed818 四格亦通過身份與完整性驗證；
- evaluator seeds `[515, 616, 818]`；
- d1=200 ms、stage4 geometry；
- lateral 1,200 steps、longitudinal 4,000 steps；
- 每格至少 1,000 completed episodes；
- invalid 0、無 `D1_INCOMPLETE.json`、source fingerprint stable；
- 收官後 37 個 D1/screen 相關測試通過。

作廢但保留的三批：

- `seed_pool_20260731_VOID_missing_delay_arg/`：漏傳 delay；目錄內舊 summary
  亦屬 VOID，不得引用；
- `seed_pool_r2_20260731_VOID_steps_default1200/`：未傳 scenario-specific
  `--steps`，longitudinal 誤用 1,200；
- `seed_pool_r3_20260731_VOID_prepatch_source_race/`：平行 session 在修補前
  source 啟動，且執行中來源版本不一致。

每 seed paired `ΔCR = CR(c100) - CR(c50)`：

| family | seed515 | seed616 | seed818 |
|---|---:|---:|---:|
| lateral | -2.261 pp | -3.408 pp | -5.073 pp |
| longitudinal | +1.419 pp | -2.250 pp | +1.777 pp |

count-pooled 結果：

| family | c50 | c100 | ΔCR |
|---|---:|---:|---:|
| lateral | 19.402% (591/3046) | 15.824% (526/3324) | **-3.578 pp** |
| longitudinal | 28.745% (1161/4039) | 29.074% (1759/6050) | +0.330 pp |
| worst direction | 28.745% | 29.074% | **+0.330 pp** |

判讀：

- lateral 改善在三個 evaluator seed 全部同向，可視為可重現的 operational signal；
- longitudinal 的 paired 方向不一致，seed818 的 `+1.777 pp` 不能視為穩定退化；
- pooled longitudinal 與 worst direction 近乎持平但略差，沒有達到預先要求的
  至少 0.5 pp 改善；
- 兩顆 checkpoint 的 corridor 絕對 hard gate 仍 FAIL；
- D1 排除了「只由 evaluator seed818 造成 lateral 改善」的疑慮，但整條訓練仍只有
  training seed42，不能宣稱一般因果。

因此 `resume_candidate=null`、`training_extension_authorized=false` 維持不變。

## SA4-D2 training-time corridor family 監控（完成）

D2 只新增監控，不改 reward、action、scene sampling、模型或 optimizer。

資料流：

- 在 `env.step()` 前，從環境權威狀態
  `_long_corridor_active`、`_long_corridor_obstacles_ready`、
  `_long_corridor_dynamic_motion_type` 與 `episode_length_buf` snapshot family；
- family 為 `lateral`、`longitudinal`、`random_2d`、`mixed`、
  `no_dynamic`、`unready`；
- episode family identity 跨 iteration 保留；只允許 deferred obstacle install
  造成一次 `unready → installed family`；
- 既有 `scene/corridor/*` 完整保留，新增 `corridor_family/*`。

每 family 記錄：

- SR、CR、TO、other outcome；
- active-step occupancy、reset share、completed-episode share；
- wall/obstacle/static/dynamic collision 與 overlap/unattributed；
- mean speed、stop、reverse、high-turn、extreme-turn；
- finite reward/signal mean、coverage 與 nonfinite fraction。

Fail-closed 帳本：

- family completed episodes 必須精確等於 `scene/corridor/episodes`；
- goal/collision/timeout/other 必須分解完整；
- collision subtype 必須可 reconciliation；
- family mask 不一致、非法 mid-episode family drift、非 finite action/total reward
  都拒絕產生可信結論。

驗證：

- 完整 CPU regression：`816 passed, 27 subtests passed`；
- 256-env、64-step isolated GPU smoke 正常結束；
- smoke 中 corridor completed episodes `22 = 8 lateral + 14 longitudinal`；
- active/reset/completed shares 各自加總為 `1.0`，
  `corridor_family/accounting/reconciliation_ok=1`；
- smoke 樣本太小，只能證明 instrumentation 與真實 Isaac lifecycle 對帳，
  **不得引用為 policy 效能證據**。

主要檔案：

- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/corridor_family_metrics.py`
- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/test_corridor_family_metrics.py`
- `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py`
- `scripts/reinforcement_learning/skrl/monitoring/detail_report.py`
- `scripts/reinforcement_learning/skrl/monitoring/README.md`

完整 paper-ready instrumentation 筆記：

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/19_D2_訓練期CorridorFamily監控_20260731.md`

## 2026-08-01 D4 固定比較結果

- SA4-R2 c50 固定 screen：lateral CR 16.95%、longitudinal CR 1.53%；仍無 accepted parent。
- D3 sustained-brake / best-turn / combined 全部 FAIL total Gate；best-turn 發生嚴重
  static/wall failure transfer。
- D4 baseline / geometry total CR 為 `16.95% / 17.56%`，兩臂皆 FAIL，沒有觀察到改善。
- D4 active frames `24,414`，其中 no-feasible `19,988` (`81.9%`)；這只適用於凍結
  selector 模型，分母不是碰撞事件。
- D5 baseline shadow 已完成：碰撞最後有解 p50 2.4 s、持續無解 onset p50 2.2 s；
  linear conflict 比 radial TTC 中位早 1.2 s。但 frozen static-feasible 只有 34.53%，
  因此先查 static model bottleneck，不直接降低 TTC 後重跑。
- D6 v2 已納入 long-corridor + arena boundary 全部有效牆面：wall points 占 61.08%，
  但移除只救回 0.96% frozen-blocked frames；residual unmatched LiDAR points 占 25.22%，
  移除救回 97.65%。h2.4→h0.4 救回 24.35%，c0.10→c0.05 救回 21.41%。
- residual 尚未證實是 sensor noise；下一步只做生成原因分解，不直接降 clearance。
- SA5 維持未授權。

## 2026-08-02 D7 residual LiDAR origin

- 權威輸出：`logs/gates/sa4_d7/residual_origin_r2_20260802/`。
- 2,603,039 residual points 完整分解：distractor winner `59.95%`；bin-centre
  reconstruction `30.82%`；dynamic attribution miss `6.81%`；raw geometry unmatched
  `1.76%`；其餘 `0.66%`。
- noise contract 對 5,760 raw rays 各自套 `p=0.002515` 與 `U(0.2,2.0)m` replacement，
  再由 72-bin min-pool 放大近距假回波；期望每 scan 14.4864 次 distractor draw。
- frozen-blocked frames 中最近 residual 為 distractor 的 descriptive share 是 `99.16%`，
  但不能當作 exact rescue rate 或 collision cause。
- 工程優先級：先稽核 distractor 統計單位/range distribution，再讓 static geometry 保留
  真正 argmin ray azimuth；不先擴充牆面 model、不直接降 clearance。
- r1 因單點 float32 boundary reconciliation 失敗而作廢；只有 r2 可引用。
- SA4 仍 HOLD，SA5 未授權。

## 2026-08-17 D4-r2 actual winner-angle result

- 有效輸出：`logs/gates/sa4_d4/argmin_angle_r2_20260817/`。
- fixed fresh baseline：n=1976，SR 83.05%，obstacle CR 16.45%，wall CR 0.51%，total CR 16.95%。
- actual-angle selector：n=1989，SR 83.36%，obstacle CR 16.44%，wall CR 0.20%，total CR 16.64%。
- 24,665 paired active frames：actual feasible 17.49%，center feasible 17.89%，net `-98 frames = -0.40pp`；`clear_candidate_recovery=False`。
- 5°中心角不是大量 no-feasible 的主因；停止沿 angle 調 selector。
- `0.002515` 是白牆 ROI 既有回波 MAD outlier fraction，不足以支持全 5,760 rays 的 uniform near-ghost Bernoulli；noise correction 必須另立 protocol。
- c6400 仍拒絕，SA4 HOLD，SA5 未啟動。

## 禁止事項

- 不直接從 SA4 c100 resume。
- 不用 aggregate corridor CR 宣告通過。
- 不立即做 108-cell qualification。
- 不同時改 reward、scene weights、optimizer。
- 不把單 training seed / evaluator seed 結論推廣成一般定理。
- 不把 best-turn/combined 的 dynamic CR 下降誤寫成整體避障 PASS。
