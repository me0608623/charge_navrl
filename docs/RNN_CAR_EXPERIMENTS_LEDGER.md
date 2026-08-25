# RNN Car Experiments Ledger

> 追蹤所有 train_rnn_car_wdclip 相關的 code 改動與實驗。

---

## training_20260817_sa4_r3_cont50 — exact optimizer continuation to it100

**日期**: 2026-08-17
**類型**: single-training-seed exact continuation + six-cell fixed screen
**執行者**: Codex

### Continuation

- run：`sa4_r3_cont50_from_r3c6400_ne1024_s42_p50_r1`
- parent：原 SA4-R3 conceptual it50，SHA-256 `fa51b5a3...fed166d`
- optimizer：載入，RL optimizer state 38 組
- config lock：除 checkpoint / optimizer-resume / metadata 外，所有欄位與 R3 相同
- fixed：stage4、1024 env、seed42、rollout128、K8、full noise、
  `valid_return_only`、delay U{0,1,2}、future-occ 0.15、corridor 50/50
- complete：50/50；0 non-finite；0/50 reconciliation failure；max KL 0.01044
- it75 SHA `7bec45d4...136f9c2`；it100 SHA `aeddf32b...8520d7d`
- 最後約 10 輪與舊 R3 layout-split evaluator 共享 GPU，僅造成 fps / wall-time confound

### Six-cell fixed screen

- output：`logs/gates/sa4_r3_cont50_checkpoint_screen/r1_20260817/`
- protocol SHA：`9ec8876c...477a14`
- seed818、d1=200 ms、64 env、runtime `valid_return_only`

| checkpoint | lateral SR/CR/TO | longitudinal SR/CR/TO | native SR/CR/TO |
|---|---|---|---|
| it75 | 83.96/16.04/0.00% FAIL | 84.94/15.06/0.00% FAIL | 94.57/5.43/0.00% PASS |
| it100 | 88.08/11.92/0.00% FAIL | 94.68/5.29/0.03% PASS | 93.60/6.40/0.00% PASS |

### 裁決

- 原先「it100 若仍卡 15-20% 就停止」的 plateau 分支未成立；lateral 仍持續改善。
- 絕對 Gate 仍失敗：it100 lateral 距 SR/CR 門檻各 1.92 pp。
- `selected_checkpoint=null`；it100 只列 diagnostic candidate，不是 accepted parent。
- SA4 HOLD、SA5 未啟動；任何 it125 探測須另行授權且限 +25 iterations。
- 權威報告：`docs/SA4_R3_CONT50_RESULT_20260817.md`。

---

## training_20260817_sa4_r3_valid_return_noise — D9 replication 與 R3 短訓

**日期**: 2026-08-17
**類型**: three-evaluator-seed diagnostic replication + single-training-seed bounded pilot
**執行者**: Codex

### D9 evaluator-seed replication

- frozen checkpoint：SA4-R2 c6400，SHA-256 `c6dbd94b...959a2197`
- fixed cell：stage4 lateral、d1=200 ms、64 env x 2500 steps/arm
- seeds：515 / 616 / 818
- output：`logs/gates/sa4_d9/noise_closed_loop_ab_replication_s515_s616_r1_20260817/`
- protocol SHA：`c1e1345e...d849ed1`
- B-A total CR：`-0.98 / -1.87 / -1.13 pp`
- B-A wall CR：`+0.37 / +0.18 / +0.27 pp`
- TO：六臂全 0；3/3 seeds 通過事前 pilot authorization rule
- pooled descriptive A→B：SR `82.69%→84.02%`、CR `17.31%→15.98%`、
  obstacle CR `16.84%→15.24%`、wall CR `0.47%→0.75%`
- evaluator-seed replication，不是 training-seed replication；`inferential_claim=false`

### SA4-R3 training

- run：`sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1`
- clean parent：SA3 c100，SHA-256 `e7da9aa0...e453f88`
- sole non-metadata config diff vs SA4-R2：
  `lidar_distractor_eligibility all_rays→valid_return_only`
- fixed：stage4、1024 env、seed42、rollout128、K8、full noise、delay U{0,1,2}、
  future-occupancy weight 0.15、no optimizer resume
- complete：50/50，1437 s，147,951 episodes；0 NaN/OOM/traceback，0/50 reconciliation failure
- max KL 0.00807、max clip 0.10134
- it25 SHA `197dbbd2...538e9f3`；it50 SHA `fa51b5a3...fed166d`

### Six-cell fixed screen

- output：`logs/gates/sa4_r3_checkpoint_screen/r1_20260817/`
- protocol SHA：`8a5ec62f...9844a51`
- seed818、d1=200 ms、64 env、runtime `valid_return_only`

| checkpoint | lateral SR/CR/TO | longitudinal SR/CR/TO | native SR/CR/TO |
|---|---|---|---|
| it25 | 78.75/21.25/0.00% FAIL | 50.39/48.97/0.64% FAIL | 95.09/4.91/0.00% PASS |
| it50 | 81.13/18.87/0.00% FAIL | 78.21/21.73/0.06% FAIL | 94.06/5.94/0.00% PASS |

### 裁決

- D9 方向在三 evaluator seeds 重現，合法授權了 bounded R3 pilot。
- R3 訓練健康，但沒有 checkpoint 同時通過 lateral、longitudinal、native。
- `selected_checkpoint=null`、`accepted_parent=null`。
- 不延長相同 recipe；SA4 HOLD；SA5 未啟動。
- 權威報告：`docs/SA4_D9_REPLICATION_AND_R3_PILOT_20260817.md`。

---

## diagnostic_20260817_sa4_d9_noise_closed_loop_ab — valid-return-only 實際 SR/CR

**日期**: 2026-08-17
**類型**: fresh-process single-seed closed-loop sensitivity A/B
**執行者**: Codex

### 固定條件

- checkpoint：SA4-R2 c6400，SHA-256 `c6dbd94b...959a2197`
- stage4 lateral、seed818、d1=200 ms、每臂 64 env x 2500 steps
- output：`logs/gates/sa4_d9/noise_closed_loop_ab_r1_20260817/`
- A=`all_rays` 歷史 full；B=`valid_return_only`，修正 sweep 真正餵 policy/critic
- 唯一行為差異為 distractor eligibility；其餘 noise/scene/action/checkpoint 固定
- protocol：`sa4_d9_noise_closed_loop_ab/v1`，SHA `e7ad89bc...ba41ca4`

### 結果

- A：n=1,976，SR 83.0466%，CR 16.9534%，obstacle 16.4474%，wall 0.5061%，TO 0%
- B：n=2,066，SR 84.1723%，CR 15.8277%，obstacle 15.0532%，wall 0.7744%，TO 0%
- B-A：SR `+1.13 pp`、CR `-1.13 pp`、obstacle `-1.39 pp`、wall `+0.27 pp`
- 事前 directional 與 >=0.5pp material descriptive improvement 規則皆通過
- A 精確重現 D8/既有 baseline，證明 `all_rays` backward compatibility

### 驗證與裁決

- 871 tests + 27 subtests；兩臂 runtime mode、d1、corridor contract 皆驗證
- source before=between=after；無 NaN/OOM/traceback；GPU 清空
- fresh-process 同 seed/同抽樣律，不是逐幀 mask bitwise paired；v1 wording 已附 interpretation erratum
- 單 seed 只支持實際點估計改善，不足以宣稱可重現或統計確證
- B 仍未過 SA4：SR 84.17% <90%，CR 15.83% >10%
- valid-return-only 仍是 sensitivity，不是 validated VLP-16 model
- SA4 HOLD、c6400 rejected、SA5 未啟動；歷史 full default 尚不變

---

## diagnostic_20260817_sa4_d8_noise_eligibility — mixed-pixel eligibility 配對 shadow

**日期**: 2026-08-17
**類型**: baseline-only privileged paired diagnostic
**執行者**: Codex

### 固定條件

- checkpoint：SA4-R2 c6400，SHA-256 `c6dbd94b...959a2197`
- stage4 lateral、seed818、d1=200 ms、64 env x 2500 steps
- output：`logs/gates/sa4_d8/noise_eligibility_r1_20260817/`
- policy/simulator 全程使用 current full noise；counterfactual 只進 D4 shadow
- counterfactual：同 masks/values，只允許 `raw_valid AND NOT hole` ray 保留 distractor
- protocol：`sa4_d8_noise_eligibility_sensitivity/v1`，SHA `1fcce8d...e701185`

### 結果

- 921,600,000 raw slots 中 2,316,395 distractors（0.2513%）
- 1,219,240（52.64%）來自 no-return/dropped ray 並由 shadow 撤銷
- 1,028,144 個 72-bin values 改變（全部 bin slots 的 8.93%）
- D4 active frames 24,534
- current no-feasible 82.73%；valid-return-only shadow 57.78%
- corrected-only feasible 6,157；current-only 34；net +6,123 frames = +24.96pp
- 救回 current no-feasible 的 30.33%；事前描述性 `major_fake_obstacle_contributor=True`
- 修正後仍 57.78% no-feasible，故不是完整解釋

### 驗證與裁決

- 864 tests + 27 subtests；160,000 identity checks、0 errors
- d1 alignment 0/157,960 errors；trace 2,500/2,500 exact；all reconciliation true
- source fingerprint stable；無 NaN/OOM/traceback
- 不能宣稱 corrected-policy SR/CR 改善，因 counterfactual 未餵 policy
- 不能稱 valid-return-only 為 faithful VLP-16 model；距離與 edge/ring/material 條件仍缺
- default full noise 不靜默修改；後續 D9 fresh closed-loop A/B 已完成，見本檔前一節
- SA4 HOLD、c6400 rejected、SA5 未啟動

---

## diagnostic_20260802_sa4_d7_residual_lidar_origin — D6 residual 點生成來源

**日期**: 2026-08-02
**類型**: baseline-only privileged realized-trace diagnostic
**執行者**: Codex

### 固定條件

- checkpoint：SA4-R2 c6400，SHA-256 `c6dbd94b...959a2197`
- stage4 lateral、seed818、d1=200 ms、64 env x 2500 steps
- output：`logs/gates/sa4_d7/residual_origin_r2_20260802/`
- protocol：`sa4_d7_lidar_residual_origin/v2`，SHA-256 `8f50b168...cdf38ca`
- policy action unchanged：160,000 identity checks、0 errors
- exact policy LiDAR trace：2,500/2,500，ambiguous 0
- D3/D6/corridor replay 與既有 D6 baseline 相同；source fingerprint stable

### 結果

2,603,039 個 final D6 residual points 的互斥來源：

- distractor winner：1,560,617（59.95%）
- 1°→5° bin-centre reconstruction：802,188（30.82%）
- actual-angle dynamic attribution miss：177,250（6.81%）
- raw geometry unmatched：45,873（1.76%）
- 其餘 range/dynamic/unresolved：17,112（0.66%）

配置以 `p=0.002515` 對 5,760 raw rays 獨立注入 `U(0.2,2.0)m` distractor，再做
72-bin minimum pooling；期望每 scan 14.4864 次 draw。Frozen-blocked frames 中最近
residual 是 distractor 的比例為 99.16%，但這不是 exact binding-cause 或 collision share。

### 裁決

主要來源是 distractor/min-pool 與丟失 argmin azimuth 的 5°中心角表示，不是大量缺失牆面。
先稽核 noise contract 並讓 static geometry 保留 argmin azimuth；不直接停用全部 noise、降低
clearance、接受 SA4 parent 或啟動 SA5。r1 單點 float32 boundary 對帳失敗，保留但不得引用。

---

## diagnostic_20260801_sa4_d6_static_feasibility — static constraint 來源與敏感度

**日期**: 2026-08-02
**類型**: baseline-only privileged diagnostic
**執行者**: Codex

### 固定條件

- checkpoint：SA4-R2 c6400，SHA-256 `c6dbd94b...959a2197`
- stage4 lateral、seed818、d1=200 ms、64 env x 2500 steps
- output：`logs/gates/sa4_d6/static_sensitivity_r3_20260801/`
- policy action unchanged；160,000 identity checks、0 errors
- protocol：`sa4_d6_static_feasibility_sensitivity/v2`

### 結果

- frozen h2.4/c0.10 static-any：34.53%；blocked frames 104,759
- all active known walls：61.08% LiDAR points；移除只救回 0.96% blocked frames
- static obstacles：13.70%；移除救回 0.93%
- residual unmatched LiDAR：25.22%；移除救回 97.65%
- horizon h2.4→h0.4（c0.10）：救回 24.35%
- clearance c0.10→c0.05（h2.4）：救回 21.41%

### 裁決

主導 frozen static constraint 的是 residual LiDAR-point partition，不是已知牆面；horizon
與 clearance 是次要、耦合的放大因子。Residual 尚未等同 sensor noise，禁止直接下調
margin、接受 SA4 parent 或啟動 SA5。下一步只做 residual 生成原因 audit。

---

## code_20260508_01 — Experiment Config Profiles (Phase 1)

**日期**: 2026-05-08
**類型**: code infrastructure
**者**: Claude Code (PC-A)

### 變更

新增 LEGO-style experiment config 系統，用 `--experiment_config <name>` 取代大量 CLI 參數。

### 修改檔案

- `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py` — 新增 --experiment_config / --print_experiment_config
- 新增 `scripts/reinforcement_learning/skrl/rnn_car_modular/experiment_config.py`
- 新增 `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/` (registry + 4 built-in configs)

### Built-in Configs

| Config | Algorithm | Aux | Entropy (lin/ang) |
|---|---|---|---|
| wd_sa2_a2c_aux | A2C-WD | WD 7D geometry | 0.05 / 0.10 |
| wd_sa2_a2c_aux_lowent | A2C-WD | WD 7D geometry | 0.01 / 0.02 |
| wd_sa2_a2c_noaux | A2C-WD | none | 0.05 / 0.10 |
| wd_sa2_ppo_noaux | PPO clip | none | 0.05 / 0.10 |

### 驗證

- py_compile: 8 檔全通
- Unit tests: 5/5 pass
- Backward compatible: 不加 --experiment_config 時行為不變

---

## smoke_20260508_01 — Experiment Config Smoke Test

**日期**: 2026-05-08
**類型**: smoke test
**者**: Hermes (PC-A)

### 測試

```
--experiment_config wd_sa2_a2c_aux_lowent --headless --num_envs 4 --timesteps 300 --rollout_length 30
```

### 結果: 全 PASS

- 38 欄位正確套用 (initial_stage=2, fixed_stage=True, use_a2c=True, aux=tbptt, ent=0.01/0.02, lidar_no_noise=True)
- CLI override 正確優先 (num_envs=4 覆蓋 config 的 1024)
- WandB 上傳成功，run name 正確
- 10 iterations 完整跑完，無 crash
- Boolean flag mapping 正確

### 結論

Phase 1 config 系統驗證通過，可進入正式實驗。

---

## diagnostic_20260731_sa4_pilot_screen — SA4 c50/c100 延長裁決

**日期**: 2026-07-31
**類型**: training-health + checkpoint diagnostic
**執行者**: Claude / Codex 協作

### 訓練

- run：`sa4_sim2real_v2_from_sa3r1_c100_ne1024_s42_p100_r1`
- parent：SA3 c100，SHA-256 `e7da9aa0...e453f88`
- outcome：100/100 正常完訓，46.8 min，無 NaN/OOM/traceback
- training corridor CR：first 25 `54.746%` → last 25 `23.459%`

### 固定 screen

- output：`logs/gates/sa4_checkpoint_screen/screen_r2_20260731/`
- 8/8 completed，invalid 0，seed818，d1=200 ms，source stable
- c50→c100 lateral CR：`19.178% → 14.105%`
- c50→c100 longitudinal CR：`27.543% → 29.320%`
- worst-direction CR：`27.543% → 29.320%`
- verdict：`resume_candidate=null`

### D1 三 evaluator seed replication

- output：`logs/gates/sa4_d1/seed_pool_r4_20260731/`
- 新格 8/8 有效、既有 seed818 四格驗證通過、invalid 0、source stable
- seeds：515 / 616 / 818；d1=200 ms；stage4 geometry
- lateral pooled CR：`19.402% → 15.824%`（`-3.578 pp`）
- longitudinal pooled CR：`28.745% → 29.074%`（`+0.330 pp`）
- worst-direction CR：`28.745% → 29.074%`（`+0.330 pp`）
- paired lateral 三 seed 全改善；paired longitudinal 為
  `+1.419 / -2.250 / +1.777 pp`，方向不一致
- r1 漏 delay、r2 漏 scenario-specific steps、r3 pre-patch source race；
  三批均標記 VOID 並保留，不得引用

### 裁決

- 不延長 SA4 c100 相同 recipe。
- 不啟動 SA5。
- c50/c100 都是 diagnostic evidence，不是 accepted parent。
- D1 已完成；不再追加 evaluator seed。
- D2 family instrumentation 已完成；本次未啟動 trainer。
- 新訓練前必須加入 lateral/longitudinal 分開的 occupancy、episode、reward 與 action
  accounting。

### 權威資料卡

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/18_SA4_sim2real_v2_Pilot與CheckpointScreen_20260731.md`

---

## code_20260731_sa4_d2_corridor_family_monitor — 訓練期方向分解監控

**日期**: 2026-07-31
**類型**: monitoring instrumentation
**執行者**: Codex

### 範圍

- 只新增訓練期 metrics、console、JSONL 與 detail report。
- 不改 reward、action、scene sampling、模型、optimizer 或 checkpoint。
- 既有 `scene/corridor/*` 保留，新增 `corridor_family/*`。

### 資料來源與 family

- 在 `env.step()` 前 snapshot 環境權威狀態，避免 terminal env 在 step 內 reset 後
  family 被下一回合覆寫。
- family：`lateral`、`longitudinal`、`random_2d`、`mixed`、
  `no_dynamic`、`unready`。
- episode family identity 跨 iteration 保留，deferred install 只允許
  `unready → installed family`。

### 指標與 fail-closed accounting

- outcome：SR / CR / TO / other。
- sampling：active-step occupancy / reset share / completed-episode share。
- behavior：speed / stop / reverse / high-turn / extreme-turn。
- collision：wall / obstacle / static / dynamic / overlap / unattributed。
- reward/signal：finite mean / coverage / nonfinite fraction。
- family episode sum 必須精確等於 `scene/corridor/episodes`；outcome 與 collision
  decomposition 也必須 reconciliation，否則拒絕可信結論。

### 驗證

- CPU：`816 passed, 27 subtests passed`。
- isolated GPU smoke：256 env × 64 steps，正常結束。
- corridor completed episodes：`22 = 8 lateral + 14 longitudinal`。
- active/reset/completed shares 各自加總為 `1.0`；
  `corridor_family/accounting/reconciliation_ok=1`。
- smoke 僅驗證 instrumentation，不是 policy performance evidence；輸出目錄已有
  `SMOKE_ONLY_NOT_EVIDENCE.md`。

### 輸出

- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/corridor_family_metrics.py`
- `scripts/reinforcement_learning/skrl/rnn_car_wdclean/test_corridor_family_metrics.py`
- `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py`
- `scripts/reinforcement_learning/skrl/monitoring/detail_report.py`
- `logs/rnn_car/smoke_monitor_d2_corridor_family_terminal_20260731/`

### 裁決

- D2 完成，不代表 SA4 capability PASS，也不授權 SA5。
- SA4-R2 若要啟動，仍須先選一個 intervention，從原 SA3 c100 parent 跑
  50-iteration bounded pilot，並同步檢查 native retention 與兩個 corridor family。

---
