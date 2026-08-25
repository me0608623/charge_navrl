# SA4-D9：valid-return-only 雜訊 closed-loop A/B

日期：2026-08-17  
狀態：`COMPLETE_VALID_REPLICATED_DIAGNOSTIC_EVIDENCE`  
範圍：單 checkpoint、單 corridor family、三 evaluator seed 的 closed-loop sensitivity

> **後續更新：** 本文件第 4 節保留原始 seed818 結果。新增 seeds 515/616 後，三個
> evaluator seeds 的 total CR 均下降，依事前規則授權了一次 SA4-R3 50-iteration pilot。
> R3 已完成，但 it25/it50 的 lateral 與 longitudinal 固定 Gate 均失敗；不延長、不接受
> parent、不啟動 SA5。權威後續報告：
> `docs/SA4_D9_REPLICATION_AND_R3_PILOT_20260817.md`。

## 1. 研究問題

SA4-D8 在同一 rollout 的 shadow 中發現，撤銷 no-return/dropped ray 上的近距 distractor，
可使 frozen-D4 no-feasible active frame 由 `82.73%` 降至 `57.78%`。D8 未把修正掃描餵給
policy，因此 D9 要回答：

> 在固定 SA4 c6400 lateral cell 中，policy 真正使用 valid-return-only LiDAR 後，episode
> SR/CR 是否也會改善？

## 2. 唯一變因

| 臂 | mixed-pixel eligibility | policy 實際輸入 |
|---|---|---|
| A `current_full` | 所有 5,760 raw ray slot | 歷史 `full` sweep |
| B `valid_return_only` | `valid AND NOT dropout_hole` | 修正 sweep |

兩臂都先抽取相同形式的 Bernoulli mask 與 `U(0.2,2.0 m)` replacement value；B 只在抽樣後
拒絕不合資格的 replacement。Bias、Gaussian displacement、dropout rate、distractor rate、
場景、checkpoint、policy、動作延遲及 rollout budget 均不變。兩臂是 fresh process、相同 seed；
closed-loop 軌跡分岔後不可解讀為逐 episode paired sample。

凍結 v1 protocol 中的「same mask/value」是指 eligibility 分支本身不增加或減少 RNG draw，
不是宣稱軌跡與 reset 分岔後仍逐幀 bitwise 配對。解讀更正保存在結果目錄的
`PROTOCOL_INTERPRETATION_ERRATUM.md`；D9 與 D8 的 paired shadow 性質不同。

## 3. 固定協定

| 項目 | 值 |
|---|---|
| checkpoint | SA4-R2 `checkpoint_6400.pt` |
| SHA-256 | `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197` |
| cell | stage4 lateral、seed818、d1=200 ms |
| budget | 每臂 64 env × 2,500 steps |
| noise preset | `full`；只改 distractor eligibility |
| action override | 無 |
| protocol | `sa4_d9_noise_closed_loop_ab/v1` |
| protocol SHA | `e7ad89bc996d9c012515e25746dd6162bf52c13a61a5f407945ae5f2dba41ca4` |

事前描述規則把 SR 上升、CR 下降且 TO 不劣化稱為 directional improvement；SR gain 與 CR
reduction 都至少 `0.5 pp` 才稱 material descriptive improvement。這不是顯著性檢定。

## 4. 結果

| outcome | A current_full | B valid_return_only | B − A |
|---|---:|---:|---:|
| episodes | 1,976 | 2,066 | +90 |
| SR | 83.0466% | 84.1723% | **+1.13 pp** |
| total CR | 16.9534% | 15.8277% | **−1.13 pp** |
| obstacle CR | 16.4474% | 15.0532% | **−1.39 pp** |
| wall CR | 0.5061% | 0.7744% | +0.27 pp |
| TO | 0.0000% | 0.0000% | 0.00 pp |

Exact outcomes：

- A：1,641 success、325 obstacle collision、10 wall collision；
- B：1,739 success、311 obstacle collision、16 wall collision。

`directional_closed_loop_improvement_observed=True`，且
`material_closed_loop_improvement_observed=True`。A 臂精確重現 D8 與既有 fixed baseline 的
`n=1,976 / SR=83.05% / CR=16.95%`，支持新增的預設 `all_rays` 沒有改變歷史行為。

## 5. 判讀

> [!success] 本輪可以下的結論
> 在這個固定 checkpoint、lateral family 與 evaluator seed 中，把 valid-return-only sweep
> 真正餵給 policy 後，實際 closed-loop SR 點估計增加 `1.13 pp`、CR 點估計下降 `1.13 pp`。
> 改善主要來自 obstacle CR 下降 `1.39 pp`，不是只改善 safety shadow 的內部指標。

但效果遠小於 D8 的 `24.96 pp` feasibility recovery。D8 測的是 frozen D4 selector 是否存在
候選；D9 測的是未重新訓練 policy 面對不同 observation 後的最終 episode outcome，兩者不是
同一估計量，不能期待線性換算。

原始第 4 節只有一個 evaluator seed。若暫以獨立 Bernoulli episode 做粗略 sanity check，CR rate
difference 的 95% Wald interval 約為 `[-3.41, +1.16] pp`，包含 0；實際 episode 也不是完全
獨立。因此本輪只支持「觀察到實際改善」，尚不足以宣稱改善可重現或具統計確證。

> [!warning] 不可下的結論
> 1. 不可稱 valid-return-only 已被驗證為 faithful VLP-16 noise model。
> 2. 不可說所有假障礙問題已解決；B 的 CR 仍為 `15.83%`。
> 3. 三個 evaluator seeds 的 wall CR 點估計皆增加，但均小於事前 +0.5 pp 容忍值；
>    仍不可把 evaluator-seed 描述直接推廣成 training failure transfer。
> 4. 不可接受 c6400 為 SA4 parent、宣告 SA4 畢業或啟動 SA5。

## 6. 驗證

| 檢查 | 結果 |
|---|---:|
| 全 `rnn_car_wdclean` tests | 871 passed + 27 subtests |
| A runtime mode | `all_rays`，policy+critic |
| B runtime mode | `valid_return_only`，policy+critic |
| actuator | d1=200 ms，decode→delay→scale→lag |
| corridor geometry/motion/penetration | 兩臂皆通過 |
| source fingerprint | before=between=after |
| NaN / OOM / traceback | 0 |
| protocol/checkpoint hash | 相符 |
| GPU cleanup | 完成，僅桌面 daemon |

## 7. 裁決與下一步

1. D9 已回答「是否有實際 closed-loop 改善」：三個固定 evaluator seeds 的點估計均改善。
2. 所有 B 臂仍未過 SA4 `SR≥90% / CR≤10% / TO≤5%`，SA4 保持 HOLD。
3. 依事前規則執行的 R3 50 輪短訓未產生合格 checkpoint，因此不延長、不接受 parent、
   不啟動 SA5。
4. 若目標是論文中的真實感測器模型，仍需補 edge/ring/range/material/no-return transition 的
   原始 VLP-16 資料，不能用本 sensitivity 取代。

## 8. 權威產物

- 結果：`logs/gates/sa4_d9/noise_closed_loop_ab_r1_20260817/`
- manifest：`suite_manifest.json`
- freeze：`docs/freeze/sa4_d9_noise_closed_loop_ab_v1.json`
- protocol/comparison：`scripts/reinforcement_learning/skrl/rnn_car_wdclean/d9_noise_closed_loop_ab.py`
- runner：`scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa4_d9_noise_closed_loop_ab.py`
- protocol 解讀更正：`PROTOCOL_INTERPRETATION_ERRATUM.md`
