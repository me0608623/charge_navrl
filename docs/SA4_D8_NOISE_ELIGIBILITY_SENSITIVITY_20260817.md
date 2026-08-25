# SA4-D8：目前雜訊與 valid-return-only 雜訊配對比較

日期：2026-08-17  
狀態：`COMPLETE_VALID_DIAGNOSTIC_EVIDENCE`  
範圍：單 checkpoint、單 evaluator seed、baseline-only paired shadow

## 1. 研究問題

目前 `full` VLP-16 preset 對全部 5,760 個 raw ray slot 獨立抽取
`Bernoulli(0.002515)`，抽中後以 `U(0.2, 2.0) m` 取代距離。這包含原本沒有命中表面的
ray，也包含已被 dropout 移除的 ray。

D8 要回答：

> 這些「沒有有效存活回波，卻被轉成近距 return」的 ghost，是否造成大量 D4
> `no-feasible` active frame？

本輪不是建立 faithful VLP-16 模型，也不估計修正版 policy 的 SR/CR。

## 2. 配對 counterfactual

同一條 rollout 中，policy 與 simulator 全程使用目前 `full` noise。觀測函式另外保存一份
shadow：

```text
目前雜訊：
  所有 raw ray slot 都可套用 mixed-pixel replacement

valid-return-only shadow：
  eligibility = raw_valid AND NOT hole
  若 distractor 抽中但 eligibility=false：
      恢復為 distractor 前的 dropout-stage range
  其他 ray 完全不變
```

兩份 sweep 共用完全相同的：

- ray hit 與 emitted-ray angle；
- bias、sigma、dropout mask；
- distractor Bernoulli mask 與 `U(0.2,2.0m)` replacement value；
- robot、行人、d1 pending command 與 19×19 action grid。

沒有重抽亂數。修正版沒有餵給 policy，也沒有修改 action。

## 3. 固定協定

| 項目 | 值 |
|---|---|
| checkpoint | SA4-R2 `checkpoint_6400.pt` |
| SHA-256 | `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197` |
| cell | stage4 lateral、seed818、d1=200 ms |
| budget | 64 env × 2,500 steps |
| policy arm | fresh identity baseline only |
| noise used by policy | current `full` preset |
| feasibility | D4 actual winner-angle、2.4 s、0.10 m clearance |
| protocol | `sa4_d8_noise_eligibility_sensitivity/v1` |
| protocol SHA | `1fcce8d769fef404d01304b184eeef96a073cce7b4275849184a1a850e701185` |

事前固定的描述性「大量來源」規則：

```text
corrected-only feasible / current no-feasible >= 25%
AND net recovery / active >= 5%
AND net recovery >= 100 frames
```

這不是統計顯著性檢定。

## 4. 驗證

| 檢查 | 結果 |
|---|---:|
| 全 `rnn_car_wdclean` | 864 passed + 27 subtests |
| GPU smoke | 24/24 records，reconciliation true |
| 正式 D3 records | 160,000 / 160,000 |
| policy action identity errors | 0 |
| d1 alignment errors | 0 / 157,960 samples |
| current trace bitwise matches | 2,500 / 2,500 calls |
| corrected range monotonic errors | 0 |
| current/corrected dynamic-grid mismatch | 0 |
| source fingerprint | rollout 前後一致 |
| NaN / OOM / traceback | 0 |

目前 policy 的結果精確重現既有 fresh baseline：`n=1,976`、SR `83.05%`、CR
`16.95%`、TO `0%`。這支持 D8 trace/shadow 沒有擾動原 rollout。

## 5. 雜訊帳本

總 raw ray slots：

```text
160,000 env-frames × 5,760 rays = 921,600,000
```

| 分類 | rays | share |
|---|---:|---:|
| realized distractor | 2,316,395 | raw slots 的 0.2513% |
| eligible：raw hit 且未 dropout | 1,097,155 | distractors 的 47.36% |
| 不合 eligibility、由 shadow 撤銷 | 1,219,240 | distractors 的 52.64% |

實測 draw rate `0.2513%` 與設定 `0.2515%` 一致。問題不在 Bernoulli sampler 是否執行，
而在其分母包含 no-return 與 dropped ray。

進一步：

- 1,028,455 個 current 72-bin winner 來自會被修正版撤銷的 distractor；
- 1,028,144 個 72-bin range 因修正而改變，占全部 bin slots 的 `8.93%`；
- 159,808 / 160,000 env-frame 至少一個 bin 改變，占 `99.88%`。

因此 `0.2515%` 的 raw-ray 機率經近距 replacement 與 `amin` 後，不是小幅 observation
擾動。

## 6. 配對 feasibility 結果

D4 frozen trigger 共產生 24,534 active frames：

| 分類 | frames | active share |
|---|---:|---:|
| both feasible | 4,202 | 17.13% |
| current only feasible | 34 | 0.14% |
| valid-return-only only feasible | 6,157 | 25.10% |
| both no-feasible | 14,141 | 57.64% |

彙總：

| 指標 | 目前雜訊 | valid-return-only shadow | 差異 |
|---|---:|---:|---:|
| jointly feasible | 17.27% | 42.22% | +24.96 pp net |
| no-feasible | 82.73% | 57.78% | -24.96 pp |
| static-any feasible | 21.42% | 50.63% | +29.21 pp |
| feasible action count | 583,183 | 1,894,010 | 3.25× |

原本 20,298 個 current no-feasible frame 中，6,157 個在修正後變成 feasible：

```text
rescued current no-feasible fraction = 6,157 / 20,298 = 30.33%
net recovery = 6,157 - 34 = 6,123 frames = 24.96% of active
```

`major_fake_obstacle_contributor=True`，通過事前固定的描述性規則。

## 7. 可以與不可以下的結論

> [!success] 可以下的結論
> 在這個固定 c6400 lateral cell 與 frozen D4 feasibility model 中，no-return/dropped
> ray 上生成的近距 ghost 是大量 no-feasible frame 的重要來源。撤銷它們可救回目前
> no-feasible frame 的 30.33%，並使 active-frame no-feasible 由 82.73% 降至 57.78%。

> [!warning] 仍不能下的結論
> 1. 不能說全部無解都由假障礙造成；修正後仍有 57.78% active frame 無解。
> 2. 不能說修正版已降低碰撞；counterfactual 沒有餵給 policy，沒有 corrected-policy SR/CR。
> 3. 不能稱此修正為 faithful VLP-16 noise model；`U(0.2,2.0m)`、edge/ring/range/material
>    條件與 `0.2515%` 的 per-return 解讀仍未被真實資料支持。
> 4. 不能據此接受 c6400、通過 SA4 或啟動 SA5。

## 8. 裁決與下一步

1. 現行 default `full` noise 暫不靜默修改；保留既有 checkpoint 與歷史結果的語意。
2. 明示的 `valid-return-only` fresh closed-loop A/B 已由 SA4-D9 完成；本固定 cell 觀察到
   SR `+1.13 pp`、CR `-1.13 pp`，詳見 `docs/SA4_D9_VALID_RETURN_NOISE_CLOSED_LOOP_AB_20260817.md`。
3. closed-loop A/B 仍只能叫 sensitivity；要改成論文中的實體感測器模型，需補 edge、ring、
   range、material 與 no-return transition 的原始量測。
4. SA4 維持 HOLD，c6400 不是 parent，SA5 未啟動。

## 9. 權威產物

- 結果目錄：`logs/gates/sa4_d8/noise_eligibility_r1_20260817/`
- 配對 JSON：`baseline_lateral_g4_d1_s818_d8.json`
- manifest：`suite_manifest.json`
- freeze：`docs/freeze/sa4_d8_noise_eligibility_sensitivity_v1.json`
- audit：`scripts/reinforcement_learning/skrl/rnn_car_wdclean/d8_noise_eligibility_sensitivity.py`
- runner：`scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa4_d8_noise_eligibility.py`
