# Mixed-corridor probe：窄縫多點驗收結果（監督端回報）

**日期**：2026-07-25
**對象**：`sa6_k8_obb_mixed_corridor_probe_c19200_s42`（config `e2e_sa6_k8_obb_mixed_corridor_probe_from_c19200`）
**產出者**：監督端（唯讀驗收，未改任何 repo 檔案）

---

## TL;DR

1. **窄縫（Gate5, 1.2m）在 probe 期間劇烈震盪，且最終 checkpoint 崩壞**：iter20 **41.9%** → iter25 **94.9%** → iter30 **13.3%**（訓練前 c19200 = 95.4%）。
2. **只驗 iter30 會誤判「全毀」，只驗 iter25 會誤判「沒事」**。單點驗收在此 run 上不可靠。
3. **建議：先解決窄縫穩定性，再延長訓練。** 若直接從 iter30 延長，等同從窄縫谷底往下挖。
4. **走廊側成果顯著**（longitudinal −24.2pp、random_2d −17.3pp、mixed −18.4pp CR），mixed replay 方向正確，值得保留；問題在保留機制不足以撐住窄縫。

---

## 1. 我跑了什麼

對 probe 的三個 checkpoint 各跑一次 1.2m 部署窄縫閘（與 `run_sa5_joint_retention_gates.py` 的 Gate5 同參數，seed 404）：

```bash
python scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py \
  --checkpoint logs/rnn_car/sa6_k8_obb_mixed_corridor_probe_c19200_s42/checkpoint_{2560,3200,3840}.pt \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
  --curriculum_version warp_drive_e2e_final20_v1 \
  --deterministic --num_envs 64 --steps 1200 --headless \
  --num_goals_override 1 --no_goal_movement \
  --stage 5 --arena_size 10 --num_static_obs 0 --num_dynamic_obs 0 --obs_near_goal_count 0 \
  --narrow_gap_eval --narrow_gap_width 1.2 --narrow_gap_yaw_limit_deg 10 --seed 404
```

log：`/tmp/narrow_probe_c2560.log`、`/tmp/narrow_probe_c3200.log`、`/tmp/narrow_probe_c3840.log`

## 2. 結果

| checkpoint | iter | 穿越率 crossing | **直接穿越率 direct** | 路徑倍率 p50 | 回頭距離 p50 | 首次穿越 p50 |
|---|---|---|---|---|---|---|
| 2560 | 20 | 41.9% | **0.0%** | 2.76× | 2.38 m | 31.4 s |
| 3200 | 25 | **94.9%** | **0.0%** | 2.24× | 0.94 m | 18.4 s |
| 3840 | 30 | **13.3%** | **0.0%** | 3.69× | 2.69 m | 31.7 s |
| c19200（訓練前） | — | 95.4% | 該版 eval 無此指標 | — | — | — |

Gate5 門檻：crossing ≥ 0.95 / cr ≤ 0.05。**三點皆未過**（iter25 的 94.9% 差 0.1pp）。

### 失敗型態（iter30，586 回合）
`撞牆 503 / 到達 78 / 超時 5` — 主要是撞縫牆，不是不敢進場或卡住。
對照 c19200 正常時（794 回合）：`到達 762 / 撞牆 32`。

## 3. 兩個判讀

### (a) 窄縫震盪，單點驗收不可靠
41.9 → 94.9 → 13.3%，10 iter 間距內擺幅超過 80pp。這與先前 corridor-deepen 期間觀察到的窄縫震盪同型（iter130 = 1.86% → iter140 = 20.4% → iter150 = 95.4%）。

**含意**：任何以單一 checkpoint 的 Gate5 結果決定「保留 / 延長 / 放棄」的判斷，在此血緣上都有高機率誤判。**建議往後 Gate5 一律至少取 3 個 checkpoint**。

### (b) rollout retention agreement 無法預測 Gate5
probe 全程 `retention_post_agreement` 線性 0.918–0.939 / 角速度 0.800–0.856，**看起來完全健康**，但同期 Gate5 crossing 在 13–95% 間亂跳。

這與先前的紀錄一致：projection8 曾有 agreement 94.1% / 86.8% 但 Gate5 crossing 僅 9.8%。**agreement 不可當作窄縫保留的驗收證據，必須跑實際 gate**。

### (c) 路徑品質：direct_crossing_rate = 0%
三個 checkpoint 的**直接穿越率皆為 0**，路徑長度是直線的 2.2–3.7 倍、回頭 0.9–2.7 m、首次穿越耗時 18–32 s（direct 門檻為 ≤10 s / 路徑倍率 ≤1.35 / 回頭 ≤0.5 m）。

**這與用戶在 GUI 上目視到的行為一致**：機器人先往旁邊牆繞、再回頭對準才鑽過去。

⚠️ **誠實界定**：c19200 的那次 Gate5 是在這些 path-quality 指標加入之前跑的，**因此無法判定 direct=0 是本次訓練造成、還是既有問題**。用戶 GUI 目視的對象正是 c19200，觀察到同樣的繞路 → 傾向是**既有問題**，非本 probe 新增。若要確認，需對 c19200 用現版 eval 重跑一次。

## 4. 走廊側成果（供權衡）

我獨立聚合的 c30 走廊 gate（3 seed，與 A/B baseline 同協定）：

| 模式 | probe c30 CR | c19200 基準 CR | 變化 |
|---|---|---|---|
| lateral | 8.05% | 5.06% | **+2.99pp（退步）** |
| longitudinal | 42.11% | 66.35% | −24.24pp |
| random_2d | 52.16% | 69.41% | −17.25pp |
| mixed（1 seed） | 30.49% | 48.85% | −18.36pp |

另註：mixed 實測 30.49% **低於**三純模式線性預測 34.11%（−3.6pp）。基準時代此偏移一直是 **+1.3 ~ +3.1pp**，方向翻轉，可能代表訓練後學到的是通用動態應對而非模式專用招式（利於泛化）。惟僅 1 seed，待其餘 seed 確認。

**lateral 退步需納入停止條件**：距 <10% 門檻的餘裕從 4.94pp 縮到 1.95pp。

## 5. 建議（供參，決定權在你）

1. **延長訓練前先處理窄縫穩定性**，而非延長後再修。理由：iter30 位於窄縫谷底（13.3%），從谷底延長等同往下挖；且震盪幅度 >80pp 表示目前的保留機制沒有把窄縫鎖住。
2. **可考慮的方向**（未驗證，僅列選項）：提高 narrow retention 權重／提高 12% 窄縫 replay 比例／對窄縫改用 CE-lock（過往紀錄顯示 CE 比 margin 更能持續施壓）／或在延長訓練中每 5 iter 自動跑 Gate5 做 early-stop。
3. **Gate5 驗收改多點取樣**（≥3 checkpoint），並建議納入 `direct_crossing_rate` 作為部署品質指標——它量化了用戶 GUI 指出的繞路問題。
4. **longitudinal 的驗收門檻建議先修再定案**：監督端鑑別實驗顯示該模式為超加成幾何陷阱（1 動態障礙 CR 21.8% → 2 動態實測 66.4%，獨立預測僅 38.8%，超出 +27.6pp），原因是兩個障礙固定在 x=±0.40 的同一中央車道。random_2d 則為真實能力缺口（1 動態 39.1%，2 動態 69.4% ≈ 獨立疊加 +6.4pp），難度誠實，可直接當訓練/驗收目標。

## 6. 未解事項

- c105 的 Gate5 窄縫 / Gate2 從未量測（若日後改由 c105 起跑需先補）。
- c19200 用現版 eval（含 path-quality 指標）重跑，以釐清 direct=0 是既有或新增。
- probe c30 的 mixed 其餘 2 seed。
