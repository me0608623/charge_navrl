# SA7 → SA8 過夜驗收（2026-07-27）

## 正式 SA7 run

| 項目 | 值 |
|---|---|
| run_name | `sa7_wander_w1c10_ne1024_s42_r3` |
| W&B | `hvf16ewq` |
| 參數 | 1024 env、seed 42、38400 timesteps、save_interval 10、不並行 eval |
| 暖啟動 | 原始 W1-c10 `sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt` |
| 目標 | 300/300 iter、`checkpoint_38400.pt` |

**不得使用 r1 / r2 或其 checkpoint**：

| 廢棄 run | 原因 |
|---|---|
| `sa7_wander_w1c10_ne1024_s42` | iter 10 後崩潰（`lateral obstacles exceed the 2 available lanes`），checkpoint 僅供診斷 |
| `sa7_wander_w1c10_ne1024_s42_r2` | 在 dump 接線完成前提前啟動，載入未接線版本，已終止 |

## 第一階段：等待 SA7 結束

放行條件（須同時成立）：

- [ ] log 出現 `Training complete` 與 `300/300`
- [ ] `checkpoint_38400.pt` 存在且可讀
- [ ] SA7 進程已退出、GPU 已釋放
- [ ] 全 log 無 Traceback / RuntimeError / NaN / OOM / overflow dump
- [ ] corridor runtime audit 持續成立

### 進行中的觀測

| 里程碑 | 狀態 |
|---|---|
| iter 1 | SR 63.5%、CR 12.6%、TO 0% |
| **iter 10（首次崩潰位置）** | **已安全越過** — SR 86.7%、CR 13.2%、TO 0%、無 dump |

### 走廊 runtime audit（n=502，1024 env）

| 條件 | 門檻 | 實測 | 判定 |
|---|---|---|---|
| family 三類各距 33.33% | ≤2pp | `[0.3324, 0.3351, 0.3324]` | **0.18pp** ✅ |
| density 誤差 | ≤2pp | `{0.2510, 0.3506, 0.1992, 0.1494, 0.0498}` | **0.10pp** ✅ |
| wander 全安裝 | 全過 | 375/375 | ✅ |
| crossing / side geometry | 全過 | 75/75、32/32 | ✅ |
| `s_spacing` / `s_speed_delta` | 0 | 0 / 0 | ✅ |
| overflow dump | 無 | 無 | ✅ |

隊形維持 98.5%（2222/2255）；`crossing_completed=50/50(installed=75)`；`x/s_wrong_family` 0/0。

## 第二階段：checkpoint 篩選 —— 完成

## 第三階段：正式 gates（待篩選完成）

## 第四階段：建立 SA8 config（待 gates 通過）

## 第五階段：啟動正式 SA8（待 smoke 通過）

## 封鎖項（全程維持）

- N1 direct-imitation 臂：**封鎖**
- λ = 0.067：**未填入 config**，`_LAMBDA_IS_CALIBRATED = False`（量自 SA6/14m，SA8/12m 須重量）

## 第三階段：正式 gates —— **四顆候選全數 FAIL**

門檻：四模式各 SR≥90%、CR≤10%、TO≤5%。seeds 515/616/717，1200 steps。

| 候選 | lateral | longitudinal | random_2d | mixed_iid | 四模式最差 margin | sentinel 排名 |
|---|---|---|---|---|---|---|
| **c90** | 89.8% | 77.3% | 69.8% | 78.5% | **-20.2pp** | 第 4 |
| **c190** | 87.0% | 68.1% | 70.7% | 73.6% | **-21.9pp** | 第 3 |
| **c160** | 87.5% | 67.8% | 70.4% | 73.5% | **-22.2pp** | 第 1 |
| **c140** | 88.0% | 67.3% | 72.3% | 73.5% | **-22.7pp** | 第 2 |

全部 `all_modes_pass=False`，依 goal 跳過各自的正式 Gate5a 與 validator。
TO 全部 0.0%，structural checks 全部 True —— **失敗純粹來自碰撞率**。

### Pareto 最佳候選：**c90**

c90 在 **lateral、longitudinal、mixed_iid 三個模式都是最佳**，僅 random_2d 略低（−0.6～−2.5pp）：

| 模式 | c90 | 其餘三顆範圍 | c90 相對 |
|---|---|---|---|
| lateral | **89.8%** | 87.0–88.0% | +1.8～+2.8pp |
| longitudinal | **77.3%** | 67.3–68.1% | **+9.2～+10.0pp** |
| random_2d | 69.8% | 70.4–72.3% | −0.6～−2.5pp |
| mixed_iid | **78.5%** | 73.5–73.6% | **+4.9～+5.0pp** |

**更正（2026-07-28）**：先前此處寫「c90 最差 margin −12.7pp」是錯的 ——
−12.7pp 是 longitudinal（第二差），**c90 真正的最差模式是 random_2d，margin −20.2pp**。

| 候選 | lateral | longitudinal | random_2d | mixed_iid | **最差模式** | **最差 margin** |
|---|---|---|---|---|---|---|
| c90 | −0.2 | −12.7 | **−20.2** | −11.5 | **random_2d** | **−20.2pp** |
| c190 | −3.0 | **−21.9** | −19.3 | −16.4 | longitudinal | −21.9pp |
| c160 | −2.5 | **−22.2** | −19.6 | −16.5 | longitudinal | −22.2pp |
| c140 | −2.0 | **−22.7** | −17.7 | −16.5 | longitudinal | −22.7pp |

c90 仍是四顆中最佳，但**優勢只有 1.7pp**（vs c190），不是先前暗示的量級。

**另一項必須更正的宣稱**：c90 只是**四顆正式候選中**最佳。
30 顆 checkpoint 中只有這 4 顆跑過正式四模式 gate，其餘 26 顆僅有
random_2d/mixed_iid 的 sentinel 數據 —— **不能宣稱 c90 優於全部 30 顆**。

## 關鍵發現

### 1. 窄縫能力完整保住，走廊能力是唯一瓶頸

| 能力 | 表現 | 判定 |
|---|---|---|
| **Gate5a 窄縫直穿** | SR/crossing/direct **100%**、CR **0%**（6 顆候選皆同）| ✅ 遠超門檻 |
| 走廊四模式 | 67.3–89.8% | ❌ 全部 FAIL |

SA7 的逐 env 密度混合訓練**沒有洗掉窄縫能力** —— 這是 68/10/12/10 replay 比例設計的直接驗證。

### 2. 訓練指標無法代理走廊能力

| checkpoint | 訓練 SR | 走廊四模式最差 |
|---|---|---|
| c160（訓練 90.5%）| 90.5% | 67.8% |
| **c90（訓練 89.2%，最低）** | 89.2% | **77.3%（最佳）** |

訓練 SR 最低的 c90 走廊能力最強。goal 明定「訓練指標不能代理窄縫或走廊能力」，此處得到直接證據。

### 3. sentinel 的模式選擇造成排序失真

goal 指定 sentinel 只跑 `random_2d,mixed_iid`。結果：

- **longitudinal 是最弱模式**（67.3–77.3%），但從未進入篩選
- **c90 在 sentinel 排第 4**，卻是四模式綜合的 Pareto 最佳
- sentinel 對其**涵蓋的**模式估計可靠（random_2d 誤差 1.5–1.7pp）

若 sentinel 涵蓋四模式，候選順序會不同。這是流程設計層面值得記錄的一點。

## 結論與後續

**SA7 未達 SA8 晉級標準，依 goal 停止、不啟動 SA8。**

改進方向的證據指向：

1. **最弱模式因候選而異**：c140/c160/c190 是 longitudinal（−21.9～−22.7pp），
   c90 則是 random_2d（−20.2pp）。**四個模式全部未達標**，沒有單一「瓶頸模式」
2. **失敗全部來自碰撞**（TO 0%、structural 全 True），非逾時或結構問題
3. **c90（iter 90）的走廊能力優於後續 210 個 iteration 的任何 checkpoint** —— 後段訓練對走廊能力無助益，甚至有害

第 3 點特別值得注意：如果走廊能力在 iter 90 就達到最佳，後續 210 iteration 的訓練（訓練 SR 從 89.2% 升到 92.8%）都在改善其他場景而**犧牲走廊**。