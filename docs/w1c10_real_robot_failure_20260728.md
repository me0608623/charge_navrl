# W1-c10 實車失敗診斷（2026-07-28）

## 結論

`w1c10_k8_e2e_1280.ts` 已列為 **REAL-ROBOT FAIL / 禁止再次部署**。

這次失敗不能只解讀成「W1 訓練不足」。目前已同時確認：

1. W1 特有的倒車解碼契約不一致。
2. 本次部署繞過 VO safety，和舊模型不是等條件比較。
3. TF 定位曾發生物理上不可能的姿態跳變。
4. W1 訓練完全沒有致動延遲、馬達滯後或 LiDAR noise。
5. W1 在實車輸入下本身也比舊 E2E K8 模型更常滿舵與換向。

因此正確順序是：**先修部署契約與安全層，再建立從 SA1 開始的
sim-to-real 新血緣**。直接延長 W1 或直接把同一份錯誤接線帶進 SA1，
都無法解決問題。

## 安全狀態

- 車端目前沒有 policy inference process。
- 不重新啟動 `w1c10_k8_e2e_1280.ts`。
- 模型 SHA-256：
  `6b8b54239e6255062154d4b6d4c562626b2606fe4e3d193e6db329d368e93556`
- 車端檔案與本機匯出檔 checksum 相同，排除「上傳錯模型」。

## 實車證據

來源：

- `/home/aa/rover_rl/logs/deploy_20260728_173557.log`
- `/home/aa/rover_rl/logs/diag/diag_20260728_173722/diag_20260728_173722.csv`
- 57.34 秒，1,145 筆 20 Hz 記錄。

### W1 動作不穩

| 指標 | W1 | 舊 E2E K8 對照 |
|---|---:|---:|
| `|rl_w| >= 0.83` | 33.7% | 26.4% |
| 明確左右換向 | 13 次 | 2 次 |
| `mean(|rl_w|)` | 0.568 rad/s | 0.498 rad/s |
| VO safety | 關閉 | 開啟 |
| raw observation | 83D | 79D |
| reverse | 允許 | 禁止 |

舊紀錄不是同日同場的嚴格因果對照，但足以支持使用者的觀察：W1 的激烈
行為不是所有模型都一樣。

### 定位跳變

記錄中 `map_yaw` 曾在約 0.05 秒跳近 97.4 度；同一時段的底盤角速度不可能
產生這個轉角。`policy_goal_ang_deg` 也隨之快速改變。

目前 `_robot_pose_in_map()` 直接接受最新 `map -> base` TF；當
`require_ndt=false` 時，即使 NDT/TF 不穩定也會繼續推論。程式雖有
`MapOdomOffsetTracker` 的跳變拒絕邏輯，但這條導航取姿路徑沒有使用它。

這是共同的上游風險，但不是 W1 特有問題的充分解釋。較合理的判讀是：
**W1 對錯誤姿態、延遲與 action-history mismatch 更敏感。**

## 已確認的部署 bug

### 倒車界線沒有進入 decoder

訓練端的速度界線是：

```text
vₜ₊₁ = clip(vₜ + aₜ Δt, -ρ_rev v_max, +v_max)
ρ_rev = 0.2
```

但車端 `ActionParams` 沒有 `reverse_velocity_scale`，decoder 使用對稱的
`[-v_max, +v_max]`。`reverse_velocity_scale=0.2` 只在後段 `CmdFilter`
做一次 clamp，已經太晚：

1. policy target 已進入訓練未見過的倒車區域；
2. 83D action history 記錄的是 filter 前 target；
3. 實際送底盤的命令又是 filter 後值。

實車證據：

- `rl_v` 最低約 `-0.557 m/s`
- `sent_v` 被 filter 夾到 `-0.200 m/s`
- `act_v` 曾到約 `-0.617 m/s`

訓練規格的最低速度應是 `-0.2 m/s`；若同時使用 `speed_rate=0.7`，
物理上對應的界線還應重新一致定義，不能讓 decoder、history、filter 各用一套。

`sent_v=-0.2` 卻量到 `act_v≈-0.6` 也顯示倒車通道需要獨立 sysid；在釐清前，
實車 RL 應禁止倒車。

### Safety 路徑不同

本次 W1：

```text
policy -> /input/nav_cmd_vel -> mux -> chassis
VO safety = false
```

舊 E2E K8 紀錄：

```text
policy -> /rover_rl/cmd_vel_desired -> VO -> /input/nav_cmd_vel
VO safety = true
```

所以「W1 撞牆、舊模型沒撞」不能只歸因於 checkpoint。W1 的直接輸出也讓
任何滿舵、倒車或定位跳變完全落到底盤。

## 致動器模型問題

W1 訓練值：

- `enable_actuator_dr=False`
- `lidar_no_noise=True`
- `obs_delay_steps=(0,0)`

實車則至少包含：

- 約 200 ms command delay；
- 車端額外 low-pass/slew filter；
- 底盤速度追蹤滯後；
- 感測與定位誤差。

83D policy 看的是 issued-command history。要讓 delayed system 近似 Markov，
可用：

```text
s_tilde = (sₜ, uₜ₋₁, uₜ₋₂)
d ~ U{0,1,2}
```

這個歷史規格是合理的，但 W1 從未在 `issued != applied` 的條件下學過。

目前 bridge 的馬達模型又只有單一 scalar `alpha=0.3`：

```text
yₜ = α uₜ₋d + (1 - α) yₜ₋₁
```

在 `Δt=0.2 s` 時，`α=0.3` 對應時間常數約 0.56 秒，不能由「dead time
約 200 ms」直接推出；線速度與角速度也不應被迫共用同一個 alpha。

新血緣至少應改成：

```text
vₜ = α_v uᵛₜ₋d + (1 - α_v) vₜ₋₁
ωₜ = α_ω uʷₜ₋d + (1 - α_ω) ωₜ₋₁
d ~ U{0,1,2}
```

其中 `α_v`、`α_ω`、velocity scale 必須由架空或低速 step-response sysid
取得，而不是沿用未校準常數。

## SA1 新血緣建議

### 啟動前阻擋

1. 修正 decoder 的 reverse bound，補 train/export/car 三端 parity test。
2. 決定唯一的 issued-command 定義；history、filter、diag 必須引用同一層。
3. 增加 TF pose-jump fail-closed：超過速度可解釋範圍就輸出零，不把跳變餵 policy。
4. 實車初期強制 `reverse=false`、VO/shield 開啟。
5. K8 policy 若部署 `speed_rate != 1`，必須先證明 frame-history 與 ego/action
   尺度仍同分布；否則訓練與部署固定 `speed_rate=1`。

### 訓練原則

主線應從 SA1 重新建立，而不是只做 W1 後接 10-iteration bridge。原因是
W1 的 encoder/policy 已在「clean LiDAR、command=applied、無 delay」下形成特徵
與控制習慣，短 bridge 已顯示只能小幅改善固定延遲 gate。

但「從 SA1 注入」不等於第一步就把所有 noise 開到最大：

1. SA1 從第一天就看見 actuator delay，最終支援固定規格 `U{0,1,2}`。
2. 先用量測中心值與窄範圍 lag/scale，再分階段放寬。
3. LiDAR noise 使用實車量測分布，從 mild ramp 到完整範圍。
4. physics DR、LiDAR DR、actuator DR 分開記帳，避免一次改多因子後無法歸因。
5. `obs_delay` 沒有獨立量測前保持關閉，不能把所有 latency 重複算兩次。

### 必要 gates

1. **Action contract gate**：所有 19x19 action、正反向邊界、slew 與 history
   train/export/car 逐值一致。
2. **Fixed-delay gate**：`d=0/1/2` 分別評估，不只評混合平均。
3. **Clean-retention gate**：加入 DR 後，無延遲 clean 能力不得崩潰。
4. **Stability gate**：滿舵率、左右換向率、`omega RMS`、主頻與倒車占比。
5. **Pose-jump gate**：人工注入 goal/TF 跳變時必須 fail-closed，不得滿舵衝出。
6. **Export parity gate**：同一 observation sequence 的 Python checkpoint 與
   TorchScript logits/action 完全一致。
7. **實車分級 gate**：motor-disabled replay -> 架空 -> `v/omega` 外層低限速 ->
   VO 開啟低速短線；任何一級異常立即停止，不直接進走廊全速測試。

## 決策

- W1-c10：永久禁止作正式實車候選。
- c10/c15 actuator bridge：維持 `NOT_A_DEPLOYMENT_CANDIDATE`。
- SA7/SA7.1：既有 gate 結論不因本事故改寫。
- 新工作主線：部署契約修復 + sysid -> SA1 sim-to-real 新血緣。
- 在上述前置完成前，不啟動長訓練，也不再上車測 W1。
