# Claude Code 定期監控 Prompt

> 貼給 Claude Code 使用。每 15 分鐘執行一次，回報訓練健康狀態。

---

## 監控 Prompt (直接複製貼上)

```
你正在監控 IsaacLab RNN car 訓練。請執行以下健康檢查並回報。

## 步驟

### 1. 確認訓練存活
```bash
PID=$(pgrep -f 'train_rnn_car_wdclip' | head -1)
echo "PID=$PID"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```
如果 PID 為空，回報「訓練已停止」並結束。

### 2. 抓取最近的訓練數據
```bash
LOG="/home/aa/IsaacLab/logs/rnn_car/sa2_cont2_wdclip.log"

# 最近 10 筆 CHARGE 摘要
grep "CHARGE S" "$LOG" | tail -10

# 最近 5 筆 AUX 數據
grep "AUX(tbptt" "$LOG" | tail -5

# 最近 5 筆 OBS 摘要 (參考用)
grep "OBS S" "$LOG" | tail -5

# 錯誤檢查
grep -ciE "nan|traceback|cuda error|oom|killed|runtimeerror" "$LOG"
```

### 3. 回報格式

用以下格式整理，**只報 CHARGE iteration 的數據** (OBS 是 obstacle policy，指標尺度不同，僅列最後一筆供參考)：

```
═══ SA2 Cont2 Monitor ═══
進度: iter N/800 (剩餘約 X min @ Y fps)
GPU: Z% | M/32607 MiB

最近 CHARGE 數據:
  iter | SR%  | CR%  | R    | ent   | h°  | gV    | ppo     | vf     | sw
  10   | 92.1 | 7.8  | 51.2 | 5.570 | 49  | +0.24 | 0.036   | 0.301  | 0.029
  30   | 66.7 | 32.4 | 31.1 | 5.503 | 63  | +0.14 | -0.073  | 0.419  | 0.031
  ...

最近 AUX 數據:
  iter | loss  | rnn_grad | rnn_delta | ph_delta | fm_delta | VE   | n1d | n2d | valid
  10   | 5.001 | 0.421    | 0.0449    | 0.004    | 0.030    | 0.706| 0   | 0   | 187710
  30   | 5.232 | 0.425    | 0.0437    | 0.004    | 0.029    | 0.583| 0   | 0   | 218176
  ...

OBS 最近: (僅參考，不算警告)
  [iter N] SR=X% CR=X% obs_ent=X

趨勢判讀:
  SR 趨勢: 上升/持平/下降 (最新 vs 前 5 筆平均)
  CR 趨勢: 上升/持平/下降
  VE 趨勢: 上升/持平/下降
  rnn_delta 趨勢: 穩定/增加

狀態: ✅ 正常 / ⚠️ 注意 / ❌ 建議中止
備註: (如有異常，說明原因和建議)
```

### 4. 判讀規則

**只對 CHARGE iteration 判讀，忽略 OBS。**

#### 正常範圍
| 指標 | 正常 | 注意 | 危險 |
|------|------|------|------|
| SR (%) | >80% | 60~80% | <60% 或連降 3 次 |
| CR (%) | <15% | 15~30% | >30% 或連升 3 次 |
| entropy | 5.0~5.6 | 4.0~5.0 或 5.6~5.8 | <4.0 (collapse) 或 >5.8 |
| h (heading°) | 30°~60° | 60°~80° | ≈90° (沒學 goal-directed) |
| VE | 0.4~0.8 | 0.2~0.4 | <0 或 < -0.5 |
| rnn_delta | 0.03~0.06 | 0.06~0.10 | >1.0 (feature drift) |
| rnn_grad | 0.3~0.6 | 0.6~1.0 | >5.0 |
| vf_loss | 0.2~0.5 | 0.5~1.0 | >3.0 |

#### 立即中止條件 (任何一項觸發就建議 kill)
1. n1d > 0 或 n2d > 0 (NaN)
2. rnn_delta > 1.0 或 rnn_feature_mean > 10
3. VE < -0.5 持續 2 次
4. entropy < 3.0 且 SR 連降
5. vf_loss > 3.0
6. 任何 Traceback / CUDA error / OOM

### 5. 特殊注意事項

- **WD 交替訓練**: 每 3 個 iteration 有 1 個 OBS (obstacle policy)，判讀只用 CHARGE 行
- **OBS 的 obs_ent 尺度不同** (正常可到 13+)，不要用 CHARGE 的 entropy 範圍判斷 OBS
- **Curriculum**: 目前 Stage 0 bootstrap，注意是否有 stage 轉換 (S0 -> S1)
- **fps 暴增**: 從 ~1100 跳到 ~4200 是正常的 (JIT warm-up 完成或 bootstrap 結束後 env 更快完成)
```

---

## 附錄: Log 格式參考

### CHARGE iteration
```
[N/800] CHARGE S{stage} | fps=X | R=X SR=X% CR=X% TO=X% | ppo=X vf=X ent=X | gV=+X gΔ=+X h=X° sw=X
  AUX(tbptt L=15): loss=X n1d=X n2d=X valid=X | rnn_grad=X rnn_delta=X ph_delta=X fm_delta=X | VE=X
```

### OBS iteration (obstacle policy，每 3 iter 出現 1 次)
```
[N/800] OBS S{stage} | fps=X | R=X SR=X% CR=X% TO=X% | obs_ppo=X obs_ent=X | gV=+X gΔ=+X h=X° sw=X
```
- OBS iteration **沒有 AUX 行**
- obs_ppo/obs_ent 是 obstacle policy 的 loss/entropy，和 CHARGE 的尺度不同

### WD 交替邏輯
```python
train_charge = (iteration % 3 != 1)  # iteration 為 0-indexed
# [N/800] 的 N = iteration + 1
# 所以 N=1,2,3 -> CHARGE,CHARGE,OBS; N=4,5,6 -> CHARGE,CHARGE,OBS; ...
```

### Curriculum 階段 (Stage 號碼對應)
```
S0 = bootstrap (10 goals, 0 obstacles)
S1 = SA1 (10 goals, 0~1 static, 0~1 dynamic, 0~2 walls)
...漸進到 S7 (1 goal, 7 static, 7 dynamic, 4~8 walls)
```

---

## 修改紀錄

- 2026-05-07: 初版，適用 wd_sa2_cont2_wdclip_s42_ne1024
- 2026-05-07: 加入 WD 交替訓練說明，區分 CHARGE/OBS 判讀規則
