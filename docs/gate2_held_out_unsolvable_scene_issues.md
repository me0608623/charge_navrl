# Gate2 held-out 不可解場景 = CR 假陽性 + 訓練污染

> 定位者：Claude（唯讀審計，2026-07-18 夜）。修復與 simulator 複驗由 Codex 完成。
> 影響：held-out det CR 卡 12%（Gate2 FAIL）有一截是「場景無解」假陽性；且訓練用同款場景，policy 在無解場景裡學到「反正到不了、亂撞算了」→ 污染訓練訊號。**很可能是 SA6 卡關 + CR 降不下來的共同根因，比 continuation/anti-spin 更根本。**

## 根因鏈

修復前 SA6 場景 = 12 靜 + 4 動 + 每 env 2~3 道長內牆
→ 障礙「goal 排斥區」總面積可能 > goal 可放區
→ goal 採樣 max_attempts 次找不到「避牆 + 避障礙 + 夠遠」的點
→ fallback 用 best 候選（**只避牆、不避障礙**）→ goal 貼障礙 = 不可達
→ 或連 best 都無 → goal 放機器人旁 +1.0（trivial 秒到）
→ held-out 大量「不可解」或「trivial」場景 → CR 假陽性 + 訓練污染

## 三個疊加根因（附行號）

### ① goal 採樣 fallback 缺陷 — `goal_command.py:375`
```python
# line 363: 完整合法性 = 四項全過
valid_goals = within_walls & clear_of_obstacles & clear_of_robot & clear_of_walls
# line 375: fallback「best 候選」資格只三項 ——★漏了 clear_of_obstacles
eligible = within_walls & clear_of_robot & clear_of_walls
```
- fallback（line 386-397）退回 best 時，best 只保證避牆 / 離機器人遠 / 邊界內，**不保證避障礙** → goal 貼障礙（min_clearance 可能為負）。
- 兜底（line 398-401）：連 best 都無 → 放機器人旁 +1.0（trivial goal）。
- clamp（line 403-409）：推到 valid_boundary 邊緣 → 貼牆。

### ② 內牆數太多 — curriculum 展開值
```python
SA6: min_walls=2, max_walls=3, target_wall_length=4.5
```
- `walls.py` 的 2~5 是函式預設值，不是 frozen SA6 runtime 值。
- SA6 `arena_shape="mix"` 目前仍是佔位鍵，runtime 為 square fallback，不是已實作的形狀 DR。

### ③ 障礙密度 vs 空間 — `deploy_dense` `_STATIC_RAMP`/`_DYNAMIC_RAMP`
- SA6 = 12 靜 + 4 動 = **16 障礙**（line 36 / 48）
- 每障礙 goal 排斥半徑 = `obstacle_safe_distance(1.0) + radius(~0.35)` = **1.35m**（`goal_command.py:304`）
- 16 障礙 × π×1.35² ≈ **91㎡ 排斥區**（扣重疊 ~60-70㎡）

**✅ SA6 room_size 已查證：= 7.0（半邊長）→ 場景 14×14m = 196㎡**
（`e2e_k8_future_frozen_base.py:128` `ROOM_SIZE_BY_STAGE[6]=7.0`；`train_rnn_car_wdclip.py:519-524`
確認 room_size 是**半邊長**，場景 = 2×room_size 見方，SA8 部署=6=12×12。之前把 7 誤當邊長=7×7，錯。）

量化（room_size=7）：
- 障礙 spawn 區 ±(room_size−2)=±5 → 100㎡；goal 可放 valid ±6.5 → 169㎡
- 16 障礙排斥區（半徑1.35）92㎡ 扣重疊 ~64㎡，集中在障礙區 100㎡ 內
- goal 可放障礙區外環 ~69㎡ + 障礙區內剩 ~36㎡ = **~100㎡ 可放，room 不算太小**

★**根因 ②「room 太小」推翻**——14×14 對 16 障礙是合理部署密度。真正主因：
(a) goal_dist 約束若強制 goal 落在障礙密集環（dist 3~5m 在障礙區 ±5 內）→ 難避障礙；
(b) **內牆（根因 ③）** 2~5 道切割空間 + 走廊形狀 = 死路（用戶明確點名「內牆太多」）。

## 修復清單（優先序）

| # | 檔案:行號 | 問題 | 建議 | 狀態 |
|---|-----------|------|------|------|
| 1 | `goal_command.py:389` | fallback goal 貼障礙不可達 | 接受門檻 `best_min_clearance` 0→0.45（碰撞門檻），貼障礙候選走 else 兜底 | **✅ 已修 push 696709a** |
| 2 | ~~room 太小~~ | 查證推翻：room_size 7 = **14×14m**（半邊長），對 16 障礙合理 | — | ~~推翻~~ |
| 3 | `e2e_final20_v1.py` `_WALL_TRIM` | 最多 3 道、每道 4.5m + 16 障礙提高切斷風險 | SA6-8 覆寫 `walls_max=2`, `wall_length=3.5` | **✅ 已修 push f0717760** |
| 4 | goal_dist 約束（`goal_command.py:288,313`）| goal 若採到障礙密集環（dist 3~5m 落 ±5 障礙區）難放 | goal_dist 偏障礙區外環，或密場景放寬約束 | 中 |

## 2026-07-18 simulator 複驗

新 SA6 runtime 明確印出 `牆壁=2~2 牆長=3.5m`。checkpoint_19200、seed303、64 env、
1200 steps 共 2,291 個完成回合：SR 92.1%、牆撞 5.1%、障礙撞 2.7%、TO 0.09%。

以 0.15m occupancy grid，將牆與障礙膨脹 `robot_radius 0.35 + buffer 0.10m`：

- walls-only：可達 100%，平均自由區 66.9%
- walls + 12 static：可達 100%，平均自由區 59.5%
- walls + initial 16 obstacles：可達 100%，平均自由區 56.7%
- goal fallback 0 次；no-best/trivial fallback 0 次；起點/goal blocked 0 次

診斷工具：`play_rnn_car.py --solvability_audit_output ...`。正式 Gate2 已為三個 held-out seed
自動輸出 `solvability_s*.json`，但可達性目前只作診斷，不改 PASS/FAIL。

## SA6 新場景 checkpoint 裁決

兩顆 checkpoint 均以 commit `f0717760` 的相同新場景跑完整四閘：

| checkpoint | Gate2 SR / CR / TO | Gate3 SR / CR | SA7 preview | path-recovery median |ω| |
|---|---|---|---|---|
| cont4 `checkpoint_6400` | 93.7% / 5.2% / 1.1% | 99.0% / 0.9% | 87.4% | 0.267 rad/s |
| observe150 `checkpoint_19200` | 92.1% / 7.7% / 0.2% | 97.4% / 2.6% | 84.0% | 0.133 rad/s |

兩者均四閘 PASS。continuation 改善回正與 timeout，但使牆撞增加；部署安全主線優先較低 CR，
因此 SA7 從真正 cont4 `checkpoint_6400` resume。SA7 run：
`sa7_e2e_k8_future_frozen_walltrim_v1_ne1024_s42_from_sa6c4`，W&B `luniwvhq`。

## 2026-07-19 SA7 no-best/trivial-goal 修復與 checkpoint 裁決

SA7 跑到 iter730 時，log 累計約 65.9 萬次 goal fallback，其中 60.3 萬次為 no-best。
根因是 goal sampler 把隱藏於 `z=-10` 或含非有限座標的 obstacle slot 納入距離矩陣；單一
`NaN` 即使 50 次候選的 `all()`/`min()` 全失效，最後落入車旁 1m trivial goal，形成立即成功、
重採樣、再成功的循環。這同時污染訓練 SR 並將 log 撐到 125MB、fps 約 2470→1470。

修正：

- goal clearance 只計 `Z>0` 且 XYZ finite 的可見 obstacle；隱藏/invalid slot 距離視為 `∞`。
- 移除 no-best 與 fallback-in-wall 的車旁 goal；若真找不到可達候選則 fail-fast，禁止靜默污染 PPO。
- Gate3 在 controller 接管前使用 1.5~3.0m 全方向 bootstrap goal，且不先放隨機內牆；controller
  接手後仍恢復正式 x=6m goal、兩面 12m 長牆與 2D blocker，任務難度不變。
- `validate_checkpoint.sh` 現在會把 simulator traceback 或缺少必要 dump 視為硬錯誤。

64-env/400-step SA7 smoke 共 563 回合：fallback=0、no-best=0、SR 89.9%、CR 10.1%。
其後在同一修正版場景完整比較：

| checkpoint | Gate2 SR / CR / TO | Gate3 SR / CR / TO | Δ|ω|@1.5~2m | SA8 preview | 裁決 |
|---|---|---|---|---|---|
| `checkpoint_76800` | 87.6% / 11.8% / 0.6% | 94.3% / 5.7% / 0.0% | +0.154 | 74.2% | Gate4 FAIL |
| `checkpoint_83200` | 87.2% / 12.4% / 0.4% | 94.6% / 5.4% / 0.0% | +0.226 | 79.8% | **四閘全 PASS** |

因此後續若 resume SA7，選 `checkpoint_83200`；`checkpoint_76800` 雖 Gate2 略優，但 SA8 preview
未達 75% 且 Gate3 方向觸發較弱。原 SA7 進程已停止，未自動 resume。

## 驗證方式
舊 2~3 道長牆 held-out CR 已作廢，不可與新 Gate2 直接比較。後續看三個 held-out seed 的
**CR + goal-resample 失敗率 + no-best 率 + 可達率**；守住 12×12/20 障礙部署密度，不砍障礙。
