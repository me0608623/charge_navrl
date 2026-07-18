# Gate2 held-out 不可解場景 = CR 假陽性 + 訓練污染

> 定位者：Claude（唯讀審計，2026-07-18 夜）。用戶 seed303 實測回報：內牆太多 + goal 貼牆 + 障礙擠成一團 → RL 物理上無法到達。
> 影響：held-out det CR 卡 12%（Gate2 FAIL）有一截是「場景無解」假陽性；且訓練用同款場景，policy 在無解場景裡學到「反正到不了、亂撞算了」→ 污染訓練訊號。**很可能是 SA6 卡關 + CR 降不下來的共同根因，比 continuation/anti-spin 更根本。**

## 根因鏈

SA6 場景 = 12 靜 + 4 動 + 每 env 2~5 道內牆，塞進小 arena
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

### ② 內牆數太多 — `walls.py:78-79, 119`
```python
min_walls: int = 2, max_walls: int = 5
num_active = torch.randint(min_walls, max_walls + 1, (N,))  # 每 env 2~5 道內牆
```
- SA6 `arena_shape="mix"`（`wd_single_agent_v3e_deploy_dense.py:67`，走廊/正方隨機）→ 走廊形狀 + 密障礙最易造死路。

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
| 3 | `walls.py:78-79`（held-out gate 牆數） | 每 env 2~5 內牆切割空間 + 走廊形狀 = 死路（**用戶點名主因**）| 降 SA6 max_walls（如 2~3）或內牆與障礙共用可行性檢查；**★先確認部署環境有無這些牆** | ★升主凶，待 Codex |
| 4 | goal_dist 約束（`goal_command.py:288,313`）| goal 若採到障礙密集環（dist 3~5m 落 ±5 障礙區）難放 | goal_dist 偏障礙區外環，或密場景放寬約束 | 中 |

## 決定性一步
先確認 SA6 room_size + 印「排斥區/可放區」比值。若 >1 = 物理過密，必須擴 room 或減障礙/牆——修 goal 邏輯（#1）治標，空間不足是治本。

## 驗證方式
修復後 A/B：改版 vs 現況，比 **held-out CR + goal-resample 失敗率 + 部署真實性**。守住底線：不偏離 12×12/20 障礙的部署密度目標（牆是額外的，減牆/擴 room OK；砍障礙謹慎）。
