---
name: v22 PhysX OOM Crash 與 6144 envs 風險
description: 6144 envs + 50 obstacles + 1h10min 訓練時間導致 PhysX GPU narrowphase kernel launch 失敗 (CUDA error 717)
type: project
---

# v22 PhysX OOM Crash 紀錄 (2026-04-08)

## 事件
v22 (M1.4 ds rebalance) 在 1h10min / Stage 2 / 16k steps 時 PhysX 崩潰：
- `GPU prepareLostFoundPairs_Stage1 fail to launch kernel`
- `GPU convexCoreConvexNphase_Kernel fail to launch`
- `Synchronizing GPU Narrowphase failed! 717` (= cudaErrorIllegalAddress)
- `Cuda context manager error, simulation will be stopped`

崩潰瞬間 GPU mem = **31.5 / 32.6 GB (96.6% full)**

## 早期警告 (init 時就出現)
```
PhysX error: The application needs to increase PxGpuDynamicsMemoryConfig::foundLostPairsCapacity to 36218880, otherwise, the simulation will miss interactions
```
這個 init warning 是 precursor — `foundLostPairsCapacity` 預設不夠 → 累積到觸發 OOM。

## 根因分析
- **6144 envs** × 50 obstacles 各 visible/hidden state
- 加上 wall_layout per-env (12 walls)
- LiDAR ray cast 5760 rays/env = 35.4M rays
- 每 step 寫入 dynamic obstacles `move_obstacles_vectorized`
- 累積 1h10min 後 PhysX 內部 buffer 滿到 OOM

## v22 收集到的數據 (崩潰前 20 rows)
- Stage 1 → 2 達成
- SR 57% → 99.8% (warm start 從 v21 best_agent 起步)
- speed 0.49 → 0.62
- KL 3.4-4.9 (warm start shock 沒 settle)
- ds reward = 0 (還沒到 Stage 5+ 有 dynamic 階段)
- **dynamic obstacle 階段沒測試到** — M1.4 ds boost 沒驗證機會

## 教訓
1. **6144 envs 對 charge_skrl + 50 obstacles 是極限**，安全 budget 是 4096
2. **PhysX init warning 不能忽略** — `foundLostPairsCapacity` 警告 = 將要 OOM
3. **長時間訓練的 memory leak**: 70min 內累積到 96%+ GPU
4. **CUDA error 717 不可恢復** — process stuck 在 sleep state，必須 kill -9

## 應對策略
- **預設 num_envs 改為 4096** (charge_skrl 訓練)
- 超過 6 小時的訓練 **必須** 用 ≤ 4096
- v23 後嘗試 6144 之前先測 30min 確認 PhysX 警告不再出現
- 監控 nvidia-smi memory usage 趨勢（線性上升 → 預警）

## 此事件對 M1.4 結論
**v22 6144 envs run 不算 valid baseline** — 在 dynamic 階段前就掛了。
重啟 v22 with 4096 envs 才是真實 reactive boost 對照組。
