---
name: MARL Migration 4 Phases
description: ManagerBasedRLEnv → DirectMARLEnv 遷移，4 階段完成 N 台車 parameter sharing + RNN + curriculum
type: project
---

# MARL Migration: ManagerBased → DirectMARLEnv (4 Phases)

完成日期: 2026-04-14

## 動機
原本用 ManagerBasedRLEnv + virtual spots hack 做 multi-agent，
改為 Isaac Lab 原生 DirectMARLEnv，匹配 WarpDrive `train_rnn_car.py` 架構。

## Phase 1: 單車 DirectMARLEnv ✅
- 檔案: `direct_marl/charge_marl_env.py`
- DirectMARLChargeSceneCfg + DirectMARLChargeCfg + DirectMARLChargeEnv
- 6 abstract methods: `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_get_states`
- 139D obs, MultiDiscrete[19,19] action, NavRL rewards
- Ground plane 用 pxr API（避免 TerrainImporter Nucleus 依賴）

## Phase 2: N 台車擴展 ✅
- `num_cars=6` 預設，動態建立 N Articulations + N LiDARs + N ContactSensors
- 所有 state tensor `[E] → [E, N]`
- TopK obstacles 包含其他 N-1 台車
- Car-to-car geometric collision detection
- **WarpDrive-style 連續 episode**: collision/goal → respawn 該車，不終止 episode，只有 shared timeout 結束

## Phase 3: Parameter Sharing + RNN + Aux Loss ✅
- 檔案: `train_marl_rnn.py`
- Parameter Sharing: `obs_dict → dict_to_flat [E*N, 139] → 共用 policy → flat_to_dict → action_dict`
- RNN: `RNNStateManager(E*N, hidden_dim=30)`
- Per-car respawn → `info["respawn_mask"]` → reset 該車 hidden to zero
- 複用 `modular_rnn_models.py`: LidarStateExtractor(96D) + PreprocessRNN(12D) + PolicyHead(91→38) + ValueHead(91→1)
- Auxiliary LiDAR prediction loss（optional，`--aux_lr > 0` 啟用）

## Phase 4: Curriculum Adaptation ✅
- 檔案: `direct_marl/marl_curriculum.py`
- `MARLCurriculum` class — 獨立 stage 管理器（不依賴 event/termination manager）
- per-car events 統計: goal/collision/car_collision → success_rate
- 複用 `CURRICULUM_CONFIGS` 的 stage 定義（baseline_v1, warp_drive_v1 等）
- `apply_to_env()`: 動態修改 obstacles, episode_length, speed_range, reward weights
- 整合到 training loop: 每 iteration 更新 stage，stage 變更時 reset optimizer momentum

## 關鍵檔案
- `source/.../charge_skrl/direct_marl/charge_marl_env.py` — env 實作
- `source/.../charge_skrl/direct_marl/marl_curriculum.py` — curriculum adapter
- `source/.../charge_skrl/direct_marl/__init__.py` — gym 註冊
- `scripts/.../skrl/train_marl_rnn.py` — 訓練腳本
- `/tmp/test_marl_env.py` — smoke test

## 啟動指令
```bash
# 訓練（含 curriculum）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_marl_rnn.py \
  --num_envs 1024 --num_cars 6 --headless \
  --curriculum_version baseline_v1 --initial_stage 1 \
  --run_name marl_rnn_v1

# 訓練（固定難度）
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_marl_rnn.py \
  --num_envs 1024 --num_cars 6 --headless --no_curriculum \
  --run_name marl_rnn_fixed

# Smoke test
./isaaclab.sh -p /tmp/test_marl_env.py --num_envs 4 --headless
```

**Why:** 原生 DirectMARLEnv 比 virtual spots hack 更乾淨，支援真正的多車物理互動（LiDAR 可見、碰撞偵測）。
**How to apply:** 新的 MARL 實驗用 `train_marl_rnn.py`，舊的 ManagerBased 實驗用 `train_rnn_car.py`。
