---
name: cli_versions
description: NavRL reward_mode / curriculum_version / CLI flags 各版本差異與演進歷史
type: reference
---

## reward_mode 演進

| 版本 | CLI 值 | 核心改動 |
|------|--------|---------|
| v1 | `navrl_ground_v1` | NavRL-Ground 初版：soft gate + soft scale |
| v2 | `navrl_ground_v2` | 移除 gate/scale + alive + 權重平衡 |
| v3 | `navrl_ground_v3` | v2 + MAX_OBSTACLES 20 + curriculum 控制權重 |
| v4 | `navrl_ground_v4` | v3 + reaching_goal 500 + alive=0 |
| v5 | `navrl_ground_v5` | v4 + collision -50→-100 |

## curriculum_version 演進

| 版本 | CLI 值 | 核心差異 |
|------|--------|---------|
| goal_first_v2 | `goal_first_v2` | 12 stage, density-adaptive weights |
| goal_first_v3 | `goal_first_v3` | v2 + Stage 3-6 ss 局部提高 |

## 其他 CLI flags

- `--dynamic_safety_mode closing_risk` — 加入接近速度風險
- `--no_walls` — 移除內牆
- `--use_cadn` — CADN 觀測預處理
- `--goal_vel_use_soft_gate` — 重啟 velocity soft gate
- `--directional_gate` — v16: 方向性 gate，只看 goal 方向 cone 的 LiDAR (±30°)
- `--gate_cone_half_bins N` — cone 半寬 (bins)，default=6
- `--gate_cone_bottom_k N` — cone 內 bottom-k，default=3
- `--gate_omni_blend F` — omnidirectional 混合比例，default=0.2

## 歷史 runs 關鍵成效

| Run | reward | curriculum | SR | CR | speed | 備註 |
|-----|--------|------------|:--:|:--:|:-----:|------|
| v3 seed1 | v3 | goal_first_v2 | 46% | 39% | 0.28 | 目標前停滯 |
| v4 seed1 | v4 | goal_first_v2 | 74% | 24% | 0.60 | 速度翻倍 |
| v5 seed1 | v5 | goal_first_v3 | 71% | 23% | 0.58 | CR↓但TO↑ |
| v15 seed1 | v1 (ground) | open_ended_v1 | 71% (S8) | 0.6% | 0.49 | bootstrap+kl0.1+gate0.15+time-0.2; S8 retreat 63.8% |
| v16 (planned) | v1 + dir_gate | open_ended_v1 | — | — | — | 方向性gate解決retreat-dominant |
