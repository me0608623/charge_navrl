---
name: run-name
description: v2 起 run_name 統一格式 sa<N>_a2c_aux_<感測器>_ne<場景數>_s<seed>
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

## ⭐ 現行規範（2026-05-28 起）— v2 階段一律使用

**格式**：`sa<N>_a2c_aux_<感測器>_ne<場景數>_s<seed>`

**範例**：
- `sa1_a2c_aux_vlp16_ne1024_s42`
- `sa2_a2c_aux_vlp16_ne1024_s42`
- `sa6_a2c_aux_vlp16_ne1024_s42`

**欄位語意**：
| 欄位 | 範例 | 說明 |
|------|------|------|
| `sa<N>` | sa1 ~ sa8 | curriculum stage（場景難度軸）|
| `a2c_aux` | a2c_aux | 演算法 = A2C + auxiliary task（v2 固定）|
| `<感測器>` | vlp16, rgbd, depth | LiDAR / 攝影機型號 |
| `ne<N>` | ne1024 | num_envs（平行場景數）|
| `s<N>` | s42 | random seed |

**特殊變體**（在 N 後加 suffix）：
- T 走廊：`sa5tc_a2c_aux_vlp16_ne1024_s42`
- v2 重訓：`sa1v2_a2c_aux_vlp16_ne1024_s42`（舊：`sa1_v2_ne1024_s42`）

**Why:** 2026-05-28 用戶明確指示「以後訓練 run name 取名規則為 `sa*_a2c_aux_感測器_ne1024(場景數目)_s42`」。固定 5 個欄位讓所有 v2 run 結構一致，便於批次掃描、表格對齊與報告生成。

**How to apply:**
- 啟動訓練前一律先核對 `run_name` 是否符合此格式（5 欄位以底線分隔）
- 不再使用舊長描述式（如 `wd_sa_v1_rawfc_freeze_env512_...`）
- 不再使用 `sa1_v2_ne1024_s42` 這種無 algo/sensor 欄位的簡短式
- 演算法非 A2C+aux 時改 `_<algo>_<aux>_`（如 `_ppo_noaux_`、`_a2c_noaux_`）
- 場景數可改 `ne256`、`ne4096`，數字必為實際 num_envs
- 寫到 [[CLAUDE]] (isaaclab_v2) + 任何 launch script 範本都要套這格式

**禁止**: test1, new, final, finalfinal, baseline_s1 等模糊名稱

---

## 舊規範（v1 baseline 期，供 retrofit 對照）

**格式範例**：
`wd_sa_v1_rawfc_freeze_env512_p1_planD_pureRL_disableaux_lidarfix_resume_u600_to_u900_0502`

舊命名元素（按序）：
- 專案/方法: `wd` (warp drive), `navrl`, `abl` (ablation)
- curriculum: `sa_v1` (single agent v1)
- 架構: `rawfc`, `wdexact`, `3branch`
- 特殊模式: `freeze`, `pureRL`, `disableaux`
- 規模: `env512`, `env4096`
- 階段: `p1`, `p1to5`
- 計劃: `planD`, `planE`
- 特殊 flags: `lidarfix`, `lidarnofix`
- 來源: `resume`, `scratch`
- 進度: `u600`, `u600_to_u900`
- 日期: `0502` (MMDD)
