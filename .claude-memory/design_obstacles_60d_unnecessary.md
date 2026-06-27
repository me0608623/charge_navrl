---
name: obstacles-60d-state-input
description: TopK obstacles 60D 在 WD 是 placeholder、在部署不可用，應從 obs 移除。RL head 和 RNN 在我們和 WD 都拿相同 state。
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

Obstacles 60D（TopK 10×6D [x,y,vx,vy,r,m]）不應存在於 state input。

**事實:**
1. 我們的 RL head 和 RNN 拿相同 113D（都走 `_wd_like_obs()`），RL 額外 +12D preprocess
2. WD 原版也相同（`module_connected.py` L438: `rl_ip = ip.detach()`）
3. WD `spot_state_obs_agent_rate = 0`，60D 幾乎全 placeholder
4. 部署端只有 LiDAR 點雲，無法提供 per-object ground-truth tracking
5. IL 和 WD 的 60D 格式不同（10×6D vs 5×10+10），直接 copy 有語義不匹配

**Why:** 43% 的 RNN 輸入是 noise/placeholder，且部署時無法提供。若 policy 依賴這段，部署會有分佈偏移。

**How to apply:** SA2 前應做消融實驗：移除 60D（113D → 53D）對比。`modular_rnn_models.py` 已有 `USED_OBS_DIM = 79` 的 non-wd layout。長期目標是 env 輸出直接降為 79D。

## ✅ 已實作（2026-05-27）— env 觀測 139D → 79D

已從 `charge_env_cfg_vlp16.py` 的 `ObservationsCfgVLP16` 移除 `obstacle_obs`（PolicyCfg + CriticCfg 兩處），env 觀測 139D → **79D**（ego4+goal2+LiDAR72+time1）。
- 新佈局：time 從 index 138 移到 **78**；舊 [78:138] 60D 障礙物已不存在
- `train_rnn_car_wdclip.py` 的 `POLICY_OBS_INDICES` 改為自動適配（79D 全選 / 139D fallback）
- **網路不變**：extractor_rnn 本來就走 `_select_79d`，rl_input 仍 91D（79+12），只是 env 不再計算 60D（省 compute）
- aux target（wd_aux_targets.py）、WD reward、asymmetric critic（extract_privileged_obs）都從 **scene 實體**抽，不受影響
- ⚠️ **checkpoint 相容性**：79D normalizer 與舊 139D checkpoint 不相容。**SA1_v2（PID 49354）目前仍用 139D 跑**（cfg 已載入記憶體不受影響）；新 fresh run 才用 79D。SA2_v2 從 SA1_v2(139D) checkpoint 續訓會 normalizer 維度不符 — 需重訓 SA1_v2 或重建 normalizer。
- ⚠️ **blast radius**：此 obs cfg 為所有 VLP16 task 共用（NavRL/WD/TCorridor）。`wd_exact_rnn` 模式（需 60D）已不相容 79D，但 v2 不使用。NavRL pipeline（train_charge_ac.py）無 139D hardcode，應可動態適配。

**Obsidian 筆記:** `/home/aa/Documents/Obsidian Vault/bug/2026-05-13_obstacles_60d_design_issue.md`
