---
name: project_head_on_behavior
description: 新增 head_on(直線迎面)障礙行為到 SA3-SA5 curriculum,對症 SA4「晚反應撞動態」;現有 crossing/near_miss 都不正面來車,故 policy 沒練過提早避讓正面障礙
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**動機（2026-06-24，用戶要求）**：訓練「障礙物直直朝車子走，讓 RL 學會提早避開」。查 `rule_behaviors.py` 確認現有 6 種動態行為**沒有一個正面來車**：path_crossing 朝場景中心(spawn 時不知 robot 位置)、near_miss **保證不碰撞**(故意擦過)、horizontal/corridor_crossing 都橫向穿越。→ 這正是 SA4 診斷「晚反應全速撞動態」的場景根因(policy 從沒練過正面接近的障礙)。

**各 Stage 障礙數據（v3e curriculum 源碼實值，給用戶報告過）**：
| Stage | goals | static | dynamic(min–max) | obs_speed | episode | 動態 behavior_mix(原始) |
|------|------|------|------|------|------|------|
| SA1/SA2 | 10 | 1 | 1–2 | 0.80 | 60s | patrol .60/random_walk .40 |
| SA3 | 6 | 3 | 3–4 | 0.85 | 60s | patrol/rw/static/horizontal_crossing |
| SA4 | 2 | 3 | 4–5 | 0.85 | 60s | +path_crossing |
| SA5 | 4 | 3 | 3–5 | 0.85 | 75s | +corridor_crossing |
| SA6 | 3 | … | … | 0.90 | 90s | +near_miss |

**實作（2026-06-24，已完成+驗證 py_compile/權重和=1.0）** `head_on` = 第 9 號行為：
- **設計**：spawn 在 LiDAR 邊界(5.5–8m)隨機方位待命(velocity=0)→ 等 activation_delay(1–3s)→ **激活當下對準 robot 當前位置算一次方向**(±8° jitter)→ 之後**等速直線**衝過去,不再重新瞄準。
- ⚠️ **刻意非持續 homing**(只激活瞬間瞄一次)→ 軌跡固定、符合「障礙不依賴 robot 動作」設計原則 + 避免與致動延遲疊成舞龍舞獅極限環([[finding_cmd_delay_limit_cycle]])。
- 用戶決定：**SA3→SA5 都加 head_on 0.20**(重新正規化其他權重,和=1.0);**只加場景、先不開減速 reward**(單變因,看 policy 能否自學提早避)。
- 改的檔案(5 處)：
  1. `obstacle_agent/behavior_config.py`:BEHAVIOR_HEAD_ON=9 + BEHAVIOR_NAMES[9] + NUM_BEHAVIOR_TYPES=10 + HeadOnConfig dataclass + BehaviorConfig.head_on。
  2. `mdp/events/rule_behaviors.py`:import + spawn_head_on + step_head_on(末尾)。
  3. `mdp/events/behavior_scheduler.py`:import + ho_ state tensors + step() 開頭快取 `self._robot_local_xy`(robot.data.root_pos_w-env_origins) + ho_mask dispatch + _spawn_single 加 BEHAVIOR_HEAD_ON 分支。
  4. `curriculum/phases/wd_single_agent_v3.py`:SA3/SA4/SA5 behavior_mix 加 head_on 0.20 + speed_overrides(SA3 .30-.55 / SA4 .30-.60 / SA5 .35-.65)。v3e deep-copy v3 故自動繼承。
- ★生效時機：SA3 launch(wd_sa3_v3f→v3e curriculum)即自動含 head_on;SA2(stage2,未改)+ 已在跑的 process 不受影響。
- 待驗證(runtime)：SA3 啟動後 log 應出現 head_on 障礙、無 crash;觀察 SR/CR 是否因正面來車先掉再學會提早避(預期 CR 暫升再降)。

相關：[[project_sa4_v3f_completion_diagnosis]] [[project_reluFix_retrain_log]] [[obs_agent_behavior_config_v2]] [[finding_cmd_delay_limit_cycle]]
