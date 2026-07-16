"""warp_drive_single_agent_v3 — 單車 WD phase/task registry (v3 with smoothness)

這個檔案是 `warp_drive_single_agent_v3` curriculum version 的「參數設定部門」。

v3 與 v1 的差異 (2026-06-08):
- LiDAR r_min 從 0.9 改為 0.25（在 charge_env_cfg_vlp16.py，配 VLP-16 實測）
- 全 stage 加 reward["penalty_smoothness"] = 0.005 → 抗單幀抽動
- 從頭訓練，不能 resume v1/v2 ckpt（obs distribution 不同）

設計目標同 v1。

設計目標：
1. 每個 phase 不只是難度 stage，而是一個完整 task spec。
2. 每個 task 集中定義：
   - scene        場景與 episode / gamma
   - reward       WD sparse reward 相關設定
   - exploration  每 phase 探索強度
   - behavior     rule-based obstacle behavior / speed overrides
   - trainer      允許 phase 微調的 trainer 超參（只同步安全欄位）
   - transition   升降階條件
3. `train_rnn_car_wdclip.py` 保持主要訓練邏輯；
   本檔負責提供每個 phase/task 的單一真實來源 (single source of truth)。

相容性原則：
- 為了不破壞既有 IsaacLab / curriculum 架構，
  本檔最後仍輸出 legacy flat stage dict 給 `goal_obstacle_curriculum.py` 使用。
- 也就是說：人類主要讀/改的是巢狀 schema；
  系統實際吃的是 bridge 後的 flat schema。

閱讀方式：
- 若你想直接調 task 定義，優先修改 STAGES。
- 若你想知道訓練時實際傳出去的欄位，請看最下面 `_flatten_phase()`。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# 全域設定（適用所有 phase）
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL = {
    # 障礙物控制模式（適用所有 phase）。
    # "rule_based": BehaviorScheduler 控制（patrol/random_walk 等確定性行為，無 NN）
    # "learned":    使用 learned FC policy（ObstaclePolicyFC），每 train_goal_rate 次交替訓練
    # "scripted":   環境內建 move_obstacles_vectorized（目標導向移動）
    "obstacle_mode": "rule_based",

    # 升階需要連續滿足條件幾次。
    # 例如設 5 代表：不是單次 SR 達標就升，而是要穩定達標 5 個檢查窗口。
    "upgrade_pass_required": 5,

    # 升階後是否清空 outcome window。
    # True = 升到新 phase 後重新累積新階段的 SR/CR/TO 統計，避免舊階段資料污染。
    "clear_window_on_promote": True,

    # 只有這些 trainer 參數允許在 phase transition 時由 training loop 動態同步。
    # 這些都是「不改 tensor shape / 不改網路結構」的安全欄位。
    "trainer_sync_allowlist": [
        "lr",                     # RL optimizer learning rate
        "rnn_lr",                 # Aux/RNN optimizer learning rate
        "vf_coeff",               # critic loss 權重
        "max_grad_norm",          # RL grad clip 上限
        "aux_grad_clip",          # aux grad clip 上限
        "wd_actor_update_clip",   # WD-style actor grad cap
        "wd_critic_update_clip",  # WD-style critic grad cap
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase/task 定義
# ═══════════════════════════════════════════════════════════════════════════
#
# 每個 stage 都包含 6 個區塊：
# 1. name         : 人類可讀的 task 名稱
# 2. scene        : 場景/回合長度/折扣率
# 3. reward       : sparse reward 相關權重
# 4. exploration  : 探索係數（目前是 linear / angular entropy）
# 5. behavior     : rule-based obstacle 行為分佈與速度覆寫
# 6. trainer      : 允許 phase 微調的安全 trainer 超參
# 7. transition   : 升降階條件
#
# 注意：
# - 這裡故意不放會破壞訓練架構的欄位，例如 num_envs / rollout_length / hidden_dim。
# - 那些屬於 run-level / architecture-level 參數，應留在 CLI 或主訓練腳本。
# ═══════════════════════════════════════════════════════════════════════════

STAGES = [
    # ═══════════════════════════════════════════════════════════════════════
    # φ 單調遞增設計表 (Monotonic φ Progression)  v2 — 2026-05-11
    # ═══════════════════════════════════════════════════════════════════════
    #
    # 設計原則：
    # 1. 每個 stage 都有 static obstacles（從 SA1 開始）
    # 2. 每個 φ 只做小幅單調遞增（goals 遞減、其餘遞增）
    # 3. 不刪除已引入的元素（behavior types 只增不減）
    # 4. reward signal density (goals/episode_s) 每步最多下降 ~35%
    #
    # | φ              | SA1 | SA2 | SA3 | SA4 | SA5 | SA6 | SA7 | SA8 |
    # |----------------|-----|-----|-----|-----|-----|-----|-----|-----|
    # | goals          |  10 |   8 |   6 |   4 |   4 |   3 |   2 |   1 |
    # | goal_dist_max  |   6 |   7 |   8 |   9 |   9 |  10 |  10 |  10 |
    # | static_obs     |   1 |   2 |   2 |   2 |   3 |   3 |   3 |   3 |
    # | dynamic_obs    |   2 |   2 |   3 |   4 |   5 |   6 |   8 |   8 |
    # | dynamic_min    |   1 |   1 |   2 |   3 |   3 |   5 |   6 |   6 |
    # | walls_max      |   1 |   1 |   2 |   2 |   3 |   3 |   3 |   3 |
    # | wall_length    | 3.0 | 3.0 | 3.5 | 4.0 | 4.0 | 4.5 | 4.5 | 5.0 |
    # | episode_s      |  60 |  60 |  60 |  60 |  75 |  90 | 120 | 180 |
    # | penalty        |  -5 |  -8 | -10 | -12 | -15 | -25 | -50 |-100 |
    # | obs_speed      |0.80 |0.80 |0.85 |0.85 |0.85 |0.90 |1.00 |1.15 |
    # | behavior_types |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   8 |
    # | signal (g/ep)  |.167 |.133 |.100 |.067 |.053 |.033 |.017 |.006 |
    # | signal Δ%      |  —  | -20 | -25 | -33 | -20 | -37 | -48 | -67 |
    # | density obs/m² |.03  |.03  |.05  |.06  |.08  |.09  |.11  |.11  |
    # | goal_mv_speed  | 0   |0.30 |0.05 |0.10 |0.15 |0.20 |0.25 |0.25 |
    # | goal_mv_radius | 0   | 2.0 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.0 |
    # | goal_mv_behav  |  —  | rw  |drift| rw  | rw  | rw  |patrl|patrl|
    # ═══════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 / SA1 — 導航 bootstrap
    # 【焦點】高 goal density + 低 penalty → 讓 agent 學會「往 goal 走」
    # 【新增】static_obstacles=1 從一開始就接觸靜態幾何
    # dφ: (baseline)
    # WD P1: spots=6, goals=20, obs=3, penalty=-5, speed=0.80, 60s
    # Budget: Sanity 512×100=15.4M | Formal 512×900=138.2M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA1_nav_bootstrap",

        "scene": {
            "goals": 10,
            "goal_distance": (2.0, 6.0),
            "static_obstacles": 1,              # v2: 0→1，從一開始就有靜態
            "dynamic_obstacles": 2,
            "dynamic_obstacles_min": 1,
            "walls_min": 0,
            "walls_max": 1,
            "wall_length": 3.0,                 # v2: 5.0→3.0，短牆降低初期阻擋
            "obs_near_goal_count": 0,           # 在 goal 附近強制生成的障礙物數量（0=關閉）
            "obs_near_goal_radius": 2.0,        # goal 附近多少米範圍內生成障礙物
            # goal 隨機移動（SA1: 關閉，agent 先學靜態 goal 導航）
            "goal_move_speed": 0.0,             # 線速度 (m/s)，0=關閉
            "goal_move_max_radius": 0.0,        # 最大漫遊半徑 (m)
            "goal_move_behavior": "random_walk", # random_walk / drift / patrol
            "goal_move_angular_speed": 0.0,     # 方向變換角速度 (rad/s)
            "episode_s": 60,
            "gamma": 0.984,
        },

        "reward": {
            "penalty_hit": -5.0,
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            # v3b (2026-06-08): 加倍 ent_coeff，避免 SA1 過早收斂
            # 前次 sa1_v3 iter 180 ent_ang 跌到 0.53（碰 YELLOW 線）+ p95_flip 反彈
            # 配合 obs_delay [0,1] 一起測試
            "entropy_linear": 0.02,
            "entropy_angular": 0.04,
        },

        "behavior": {
            "obstacle_speed": 0.80,
            "behavior_mix": {
                "patrol": 0.60,
                "random_walk": 0.40,
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.15, 0.35)},
                "random_walk": {"speed_range": (0.10, 0.30)},
            },
        },

        "trainer": {
            "lr": 2e-4,
            "rnn_lr": 5e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 1.0,
            "aux_grad_clip": 0.5,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.72,
            "upgrade_max_cr": 1.0,              # SA1 先專注導航，不限碰撞
            "upgrade_max_to": 0.30,
            "upgrade_min_dyn_sr": 0.0,
            "min_stage_updates": 50,
            "downgrade_sr": 0.0,                # SA1 幾乎不降
            "downgrade_min_cr": 1.0,
            "downgrade_min_to": 1.0,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 / SA2 — 導航強化 + 靜態辨識
    # 【焦點】penalty ↑, 引入 "static" behavior 讓 RNN 區分靜態 vs 動態
    # dφ from SA1: goals -2, penalty -3, +static behavior type
    # WD P2: spots=3, goals=16, obs=3, penalty=-8, speed=0.85, 60s
    # Budget: Formal 1024×600=184.3M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA2_nav_static",

        "scene": {
            "goals": 10,                        # 【Test G】SA1 clone：8→10
            "goal_distance": (2.0, 6.0),        # 【Test G】SA1 clone
            "static_obstacles": 1,
            "dynamic_obstacles": 2,
            "dynamic_obstacles_min": 1,
            "walls_min": 0,
            "walls_max": 1,
            "wall_length": 3.0,
            "boundary": 8.5,                    # 【Test G】SA1 clone：7.0→8.5
            "obs_near_goal_count": 0,
            "obs_near_goal_radius": 2.0,        # 【Test G】SA1 clone
            "goal_move_speed": 0.0,
            "goal_move_max_radius": 0.0,        # 【Test G】SA1 clone
            "goal_move_behavior": "random_walk",
            "goal_move_angular_speed": 0.0,     # 【Test G】SA1 clone
            "episode_s": 60,
            "gamma": 0.984,
        },

        "reward": {
            "penalty_hit": -8.0,
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.03,             # ↑ from 0.01，打破 policy 僵化
            "entropy_angular": 0.05,            # ↑ from 0.02
        },

        "behavior": {
            "obstacle_speed": 0.80,
            "behavior_mix": {                   # 【Test G】SA1 clone：移除 static
                "patrol": 0.60,
                "random_walk": 0.40,
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.15, 0.35)},     # 【Test G】SA1 clone
                "random_walk": {"speed_range": (0.10, 0.30)},
            },
        },

        "trainer": {
            "lr": 5e-4,                         # ↑ from 2e-4，加大 actor gradient 效果
            "rnn_lr": 5e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 1.0,
            "aux_grad_clip": 0.5,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.65,
            "upgrade_max_cr": 0.40,
            "upgrade_max_to": 0.30,
            "upgrade_min_dyn_sr": 0.0,
            "min_stage_updates": 65,
            "downgrade_sr": 0.15,
            "downgrade_min_cr": 0.70,
            "downgrade_min_to": 0.65,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 3 / SA3 — 障礙密度 ↑ + 牆壁 + 首個 crossing
    # 【焦點】更多靜態/動態 + 引入牆壁 + horizontal_crossing
    # dφ from SA2: static +1, dynamic +1, walls 0-1→1-2, penalty -2,
    #              goals -2, +horizontal_crossing
    # WD P3: spots=3, goals=1, obs=2, penalty=-12, speed=0.85, 90s
    # Budget: Formal 1024×900=276.5M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA3_walls_crossing",

        "scene": {
            "goals": 6,                         # 保持足夠 reward signal
            "goal_distance": (5.0, 9.0),        # 同 SA2，維持長距離導航
            "static_obstacles": 3,              # 同 SA2，保持密度
            "dynamic_obstacles": 4,             # ↑ from SA2(3)，增加動態挑戰
            "dynamic_obstacles_min": 3,         # ↑ from SA2(2)
            "walls_min": 1,                     # ↑ from SA2(0)
            "walls_max": 2,                     # ↑ from SA2(1)
            "wall_length": 3.5,                 # ↑ from SA2(3.0)
            "boundary": 7.0,                    # 同 SA2
            # 目標附近障礙物
            "obs_near_goal_count": 2,           # 同 SA2，goal 附近 2 個障礙
            "obs_near_goal_radius": 2.5,        # 同 SA2
            # goal 隨機移動（SA3: 朝障礙物移動，迫使 agent 在 obs 附近導航）
            "goal_move_speed": 0.15,            # 線速度 (m/s)，中等速度朝 obs 靠近
            "goal_move_max_radius": 3.0,        # 最大漫遊半徑 (m)
            "goal_move_behavior": "toward_obstacle",  # 主動靠近障礙物
            "goal_move_angular_speed": 0.3,     # 方向變換角速度 (rad/s)
            "episode_s": 60,
            "gamma": 0.990,
        },

        "reward": {
            "penalty_hit": -12.0,               # ↑ from SA2(-8)，WD P3 = -12
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.005,            # ↓ 6x from 0.03，避免 entropy 主導 total_loss
            "entropy_angular": 0.01,            # ↓ 5x from 0.05
        },

        "behavior": {
            "obstacle_speed": 0.85,             # ↑ from 0.80
            "behavior_mix": {
                "patrol": 0.25,
                "random_walk": 0.20,
                "static": 0.15,
                "horizontal_crossing": 0.20,
                "head_on": 0.20,                # 新增：直線迎面 (訓練提早避讓正面來車)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.25, 0.50)},
                "random_walk": {"speed_range": (0.20, 0.45)},
                "horizontal_crossing": {"speed_range": (0.25, 0.50)},
                "head_on": {"speed_range": (0.30, 0.55)},
            },
        },

        "trainer": {
            "lr": 5e-4,                         # 同 SA2 v4，避免 policy 凍結
            "rnn_lr": 5e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 1.0,
            "aux_grad_clip": 0.5,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.60,
            "upgrade_max_cr": 0.40,
            "upgrade_max_to": 0.30,
            "upgrade_min_dyn_sr": 0.20,
            "min_stage_updates": 80,
            "downgrade_sr": 0.15,
            "downgrade_min_cr": 0.70,
            "downgrade_min_to": 0.65,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 4 / SA4 — 空間規劃 + path crossing
    # 【焦點】更多動態 + penalty ↑ + path_crossing
    # dφ from SA3: dynamic +1, goals -2, penalty -2, +path_crossing
    # WD P4: spots=3, goals=1, obs=2, penalty=-12, speed=0.85, 60s, wall=1/4.5m
    # Budget: Formal 1024×900=276.5M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA4_spatial_plan",

        "scene": {
            "goals": 2,                         # ↓ from SA3(6)，大幅減少 → 更長導航路徑
            "goal_distance": (5.0, 9.0),        # 同 SA2/SA3，維持長距離
            "static_obstacles": 3,              # 同 SA3
            "dynamic_obstacles": 5,             # ↑ from SA3(4)，漸進增加
            "dynamic_obstacles_min": 4,         # ↑ from SA3(3)
            "walls_min": 1,
            "walls_max": 2,
            "wall_length": 4.0,                 # ↑ from SA3(3.5)
            "boundary": 7.0,                    # 同 SA2/SA3
            # 目標附近障礙物
            "obs_near_goal_count": 2,           # 同 SA3
            "obs_near_goal_radius": 2.5,        # 同 SA3
            # narrow-gap pairs: SA4 intro 寬通道（D 2.2-2.4m → LiDAR 通道 0.8-1.0m）
            # 偶爾生成（20% env），goal 在 pair 後方逼穿越
            "narrow_gap_prob": 0.20,
            "narrow_gap_max_pairs": 1,
            "narrow_gap_center_dist_range": (2.2, 2.4),
            "narrow_gap_align_to_goal": True,
            # goal 隨機移動（SA4: 首次真正啟用！per-goal random [0.3,0.6] m/s）
            "goal_move_speed": 0.1,             # > 0 即啟用 per-goal movement
            "goal_move_max_radius": 3.0,        # 最大漫遊半徑
            "goal_move_behavior": "random_walk",
            "goal_move_angular_speed": 0.3,
            "episode_s": 60,
            "gamma": 0.994,
        },

        "reward": {
            "penalty_hit": -15.0,               # ↑ from SA3(-12)，漸進增加
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.005,            # 同 SA3 low-ent
            "entropy_angular": 0.01,            # 同 SA3 low-ent
        },

        "behavior": {
            "obstacle_speed": 0.85,
            "behavior_mix": {
                "patrol": 0.20,
                "random_walk": 0.15,
                "static": 0.10,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.20,          # robot→goal 路徑交叉
                "head_on": 0.20,                # 新增：直線迎面 (訓練提早避讓正面來車)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.25, 0.55)},
                "random_walk": {"speed_range": (0.20, 0.50)},
                "horizontal_crossing": {"speed_range": (0.30, 0.60)},
                "path_crossing": {"speed_range": (0.25, 0.55)},
                "head_on": {"speed_range": (0.30, 0.60)},
            },
        },

        "trainer": {
            "lr": 5e-4,                         # 同 SA2/SA3，避免 policy 凍結
            "rnn_lr": 5e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 1.0,
            "aux_grad_clip": 0.5,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.60,
            "upgrade_max_cr": 0.35,
            "upgrade_max_to": 0.30,
            "upgrade_min_dyn_sr": 0.30,
            "min_stage_updates": 95,
            "downgrade_sr": 0.15,
            "downgrade_min_cr": 0.70,
            "downgrade_min_to": 0.65,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 5 / SA5 — 耐久導航 + corridor
    # 【焦點】首次延長 episode + 更多靜態/牆 + corridor_crossing
    # 【關鍵】goals 保持 4 不減，維持 reward signal (4/75=0.053)
    # dφ from SA4: static +1, dynamic +1, walls +1, episode +15s,
    #              penalty -3, +corridor_crossing
    # WD P5: spots=2, goals=1, obs=2, penalty=-12, speed=0.85, 100s, wall=1/5.0m
    # Budget: Formal 1024×900=276.5M | Long 1024×1500=460.8M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA5_endurance",

        "scene": {
            "goals": 4,                         # 同 SA4！維持 reward signal
            "goal_distance": (2.0, 9.0),
            "static_obstacles": 3,              # ↑ from 2
            "dynamic_obstacles": 5,             # ↑ from 4
            "dynamic_obstacles_min": 3,
            "walls_min": 2,                     # ↑ from 1
            "walls_max": 3,                     # ↑ from 2
            "wall_length": 4.0,
            # 目標附近障礙物
            "obs_near_goal_count": 1,           # goal 附近強制生成的障礙物數量（0=關閉）
            "obs_near_goal_radius": 2.0,        # goal 附近多少米範圍內生成障礙物
            # narrow-gap pairs: SA5 加 1 對中通道（D 2.0-2.3m → LiDAR 通道 0.7-0.95m）
            "narrow_gap_prob": 0.30,
            "narrow_gap_max_pairs": 1,
            "narrow_gap_center_dist_range": (2.0, 2.3),
            "narrow_gap_align_to_goal": True,
            # goal 隨機移動（SA5: 中速 random_walk）
            "goal_move_speed": 0.15,            # 線速度 (m/s)，↑ from 0.10
            "goal_move_max_radius": 2.0,        # 最大漫遊半徑 (m)，↑ from 1.5
            "goal_move_behavior": "random_walk", # 隨機方向 + 平滑轉向
            "goal_move_angular_speed": 0.4,     # 方向變換角速度 (rad/s)
            "episode_s": 75,                    # ↑ from 60（漸進，非 60→90 跳躍）
            "gamma": 0.995,
        },

        "reward": {
            "penalty_hit": -15.0,               # ↑ from -12
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.008,            # ↓ from 0.01
            "entropy_angular": 0.015,
        },

        "behavior": {
            "obstacle_speed": 0.85,
            "behavior_mix": {
                "patrol": 0.15,
                "random_walk": 0.10,
                "static": 0.10,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.10,
                "corridor_crossing": 0.20,      # 狹窄通道穿越
                "head_on": 0.20,                # 新增：直線迎面 (訓練提早避讓正面來車)
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.30, 0.65)},
                "random_walk": {"speed_range": (0.25, 0.60)},
                "horizontal_crossing": {"speed_range": (0.35, 0.70)},
                "path_crossing": {"speed_range": (0.30, 0.65)},
                "corridor_crossing": {"speed_range": (0.20, 0.45)},
                "head_on": {"speed_range": (0.35, 0.65)},
            },
        },

        "trainer": {
            "lr": 1.5e-4,                       # ↓ from 2e-4
            "rnn_lr": 3e-4,                     # ↓ from 4e-4
            "vf_coeff": 0.5,
            "max_grad_norm": 0.8,               # ↓ from 1.0
            "aux_grad_clip": 0.4,               # ↓ from 0.5
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.55,
            "upgrade_max_cr": 0.35,
            "upgrade_max_to": 0.30,
            "upgrade_min_dyn_sr": 0.35,
            "min_stage_updates": 110,
            "downgrade_sr": 0.15,
            "downgrade_min_cr": 0.70,
            "downgrade_min_to": 0.65,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 6 / SA6 — dense 避障 + near_miss
    # 【焦點】更多動態 + 延長 episode + penalty ↑ + near_miss
    # dφ from SA5: dynamic +1, goals -1, episode +15s, penalty -10,
    #              +near_miss, obs_speed ↑
    # WD P6: spots=3, goals=1, obs=10, penalty=-15, speed=0.85, 210s
    # Budget: Formal 1024×1500=460.8M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA6_dense_avoid",

        "scene": {
            "goals": 3,
            "goal_distance": (3.0, 10.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 6,             # ↑ from 5
            "dynamic_obstacles_min": 5,         # ↑ from 3
            "walls_min": 2,
            "walls_max": 3,
            "wall_length": 4.5,                 # ↑ from 4.0
            # 目標附近障礙物
            "obs_near_goal_count": 2,           # goal 附近強制生成的障礙物數量（0=關閉）
            "obs_near_goal_radius": 2.0,        # goal 附近多少米範圍內生成障礙物
            # narrow-gap pairs: SA6 機率提升 + max 2 對（D 1.9-2.2m → LiDAR 通道 0.65-0.9m）
            "narrow_gap_prob": 0.40,
            "narrow_gap_max_pairs": 2,
            "narrow_gap_center_dist_range": (1.9, 2.2),
            "narrow_gap_align_to_goal": True,
            # goal 隨機移動（SA6: 中快速 random_walk）
            "goal_move_speed": 0.20,            # 線速度 (m/s)，↑ from 0.15
            "goal_move_max_radius": 2.5,        # 最大漫遊半徑 (m)，↑ from 2.0
            "goal_move_behavior": "random_walk", # 隨機方向 + 平滑轉向
            "goal_move_angular_speed": 0.5,     # 方向變換角速度 (rad/s)
            "episode_s": 90,                    # ↑ from 75
            "gamma": 0.996,
        },

        "reward": {
            "penalty_hit": -25.0,               # ↑ from -15（漸進，非 -15→-85 跳躍）
            "penalty_timeout": -12.5,           # = penalty_hit / 2（SA6+ timeout 懲罰）
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.008,
            "entropy_angular": 0.015,
        },

        "behavior": {
            "obstacle_speed": 0.90,             # ↑ from 0.85
            "behavior_mix": {
                "patrol": 0.15,
                "random_walk": 0.15,
                "static": 0.05,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.15,
                "corridor_crossing": 0.15,
                "near_miss": 0.20,              # 新增：高速擦身
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.35, 0.75)},
                "random_walk": {"speed_range": (0.30, 0.70)},
                "horizontal_crossing": {"speed_range": (0.40, 0.80)},
                "path_crossing": {"speed_range": (0.35, 0.75)},
                "corridor_crossing": {"speed_range": (0.25, 0.55)},
                "near_miss": {"speed_range": (0.30, 0.65)},
            },
        },

        "trainer": {
            "lr": 1.5e-4,
            "rnn_lr": 3e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 0.8,
            "aux_grad_clip": 0.4,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.50,
            "upgrade_max_cr": 0.30,
            "upgrade_max_to": 0.35,
            "upgrade_min_dyn_sr": 0.40,
            "min_stage_updates": 120,
            "downgrade_sr": 0.10,
            "downgrade_min_cr": 0.75,
            "downgrade_min_to": 0.70,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 7 / SA7 — 高壓綜合 + occlusion
    # 【焦點】更多動態 + 延長 episode + 高 penalty + occlusion + 高速
    # dφ from SA6: dynamic +2, goals -1, episode +30s, penalty -25,
    #              +occlusion, obs_speed ↑
    # WD P7: spots=2, goals=1, obs=10, penalty=-85, speed=1.10, 210s, wall=2/3.5m
    # Budget: Formal 1024×1500=460.8M
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA7_high_pressure",

        "scene": {
            "goals": 2,
            "goal_distance": (3.0, 10.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 8,             # ↑ from 6
            "dynamic_obstacles_min": 6,         # ↑ from 5
            "walls_min": 2,
            "walls_max": 3,
            "wall_length": 4.5,
            # 目標附近障礙物
            "obs_near_goal_count": 2,           # goal 附近強制生成的障礙物數量（0=關閉）
            "obs_near_goal_radius": 2.0,        # goal 附近多少米範圍內生成障礙物
            # narrow-gap pairs: SA7 最高機率 + max 2 對（D 1.8-2.1m → LiDAR 通道 0.6-0.8m）
            "narrow_gap_prob": 0.50,
            "narrow_gap_max_pairs": 2,
            "narrow_gap_center_dist_range": (1.8, 2.1),
            "narrow_gap_align_to_goal": True,
            # goal 隨機移動（SA7: 快速 patrol）
            "goal_move_speed": 0.25,            # 線速度 (m/s)，↑ from 0.20
            "goal_move_max_radius": 3.0,        # 最大漫遊半徑 (m)，↑ from 2.5
            "goal_move_behavior": "patrol",     # waypoint 間巡邏（切換自 random_walk）
            "goal_move_angular_speed": 0.5,     # 方向變換角速度 (rad/s)
            "episode_s": 120,                   # ↑ from 90（漸進，非 90→210 跳躍）
            "gamma": 0.997,
        },

        "reward": {
            "penalty_hit": -50.0,               # ↑ from -25（漸進，非 -15→-85 跳躍）
            "penalty_timeout": -25.0,           # = penalty_hit / 2
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.005,
            "entropy_angular": 0.01,
        },

        "behavior": {
            "obstacle_speed": 1.00,             # ↑ from 0.90
            "behavior_mix": {
                "patrol": 0.10,
                "random_walk": 0.10,
                "static": 0.05,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.10,
                "corridor_crossing": 0.10,
                "near_miss": 0.20,
                "occlusion": 0.20,              # 新增：遮蔽後突現
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.45, 0.90)},
                "random_walk": {"speed_range": (0.40, 0.85)},
                "horizontal_crossing": {"speed_range": (0.50, 0.95)},
                "path_crossing": {"speed_range": (0.45, 0.90)},
                "corridor_crossing": {"speed_range": (0.35, 0.70)},
                "near_miss": {"speed_range": (0.40, 0.80)},
                "occlusion": {"speed_range": (0.25, 0.60)},
            },
        },

        "trainer": {
            "lr": 1e-4,
            "rnn_lr": 2e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 0.6,
            "aux_grad_clip": 0.3,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 0.45,
            "upgrade_max_cr": 0.25,
            "upgrade_max_to": 0.35,
            "upgrade_min_dyn_sr": 0.35,
            "min_stage_updates": 130,
            "downgrade_sr": 0.10,
            "downgrade_min_cr": 0.80,
            "downgrade_min_to": 0.75,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 8 / SA8 — 最終評估級
    # 【焦點】penalty=-100 最終標準 + 最長 episode + 最高速
    # dφ from SA7: goals -1, episode +60s, penalty -50, obs_speed ↑,
    #              wall_length ↑
    # WD P8: spots=2, goals=1, obs=6, penalty=-100, speed=1.15, 210s, wall=2/3.5m
    # Budget: Formal 1024×1500=460.8M | Eval: deterministic play
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA8_final",

        "scene": {
            "goals": 1,
            "goal_distance": (3.0, 10.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 8,
            "dynamic_obstacles_min": 6,
            "walls_min": 2,
            "walls_max": 3,
            "wall_length": 5.0,                 # ↑ from 4.5
            # 目標附近障礙物
            "obs_near_goal_count": 2,           # goal 附近強制生成的障礙物數量（0=關閉）
            "obs_near_goal_radius": 2.0,        # goal 附近多少米範圍內生成障礙物
            # goal 隨機移動（SA8: 最終標準 patrol）
            "goal_move_speed": 0.25,            # 線速度 (m/s)，同 SA7
            "goal_move_max_radius": 3.0,        # 最大漫遊半徑 (m)，同 SA7
            "goal_move_behavior": "patrol",     # waypoint 間巡邏
            "goal_move_angular_speed": 0.5,     # 方向變換角速度 (rad/s)
            "episode_s": 180,                   # ↑ from 120
            "gamma": 0.998,
        },

        "reward": {
            "penalty_hit": -100.0,
            "penalty_timeout": -50.0,           # = penalty_hit / 2
            "reward_get_goal": 40.0,
            "penalty_smoothness": 0.005,        # v3: anti-jitter |Δratio_ang| penalty
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.003,
            "entropy_angular": 0.008,
        },

        "behavior": {
            "obstacle_speed": 1.15,             # ↑ from 1.00
            "behavior_mix": {
                "patrol": 0.10,
                "random_walk": 0.10,
                "static": 0.05,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.10,
                "corridor_crossing": 0.10,
                "near_miss": 0.20,
                "occlusion": 0.20,
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.50, 1.00)},
                "random_walk": {"speed_range": (0.45, 0.95)},
                "horizontal_crossing": {"speed_range": (0.60, 1.10)},
                "path_crossing": {"speed_range": (0.55, 1.00)},
                "corridor_crossing": {"speed_range": (0.45, 0.80)},
                "near_miss": {"speed_range": (0.50, 0.95)},
                "occlusion": {"speed_range": (0.35, 0.75)},
            },
        },

        "trainer": {
            "lr": 1e-4,
            "rnn_lr": 2e-4,
            "vf_coeff": 0.5,
            "max_grad_norm": 0.5,
            "aux_grad_clip": 0.3,
            "wd_actor_update_clip": 8.0,
            "wd_critic_update_clip": 30.0,
        },

        "transition": {
            "upgrade_sr": 1.0,                  # 最終 phase，不可能升階
            "upgrade_max_cr": 0.0,
            "upgrade_max_to": 0.0,
            "upgrade_min_dyn_sr": 1.0,
            "min_stage_updates": 9999,
            "downgrade_sr": 0.08,
            "downgrade_min_cr": 0.85,
            "downgrade_min_to": 0.80,
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Legacy bridge
# ═══════════════════════════════════════════════════════════════════════════
#
# 目的：
# - 人類在上面使用巢狀 schema 編寫 task。
# - curriculum / training 舊程式碼仍偏好 flat dict。
# - 所以這個 bridge 會把巢狀 phase 攤平成 legacy 欄位。
#
# 關鍵原則：
# - 只做 schema 轉換，不改變原本訓練架構。
# - 不在這裡做任何 IsaacLab core 行為修改。
# ═══════════════════════════════════════════════════════════════════════════


def _flatten_phase(phase: dict) -> dict:
    """把巢狀 phase/task schema 攤平成 curriculum 既有可用格式。

    回傳結果會被 `CONFIG["stages"]` 使用，
    供 `goal_obstacle_curriculum.py` 與 `train_rnn_car_wdclip.py` 讀取。
    """

    # 若輸入本來就是 flat/legacy schema，直接回傳副本，保留相容性。
    if "scene" not in phase:
        return dict(phase)

    # 取出各區塊，方便下方轉換。
    scene = phase.get("scene", {})
    reward = phase.get("reward", {})
    exploration = phase.get("exploration", {})
    behavior = phase.get("behavior", {})
    trainer = phase.get("trainer", {})
    transition = phase.get("transition", {})

    # 障礙物數量與 ratio 轉換。
    # 既有 curriculum event 需要的是 static_ratio / dynamic_ratio / empty_ratio / mixed_ratio，
    # 當同時有 static 和 dynamic 障礙物時，必須用 mixed mode (difficulty=3)
    # 才能在同一個 env 內正確區分：i < num_static 靜止，i >= num_static 移動。
    # 參照 goal_obstacle_curriculum.py L1049-1053 的邏輯。
    n_s = int(scene.get("static_obstacles", 0))
    n_d = int(scene.get("dynamic_obstacles", 0))
    total = n_s + n_d

    if total == 0:
        # 無障礙物 → 全部 empty
        empty = 1.0
        s_ratio = 0.0
        d_ratio = 0.0
        m_ratio = 0.0
    elif n_s > 0 and n_d > 0:
        # 同時有 static + dynamic → mixed mode，所有 env 都混合配置
        empty = 0.0
        s_ratio = 0.0
        d_ratio = 0.0
        m_ratio = 1.0
    elif n_d > 0:
        # 純 dynamic
        empty = 0.0
        s_ratio = 0.0
        d_ratio = 1.0
        m_ratio = 0.0
    else:
        # 純 static
        empty = 0.0
        s_ratio = 1.0
        d_ratio = 0.0
        m_ratio = 0.0

    # 主要 flat schema。
    flat = {
        # 基本識別
        "name": phase["name"],

        # scene -> legacy scene fields
        "num_goals": int(scene["goals"]),
        "goal_distance": scene["goal_distance"],
        "num_obstacles_static": n_s,
        "num_obstacles_dynamic": n_d,
        "min_obstacles_dynamic": int(scene.get("dynamic_obstacles_min", n_d)),
        "empty_ratio": empty,
        "static_ratio": s_ratio,
        "dynamic_ratio": d_ratio,
        "mixed_ratio": m_ratio,
        "gamma": float(scene["gamma"]),
        "episode_length_s": float(scene["episode_s"]),
        "min_walls": int(scene["walls_min"]),
        "max_walls": int(scene["walls_max"]),
        "target_wall_length": float(scene.get("wall_length", 0.0)),

        # reward -> WD sparse reward fields
        "spot_penalty_hit": float(reward.get("penalty_hit", -5.0)),
        "spot_reward_get_goal": float(reward.get("reward_get_goal", 40.0)),
        "spot_cost_operate": float(reward.get("cost_operate", 0.0)),
        "spot_penalty_timeout": float(reward.get("penalty_timeout", 0.0)),
        "spot_penalty_smoothness": float(reward.get("penalty_smoothness", 0.0)),  # v3
        "spot_penalty_speed_near_obs": float(reward.get("penalty_speed_near_obs", 0.0)),  # v3f-react

        # exploration -> entropy fields
        "ent_coeff_linear": float(exploration.get("entropy_linear", 0.01)),
        "ent_coeff_angular": float(exploration.get("entropy_angular", 0.02)),

        # behavior -> obstacle speed / behavior fields
        "obstacle_speed_rate": float(behavior.get("obstacle_speed", 0.8)),

        # boundary (obstacle spawn range)
        "boundary": float(scene.get("boundary", 8.5)),

        # goal 附近障礙物
        "obs_near_goal_count": float(scene.get("obs_near_goal_count", 0)),
        "obs_near_goal_radius": float(scene.get("obs_near_goal_radius", 2.0)),

        # narrow-gap pairs: 機率性窄通道（policy 偶爾遇到通道，學穿越）
        # narrow_gap_prob: 每對 pair 對每個 env 獨立 roll 的命中機率
        # narrow_gap_max_pairs: 每次 reset 最多嘗試生成幾對
        # 舊欄位 narrow_gap_pairs 維持向後相容（=N 時 → prob=1.0, max_pairs=N）
        "narrow_gap_prob": float(scene.get("narrow_gap_prob",
                                           1.0 if int(scene.get("narrow_gap_pairs", 0)) > 0 else 0.0)),
        "narrow_gap_max_pairs": int(scene.get("narrow_gap_max_pairs",
                                              int(scene.get("narrow_gap_pairs", 0)))),
        "narrow_gap_center_dist_range": tuple(scene.get("narrow_gap_center_dist_range", (1.8, 2.4))),
        "narrow_gap_align_to_goal": bool(scene.get("narrow_gap_align_to_goal", True)),

        # goal 隨機移動
        "goal_move_speed": float(scene.get("goal_move_speed", 0.0)),
        "goal_move_max_radius": float(scene.get("goal_move_max_radius", 0.0)),
        "goal_move_behavior": str(scene.get("goal_move_behavior", "random_walk")),
        "goal_move_angular_speed": float(scene.get("goal_move_angular_speed", 0.0)),

        # 目前單車任務不主用，但保留 legacy 欄位相容性
        "num_virtual_spots": float(scene.get("virtual_spots", 0)),
        "goal_speed_rate": float(scene.get("goal_speed", 0.7)),

        # transition -> curriculum threshold fields
        "upgrade_sr": float(transition["upgrade_sr"]),
        "upgrade_max_cr": float(transition["upgrade_max_cr"]),
        "upgrade_max_to": float(transition["upgrade_max_to"]),
        "upgrade_min_dyn_sr": float(transition.get("upgrade_min_dyn_sr", 0.0)),
        "min_stage_updates": int(transition.get("min_stage_updates", 0)),
        "downgrade_sr": float(transition["downgrade_sr"]),
        "downgrade_min_cr": float(transition["downgrade_min_cr"]),
        "downgrade_min_to": float(transition["downgrade_min_to"]),

        # trainer runtime keys：
        # 這些欄位不直接改 IsaacLab core，只是讓 training loop 能安全讀取並同步。
        "trainer_lr": float(trainer.get("lr", 2e-4)),
        "trainer_rnn_lr": float(trainer.get("rnn_lr", 5e-4)),
        "trainer_vf_coeff": float(trainer.get("vf_coeff", 0.5)),
        "trainer_max_grad_norm": float(trainer.get("max_grad_norm", 1.0)),
        "trainer_aux_grad_clip": float(trainer.get("aux_grad_clip", trainer.get("max_grad_norm", 1.0))),
        "trainer_wd_actor_update_clip": float(trainer.get("wd_actor_update_clip", 8.0)),
        "trainer_wd_critic_update_clip": float(trainer.get("wd_critic_update_clip", 30.0)),
    }

    # 可選欄位：只有存在時才傳遞，避免污染沒有用到的 phase。
    optional_map = {
        # scene 額外隨機化欄位
        "obs_size_rand": scene.get("obs_size_rand"),
        "scene_bound_rand": scene.get("scene_bound_rand"),

        # corridor-crossing injector fraction（e2e curriculum 專用，非 e2e phase 不含此 key）
        "corridor_crossing_fraction": scene.get("corridor_crossing_fraction"),

        # reward manager 權重表
        "reward_weights": reward.get("reward_weights"),

        # behavior scheduler 相關設定
        "behavior_mix": behavior.get("behavior_mix"),
        "speed_overrides": behavior.get("speed_overrides"),
        "safety_overrides": behavior.get("safety_overrides"),

        # 若未來某 phase 想自定安全同步白名單，可放這裡
        "trainer_sync_allowlist": trainer.get("sync_allowlist"),
    }

    for key, value in optional_map.items():
        if value is not None:
            flat[key] = value

    return flat


# 向後相容：其他 phase 檔案（例如 wd_goal_first.py）仍可能 import 舊名稱 `_to_legacy_stage`。
# 這裡保留同義 alias，避免重構後打斷既有 import chain。
def _to_legacy_stage(phase: dict) -> dict:
    return _flatten_phase(phase)


# 供 phases registry 匯出的標準格式。
CONFIG = {
    # 障礙物控制模式（全局）
    "obstacle_mode": GLOBAL["obstacle_mode"],

    # 全局升階連續通過次數
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],

    # 升階後是否清空 window
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],

    # 可安全動態同步的 trainer 欄位白名單
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],

    # curriculum 真正使用的 stage 列表（已 flatten）
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
