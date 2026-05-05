"""warp_drive_single_agent_v1 — 你目前正在用的 curriculum

Single-agent 版 WD curriculum。
Phase 1 就有 dynamic obstacle + wall，用多 goals + 短距離補回低成功事件頻率。

────────────────────────────────────────────────────────────────────────────
修改指南：
  - 改 phase 數量/順序 → 直接增減 STAGES 列表
  - 改升階條件 → 修改對應 phase 的 upgrade_* 欄位
  - 改場景內容 → 修改 goals / obstacles / walls / goal_distance
  - 改 reward 信號 → 修改 penalty_hit / cost_operate
  - 改探索程度 → 修改 entropy_linear / entropy_angular
────────────────────────────────────────────────────────────────────────────
"""

# ═══════════════════════════════════════════════════════════════════════════
# 全域設定（適用所有 phase）
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL = {
    "upgrade_pass_required": 5,       # 連續通過幾次才真正升階
    "clear_window_on_promote": True,  # 升階後清空 outcome window
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 定義（按順序排列，index 0 = Stage 1）
#
# 每個 phase 是一個 dict，欄位分三區：
#   [場景] goals, obstacles, walls, episode, goal_distance
#   [RL]   gamma, penalty_hit, reward_get_goal, cost_operate, entropy, obs_speed
#   [升降] upgrade 條件, downgrade 條件, min_stage_updates
# ═══════════════════════════════════════════════════════════════════════════

STAGES = [
    # ──────────────────────────────────────────────────────────────────────
    # SA1: 近距多目標 + 少量動態障礙 + 少量牆壁
    #   目的: 高成功事件頻率啟動學習，RNN 從頭接觸 dynamic
    #   bounded random: 有些 env 容易 (0 wall, 1 dyn)，有些稍難 (1 wall, 2 dyn)
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA1_goal_wall_dyn",

        # --- 場景 ---
        "goals": 10,
        "static_obstacles": 0,
        "dynamic_obstacles": 2,          # 上限
        "dynamic_obstacles_min": 1,      # 下限 (per-env random 在 min~max)
        "walls_min": 0,
        "walls_max": 1,
        "wall_length": 5.0,             # 內牆目標長度 (m)，WD Phase 1: 6.0
        "goal_distance": (2.0, 6.0),     # (min_m, max_m)
        "episode_s": 60,

        # --- RL 超參 ---
        "gamma": 0.990,
        "penalty_hit": -5.0,             # 碰撞懲罰
        "reward_get_goal": 40.0,         # 到達目標獎勵
        "cost_operate": 0.03,            # 動作成本/fps (Phase 1 專屬)
        "entropy_linear": 0.10,          # 低 → 先學基本走
        "entropy_angular": 0.375,
        "obstacle_speed": 0.8,           # 障礙物速度倍率

        # --- 升階條件 ---
        "upgrade_sr": 0.72,              # Success Rate ≥ 72%
        "upgrade_max_cr": 1.0,           # Collision Rate 不限 (Phase 1 容許碰撞)
        "upgrade_max_to": 0.30,          # Timeout Rate ≤ 30%
        "upgrade_min_dyn_sr": 0.0,       # Dynamic SR 不限
        "min_stage_updates": 50,         # 至少跑 50 rollout cycles

        # --- 降階條件 ---
        "downgrade_sr": 0.0,             # Phase 1 不降階
        "downgrade_min_cr": 1.0,
        "downgrade_min_to": 1.0,
    },

    # ──────────────────────────────────────────────────────────────────────
    # SA2: 目標距離稍遠，探索增加
    #   目的: 延伸導航範圍，entropy 提高鼓勵探索
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA2_goal_wall_2dyn",

        # --- 場景 ---
        "goals": 8,
        "static_obstacles": 0,
        "dynamic_obstacles": 2,
        "dynamic_obstacles_min": 1,
        "walls_min": 0,
        "walls_max": 1,
        "wall_length": 5.0,
        "goal_distance": (2.0, 7.0),
        "episode_s": 60,

        # --- RL 超參 ---
        "gamma": 0.991,
        "penalty_hit": -5.0,
        "reward_get_goal": 40.0,
        "cost_operate": 0.0,             # Phase 2+ 不收動作成本
        "entropy_linear": 0.20,          # 提高探索
        "entropy_angular": 0.375,
        "obstacle_speed": 0.8,

        # --- 升階條件 ---
        "upgrade_sr": 0.65,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0,
        "min_stage_updates": 65,

        # --- 降階條件 ---
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.70,
        "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # SA3: 動態障礙增加 + 牆壁下限提升
    #   目的: 開始學避障，碰撞懲罰加重
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA3_dyn_intro",

        # --- 場景 ---
        "goals": 6,
        "static_obstacles": 0,
        "dynamic_obstacles": 3,
        "dynamic_obstacles_min": 2,
        "walls_min": 1,
        "walls_max": 2,
        "wall_length": 4.5,             # WD Phase 4 對齊
        "goal_distance": (2.0, 8.0),
        "episode_s": 60,

        # --- RL 超參 ---
        "gamma": 0.993,
        "penalty_hit": -8.0,
        "reward_get_goal": 40.0,
        "cost_operate": 0.0,
        "entropy_linear": 0.30,
        "entropy_angular": 0.375,
        "obstacle_speed": 0.85,

        # --- 升階條件 ---
        "upgrade_sr": 0.65,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35,      # 需要 dynamic 環境也有 35% SR
        "min_stage_updates": 80,

        # --- 降階條件 ---
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.70,
        "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # SA4: 更多動態 + episode 加長 + 目標漸遠
    #   目的: 學習在持續干擾下完成長距離導航
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA4_nav_avoid",

        # --- 場景 ---
        "goals": 4,
        "static_obstacles": 0,
        "dynamic_obstacles": 4,
        "dynamic_obstacles_min": 3,
        "walls_min": 1,
        "walls_max": 2,
        "wall_length": 4.0,             # 牆壁漸短，增加繞行路線
        "goal_distance": (2.0, 9.0),
        "episode_s": 75,

        # --- RL 超參 ---
        "gamma": 0.994,
        "penalty_hit": -12.0,
        "reward_get_goal": 40.0,
        "cost_operate": 0.0,
        "entropy_linear": 0.30,
        "entropy_angular": 0.375,
        "obstacle_speed": 0.85,

        # --- 升階條件 ---
        "upgrade_sr": 0.65,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35,
        "min_stage_updates": 95,

        # --- 降階條件 ---
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.70,
        "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # SA5: 最終階段 — 高密度動態
    #   目的: 最終難度，4~6 dynamic + 遠目標
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA5_medium",

        # --- 場景 ---
        "goals": 2,
        "static_obstacles": 0,
        "dynamic_obstacles": 6,
        "dynamic_obstacles_min": 4,
        "walls_min": 1,
        "walls_max": 2,
        "wall_length": 3.5,             # WD Phase 7+ 對齊：短牆多路線
        "goal_distance": (2.0, 10.0),
        "episode_s": 90,

        # --- RL 超參 ---
        "gamma": 0.995,
        "penalty_hit": -15.0,
        "reward_get_goal": 40.0,
        "cost_operate": 0.0,
        "entropy_linear": 0.30,
        "entropy_angular": 0.375,
        "obstacle_speed": 0.85,

        # --- 升階條件 (最終階段 — 不會再升) ---
        "upgrade_sr": 0.60,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.30,
        "min_stage_updates": 110,

        # --- 降階條件 ---
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.70,
        "downgrade_min_to": 0.65,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 匯出格式（供 goal_obstacle_curriculum.py 使用）
# ═══════════════════════════════════════════════════════════════════════════

def _to_legacy_stage(phase: dict) -> dict:
    """將清晰格式轉換為 _make_stage 產出的 legacy dict。"""
    n_s = phase["static_obstacles"]
    n_d = phase["dynamic_obstacles"]
    total = n_s + n_d
    empty = 1.0 if total == 0 else 0.0
    if total > 0:
        s_ratio = round((1.0 - empty) * n_s / total, 2)
        d_ratio = round(1.0 - empty - s_ratio, 2)
    else:
        s_ratio = d_ratio = 0.0

    return {
        "name": phase["name"],
        "num_goals": phase["goals"],
        "goal_distance": phase["goal_distance"],
        "num_obstacles_static": n_s,
        "num_obstacles_dynamic": n_d,
        "min_obstacles_dynamic": phase.get("dynamic_obstacles_min", n_d),
        "empty_ratio": empty,
        "static_ratio": s_ratio,
        "dynamic_ratio": d_ratio,
        "gamma": phase["gamma"],
        "episode_length_s": float(phase["episode_s"]),
        "min_walls": phase["walls_min"],
        "max_walls": phase["walls_max"],
        "target_wall_length": phase.get("wall_length", 0.0),
        # WD reward params
        "spot_penalty_hit": phase["penalty_hit"],
        "spot_reward_get_goal": phase["reward_get_goal"],
        "spot_cost_operate": phase["cost_operate"],
        # WD entropy params
        "ent_coeff_linear": phase["entropy_linear"],
        "ent_coeff_angular": phase["entropy_angular"],
        # WD obstacle speed
        "obstacle_speed_rate": phase["obstacle_speed"],
        # MARL (unused here)
        "num_virtual_spots": phase.get("virtual_spots", 0),
        "goal_speed_rate": phase.get("goal_speed", 0.7),
        # Upgrade/downgrade
        "upgrade_sr": phase["upgrade_sr"],
        "upgrade_max_cr": phase["upgrade_max_cr"],
        "upgrade_max_to": phase["upgrade_max_to"],
        "upgrade_min_dyn_sr": phase["upgrade_min_dyn_sr"],
        "min_stage_updates": phase["min_stage_updates"],
        "downgrade_sr": phase["downgrade_sr"],
        "downgrade_min_cr": phase["downgrade_min_cr"],
        "downgrade_min_to": phase["downgrade_min_to"],
        # Optional
        **({k: phase[k] for k in ("obs_size_rand", "scene_bound_rand", "reward_weights",
                                    "behavior_mix", "speed_overrides", "safety_overrides")
            if k in phase}),
    }


CONFIG = {
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "stages": [_to_legacy_stage(s) for s in STAGES],
}
