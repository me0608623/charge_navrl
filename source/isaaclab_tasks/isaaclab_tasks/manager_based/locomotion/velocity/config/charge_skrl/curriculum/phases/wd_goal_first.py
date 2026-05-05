"""warp_drive_goal_first — Goal → Static → Dynamic 漸進課程

設計理念:
  Phase 1-2: 純導航（goal-reaching + walls），無障礙物
  Phase 3-4: 加入 static obstacles（LiDAR 學習靜態避障）
  Phase 5-8: 加入 dynamic obstacles（obs_agent policy 啟動）

────────────────────────────────────────────────────────────────────────────
修改指南：同 wd_single_agent_v1.py
────────────────────────────────────────────────────────────────────────────
"""

GLOBAL = {
    "upgrade_pass_required": 5,
    "clear_window_on_promote": True,
}

STAGES = [
    # ──────────────────────────────────────────────────────────────────────
    # GF1: 純導航 — 多 goal，無障礙物，obs_agent OFF
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF1_goal_open",
        "goals": 8, "static_obstacles": 0, "dynamic_obstacles": 0,
        "walls_min": 0, "walls_max": 0,
        "goal_distance": (2.0, 13.0), "episode_s": 45,
        "gamma": 0.990,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.03,
        "entropy_linear": 0.10, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.85, "upgrade_max_cr": 1.0, "upgrade_max_to": 0.20,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 50,
        "downgrade_sr": 0.0, "downgrade_min_cr": 1.0, "downgrade_min_to": 1.0,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF2: 導航 + 牆壁 — 學繞路, obs_agent OFF
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF2_goal_walls",
        "goals": 6, "static_obstacles": 0, "dynamic_obstacles": 0,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 50,
        "gamma": 0.990,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.82, "upgrade_max_cr": 1.0, "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 60,
        "downgrade_sr": 0.30, "downgrade_min_cr": 1.0, "downgrade_min_to": 0.80,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF3: Static 初階 — 2 static + 1 wall, obs_agent OFF
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF3_static_intro",
        "goals": 4, "static_obstacles": 2, "dynamic_obstacles": 0,
        "walls_min": 1, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 55,
        "gamma": 0.992,
        "penalty_hit": -8.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "obs_size_rand": 0.1,
        "upgrade_sr": 0.78, "upgrade_max_cr": 0.35, "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 70,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF4: Static 密集 — 4 static + 1 wall, obs_agent OFF
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF4_static_dense",
        "goals": 2, "static_obstacles": 4, "dynamic_obstacles": 0,
        "walls_min": 1, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 60,
        "gamma": 0.993,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.2,
        "upgrade_sr": 0.75, "upgrade_max_cr": 0.35, "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 85,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF5: Dynamic 引入 — 4 static + 2 dynamic, obs_agent ON
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF5_dynamic_intro",
        "goals": 1, "static_obstacles": 4, "dynamic_obstacles": 2,
        "walls_min": 1, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 70,
        "gamma": 0.994,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.3, "scene_bound_rand": 0.5,
        "upgrade_sr": 0.70, "upgrade_max_cr": 0.35, "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 100,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF6: Dynamic 中階 — 4 static + 6 dynamic
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF6_dynamic_medium",
        "goals": 1, "static_obstacles": 4, "dynamic_obstacles": 6,
        "walls_min": 1, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 90,
        "gamma": 0.996,
        "penalty_hit": -40.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.35, "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 120,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF7: Dense dynamic — 4 static + 10 dynamic + 2 walls
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF7_dense",
        "goals": 1, "static_obstacles": 4, "dynamic_obstacles": 10,
        "walls_min": 1, "walls_max": 2,
        "goal_distance": (2.0, 13.0), "episode_s": 210,
        "gamma": 0.998,
        "penalty_hit": -85.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 1.15,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "upgrade_sr": 0.55, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.30, "min_stage_updates": 150,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # GF8: 最終穩定 — penalty=-200 (不再升階)
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "GF8_stable",
        "goals": 1, "static_obstacles": 4, "dynamic_obstacles": 6,
        "walls_min": 1, "walls_max": 2,
        "goal_distance": (2.0, 13.0), "episode_s": 210,
        "gamma": 0.998,
        "penalty_hit": -200.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 1.15,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "upgrade_sr": 1.0, "upgrade_max_cr": 0.0, "upgrade_max_to": 0.0,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 0,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 匯出
# ═══════════════════════════════════════════════════════════════════════════

from .wd_single_agent_v1 import _to_legacy_stage

CONFIG = {
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "stages": [_to_legacy_stage(s) for s in STAGES],
}
