"""warp_drive_v1 — 完全照 Warp Drive 原始 Phase 1-8 設計

核心: 從 Phase 1 就有 3 dynamic，無 static（WD 沒有 static obstacle 概念）。
所有 obstacles 由 learned obstacle policy 控制。

| Phase | Goals | Dynamic | Walls | Episode | penalty | Virtual Spots |
|-------|-------|---------|-------|---------|---------|---------------|
| WD1   | 8     | 3       | 0~1   | 60s     | -5      | 5             |
| WD2   | 8     | 3       | 0~1   | 60s     | -8      | 2             |
| WD3   | 1     | 2       | 0~1   | 90s     | -12     | 2             |
| WD4   | 1     | 2       | 0~1   | 60s     | -12     | 2             |
| WD5   | 1     | 2       | 0~1   | 100s    | -12     | 1             |
| WD6   | 1     | 10      | 0~1   | 210s    | -15     | 2             |
| WD7   | 1     | 6       | 0~2   | 210s    | -85     | 1             |
| WD8   | 1     | 6       | 0~2   | 210s    | -200    | 1             |
"""

GLOBAL = {
    "upgrade_pass_required": 5,
    "clear_window_on_promote": True,
}

STAGES = [
    {
        "name": "WD1_nav_3obs_6spot",
        "goals": 8, "static_obstacles": 0, "dynamic_obstacles": 3,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 60,
        "gamma": 0.990,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.03,
        "entropy_linear": 0.10, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "virtual_spots": 5, "goal_speed": 0.7,
        "upgrade_sr": 0.72, "upgrade_max_cr": 1.0, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 50,
        "downgrade_sr": 0.0, "downgrade_min_cr": 1.0, "downgrade_min_to": 1.0,
    },
    {
        "name": "WD2_3obs_3spot",
        "goals": 8, "static_obstacles": 0, "dynamic_obstacles": 3,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 60,
        "gamma": 0.991,
        "penalty_hit": -8.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "virtual_spots": 2, "goal_speed": 0.7,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 65,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD3_single_3spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 2,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 90,
        "gamma": 0.993,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.1, "scene_bound_rand": 0.5,
        "virtual_spots": 2, "goal_speed": 0.6,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 80,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD4_wall_3spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 2,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 60,
        "gamma": 0.994,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.2, "scene_bound_rand": 1.0,
        "virtual_spots": 2, "goal_speed": 0.6,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 95,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD5_long_2spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 2,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 100,
        "gamma": 0.995,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.3, "scene_bound_rand": 1.0,
        "virtual_spots": 1, "goal_speed": 0.6,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 110,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD6_10obs_3spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 10,
        "walls_min": 0, "walls_max": 1,
        "goal_distance": (2.0, 13.0), "episode_s": 210,
        "gamma": 0.997,
        "penalty_hit": -15.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.85,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "virtual_spots": 2, "goal_speed": 0.6,
        "upgrade_sr": 0.60, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.30, "min_stage_updates": 130,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD7_main_2spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 6,
        "walls_min": 0, "walls_max": 2,
        "goal_distance": (2.0, 13.0), "episode_s": 210,
        "gamma": 0.998,
        "penalty_hit": -85.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 1.15,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "virtual_spots": 1, "goal_speed": 0.5,
        "upgrade_sr": 0.55, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.30, "min_stage_updates": 150,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "WD8_stable_2spot",
        "goals": 1, "static_obstacles": 0, "dynamic_obstacles": 6,
        "walls_min": 0, "walls_max": 2,
        "goal_distance": (2.0, 13.0), "episode_s": 210,
        "gamma": 0.998,
        "penalty_hit": -200.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 1.15,
        "obs_size_rand": 0.4, "scene_bound_rand": 1.0,
        "virtual_spots": 1, "goal_speed": 0.5,
        "upgrade_sr": 1.0, "upgrade_max_cr": 0.0, "upgrade_max_to": 0.0,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 0,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
]

from .wd_single_agent_v1 import _to_legacy_stage

CONFIG = {
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "stages": [_to_legacy_stage(s) for s in STAGES],
}
