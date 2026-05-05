"""baseline_v1 — 原始 v12 線性 8 階段

最簡單: 每階 -1G, +1S, +1D（對稱增長）

| Phase | Goals | Static | Dynamic | Walls | γ     | Episode |
|-------|-------|--------|---------|-------|-------|---------|
| 1     | 8     | 0      | 0       | 0~2   | 0.990 | 45s     |
| 2     | 7     | 1      | 1       | 0~3   | 0.991 | 51s     |
| 3     | 6     | 2      | 2       | 1~4   | 0.993 | 56s     |
| 4     | 5     | 3      | 3       | 1~5   | 0.994 | 61s     |
| 5     | 4     | 4      | 4       | 2~6   | 0.996 | 67s     |
| 6     | 3     | 5      | 5       | 2~7   | 0.997 | 72s     |
| 7     | 2     | 6      | 6       | 3~8   | 0.998 | 78s     |
| 8     | 1     | 7      | 7       | 4~8   | 0.998 | 90s     |
"""

GLOBAL = {
    "upgrade_pass_required": 5,
    "clear_window_on_promote": True,
}

STAGES = [
    {
        "name": "純導航",
        "goals": 8, "static_obstacles": 0, "dynamic_obstacles": 0,
        "walls_min": 0, "walls_max": 2,
        "goal_distance": (2.0, 13.0), "episode_s": 45,
        "gamma": 0.990,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.72, "upgrade_max_cr": 1.0, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 50,
        "downgrade_sr": 0.0, "downgrade_min_cr": 1.0, "downgrade_min_to": 1.0,
    },
    {
        "name": "1S+1D",
        "goals": 7, "static_obstacles": 1, "dynamic_obstacles": 1,
        "walls_min": 0, "walls_max": 3,
        "goal_distance": (2.0, 13.0), "episode_s": 51,
        "gamma": 0.991,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 65,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "2S+2D",
        "goals": 6, "static_obstacles": 2, "dynamic_obstacles": 2,
        "walls_min": 1, "walls_max": 4,
        "goal_distance": (2.0, 13.0), "episode_s": 56,
        "gamma": 0.993,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 80,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "3S+3D",
        "goals": 5, "static_obstacles": 3, "dynamic_obstacles": 3,
        "walls_min": 1, "walls_max": 5,
        "goal_distance": (2.0, 13.0), "episode_s": 61,
        "gamma": 0.994,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 95,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "4S+4D",
        "goals": 4, "static_obstacles": 4, "dynamic_obstacles": 4,
        "walls_min": 2, "walls_max": 6,
        "goal_distance": (2.0, 13.0), "episode_s": 67,
        "gamma": 0.996,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 110,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "5S+5D",
        "goals": 3, "static_obstacles": 5, "dynamic_obstacles": 5,
        "walls_min": 2, "walls_max": 7,
        "goal_distance": (2.0, 13.0), "episode_s": 72,
        "gamma": 0.997,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 125,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "6S+6D",
        "goals": 2, "static_obstacles": 6, "dynamic_obstacles": 6,
        "walls_min": 3, "walls_max": 8,
        "goal_distance": (2.0, 13.0), "episode_s": 78,
        "gamma": 0.998,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.40, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 140,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
    {
        "name": "終極挑戰",
        "goals": 1, "static_obstacles": 7, "dynamic_obstacles": 7,
        "walls_min": 4, "walls_max": 8,
        "goal_distance": (2.0, 13.0), "episode_s": 90,
        "gamma": 0.998,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375, "obstacle_speed": 0.8,
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
