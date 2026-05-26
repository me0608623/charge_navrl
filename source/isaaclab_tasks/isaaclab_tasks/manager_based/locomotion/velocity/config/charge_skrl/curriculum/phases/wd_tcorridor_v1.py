"""wd_tcorridor_v1 — T 走廊實驗用 curriculum

Stage 1: TC1_endurance_g1      — 基於 SA5: goals=1, penalty=-30
Stage 2: TC2_sa6_dense         — SA6 T 走廊版: goals=1, penalty=-25, timeout=-12.5, near_miss 20%
Stage 3: TC3_sa7_high_pressure — SA7 T 走廊版: goals=1, penalty=-50, timeout=-25, occlusion 20%, dynamic=8
"""

from __future__ import annotations

from .wd_single_agent_v1 import _flatten_phase

GLOBAL = {
    "obstacle_mode": "rule_based",
    "upgrade_pass_required": 5,
    "clear_window_on_promote": True,
    "trainer_sync_allowlist": [
        "lr", "rnn_lr", "vf_coeff", "max_grad_norm",
        "aux_grad_clip", "wd_actor_update_clip", "wd_critic_update_clip",
    ],
}

STAGES = [
    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 (唯一) — T corridor endurance: 1 goal, penalty -30
    # 基於 SA5_endurance，goals 1, penalty ↑↑
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "TC1_endurance_g1",

        "scene": {
            "goals": 1,
            "goal_distance": (2.0, 9.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 5,
            "dynamic_obstacles_min": 3,
            "walls_min": 0,
            "walls_max": 0,
            "wall_length": 0.0,
            "obs_near_goal_count": 1,
            "obs_near_goal_radius": 2.0,
            "goal_move_speed": 0.15,
            "goal_move_max_radius": 2.0,
            "goal_move_behavior": "random_walk",
            "goal_move_angular_speed": 0.4,
            "episode_s": 75,
            "gamma": 0.995,
        },

        "reward": {
            "penalty_hit": -30.0,
            "reward_get_goal": 40.0,
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.008,
            "entropy_angular": 0.015,
        },

        "behavior": {
            "obstacle_speed": 0.85,
            "behavior_mix": {
                "patrol": 0.20,
                "random_walk": 0.15,
                "static": 0.10,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.15,
                "corridor_crossing": 0.25,
            },
            "speed_overrides": {
                "patrol": {"speed_range": (0.30, 0.65)},
                "random_walk": {"speed_range": (0.25, 0.60)},
                "horizontal_crossing": {"speed_range": (0.35, 0.70)},
                "path_crossing": {"speed_range": (0.30, 0.65)},
                "corridor_crossing": {"speed_range": (0.20, 0.45)},
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
            "upgrade_sr": 0.99,
            "upgrade_max_cr": 0.01,
            "upgrade_max_to": 0.01,
            "upgrade_min_dyn_sr": 0.99,
            "min_stage_updates": 99999,
            "downgrade_sr": 0.0,
            "downgrade_min_cr": 1.0,
            "downgrade_min_to": 1.0,
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 — TC2: SA6 T 走廊版 dense 避障
    # 基於 SA6_dense_avoid T 走廊化：goals=1, dynamic ↑6, near_miss 20%,
    # penalty -25, timeout -12.5, episode 90s, no random walls
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "TC2_sa6_dense",

        "scene": {
            "goals": 1,                         # TC 維持 1 goal
            "goal_distance": (2.0, 9.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 6,             # ↑ from 5 (SA6)
            "dynamic_obstacles_min": 5,         # ↑ from 3
            "walls_min": 0,                     # TC 不用隨機牆
            "walls_max": 0,
            "wall_length": 0.0,
            "obs_near_goal_count": 2,
            "obs_near_goal_radius": 2.0,
            "goal_move_speed": 0.20,            # SA6 level
            "goal_move_max_radius": 2.5,
            "goal_move_behavior": "random_walk",
            "goal_move_angular_speed": 0.5,
            "episode_s": 90,                    # ↑ from 75
            "gamma": 0.996,
        },

        "reward": {
            "penalty_hit": -25.0,               # SA6 level
            "penalty_timeout": -12.5,           # = penalty_hit / 2
            "reward_get_goal": 40.0,
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.008,
            "entropy_angular": 0.015,
        },

        "behavior": {
            "obstacle_speed": 0.90,             # SA6 level
            "behavior_mix": {
                "patrol": 0.15,
                "random_walk": 0.15,
                "static": 0.05,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.15,
                "corridor_crossing": 0.15,      # TC 走廊穿越
                "near_miss": 0.20,              # 高速擦身
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
    # Stage 3 — TC3: SA7 T 走廊版 high pressure
    # 基於 SA7_high_pressure T 走廊化：goals=1, dynamic ↑8, occlusion 20%,
    # penalty -50, timeout -25, episode 120s, obs_speed 1.0
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "TC3_sa7_high_pressure",

        "scene": {
            "goals": 1,                         # TC 維持 1 goal
            "goal_distance": (2.0, 9.0),
            "static_obstacles": 3,
            "dynamic_obstacles": 8,             # ↑ from 6 (SA7)
            "dynamic_obstacles_min": 6,         # ↑ from 5
            "walls_min": 0,                     # TC 不用隨機牆
            "walls_max": 0,
            "wall_length": 0.0,
            "obs_near_goal_count": 2,
            "obs_near_goal_radius": 2.0,
            "goal_move_speed": 0.25,            # SA7 level ↑ from 0.20
            "goal_move_max_radius": 3.0,        # ↑ from 2.5
            "goal_move_behavior": "patrol",     # SA7: waypoint 巡邏
            "goal_move_angular_speed": 0.5,
            "episode_s": 120,                   # ↑ from 90
            "gamma": 0.997,
        },

        "reward": {
            "penalty_hit": -50.0,               # SA7 level ↑ from -25
            "penalty_timeout": -25.0,           # = penalty_hit / 2
            "reward_get_goal": 40.0,
            "cost_operate": 0.03,
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.005,
            "entropy_angular": 0.01,
        },

        "behavior": {
            "obstacle_speed": 1.00,             # SA7 level ↑ from 0.90
            "behavior_mix": {
                "patrol": 0.10,
                "random_walk": 0.10,
                "static": 0.05,
                "horizontal_crossing": 0.15,
                "path_crossing": 0.10,
                "corridor_crossing": 0.10,      # TC 走廊穿越
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
            "upgrade_sr": 0.99,
            "upgrade_max_cr": 0.01,
            "upgrade_max_to": 0.01,
            "upgrade_min_dyn_sr": 0.99,
            "min_stage_updates": 99999,
            "downgrade_sr": 0.0,
            "downgrade_min_cr": 1.0,
            "downgrade_min_to": 1.0,
        },
    },
]

CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
