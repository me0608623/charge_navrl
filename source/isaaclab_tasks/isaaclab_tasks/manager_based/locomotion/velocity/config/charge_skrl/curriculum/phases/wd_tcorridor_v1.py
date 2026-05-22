"""wd_tcorridor_v1 — T 走廊實驗用 single-stage curriculum

基於 SA5_endurance 修改：goals=1, penalty=-30。
只有 1 個 stage (fixed_stage)，不做自動升降階。
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
]

CONFIG = {
    "obstacle_mode": GLOBAL["obstacle_mode"],
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
