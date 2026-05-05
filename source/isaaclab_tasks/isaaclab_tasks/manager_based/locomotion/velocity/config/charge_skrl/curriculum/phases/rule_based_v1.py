"""rule_based_v1 — Rule-Based Obstacle Behavior Curriculum (6 stages)

每個 stage 使用 behavior_mix 指定 obstacle 行為分佈比例。
BehaviorScheduler 根據此比例為每個 env 的每個 obstacle slot 分配行為。

物理場景: 20×20m, dt=0.2s, robot v_max=1.0 m/s
"""

GLOBAL = {
    "upgrade_pass_required": 5,
    "clear_window_on_promote": True,
}

STAGES = [
    # ──────────────────────────────────────────────────────────────────────
    # Stage 1: 基礎導航 + 靜態 LiDAR baseline
    #   目的: 高成功率啟動學習。static obstacle 提供 persistent geometry baseline，
    #         讓 RNN 學會「不變的回波 = 不需時序追蹤」。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB1_static_only",
        "goals": 10,
        "static_obstacles": 3,
        "dynamic_obstacles": 0,
        "walls_min": 0, "walls_max": 1,
        "wall_length": 5.0,
        "goal_distance": (2.0, 6.0),
        "episode_s": 45,
        "gamma": 0.990,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.03,
        "entropy_linear": 0.10, "entropy_angular": 0.375,
        "obstacle_speed": 0.0,

        # --- Behavior 配置 ---
        "behavior_mix": {
            "static": 1.0,
        },

        # --- 升階條件 ---
        "upgrade_sr": 0.70, "upgrade_max_cr": 0.15, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 50,
        "downgrade_sr": 0.0, "downgrade_min_cr": 1.0, "downgrade_min_to": 1.0,
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2: 引入可預測動態 (patrol + random walk)
    #   目的: 讓 RNN 開始學習週期性/隨機性時間模式。
    #         Policy 學會「等待」或「加速通過」。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB2_predictable_dynamic",
        "goals": 8,
        "static_obstacles": 2,
        "dynamic_obstacles": 3,
        "walls_min": 0, "walls_max": 1,
        "wall_length": 5.0,
        "goal_distance": (2.0, 7.0),
        "episode_s": 51,
        "gamma": 0.992,
        "penalty_hit": -5.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.20, "entropy_angular": 0.375,
        "obstacle_speed": 0.5,

        # --- Behavior 配置 ---
        "behavior_mix": {
            "static": 0.40,
            "patrol": 0.40,
            "random_walk": 0.20,
        },
        "speed_overrides": {
            "patrol": {"speed_range": (0.2, 0.4)},
            "random_walk": {"speed_range": (0.15, 0.4)},
        },

        # --- 升階條件 ---
        "upgrade_sr": 0.65, "upgrade_max_cr": 0.20, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 65,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 3: 引入 crossing + 速度提升
    #   目的: 訓練 RNN 偵測橫向穿越信號。
    #         Policy 學會 crossing timing。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB3_crossing_intro",
        "goals": 6,
        "static_obstacles": 2,
        "dynamic_obstacles": 5,
        "walls_min": 1, "walls_max": 2,
        "wall_length": 4.5,
        "goal_distance": (2.0, 8.0),
        "episode_s": 56,
        "gamma": 0.994,
        "penalty_hit": -8.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375,
        "obstacle_speed": 0.7,

        # --- Behavior 配置 ---
        "behavior_mix": {
            "static": 0.28,
            "patrol": 0.28,
            "random_walk": 0.14,
            "horizontal_crossing": 0.15,
            "path_crossing": 0.15,
        },
        "speed_overrides": {
            "patrol": {"speed_range": (0.3, 0.6)},
            "random_walk": {"speed_range": (0.2, 0.6)},
            "horizontal_crossing": {"speed_range": (0.4, 0.7)},
            "path_crossing": {"speed_range": (0.3, 0.7)},
        },

        # --- 升階條件 ---
        "upgrade_sr": 0.60, "upgrade_max_cr": 0.25, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 80,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 4: 引入 near-miss + corridor (P2 實作)
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB4_tight_clearance",
        "goals": 4,
        "static_obstacles": 1,
        "dynamic_obstacles": 7,
        "walls_min": 1, "walls_max": 2,
        "wall_length": 4.0,
        "goal_distance": (2.0, 9.0),
        "episode_s": 65,
        "gamma": 0.995,
        "penalty_hit": -12.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375,
        "obstacle_speed": 0.85,

        "behavior_mix": {
            "static": 0.125,
            "patrol": 0.25,
            "random_walk": 0.125,
            "horizontal_crossing": 0.125,
            "path_crossing": 0.125,
            "near_miss": 0.125,
            "corridor_crossing": 0.125,
        },
        "speed_overrides": {
            "patrol": {"speed_range": (0.3, 0.7)},
            "random_walk": {"speed_range": (0.3, 0.7)},
            "horizontal_crossing": {"speed_range": (0.5, 0.9)},
            "path_crossing": {"speed_range": (0.4, 0.8)},
            "near_miss": {"speed_range": (0.4, 0.8)},
            "corridor_crossing": {"speed_range": (0.3, 0.6)},
        },

        "upgrade_sr": 0.55, "upgrade_max_cr": 0.30, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.35, "min_stage_updates": 100,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 5: 引入 occlusion + 全速 (P3 實作)
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB5_occlusion",
        "goals": 3,
        "static_obstacles": 1,
        "dynamic_obstacles": 9,
        "walls_min": 1, "walls_max": 2,
        "wall_length": 3.5,
        "goal_distance": (2.0, 10.0),
        "episode_s": 75,
        "gamma": 0.997,
        "penalty_hit": -15.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375,
        "obstacle_speed": 1.0,

        "behavior_mix": {
            "static": 0.10,
            "patrol": 0.15,
            "random_walk": 0.15,
            "horizontal_crossing": 0.10,
            "path_crossing": 0.10,
            "near_miss": 0.15,
            "corridor_crossing": 0.10,
            "occlusion": 0.15,
        },
        "speed_overrides": {
            "patrol": {"speed_range": (0.4, 0.8)},
            "random_walk": {"speed_range": (0.3, 0.8)},
            "horizontal_crossing": {"speed_range": (0.5, 1.0)},
            "path_crossing": {"speed_range": (0.4, 0.9)},
            "near_miss": {"speed_range": (0.5, 1.0)},
            "corridor_crossing": {"speed_range": (0.3, 0.7)},
        },

        "upgrade_sr": 0.50, "upgrade_max_cr": 0.35, "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.30, "min_stage_updates": 120,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 6: 全 behavior 壓力測試 (最終)
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "RB6_full_pressure",
        "goals": 2,
        "static_obstacles": 1,
        "dynamic_obstacles": 9,
        "walls_min": 1, "walls_max": 3,
        "wall_length": 3.5,
        "goal_distance": (2.0, 12.0),
        "episode_s": 90,
        "gamma": 0.998,
        "penalty_hit": -15.0, "reward_get_goal": 40.0, "cost_operate": 0.0,
        "entropy_linear": 0.30, "entropy_angular": 0.375,
        "obstacle_speed": 1.0,

        "behavior_mix": {
            "static": 0.10,
            "patrol": 0.10,
            "random_walk": 0.15,
            "horizontal_crossing": 0.10,
            "path_crossing": 0.10,
            "near_miss": 0.20,
            "corridor_crossing": 0.10,
            "occlusion": 0.15,
        },
        "speed_overrides": {
            "patrol": {"speed_range": (0.5, 0.85)},
            "random_walk": {"speed_range": (0.4, 0.85)},
            "horizontal_crossing": {"speed_range": (0.6, 1.0)},
            "path_crossing": {"speed_range": (0.5, 1.0)},
            "near_miss": {"speed_range": (0.6, 1.0)},
            "corridor_crossing": {"speed_range": (0.4, 0.8)},
        },

        # 最終 stage — 不升階
        "upgrade_sr": 1.0, "upgrade_max_cr": 0.0, "upgrade_max_to": 0.0,
        "upgrade_min_dyn_sr": 0.0, "min_stage_updates": 0,
        "downgrade_sr": 0.15, "downgrade_min_cr": 0.70, "downgrade_min_to": 0.65,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 匯出 (legacy format for goal_obstacle_curriculum.py)
# ═══════════════════════════════════════════════════════════════════════════

from .wd_single_agent_v1 import _to_legacy_stage

CONFIG = {
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],
    "stages": [_to_legacy_stage(s) for s in STAGES],
}
