"""獎勵模組

此模組包含所有獎勵函數，分為以下類別：
- utils: 工具函數（診斷、檢查）
- goal_rewards: 目標相關獎勵
- safety_rewards: 安全相關獎勵（避障、碰撞）
- motion_rewards: 運動相關獎勵（速度、超時）
"""

from .utils import (
    _print_diagnostics,
    _check_reward_term,
)

from .goal_rewards import (
    velocity_toward_goal,
    velocity_toward_goal_smooth,
    progress_to_goal,
    reaching_goal,
    approaching_goal_bonus,
    heading_to_goal,
    heading_to_goal_distance_weighted,
    velocity_toward_goal_distance_weighted,
    reverse_toward_goal_distance_weighted,
    velocity_toward_goal_dynamic_gated,
    alignment_reward,
)

from .safety_rewards import (
    obstacle_avoidance_reward,
    collision_penalty,
    progressive_collision_penalty,
    safe_navigation_bonus,
    speed_control_near_obstacles,
    wall_collision_penalty,
    collision_occurred,
    collision_contact_occurred,
)

from .motion_rewards import (
    forward_velocity_reward,
    time_out_penalty,
    forward_motion_reward,
    move_reward,
    action_rate_penalty,
)

from .expected_value_tracker import ExpectedValueTracker

__all__ = [
    # 工具函數
    "_print_diagnostics",
    "_check_reward_term",
    # 目標獎勵
    "velocity_toward_goal",
    "velocity_toward_goal_smooth",
    "progress_to_goal",
    "reaching_goal",
    "approaching_goal_bonus",
    "heading_to_goal",
    "heading_to_goal_distance_weighted",
    "velocity_toward_goal_distance_weighted",
    "reverse_toward_goal_distance_weighted",
    "velocity_toward_goal_dynamic_gated",
    # 安全獎勵
    "obstacle_avoidance_reward",
    "collision_penalty",
    "progressive_collision_penalty",
    "safe_navigation_bonus",
    "speed_control_near_obstacles",
    "wall_collision_penalty",
    "collision_occurred",
    "collision_contact_occurred",
    # 運動獎勵
    "forward_velocity_reward",
    "time_out_penalty",
    "forward_motion_reward",
    "move_reward",
    "action_rate_penalty",
    # 對齊獎勵
    "alignment_reward",
    # 期望值追蹤
    "ExpectedValueTracker",
]
