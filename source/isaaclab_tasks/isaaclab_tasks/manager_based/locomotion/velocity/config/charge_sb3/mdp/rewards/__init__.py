"""獎勵模組

此模組包含所有獎勵函數，分為以下類別：
- utils: 工具函數（診斷、檢查）
- goal_rewards: 目標相關獎勵
- safety_rewards: 安全相關獎�勵（避障、碰撞）
- motion_rewards: 運動相關獎勵（速度、超時）
- path_following: A* + RL 層級式導航路徑跟隨獎勵
- aitstar_guided: AIT* 引導獎勵（課程學習）
- corridor_based: 走廊式獎勵（解決獎勵衝突）⭐
- cbf_penalty: CBF 修正懲罰（用於 SB3 PPO + CBF 整合）⚠️
- perception_rewards: 感知感知獎勵（v2 優化）🆕
- dynamic_corridor: 動態廊道獎勵（v3 優化）🆕
- navrl_rewards: NavRL 風格獎勵（移植自 NavRL 論文）🆕
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
    action_smoothness_linear,  # 新增：線性平滑度懲罰
)

from .path_following import (
    progress_along_path,
    cross_track_error,
    path_direction_reward,
    path_following_reward,
    generate_straight_path,
)

# AIT* 引導獎勵（用於課程學習）
from .aitstar_guided import (
    AITStarRewardWeights,
    aitstar_progress_reward,
    aitstar_cross_track_error,
    aitstar_alignment_reward,
    aitstar_velocity_reward,
    aitstar_guided_reward,
    curriculum_adjusted_reward,
)

# 走廊式獎勵（解決獎勵衝突）⭐
from .corridor_based import (
    CorridorRewardConfig,
    corridor_following_reward,
    dynamic_corridor_width,
    corridor_guided_total_reward,
    survival_priority_reward,
    AdaptiveRewardWeights,
)

from .expected_value_tracker import ExpectedValueTracker

# CBF 修正懲罰（用於 SB3 PPO + CBF 整合）
from .cbf_penalty import (
    cbf_correction_penalty,
    cbf_intervention_penalty,
    cbf_adaptive_penalty,
)

# 感知感知獎勵（v2 優化）
from .perception_rewards import (
    lidar_clearance_reward,
    too_close_penalty,
    lidar_utilization_bonus,
    safety_field_penalty,  # 新增：安場懲罰 (Safety Field Penalty)
    collision_penalty_reward,  # 新增：連續碰撞懲罰
    wall_proximity_penalty,  # 新增：牆壁過近懲罰
)

# 動態廊道獎勵（v3 優化）
from .dynamic_corridor import (
    dynamic_corridor_reward,
    narrow_gate_bonus,
)

# 層級式導航獎勵（AIT* + RL 整合）🆕
from .hierarchical_rewards import (
    local_goal_reached_reward,
    progress_to_local_goal,
    heading_alignment_reward,
    collision_penalty as hierarchical_collision_penalty,
    action_smoothness_penalty,
    proximity_to_obstacle_penalty,
    hierarchical_navigation_reward,
    reset_reward_tracking,
)

# 🆕 動態 Goal 重置獎勵
from .dynamic_goal_rewards import (
    dynamic_goal_respawn_reward,
    dynamic_goal_bonus_reward,
    dynamic_goal_progress_reward,
)

# 🆕 NavRL 風格獎勵（移植自 NavRL 論文）
from .navrl_rewards import (
    navrl_velocity_reward,
    navrl_safety_reward_lidar,
    navrl_dynamic_obstacle_safety_reward,
    navrl_smoothness_penalty,
    navrl_height_penalty,
    navrl_total_reward,
)

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
    "action_smoothness_linear",  # 新增：線性平滑度懲罰
    # 對齊獎勵
    "alignment_reward",
    # 路徑跟隨獎勵 (A* + RL 層級式導航)
    "progress_along_path",
    "cross_track_error",
    "path_direction_reward",
    "path_following_reward",
    "generate_straight_path",
    # AIT* 引導獎勵 (課程學習)
    "AITStarRewardWeights",
    "aitstar_progress_reward",
    "aitstar_cross_track_error",
    "aitstar_alignment_reward",
    "aitstar_velocity_reward",
    "aitstar_guided_reward",
    "curriculum_adjusted_reward",
    # 走廊式獎勵 (解決獎勵衝突) ⭐
    "CorridorRewardConfig",
    "corridor_following_reward",
    "dynamic_corridor_width",
    "corridor_guided_total_reward",
    "survival_priority_reward",
    "AdaptiveRewardWeights",
    # 期望值追蹤
    "ExpectedValueTracker",
    # CBF 修正懲罰 (SB3 PPO + CBF 整合)
    "cbf_correction_penalty",
    "cbf_intervention_penalty",
    "cbf_adaptive_penalty",
    # 感知感知獎勵 (v2 優化)
    "lidar_clearance_reward",
    "too_close_penalty",
    "lidar_utilization_bonus",
    "safety_field_penalty",  # 新增：安場懲罰 (Safety Field Penalty)
    "collision_penalty_reward",  # 新增：連續碰撞懲罰
    # 動態廊道獎勵 (v3 優化)
    "dynamic_corridor_reward",
    "narrow_gate_bonus",
    # 層級式導航獎勵 (AIT* + RL 整合) 🆕
    "local_goal_reached_reward",
    "progress_to_local_goal",
    "heading_alignment_reward",
    "hierarchical_collision_penalty",
    "action_smoothness_penalty",
    "proximity_to_obstacle_penalty",
    "hierarchical_navigation_reward",
    "reset_reward_tracking",
    # 🆕 動態 Goal 重置獎勵
    "dynamic_goal_respawn_reward",
    "dynamic_goal_bonus_reward",
    "dynamic_goal_progress_reward",
    # 🆕 NavRL 風格獎勵
    "navrl_velocity_reward",
    "navrl_safety_reward_lidar",
    "navrl_dynamic_obstacle_safety_reward",
    "navrl_smoothness_penalty",
    "navrl_height_penalty",
    "navrl_total_reward",
]
