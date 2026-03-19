"""Charge 導航 MDP 模組 — VLP16 訓練使用的子模組

子模組：
- actions: 離散差動驅動動作
- observations: VLP16 觀測函數
- rewards: 獎勵函數（PBRS + 目標 + 安全）
- terminations: 終止條件
- events: 事件函數（重置、障礙物）
- wall_layout: 牆壁幾何工具（原 core/wall_layout.py）

未使用的模組已移至 temp/ 目錄。
"""

# ============================================================================
# 動作 (Actions)
# ============================================================================
from .actions import (
    DiscreteDifferentialDriveAction,
    DiscreteDifferentialDriveActionCfg,
)

# ============================================================================
# 觀測 (Observations)
# ============================================================================
from .observations import (
    check_finite,
    goal_position_in_robot_frame,
    goal_distance,
    base_velocity_xy,
    base_angular_velocity_z,
    dynamic_obstacles_state,
    lidar_vlp16_to_2d_bins,
    topk_obstacles_body_frame,
    robot_heading_normalized,
    robot_position_local,
    discrete_applied_action,
)

# ============================================================================
# 獎勵 (Rewards)
# ============================================================================
from .rewards import (
    _print_diagnostics,
    _check_reward_term,
    reaching_goal,
    collision_occurred,
    collision_contact_occurred,
    potential_progress_reward,
    smooth_collision_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    acceleration_squared_penalty,
    discrete_acceleration_squared_penalty,
    angular_velocity_squared_penalty,
    velocity_too_low_penalty,
)

# ============================================================================
# 終止條件 (Terminations)
# ============================================================================
from .terminations import (
    goal_reached,
    robot_tipped_over,
    robot_flying,
    physics_explosion,
)

# ============================================================================
# 事件 (Events)
# ============================================================================
from .events import (
    reset_obstacles,
    move_obstacles_vectorized,
    reset_root_state_random_safe,
    randomize_obstacles_by_difficulty,
    set_obstacle_metadata,
    get_obstacle_num,
    get_obstacle_sizes,
    get_obstacle_metadata,
    init_perenv_walls,
    randomize_walls,
)

# ============================================================================
# 牆壁幾何（原 core/wall_layout.py）
# ============================================================================
from .wall_layout import (
    MAZE_WALLS,
    WALL_SLOT_SPECS,
    MAX_WALL_SLOTS,
    get_wall_tensors,
    get_all_wall_tensors,
    check_wall_proximity_batch,
    check_los_batch,
    check_wall_proximity_perenv,
    check_los_perenv,
    get_combined_wall_data,
)

__all__ = [
    # 動作
    "DiscreteDifferentialDriveAction",
    "DiscreteDifferentialDriveActionCfg",
    # 觀測
    "check_finite",
    "goal_position_in_robot_frame",
    "goal_distance",
    "base_velocity_xy",
    "base_angular_velocity_z",
    "dynamic_obstacles_state",
    "lidar_vlp16_to_2d_bins",
    "topk_obstacles_body_frame",
    "robot_heading_normalized",
    "robot_position_local",
    "discrete_applied_action",
    # 獎勵
    "_print_diagnostics",
    "_check_reward_term",
    "reaching_goal",
    "collision_occurred",
    "collision_contact_occurred",
    "potential_progress_reward",
    "smooth_collision_penalty",
    "collision_terminal_penalty",
    "per_step_time_penalty",
    "acceleration_squared_penalty",
    "discrete_acceleration_squared_penalty",
    "angular_velocity_squared_penalty",
    "velocity_too_low_penalty",
    # 終止條件
    "goal_reached",
    "robot_tipped_over",
    "robot_flying",
    "physics_explosion",
    # 事件
    "reset_obstacles",
    "move_obstacles_vectorized",
    "reset_root_state_random_safe",
    "randomize_obstacles_by_difficulty",
    "set_obstacle_metadata",
    "get_obstacle_num",
    "get_obstacle_sizes",
    "get_obstacle_metadata",
    "init_perenv_walls",
    "randomize_walls",
    # 牆壁幾何
    "MAZE_WALLS",
    "WALL_SLOT_SPECS",
    "MAX_WALL_SLOTS",
    "get_wall_tensors",
    "get_all_wall_tensors",
    "check_wall_proximity_batch",
    "check_los_batch",
    "check_wall_proximity_perenv",
    "check_los_perenv",
    "get_combined_wall_data",
]
