# ============================================================================
# MDP 模組統一導出介面
# ============================================================================
"""
Charge 導航 MDP 模組

此模組包含強化學習環境的核心實現：
- Actions: 動作類
- Observations: 觀測函數
- Rewards: 獎勵函數
- Terminations: 終止條件
- Events: 事件函數
- Path Planner: AIT* 路徑規劃器 (層級式導航)
- Curriculum: 課程學習管理器 (4階段漸進式學習)

統一從此模組導出所有 MDP 相關的類和函數。
"""

# ============================================================================
# 動作 (Actions)
# ============================================================================
from .actions import (
    DifferentialDriveAction,
    DifferentialDriveActionCfg,
)

# ============================================================================
# 觀測 (Observations)
# ============================================================================
from .observations import (
    check_finite,
    lidar_scan,
    lidar_scan_2d_sweep,
    goal_position_in_robot_frame,
    goal_distance,
    base_velocity_xy,
    base_angular_velocity_z,
    time_remaining_ratio,
    alive_flag,
    heading_error_to_goal,
    safe_last_action,
    dynamic_obstacles_state,
    charge_dies_at_birth_probability,
)

# ============================================================================
# 獎勵 (Rewards)
# ============================================================================
from .rewards import (
    # 目標獎勵
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
    # 安全獎勵
    obstacle_avoidance_reward,
    collision_penalty,
    progressive_collision_penalty,
    safe_navigation_bonus,
    speed_control_near_obstacles,
    wall_collision_penalty,
    collision_occurred,
    collision_contact_occurred,
    # 運動獎勵
    forward_velocity_reward,
    time_out_penalty,
    forward_motion_reward,
    move_reward,
    action_rate_penalty,
    # 對齊獎勵
    alignment_reward,
    # 路徑跟隨獎勵 (A* + RL 層級式導航)
    progress_along_path,
    cross_track_error,
    path_direction_reward,
    path_following_reward,
    generate_straight_path,
    # 工具
    _print_diagnostics,
    _check_reward_term,
    ExpectedValueTracker,
)

# ============================================================================
# 終止條件 (Terminations)
# ============================================================================
from .terminations import (
    goal_reached,
    robot_tipped_over,
    robot_flying,
    wall_collision,
)

# ============================================================================
# 事件 (Events)
# ============================================================================
from .events import reset_obstacles, reset_root_state_fixed_per_env

# ============================================================================
# 核心狀態管理 (Core)
# ============================================================================
from .core import (
    set_obstacle_metadata,
    get_obstacle_num,
    get_obstacle_sizes,
    get_obstacle_metadata,
    reset_obstacle_targets,
    get_obstacle_start_positions,
    get_obstacle_directions,
    update_obstacle_start_position,
    update_obstacle_direction,
)

# ============================================================================
# 路徑規劃器 (Path Planner) - AIT* 全域路徑規劃
# ============================================================================
from .path_planner import (
    AITStarPathPlanner,
    AITStarPlannerCfg,
    IsaacLabCollisionChecker,
    create_aitstar_planner,
    CurriculumManager,
    CurriculumStageConfig,
    CurriculumMetrics,
    CURRICULUM_STAGES,
    create_curriculum_manager,
    EnvironmentMap,
    EnvironmentMapCfg,
    PathSmoother,
    PathSmootherCfg,
    # Frenet Frame 座標轉換 (Global -> Local) ⭐
    FrenetTransform,
    robot_frenet_observations,
    frenet_to_global,
    # 局部目標提取 (Carrot-on-stick) ⭐
    LocalGoalConfig,
    LocalGoalExtractor,
    extract_local_goals_batch,
    # 牆壁幾何採樣 ⭐
    get_wall_bounding_boxes,
)

# ============================================================================
# 統一導出列表
# ============================================================================
__all__ = [
    # 動作
    "DifferentialDriveAction",
    "DifferentialDriveActionCfg",
    # 觀測
    "check_finite",
    "lidar_scan",
    "lidar_scan_2d_sweep",
    "goal_position_in_robot_frame",
    "goal_distance",
    "base_velocity_xy",
    "base_angular_velocity_z",
    "time_remaining_ratio",
    "alive_flag",
    "heading_error_to_goal",
    "safe_last_action",
    "dynamic_obstacles_state",
    "charge_dies_at_birth_probability",
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
    # 路徑跟隨獎勵 (A* + RL 層級式導航)
    "progress_along_path",
    "cross_track_error",
    "path_direction_reward",
    "path_following_reward",
    "generate_straight_path",
    # 獎勵工具
    "_print_diagnostics",
    "_check_reward_term",
    "ExpectedValueTracker",
    # 終止條件
    "goal_reached",
    "robot_tipped_over",
    "robot_flying",
    "wall_collision",
    # 事件
    "reset_obstacles",
    "reset_root_state_fixed_per_env",
    # 核心
    "set_obstacle_metadata",
    "get_obstacle_num",
    "get_obstacle_sizes",
    "get_obstacle_metadata",
    "reset_obstacle_targets",
    "get_obstacle_start_positions",
    "get_obstacle_directions",
    "update_obstacle_start_position",
    "update_obstacle_direction",
    # 路徑規劃器 (AIT* 全域路徑規劃)
    "AITStarPathPlanner",
    "AITStarPlannerCfg",
    "IsaacLabCollisionChecker",
    "create_aitstar_planner",
    # 課程學習管理器
    "CurriculumManager",
    "CurriculumStageConfig",
    "CurriculumMetrics",
    "CURRICULUM_STAGES",
    "create_curriculum_manager",
    # 環境地圖
    "EnvironmentMap",
    "EnvironmentMapCfg",
    # 路徑平滑器
    "PathSmoother",
    "PathSmootherCfg",
    # Frenet Frame 座標轉換 ⭐
    "FrenetTransform",
    "robot_frenet_observations",
    "frenet_to_global",
    # 局部目標提取 ⭐
    "LocalGoalConfig",
    "LocalGoalExtractor",
    "extract_local_goals_batch",
    # 牆壁幾何採樣 ⭐
    "get_wall_bounding_boxes",
    "get_wall_prims_from_usd",  # 🆕 直接從 USD 獲取牆壁 prims
]
