"""觀測模組

提供所有機器人觀測函數,包括:
- 雷達觀測 (lidar_scan, lidar_scan_2d_sweep)
- 目標觀測 (goal_position_in_robot_frame, goal_distance)
- 機器人狀態 (base_velocity_xy, safe_last_action, time_remaining_ratio, alive_flag)
- 障礙物觀測 (dynamic_obstacles_state, static_obstacles_relative_state)
- 路徑觀測 (path_relative: 相對路徑、進度、CTE等)
- 診斷觀測 (charge_dies_at_birth_probability)
- 工具函數 (check_finite)
- 固定拓撲觀測 (fixed_topology: 122 維固定觀測空間)
"""

from .utils import check_finite
from .functions import (
    # 雷達觀測
    lidar_scan,
    lidar_scan_2d_sweep,
    # 目標觀測
    goal_position_in_robot_frame,
    goal_distance,
    # 機器人狀態
    safe_last_action,
    base_velocity_xy,
    base_angular_velocity_z,
    time_remaining_ratio,
    alive_flag,
    # 朝向觀測
    heading_error_to_goal,
    # 障礙物觀測
    dynamic_obstacles_state,
    # 診斷觀測
    charge_dies_at_birth_probability,
)

# AIT* 路徑觀測（用於層級式導航）
from .path_relative import (
    relative_path_displacement,
    path_tangent_direction,
    path_progress,
    cross_track_error_obs,
    aitstar_heuristic_field,
    combined_path_observations,
)

# 層級式導航觀測（AIT* + RL 整合）
from .hierarchical_navigation import (
    local_goal_polar,
    local_goal_cartesian,
    navigation_features,
    lidar_with_navigation,
    cross_track_error_with_heading,
    lookahead_goal_obs,
    update_local_goal_from_aitstar,
)

# 障礙物觀測（Phase 1+ 擴展）
from .obstacle_observations import (
    static_obstacles_relative_state,
    static_obstacles_polar,
    dynamic_obstacles_relative_state,
    topk_obstacles_goal_centric,
    collect_obstacle_info_from_scene,
)

# 固定拓撲觀測系統 (122 維固定觀測空間)
from .fixed_topology import (
    # KNN 障礙物觀測
    nearest_static_obstacles,
    nearest_dynamic_obstacles,
    # 導航指令
    navigation_command,
    # 本體感覺
    proprioception,
    # 維度常量
    LIDAR_DIM,
    STATIC_SLOTS_DIM,
    DYNAMIC_SLOTS_DIM,
    NAV_DIM,
    PROPRIOCEPTION_DIM,
    TOTAL_OBS_DIM,
)

__all__ = [
    # 工具
    "check_finite",
    # 雷達
    "lidar_scan",
    "lidar_scan_2d_sweep",
    # 目標
    "goal_position_in_robot_frame",
    "goal_distance",
    # 機器人狀態
    "safe_last_action",
    "base_velocity_xy",
    "base_angular_velocity_z",
    "time_remaining_ratio",
    "alive_flag",
    # 朝向觀測
    "heading_error_to_goal",
    # 障礙物
    "dynamic_obstacles_state",
    # 診斷觀測
    "charge_dies_at_birth_probability",
    # AIT* 路徑觀測
    "relative_path_displacement",
    "path_tangent_direction",
    "path_progress",
    "cross_track_error_obs",
    "aitstar_heuristic_field",
    "combined_path_observations",
    # 層級式導航觀測
    "local_goal_polar",
    "local_goal_cartesian",
    "navigation_features",
    "lidar_with_navigation",
    "cross_track_error_with_heading",
    "lookahead_goal_obs",
    "update_local_goal_from_aitstar",
    # 障礙物觀測（Phase 1+）
    "static_obstacles_relative_state",
    "static_obstacles_polar",
    "dynamic_obstacles_relative_state",
    "topk_obstacles_goal_centric",
    "collect_obstacle_info_from_scene",
    # 固定拓撲觀測系統 (122 維)
    "nearest_static_obstacles",
    "nearest_dynamic_obstacles",
    "navigation_command",
    "proprioception",
    "LIDAR_DIM",
    "STATIC_SLOTS_DIM",
    "DYNAMIC_SLOTS_DIM",
    "NAV_DIM",
    "PROPRIOCEPTION_DIM",
    "TOTAL_OBS_DIM",
]