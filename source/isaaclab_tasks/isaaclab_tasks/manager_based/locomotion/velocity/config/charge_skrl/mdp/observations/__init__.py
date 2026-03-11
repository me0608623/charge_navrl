"""觀測模組 — VLP16 訓練使用的觀測函數"""

from .utils import check_finite

from .functions import (
    # v2 ego state (4D)
    normalized_linear_acceleration,
    normalized_linear_velocity,
    base_angular_velocity_z,
    robot_radius_obs,
    # goal state (2D)
    goal_position_in_robot_frame,
    # time state (1D)
    time_remaining_ratio,
    # static LiDAR (72D) — base config 使用
    lidar_scan_2d_sweep,
    # legacy (保留但 v2 不使用)
    goal_distance,
    base_velocity_xy,
    dynamic_obstacles_state,
    safe_last_action,
    discrete_last_action,
    heading_error_to_goal,
    alive_flag,
    charge_dies_at_birth_probability,
)

from .obs_functions import (
    # static state (72D)
    lidar_vlp16_to_2d_bins,
    # obs state (60D) — v2 dynamic obstacle observation
    topk_obstacles_6d,
    # goal-centric obstacle observation (phase0 使用)
    topk_obstacles_goal_centric,
    # legacy (保留但 v2 不使用)
    topk_obstacles_body_frame,
    robot_heading_normalized,
    robot_position_local,
    discrete_applied_action,
)

__all__ = [
    "check_finite",
    # v2 observation design (79D)
    "normalized_linear_acceleration",
    "normalized_linear_velocity",
    "base_angular_velocity_z",
    "robot_radius_obs",
    "goal_position_in_robot_frame",
    "time_remaining_ratio",
    "lidar_vlp16_to_2d_bins",
    "lidar_scan_2d_sweep",
    "topk_obstacles_goal_centric",
    # legacy
    "goal_distance",
    "base_velocity_xy",
    "dynamic_obstacles_state",
    "safe_last_action",
    "discrete_last_action",
    "heading_error_to_goal",
    "alive_flag",
    "charge_dies_at_birth_probability",
    "topk_obstacles_body_frame",
    "robot_heading_normalized",
    "robot_position_local",
    "discrete_applied_action",
]
