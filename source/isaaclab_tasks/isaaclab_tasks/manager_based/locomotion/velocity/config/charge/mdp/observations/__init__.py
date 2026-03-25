"""觀測模組

提供所有機器人觀測函數,包括:
- 雷達觀測 (lidar_scan, lidar_scan_2d_sweep)
- 目標觀測 (goal_position_in_robot_frame, goal_distance)
- 機器人狀態 (base_velocity_xy, safe_last_action, time_remaining_ratio, alive_flag)
- 障礙物觀測 (dynamic_obstacles_state)
- 診斷觀測 (charge_dies_at_birth_probability)
- 工具函數 (check_finite)
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
    base_angular_velocity_z,  # 新增：角速度觀測
    time_remaining_ratio,
    alive_flag,
    # 朝向觀測
    heading_error_to_goal,  # 新增：朝向誤差觀測
    # 障礙物觀測
    dynamic_obstacles_state,
    # 診斷觀測
    charge_dies_at_birth_probability,
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
    "base_angular_velocity_z",  # 新增：角速度觀測
    "time_remaining_ratio",
    "alive_flag",
    # 朝向觀測
    "heading_error_to_goal",  # 新增：朝向誤差觀測
    # 障礙物
    "dynamic_obstacles_state",
    # 診斷觀測
    "charge_dies_at_birth_probability",
]