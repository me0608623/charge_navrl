"""核心模組

提供狀態管理功能,包括障礙物元數據和動態障礙物狀態。
"""

from .state import (
    # 障礙物元數據
    set_obstacle_metadata,
    get_obstacle_num,
    get_obstacle_sizes,
    get_obstacle_metadata,
    # 動態障礙物狀態
    reset_obstacle_targets,
    get_obstacle_start_positions,
    get_obstacle_directions,
    update_obstacle_start_position,
    update_obstacle_direction,
)

__all__ = [
    # 障礙物元數據
    "set_obstacle_metadata",
    "get_obstacle_num",
    "get_obstacle_sizes",
    "get_obstacle_metadata",
    # 動態障礙物狀態
    "reset_obstacle_targets",
    "get_obstacle_start_positions",
    "get_obstacle_directions",
    "update_obstacle_start_position",
    "update_obstacle_direction",
]
