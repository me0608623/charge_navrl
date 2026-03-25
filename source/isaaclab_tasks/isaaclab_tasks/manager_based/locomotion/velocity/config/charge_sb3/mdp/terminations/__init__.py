"""終止條件模組

包含所有終止條件相關的實現：
- goal: 目標相關終止條件
- dynamic_goal: 🆕 動態 Goal 重置相關終止條件
- robot_state: 機器人狀態相關終止條件
- collision: 碰撞相關終止條件（牆壁、LiDAR、PhysX）
"""

from .goal import (
    goal_reached,
)

from .dynamic_goal import (
    goal_reached_dynamic,
    goal_reached_with_count,
)

from .robot_state import (
    robot_tipped_over,
    robot_flying,
)

from .collision import (
    wall_collision,
    lidar_collision,  # 🆕 LiDAR 碰撞終止
    physx_contact_collision,  # 🆕 PhysX 物理碰撞終止
)

__all__ = [
    # 目標終止
    "goal_reached",
    # 🆕 動態 Goal 終止
    "goal_reached_dynamic",
    "goal_reached_with_count",
    # 機器人狀態終止
    "robot_tipped_over",
    "robot_flying",
    # 碰撞終止
    "wall_collision",
    "lidar_collision",  # 🆕 LiDAR 碰撞
    "physx_contact_collision",  # 🆕 PhysX 碰撞
]
