"""終止條件模組

包含所有終止條件相關的實現：
- goal: 目標相關終止條件
- robot_state: 機器人狀態相關終止條件
- collision: 碰撞相關終止條件
"""

from .goal import (
    goal_reached,
)

from .robot_state import (
    robot_tipped_over,
    robot_flying,
)

from .collision import (
    wall_collision,
)

__all__ = [
    # 目標終止
    "goal_reached",
    # 機器人狀態終止
    "robot_tipped_over",
    "robot_flying",
    # 碰撞終止
    "wall_collision",
]
