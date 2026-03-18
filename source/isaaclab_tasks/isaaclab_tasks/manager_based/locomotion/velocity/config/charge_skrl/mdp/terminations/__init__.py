"""終止條件模組 — VLP16 訓練使用的終止條件"""

from .goal import (
    goal_reached,
)

from .robot_state import (
    robot_tipped_over,
    robot_flying,
    physics_explosion,
    wall_collision_termination,
)

__all__ = [
    "goal_reached",
    "robot_tipped_over",
    "robot_flying",
    "physics_explosion",
    "wall_collision_termination",
]
