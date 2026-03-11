"""事件模組 — VLP16 訓練使用的事件函數

保留的模組：
- obstacles: 障礙物重置和移動
- mixed_parallel: 混合難度障礙物配置
- reset: 安全隨機重置
- state: 障礙物狀態管理（從 core/ 遷移）
"""

from .obstacles import (
    reset_obstacles,
    move_obstacles,
    move_obstacles_goal_directed,
    enforce_obstacle_bounds,
    move_obstacles_vectorized,
)

from .reset import (
    reset_obstacles_with_adaptive_params,
    hide_unused_obstacles,
    reset_root_state_fixed_per_env,
    reset_root_state_random_safe,
)

from .mixed_parallel import (
    randomize_obstacles_by_difficulty,
    get_env_difficulty,
    get_env_num_visible_obstacles,
)

from .state import (
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

__all__ = [
    # 障礙物事件
    "reset_obstacles",
    "move_obstacles",
    "move_obstacles_goal_directed",
    "enforce_obstacle_bounds",
    "move_obstacles_vectorized",
    # 安全重置
    "reset_obstacles_with_adaptive_params",
    "hide_unused_obstacles",
    "reset_root_state_fixed_per_env",
    "reset_root_state_random_safe",
    # 混合平行環境事件
    "randomize_obstacles_by_difficulty",
    "get_env_difficulty",
    "get_env_num_visible_obstacles",
    # 障礙物狀態管理（原 core/state.py）
    "set_obstacle_metadata",
    "get_obstacle_num",
    "get_obstacle_sizes",
    "get_obstacle_metadata",
    "reset_obstacle_targets",
    "get_obstacle_start_positions",
    "get_obstacle_directions",
    "update_obstacle_start_position",
    "update_obstacle_direction",
]
