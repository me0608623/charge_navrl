"""事件模組

包含所有事件相關的實現：
- obstacles: 障礙物重置和移動
- curriculum: 自適應課程學習（障礙物數量）
- goal_distance_curriculum: 目標距離課程學習（Phase 3）
- reset: 自適應參數重置
- curriculum_strategies: 課程學習策略（用於系統性比較實驗）
"""

from .obstacles import (
    reset_obstacles,
    move_obstacles,
)

from .curriculum import (
    initialize_adaptive_curriculum,
    get_success_rate,
    get_collision_rate,
    update_adaptive_curriculum,
    adaptive_curriculum_update,
)

# Phase 3：目標距離課程學習
from .goal_distance_curriculum import (
    initialize_goal_distance_curriculum,
    get_success_rate_for_distance,
    update_goal_distance_curriculum,
    adaptive_goal_distance_update,
    DIFFICULTY_DISTANCE_MAP,
)

from .reset import (
    reset_obstacles_with_adaptive_params,
    hide_unused_obstacles,
    reset_root_state_fixed_per_env,
)

# 課程學習策略（用於系統性比較實驗）
from .curriculum_strategies import (
    CurriculumStrategy,
    CurriculumMetrics,
    NoCurriculum,
    LinearCurriculum,
    ExponentialCurriculum,
    LogarithmicCurriculum,
    AdaptiveCurriculum,
    SigmoidCurriculum,
    get_curriculum_strategy,
    list_available_strategies,
    compare_strategies_theoretical,
    CurriculumExperimentConfig,
)

__all__ = [
    # 障礙物事件
    "reset_obstacles",
    "move_obstacles",
    # 課程學習事件（障礙物數量）
    "initialize_adaptive_curriculum",
    "get_success_rate",
    "get_collision_rate",
    "update_adaptive_curriculum",
    "adaptive_curriculum_update",
    # Phase 3：目標距離課程學習
    "initialize_goal_distance_curriculum",
    "get_success_rate_for_distance",
    "update_goal_distance_curriculum",
    "adaptive_goal_distance_update",
    "DIFFICULTY_DISTANCE_MAP",
    # 自適應重置事件
    "reset_obstacles_with_adaptive_params",
    "hide_unused_obstacles",
    # Agent固定位置重置
    "reset_root_state_fixed_per_env",
    # 課程學習策略（用於系統性比較實驗）
    "CurriculumStrategy",
    "CurriculumMetrics",
    "NoCurriculum",
    "LinearCurriculum",
    "ExponentialCurriculum",
    "LogarithmicCurriculum",
    "AdaptiveCurriculum",
    "SigmoidCurriculum",
    "get_curriculum_strategy",
    "list_available_strategies",
    "compare_strategies_theoretical",
    "CurriculumExperimentConfig",
]

