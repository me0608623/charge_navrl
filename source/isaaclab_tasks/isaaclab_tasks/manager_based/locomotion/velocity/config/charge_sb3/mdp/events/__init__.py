"""事件模組

包含所有事件相關的實現：
- obstacles: 障礙物重置和移動
- curriculum: 自適應課程學習（障礙物數量）
- goal_distance_curriculum: 目標距離課程學習（Phase 3）
- dynamic_goal: 🆕 Episode 內動態 Goal 重置
- reset: 自適應參數重置
- aitstar_integration: AIT* 全域路徑規劃整合 ⭐
- curriculum_strategies: 課程學習策略（用於系統性比較實驗）
"""

from .obstacles import (
    reset_obstacles,
    move_obstacles,
    move_obstacles_goal_directed,
    enforce_obstacle_bounds,
    move_obstacles_vectorized,
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

# AIT* 路徑規劃整合事件 ⭐
from .aitstar_integration import (
    plan_aitstar_and_update_local_goal,
    sync_local_goal_from_command,
    update_local_goal_during_episode,
)

# 🆕 動態 Goal 重置事件
from .dynamic_goal import (
    initialize_dynamic_goal_respawn,
    respawn_goal_on_reach,
    should_allow_termination,
    reset_dynamic_goal_count,
    get_dynamic_goal_stats,
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

# 🆕 混合平行環境事件
from .mixed_parallel import (
    randomize_obstacles_by_difficulty,
    get_env_difficulty,
    get_env_num_visible_obstacles,
)

__all__ = [
    # 障礙物事件
    "reset_obstacles",
    "move_obstacles",
    "move_obstacles_goal_directed",
    "enforce_obstacle_bounds",
    "move_obstacles_vectorized",
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
    # AIT* 路徑規劃整合事件 ⭐
    "plan_aitstar_and_update_local_goal",
    "sync_local_goal_from_command",
    "update_local_goal_during_episode",
    # 🆕 動態 Goal 重置事件
    "initialize_dynamic_goal_respawn",
    "respawn_goal_on_reach",
    "should_allow_termination",
    "reset_dynamic_goal_count",
    "get_dynamic_goal_stats",
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
    # 🆕 混合平行環境事件
    "randomize_obstacles_by_difficulty",
    "get_env_difficulty",
    "get_env_num_visible_obstacles",
]

