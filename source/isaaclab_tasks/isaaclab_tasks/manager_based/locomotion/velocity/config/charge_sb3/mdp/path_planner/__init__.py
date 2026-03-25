"""
路徑規劃模組 (Path Planning Module)

此模組實現分層導航系統：
- AITStarPathPlanner: AIT* 路徑規劃器（推理模式，全局路徑規劃）
- VirtualPlanner: 虛擬規劃器（訓練模式，隨機目標點，FPS 高）⭐
- IsaacLabCollisionChecker: Isaac Lab 環境碰撞檢測器
- EnvironmentMap: 環境地圖表示（障礙物、邊界）
- WallGeometrySampler: 從配置直接生成牆壁幾何（數學方法，精準且快速）⭐
- PathSmoother: 路徑平滑化（減少路徑點數量）
- CurriculumManager: 課程學習管理器（4階段漸進式學習）
- FrenetTransform: Frenet Frame 座標轉換器 (Global -> Local) ⭐
- LocalGoalExtractor: 局部目標提取器 (Carrot-on-stick) ⭐

Pipeline (配置 → AIT*):
1. wall_geometry_sampler.get_phase1_wall_configs()
   → 從配置獲取牆壁參數
2. add_walls_from_config()
   → 直接在 occupancy grid 中畫長方形（數學幾何）
3. AITStarPathPlanner.plan_path()
   → 使用精準的障礙物地圖規劃路徑

訓練/推理架構：
- 訓練模式：使用 VirtualPlanner（隨機目標點，高 FPS）
- 推理模式：使用 AITStarPathPlanner（真實全局路徑規劃）

使用方式：
    # 訓練模式
    planner = VirtualPlanner(phase="phase0", min_distance=2.0, max_distance=5.0)
    path = planner.plan_path(start_pos, goal_pos, env)

    # 推理模式
    planner = create_aitstar_planner(
        map_size=(20, 20),
        grid_resolution=0.1,
        robot_radius=0.3,
    )
    path = planner.plan_path(start_pos, goal_pos, env)

課程學習：
    curriculum = create_curriculum_manager(initial_stage=1)

座標轉換（關鍵！）：
    frenet_obs = robot_frenet_observations(env, path)

AIT* 參考實現：/globle_planner/src/aitstar_path_planner/
"""

from .aitstar_adapter import (
    AITStarPathPlanner,
    AITStarPlannerCfg,
    IsaacLabCollisionChecker,
    create_aitstar_planner,
)
from .curriculum_manager import (
    CurriculumManager,
    CurriculumStageConfig,
    CurriculumMetrics,
    CURRICULUM_STAGES,
    create_curriculum_manager,
)
from .environment_map import EnvironmentMap, EnvironmentMapCfg
from .frenet_transform import (
    FrenetTransform,
    robot_frenet_observations,
    frenet_to_global,
)
from .local_goal_extractor import (
    LocalGoalConfig,
    LocalGoalExtractor,
    extract_local_goals_batch,
)
from .path_smoother import PathSmoother, PathSmootherCfg
from .path_visualizer import (
    AITStarPathVisualizer,
    MultiEnvPathVisualizer,
)
from .virtual_planner import VirtualPlanner
from .wall_geometry_sampler import (
    # 推薦使用的新方法（數學幾何，精準且快速）
    WallConfig,
    get_phase0_wall_configs,  # Phase 0: 只有四面邊界牆
    get_phase1_wall_configs,
    get_phase2_wall_configs,
    get_phase3_wall_configs,
    add_rectangle_to_occupancy_grid,
    add_walls_from_config,
    # 舊方法（保留用於向後兼容）
    add_wall_points_to_occupancy_grid,
    WallGeometryVisualizer,
    get_wall_bounding_boxes,
    get_wall_prims_from_usd,
)

__all__ = [
    # 規劃器
    "AITStarPathPlanner",        # AIT* 規劃器（推理模式）
    "AITStarPlannerCfg",
    "IsaacLabCollisionChecker",
    "create_aitstar_planner",
    "VirtualPlanner",             # 虛擬規劃器（訓練模式，FPS 高）⭐
    # 課程學習
    "CurriculumManager",
    "CurriculumStageConfig",
    "CurriculumMetrics",
    "CURRICULUM_STAGES",
    "create_curriculum_manager",
    # 地圖
    "EnvironmentMap",
    "EnvironmentMapCfg",
    # 牆壁幾何（推薦新方法：數學幾何）⭐
    "WallConfig",
    "get_phase0_wall_configs",
    "get_phase1_wall_configs",
    "get_phase2_wall_configs",
    "get_phase3_wall_configs",
    "add_rectangle_to_occupancy_grid",
    "add_walls_from_config",
    # 牆壁幾何（舊方法，保留用於向後兼容）
    "add_wall_points_to_occupancy_grid",
    "WallGeometryVisualizer",
    "get_wall_bounding_boxes",
    "get_wall_prims_from_usd",
    # Frenet Frame 座標轉換 (關鍵！)
    "FrenetTransform",
    "robot_frenet_observations",
    "frenet_to_global",
    # 局部目標提取
    "LocalGoalConfig",
    "LocalGoalExtractor",
    "extract_local_goals_batch",
    # 平滑器
    "PathSmoother",
    "PathSmootherCfg",
    # 視覺化
    "AITStarPathVisualizer",
    "MultiEnvPathVisualizer",
]
