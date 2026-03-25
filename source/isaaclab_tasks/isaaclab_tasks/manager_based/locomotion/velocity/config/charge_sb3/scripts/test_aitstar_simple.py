#!/usr/bin/env python3
"""
AIT* 路徑規劃器獨立測試（不依賴 Isaac Lab 核心）

直接測試 AIT* 算法和適配器的基本功能。
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 添加 AIT* 路徑
_AITSTAR_PATH = "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/globle_planner/src/aitstar_path_planner"
sys.path.insert(0, _AITSTAR_PATH)

# 添加 MDP 路徑（用於適配器）
_MDP_PATH = "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/mdp"
sys.path.insert(0, _MDP_PATH)

print("=" * 60)
print("    AIT* 路徑規劃器獨立測試（Phase 0：空地直線）")
print("=" * 60)

# 測試 1: AIT* 原始算法
print("\n[測試 1] AIT* 原始算法...")
try:
    from aitstar_path_planner.aitstar import (
        AITStar,
        SimpleCollisionChecker,
    )
    print("✅ AIT* 模組導入成功")

    # 創建簡單的障礙物配置（Phase 0: 無障礙物）
    obstacles = []  # Phase 0: 空地

    # 邊界（16x16m 房間）
    bounds_min = np.array([-8.0, -8.0])
    bounds_max = np.array([8.0, 8.0])

    # 創建碰撞檢測器
    collision_checker = SimpleCollisionChecker(
        obstacles=obstacles,
        bounds=(bounds_min, bounds_max),
        robot_radius=0.3,
    )
    print("✅ 碰撞檢測器創建成功")

    # 創建 AIT* 規劃器
    planner = AITStar(
        collision_checker=collision_checker,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        batch_size=100,
        rewire_factor=1.1,
        goal_bias=0.05,
        max_iterations=1000,
    )
    print("✅ AIT* 規劃器創建成功")

    # 測試路徑規劃（Phase 0: 空地直線）
    # 注意: planner.plan() 接受 np.ndarray，不是 State 對象
    start_pos = np.array([-3.0, -3.0])
    goal_pos = np.array([3.0, 3.0])

    print(f"\n執行路徑規劃...")
    print(f"  起點: {start_pos}")
    print(f"  終點: {goal_pos}")

    path, cost = planner.plan(start_pos, goal_pos)

    if path is not None and len(path) > 0:
        print(f"\n✅ 規劃成功!")
        print(f"  路徑點數: {len(path)}")
        print(f"  路徑代價: {cost:.2f}")

        # 檢查是否為直線（Phase 0 空地應該是直線）
        if len(path) == 2:
            print(f"  路徑類型: 直線（符合 Phase 0 預期）")
        else:
            print(f"  路徑類型: 曲線（{len(path)} 個路徑點）")
    else:
        print(f"\n❌ 規劃失敗")

except ImportError as e:
    print(f"❌ AIT* 模組導入失敗: {e}")
    print(f"   請檢查路徑: {_AITSTAR_PATH}")
    sys.exit(1)

# 測試 2: IsaacLab 適配器（如果可用）
print("\n" + "=" * 60)
print("[測試 2] IsaacLab 適配器...")

try:
    # 這部分需要 Isaac Lab 環境，如果失敗就跳過
    from path_planner.aitstar_adapter import AITStarPathPlanner, AITStarPlannerCfg
    from path_planner.environment_map import EnvironmentMap, EnvironmentMapCfg

    print("✅ IsaacLab 適配器導入成功")

    # 創建環境地圖
    map_cfg = EnvironmentMapCfg(
        map_size=[16.0, 16.0],
        grid_resolution=0.1,
        obstacle_inflation=0.3,
    )
    env_map = EnvironmentMap(map_cfg)
    print("✅ 環境地圖創建成功")

    # 創建規劃器配置
    planner_cfg = AITStarPlannerCfg(
        map_cfg=map_cfg,
        robot_radius=0.3,
        max_iterations=1000,
        batch_size=100,
        path_smoothing=True,
        waypoint_spacing=0.2,
    )
    print("✅ 規劃器配置創建成功")

    # 創建規劃器
    adapter = AITStarPathPlanner(planner_cfg)
    print("✅ AITStar 適配器創建成功")

    # 測試規劃
    start = torch.tensor([-3.0, -3.0])
    goal = torch.tensor([3.0, 3.0])

    path = adapter.plan_path(start, goal)

    if len(path) > 0:
        print(f"✅ 適配器規劃成功: {len(path)} 個路徑點")
    else:
        print(f"❌ 適配器規劃失敗")

except ImportError as e:
    print(f"⚠️  IsaacLab 適配器需要 Isaac Lab 環境")
    print(f"   錯誤: {e}")
    print(f"   跳過適配器測試")

# 測試 3: LocalGoalExtractor（如果可用）
print("\n" + "=" * 60)
print("[測試 3] LocalGoalExtractor...")

try:
    from path_planner.local_goal_extractor import LocalGoalExtractor, LocalGoalConfig

    local_cfg = LocalGoalConfig(
        lookahead_distance=2.0,
        min_update_distance=0.3,
        smoothing_factor=0.3,
    )
    extractor = LocalGoalExtractor(local_cfg, num_envs=1)

    # 模擬直線路徑
    robot_pos = torch.tensor([0.0, 0.0])
    path = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],  # 預期局部目標在這裡
        [3.0, 0.0],
    ])

    local_goal = extractor.extract_local_goal(
        robot_pos=robot_pos,
        aitstar_path=path,
        env_id=0,
    )

    print(f"✅ LocalGoalExtractor 測試成功")
    print(f"  機器人位置: {robot_pos.numpy()}")
    print(f"  局部目標: {local_goal.numpy()}")
    print(f"  預期目標: [2.0, 0.0] (lookahead=2.0m)")

    # 驗證
    expected = torch.tensor([2.0, 0.0])
    if torch.allclose(local_goal, expected, atol=0.1):
        print(f"  ✅ 局部目標正確 (預期 {expected.numpy()})")
    else:
        print(f"  ⚠️ 局部目標偏差 (實際 {local_goal.numpy()}, 預期 {expected.numpy()})")

except ImportError as e:
    print(f"⚠️  LocalGoalExtractor 需要 Isaac Lab 環境")
    print(f"   錯誤: {e}")
    print(f"   跳過 LocalGoalExtractor 測試")

print("\n" + "=" * 60)
print("    測試完成")
print("=" * 60)
