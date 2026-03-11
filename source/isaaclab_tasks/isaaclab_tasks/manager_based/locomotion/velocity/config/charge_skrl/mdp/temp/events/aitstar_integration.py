"""
AIT* 整合事件 (AIT* Integration Events)

完整的 USD → AIT* Pipeline 驗證：

1. USD Stage 牆壁幾何
   ↓
2. 採樣牆壁表面點 (每 5cm 一點)
   ↓
3. 轉換為局部座標系 (local = world - env_origin)
   ↓
4. 添加到 AIT* occupancy grid
   ↓
5. AIT* 規劃路徑（輸出局部坐標路徑）
   ↓
6. 提取局部目標（Carrot-on-stick，局部坐標）
   ↓
7. 🔥 轉換為世界坐標後存儲：local_goal_world = local_goal + env_origin
   ↓
8. RL 觀測函數讀取世界坐標的局部目標

座標轉換說明：
- Isaac Lab 使用「世界座標系」，每個環境有各自的 env_origin 偏移
- AIT* 規劃器使用「環境局部座標系」（16x16m 地圖，中心在 (0,0)）
- 規劃前轉換：local_pos = world_pos - env_origin
- 🔥 關鍵：存儲到 env._local_goal_world 時必須轉回世界坐標
- 視測函數期望接收世界坐標，用於計算相對位置

使用方式：
    events: EventCfg = EventCfg(
        plan_aitstar_path = EventTerm(
            func=plan_aitstar_and_update_local_goal,
            mode="reset",
            params={
                "lookahead_distance": 2.0,
                "map_size": (16.0, 16.0),
                "robot_radius": 0.3,
                "visualize_path": True,
                "sample_walls_from_usd": True,  # 🔥 從 USD 採樣牆壁幾何
                "run_verification": True,      # 🆕 運行 8 個驗收測試
                "phase": "phase1",              # 🆕 指定 phase 用於門口位置檢查
            },
        ),
    )
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def plan_aitstar_and_update_local_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    lookahead_distance: float = 2.0,
    map_size: tuple = (16.0, 16.0),
    robot_radius: float = 0.3,
    visualize_path: bool = True,
    sample_walls_from_usd: bool = True,  # 🔥 從 USD 採樣牆壁幾何
    run_verification: bool = False,     # 🆕 運行驗收測試
    phase: str = "phase1",              # 🆕 指定 phase
    use_astar: bool = False,             # 🔥 使用 A* 代替 AIT*
    use_virtual_planner: bool = False,  # 🆕 使用虛擬規劃器（訓練模式）
):
    """在環境重置時使用 AIT* 規劃路徑並更新局部目標

    完整 Pipeline（USD → AIT*）：
    1. 獲取機器人位置和目標位置
    2. 🔥 從 USD stage 採樣牆壁幾何
    3. 將採樣點轉換為局部座標
    4. 添加到 AIT* occupancy grid
    5. 🆕 運行 8 個驗收測試（如果啟用）
    6. 調用 AIT* 規劃路徑
    7. 使用 LocalGoalExtractor 提取局部目標
    8. 可視化驗證

    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
        lookahead_distance: 前瞻距離（米）
        map_size: 地圖大小 (width, height) 米
        robot_radius: 機器人半徑（米）
        visualize_path: 是否可視化路徑
        sample_walls_from_usd: 是否從 USD 採樣牆壁幾何
        run_verification: 是否運行 8 個驗收測試
        phase: 指定 phase（用於門口位置檢查）
    """
    from isaaclab.assets import Articulation

    # 獲取機器人
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    # =========================================================================
    # 第 1 步：初始化規劃器和可視化器
    # =========================================================================
    if not hasattr(env, "_path_planner"):
        from ..path_planner import (
            AITStarPathPlanner,
            AITStarPlannerCfg,
            EnvironmentMapCfg,
            LocalGoalExtractor,
            LocalGoalConfig,
            MultiEnvPathVisualizer,
            WallGeometryVisualizer,
        )
        from ..path_planner.virtual_planner import VirtualPlanner

        if use_virtual_planner:
            # 🆕 訓練模式：使用虛擬規劃器（隨機目標點）
            # Phase 2+ 需要障礙物檢測，創建共享的 EnvironmentMap
            env_map = None
            if phase in ["phase2", "phase3"]:
                map_cfg = EnvironmentMapCfg(
                    map_size=list(map_size),
                    grid_resolution=0.1,
                    obstacle_inflation=robot_radius,
                )
                from ..path_planner.environment_map import EnvironmentMap
                env_map = EnvironmentMap(map_cfg)

            env._path_planner = VirtualPlanner(
                phase=phase,
                min_distance=lookahead_distance,
                max_distance=lookahead_distance * 2.0,
                map_size=map_size,
                env_map=env_map,
            )
            env._use_virtual_planner = True

            # 設置 _aitstar_planner 引用（用於後續添加牆壁/障礙物）
            if env_map is not None:
                env._environment_map = env_map
                env._aitstar_planner = type('obj', (object,), {'map': env_map})()
            else:
                env._aitstar_planner = env._path_planner
        else:
            # 推理模式：使用真實的 AIT* 規劃器
            print(f"\n[路徑規劃] 使用 AIT* 算法（全局路徑規劃 - 推理模式）")

            # 創建 AIT* 規劃器配置
            map_cfg = EnvironmentMapCfg(
                map_size=list(map_size),
                grid_resolution=0.1,
                obstacle_inflation=robot_radius,
            )

            planner_cfg = AITStarPlannerCfg(
                map_cfg=map_cfg,
                robot_radius=robot_radius,
                max_iterations=1000,
                batch_size=200,
                rewire_factor=2.0,
                path_smoothing=True,
                waypoint_spacing=0.2,
                convergence_threshold=1e-3,
            )

            env._path_planner = AITStarPathPlanner(planner_cfg)
            env._use_virtual_planner = False
            env._aitstar_planner = env._path_planner  # AIT* 有自己的 map

        # 創建局部目標提取器
        local_goal_cfg = LocalGoalConfig(
            lookahead_distance=lookahead_distance,
            min_update_distance=0.3,
            smoothing_factor=0.3,
            corridor_half_width=0.5,
            max_goal_change=0.5,
        )
        env._local_goal_extractor = LocalGoalExtractor(
            local_goal_cfg, num_envs=num_envs
        )

        # 初始化局部目標存儲
        env._local_goal_world = torch.zeros(num_envs, 2, device=device)
        env._aitstar_path_version = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 初始化 AIT* 路徑可視化器（僅 AIT* 模式）
        # 虛擬規劃器模式下不需要可視化（訓練效率優先）
        if visualize_path and not use_virtual_planner:
            env._aitstar_visualizer = MultiEnvPathVisualizer(
                num_envs=num_envs,
                prim_path="/World/AITStarPaths",
                max_envs_to_visualize=min(4, num_envs),
            )

        # 🔥 初始化牆壁幾何可視化器（驗證 USD 採樣）
        # 注意：虛擬規劃器不需要牆壁幾何可視化
        if sample_walls_from_usd and not use_virtual_planner:
            env._wall_visualizer = WallGeometryVisualizer(
                num_envs=num_envs,
                prim_path="/Visuals/WallPoints",
            )

    # =========================================================================
    # 第 2 步：獲取機器人和目標位置（必須在條件塊外定義）
    # =========================================================================
    robot_pos_world = robot.data.root_pos_w[env_ids, :2]
    goal_pos_world = env.command_manager.get_command("goal_command")[env_ids, :2]
    env_origins = env.scene.env_origins[env_ids, :2]

    # 🔥 重要：清理所有輸入數據的 NaN/Inf
    robot_pos_world = torch.nan_to_num(robot_pos_world, nan=0.0, posinf=100.0, neginf=-100.0)
    goal_pos_world = torch.nan_to_num(goal_pos_world, nan=0.0, posinf=100.0, neginf=-100.0)
    env_origins = torch.nan_to_num(env_origins, nan=0.0, posinf=100.0, neginf=-100.0)

    robot_pos_local = robot_pos_world - env_origins
    goal_pos_local = goal_pos_world - env_origins

    # 確保目標在範圍內
    goal_pos_local_clamped = torch.clamp(goal_pos_local, -7.0, 7.0)

    # =========================================================================
    # 第 2.5 步：確保 goal_command 已經被重置（只在第一幀重置時）
    # =========================================================================
    if env.episode_length_buf[env_ids[0]] == 0:
        goal_command_term = env.command_manager._terms.get("goal_command")
        if goal_command_term is not None:
            goal_pos_w_check = env.command_manager.get_command("goal_command")[env_ids, :2]
            goal_needs_reset = torch.all(goal_pos_w_check.abs() < 0.1)

            if goal_needs_reset:
                goal_command_term._resample_command(env_ids)

            # 重新獲取可能更新的目標位置
            goal_pos_world = env.command_manager.get_command("goal_command")[env_ids, :2]
            goal_pos_local = goal_pos_world - env_origins
            goal_pos_local_clamped = torch.clamp(goal_pos_local, -7.0, 7.0)

    # =========================================================================
    # 第 3 步：🔥 從配置直接添加牆壁/障礙物到地圖（數學幾何方法）
    # =========================================================================
    # 判斷是否需要添加牆壁/障礙物：
    # 1. sample_walls_from_usd=True
    # 2. 第一幀重置時
    # 3. (AIT* 模式) OR (VirtualPlanner + Phase 2+)
    needs_walls = sample_walls_from_usd and env.episode_length_buf[env_ids[0]] == 0
    needs_walls = needs_walls and (not use_virtual_planner or phase in ["phase2", "phase3"])

    if needs_walls:
        from ..path_planner import add_walls_from_config

        planner_type = "虛擬規劃器 (Virtual Planner)" if use_virtual_planner else "AIT* 規劃器"
        print(f"\n{'='*70}")
        print(f"牆壁/障礙物 → {planner_type} 地圖（數學幾何方法）")
        print(f"{'='*70}")

        total_marked_grids = 0

        for i, env_id in enumerate(env_ids[:4]):  # 只處理前 4 個環境
            # 🔥 直接從配置添加牆壁（數學幾何，精準且快速）
            num_marked = add_walls_from_config(
                env=env,
                env_id=env_id.item(),
                occupancy_grid=env._aitstar_planner.map.occupancy_grid,
                map_origin=env._aitstar_planner.map.map_origin,
                grid_resolution=0.1,
                grid_width=env._aitstar_planner.map.grid_width,
                grid_depth=env._aitstar_planner.map.grid_depth,
                phase=phase,
                debug=True,  # 🔥 啟用調試輸出
            )

            total_marked_grids += num_marked

        print(f"{'='*70}")
        print(f"總計: {total_marked_grids} 個網格被標記為牆壁/障礙物")
        print(f"{planner_type} 現在「看到」牆壁拓撲結構！")

        # 🔥 可視化牆壁（僅 AIT* 模式）
        if not use_virtual_planner and hasattr(env, "_wall_visualizer"):
            for i, env_id in enumerate(env_ids[:4]):
                env_id_int = env_id.item()
                env._wall_visualizer.visualize_from_grid(
                    env_idx=env_id_int,
                    occupancy_grid=env._aitstar_planner.map.occupancy_grid,
                    map_origin=env._aitstar_planner.map.map_origin,
                    grid_resolution=0.1,
                )
            print(f"🔥 牆壁可視化已添加（紫色方塊 = 牆壁網格）")
        print(f"{'='*70}\n")

    # =========================================================================
    # 第 3.5 步：🆕 運行驗收測試（如果啟用）
    # VirtualPlanner + Phase 2+ 也需要驗收測試
    # =========================================================================
    if run_verification and env.episode_length_buf[env_ids[0]] == 0:
        # 🔥 VirtualPlanner 在 Phase 2+ 才有地圖
        if use_virtual_planner and phase not in ["phase2", "phase3"]:
            print(f"[跳過驗收測試] VirtualPlanner Phase {phase} 沒有障礙物地圖")

        from ..path_planner import get_phase0_wall_configs, get_phase1_wall_configs, get_phase2_wall_configs, get_phase3_wall_configs

        print(f"\n{'='*70}")
        print(f"🔍 牆壁/障礙物幾何驗收測試（數學幾何方法）")
        print(f"{'='*70}")

        # 獲取對應 phase 的牆壁配置
        env_id_0 = env_ids[0].item()
        if phase == "phase0":
            wall_configs = get_phase0_wall_configs()
        elif phase == "phase1":
            wall_configs = get_phase1_wall_configs()
        elif phase == "phase2":
            wall_configs = get_phase2_wall_configs()
        elif phase == "phase3":
            wall_configs = get_phase3_wall_configs()
        else:
            wall_configs = []

        print(f"Phase {phase} 牆壁配置:")
        print(f"  牆壁數量: {len(wall_configs)}")
        for wall in wall_configs:
            print(f"    {wall.name}: center=({wall.center[0]:.2f}, {wall.center[1]:.2f}), "
                  f"size=({wall.size[0]:.2f}m x {wall.size[1]:.2f}m)")

        # 計算 grid 標記數量
        grid_marked_count = torch.sum(env._aitstar_planner.map.occupancy_grid).item()

        print(f"{'='*70}")
        print(f"驗收測試結果:")
        print(f"  總共標記了 {grid_marked_count} 個網格為障礙物")
        print(f"  網格解析度: 0.1m")
        print(f"  地圖尺寸: {map_size[0]}m x {map_size[1]}m")
        print(f"{'='*70}\n")

    # =========================================================================
    # 第 4 步：為每個環境規劃路徑
    # =========================================================================
    for i, env_id in enumerate(env_ids):
        start = robot_pos_local[i]
        goal = goal_pos_local_clamped[i]

        # 🔥 訓練時關閉除錯輸出：只在非虛擬規劃器模式下打印詳細信息
        # 🔥 詳細調試：env_0 的規劃資訊
        if env_id == 0 and not hasattr(env, "_path_debug_printed") and not use_virtual_planner:
            env._path_debug_printed = True
            planner_type = "虛擬規劃器 (Virtual Planner)" if use_virtual_planner else "AIT* 規劃器"
            print(f"\n{'='*60}")
            print(f"{planner_type} 路徑規劃調試 (env_0)")
            print(f"{'='*60}")
            print(f"機器人世界坐標: {robot_pos_world[0].cpu().numpy()}")
            print(f"目標世界坐標: {goal_pos_world[0].cpu().numpy()}")
            print(f"環境原點: {env_origins[0].cpu().numpy()}")
            print(f"機器人局部坐標: {start.cpu().numpy()}")
            print(f"目標局部坐標: {goal.cpu().numpy()}")
            print(f"--- 輸入給規劃器的坐標 ---")
            print(f"start (傳入): {start.cpu().numpy()}")
            print(f"goal (傳入): {goal.cpu().numpy()}")
            print(f"{'='*60}\n")

        # 調試：檢查規劃器類型（只打印一次）
        if not hasattr(env, "_planner_type_checked") and not use_virtual_planner:
            env._planner_type_checked = True
            planner_type = "虛擬規劃器 (Virtual Planner)" if use_virtual_planner else "AIT* 規劃器"
            print(f"\n[路徑規劃] 使用 {planner_type}")
            print(f"[路徑規劃] phase: {phase}")

        # 調試：打印所有環境的路徑規劃信息（只在非虛擬規劃器模式下打印詳細信息）
        if not use_virtual_planner:
            print(f"\n{'='*70}")
            print(f"[AIT* Debug] env_{env_id} - 座標對比")
            print(f"{'='*70}")
            print(f"🎯 紅色箭頭（目標位置 GoalCommand）:")
            print(f"   世界坐標: {goal_pos_world[i].cpu().numpy()}")
            print(f"   局部坐標: {goal.cpu().numpy()}")
            print(f"")
            print(f"🤖 機器人起點:")
            print(f"   世界坐標: {robot_pos_world[i].cpu().numpy()}")
            print(f"   局部坐標: {start.cpu().numpy()}")
            print(f"")
            print(f"🔍 AIT* 規劃器:")

        # 執行路徑規劃
        path = env._path_planner.plan_path(start, goal, env=None)

        # 🔥 訓練時關閉除錯輸出（使用虛擬規劃器時不打印）
        if not use_virtual_planner:
            print(f"   AIT* 輸入: start={start.cpu().numpy()}, goal={goal.cpu().numpy()}")
            print(f"   AIT* 輸出: shape={path.shape}")
            print(f"")
            print(f"📍 AIT* 路徑終點（局部坐標）:")
            print(f"   path[-1]: {path[-1].cpu().numpy()}")
            print(f"")
            print(f"📍 期望終點（局部坐標）:")
            print(f"   goal:    {goal.cpu().numpy()}")

            # 檢查路徑是否從 start 開始
            start_diff = torch.norm(path[0] - start).item()
            goal_diff = torch.norm(path[-1] - goal).item()
            print(f"")
            print(f"✓ 座標驗證:")
            print(f"   start 誤差: {start_diff:.6f}m  {'✓ 正確' if start_diff < 0.01 else '✗ 錯誤'}")
            print(f"   goal 誤差:  {goal_diff:.6f}m   {'✓ 正確' if goal_diff < 0.01 else '✗ 錯誤'}")
            print(f"")
            print(f"📊 路徑航點數量: {len(path)}")

            if len(path) > 2:
                print(f"   ✓ AIT* 成功！有 {len(path)-2} 個中間航點")
            elif len(path) == 2:
                print(f"   ⚠️  AIT* 失敗！只有起點和終點（直線）")
            print(f"{'='*70}\n")

            print(f"[AIT* Debug]   start 誤差: {start_diff:.4f}m")
            print(f"[AIT* Debug]   goal 誤差: {goal_diff:.4f}m")

            if start_diff > 0.1:
                print(f"[AIT* WARNING] env_{env_id} 路徑起點與機器人位置不匹配！")
            if goal_diff > 0.1:
                print(f"[AIT* WARNING] env_{env_id} 路徑終點與目標位置不匹配！")

        if len(path) > 0:
            # 🔥 訓練時：使用虛擬規劃器直接追蹤紅色箭頭（goal_command）
            # 訓練目標：讓 agent 學會追蹤紅色箭頭（這是推理時全局路徑中的一個waypoint）
            if use_virtual_planner:
                # 虛擬規劃器模式下，直接使用目標（紅色箭頭）
                local_goal = goal
            else:
                # AIT* 模式下，從路徑中提取局部目標（Carrot-on-stick）
                local_goal = env._local_goal_extractor.extract_local_goal(
                    robot_pos=start,
                    aitstar_path=path,
                    path_version=0,
                    env_id=env_id,
                )

            # 可視化 AIT* 路徑
            if visualize_path and hasattr(env, "_aitstar_visualizer"):
                origin = env_origins[i].to(path.device)
                path_world = path + origin
                start_world = robot_pos_world[i]

                env._aitstar_visualizer.visualize(
                    env_idx=env_id,
                    path_points=path_world,
                    start_pos=start_world,
                )
        else:
            # 規劃失敗，直接用目標
            local_goal = goal

            # 清除可視化
            if visualize_path and env_id == 0 and hasattr(env, "_aitstar_visualizer"):
                env._aitstar_visualizer.clear()

        # 🔥 修復坐標系 BUG：將局部坐標轉換為世界坐標後存儲
        # local_goal 是相對 env_origin 的局部坐標，需要轉換為世界坐標
        env_origin = env_origins[i].to(local_goal.device)
        local_goal_world = local_goal + env_origin
        env._local_goal_world[env_id] = local_goal_world

        # 🔍 調試輸出（僅 env_0 第一次，訓練時關閉）
        if env_id == 0 and not hasattr(env, "_coord_fix_debug_printed") and not use_virtual_planner:
            env._coord_fix_debug_printed = True
            print(f"\n{'='*60}")
            print(f"🔧 坐標系修復驗證 (env_0)")
            print(f"{'='*60}")
            print(f"env_origin:     {env_origin.cpu().numpy()}")
            print(f"local_goal (局部坐標):  {local_goal.cpu().numpy()}")
            print(f"local_goal_world (世界坐標): {local_goal_world.cpu().numpy()}")
            print(f"robot_pos_world:         {robot_pos_world[0].cpu().numpy()}")
            print(f"")
            print(f"驗證計算：")
            print(f"  local_goal_world = local_goal + env_origin")
            print(f"  {local_goal_world.cpu().numpy()} = {local_goal.cpu().numpy()} + {env_origin.cpu().numpy()}")
            print(f"")
            print(f"相對位置（觀測函數將計算）：")
            relative_pos = local_goal_world - robot_pos_world[0]
            print(f"  relative = local_goal_world - robot_pos_world")
            print(f"  {relative_pos.cpu().numpy()} = {local_goal_world.cpu().numpy()} - {robot_pos_world[0].cpu().numpy()}")
            print(f"{'='*60}\n")


def sync_local_goal_from_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """每步同步 _local_goal_world 為 goal_command 的世界座標位置。

    修復 _reset_idx() 執行順序 Bug：
        Isaac Lab 的 _reset_idx() 先執行 event_manager.apply("reset")
        再執行 command_manager.reset()。這導致 plan_aitstar 事件讀到的是
        舊的 goal_command，而 command_manager.reset() 隨後重新採樣到新位置。
        結果：_local_goal_world ≠ goal_command（綠色箭頭），二者永遠不同步。

    此函數作為 interval 事件（每步執行），將 _local_goal_world 強制同步為
    goal_command 當前值，確保 goal_reached / rewards / observations 看到的
    目標位置與綠色箭頭一致。

    僅在 Phase 0 虛擬規劃器模式下使用。真實 AIT* 模式由 AIT* 路徑規劃器
    獨立管理 _local_goal_world，不需要此同步。
    """
    if not hasattr(env, "_local_goal_world") or env._local_goal_world is None:
        return

    # goal_command 返回世界座標 [num_envs, 2+]
    goal_pos_w = env.command_manager.get_command("goal_command")[:, :2]

    # 強制同步全部環境（不只是 env_ids，因為 command_manager.reset
    # 可能已經改變了任意環境的目標）
    env._local_goal_world[:] = goal_pos_w


def update_local_goal_during_episode(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    update_interval: int = 10,
):
    """在 episode 期間定期更新局部目標（可選）"""
    if not hasattr(env, "_local_goal_extractor"):
        return

    from isaaclab.assets import Articulation
    robot = env.scene["robot"]

    current_step = env.episode_length_buf[env_ids[0]]
    if current_step % update_interval != 0:
        return

    robot_pos_world = robot.data.root_pos_w[env_ids, :2]
    goal_pos_world = env.command_manager.get_command("goal_command")[env_ids, :2]

    env_origins = env.scene.env_origins[env_ids, :2]
    robot_pos_local = robot_pos_world - env_origins
    goal_pos_local = goal_pos_world - env_origins

    for i, env_id in enumerate(env_ids):
        start = robot_pos_local[i]
        goal = goal_pos_local[i]

        path = env._path_planner.plan_path(start, goal, env=None)

        # 🔥 虚拟规划器模式：直接使用目标（红色箭头）
        if hasattr(env, "_use_virtual_planner") and env._use_virtual_planner:
            local_goal = goal
        elif len(path) > 0:
            local_goal = env._local_goal_extractor.extract_local_goal(
                robot_pos=start,
                aitstar_path=path,
                path_version=env._aitstar_path_version[env_id].item(),
                env_id=env_id,
            )
        else:
            local_goal = goal

        # 🔥 修復坐標系 BUG：將局部坐標轉換為世界坐標後存儲
        env_origin = env_origins[i].to(local_goal.device)
        local_goal_world = local_goal + env_origin
        env._local_goal_world[env_id] = local_goal_world
        env._aitstar_path_version[env_id] += 1


__all__ = [
    "plan_aitstar_and_update_local_goal",
    "update_local_goal_during_episode",
]
