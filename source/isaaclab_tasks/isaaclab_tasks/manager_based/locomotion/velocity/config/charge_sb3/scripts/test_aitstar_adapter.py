#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIT* 路徑規劃器適配器測試腳本

測試 Isaac Lab 適配的 AIT* 路徑規劃器：
1. 創建帶障礙物的測試環境
2. 使用 AIT* 規劃路徑
3. 可視化結果
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 添加 globle_planner 路徑
_AITSTAR_PATH = "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3/globle_planner/src/aitstar_path_planner"
sys.path.insert(0, _AITSTAR_PATH)

from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.path_planner import (
    AITStarPathPlanner,
    AITStarPlannerCfg,
    EnvironmentMapCfg,
    create_aitstar_planner,
)


def create_test_environment(map_size: tuple = (20.0, 20.0)):
    """創建測試環境

    Returns:
        (obstacles, start, goal)
        obstacles: [(x, y, radius), ...]
        start: (x, y)
        goal: (x, y)
    """
    # 定義障礙物（圓形）
    obstacles = [
        (5.0, 5.0, 1.0),
        (8.0, 7.0, 1.5),
        (3.0, 8.0, 0.8),
        (7.0, 3.0, 1.2),
        (-5.0, 0.0, 1.0),
        (0.0, 5.0, 0.8),
    ]

    # 起點和終點
    start = (-8.0, -8.0)
    goal = (8.0, 8.0)

    return obstacles, start, goal


def setup_obstacles_on_map(planner: AITStarPathPlanner, obstacles: list):
    """在規劃器的地圖上設置障礙物

    Args:
        planner: AIT* 規劃器
        obstacles: [(x, y, radius), ...]
    """
    for x, y, radius in obstacles:
        pos = torch.tensor([x, y])
        # 將圓形轉換為方形（保守估計）
        size = torch.tensor([radius * 2, radius * 2])
        planner.map.add_obstacle(pos, size, shape="cuboid")


def visualize_planning_result(
    obstacles: list,
    start: tuple,
    goal: tuple,
    path: torch.Tensor,
    search_tree: list,
    stats: dict,
    map_size: tuple = (20.0, 20.0),
    save_path: str = None,
):
    """可視化規劃結果

    Args:
        obstacles: 障礙物列表
        start: 起點
        goal: 終點
        path: 規劃的路徑
        search_tree: 搜索樹邊
        stats: 統計資訊
        map_size: 地圖大小
        save_path: 保存圖像路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 路徑結果（左圖）
    ax = axes[0]
    ax.set_title("AIT* Path Planning Result", fontsize=14)

    # 繪製障礙物
    for x, y, radius in obstacles:
        circle = Circle((x, y), radius, color='red', alpha=0.3)
        ax.add_patch(circle)
        ax.add_patch(Circle((x, y), radius, fill=False, edgecolor='red', linestyle='--'))

    # 繪製邊界
    w, h = map_size
    ax.set_xlim(-w/2, w/2)
    ax.set_ylim(-h/2, h/2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 繪製搜索樹（淺色）
    for p1, p2 in search_tree:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.2, linewidth=0.5)

    # 繪製路徑
    if len(path) > 0:
        path_np = path.numpy()
        ax.plot(path_np[:, 0], path_np[:, 1], 'b-', linewidth=3, label='AIT* Path')
        ax.scatter(path_np[:, 0], path_np[:, 1], c='blue', s=20, alpha=0.6, zorder=3)

    # 繪製起點和終點
    ax.scatter(start[0], start[1], c='green', s=200, marker='o', label='Start', zorder=5, edgecolors='black')
    ax.scatter(goal[0], goal[1], c='red', s=200, marker='*', label='Goal', zorder=5, edgecolors='black')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')

    # 2. 統計資訊（右圖）
    ax = axes[1]
    ax.axis('off')
    ax.set_title("Planning Statistics", fontsize=14)

    # 計算路徑長度
    if len(path) > 1:
        path_length = torch.norm(path[1:] - path[:-1], dim=1).sum().item()
    else:
        path_length = 0.0

    # 直線距離
    start_arr = np.array(start)
    goal_arr = np.array(goal)
    straight_dist = np.linalg.norm(goal_arr - start_arr)

    # 路徑效率
    efficiency = (straight_dist / path_length * 100) if path_length > 0 else 0

    stats_text = f"""
    ╔══════════════════════════════════════════════════╗
    ║          AIT* Path Planning Statistics           ║
    ╠══════════════════════════════════════════════════╣
    ║ Configuration                                    ║
    ║  Map Size:            {map_size[0]:>6.1f} x {map_size[1]:<6.1f} m          ║
    ║  Obstacles:           {len(obstacles):>6}                              ║
    ║  Max Iterations:      {stats.get('iterations', 0):>6}                              ║
    ╠──────────────────────────────────────────────────╣
    ║ Planning Results                                 ║
    ║  Solution Found:      {'YES' if len(path) > 0 else 'NO':>6}                              ║
    ║  Solution Cost:       {stats.get('solution_cost', float('inf')):>6.2f}                          ║
    ║  Path Length:         {path_length:>6.2f} m                         ║
    ║  Waypoints:           {len(path):>6}                              ║
    ║  Straight Distance:    {straight_dist:>6.2f} m                         ║
    ║  Path Efficiency:      {efficiency:>5.1f}%                          ║
    ╠──────────────────────────────────────────────────╣
    ║ Search Statistics                                ║
    ║  Total Vertices:      {stats.get('total_vertices', 0):>6}                              ║
    ║  Vertices Sampled:    {stats.get('vertices_sampled', 0):>6}                              ║
    ║  Edges Checked:       {stats.get('edges_checked', 0):>6}                              ║
    ║  Collision Checks:    {stats.get('collision_checks', 0):>6}                              ║
    ║  Solutions Found:     {stats.get('solutions_found', 0):>6}                              ║
    ╚══════════════════════════════════════════════════╝
    """

    ax.text(0.1, 0.5, stats_text, family='monospace', fontsize=11, va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 圖表已保存至: {save_path}")

    plt.show()


def test_aitstar_planner():
    """測試 AIT* 規劃器"""
    print("\n" + "=" * 60)
    print("    AIT* 路徑規劃器測試")
    print("=" * 60)

    # 創建測試環境
    obstacles, start, goal = create_test_environment()

    print(f"\n測試環境:")
    print(f"  地圖大小: 20m x 20m")
    print(f"  障礙物數量: {len(obstacles)}")
    print(f"  起點: {start}")
    print(f"  終點: {goal}")

    # 創建規劃器
    print(f"\n創建 AIT* 規劃器...")

    planner = create_aitstar_planner(
        map_size=(20.0, 20.0),
        grid_resolution=0.1,
        obstacle_inflation=0.3,
        robot_radius=0.3,
        max_iterations=1000,
        batch_size=100,
        rewire_factor=1.1,
        goal_bias=0.05,
        path_smoothing=True,
        waypoint_spacing=0.2,
        seed=42,
    )

    print(f"  {planner}")

    # 設置障礙物
    print(f"\n設置障礙物...")
    setup_obstacles_on_map(planner, obstacles)

    # 規劃路徑
    print(f"\n執行路徑規劃...")
    import time
    start_time = time.time()

    start_tensor = torch.tensor([start[0], start[1]])
    goal_tensor = torch.tensor([goal[0], goal[1]])

    path = planner.plan_path(start_tensor, goal_tensor)

    planning_time = time.time() - start_time

    print(f"  規劃時間: {planning_time:.3f} 秒")

    # 獲取統計資訊
    stats = planner.get_statistics()
    print(f"  迭代次數: {stats['iterations']}")
    print(f"  頂點數量: {stats['total_vertices']}")
    print(f"  解代價: {stats['solution_cost']:.2f}")

    if len(path) > 0:
        print(f"  路徑點數: {len(path)}")
        print(f"  路徑:\n{path}")
    else:
        print(f"  ❌ 未找到路徑")

    # 獲取搜索樹
    search_tree = planner.get_search_tree()
    print(f"  搜索樹邊數: {len(search_tree)}")

    # 可視化
    print(f"\n生成可視化...")
    visualize_planning_result(
        obstacles=obstacles,
        start=start,
        goal=goal,
        path=path,
        search_tree=search_tree,
        stats=stats,
    )

    return path, stats


def main():
    """主測試函數"""
    try:
        path, stats = test_aitstar_planner()
        print("\n" + "=" * 60)
        print("    ✅ 測試完成")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
