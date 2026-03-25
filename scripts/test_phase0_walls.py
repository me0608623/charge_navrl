#!/usr/bin/env python3
"""测试 Phase 0 墙壁配置"""

import sys
sys.path.insert(0, "/home/aa/IsaacLab/source/isaaclab_tasks")

from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_sb3.mdp.path_planner import (
    get_phase0_wall_configs,
    add_rectangle_to_occupancy_grid,
)

import torch
import numpy as np

# 创建 occupancy grid
grid_resolution = 0.1
grid_width = int(16.0 / grid_resolution)  # 160
grid_depth = int(16.0 / grid_resolution)  # 160
map_origin = torch.tensor([-8.0, -8.0])

occupancy_grid = torch.zeros((grid_width, grid_depth), dtype=torch.uint8)

# 获取 Phase 0 墙壁配置
wall_configs = get_phase0_wall_configs()

print(f"\n{'='*60}")
print(f"Phase 0 墙壁配置测试")
print(f"{'='*60}")
print(f"牆壁数量: {len(wall_configs)}")
print(f"地图大小: {grid_width} x {grid_depth} = {grid_width * grid_depth} 网格")
print(f"地图原点: {map_origin.numpy()}")
print(f"{'='*60}\n")

# 添加牆壁（不使用 env_origin，因為 center 已經是局部坐標）
total_marked = 0
for wall in wall_configs:
    print(f"添加牆壁: {wall.name}")
    print(f"  center = {wall.center}")
    print(f"  size = {wall.size}")
    marked = add_rectangle_to_occupancy_grid(
        occupancy_grid=occupancy_grid,
        map_origin=map_origin,
        grid_resolution=grid_resolution,
        grid_width=grid_width,
        grid_depth=grid_depth,
        center=wall.center,
        size=wall.size,
        yaw=wall.yaw,
        env_origin=None,  # Phase 0 使用局部坐標，不需要 env_origin
    )
    print(f"  標記了 {marked} 個網格")
    total_marked += marked

print(f"\n{'='*60}")
print(f"總計標記了 {total_marked} 個網格為障礙物")
print(f"{'='*60}\n")

# 驗證邊界牆是否正確
# 北牆（Y = +8m）應該在 grid_y = 160
# 南牆（Y = -8m）應該在 grid_y = 0
# 東牆（X = +8m）應該在 grid_x = 160
# 西牆（X = -8m）應該在 grid_x = 0

print("\n驗證邊界牆位置:")
print(f"  北牆 (Y=+8m): grid_y 範圍應該接近 158-160")
print(f"  南牆 (Y=-8m): grid_y 範圍應該接近 0-2")
print(f"  東牆 (X=+8m): grid_x 範圍應該接近 158-160")
print(f"  西牆 (X=-8m): grid_x 範圍應該接近 0-2")

# 檢查角落是否被標記（角落應該被兩面牆都標記）
corner_count = 0
for i in [0, 1]:
    for j in [0, 1]:
        if occupancy_grid[i * 159, j * 159] > 0:
            corner_count += 1
print(f"\n角落標記檢查: {corner_count}/4 個角落被標記")

# 統計障礙物分佈
marked_indices = torch.nonzero(occupancy_grid)
if len(marked_indices) > 0:
    x_coords = marked_indices[:, 0].float()
    y_coords = marked_indices[:, 1].float()
    print(f"\n障礙物分佈:")
    print(f"  X 範圍: {x_coords.min().item():.0f} - {x_coords.max().item():.0f}")
    print(f"  Y 範圍: {y_coords.min().item():.0f} - {y_coords.max().item():.0f}")

print("\n✓ Phase 0 墙壁配置测试完成!")
