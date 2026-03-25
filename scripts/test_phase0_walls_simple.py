#!/usr/bin/env python3
"""测试 Phase 0 墙壁配置（不依赖 Isaac Lab）"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# 直接复制必要的类定义
@dataclass
class WallConfig:
    """單面牆的配置"""
    name: str
    center: Tuple[float, float]  # (x, y) 中心位置（環境局部坐標）
    size: Tuple[float, float]  # (length, thickness) 長度和厚度
    yaw: float = 0.0

def get_phase0_wall_configs() -> List[WallConfig]:
    """獲取 Phase 0 的牆壁配置（只有四面邊界牆）"""
    room_size = 8.0
    wall_thickness = 0.2
    walls = []

    # 四面邊界牆
    walls.append(WallConfig("wall_north", (0.0, room_size), (16.0, wall_thickness)))
    walls.append(WallConfig("wall_south", (0.0, -room_size), (16.0, wall_thickness)))
    walls.append(WallConfig("wall_east", (room_size, 0.0), (wall_thickness, 16.0)))
    walls.append(WallConfig("wall_west", (-room_size, 0.0), (wall_thickness, 16.0)))

    return walls

def add_rectangle_to_occupancy_grid(
    occupancy_grid: torch.Tensor,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    center: Tuple[float, float],
    size: Tuple[float, float],
    yaw: float = 0.0,
) -> int:
    """直接標記矩形障礙物到 occupancy grid"""
    origin_np = map_origin.cpu().numpy()
    center_local = center

    min_grid_x = int((center_local[0] - size[0]/2 - origin_np[0]) / grid_resolution)
    max_grid_x = int((center_local[0] + size[0]/2 - origin_np[0]) / grid_resolution)
    min_grid_y = int((center_local[1] - size[1]/2 - origin_np[1]) / grid_resolution)
    max_grid_y = int((center_local[1] + size[1]/2 - origin_np[1]) / grid_resolution)

    marked_count = 0
    for gx in range(min_grid_x, max_grid_x + 1):
        for gy in range(min_grid_y, max_grid_y + 1):
            if 0 <= gx < grid_width and 0 <= gy < grid_depth:
                occupancy_grid[gx, gy] = 1
                marked_count += 1

    return marked_count

# 测试
grid_resolution = 0.1
grid_width = int(16.0 / grid_resolution)  # 160
grid_depth = int(16.0 / grid_resolution)  # 160
map_origin = torch.tensor([-8.0, -8.0])
occupancy_grid = torch.zeros((grid_width, grid_depth), dtype=torch.uint8)

wall_configs = get_phase0_wall_configs()

print(f"\n{'='*60}")
print(f"Phase 0 墙壁配置测试")
print(f"{'='*60}")
print(f"牆壁数量: {len(wall_configs)}")
print(f"地图大小: {grid_width} x {grid_depth} = {grid_width * grid_depth} 网格")
print(f"地图原点: {map_origin.numpy()}")
print(f"{'='*60}\n")

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
    )
    print(f"  標記了 {marked} 個網格")
    total_marked += marked

print(f"\n{'='*60}")
print(f"總計標記了 {total_marked} 個網格為障礙物")
print(f"{'='*60}\n")

# 驗證
print("驗證邊界牆位置:")

marked_indices = torch.nonzero(occupancy_grid)
if len(marked_indices) > 0:
    x_coords = marked_indices[:, 0].float()
    y_coords = marked_indices[:, 1].float()
    print(f"  X 範圍: {x_coords.min().item():.0f} - {x_coords.max().item():.0f} (應該是 0-159)")
    print(f"  Y 範圍: {y_coords.min().item():.0f} - {y_coords.max().item():.0f} (應該是 0-159)")

    # 檢查北牆（Y = +8m → grid_y ≈ 158-159）
    north_wall_y = y_coords[x_coords > 50]  # 取中間 X 區域的 Y
    if len(north_wall_y) > 0:
        print(f"  北牆 Y 範圍: {north_wall_y.min().item():.0f} - {north_wall_y.max().item():.0f}")

    # 檢查南牆（Y = -8m → grid_y ≈ 0-1）
    south_wall_y = y_coords[x_coords > 50]
    if len(south_wall_y) > 0:
        print(f"  南牆 Y 範圍: {south_wall_y.min().item():.0f} - {south_wall_y.max().item():.0f}")

    # 檢查東牆（X = +8m → grid_x ≈ 158-159）
    east_wall_x = x_coords[y_coords > 50]
    if len(east_wall_x) > 0:
        print(f"  東牆 X 範圍: {east_wall_x.min().item():.0f} - {east_wall_x.max().item():.0f}")

    # 檢查西牆（X = -8m → grid_x ≈ 0-1）
    west_wall_x = x_coords[y_coords > 50]
    if len(west_wall_x) > 0:
        print(f"  西牆 X 範圍: {west_wall_x.min().item():.0f} - {west_wall_x.max().item():.0f}")

print("\n✓ Phase 0 墙壁配置测试完成!")
