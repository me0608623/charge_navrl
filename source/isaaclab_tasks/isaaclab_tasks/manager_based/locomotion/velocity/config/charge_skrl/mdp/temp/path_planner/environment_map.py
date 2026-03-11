"""
環境地圖表示 (Environment Map)

將 Isaac Lab 環境轉換為網格化地圖，用於 A* 路徑規劃。
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class EnvironmentMapCfg:
    """環境地圖配置類別

    Attributes:
        map_size: 地圖大小 [width, depth] (米)
        grid_resolution: 網格解析度 (米/格)，越小越精確但計算量越大
        obstacle_inflation: 障礙物膨脹距離 (米)，用於安全邊界
        boundary_margin: 邊界安全距離 (米)
        dynamic_obstacles: 是否考慮動態障礙物
    """
    map_size: Tuple[float, float] = (20.0, 20.0)
    grid_resolution: float = 0.1  # 10cm per grid
    obstacle_inflation: float = 0.3  # 30cm safety margin
    boundary_margin: float = 0.5  # 50cm boundary safety
    dynamic_obstacles: bool = False


class EnvironmentMap:
    """環境地圖類別

    將連續的 Isaac Lab 環境轉換為離散的網格地圖，
    並提供障礙物查詢、碰撞檢測等功能。
    """

    def __init__(self, cfg: EnvironmentMapCfg):
        """初始化環境地圖

        Args:
            cfg: 地圖配置
        """
        self.cfg = cfg

        # 計算網格尺寸
        self.grid_width = int(cfg.map_size[0] / cfg.grid_resolution)
        self.grid_depth = int(cfg.map_size[1] / cfg.grid_resolution)

        # 創建網格地圖 (0 = 可通行, 1 = 障礙物)
        self.occupancy_grid = torch.zeros(
            (self.grid_width, self.grid_depth),
            dtype=torch.uint8,
            device='cpu'
        )

        # 地圖原點（左下角在世界坐標中的位置）
        self.map_origin = torch.tensor([
            -cfg.map_size[0] / 2,
            -cfg.map_size[1] / 2,
        ], device='cpu')

        # 障礙物列表（用於動態更新）
        self.obstacles: List[dict] = []

        # 🔥 關鍵修復：初始化時立即添加邊界牆壁
        # 這樣 AIT* 規劃器才能看到牆壁，不會規劃穿牆路徑
        self.add_boundary_walls()

    def world_to_grid(self, world_pos: torch.Tensor) -> Tuple[int, int]:
        """將世界坐標轉換為網格坐標

        Args:
            world_pos: [2] or [num_pos, 2] 世界坐標位置 (米)

        Returns:
            (grid_x, grid_y) 網格坐標
        """
        if world_pos.dim() == 1:
            world_pos = world_pos.unsqueeze(0)

        # 相對於地圖原點的位置
        relative_pos = world_pos - self.map_origin

        # 轉換為網格坐標
        grid_x = (relative_pos[:, 0] / self.cfg.grid_resolution).long()
        grid_y = (relative_pos[:, 1] / self.cfg.grid_resolution).long()

        # 限制在有效範圍內
        grid_x = torch.clamp(grid_x, 0, self.grid_width - 1)
        grid_y = torch.clamp(grid_y, 0, self.grid_depth - 1)

        if world_pos.shape[0] == 1:
            return int(grid_x[0].item()), int(grid_y[0].item())
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> torch.Tensor:
        """將網格坐標轉換為世界坐標

        Args:
            grid_x: X 方向網格坐標
            grid_y: Y 方向網格坐標

        Returns:
            [2] 世界坐標位置 (米)
        """
        world_x = self.map_origin[0] + (grid_x + 0.5) * self.cfg.grid_resolution
        world_y = self.map_origin[1] + (grid_y + 0.5) * self.cfg.grid_resolution
        return torch.tensor([world_x, world_y])

    def is_occupied(self, grid_x: int, grid_y: int) -> bool:
        """檢查網格是否被佔據

        Args:
            grid_x: X 方向網格坐標
            grid_y: Y 方向網格坐標

        Returns:
            True if occupied, False otherwise
        """
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_depth):
            return True  # 超出邊界視為障礙物
        return self.occupancy_grid[grid_x, grid_y].item() > 0

    def is_occupied_world(self, world_pos: torch.Tensor) -> bool:
        """檢查世界坐標位置是否被佔據

        Args:
            world_pos: [2] 世界坐標位置 (米)

        Returns:
            True if occupied, False otherwise
        """
        grid_x, grid_y = self.world_to_grid(world_pos)
        return self.is_occupied(grid_x, grid_y)

    def add_obstacle(
        self,
        position: torch.Tensor,
        size: torch.Tensor,
        shape: str = "cuboid",
    ) -> None:
        """添加障礙物到地圖

        Args:
            position: [2] or [3] 障礙物中心位置 (米)
            size: [2] or [3] 障礙物尺寸 (米)
            shape: 障礙物形狀 ("cuboid", "cylinder")
        """
        pos_xy = position[:2] if position.shape[0] >= 2 else position
        size_xy = size[:2] if size.shape[0] >= 2 else size

        # 添加膨脹距離
        half_size = size_xy / 2 + self.cfg.obstacle_inflation

        # 計算障礙物覆蓋的網格範圍
        min_world = pos_xy - half_size
        max_world = pos_xy + half_size

        min_grid = self.world_to_grid(min_world)
        max_grid = self.world_to_grid(max_world)

        # 標記網格為障礙物
        for x in range(min_grid[0], max_grid[0] + 1):
            for y in range(min_grid[1], max_grid[1] + 1):
                if 0 <= x < self.grid_width and 0 <= y < self.grid_depth:
                    self.occupancy_grid[x, y] = 1

        # 記錄障礙物
        self.obstacles.append({
            'position': pos_xy.clone(),
            'size': size_xy.clone(),
            'shape': shape,
        })

    def add_boundary_walls(self) -> None:
        """添加邊界牆壁到地圖"""
        margin = self.cfg.boundary_margin
        map_w, map_d = self.cfg.map_size

        # 🔍 調試：打印牆壁信息
        if not hasattr(self, '_wall_debug_printed'):
            self._wall_debug_printed = True
            print(f"\n{'='*60}")
            print("EnvironmentMap - 添加邊界牆壁")
            print(f"{'='*60}")
            print(f"  地圖大小: {map_w}m x {map_d}m")
            print(f"  地圖原點: ({self.map_origin[0]:.2f}, {self.map_origin[1]:.2f})")
            print(f"  網格尺寸: {self.grid_width} x {self.grid_depth}")
            print(f"  邊界邊距: {margin}m")
            print(f"{'='*60}\n")

        # 四面牆 - 邊界牆不使用 obstacle_inflation，直接標記網格
        walls = [
            # (position, size)
            # 上牆：y 正方向，x 跨越整個寬度
            (torch.tensor([0, map_d / 2]), torch.tensor([map_w, margin])),
            # 下牆：y 負方向
            (torch.tensor([0, -map_d / 2]), torch.tensor([map_w, margin])),
            # 右牆：x 正方向，y 跨越整個深度
            (torch.tensor([map_w / 2, 0]), torch.tensor([margin, map_d])),
            # 左牆：x 負方向
            (torch.tensor([-map_w / 2, 0]), torch.tensor([margin, map_d])),
        ]

        for pos, size in walls:
            self._add_boundary_obstacle(pos, size)

    def _add_boundary_obstacle(
        self,
        position: torch.Tensor,
        size: torch.Tensor,
    ) -> None:
        """添加邊界牆障礙物（不使用 inflation）"""
        pos_xy = position[:2] if position.shape[0] >= 2 else position
        size_xy = size[:2] if size.shape[0] >= 2 else size

        # 邊界牆不添加膨脹距離
        half_size = size_xy / 2

        # 計算障礙物覆蓋的網格範圍
        min_world = pos_xy - half_size
        max_world = pos_xy + half_size

        min_grid = self.world_to_grid(min_world)
        max_grid = self.world_to_grid(max_world)

        # 標記網格為障礙物
        for x in range(min_grid[0], max_grid[0] + 1):
            for y in range(min_grid[1], max_grid[1] + 1):
                if 0 <= x < self.grid_width and 0 <= y < self.grid_depth:
                    self.occupancy_grid[x, y] = 1

    def update_from_env(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: str = "robot",
        obstacle_cfg_prefix: str = "obstacle_",
    ) -> None:
        """從環境更新障礙物地圖

        Args:
            env: Isaac Lab 環境實例
            robot_cfg: 機器人配置名稱
            obstacle_cfg_prefix: 障礙物配置名稱前綴
        """
        # 清空現有障礙物
        self.occupancy_grid.zero_()
        self.obstacles.clear()

        # 添加邊界牆壁
        self.add_boundary_walls()

        # 從場景中獲取障礙物
        scene = env.scene

        # 遍歷場景中的所有物體
        for name, obj in scene.items():
            if name.startswith(obstacle_cfg_prefix):
                # 獲取障礙物位置和尺寸
                if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                    pos = obj.data.root_pos_w[0, :2]  # 取第一個環境
                    # 假設障礙物有 collision 形狀信息
                    # 這裡使用默認尺寸，實際應從配置中獲取
                    size = torch.tensor([0.5, 0.5])
                    self.add_obstacle(pos, size)

    def get_neighbors(
        self,
        grid_x: int,
        grid_y: int,
        diagonal: bool = True,
    ) -> List[Tuple[int, int]]:
        """獲取相鄰的可通行網格

        Args:
            grid_x: 當前 X 網格坐標
            grid_y: 當前 Y 網格坐標
            diagonal: 是否包含對角線方向

        Returns:
            相鄰可通行網格坐標列表 [(x1, y1), (x2, y2), ...]
        """
        neighbors = []

        # 8 方向移動
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # 上下左右
        ]

        if diagonal:
            directions.extend([
                (1, 1), (1, -1), (-1, 1), (-1, -1),  # 對角線
            ])

        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy

            # 檢查是否在範圍內且可通行
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_depth:
                if not self.is_occupied(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def clear(self) -> None:
        """清空地圖（保留邊界牆壁）"""
        self.occupancy_grid.zero_()
        self.obstacles.clear()
        self.add_boundary_walls()

    def get_occupancy_numpy(self) -> np.ndarray:
        """獲取佔據網格的 numpy 數組（用於可視化）

        Returns:
            [grid_width, grid_depth] uint8 數組
        """
        return self.occupancy_grid.cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"EnvironmentMap("
            f"size={self.cfg.map_size}, "
            f"resolution={self.cfg.grid_resolution}m, "
            f"grid=({self.grid_width}x{self.grid_depth}), "
            f"obstacles={len(self.obstacles)})"
        )


__all__ = [
    "EnvironmentMap",
    "EnvironmentMapCfg",
]
