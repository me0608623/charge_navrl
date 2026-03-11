"""Frontier 探索模組 (Frontier Exploration Module)

用於完全未知環境的自主探索。

核心思想：
- Frontier = 已知地圖與未知區域的交界線
- 策略：將 Frontier 設為 AIT* 的目標，驅使 RL 前往探索
- 循環：探索 → 更新地圖 → 找新 Frontier → 繼續探索
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .hierarchical_navigation_manager import HierarchicalNavigationManager


@dataclass
class Frontier:
    """Frontier 數據結構
    
    代表已知地圖與未知區域的交界線。
    """
    center: torch.Tensor  # [2] Frontier 中心點（世界坐標）
    points: torch.Tensor  # [N, 2] Frontier 點集
    size: float  # Frontier 大小（米）
    priority: float  # 優先級（越大越重要）


class FrontierExplorer:
    """Frontier 探索器
    
    在未知環境中自動尋找探索目標。
    """
    
    def __init__(
        self,
        map_size: Tuple[float, float] = (20, 20),
        grid_resolution: float = 0.1,
        frontier_min_size: float = 0.5,
        sensor_range: float = 5.0,
    ):
        """初始化 Frontier 探索器
        
        Args:
            map_size: 地圖大小 [width, height]（米）
            grid_resolution: 網格解析度（米/格）
            frontier_min_size: 最小 Frontier 大小（米）
            sensor_range: 感測器範圍（米）
        """
        self.map_size = map_size
        self.grid_resolution = grid_resolution
        self.frontier_min_size = frontier_min_size
        self.sensor_range = sensor_range
        
        # 地圖網格
        self.grid_width = int(map_size[0] / grid_resolution)
        self.grid_height = int(map_size[1] / grid_resolution)
        
        # 探索狀態
        # 0 = 未知, 1 = 已探索空閒, 2 = 已探索障礙
        self.exploration_map = torch.zeros(
            (1, self.grid_height, self.grid_width),
            dtype=torch.long,
            device=torch.device("cpu")
        )
        
        # 已探索區域邊界
        self.explored_boundary = torch.zeros(
            (1, self.grid_height, self.grid_width),
            dtype=torch.bool,
            device=torch.device("cpu")
        )
        
    def update_exploration_map(
        self,
        robot_pos: torch.Tensor,
        lidar_data: torch.Tensor,
    ) -> None:
        """更新探索地圖
        
        根據機器人位置和 LiDAR 數據更新探索狀態。
        
        Args:
            robot_pos: [2] 機器人位置（世界坐標）
            lidar_data: [N] LiDAR 距離數據（歸一化 0-1）
        """
        # 將機器人位置轉換為網格坐標
        grid_x = int((robot_pos[0].item() + self.map_size[0] / 2) / self.grid_resolution)
        grid_y = int((robot_pos[1].item() + self.map_size[1] / 2) / self.grid_resolution)
        
        # 標記機器人周圍為已探索
        explore_radius = int(self.sensor_range / self.grid_resolution)
        
        for dy in range(-explore_radius, explore_radius + 1):
            for dx in range(-explore_radius, explore_radius + 1):
                gy, gx = grid_y + dy, grid_x + dx
                
                if 0 <= gy < self.grid_height and 0 <= gx < self.grid_width:
                    # 計算距離
                    dist = np.sqrt(dx**2 + dy**2) * self.grid_resolution
                    
                    if dist <= self.sensor_range:
                        # 在感測範圍內，標記為已探索
                        self.exploration_map[0, gy, gx] = 1
                        
                        # 如果 LiDAR 檢測到障礙物，標記為障礙
                        # 這裡簡化處理，實際應該根據 LiDAR 數據判斷
                        # lidar_data 對應的角度
                        angle_idx = int(np.arctan2(dy, dx) / (2 * np.pi) * len(lidar_data))
                        angle_idx = (angle_idx + len(lidar_data)) % len(lidar_data)
                        
                        if lidar_data[angle_idx] < 0.3:  # 很近 = 障礙物
                            self.exploration_map[0, gy, gx] = 2
        
        # 更新邊界（已探索與未知的交界）
        self._update_boundary()
    
    def _update_boundary(self):
        """更新探索邊界
        
        找出已探索區域與未知區域的交界線。
        """
        # 使用卷積找到邊界
        # 未知且鄰接已探索 = Frontier
        unknown = (self.exploration_map == 0).float()
        explored = (self.exploration_map >= 1).float()
        
        # 簡化邊界檢測：使用形態學梯度
        # 實際實現可能需要更複雜的算法
        pass
    
    def find_frontiers(
        self,
        robot_pos: torch.Tensor,
        max_frontiers: int = 5,
    ) -> List[Frontier]:
        """尋找 Frontier
        
        Args:
            robot_pos: [2] 機器人位置
            max_frontiers: 最多返回的 Frontier 數量
        
        Returns:
            Frontier 列表，按優先級排序
        """
        frontiers = []
        
        # 簡化實現：基於探索地圖找邊界點
        # 實際應該使用更高效的算法（如 Wavefront Frontier Detection）
        
        # 找出所有未知且鄰近已探索的網格
        explored_mask = (self.exploration_map >= 1).squeeze(0)
        unknown_mask = (self.exploration_map == 0).squeeze(0)
        
        # 找邊界點（未知但鄰近已探索）
        boundary_points = torch.argwhere(
            unknown_mask & self._is_adjacent_to_explored(explored_mask)
        )
        
        if len(boundary_points) == 0:
            return frontiers
        
        # 聚類邊界點為 Frontiers
        # 簡化實現：使用 DBSCAN 或簡單的距離聚類
        frontiers = self._cluster_frontiers(boundary_points, robot_pos, max_frontiers)
        
        return frontiers
    
    def _is_adjacent_to_explored(
        self,
        explored_mask: torch.Tensor,
    ) -> torch.Tensor:
        """檢查每個未知格是否鄰近已探索格
        
        Args:
            explored_mask: [H, W] 已探索掩碼
        
        Returns:
            [H, W] 是否鄰近已探索
        """
        h, w = explored_mask.shape
        adjacent = torch.zeros_like(explored_mask, dtype=torch.bool)
        
        # 檢查 8 鄰域
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                shifted = torch.roll(explored_mask, shifts=(dy, dx), dims=(0, 1))
                adjacent |= shifted
        
        return adjacent
    
    def _cluster_frontiers(
        self,
        boundary_points: torch.Tensor,
        robot_pos: torch.Tensor,
        max_frontiers: int,
    ) -> List[Frontier]:
        """將邊界點聚類為 Frontiers
        
        Args:
            boundary_points: [N, 2] 邊界點坐標（網格坐標）
            robot_pos: [2] 機器人位置
            max_frontiers: 最多返回的 Frontier 數量
        
        Returns:
            Frontier 列表
        """
        frontiers = []
        
        if len(boundary_points) == 0:
            return frontiers
        
        # 計算每個邊界點到機器人的距離
        robot_grid_x = int((robot_pos[0].item() + self.map_size[0] / 2) / self.grid_resolution)
        robot_grid_y = int((robot_pos[1].item() + self.map_size[1] / 2) / self.grid_resolution)
        
        distances = torch.sqrt(
            (boundary_points[:, 0].float() - robot_grid_y)**2 +
            (boundary_points[:, 1].float() - robot_grid_x)**2
        ) * self.grid_resolution
        
        # 按距離排序
        sorted_indices = torch.argsort(distances)
        
        # 簡化聚類：直接選取最近的幾個點作為不同的 Frontier
        for i in range(min(max_frontiers, len(sorted_indices))):
            idx = sorted_indices[i].item()
            point = boundary_points[idx]
            
            # 轉換為世界坐標
            world_x = point[1].item() * self.grid_resolution - self.map_size[0] / 2
            world_y = point[0].item() * self.grid_resolution - self.map_size[1] / 2
            
            center = torch.tensor([world_x, world_y])
            
            # 計算優先級（距離越近優先級越高）
            priority = 1.0 / (distances[idx].item() + 1.0)
            
            frontier = Frontier(
                center=center,
                points=center.unsqueeze(0),  # 簡化：只有中心點
                size=self.frontier_min_size,
                priority=priority,
            )
            
            frontiers.append(frontier)
        
        return frontiers
    
    def select_exploration_goal(
        self,
        robot_pos: torch.Tensor,
        frontiers: List[Frontier],
    ) -> Optional[torch.Tensor]:
        """選擇探索目標
        
        Args:
            robot_pos: [2] 機器人位置
            frontiers: Frontier 列表
        
        Returns:
            [2] 選定的目標位置，如果沒有 Frontier 則返回 None
        """
        if not frontiers:
            return None
        
        # 選擇優先級最高的 Frontier
        best_frontier = max(frontiers, key=lambda f: f.priority)
        
        return best_frontier.center


class FrontierBasedNavigation:
    """基於 Frontier 的導航系統
    
    用於完全未知環境的自主探索。
    """
    
    def __init__(
        self,
        explorer: FrontierExplorer,
        hierarchical_manager: HierarchicalNavigationManager,
    ):
        """初始化 Frontier 導航系統
        
        Args:
            explorer: Frontier 探索器
            hierarchical_manager: 層級式導航管理器
        """
        self.explorer = explorer
        self.hierarchical_manager = hierarchical_manager
        
        # 探索狀態
        self._current_frontier: Optional[Frontier] = None
        self._exploration_complete = False
    
    def update(
        self,
        env_id: int,
        robot_pos: torch.Tensor,
        lidar_data: torch.Tensor,
        dt: float,
    ) -> dict:
        """更新探索導航
        
        Args:
            env_id: 環境 ID
            robot_pos: [2] 機器人位置
            lidar_data: [N] LiDAR 數據
            dt: 時間步長
        
        Returns:
            調試信息
        """
        debug_info = {}
        
        # 更新探索地圖
        self.explorer.update_exploration_map(robot_pos, lidar_data)
        
        # 檢查是否到達當前 Frontier
        if self._current_frontier is not None:
            dist = torch.norm(robot_pos - self._current_frontier.center).item()
            
            if dist < 1.0:  # 到達 Frontier
                self._current_frontier = None
        
        # 如果沒有當前 Frontier，尋找新的
        if self._current_frontier is None:
            frontiers = self.explorer.find_frontiers(robot_pos)
            
            if frontiers:
                # 有新的 Frontier，設為目標
                goal = self.explorer.select_exploration_goal(robot_pos, frontiers)
                
                if goal is not None:
                    self._current_frontier = frontiers[0]  # 保存選中的 Frontier
                    
                    # 更新目標命令
                    self.hierarchical_manager.env.command_manager.get_command
                    debug_info["new_frontier"] = goal.numpy()
            else:
                # 沒有更多 Frontier，探索完成
                self._exploration_complete = True
                debug_info["exploration_complete"] = True
        
        return debug_info
    
    def is_exploration_complete(self) -> bool:
        """檢查探索是否完成"""
        return self._exploration_complete
    
    def get_exploration_progress(self) -> float:
        """獲取探索進度
        
        Returns:
            進度百分比 [0, 1]
        """
        total_cells = self.explorer.grid_width * self.explorer.grid_height
        explored_cells = (self.explorer.exploration_map >= 1).sum().item()
        
        return explored_cells / total_cells


__all__ = [
    "Frontier",
    "FrontierExplorer",
    "FrontierBasedNavigation",
]
