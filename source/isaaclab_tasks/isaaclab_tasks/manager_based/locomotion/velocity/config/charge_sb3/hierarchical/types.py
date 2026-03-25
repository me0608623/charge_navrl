"""分層式導航類型定義

此模組定義分層式導航系統的共用類型。
"""

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Optional, List, Tuple

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ============================================================================
# 路徑規劃相關類型
# ============================================================================

class Waypoint(NamedTuple):
    """路徑航點

    Attributes:
        position: 2D 位置 [x, y]
        reached: 是否已到達
    """
    position: Tuple[float, float]
    reached: bool = False


class Path:
    """路徑（航點序列）

    Attributes:
        waypoints: 航點列表
        current_index: 當前目標航點索引
        total_length: 路徑總長度
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, float]],
    ):
        self.waypoints: List[Waypoint] = [
            Waypoint(pos, reached=False) for pos in waypoints
        ]
        self.current_index: int = 0
        self.total_length: float = self._calculate_total_length()

    def _calculate_total_length(self) -> float:
        """計算路徑總長度"""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            pos1 = self.waypoints[i].position
            pos2 = self.waypoints[i + 1].position
            total += ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5

        return total

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """獲取當前目標航點"""
        if self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    def advance_waypoint(self) -> bool:
        """前進到下一個航點

        Returns:
            是否成功前進（如果已到達終點，返回 False）
        """
        if self.current_index < len(self.waypoints) - 1:
            self.waypoints[self.current_index] = self.waypoints[self.current_index]._replace(reached=True)
            self.current_index += 1
            return True
        elif self.current_index == len(self.waypoints) - 1:
            self.waypoints[self.current_index] = self.waypoints[self.current_index]._replace(reached=True)
            return False
        return False

    def is_complete(self) -> bool:
        """檢查路徑是否已完成"""
        return self.current_index >= len(self.waypoints) - 1

    def get_remaining_length(self) -> float:
        """獲取剩餘路徑長度"""
        if self.is_complete():
            return 0.0

        total = 0.0
        for i in range(self.current_index, len(self.waypoints) - 1):
            pos1 = self.waypoints[i].position
            pos2 = self.waypoints[i + 1].position
            total += ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5

        return total

    def __len__(self) -> int:
        """路徑航點數量"""
        return len(self.waypoints)

    def __repr__(self) -> str:
        return f"Path(waypoints={len(self)}, current={self.current_index}, remaining={self.get_remaining_length():.2f}m)"


# ============================================================================
# 安全過濾器相關類型
# ============================================================================

class SafetyAction(NamedTuple):
    """安全動作（過濾後的動作）

    Attributes:
        linear_speed: 線性速度
        angular_speed: 角速度
        safe: 是否安全
        reason: 不安全原因（如果不安全）
    """
    linear_speed: float
    angular_speed: float
    safe: bool = True
    reason: str = ""

    @classmethod
    def unsafe(cls, linear_speed: float, angular_speed: float, reason: str) -> "SafetyAction":
        """創建不安全的動作"""
        return cls(linear_speed, angular_speed, safe=False, reason=reason)

    @classmethod
    def safe_action(cls, linear_speed: float, angular_speed: float) -> "SafetyAction":
        """創建安全的動作"""
        return cls(linear_speed, angular_speed, safe=True, reason="")

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """轉換為 Tensor

        Returns:
            [linear_speed, angular_speed]
        """
        return torch.tensor(
            [self.linear_speed, self.angular_speed],
            dtype=torch.float32,
            device=device,
        )


# ============================================================================
# 狀態管理相關類型
# ============================================================================

class NavigationState(NamedTuple):
    """導航狀態

    Attributes:
        current_path: 當前路徑
        goal_position: 終點位置（世界座標）
        robot_position: 機器人位置（世界座標）
        sub_goal_reached: sub-goal 是否已到達
        navigation_complete: 導航是否完成
    """
    current_path: Optional[Path]
    goal_position: Tuple[float, float]
    robot_position: Tuple[float, float]
    sub_goal_reached: bool = False
    navigation_complete: bool = False

    @classmethod
    def initial(cls, goal_position: Tuple[float, float]) -> "NavigationState":
        """創建初始狀態"""
        return cls(
            current_path=None,
            goal_position=goal_position,
            robot_position=(0.0, 0.0),
            sub_goal_reached=False,
            navigation_complete=False,
        )


# ============================================================================
# 地圖相關類型
# ============================================================================

class Obstacle:
    """障礙物

    Attributes:
        position: 中心位置 [x, y]
        size: 尺寸（半徑或邊長）
        shape: 形状 ('circle' 或 'rectangle')
    """

    def __init__(
        self,
        position: Tuple[float, float],
        size: float,
        shape: str = "circle",
    ):
        self.position = position
        self.size = size
        self.shape = shape

    def is_collision(
        self,
        point: Tuple[float, float],
        robot_radius: float = 0.5,
    ) -> bool:
        """檢查點是否與障礙物碰撞"""
        if self.shape == "circle":
            # 圓形障礙物：檢查距離
            dist = ((point[0] - self.position[0]) ** 2 + (point[1] - self.position[1]) ** 2) ** 0.5
            return dist < (self.size / 2 + robot_radius)
        elif self.shape == "rectangle":
            # 矩形障礙物：檢查點是否在矩形內（擴展 robot_radius）
            half_size = (self.size / 2 + robot_radius)
            return (
                self.position[0] - half_size <= point[0] <= self.position[0] + half_size
                and self.position[1] - half_size <= point[1] <= self.position[1] + half_size
            )
        return False


class GridMap:
    """網格地圖（用於路徑規劃）

    Attributes:
        resolution: 網格解析度（米/格子）
        size_x: x 方向大小（格子數）
        size_y: y 方向大小（格子數）
        origin: 地圖原點（世界座標）
        grid: 2D 網格（True 表示障礙，False 表示自由）
    """

    def __init__(
        self,
        resolution: float = 0.1,
        size_x: int = 100,
        size_y: int = 100,
        origin: Tuple[float, float] = (-5.0, -5.0),
    ):
        self.resolution = resolution
        self.size_x = size_x
        self.size_y = size_y
        self.origin = origin
        self.grid = torch.zeros(size_y, size_x, dtype=torch.bool)

    def world_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """世界座標轉網格座標"""
        x = int((position[0] - self.origin[0]) / self.resolution)
        y = int((position[1] - self.origin[1]) / self.resolution)
        return (x, y)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """網格座標轉世界座標"""
        x = grid_pos[0] * self.resolution + self.origin[0]
        y = grid_pos[1] * self.resolution + self.origin[1]
        return (x, y)

    def set_obstacle(
        self,
        obstacle: Obstacle,
        robot_radius: float = 0.5,
    ):
        """設置障礙物到地圖"""
        if obstacle.shape == "circle":
            # 圓形障礙物：填充圓形區域
            center_grid = self.world_to_grid(obstacle.position)
            radius_grid = int((obstacle.size / 2 + robot_radius) / self.resolution)

            y, x = torch.meshgrid(
                torch.arange(self.size_y),
                torch.arange(self.size_x),
                indexing="ij",
            )
            dist_sq = (x - center_grid[0]) ** 2 + (y - center_grid[1]) ** 2
            mask = dist_sq <= radius_grid ** 2
            self.grid[mask] = True

        elif obstacle.shape == "rectangle":
            # 矩形障礙物：填充矩形區域
            center_grid = self.world_to_grid(obstacle.position)
            half_size = int((obstacle.size / 2 + robot_radius) / self.resolution)

            x_min = max(0, center_grid[0] - half_size)
            x_max = min(self.size_x, center_grid[0] + half_size + 1)
            y_min = max(0, center_grid[1] - half_size)
            y_max = min(self.size_y, center_grid[1] + half_size + 1)

            self.grid[y_min:y_max, x_min:x_max] = True

    def is_collision(
        self,
        position: Tuple[float, float],
        robot_radius: float = 0.5,
    ) -> bool:
        """檢查位置是否與地圖中的障礙物碰撞"""
        grid_pos = self.world_to_grid(position)
        x, y = grid_pos

        # 邊界檢查
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return True

        # 檢查周圍網格（考慮機器人半徑）
        radius_grid = int(robot_radius / self.resolution)

        y_min = max(0, y - radius_grid)
        y_max = min(self.size_y, y + radius_grid + 1)
        x_min = max(0, x - radius_grid)
        x_max = min(self.size_x, x + radius_grid + 1)

        return self.grid[y_min:y_max, x_min:x_max].any().item()

    def __repr__(self) -> str:
        return f"GridMap(res={self.resolution}, size={self.size_x}x{self.size_y}, origin={self.origin})"
