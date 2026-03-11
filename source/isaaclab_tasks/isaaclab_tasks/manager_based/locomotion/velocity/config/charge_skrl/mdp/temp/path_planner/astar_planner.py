"""
A* 路徑規劃器 (A* Path Planner)

實現經典 A* 演算法進行網格路徑規劃。
"""

from __future__ import annotations

import heapq
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .environment_map import EnvironmentMap, EnvironmentMapCfg


@dataclass
class AStarPathPlannerCfg:
    """A* 路徑規劃器配置類別

    Attributes:
        map_cfg: 環境地圖配置
        heuristic_weight: 啟發式函數權重（越大越傾向於目標方向）
        allow_diagonal: 是否允許對角線移動
        path_smoothing: 是否啟用路徑平滑
        max_iterations: 最大搜尋迭代次數（防止無限循環）
        waypoint_spacing: 路徑點間隔（米）
    """
    map_cfg: EnvironmentMapCfg = field(default_factory=EnvironmentMapCfg)
    heuristic_weight: float = 1.0
    allow_diagonal: bool = True
    path_smoothing: bool = True
    max_iterations: int = 100000
    waypoint_spacing: float = 0.2  # 20cm between waypoints


class AStarNode:
    """A* 節點類別"""

    __slots__ = ['x', 'y', 'g', 'h', 'f', 'parent']

    def __init__(
        self,
        x: int,
        y: int,
        g: float = 0.0,
        h: float = 0.0,
        parent: Optional['AStarNode'] = None,
    ):
        self.x = x  # 網格 X 坐標
        self.y = y  # 網格 Y 坐標
        self.g = g  # 從起點的實際代價
        self.h = h  # 到終點的啟發式估計代價
        self.f = g + h  # 總估計代價
        self.parent = parent  # 父節點

    def __lt__(self, other: 'AStarNode') -> bool:
        """用於 heapq 比較"""
        return self.f < other.f

    def __eq__(self, other: object) -> bool:
        """判斷兩個節點是否相同（位置相同）"""
        if not isinstance(other, AStarNode):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """用於 set 和 dict 的哈希"""
        return hash((self.x, self.y))


class AStarPathPlanner:
    """A* 路徑規劃器類別

    使用 A* 演算法在網格地圖上找到從起點到終點的最短路徑。
    支援對角線移動和啟發式函數權重調整。
    """

    def __init__(self, cfg: AStarPathPlannerCfg):
        """初始化 A* 路徑規劃器

        Args:
            cfg: 規劃器配置
        """
        self.cfg = cfg
        self.map = EnvironmentMap(cfg.map_cfg)

        # 預計算移動代價
        self.straight_cost = 1.0
        self.diagonal_cost = np.sqrt(2)  # ~1.414

    def heuristic(self, node: AStarNode, goal: AStarNode) -> float:
        """計算啟發式函數（估計從 node 到 goal 的代價）

        使用歐幾里得距離作為啟發式函數。

        Args:
            node: 當前節點
            goal: 目標節點

        Returns:
            啟發式估計代價
        """
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)

        if self.cfg.allow_diagonal:
            # 對角線距離
            return self.cfg.heuristic_weight * max(dx, dy)
        else:
            # 曼哈頓距離
            return self.cfg.heuristic_weight * (dx + dy)

    def plan_path(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        env: Optional[ManagerBasedRLEnv] = None,
    ) -> torch.Tensor:
        """規劃從起點到終點的路徑

        Args:
            start_pos: [2] or [num_envs, 2] 起點位置（世界坐標，米）
            goal_pos: [2] or [num_envs, 2] 終點位置（世界坐標，米）
            env: 環境實例（可選，用於更新障礙物）

        Returns:
            [num_waypoints, 2] 路徑點坐標（世界坐標，米）
            如果找不到路徑，返回空張量
        """
        # 確保輸入形狀正確
        if start_pos.dim() == 1:
            start_pos = start_pos.unsqueeze(0)
        if goal_pos.dim() == 1:
            goal_pos = goal_pos.unsqueeze(0)

        # 只處理單個環境
        start = start_pos[0]
        goal = goal_pos[0]

        # 從環境更新障礙物（如果提供）
        if env is not None:
            self.map.update_from_env(env)

        # 轉換為網格坐標
        start_grid = self.map.world_to_grid(start)
        goal_grid = self.map.world_to_grid(goal)

        # 檢查起點和終點是否有效
        if self.map.is_occupied(*start_grid):
            print(f"[A*] 警告：起點 {start.cpu().numpy()} 在障礙物內")
            # 嘗試找到最近的自由網格
            start_grid = self._find_nearest_free_cell(start_grid)

        if self.map.is_occupied(*goal_grid):
            print(f"[A*] 警告：終點 {goal.cpu().numpy()} 在障礙物內")
            goal_grid = self._find_nearest_free_cell(goal_grid)

        # 執行 A* 搜尋
        path_grid = self._astar_search(start_grid, goal_grid)

        if path_grid is None or len(path_grid) == 0:
            print(f"[A*] 無法找到從 {start.cpu().numpy()} 到 {goal.cpu().numpy()} 的路徑")
            return torch.empty((0, 2), dtype=torch.float32)

        # 轉換為世界坐標
        path_world = torch.stack([
            self.map.grid_to_world(x, y)
            for x, y in path_grid
        ])

        # 路徑平滑（減少路徑點）
        if self.cfg.path_smoothing and len(path_world) > 2:
            path_world = self._smooth_path(path_world)

        # 根據間隔取樣路徑點
        if self.cfg.waypoint_spacing > 0:
            path_world = self._subsample_path(path_world, self.cfg.waypoint_spacing)

        return path_world

    def _astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """執行 A* 搜尋

        Args:
            start: 起點網格坐標 (x, y)
            goal: 終點網格坐標 (x, y)

        Returns:
            路徑網格坐標列表 [(x1, y1), (x2, y2), ...]
            如果找不到路徑，返回 None
        """
        start_node = AStarNode(start[0], start[1])
        goal_node = AStarNode(goal[0], goal[1])

        # 開放列表（待探索節點）
        open_list: List[AStarNode] = []
        heapq.heappush(open_list, start_node)

        # 關閉列表（已探索節點）
        closed_set: set = set()

        # g_score[n]: 從起點到 n 的最小代價
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        # 迭代計數
        iterations = 0

        while open_list and iterations < self.cfg.max_iterations:
            iterations += 1

            # 取出 f 值最小的節點
            current = heapq.heappop(open_list)

            # 檢查是否到達目標
            if current == goal_node:
                return self._reconstruct_path(current)

            # 標記為已探索
            closed_set.add((current.x, current.y))

            # 探索相鄰節點
            neighbors = self.map.get_neighbors(
                current.x, current.y,
                diagonal=self.cfg.allow_diagonal
            )

            for nx, ny in neighbors:
                if (nx, ny) in closed_set:
                    continue

                # 計算移動代價
                is_diagonal = (abs(nx - current.x) == 1 and abs(ny - current.y) == 1)
                move_cost = self.diagonal_cost if is_diagonal else self.straight_cost

                tentative_g = current.g + move_cost

                # 檢查是否找到更短路徑
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g

                    neighbor = AStarNode(
                        nx, ny,
                        g=tentative_g,
                        h=self.heuristic(AStarNode(nx, ny), goal_node),
                        parent=current,
                    )

                    # 檢查是否已在開放列表中
                    in_open = False
                    for i, node in enumerate(open_list):
                        if node == neighbor:
                            if tentative_g < node.g:
                                open_list[i] = neighbor  # 替換為更優的節點
                                heapq.heapify(open_list)
                            in_open = True
                            break

                    if not in_open:
                        heapq.heappush(open_list, neighbor)

        print(f"[A*] 搜尋超過最大迭代次數 ({self.cfg.max_iterations})")
        return None

    def _reconstruct_path(self, node: AStarNode) -> List[Tuple[int, int]]:
        """從終點回溯重建路徑

        Args:
            node: 終點節點

        Returns:
            路徑坐標列表（從起點到終點）
        """
        path = []
        current = node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()  # 反轉使從起點到終點
        return path

    def _find_nearest_free_cell(
        self,
        grid_pos: Tuple[int, int],
        max_radius: int = 10,
    ) -> Tuple[int, int]:
        """找到最近的自由網格

        Args:
            grid_pos: 網格坐標
            max_radius: 最大搜尋半徑

        Returns:
            最近的自由網格坐標
        """
        for radius in range(max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = grid_pos[0] + dx, grid_pos[1] + dy
                    if 0 <= x < self.map.grid_width and 0 <= y < self.map.grid_depth:
                        if not self.map.is_occupied(x, y):
                            return (x, y)

        return grid_pos  # 找不到，返回原位置

    def _smooth_path(
        self,
        path: torch.Tensor,
        max_deviation: float = 0.5,
    ) -> torch.Tensor:
        """平滑路徑（移除不必要的路徑點）

        使用直線可見性檢查：如果兩點之間沒有障礙物，移除中間的點。

        Args:
            path: [N, 2] 原始路徑
            max_deviation: 允許的最大偏離距離

        Returns:
            平滑後的路徑
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]  # 保留起點
        current_idx = 0

        while current_idx < len(path) - 1:
            # 找到最遠的可見點
            farthest_idx = current_idx + 1

            for i in range(current_idx + 2, len(path)):
                if self._is_line_clear(path[current_idx], path[i]):
                    farthest_idx = i
                else:
                    break

            smoothed.append(path[farthest_idx])
            current_idx = farthest_idx

        # 確保終點被包含
        if not torch.allclose(smoothed[-1], path[-1]):
            smoothed.append(path[-1])

        return torch.stack(smoothed)

    def _is_line_clear(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        step_size: float = 0.1,
    ) -> bool:
        """檢查兩點之間的直線是否沒有障礙物

        Args:
            start: 起點坐標 [2]
            end: 終點坐標 [2]
            step_size: 採樣步長

        Returns:
            True 如果直線上沒有障礙物
        """
        direction = end - start
        distance = torch.norm(direction)

        if distance < step_size:
            return True

        num_steps = int(distance / step_size) + 1
        direction = direction / distance

        for i in range(1, num_steps):
            t = i / num_steps
            check_pos = start + t * (end - start)

            if self.map.is_occupied_world(check_pos):
                return False

        return True

    def _subsample_path(
        self,
        path: torch.Tensor,
        spacing: float,
    ) -> torch.Tensor:
        """根據間隔取樣路徑點

        Args:
            path: [N, 2] 原始路徑
            spacing: 取樣間隔（米）

        Returns:
            取樣後的路徑
        """
        if len(path) <= 2:
            return path

        subsampled = [path[0]]
        last_pos = path[0]
        accumulated_distance = 0.0

        for i in range(1, len(path) - 1):
            segment_distance = torch.norm(path[i] - last_pos)
            accumulated_distance += segment_distance

            if accumulated_distance >= spacing:
                subsampled.append(path[i])
                last_pos = path[i]
                accumulated_distance = 0.0

        # 確保終點被包含
        subsampled.append(path[-1])

        return torch.stack(subsampled)

    def update_obstacles(
        self,
        obstacles: List[dict],
    ) -> None:
        """更新障礙物地圖

        Args:
            obstacles: 障礙物列表
                每個障礙物包含 {'position': [x, y], 'size': [w, h], 'shape': str}
        """
        self.map.clear()
        for obs in obstacles:
            pos = torch.tensor(obs['position'])
            size = torch.tensor(obs['size'])
            shape = obs.get('shape', 'cuboid')
            self.map.add_obstacle(pos, size, shape)

    def visualize_path(
        self,
        path: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
    ) -> np.ndarray:
        """生成路徑可視化圖像

        Args:
            path: [N, 2] 路徑點
            start: [2] 起點
            goal: [2] 終點

        Returns:
            [H, W, 3] RGB 圖像
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use('Agg')

        # 獲取佔據網格
        occupancy = self.map.get_occupancy_numpy().T  # 轉置以正確顯示

        # 創建圖像
        fig, ax = plt.subplots(figsize=(8, 8))

        # 繪製佔據網格
        ax.imshow(occupancy, cmap='gray_r', origin='lower',
                  extent=[-self.map.cfg.map_size[0]/2, self.map.cfg.map_size[0]/2,
                         -self.map.cfg.map_size[1]/2, self.map.cfg.map_size[1]/2])

        # 繪製路徑
        if len(path) > 0:
            path_np = path.cpu().numpy()
            ax.plot(path_np[:, 0], path_np[:, 1], 'b-', linewidth=2, label='A* Path')
            ax.scatter(path_np[:, 0], path_np[:, 1], c='blue', s=20, alpha=0.5)

        # 繪製起點和終點
        ax.scatter(start[0].item(), start[1].item()], c='green', s=200, marker='o', label='Start', zorder=5)
        ax.scatter(goal[0].item(), goal[1].item(), c='red', s=200, marker='*', label='Goal', zorder=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'A* Path Planning ({self.map})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 轉換為圖像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return image

    def __repr__(self) -> str:
        return (
            f"AStarPathPlanner("
            f"heuristic_weight={self.cfg.heuristic_weight}, "
            f"diagonal={self.cfg.allow_diagonal}, "
            f"smoothing={self.cfg.path_smoothing}, "
            f"map={self.map})"
        )


__all__ = [
    "AStarPathPlanner",
    "AStarPathPlannerCfg",
    "AStarNode",
]
