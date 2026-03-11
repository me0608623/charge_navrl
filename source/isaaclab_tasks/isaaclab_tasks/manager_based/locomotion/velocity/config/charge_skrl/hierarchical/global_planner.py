"""全域路徑規劃器模組

實現分層式導航的上層：全域路徑規劃
支持算法：A*, Dijkstra, RRT*
"""

from __future__ import annotations

import heapq
import math
import random
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

from .types import Obstacle, Path, Waypoint, GridMap


# ============================================================================
# 基礎路徑規劃器類
# ============================================================================

class GlobalPlanner:
    """全域路徑規劃器基類

    負責在已知地圖上規劃從起點到終點的路徑。
    """

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
        grid_map: Optional[GridMap] = None,
    ) -> Path:
        """規劃路徑

        Args:
            start: 起點位置 [x, y]
            goal: 終點位置 [x, y]
            obstacles: 障礙物列表
            grid_map: 網格地圖（可選）

        Returns:
            路徑對象
        """
        raise NotImplementedError("子類必須實現 plan 方法")


# ============================================================================
# A* 路徑規劃器
# ============================================================================

class AStarPlanner(GlobalPlanner):
    """A* 路徑規劃器

    使用 A* 算法在網格地圖上規劃最短路徑。
    """

    def __init__(
        self,
        grid_map: GridMap,
        diagonal_movement: bool = True,
    ):
        """初始化 A* 規劃器

        Args:
            grid_map: 網格地圖
            diagonal_movement: 是否允許對角線移動
        """
        self.grid_map = grid_map
        self.diagonal_movement = diagonal_movement

        # 預計算移動代價
        self.movement_costs = self._compute_movement_costs()

    def _compute_movement_costs(self) -> List[Tuple[int, int, float]]:
        """計算移動代價

        Returns:
            [(dx, dy, cost), ...] 移動方向列表
        """
        costs = []

        # 四方向移動
        costs.extend([
            (0, 1, 1.0),     # 上
            (0, -1, 1.0),    # 下
            (1, 0, 1.0),     # 右
            (-1, 0, 1.0),    # 左
        ])

        # 對角線移動
        if self.diagonal_movement:
            costs.extend([
                (1, 1, math.sqrt(2)),      # 右上
                (1, -1, math.sqrt(2)),     # 右下
                (-1, 1, math.sqrt(2)),     # 左上
                (-1, -1, math.sqrt(2)),    # 左下
            ])

        return costs

    def _heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:
        """計算啟發函數

        Args:
            a: 網格坐標 a
            b: 網格坐標 b

        Returns:
            啟發值（估計距離）
        """
        if self.diagonal_movement:
            # 八方向移動：使用切比雪夫距離
            return max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        else:
            # 四方向移動：使用曼哈頓距離
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

    def _get_neighbors(
        self,
        node: Tuple[int, int],
        closed_set: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """獲取鄰居節點

        Args:
            node: 當前節點
            closed_set: 已訪問節點集合

        Returns:
            可訪問的鄰居節點列表
        """
        neighbors = []

        for dx, dy, _ in self.movement_costs:
            nx, ny = node[0] + dx, node[1] + dy

            # 邊界檢查
            if not (0 <= nx < self.grid_map.size_x and 0 <= ny < self.grid_map.size_y):
                continue

            # 障礙物檢查
            if (nx, ny) in closed_set or self.grid_map.grid[ny, nx]:
                continue

            neighbors.append((nx, ny))

        return neighbors

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
        grid_map: Optional[GridMap] = None,
    ) -> Path:
        """A* 路徑規劃

        Args:
            start: 起點位置（世界坐標）
            goal: 終點位置（世界坐標）
            obstacles: 障礙物列表
            grid_map: 網格地圖（如果為 None，使用 self.grid_map）

        Returns:
            路徑對象
        """
        # 使用提供的 grid_map 或默認 grid_map
        gm = grid_map if grid_map is not None else self.grid_map

        # 更新障礙物到地圖
        for obs in obstacles:
            gm.set_obstacle(obs)

        # 轉換為網格坐標
        start_grid = gm.world_to_grid(start)
        goal_grid = gm.world_to_grid(goal)

        # 邊界檢查
        if not (0 <= start_grid[0] < gm.size_x and 0 <= start_grid[1] < gm.size_y):
            return self._fallback_path(start, goal)
        if not (0 <= goal_grid[0] < gm.size_x and 0 <= goal_grid[1] < gm.size_y):
            return self._fallback_path(start, goal)

        # A* 算法
        open_set = []
        heapq.heappush(open_set, (0, start_grid))

        came_from = {}  # 記錄路徑
        g_score = {start_grid: 0}  # 起點到當前節點的代價
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}  # 總代價

        closed_set = set()

        while open_set:
            # 獲取 f_score 最小的節點
            current = heapq.heappop(open_set)[1]

            # 到達目標
            if current == goal_grid:
                return self._reconstruct_path(came_from, current, gm)

            closed_set.add(current)

            # 探索鄰居
            for neighbor in self._get_neighbors(current, closed_set):
                tentative_g_score = g_score[current] + self._movement_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # 未找到路徑
        return self._fallback_path(start, goal)

    def _movement_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """計算從 a 到 b 的移動代價"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        for mx, my, cost in self.movement_costs:
            if dx == mx and dy == my:
                return cost

        return 1.0

    def _reconstruct_path(
        self,
        came_from: dict,
        current: Tuple[int, int],
        grid_map: GridMap,
    ) -> Path:
        """重建路徑

        Args:
            came_from: 節點到前驅節點的映射
            current: 終點節點
            grid_map: 網格地圖

        Returns:
            路徑對象
        """
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        # 轉換為世界坐標
        waypoints = [grid_map.grid_to_world(p) for p in path]

        return Path(waypoints)

    def _fallback_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """失敗回退路徑（直線路徑）

        Args:
            start: 起點
            goal: 終點

        Returns:
            直線路徑
        """
        waypoints = [start, goal]
        return Path(waypoints)


# ============================================================================
# RRT* 路徑規劃器
# ============================================================================

class RRTStarPlanner(GlobalPlanner):
    """RRT* 路徑規劃器

    使用 RRT* (Rapidly-exploring Random Tree Star) 算法規劃路徑。
    適合連續空間和高維配置空間。
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
        max_iterations: int = 10000,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.1,
        neighbor_radius: float = 1.0,
        robot_radius: float = 0.5,
    ):
        """初始化 RRT* 規劃器

        Args:
            bounds: 空間邊界 [x_min, x_max, y_min, y_max]
            max_iterations: 最大迭代次數
            step_size: 擴展步長
            goal_sample_rate: 目標採樣率
            neighbor_radius: 鄰居搜索半徑
            robot_radius: 機器人半徑
        """
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius = neighbor_radius
        self.robot_radius = robot_radius

    def _sample_random(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """隨機採樣

        Args:
            goal: 目標位置

        Returns:
            採樣點
        """
        # 以一定概率採樣目標
        if random.random() < self.goal_sample_rate:
            return goal

        # 隨機採樣
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        return (x, y)

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """計算兩點的歐幾里得距離"""
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def _steer(
        self,
        from_node: Tuple[float, float],
        to_point: Tuple[float, float],
    ) -> Tuple[float, float]:
        """朝目標方向擴展

        Args:
            from_node: 起始節點
            to_point: 目標點

        Returns:
            新節點
        """
        dist = self._distance(from_node, to_point)

        if dist <= self.step_size:
            return to_point

        # 限制步長
        ratio = self.step_size / dist
        x = from_node[0] + ratio * (to_point[0] - from_node[0])
        y = from_node[1] + ratio * (to_point[1] - from_node[1])

        return (x, y)

    def _is_collision_free(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        obstacles: List[Obstacle],
    ) -> bool:
        """檢查路徑是否碰撞

        Args:
            a: 起點
            b: 終點
            obstacles: 障礙物列表

        Returns:
            是否無碰撞
        """
        # 在路徑上採樣點檢查碰撞
        num_samples = int(self._distance(a, b) / 0.1) + 1

        for i in range(num_samples + 1):
            t = i / num_samples
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])

            for obs in obstacles:
                if obs.is_collision((x, y), self.robot_radius):
                    return False

        return True

    def _find_neighbors(
        self,
        node: Tuple[float, float],
        nodes: List[Tuple[float, float]],
    ) -> List[Tuple[int, float]]:
        """查找鄰居節點

        Args:
            node: 目標節點
            nodes: 所有節點

        Returns:
            [(idx, distance), ...] 鄰居索引和距離
        """
        neighbors = []

        for idx, n in enumerate(nodes):
            dist = self._distance(node, n)
            if dist <= self.neighbor_radius:
                neighbors.append((idx, dist))

        return neighbors

    def _cost(self, parent: Tuple[float, float], child: Tuple[float, float]) -> float:
        """計算代價（距離）"""
        return self._distance(parent, child)

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
        grid_map: Optional[GridMap] = None,
    ) -> Path:
        """RRT* 路徑規劃

        Args:
            start: 起點
            goal: 終點
            obstacles: 障礙物列表
            grid_map: 網格地圖（未使用）

        Returns:
            路徑對象
        """
        # 初始化樹
        nodes = [start]
        parents = {0: None}  # 節點索引 -> 父節點索引
        costs = {0: 0.0}  # 節點索引 -> 起點到節點的代價

        for _ in range(self.max_iterations):
            # 隨機採樣
            x_rand = self._sample_random(goal)

            # 找到最近節點
            nearest_idx = min(
                range(len(nodes)),
                key=lambda i: self._distance(nodes[i], x_rand),
            )
            x_nearest = nodes[nearest_idx]

            # 擴展樹
            x_new = self._steer(x_nearest, x_rand)

            # 碰撞檢查
            if not self._is_collision_free(x_nearest, x_new, obstacles):
                continue

            # 添加新節點
            new_idx = len(nodes)
            nodes.append(x_new)

            # 查找鄰居
            neighbors = self._find_neighbors(x_new, nodes)

            # 選擇最佳父節點
            min_cost = costs[nearest_idx] + self._cost(x_nearest, x_new)
            best_parent = nearest_idx

            for idx, dist in neighbors:
                if idx == new_idx:
                    continue

                tentative_cost = costs[idx] + self._cost(nodes[idx], x_new)
                if tentative_cost < min_cost:
                    min_cost = tentative_cost
                    best_parent = idx

            parents[new_idx] = best_parent
            costs[new_idx] = min_cost

            # 重新布線
            for idx, dist in neighbors:
                if idx == new_idx or idx == best_parent:
                    continue

                if self._is_collision_free(nodes[idx], x_new, obstacles):
                    tentative_cost = min_cost + self._cost(x_new, nodes[idx])
                    if tentative_cost < costs[idx]:
                        parents[idx] = new_idx
                        costs[idx] = tentative_cost

            # 檢查是否到達目標
            if self._distance(x_new, goal) < self.step_size:
                if self._is_collision_free(x_new, goal, obstacles):
                    # 添加目標節點
                    goal_idx = len(nodes)
                    nodes.append(goal)
                    parents[goal_idx] = new_idx

                    # 重建路徑
                    path = self._reconstruct_path_rrt(parents, nodes, goal_idx)

                    if len(path) > 0:
                        return Path([tuple(p) for p in path])

        # 未找到路徑，返回直線路徑
        return self._fallback_path(start, goal)

    def _reconstruct_path_rrt(
        self,
        parents: dict,
        nodes: List[Tuple[float, float]],
        goal_idx: int,
    ) -> List[Tuple[float, float]]:
        """重建 RRT 路徑

        Args:
            parents: 父節點映射
            nodes: 所有節點
            goal_idx: 目標節點索引

        Returns:
            路徑點列表
        """
        path = [nodes[goal_idx]]
        current = goal_idx

        while parents[current] is not None:
            current = parents[current]
            path.append(nodes[current])

        path.reverse()
        return path

    def _fallback_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Path:
        """失敗回退路徑"""
        waypoints = [start, goal]
        return Path(waypoints)
