"""分層式導航管理器

結合全局規劃器、局部 RL 模型和安全過濾器，
實現完整的分層式導航架構。
"""

from __future__ import annotations

import math
import torch
from typing import List, Optional, Tuple

from .types import (
    NavigationState,
    Obstacle,
    GridMap,
    SafetyAction,
)
from .global_planner import GlobalPlanner, AStarPlanner, RRTStarPlanner
from .safety_shield import SafetyShield, RuleBasedShield, CBFShield


# ============================================================================
# 分層式導航管理器
# ============================================================================

class HierarchicalNavigationManager:
    """分層式導航管理器

    協調全局規劃器、局部 RL 模型和安全過濾器，
    實現從起點到終點的安全導航。
    """

    def __init__(
        self,
        global_planner: GlobalPlanner,
        safety_shield: SafetyShield,
        sub_goal_distance: float = 3.0,
        sub_goal_tolerance: float = 0.5,
    ):
        """初始化分層式導航管理器

        Args:
            global_planner: 全局路徑規劃器
            safety_shield: 安全過濾器
            sub_goal_distance: sub-goal 與 sub-goal 之間的距離（米）
            sub_goal_tolerance: sub-goal 到達容忍度（米）
        """
        self.global_planner = global_planner
        self.safety_shield = safety_shield
        self.sub_goal_distance = sub_goal_distance
        self.sub_goal_tolerance = sub_goal_tolerance

        # 導航狀態
        self.state: Optional[NavigationState] = None
        self.current_sub_goal: Optional[Tuple[float, float]] = None

    def reset(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
    ) -> NavigationState:
        """重置導航狀態

        Args:
            start: 起點位置（世界坐標）
            goal: 終點位置（世界坐標）
            obstacles: 障礙物列表

        Returns:
            導航狀態
        """
        # 使用全局規劃器規劃路徑
        path = self.global_planner.plan(start, goal, obstacles)

        # 初始化導航狀態
        self.state = NavigationState.initial(goal_position=goal)
        self.state.current_path = path

        # 設置第一個 sub-goal（路徑的起點）
        if len(path.waypoints) > 0:
            self.current_sub_goal = path.waypoints[0].position

        return self.state

    def update_sub_goal(
        self,
        robot_pos: Tuple[float, float],
    ) -> bool:
        """更新 sub-goal

        檢查當前 sub-goal 是否到達，如果是則前進到下一個。

        Args:
            robot_pos: 機器人位置

        Returns:
            sub-goal 是否更新
        """
        if self.state is None or self.state.current_path is None:
            return False

        # 檢查是否到達當前 sub-goal
        if self.current_sub_goal is not None:
            dist = math.sqrt(
                (robot_pos[0] - self.current_sub_goal[0]) ** 2
                + (robot_pos[1] - self.current_sub_goal[1]) ** 2
            )

            if dist < self.sub_goal_tolerance:
                # 到達當前 sub-goal，前進到下一個
                self.current_sub_goal = self._get_next_sub_goal(robot_pos)
                return True

        return False

    def _get_next_sub_goal(
        self,
        robot_pos: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """獲取下一個 sub-goal

        從路徑中選擇下一個合適的 sub-goal。

        Args:
            robot_pos: 機器人位置

        Returns:
            下一個 sub-goal 位置
        """
        if self.state is None or self.state.current_path is None:
            return None

        path = self.state.current_path

        # 找到路徑中距離機器人大於 sub_goal_distance 的第一個航點
        for i in range(path.current_index, len(path.waypoints)):
            waypoint_pos = path.waypoints[i].position
            dist = math.sqrt(
                (waypoint_pos[0] - robot_pos[0]) ** 2
                + (waypoint_pos[1] - robot_pos[1]) ** 2
            )

            if dist > self.sub_goal_distance:
                return waypoint_pos

        # 如果沒有找到距離大於 sub_goal_distance 的航點，使用終點
        if len(path.waypoints) > 0:
            return path.waypoints[-1].position

        return None

    def get_sub_goal_relative(
        self,
        robot_pos: Tuple[float, float],
        robot_yaw: float,
    ) -> Tuple[float, float]:
        """獲取 sub-goal 的相對位置（機器人坐標系）

        Args:
            robot_pos: 機器人位置（世界坐標）
            robot_yaw: 機器人偏航角（弧度）

        Returns:
            sub-goal 的相對位置 [x, y]
        """
        if self.current_sub_goal is None:
            return (0.0, 0.0)

        # 計算世界坐標系下的相對位置
        dx = self.current_sub_goal[0] - robot_pos[0]
        dy = self.current_sub_goal[1] - robot_pos[1]

        # 旋轉到機器人坐標系
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        rx = dx * cos_yaw + dy * sin_yaw
        ry = -dx * sin_yaw + dy * cos_yaw

        return (rx, ry)

    def filter_action(
        self,
        rl_action: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        lidar_scan: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """過濾 RL 動作

        使用安全過濾器檢查並修改 RL 輸出的動作。

        Args:
            rl_action: RL 輸出的動作 [linear_speed, angular_speed]
            robot_pos: 機器人位置
            robot_vel: 機器人速度
            lidar_scan: Lidar 掃描數據
            obstacles: 障礙物列表

        Returns:
            過濾後的安全動作
        """
        return self.safety_shield.filter_action(
            rl_action, robot_pos, robot_vel, lidar_scan, obstacles
        )

    def is_complete(self) -> bool:
        """檢查導航是否完成

        Returns:
            是否到達終點
        """
        if self.state is None or self.state.current_path is None:
            return False

        return self.state.current_path.is_complete()

    def get_progress(self) -> float:
        """獲取導航進度

        Returns:
            進度百分比 [0, 1]
        """
        if self.state is None or self.state.current_path is None:
            return 0.0

        total_length = self.state.current_path.total_length
        remaining_length = self.state.current_path.get_remaining_length()

        if total_length <= 0:
            return 1.0

        return 1.0 - (remaining_length / total_length)


# ============================================================================
# 工廠函數：創建常用的分層式導航系統
# ============================================================================

def create_astar_hierarchical_navigation(
    grid_resolution: float = 0.1,
    grid_size_x: int = 100,
    grid_size_y: int = 100,
    grid_origin: Tuple[float, float] = (-5.0, -5.0),
    shield_type: str = "rule",  # "rule", "cbf", "composite"
    sub_goal_distance: float = 3.0,
) -> HierarchicalNavigationManager:
    """創建基於 A* 的分層式導航系統

    Args:
        grid_resolution: 網格解析度（米/格子）
        grid_size_x: x 方向格子數
        grid_size_y: y 方向格子數
        grid_origin: 地圖原點（世界坐標）
        shield_type: 安全過濾器類型
        sub_goal_distance: sub-goal 距離

    Returns:
        分層式導航管理器
    """
    # 創建網格地圖
    grid_map = GridMap(
        resolution=grid_resolution,
        size_x=grid_size_x,
        size_y=grid_size_y,
        origin=grid_origin,
    )

    # 創建全局規劃器
    global_planner = AStarPlanner(grid_map=grid_map, diagonal_movement=True)

    # 創建安全過濾器
    if shield_type == "rule":
        safety_shield = RuleBasedShield()
    elif shield_type == "cbf":
        safety_shield = CBFShield()
    elif shield_type == "composite":
        from .safety_shield import CompositeShield
        safety_shield = CompositeShield()
    else:
        raise ValueError(f"未知的 shield_type: {shield_type}")

    # 創建分層式導航管理器
    return HierarchicalNavigationManager(
        global_planner=global_planner,
        safety_shield=safety_shield,
        sub_goal_distance=sub_goal_distance,
    )


def create_rrtstar_hierarchical_navigation(
    bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
    max_iterations: int = 10000,
    step_size: float = 0.5,
    shield_type: str = "rule",  # "rule", "cbf", "composite"
    sub_goal_distance: float = 3.0,
) -> HierarchicalNavigationManager:
    """創建基於 RRT* 的分層式導航系統

    Args:
        bounds: 空間邊界 [x_min, x_max, y_min, y_max]
        max_iterations: 最大迭代次數
        step_size: 擴展步長
        shield_type: 安全過濾器類型
        sub_goal_distance: sub-goal 距離

    Returns:
        分層式導航管理器
    """
    # 創建全局規劃器
    global_planner = RRTStarPlanner(
        bounds=bounds,
        max_iterations=max_iterations,
        step_size=step_size,
    )

    # 創建安全過濾器
    if shield_type == "rule":
        safety_shield = RuleBasedShield()
    elif shield_type == "cbf":
        safety_shield = CBFShield()
    elif shield_type == "composite":
        from .safety_shield import CompositeShield
        safety_shield = CompositeShield()
    else:
        raise ValueError(f"未知的 shield_type: {shield_type}")

    # 創建分層式導航管理器
    return HierarchicalNavigationManager(
        global_planner=global_planner,
        safety_shield=safety_shield,
        sub_goal_distance=sub_goal_distance,
    )
