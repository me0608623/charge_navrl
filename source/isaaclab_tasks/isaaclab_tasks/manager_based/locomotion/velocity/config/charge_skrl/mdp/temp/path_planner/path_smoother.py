"""
路徑平滑器 (Path Smoother)

提供多種路徑平滑算法，使 A* 生成的路徑更適合機器人跟隨。
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class PathSmootherCfg:
    """路徑平滑器配置類別

    Attributes:
        method: 平滑方法 ("chaikin", "bezier", "spline")
        iterations: 迭代次數（適用於 Chaikin）
        tension: 張力參數（適用於樣條）
        min_waypoint_distance: 最小路徑點間距（米）
    """
    method: str = "chaikin"  # chaikin, bezier, spline
    iterations: int = 3
    tension: float = 0.5
    min_waypoint_distance: float = 0.2


class PathSmoother:
    """路徑平滑器類別

    提供多種路徑平滑算法：
    - Chaikin's corner cutting: 快速平滑，保留整體形狀
    - Bezier 曲線: 生成平滑曲線
    - 樣條插值: 更平滑但計算量較大
    """

    def __init__(self, cfg: PathSmootherCfg = None):
        """初始化路徑平滑器

        Args:
            cfg: 平滑器配置
        """
        self.cfg = cfg or PathSmootherCfg()

    def smooth(
        self,
        path: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """平滑路徑

        Args:
            path: [N, 2] 原始路徑點
            method: 平滑方法（覆蓋配置）

        Returns:
            [M, 2] 平滑後的路徑點
        """
        if len(path) <= 2:
            return path.clone()

        method = method or self.cfg.method

        if method == "chaikin":
            return self._chaikin_smooth(path)
        elif method == "bezier":
            return self._bezier_smooth(path)
        elif method == "spline":
            return self._spline_smooth(path)
        else:
            print(f"[PathSmoother] 未知的方法: {method}，使用原始路徑")
            return path.clone()

    def _chaikin_smooth(
        self,
        path: torch.Tensor,
        iterations: Optional[int] = None,
    ) -> torch.Tensor:
        """Chaikin's corner cutting 算法

        通過反覆切割角點生成平滑路徑。

        Args:
            path: [N, 2] 原始路徑
            iterations: 迭代次數

        Returns:
            平滑後的路徑
        """
        iterations = iterations or self.cfg.iterations
        current = path.clone()

        for _ in range(iterations):
            if len(current) <= 2:
                break

            new_path = [current[0]]  # 保留起點

            for i in range(len(current) - 1):
                p0 = current[i]
                p1 = current[i + 1]

                # Chaikin 的切割點
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1

                new_path.append(q)
                new_path.append(r)

            new_path.append(current[-1])  # 保留終點
            current = torch.stack(new_path)

        return current

    def _bezier_smooth(
        self,
        path: torch.Tensor,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """Bezier 曲線平滑

        使用二次 Bezier 曲線連接路徑點。

        Args:
            path: [N, 2] 原始路徑
            num_points: 輸出路徑點數量

        Returns:
            平滑後的路徑
        """
        if len(path) <= 2:
            return path.clone()

        num_points = num_points or (len(path) * 2)
        result = []

        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]

            # 計算控制點（簡單版本使用兩點中點）
            if i < len(path) - 2:
                p2 = path[i + 2]
                control = p1
            else:
                control = p1

            # 生成 Bezier 曲線段
            segment_points = self._quadratic_bezier_segment(
                p0, control, p1,
                num_points=max(2, num_points // (len(path) - 1))
            )
            result.extend(segment_points[:-1])  # 避免重複端點

        result.append(path[-1])
        return torch.stack(result)

    def _quadratic_bezier_segment(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        num_points: int = 10,
    ) -> List[torch.Tensor]:
        """生成二次 Bezier 曲線段

        B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2

        Args:
            p0: 起點
            p1: 控制點
            p2: 終點
            num_points: 採樣點數量

        Returns:
            曲線點列表
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            points.append(point)
        return points

    def _spline_smooth(
        self,
        path: torch.Tensor,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """樣條曲線平滑

        使用 Catmull-Rom 樣條曲線。

        Args:
            path: [N, 2] 原始路徑
            num_points: 輸出路徑點數量

        Returns:
            平滑後的路徑
        """
        if len(path) <= 3:
            return path.clone()

        num_points = num_points or (len(path) * 3)
        result = [path[0]]

        # Catmull-Rom 樣條需要每個段的前後點
        for i in range(len(path) - 1):
            p0 = path[max(0, i - 1)]
            p1 = path[i]
            p2 = path[i + 1]
            p3 = path[min(len(path) - 1, i + 2)]

            segment_points = self._catmull_rom_segment(
                p0, p1, p2, p3,
                num_points=max(2, num_points // (len(path) - 1))
            )
            result.extend(segment_points[:-1])

        result.append(path[-1])
        return torch.stack(result)

    def _catmull_rom_segment(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        num_points: int = 10,
        tension: float = 0.5,
    ) -> List[torch.Tensor]:
        """生成 Catmull-Rom 樣條曲線段

        Args:
            p0, p1, p2, p3: 控制點
            num_points: 採樣點數量
            tension: 張力參數

        Returns:
            曲線點列表
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom 基函數
            s = (1 - tension) / 2

            c0 = -s * t3 + 2 * s * t2 - s * t
            c1 = (2 - s) * t3 + (s - 3) * t2 + 1
            c2 = (s - 2) * t3 + (3 - 2 * s) * t2 + s * t
            c3 = s * t3 - s * t2

            point = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3
            points.append(point)

        return points

    def subsample(
        self,
        path: torch.Tensor,
        min_distance: Optional[float] = None,
    ) -> torch.Tensor:
        """根據最小間距取樣路徑點

        Args:
            path: [N, 2] 輸入路徑
            min_distance: 最小間距（米）

        Returns:
            取樣後的路徑
        """
        min_distance = min_distance or self.cfg.min_waypoint_distance

        if len(path) <= 2:
            return path.clone()

        result = [path[0]]
        last_pos = path[0]

        for i in range(1, len(path)):
            if torch.norm(path[i] - last_pos) >= min_distance:
                result.append(path[i])
                last_pos = path[i]

        # 確保終點被包含
        if not torch.allclose(result[-1], path[-1]):
            result.append(path[-1])

        return torch.stack(result)


def create_smooth_trajectory(
    waypoints: torch.Tensor,
    num_points: int = 100,
    method: str = "spline",
) -> torch.Tensor:
    """從路徑點創建平滑軌跡

    Args:
        waypoints: [N, 2] 路徑點
        num_points: 輸出軌跡點數量
        method: 平滑方法

    Returns:
        [num_points, 2] 平滑軌跡
    """
    smoother = PathSmoother(PathSmootherCfg(method=method))
    smoothed = smoother.smooth(waypoints)

    # 插值到指定數量的點
    if len(smoothed) >= 2:
        # 計算累積路徑長度
        diffs = smoothed[1:] - smoothed[:-1]
        segment_lengths = torch.norm(diffs, dim=1)
        total_length = segment_lengths.sum()

        if total_length > 0:
            # 生成均勻採樣的參數
            t_values = torch.linspace(0, 1, num_points)

            # 沿著路徑插值
            result = []
            for t in t_values:
                # 找到對應的段
                target_dist = t * total_length
                current_dist = 0.0

                for i, seg_len in enumerate(segment_lengths):
                    if current_dist + seg_len >= target_dist:
                        # 在這一段內插值
                        local_t = (target_dist - current_dist) / seg_len
                        point = smoothed[i] + local_t * (smoothed[i + 1] - smoothed[i])
                        result.append(point)
                        break
                    current_dist += seg_len
                else:
                    # 超出範圍，使用最後一點
                    result.append(smoothed[-1])

            return torch.stack(result)

    return smoothed


__all__ = [
    "PathSmoother",
    "PathSmootherCfg",
    "create_smooth_trajectory",
]
