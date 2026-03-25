"""
Frenet Frame 座標轉換器

將 AIT* 全局路徑轉換為機器人局部座標系（Frenet Frame）。

Frenet Frame 定義：
- s (longitudinal): 沿路徑方向（切線）
- t (lateral): 垂直於路徑方向（法線）

這解決了 Global Frame vs Local Frame 的混亂問題。
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FrenetTransform:
    """Frenet Frame 座標轉換器

    將全局座標轉換為相對於路徑的 Frenet 座標：
    - longitudinal (s): 沿路徑方向的距離
    - lateral (t): 垂直路徑方向的距離
    - heading_diff: 機器人朝向與路徑方向的夾角
    """

    def __init__(self):
        """初始化 Frenet 轉換器"""
        self._cached_path: Optional[torch.Tensor] = None
        self._cached_tangents: Optional[torch.Tensor] = None
        self._path_lengths: Optional[torch.Tensor] = None

    def transform_to_frenet(
        self,
        global_points: torch.Tensor,
        path: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """將全局坐標點轉換為 Frenet 坐標

        Args:
            global_points: [N, 2] or [2] 全局坐標點
            path: [M, 2] AIT* 路徑點

        Returns:
            (longitudinal, lateral) Frenet 坐標
            - longitudinal: 沿路徑方向的距離
            - lateral: 垂直路徑方向的距離（左負右正）
        """
        if global_points.dim() == 1:
            global_points = global_points.unsqueeze(0)

        n_points = global_points.shape[0]
        device = global_points.device

        # 計算路徑切線和累積長度
        if not torch.is_tensor(self._cached_path) or not torch.equal(path, self._cached_path):
            self._compute_path_properties(path)

        s = torch.zeros(n_points, device=device)  # longitudinal
        t = torch.zeros(n_points, device=device)  # lateral

        for i in range(n_points):
            point = global_points[i]

            # 找到最近的路徑段
            min_dist = float('inf')
            best_s = 0.0
            best_t = 0.0

            for j in range(len(path) - 1):
                p1 = path[j]
                p2 = path[j + 1]

                # 路徑段向量
                segment = p2 - p1
                segment_length = torch.norm(segment)

                if segment_length < 1e-6:
                    continue

                # 切線方向
                tangent = segment / segment_length

                # 計算投影參數
                to_point = point - p1
                proj_length = torch.dot(to_point, tangent)

                if 0 <= proj_length <= segment_length:
                    # 投影點在線段上
                    projection = p1 + proj_length * tangent
                    lateral_dist = torch.norm(point - projection)

                    # 計算側向（使用叉積判斷左右）
                    cross = tangent[0] * (point[1] - projection[1]) - tangent[1] * (point[0] - projection[0])
                    lateral_signed = lateral_dist if cross > 0 else -lateral_dist

                    s_val = self._path_lengths[j].item() + proj_length.item()
                    dist = lateral_dist

                    if dist < min_dist:
                        min_dist = dist.item()
                        best_s = s_val
                        best_t = lateral_signed.item()
                else:
                    # 投影點在線段外，使用端點
                    if proj_length < 0:
                        s_val = self._path_lengths[j].item()
                        dist = torch.norm(point - p1)
                    else:
                        s_val = self._path_lengths[j + 1].item()
                        dist = torch.norm(point - p2)

                    if dist < min_dist:
                        min_dist = dist.item()
                        best_s = s_val
                        best_t = 0.0

            s[i] = best_s
            t[i] = best_t

        return s, t

    def transform_single_to_frenet(
        self,
        global_point: torch.Tensor,
        path: torch.Tensor,
    ) -> Tuple[float, float]:
        """轉換單個點到 Frenet 坐標

        Args:
            global_point: [2] 全局坐標點
            path: [M, 2] AIT* 路徑點

        Returns:
            (s, t) Frenet 坐標
        """
        s, t = self.transform_to_frenet(global_point, path)
        return s[0].item(), t[0].item()

    def _compute_path_properties(self, path: torch.Tensor):
        """計算路徑屬性（切線、累積長度）"""
        self._cached_path = path.clone()

        num_points = len(path)
        device = path.device

        # 計算累積路徑長度
        self._path_lengths = torch.zeros(num_points, device=device)
        for i in range(1, num_points):
            self._path_lengths[i] = self._path_lengths[i-1] + torch.norm(path[i] - path[i-1])

    def get_path_curvature(
        self,
        path: torch.Tensor,
        s_position: float,
    ) -> float:
        """計算路徑在給定位置的曲率

        Args:
            path: [M, 2] AIT* 路徑點
            s_position: 沿路徑的位置

        Returns:
            曲率值（1/半徑）
        """
        # 找到對應的路徑段
        for i in range(len(path) - 1):
            if self._path_lengths[i] <= s_position <= self._path_lengths[i + 1]:
                # 使用三點計算曲率
                idx = max(1, min(i, len(path) - 2))
                p_prev = path[idx - 1]
                p_curr = path[idx]
                p_next = path[idx + 1]

                # 使用 Menger 曲率公式
                # κ = 4 * Area / (a * b * c)
                a = torch.norm(p_curr - p_prev)
                b = torch.norm(p_next - p_curr)
                c = torch.norm(p_next - p_prev)

                # 三角形面積（叉積）
                area = 0.5 * abs(
                    (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1]) -
                    (p_curr[1] - p_prev[1]) * (p_next[0] - p_prev[0])
                )

                if a * b * c < 1e-6:
                    return 0.0

                curvature = 4 * area / (a * b * c)
                return curvature.item()

        return 0.0


def robot_frenet_observations(
    env: ManagerBasedRLEnv,
    path: torch.Tensor,
    robot_cfg: str = "robot",
) -> dict[str, torch.Tensor]:
    """計算機器人在 Frenet Frame 中的觀測

    這是 RL 模型應該接收的輸入格式！

    Args:
        env: 環境實例
        path: [M, 2] AIT* 路徑點
        robot_cfg: 機器人配置名稱

    Returns:
        字典包含：
        - 's': [num_envs] longitudinal 位置（沿路徑距離）
        - 't': [num_envs] lateral 偏移（左負右正）
        - 'heading_error': [num_envs] 朝向誤差（弧度）
        - 'curvature': [num_envs] 前方路徑曲率
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_yaw = robot.data.root_quat_w

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    # 計算機器人朝向（yaw）
    w, x, y, z = robot_yaw[:, 0], robot_yaw[:, 1], robot_yaw[:, 2], robot_yaw[:, 3]
    robot_heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 初始化 Frenet 轉換器
    transform = FrenetTransform()

    # 計算每個機器人的 Frenet 坐標
    s = torch.zeros(num_envs, device=device)
    t = torch.zeros(num_envs, device=device)
    heading_error = torch.zeros(num_envs, device=device)
    curvature = torch.zeros(num_envs, device=device)

    for env_id in range(num_envs):
        pos = robot_pos[env_id]
        heading = robot_heading[env_id]

        # 轉換到 Frenet 坐標
        s_val, t_val = transform.transform_single_to_frenet(pos, path)
        s[env_id] = s_val
        t[env_id] = t_val

        # 找到對應的路徑段方向
        for i in range(len(path) - 1):
            if transform._path_lengths[i] <= s_val <= transform._path_lengths[i + 1]:
                path_tangent = path[i + 1] - path[i]
                path_heading = torch.atan2(path_tangent[1], path_tangent[0])
                heading_error[env_id] = heading - path_heading
                # 歸一化到 [-π, π]
                heading_error[env_id] = torch.atan2(
                    torch.sin(heading_error[env_id]),
                    torch.cos(heading_error[env_id])
                ).item()
                break

        # 計算前方曲率
        curvature[env_id] = transform.get_path_curvature(path, s_val + 1.0)

    return {
        's': s,
        't': t,
        'heading_error': heading_error,
        'curvature': curvature,
    }


def frenet_to_global(
    s: float,
    t: float,
    path: torch.Tensor,
) -> torch.Tensor:
    """將 Frenet 坐標轉換回全局坐標

    Args:
        s: longitudinal 坐標
        t: lateral 坐標
        path: [M, 2] AIT* 路徑點

    Returns:
        [2] 全局坐標點
    """
    transform = FrenetTransform()
    transform._compute_path_properties(path)

    # 找到 s 對應的路徑位置
    for i in range(len(path) - 1):
        if transform._path_lengths[i] <= s <= transform._path_lengths[i + 1]:
            # 找到線段上的對應點
            segment_length = (transform._path_lengths[i + 1] - transform._path_lengths[i]).item()
            if segment_length < 1e-6:
                base_point = path[i]
                tangent = torch.zeros(2)
            else:
                progress = (s - transform._path_lengths[i].item()) / segment_length
                base_point = path[i] + progress * (path[i + 1] - path[i])
                tangent = (path[i + 1] - path[i]) / segment_length

            # 計算法線方向（切線旋轉90度）
            normal = torch.tensor([-tangent[1], tangent[0]])

            # base_point + t * normal
            global_point = base_point + t * normal
            return global_point

    # 如果超出範圍，使用最近端點
    if s <= transform._path_lengths[0]:
        return path[0]
    else:
        return path[-1]


__all__ = [
    "FrenetTransform",
    "robot_frenet_observations",
    "frenet_to_global",
]
