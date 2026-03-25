"""AIT* 路徑視覺化器 (Isaac Lab 版本)

使用 Isaac Lab 原生 VisualizationMarkers API 來顯示 AIT* 規劃的路徑。
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class AITStarPathVisualizer:
    """AIT* 路徑視覺化器

    在 Isaac Sim 視口中繪製 AIT* 規劃的路徑，包含：
    - 路徑線條（連接各路徑點）
    - 路徑點（小球標記）
    - 起點標記（綠色大球）
    - 終點標記由 goal_command 管理（紅色箭頭）
    """

    def __init__(
        self,
        prim_path: str = "/World/AITStarPath",
        path_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        waypoint_radius: float = 0.15,  # 增大尺寸以便觀察
        line_radius: float = 0.08,     # 增大尺寸以便觀察
    ):
        """初始化視覺化器

        Args:
            prim_path: 標記物體的 prim 路徑
            path_color: 路徑顏色 (R, G, B)
            waypoint_radius: 路徑點球體半徑
            line_radius: 路徑線條粗細
        """
        # 路徑點標記配置（小球）
        self._waypoint_cfg = VisualizationMarkersCfg(
            prim_path=f"{prim_path}/waypoints",
            markers={
                "waypoint": sim_utils.SphereCfg(
                    radius=waypoint_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=path_color,
                    ),
                ),
            },
        )

        # 起點標記配置（綠色大球）
        self._start_cfg = VisualizationMarkersCfg(
            prim_path=f"{prim_path}/start",
            markers={
                "start": sim_utils.SphereCfg(
                    radius=waypoint_radius * 1.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                    ),
                ),
            },
        )

        # 創建標記物體（延遲創建，直到第一次 visualize 調用）
        self._waypoint_markers: Optional[VisualizationMarkers] = None
        self._start_marker: Optional[VisualizationMarkers] = None

        # 用於繪製線條的連接柱標記
        self._line_cfg = VisualizationMarkersCfg(
            prim_path=f"{prim_path}/lines",
            markers={
                "line": sim_utils.CylinderCfg(
                    radius=line_radius,
                    height=1.0,  # 初始高度，會通過 scale 動態調整
                    axis="X",  # X 軸對齊，方便水平旋轉
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=path_color,  # RGB only, no alpha
                        metallic=0.0,
                        roughness=0.5,
                    ),
                ),
            },
        )
        self._line_markers: Optional[VisualizationMarkers] = None

        self.prim_path = prim_path
        self._initialized = False

    def _ensure_initialized(self):
        """確保標記物體已創建"""
        if self._initialized:
            return

        self._waypoint_markers = VisualizationMarkers(self._waypoint_cfg)
        self._start_marker = VisualizationMarkers(self._start_cfg)
        self._line_markers = VisualizationMarkers(self._line_cfg)
        self._initialized = True

    def visualize(
        self,
        path_points: torch.Tensor | np.ndarray | List,
        start_pos: Optional[torch.Tensor | np.ndarray] = None,
        goal_pos: Optional[torch.Tensor | np.ndarray] = None,  # 保留參數以兼容舊代碼，但不使用
    ):
        """視覺化路徑

        Args:
            path_points: [N, 2] 或 [N, 3] 路徑點坐標（世界坐標系）
            start_pos: [2] 或 [3] 起點坐標（可選，顯示綠色大球）
            goal_pos: [2] 或 [3] 終點坐標（可選，保留參數以兼容，但不顯示）
                     （終點使用紅色箭頭標記，由 goal_command 管理）
        """
        self._ensure_initialized()

        # 轉換為 numpy
        if isinstance(path_points, torch.Tensor):
            path_points = path_points.detach().cpu().numpy()
        elif isinstance(path_points, list):
            path_points = np.array(path_points)

        # 確保是 3D 坐標
        if path_points.shape[1] == 2:
            # 添加 z 坐標（與目標箭頭一致，貼地面）
            path_points = np.concatenate([
                path_points,
                np.full((path_points.shape[0], 1), 0.1)  # z = 0.1m（略高於地面，避免穿模）
            ], axis=1)

        num_points = len(path_points)

        if num_points == 0:
            # 清空視覺化
            if self._waypoint_markers:
                self._waypoint_markers.visualize(translations=np.zeros((0, 3)))
            if self._line_markers:
                self._line_markers.visualize(translations=np.zeros((0, 3)))
            return

        # 如果路徑點太少（只有起點和終點），添加中間插值點以便視覺化
        if num_points == 2:
            # 在起點和終點之間插入中間點
            p1 = path_points[0]
            p2 = path_points[1]
            # 計算總距離
            total_distance = np.linalg.norm(p2 - p1)
            # 根據距離決定插值點數量
            num_interpolated = max(3, int(total_distance / 0.5))  # 每 0.5m 一個點
            interpolated_points = []
            for i in range(num_interpolated + 1):
                t = i / num_interpolated
                point = p1 + t * (p2 - p1)
                interpolated_points.append(point)
            path_points = np.array(interpolated_points)
            num_points = len(path_points)

        # 繪製路徑點
        self._waypoint_markers.visualize(translations=path_points)

        # 繪製連接線
        if num_points > 1:
            line_positions = []
            line_orientations = []
            line_scales = []

            for i in range(num_points - 1):
                p1 = path_points[i]
                p2 = path_points[i + 1]

                # 計算中點
                midpoint = (p1 + p2) / 2
                line_positions.append(midpoint)

                # 計算方向（將柱體旋轉對齊兩點）
                direction = p2 - p1
                distance = np.linalg.norm(direction[:2])  # 只看水平距離

                # 計算旋轉角度（繞 Z 軸的 yaw）
                # 圓柱的 X 軸對齊到方向向量
                yaw = np.arctan2(direction[1], direction[0])

                # 四元數 (w, x, y, z) - 只繞 Z 軸旋轉 yaw
                cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
                qw, qx, qy, qz = cy, 0.0, 0.0, sy  # 繞 Z 軸旋轉

                line_orientations.append([qw, qx, qy, qz])

                # 柱體 X 軸長度 = 距離，Y 和 Z 保持半徑
                # 轉換為 Python float 避免 USD 類型錯誤
                line_scales.append([float(distance), float(1.0), float(1.0)])

            if line_positions:
                self._line_markers.visualize(
                    translations=np.array(line_positions),
                    orientations=np.array(line_orientations),
                    scales=np.array(line_scales, dtype=np.float32),
                )

        # 繪製起點
        if start_pos is not None:
            if isinstance(start_pos, torch.Tensor):
                start_pos = start_pos.detach().cpu().numpy()
            if len(start_pos) == 2:
                start_pos = np.array([start_pos[0], start_pos[1], 0.3])  # 提高高度
            self._start_marker.visualize(translations=start_pos.reshape(1, 3))

        # 終點標記由 goal_command 管理（紅色箭頭），此處不再繪製

    def clear(self):
        """清除所有視覺化"""
        if self._waypoint_markers:
            self._waypoint_markers.visualize(translations=np.zeros((0, 3)))
        if self._line_markers:
            self._line_markers.visualize(translations=np.zeros((0, 3)))
        if self._start_marker:
            self._start_marker.visualize(translations=np.zeros((0, 3)))

    def set_visibility(self, visible: bool):
        """設置可見性"""
        if not self._initialized:
            return
        self._waypoint_markers.set_visibility(visible)
        self._line_markers.set_visibility(visible)
        self._start_marker.set_visibility(visible)


class MultiEnvPathVisualizer:
    """多環境路徑視覺化器

    為每個環境創建獨立的路徑視覺化。
    主要用於調試和演示，訓練時建議只視覺化第一個環境。
    """

    def __init__(
        self,
        num_envs: int,
        prim_path: str = "/World/AITStarPaths",
        max_envs_to_visualize: int = 1,  # 只視覺化前幾個環境
    ):
        """初始化多環境視覺化器

        Args:
            num_envs: 環境總數
            prim_path: 基礎 prim 路徑
            max_envs_to_visualize: 最多視覺化幾個環境（避免性能問題）
        """
        self.num_envs = num_envs
        self.max_envs_to_visualize = min(max_envs_to_visualize, num_envs)

        # 為每個要視覺化的環境創建視覺化器
        self.visualizers: list[AITStarPathVisualizer] = []
        for i in range(self.max_envs_to_visualize):
            env_prim_path = f"{prim_path}/env_{i}"
            self.visualizers.append(AITStarPathVisualizer(prim_path=env_prim_path))

    def visualize(
        self,
        env_idx: int,
        path_points: torch.Tensor | np.ndarray | List,
        start_pos: Optional[torch.Tensor | np.ndarray] = None,
        goal_pos: Optional[torch.Tensor | np.ndarray] = None,
    ):
        """視覺化指定環境的路徑

        Args:
            env_idx: 環境索引
            path_points: 路徑點
            start_pos: 起點位置
            goal_pos: 終點位置
        """
        if env_idx >= self.max_envs_to_visualize:
            return  # 不視覺化此環境

        self.visualizers[env_idx].visualize(path_points, start_pos, goal_pos)

    def visualize_all(
        self,
        paths: list[torch.Tensor] | torch.Tensor,
        starts: Optional[list[torch.Tensor]] = None,
        goals: Optional[list[torch.Tensor]] = None,
    ):
        """視覺化所有環境的路徑

        Args:
            paths: 路徑列表，每個元素是一個環境的路徑
            starts: 起點位置列表
            goals: 終點位置列表
        """
        for i in range(min(len(paths), self.max_envs_to_visualize)):
            start = starts[i] if starts and i < len(starts) else None
            goal = goals[i] if goals and i < len(goals) else None
            self.visualize(i, paths[i], start, goal)

    def clear(self):
        """清除所有視覺化"""
        for viz in self.visualizers:
            viz.clear()

    def set_visibility(self, visible: bool):
        """設置所有視覺化器的可見性"""
        for viz in self.visualizers:
            viz.set_visibility(visible)


__all__ = [
    "AITStarPathVisualizer",
    "MultiEnvPathVisualizer",
]
