"""
虛擬規劃器 (Virtual/Fake Planner) - 用於訓練階段

====================================================================
Sim-to-Real 最佳實踐：訓練時移除 AIT*，使用隨機局部目標
====================================================================

設計理念：
---------
1. FPS is King: 訓練時不調用昂貴的 AIT* 算法
2. 觀測一致性: 輸出格式與 AIT* 完全相同，確保 Sim-to-Real 遷移
3. 局部導航訓練: RL 專注於「如何到達局部目標」而非「如何規劃路徑」

數據流：
---------
訓練時: VirtualPlanner → 隨機局部目標 → RL 學會局部導航
推理時: AIT* → 全局路徑航點 → RL 使用相同的局部導航技能

Phase 策略：
-----------
- Phase 0: 空地環境，扇形區域隨機（模擬直線跟隨）
- Phase 1: 有邊界牆，確保目標在室內
- Phase 2: 有障礙物，避開障礙物重新採樣
- Phase 3: 窄門場景，目標設在門另一端
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VirtualPlanner:
    """虛擬規劃器 - 訓練時使用隨機目標點代替 AIT*

    核心功能：
    1. _sample_random_local_goal(): 在機器人周圍生成隨機局部目標
    2. plan_path(): 返回與 AIT* 相同格式的路徑 [start, waypoint, goal]
    3. 障礙物檢測: Phase 2+ 支持重新採樣避開障礙物

    使用方式：
        # 訓練時 (use_virtual_planner=True)
        planner = VirtualPlanner(phase="phase2", env_map=environment_map)
        path = planner.plan_path(start, goal, env)  # [N, 2] 路徑點

        # 觀測函數完全一致
        local_goal = extract_local_goal(path)  # 與 AIT* 相同
    """

    def __init__(
        self,
        phase: str = "phase0",
        min_distance: float = 3.0,   # 🔥 最小目標距離 3m (用戶需求)
        max_distance: float = 6.0,   # 🔥 最大目標距離 6m (用戶需求)
        map_size: Tuple[float, float] = (16.0, 16.0),
        env_map: Optional[object] = None,  # 🔥 環境地圖，用於障礙物檢測
    ):
        """初始化虛擬規劃器

        Args:
            phase: 當前訓練階段 ("phase0", "phase1", "phase2", "phase3")
            min_distance: 目標點的最小距離 (米)，默認 3m
            max_distance: 目標點的最大距離 (米)，默認 6m
            map_size: 地圖大小 [width, depth] (米)
            env_map: EnvironmentMap 實例，用於障礙物檢測 (Phase 2+)
        """
        self.phase = phase
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.map_size = map_size
        self.env_map = env_map

        # 採樣參數
        self.max_resample_attempts = 20  # 最大重新採樣次數
        self.safety_margin = 0.3  # 安全邊距 (米)

    def _sample_random_local_goal(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """🔥 核心函數：在機器人周圍生成隨機局部目標

        這是與 AIT* 的關鍵區別：
        - AIT*: 全局路徑規劃，考慮障礙物和最優路徑
        - VirtualPlanner: 隨機採樣，模擬「某個方向上的航點」

        Args:
            robot_pos: [2] 機器人位置 (局部坐標)
            goal_pos: [2] 終點位置 (局部坐標)

        Returns:
            [2] 隨機局部目標位置 (局部坐標)
        """
        # 計算朝向終點的大致方向
        direction_to_goal = goal_pos - robot_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)

        # 如果已經到達目標附近
        if distance_to_goal < 0.5:
            return goal_pos

        # 歸一化方向
        if distance_to_goal > 1e-6:
            unit_direction = direction_to_goal / distance_to_goal
        else:
            unit_direction = np.array([1.0, 0.0])

        # 🔥 隨機採樣距離：在 [min_distance, max_distance] 範圍內
        # 但不超過到目標的總距離
        sample_distance = np.random.uniform(
            self.min_distance,
            min(self.max_distance, distance_to_goal * 0.8)  # 不超過 80% 目標距離
        )

        # 🔥 隨機角度偏移：模擬路徑跟隨時的航點偏移
        # Phase 0: ±45 度 (空地，可以大角度)
        # Phase 1-3: ±30 度 (有障礙物，角度小一點)
        max_angle_offset = np.radians(45 if self.phase == "phase0" else 30)
        angle_offset = np.random.uniform(-max_angle_offset, max_angle_offset)

        # 旋轉方向向量
        cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
        rotated_direction = np.array([
            unit_direction[0] * cos_a - unit_direction[1] * sin_a,
            unit_direction[0] * sin_a + unit_direction[1] * cos_a,
        ])

        # 計算採樣點位置
        sampled_point = robot_pos + rotated_direction * sample_distance

        return sampled_point

    def _check_obstacle_collision(
        self,
        point: np.ndarray,
    ) -> bool:
        """檢查點是否在障礙物內部

        Args:
            point: [2] 要檢查的位置 (局部坐標)

        Returns:
            True 如果在障礙物內，False 否則
        """
        if self.env_map is None:
            return False

        # 轉換為 world 坐標 (env_map 使用世界坐標系)
        # 注意：局部坐標系原點在 (0,0)，world 坐標系需要考慮 map_origin
        world_pos = torch.from_numpy(point).float()

        return self.env_map.is_occupied_world(world_pos)

    def _resample_if_obstacle(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """如果採樣點在障礙物內，重新採樣

        Phase 2+ 使用：確保目標不在障礙物內

        Args:
            robot_pos: [2] 機器人位置
            goal_pos: [2] 終點位置

        Returns:
            [2] 合法的局部目標位置
        """
        for attempt in range(self.max_resample_attempts):
            waypoint = self._sample_random_local_goal(robot_pos, goal_pos)

            # 檢查是否在障礙物內
            if not self._check_obstacle_collision(waypoint):
                return waypoint

        # 如果多次嘗試都失敗，返回朝向目標的點
        direction = goal_pos - robot_pos
        distance = np.linalg.norm(direction)
        safe_distance = min(self.max_distance, distance)
        return robot_pos + (direction / distance) * safe_distance

    def _generate_phase0_waypoint(
        self,
        robot: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Phase 0: 空地環境，扇形區域隨機"""
        return self._sample_random_local_goal(robot, goal)

    def _generate_phase1_waypoint(
        self,
        robot: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Phase 1: 有邊界牆，確保目標在地圖範圍內"""
        waypoint = self._sample_random_local_goal(robot, goal)

        # 檢查邊界 (16x16m 地圖，範圍 -8 ~ 8)
        half_w, half_d = self.map_size[0] / 2, self.map_size[1] / 2
        margin = 0.5  # 距離牆壁至少 0.5m

        waypoint[0] = np.clip(waypoint[0], -half_w + margin, half_w - margin)
        waypoint[1] = np.clip(waypoint[1], -half_d + margin, half_d - margin)

        return waypoint

    def _generate_phase2_waypoint(
        self,
        robot: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Phase 2: 有障礙物，避開障礙物重新採樣"""
        return self._resample_if_obstacle(robot, goal)

    def _generate_phase3_waypoint(
        self,
        robot: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Phase 3: 窄門場景，目標設在門另一端

        TODO: 實現窄門感知邏輯
        目前先用 Phase 2 的障礙物避讓策略
        """
        return self._resample_if_obstacle(robot, goal)

    def generate_virtual_waypoint(
        self,
        robot_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        env: Optional[ManagerBasedRLEnv] = None,
    ) -> torch.Tensor:
        """生成虛擬航點（與 AIT* 輸出格式完全一致）

        🔥 關鍵：這是與 AIT* 的接口兼容點
        AIT* 輸出: [N, 2] 路徑點
        VirtualPlanner 輸出: [3, 2] 路徑點 = [start, waypoint, goal]

        Args:
            robot_pos: [2] 或 [num_envs, 2] 機器人位置
            goal_pos: [2] 或 [num_envs, 2] 終點位置
            env: 環境實例（可選，用於動態獲取地圖）

        Returns:
            [2] 或 [num_envs, 2] 虛擬航點位置
        """
        # 確保形狀正確
        if robot_pos.dim() == 1:
            robot_pos = robot_pos.unsqueeze(0)
            goal_pos = goal_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_envs = robot_pos.shape[0]
        device = robot_pos.device

        virtual_waypoints = []

        for i in range(num_envs):
            robot = robot_pos[i].cpu().numpy()
            goal = goal_pos[i].cpu().numpy()

            # 根據 Phase 使用不同的生成策略
            if self.phase == "phase0":
                waypoint = self._generate_phase0_waypoint(robot, goal)
            elif self.phase == "phase1":
                waypoint = self._generate_phase1_waypoint(robot, goal)
            elif self.phase == "phase2":
                waypoint = self._generate_phase2_waypoint(robot, goal)
            elif self.phase == "phase3":
                waypoint = self._generate_phase3_waypoint(robot, goal)
            else:
                # 默認：簡單的扇形區域隨機
                waypoint = self._generate_phase0_waypoint(robot, goal)

            virtual_waypoints.append(waypoint)

        # 轉換為 tensor
        result = torch.from_numpy(np.array(virtual_waypoints)).float().to(device)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def plan_path(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        env: Optional[ManagerBasedRLEnv] = None,
    ) -> torch.Tensor:
        """生成虛擬路徑（🔥 與 AIT* 接口完全兼容）

        這是關鍵接口！確保：
        1. 輸入格式與 AIT* 相同
        2. 輸出格式與 AIT* 相同
        3. 觀測函數使用時完全一致

        AIT* 輸出: [N, 2] 路徑點 (局部坐標)
        VirtualPlanner 輸出: [3, 2] = [start, virtual_waypoint, goal]

        Args:
            start_pos: [2] or [num_envs, 2] 起點位置 (局部坐標)
            goal_pos: [2] or [num_envs, 2] 終點位置 (局部坐標)
            env: 環境實例（可選，用於動態獲取地圖）

        Returns:
            [num_waypoints, 2] 路徑點坐標 (局部坐標)
            對於虛擬規劃器，返回 [start, virtual_waypoint, goal]
        """
        # 🔥 動態獲取環境地圖（如果提供了 env）
        if env is not None and hasattr(env, "_aitstar_planner"):
            if hasattr(env._aitstar_planner, "map"):
                self.env_map = env._aitstar_planner.map

        # 確保形狀正確
        if start_pos.dim() == 1:
            start_pos = start_pos.unsqueeze(0)
            goal_pos = goal_pos.unsqueeze(0)

        num_envs = start_pos.shape[0]
        device = start_pos.device

        paths = []

        for i in range(num_envs):
            start = start_pos[i]
            goal = goal_pos[i]

            # 生成虛擬航點
            virtual_waypoint = self.generate_virtual_waypoint(start, goal, env)

            # 🔥 構建路徑：起點 -> 虛擬航點 -> 終點
            # 這與 AIT* 的輸出格式完全一致
            path = torch.stack([start, virtual_waypoint, goal])
            paths.append(path)

        # 對於單個環境，返回 [3, 2] 的路徑
        if num_envs == 1:
            return paths[0]

        # 對於多個環境，返回第一個環境的路徑
        # （AIT* 也只處理單個環境）
        return paths[0]


__all__ = ["VirtualPlanner"]
