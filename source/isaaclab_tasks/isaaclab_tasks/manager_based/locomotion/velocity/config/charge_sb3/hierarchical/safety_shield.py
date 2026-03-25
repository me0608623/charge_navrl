"""安全過濾器模組

實現分層式導航的底層：安全過濾器
支持基於規則的安全過濾和控制屏障函數 (CBF)
"""

from __future__ import annotations

import math
import torch
from typing import List, Tuple, Optional

from .types import SafetyAction, Obstacle


# ============================================================================
# 基礎安全過濾器類
# ============================================================================

class SafetyShield:
    """安全過濾器基類

    負責檢查和過濾 RL 算法輸出的動作，確保不會導致碰撞。
    """

    def filter_action(
        self,
        action: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        lidar_scan: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """過濾動作

        Args:
            action: RL 輸出的動作 [linear_speed, angular_speed]
            robot_pos: 機器人位置 [x, y]
            robot_vel: 機器人速度 [vx, vy]
            lidar_scan: Lidar 掃描數據
            obstacles: 障礙物列表

        Returns:
            過濾後的安全動作
        """
        raise NotImplementedError("子類必須實現 filter_action 方法")


# ============================================================================
# 基於規則的安全過濾器
# ============================================================================

class RuleBasedShield(SafetyShield):
    """基於規則的安全過濾器

    使用簡單的規則來判斷動作是否安全：
    1. Lidar 障值判斷
    2. 預測碰撞
    3. 緊急煞車
    """

    def __init__(
        self,
        safety_distance: float = 0.8,
        danger_distance: float = 0.5,
        emergency_distance: float = 0.3,
        max_linear_speed: float = 1.5,
        max_angular_speed: float = 1.5,
    ):
        """初始化基於規則的安全過濾器

        Args:
            safety_distance: 安全距離（米）
            danger_distance: 危險距離（米）
            emergency_distance: 緊急距離（米）
            max_linear_speed: 最大線性速度
            max_angular_speed: 最大角速度
        """
        self.safety_distance = safety_distance
        self.danger_distance = danger_distance
        self.emergency_distance = emergency_distance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    def filter_action(
        self,
        action: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        lidar_scan: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """基於規則過濾動作

        Args:
            action: RL 輸出的動作 [linear_speed, angular_speed]
            robot_pos: 機器人位置 [x, y]
            robot_vel: 機器人速度 [vx, vy]
            lidar_scan: Lidar 掃描數據（已歸一化，值越大表示越近）
            obstacles: 障礙物列表

        Returns:
            過濾後的安全動作
        """
        linear_speed, angular_speed = action.tolist()

        # 檢查 Lidar 數據中的最近障礙物
        # 假設 lidar_scan 是已歸一化的，值越大表示越近
        min_lidar_dist = lidar_scan.max().item() if lidar_scan.numel() > 0 else 0.0

        # 將歸一化的 Lidar 距離轉換為實際距離
        # 假設 Lidar 範圍是 0-10 米，歸一化到 0-1
        actual_min_dist = min_lidar_dist * 10.0

        # 檢查安全狀態
        if actual_min_dist < self.emergency_distance:
            # 緊急情況：強制停止
            return SafetyAction.unsafe(
                linear_speed=0.0,
                angular_speed=0.0,
                reason=f"緊急停止：障礙物距離 {actual_min_dist:.2f}m < {self.emergency_distance}m"
            )

        if actual_min_dist < self.danger_distance:
            # 危險情況：大幅降低速度
            speed_scale = (actual_min_dist - self.emergency_distance) / (
                self.danger_distance - self.emergency_distance
            )
            speed_scale = max(0.0, min(1.0, speed_scale))

            linear_speed *= speed_scale * 0.5  # 大幅降低速度
            angular_speed *= 0.5  # 降低轉向速度

            return SafetyAction.unsafe(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                reason=f"危險警告：障礙物距離 {actual_min_dist:.2f}m，速度縮放 {speed_scale:.2f}"
            )

        if actual_min_dist < self.safety_distance:
            # 安全警告：輕微降低速度
            speed_scale = (actual_min_dist - self.danger_distance) / (
                self.safety_distance - self.danger_distance
            )
            speed_scale = max(0.0, min(1.0, speed_scale))

            linear_speed *= speed_scale

            return SafetyAction.unsafe(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
                reason=f"安全警告：障礙物距離 {actual_min_dist:.2f}m，速度縮放 {speed_scale:.2f}"
            )

        # 預測碰撞（0.5 秒後）
        if self._will_collide(
            robot_pos,
            robot_vel,
            action,
            obstacles,
            prediction_horizon=0.5,
        ):
            # 預測會碰撞：停止
            return SafetyAction.unsafe(
                linear_speed=0.0,
                angular_speed=0.0,
                reason="預測 0.5 秒後會碰撞"
            )

        # 動作是安全的
        return SafetyAction.safe_action(
            linear_speed=linear_speed,
            angular_speed=angular_speed,
        )

    def _will_collide(
        self,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        action: torch.Tensor,
        obstacles: List[Obstacle],
        prediction_horizon: float = 0.5,
    ) -> bool:
        """預測是否會碰撞

        Args:
            robot_pos: 機器人位置 [x, y]
            robot_vel: 機器人速度 [vx, vy]
            action: 動作 [linear_speed, angular_speed]
            obstacles: 障礙物列表
            prediction_horizon: 預測時間範圍（秒）

        Returns:
            是否會碰撞
        """
        linear_speed, angular_speed = action.tolist()
        vx, vy = robot_vel.tolist()

        # 預測 0.5 秒後的位置
        # 簡化假設：速度不變，忽略角速度的影響
        future_x = robot_pos[0].item() + vx * prediction_horizon
        future_y = robot_pos[1].item() + vy * prediction_horizon

        # 檢查是否會與任何障礙物碰撞
        for obs in obstacles:
            if obs.is_collision((future_x, future_y), robot_radius=0.5):
                return True

        return False


# ============================================================================
# 控制屏障函數 (CBF) 安全過濾器
# ============================================================================

class CBFShield(SafetyShield):
    """控制屏障函數 (Control Barrier Function) 安全過濾器

    使用 CBF 來確保系統狀態始終保持在安全集合內。
    CBF 原理：如果 h(x) >= 0 表示安全，則需要滿足 Lf h(x) + α h(x) >= 0
    其中 Lf h(x) 是李導數，α > 0 是 Class-K 函數。
    """

    def __init__(
        self,
        safety_distance: float = 0.8,
        alpha: float = 1.0,
        max_linear_speed: float = 1.5,
        max_angular_speed: float = 1.5,
    ):
        """初始化 CBF 安全過濾器

        Args:
            safety_distance: 安全距離（米）
            alpha: CBF 參數，控制收斂速度
            max_linear_speed: 最大線性速度
            max_angular_speed: 最大角速度
        """
        self.safety_distance = safety_distance
        self.alpha = alpha
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    def filter_action(
        self,
        action: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        lidar_scan: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """使用 CBF 過濾動作

        Args:
            action: RL 輸出的動作 [linear_speed, angular_speed]
            robot_pos: 機器人位置 [x, y]
            robot_vel: 機器人速度 [vx, vy]
            lidar_scan: Lidar 掃描數據
            obstacles: 障礙物列表

        Returns:
            過濾後的安全動作
        """
        linear_speed, angular_speed = action.tolist()

        # 找到最近的障礙物
        nearest_dist, nearest_obstacle = self._find_nearest_obstacle(robot_pos, obstacles)

        if nearest_obstacle is None:
            # 沒有障礙物，動作是安全的
            return SafetyAction.safe_action(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
            )

        # 計算 CBF 約束
        h = self._compute_barrier_function(robot_pos, nearest_obstacle)
        Lf_h = self._compute_lie_derivative(robot_pos, robot_vel, nearest_obstacle)

        # 檢查約束是否滿足
        if h >= 0:
            # 已經在安全區域內
            return SafetyAction.safe_action(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
            )

        # 計算安全控制約束
        constraint = self.alpha * h - Lf_h

        if constraint >= 0:
            # 約束滿足，動作是安全的
            return SafetyAction.safe_action(
                linear_speed=linear_speed,
                angular_speed=angular_speed,
            )

        # 約束不滿足，需要修改動作
        # 簡化處理：降低速度
        speed_scale = max(0.0, min(1.0, -constraint / 10.0))
        linear_speed *= speed_scale
        angular_speed *= speed_scale

        return SafetyAction.unsafe(
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            reason=f"CBF 約束不滿足：h={h:.3f}, Lf_h={Lf_h:.3f}, constraint={constraint:.3f}",
        )

    def _find_nearest_obstacle(
        self,
        robot_pos: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> Tuple[float, Optional[Obstacle]]:
        """找到最近的障礙物

        Args:
            robot_pos: 機器人位置
            obstacles: 障礙物列表

        Returns:
            (最近距離, 最近障礙物)
        """
        if not obstacles:
            return float("inf"), None

        min_dist = float("inf")
        nearest_obs = None

        for obs in obstacles:
            dist = math.sqrt(
                (obs.position[0] - robot_pos[0].item()) ** 2
                + (obs.position[1] - robot_pos[1].item()) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        return min_dist, nearest_obs

    def _compute_barrier_function(
        self,
        robot_pos: torch.Tensor,
        obstacle: Obstacle,
    ) -> float:
        """計算屏障函數 h(x)

        Args:
            robot_pos: 機器人位置
            obstacle: 障礙物

        Returns:
            屏障函數值（h >= 0 表示安全）
        """
        # 簡化：使用距離作為屏障函數
        # h(x) = d(x, obstacle) - safety_distance
        dist = math.sqrt(
            (obstacle.position[0] - robot_pos[0].item()) ** 2
            + (obstacle.position[1] - robot_pos[1].item()) ** 2
        )

        return dist - self.safety_distance

    def _compute_lie_derivative(
        self,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        obstacle: Obstacle,
    ) -> float:
        """計算李導數 Lf h(x)

        Args:
            robot_pos: 機器人位置
            robot_vel: 機器人速度
            obstacle: 障礙物

        Returns:
            李導數
        """
        # 簡化：假設 h(x) = ||x - obs|| - d
        # 則 Lf h(x) = ∇h · f(x) = (x - obs) / ||x - obs|| · v

        dx = robot_pos[0].item() - obstacle.position[0]
        dy = robot_pos[1].item() - obstacle.position[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            return 0.0

        # 計算單位向量
        nx = dx / dist
        ny = dy / dist

        # 計算李導數
        vx = robot_vel[0].item()
        vy = robot_vel[1].item()

        return nx * vx + ny * vy


# ============================================================================
# 複合安全過濾器
# ============================================================================

class CompositeShield(SafetyShield):
    """複合安全過濾器

    結合多種安全過濾器，提供多層次的安全保護。
    """

    def __init__(
        self,
        rule_shield: Optional[RuleBasedShield] = None,
        cbf_shield: Optional[CBFShield] = None,
    ):
        """初始化複合安全過濾器

        Args:
            rule_shield: 基於規則的過濾器
            cbf_shield: CBF 過濾器
        """
        self.rule_shield = rule_shield or RuleBasedShield()
        self.cbf_shield = cbf_shield or CBFShield()

    def filter_action(
        self,
        action: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        lidar_scan: torch.Tensor,
        obstacles: List[Obstacle],
    ) -> SafetyAction:
        """使用多個過濾器過濾動作

        Args:
            action: RL 輸出的動作
            robot_pos: 機器人位置
            robot_vel: 機器人速度
            lidar_scan: Lidar 掃描數據
            obstacles: 障礙物列表

        Returns:
            過濾後的安全動作
        """
        # 第一層：基於規則的過濾（快速，簡單）
        rule_action = self.rule_shield.filter_action(
            action, robot_pos, robot_vel, lidar_scan, obstacles
        )

        # 如果規則過濾器已經判定為不安全，直接返回
        if not rule_action.safe:
            return rule_action

        # 第二層：CBF 過濾（更精確，但較慢）
        cbf_action = self.cbf_shield.filter_action(
            action, robot_pos, robot_vel, lidar_scan, obstacles
        )

        # 返回 CBF 的結果（如果 CBF 判定安全，則動作是安全的）
        return cbf_action
