"""
AIT* 路徑規劃器 - Isaac Lab 適配器

將現有的 AIT* (Adaptive Informed Trees) 實現適配到 Isaac Lab 環境。
參考：/globle_planner/src/aitstar_path_planner/aitstar_path_planner/aitstar.py
"""

from __future__ import annotations

import sys
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 導入現有的 AIT* 實現
# 添加 globle_planner 路徑到 Python 模組搜索路徑
_AITSTAR_PATH = "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/globle_planner/src/aitstar_path_planner"
if _AITSTAR_PATH not in sys.path:
    sys.path.insert(0, _AITSTAR_PATH)

try:
    from aitstar_path_planner.aitstar import (
        AITStar,
        State,
        Vertex,
        CollisionChecker,
        SimpleCollisionChecker,
        PathSmoother as AITPathSmoother,
    )
    _AITSTAR_AVAILABLE = True
except ImportError:
    _AITSTAR_AVAILABLE = False
    print("[警告] 無法導入 AIT* 模組，請確保路徑正確")

from .environment_map import EnvironmentMap, EnvironmentMapCfg


@dataclass
class AITStarPlannerCfg:
    """AIT* 路徑規劃器配置類別

    Attributes:
        map_cfg: 環境地圖配置
        batch_size: 每批次新增的頂點數
        rewire_factor: 重連半徑因子 (> 1.0)
        goal_bias: 目標偏向機率
        max_iterations: 最大迭代次數
        convergence_threshold: 收斂閾值
        robot_radius: 機器人半徑（米）
        path_smoothing: 是否啟用路徑平滑
        waypoint_spacing: 路徑點間隔（米）
        seed: 隨機種子
    """
    map_cfg: EnvironmentMapCfg = field(default_factory=EnvironmentMapCfg)
    batch_size: int = 50      # 降低批次大小以加快速度
    rewire_factor: float = 1.1
    goal_bias: float = 0.05
    max_iterations: int = 100  # 降低迭代次数以加快速度
    convergence_threshold: float = 1e-3
    robot_radius: float = 0.3
    path_smoothing: bool = True
    waypoint_spacing: float = 0.2
    seed: Optional[int] = None


class IsaacLabCollisionChecker(CollisionChecker):
    """Isaac Lab 環境的碰撞檢測器

    實現 AIT* 所需的 CollisionChecker 介面，
    直接使用 Isaac Lab 的環境數據進行碰撞檢測。
    """

    def __init__(
        self,
        env_map: EnvironmentMap,
        robot_radius: float = 0.3,
    ):
        """初始化碰撞檢測器

        Args:
            env_map: 環境地圖實例
            robot_radius: 機器人半徑（米）
        """
        self.env_map = env_map
        self.robot_radius = robot_radius

        # 設定邊界
        self.bounds_min = self.env_map.map_origin.cpu().numpy()
        self.bounds_max = (
            self.env_map.map_origin.cpu().numpy() +
            np.array([self.env_map.cfg.map_size[0], self.env_map.cfg.map_size[1]])
        )

    def is_state_valid(self, state: State) -> bool:
        """檢查狀態是否有效（無碰撞）"""
        pos = state.position

        # 檢查邊界
        if np.any(pos < self.bounds_min) or np.any(pos > self.bounds_max):
            return False

        # 使用 EnvironmentMap 檢查障礙物（已考慮機器人半徑）
        grid_x, grid_y = self.env_map.world_to_grid(
            torch.from_numpy(pos).float()
        )

        # 調試：檢查坐標是否有效
        if not (0 <= grid_x < self.env_map.grid_width and 0 <= grid_y < self.env_map.grid_depth):
            return False

        return not self.env_map.is_occupied(grid_x, grid_y)

    def is_edge_valid(
        self,
        state1: State,
        state2: State,
        resolution: float = 0.1,
    ) -> bool:
        """檢查兩狀態之間的邊是否有效（無碰撞）"""
        if not self.is_state_valid(state1) or not self.is_state_valid(state2):
            return False

        direction = state2.position - state1.position
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return True

        n_steps = max(2, int(np.ceil(distance / resolution)))

        for i in range(1, n_steps):
            t = i / n_steps
            intermediate_pos = state1.position + t * direction
            intermediate_state = State(intermediate_pos)

            if not self.is_state_valid(intermediate_state):
                return False

        return True


class AITStarPathPlanner:
    """AIT* 路徑規劃器 - Isaac Lab 版本

    適配器類別，將現有的 AIT* 實現整合到 Isaac Lab 環境中。
    """

    def __init__(self, cfg: AITStarPlannerCfg):
        """初始化 AIT* 路徑規劃器

        Args:
            cfg: 規劃器配置
        """
        if not _AITSTAR_AVAILABLE:
            raise RuntimeError(
                "AIT* 模組無法導入。請確保 globle_planner 路徑正確：\n"
                f"  {_AITSTAR_PATH}"
            )

        self.cfg = cfg

        # 創建環境地圖
        self.map = EnvironmentMap(cfg.map_cfg)

        # 創建碰撞檢測器
        self.collision_checker = IsaacLabCollisionChecker(
            env_map=self.map,
            robot_radius=cfg.robot_radius,
        )

        # 設定邊界
        bounds_min = self.collision_checker.bounds_min
        bounds_max = self.collision_checker.bounds_max

        # 創建 AIT* 規劃器
        self.planner = AITStar(
            collision_checker=self.collision_checker,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            batch_size=cfg.batch_size,
            rewire_factor=cfg.rewire_factor,
            goal_bias=cfg.goal_bias,
            max_iterations=cfg.max_iterations,
            convergence_threshold=cfg.convergence_threshold,
            seed=cfg.seed,
        )

        # 路徑平滑器
        self.smoother = AITPathSmoother(self.collision_checker)

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
            如果找不到路徑，返回直線路徑
        """
        # 確保輸入形狀正確
        if start_pos.dim() == 1:
            start_pos = start_pos.unsqueeze(0)
        if goal_pos.dim() == 1:
            goal_pos = goal_pos.unsqueeze(0)

        # 只處理單個環境
        start = start_pos[0].cpu().numpy()
        goal = goal_pos[0].cpu().numpy()
        device = start_pos.device

        # 調試：檢查起始和目標位置
        start_valid = self.collision_checker.is_state_valid(State(start))
        goal_valid = self.collision_checker.is_state_valid(State(goal))
        print(f"[AIT* Debug] start valid: {start_valid}, goal valid: {goal_valid}")
        print(f"[AIT* Debug] bounds_min: {self.collision_checker.bounds_min}")
        print(f"[AIT* Debug] bounds_max: {self.collision_checker.bounds_max}")

        # 從環境更新障礙物（如果提供）
        if env is not None:
            self._update_map_from_env(env)

        # 執行 AIT* 規劃
        print(f"[AIT* Debug] 開始執行 AIT* 規劃...")
        print(f"[AIT* Debug] max_iterations: {self.cfg.max_iterations}")
        print(f"[AIT* Debug] batch_size: {self.cfg.batch_size}")
        try:
            path_np, cost = self.planner.plan(start, goal)
            print(f"[AIT* Debug] 規劃完成，路徑點數: {len(path_np) if path_np is not None else 0}, cost: {cost}")
        except ValueError as e:
            # AIT* 規劃失敗（例如起始位置碰撞），使用直線路徑
            print(f"[AIT*] 規劃失敗: {e}，使用直線路徑")
            path_np = None
        except Exception as e:
            # 捕获其他异常
            print(f"[AIT*] 規劃異常: {type(e).__name__}: {e}，使用直線路徑")
            path_np = None

        if path_np is None or len(path_np) == 0:
            # 回退到直線路徑
            print(f"[AIT*] 無法找到從 {start} 到 {goal} 的路徑，使用直線路徑")
            # 創建直線路徑（起點 -> 終點）
            path_np = np.array([start, goal])

        # 確保 path_np 是 numpy 數組
        if isinstance(path_np, list):
            path_np = np.array(path_np)

        # 轉換為 PyTorch 張量（保持與輸入相同的設備）
        path = torch.from_numpy(path_np).float().to(device)

        # 路徑平滑（如果啟用且不是直線）
        if self.cfg.path_smoothing and len(path) > 2:
            path = self._smooth_path(path)

        # 根據間隔取樣路徑點
        if self.cfg.waypoint_spacing > 0:
            path = self._subsample_path(path, self.cfg.waypoint_spacing)

        return path

    def _update_map_from_env(self, env: ManagerBasedRLEnv) -> None:
        """從環境更新障礙物地圖

        架構說明：
        - 靜態幾何（牆壁、房間結構）→ 給 Global Planner
        - 動態障礙物（箱子、行人）→ 交給 Local Policy (LiDAR + RL)
        - 這是標準的 Global + Local 分層導航架構

        因此這個方法：
        - ✓ 保留靜態牆壁（已通過 EnvironmentMap.__init__ 添加）
        - ✗ 不添加動態障礙物（obstacle_*）

        Args:
            env: Isaac Lab 環境實例
        """
        # 清空並重新添加靜態牆壁（不添加動態障礙物）
        self.map.clear()
        # clear() 會自動調用 add_boundary_walls()，所以牆壁已經在這裡了

        # 🔥 關鍵架構決策：不添加動態障礙物到 Planner
        # 動態障礙物會在 reset_obstacles 事件中隨機移動
        # 如果添加到 Planner，會導致：
        #   1. Planner Thrashing（路徑瘋狂重算）
        #   2. 規劃震盪
        #   3. 效能崩潰
        # 動態障礙物交給：
        #   - LiDAR / Raycast (感知)
        #   - RL Policy (即時閃避)

    def _smooth_path(
        self,
        path: torch.Tensor,
        max_iterations: int = 50,
    ) -> torch.Tensor:
        """使用 AIT* 的路徑平滑器

        Args:
            path: [N, 2] 原始路徑
            max_iterations: 最大迭代次數

        Returns:
            平滑後的路徑
        """
        # 🔥 修復 CUDA tensor 轉換問題：先移到 CPU 再轉 numpy
        path_np = [p.cpu().numpy() for p in path]
        smoothed_np = self.smoother.shortcut(path_np, max_iterations=max_iterations)

        if len(smoothed_np) < 2:
            return path

        # 轉換回原本的設備
        return torch.from_numpy(np.array(smoothed_np)).float().to(path.device)

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

    def get_statistics(self) -> Dict:
        """獲取規劃統計資訊"""
        return self.planner.get_statistics()

    def get_search_tree(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """獲取搜索樹（用於可視化）"""
        return self.planner.get_tree()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"AITStarPathPlanner("
            f"iterations={stats.get('iterations', 0)}, "
            f"vertices={stats.get('total_vertices', 0)}, "
            f"solution_cost={stats.get('solution_cost', float('inf')):.2f}, "
            f"map={self.map})"
        )


def create_aitstar_planner(
    map_size: Tuple[float, float] = (20.0, 20.0),
    grid_resolution: float = 0.1,
    obstacle_inflation: float = 0.3,
    robot_radius: float = 0.3,
    max_iterations: int = 1000,
    **kwargs
) -> AITStarPathPlanner:
    """工廠函數：創建 AIT* 路徑規劃器

    Args:
        map_size: 地圖大小 [width, depth] (米)
        grid_resolution: 網格解析度 (米/格)
        obstacle_inflation: 障礙物膨脹距離 (米)
        robot_radius: 機器人半徑 (米)
        max_iterations: 最大迭代次數
        **kwargs: 其他配置參數

    Returns:
        AITStarPathPlanner 實例
    """
    map_cfg = EnvironmentMapCfg(
        map_size=map_size,
        grid_resolution=grid_resolution,
        obstacle_inflation=obstacle_inflation,
    )

    planner_cfg = AITStarPlannerCfg(
        map_cfg=map_cfg,
        robot_radius=robot_radius,
        max_iterations=max_iterations,
        **kwargs
    )

    return AITStarPathPlanner(planner_cfg)


__all__ = [
    "AITStarPathPlanner",
    "AITStarPlannerCfg",
    "IsaacLabCollisionChecker",
    "create_aitstar_planner",
]
