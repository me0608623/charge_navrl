"""層級式導航環境配置 (Hierarchical Navigation Environment)

整合 AIT* 全域規劃器 + RL (PPO) 局部控制器，
實現動態重規劃和域隨機化。

架構：
┌───────────────────────────────────────────────────────────────────┐
│  AIT* Global Planner (1-5 Hz)                                    │
│  ├─ Static Map: 初始地圖                                        │
│  ├─ Dynamic Re-planning: RL 卡住時觸發                          │
│  └─ Frontier Exploration: 未知環境探索                           │
├───────────────────────────────────────────────────────────────────┤
│  Local Goal Extractor (10-20 Hz)                                │
│  └─ Carrot-on-stick: lookahead distance                        │
├───────────────────────────────────────────────────────────────────┤
│  RL PPO Controller (50-100 Hz)                                  │
│  ├─ Observations: LiDAR + Navigation Features                   │
│  └─ Actions: [linear_vel, angular_vel]                          │
├───────────────────────────────────────────────────────────────────┤
│  Domain Randomization                                           │
│  ├─ Physics: 地面摩擦、機器人質量                                │
│  ├─ Sensor: LiDAR 雜訊、丟失                                    │
│  └─ Dynamics: 動力學參數隨機化                                  │
└───────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg

# 導入層級式導航組件
from ..mdp.path_planner.local_goal_extractor import (
    LocalGoalExtractor,
    LocalGoalConfig,
)
from ..mdp.path_planner.aitstar_adapter import (
    AITStarPathPlanner,
    AITStarPlannerCfg,
    create_aitstar_planner,
)
from ..mdp.path_planner.path_visualizer import (
    AITStarPathVisualizer,
    MultiEnvPathVisualizer,
)


# ============================================================================
# 動態重規劃條件判斷
# ============================================================================

class ReplanTrigger:
    """重規劃觸發器

    決定何時需要呼叫 AIT* 重新規劃路徑。
    """

    def __init__(
        self,
        stuck_threshold: float = 0.1,      # 速度閾值（米/秒）
        stuck_duration: float = 2.0,        # 停滯時長（秒）
        off_path_threshold: float = 1.5,    # 偏離路徑閾值（米）
        progress_threshold: float = 0.2,    # 進步閾值（米/秒）
    ):
        """初始化重規劃觸發器

        Args:
            stuck_threshold: 速度低於此值視為「卡住」
            stuck_duration: 持續多久視為「真的卡住」
            off_path_threshold: 偏離路徑超過此值觸發重規劃
            progress_threshold: 進步速度低於此值觸發警告
        """
        self.stuck_threshold = stuck_threshold
        self.stuck_duration = stuck_duration
        self.off_path_threshold = off_path_threshold
        self.progress_threshold = progress_threshold

        # 追蹤狀態
        self._stuck_timers: dict[int, float] = {}
        self._last_positions: dict[int, torch.Tensor] = {}
        self._last_progress_time: dict[int, float] = {}

    def should_replan(
        self,
        env_id: int,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        aitstar_path: torch.Tensor,
        current_time: float,
    ) -> tuple[bool, str]:
        """判斷是否需要重規劃

        Args:
            env_id: 環境 ID
            robot_pos: [2] 機器人位置
            robot_vel: [2] 機器人速度
            aitstar_path: [N, 2] AIT* 路徑
            current_time: 當前時間

        Returns:
            (should_replan, reason): 是否需要重規劃及原因
        """
        speed = torch.norm(robot_vel).item()

        # 條件 1: 機器人卡住（速度太低持續一段時間）
        if speed < self.stuck_threshold:
            if env_id not in self._stuck_timers:
                self._stuck_timers[env_id] = current_time

            stuck_time = current_time - self._stuck_timers[env_id]
            if stuck_time > self.stuck_duration:
                return True, f"機器人卡住 {stuck_time:.1f}秒"
        else:
            if env_id in self._stuck_timers:
                del self._stuck_timers[env_id]

        # 條件 2: 偏離路徑太遠
        if len(aitstar_path) > 0:
            distances = torch.norm(aitstar_path - robot_pos, dim=1)
            min_dist = torch.min(distances).item()

            if min_dist > self.off_path_threshold:
                return True, f"偏離路徑 {min_dist:.2f}米"

        # 條件 3: 沒有進步（長時間沒靠近目標）
        if env_id in self._last_positions:
            last_pos = self._last_positions[env_id]
            displacement = torch.norm(robot_pos - last_pos).item()

            # 計算平均進步速度
            time_delta = current_time - self._last_progress_time.get(env_id, current_time - 1.0)
            if time_delta > 0:
                progress_rate = displacement / time_delta

                if progress_rate < self.progress_threshold and time_delta > 3.0:
                    return True, f"進步太慢 ({progress_rate:.3f} m/s)"

        # 更新追蹤狀態
        self._last_positions[env_id] = robot_pos.clone()
        self._last_progress_time[env_id] = current_time

        return False, ""

    def reset_env(self, env_id: int):
        """重置特定環境的觸發狀態"""
        if env_id in self._stuck_timers:
            del self._stuck_timers[env_id]
        if env_id in self._last_positions:
            del self._last_positions[env_id]
        if env_id in self._last_progress_time:
            del self._last_progress_time[env_id]


# ============================================================================
# 域隨機化配置
# ============================================================================

class DomainRandomizationConfig:
    """域隨機化配置

    定義各種隨機化參數，用於提高 Sim-to-Real 泛化能力。
    """

    # 物理隨機化
    friction_range: tuple = (0.3, 0.9)      # 地面摩擦係數
    restitution_range: tuple = (0.1, 0.5)   # 彈性係數
    mass_range: tuple = (0.8, 1.2)          # 機器人質量乘數

    # 感測器隨機化
    lidar_noise_std: float = 0.02           # LiDAR 高斯雜訊標準差
    lidar_dropout_rate: float = 0.05        # LiDAR 射線丟失率
    lidar_max_dropout: int = 5             # 最多丟失射線數

    # 動力學隨機化
    torque_range: tuple = (0.9, 1.1)       # 扭矩常數乘數

    # 環境隨機化
    obstacle_randomization: bool = True    # 隨機障礙物位置
    goal_randomization: bool = True        # 隨機目標位置


def apply_domain_randomization(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cfg: DomainRandomizationConfig,
) -> None:
    """應用域隨機化到指定環境

    Args:
        env: 環境實例
        env_ids: 要隨機化的環境 ID
        cfg: 隨機化配置
    """
    if len(env_ids) == 0:
        return

    device = env.device
    num_envs = len(env_ids)

    # 隨機化物理參數（如果環境支持）
    if hasattr(env.scene, "ground"):
        # 這需要環境實現物理隨機化接口
        pass

    # 隨機化機器人質量
    if hasattr(env.scene, "robot") and cfg.mass_range != (1.0, 1.0):
        robot: Articulation = env.scene["robot"]
        # 注意：質量隨機化需要在 PhysX 層實現
        # 這裡只是一個示例
        pass


# ============================================================================
# 層級式導航管理器
# ============================================================================

class HierarchicalNavigationManager:
    """層級式導航管理器

    整合 AIT* 全域規劃器、局部目標提取器、重規劃觸發器和視覺化器。
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: str = "robot",
        lidar_cfg: str = "lidar",
        goal_command_cfg: str = "goal_command",
        lookahead_distance: float = 2.0,
        replan_interval: float = 1.0,        # AIT* 規劃間隔（秒）
        enable_visualization: bool = False,
    ):
        """初始化層級式導航管理器

        Args:
            env: 環境實例
            robot_cfg: 機器人配置名稱
            lidar_cfg: LiDAR 配置名稱
            goal_command_cfg: 目標命令配置名稱
            lookahead_distance: 前瞻距離（米）
            replan_interval: AIT* 重規劃間隔（秒）
            enable_visualization: 是否啟用視覺化
        """
        self.env = env
        self.robot_cfg = robot_cfg
        self.lidar_cfg = lidar_cfg
        self.goal_command_cfg = goal_command_cfg
        self.lookahead_distance = lookahead_distance
        self.replan_interval = replan_interval
        self.enable_visualization = enable_visualization

        self.num_envs = env.num_envs
        self.device = env.device

        # 創建 AIT* 規劃器（共享給所有環境）
        self.aitstar_planner: Optional[AITStarPathPlanner] = None
        self._init_aitstar_planner()

        # 創建局部目標提取器
        local_goal_cfg = LocalGoalConfig(
            lookahead_distance=lookahead_distance,
            min_update_distance=0.3,
            smoothing_factor=0.3,
        )
        self.local_goal_extractor = LocalGoalExtractor(local_goal_cfg, self.num_envs)

        # 創建重規劃觸發器
        self.replan_trigger = ReplanTrigger(
            stuck_threshold=0.1,
            stuck_duration=2.0,
            off_path_threshold=1.5,
            progress_threshold=0.2,
        )

        # 創建視覺化器
        if enable_visualization:
            self.visualizer = MultiEnvPathVisualizer(
                num_envs=self.num_envs,
                prim_path="/World/AITStarPaths",
                max_envs_to_visualize=min(4, self.num_envs),
            )
        else:
            self.visualizer = None

        # 追蹤狀態
        self._last_replan_time = torch.zeros(self.num_envs, device=self.device)
        self._current_time = torch.zeros(self.num_envs, device=self.device)

        # 存儲當前路徑
        self._current_paths: list[Optional[torch.Tensor]] = [None] * self.num_envs
        self._path_version: list[int] = [0] * self.num_envs

    def _init_aitstar_planner(self):
        """初始化 AIT* 規劃器"""
        try:
            self.aitstar_planner = create_aitstar_planner(
                map_size=(20, 20),
                grid_resolution=0.1,
                robot_radius=0.3,
            )
        except Exception as e:
            print(f"[警告] 無法初始化 AIT* 規劃器: {e}")
            self.aitstar_planner = None

    def update(
        self,
        dt: float,
    ) -> dict:
        """更新導航系統

        每個環境步驟調用一次，處理：
        1. 檢查是否需要重規劃
        2. 更新局部目標
        3. 更新視覺化

        Args:
            dt: 時間步長（秒）

        Returns:
            調試信息字典
        """
        debug_info = {
            "replan_count": 0,
            "replan_reasons": [],
        }

        self._current_time += dt

        # 獲取機器人狀態
        robot = self.env.scene[self.robot_cfg]
        robot_pos = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
        robot_vel = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]

        # 獲取目標位置
        goal_pos = self.env.command_manager.get_command(self.goal_command_cfg)[:, :2]

        # 檢查每個環境是否需要重規劃
        for env_id in range(self.num_envs):
            current_time = self._current_time[env_id].item()

            # 檢查是否到達重規劃間隔
            time_since_replan = current_time - self._last_replan_time[env_id].item()

            # 檢查是否需要重規劃（時間觸發或條件觸發）
            need_replan = False
            replan_reason = ""

            # 條件 1: 定期重規劃
            if time_since_replan >= self.replan_interval:
                need_replan = True
                replan_reason = f"定期重規劃 ({time_since_replan:.1f}s)"

            # 條件 2: 緊急重規劃（卡住或偏離路徑）
            elif self._current_paths[env_id] is not None:
                should_replan, reason = self.replan_trigger.should_replan(
                    env_id,
                    robot_pos[env_id],
                    robot_vel[env_id],
                    self._current_paths[env_id],
                    current_time,
                )
                if should_replan:
                    need_replan = True
                    replan_reason = reason

            # 執行重規劃
            if need_replan and self.aitstar_planner is not None:
                self._replan(env_id, robot_pos[env_id], goal_pos[env_id])
                self._last_replan_time[env_id] = current_time
                self._path_version[env_id] += 1
                debug_info["replan_count"] += 1
                debug_info["replan_reasons"].append(f"Env {env_id}: {replan_reason}")

            # 更新局部目標
            if self._current_paths[env_id] is not None:
                local_goal = self.local_goal_extractor.extract_local_goal(
                    robot_pos[env_id],
                    self._current_paths[env_id],
                    path_version=self._path_version[env_id],
                    env_id=env_id,
                )
            else:
                # 沒有路徑，直接用目標作為局部目標
                local_goal = goal_pos[env_id]

            # 保存局部目標到環境（供觀測函數使用）
            if not hasattr(self.env, '_local_goal_world'):
                self.env._local_goal_world = torch.zeros(self.num_envs, 2, device=self.device)
            self.env._local_goal_world[env_id] = local_goal

        # 更新視覺化
        if self.visualizer is not None:
            for env_id in range(min(4, self.num_envs)):
                if self._current_paths[env_id] is not None:
                    self.visualizer.visualize(
                        env_id,
                        self._current_paths[env_id],
                        start_pos=robot_pos[env_id],
                        goal_pos=goal_pos[env_id],
                    )

        return debug_info

    def _replan(
        self,
        env_id: int,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ):
        """執行 AIT* 重規劃

        Args:
            env_id: 環境 ID
            start_pos: [2] 起點位置
            goal_pos: [2] 終點位置
        """
        if self.aitstar_planner is None:
            return

        try:
            # 更新地圖（從環境獲取當前障礙物）
            self.aitstar_planner._update_map_from_env(self.env)

            # 規劃新路徑
            new_path = self.aitstar_planner.plan_path(
                start_pos.unsqueeze(0),
                goal_pos.unsqueeze(0),
                self.env,
            )

            if len(new_path) > 0:
                self._current_paths[env_id] = new_path
                print(f"[AIT*] Env {env_id}: 重規劃成功，路徑包含 {len(new_path)} 個點")
            else:
                print(f"[AIT*] Env {env_id}: 重規劃失敗，找不到路徑")

        except Exception as e:
            print(f"[AIT*] Env {env_id}: 重規劃出錯: {e}")

    def reset_env(self, env_id: int):
        """重置特定環境的狀態

        Args:
            env_id: 環境 ID
        """
        self.replan_trigger.reset_env(env_id)
        self._last_replan_time[env_id] = 0.0
        self._current_paths[env_id] = None
        self._path_version[env_id] = 0

    def reset_all(self):
        """重置所有環境"""
        for env_id in range(self.num_envs):
            self.reset_env(env_id)


__all__ = [
    "ReplanTrigger",
    "DomainRandomizationConfig",
    "apply_domain_randomization",
    "HierarchicalNavigationManager",
]
