"""
課程學習管理器 (Curriculum Learning Manager)

管理 AIT* + RL 層級式導航的訓練課程，實現由易到難的漸進式學習。

課程階段設計：
- Stage 1: Goal-Oriented Following (無障礙物)
- Stage 2: Static Obstacle Navigation (簡單障礙物)
- Stage 3: Complex Topology (窄門、迷宮)
- Stage 4: Dynamic & Noisy Environment (動態障礙物)
"""

from __future__ import annotations

import math
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .aitstar_adapter import AITStarPathPlanner, AITStarPlannerCfg


@dataclass
class CurriculumStageConfig:
    """單個課程階段的配置

    Attributes:
        name: 階段名稱
        stage_id: 階段編號 (1-4)
        description: 階段描述
        obstacle_density: 障礙物密度 (0-1)
        obstacle_types: 障礙物類型列表
        has_dynamic_obstacles: 是否有動態障礙物
        goal_distance_range: 目標距離範圍 [min, max] (米)
        path_complexity_target: 目標路徑複雜度 (1.0-5.0)
        success_threshold: 晉升下一階的成功率閾值
        min_episodes: 最少訓練回合數
        sensor_noise: 感測器雜訊水平
        has_virtual_walls: 是否有虛擬牆壁（防止災難性遺忘）
        wall_penalty: 撞牆懲罰值
        mixed_training_ratio: 混合訓練比例（回頭練習早期階段的頻率）
    """
    name: str
    stage_id: int
    description: str
    obstacle_density: float = 0.0
    obstacle_types: List[str] = field(default_factory=list)
    has_dynamic_obstacles: bool = False
    goal_distance_range: Tuple[float, float] = (5.0, 10.0)
    path_complexity_target: float = 1.0
    success_threshold: float = 0.8
    min_episodes: int = 1000
    sensor_noise: float = 0.0
    has_virtual_walls: bool = True
    wall_penalty: float = 5.0
    mixed_training_ratio: float = 0.1


# 預定義的課程階段配置
CURRICULUM_STAGES = {
    1: CurriculumStageConfig(
        name="Goal-Oriented Following",
        stage_id=1,
        description="無障礙物的空地，學習基礎運動控制（加速、轉向）",
        obstacle_density=0.0,
        obstacle_types=[],
        has_dynamic_obstacles=False,
        goal_distance_range=(5.0, 15.0),
        path_complexity_target=1.0,
        success_threshold=0.85,
        min_episodes=500,
        sensor_noise=0.0,
        has_virtual_walls=True,  # ⚠️ 關鍵修復：Stage 1 也要有邊界限制！
        wall_penalty=5.0,  # 撞牆懲罰
    ),
    2: CurriculumStageConfig(
        name="Static Obstacle Navigation",
        stage_id=2,
        description="簡單的幾何障礙物，學習感知偏差處理",
        obstacle_density=0.15,
        obstacle_types=["cuboid", "cylinder"],
        has_dynamic_obstacles=False,
        goal_distance_range=(8.0, 18.0),
        path_complexity_target=2.0,
        success_threshold=0.80,
        min_episodes=1500,
        sensor_noise=0.02,
        has_virtual_walls=True,
        wall_penalty=5.0,
        mixed_training_ratio=0.15,  # 15% 機率回 Stage 1 複習
    ),
    3: CurriculumStageConfig(
        name="Complex Topology",
        stage_id=3,
        description="窄門、U型彎、牆壁，學習狹窄空間精準控制",
        obstacle_density=0.35,
        obstacle_types=["cuboid", "cylinder", "wall", "narrow_passage"],
        has_dynamic_obstacles=False,
        goal_distance_range=(10.0, 20.0),
        path_complexity_target=3.5,
        success_threshold=0.75,
        min_episodes=2500,
        sensor_noise=0.05,
        has_virtual_walls=True,
        wall_penalty=8.0,
        mixed_training_ratio=0.20,  # 20% 機率回早期階段複習
    ),
    4: CurriculumStageConfig(
        name="Dynamic & Noisy Environment",
        stage_id=4,
        description="動態障礙物、感測器雜訊，學習即時反應",
        obstacle_density=0.40,
        obstacle_types=["cuboid", "cylinder", "wall", "dynamic"],
        has_dynamic_obstacles=True,
        goal_distance_range=(12.0, 25.0),
        path_complexity_target=5.0,
        success_threshold=0.70,
        min_episodes=5000,
        sensor_noise=0.10,
        has_virtual_walls=True,
        wall_penalty=10.0,
        mixed_training_ratio=0.25,  # 25% 機率回早期階段複習
    ),
}


@dataclass
class CurriculumMetrics:
    """課程學習指標

    追蹤當前階段的訓練進度，用於決定何時晉升到下一階段。
    """
    episodes_completed: int = 0
    episodes_successful: int = 0
    average_reward: float = 0.0
    average_path_length: float = 0.0
    average_time_to_goal: float = 0.0
    collision_rate: float = 0.0
    goal_reached_rate: float = 0.0

    def success_rate(self) -> float:
        """計算成功率"""
        if self.episodes_completed == 0:
            return 0.0
        return self.episodes_successful / self.episodes_completed

    def should_progress(self, stage_config: CurriculumStageConfig) -> bool:
        """判斷是否應該晉升到下一階段"""
        return (
            self.episodes_completed >= stage_config.min_episodes and
            self.success_rate() >= stage_config.success_threshold
        )


class CurriculumManager:
    """課程學習管理器

    管理 AIT* + RL 層級式導航的完整訓練課程。
    負責：
    1. 追蹤當前訓練階段
    2. 根據表現自動晉升/降級
    3. 為當前階段生成環境配置
    4. 提供 AIT* 路徑規劃器
    """

    def __init__(
        self,
        initial_stage: int = 1,
        max_stage: int = 4,
        auto_progress: bool = True,
        planner_cfg: Optional[AITStarPlannerCfg] = None,
    ):
        """初始化課程管理器

        Args:
            initial_stage: 初始階段 (1-4)
            max_stage: 最高階段
            auto_progress: 是否自動晉升
            planner_cfg: AIT* 規劃器配置
        """
        self.current_stage = initial_stage
        self.max_stage = max_stage
        self.auto_progress = auto_progress

        # 獲取當前階段配置
        self.stage_config = CURRICULUM_STAGES[self.current_stage]

        # 創建 AIT* 規劃器
        if planner_cfg is None:
            planner_cfg = AITStarPlannerCfg()
        self.planner = AITStarPathPlanner(planner_cfg)

        # 訓練指標
        self.metrics = CurriculumMetrics()

        # 歷史記錄
        self.stage_history: List[int] = [self.current_stage]
        self.episode_rewards: List[float] = []

    def get_current_stage(self) -> int:
        """獲取當前階段"""
        return self.current_stage

    def get_stage_config(self, stage: Optional[int] = None) -> CurriculumStageConfig:
        """獲取階段配置"""
        if stage is None:
            stage = self.current_stage
        return CURRICULUM_STAGES.get(stage, CURRICULUM_STAGES[1])

    def update_metrics(
        self,
        reward: float,
        success: bool,
        path_length: float = 0.0,
        time_to_goal: float = 0.0,
        collided: bool = False,
    ):
        """更新訓練指標

        Args:
            reward: 本回合獎勵
            success: 是否成功到達目標
            path_length: 行駛路徑長度
            time_to_goal: 到達目標時間
            collided: 是否發生碰撞
        """
        self.metrics.episodes_completed += 1
        self.episode_rewards.append(reward)

        if success:
            self.metrics.episodes_successful += 1

        # 更新平均值（使用移動平均）
        alpha = 0.1  # 平滑因子
        n = self.metrics.episodes_completed

        self.metrics.average_reward = (
            (1 - alpha) * self.metrics.average_reward + alpha * reward
        )

        if path_length > 0:
            self.metrics.average_path_length = (
                (1 - alpha) * self.metrics.average_path_length + alpha * path_length
            )

        if time_to_goal > 0:
            self.metrics.average_time_to_goal = (
                (1 - alpha) * self.metrics.average_time_to_goal + alpha * time_to_goal
            )

        if collided:
            self.metrics.collision_rate = (
                (1 - alpha) * self.metrics.collision_rate + alpha * 1.0
            )
        else:
            self.metrics.collision_rate = (1 - alpha) * self.metrics.collision_rate

        self.metrics.goal_reached_rate = self.metrics.success_rate()

        # 檢查是否應該晉升
        if self.auto_progress and self._should_progress():
            self._progress_to_next_stage()

    def _should_progress(self) -> bool:
        """判斷是否應該晉升到下一階段"""
        if self.current_stage >= self.max_stage:
            return False
        return self.metrics.should_progress(self.stage_config)

    def _progress_to_next_stage(self):
        """晉升到下一階段"""
        if self.current_stage < self.max_stage:
            self.current_stage += 1
            self.stage_config = CURRICULUM_STAGES[self.current_stage]
            self.stage_history.append(self.current_stage)
            self.metrics = CurriculumMetrics()  # 重置指標
            print(f"\n{'='*60}")
            print(f"🎓 課程晉升: Stage {self.current_stage - 1} → Stage {self.current_stage}")
            print(f"   {self.stage_config.name}")
            print(f"   {self.stage_config.description}")
            print(f"{'='*60}\n")

    def regress_to_previous_stage(self):
        """降級到上一階段（當表現不佳時）"""
        if self.current_stage > 1:
            self.current_stage -= 1
            self.stage_config = CURRICULUM_STAGES[self.current_stage]
            self.stage_history.append(self.current_stage)
            self.metrics = CurriculumMetrics()
            print(f"\n⚠️  課程降級: Stage {self.current_stage + 1} → Stage {self.current_stage}")

    def generate_obstacles_for_stage(
        self,
        env: ManagerBasedRLEnv,
        num_envs: int,
    ) -> List[Dict]:
        """為當前階段生成障礙物配置

        Args:
            env: Isaac Lab 環境
            num_envs: 環境數量

        Returns:
            障礙物配置列表
        """
        config = self.stage_config
        obstacles = []

        # 計算障礙物數量
        map_area = config.goal_distance_range[1] ** 2
        num_obstacles = int(map_area * config.obstacle_density / 10)

        for env_id in range(num_envs):
            env_obstacles = []

            for _ in range(num_obstacles):
                # 隨機選擇障礙物類型
                obs_type = random.choice(config.obstacle_types) if config.obstacle_types else "cuboid"

                # 隨機位置（避免在起點附近）
                angle = random.uniform(0, 2 * math.pi)
                min_dist = config.goal_distance_range[0] * 0.3
                max_dist = config.goal_distance_range[1] * 0.8
                distance = random.uniform(min_dist, max_dist)

                pos_x = distance * math.cos(angle)
                pos_y = distance * math.sin(angle)

                # 隨機尺寸
                if obs_type == "cylinder":
                    radius = random.uniform(0.3, 0.8)
                    size = [radius * 2, radius * 2, 1.0]
                elif obs_type == "wall":
                    width = random.uniform(2.0, 5.0)
                    height = random.uniform(0.2, 0.5)
                    size = [width, height, 1.2]
                else:  # cuboid
                    size = [
                        random.uniform(0.5, 1.5),
                        random.uniform(0.5, 1.5),
                        random.uniform(0.8, 1.5),
                    ]

                env_obstacles.append({
                    "type": obs_type,
                    "position": [pos_x, pos_y, 0.0],
                    "size": size,
                    "is_dynamic": obs_type == "dynamic" if config.has_dynamic_obstacles else False,
                })

            obstacles.append(env_obstacles)

        return obstacles

    def generate_goal_for_stage(
        self,
        env: ManagerBasedRLEnv,
        env_id: int = 0,
    ) -> torch.Tensor:
        """為當前階段生成目標位置

        Args:
            env: Isaac Lab 環境
            env_id: 環境 ID

        Returns:
            [2] 目標位置
        """
        config = self.stage_config

        # 隨機距離和角度
        distance = random.uniform(*config.goal_distance_range)
        angle = random.uniform(0, 2 * math.pi)

        goal_x = distance * math.cos(angle)
        goal_y = distance * math.sin(angle)

        return torch.tensor([goal_x, goal_y])

    def plan_path_with_aitstar(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        env: Optional[ManagerBasedRLEnv] = None,
    ) -> torch.Tensor:
        """使用 AIT* 規劃路徑

        Args:
            start_pos: [2] 起點位置
            goal_pos: [2] 終點位置
            env: 環境實例（可選）

        Returns:
            [N, 2] 路徑點
        """
        return self.planner.plan_path(start_pos, goal_pos, env)

    def generate_guided_sampling_action(
        self,
        current_pos: torch.Tensor,
        path: torch.Tensor,
        noise_std: float = 0.5,
    ) -> torch.Tensor:
        """生成引導採樣動作（用於訓練初期）

        在 AIT* 路徑附近添加擾動，幫助 RL 探索。

        Args:
            current_pos: [2] 當前位置
            path: [N, 2] AIT* 路徑
            noise_std: 擾動標準差

        Returns:
            [2] 建議的移動方向
        """
        if len(path) < 2:
            return torch.zeros(2)

        # 找到最近的目標路徑點
        distances = torch.norm(path - current_pos, dim=1)
        target_idx = torch.argmin(distances)

        # 使用下一個路徑點作為目標
        if target_idx < len(path) - 1:
            target_idx = target_idx + 1

        target = path[target_idx]

        # 計算方向
        direction = target - current_pos
        direction = direction / (torch.norm(direction) + 1e-6)

        # 添加擾動
        noise = torch.randn(2) * noise_std
        guided_direction = direction + noise
        guided_direction = guided_direction / (torch.norm(guided_direction) + 1e-6)

        return guided_direction

    def compute_heuristic_field(
        self,
        path: torch.Tensor,
        goal_pos: torch.Tensor,
        grid_size: int = 50,
    ) -> torch.Tensor:
        """計算 AIT* 啟發式場（用於 RL 特徵輸入）

        啟發式場表示每個網格到目標的「代價」，
        可以幫助 RL 理解哪些方向長遠來看代價更低。

        Args:
            path: [N, 2] AIT* 路徑
            goal_pos: [2] 目標位置
            grid_size: 網格大小

        Returns:
            [grid_size, grid_size] 啟發式場
        """
        # 創建網格
        field = torch.zeros((grid_size, grid_size))

        # 簡化版：使用到路徑的距離作為啟發式值
        # 實際實現應使用 AIT* 的逆向搜尋代價場
        for i in range(grid_size):
            for j in range(grid_size):
                # 網格坐標轉世界坐標
                x = (i - grid_size / 2) * 0.5
                y = (j - grid_size / 2) * 0.5
                pos = torch.tensor([x, y])

                # 計算到路徑的最小距離
                distances = torch.norm(path - pos, dim=1)
                min_dist = distances.min().item()

                # 計算到目標的直線距離
                goal_dist = torch.norm(goal_pos - pos).item()

                # 啟發式值 = 路徑偏離 + 目標距離
                field[i, j] = min_dist + goal_dist

        # 歸一化
        field = (field - field.min()) / (field.max() - field.min() + 1e-6)

        return field

    def get_guidance_signal(
        self,
        robot_pos: torch.Tensor,
        path: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """獲取 AIT* 對 RL 的引導信號

        Args:
            robot_pos: [2] 機器人位置
            path: [N, 2] AIT* 路徑

        Returns:
            引導信號字典：
            - 'nearest_point': 最近的路徑點
            - 'tangent_direction': 路徑切線方向
            - 'progress_along_path': 沿路徑進度 (0-1)
            - 'deviation': 偏離路徑距離
        """
        if len(path) < 2:
            return {
                'nearest_point': robot_pos.clone(),
                'tangent_direction': torch.zeros(2),
                'progress_along_path': torch.tensor(0.0),
                'deviation': torch.tensor(0.0),
            }

        # 找最近路徑點
        distances = torch.norm(path - robot_pos, dim=1)
        nearest_idx = torch.argmin(distances)
        nearest_point = path[nearest_idx]
        deviation = distances[nearest_idx]

        # 計算路徑進度
        progress = nearest_idx.item() / (len(path) - 1)

        # 計算切線方向（指向下一個路徑點）
        if nearest_idx < len(path) - 1:
            next_point = path[nearest_idx + 1]
            tangent = next_point - nearest_point
            tangent = tangent / (torch.norm(tangent) + 1e-6)
        else:
            tangent = torch.zeros(2)

        return {
            'nearest_point': nearest_point,
            'tangent_direction': tangent,
            'progress_along_path': torch.tensor(progress),
            'deviation': deviation,
        }

    def get_state_summary(self) -> Dict:
        """獲取當前狀態摘要"""
        return {
            'current_stage': self.current_stage,
            'stage_name': self.stage_config.name,
            'episodes_completed': self.metrics.episodes_completed,
            'success_rate': self.metrics.success_rate(),
            'average_reward': self.metrics.average_reward,
            'collision_rate': self.metrics.collision_rate,
            'stage_history': self.stage_history.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumManager(\n"
            f"  current_stage={self.current_stage}/4,\n"
            f"  stage_name='{self.stage_config.name}',\n"
            f"  episodes={self.metrics.episodes_completed},\n"
            f"  success_rate={self.metrics.success_rate():.2%},\n"
            f"  auto_progress={self.auto_progress}\n"
            f")"
        )


def create_curriculum_manager(
    initial_stage: int = 1,
    max_stage: int = 4,
    auto_progress: bool = True,
    **kwargs
) -> CurriculumManager:
    """工廠函數：創建課程管理器

    Args:
        initial_stage: 初始階段
        max_stage: 最高階段
        auto_progress: 是否自動晉升
        **kwargs: 其他配置參數

    Returns:
        CurriculumManager 實例
    """
    return CurriculumManager(
        initial_stage=initial_stage,
        max_stage=max_stage,
        auto_progress=auto_progress,
        **kwargs
    )


__all__ = [
    "CurriculumManager",
    "CurriculumStageConfig",
    "CurriculumMetrics",
    "CURRICULUM_STAGES",
    "create_curriculum_manager",
]
