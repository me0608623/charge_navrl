"""混合課程學習調度器 (Mixed Curriculum Scheduler)

用於防止災難性遺忘 (Catastrophic Forgetting)：
在訓練高階段時，持續混合低階段環境，確保基本能力不會退化。

訓練階段混合比例：
- Stage 1: 100% Stage 1 環境
- Stage 2: 80% Stage 2 + 20% Stage 1
- Stage 3: 60% Stage 3 + 30% Stage 2 + 10% Stage 1
- Stage 4: 40% Stage 4 + 30% Stage 3 + 20% Stage 2 + 10% Stage 1
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MixedCurriculumScheduler:
    """混合課程學習調度器

    在訓練過程中動態調整不同環境的採樣比例，
    確保 Agent 在學習高階技能時不會忘記低階基本能力。

    使用方式：
        scheduler = MixedCurriculumScheduler(
            total_iterations=20000,
            num_envs_per_stage=64
        )

        # 在訓練循環中
        env_ids = scheduler.sample_envs(current_iteration)
    """

    def __init__(
        self,
        total_iterations: int = 20000,
        num_envs_per_stage: int = 64,
        stage_boundaries: List[int] = None,
    ):
        """初始化混合課程學習調度器

        Args:
            total_iterations: 總訓練迭代次數
            num_envs_per_stage: 每個階段的環境數量
            stage_boundaries: 各階段邊界（迭代次數）
        """
        self.total_iterations = total_iterations
        self.num_envs_per_stage = num_envs_per_stage

        # 定義階段邊界（迭代次數）
        if stage_boundaries is None:
            stage_boundaries = [
                int(total_iterations * 0.25),   # Stage 1 → Stage 2
                int(total_iterations * 0.50),   # Stage 2 → Stage 3
                int(total_iterations * 0.75),   # Stage 3 → Stage 4
            ]
        self.stage_boundaries = stage_boundaries

        # 定義混合比例表
        # [Stage1, Stage2, Stage3, Stage4]
        self.mix_schedule = {
            "early": {    # 0 ~ 25%: Stage 1 only
                "stage1": 1.0,
                "stage2": 0.0,
                "stage3": 0.0,
                "stage4": 0.0,
            },
            "mid_early": { # 25% ~ 50%: Stage 1 + 2
                "stage1": 0.2,
                "stage2": 0.8,
                "stage3": 0.0,
                "stage4": 0.0,
            },
            "mid_late": {  # 50% ~ 75%: Stage 1 + 2 + 3
                "stage1": 0.1,
                "stage2": 0.3,
                "stage3": 0.6,
                "stage4": 0.0,
            },
            "late": {      # 75% ~ 100%: All stages
                "stage1": 0.1,
                "stage2": 0.2,
                "stage3": 0.3,
                "stage4": 0.4,
            },
        }

        # 溫存環境 ID（由外部設置）
        self.env_ids_by_stage = {
            "stage1": [],
            "stage2": [],
            "stage3": [],
            "stage4": [],
        }

    def get_mix_ratio(self, iteration: int) -> dict:
        """獲取當前迭代應使用的混合比例

        Args:
            iteration: 當前迭代次數

        Returns:
            各階段的採樣比例字典
        """
        if iteration < self.stage_boundaries[0]:
            return self.mix_schedule["early"]
        elif iteration < self.stage_boundaries[1]:
            return self.mix_schedule["mid_early"]
        elif iteration < self.stage_boundaries[2]:
            return self.mix_schedule["mid_late"]
        else:
            return self.mix_schedule["late"]

    def sample_env_ids(
        self,
        iteration: int,
        num_envs: int,
        device: torch.device,
    ) -> torch.Tensor:
        """根據混合比例採樣環境 ID

        Args:
            iteration: 當前迭代次數
            num_envs: 總環境數量
            device: 設備

        Returns:
            shape [num_envs]：採樣的環境 ID
        """
        mix_ratio = self.get_mix_ratio(iteration)

        # 計算每個階段應採樣的環境數量
        num_stage1 = int(num_envs * mix_ratio["stage1"])
        num_stage2 = int(num_envs * mix_ratio["stage2"])
        num_stage3 = int(num_envs * mix_ratio["stage3"])
        num_stage4 = num_envs - num_stage1 - num_stage2 - num_stage3

        # 構建環境 ID 列表
        env_ids = []

        # Stage 1 IDs: 0 ~ num_envs_per_stage-1
        for _ in range(num_stage1):
            env_ids.append(torch.randint(0, self.num_envs_per_stage, (1,), device=device).item())

        # Stage 2 IDs: num_envs_per_stage ~ 2*num_envs_per_stage-1
        for _ in range(num_stage2):
            env_ids.append(torch.randint(self.num_envs_per_stage, 2 * self.num_envs_per_stage, (1,), device=device).item())

        # Stage 3 IDs: 2*num_envs_per_stage ~ 3*num_envs_per_stage-1
        for _ in range(num_stage3):
            env_ids.append(torch.randint(2 * self.num_envs_per_stage, 3 * self.num_envs_per_stage, (1,), device=device).item())

        # Stage 4 IDs: 3*num_envs_per_stage ~ 4*num_envs_per_stage-1
        for _ in range(num_stage4):
            env_ids.append(torch.randint(3 * self.num_envs_per_stage, 4 * self.num_envs_per_stage, (1,), device=device).item())

        # 隨機打亂順序
        env_ids = torch.tensor(env_ids, device=device)
        indices = torch.randperm(len(env_ids), device=device)
        env_ids = env_ids[indices]

        return env_ids

    def get_stage_name(self, iteration: int) -> str:
        """獲取當前階段名稱（用於日誌）"""
        if iteration < self.stage_boundaries[0]:
            return "Stage1_Transition"
        elif iteration < self.stage_boundaries[1]:
            return "Stage2_Integration"
        elif iteration < self.stage_boundaries[2]:
            return "Stage3_Refinement"
        else:
            return "Stage4_Advanced"


class ReplayBufferSampler:
    """重播緩衝區採樣器：用於持續訓練時混合舊數據

    在最終階段，除了混合環境外，還應該隨機重播之前的訓練數據，
    這能進一步防止災難性遺忘。
    """

    def __init__(
        self,
        replay_ratio: float = 0.3,  # 30% 的步驟使用重播數據
        buffer_capacity: int = 100000,
    ):
        """初始化重播緩衝區採樣器

        Args:
            replay_ratio: 重播比例（0-1）
            buffer_capacity: 緩衝區容量
        """
        self.replay_ratio = replay_ratio
        self.buffer_capacity = buffer_capacity
        self.buffer = []

    def add_transition(self, transition: dict):
        """添加轉換到緩衝區"""
        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)  # 移除最舊的數據
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        """採樣歷史轉換

        Args:
            batch_size: 採樣批次大小

        Returns:
            採樣的轉換列表
        """
        if len(self.buffer) == 0:
            return []

        sample_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)
        return [self.buffer[i] for i in indices]

    def should_use_replay(self) -> bool:
        """決定當前步驟是否使用重播數據"""
        return len(self.buffer) > 0 and np.random.random() < self.replay_ratio


__all__ = [
    "MixedCurriculumScheduler",
    "ReplayBufferSampler",
]
