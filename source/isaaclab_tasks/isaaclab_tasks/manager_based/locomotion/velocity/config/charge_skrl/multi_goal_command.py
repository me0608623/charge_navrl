"""多目標位置命令生成器

同時生成 N 個目標位置，機器人追蹤最近的一個。
到達最近目標後，episode 結束，重新生成 N 個新目標。

核心功能：
1. 同時生成 N 個目標（num_goals）
2. 動態輸出「離機器人最近的目標」作為當前命令
3. 所有目標的可視化（綠色箭頭）

設計：
- 對外表現與單個 GoalCommand 一致（command 屬性返回最近目標）
- 下游模組（觀測/獎勵/終止）無需修改
- VisualizationMarkers 支援動態數量，直接傳 num_envs * num_goals 個 marker
"""

from __future__ import annotations

import sys
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .goal_command import GoalCommand, GoalCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# Command class FIRST (before Cfg) — so Cfg can reference it directly
# ============================================================================

class MultiGoalCommand(GoalCommand):
    """多目標位置命令生成器

    維護 num_goals 個目標位置，command 屬性始終返回離機器人最近的一個。
    到達最近目標 → goal_reached 終止 → 新 episode 重新生成所有目標。
    """

    cfg: MultiGoalCommandCfg  # Forward ref OK due to `from __future__ import annotations`

    def __init__(self, cfg: MultiGoalCommandCfg, env: ManagerBasedRLEnv):
        # 先呼叫父類 __init__（會創建 goal_visualizer）
        super().__init__(cfg, env)

        # 多目標緩存 [num_envs, num_goals, 3]
        self.all_goals_pos_w = torch.zeros(
            self.num_envs, self.cfg.num_goals, 3,
            device=self.device,
        )

        # 當前最近目標索引 [num_envs]
        self.nearest_goal_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )

        print(
            f"[MultiGoalCommand.__init__] num_envs={self.num_envs}, "
            f"num_goals={self.cfg.num_goals}, "
            f"has_visualizer={hasattr(self, 'goal_visualizer')}, "
            f"debug_vis={self.cfg.debug_vis}",
            flush=True,
        )

    @property
    def command(self) -> torch.Tensor:
        """返回每個環境中離機器人最近的目標位置。

        Returns:
            [num_envs, 3] 最近目標的世界座標
        """
        robot_pos = self.robot.data.root_pos_w[:, :2]  # [num_envs, 2]
        goals_xy = self.all_goals_pos_w[:, :, :2]  # [num_envs, num_goals, 2]

        # 計算距離 [num_envs, num_goals]
        dists = torch.norm(goals_xy - robot_pos.unsqueeze(1), dim=2)

        # 最近目標
        _, min_indices = torch.min(dists, dim=1)
        self.nearest_goal_idx = min_indices

        # gather 最近目標 [num_envs, 3]
        idx = min_indices.view(-1, 1, 1).expand(-1, 1, 3)
        nearest = torch.gather(self.all_goals_pos_w, 1, idx).squeeze(1)

        return nearest

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Override reset to ensure multi-goal resample + visualization."""
        # 處理 env_ids
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids = list(range(self.num_envs))

        # 生成多目標
        self._resample_command(env_ids)

        # 更新可視化
        if self.cfg.debug_vis:
            self._update_goal_markers()

        return {}

    def _resample_command(self, env_ids: Sequence[int]):
        """為指定環境生成 num_goals 個新目標。"""
        if len(env_ids) == 0:
            return

        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        for i in range(self.cfg.num_goals):
            # 調用父類生成 1 個目標（會更新 self.goal_pos_w[env_ids]）
            super()._resample_command(env_ids)
            # 保存到多目標緩存
            self.all_goals_pos_w[env_ids_t, i] = self.goal_pos_w[env_ids_t].clone()

        # goal_pos_w 保留最後一個目標（不影響下游，因 command 屬性回傳最近目標）

    def _update_goal_markers(self):
        """更新可視化：顯示所有 num_goals 個目標（每個環境）。"""
        has_vis = hasattr(self, "goal_visualizer")
        if not has_vis:
            print("[MultiGoalCommand._update_goal_markers] NO goal_visualizer!", flush=True)
            return

        ng = self.cfg.num_goals
        total = self.num_envs * ng

        # [num_envs, num_goals, 3] → [num_envs * num_goals, 3]
        marker_pos = self.all_goals_pos_w.reshape(-1, 3).clone()
        marker_pos[:, 2] = 0.3  # 抬高避免陷入地面

        marker_quat = torch.zeros(total, 4, device=self.device)
        marker_quat[:, 0] = 1.0  # 單位四元數

        # 只在前幾次 print（避免 spam）
        if not hasattr(self, "_marker_print_count"):
            self._marker_print_count = 0
        if self._marker_print_count < 3:
            print(
                f"[MultiGoalCommand] Visualizing {total} markers "
                f"(envs={self.num_envs}, goals={ng}), "
                f"pos sample env0 goal0: {marker_pos[0].tolist()}",
                flush=True,
            )
            self._marker_print_count += 1

        self.goal_visualizer.visualize(marker_pos, marker_quat)


# ============================================================================
# Config class AFTER command class — can set class_type directly
# ============================================================================

@configclass
class MultiGoalCommandCfg(GoalCommandCfg):
    """多目標位置命令配置

    Attributes:
        num_goals: 同時存在的目標數量（預設 10）
    """

    class_type: type = MultiGoalCommand

    num_goals: int = 10
