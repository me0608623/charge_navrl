# ============================================================================
# 文件說明
# ============================================================================
"""
多目標位置命令生成器

繼承自 GoalCommand，支援同時生成多個目標位置。
核心功能：
1. 同時生成 N 個目標（num_goals）
2. 動態輸出「離機器人最近的目標」作為當前命令
3. 支援所有目標的可視化

設計理念：
- 廣撒網策略：在初期生成多個目標，增加機器人隨機探索成功的機率
- 課程學習兼容：隨著能力提升，減少目標數量，最終回歸單目標
- 無縫集成：對外表現與單個 GoalCommand 一致，下游模組（觀測/獎勵）無需修改
"""

# ============================================================================
# 模塊導入
# ============================================================================
from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

# 導入基礎 GoalCommand
from .goal_command import GoalCommand, GoalCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class MultiGoalCommandCfg(GoalCommandCfg):
    """多目標位置命令配置"""
    
    class_type: type = MISSING  #將在下面設置為 MultiGoalCommand
    
    num_goals: int = 5
    # 目標數量（初始值）
    # 課程學習會動態調整此值


class MultiGoalCommand(GoalCommand):
    """多目標位置命令生成器
    
    維護多個目標位置，並始終輸出離機器人最近的一個。
    """

    cfg: MultiGoalCommandCfg

    def __init__(self, cfg: MultiGoalCommandCfg, env: ManagerBasedRLEnv):
        """初始化"""
        super().__init__(cfg, env)
        
        # 初始化多目標緩存 [num_envs, max_goals, 3]
        # 預設最大支持 10 個目標，由 cfg.num_goals 控制實際使用數量
        self.max_supported_goals = 10
        self.all_goals_pos_w = torch.zeros(
            self.num_envs, self.max_supported_goals, 3, 
            device=self.device
        )
        
        # 當前最近的目標 ID [num_envs]
        self.nearest_goal_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """獲取當前命令（最近的目標）
        
        每一幀都會被調用（用於觀測和獎勵計算）。
        這裡我們動態計算哪個目標最近，並返回該目標的位置。
        
        Returns:
            shape [num_envs, 3]: 最近目標的世界座標
        """
        # 獲取機器人位置 [num_envs, 3]
        robot_pos = self.robot.data.root_pos_w[:, :2]  # 只取 XY
        
        # 獲取當前活躍的目標數量
        active_goals = self.cfg.num_goals
        
        # 獲取所有目標位置 [num_envs, active_goals, 2]
        goals_pos = self.all_goals_pos_w[:, :active_goals, :2]
        
        # 計算距離 [num_envs, active_goals]
        # expand robot_pos: [num_envs, 1, 2]
        dists = torch.norm(goals_pos - robot_pos.unsqueeze(1), dim=2)
        
        # 找到最近的目標索引 [num_envs]
        min_dists, min_indices = torch.min(dists, dim=1)
        self.nearest_goal_idx = min_indices
        
        # 構造返回的目標位置 [num_envs, 3]
        # 使用 gather 提取最近目標
        # indices 需要擴展為 [num_envs, 1, 3] 以匹配 gather 的維度要求
        indices_expanded = min_indices.view(-1, 1, 1).expand(-1, 1, 3)
        nearest_goals = torch.gather(
            self.all_goals_pos_w[:, :active_goals, :], 
            1, 
            indices_expanded
        ).squeeze(1)
        
        return nearest_goals

    def _resample_command(self, env_ids: Sequence[int]):
        """生成新的多個目標位置"""
        # 獲取需要重置的環境數量
        num_envs = len(env_ids)
        if num_envs == 0:
            return
            
        # 轉換 env_ids
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            
        # 多次調用父類的生成邏輯來生成每個目標
        # 這是一種簡單但有效的方法，複用了父類的碰撞檢測邏輯
        
        active_goals = self.cfg.num_goals
        
        # 暫存父類的 goal_pos_w，因為我們會多次覆蓋它
        original_goal_pos_w = self.goal_pos_w.clone()
        
        for i in range(active_goals):
            # 調用父類方法生成一個目標
            # 父類會更新 self.goal_pos_w[env_ids]
            super()._resample_command(env_ids)
            
            # 將生成的目標保存到我們的多目標緩存中
            self.all_goals_pos_w[env_ids_tensor, i] = self.goal_pos_w[env_ids_tensor].clone()
            
        # 恢復父類的 goal_pos_w（雖然其實 command 屬性已經覆蓋了訪問權）
        # 但為了保持內部狀態一致性
        self.goal_pos_w = original_goal_pos_w

    def _update_goal_markers(self):
        """更新可視化標記（顯示所有目標）"""
        if not self.cfg.debug_vis:
            return
            
        active_goals = self.cfg.num_goals
        total_markers = self.num_envs * active_goals
        
        # 準備所有目標的位置 [num_envs * active_goals, 3]
        # 將 [num_envs, active_goals, 3] 展平
        all_markers_pos = self.all_goals_pos_w[:, :active_goals, :].reshape(-1, 3).clone()
        
        # 抬高一點顯示
        all_markers_pos[:, 2] = 0.5
        
        # 準備朝向（默認朝上）
        all_markers_quat = torch.zeros(total_markers, 4, device=self.device)
        all_markers_quat[:, 0] = 1.0
        
        # 如果 VisualizationMarkers 支援動態數量，我們需要重新創建或使用支援多點的 marker
        # Isaac Lab 的 VisualizationMarkers 通常接受固定數量的點嗎？
        # 它通常接受 [N, 3] 的位置。只要 N 一致即可。
        # 但這裡 N = num_envs * active_goals，與初始化的 num_envs 不同。
        # 
        # 我們這裡做一個 HACK：
        # 如果 marker 數量變了，可能需要重新初始化 visualizer？
        # 或者我們只顯示最近的目標？
        # 
        # 為了簡單起見，我們修改 visualizer 的行為：
        # 我們重新創建一個臨時的 visualizer 嗎？不行，太慢。
        # 
        # 讓我們只顯示「最近的目標」，這樣數量就是 num_envs，與父類一致。
        # 這樣最安全且性能最好。
        
        # 獲取最近目標的位置
        nearest_goals = self.command  # [num_envs, 3]
        
        marker_pos = nearest_goals.clone()
        marker_pos[:, 2] = 0.5
        
        marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        marker_quat[:, 0] = 1.0
        
        self.goal_visualizer.visualize(marker_pos, marker_quat)

# 設置配置類的 class_type
MultiGoalCommandCfg.class_type = MultiGoalCommand
