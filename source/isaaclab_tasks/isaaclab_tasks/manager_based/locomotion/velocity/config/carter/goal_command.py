"""
目標位置命令生成器模組

此模組實現了目標導航任務的命令生成器，用於在環境中隨機生成目標位置。
機器人需要導航到這些目標點，完成導航任務。
"""

from __future__ import annotations  # 啟用延遲類型註解評估

import torch  # PyTorch 張量運算庫
from collections.abc import Sequence  # 序列類型提示
from dataclasses import MISSING  # 缺失值標記
from typing import TYPE_CHECKING  # 類型檢查工具

from isaaclab.assets import Articulation  # 關節式機器人類別
from isaaclab.managers import CommandTerm, CommandTermCfg  # 命令項基類和配置
from isaaclab.markers import VisualizationMarkers  # 可視化標記類別
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG  # 標記配置
from isaaclab.utils import configclass  # 配置類別裝飾器

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 類型檢查時導入環境類別


class GoalCommand(CommandTerm):
    """
    目標位置命令生成器類別
    
    在環境中隨機生成目標位置，機器人需要導航到目標點。
    目標位置相對於機器人當前位置生成，使用極座標（距離和角度）然後轉換為笛卡爾座標。
    
    支援可視化標記，可以在模擬中顯示目標位置。
    """

    cfg: GoalCommandCfg  # 命令配置
    """命令配置"""

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        """
        初始化命令生成器
        
        Args:
            cfg: 命令配置
            env: 環境實例
        """
        super().__init__(cfg, env)

        # 獲取機器人資產
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 目標位置緩衝區 (x, y, z)，形狀為 [num_envs, 3]
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 創建可視化標記（如果啟用除錯視覺化）
        if self.cfg.debug_vis:
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()  # 複製綠色箭頭標記配置
            marker_cfg.prim_path = "/Visuals/GoalCommand"  # 設置標記在場景圖中的路徑
            marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)  # 設置標記縮放（縮小到 0.5 倍）
            self.goal_visualizer = VisualizationMarkers(marker_cfg)  # 創建可視化標記實例

    def __str__(self) -> str:
        """
        字串表示
        
        Returns:
            命令生成器的字串描述
        """
        msg = "GoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"  # 命令維度
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"  # 重新採樣時間範圍
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        目標位置命令屬性
        
        Returns:
            目標位置張量，形狀為 [num_envs, 3]，單位為公尺（世界座標系）
        """
        return self.goal_pos_w

    def _update_metrics(self):
        """
        更新指標（每步調用）
        
        目標導航不需要特殊指標，此方法為空實現。
        """
        pass  # 目標導航不需要特殊指標

    def _resample_command(self, env_ids: Sequence[int]):
        """
        重新採樣命令
        
        為指定的環境生成新的目標位置。目標位置使用極座標（距離和角度）生成，
        然後轉換為笛卡爾座標（相對於機器人當前位置）。
        
        Args:
            env_ids: 需要重新採樣的環境 ID 列表
        """
        # 生成新的目標位置
        num_envs = len(env_ids)  # 需要重新採樣的環境數量
        
        # 距離範圍：在配置的距離範圍內隨機生成
        distance = (
            torch.rand(num_envs, device=self.device)  # 生成 [0, 1] 的隨機數
            * (self.cfg.ranges.distance[1] - self.cfg.ranges.distance[0])  # 乘以範圍寬度
            + self.cfg.ranges.distance[0]  # 加上最小值
        )
        
        # 角度範圍：在配置的角度範圍內隨機生成
        angle = (
            torch.rand(num_envs, device=self.device)  # 生成 [0, 1] 的隨機數
            * (self.cfg.ranges.angle[1] - self.cfg.ranges.angle[0])  # 乘以範圍寬度
            + self.cfg.ranges.angle[0]  # 加上最小值
        )
        
        # 轉換為笛卡爾座標（相對於機器人當前位置）
        robot_pos = self.robot.data.root_pos_w[env_ids, :2]  # 機器人位置（僅 X, Y）
        
        # 使用極座標轉換公式：x = r * cos(θ), y = r * sin(θ)
        self.goal_pos_w[env_ids, 0] = robot_pos[:, 0] + distance * torch.cos(angle)  # X 座標
        self.goal_pos_w[env_ids, 1] = robot_pos[:, 1] + distance * torch.sin(angle)  # Y 座標
        self.goal_pos_w[env_ids, 2] = 0.0  # Z 座標固定為 0（地面高度）

    def _update_command(self):
        """
        更新命令（每步調用）
        
        目標位置不需要每步更新，此方法為空實現。
        """
        pass  # 目標位置不需要每步更新

    def compute(self, dt: float):
        """
        計算命令（每步調用）
        
        目標位置在 reset 時生成，不需要每步更新，此方法為空實現。
        
        Args:
            dt: 時間步長（未使用）
        """
        pass  # 目標位置在 reset 時生成，不需要每步更新

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """
        重置命令
        
        為指定環境生成新的目標位置。這通常在每個回合開始時調用。
        
        Args:
            env_ids: 需要重置的環境 ID 列表，None 表示重置所有環境
            
        Returns:
            空字典（無額外資訊返回）
        """
        if env_ids is None:
            env_ids = slice(None)  # 如果為 None，重置所有環境
        
        # 調用 _resample_command 生成目標
        if isinstance(env_ids, slice):
            env_ids = list(range(self.num_envs))  # 將切片轉換為 ID 列表
        
        self._resample_command(env_ids)  # 重新採樣目標位置

        # 更新可視化標記（如果啟用）
        if self.cfg.debug_vis:
            self._update_goal_markers()

        return {}

    def _update_goal_markers(self):
        """
        更新目標位置的可視化標記
        
        在模擬場景中顯示目標位置的標記，方便視覺化調試。
        """
        # 標記位置：使用目標位置，但稍微抬高一點以便觀察
        marker_pos = self.goal_pos_w.clone()  # 複製目標位置
        marker_pos[:, 2] = 0.1  # 稍微抬高一點（0.1 公尺），避免與地面重疊

        # 標記朝向（指向上方）：使用單位四元數（無旋轉）
        marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        marker_quat[:, 0] = 1.0  # w=1, x=y=z=0（單位四元數，無旋轉）

        # 更新標記：在場景中顯示目標位置
        self.goal_visualizer.visualize(marker_pos, marker_quat)


@configclass
class GoalCommandCfg(CommandTermCfg):
    """
    目標位置命令配置類別
    
    定義目標位置命令生成器的所有配置參數，包括：
    - 機器人資產名稱
    - 重新採樣時間範圍
    - 可視化選項
    - 目標生成範圍（距離和角度）
    """

    class_type: type = GoalCommand  # 命令類別：使用 GoalCommand

    asset_name: str = MISSING  # 機器人資產名稱（必須指定）
    """機器人資產名稱"""

    resampling_time_range: tuple[float, float] = MISSING  # 重新採樣時間範圍（秒）
    """重新採樣時間範圍 (s)"""

    debug_vis: bool = True  # 是否顯示可視化標記（預設為 True）
    """是否顯示可視化標記"""

    @configclass
    class Ranges:
        """
        目標生成範圍配置類別
        
        定義目標位置的生成範圍，使用極座標（距離和角度）。
        """
        distance: tuple[float, float] = MISSING  # 距離範圍（公尺）
        """距離範圍 (m)"""
        angle: tuple[float, float] = MISSING  # 角度範圍（弧度）
        """角度範圍 (rad)"""

    ranges: Ranges = MISSING  # 目標生成範圍配置（必須指定）
    """目標生成範圍配置"""
