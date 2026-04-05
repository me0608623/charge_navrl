# ============================================================================
# 文件說明
# ============================================================================
"""
目標位置命令生成器

這個文件實現了「目標位置生成器」，負責：
1. 在環境中隨機生成目標位置（機器人要去的地方）
2. 在模擬器中顯示綠色箭頭標記（給人類看的）
3. 在環境重置時更新目標位置

這個類是 Isaac Lab 的 CommandTerm 系統的一部分：
- CommandTerm：命令項（定義任務目標）
- 其他例子：速度命令、位置命令、軌跡命令等
"""

# ============================================================================
# 模塊導入
# ============================================================================
from __future__ import annotations  # 允許使用字符串形式的類型提示（Python 3.7+）

import torch  # PyTorch：用於張量運算（生成隨機數、座標計算）
from collections.abc import Sequence  # 序列類型（用於 env_ids 參數）
from dataclasses import MISSING  # 標記必填參數（配置驗證）
from typing import TYPE_CHECKING  # 類型檢查標記（避免循環導入）

# Isaac Lab 核心模塊
from isaaclab.assets import Articulation  # 關節機器人類（機器人資產）
from isaaclab.managers import CommandTerm, CommandTermCfg  # 命令項基類和配置
from isaaclab.markers import VisualizationMarkers  # 可視化標記類（顯示箭頭）
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,  # 藍色箭頭標記配置（未使用，但可選）
    GREEN_ARROW_X_MARKER_CFG,  # 綠色箭頭標記配置（用於顯示目標）
)
from isaaclab.utils import configclass  # 配置類裝飾器

# 類型檢查時才導入（避免運行時循環導入）
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# 目標位置命令類
# ============================================================================
# 這個類負責生成和管理目標位置
# ============================================================================

class GoalCommand(CommandTerm):
    """目標位置命令生成器
    
    功能：
    1. 在環境重置時，為每個環境隨機生成一個目標位置
    2. 目標位置以極座標形式生成（距離 + 角度），然後轉換為笛卡爾座標
    3. 在模擬器中顯示綠色箭頭標記（可選）
    4. 提供目標位置給其他模塊（觀測、獎勵函數等）
    
    繼承自 CommandTerm：
    - CommandTerm 是 Isaac Lab 的命令系統基類
    - 提供統一的接口（reset、compute、command 等）
    - 其他命令類型：速度命令、位置命令等
    """

    # 類型提示
    cfg: GoalCommandCfg  # 命令配置（包含生成範圍、可視化選項等）

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令生成器
        
        Args:
            cfg: 命令配置（目標生成範圍、可視化選項等）
            env: 環境實例（用於訪問場景和設備）
        
        初始化流程：
        1. 調用父類初始化
        2. 獲取機器人資產（用於計算相對位置）
        3. 初始化目標位置緩存
        4. 創建可視化標記（如果啟用）
        """
        super().__init__(cfg, env)  # 調用父類初始化（設置設備、環境數量等）

        # ------------------------------------------------------------------------
        # 獲取機器人資產
        # ------------------------------------------------------------------------
        self.robot: Articulation = env.scene[cfg.asset_name]
        # 從場景中獲取名為 "robot" 的資產
        # 為什麼需要機器人？
        # - 目標位置是「相對於機器人」生成的（極座標）
        # - 需要機器人的位置來計算目標的世界座標

        # ------------------------------------------------------------------------
        # 初始化目標位置緩存
        # ------------------------------------------------------------------------
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # shape: [num_envs, 3] = [X, Y, Z] 世界座標
        # 例如：[128, 3] = 128 個環境，每個環境一個目標位置
        # 
        # 初始值全為 0（會在 reset 時更新）
        # 
        # 為什麼用 torch.zeros？
        # - 需要在 GPU 上運行（如果使用 GPU）
        # - 統一的數據類型（方便後續計算）

        # ------------------------------------------------------------------------
        # 創建可視化標記（綠色箭頭）
        # ------------------------------------------------------------------------
        if self.cfg.debug_vis:
            # 如果啟用調試可視化
            
            # 複製綠色箭頭配置模板
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            # GREEN_ARROW_X_MARKER_CFG 是 Isaac Lab 預設的綠色箭頭配置
            # .copy() 創建副本（避免修改原始配置）
            
            marker_cfg.prim_path = "/Visuals/GoalCommand"
            # 設定箭頭在場景中的路徑
            # /Visuals/GoalCommand 是全局路徑（所有環境共享）
            # 注意：這裡沒有用 {ENV_REGEX_NS}，所以所有環境的箭頭會重疊
            # （但因為每個環境的目標位置不同，實際顯示時會分開）
            
            marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            # 設定箭頭大小：縮放到 0.5 倍
            # (X, Y, Z) 三個軸的縮放比例
            # 0.5 = 原始大小的一半（更小，不擋視線）
            
            # 創建可視化標記物件
            self.goal_visualizer = VisualizationMarkers(marker_cfg)
            # 這個物件會在模擬器中顯示綠色箭頭
            # 箭頭位置會在 _update_goal_markers() 中更新

    # ------------------------------------------------------------------------
    # 字符串表示（用於調試）
    # ------------------------------------------------------------------------
    def __str__(self) -> str:
        """字符串表示（用於打印和調試）
        
        Returns:
            描述命令配置的字符串
        
        例子：
            GoalCommand:
                Command dimension: (3,)
                Resampling time range: (1e+10, 1e+10)
        """
        msg = "GoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        # command.shape = [num_envs, 3]
        # shape[1:] = [3]（去掉批次維度）
        # 表示每個環境的目標位置是 3 維（X, Y, Z）
        
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        # 顯示重新採樣時間範圍（何時生成新目標）
        
        return msg

    # ------------------------------------------------------------------------
    # 屬性方法
    # ------------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """目標位置命令（供其他模塊使用）
        
        Returns:
            shape [num_envs, 3]：目標位置 (X, Y, Z) 在世界座標系
        
        這個屬性會被以下模塊使用：
        - 觀測函數：goal_position_in_robot_frame()、goal_distance()
        - 獎勵函數：velocity_toward_goal()、progress_to_goal()
        - 終止條件：goal_reached()
        """
        return self.goal_pos_w

    # ------------------------------------------------------------------------
    # 命令更新方法（父類接口）
    # ------------------------------------------------------------------------
    def _update_metrics(self):
        """更新指標（每步調用）
        
        這個方法在每個時間步都會被調用，用於更新統計指標。
        目標導航任務不需要特殊指標，所以留空。
        
        其他命令類型可能會在這裡更新：
        - 速度命令：當前速度、加速度
        - 位置命令：位置誤差、速度誤差
        """
        pass  # 目標導航不需要特殊指標

    def _resample_command(self, env_ids: Sequence[int]):
        """重新採樣命令（生成新的目標位置）
        
        這是核心方法！負責生成目標位置。
        
        生成策略：
        1. 使用極座標（距離 + 角度）生成，相對於機器人當前位置
        2. 檢查是否在牆壁邊界內
        3. 檢查是否與障礙物碰撞
        4. 如果不合法，重新生成（最多重試 max_resample_attempts 次）
        5. 轉換為笛卡爾座標（X, Y, Z）
        
        Args:
            env_ids: 需要重新生成目標的環境 ID 列表
                     例如：[0, 5, 12] = 第 0、5、12 個環境
        """
        # 獲取需要重置的環境數量
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        
        # 將 env_ids 轉為 tensor
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # 獲取配置參數
        wall_boundary = self.cfg.wall_boundary
        wall_safe_margin = self.cfg.wall_safe_margin
        obstacle_safe_distance = self.cfg.obstacle_safe_distance
        max_attempts = self.cfg.max_resample_attempts
        num_obstacles = self.cfg.num_obstacles
        
        # 獲取環境原點（每個環境的座標偏移）
        env_origins = self._env.scene.env_origins[env_ids_tensor, :2]  # [num_envs, 2]
        
        # 獲取機器人在環境局部座標系中的位置
        robot_pos_world = self.robot.data.root_pos_w[env_ids_tensor, :2]  # [num_envs, 2]
        robot_pos_local = robot_pos_world - env_origins  # [num_envs, 2]
        
        # ------------------------------------------------------------------------
        # 收集所有障礙物位置（環境局部座標系）
        # ------------------------------------------------------------------------
        obstacle_positions = []
        for i in range(num_obstacles):
            try:
                obstacle = self._env.scene[f"obstacle_{i}"]
                obs_pos_world = obstacle.data.root_pos_w[env_ids_tensor, :2]  # [num_envs, 2]
                obs_pos_local = obs_pos_world - env_origins  # [num_envs, 2]
                obstacle_positions.append(obs_pos_local)
            except (KeyError, AttributeError):
                # 障礙物不存在，跳過
                continue
        
        # 將障礙物位置堆疊為 [num_envs, num_valid_obstacles, 2]
        if obstacle_positions:
            all_obstacles = torch.stack(obstacle_positions, dim=1)  # [num_envs, num_obstacles, 2]
        else:
            all_obstacles = None
        
        # ------------------------------------------------------------------------
        # 為每個環境生成有效的目標位置
        # ------------------------------------------------------------------------
        # 初始化：標記哪些環境還需要生成有效目標
        needs_resample = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        
        # 儲存候選目標位置（環境局部座標系）
        candidate_goals_local = torch.zeros(num_envs, 2, device=self.device)
        
        for attempt in range(max_attempts):
            # 找出還需要重新採樣的環境
            resample_mask = needs_resample
            num_resample = resample_mask.sum().item()
            
            if num_resample == 0:
                break  # 所有環境都已生成有效目標
            
            # 生成隨機距離和角度
            distance = (
                torch.rand(num_resample, device=self.device)
                * (self.cfg.ranges.distance[1] - self.cfg.ranges.distance[0])
                + self.cfg.ranges.distance[0]
            )
            angle = (
                torch.rand(num_resample, device=self.device)
                * (self.cfg.ranges.angle[1] - self.cfg.ranges.angle[0])
                + self.cfg.ranges.angle[0]
            )
            
            # 計算候選目標位置（環境局部座標系）
            robot_pos_resample = robot_pos_local[resample_mask]  # [num_resample, 2]
            goal_x = robot_pos_resample[:, 0] + distance * torch.cos(angle)
            goal_y = robot_pos_resample[:, 1] + distance * torch.sin(angle)
            
            # 更新候選位置
            candidate_goals_local[resample_mask, 0] = goal_x
            candidate_goals_local[resample_mask, 1] = goal_y
            
            # ------------------------------------------------------------------------
            # 檢查 1：牆壁邊界（環境局部座標系）
            # ------------------------------------------------------------------------
            valid_boundary = wall_boundary - wall_safe_margin
            within_walls = (
                (candidate_goals_local[:, 0].abs() < valid_boundary) &
                (candidate_goals_local[:, 1].abs() < valid_boundary)
            )
            
            # ------------------------------------------------------------------------
            # 檢查 2：障礙物碰撞
            # ------------------------------------------------------------------------
            if all_obstacles is not None:
                # 計算目標與所有障礙物的距離
                # candidate_goals_local: [num_envs, 2]
                # all_obstacles: [num_envs, num_obstacles, 2]
                goal_expanded = candidate_goals_local.unsqueeze(1)  # [num_envs, 1, 2]
                distances_to_obstacles = torch.norm(goal_expanded - all_obstacles, dim=2)  # [num_envs, num_obstacles]
                
                # 檢查最小距離是否大於安全距離
                min_dist_to_obstacle = distances_to_obstacles.min(dim=1)[0]  # [num_envs]
                clear_of_obstacles = min_dist_to_obstacle > obstacle_safe_distance
            else:
                clear_of_obstacles = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            
            # ------------------------------------------------------------------------
            # 檢查 3：與機器人的距離（確保不會生成在機器人身上）
            # ------------------------------------------------------------------------
            dist_to_robot = torch.norm(candidate_goals_local - robot_pos_local, dim=1)
            clear_of_robot = dist_to_robot > 0.5  # 至少離機器人 0.5 米
            
            # ------------------------------------------------------------------------
            # 綜合判斷：所有條件都滿足
            # ------------------------------------------------------------------------
            valid_goals = within_walls & clear_of_obstacles & clear_of_robot
            
            # 更新需要重新採樣的環境
            needs_resample = needs_resample & (~valid_goals)
        
        # ------------------------------------------------------------------------
        # 最終處理：如果還有環境沒有找到有效目標，放寬條件
        # ------------------------------------------------------------------------
        if needs_resample.any():
            # 對於無法找到有效目標的環境，只確保在牆壁邊界內
            failed_envs = needs_resample
            candidate_goals_local[failed_envs, 0] = torch.clamp(
                candidate_goals_local[failed_envs, 0],
                -valid_boundary, valid_boundary
            )
            candidate_goals_local[failed_envs, 1] = torch.clamp(
                candidate_goals_local[failed_envs, 1],
                -valid_boundary, valid_boundary
            )
        
        # ------------------------------------------------------------------------
        # 轉換為世界座標系並更新目標位置
        # ------------------------------------------------------------------------
        goal_pos_world = candidate_goals_local + env_origins
        
        self.goal_pos_w[env_ids_tensor, 0] = goal_pos_world[:, 0]
        self.goal_pos_w[env_ids_tensor, 1] = goal_pos_world[:, 1]
        self.goal_pos_w[env_ids_tensor, 2] = 0.0  # Z 座標 = 0.0（貼地）

    def _update_command(self):
        """更新命令（每步調用）
        
        這個方法在每個時間步都會被調用，用於更新命令。
        目標位置是靜態的（生成後不變），所以不需要每步更新。
        
        其他命令類型可能會在這裡更新：
        - 速度命令：根據時間變化速度
        - 軌跡命令：沿著軌跡移動目標點
        """
        pass  # 目標位置不需要每步更新

    def compute(self, dt: float):
        """計算命令（每步調用）
        
        Args:
            dt: 時間步長（秒）
        
        這個方法在每個時間步都會被調用，用於計算命令。
        目標位置在 reset 時生成，不需要每步計算。
        
        其他命令類型可能會在這裡計算：
        - 速度命令：根據加速度計算速度
        - 位置命令：根據速度計算位置
        """
        pass  # 目標位置在 reset 時生成，不需要每步更新

    # ------------------------------------------------------------------------
    # 重置方法
    # ------------------------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """重置命令（環境重置時調用）
        
        這是主要入口！當環境重置時，為指定環境生成新的目標位置。
        
        Args:
            env_ids: 需要重置的環境 ID 列表
                     None = 重置所有環境
                     [0, 5] = 只重置第 0 和第 5 個環境
        
        Returns:
            空字典（父類接口要求）
        
        執行流程：
        1. 處理 env_ids 參數（None → slice → list）
        2. 調用 _resample_command() 生成新目標
        3. 更新可視化標記（如果啟用）
        """
        # ------------------------------------------------------------------------
        # 步驟 1：處理 env_ids 參數
        # ------------------------------------------------------------------------
        if env_ids is None:
            env_ids = slice(None)
            # slice(None) 表示「所有元素」
            # 等同於 [0, 1, 2, ..., num_envs-1]

        # ------------------------------------------------------------------------
        # 步驟 2：轉換 slice 為 list
        # ------------------------------------------------------------------------
        if isinstance(env_ids, slice):
            env_ids = list(range(self.num_envs))
            # 如果是 slice，轉換為完整的環境 ID 列表
            # 例如：num_envs=128 → [0, 1, 2, ..., 127]

        # ------------------------------------------------------------------------
        # 步驟 3：生成新目標位置
        # ------------------------------------------------------------------------
        self._resample_command(env_ids)
        # 調用核心方法生成目標位置
        # 會更新 self.goal_pos_w[env_ids]

        # ------------------------------------------------------------------------
        # 步驟 4：更新可視化標記
        # ------------------------------------------------------------------------
        if self.cfg.debug_vis:
            self._update_goal_markers()
            # 如果啟用可視化，更新綠色箭頭的位置

        return {}  # 返回空字典（父類接口要求）

    # ------------------------------------------------------------------------
    # 可視化方法
    # ------------------------------------------------------------------------
    def _update_goal_markers(self):
        """更新目標位置的可視化標記（綠色箭頭）
        
        這個方法會將目標位置顯示在模擬器中（綠色箭頭）。
        箭頭會出現在目標位置，幫助人類觀察機器人的任務。
        
        注意：
        - 箭頭只是給人類看的，機器人不會「看到」箭頭
        - 機器人通過觀測函數獲取目標位置（數字形式）
        """
        # ------------------------------------------------------------------------
        # 步驟 1：設定箭頭位置
        # ------------------------------------------------------------------------
        marker_pos = self.goal_pos_w.clone()
        # 複製目標位置（避免修改原始數據）
        # shape: [num_envs, 3]

        marker_pos[:, 2] = 0.1  # 稍微抬高一點（0.1 米）
        # 為什麼抬高？
        # - 讓箭頭浮在地面上方（不會被地面遮擋）
        # - 更容易看到（視覺效果更好）
        # 
        # 注意：目標位置本身是 0.0（貼地），但箭頭顯示在 0.1 米高度

        # ------------------------------------------------------------------------
        # 步驟 2：設定箭頭朝向
        # ------------------------------------------------------------------------
        marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        # 初始化四元數（表示旋轉）
        # shape: [num_envs, 4] = [w, x, y, z]

        marker_quat[:, 0] = 1.0  # w=1, x=y=z=0
        # 四元數 (1, 0, 0, 0) = 不旋轉（默認朝向）
        # 箭頭會垂直向上（Z 軸方向）
        # 
        # 如果想讓箭頭指向機器人，需要計算旋轉角度（見下面的說明）

        # ------------------------------------------------------------------------
        # 步驟 3：更新標記
        # ------------------------------------------------------------------------
        self.goal_visualizer.visualize(marker_pos, marker_quat)
        # 在模擬器中顯示/更新箭頭
        # marker_pos：箭頭位置 [num_envs, 3]
        # marker_quat：箭頭朝向 [num_envs, 4]
        # 
        # 這會在所有環境中顯示綠色箭頭（每個環境一個）

    # ------------------------------------------------------------------------
    # 可選：讓箭頭指向機器人（進階功能）
    # ------------------------------------------------------------------------
    # 如果你想讓箭頭指向機器人，可以這樣修改：
    #
    # def _update_goal_markers(self):
    #     marker_pos = self.goal_pos_w.clone()
    #     marker_pos[:, 2] = 0.1
    #     
    #     # 計算從目標指向機器人的方向
    #     robot_pos = self.robot.data.root_pos_w[:, :2]
    #     goal_pos = self.goal_pos_w[:, :2]
    #     direction = robot_pos - goal_pos  # 從目標指向機器人
    #     
    #     # 計算旋轉角度（在 XY 平面）
    #     angle = torch.atan2(direction[:, 1], direction[:, 0])
    #     
    #     # 轉換成四元數（繞 Z 軸旋轉）
    #     marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
    #     marker_quat[:, 0] = torch.cos(angle / 2)  # w
    #     marker_quat[:, 3] = torch.sin(angle / 2)  # z
    #     
    #     self.goal_visualizer.visualize(marker_pos, marker_quat)


# ============================================================================
# 目標位置命令配置類
# ============================================================================
# 定義目標生成的參數（距離範圍、角度範圍、可視化選項等）
# ============================================================================

@configclass
class GoalCommandCfg(CommandTermCfg):
    """目標位置命令配置
    
    這個配置類定義了目標生成的參數：
    1. 距離範圍：目標離機器人多遠
    2. 角度範圍：目標在機器人的哪個方向
    3. 可視化選項：是否顯示綠色箭頭
    4. 重新採樣時間：何時生成新目標
    """

    # ------------------------------------------------------------------------
    # 基本配置
    # ------------------------------------------------------------------------
    class_type: type = GoalCommand
    # 關聯的命令類（Isaac Lab 會用這個來創建實例）

    asset_name: str = MISSING
    # 機器人資產名稱（必填）
    # MISSING 表示必須在配置時提供
    # 在 charge_env_cfg.py 中會設為 "robot"

    resampling_time_range: tuple[float, float] = MISSING
    # 重新採樣時間範圍（秒）（必填）
    # (最小時間, 最大時間)：何時生成新目標
    # 
    # 例子：
    #   (10, 20)：每 10-20 秒生成新目標（移動目標）
    #   (1e10, 1e10)：極大值，只在 reset 時生成（靜態目標，當前使用）

    debug_vis: bool = True
    # 是否顯示可視化標記（綠色箭頭）
    # True = 顯示（調試和展示時用）
    # False = 不顯示（訓練時可以關閉以提升性能）

    # ------------------------------------------------------------------------
    # 目標生成範圍配置
    # ------------------------------------------------------------------------
    @configclass
    class Ranges:
        """目標生成範圍
        
        定義目標位置的生成範圍（極座標形式）。
        """
        distance: tuple[float, float] = MISSING
        # 距離範圍（米）（必填）
        # (最小距離, 最大距離)
        # 
        # 例子：
        #   (1.5, 3.0)：目標在 1.5-3.0 米範圍內
        #   (0.5, 5.0)：目標在 0.5-5.0 米範圍內（更寬的範圍）
        # 
        # 設計考量：
        #   - 太近（<1米）：任務太簡單，機器人不需要移動
        #   - 太遠（>5米）：任務太難，可能被障礙物擋住
        #   - 1.5-3.0 米：平衡難度和可行性

        angle: tuple[float, float] = MISSING
        # 角度範圍（弧度）（必填）
        # (最小角度, 最大角度)
        # 
        # 例子：
        #   (-π, π)：全方位（360度，當前使用）
        #   (-π/2, π/2)：只在前方 180 度（更簡單）
        #   (0, π/2)：只在右前方 90 度（最簡單）
        # 
        # 角度定義（機器人座標系）：
        #   0 弧度 = 正前方（X 軸正方向）
        #   π/2 弧度 = 左側（Y 軸正方向）
        #   -π/2 弧度 = 右側（Y 軸負方向）
        #   π 弧度 = 正後方（X 軸負方向）

    ranges: Ranges = MISSING
    # 目標生成範圍配置（必填）
    # 在 charge_env_cfg.py 中會這樣設定：
    #   ranges=GoalCommandCfg.Ranges(
    #       distance=(1.5, 3.0),
    #       angle=(-math.pi, math.pi),
    #   )

    # ------------------------------------------------------------------------
    # 碰撞檢查配置（避免目標生成在障礙物或牆壁附近）
    # ------------------------------------------------------------------------
    wall_boundary: float = 5.0
    # 牆壁邊界距離（米）
    # 目標不會生成在 |X| > boundary 或 |Y| > boundary 的位置

    wall_safe_margin: float = 0.5
    # 牆壁安全邊距（米）
    # 目標與牆壁的最小距離：goal 必須在 |X|, |Y| < (boundary - margin) 範圍內

    obstacle_safe_distance: float = 1.0
    # 障礙物安全距離（米）
    # 目標與任何障礙物中心的最小距離

    max_resample_attempts: int = 50
    # 最大重試次數
    # 如果連續 N 次生成的目標都不合法，則放寬條件或使用最後一次生成的位置

    num_obstacles: int = 10
    # 障礙物數量（用於碰撞檢查）
