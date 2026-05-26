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
        # 創建可視化標記（綠色箭頭）- 批量可視化版本
        # ------------------------------------------------------------------------
        if self.cfg.debug_vis:
            # 🔥 使用單個可視化器實例，批量繪製所有環境的箭頭
            # 這是 Isaac Lab 的標準做法，避免為每個環境創建獨立的 USD prim
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/GoalCommand"
            marker_cfg.markers["arrow"].scale = (1.0, 1.0, 1.0)
            marker_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.goal_visualizer = VisualizationMarkers(marker_cfg)

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
        # 牆壁 proximity 檢查工具 (per-env)
        from .mdp.wall_layout import check_wall_proximity_perenv, get_combined_wall_data

        # 獲取需要重置的環境數量
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        
        # 將 env_ids 轉為 tensor
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.detach().clone().to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # 獲取配置參數（支援非對稱邊界）
        _wb = self.cfg.wall_boundary
        if isinstance(_wb, (list, tuple)):
            wall_boundary_x, wall_boundary_y = _wb
        else:
            wall_boundary_x = wall_boundary_y = _wb
        wall_safe_margin = self.cfg.wall_safe_margin
        num_obstacles = self.cfg.num_obstacles
        max_attempts = self.cfg.max_resample_attempts

        # 障礙物越密集 → 自動縮小安全距離，避免找不到合法位置
        base_safe_dist = self.cfg.obstacle_safe_distance
        if num_obstacles > 30:
            # 30→0.8, 50→0.6, 70→0.5, 100→0.5
            obstacle_safe_distance = max(0.5, base_safe_dist - 0.01 * (num_obstacles - 30))
        else:
            obstacle_safe_distance = base_safe_dist
        
        # 獲取環境原點（每個環境的座標偏移）
        env_origins = self._env.scene.env_origins[env_ids_tensor, :2]  # [num_envs, 2]
        
        # 獲取機器人在環境局部座標系中的位置
        robot_pos_world = self.robot.data.root_pos_w[env_ids_tensor, :2]  # [num_envs, 2]
        robot_pos_local = robot_pos_world - env_origins  # [num_envs, 2]
        
        # ------------------------------------------------------------------------
        # 收集所有障礙物位置和大小（環境局部座標系）
        # ------------------------------------------------------------------------
        obstacle_positions = []
        obstacle_radii = []
        
        # 獲取障礙物尺寸（如果可用）
        from .mdp.events.state import get_obstacle_sizes
        obstacle_sizes = get_obstacle_sizes()
        
        for i in range(num_obstacles):
            try:
                obstacle = self._env.scene[f"obstacle_{i}"]
                obs_pos_world = obstacle.data.root_pos_w[env_ids_tensor, :2]  # [num_envs, 2]
                obs_pos_local = obs_pos_world - env_origins  # [num_envs, 2]
                obstacle_positions.append(obs_pos_local)
                
                # 獲取障礙物半徑（如果可用）
                if obstacle_sizes is not None and i < len(obstacle_sizes):
                    # obstacle_sizes[i] 是直徑（size_scalar = radius * 2.0）
                    radius = obstacle_sizes[i] / 2.0  # 轉換為半徑
                else:
                    # 默認半徑：使用最大可能半徑（0.5 米）作為保守估計
                    radius = 0.5
                obstacle_radii.append(radius)
            except (KeyError, AttributeError):
                # 障礙物不存在，跳過
                continue
        
        # 將障礙物位置堆疊為 [num_envs, num_valid_obstacles, 2]
        if obstacle_positions:
            all_obstacles = torch.stack(obstacle_positions, dim=1)  # [num_envs, num_obstacles, 2]
            # 將障礙物半徑轉換為張量 [num_valid_obstacles]
            all_obstacle_radii = torch.tensor(obstacle_radii, device=self.device, dtype=torch.float32)
        else:
            all_obstacles = None
            all_obstacle_radii = None
        
        # ------------------------------------------------------------------------
        # per-env 牆壁數據（迷宮內部 + 邊界牆壁 proximity 檢查）
        # ------------------------------------------------------------------------
        wall_c_all, wall_s_all, wall_mask_all = get_combined_wall_data(self._env)
        # 索引需要重新採樣的 env
        wall_c = wall_c_all[env_ids_tensor]      # [num_envs, W, 2]
        wall_s = wall_s_all[env_ids_tensor]      # [num_envs, W, 2]
        wall_m = wall_mask_all[env_ids_tensor]   # [num_envs, W]

        # ------------------------------------------------------------------------
        # 為每個環境生成有效的目標位置
        # ------------------------------------------------------------------------
        # 初始化：標記哪些環境還需要生成有效目標
        needs_resample = torch.ones(num_envs, device=self.device, dtype=torch.bool)

        # 儲存候選目標位置（環境局部座標系）
        candidate_goals_local = torch.zeros(num_envs, 2, device=self.device)

        # 追蹤每個 env 歷史最佳候選（離障礙物最遠的）
        best_goals_local = torch.zeros(num_envs, 2, device=self.device)
        best_min_clearance = torch.full((num_envs,), -1.0, device=self.device)

        # 預計算常數（避免每次迴圈重複存取）
        d_min, d_max = self.cfg.ranges.distance[0], self.cfg.ranges.distance[1]
        a_min, a_max = self.cfg.ranges.angle[0], self.cfg.ranges.angle[1]
        valid_boundary_x = wall_boundary_x - wall_safe_margin
        valid_boundary_y = wall_boundary_y - wall_safe_margin

        # 預計算障礙物 required_distances（不隨迴圈變化）
        if all_obstacles is not None and all_obstacle_radii is not None:
            obstacle_radii_expanded = all_obstacle_radii.unsqueeze(0).expand(num_envs, -1)
            required_distances = obstacle_safe_distance + obstacle_radii_expanded
        else:
            required_distances = None

        for attempt in range(max_attempts):
            if not needs_resample.any():
                break  # 所有環境都已生成有效目標

            # 對全部 envs 生成隨機距離和角度（GPU 利用率低，不需節省）
            distance = torch.rand(num_envs, device=self.device) * (d_max - d_min) + d_min
            angle = torch.rand(num_envs, device=self.device) * (a_max - a_min) + a_min

            # 計算候選目標位置（環境局部座標系）
            goal_x = robot_pos_local[:, 0] + distance * torch.cos(angle)
            goal_y = robot_pos_local[:, 1] + distance * torch.sin(angle)

            # 只更新仍需重新採樣的環境
            candidate_goals_local[needs_resample, 0] = goal_x[needs_resample]
            candidate_goals_local[needs_resample, 1] = goal_y[needs_resample]

            # ------------------------------------------------------------------------
            # 檢查 1：牆壁邊界（環境局部座標系）
            # ------------------------------------------------------------------------
            within_walls = (
                (candidate_goals_local[:, 0].abs() < valid_boundary_x) &
                (candidate_goals_local[:, 1].abs() < valid_boundary_y)
            )
            
            # ------------------------------------------------------------------------
            # 檢查 2：障礙物碰撞（考慮障礙物半徑）
            # ------------------------------------------------------------------------
            if all_obstacles is not None:
                goal_expanded = candidate_goals_local.unsqueeze(1)  # [num_envs, 1, 2]
                distances_to_obstacle_centers = torch.norm(goal_expanded - all_obstacles, dim=2)  # [num_envs, num_obstacles]

                if required_distances is not None:
                    clear_of_obstacles = (distances_to_obstacle_centers > required_distances).all(dim=1)
                else:
                    conservative_distance = obstacle_safe_distance + 0.5
                    clear_of_obstacles = distances_to_obstacle_centers.min(dim=1)[0] > conservative_distance
            else:
                clear_of_obstacles = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            
            # ------------------------------------------------------------------------
            # 檢查 3：與機器人的距離（確保不會生成在機器人身上）
            # ------------------------------------------------------------------------
            dist_to_robot = torch.norm(candidate_goals_local - robot_pos_local, dim=1)
            clear_of_robot = dist_to_robot > 0.5  # 至少離機器人 0.5 米
            
            # ------------------------------------------------------------------------
            # 檢查 4：per-env 迷宮牆壁（確保目標不在牆壁內或過近）
            # ------------------------------------------------------------------------
            clear_of_walls = ~check_wall_proximity_perenv(
                candidate_goals_local, wall_c, wall_s, wall_m, wall_safe_margin
            )

            # ------------------------------------------------------------------------
            # 綜合判斷：所有條件都滿足
            # ------------------------------------------------------------------------
            valid_goals = within_walls & clear_of_obstacles & clear_of_robot & clear_of_walls

            # 追蹤歷史最佳候選（即使不合法，也記錄離障礙物最遠的位置）
            if all_obstacles is not None:
                goal_exp = candidate_goals_local.unsqueeze(1)  # [N, 1, 2]
                dists = torch.norm(goal_exp - all_obstacles, dim=2)  # [N, num_obs]
                if all_obstacle_radii is not None:
                    dists = dists - all_obstacle_radii.unsqueeze(0)  # 扣除半徑
                min_clearance = dists.min(dim=1).values  # [N]
            else:
                min_clearance = torch.full((num_envs,), 999.0, device=self.device)
            # 在邊界內 & 離機器人夠遠 & 不在牆壁內的候選才有資格當 best
            eligible = within_walls & clear_of_robot & clear_of_walls
            improved = eligible & (min_clearance > best_min_clearance)
            best_goals_local[improved] = candidate_goals_local[improved]
            best_min_clearance[improved] = min_clearance[improved]

            # 更新需要重新採樣的環境
            needs_resample = needs_resample & (~valid_goals)

        # ------------------------------------------------------------------------
        # 最終處理：如果還有環境沒有找到有效目標，使用最佳候選
        # ------------------------------------------------------------------------
        if needs_resample.any():
            n_failed = needs_resample.sum().item()
            failed_envs = needs_resample
            # 使用歷史最佳候選（離障礙物最遠且不在牆壁內的位置）
            has_best = failed_envs & (best_min_clearance >= 0)
            best_clr = best_min_clearance[has_best].min().item() if has_best.any() else -1
            print(f"[WARN] Goal resample fallback: {n_failed}/{num_envs} envs "
                  f"failed after {max_attempts} attempts. "
                  f"Using best candidate (min clearance={best_clr:.2f}m, "
                  f"safe_dist={obstacle_safe_distance:.2f}m)")
            if has_best.any():
                candidate_goals_local[has_best] = best_goals_local[has_best]
            else:
                # 所有嘗試都落在牆壁內（T 走廊等窄場景常見）→ 放在機器人附近
                candidate_goals_local[failed_envs] = robot_pos_local[failed_envs] + 1.0
                print(f"[WARN] No valid best candidate found, placing goal near robot")
            # 兜底: clamp 到邊界內
            candidate_goals_local[failed_envs, 0] = torch.clamp(
                candidate_goals_local[failed_envs, 0],
                -valid_boundary_x, valid_boundary_x
            )
            candidate_goals_local[failed_envs, 1] = torch.clamp(
                candidate_goals_local[failed_envs, 1],
                -valid_boundary_y, valid_boundary_y
            )
            # 最後安全檢查：fallback 候選仍在牆壁內 → 強制放回機器人前方 1m
            still_in_wall = check_wall_proximity_perenv(
                candidate_goals_local, wall_c, wall_s, wall_m, wall_safe_margin
            )
            if (failed_envs & still_in_wall).any():
                stuck = failed_envs & still_in_wall
                candidate_goals_local[stuck] = robot_pos_local[stuck] + 1.0
                print(f"[WARN] {stuck.sum().item()} envs fallback still in wall, "
                      f"placing goal 1m from robot")
        
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
        # 步驟 3.5：重新放置 near-goal 障礙物
        # ------------------------------------------------------------------------
        # _reset_idx 中 event_manager.apply("reset") 在 command_manager.reset()
        # 之前執行，導致 BehaviorScheduler._place_near_goal() 讀到的是上一局
        # 的 goal 位置。在此處 goal 已更新後，重新觸發放置。
        self._relocate_near_goal_obstacles(env_ids)

        # ------------------------------------------------------------------------
        # 步驟 4：更新可視化標記
        # ------------------------------------------------------------------------
        if self.cfg.debug_vis:
            self._update_goal_markers()
            # 如果啟用可視化，更新綠色箭頭的位置

        return {}  # 返回空字典（父類接口要求）

    def _relocate_near_goal_obstacles(self, env_ids) -> None:
        """Goal 位置更新後，重新放置 near-goal 障礙物。

        解決 _reset_idx 中 event_manager 先於 command_manager 的時序問題。
        """
        sched = getattr(self._env, '_behavior_scheduler', None)
        if sched is None or sched.obs_near_goal_count <= 0:
            return

        import torch
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        sched._place_near_goal(env_ids, self._env)
        sched._enforce_spawn_constraints(env_ids, self._env)
        sched._write_positions_to_sim(self._env)

    # ------------------------------------------------------------------------
    # 可視化方法
    # ------------------------------------------------------------------------
    def _update_goal_markers(self):
        """更新目標位置的可視化標記（綠色箭頭）- 批量可視化版本

        使用單個可視化器實例批量繪製所有環境的箭頭（Isaac Lab 標準做法）。
        """
        if not hasattr(self, "goal_visualizer"):
            return

        # 批量繪製所有環境的箭頭
        marker_pos = self.goal_pos_w.clone()  # [num_envs, 3]
        marker_pos[:, 2] = 0.1  # 抬高一點（避免陷入地面）

        # 設定箭頭朝向（不旋轉，保持預設方向）
        marker_quat = torch.zeros(self.num_envs, 4, device=self.device)
        marker_quat[:, 0] = 1.0  # w=1, x=y=z=0（單位四元數）

        # 批量可視化
        self.goal_visualizer.visualize(marker_pos, marker_quat)

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
    # True = 顯示（GUI 模式預設）
    # False = 不顯示（headless 訓練時自動禁用）

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
    wall_boundary: float | tuple[float, float] = 5.0
    # 牆壁邊界距離（米）
    # float → 對稱 (x, y 同值)
    # (float, float) → 非對稱 (boundary_x, boundary_y)，用於 T 走廊等非正方形場景
    # 目標不會生成在 |X| > boundary_x 或 |Y| > boundary_y 的位置

    wall_safe_margin: float = 0.5
    # 牆壁安全邊距（米）
    # 目標與牆壁的最小距離：goal 必須在 |X|, |Y| < (boundary - margin) 範圍內

    obstacle_safe_distance: float = 1.0
    # 障礙物安全距離（米）
    # 目標與任何障礙物中心的最小距離

    max_resample_attempts: int = 50
    # 最大重試次數
    # 密集障礙物場景需要更多嘗試；fallback 使用歷史最佳候選（離障礙最遠的位置）

    num_obstacles: int = 10
    # 障礙物數量（用於碰撞檢查）
