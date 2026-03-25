# ============================================================================
# 差速驅動動作類
# ============================================================================
"""差速驅動動作實現

此模組實現了差速驅動(Differential Drive)控制:
- 兩輪獨立控制,類似坦克移動
- 輸入: [前進速度, 旋轉速度] 範圍 [-1, 1]
- 輸出: 機器人實際速度(帶加速度限制和平滑控制)

主要特點:
1. 動態加速度限制(根據與目標的距離自動調整)
2. 速度平滑控制(避免突然加減速)
3. 座標系轉換(機器人本地座標 → 世界座標)
4. Debug 可視化(顯示目標速度和當前速度箭頭)
"""

from __future__ import annotations

import torch
from typing import Sequence

# Isaac Lab 工具
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass


class DifferentialDriveAction(ActionTerm):
    """差速驅動動作類
    
    差速驅動（Differential Drive）：
    - 兩個輪子獨立控制（像坦克）
    - 輸入：[前進速度, 旋轉速度]
    - 不是：[左輪速度, 右輪速度]（這是更底層的控制）
    
    這個類接收神經網絡的輸出 [-1, 1]，轉換成實際的速度指令。
    """
    
    # 類型提示
    cfg: DifferentialDriveActionCfg  # 動作配置
    _asset: Articulation  # 機器人資產（可移動的關節物體）
    
    def __init__(self, cfg: DifferentialDriveActionCfg, env):
        """初始化動作項
        
        Args:
            cfg: 動作配置（包含速度限制等參數）
            env: 環境實例（用於訪問場景和設備）
        """
        super().__init__(cfg, env)  # 調用父類初始化
        
        # 獲取機器人資產
        self._asset = env.scene[cfg.asset_name]  # 從場景中獲取名為 "robot" 的資產
        
        # 初始化動作緩存（存儲動作數據）
        # 為什麼用 torch.zeros？因為需要在 GPU 上運行（快！）
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device)
        # shape: [num_envs, 2]
        # 例如：[128, 2] = 128 個環境，每個環境 2 個動作值
        # 存儲原始動作：神經網絡輸出的 [-1, 1] 範圍的值
        
        self._processed_actions = torch.zeros(env.num_envs, 2, device=self.device)
        # shape: [num_envs, 2]
        # 存儲處理後的動作：實際的速度（m/s 和 rad/s）
        
        # 初始化上一步的速度（用於加速度限制）
        self._prev_velocity = torch.zeros(env.num_envs, 2, device=self.device)
        # shape: [num_envs, 2]
        # 存儲上一步的速度：[線速度, 角速度]
        # 用於計算速度變化率，限制加速度
    
    # ------------------------------------------------------------------------
    # 屬性方法（Properties）
    # ------------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """動作維度
        
        Returns:
            2：[前進速度, 旋轉速度]
        """
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """原始動作（神經網絡輸出）
        
        Returns:
            shape [num_envs, 2]：範圍 [-1, 1] 的動作值
        """
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """處理後的動作（實際速度）
        
        Returns:
            shape [num_envs, 2]：實際的 [線速度 m/s, 角速度 rad/s]
        """
        return self._processed_actions
    
    # ------------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------------
    def process_actions(self, actions: torch.Tensor):
        """處理動作：將神經網絡輸出轉換為實際速度（帶加速度限制）
        
        神經網絡輸出 [-1, 1] → 實際速度 [m/s, rad/s]
        
        這個方法會：
        1. 將動作縮放到目標速度
        2. 限制速度變化率（加速度限制）
        3. 確保速度變化不超過最大加速度
        
        Args:
            actions: shape [num_envs, 2]，範圍 [-1, 1]
                     actions[:, 0] = 前進動作（-1=最快後退, +1=最快前進）
                     actions[:, 1] = 旋轉動作（-1=最快右轉, +1=最快左轉）
        
        數學公式：
            目標速度 = 動作值 × 最大速度
            速度變化 = clamp(目標速度 - 上一步速度, -最大加速度, +最大加速度)
            實際速度 = 上一步速度 + 速度變化
        """
        # 保存原始動作
        self._raw_actions[:] = actions
        # [:] 表示原地賦值（不創建新張量，節省記憶體）
        
        # ====================================================================
        # 步驟 1：計算目標速度（神經網絡期望的速度）
        # ====================================================================
        target_velocity = torch.zeros_like(self._processed_actions)
        target_velocity[:, 0] = actions[:, 0] * self.cfg.max_linear_velocity
        # 例如：actions[:, 0] = 0.5, max_linear_velocity = 1.5
        #       → 目標前進速度 = 0.5 × 1.5 = 0.75 m/s
        
        target_velocity[:, 1] = actions[:, 1] * self.cfg.max_angular_velocity
        # 例如：actions[:, 1] = -1.0, max_angular_velocity = 1.5
        #       → 目標旋轉速度 = -1.0 × 1.5 = -1.5 rad/s（左轉）
        
        # ====================================================================
        # 步驟 2：計算速度變化（加速度限制，根據距離動態調整）
        # ====================================================================
        # 獲取控制頻率（用於計算每步的最大速度變化）
        # 控制頻率 = 1 / (decimation * sim.dt)
        # 例如：decimation=4, sim.dt=0.01 → 控制頻率 = 25 Hz → dt_control = 0.04 秒
        dt_control = self._env.cfg.decimation * self._env.cfg.sim.dt
        
        # 計算速度變化
        velocity_change = target_velocity - self._prev_velocity
        # shape: [num_envs, 2]
        # 例如：目標速度 = [1.5, 0.5], 上一步速度 = [0.0, 0.0]
        #       → 速度變化 = [1.5, 0.5]
        
        # ====================================================================
        # 根據距離目標的遠近動態調整加速度限制
        # ====================================================================
        # 在接近目標時，降低加速度限制（更平滑的運動）
        # 在遠離目標時，使用正常加速度限制（快速響應）
        if self.cfg.max_linear_acceleration is not None:
            # 獲取目標距離（用於動態調整加速度）
            try:
                goal_pos_w = self._env.command_manager.get_command("goal_command")
                robot_pos_w = self._asset.data.root_pos_w[:, :2]
                goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
                
                # 動態調整線性加速度限制
                # 距離 < 2.0 米：降低加速度到 50%（更平滑）
                # 距離 > 4.0 米：使用正常加速度（快速響應）
                # 中間距離：線性插值
                close_distance = 2.0
                far_distance = 4.0
                accel_scale = torch.clamp(
                    (goal_distance - close_distance) / (far_distance - close_distance + 1e-6),
                    min=0.5,  # 最小縮放：50%（接近目標時）
                    max=1.0   # 最大縮放：100%（遠離目標時）
                )
                dynamic_max_linear_accel = self.cfg.max_linear_acceleration * accel_scale
            except:
                # 如果無法獲取目標距離，使用默認加速度
                dynamic_max_linear_accel = self.cfg.max_linear_acceleration
            
            max_linear_change = dynamic_max_linear_accel * dt_control
            velocity_change[:, 0] = torch.clamp(
                velocity_change[:, 0],
                -max_linear_change,
                max_linear_change
            )
            # 例如：接近目標時（距離 < 2米）
            #       max_linear_acceleration = 2.0 m/s² → 動態調整為 1.0 m/s²
            #       dt_control = 0.04 秒 → max_linear_change = 0.04 m/s（更平滑）
            #       遠離目標時（距離 > 4米）
            #       max_linear_acceleration = 2.0 m/s² → 動態調整為 2.0 m/s²
            #       dt_control = 0.04 秒 → max_linear_change = 0.08 m/s（正常）
        
        if self.cfg.max_angular_acceleration is not None:
            # 角加速度也根據距離動態調整
            try:
                goal_pos_w = self._env.command_manager.get_command("goal_command")
                robot_pos_w = self._asset.data.root_pos_w[:, :2]
                goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
                
                # 動態調整角加速度限制
                # 接近目標時降低角加速度（更平滑的旋轉）
                close_distance = 2.0
                far_distance = 4.0
                accel_scale = torch.clamp(
                    (goal_distance - close_distance) / (far_distance - close_distance + 1e-6),
                    min=0.6,  # 最小縮放：60%（接近目標時）
                    max=1.0   # 最大縮放：100%（遠離目標時）
                )
                dynamic_max_angular_accel = self.cfg.max_angular_acceleration * accel_scale
            except:
                dynamic_max_angular_accel = self.cfg.max_angular_acceleration
            
            max_angular_change = dynamic_max_angular_accel * dt_control
            velocity_change[:, 1] = torch.clamp(
                velocity_change[:, 1],
                -max_angular_change,
                max_angular_change
            )
        
        # ====================================================================
        # 步驟 3：計算實際速度（上一步速度 + 限制後的速度變化）
        # ====================================================================
        self._processed_actions[:] = self._prev_velocity + velocity_change
        
        # 確保速度不超過最大速度限制（雙重保險）
        self._processed_actions[:, 0] = torch.clamp(
            self._processed_actions[:, 0],
            -self.cfg.max_linear_velocity,
            self.cfg.max_linear_velocity
        )
        self._processed_actions[:, 1] = torch.clamp(
            self._processed_actions[:, 1],
            -self.cfg.max_angular_velocity,
            self.cfg.max_angular_velocity
        )
        
        # ====================================================================
        # 步驟 4：更新上一步速度（用於下一次計算）
        # ====================================================================
        self._prev_velocity[:] = self._processed_actions
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """重置動作狀態（環境重置時調用）
        
        當環境重置時，需要重置上一步的速度，確保新 episode 從零速度開始。
        
        Args:
            env_ids: 需要重置的環境 ID 列表。如果為 None，則重置所有環境。
        """
        if env_ids is None:
            # 重置所有環境
            self._prev_velocity[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            # 只重置指定的環境
            # 將 env_ids 轉換為張量索引
            if isinstance(env_ids, slice):
                self._prev_velocity[env_ids] = 0.0
                self._processed_actions[env_ids] = 0.0
            else:
                # 轉換為張量索引
                if isinstance(env_ids, torch.Tensor):
                    env_ids_tensor = env_ids.detach().clone().to(device=self.device, dtype=torch.long)
                else:
                    env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                self._prev_velocity[env_ids_tensor] = 0.0
                self._processed_actions[env_ids_tensor] = 0.0
    
    def apply_actions(self):
        """應用動作：將速度指令發送到模擬器
        
        這是差速驅動的核心實現！
        
        步驟：
        1. 獲取機器人當前朝向（四元數）
        2. 構造本地座標系的速度向量
        3. 轉換到世界座標系
        4. 應用到機器人
        
        為什麼需要座標轉換？
        - 神經網絡輸出的是「往前走 1 m/s」（相對於機器人）
        - 但物理引擎需要「往東走 0.7 m/s，往北走 0.7 m/s」（世界座標）
        """
        # ====================================================================
        # 步驟 1：獲取機器人當前朝向
        # ====================================================================
        robot_quat_w = self._asset.data.root_quat_w
        # shape: [num_envs, 4]
        # 四元數 (w, x, y, z)：表示機器人的旋轉姿態
        # 例如：(1, 0, 0, 0) = 不旋轉
        #       (0.707, 0, 0, 0.707) = 繞 Z 軸旋轉 90 度
        
        # ====================================================================
        # 步驟 2：構造本地座標系的速度向量
        # ====================================================================
        num_envs = self._env.num_envs
        local_velocity = torch.zeros(num_envs, 3, device=self.device)
        # shape: [num_envs, 3] = [X, Y, Z] 速度
        # 初始化為零向量
        
        local_velocity[:, 0] = self._processed_actions[:, 0]  # X 軸是前進方向
        # 在機器人座標系中：
        # X 軸 = 前方（車頭方向）
        # Y 軸 = 左側
        # Z 軸 = 上方
        # 
        # 例如：processed_actions[:, 0] = 1.5 m/s
        #       → local_velocity = [1.5, 0, 0]（往前走 1.5 m/s）
        
        # ====================================================================
        # 步驟 3：轉換到世界座標系
        # ====================================================================
        global_linear_velocity = math_utils.quat_apply(robot_quat_w, local_velocity)
        # quat_apply：用四元數旋轉向量（quat_rotate 的更快替代方法）
        # 
        # 例子：
        #   機器人朝向東北（45度）
        #   本地速度 = [1.0, 0, 0]（往前 1 m/s）
        #   ↓ 旋轉變換
        #   世界速度 = [0.707, 0.707, 0]（往東 0.707 + 往北 0.707）
        
        # ====================================================================
        # 步驟 4：構造完整的 6D 速度向量
        # ====================================================================
        root_velocity = torch.zeros(num_envs, 6, device=self.device)
        # shape: [num_envs, 6] = [vx, vy, vz, wx, wy, wz]
        # 前 3 個：線速度（平移）
        # 後 3 個：角速度（旋轉）
        
        root_velocity[:, 0:3] = global_linear_velocity  # 設定全局線速度
        # 例如：[0.707, 0.707, 0] = 往東北方移動
        
        root_velocity[:, 5] = self._processed_actions[:, 1]  # 設定全局 Z 軸角速度
        # Z 軸角速度 = 繞垂直軸旋轉（偏航角/Yaw）
        # 正值 = 逆時針旋轉（左轉）
        # 負值 = 順時針旋轉（右轉）
        # 
        # 注意：這裡不需要座標轉換，因為 Z 軸在世界座標系和本地座標系都是垂直向上
        
        # ====================================================================
        # 步驟 5：應用速度到模擬器
        # ====================================================================
        self._asset.write_root_velocity_to_sim(root_velocity)
        # 直接設定機器人的速度（高階控制）
        # 物理引擎會自動計算輪子需要轉多快
        # 
        # 這比直接控制每個輪子簡單得多！

    # ------------------------------------------------------------------------
    # 可視化方法（Debug Visualization）
    # ------------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        """設置調試可視化的可見性
        
        當 debug_vis=True 時，創建並顯示速度箭頭：
        - 綠色箭頭：目標速度命令（神經網絡輸出的速度）
        - 藍色箭頭：當前實際速度（機器人正在執行的速度）
        
        Args:
            debug_vis: 是否啟用可視化
        """
        if debug_vis:
            # 創建標記（如果尚未創建）
            if not hasattr(self, "vel_goal_visualizer"):
                # 綠色箭頭：目標速度命令
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                
                # 藍色箭頭：當前實際速度
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.vel_current_visualizer = VisualizationMarkers(marker_cfg)
            
            # 設置可見性為 True
            self.vel_goal_visualizer.set_visibility(True)
            self.vel_current_visualizer.set_visibility(True)
        else:
            # 設置可見性為 False
            if hasattr(self, "vel_goal_visualizer"):
                self.vel_goal_visualizer.set_visibility(False)
                self.vel_current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """更新可視化標記的回調函數
        
        這個方法在每個渲染幀都會被調用，用於更新箭頭的位置和方向。
        
        Args:
            event: 渲染事件（由 Isaac Lab 提供）
        """
        # 檢查可視化器是否已創建
        if not hasattr(self, "vel_goal_visualizer"):
            return
        
        # 檢查機器人是否已初始化
        if not self._asset.is_initialized:
            return
        
        # 獲取機器人基座位置（箭頭顯示在基座上方）
        base_pos_w = self._asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5  # 抬高 0.5 米，避免被機器人遮擋
        
        # 計算目標速度箭頭的方向和長度
        # 目標速度是處理後的動作（實際速度命令）
        vel_goal_xy = self._processed_actions[:, 0:1]  # 只取線速度（前進速度）
        # 注意：差速驅動只有前進速度，沒有 Y 方向速度
        vel_goal_scale, vel_goal_quat = self._resolve_velocity_to_arrow(vel_goal_xy)
        
        # 計算當前速度箭頭的方向和長度
        # 當前速度是機器人實際的線速度（在世界座標系）
        robot_vel_w = self._asset.data.root_lin_vel_w[:, :2]  # XY 速度
        vel_current_scale, vel_current_quat = self._resolve_velocity_to_arrow(robot_vel_w)
        
        # 更新可視化標記
        self.vel_goal_visualizer.visualize(base_pos_w, vel_goal_quat, vel_goal_scale)
        self.vel_current_visualizer.visualize(base_pos_w, vel_current_quat, vel_current_scale)

    def _resolve_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """將 XY 速度轉換為箭頭的方向和縮放
        
        這個方法將速度向量轉換為箭頭的旋轉（四元數）和長度（縮放）。
        
        Args:
            xy_velocity: XY 速度向量，shape [num_envs, 1] 或 [num_envs, 2]
                        [num_envs, 1] = 只有 X 方向速度（前進速度）
                        [num_envs, 2] = X 和 Y 方向速度
        
        Returns:
            arrow_scale: 箭頭縮放，shape [num_envs, 3]
            arrow_quat: 箭頭旋轉（四元數），shape [num_envs, 4]
        """
        # 獲取默認縮放（使用配置中的縮放值，如果可視化器已創建則從中獲取，否則使用默認值）
        if hasattr(self, "vel_goal_visualizer"):
            default_scale = self.vel_goal_visualizer.cfg.markers["arrow"].scale
        else:
            default_scale = (0.5, 0.5, 0.5)  # 默認縮放值
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        
        # 計算速度大小（用於調整箭頭長度）
        if xy_velocity.shape[1] == 1:
            # 只有 X 方向速度（前進速度）
            vel_magnitude = torch.abs(xy_velocity[:, 0])
            # 根據速度符號決定方向：正數=前方（0度），負數=後方（180度）
            heading_angle = torch.where(
                xy_velocity[:, 0] >= 0,
                torch.zeros_like(xy_velocity[:, 0]),  # 前進：0 度（前方）
                torch.full_like(xy_velocity[:, 0], 3.14159),  # 後退：π 弧度（後方）
            )
        else:
            # 有 X 和 Y 方向速度
            vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
            heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        
        # 根據速度大小調整箭頭長度（X 軸方向）
        arrow_scale[:, 0] *= vel_magnitude * 3.0
        
        # 計算箭頭朝向（繞 Z 軸旋轉）
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # 轉換到世界座標系（考慮機器人的朝向）
        base_quat_w = self._asset.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        
        return arrow_scale, arrow_quat


# ----------------------------------------------------------------------------
# 動作配置類
# ----------------------------------------------------------------------------
@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """差速驅動動作配置
    
    這個配置類定義了動作的參數（速度限制和加速度限制）。
    """
    class_type: type = DifferentialDriveAction  # 關聯的動作類
    asset_name: str = "robot"  # 控制的資產名稱
    
    # 速度限制
    max_linear_velocity: float = 2.0  # 最大線速度：2.0 m/s
    max_angular_velocity: float = 2.0  # 最大角速度：2.0 rad/s
    
    # 加速度限制（新增）
    max_linear_acceleration: float | None = None  # 最大線性加速度：None = 無限制，或設置為 m/s²
    # 例如：2.0 m/s² 表示每秒最多改變 2.0 m/s 的速度
    # 如果設為 None，則不限制加速度（直接達到目標速度）
    
    max_angular_acceleration: float | None = None  # 最大角加速度：None = 無限制，或設置為 rad/s²
    # 例如：3.0 rad/s² 表示每秒最多改變 3.0 rad/s 的角速度
    
    debug_vis: bool = True  # 是否顯示調試可視化（速度箭頭）
