# ============================================================================
# 文件說明
# ============================================================================
"""
Charge 導航 MDP 函數

MDP = Markov Decision Process（馬爾可夫決策過程）
這個文件包含強化學習環境的核心實現：

1. 動作類（Actions）：機器人如何移動
2. 觀測函數（Observations）：機器人看到什麼
3. 獎勵函數（Rewards）：什麼行為得分/扣分
4. 終止條件（Terminations）：何時遊戲結束
5. 事件函數（Events）：重置時的操作

這些函數會被 charge_env_cfg.py 調用。
"""

# ============================================================================
# 模塊導入
# ============================================================================
from __future__ import annotations  # 允許使用字符串形式的類型提示（Python 3.7+）

import math  # 數學函數（用於角度計算）
import torch  # PyTorch：深度學習框架（用於張量運算）
from typing import TYPE_CHECKING, Optional, Sequence  # 類型檢查標記（避免循環導入）

# Isaac Lab 工具
import isaaclab.utils.math as math_utils  # 數學工具：四元數旋轉、座標變換
from isaaclab.assets import Articulation  # 關節機器人類
from isaaclab.managers import SceneEntityCfg  # 場景實體配置（引用場景中的物體）
from isaaclab.sensors import RayCaster, ContactSensor  # 射線投射器（雷達）類、接觸感測器類
from isaaclab.managers import ActionTerm, ActionTermCfg  # 動作項基類和配置
from isaaclab.markers import VisualizationMarkers  # 可視化標記類（顯示箭頭）
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,  # 藍色箭頭標記配置（用於顯示當前速度）
    GREEN_ARROW_X_MARKER_CFG,  # 綠色箭頭標記配置（用於顯示目標速度）
)
from isaaclab.utils import configclass  # 配置類裝飾器

# 類型檢查時才導入（避免運行時循環導入）
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# 障礙物設定（避免寫入 scene cfg）
# ============================================================================
_OBSTACLE_NUM: int | None = None
_OBSTACLE_SIZES: list[float] | None = None


def set_obstacle_metadata(num_obstacles: int, obstacle_sizes: list[float]) -> None:
    """保存障礙物數量與尺寸（供觀測與事件使用）"""
    global _OBSTACLE_NUM, _OBSTACLE_SIZES
    _OBSTACLE_NUM = num_obstacles
    _OBSTACLE_SIZES = obstacle_sizes


def _get_obstacle_num(default: int = 8) -> int:
    if _OBSTACLE_NUM is not None:
        return _OBSTACLE_NUM
    return default


def _get_obstacle_sizes() -> list[float] | None:
    return _OBSTACLE_SIZES


def get_obstacle_metadata():
    """獲取障礙物元數據（用於動態障礙物）"""
    return (_OBSTACLE_NUM or 3, _OBSTACLE_SIZES or [])


# ============================================================================
# 動態障礙物移動邏輯（Phase 2）
# ============================================================================
# 全局變量：記錄每個障礙物的移動狀態
_obstacle_start_positions: torch.Tensor | None = None  # 記錄移動起始位置
_obstacle_directions: torch.Tensor | None = None  # 記錄移動方向（角度，弧度）


def reset_obstacle_targets(env, env_ids: torch.Tensor, num_obstacles: int, boundary: float = 5.0):
    """重置障礙物的移動狀態（來回移動模式）"""
    global _obstacle_start_positions, _obstacle_directions
    
    if _obstacle_start_positions is None:
        _obstacle_start_positions = torch.zeros(env.num_envs, num_obstacles, 2, device=env.device)
    
    if _obstacle_directions is None:
        _obstacle_directions = torch.zeros(env.num_envs, num_obstacles, device=env.device)
    
    # 為每個障礙物隨機生成移動方向（0 到 2π）
    if _obstacle_start_positions is not None and _obstacle_directions is not None:
        for i in range(num_obstacles):
            # 隨機方向（0 到 2π）
            _obstacle_directions[env_ids, i] = torch.empty(len(env_ids), device=env.device).uniform_(0, 2 * math.pi)
            # 記錄起始位置（當前位置會在 reset_obstacles 中設置）


def reset_obstacles_with_targets(
    env,
    env_ids,
    num_obstacles: int = 10,
    boundary: float = 5.0,
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
):
    """重置障礙物位置並初始化移動狀態（Phase 2 動態障礙物）"""
    global _obstacle_start_positions
    
    # 先調用原有的 reset_obstacles 函數
    reset_obstacles(
        env,
        env_ids,
        speed_range=0.5,
        min_speed=0.05,
        min_robot_distance=min_robot_distance,
        min_goal_distance=min_goal_distance,
        min_obstacle_spacing=min_obstacle_spacing,
        max_spawn_attempts=max_spawn_attempts,
    )
    
    # 初始化移動方向
    reset_obstacle_targets(env, env_ids, num_obstacles, boundary)
    
    # 記錄重置後的起始位置
    if _obstacle_start_positions is not None:
        for i in range(num_obstacles):
            try:
                obstacle = env.scene[f"obstacle_{i}"]
                _obstacle_start_positions[env_ids, i] = obstacle.data.root_pos_w[env_ids, :2]
            except KeyError:
                continue


def move_obstacles_toward_target(
    env,
    env_ids: torch.Tensor | None,
    num_obstacles: int = 10,
    max_velocity: float = 0.5,
    boundary: float = 5.0,
    travel_distance: float = 3.0,
):
    """
    讓障礙物水平來回移動
    
    邏輯：
    1. 每個障礙物有一個移動方向（角度）
    2. 障礙物沿該方向移動，記錄從起始位置的距離
    3. 移動 3 米後，反轉方向（角度 + π）
    4. 繼續移動 3 米後，再次反轉
    5. 形成來回移動的模式
    
    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID（None 表示所有環境）
        num_obstacles: 障礙物數量
        max_velocity: 移動速度（m/s）
        boundary: 活動邊界（米），超出邊界時反轉方向
        travel_distance: 單向移動距離（米），移動此距離後反轉方向
    """
    global _obstacle_start_positions, _obstacle_directions
    
    # 處理 env_ids 參數
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    
    # 類型斷言：確保 env_ids 是 Tensor（消除 linter 警告）
    assert isinstance(env_ids, torch.Tensor)
    
    actual_count, _ = get_obstacle_metadata()
    num_obstacles = min(num_obstacles, actual_count)
    
    # 初始化移動狀態（如果還沒有）
    if _obstacle_start_positions is None or _obstacle_directions is None:
        reset_obstacle_targets(env, torch.arange(env.num_envs, device=env.device), actual_count, boundary)
        # 記錄起始位置
        if _obstacle_start_positions is not None:
            for i in range(actual_count):
                try:
                    obstacle = env.scene[f"obstacle_{i}"]
                    _obstacle_start_positions[:, i] = obstacle.data.root_pos_w[:, :2]
                except KeyError:
                    continue
    
    if _obstacle_start_positions is None or _obstacle_directions is None:
        return  # 如果初始化失敗，直接返回
    
    # 類型斷言：確保變量不為 None（消除 linter 警告）
    # 在上面的檢查後，這些變量必定不為 None
    assert _obstacle_start_positions is not None
    assert _obstacle_directions is not None
    
    # 創建局部引用並明確類型（解決 pyright 類型推斷問題）
    obstacle_start_positions: torch.Tensor = _obstacle_start_positions
    obstacle_directions: torch.Tensor = _obstacle_directions
    
    # ========================================================================
    # 獲取環境原點（用於將世界座標轉換為環境內相對座標）
    # ========================================================================
    # 重要：root_pos_w 是「世界座標」，但邊界檢查需要「環境內相對座標」
    # 例如：env_0 中心在 (0,0)，env_1 中心在 (15,0)（env_spacing=15）
    # 如果不轉換，env_1 的障礙物會被誤判為超出邊界
    env_origins = env.scene.env_origins[env_ids, :2]  # (len(env_ids), 2)
    
    for i in range(num_obstacles):
        try:
            obstacle = env.scene[f"obstacle_{i}"]
            
            # 獲取當前位置（世界座標）
            pos_world = obstacle.data.root_pos_w[env_ids, :2]  # (len(env_ids), 2)
            
            # 轉換為環境內相對座標（減去環境原點）
            pos_local = pos_world - env_origins  # (len(env_ids), 2)
            
            # 獲取移動狀態（使用局部引用）
            start_pos = obstacle_start_positions[env_ids, i]  # (len(env_ids), 2)
            direction_angle = obstacle_directions[env_ids, i]  # (len(env_ids),)
            
            # 計算從起始位置的位移（使用世界座標，因為起始位置也是世界座標）
            displacement = pos_world - start_pos
            
            # 計算沿移動方向的距離（投影）
            # 移動方向單位向量
            dir_vec = torch.stack([
                torch.cos(direction_angle),
                torch.sin(direction_angle)
            ], dim=-1)  # (len(env_ids), 2)
            
            # 投影距離
            travel_dist = torch.sum(displacement * dir_vec, dim=-1)  # (len(env_ids),)
            
            # 檢查是否需要反轉方向：
            # 1. 移動距離超過 travel_distance
            # 2. 超出邊界（使用環境內相對座標！）
            # 重要：使用 OR 合併條件，避免雙重反轉（方向加兩次 π 會導致方向不變）
            need_reverse_travel = (travel_dist.abs() >= travel_distance)
            out_of_bounds = (pos_local.abs() > boundary).any(dim=-1)  # 使用相對座標
            need_reverse = need_reverse_travel | out_of_bounds  # 合併條件，只反轉一次
            
            # 反轉方向（角度 + π）
            if need_reverse.any():
                obstacle_directions[env_ids[need_reverse], i] += math.pi
                # 更新起始位置為當前位置（新的移動起點，使用世界座標）
                obstacle_start_positions[env_ids[need_reverse], i] = pos_world[need_reverse]
            
            # 邊界保護：將超出邊界的障礙物位置 clamp 回邊界內
            # 重要：必須寫回模擬器，否則下次檢查仍然超出邊界
            if out_of_bounds.any():
                # 獲取超出邊界的環境 ID
                oob_env_ids = env_ids[out_of_bounds]
                oob_env_origins = env_origins[out_of_bounds]
                
                # 獲取當前完整位置和姿態（世界座標）
                full_pos = obstacle.data.root_pos_w[oob_env_ids, :3].clone()
                full_quat = obstacle.data.root_quat_w[oob_env_ids, :].clone()
                
                # 將世界座標轉換為相對座標，clamp，再轉回世界座標
                local_xy = full_pos[:, :2] - oob_env_origins
                local_xy = torch.clamp(local_xy, -boundary, boundary)
                full_pos[:, :2] = local_xy + oob_env_origins
                
                # 寫回模擬器
                obstacle.write_root_pose_to_sim(
                    torch.cat([full_pos, full_quat], dim=-1),
                    env_ids=oob_env_ids
                )
                # 更新起始位置為 clamp 後的位置（世界座標）
                obstacle_start_positions[oob_env_ids, i] = full_pos[:, :2]
            
            # 獲取當前方向（可能已經更新）
            current_direction = obstacle_directions[env_ids, i]
            
            # 計算速度向量
            velocity = torch.stack([
                torch.cos(current_direction) * max_velocity,
                torch.sin(current_direction) * max_velocity
            ], dim=-1)  # (len(env_ids), 2)
            
            # 設置速度 (vx, vy, vz, wx, wy, wz)
            vel_3d = torch.zeros(len(env_ids), 6, device=env.device)
            vel_3d[:, 0] = velocity[:, 0]  # vx
            vel_3d[:, 1] = velocity[:, 1]  # vy
            # vz, wx, wy, wz 保持為 0
            
            obstacle.write_root_velocity_to_sim(vel_3d, env_ids=env_ids)
            
        except KeyError:
            continue


# ============================================================================
# 動作實現（Actions）
# ============================================================================
# 定義機器人如何根據神經網絡輸出來移動
# ============================================================================

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
                     actions[:, 1] = 旋轉動作（-1=最快左轉, +1=最快右轉）
        
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


# ============================================================================
# 觀測函數（Observations）
# ============================================================================
# 定義機器人每一步能「感知」到什麼信息
# 這些函數返回的數據會被餵給神經網絡
# ============================================================================

def _check_finite(name: str, x: torch.Tensor, raise_on_error: bool = True) -> bool:
    """檢查張量是否包含 NaN/Inf 值（用於定位 PPO std>=0 錯誤）
    
    Args:
        name: 觀測項名稱（用於錯誤訊息）
        x: 要檢查的張量
        raise_on_error: 如果發現 NaN/Inf 是否立即拋出異常
    
    Returns:
        True 如果所有值都是有限的，False 如果有 NaN/Inf
    
    這個函數是修復 PPO std>=0 錯誤的關鍵工具。
    當 policy 的 std 變成 NaN/Inf 時，通常是因為觀測或獎勵中出現了異常值。
    """
    if not torch.isfinite(x).all():
        bad = ~torch.isfinite(x)
        num_bad = bad.sum().item()
        num_total = x.numel()
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        
        # 計算統計值（用 NaN/Inf 安全的方式）
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = x_safe.min().item()
        max_val = x_safe.max().item()
        
        error_msg = (
            f"[NaN/Inf Check] {name}: "
            f"shape={tuple(x.shape)} "
            f"bad_count={num_bad}/{num_total} "
            f"nan={nan_count} inf={inf_count} "
            f"min={min_val:.6f} max={max_val:.6f}"
        )
        
        if raise_on_error:
            raise RuntimeError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
        
        return False
    return True


def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """激光雷達掃描觀測（舊版 VLP16 版本 - 已註釋保留）
    
    從雷達獲取距離數據，並進行安全處理（防止 NaN、無窮大）。
    這是修復 PPO std>=0 錯誤的關鍵函數。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用場景中的雷達）
    
    Returns:
        shape [num_envs, num_rays]：歸一化的距離值 [0, 1]
        0 = 非常近（0米）
        1 = 最遠（max_distance）
    """
    # 獲取雷達傳感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # 計算從感測器位置到每個碰撞點的歐幾里得距離
    # sensor_pos.unsqueeze(1) 將 [num_envs, 3] 擴展為 [num_envs, 1, 3] 以便廣播
    # hit_points - sensor_pos.unsqueeze(1) 得到每個射線的位移向量 [num_envs, num_rays, 3]
    # torch.norm(..., dim=-1) 計算每個向量的長度（距離）[num_envs, num_rays]
    distances = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)
    
    # ========================================================================
    # 關鍵：NaN/Inf 消毒（防止 PPO std>=0 錯誤）
    # ========================================================================
    # 1) 把 NaN/Inf 替換為 max_distance（射線沒打到任何東西）
    max_range = sensor.cfg.max_distance
    distances = torch.nan_to_num(distances, nan=max_range, posinf=max_range, neginf=0.0)
    
    # 2) clip 到合理範圍 [0, max_range]
    distances = torch.clamp(distances, 0.0, max_range)
    
    # 3) normalize 到 [0, 1]（超重要，避免數值爆）
    normalized = distances / max_range
    
    # 4) 再次消毒（防止除法產生異常）
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # 5) 最終 finite 檢查（如果還有問題會立即報錯，方便定位）
    _check_finite("lidar_scan", normalized, raise_on_error=True)
    
    return normalized


def lidar_scan_2d_sweep(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """2D 平面掃描觀測（72 個角度）
    
    將 3D 點雲資料轉換為鳥瞰圖的可行駛區域後，計算每個角度的最近障礙物距離。
    設計：將機器人周圍 360 度切分為 72 個等份（每 5 度一格）。
    內容：每個角度紀錄「從機器狗中心點到最近障礙物的距離」。
    這代表機器人只知道「哪個方向多遠有牆」，而不知道那個牆是什麼顏色的。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用場景中的雷達）
    
    Returns:
        shape [num_envs, 72]：歸一化的距離值 [0, 1]
        0 = 非常近（0米）
        1 = 最遠（max_distance）
        每個值對應一個 5 度扇形區域的最近障礙物距離
    """
    # 獲取雷達傳感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # ========================================================================
    # 步驟 1：將 3D 點雲轉換為鳥瞰圖（2D 平面投影）
    # ========================================================================
    # 計算從感測器位置到每個碰撞點的 2D 平面距離（忽略高度 Z）
    # 只考慮 X-Y 平面的距離（鳥瞰圖視角）
    #
    # 為什麼使用 2D 投影而不是 3D 距離？
    # 1. 導航任務關心的是「水平方向能否通過」，而不是斜向距離
    # 2. 2D 投影直接對應機器人底盤的可通行空間
    # 3. 障礙物高度（0.8-1.5m）完全覆蓋雷達高度（0.5m），不存在「鑽過去」的情況
    # 4. 射線是水平發射的（vertical_fov_range=(0.0, 0.0)），高度差影響極小
    #
    # 注意：如果需要更精確的距離計算（例如：高低差異大的場景），可以改用：
    #   distances_3d = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)
    # 但對於當前的 2D 導航任務，2D 投影是更合適的選擇。
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2] 只取 X, Y 座標
    
    # 計算 2D 平面距離（從機器狗中心點到障礙物的水平距離）
    # sensor_pos_2d.unsqueeze(1) 將 [num_envs, 2] 擴展為 [num_envs, 1, 2] 以便廣播
    # hit_points_2d - sensor_pos_2d.unsqueeze(1) 得到每個射線的 2D 位移向量
    # torch.norm(..., dim=-1) 計算每個向量的長度（2D 距離）[num_envs, num_rays]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    
    # ========================================================================
    # 步驟 2：由於雷達已配置為 72 個角度（每 5 度一個），
    #         射線投射器會自動生成 72 條射線，每條射線對應一個角度。
    #         我們直接使用這些射線的 2D 距離數據即可。
    # ========================================================================
    # 注意：如果射線數不等於 72，我們需要重新分組（見下方備用實現）
    # 但通常情況下，配置為 72 個角度時，射線數應該就是 72
    num_rays = distances_2d.shape[1]
    num_sectors = 72  # 對應 horizontal_res=5.0（360/5=72）
    
    # 如果射線數正好是 72，直接使用（最簡單的情況）
    if num_rays == num_sectors:
        distances_2d_sectors = distances_2d
    else:
        # 備用實現：如果射線數不是 72，需要重新分組（通常不會進入這裡）
        # 計算每個射線的角度並分組到 72 個扇形區域
        vectors_2d = hit_points_2d - sensor_pos_2d.unsqueeze(1)  # [num_envs, num_rays, 2]
        angles = torch.atan2(vectors_2d[:, :, 1], vectors_2d[:, :, 0])  # [num_envs, num_rays]
        angles_deg = torch.rad2deg(angles + torch.pi)  # [num_envs, num_rays]，範圍 [0, 360]
        
        sector_size = 360.0 / num_sectors  # 5 度
        sector_indices = (angles_deg / sector_size).long()  # [num_envs, num_rays]
        sector_indices = torch.clamp(sector_indices, 0, num_sectors - 1)
        
        # 對每個扇形區域，找到最近的障礙物距離
        max_range = sensor.cfg.max_distance
        num_envs = distances_2d.shape[0]
        device = distances_2d.device
        
        distances_2d_sectors = torch.full(
            (num_envs, num_sectors),
            max_range,
            dtype=distances_2d.dtype,
            device=device
        )
        
        for sector_idx in range(num_sectors):
            mask = (sector_indices == sector_idx)  # [num_envs, num_rays]
            sector_distances = torch.where(
                mask,
                distances_2d,
                torch.full_like(distances_2d, float('inf'))
            )
            min_dist, _ = torch.min(sector_distances, dim=1)  # [num_envs]
            min_dist = torch.where(
                torch.isfinite(min_dist),
                min_dist,
                torch.full_like(min_dist, max_range)
            )
            distances_2d_sectors[:, sector_idx] = min_dist
    
    # ========================================================================
    # 步驟 3：安全處理和歸一化
    # ========================================================================
    max_range = sensor.cfg.max_distance
    
    # 1) 把 NaN/Inf 替換為 max_distance
    distances_2d_sectors = torch.nan_to_num(
        distances_2d_sectors, nan=max_range, posinf=max_range, neginf=0.0
    )
    
    # 2) clip 到合理範圍 [0, max_range]
    distances_2d_sectors = torch.clamp(distances_2d_sectors, 0.0, max_range)
    
    # 3) normalize 到 [0, 1]（超重要，避免數值爆）
    normalized = distances_2d_sectors / max_range
    
    # 4) 再次消毒（防止除法產生異常）
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # 5) 最終 finite 檢查（如果還有問題會立即報錯，方便定位）
    _check_finite("lidar_scan_2d_sweep", normalized, raise_on_error=True)
    
    return normalized


def safe_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """安全的上一步動作觀測
    
    返回上一步執行的動作（記憶），並清理異常值。
    
    Args:
        env: 環境實例
    
    Returns:
        shape [num_envs, 2]：上一步的動作 [-1, 1]
    
    為什麼需要這個觀測？
    - 幫助動作平滑（避免突然急轉）
    - 給神經網絡「短期記憶」
    """
    # 獲取上一步動作
    actions = env.action_manager.action
    # shape: [num_envs, 2]
    
    # 清理異常值
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    # 如果出現 NaN → 設為 0（靜止）
    # 如果出現 ±inf → 限制到 ±1
    
    # 限制範圍
    actions = torch.clamp(actions, -1.0, 1.0)
    # 確保在 [-1, 1] 範圍內
    
    # 檢查 finite（防止 PPO std>=0 錯誤）
    _check_finite("actions", actions, raise_on_error=True)
    
    return actions


def goal_position_in_robot_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標相對位置觀測（機器人座標系）
    
    計算目標位置相對於機器人的位置（前後、左右）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs, 2]：相對位置 [X, Y] 在機器人座標系
        例如：[2.5, -1.0] = 前方 2.5 米，右邊 1 米
    
    為什麼要轉換到機器人座標系？
    - 神經網絡更容易學習相對關係
    - 世界座標系的位置沒有意義（機器人不知道「東西南北」）
    """
    # 獲取機器人
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取數據
    goal_pos_w = env.command_manager.get_command("goal_command")
    # shape: [num_envs, 3]：目標在世界座標系的位置 (X, Y, Z)
    
    robot_pos_w = asset.data.root_pos_w[:, :3]
    # shape: [num_envs, 3]：機器人在世界座標系的位置
    
    robot_quat_w = asset.data.root_quat_w
    # shape: [num_envs, 4]：機器人的朝向（四元數）
    
    # 座標系變換：世界座標 → 機器人座標
    goal_vec_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w,  # 機器人的座標系
        goal_pos_w, torch.zeros_like(robot_quat_w)  # 目標的座標系（無旋轉）
    )
    # subtract_frame_transforms：計算兩個座標系之間的相對變換
    # 返回目標相對於機器人的位置向量
    # 
    # 例子：
    #   機器人在 (0, 0)，朝東
    #   目標在 (3, 2)
    #   → 在機器人座標系中：前方 3 米，左邊 2 米
    
    # 只保留 XY（忽略 Z 高度）
    result = goal_vec_b[:, :2]
    # shape: [num_envs, 2]
    
    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = torch.clamp(result, -10.0, 10.0)
    # 限制在 ±10 米範圍內
    
    # 檢查 finite（防止 PPO std>=0 錯誤）
    _check_finite("goal_position", result, raise_on_error=True)
    
    return result


def goal_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標距離觀測
    
    計算機器人到目標的直線距離。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs, 1]：距離（米）
    
    為什麼需要這個觀測（goal_position 已經有距離信息了）？
    - 顯式提供距離信息，幫助神經網絡學習
    - 有些策略會根據距離改變行為（遠距離→快速接近，近距離→小心靠近）
    """
    # 獲取機器人
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取位置
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]  # 只要 XY（忽略高度）
    
    # 計算歐幾里得距離
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1, keepdim=True)
    
    # 安全處理：清理 NaN/Inf
    distance = torch.nan_to_num(distance, nan=10.0, posinf=10.0, neginf=0.0)
    distance = torch.clamp(distance, 0.0, 10.0)
    
    # 檢查 finite（防止 PPO std>=0 錯誤）
    _check_finite("goal_distance", distance, raise_on_error=True)
    # torch.norm：計算向量的長度（L2 範數）
    # dim=1：沿著第 1 維計算（對每個環境分別計算）
    # keepdim=True：保持維度 [num_envs, 1]
    # 
    # 例如：
    #   robot_pos = [0, 0], goal_pos = [3, 4]
    #   distance = sqrt((3-0)^2 + (4-0)^2) = sqrt(25) = 5.0 米
    
    # 安全處理
    distance = torch.nan_to_num(distance, nan=10.0, posinf=10.0, neginf=0.0)
    distance = torch.clamp(distance, 0.0, 20.0)
    # 限制在 [0, 20] 米範圍內
    
    return distance


def base_velocity_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """速度觀測（機器人座標系）
    
    回傳機器人座標系下的線速度 [vx, vy]。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs, 2]：速度 (vx, vy) in robot frame
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 取得世界座標速度與機器人姿態
    vel_w = asset.data.root_lin_vel_w[:, :3]
    quat_w = asset.data.root_quat_w
    
    # 世界座標 → 機器人座標
    vel_b = math_utils.quat_apply_inverse(quat_w, vel_w)
    result = vel_b[:, :2]
    
    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = torch.clamp(result, -10.0, 10.0)
    _check_finite("base_velocity_xy", result, raise_on_error=True)
    
    return result


def time_remaining_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """時間剩餘比例觀測
    
    回傳 episode 剩餘時間比例（1 = 剛開始，0 = 即將超時）。
    """
    # 最大步數（環境步）
    if hasattr(env, "max_episode_length"):
        max_steps = float(env.max_episode_length)
    else:
        max_steps = float(env.cfg.episode_length_s / (env.cfg.sim.dt * env.cfg.decimation))
    
    # 當前步數
    curr_steps = env.episode_length_buf.to(dtype=torch.float32)
    
    # 剩餘比例
    ratio = 1.0 - (curr_steps / (max_steps + 1e-6))
    ratio = torch.clamp(ratio, 0.0, 1.0)
    ratio = ratio.unsqueeze(1)
    
    _check_finite("time_remaining_ratio", ratio, raise_on_error=True)
    return ratio


def alive_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活狀態觀測（死亡狀態標記）
    
    這是一個關鍵的狀態特徵，用於：
    1. 幫助 Critic Network 正確估算期望值（Value）
       - 當 Agent 死亡（碰撞）後，Value 應該為 0（無未來獎勵）
       - 當 Agent 存活時，Value 應該是正常估算的期望獎勵
    
    2. 強制融合「避障」與「抵達目標」的能力
       - 單純的負向碰撞獎勵不足以讓 Agent 學會避障
       - 因為如果抵達目標的獎勵夠高，Agent 可能會選擇硬衝
       - 有了死亡標記，Critic 會知道碰撞後 Value = 0
       - 這迫使 Agent 明白：「活著」是獲得任何獎勵的前提
    
    輸出：
        1.0 = 存活（可以繼續獲得獎勵）
        0.0 = 死亡（因碰撞/翻倒等被終止，無法獲得後續獎勵）
    
    注意：
        - 只有「terminated」（真正終止，如碰撞）才算死亡
        - 「time_outs」（超時截斷）不算死亡，因為這只是訓練時的截斷
        - 這樣設計讓 Critic 能正確區分：
          - 碰撞終止 → Value = 0
          - 超時截斷 → Value = bootstrap（繼續估算未來獎勵）
    """
    # 檢查是否有 reset_terminated（因碰撞/翻倒等真正終止）
    if hasattr(env, "reset_terminated"):
        # 只有「terminated」才算死亡，不包括「time_outs」
        # reset_terminated = True → 死亡（碰撞、翻倒等）
        # reset_terminated = False → 存活
        died = env.reset_terminated.to(dtype=torch.float32)
        alive = 1.0 - died
    elif hasattr(env, "reset_buf"):
        # 備用：如果沒有 reset_terminated，使用 reset_buf
        # 但這不夠精確，因為 reset_buf 包含了 time_outs
        alive = 1.0 - env.reset_buf.to(dtype=torch.float32)
    else:
        # 默認：所有 Agent 都存活
        alive = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    
    alive = torch.clamp(alive, 0.0, 1.0).unsqueeze(1)
    _check_finite("alive_flag", alive, raise_on_error=True)
    return alive


def dynamic_obstacles_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    num_obstacles: int = 10,
    max_obstacles: int | None = None,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """障礙物狀態觀測（每個障礙物 5 維，固定維度支持權重遷移）
    
    每個障礙物特徵：
      [x, y, dir, v, size]
      - x, y：障礙物相對於機器人的位置（機器人座標系）
      - dir：障礙物方向（相對於機器人，弧度）
      - v：障礙物速度（m/s，取平面速度大小）
      - size：障礙物尺寸（近似值），-1.0 表示「不存在」（padding）
    
    2024-01 架構修復：
    - 核心問題：觀測維度必須固定，否則無法在不同階段之間遷移權重
    - Phase 1: 3 個障礙物 → 使用 padding 填充到 max_obstacles 個
    - Phase 2: 10 個障礙物 → 直接使用 max_obstacles 個
    - Phase 2.5: 動態調整障礙物數量（3-10 個），使用 env._num_obstacles
    - Padding 策略：不存在的障礙物用 [0, 0, 0, 0, -1] 填充
      - size = -1.0 明確標記「不存在」（避免誤解為「原點有障礙物」）
    
    Phase 1 設計說明：
    - 障礙物是靜態的（kinematic_enabled=True）
    - 速度 v 會正確設為 0（與物理引擎一致）
    - 方向 dir 使用障礙物的朝向（因為速度為 0，無法從速度推斷方向）
    
    Phase 2 升級（動態障礙物）：
    - 速度 v 會反映實際移動速度
    - 方向 dir 會從速度向量推斷
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        num_obstacles: 實際障礙物數量（默認值，會被 env._num_obstacles 覆蓋）
        max_obstacles: 最大障礙物數量（用於固定觀測維度），None 時使用 num_obstacles
        max_distance: 最大觀測距離（米）
    
    Returns:
        shape [num_envs, max_obstacles * 5]：固定維度的障礙物觀測
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    
    # Phase 2.5：優先使用環境中的動態障礙物數量（如果存在）
    if hasattr(env, "_num_obstacles") and env._num_obstacles is not None:
        num_obstacles = env._num_obstacles
    
    # 如果未指定 max_obstacles，使用 num_obstacles（向後兼容）
    if max_obstacles is None:
        max_obstacles = num_obstacles
    
    # 機器人位置與姿態
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)
    
    # 準備輸出（固定維度：max_obstacles × 5）
    obs = torch.zeros(num_envs, max_obstacles, 5, device=env.device, dtype=torch.float32)
    
    # 取得障礙物尺寸（若有）
    obstacle_sizes = getattr(env, "_obstacle_sizes", None)
    if obstacle_sizes is None:
        obstacle_sizes = _get_obstacle_sizes()
    
    # 取得障礙物速度緩存，如果不存在則初始化為全零
    # 這樣確保即使在 reset_obstacles 被調用之前，觀測函數也能正常工作
    obstacle_velocities = getattr(env, "_obstacle_velocities", None)
    if obstacle_velocities is None:
        # 初始化速度緩存為全零（Phase 1：靜態障礙物）
        env._obstacle_velocities = torch.zeros(
            num_envs, num_obstacles, 2, device=env.device, dtype=torch.float32
        )
        obstacle_velocities = env._obstacle_velocities
    
    # 遍歷障礙物
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        if not hasattr(env.scene, obstacle_name):
            continue
        
        obstacle = getattr(env.scene, obstacle_name)
        obs_pos_w = obstacle.data.root_pos_w[:, :3]
        
        # 相對位置（機器人座標系）
        rel_pos_b, _ = math_utils.subtract_frame_transforms(
            robot_pos_w, robot_quat_w,
            obs_pos_w, torch.zeros_like(robot_quat_w),
        )
        rel_xy = rel_pos_b[:, :2]
        rel_xy = torch.clamp(rel_xy, -max_distance, max_distance)
        
        # 方向與速度（使用內部速度緩存，沒有則退回到物體速度）
        # Phase 1：速度為 0 時，使用障礙物朝向；Phase 2：速度不為 0 時，使用速度方向
        obs_quat_w = obstacle.data.root_quat_w if hasattr(obstacle.data, "root_quat_w") else robot_quat_w
        _, _, obs_yaw = math_utils.euler_xyz_from_quat(obs_quat_w)
        
        if obstacle_velocities is not None and obstacle_velocities.shape[1] > i:
            vel_w = obstacle_velocities[:, i, :]
            speed = torch.linalg.norm(vel_w, dim=1)
            # 如果速度為 0（Phase 1 靜態障礙物），使用障礙物朝向
            # 如果速度不為 0（Phase 2 動態障礙物），使用速度方向
            has_velocity = speed > 1e-6  # 速度閾值：> 0.001 m/s 才算有速度
            vel_yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])
            rel_yaw = torch.where(has_velocity, vel_yaw - robot_yaw, obs_yaw - robot_yaw)
        else:
            # 沒有速度緩存：使用障礙物朝向和物理引擎速度
            rel_yaw = obs_yaw - robot_yaw
            if hasattr(obstacle.data, "root_lin_vel_w"):
                vel_w = obstacle.data.root_lin_vel_w[:, :2]
                speed = torch.linalg.norm(vel_w, dim=1)
            else:
                speed = torch.zeros(num_envs, device=env.device)
        
        # 將方向角限制到 [-pi, pi]
        rel_yaw = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))
        
        # 尺寸（近似值）
        if obstacle_sizes is not None and i < len(obstacle_sizes):
            size_val = float(obstacle_sizes[i])
        else:
            size_val = 0.0
        size = torch.full((num_envs,), size_val, device=env.device, dtype=torch.float32)
        
        # 寫入真實障礙物資訊
        obs[:, i, 0:2] = rel_xy
        obs[:, i, 2] = rel_yaw
        obs[:, i, 3] = speed
        obs[:, i, 4] = size
    
    # ------------------------------------------------------------------------
    # Padding：填充不存在的障礙物（2024-01 架構修復）
    # ------------------------------------------------------------------------
    # 為 num_obstacles 到 max_obstacles 之間的「空槽」填充特殊標記
    # Padding 策略：[0, 0, 0, 0, -1]
    # - x, y = 0：位置為原點（但不會誤解，因為 size = -1）
    # - dir = 0：方向為 0
    # - v = 0：速度為 0
    # - size = -1：特殊標記「不存在」（⭐ 關鍵：明確標記，避免誤解）
    for i in range(num_obstacles, max_obstacles):
        obs[:, i, 0] = 0.0  # x = 0
        obs[:, i, 1] = 0.0  # y = 0
        obs[:, i, 2] = 0.0  # direction = 0
        obs[:, i, 3] = 0.0  # velocity = 0
        obs[:, i, 4] = -1.0  # size = -1（⭐ 特殊標記：不存在）
    
    # reshape 成 [num_envs, max_obstacles * 5]（固定維度）
    obs = obs.reshape(num_envs, max_obstacles * 5)
    
    # 安全處理
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    obs = torch.clamp(obs, -max_distance, max_distance)
    _check_finite("dynamic_obstacles_state", obs, raise_on_error=True)
    
    return obs


# ============================================================================
# 獎勵函數（Rewards）
# ============================================================================
# 定義什麼行為會得分/扣分（強化學習的核心！）
# ============================================================================

def _print_diagnostics(env: ManagerBasedRLEnv, env_id: int, reward_value: Optional[float] = None):
    """打印環境的詳細診斷信息
    
    Args:
        env: 環境實例
        env_id: 環境 ID
        reward_value: 異常的 reward 值（可選）
    """
    print(f"\n  [Diagnostics for env {env_id}]:")
    if reward_value is not None:
        print(f"    Reward value: {reward_value:.6e}")
    
    try:
        # 獲取機器人狀態
        asset = env.scene["robot"]
        robot_pos = asset.data.root_pos_w[env_id, :3].cpu()
        robot_vel = asset.data.root_lin_vel_w[env_id, :2].cpu()
        robot_ang_vel = asset.data.root_ang_vel_w[env_id, :].cpu()  # 全部角速度
        
        # 獲取目標位置
        goal_pos = env.command_manager.get_command("goal_command")[env_id, :3].cpu()
        goal_dist = torch.norm(goal_pos[:2] - robot_pos[:2]).item()
        
        # 獲取雷達數據
        sensor = env.scene.sensors["lidar"]
        sensor_pos = sensor.data.pos_w[env_id, :3].cpu()
        hit_points = sensor.data.ray_hits_w[env_id, :, :3].cpu()
        distances = torch.norm(hit_points - sensor_pos.unsqueeze(0), dim=-1)
        min_lidar_dist = distances[torch.isfinite(distances)].min().item() if torch.isfinite(distances).any() else float('inf')
        max_lidar_dist = distances[torch.isfinite(distances)].max().item() if torch.isfinite(distances).any() else float('inf')
        
        # 獲取動作
        if hasattr(env.action_manager, 'action'):
            action = env.action_manager.action[env_id, :].cpu()
        else:
            action = torch.zeros(2)
        
        # 檢查狀態中的 NaN/Inf
        pos_has_nan = torch.isnan(robot_pos).any().item() or torch.isinf(robot_pos).any().item()
        vel_has_nan = torch.isnan(robot_vel).any().item() or torch.isinf(robot_vel).any().item()
        goal_has_nan = torch.isnan(goal_pos).any().item() or torch.isinf(goal_pos).any().item()
        
        print(f"    Robot position: {robot_pos.tolist()} {'[HAS NaN/Inf!]' if pos_has_nan else ''}")
        print(f"    Robot velocity (XY): {robot_vel.tolist()} (|v|={torch.norm(robot_vel).item():.6f} m/s) {'[HAS NaN/Inf!]' if vel_has_nan else ''}")
        print(f"    Robot angular vel: {robot_ang_vel.tolist()} rad/s")
        print(f"    Goal position: {goal_pos.tolist()} {'[HAS NaN/Inf!]' if goal_has_nan else ''}")
        print(f"    Goal distance: {goal_dist:.6f} m")
        print(f"    Min lidar distance: {min_lidar_dist:.6f} m")
        print(f"    Max lidar distance: {max_lidar_dist:.6f} m")
        print(f"    Action: {action.tolist()}")
        print(f"    Episode length: {env.episode_length_buf[env_id].item()}")
        
        # 檢查是否有異常大的狀態值
        vel_mag = torch.norm(robot_vel).item()
        if vel_mag > 100.0:
            print(f"    ⚠️  WARNING: Robot velocity magnitude is very large: {vel_mag:.6f} m/s")
        if abs(goal_dist) > 100.0:
            print(f"    ⚠️  WARNING: Goal distance is very large: {goal_dist:.6f} m")
        if min_lidar_dist < 0.0 or not torch.isfinite(torch.tensor(min_lidar_dist)):
            print(f"    ⚠️  WARNING: Min lidar distance is invalid: {min_lidar_dist}")
    except Exception as e:
        print(f"    ⚠️  Error getting diagnostics: {e}")


def _check_reward_term(name: str, reward: torch.Tensor, env: ManagerBasedRLEnv, raise_on_error: bool = True) -> torch.Tensor:
    """檢查 reward term 是否包含 NaN/Inf 或異常值（用於定位 PPO std>=0 錯誤）
    
    Args:
        name: reward term 名稱（用於錯誤訊息）
        reward: reward 張量
        env: 環境實例（用於獲取診斷信息）
        raise_on_error: 如果發現異常是否立即拋出異常
    
    Returns:
        清理後的 reward 張量
    
    這個函數會：
    1. 檢查 NaN/Inf
    2. 檢查異常大的值（> 1e6 或 min/max 突然變大）
    3. 如果發現問題，打印詳細的診斷信息（機器人狀態、距離、速度、角速度、min lidar range、action 等）
    4. 嘗試清理異常值
    """
    # 計算統計值（用於檢查 min/max 是否異常）
    reward_finite = reward[torch.isfinite(reward)]
    if reward_finite.numel() > 0:
        min_val = reward_finite.min().item()
        max_val = reward_finite.max().item()
        mean_val = reward_finite.mean().item()
        std_val = reward_finite.std().item()
    else:
        min_val = float('nan')
        max_val = float('nan')
        mean_val = float('nan')
        std_val = float('nan')
    
    # 檢查 NaN/Inf
    if not torch.isfinite(reward).all():
        bad_mask = ~torch.isfinite(reward)
        bad_env_ids = bad_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        num_bad = bad_mask.sum().item()
        num_total = reward.numel()
        nan_count = torch.isnan(reward).sum().item()
        inf_count = torch.isinf(reward).sum().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD ERROR] {name}: Non-finite values detected!")
        print(f"{'='*80}")
        print(f"  Bad count: {num_bad}/{num_total}")
        print(f"  NaN count: {nan_count}")
        print(f"  Inf count: {inf_count}")
        print(f"  Bad env IDs: {bad_env_ids[:10]}..." if len(bad_env_ids) > 10 else f"  Bad env IDs: {bad_env_ids}")
        
        # 打印 reward 統計
        print(f"\n  [Reward statistics (finite values only)]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印診斷信息（只對第一個有問題的環境）
        if len(bad_env_ids) > 0:
            env_id = bad_env_ids[0]
            bad_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=bad_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains non-finite values! "
                f"nan={nan_count} inf={inf_count} bad_envs={bad_env_ids[:5]}"
            )
        
        # 嘗試清理（如果允許）
        reward_cleaned = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward_cleaned = torch.clamp(reward_cleaned, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查異常大的值（> 1e6 或 > 1e10）
    abs_reward = torch.abs(reward)
    
    # 檢查是否有 > 1e10 的值（極端異常）
    extreme_mask = abs_reward > 1e10
    if extreme_mask.any():
        extreme_env_ids = extreme_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        extreme_max_val = abs_reward.max().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD CRITICAL ERROR] {name}: EXTREMELY large values detected (> 1e10)!")
        print(f"{'='*80}")
        print(f"  Max absolute value: {extreme_max_val:.2e}")
        print(f"  Extreme env IDs: {extreme_env_ids[:10]}..." if len(extreme_env_ids) > 10 else f"  Extreme env IDs: {extreme_env_ids}")
        
        print(f"\n  [Reward statistics]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印詳細診斷信息
        if len(extreme_env_ids) > 0:
            env_id = extreme_env_ids[0]
            extreme_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=extreme_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains EXTREMELY large values! "
                f"max={extreme_max_val:.2e} extreme_envs={extreme_env_ids[:5]}"
            )
        
        # 嘗試清理
        reward_cleaned = torch.clamp(reward, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查是否有 > 1e6 的值（較大異常）
    large_mask = abs_reward > 1e6
    if large_mask.any():
        large_env_ids = large_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        large_max_val = abs_reward.max().item()
        
        print(f"\n{'='*80}")
        print(f"[REWARD WARNING] {name}: Very large values detected (> 1e6)!")
        print(f"{'='*80}")
        print(f"  Max absolute value: {large_max_val:.2e}")
        print(f"  Large env IDs: {large_env_ids[:10]}..." if len(large_env_ids) > 10 else f"  Large env IDs: {large_env_ids}")
        
        print(f"\n  [Reward statistics]:")
        print(f"    Min: {min_val:.6e}")
        print(f"    Max: {max_val:.6e}")
        print(f"    Mean: {mean_val:.6e}")
        print(f"    Std: {std_val:.6e}")
        
        # 打印詳細診斷信息
        if len(large_env_ids) > 0:
            env_id = large_env_ids[0]
            large_value = reward[env_id].item()
            _print_diagnostics(env, env_id, reward_value=large_value)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term contains very large values! "
                f"max={large_max_val:.2e} large_envs={large_env_ids[:5]}"
            )
        
        # 嘗試清理
        reward_cleaned = torch.clamp(reward, -1000.0, 1000.0)
        return reward_cleaned
    
    # 檢查 min/max 是否突然變得很大（即使沒有單個值 > 1e6，但 min/max 差距很大也可能有問題）
    if abs(min_val) > 1e4 or abs(max_val) > 1e4:
        print(f"\n{'='*80}")
        print(f"[REWARD WARNING] {name}: Min/Max values are suspiciously large!")
        print(f"{'='*80}")
        print(f"  Min: {min_val:.6e}")
        print(f"  Max: {max_val:.6e}")
        print(f"  Mean: {mean_val:.6e}")
        print(f"  Std: {std_val:.6e}")
        
        # 找出 min/max 對應的環境
        if abs(min_val) > 1e4:
            min_env_id = (reward == min_val).nonzero(as_tuple=False)[0, 0].item()
            _print_diagnostics(env, min_env_id, reward_value=min_val)
        if abs(max_val) > 1e4:
            max_env_id = (reward == max_val).nonzero(as_tuple=False)[0, 0].item()
            _print_diagnostics(env, max_env_id, reward_value=max_val)
        
        if raise_on_error:
            raise RuntimeError(
                f"[{name}] Reward term has suspiciously large min/max values! "
                f"min={min_val:.2e} max={max_val:.2e}"
            )
    
    return reward

def velocity_toward_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float = 1.0) -> torch.Tensor:
    """朝向目標的速度獎勵
    
    獎勵機器人「朝著目標移動」的速度分量。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_dist: 最小距離（米），低於此距離不給獎勵（避免衝撞）
    
    Returns:
        shape [num_envs]：獎勵值 [0, ~1.5]
        值越大表示越快朝目標移動
    
    為什麼用速度獎勵？
    - 鼓勵機器人「積極移動」
    - 避免機器人「磨蹭」（慢慢走）
    - 比單純的距離獎勵更直接
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # 只要 XY 速度（忽略 Z）
    # shape: [num_envs, 2]：[vx, vy] 在世界座標系
    
    # ------------------------------------------------------------------------
    # 步驟 1：計算目標方向（歸一化單位向量）
    # ------------------------------------------------------------------------
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    # 從機器人指向目標的向量
    # 例如：robot=[0,0], goal=[3,4] → direction=[3,4]
    
    goal_distance = torch.norm(goal_direction, dim=1)
    # 計算距離：sqrt(3^2 + 4^2) = 5.0 米
    
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    # 歸一化為單位向量（長度=1）
    # +1e-6 防止除以零
    # 例如：[3,4] / 5.0 = [0.6, 0.8]（單位向量）
    
    # ------------------------------------------------------------------------
    # 步驟 2：計算速度在目標方向上的投影
    # ------------------------------------------------------------------------
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    # 向量點積（dot product）：v · d = |v| |d| cos(θ)
    # 當 d 是單位向量時：v · d = 速度在 d 方向的分量
    # 
    # 例子：
    #   robot_vel = [1.0, 0.5]（往東北移動）
    #   goal_direction = [1.0, 0.0]（目標在正東）
    #   projection = 1.0*1.0 + 0.5*0.0 = 1.0 m/s
    #   （機器人以 1 m/s 的速度朝目標移動）
    
    velocity_projection = torch.clamp(velocity_projection, min=0.0)
    # 只獎勵正向速度（朝目標移動）
    # 如果背離目標（負值）→ 設為 0（不給獎勵，但不懲罰）
    
    # ------------------------------------------------------------------------
    # 步驟 3：距離閘門（避免衝撞）
    # ------------------------------------------------------------------------
    gated_velocity = torch.where(
        goal_distance > min_dist,  # 條件：距離 > 1.0 米
        velocity_projection,  # True：給速度獎勵
        torch.zeros_like(velocity_projection),  # False：不給獎勵
    )
    # 為什麼需要這個？
    # - 防止機器人「衝撞」目標（撞過頭）
    # - 在目標附近（<1米）需要減速，不應獎勵高速
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    gated_velocity = torch.nan_to_num(gated_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    gated_velocity = torch.clamp(gated_velocity, 0.0, 10.0)  # 限制到合理範圍
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    gated_velocity = _check_reward_term("velocity_toward_goal", gated_velocity, env, raise_on_error=True)
    
    return gated_velocity
    # 返回值範圍：[0, max_linear_velocity]
    # 例如：[0, 1.5] m/s


def velocity_toward_goal_smooth(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    slow_distance: float = 1.0,
    stop_distance: float = 0.28,
    max_reward_speed: float = 1.5,
    min_reward_speed: float = 0.3,
) -> torch.Tensor:
    """漸進式速度獎勵：距離越近，獎勵的速度越低
    
    這個函數鼓勵機器人在接近目標時自動減速，避免衝撞。
    獎勵機制：
    - 距離 > slow_distance：獎勵高速（max_reward_speed）
    - stop_distance < 距離 < slow_distance：線性插值期望速度
    - 距離 < stop_distance：不給獎勵（避免衝撞）
    
    獎勵計算：
    - 使用指數衰減：速度越接近「期望速度」，獎勵越高
    - 完美匹配期望速度 → 獎勵 1.0
    - 偏離期望速度 → 獎勵指數衰減
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        slow_distance: 開始減速的距離（米），低於此距離開始降低期望速度
        stop_distance: 停止獎勵的距離（米），低於此距離不給獎勵
        max_reward_speed: 遠距離的目標速度（m/s），距離 > slow_distance 時的期望速度
        min_reward_speed: 近距離的目標速度（m/s），距離 = stop_distance 時的期望速度
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
        1.0 = 完美匹配期望速度
        0.0 = 距離太近或速度偏離太大
    
    設計理念：
    - 遠距離：鼓勵快速接近（1.5 m/s）
    - 中距離：鼓勵適度減速（0.3-1.5 m/s 之間）
    - 近距離：鼓勵低速接近（0.3 m/s）
    - 極近距離：停止獎勵（避免衝撞）
    
    例子：
        距離 = 3.0m → 期望速度 1.5 m/s → 以 1.5 m/s 前進得最高獎勵
        距離 = 1.0m → 期望速度 0.3 m/s → 開始自動減速
        距離 = 0.5m → 期望速度 0.15 m/s → 繼續減速
        距離 = 0.28m → 期望速度 0 m/s → 停止獎勵
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # 只要 XY 速度（忽略 Z）
    
    # ------------------------------------------------------------------------
    # 步驟 1：計算目標方向和距離
    # ------------------------------------------------------------------------
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    goal_distance = torch.norm(goal_direction, dim=1)
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    
    # ------------------------------------------------------------------------
    # 步驟 2：計算速度在目標方向上的投影
    # ------------------------------------------------------------------------
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    velocity_projection = torch.clamp(velocity_projection, min=0.0)  # 只獎勵正向速度
    
    # ------------------------------------------------------------------------
    # 步驟 3：計算「期望速度」（距離加權）
    # ------------------------------------------------------------------------
    # 遠距離：期望高速（max_reward_speed）
    # 近距離：期望低速（min_reward_speed）
    # 中間距離：線性插值
    target_speed = torch.where(
        goal_distance > slow_distance,
        # 距離 > slow_distance：期望高速
        torch.full_like(goal_distance, max_reward_speed),
        torch.where(
            goal_distance > stop_distance,
            # stop_distance < 距離 < slow_distance：線性插值
            min_reward_speed + (max_reward_speed - min_reward_speed) * 
            ((goal_distance - stop_distance) / (slow_distance - stop_distance + 1e-6)),
            # 距離 < stop_distance：期望速度為 0（停止）
            torch.zeros_like(goal_distance)
        )
    )
    
    # ------------------------------------------------------------------------
    # 步驟 4：計算獎勵（越接近目標速度越好）
    # ------------------------------------------------------------------------
    # 使用指數衰減：完美匹配 target_speed 時獎勵 = 1.0
    # 偏離越大，獎勵越小
    speed_error = torch.abs(velocity_projection - target_speed)
    # 使用 exp(-error) 作為獎勵，error=0 時 reward=1.0，error 越大 reward 越小
    # 為了讓獎勵更平滑，使用 exp(-error / scale)，scale 控制衰減速度
    scale = max_reward_speed * 0.5  # 衰減尺度：當誤差 = scale 時，獎勵約為 0.6
    reward = torch.exp(-speed_error / (scale + 1e-6))
    
    # ------------------------------------------------------------------------
    # 步驟 5：距離閘門（避免衝撞）
    # ------------------------------------------------------------------------
    reward = torch.where(
        goal_distance > stop_distance,  # 距離 > stop_distance 才給獎勵
        reward,
        torch.zeros_like(reward)  # 距離太近：不給獎勵
    )
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("velocity_toward_goal_smooth", reward, env, raise_on_error=True)
    
    return reward


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_speed: float = 0.0,
    max_speed: float | None = None,
) -> torch.Tensor:
    """鼓勵機器人向前走的速度獎勵
    
    獎勵機器人「朝自己前方」的速度分量（機器人座標系 X 軸）。
    只獎勵前進速度，不獎勵後退或側向。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_speed: 最小速度門檻（低於此速度不給獎勵）
        max_speed: 最大速度截斷（避免獎勵過大），None 表示不截斷
    
    Returns:
        shape [num_envs]：獎勵值 [0, max_speed]
    """
    # 獲取機器人狀態
    asset: Articulation = env.scene[asset_cfg.name]
    robot_vel_w = asset.data.root_lin_vel_w[:, :3]  # 世界座標速度
    robot_quat_w = asset.data.root_quat_w  # 世界座標姿態
    
    # 計算機器人前方向量（世界座標）
    num_envs = robot_vel_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # 機器人座標系 X 軸是前方
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 計算前進速度分量（dot product）
    forward_speed = torch.sum(robot_vel_w * forward_w, dim=1)
    forward_speed = torch.clamp(forward_speed, min=0.0)  # 只獎勵前進
    
    # 最小速度門檻
    if min_speed > 0.0:
        forward_speed = torch.where(
            forward_speed >= min_speed,
            forward_speed,
            torch.zeros_like(forward_speed),
        )
    
    # 最大速度截斷
    if max_speed is not None:
        forward_speed = torch.clamp(forward_speed, 0.0, max_speed)
    
    # 清理 NaN/Inf
    forward_speed = torch.nan_to_num(forward_speed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 檢查 reward term
    forward_speed = _check_reward_term("forward_velocity_reward", forward_speed, env, raise_on_error=True)
    
    return forward_speed


def progress_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標進度獎勵
    
    獎勵「每一步接近目標的距離變化」。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs]：獎勵值（可正可負）
        正值 = 接近目標（好！）
        負值 = 遠離目標（不好）
    
    為什麼需要這個（已經有速度獎勵了）？
    - 防止機器人「繞圈」刷速度獎勵
    - 直接測量「有效進度」
    - 補充速度獎勵（兩者結合效果最好）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    
    # 計算當前距離
    curr_dist = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    # shape: [num_envs]

    # ------------------------------------------------------------------------
    # 初始化「上一步距離」緩存
    # ------------------------------------------------------------------------
    if not hasattr(env, "_prev_goal_dist"):
        # 第一次調用時，創建緩存
        env._prev_goal_dist = curr_dist.clone()
        # clone() 創建副本（不共享記憶體）

    # ------------------------------------------------------------------------
    # 計算進度
    # ------------------------------------------------------------------------
    reward = env._prev_goal_dist - curr_dist
    # 進度 = 上一步距離 - 這一步距離
    # 
    # 例子 1（接近目標）：
    #   上一步：5.0 米，這一步：4.5 米
    #   進度 = 5.0 - 4.5 = +0.5（獎勵！）
    # 
    # 例子 2（遠離目標）：
    #   上一步：5.0 米，這一步：5.3 米
    #   進度 = 5.0 - 5.3 = -0.3（懲罰）

    # ------------------------------------------------------------------------
    # 重置保護
    # ------------------------------------------------------------------------
    reward = torch.where(
        env.episode_length_buf == 0,  # 條件：剛重置的環境（第 0 步）
        torch.zeros_like(reward),  # True：不給獎勵（因為沒有「上一步」）
        reward,  # False：正常給獎勵
    )
    # 為什麼需要這個？
    # - 剛重置時，_prev_goal_dist 是舊的值（上一個 episode）
    # - 用舊值計算會產生錯誤的獎勵
    # - 所以第 0 步不給獎勵

    # ------------------------------------------------------------------------
    # 更新緩存
    # ------------------------------------------------------------------------
    env._prev_goal_dist = curr_dist.detach()
    # detach()：從計算圖中分離（不追蹤梯度）
    # 為什麼？因為這是緩存，不需要計算梯度
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = torch.clamp(reward, -10.0, 10.0)  # 限制進度獎勵範圍
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("progress_to_goal", reward, env, raise_on_error=True)
    
    return reward


def reaching_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """到達目標獎勵
    
    當機器人到達目標時給予大獎勵。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米），低於此距離算「到達」
    
    Returns:
        shape [num_envs]：0 或 1
        1 = 到達目標
        0 = 未到達
    
    這是最重要的獎勵！
    - 配合大權重（weight=100），成為主要目標
    - 稀疏獎勵（大部分時間是 0，到達時才給）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    
    # 計算距離（扣除機器人體半徑，使用「身體到目標」距離）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)
    
    # 安全處理：清理 NaN/Inf
    distance = torch.nan_to_num(distance, nan=100.0, posinf=100.0, neginf=0.0)
    distance = torch.clamp(distance, 0.0, 100.0)
    
    # 判斷是否到達
    reward = (distance < threshold).float()
    # (distance < 0.5) 返回布林值 [True, False, ...]
    # .float() 轉換為浮點數 [1.0, 0.0, ...]
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("reaching_goal", reward, env, raise_on_error=True)
    
    return reward


def approaching_goal_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    close_threshold: float = 2.0,
    decay_scale: float = 0.5,
) -> torch.Tensor:
    """接近目標時給予額外獎勵（鼓勵 agent 敢於靠近）
    
    這個函數鼓勵 agent 接近目標，即使還沒到達。
    使用指數衰減：距離越近，獎勵越高。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        close_threshold: 接近閾值（米），進入此距離內開始給獎勵
        decay_scale: 衰減尺度係數（0-1），控制指數衰減的陡峭程度
                     0.5 = 較平緩的衰減（推薦）
                     1.0 = 較陡峭的衰減
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
        1.0 = 距離為 0（已到達）
        0.0 = 距離 >= close_threshold（太遠）
    
    設計理念：
    - 鼓勵 agent 敢於接近目標，而不是「遠離一切」
    - 與 progressive_collision 形成平衡：可以接近障礙物，但不要太近
    - 幫助 agent 克服「恐懼」，積極嘗試到達目標
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    
    # 計算距離
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算衰減尺度：使用 close_threshold * decay_scale 作為分母
    # decay_scale=0.5 時：
    #   距離 = 2.0m → exp(-2.0/1.0) = exp(-2) ≈ 0.14
    #   距離 = 1.0m → exp(-1.0/1.0) = exp(-1) ≈ 0.37
    #   距離 = 0.5m → exp(-0.5/1.0) = exp(-0.5) ≈ 0.61
    #   距離 = 0.0m → exp(0) = 1.0
    # 
    # 這比之前更平緩：在閾值邊界（2.0m）處獎勵約 0.14，而不是 0.37
    # 這樣可以避免衰減過於陡峭導致 agent 難以學習
    scale = close_threshold * decay_scale
    
    # 距離閘門：只對接近的環境給獎勵
    # 距離 < close_threshold 時，使用指數衰減計算獎勵
    # 距離 >= close_threshold 時，獎勵為 0
    reward = torch.where(
        goal_distance < close_threshold,
        torch.exp(-goal_distance / (scale + 1e-6)),  # 指數衰減：距離越近獎勵越高
        torch.zeros_like(goal_distance)  # 距離太遠：不給獎勵
    )
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    
    # 檢查 reward term
    reward = _check_reward_term("approaching_goal_bonus", reward, env, raise_on_error=True)
    
    return reward


def obstacle_avoidance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
    collision_distance: float = 0.3,
) -> torch.Tensor:
    """避障獎勵（未使用）
    
    這個函數沒有在 charge_env_cfg.py 中使用，但保留以供參考。
    
    根據最近障礙物的距離給予獎勵：
    - 距離 > safe_distance：獎勵 +1
    - 距離 < collision_distance：懲罰 -1
    - 中間：線性插值
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = sensor.data.ray_hits_w[..., 0]
    min_distance = torch.min(distances, dim=1)[0]
    # 找出 360 條射線中最近的障礙物
    
    reward = torch.clamp(
        (min_distance - collision_distance) / (safe_distance - collision_distance),
        min=-1.0,
        max=1.0,
    )
    # 線性映射：[0.3, 1.0] → [-1, 1]
    return reward


def heading_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """朝向目標獎勵
    
    獎勵機器人「面向」目標的角度。這個獎勵鼓勵機器人先對準目標方向，再加速前進。
    
    Returns:
        shape [num_envs]：cos(angle) 範圍 [-1, 1]
        1.0 = 正對目標（角度 0°）
        0.0 = 側對目標（角度 90°）
        -1.0 = 背對目標（角度 180°）
    
    設計理念：
    - 在接近目標時，朝向獎勵更重要（先對準方向）
    - 在遠離目標時，速度獎勵更重要（快速接近）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    
    # 計算目標方向（世界座標系）
    goal_direction_w = goal_pos_w - robot_pos_w
    goal_distance = torch.norm(goal_direction_w[:, :2], dim=1)
    goal_direction_w = goal_direction_w / (torch.norm(goal_direction_w, dim=1, keepdim=True) + 1e-6)
    # 歸一化單位向量
    
    # 創建機器人前方向量（本地座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # X 軸是前方
    
    # 轉換到世界座標系
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 計算夾角的余弦值（向量點積）
    heading_reward = torch.sum(forward_w[:, :2] * goal_direction_w[:, :2], dim=1)
    # cos(0°) = 1.0（正對）
    # cos(90°) = 0.0（側對）
    # cos(180°) = -1.0（背對）
    
    # 只獎勵正向朝向（不懲罰背對，因為速度獎勵會處理）
    heading_reward = torch.clamp(heading_reward, min=0.0)
    
    # 安全處理
    heading_reward = torch.nan_to_num(heading_reward, nan=0.0, posinf=1.0, neginf=0.0)
    heading_reward = torch.clamp(heading_reward, 0.0, 1.0)
    
    return heading_reward


def heading_to_goal_distance_weighted(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """朝向目標獎勵（根據距離加權）
    
    根據距離目標的遠近，動態調整朝向獎勵的權重：
    - 接近目標時（< close_distance）：強烈獎勵朝向目標
    - 遠離目標時（> far_distance）：較少獎勵朝向（速度更重要）
    - 中間距離：線性插值
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        close_distance: 接近距離閾值（米），低於此距離時朝向獎勵權重最大
        far_distance: 遠距離閾值（米），高於此距離時朝向獎勵權重最小
    
    Returns:
        shape [num_envs]：加權後的朝向獎勵 [0, 1]
    """
    # 獲取朝向獎勵
    heading_reward = heading_to_goal(env, asset_cfg)
    
    # 獲取距離
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算距離權重（線性插值）
    # 距離 < close_distance → 權重 = 1.0（最大）
    # 距離 > far_distance → 權重 = 0.2（最小）
    # 中間距離 → 線性插值
    weight = torch.clamp(
        (far_distance - goal_distance) / (far_distance - close_distance + 1e-6),
        min=0.2,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = heading_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=1.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 1.0)
    
    return weighted_reward


def velocity_toward_goal_distance_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.5,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """朝向目標的速度獎勵（根據距離動態調整）
    
    根據距離目標的遠近，動態調整速度獎勵：
    - 接近目標時（< close_distance）：降低速度獎勵權重（優先朝向）
    - 遠離目標時（> far_distance）：提高速度獎勵權重（快速接近）
    - 中間距離：線性插值
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        min_dist: 最小距離（米），低於此距離不給速度獎勵
        close_distance: 接近距離閾值（米）
        far_distance: 遠距離閾值（米）
    
    Returns:
        shape [num_envs]：加權後的速度獎勵
    """
    # 獲取原始速度獎勵
    base_reward = velocity_toward_goal(env, asset_cfg, min_dist=min_dist)
    
    # 獲取距離
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    goal_distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    
    # 計算距離權重（線性插值）
    # 距離 < close_distance → 權重 = 0.3（降低速度獎勵，優先朝向）
    # 距離 > far_distance → 權重 = 1.0（提高速度獎勵，快速接近）
    # 中間距離 → 線性插值
    weight = torch.clamp(
        (goal_distance - close_distance) / (far_distance - close_distance + 1e-6),
        min=0.3,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = base_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    
    # 檢查 reward term
    weighted_reward = _check_reward_term("velocity_toward_goal_distance_weighted", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward


def reverse_toward_goal_distance_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.28,
    close_distance: float = 2.0,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """倒車靠近目標的速度獎勵（根據距離動態調整）
    
    當機器人「背對」目標時，鼓勵用倒車靠近目標。
    - 背對目標：朝向角度 > 90°（dot < 0）
    - 只獎勵倒車方向的速度分量
    - 接近目標時權重較高（避免繞圈）
    """
    # 獲取數據
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]
    
    # 目標方向（世界座標系）
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    goal_distance = torch.norm(goal_direction, dim=1)
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    
    # 機器人前方向量（世界座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0
    forward_w = math_utils.quat_apply(robot_quat_w, forward_vec)
    
    # 朝向判斷（dot < 0 表示背對目標）
    heading_dot = torch.sum(forward_w[:, :2] * goal_direction, dim=1)
    is_facing_away = heading_dot < 0.0
    
    # 速度在目標方向的投影（正值=前進，負值=倒車靠近）
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    reverse_speed = torch.clamp(-velocity_projection, min=0.0)
    
    # 距離閘門（避免已到達仍拿獎勵）
    reverse_speed = torch.where(
        goal_distance > min_dist,
        reverse_speed,
        torch.zeros_like(reverse_speed),
    )
    
    # 只有背對目標時才啟用倒車獎勵
    reverse_speed = torch.where(
        is_facing_away,
        reverse_speed,
        torch.zeros_like(reverse_speed),
    )
    
    # 距離權重（接近目標更重要）
    weight = torch.clamp(
        (far_distance - goal_distance) / (far_distance - close_distance + 1e-6),
        min=0.2,
        max=1.0,
    )
    weighted_reward = reverse_speed * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    
    # 檢查 reward term
    weighted_reward = _check_reward_term("reverse_toward_goal_distance_weighted", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward


def collision_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.3) -> torch.Tensor:
    """碰撞懲罰（二元懲罰）
    
    當機器人過於接近障礙物時給予懲罰。
    這是一個二元懲罰：要麼發生碰撞風險（1.0），要麼沒有（0.0）。
    
    注意：使用 2D 平面距離（忽略高度），與 collision_occurred 保持一致。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），低於此距離算「碰撞」（使用 2D 平面距離）
    
    Returns:
        shape [num_envs]：0 或 1
        1 = 發生碰撞（懲罰！）
        0 = 安全
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # ========================================================================
    # 使用 2D 平面距離（忽略高度 Z），與 collision_occurred 保持一致
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2] 只取 X, Y 座標
    
    # 計算 2D 平面距離（從雷達投影到地面的點到障礙物的水平距離）
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    # shape: [num_envs, num_rays]
    
    # 處理無效碰撞（射線沒打到任何東西，距離為 inf）
    # 將 inf 替換為 max_distance，這樣不會誤判為碰撞
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出每個環境中最近的障礙物距離（2D 平面距離）
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    # torch.min(tensor, dim=1) 返回 (values, indices)
    # [0] 取 values（最小距離）
    
    # 安全處理：確保距離在合理範圍
    min_distance = torch.clamp(min_distance, 0.0, sensor.cfg.max_distance)
    
    # 判斷是否碰撞：最近距離 < 閾值
    reward = (min_distance < threshold).float()
    # min_distance < 0.6 米 → 1.0（碰撞）
    # 配合 weight=-200，變成 -200 分懲罰（極重懲罰！）
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("collision_penalty", reward, env, raise_on_error=True)
    
    return reward


def progressive_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.5,
    danger_distance: float = 0.8,
    collision_distance: float = 0.6,
) -> torch.Tensor:
    """漸進式碰撞懲罰（距離越近懲罰越大）
    
    根據機器人與障礙物的距離給予漸進式懲罰：
    - 距離 >= safe_distance：無懲罰（0.0）
    - safe_distance > 距離 >= danger_distance：輕微懲罰（0.0 到 0.5）
    - danger_distance > 距離 >= collision_distance：中等懲罰（0.5 到 1.0）
    - 距離 < collision_distance：最大懲罰（1.0）
    
    這種設計讓機器人在接近障礙物時逐漸感受到懲罰，而不是突然的二元懲罰。
    有助於機器人學習更平滑的避障行為。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        safe_distance: 安全距離（米），超過此距離無懲罰
        danger_distance: 危險距離（米），低於此距離開始中等懲罰
        collision_distance: 碰撞距離（米），低於此距離最大懲罰
    
    Returns:
        shape [num_envs]：懲罰值 [0, 1]
        0 = 安全距離，無懲罰
        1 = 碰撞距離，最大懲罰
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3]
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    
    # 使用 2D 平面距離
    sensor_pos_2d = sensor_pos[:, :2]
    hit_points_2d = hit_points[:, :, :2]
    
    # 計算 2D 平面距離
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出最近距離
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    min_distance = torch.clamp(min_distance, 0.0, sensor.cfg.max_distance)
    
    # 計算漸進式懲罰
    # 距離 >= safe_distance：0.0（無懲罰）
    # safe_distance > 距離 >= danger_distance：線性插值 0.0 到 0.5
    # danger_distance > 距離 >= collision_distance：線性插值 0.5 到 1.0
    # 距離 < collision_distance：1.0（最大懲罰）
    
    penalty = torch.zeros_like(min_distance)
    
    # 區域 1：danger_distance 到 collision_distance（中等懲罰 0.5 到 1.0）
    mask1 = (min_distance < danger_distance) & (min_distance >= collision_distance)
    if mask1.any():
        penalty[mask1] = 0.5 + 0.5 * (1.0 - (min_distance[mask1] - collision_distance) / (danger_distance - collision_distance + 1e-6))
    
    # 區域 2：safe_distance 到 danger_distance（輕微懲罰 0.0 到 0.5）
    mask2 = (min_distance < safe_distance) & (min_distance >= danger_distance)
    if mask2.any():
        penalty[mask2] = 0.5 * (1.0 - (min_distance[mask2] - danger_distance) / (safe_distance - danger_distance + 1e-6))
    
    # 區域 3：距離 < collision_distance（最大懲罰 1.0）
    mask3 = min_distance < collision_distance
    if mask3.any():
        penalty[mask3] = 1.0
    
    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 1.0)
    
    # 檢查 reward term
    penalty = _check_reward_term("progressive_collision_penalty", penalty, env, raise_on_error=True)
    
    return penalty


def safe_navigation_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    comfort_distance: float = 2.5,
) -> torch.Tensor:
    """安全導航獎勵：獎勵與障礙物保持安全距離（Phase 2 專屬）
    
    設計理念：
    - 最近障礙物 > comfort_distance → 獎勵 +1.0
    - 最近障礙物 < comfort_distance → 線性衰減到 0
    - 這會讓 Agent 「主動繞路」而非「擦邊通過」
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        comfort_distance: 舒適距離（米），超過此距離給滿分獎勵
    
    Returns:
        shape [num_envs]：獎勵值 [0, 1]
    """
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取最近障礙物距離（2D）
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]
    
    # 計算獎勵（線性衰減）
    reward = torch.clamp(min_distance / comfort_distance, 0.0, 1.0)
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    reward = torch.clamp(reward, 0.0, 1.0)
    reward = _check_reward_term("safe_navigation_bonus", reward, env, raise_on_error=True)
    
    return reward


def speed_control_near_obstacles(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    warning_distance: float = 2.0,
    max_safe_speed: float = 0.5,
) -> torch.Tensor:
    """接近障礙物時的速度控制獎勵
    
    設計理念：
    - 當機器人接近障礙物時（距離 < warning_distance），鼓勵降低速度
    - 距離越近，速度應該越慢
    - 如果速度超過 max_safe_speed，給予懲罰
    - 如果速度低於 max_safe_speed，給予獎勵
    
    這會讓機器人在接近障礙物時主動減速，提高安全性。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        warning_distance: 警告距離（米），低於此距離開始速度控制
        max_safe_speed: 最大安全速度（m/s），接近障礙物時不應超過此速度
    
    Returns:
        shape [num_envs]：獎勵值 [-1, 1]
        - 接近障礙物且速度過快 → 負值（懲罰）
        - 接近障礙物且速度適中 → 正值（獎勵）
        - 遠離障礙物 → 0（無影響）
    """
    # 獲取機器人速度（機器人座標系）
    robot: Articulation = env.scene[asset_cfg.name]
    robot_vel_b = robot.data.root_lin_vel_b  # [num_envs, 3]
    robot_speed = torch.norm(robot_vel_b[:, :2], dim=1)  # [num_envs] 2D 速度大小
    
    # 獲取雷達數據
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取最近障礙物距離（2D）
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    
    # 計算速度控制獎勵
    # 只在接近障礙物時（距離 < warning_distance）才生效
    close_to_obstacle = min_distance < warning_distance
    
    # 計算速度偏差（相對於最大安全速度）
    speed_excess = robot_speed - max_safe_speed  # [num_envs]
    
    # 計算距離權重（距離越近，影響越大）
    # 距離 = warning_distance → 權重 = 0
    # 距離 = 0 → 權重 = 1
    distance_weight = torch.clamp(
        1.0 - (min_distance / warning_distance),
        0.0, 1.0
    )  # [num_envs]
    
    # 計算獎勵
    # 如果速度超過安全速度 → 懲罰（負值）
    # 如果速度低於安全速度 → 獎勵（正值）
    # 獎勵值 = -speed_excess * distance_weight
    # 例如：速度 = 1.0 m/s, max_safe_speed = 0.5 m/s
    #      speed_excess = 0.5
    #      距離 = 1.0 米, warning_distance = 2.0 米
    #      distance_weight = 0.5
    #      獎勵 = -0.5 * 0.5 = -0.25（懲罰）
    reward = -speed_excess * distance_weight
    
    # 只在接近障礙物時生效
    reward = torch.where(close_to_obstacle, reward, torch.zeros_like(reward))
    
    # 歸一化到 [-1, 1] 範圍
    # 假設最大速度約為 1.5 m/s，則最大 speed_excess = 1.0
    # 所以 reward 範圍約為 [-1.0, 1.0]
    reward = torch.clamp(reward, -1.0, 1.0)
    
    # 安全處理
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = _check_reward_term("speed_control_near_obstacles", reward, env, raise_on_error=True)
    
    return reward


def velocity_toward_goal_dynamic_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_dist: float = 0.5,
    safe_clearance: float = 1.5,
) -> torch.Tensor:
    """動態門控速度獎勵：根據障礙物距離調整獎勵（Phase 2 專屬）
    
    核心邏輯：
    - 障礙物 > safe_clearance → 正常速度獎勵（權重 1.0）
    - 障礙物 < safe_clearance → 降低速度獎勵（權重 0.2-1.0）
    - 這會讓 Agent 在接近障礙物時「自動減速」
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        sensor_cfg: 傳感器配置（引用雷達）
        min_dist: 最小距離（米），低於此距離不給獎勵
        safe_clearance: 安全距離（米），低於此距離降低速度獎勵
    
    Returns:
        shape [num_envs]：加權後的速度獎勵
    """
    # 獲取基礎速度獎勵
    base_reward = velocity_toward_goal(env, asset_cfg, min_dist=min_dist)
    
    # 獲取最近障礙物距離
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_pos_2d = sensor.data.pos_w[:, :2]
    hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    min_distance = torch.min(distances_2d, dim=1)[0]
    
    # 計算速度權重（障礙物越近權重越低）
    # > safe_clearance: 權重 1.0
    # < safe_clearance: 權重線性衰減到 0.2
    weight = torch.clamp(
        (min_distance - 0.5) / (safe_clearance - 0.5 + 1e-6),
        min=0.2,
        max=1.0
    )
    
    # 應用權重
    weighted_reward = base_reward * weight
    
    # 安全處理
    weighted_reward = torch.nan_to_num(weighted_reward, nan=0.0, posinf=10.0, neginf=0.0)
    weighted_reward = torch.clamp(weighted_reward, 0.0, 10.0)
    weighted_reward = _check_reward_term("velocity_toward_goal_dynamic_gated", weighted_reward, env, raise_on_error=True)
    
    return weighted_reward


def time_out_penalty(env: ManagerBasedRLEnv, weight: float = -1.0) -> torch.Tensor:
    """超時懲罰
    
    只對超時的環境施加懲罰，不對成功到達目標的環境懲罰。
    
    Args:
        env: 環境實例
        weight: 懲罰權重（已棄用，由 charge_env_cfg.py 控制）
    
    Returns:
        shape [num_envs]：0 或 weight
        只有超時終止的環境才會得到懲罰，成功到達目標的環境不會被懲罰
    
    修復說明：
    - 使用 time_outs 屬性而不是 terminated 屬性
    - time_outs 只包含超時終止，不包含成功/失敗終止
    - 這樣可以避免對成功完成的任務也施加懲罰
    """
    # 獲取超時狀態（只包含超時終止，不包含成功/失敗終止）
    time_outs = env.termination_manager.time_outs
    # shape: [num_envs]：布林張量
    # True = 環境因為超時而終止
    # False = 環境未超時（可能還在運行，或因為其他原因終止）
    
    # 創建懲罰（只有超時才懲罰）
    reward = torch.where(
        time_outs,  # 只有超時的環境才給懲罰
        torch.full_like(time_outs, weight, dtype=torch.float32),  # 給懲罰
        torch.zeros_like(time_outs, dtype=torch.float32)  # 不給懲罰
    )
    
    # 安全處理：清理 NaN/Inf 並限制範圍
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    reward = torch.clamp(reward, -100.0, 0.0)  # 限制超時懲罰範圍
    
    # 檢查 reward term（防止 PPO std>=0 錯誤）
    reward = _check_reward_term("time_out_penalty", reward, env, raise_on_error=True)
    
    return reward


# ============================================================================
# 終止條件函數（Terminations）
# ============================================================================
# 定義什麼情況下「遊戲結束」，環境會被重置
# ============================================================================

def goal_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """終止條件：到達目標
    
    當機器人到達目標時終止 episode（成功！）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米）
    
    Returns:
        shape [num_envs]：布林張量
        True = 到達目標，終止
        False = 未到達，繼續
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)
    return distance < threshold
    # 距離 < 0.5 米 → True（終止）


def collision_occurred(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.3) -> torch.Tensor:
    """終止條件：發生碰撞
    
    當機器人撞到障礙物時終止 episode（失敗）。
    
    注意：使用 2D 平面距離（忽略高度），因為：
    1. 雷達安裝在機器人上方 0.5 米處
    2. 使用 3D 距離會導致距離偏大（考慮了高度差）
    3. 2D 距離更能反映機器人底盤與障礙物的實際距離
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），使用 2D 平面距離
    
    Returns:
        shape [num_envs]：布林張量
        True = 碰撞，終止
        False = 安全，繼續
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點（世界座標系）
    
    # ========================================================================
    # 使用 2D 平面距離（忽略高度 Z），因為：
    # 1. 雷達安裝在機器人上方 0.5 米處
    # 2. 我們關心的是機器人底盤與障礙物的水平距離
    # 3. 使用 3D 距離會因為高度差導致距離偏大
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2] 只取 X, Y 座標
    
    # 計算 2D 平面距離（從雷達投影到地面的點到障礙物的水平距離）
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    # shape: [num_envs, num_rays]
    
    # 處理無效碰撞（射線沒打到任何東西，距離為 inf）
    # 將 inf 替換為 max_distance，這樣不會誤判為碰撞
    distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)
    
    # 找出每個環境中最近的障礙物距離（2D 平面距離）
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]
    
    # 判斷是否碰撞：最近距離 < 閾值
    return min_distance < threshold
    # 最近障礙物 < threshold 米 → True（碰撞，終止）


def collision_contact_occurred(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：接觸感測器檢測到碰撞
    
    當機器人的碰撞框與障礙物的碰撞框接觸時終止 episode（失敗）。
    這比雷達距離判定更準確，因為它使用物理引擎的真實碰撞檢測。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用接觸感測器）
    
    Returns:
        shape [num_envs]：布林張量
        True = 檢測到碰撞接觸，終止
        False = 無碰撞，繼續
    
    優勢：
    - 使用真實的碰撞框（物理引擎的碰撞檢測）
    - 比雷達距離判定更準確（不會因為雷達安裝位置產生誤差）
    - 可以檢測機器人任何部件與障礙物的碰撞（不只是底盤）
    
    注意：
    - 接觸感測器必須在場景配置中啟用
    - 必須設置 filter_prim_paths_expr 來過濾障礙物
    """
    # 獲取接觸感測器
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 獲取接觸力數據
    # force_matrix_w: shape [num_envs, num_bodies, num_filtered_objects, 3]
    # - num_envs: 環境數量
    # - num_bodies: 每個感測器的機器人部件數量（例如：底盤、輪子等）
    # - num_filtered_objects: 過濾的物體數量（障礙物數量）
    # - 3: 力的 XYZ 分量（世界座標系）
    contact_forces = sensor.data.force_matrix_w
    
    # 如果沒有設置過濾器，force_matrix_w 會是 None
    if contact_forces is None:
        # 沒有過濾器：無法檢測與障礙物的碰撞
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    # 計算每個環境的總接觸力大小
    # 對所有機器人部件和所有障礙物的接觸力求和
    total_force = torch.sum(contact_forces, dim=(1, 2))  # [num_envs, 3]：所有部件與所有障礙物的總接觸力
    force_magnitude = torch.norm(total_force, dim=1)  # [num_envs]：總接觸力的大小
    
    # 判斷是否碰撞：接觸力 > 0 表示有接觸
    # 使用一個小的閾值（0.1 N）來避免數值誤差
    collision_threshold = 0.1  # 接觸力閾值：0.1 牛頓
    return force_magnitude > collision_threshold
    # 接觸力 > 0.1 N → True（碰撞，終止）


def robot_tipped_over(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：機器人翻倒
    
    當機器人翻倒時終止 episode（異常）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs]：布林張量
        True = 翻倒，終止
        False = 正常，繼續
    
    如何判斷翻倒？
    - 檢查機器人的 Z 軸（本地座標系的「上方」）
    - 如果 Z 軸在世界座標系中不向上 → 翻倒了
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取機器人姿態（四元數）
    quat = asset.data.root_quat_w
    # shape: [num_envs, 4]
    
    # 創建本地 Z 軸向量
    num_envs = quat.shape[0]
    z_vec = torch.zeros(num_envs, 3, device=env.device)
    z_vec[:, 2] = 1.0  # [0, 0, 1] = 垂直向上（本地座標系）
    
    # 轉換到世界座標系
    z_axis = math_utils.quat_apply(quat, z_vec)
    # 如果機器人正常：z_axis ≈ [0, 0, 1]（向上）
    # 如果機器人翻倒：z_axis ≈ [0, 0, -1]（向下）或 [1, 0, 0]（側倒）
    
    # 判斷翻倒
    is_tipped = z_axis[:, 2] < 0.5
    # z_axis[:, 2] 是 Z 軸在世界座標系中的垂直分量
    # < 0.5 表示傾斜超過 60 度（cos(60°) = 0.5）
    # 
    # 例子：
    #   正常站立：z_axis = [0, 0, 1] → z_axis[:, 2] = 1.0 > 0.5 ✅
    #   側倒 90°：z_axis = [1, 0, 0] → z_axis[:, 2] = 0.0 < 0.5 ❌（翻倒）
    #   倒立：z_axis = [0, 0, -1] → z_axis[:, 2] = -1.0 < 0.5 ❌（翻倒）
    
    return is_tipped


def robot_flying(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：機器人飛起來（異常）
    
    當機器人離地太高時終止 episode（物理引擎錯誤）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs]：布林張量
        True = 飛起來了，終止
        False = 在地上，繼續
    
    為什麼需要這個？
    - 物理引擎有時會出錯（穿模、爆炸）
    - 機器人可能「升天」
    - 這種情況下應該終止並重置
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取高度
    height = asset.data.root_pos_w[:, 2]
    # shape: [num_envs]：Z 座標（高度）
    
    # 判斷是否飛起來
    is_flying = height > 1.0
    # 高度 > 1 米 → True（異常，終止）
    # 正常情況下，Charge 底盤高度約 0.1-0.2 米
    
    return is_flying


def wall_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    boundary: float = 5.0,
    robot_radius: float = 0.28,
) -> torch.Tensor:
    """終止條件：機器人撞到邊界牆壁

    當機器人的位置超出邊界（減去機器人半徑）時，判定為撞牆。

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        boundary: 邊界距離（米），默認 5.0 米
        robot_radius: 機器人半徑（米），默認 0.28 米

    Returns:
        shape [num_envs]：布林張量
        True = 撞牆，終止
        False = 安全，繼續
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 獲取機器人位置（世界座標）
    robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # 獲取環境原點（用於計算相對位置）
    env_origins = env.scene.env_origins[:, :2]  # [num_envs, 2]
    
    # 計算相對於環境原點的位置
    robot_pos_local = robot_pos_w - env_origins  # [num_envs, 2]
    
    # 計算有效邊界（邊界減去機器人半徑）
    effective_boundary = boundary - robot_radius
    
    # 判斷是否超出邊界
    # 任一軸超出邊界就算撞牆
    out_of_bounds_x = torch.abs(robot_pos_local[:, 0]) > effective_boundary
    out_of_bounds_y = torch.abs(robot_pos_local[:, 1]) > effective_boundary
    
    is_wall_collision = out_of_bounds_x | out_of_bounds_y
    
    return is_wall_collision


def wall_collision_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    boundary: float = 5.0,
    robot_radius: float = 0.28,
    safe_distance: float = 1.0,
) -> torch.Tensor:
    """漸進式牆壁碰撞懲罰：距離牆壁越近，懲罰越大

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        boundary: 邊界距離（米），默認 5.0 米
        robot_radius: 機器人半徑（米），默認 0.28 米
        safe_distance: 安全距離（米），低於此距離開始懲罰

    Returns:
        shape [num_envs]：懲罰值 [0, 1]
        0 = 遠離牆壁，無懲罰
        1 = 撞牆，最大懲罰
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 獲取機器人位置（世界座標）
    robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # 獲取環境原點（用於計算相對位置）
    env_origins = env.scene.env_origins[:, :2]  # [num_envs, 2]
    
    # 計算相對於環境原點的位置
    robot_pos_local = robot_pos_w - env_origins  # [num_envs, 2]
    
    # 計算到各牆壁的距離
    # 東西牆（X 方向）
    dist_to_east_wall = boundary - robot_pos_local[:, 0]  # 正值 = 在邊界內
    dist_to_west_wall = boundary + robot_pos_local[:, 0]  # 正值 = 在邊界內
    # 南北牆（Y 方向）
    dist_to_north_wall = boundary - robot_pos_local[:, 1]
    dist_to_south_wall = boundary + robot_pos_local[:, 1]
    
    # 取最小距離（減去機器人半徑）
    min_dist_x = torch.min(dist_to_east_wall, dist_to_west_wall) - robot_radius
    min_dist_y = torch.min(dist_to_north_wall, dist_to_south_wall) - robot_radius
    min_wall_dist = torch.min(min_dist_x, min_dist_y)
    
    # 計算懲罰（距離越近懲罰越大）
    # 距離 >= safe_distance → 懲罰 0
    # 距離 <= 0 → 懲罰 1（撞牆）
    # 中間線性插值
    penalty = torch.clamp(1.0 - min_wall_dist / safe_distance, 0.0, 1.0)
    
    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=1.0, neginf=0.0)
    penalty = _check_reward_term("wall_collision_penalty", penalty, env, raise_on_error=True)
    
    return penalty


# ============================================================================
# 事件函數（Events）
# ============================================================================
# 定義在特定時機（如環境重置）觸發的操作
# ============================================================================

def reset_obstacles(
    env,
    env_ids,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    min_robot_distance: float = 1.5,
    min_goal_distance: float = 1.0,
    min_obstacle_spacing: float = 1.0,
    max_spawn_attempts: int = 50,
    boundary: float = 5.0,  # 場景邊界（米），默認 5.0 米
):
    """重置障礙物位置（Phase 1：完全靜態，含碰撞檢查）

    在環境重置時，將所有障礙物隨機重新擺放到新位置。
    Phase 1 設計：障礙物完全靜態（速度設為 0），只在每個 episode 開始時移動。
    
    2024-01 修正：加入碰撞檢查，確保障礙物不會與機器人、目標或其他障礙物重疊。

    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
                 例如：[0, 5, 12] = 第 0、5、12 個環境需要重置
        speed_range: 速度範圍（Phase 1 不使用，保留用於 Phase 2）
        min_speed: 最小速度（Phase 1 不使用，保留用於 Phase 2）
        min_robot_distance: 障礙物與機器人的最小距離（米）
        min_goal_distance: 障礙物與目標的最小距離（米）
        min_obstacle_spacing: 障礙物之間的最小距離（米）
        max_spawn_attempts: 每個障礙物最大嘗試生成次數
        boundary: 場景邊界（米），默認 5.0 米（Phase 1/2），Phase 3 為 8.0 米

    Phase 1 設計說明：
    - 障礙物設為 kinematic（固定不動）
    - 速度設為 0（與物理引擎一致）
    - 只在環境重置時隨機擺放位置
    - 避免「瞬移」問題（不會在 episode 中間突然移動）
    - 碰撞檢查確保合理的初始配置

    Phase 2 升級（動態障礙物）：
    - 將 rigid_props.kinematic_enabled 設為 False
    - 使用 move_obstacles 事件進行連續移動
    - 此時速度才會被實際使用
    """
    # reset 模式可能傳入 env_ids=None，代表所有環境
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_resets = len(env_ids)  # 需要重置的環境數量
    device = env.device  # 設備（CPU 或 GPU）

    # 遍歷所有可能的障礙物
    # Phase 2.5：優先使用環境中已設置的 _num_obstacles（由課程學習設置）
    # 只有在未設置時才使用全局配置
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
        env._num_obstacles = num_obstacles
    # 注意：如果 env._num_obstacles 已經設置（例如由 reset_obstacles_with_adaptive_params 設置），
    # 則使用該值，不覆蓋它
    if not hasattr(env, "_obstacle_sizes"):
        env._obstacle_sizes = _get_obstacle_sizes()

    # 初始化障礙物速度緩存（用於觀測）
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)

    # Phase 1：速度設為 0（障礙物完全靜態）
    # 這樣觀測中的速度資訊會正確反映物理引擎的狀態（靜止）
    env._obstacle_velocities[env_ids, :, :] = 0.0

    # ------------------------------------------------------------------------
    # 獲取機器人和目標位置（用於碰撞檢查）
    # ------------------------------------------------------------------------
    robot_pos_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]  # [num_resets, 2]
    
    # 嘗試獲取目標位置（可能還未初始化）
    try:
        goal_pos = env.command_manager.get_command("goal_command")
        goal_pos_xy = goal_pos[env_ids, :2]  # [num_resets, 2]
        has_goal = True
    except (AttributeError, KeyError, IndexError):
        goal_pos_xy = None
        has_goal = False

    # 儲存已放置的障礙物位置（用於障礙物間碰撞檢查）
    # shape: [num_resets, num_obstacles, 2]
    placed_positions = torch.full(
        (num_resets, num_obstacles, 2), float('inf'), device=device, dtype=torch.float32
    )

    # ------------------------------------------------------------------------
    # 分層採樣設計：確保障礙物均勻分布在場景各區域
    # ------------------------------------------------------------------------
    # 將場景分成 4 個象限，每個象限分配固定數量的障礙物
    # 這樣可以避免障礙物集中在某一側
    #
    # 象限劃分（以原點為中心）：
    #   象限 0: X > 0, Y > 0（右上）
    #   象限 1: X < 0, Y > 0（左上）
    #   象限 2: X < 0, Y < 0（左下）
    #   象限 3: X > 0, Y < 0（右下）
    #
    # 邊界值：使用傳入的 boundary 參數（但留安全邊距，實際生成範圍 boundary - safe_margin）
    # ------------------------------------------------------------------------
    
    # boundary 已作為參數傳入（默認 5.0 米，Phase 3 為 8.0 米）
    safe_margin = 0.5  # 安全邊距（避免太靠近牆壁）
    spawn_range = boundary - safe_margin  # 實際生成範圍
    
    # 定義 4 個象限的範圍
    # 格式：(x_min, x_max, y_min, y_max)
    quadrants = [
        (0.3, spawn_range, 0.3, spawn_range),      # 象限 0: 右上（避開原點附近）
        (-spawn_range, -0.3, 0.3, spawn_range),    # 象限 1: 左上
        (-spawn_range, -0.3, -spawn_range, -0.3),  # 象限 2: 左下
        (0.3, spawn_range, -spawn_range, -0.3),    # 象限 3: 右下
    ]
    
    # 計算每個象限分配多少障礙物
    # 例如：10 個障礙物 → 每象限 2-3 個
    obstacles_per_quadrant = num_obstacles // 4  # 基礎數量
    remainder = num_obstacles % 4  # 餘數分配給前幾個象限
    
    # 建立障礙物到象限的映射
    obstacle_to_quadrant = []
    for q in range(4):
        count = obstacles_per_quadrant + (1 if q < remainder else 0)
        obstacle_to_quadrant.extend([q] * count)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"

        # 檢查障礙物是否存在
        if not hasattr(env.scene, obstacle_name):
            continue

        # ------------------------------------------------------------------------
        # 取得此障礙物應該生成的象限
        # ------------------------------------------------------------------------
        quadrant_idx = obstacle_to_quadrant[i] if i < len(obstacle_to_quadrant) else i % 4
        x_min, x_max, y_min, y_max = quadrants[quadrant_idx]

        # ------------------------------------------------------------------------
        # 生成隨機位置（含碰撞檢查）
        # ------------------------------------------------------------------------
        pos = torch.zeros(num_resets, 3, device=device, dtype=torch.float32)
        pos[:, 2] = 0.5  # Z 座標：固定高度

        # 追蹤哪些環境還需要找到有效位置
        needs_position = torch.ones(num_resets, dtype=torch.bool, device=device)
        
        for attempt in range(max_spawn_attempts):
            if not needs_position.any():
                break  # 所有環境都找到有效位置了
            
            # 為需要位置的環境生成隨機候選位置（在指定象限內）
            num_need = needs_position.sum().item()
            
            # 在指定象限範圍內生成隨機位置
            candidate_x = torch.rand(num_need, device=device) * (x_max - x_min) + x_min
            candidate_y = torch.rand(num_need, device=device) * (y_max - y_min) + y_min
            
            # 暫存候選位置
            pos[needs_position, 0] = candidate_x
            pos[needs_position, 1] = candidate_y
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 1：與機器人距離
            # ------------------------------------------------------------------------
            dist_to_robot = torch.norm(pos[:, :2] - robot_pos_xy, dim=1)
            valid_robot = dist_to_robot >= min_robot_distance
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 2：與目標距離
            # ------------------------------------------------------------------------
            if has_goal:
                dist_to_goal = torch.norm(pos[:, :2] - goal_pos_xy, dim=1)
                valid_goal = dist_to_goal >= min_goal_distance
            else:
                valid_goal = torch.ones(num_resets, dtype=torch.bool, device=device)
            
            # ------------------------------------------------------------------------
            # 碰撞檢查 3：與其他已放置障礙物的距離
            # ------------------------------------------------------------------------
            valid_obstacles = torch.ones(num_resets, dtype=torch.bool, device=device)
            for j in range(i):  # 只檢查已放置的障礙物
                dist_to_other = torch.norm(pos[:, :2] - placed_positions[:, j, :], dim=1)
                valid_obstacles &= dist_to_other >= min_obstacle_spacing
            
            # ------------------------------------------------------------------------
            # 綜合判斷：所有檢查都通過才算有效
            # ------------------------------------------------------------------------
            valid_this_attempt = valid_robot & valid_goal & valid_obstacles & needs_position
            
            # 更新 needs_position：有效的不再需要
            needs_position = needs_position & ~valid_this_attempt
        
        # 如果達到最大嘗試次數仍有環境找不到有效位置
        # 允許在整個場景範圍內重新嘗試（降級策略）
        if needs_position.any():
            num_failed = needs_position.sum().item()
            # 使用全場景範圍重新生成
            fallback_x = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            fallback_y = torch.rand(num_failed, device=device) * (2 * spawn_range) - spawn_range
            pos[needs_position, 0] = fallback_x
            pos[needs_position, 1] = fallback_y
        
        # 記錄已放置的位置
        placed_positions[:, i, :] = pos[:, :2]

        # ------------------------------------------------------------------------
        # 生成姿態（不旋轉）
        # ------------------------------------------------------------------------
        quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
        quat[:, 0] = 1.0  # (w=1, x=0, y=0, z=0) = 不旋轉

        # ------------------------------------------------------------------------
        # 應用到模擬器
        # ------------------------------------------------------------------------
        getattr(env.scene, obstacle_name).write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),  # 拼接位置和姿態 [7]
            env_ids=env_ids  # 只更新指定的環境
        )
        # write_root_pose_to_sim：直接設定物體的位置和姿態
        # env_ids：指定哪些環境需要更新（避免影響其他環境）


def move_obstacles(
    env,
    env_ids,
    move_dt: float = 0.2,
    speed_range: float = 0.5,
    min_speed: float = 0.05,
    speed_jitter: float = 0.05,
    area_limit: float = 6.0,
):
    """動態移動障礙物（interval 事件）
    
    透過「連續速度 + 小擾動」讓障礙物平滑移動。
    
    Args:
        env: 環境實例
        env_ids: 需要更新的環境 ID（可能是 None、張量或列表）
        move_dt: 每次更新的時間步長（秒）
        speed_range: 最大速度（m/s）
        min_speed: 最小速度（m/s）
        speed_jitter: 速度小擾動（m/s）
        area_limit: 障礙物活動範圍限制（正負範圍）
    """
    # 處理 env_ids 參數（interval 模式可能傳入 None 或張量）
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, (list, tuple)):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor([env_ids], device=env.device, dtype=torch.long)
    
    num_resets = len(env_ids)
    device = env.device
    
    num_obstacles = getattr(env, "_num_obstacles", None)
    if num_obstacles is None:
        num_obstacles = _get_obstacle_num()
    if not hasattr(env, "_obstacle_velocities") or env._obstacle_velocities.shape[1] != num_obstacles:
        env._obstacle_velocities = torch.zeros(env.num_envs, num_obstacles, 2, device=device)
    
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        if not hasattr(env.scene, obstacle_name):
            continue
        
        obstacle = getattr(env.scene, obstacle_name)
        
        # 取出當前位置與速度
        pos = obstacle.data.root_pos_w[env_ids, :3].clone()
        vel = env._obstacle_velocities[env_ids, i, :].clone()
        
        # 如果速度為 0（初始狀態），初始化隨機速度
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        zero_speed_mask = (speed.squeeze(1) < 1e-6)
        if zero_speed_mask.any():
            # 為速度為 0 的障礙物初始化隨機速度
            num_zero = zero_speed_mask.sum().item()
            angle = torch.rand(num_zero, device=device) * 2.0 * math.pi - math.pi  # [-π, π]
            init_speed = torch.rand(num_zero, device=device) * (speed_range - min_speed) + min_speed
            vel[zero_speed_mask, 0] = init_speed * torch.cos(angle)
            vel[zero_speed_mask, 1] = init_speed * torch.sin(angle)
        
        # 速度小擾動（讓方向慢慢變化）
        if speed_jitter > 0.0:
            vel += (torch.rand(num_resets, 2, device=device) - 0.5) * 2.0 * speed_jitter
        
        # 速度限制
        speed = torch.linalg.norm(vel, dim=1, keepdim=True)
        speed = torch.clamp(speed, min_speed, speed_range)
        vel = vel / (torch.linalg.norm(vel, dim=1, keepdim=True) + 1e-6) * speed
        
        # 平滑位移
        pos[:, 0:2] += vel * move_dt
        
        # 邊界反彈
        hit_x = (pos[:, 0] <= -area_limit) | (pos[:, 0] >= area_limit)
        hit_y = (pos[:, 1] <= -area_limit) | (pos[:, 1] >= area_limit)
        vel[hit_x, 0] *= -1.0
        vel[hit_y, 1] *= -1.0
        
        pos[:, 0] = torch.clamp(pos[:, 0], -area_limit, area_limit)
        pos[:, 1] = torch.clamp(pos[:, 1], -area_limit, area_limit)
        
        # 取出當前姿態（保持不變）
        if hasattr(obstacle.data, "root_quat_w"):
            quat = obstacle.data.root_quat_w[env_ids, :4].clone()
        else:
            quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
            quat[:, 0] = 1.0
        
        # 應用到模擬器
        obstacle.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1),
            env_ids=env_ids,
        )
        
        # 回寫速度
        env._obstacle_velocities[env_ids, i, :] = vel


# ============================================================================
# 自適應課程學習框架（Phase 2.5）
# ============================================================================

# 全局變量：追蹤訓練統計
_adaptive_curriculum_stats = {
    "success_rate_history": [],
    "collision_rate_history": [],
    "window_size": 100,  # 滑動窗口大小（episodes）
    "current_difficulty": 3,  # 當前難度等級（障礙物數量）
    "min_difficulty": 3,  # 最小難度（3 個障礙物）
    "max_difficulty": 10,  # 最大難度（10 個障礙物）
}


def initialize_adaptive_curriculum(env: "ManagerBasedRLEnv", initial_difficulty: int = 3):
    """初始化自適應課程學習
    
    Args:
        env: 環境實例
        initial_difficulty: 初始難度等級（默認 3）
    """
    global _adaptive_curriculum_stats
    _adaptive_curriculum_stats["current_difficulty"] = initial_difficulty
    
    # 設置環境中的初始障礙物數量
    env._num_obstacles = initial_difficulty
    
    # 初始化課程學習參數
    if not hasattr(env, "_adaptive_curriculum_params"):
        env._adaptive_curriculum_params = {
            "num_obstacles": initial_difficulty,
            "min_robot_distance": 3.5,
            "min_goal_distance": 2.0,
            "min_obstacle_spacing": 2.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "difficulty": initial_difficulty,
        }


def get_success_rate(env: "ManagerBasedRLEnv") -> float:
    """計算當前成功率
    
    從終止管理器的統計信息中獲取成功率。
    成功率 = 到達目標的環境數 / 總環境數
    
    Args:
        env: 環境實例
        
    Returns:
        成功率（0.0-1.0）
    """
    termination_manager = env.termination_manager
    
    # 方法 1：從終止管理器的 _last_episode_dones 中獲取
    if hasattr(termination_manager, "_term_names") and hasattr(termination_manager, "_last_episode_dones"):
        # 查找 goal_reached 終止條件
        goal_reached_idx = None
        for i, name in enumerate(termination_manager._term_names):
            if name == "goal_reached":
                goal_reached_idx = i
                break
        
        if goal_reached_idx is not None:
            # 獲取最近一個 episode 的終止狀態
            goal_reached = termination_manager._last_episode_dones[:, goal_reached_idx]
            if goal_reached.numel() > 0 and goal_reached.any():
                success_rate = goal_reached.float().mean().item()
                return success_rate
    
    # 方法 2：從 extras 中獲取（如果有的話）
    if "log" in env.extras:
        # 檢查是否有 Episode_Termination 統計
        for key, value in env.extras["log"].items():
            if "Episode_Termination" in key and "goal_reached" in key:
                if isinstance(value, (int, float)):
                    return float(value)
        # 檢查是否有直接的 success_rate
        if "success_rate" in env.extras["log"]:
            return float(env.extras["log"]["success_rate"])
    
    # 默認返回 0.0（沒有統計信息）
    return 0.0


def get_collision_rate(env: "ManagerBasedRLEnv") -> float:
    """計算當前碰撞率
    
    從終止管理器的統計信息中獲取碰撞率。
    碰撞率 = (collision_contact 或 collision) 的環境數 / 總環境數
    
    Args:
        env: 環境實例
        
    Returns:
        碰撞率（0.0-1.0）
    """
    termination_manager = env.termination_manager
    
    # 方法 1：從終止管理器的 _last_episode_dones 中獲取
    if hasattr(termination_manager, "_term_names") and hasattr(termination_manager, "_last_episode_dones"):
        # 查找 collision 或 collision_contact 終止條件
        collision_idx = None
        collision_contact_idx = None
        
        for i, name in enumerate(termination_manager._term_names):
            if name == "collision":
                collision_idx = i
            elif name == "collision_contact":
                collision_contact_idx = i
        
        # 合併兩個碰撞終止條件（任一觸發都算碰撞）
        collision_combined = None
        if collision_idx is not None and collision_contact_idx is not None:
            collision_combined = (
                termination_manager._last_episode_dones[:, collision_idx] |
                termination_manager._last_episode_dones[:, collision_contact_idx]
            )
        elif collision_idx is not None:
            collision_combined = termination_manager._last_episode_dones[:, collision_idx]
        elif collision_contact_idx is not None:
            collision_combined = termination_manager._last_episode_dones[:, collision_contact_idx]
        
        if collision_combined is not None and collision_combined.any():
            collision_rate = collision_combined.float().mean().item()
            return collision_rate
    
    # 方法 2：從 extras 中獲取（如果有的話）
    if "log" in env.extras:
        # 檢查是否有 Episode_Termination 統計
        collision_rate = 0.0
        for key, value in env.extras["log"].items():
            if "Episode_Termination" in key and ("collision" in key or "collision_contact" in key):
                if isinstance(value, (int, float)):
                    collision_rate = max(collision_rate, float(value))  # 取最大值（任一觸發都算）
        if collision_rate > 0.0:
            return collision_rate
        # 檢查是否有直接的 collision_rate
        if "collision_rate" in env.extras["log"]:
            return float(env.extras["log"]["collision_rate"])
    
    # 默認返回 0.0（沒有統計信息）
    return 0.0


def update_adaptive_curriculum(
    env: "ManagerBasedRLEnv",
    success_rate: float,
    collision_rate: float,
    target_success_rate: float = 0.7,
    max_collision_rate: float = 0.3,
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.5,
) -> dict:
    """更新自適應課程學習難度
    
    根據成功率和碰撞率動態調整環境難度：
    - 如果成功率 > 80% 且碰撞率 < 20%：增加難度（增加障礙物數量）
    - 如果成功率 < 50% 或碰撞率 > 40%：降低難度（減少障礙物數量）
    - 否則：保持當前難度
    
    Args:
        env: 環境實例
        success_rate: 當前成功率（0.0-1.0）
        collision_rate: 當前碰撞率（0.0-1.0）
        target_success_rate: 目標成功率（默認 0.7）
        max_collision_rate: 最大可接受碰撞率（默認 0.3）
        difficulty_increase_threshold: 難度增加閾值（默認 0.8）
        difficulty_decrease_threshold: 難度降低閾值（默認 0.5）
        
    Returns:
        包含更新後難度參數的字典
    """
    global _adaptive_curriculum_stats
    
    # 更新歷史記錄
    _adaptive_curriculum_stats["success_rate_history"].append(success_rate)
    _adaptive_curriculum_stats["collision_rate_history"].append(collision_rate)
    
    # 保持歷史記錄在窗口大小內
    if len(_adaptive_curriculum_stats["success_rate_history"]) > _adaptive_curriculum_stats["window_size"]:
        _adaptive_curriculum_stats["success_rate_history"].pop(0)
        _adaptive_curriculum_stats["collision_rate_history"].pop(0)
    
    # 計算滑動平均
    if len(_adaptive_curriculum_stats["success_rate_history"]) >= 10:
        avg_success_rate = sum(_adaptive_curriculum_stats["success_rate_history"][-10:]) / 10
        avg_collision_rate = sum(_adaptive_curriculum_stats["collision_rate_history"][-10:]) / 10
    else:
        avg_success_rate = success_rate
        avg_collision_rate = collision_rate
    
    current_difficulty = _adaptive_curriculum_stats["current_difficulty"]
    min_difficulty = _adaptive_curriculum_stats["min_difficulty"]
    max_difficulty = _adaptive_curriculum_stats["max_difficulty"]
    
    # 決定是否調整難度
    should_increase = (
        avg_success_rate >= difficulty_increase_threshold and
        avg_collision_rate < max_collision_rate / 2 and
        current_difficulty < max_difficulty
    )
    
    should_decrease = (
        (avg_success_rate < difficulty_decrease_threshold or avg_collision_rate > max_collision_rate) and
        current_difficulty > min_difficulty
    )
    
    # 調整難度
    if should_increase:
        new_difficulty = min(current_difficulty + 1, max_difficulty)
        _adaptive_curriculum_stats["current_difficulty"] = new_difficulty
        print(f"[Adaptive Curriculum] 增加難度: {current_difficulty} -> {new_difficulty} "
              f"(成功率: {avg_success_rate:.2%}, 碰撞率: {avg_collision_rate:.2%})")
    elif should_decrease:
        new_difficulty = max(current_difficulty - 1, min_difficulty)
        _adaptive_curriculum_stats["current_difficulty"] = new_difficulty
        print(f"[Adaptive Curriculum] 降低難度: {current_difficulty} -> {new_difficulty} "
              f"(成功率: {avg_success_rate:.2%}, 碰撞率: {avg_collision_rate:.2%})")
    else:
        new_difficulty = current_difficulty
    
    # 根據難度等級計算參數
    # 難度 3-10 對應障礙物數量 3-10
    num_obstacles = new_difficulty
    
    # 根據難度調整距離參數
    # 難度越高，距離參數越小（更困難）
    difficulty_factor = (new_difficulty - min_difficulty) / (max_difficulty - min_difficulty)
    
    # 基礎距離參數（Phase 2 的設置）
    base_min_robot_distance = 3.5
    base_min_goal_distance = 2.0
    base_min_obstacle_spacing = 2.0
    
    # 根據難度調整（難度越高，距離越小）
    min_robot_distance = base_min_robot_distance * (1.0 - difficulty_factor * 0.3)  # 3.5 -> 2.45
    min_goal_distance = base_min_goal_distance * (1.0 - difficulty_factor * 0.3)  # 2.0 -> 1.4
    min_obstacle_spacing = base_min_obstacle_spacing * (1.0 - difficulty_factor * 0.3)  # 2.0 -> 1.4
    
    return {
        "num_obstacles": num_obstacles,
        "min_robot_distance": min_robot_distance,
        "min_goal_distance": min_goal_distance,
        "min_obstacle_spacing": min_obstacle_spacing,
        "success_rate": avg_success_rate,
        "collision_rate": avg_collision_rate,
        "difficulty": new_difficulty,
    }


def adaptive_curriculum_update(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    target_success_rate: float = 0.7,
    max_collision_rate: float = 0.3,
    difficulty_increase_threshold: float = 0.8,
    difficulty_decrease_threshold: float = 0.5,
):
    """自適應課程學習更新函數（用於 CurriculumTerm）
    
    此函數在每個 step 結束時被調用，檢查是否有環境重置，並更新統計和難度。
    實際的參數調整會在 reset 事件中進行。
    
    返回值：
        dict[str, float]: 包含課程學習狀態的字典，用於日誌記錄
        - difficulty: 當前難度等級
        - success_rate: 當前成功率
        - collision_rate: 當前碰撞率
    
    Args:
        env: 環境實例
        env_ids: 環境 ID 列表（Isaac Lab 課程學習管理器自動傳入）
        target_success_rate: 目標成功率
        max_collision_rate: 最大可接受碰撞率
        difficulty_increase_threshold: 難度增加閾值
        difficulty_decrease_threshold: 難度降低閾值
    """
    global _adaptive_curriculum_stats
    
    # 獲取當前難度等級（如果尚未初始化，使用默認值）
    current_difficulty = _adaptive_curriculum_stats.get("current_difficulty", 3)
    
    # 只在有環境重置時更新統計（episode 結束時）
    if not hasattr(env, "reset_buf") or not env.reset_buf.any():
        # 返回當前狀態（不更新）
        return {
            "difficulty": float(current_difficulty),
            "success_rate": 0.0,
            "collision_rate": 0.0,
        }
    
    # 獲取當前統計（從終止管理器的 _last_episode_dones 中獲取）
    success_rate = get_success_rate(env)
    collision_rate = get_collision_rate(env)
    
    # 更新難度（只在有有效統計時，或者至少有一些環境重置了）
    # 注意：即使統計為 0，也可能是因為沒有環境成功/碰撞，這本身也是信息
    if env.reset_buf.any():
        difficulty_params = update_adaptive_curriculum(
            env,
            success_rate,
            collision_rate,
            target_success_rate,
            max_collision_rate,
            difficulty_increase_threshold,
            difficulty_decrease_threshold,
        )
        
        # 將更新後的參數存儲在環境中，供 reset 事件使用
        if not hasattr(env, "_adaptive_curriculum_params"):
            env._adaptive_curriculum_params = {}
        env._adaptive_curriculum_params.update(difficulty_params)
        
        # 更新當前難度
        current_difficulty = difficulty_params.get("difficulty", current_difficulty)
    
    # 返回課程學習狀態字典（用於日誌記錄）
    # RSL-RL 會將這些值記錄到日誌中，方便監控訓練進度
    return {
        "difficulty": float(current_difficulty),
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
    }


def reset_obstacles_with_adaptive_params(
    env,
    env_ids,
    speed_range: float = 0.0,
    min_speed: float = 0.0,
    min_robot_distance: float = 3.5,
    min_goal_distance: float = 2.0,
    min_obstacle_spacing: float = 2.0,
    max_spawn_attempts: int = 50,
    boundary: float = 8.0,
):
    """重置障礙物位置（Phase 2.5：使用自適應課程學習參數）
    
    此函數會根據自適應課程學習的結果動態調整參數。
    策略：場景中創建 MAX_OBSTACLES 個障礙物，但根據難度等級只啟用前 N 個。
    
    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
        speed_range: 速度範圍（靜態障礙物不使用）
        min_speed: 最小速度（靜態障礙物不使用）
        min_robot_distance: 機器人與障礙物最小距離（會被動態調整）
        min_goal_distance: 目標與障礙物最小距離（會被動態調整）
        min_obstacle_spacing: 障礙物之間最小間距（會被動態調整）
        max_spawn_attempts: 最大生成嘗試次數
        boundary: 場景邊界
    """
    # 初始化課程學習（如果尚未初始化）
    if not hasattr(env, "_adaptive_curriculum_params") or not env._adaptive_curriculum_params:
        initialize_adaptive_curriculum(env, initial_difficulty=3)
    
    # 確保 env._num_obstacles 已設置（初始化時設置為 3）
    if not hasattr(env, "_num_obstacles") or env._num_obstacles is None:
        env._num_obstacles = 3  # 初始難度：3 個障礙物
    
    # 從環境中獲取課程學習參數（由 adaptive_curriculum_update 設置）
    if hasattr(env, "_adaptive_curriculum_params") and env._adaptive_curriculum_params:
        # 使用課程學習更新後的參數
        difficulty_params = env._adaptive_curriculum_params
        adaptive_min_robot_distance = difficulty_params.get("min_robot_distance", min_robot_distance)
        adaptive_min_goal_distance = difficulty_params.get("min_goal_distance", min_goal_distance)
        adaptive_min_obstacle_spacing = difficulty_params.get("min_obstacle_spacing", min_obstacle_spacing)
        adaptive_num_obstacles = difficulty_params.get("num_obstacles", env._num_obstacles)
    else:
        # 沒有課程學習參數時，使用初始參數
        adaptive_min_robot_distance = min_robot_distance
        adaptive_min_goal_distance = min_goal_distance
        adaptive_min_obstacle_spacing = min_obstacle_spacing
        adaptive_num_obstacles = env._num_obstacles  # 使用已設置的值（應該是 3）
    
    # 更新環境中的障礙物數量記錄
    env._num_obstacles = adaptive_num_obstacles
    
    # 準備環境 ID 列表（用於移動未使用的障礙物）
    if env_ids is None:
        env_ids_list = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_list = env_ids
    else:
        env_ids_list = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    # ⚠️ 重要：先將未使用的障礙物移動到場景外，再重置使用的障礙物
    # 這樣可以確保在 reset_obstacles 執行時，場景中只有 N 個障礙物
    max_obstacles = 10  # MAX_OBSTACLES
    for i in range(adaptive_num_obstacles, max_obstacles):
        obstacle_name = f"obstacle_{i}"
        if hasattr(env.scene, obstacle_name):
            obstacle = getattr(env.scene, obstacle_name)
            # 移動到場景外（Z 軸下方）
            pos = obstacle.data.root_pos_w[env_ids_list, :3].clone()
            pos[:, 2] = -10.0  # 移動到 Z = -10 米（場景外）
            quat = obstacle.data.root_quat_w[env_ids_list, :4].clone()
            
            obstacle.write_root_pose_to_sim(
                torch.cat([pos, quat], dim=-1),
                env_ids=env_ids_list,
            )
    
    # 調用原始的 reset_obstacles 函數，但使用動態參數
    # reset_obstacles 會根據 env._num_obstacles 只重置前 N 個障礙物
    # 注意：此時未使用的障礙物已經被移動到場景外，不會影響重置邏輯
    reset_obstacles(
        env,
        env_ids,
        speed_range=speed_range,
        min_speed=min_speed,
        min_robot_distance=adaptive_min_robot_distance,
        min_goal_distance=adaptive_min_goal_distance,
        min_obstacle_spacing=adaptive_min_obstacle_spacing,
        max_spawn_attempts=max_spawn_attempts,
        boundary=boundary,
    )