"""
Carter 導航 MDP 函數模組

此模組定義了 Carter 機器人導航任務的所有 MDP（馬可夫決策過程）組件：
- 動作函數：差速驅動動作實現
- 觀測函數：雷射雷達掃描、目標位置、目標距離等
- 獎勵函數：朝向目標速度、距離進度、到達目標、碰撞懲罰等
- 終止條件：到達目標、碰撞、翻倒、懸空等
- 事件函數：重置障礙物位置

所有函數都使用 PyTorch 張量進行批量計算，支援並行環境。
"""

from __future__ import annotations  # 啟用延遲類型註解評估（Python 3.7+）

import torch  # PyTorch 張量運算庫
from typing import TYPE_CHECKING  # 類型檢查工具

import isaaclab.utils.math as math_utils  # Isaac Lab 數學工具（四元數旋轉等）
from isaaclab.assets import Articulation  # 關節式機器人類別
from isaaclab.managers import SceneEntityCfg  # 場景實體配置
from isaaclab.sensors import RayCaster  # 射線投射感測器類別
from isaaclab.managers import ActionTerm, ActionTermCfg  # 動作項基類和配置
from isaaclab.utils import configclass  # 配置類別裝飾器

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 類型檢查時導入環境類別

"""
======================
差速驅動動作 (Differential Drive Action)
======================
"""

def _get_lidar_distances(sensor: RayCaster) -> torch.Tensor:
    """
    從 RayCaster 獲取實際距離
    
    計算每條射線從感測器位置到命中點的實際距離。
    這比直接使用 ray_hits_w 更準確，因為考慮了感測器的實際位置。
    
    Args:
        sensor: 雷射雷達感測器實例
        
    Returns:
        距離張量，形狀為 [num_envs, num_rays]，單位為公尺
    """
    sensor_pos = sensor.data.pos_w  # 感測器位置 [num_envs, 3]（世界座標系）
    hit_points = sensor.data.ray_hits_w  # 射線命中點 [num_envs, num_rays, 3]（世界座標系）
    # 計算從感測器位置到每個命中點的歐幾里得距離
    return torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)

class DifferentialDriveAction(ActionTerm):
    """
    差速驅動動作類別
    
    實現差速驅動機器人的速度控制。動作空間為 2 維：
    - [0]: 線性速度（前進/後退）
    - [1]: 角速度（旋轉）
    
    動作值範圍為 [-1, 1]，會被縮放到實際的速度範圍。
    速度會從機器人局部座標系轉換到世界座標系後應用到機器人基座。
    """
    
    cfg: DifferentialDriveActionCfg  # 動作配置
    _asset: Articulation  # 機器人資產實例
    
    def __init__(self, cfg: DifferentialDriveActionCfg, env):
        """
        初始化差速驅動動作
        
        Args:
            cfg: 動作配置
            env: 環境實例
        """
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]  # 獲取機器人資產
        # 初始化動作緩衝區：原始動作（[-1, 1]）和處理後的動作（實際速度）
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 2, device=self.device)
    
    @property
    def action_dim(self) -> int:
        """動作維度：2（線性速度和角速度）"""
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """原始動作：[-1, 1] 範圍的標準化動作"""
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """處理後的動作：實際的速度值（m/s 和 rad/s）"""
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        """
        處理動作：將標準化動作 [-1, 1] 轉換為實際速度
        
        Args:
            actions: 標準化動作張量，形狀為 [num_envs, 2]
                    - actions[:, 0]: 線性速度（-1 到 1）
                    - actions[:, 1]: 角速度（-1 到 1）
        """
        self._raw_actions[:] = actions  # 保存原始動作
        # 將標準化動作縮放到實際速度範圍
        self._processed_actions[:, 0] = actions[:, 0] * self.cfg.max_linear_velocity  # 線性速度
        self._processed_actions[:, 1] = actions[:, 1] * self.cfg.max_angular_velocity  # 角速度
    
    def apply_actions(self):
        """
        應用動作：將速度從機器人局部座標系轉換到世界座標系並應用到機器人
        
        這是正確的差速驅動實現，確保機器人無論朝向如何都能正確移動。
        """
        # 獲取機器人當前朝向（四元數）
        robot_quat_w = self._asset.data.root_quat_w
        
        # 構建機器人局部座標系的速度向量 [v_forward, 0, 0]
        # 在機器人局部座標系中，X 軸是前進方向
        num_envs = self._env.num_envs
        local_velocity = torch.zeros(num_envs, 3, device=self.device)
        local_velocity[:, 0] = self._processed_actions[:, 0]  # X 軸是前進方向
        
        # 將局部速度轉換到全局（世界）座標系
        # 使用四元數旋轉將局部速度向量轉換為世界座標系
        global_linear_velocity = math_utils.quat_rotate(robot_quat_w, local_velocity)
        
        # 構造完整的 6D 速度向量 [vx, vy, vz, wx, wy, wz]
        root_velocity = torch.zeros(num_envs, 6, device=self.device)
        root_velocity[:, 0:3] = global_linear_velocity  # 全局線速度（X, Y, Z）
        root_velocity[:, 5] = self._processed_actions[:, 1]  # 全局 Z 軸角速度（繞垂直軸旋轉）
        
        # 將速度應用到機器人基座（直接設置根速度）
        self._asset.write_root_velocity_to_sim(root_velocity)


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """
    差速驅動動作配置類別
    
    定義差速驅動動作的參數，包括最大線性速度和最大角速度。
    """
    class_type: type = DifferentialDriveAction  # 動作類別：使用 DifferentialDriveAction
    asset_name: str = "robot"  # 資產名稱：機器人在場景中的名稱
    max_linear_velocity: float = 2.0   # 最大線性速度：2.0 m/s（前進/後退的最大速度）
    max_angular_velocity: float = 2.0  # 最大角速度：2.0 rad/s（旋轉的最大角速度，約 115 度/秒）

"""
======================
觀測函數 (Observation Functions)
======================
"""

def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    雷射雷達掃描觀測函數 - 帶安全檢查
    
    從雷射雷達感測器獲取距離數據，進行歸一化和安全處理。
    返回歸一化的距離值（0 到 1），其中 1 表示達到最大檢測距離。
    
    Args:
        env: 環境實例
        sensor_cfg: 感測器配置（包含感測器名稱）
        
    Returns:
        歸一化的距離張量，形狀為 [num_envs, num_rays]，值範圍 [0, 1]
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]  # 獲取雷射雷達感測器
    distances = _get_lidar_distances(sensor)  # 計算實際距離
    # 備用方法（已註釋）：直接使用 ray_hits_w 的第一個分量
    #distances = sensor.data.ray_hits_w[..., 0]
    #distances = torch.nan_to_num(distances, nan=10.0, posinf=10.0, neginf=0.0)
    
    # 將距離限制在 [0, max_distance] 範圍內
    distances = torch.clamp(distances, 0.0, 10.0)
    
    # 歸一化：將距離除以最大檢測距離，得到 [0, 1] 範圍的值
    normalized = distances / sensor.cfg.max_distance
    
    # 安全檢查：處理 NaN 和無窮大值
    # - NaN（未命中）→ 1.0（表示達到最大距離）
    # - 正無窮大 → 1.0
    # - 負無窮大 → 0.0
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    # 再次限制範圍，確保值在 [0, 1] 內
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    return normalized

def safe_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    安全的上一步動作觀測 - 防止 NaN 傳播
    
    獲取上一個時間步的動作，並進行安全處理，防止異常值影響策略學習。
    這對於學習平滑的動作序列很重要。
    
    Args:
        env: 環境實例
        
    Returns:
        清理後的動作張量，形狀為 [num_envs, action_dim]，值範圍 [-1, 1]
    """
    actions = env.action_manager.action  # 獲取當前動作（實際上是上一步的動作）
    
    # 清理動作：處理異常值
    # - NaN → 0.0（無動作）
    # - 正無窮大 → 1.0（最大正向動作）
    # - 負無窮大 → -1.0（最大負向動作）
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    # 限制動作範圍在 [-1, 1] 內
    actions = torch.clamp(actions, -1.0, 1.0)
    
    return actions

def goal_position_in_robot_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    目標相對位置觀測 - 帶安全檢查
    
    計算目標點在機器人座標系中的位置（僅 X, Y 分量）。
    這對於導航決策很重要，因為策略需要知道目標相對於機器人的方向。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        目標相對位置張量，形狀為 [num_envs, 2]，單位為公尺
        - [:, 0]: X 方向（機器人前方為正）
        - [:, 1]: Y 方向（機器人左側為正）
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置（世界座標系）
    robot_pos_w = asset.data.root_pos_w[:, :3]  # 機器人位置（世界座標系）
    robot_quat_w = asset.data.root_quat_w  # 機器人朝向（四元數）
    
    # 將目標位置從世界座標系轉換到機器人座標系
    # subtract_frame_transforms 計算兩個座標系之間的變換
    goal_vec_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w,  # 源座標系（機器人）
        goal_pos_w, torch.zeros_like(robot_quat_w)  # 目標座標系（目標點，無旋轉）
    )
    
    # 只取 X, Y 分量（忽略 Z 高度）
    result = goal_vec_b[:, :2]
    # 安全檢查：處理異常值
    result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    # 限制範圍在 [-10, 10] 公尺內
    result = torch.clamp(result, -10.0, 10.0)
    
    return result


def goal_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    目標距離觀測 - 帶安全檢查
    
    計算機器人到目標點的直線距離（僅考慮 X, Y 平面）。
    用於判斷是否接近目標，以及作為獎勵函數的輸入。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        距離張量，形狀為 [num_envs, 1]，單位為公尺，值範圍 [0, 20]
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :2]  # 機器人位置（僅 X, Y）
    
    # 計算歐幾里得距離（X, Y 平面）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1, keepdim=True)
    
    # 安全檢查：處理異常值
    # - NaN → 10.0（假設距離較遠）
    # - 正無窮大 → 10.0
    # - 負無窮大 → 0.0
    distance = torch.nan_to_num(distance, nan=10.0, posinf=10.0, neginf=0.0)
    # 限制距離範圍在 [0, 20] 公尺內
    distance = torch.clamp(distance, 0.0, 20.0)
    
    return distance


"""
======================
獎勵函數 (Reward Functions)
======================
"""

def velocity_toward_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float=1.0) -> torch.Tensor:
    """
    朝向目標的速度獎勵
    
    直接獎勵機器人朝向目標移動的速度分量，這比距離獎勵更直接有效。
    避免了進度追蹤的複雜性，鼓勵機器人快速接近目標。
    
    當機器人距離目標小於 min_dist 時，不給予速度獎勵（避免在目標附近振盪）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        min_dist: 最小距離閾值（公尺），小於此距離時不給予速度獎勵
        
    Returns:
        速度獎勵張量，形狀為 [num_envs]，值範圍 [0, max_linear_velocity]
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :3]  # 機器人位置
    robot_vel_w = asset.data.root_lin_vel_w[:, :2]  # 機器人線速度（僅 X, Y 分量）
    
    # 計算目標方向向量（從機器人指向目標）
    goal_direction = goal_pos_w[:, :2] - robot_pos_w[:, :2]
    goal_distance = torch.norm(goal_direction, dim=1)  # 計算距離
    # 歸一化方向向量（避免除零）
    goal_direction = goal_direction / (goal_distance.unsqueeze(1) + 1e-6)
    
    # 計算速度在目標方向上的投影（點積）
    # 正值表示朝向目標移動，負值表示遠離目標
    velocity_projection = torch.sum(robot_vel_w * goal_direction, dim=1)
    # 只獎勵正向速度（朝向目標移動），負速度設為 0
    velocity_projection = torch.clamp(velocity_projection, min=0.0)
    
    # 閘控：當距離目標小於 min_dist 時，不給予速度獎勵
    # 這防止機器人在目標附近快速振盪以獲得獎勵
    gated_velocity = torch.where(
        goal_distance > min_dist,  # 距離大於閾值
        velocity_projection,  # 給予速度獎勵
        torch.zeros_like(velocity_projection),  # 否則為 0
    )
    
    return gated_velocity


# 備用獎勵函數（已註釋）：距離懲罰
# def distance_to_goal_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     距離懲罰獎勵（指數衰減）
#     距離越遠，獎勵越小（指數衰減）
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     goal_pos_w = env.command_manager.get_command("goal_command")
#     robot_pos_w = asset.data.root_pos_w[:, :2]
#     
#     distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)
#     
#     # 距離越遠，獎勵越小（指數衰減）
#     return torch.exp(-distance / 5.0)

def progress_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    朝向目標的距離進展獎勵（progress reward）
    
    只有在本時間步比上一步更接近目標時才有正獎勵。
    這防止機器人在目標附近振盪刷分，鼓勵持續接近目標。
    
    關鍵設計：剛重置的環境（episode_length_buf == 0）不給予進展獎勵，
    因為沒有上一步的距離可以比較。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        進展獎勵張量，形狀為 [num_envs]，正值表示接近目標，負值表示遠離目標
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :2]  # 機器人位置（僅 X, Y）
    curr_dist = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)  # 當前距離

    # 初始化緩衝區（只在第一次調用時執行）
    # 用於儲存上一步的距離，以便計算進展
    if not hasattr(env, "_prev_goal_dist"):
        env._prev_goal_dist = curr_dist.clone()

    # 計算進展：上一步距離 - 當前距離
    # 正值表示接近目標，負值表示遠離目標
    reward = env._prev_goal_dist - curr_dist

    # 關鍵設計：剛重置的環境（episode_length_buf == 0）不給予進展獎勵
    # 因為沒有上一步的距離可以比較，避免錯誤的獎勵信號
    reward = torch.where(
        env.episode_length_buf == 0,  # 如果是剛重置的環境
        torch.zeros_like(reward),  # 獎勵為 0
        reward,  # 否則使用計算出的進展獎勵
    )

    # 更新上一步距離（使用 detach 避免梯度傳播）
    env._prev_goal_dist = curr_dist.detach()
    return reward

def reaching_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """
    到達目標獎勵
    
    當機器人成功到達目標（距離小於閾值）時給予獎勵。
    這是最重要的獎勵項，確保策略優先學習到達目標。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        threshold: 到達閾值（公尺），距離小於此值視為到達目標
        
    Returns:
        到達獎勵張量，形狀為 [num_envs]，值為 1.0（到達）或 0.0（未到達）
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :2]  # 機器人位置（僅 X, Y）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)  # 計算距離
    # 返回布林值轉換為浮點數：到達為 1.0，未到達為 0.0
    return (distance < threshold).float()


def obstacle_avoidance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
    collision_distance: float = 0.3,
) -> torch.Tensor:
    """
    避障獎勵
    
    根據機器人與障礙物的最小距離給予獎勵。
    距離越遠，獎勵越高；距離越近，獎勵越低（甚至為負）。
    
    Args:
        env: 環境實例
        sensor_cfg: 感測器配置（包含雷射雷達名稱）
        safe_distance: 安全距離（公尺），達到此距離時獎勵為 1.0
        collision_distance: 碰撞距離（公尺），小於此距離時獎勵為 -1.0
        
    Returns:
        避障獎勵張量，形狀為 [num_envs]，值範圍 [-1, 1]
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]  # 獲取雷射雷達感測器
    distances = sensor.data.ray_hits_w[..., 0]  # 獲取距離數據（簡化版本）
    min_distance = torch.min(distances, dim=1)[0]  # 找到每個環境的最小距離
    
    # 線性插值：將距離映射到 [-1, 1] 範圍
    # 距離 = collision_distance → 獎勵 = -1.0
    # 距離 = safe_distance → 獎勵 = 1.0
    reward = torch.clamp(
        (min_distance - collision_distance) / (safe_distance - collision_distance),
        min=-1.0,
        max=1.0,
    )
    return reward

def heading_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    朝向目標獎勵
    
    獎勵機器人朝向目標的朝向角度。使用夾角的餘弦值作為獎勵：
    - 正對目標（夾角 0°）→ 1.0
    - 垂直目標（夾角 90°）→ 0.0
    - 背對目標（夾角 180°）→ -1.0
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        朝向獎勵張量，形狀為 [num_envs]，值範圍 [-1, 1]
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :3]  # 機器人位置
    robot_quat_w = asset.data.root_quat_w  # 機器人朝向（四元數）
    
    # 計算目標方向向量（世界座標系）
    goal_direction_w = goal_pos_w - robot_pos_w
    # 歸一化方向向量
    goal_direction_w = goal_direction_w / (torch.norm(goal_direction_w, dim=1, keepdim=True) + 1e-6)
    
    # 創建批量前向向量（機器人局部座標系）
    num_envs = robot_quat_w.shape[0]
    forward_vec = torch.zeros(num_envs, 3, device=env.device)
    forward_vec[:, 0] = 1.0  # X 軸是前方（機器人局部座標系）
    
    # 將前向向量轉換到世界座標系
    forward_w = math_utils.quat_rotate(robot_quat_w, forward_vec)
    
    # 計算夾角的餘弦值（點積）
    # 這表示機器人前方向與目標方向的對齊程度
    heading_reward = torch.sum(forward_w * goal_direction_w, dim=1)
    
    return heading_reward


def collision_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.3) -> torch.Tensor:
    """
    碰撞懲罰
    
    當機器人過於接近障礙物（最小距離小於閾值）時給予懲罰。
    這是一個二元懲罰：要麼發生碰撞風險（1.0），要麼沒有（0.0）。
    
    Args:
        env: 環境實例
        sensor_cfg: 感測器配置（包含雷射雷達名稱）
        threshold: 碰撞距離閾值（公尺），小於此值視為碰撞風險
        
    Returns:
        碰撞懲罰張量，形狀為 [num_envs]，值為 1.0（碰撞風險）或 0.0（安全）
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]  # 獲取雷射雷達感測器
    distances = _get_lidar_distances(sensor)  # 計算實際距離
    # 備用方法（已註釋）：直接使用 ray_hits_w
    #distances = sensor.data.ray_hits_w[..., 0]
    min_distance = torch.min(distances, dim=1)[0]  # 找到每個環境的最小距離
    # 返回布林值轉換為浮點數：碰撞風險為 1.0，安全為 0.0
    return (min_distance < threshold).float()

def time_out_penalty(env: ManagerBasedRLEnv, weight: float = -1.0) -> torch.Tensor:
    """
    超時懲罰
    
    對超時的環境施加懲罰，鼓勵策略在時間限制內完成任務。
    只有超時的環境（terminated_buf == 1）才會受到懲罰。
    
    Args:
        env: 環境實例
        weight: 懲罰權重（通常為負值，如 -1.0）
        
    Returns:
        超時懲罰張量，形狀為 [num_envs]
        - 超時的環境：weight（通常為負值）
        - 未超時的環境：0.0
    """
    # terminated 已經是張量 [num_envs]
    terminated_buf = env.termination_manager.terminated  # [num_envs] 終止狀態緩衝區
    
    # 創建獎勵張量，只有超時的環境才懲罰
    # 假設超時對應 terminated_buf == 1
    reward = torch.where(
        terminated_buf == 1,  # 如果環境已終止（超時）
        torch.full_like(terminated_buf, weight),  # 給予懲罰
        torch.zeros_like(terminated_buf)  # 否則為 0
    )
    
    return reward  # [num_envs]


"""
======================
終止條件 (Termination Conditions)
======================
"""

def goal_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """
    到達目標終止條件
    
    檢查機器人是否成功到達目標點。當距離小於閾值時，視為到達目標，回合結束。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        threshold: 到達閾值（公尺），距離小於此值視為到達目標
        
    Returns:
        終止標誌張量，形狀為 [num_envs]，True 表示到達目標，False 表示未到達
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    goal_pos_w = env.command_manager.get_command("goal_command")  # 獲取目標位置
    robot_pos_w = asset.data.root_pos_w[:, :2]  # 機器人位置（僅 X, Y）
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1)  # 計算距離
    # 返回布林值：距離小於閾值為 True（到達目標）
    return distance < threshold


def collision_occurred(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.3) -> torch.Tensor:
    """
    碰撞發生終止條件
    
    檢查機器人是否與障礙物發生碰撞。當最小距離小於閾值時，視為發生碰撞，回合結束。
    
    Args:
        env: 環境實例
        sensor_cfg: 感測器配置（包含雷射雷達名稱）
        threshold: 碰撞距離閾值（公尺），小於此值視為發生碰撞
        
    Returns:
        終止標誌張量，形狀為 [num_envs]，True 表示發生碰撞，False 表示安全
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]  # 獲取雷射雷達感測器
    distances = _get_lidar_distances(sensor)  # 計算實際距離
    # 備用方法（已註釋）：直接使用 ray_hits_w
    #distances = sensor.data.ray_hits_w[..., 0]
    min_distance = torch.min(distances, dim=1)[0]  # 找到每個環境的最小距離
    # 返回布林值：最小距離小於閾值為 True（發生碰撞）
    return min_distance < threshold

def robot_tipped_over(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    機器人翻倒終止條件
    
    檢查機器人是否翻倒。通過檢查機器人 Z 軸（垂直軸）在世界座標系中的垂直分量來判斷。
    如果垂直分量小於 0.5，表示機器人傾斜過大，視為翻倒。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        終止標誌張量，形狀為 [num_envs]，True 表示翻倒，False 表示正常
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    
    # 檢查 Z 軸朝向（應該向上）
    quat = asset.data.root_quat_w  # 機器人朝向（四元數）
    
    # 創建批量 Z 軸向量（機器人局部座標系）
    num_envs = quat.shape[0]
    z_vec = torch.zeros(num_envs, 3, device=env.device)
    z_vec[:, 2] = 1.0  # 局部 Z 軸（垂直向上）
    
    # 將 Z 軸向量旋轉到世界座標系
    z_axis = math_utils.quat_rotate(quat, z_vec)
    
    # 如果 Z 軸的垂直分量 < 0.5，認為翻倒
    # 正常情況下，Z 軸應該垂直向上，z_axis[:, 2] 應該接近 1.0
    is_tipped = z_axis[:, 2] < 0.5
    
    return is_tipped


def robot_flying(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    機器人懸空終止條件
    
    檢查機器人是否懸空（離開地面）。當機器人高度超過閾值時，視為異常狀態，回合結束。
    這通常表示機器人發生了物理異常（如被彈飛）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（包含機器人名稱）
        
    Returns:
        終止標誌張量，形狀為 [num_envs]，True 表示懸空，False 表示正常
    """
    asset: Articulation = env.scene[asset_cfg.name]  # 獲取機器人資產
    
    # 檢查高度（Z 座標）
    height = asset.data.root_pos_w[:, 2]  # 機器人 Z 座標（高度）
    is_flying = height > 1.0  # 超過 1 公尺認為異常（正常情況下應該貼地）
    
    return is_flying


"""
======================
事件函數 (Event Functions)
======================
"""

def reset_obstacles(env, env_ids):
    """
    重置所有障礙物位置
    
    在環境重置時，隨機重新生成所有障礙物的位置。
    這確保每個回合的障礙物配置都不同，提高訓練的多樣性。
    
    Args:
        env: 環境實例
        env_ids: 需要重置的環境 ID 列表
    """
    num_resets = len(env_ids)  # 需要重置的環境數量
    device = env.device  # 設備（CPU 或 GPU）
    
    # 遍歷所有障礙物（共 8 個）
    for i in range(8):
        obstacle_name = f"obstacle_{i}"  # 障礙物名稱：obstacle_0, obstacle_1, ...
        # 檢查場景中是否存在該障礙物
        if hasattr(env.scene, obstacle_name):
            # 生成隨機位置（X, Y 在 [-6, 6] 範圍內，Z 固定為 0.5）
            pos = torch.zeros(num_resets, 3, device=device, dtype=torch.float32)
            pos[:, 0] = torch.rand(num_resets, device=device) * 12.0 - 6.0  # X: [-6, 6]
            pos[:, 1] = torch.rand(num_resets, device=device) * 12.0 - 6.0  # Y: [-6, 6]
            pos[:, 2] = 0.5  # Z: 0.5（使底部貼地）
            
            # 生成姿態（四元數，無旋轉）
            quat = torch.zeros(num_resets, 4, device=device, dtype=torch.float32)
            quat[:, 0] = 1.0  # w=1, x=y=z=0（無旋轉）
            
            # 將位置和姿態應用到障礙物
            getattr(env.scene, obstacle_name).write_root_pose_to_sim(
                torch.cat([pos, quat], dim=-1),  # 合併位置和姿態 [num_resets, 7]
                env_ids=env_ids  # 指定需要重置的環境 ID
            )