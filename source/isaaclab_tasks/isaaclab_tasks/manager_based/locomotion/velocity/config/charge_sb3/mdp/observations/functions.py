"""觀測函數集合

包含所有機器人觀測函數:
- 雷達觀測 (lidar_scan, lidar_scan_2d_sweep)
- 目標觀測 (goal_position_in_robot_frame, goal_distance)
- 機器人狀態 (base_velocity_xy, safe_last_action, time_remaining_ratio, alive_flag)
- 障礙物觀測 (dynamic_obstacles_state)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
import isaaclab.utils.math as math_utils

from .utils import check_finite

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# 雷達觀測
# ============================================================================

def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """激光雷達掃描觀測(舊版 VLP16 版本 - 已註釋保留)
    
    從雷達獲取距離數據,並進行安全處理(防止 NaN、無窮大)。
    這是修復 PPO std>=0 錯誤的關鍵函數。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置(引用場景中的雷達)
    
    Returns:
        shape [num_envs, num_rays]: 歸一化的距離值 [0, 1]
        0 = 非常近(0米)
        1 = 最遠(max_distance)
    """
    # 獲取雷達傳感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置(世界座標系)
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點(世界座標系)
    
    # 計算從感測器位置到每個碰撞點的歐幾里得距離
    distances = torch.norm(hit_points - sensor_pos.unsqueeze(1), dim=-1)
    
    # ========================================================================
    # 關鍵: NaN/Inf 消毒(防止 PPO std>=0 錯誤)
    # ========================================================================
    # 1) 把 NaN/Inf 替換為 max_distance(射線沒打到任何東西)
    max_range = sensor.cfg.max_distance
    distances = torch.nan_to_num(distances, nan=max_range, posinf=max_range, neginf=0.0)
    
    # 2) clip 到合理範圍 [0, max_range]
    distances = torch.clamp(distances, 0.0, max_range)
    
    # 3) normalize 到 [0, 1](超重要,避免數值爆)
    normalized = distances / max_range
    
    # 4) 再次消毒(防止除法產生異常)
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # 5) 最終 finite 檢查(如果還有問題會立即報錯,方便定位)
    check_finite("lidar_scan", normalized, raise_on_error=True)
    
    return normalized


def lidar_scan_2d_sweep(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """2D 平面掃描觀測(72 個角度)
    
    將 3D 點雲資料轉換為鳥瞰圖的可行駛區域後,計算每個角度的最近障礙物距離。
    設計:將機器人周圍 360 度切分為 72 個等份(每 5 度一格)。
    內容:每個角度紀錄「從機器狗中心點到最近障礙物的距離」。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置(引用場景中的雷達)
    
    Returns:
        shape [num_envs, 72]: 歸一化的距離值 [0, 1]
        0 = 非常近(0米)
        1 = 最遠(max_distance)
        每個值對應一個 5 度扇形區域的最近障礙物距離
    """
    # 獲取雷達傳感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3]
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    
    # ========================================================================
    # 步驟 1: 將 3D 點雲轉換為鳥瞰圖(2D 平面投影)
    # ========================================================================
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2] 只取 X, Y 座標
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2]
    
    # 計算 2D 平面距離
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    
    # ========================================================================
    # 步驟 2: 處理射線數據(通常已經是 72 個角度)
    # ========================================================================
    num_rays = distances_2d.shape[1]
    num_sectors = 72  # 對應 horizontal_res=5.0(360/5=72)
    
    # 如果射線數正好是 72,直接使用
    if num_rays == num_sectors:
        distances_2d_sectors = distances_2d
    else:
        # 備用實現:如果射線數不是 72,需要重新分組
        vectors_2d = hit_points_2d - sensor_pos_2d.unsqueeze(1)
        angles = torch.atan2(vectors_2d[:, :, 1], vectors_2d[:, :, 0])
        angles_deg = torch.rad2deg(angles + torch.pi)  # [0, 360]
        
        sector_size = 360.0 / num_sectors  # 5 度
        sector_indices = (angles_deg / sector_size).long()
        sector_indices = torch.clamp(sector_indices, 0, num_sectors - 1)
        
        # 對每個扇形區域,找到最近的障礙物距離
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
            mask = (sector_indices == sector_idx)
            sector_distances = torch.where(
                mask,
                distances_2d,
                torch.full_like(distances_2d, float('inf'))
            )
            min_dist, _ = torch.min(sector_distances, dim=1)
            min_dist = torch.where(
                torch.isfinite(min_dist),
                min_dist,
                torch.full_like(min_dist, max_range)
            )
            distances_2d_sectors[:, sector_idx] = min_dist
    
    # ========================================================================
    # 步驟 3: 安全處理和歸一化
    # ========================================================================
    max_range = sensor.cfg.max_distance
    
    # 1) 把 NaN/Inf 替換為 max_distance
    distances_2d_sectors = torch.nan_to_num(
        distances_2d_sectors, nan=max_range, posinf=max_range, neginf=0.0
    )
    
    # 2) clip 到合理範圍 [0, max_range]
    distances_2d_sectors = torch.clamp(distances_2d_sectors, 0.0, max_range)
    
    # 3) normalize 到 [0, 1]
    normalized = distances_2d_sectors / max_range
    
    # 4) 再次消毒
    normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # 5) 最終 finite 檢查
    check_finite("lidar_scan_2d_sweep", normalized, raise_on_error=True)
    
    return normalized


# ============================================================================
# 目標觀測
# ============================================================================

def goal_position_in_robot_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標相對位置觀測(機器人座標系)

    計算目標位置相對於機器人的位置(前後、左右)。

    🔥 訓練策略：agent 追蹤紅色箭頭（goal_command）
    - 訓練模式（VirtualPlanner）：_local_goal_world = goal_command（紅色箭頭）
    - 推理模式（AIT*）：_local_goal_world = AIT* 路徑中的局部航點

    優先順序：
    1. env._local_goal_world (訓練時=紅色箭頭, 推理時=AIT*航點)
    2. goal_command (最終目標，紅色箭頭)

    Args:
        env: 環境實例
        asset_cfg: 資產配置(引用機器人)

    Returns:
        shape [num_envs, 2]: 相對位置 [X, Y] 在機器人座標系
        例如: [2.5, -1.0] = 前方 2.5 米,右邊 1 米
    """
    # 獲取機器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 🔥 關鍵：_local_goal_world 在訓練時就是紅色箭頭位置
    # （見 aitstar_integration.py 中的 plan_aitstar_and_update_local_goal）
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world  # [N, 2] 訓練時=紅色箭頭, 推理時=AIT*航點
        # 🔥 重要：清理 _local_goal_world 的 NaN/Inf
        goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)
        # 擴展為 [N, 3] 以匹配 subtract_frame_transforms 的要求
        goal_pos_w = torch.cat([goal_pos_w, torch.zeros(goal_pos_w.shape[0], 1, device=goal_pos_w.device)], dim=1)
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")  # 紅色箭頭（最終目標）
        # 🔥 清理 goal_command 的 NaN/Inf
        goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w

    # 🔥 清理機器人狀態的 NaN/Inf
    robot_pos_w = torch.nan_to_num(robot_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)
    robot_quat_w = torch.nan_to_num(robot_quat_w, nan=0.0, posinf=1.0, neginf=-1.0)
    # 歸一化四元數
    robot_quat_w = robot_quat_w / (torch.norm(robot_quat_w, dim=1, keepdim=True) + 1e-6)

    # 座標系變換: 世界座標 → 機器人座標
    goal_vec_b, _ = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w,
        goal_pos_w, torch.zeros_like(robot_quat_w)
    )

    # 只保留 XY(忽略 Z 高度)
    result = goal_vec_b[:, :2]

    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = torch.clamp(result, -10.0, 10.0)

    check_finite("goal_position", result, raise_on_error=True)

    return result


def goal_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_distance: float = 8.0) -> torch.Tensor:
    """目標距離觀測（歸一化版本）

    計算機器人到目標的直線距離，並歸一化到 [0, 1] 範圍。

    🔥 Bug 修復：歸一化到 [0, 1]，避免神經網路權重崩潰！
    - 之前：返回 [0, 20] 米的裸數值
    - 現在：返回 [0, 1] 的歸一化值
    - 0 = 已到達目標，1 = 最遠距離

    🔥 訓練策略：計算到紅色箭頭的距離
    - 訓練模式（VirtualPlanner）：_local_goal_world = goal_command（紅色箭頭）
    - 推理模式（AIT*）：_local_goal_world = AIT* 路徑中的局部航點

    優先順序：
    1. env._local_goal_world (訓練時=紅色箭頭, 推理時=AIT*航點)
    2. goal_command (最終目標，紅色箭頭)

    Args:
        env: 環境實例
        asset_cfg: 資產配置(引用機器人)
        max_distance: 最大距離（米），用於歸一化。默認 8.0 米（對應目標生成範圍）

    Returns:
        shape [num_envs, 1]: 歸一化距離 [0, 1]
        0 = 已到達目標
        1 = 最遠距離 (max_distance 米)
    """
    # 獲取機器人
    asset: Articulation = env.scene[asset_cfg.name]

    # 🔥 關鍵修復：優先使用規劃器的局部目標
    if hasattr(env, '_local_goal_world'):
        goal_pos_w = env._local_goal_world
        # 🔥 清理 NaN/Inf
        goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)
    else:
        goal_pos_w = env.command_manager.get_command("goal_command")
        # 🔥 清理 NaN/Inf
        goal_pos_w = torch.nan_to_num(goal_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

    robot_pos_w = asset.data.root_pos_w[:, :2]  # 只要 XY

    # 🔥 清理機器人位置的 NaN/Inf
    robot_pos_w = torch.nan_to_num(robot_pos_w, nan=0.0, posinf=100.0, neginf=-100.0)

    # 計算歐幾里得距離
    distance = torch.norm(goal_pos_w[:, :2] - robot_pos_w, dim=1, keepdim=True)

    # 🔥 Bug 修復：歸一化到 [0, 1]
    # 這是關鍵！避免神經網路權重被大數值拉偏
    distance = torch.clamp(distance, 0.0, max_distance)
    normalized_distance = distance / max_distance

    # 安全處理
    normalized_distance = torch.nan_to_num(normalized_distance, nan=0.0, posinf=1.0, neginf=0.0)
    normalized_distance = torch.clamp(normalized_distance, 0.0, 1.0)

    check_finite("goal_distance", normalized_distance, raise_on_error=True)

    return normalized_distance


# ============================================================================
# 機器人狀態觀測
# ============================================================================

def safe_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """安全的上一步動作觀測
    
    返回上一步執行的動作(記憶),並清理異常值。
    
    Args:
        env: 環境實例
    
    Returns:
        shape [num_envs, 2]: 上一步的動作 [-1, 1]
    """
    # 獲取上一步動作
    actions = env.action_manager.action
    
    # 清理異常值
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = torch.clamp(actions, -1.0, 1.0)
    
    check_finite("actions", actions, raise_on_error=True)
    
    return actions


def base_velocity_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_linear_velocity: float = 1.5) -> torch.Tensor:
    """速度觀測(機器人座標系，歸一化版本)

    回傳機器人座標系下的線速度 [vx, vy]，歸一化到 [-1, 1] 範圍。

    🔥 Bug 修復：歸一化到 [-1, 1]，避免神經網路權重崩潰！
    - 之前：返回裸數值 (例如 1.5 m/s)
    - 現在：返回歸一化值 [-1, 1]
    - 1.0 = 最大前進速度 (max_linear_velocity)
    - -1.0 = 最大後退速度

    Args:
        env: 環境實例
        asset_cfg: 資產配置(引用機器人)
        max_linear_velocity: 最大線速度（米/秒），默認 1.5 m/s（對應 ActionsCfg）

    Returns:
        shape [num_envs, 2]: 歸一化速度 [vx, vy] in robot frame, 範圍 [-1, 1]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 取得世界座標速度與機器人姿態
    vel_w = asset.data.root_lin_vel_w[:, :3]
    quat_w = asset.data.root_quat_w

    # 🔥 重要：清理輸入數據的 NaN/Inf
    vel_w = torch.nan_to_num(vel_w, nan=0.0, posinf=10.0, neginf=-10.0)
    quat_w = torch.nan_to_num(quat_w, nan=0.0, posinf=1.0, neginf=-1.0)
    # 歸一化四元數
    quat_w = quat_w / (torch.norm(quat_w, dim=1, keepdim=True) + 1e-6)

    # 世界座標 → 機器人座標
    vel_b = math_utils.quat_apply_inverse(quat_w, vel_w)
    vel_xy = vel_b[:, :2]

    # 🔥 Bug 修復：歸一化到 [-1, 1]
    # 除以最大速度，讓數值在合理範圍內
    result = vel_xy / max_linear_velocity

    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    result = torch.clamp(result, -1.0, 1.0)
    check_finite("base_velocity_xy", result, raise_on_error=True)

    return result


def base_angular_velocity_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_angular_velocity: float = 1.5) -> torch.Tensor:
    """角速度觀測（yaw rate / ω_z，歸一化版本）

    提供機器人繞 Z 軸的角速度，幫助策略實現角速度阻尼。
    這是消除蛇行行為的關鍵觀測：
    - 策略可以感知「當前正在轉多快」
    - 從而學習在角速度過大時減少旋轉輸出
    - 實現類似 PD 控制器的阻尼效果

    🔥 Bug 修復：歸一化到 [-1, 1]，避免神經網路權重崩潰！
    - 之前：返回裸數值 (rad/s，例如 1.5)
    - 現在：返回歸一化值 [-1, 1]
    - 1.0 = 最大左轉速度 (max_angular_velocity)
    - -1.0 = 最大右轉速度

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        max_angular_velocity: 最大角速度（弧度/秒），默認 1.5 rad/s（對應 ActionsCfg）

    Returns:
        shape [num_envs, 1]: 歸一化角速度 ω_z，範圍 [-1, 1]
        正值 = 逆時針旋轉（左轉）
        負值 = 順時針旋轉（右轉）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 獲取機器人本體座標系的角速度
    angular_vel_b = asset.data.root_ang_vel_b  # [num_envs, 3]

    # 🔥 重要：清理輸入數據的 NaN/Inf
    angular_vel_b = torch.nan_to_num(angular_vel_b, nan=0.0, posinf=5.0, neginf=-5.0)

    omega_z = angular_vel_b[:, 2:3]  # 只取 Z 軸（yaw）, 保持 [num_envs, 1] 形狀

    # 🔥 Bug 修復：歸一化到 [-1, 1]
    # 除以最大角速度，讓數值在合理範圍內
    omega_z_normalized = omega_z / max_angular_velocity

    # 安全處理
    omega_z_normalized = torch.nan_to_num(omega_z_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    omega_z_normalized = torch.clamp(omega_z_normalized, -1.0, 1.0)  # 限制在 [-1, 1]

    check_finite("base_angular_velocity_z", omega_z_normalized, raise_on_error=True)

    return omega_z_normalized


def heading_error_to_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """目標朝向誤差觀測

    返回 [sin(θ), cos(θ)]，其中 θ 是車頭與目標方向的夾角。
    這比單獨的 goal_position 更直接地告訴策略「要轉多少」。

    設計理念：
    - sin(θ) 表示「要往哪邊轉」：正值=左轉，負值=右轉
    - cos(θ) 表示「對準程度」：1.0=正對，0.0=側對，-1.0=背對
    - 使用 [sin, cos] 形式避免角度跳變問題（-π 到 +π 的不連續）

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）

    Returns:
        shape [num_envs, 2]: [sin(heading_error), cos(heading_error)]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 統一目標來源：優先使用局部航點（分層架構核心）
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos_w = env._local_goal_world
    else:
        # Fallback: 系統尚未生成局部航點時暫用全域目標
        goal_pos_w = env.command_manager.get_command("goal_command")
    robot_pos_w = asset.data.root_pos_w[:, :2]
    robot_quat_w = asset.data.root_quat_w

    # 目標方向（世界座標）
    goal_dir = goal_pos_w[:, :2] - robot_pos_w
    goal_angle = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])

    # 機器人朝向
    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)

    # 誤差角（歸一化到 [-π, π]）
    heading_error = goal_angle - robot_yaw
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

    # 返回 [sin, cos] 形式
    result = torch.stack([torch.sin(heading_error), torch.cos(heading_error)], dim=1)

    # 安全處理
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    result = torch.clamp(result, -1.0, 1.0)

    check_finite("heading_error_to_goal", result, raise_on_error=True)

    return result


def time_remaining_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """時間剩餘比例觀測
    
    回傳 episode 剩餘時間比例(1 = 剛開始,0 = 即將超時)。
    
    Args:
        env: 環境實例
    
    Returns:
        shape [num_envs, 1]: 剩餘時間比例 [0, 1]
    """
    # 最大步數
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
    
    check_finite("time_remaining_ratio", ratio, raise_on_error=True)
    return ratio


def alive_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活狀態觀測(死亡狀態標記)
    
    這是一個關鍵的狀態特徵,用於:
    1. 幫助 Critic Network 正確估算期望值(Value)
    2. 強制融合「避障」與「抵達目標」的能力
    
    輸出:
        1.0 = 存活(可以繼續獲得獎勵)
        0.0 = 死亡(因碰撞/翻倒等被終止,無法獲得後續獎勵)
    
    注意:
        - 只有「terminated」(真正終止,如碰撞)才算死亡
        - 「time_outs」(超時截斷)不算死亡
    
    Args:
        env: 環境實例
    
    Returns:
        shape [num_envs, 1]: 存活標記 [0, 1]
    """
    # 檢查是否有 reset_terminated
    if hasattr(env, "reset_terminated"):
        died = env.reset_terminated.to(dtype=torch.float32)
        alive = 1.0 - died
    elif hasattr(env, "reset_buf"):
        alive = 1.0 - env.reset_buf.to(dtype=torch.float32)
    else:
        alive = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    
    alive = torch.clamp(alive, 0.0, 1.0).unsqueeze(1)
    check_finite("alive_flag", alive, raise_on_error=True)
    return alive


# ============================================================================
# 障礙物觀測
# ============================================================================

def dynamic_obstacles_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    num_obstacles: int = 10,
    max_obstacles: int | None = None,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """障礙物狀態觀測（Critic 上帝視角，每個障礙物 5 維）

    ═══════════════════════════════════════════════════════════════════════════
                    Critic 觀測空間 - 混合平行環境支援
    ═══════════════════════════════════════════════════════════════════════════

    每個障礙物特徵 [x, y, dir, v, size]:
      - x, y: 障礙物相對於機器人的位置（機器人座標系）
      - dir: 障礙物方向（相對於機器人，弧度）
      - v: 障礙物速度（m/s，取平面速度大小）
      - size: 障礙物尺寸（近似值）

    🔥 混合平行環境支援（論文核心亮點）：
    ────────────────────────────────────────────────────────────────────────────────
    地底遮蔽檢查：
      - 讀取障礙物的 Z 座標
      - 使用 torch.where 進行遮罩處理
      - 如果 Z < 0（地底下），該障礙物的 5 維特徵 [x, y, dir, v, size] 全部設為 0.0
      - 確保 Critic 在 Empty 環境中看到的是乾淨的全零狀態

    Empty 環境（20%）：
      - 所有障礙物 Z = -10.0 → 觀測全為 [0, 0, 0, 0, 0]
      - Critic 學習評估「無障礙物」場景的價值

    Static 環境（50%）：
      - 前 num_obstacles_static 個障礙物 Z = 0.5 → 觀測為真實值
      - 其餘障礙物 Z = -10.0 → 觀測為 [0, 0, 0, 0, 0]

    Dynamic 環境（30%）：
      - 前 num_obstacles_dynamic 個障礙物 Z = 0.5 → 觀測為真實值（含速度）
      - 其餘障礙物 Z = -10.0 → 觀測為 [0, 0, 0, 0, 0]

    架構說明：
      - 觀測維度固定為 max_obstacles × 5（支持權重遷移）
      - 使用 PyTorch Tensor 操作，避免 Python for 循環遍歷環境

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        num_obstacles: 實際障礙物數量
        max_obstacles: 最大障礙物數量（用於固定觀測維度）
        max_distance: 最大觀測距離（米）

    Returns:
        shape [num_envs, max_obstacles * 5]: 固定維度的障礙物觀測
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = env.num_envs

    # 優先使用環境中的動態障礙物數量
    if hasattr(env, "_num_obstacles") and env._num_obstacles is not None:
        num_obstacles = env._num_obstacles

    # 如果未指定 max_obstacles，使用 num_obstacles
    if max_obstacles is None:
        max_obstacles = num_obstacles

    # 機器人位置與姿態
    robot_pos_w = asset.data.root_pos_w[:, :3]
    robot_quat_w = asset.data.root_quat_w
    _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot_quat_w)

    # 準備輸出（固定維度: max_obstacles × 5）
    # 🔥 關鍵：初始值全為 0.0，隱藏障礙物不需要額外處理
    obs = torch.zeros(num_envs, max_obstacles, 5, device=env.device, dtype=torch.float32)

    # 取得障礙物尺寸
    from ..core.state import get_obstacle_sizes

    obstacle_sizes = getattr(env, "_obstacle_sizes", None)
    if obstacle_sizes is None:
        obstacle_sizes = get_obstacle_sizes()

    # 取得障礙物速度緩存
    obstacle_velocities = getattr(env, "_obstacle_velocities", None)
    if obstacle_velocities is None:
        env._obstacle_velocities = torch.zeros(
            num_envs, num_obstacles, 2, device=env.device, dtype=torch.float32
        )
        obstacle_velocities = env._obstacle_velocities

    # ========================================================================
    # 遍歷障礙物，使用 torch.where 處理地底遮蔽
    # ========================================================================
    for i in range(num_obstacles):
        obstacle_name = f"obstacle_{i}"
        # 🔥 修正：InteractiveScene 只支援 scene["name"]，不支援 hasattr/getattr
        if obstacle_name not in env.scene.keys():
            continue

        obstacle = env.scene[obstacle_name]
        obs_pos_w = obstacle.data.root_pos_w[:, :3]

        # 🔥 地底遮蔽檢查：Z < 0 表示障礙物被隱藏
        # 使用 torch.where 確保 Critic 在 Empty 環境中看到全零狀態
        hidden_mask = obs_pos_w[:, 2] < 0.0  # [num_envs]
        visible_mask = ~hidden_mask

        # 相對位置（機器人座標系）
        rel_pos_b, _ = math_utils.subtract_frame_transforms(
            robot_pos_w, robot_quat_w,
            obs_pos_w, torch.zeros_like(robot_quat_w),
        )
        rel_xy = rel_pos_b[:, :2]
        rel_xy = torch.clamp(rel_xy, -max_distance, max_distance)

        # 方向與速度
        obs_quat_w = obstacle.data.root_quat_w if hasattr(obstacle.data, "root_quat_w") else robot_quat_w
        _, _, obs_yaw = math_utils.euler_xyz_from_quat(obs_quat_w)

        if obstacle_velocities is not None and obstacle_velocities.shape[1] > i:
            vel_w = obstacle_velocities[:, i, :]
            speed = torch.linalg.norm(vel_w, dim=1)
            has_velocity = speed > 1e-6
            vel_yaw = torch.atan2(vel_w[:, 1], vel_w[:, 0])
            rel_yaw = torch.where(has_velocity, vel_yaw - robot_yaw, obs_yaw - robot_yaw)
        else:
            rel_yaw = obs_yaw - robot_yaw
            if hasattr(obstacle.data, "root_lin_vel_w"):
                vel_w = obstacle.data.root_lin_vel_w[:, :2]
                speed = torch.linalg.norm(vel_w, dim=1)
            else:
                speed = torch.zeros(num_envs, device=env.device)

        # 將方向角限制到 [-pi, pi]
        rel_yaw = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))

        # 尺寸
        if obstacle_sizes is not None and i < len(obstacle_sizes):
            size_val = float(obstacle_sizes[i])
        else:
            size_val = 0.0
        size = torch.full((num_envs,), size_val, device=env.device, dtype=torch.float32)

        # 🔥 混合平行環境：地底遮蔽處理
        # 使用 torch.where 確保隱藏障礙物的 5 維特徵全部為 0.0
        # 這樣 Critic 在 Empty 環境中會看到乾淨的全零狀態
        obs[:, i, 0] = torch.where(hidden_mask, torch.zeros_like(rel_xy[:, 0]), rel_xy[:, 0])  # x
        obs[:, i, 1] = torch.where(hidden_mask, torch.zeros_like(rel_xy[:, 1]), rel_xy[:, 1])  # y
        obs[:, i, 2] = torch.where(hidden_mask, torch.zeros_like(rel_yaw), rel_yaw)            # dir
        obs[:, i, 3] = torch.where(hidden_mask, torch.zeros_like(speed), speed)                # v
        obs[:, i, 4] = torch.where(hidden_mask, torch.zeros_like(size), size)                  # size

    # Padding: 填充不存在的障礙物（索引 > num_obstacles）
    # 這些位置保持為 0.0（因為 obs 初始化為全零）

    # reshape 成 [num_envs, max_obstacles * 5]
    obs = obs.reshape(num_envs, max_obstacles * 5)

    # 安全處理
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    # 注意：不 clamp size，因為 0.0 是有效的「不存在」標記
    obs[:, :max_obstacles * 4] = torch.clamp(obs[:, :max_obstacles * 4], -max_distance, max_distance)
    check_finite("dynamic_obstacles_state", obs, raise_on_error=True)

    return obs


# ============================================================================
# 診斷觀測
# ============================================================================

def charge_dies_at_birth_probability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.3,
    birth_window_steps: int = 10,
) -> torch.Tensor:
    """觀測：Charge 在重置後一定時間步內撞到障礙物的機率
    
    此觀測用於診斷訓練過程中的問題，監控機器人在重置後的「出生窗口期」內
    是否會與障礙物發生碰撞。這是一個重要的診斷指標，可以幫助識別：
    - 障礙物生成邏輯是否有問題
    - 機器人起始位置是否合理
    - 訓練環境配置是否正確
    
    設計理念：
    - 只在重置後的特定時間窗口內（例如前 10 步）檢測碰撞
    - 如果在這個時間窗口內發生碰撞，則標記為「出生即死」（1.0）
    - 一旦標記為 1.0，在該 episode 的剩餘時間內保持為 1.0
    
    注意：使用 2D 平面距離（忽略高度），與 collision_occurred 保持一致。
    
    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），低於此距離算「碰撞」（使用 2D 平面距離）
        birth_window_steps: 出生窗口期步數（默認 10 步），在此時間內碰撞才算「出生即死」
    
    Returns:
        shape [num_envs]：浮點數張量，值為 0.0 或 1.0
        1.0 = 在重置後的出生窗口期內發生碰撞（機器人「出生即死」）
        0.0 = 安全，沒有在出生窗口期內碰撞
    """
    # 初始化追蹤變量（如果尚未初始化）
    if not hasattr(env, "_dies_at_birth_flags"):
        # 追蹤每個環境是否在出生窗口期內發生碰撞
        # shape: [num_envs]，1.0 = 已標記為「出生即死」，0.0 = 尚未發生
        env._dies_at_birth_flags = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    # 獲取當前 episode 步數
    if hasattr(env, "episode_length_buf"):
        episode_steps = env.episode_length_buf  # [num_envs]
    else:
        # 如果沒有 episode_length_buf，無法追蹤，返回全零
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    # 檢測哪些環境剛被重置（episode_steps == 0，即重置後的第一步）
    # 重置時清除對應環境的標記，開始新的追蹤
    reset_mask = (episode_steps == 0)
    if reset_mask.any():
        env._dies_at_birth_flags[reset_mask] = 0.0
    
    # 檢查哪些環境在出生窗口期內（episode_steps <= birth_window_steps）
    in_birth_window = episode_steps <= birth_window_steps
    
    # 只對在出生窗口期內且尚未標記為「出生即死」的環境進行碰撞檢測
    needs_check = in_birth_window & (env._dies_at_birth_flags == 0.0)
    
    if needs_check.any():
        # 獲取雷達傳感器
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        
        # 獲取感測器位置和射線碰撞點
        sensor_pos = sensor.data.pos_w  # [num_envs, 3]
        hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
        
        # ========================================================================
        # 使用 2D 平面距離（忽略高度 Z），與 collision_occurred 保持一致
        # ========================================================================
        sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2]
        hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2]
        
        # 計算 2D 平面距離
        distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
        # shape: [num_envs, num_rays]
        
        # 處理無效碰撞（射線沒打到任何東西，距離為 inf）
        distances_2d = torch.nan_to_num(
            distances_2d, 
            nan=sensor.cfg.max_distance, 
            posinf=sensor.cfg.max_distance
        )
        
        # 找出每個環境中最近的障礙物距離（2D 平面距離）
        min_distances = torch.min(distances_2d, dim=1)[0]  # [num_envs]
        
        # 判斷是否碰撞：距離 < threshold 表示碰撞
        collision_mask = min_distances < threshold
        
        # 只更新需要檢查的環境：在出生窗口期內且發生碰撞的環境
        new_dies_at_birth = needs_check & collision_mask
        if new_dies_at_birth.any():
            env._dies_at_birth_flags[new_dies_at_birth] = 1.0
    
    # 返回標記（已經是浮點數，值為 0.0 或 1.0）
    probability = env._dies_at_birth_flags.clone()
    
    # 安全處理：確保輸出是有效的浮點數
    probability = torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    probability = torch.clamp(probability, 0.0, 1.0)
    
    return probability