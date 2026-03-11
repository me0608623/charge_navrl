"""碰撞相關終止條件

包含與碰撞相關的終止條件：
- 牆壁碰撞終止
- LiDAR 碰撞終止（雷達檢測）
- PhysX 物理碰撞終止（接觸感測器）

注意：
- collision_occurred 和 collision_contact_occurred 原本位於 mdp/rewards/safety_rewards.py
- 這裡提供純粹的終止條件版本
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    判斷邏輯：
    - 計算機器人相對於環境原點的位置
    - 計算有效邊界（邊界減去機器人半徑）
    - 任一軸（X 或 Y）超出有效邊界即判定為撞牆
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


def lidar_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """🔥 LiDAR 碰撞終止條件

    當 LiDAR 檢測到障礙物距離 < threshold 時終止 episode。

    注意：使用 2D 平面距離（忽略高度），因為：
    1. 雷達安裝在機器人上方 0.5 米處
    2. 使用 3D 距離會導致距離偏大（考慮了高度差）
    3. 2D 距離更能反映機器人底盤與障礙物的實際距離

    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用雷達）
        threshold: 碰撞閾值（米），默認 0.5m

    Returns:
        shape [num_envs]：布林張量
        True = 碰撞，終止
        False = 安全，繼續
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 獲取感測器位置和射線碰撞點
    sensor_pos = sensor.data.pos_w  # [num_envs, 3] 感測器位置（世界座標系）
    hit_points = sensor.data.ray_hits_w  # [num_envs, num_rays, 3] 射線碰撞點

    # 使用 2D 平面距離（忽略高度 Z）
    sensor_pos_2d = sensor_pos[:, :2]  # [num_envs, 2]
    hit_points_2d = hit_points[:, :, :2]  # [num_envs, num_rays, 2]

    # 計算 2D 平面距離
    distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
    # shape: [num_envs, num_rays]

    # 處理無效碰撞（射線沒打到任何東西）
    distances_2d = torch.nan_to_num(
        distances_2d,
        nan=sensor.cfg.max_distance,
        posinf=sensor.cfg.max_distance
    )

    # 找出每個環境中最近的障礙物距離
    min_distance = torch.min(distances_2d, dim=1)[0]  # [num_envs]

    # 判斷是否碰撞：最近距離 < 閾值
    return min_distance < threshold


def physx_contact_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """🔥 PhysX 物理碰撞終止條件

    當接觸感測器檢測到機器人與障礙物發生物理接觸時終止 episode。
    這使用物理引擎的真實碰撞檢測，比 LiDAR 距離判定更準確。

    Args:
        env: 環境實例
        sensor_cfg: 傳感器配置（引用接觸感測器）
        force_threshold: 接觸力閾值（牛頓），默認 0.1 N

    Returns:
        shape [num_envs]：布林張量
        True = 檢測到碰撞，終止
        False = 無碰撞，繼續

    優勢：
    - 使用真實的碰撞框（物理引擎的碰撞檢測）
    - 比 LiDAR 距離判定更準確
    - 可以檢測機器人任何部件與障礙物的碰撞

    注意：
    - 接觸感測器必須在場景配置中啟用
    - 必須設置 filter_prim_paths_expr 來過濾障礙物
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 獲取接觸力數據
    # force_matrix_w: [num_envs, num_bodies, num_filtered_objects, 3]
    contact_forces = sensor.data.force_matrix_w

    # 如果沒有設置過濾器，無法檢測
    if contact_forces is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # 計算每個環境的總接觸力大小
    total_force = torch.sum(contact_forces, dim=(1, 2))  # [num_envs, 3]
    force_magnitude = torch.norm(total_force, dim=1)  # [num_envs]

    # 判斷是否碰撞
    return force_magnitude > force_threshold

