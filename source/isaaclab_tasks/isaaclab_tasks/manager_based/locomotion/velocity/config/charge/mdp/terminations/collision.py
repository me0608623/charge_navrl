"""碰撞相關終止條件

包含與碰撞相關的終止條件：
- 牆壁碰撞終止

注意：
- collision_occurred 和 collision_contact_occurred 位於 mdp/rewards/safety_rewards.py
- 因為它們同時作為獎勵函數和終止條件使用
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

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
