"""機器人狀態相關終止條件

包含與機器人物理狀態相關的終止條件：
- 機器人翻倒終止
- 機器人飛起終止（異常檢測）
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    
    判斷邏輯：
    - 正常站立：z_axis = [0, 0, 1] → z_axis[:, 2] = 1.0 > 0.5 ✅
    - 側倒 90°：z_axis = [1, 0, 0] → z_axis[:, 2] = 0.0 < 0.5 ❌（翻倒）
    - 倒立：z_axis = [0, 0, -1] → z_axis[:, 2] = -1.0 < 0.5 ❌（翻倒）
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
    
    注意：
    - 正常情況下，Charge 底盤高度約 0.1-0.2 米
    - 高度 > 1.0 米視為異常
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取高度
    height = asset.data.root_pos_w[:, 2]
    # shape: [num_envs]：Z 座標（高度）
    
    # 判斷是否飛起來
    is_flying = height > 1.0
    # 高度 > 1 米 → True（異常，終止）
    
    return is_flying
