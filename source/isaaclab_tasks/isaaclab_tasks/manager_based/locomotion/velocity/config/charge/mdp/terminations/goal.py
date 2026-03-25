"""目標相關終止條件

包含與目標達成相關的終止條件：
- 到達目標終止
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        body_radius: 機器人本體半徑（米），用於調整距離計算
    
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
    # 距離 < threshold 米 → True（終止）
