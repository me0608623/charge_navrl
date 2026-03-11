"""目標達成終止條件

VLP16 訓練使用:
    goal_reached: 機器人到達目標航點時終止 episode (成功)

參數:
    threshold: float = 0.5  — GOAL_REACH_THRESHOLD [m]
    body_radius: float = 0.5 — ROBOT_BODY_RADIUS [m]

判定邏輯:
    dist = ||robot_pos_2d - goal_pos_2d|| - body_radius
    → dist <= threshold → 終止 (time_out=False → value 不 bootstrap)

目標來源:
    優先使用 env._local_goal_world (層級導航的局部航點)
    Fallback: env.command_manager.get_command("goal_command")
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
    """終止條件：到達局部航點 (Waypoint)

    分層架構設計：
    - RL 的任務是到達局部航點 (_local_goal_world)，而非全域目標
    - 到達局部航點 → episode 成功終止
    - Fallback: 系統尚未初始化局部航點時，暫用 goal_command

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 距離閾值（米）
        body_radius: 機器人本體半徑（米），用於調整距離計算

    Returns:
        shape [num_envs]：布林張量
        True = 到達局部航點，終止
        False = 未到達，繼續
    """
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = asset.data.root_pos_w[:, :2]

    # 統一目標來源：優先使用局部航點（分層架構核心）
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos_w = env._local_goal_world[:, :2]
    else:
        # Fallback: 系統尚未生成局部航點時暫用全域目標
        goal_pos_w = env.command_manager.get_command("goal_command")[:, :2]

    distance = torch.norm(goal_pos_w - robot_pos_w, dim=1)
    if body_radius > 0.0:
        distance = torch.clamp(distance - body_radius, min=0.0)
    return distance < threshold
