"""競爭式終止條件 (Competitive Terminations)

為競賽組訓練提供的終止條件：
- group_goal_reached: 同組任一 agent 到達目標 -> 全組終止
"""

from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger("CompetitionGroup")


def group_goal_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
    threshold: float = 1.0,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """群組終止：同組任一 agent 到達目標 -> 全組終止。

    工作原理：
    1. 計算每個 env 到目標的距離
    2. 找出所有到達的 env
    3. 將到達信號擴展到同組所有成員

    Args:
        env: 環境實例（需要有 competition 屬性）
        asset_cfg: 機器人資產配置
        threshold: 到達判定距離（米）
        body_radius: 機器人本體半徑

    Returns:
        [num_envs] — bool tensor, True = 該 env 所在組有人到達
    """
    if not hasattr(env, "competition"):
        # Fallback: 無競爭模式時，退化為普通 goal_reached
        asset: Articulation = env.scene[asset_cfg.name]
        robot_pos = asset.data.root_pos_w[:, :2]
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]
        dist = torch.norm(robot_pos - goal_pos, dim=1)
        if body_radius > 0.0:
            dist = torch.clamp(dist - body_radius, min=0.0)
        return dist < threshold

    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]

    # 統一目標來源
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world[:, :2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    dist = torch.norm(robot_pos - goal_pos, dim=1)
    if body_radius > 0.0:
        dist = torch.clamp(dist - body_radius, min=0.0)

    reached_any = dist < threshold
    reached_ids = reached_any.nonzero(as_tuple=False).squeeze(-1)

    if len(reached_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 擴展到全組
    all_terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    group_members = env.competition.get_group_members(reached_ids)
    all_terminate[group_members] = True

    n_reached = len(reached_ids)
    n_terminated = all_terminate.sum().item()
    group_ids = env.competition.env_to_group[reached_ids].unique()
    logger.info(
        f"[群組終止] {n_reached} 個 env 到達目標 → {int(n_terminated)} 個 env 終止"
        f"（{len(group_ids)} 組: {group_ids.tolist()[:8]}{'...' if len(group_ids) > 8 else ''}）"
    )

    return all_terminate
