"""競爭式獎勵函數 (Competitive Rewards)

為競賽組訓練提供的獎勵函數：
- competitive_reaching_goal: 只有 group 內第一個到達的 agent 獲得正獎勵
- competitive_loser_penalty: group 中有 winner 但自己不是 winner 的懲罰
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


def competitive_reaching_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
    threshold: float = 1.0,
    body_radius: float = 0.0,
) -> torch.Tensor:
    """競爭式目標到達獎勵 — 只有 group 內第一個到達的 agent 獲得獎勵。

    工作原理：
    1. 計算每個 env 到目標的距離
    2. 判斷是否到達（距離 < threshold）
    3. 檢查所在 group 是否已有 winner
    4. 只有「首次到達且無 winner」的 agent 獲得 1.0

    Args:
        env: 環境實例（需要有 competition 屬性）
        asset_cfg: 機器人資產配置
        threshold: 到達判定距離（米）
        body_radius: 機器人本體半徑，用於調整距離計算

    Returns:
        [num_envs] — 1.0 for winner, 0.0 for others
    """
    if not hasattr(env, "competition"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]

    # 統一目標來源：優先使用局部航點
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        goal_pos = env._local_goal_world[:, :2]
    else:
        goal_pos = env.command_manager.get_command("goal_command")[:, :2]

    dist = torch.norm(robot_pos - goal_pos, dim=1)
    if body_radius > 0.0:
        dist = torch.clamp(dist - body_radius, min=0.0)

    # 到達判定
    reached = dist < threshold
    n_reached = reached.sum().item()

    # 檢查 group 是否已有 winner
    already_won = env.competition.group_has_winner[env.competition.env_to_group]
    is_first = reached & (~already_won)

    # 標記 winner
    first_ids = is_first.nonzero(as_tuple=False).squeeze(-1)
    n_first = len(first_ids)
    env.competition.mark_winners(first_ids)

    if n_first > 0:
        group_ids = env.competition.env_to_group[first_ids].unique()
        min_dist_val = dist[first_ids].min().item()
        logger.info(
            f"[競爭獎勵] 新增 {n_first} 個勝者"
            f"（到達={n_reached}，已有勝者={n_reached - n_first}）| "
            f"勝者 env={first_ids.tolist()[:8]}{'...' if n_first > 8 else ''} | "
            f"所屬組={group_ids.tolist()[:8]} | 最近距離={min_dist_val:.3f}m"
        )

    return is_first.float()


def competitive_loser_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """失敗者懲罰 — group 中有 winner 但自己不是 winner。

    當同組有 agent 已到達目標時，其餘 agent 會收到此懲罰。
    配合 group_goal_reached 終止條件，此懲罰通常只生效 1 步。

    Args:
        env: 環境實例（需要有 competition 屬性）

    Returns:
        [num_envs] — 1.0 for losers in groups with winner, 0.0 for others
    """
    if not hasattr(env, "competition"):
        return torch.zeros(env.num_envs, device=env.device)

    has_winner = env.competition.group_has_winner[env.competition.env_to_group]
    n_losers = has_winner.sum().item()

    if n_losers > 0:
        logger.debug(
            f"[失敗懲罰] {int(n_losers)} 個 env 被懲罰（同組已有勝者）"
        )

    return has_winner.float()
