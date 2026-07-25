"""Goal ownership helpers for deterministic reset-injected scenes."""

from __future__ import annotations

from collections.abc import Sequence

import torch


FIXED_SCENE_GOAL_FIELDS = (
    ("_long_corridor_active", "_long_corridor_goal_w"),
    ("_narrow_bridge_active", "_narrow_bridge_goal_w"),
)


def normalize_env_ids(env, env_ids: Sequence[int] | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def fixed_scene_goal_mask(env, env_ids: Sequence[int] | None) -> torch.Tensor:
    ids = normalize_env_ids(env, env_ids)
    fixed = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)
    for active_name, goal_name in FIXED_SCENE_GOAL_FIELDS:
        active = getattr(env, active_name, None)
        fixed_goal = getattr(env, goal_name, None)
        if active is not None and fixed_goal is not None:
            fixed |= active[ids]
    return fixed


def restore_fixed_scene_goals(
    env, goal_term, env_ids: Sequence[int] | None
) -> torch.Tensor:
    ids = normalize_env_ids(env, env_ids)
    claimed = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)

    for active_name, goal_name in FIXED_SCENE_GOAL_FIELDS:
        active = getattr(env, active_name, None)
        fixed_goal = getattr(env, goal_name, None)
        if active is None or fixed_goal is None:
            continue

        source_mask = active[ids]
        if bool((claimed & source_mask).any()):
            raise RuntimeError(
                "multiple fixed-scene injectors claimed the same environment"
            )
        selected = ids[source_mask]
        if selected.numel() == 0:
            continue

        goal = fixed_goal[selected]
        goal_term.goal_pos_w[selected] = goal
        if hasattr(goal_term, "all_goals_pos_w"):
            goal_term.all_goals_pos_w[selected] = goal[:, None, :]
        local_goal = getattr(env, "_local_goal_world", None)
        if local_goal is not None:
            local_goal[selected, :2] = goal[:, :2]
        claimed |= source_mask

    return claimed
