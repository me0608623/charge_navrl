"""Tensor helpers for obstacle-aware goal sampling."""

from __future__ import annotations

import torch


def record_unsolvable_scene(env, count: int) -> int:
    """Record rejected scenes on the environment and return the cumulative count."""
    total = int(getattr(env, "_unsolvable_scene_count_total", 0)) + int(count)
    env._unsolvable_scene_count_total = total
    return total


def regenerate_unsolvable_scenes(env, env_ids: torch.Tensor) -> None:
    """Reset and rerun scene-randomization events only for rejected environments."""
    env.scene.reset(env_ids)
    env_step_count = env._sim_step_counter // env.cfg.decimation
    env.event_manager.apply(
        mode="reset",
        env_ids=env_ids,
        global_env_step_count=env_step_count,
    )


def visible_obstacle_xy(
    root_positions_w: torch.Tensor,
    env_origins_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local XY positions and a mask for physically visible obstacles."""
    finite = torch.isfinite(root_positions_w).all(dim=-1)
    visible = finite & (root_positions_w[:, 2] > 0.0)
    safe_xy = torch.nan_to_num(
        root_positions_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
    )
    return safe_xy - env_origins_xy, visible


def masked_obstacle_distances(
    candidate_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_visible: torch.Tensor,
) -> torch.Tensor:
    """Compute distances while treating hidden and invalid slots as infinity."""
    distances = torch.linalg.vector_norm(candidate_xy.unsqueeze(1) - obstacle_xy, dim=-1)
    return distances.masked_fill(~obstacle_visible, torch.inf)
