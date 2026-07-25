"""Per-environment replay of the accepted SA5 general-navigation scene.

This injector is intentionally narrower than a full curriculum rollback. It
replays the scene variables that define the fixed SA5 Gate2 distribution:
10 static plus 3 dynamic obstacles and 2-3 internal walls near 4 m long.
The current stage keeps ownership of reward, discount, episode horizon, goal
count, and the shared outer arena.
"""

from __future__ import annotations

import torch


MAX_PREVIOUS_STAGE_REPLAY_FRACTION = 0.60


def configure_previous_stage_replay(
    env_cfg,
    *,
    fraction: float,
    static_obstacles: int = 10,
    dynamic_obstacles: int = 3,
    min_walls: int = 2,
    max_walls: int = 3,
    wall_length: float = 4.0,
    obstacle_boundary: float = 5.5,
) -> None:
    """Configure the reset event without adding any scene assets."""
    if not 0.0 <= float(fraction) <= MAX_PREVIOUS_STAGE_REPLAY_FRACTION:
        raise ValueError(
            "previous-stage replay fraction must be in "
            f"[0, {MAX_PREVIOUS_STAGE_REPLAY_FRACTION:.2f}], got {fraction}"
        )
    if static_obstacles < 0 or dynamic_obstacles < 0:
        raise ValueError("previous-stage obstacle counts must be non-negative")
    if static_obstacles + dynamic_obstacles <= 0:
        raise ValueError("previous-stage replay requires at least one obstacle")
    if not 0 <= min_walls <= max_walls <= 8:
        raise ValueError(
            f"invalid previous-stage wall range: {min_walls}..{max_walls}"
        )
    if wall_length <= 0.0 or obstacle_boundary <= 0.0:
        raise ValueError("wall length and obstacle boundary must be positive")

    event = getattr(env_cfg.events, "previous_stage_replay", None)
    if event is None:
        raise RuntimeError("environment config has no previous_stage_replay event")
    event.params.update(
        {
            "fraction": float(fraction),
            "static_obstacles": int(static_obstacles),
            "dynamic_obstacles": int(dynamic_obstacles),
            "min_walls": int(min_walls),
            "max_walls": int(max_walls),
            "wall_length": float(wall_length),
            "obstacle_boundary": float(obstacle_boundary),
        }
    )
    print(
        "[PREVIOUS-STAGE-CONFIG] "
        f"fraction={fraction:.3f} scene=SA5-general "
        f"obstacles={static_obstacles}S+{dynamic_obstacles}D "
        f"walls={min_walls}-{max_walls}@{wall_length:.1f}m "
        f"obstacle_boundary=+/-{obstacle_boundary:.1f}m "
        "reward_horizon_outer_arena=current_stage",
        flush=True,
    )


def deployment_behavior_mix(
    static_obstacles: int, dynamic_obstacles: int
) -> dict[str, float]:
    """Match the ordered behavior mix used by e2e_final20_v1."""
    total = int(static_obstacles) + int(dynamic_obstacles)
    if total <= 0:
        raise ValueError("behavior mix requires at least one obstacle")
    if dynamic_obstacles <= 0:
        return {"static": 1.0}
    static_fraction = float(static_obstacles) / float(total)
    dynamic_fraction = float(dynamic_obstacles) / float(total)
    return {
        "static": static_fraction,
        "horizontal_crossing": dynamic_fraction * 0.30,
        "path_crossing": dynamic_fraction * 0.25,
        "patrol": dynamic_fraction * 0.20,
        "random_walk": dynamic_fraction * 0.15,
        "head_on": dynamic_fraction * 0.10,
    }


def _as_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_state(env) -> None:
    if hasattr(env, "_previous_stage_replay_active"):
        return
    env._previous_stage_replay_active = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._previous_stage_replay_pending = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._previous_stage_replay_reset_count = 0
    env._previous_stage_replay_injected_count = 0
    env._previous_stage_replay_installed_count = 0


def _clear_scheduler_state(scheduler, env_ids: torch.Tensor) -> None:
    from .behavior_scheduler import BEHAVIOR_INACTIVE

    scheduler.behavior_type[env_ids] = BEHAVIOR_INACTIVE
    scheduler.positions[env_ids] = 0.0
    scheduler.velocities[env_ids] = 0.0
    scheduler.phase_timer[env_ids] = 0
    scheduler.patrol_pause_remaining[env_ids] = 0


def _install_obstacles(
    env,
    selected: torch.Tensor,
    static_obstacles: int,
    dynamic_obstacles: int,
) -> bool:
    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    if scheduler is None:
        env._previous_stage_replay_pending[selected] = True
        return False

    total = int(static_obstacles) + int(dynamic_obstacles)
    if total > scheduler.max_obstacles:
        raise RuntimeError(
            "previous-stage replay requires "
            f"{total} obstacle slots but scheduler has {scheduler.max_obstacles}"
        )

    mix = deployment_behavior_mix(static_obstacles, dynamic_obstacles)
    counts = scheduler._allocate_counts(mix, total)
    _clear_scheduler_state(scheduler, selected)

    ordered_names = sorted(
        counts.keys(), key=lambda name: 0 if name == "static" else 1
    )
    slot_index = 0
    for behavior_name in ordered_names:
        count = int(counts[behavior_name])
        behavior_type = scheduler._name_to_id(behavior_name)
        for _ in range(count):
            if slot_index >= total:
                break
            scheduler.behavior_type[selected, slot_index] = behavior_type
            scheduler._spawn_single(
                selected, slot_index, behavior_type
            )
            slot_index += 1

    if slot_index != total:
        raise RuntimeError(
            f"previous-stage replay allocated {slot_index}/{total} obstacle slots"
        )
    from .behavior_scheduler import BEHAVIOR_INACTIVE, BEHAVIOR_STATIC

    selected_types = scheduler.behavior_type[selected]
    active_counts = (selected_types != BEHAVIOR_INACTIVE).sum(dim=1)
    static_counts = (selected_types == BEHAVIOR_STATIC).sum(dim=1)
    if not bool((active_counts == total).all()):
        raise RuntimeError(
            "previous-stage replay active obstacle count does not match "
            f"{total}"
        )
    if not bool((static_counts == int(static_obstacles)).all()):
        raise RuntimeError(
            "previous-stage replay static obstacle count does not match "
            f"{static_obstacles}"
        )
    scheduler._enforce_spawn_constraints(selected, env)
    scheduler._write_positions_to_sim(env)
    env._previous_stage_replay_pending[selected] = False
    env._previous_stage_replay_installed_count += int(selected.numel())
    return True


def setup_previous_stage_replay(
    env,
    env_ids,
    *,
    fraction: float = 0.0,
    static_obstacles: int = 10,
    dynamic_obstacles: int = 3,
    min_walls: int = 2,
    max_walls: int = 3,
    wall_length: float = 4.0,
    obstacle_boundary: float = 5.5,
) -> None:
    """Replace a fraction of native resets with the accepted SA5 scene mix."""
    if fraction <= 0.0:
        return
    if not 0.0 < float(fraction) <= MAX_PREVIOUS_STAGE_REPLAY_FRACTION:
        raise ValueError(
            "previous-stage replay fraction must be in "
            f"(0, {MAX_PREVIOUS_STAGE_REPLAY_FRACTION:.2f}], got {fraction}"
        )

    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    _ensure_state(env)
    env._previous_stage_replay_fraction = float(fraction)
    env._previous_stage_replay_obstacle_counts = (
        int(static_obstacles),
        int(dynamic_obstacles),
    )
    env._previous_stage_replay_active[ids] = False
    env._previous_stage_replay_pending[ids] = False
    env._previous_stage_replay_reset_count += int(ids.numel())

    selected = ids[
        torch.rand(ids.numel(), device=env.device) < float(fraction)
    ]
    if selected.numel() == 0:
        return

    from .walls import randomize_walls

    randomize_walls(
        env,
        selected,
        min_walls=int(min_walls),
        max_walls=int(max_walls),
        boundary=float(obstacle_boundary),
        target_wall_length=float(wall_length),
    )
    env._previous_stage_replay_active[selected] = True
    env._previous_stage_replay_pending[selected] = True
    installed = _install_obstacles(
        env,
        selected,
        int(static_obstacles),
        int(dynamic_obstacles),
    )
    env._previous_stage_replay_injected_count += int(selected.numel())

    if not getattr(env, "_previous_stage_replay_logged", False):
        env._previous_stage_replay_logged = True
        print(
            "[PREVIOUS-STAGE] injector FIRED: "
            f"{selected.numel()}/{ids.numel()} envs "
            f"scene=SA5-general obstacles={static_obstacles}S+"
            f"{dynamic_obstacles}D walls={min_walls}-{max_walls}"
            f"@{wall_length:.1f}m "
            f"pending_scheduler={0 if installed else selected.numel()} "
            "narrow_gap_pairs=OFF near_goal_override=OFF",
            flush=True,
        )


def maintain_previous_stage_replay(env, env_ids=None) -> None:
    """Finish initial-reset obstacle installation after scheduler creation."""
    if not hasattr(env, "_previous_stage_replay_pending"):
        return
    ids = _as_env_ids(env, env_ids)
    pending = ids[env._previous_stage_replay_pending[ids]]
    if pending.numel() == 0:
        return
    static_obstacles, dynamic_obstacles = (
        env._previous_stage_replay_obstacle_counts
    )
    if not _install_obstacles(
        env,
        pending,
        int(static_obstacles),
        int(dynamic_obstacles),
    ):
        return
    print(
        "[PREVIOUS-STAGE] completed pending scheduler install: "
        f"{pending.numel()} envs",
        flush=True,
    )


__all__ = [
    "configure_previous_stage_replay",
    "deployment_behavior_mix",
    "maintain_previous_stage_replay",
    "setup_previous_stage_replay",
]
