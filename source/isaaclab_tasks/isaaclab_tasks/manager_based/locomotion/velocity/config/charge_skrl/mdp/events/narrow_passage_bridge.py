"""Reset injector for the SA5 narrow-passage warm-start bridge."""

from __future__ import annotations

import math

import torch

from .narrow_passage_bridge_geometry import schedule_at, validate_constructed_scene


_ASSET_NAMES = ("narrow_bridge_wall_0", "narrow_bridge_wall_1")
_HIDDEN_Z = -10.0


def configure_narrow_passage_assets(
    env_cfg,
    *,
    fraction: float,
    schedule_steps: int,
    room_half_extent: float,
    segment_length: float = 9.0,
    final_stress_ratio: float = 0.25,
    fixed_width_range: tuple[float, float] | None = None,
    fixed_yaw_limit_deg: float | None = None,
    exact_width: float | None = None,
    exact_width_ratio: float = 0.0,
) -> None:
    """Add two bridge-only wall assets and configure the reset event.

    The assets are not part of the eight random wall slots. They remain hidden
    outside selected narrow-replay episodes.
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.55, 0.78), metallic=0.1)
    for index, name in enumerate(_ASSET_NAMES):
        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall_NarrowBridge_{index}",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, float(segment_length), 3.0),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    visual_material=visual,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _HIDDEN_Z)),
            ),
        )

    event = getattr(env_cfg.events, "narrow_passage_bridge", None)
    if event is None:
        raise RuntimeError("environment config has no narrow_passage_bridge reset event")
    event.params.update(
        {
            "fraction": float(fraction),
            "schedule_steps": int(schedule_steps),
            "room_half_extent": float(room_half_extent),
            "segment_length": float(segment_length),
            "final_stress_ratio": float(final_stress_ratio),
            "fixed_width_range": fixed_width_range,
            "fixed_yaw_limit_deg": fixed_yaw_limit_deg,
            "exact_width": exact_width,
            "exact_width_ratio": float(exact_width_ratio),
        }
    )
    if not 0.0 <= float(exact_width_ratio) <= 1.0:
        raise ValueError(
            f"exact narrow-passage width ratio must be in [0, 1]: {exact_width_ratio}"
        )
    if (exact_width is None) != (float(exact_width_ratio) == 0.0):
        raise ValueError(
            "exact_width and a positive exact_width_ratio must be configured together"
        )
    if exact_width is not None:
        if fixed_width_range is None:
            raise ValueError("exact-width replay requires a fixed width range")
        width_min, width_max = map(float, fixed_width_range)
        if not width_min <= float(exact_width) <= width_max:
            raise ValueError(
                f"exact width {exact_width} is outside fixed range {fixed_width_range}"
            )
    schedule_mode = (
        f"fixed widths={fixed_width_range} yaw=+/-{fixed_yaw_limit_deg}deg stress=0"
        if fixed_width_range is not None
        else f"ramp final_stress_ratio={final_stress_ratio:.3f}"
    )
    print(
        "[NARROW-BRIDGE-CONFIG] "
        f"fraction={fraction:.3f} "
        f"schedule_steps={schedule_steps} room_half={room_half_extent:.2f} "
        f"segment_length={segment_length:.2f} schedule={schedule_mode} "
        f"exact_width={exact_width} exact_ratio={float(exact_width_ratio):.3f} "
        "reward_unchanged=True",
        flush=True,
    )


def _as_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_wall_state(env) -> None:
    if hasattr(env, "_narrow_bridge_wall_centers"):
        return
    env._narrow_bridge_wall_centers = torch.zeros(env.num_envs, 2, 2, device=env.device)
    env._narrow_bridge_wall_sizes = torch.zeros(env.num_envs, 2, 2, device=env.device)
    env._narrow_bridge_wall_mask = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    env._narrow_bridge_active = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._narrow_bridge_goal_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._narrow_bridge_reset_count = 0
    env._narrow_bridge_injected_count = 0
    env._narrow_bridge_unsolvable_count = 0


def _hide_bridge_walls(env, env_ids: torch.Tensor) -> None:
    env._narrow_bridge_wall_mask[env_ids] = False
    env._narrow_bridge_active[env_ids] = False
    origins = env.scene.env_origins[env_ids]
    for name in _ASSET_NAMES:
        wall = env.scene[name]
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        wall.write_root_pose_to_sim(pose, env_ids=env_ids)


def _hide_original_geometry(env, env_ids: torch.Tensor) -> None:
    origins = env.scene.env_origins[env_ids]
    if hasattr(env, "_maze_wall_mask"):
        env._maze_wall_mask[env_ids] = False
    for slot in range(getattr(env, "_maze_wall_mask", torch.empty(0, 0)).shape[1]):
        name = f"wall_internal_{slot}"
        if name not in env.scene.keys():
            continue
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)

    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    if scheduler is not None:
        from .behavior_scheduler import BEHAVIOR_INACTIVE

        scheduler.behavior_type[env_ids] = BEHAVIOR_INACTIVE
        scheduler.positions[env_ids] = 0.0
        scheduler.velocities[env_ids] = 0.0
        scheduler.phase_timer[env_ids] = 0
        scheduler.pc_velocity[env_ids] = 0.0
        scheduler.pc_done[env_ids] = True
        obstacle_slots = scheduler.max_obstacles
    else:
        obstacle_slots = 100

    # Write only selected bridge envs. BehaviorScheduler._write_positions_to_sim
    # rewrites every obstacle in every env, which is unnecessarily expensive for
    # the small reset subset used by this injector.
    for slot in range(obstacle_slots):
        name = f"obstacle_{slot}"
        if name not in env.scene.keys():
            continue
        obstacle = env.scene[name]
        pose = obstacle.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=env.device), env_ids=env_ids
        )


def setup_narrow_passage_bridge(
    env,
    env_ids,
    *,
    fraction: float = 0.0,
    schedule_steps: int = 19200,
    room_half_extent: float = 7.5,
    boundary_wall_width: float = 1.0,
    segment_length: float = 9.0,
    gap_center_limit: float = 1.0,
    barrier_x_limit: float = 0.5,
    start_goal_distance: float = 3.0,
    final_stress_ratio: float = 0.25,
    fixed_width_range: tuple[float, float] | None = None,
    fixed_yaw_limit_deg: float | None = None,
    exact_width: float | None = None,
    exact_width_ratio: float = 0.0,
    wall_z: float = 1.5,
) -> None:
    """Replace a fraction of reset episodes with guaranteed-solvable wall gaps."""
    if fraction <= 0.0:
        return

    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    _ensure_wall_state(env)
    _hide_bridge_walls(env, ids)
    env._narrow_bridge_reset_count += int(ids.numel())

    # Previous-stage and deployment-corridor events run before this event. Keep
    # all replay classes disjoint and compensate the Bernoulli probability so
    # `fraction` remains the absolute narrow share of all resets.
    eligible = ids
    conditional_fraction = float(fraction)
    reserved_fraction = 0.0
    if hasattr(env, "_previous_stage_replay_active"):
        eligible = eligible[
            ~env._previous_stage_replay_active[eligible]
        ]
        reserved_fraction += float(
            getattr(env, "_previous_stage_replay_fraction", 0.0)
        )
    if hasattr(env, "_long_corridor_active"):
        eligible = eligible[~env._long_corridor_active[eligible]]
        reserved_fraction += float(
            getattr(env, "_long_corridor_fraction", 0.0)
        )
    conditional_fraction = min(
        float(fraction) / max(1.0 - reserved_fraction, 1e-6),
        1.0,
    )
    if eligible.numel() == 0:
        return
    selected = eligible[
        torch.rand(eligible.numel(), device=env.device) < conditional_fraction
    ]
    if selected.numel() == 0:
        return

    common_steps = int(getattr(env, "common_step_counter", 0))
    progress = common_steps / max(int(schedule_steps), 1)
    schedule = schedule_at(
        progress,
        final_stress_ratio,
        fixed_width_range=fixed_width_range,
        fixed_yaw_limit_deg=fixed_yaw_limit_deg,
    )
    count = selected.numel()

    gap_width = torch.empty(count, device=env.device).uniform_(
        schedule.width_min, schedule.width_max
    )
    exact_mask = torch.zeros(count, dtype=torch.bool, device=env.device)
    if exact_width is not None and exact_width_ratio > 0.0:
        exact_mask = (
            torch.rand(count, device=env.device) < float(exact_width_ratio)
        )
        gap_width[exact_mask] = float(exact_width)
    if schedule.stress_ratio > 0.0:
        stress = torch.rand(count, device=env.device) < schedule.stress_ratio
        if stress.any():
            gap_width[stress] = torch.empty(
                int(stress.sum().item()), device=env.device
            ).uniform_(1.0, 1.2)
    else:
        stress = torch.zeros(count, dtype=torch.bool, device=env.device)

    gap_center = torch.empty(count, device=env.device).uniform_(
        -gap_center_limit, gap_center_limit
    )
    barrier_x = torch.empty(count, device=env.device).uniform_(
        -barrier_x_limit, barrier_x_limit
    )
    direction = torch.where(
        torch.rand(count, device=env.device) < 0.5,
        torch.full((count,), -1.0, device=env.device),
        torch.ones(count, device=env.device),
    )
    start_x = barrier_x - direction * float(start_goal_distance)
    goal_x = barrier_x + direction * float(start_goal_distance)

    valid = []
    reasons = []
    for i in range(count):
        ok, reason = validate_constructed_scene(
            gap_width=float(gap_width[i]),
            gap_center=float(gap_center[i]),
            barrier_x=float(barrier_x[i]),
            start_x=float(start_x[i]),
            goal_x=float(goal_x[i]),
            room_half_extent=room_half_extent,
            boundary_wall_width=boundary_wall_width,
            segment_length=segment_length,
        )
        valid.append(ok)
        reasons.append(reason)
    valid_t = torch.tensor(valid, dtype=torch.bool, device=env.device)
    if not bool(valid_t.all()):
        env._narrow_bridge_unsolvable_count += int((~valid_t).sum().item())
        bad = valid_t.logical_not().nonzero(as_tuple=False).flatten()[0].item()
        raise RuntimeError(
            f"narrow bridge generated an unsolvable scene: env={int(selected[bad])} "
            f"reason={reasons[bad]}"
        )

    _hide_original_geometry(env, selected)
    origins = env.scene.env_origins[selected]

    # Long fixed wall segments start exactly at each gap edge and overhang beyond
    # the outer walls. Inside the arena this forms a complete, non-bypassable barrier.
    gap_lo = gap_center - 0.5 * gap_width
    gap_hi = gap_center + 0.5 * gap_width
    centers_y = torch.stack(
        [gap_hi + 0.5 * segment_length, gap_lo - 0.5 * segment_length], dim=1
    )
    for index, name in enumerate(_ASSET_NAMES):
        pose = torch.zeros(count, 7, device=env.device)
        pose[:, 0] = origins[:, 0] + barrier_x
        pose[:, 1] = origins[:, 1] + centers_y[:, index]
        pose[:, 2] = wall_z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=selected)
        env._narrow_bridge_wall_centers[selected, index, 0] = barrier_x
        env._narrow_bridge_wall_centers[selected, index, 1] = centers_y[:, index]
        env._narrow_bridge_wall_sizes[selected, index, 0] = 1.0
        env._narrow_bridge_wall_sizes[selected, index, 1] = segment_length
        env._narrow_bridge_wall_mask[selected, index] = True

    # Random left/right offset is bounded by the straight OBB clearance.
    center_clearance = 0.5 * gap_width - 0.40
    offset_limit = torch.minimum(
        torch.full_like(center_clearance, 0.30),
        (0.65 * center_clearance).clamp(min=0.0),
    )
    start_y = gap_center + (2.0 * torch.rand(count, device=env.device) - 1.0) * offset_limit
    yaw_error = torch.empty(count, device=env.device).uniform_(
        -math.radians(schedule.yaw_limit_deg), math.radians(schedule.yaw_limit_deg)
    )
    base_yaw = torch.where(direction > 0.0, torch.zeros_like(direction), torch.full_like(direction, math.pi))
    yaw = base_yaw + yaw_error

    robot = env.scene["robot"]
    robot_pose = robot.data.default_root_state[selected, :7].clone()
    robot_pose[:, 0] = origins[:, 0] + start_x
    robot_pose[:, 1] = origins[:, 1] + start_y
    robot_pose[:, 3:7] = 0.0
    robot_pose[:, 3] = torch.cos(0.5 * yaw)
    robot_pose[:, 6] = torch.sin(0.5 * yaw)
    robot.write_root_pose_to_sim(robot_pose, env_ids=selected)
    robot.write_root_velocity_to_sim(
        torch.zeros(count, 6, device=env.device), env_ids=selected
    )

    goal = torch.zeros(count, 3, device=env.device)
    goal[:, 0] = origins[:, 0] + goal_x
    goal[:, 1] = origins[:, 1] + gap_center
    goal_term = env.command_manager.get_term("goal_command")
    goal_term.goal_pos_w[selected] = goal
    if hasattr(goal_term, "all_goals_pos_w"):
        goal_term.all_goals_pos_w[selected] = goal[:, None, :]
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        env._local_goal_world[selected, :2] = goal[:, :2]
    env._narrow_bridge_goal_w[selected] = goal
    env._narrow_bridge_active[selected] = True

    env._narrow_bridge_injected_count += int(count)
    env._narrow_bridge_last_progress = schedule.progress
    env._narrow_bridge_last_width_min = float(gap_width.min())
    env._narrow_bridge_last_width_max = float(gap_width.max())
    env._narrow_bridge_last_yaw_limit_deg = schedule.yaw_limit_deg
    env._narrow_bridge_last_stress_ratio = schedule.stress_ratio

    if not getattr(env, "_narrow_bridge_logged", False):
        env._narrow_bridge_logged = True
        print(
            "[NARROW-BRIDGE] injector FIRED: "
            f"{count}/{ids.numel()} envs progress={schedule.progress:.3f} "
            f"gap=[{float(gap_width.min()):.3f},{float(gap_width.max()):.3f}]m "
            f"exact={int(exact_mask.sum())}/{count} "
            f"yaw=+/-{schedule.yaw_limit_deg:.1f}deg stress={int(stress.sum())}/{count} "
            f"constructive_solvability=100% mirror_left_right=True",
            flush=True,
        )


def maintain_narrow_passage_goal(env, env_ids=None) -> None:
    """Keep bridge goals at the opposite side after the SA5 moving-goal event."""
    if not hasattr(env, "_narrow_bridge_active"):
        return
    active = env._narrow_bridge_active
    if env_ids is not None:
        ids = _as_env_ids(env, env_ids)
        selected = ids[active[ids]]
    else:
        selected = active.nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return

    goal = env._narrow_bridge_goal_w[selected]
    goal_term = env.command_manager.get_term("goal_command")
    goal_term.goal_pos_w[selected] = goal
    if hasattr(goal_term, "all_goals_pos_w"):
        goal_term.all_goals_pos_w[selected] = goal[:, None, :]
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        env._local_goal_world[selected, :2] = goal[:, :2]
