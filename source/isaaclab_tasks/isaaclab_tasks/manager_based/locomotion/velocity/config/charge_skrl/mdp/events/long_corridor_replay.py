"""Reset injector for the 4 m x 10 m deployment-corridor replay scene."""

from __future__ import annotations

import math

import torch

from .long_corridor_replay_geometry import (
    LongCorridorSpec,
    layout_is_constructively_solvable,
    sample_obstacle_layout,
    validate_obstacle_counts,
    validate_spec,
    wall_geometry,
)


_ASSET_NAMES = ("long_corridor_wall_0", "long_corridor_wall_1")
_HIDDEN_Z = -10.0


def configure_long_corridor_assets(
    env_cfg,
    *,
    fraction: float,
    free_width: float = 4.0,
    length: float = 10.0,
    static_obstacles: int = 4,
    dynamic_obstacles: int = 2,
    dynamic_speed_range: tuple[float, float] = (0.30, 0.60),
) -> None:
    """Add dedicated corridor walls and configure the reset event."""
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    spec = LongCorridorSpec(free_width=float(free_width), length=float(length))
    validate_spec(spec)
    validate_obstacle_counts(static_obstacles, dynamic_obstacles)
    speed_min, speed_max = map(float, dynamic_speed_range)
    if not (0.0 < speed_min <= speed_max):
        raise ValueError("dynamic corridor speed range must be positive and ordered")

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True, disable_gravity=True
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.32, 0.38, 0.42), metallic=0.1
    )
    for index, name in enumerate(_ASSET_NAMES):
        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall_LongCorridor_{index}",
                spawn=sim_utils.CuboidCfg(
                    size=(spec.wall_thickness, spec.length, spec.wall_height),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    visual_material=visual,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, _HIDDEN_Z)
                ),
            ),
        )

    event = getattr(env_cfg.events, "long_corridor_replay", None)
    if event is None:
        raise RuntimeError("environment config has no long_corridor_replay reset event")
    event.params.update(
        {
            "fraction": float(fraction),
            "free_width": spec.free_width,
            "length": spec.length,
            "static_obstacles": int(static_obstacles),
            "dynamic_obstacles": int(dynamic_obstacles),
            "dynamic_speed_min": speed_min,
            "dynamic_speed_max": speed_max,
        }
    )
    print(
        "[LONG-CORRIDOR-CONFIG] "
        f"fraction={fraction:.3f} free_width={spec.free_width:.2f}m "
        f"length={spec.length:.2f}m obstacles={static_obstacles}S+"
        f"{dynamic_obstacles}D speed=[{speed_min:.2f},{speed_max:.2f}]m/s "
        "reward_unchanged=True",
        flush=True,
    )


def _as_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_state(env) -> None:
    if hasattr(env, "_long_corridor_wall_centers"):
        return
    env._long_corridor_wall_centers = torch.zeros(
        env.num_envs, 2, 2, device=env.device
    )
    env._long_corridor_wall_sizes = torch.zeros(
        env.num_envs, 2, 2, device=env.device
    )
    env._long_corridor_wall_mask = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    env._long_corridor_active = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._long_corridor_pending_obstacles = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._long_corridor_goal_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._long_corridor_dynamic_start = torch.zeros(
        env.num_envs, 2, 2, device=env.device
    )
    env._long_corridor_reset_count = 0
    env._long_corridor_injected_count = 0
    env._long_corridor_unsolvable_count = 0


def _hide_corridor_walls(env, env_ids: torch.Tensor) -> None:
    env._long_corridor_wall_mask[env_ids] = False
    env._long_corridor_active[env_ids] = False
    env._long_corridor_pending_obstacles[env_ids] = False
    origins = env.scene.env_origins[env_ids]
    for name in _ASSET_NAMES:
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)


def _hide_original_geometry(env, env_ids: torch.Tensor) -> None:
    """Remove random walls and obstacles from selected replay envs."""
    origins = env.scene.env_origins[env_ids]
    if hasattr(env, "_maze_wall_mask"):
        env._maze_wall_mask[env_ids] = False
    wall_slots = getattr(env, "_maze_wall_mask", torch.empty(0, 0)).shape[1]
    for slot in range(wall_slots):
        name = f"wall_internal_{slot}"
        if name not in env.scene.keys():
            continue
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)

    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    obstacle_slots = scheduler.max_obstacles if scheduler is not None else 100
    if scheduler is not None:
        from .behavior_scheduler import BEHAVIOR_INACTIVE

        scheduler.behavior_type[env_ids] = BEHAVIOR_INACTIVE
        scheduler.positions[env_ids] = 0.0
        scheduler.velocities[env_ids] = 0.0
        scheduler.phase_timer[env_ids] = 0
        scheduler.patrol_pause_remaining[env_ids] = 0

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


def _install_obstacles(
    env,
    selected: torch.Tensor,
    spec: LongCorridorSpec,
    speed_min: float,
    speed_max: float,
    static_obstacles: int,
    dynamic_obstacles: int,
) -> bool:
    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    if scheduler is None:
        env._long_corridor_pending_obstacles[selected] = True
        return False
    validate_obstacle_counts(static_obstacles, dynamic_obstacles)
    if scheduler.max_obstacles < 4 + dynamic_obstacles:
        raise RuntimeError(
            "long corridor requires enough BehaviorScheduler obstacle slots"
        )

    from .behavior_scheduler import (
        BEHAVIOR_INACTIVE,
        BEHAVIOR_PATROL,
        BEHAVIOR_STATIC,
    )

    count = selected.numel()
    static, dynamic, waypoints = sample_obstacle_layout(count, spec, env.device)
    valid = layout_is_constructively_solvable(static, dynamic, waypoints, spec)
    if not bool(valid.all()):
        bad_count = int((~valid).sum().item())
        env._long_corridor_unsolvable_count += bad_count
        raise RuntimeError(
            f"long corridor generated {bad_count} non-constructive layouts"
        )

    scheduler.behavior_type[selected] = BEHAVIOR_INACTIVE
    scheduler.positions[selected] = 0.0
    scheduler.velocities[selected] = 0.0
    scheduler.phase_timer[selected] = 0
    scheduler.patrol_pause_remaining[selected] = 0
    env._long_corridor_dynamic_start[selected] = 0.0

    if static_obstacles > 0:
        static_slots = slice(0, static_obstacles)
        scheduler.behavior_type[selected, static_slots] = BEHAVIOR_STATIC
        scheduler.positions[selected, static_slots] = static[
            :, :static_obstacles
        ]

    if dynamic_obstacles > 0:
        dynamic_slots = slice(4, 4 + dynamic_obstacles)
        active_dynamic = dynamic[:, :dynamic_obstacles]
        active_waypoints = waypoints[:, :dynamic_obstacles]
        scheduler.behavior_type[selected, dynamic_slots] = BEHAVIOR_PATROL
        scheduler.positions[selected, dynamic_slots] = active_dynamic
        scheduler.patrol_waypoints[selected, dynamic_slots, :2] = (
            active_waypoints
        )
        scheduler.patrol_waypoints[selected, dynamic_slots, 2:] = 0.0
        scheduler.patrol_num_waypoints[selected, dynamic_slots] = 2
        direction_right = (
            torch.rand(count, dynamic_obstacles, device=env.device) < 0.5
        )
        scheduler.patrol_wp_index[selected, dynamic_slots] = (
            direction_right.long()
        )
        speeds = torch.empty(
            count, dynamic_obstacles, device=env.device
        ).uniform_(float(speed_min), float(speed_max))
        scheduler.patrol_speed[selected, dynamic_slots] = speeds
        velocity_sign = torch.where(
            direction_right,
            torch.ones_like(speeds),
            -torch.ones_like(speeds),
        )
        scheduler.velocities[selected, dynamic_slots, 0] = (
            velocity_sign * speeds
        )
        scheduler.velocities[selected, dynamic_slots, 1] = 0.0
        env._long_corridor_dynamic_start[selected, :dynamic_obstacles] = (
            active_dynamic
        )
    env._long_corridor_pending_obstacles[selected] = False
    scheduler._write_positions_to_sim(env)
    return True


def _write_goal(
    env, selected: torch.Tensor, *, update_markers: bool = False
) -> None:
    goal = env._long_corridor_goal_w[selected]
    goal_term = env.command_manager.get_term("goal_command")
    goal_term.goal_pos_w[selected] = goal
    if hasattr(goal_term, "all_goals_pos_w"):
        goal_term.all_goals_pos_w[selected] = goal[:, None, :]
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        env._local_goal_world[selected, :2] = goal[:, :2]
    if update_markers and hasattr(goal_term, "_update_goal_markers"):
        goal_term._update_goal_markers()


def setup_long_corridor_replay(
    env,
    env_ids,
    *,
    fraction: float = 0.0,
    free_width: float = 4.0,
    length: float = 10.0,
    static_obstacles: int = 4,
    dynamic_obstacles: int = 2,
    dynamic_speed_min: float = 0.30,
    dynamic_speed_max: float = 0.60,
    wall_z: float = 1.5,
) -> None:
    """Replace a fraction of resets with the frozen deployment corridor."""
    if fraction <= 0.0:
        return
    validate_obstacle_counts(static_obstacles, dynamic_obstacles)

    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    spec = LongCorridorSpec(free_width=float(free_width), length=float(length))
    validate_spec(spec)
    _ensure_state(env)
    env._long_corridor_fraction = float(fraction)
    env._long_corridor_spec = spec
    env._long_corridor_speed_range = (
        float(dynamic_speed_min),
        float(dynamic_speed_max),
    )
    env._long_corridor_obstacle_counts = (
        int(static_obstacles),
        int(dynamic_obstacles),
    )
    _hide_corridor_walls(env, ids)
    env._long_corridor_reset_count += int(ids.numel())

    # Previous-stage replay runs immediately before this event. Exclude those
    # envs and compensate the Bernoulli probability so `fraction` remains the
    # absolute long-corridor share of all resets.
    eligible = ids
    conditional_fraction = float(fraction)
    if hasattr(env, "_previous_stage_replay_active"):
        eligible = ids[~env._previous_stage_replay_active[ids]]
        previous_fraction = float(
            getattr(env, "_previous_stage_replay_fraction", 0.0)
        )
        conditional_fraction = min(
            float(fraction) / max(1.0 - previous_fraction, 1e-6),
            1.0,
        )
    if eligible.numel() == 0:
        return
    selected = eligible[
        torch.rand(eligible.numel(), device=env.device)
        < conditional_fraction
    ]
    if selected.numel() == 0:
        return

    _hide_original_geometry(env, selected)
    origins = env.scene.env_origins[selected]
    centers, sizes = wall_geometry(selected.numel(), spec, env.device)
    for index, name in enumerate(_ASSET_NAMES):
        pose = torch.zeros(selected.numel(), 7, device=env.device)
        pose[:, 0] = origins[:, 0] + centers[:, index, 0]
        pose[:, 1] = origins[:, 1] + centers[:, index, 1]
        pose[:, 2] = wall_z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=selected)
    env._long_corridor_wall_centers[selected] = centers
    env._long_corridor_wall_sizes[selected] = sizes
    env._long_corridor_wall_mask[selected] = True

    robot = env.scene["robot"]
    robot_pose = robot.data.default_root_state[selected, :7].clone()
    robot_pose[:, 0] = origins[:, 0]
    robot_pose[:, 1] = origins[:, 1] + spec.robot_start_y
    robot_pose[:, 3:7] = 0.0
    robot_pose[:, 3] = math.cos(math.pi / 4.0)
    robot_pose[:, 6] = math.sin(math.pi / 4.0)
    robot.write_root_pose_to_sim(robot_pose, env_ids=selected)
    robot.write_root_velocity_to_sim(
        torch.zeros(selected.numel(), 6, device=env.device), env_ids=selected
    )

    goal = origins.clone()
    goal[:, 1] = origins[:, 1] + spec.goal_y
    goal[:, 2] = 0.0
    env._long_corridor_goal_w[selected] = goal
    env._long_corridor_active[selected] = True
    _write_goal(env, selected, update_markers=True)
    installed = _install_obstacles(
        env,
        selected,
        spec,
        dynamic_speed_min,
        dynamic_speed_max,
        static_obstacles,
        dynamic_obstacles,
    )

    env._long_corridor_injected_count += int(selected.numel())
    if not getattr(env, "_long_corridor_logged", False):
        env._long_corridor_logged = True
        pending = 0 if installed else selected.numel()
        print(
            "[LONG-CORRIDOR] injector FIRED: "
            f"{selected.numel()}/{ids.numel()} envs "
            f"free_width={spec.free_width:.2f}m length={spec.length:.2f}m "
            f"walls_x=+/-{spec.wall_center_offset:.2f}m "
            f"obstacles={static_obstacles}S+{dynamic_obstacles}D "
            f"speed=[{dynamic_speed_min:.2f},"
            f"{dynamic_speed_max:.2f}]m/s pending_scheduler={pending} "
            "constructive_solvability=100%",
            flush=True,
        )


def maintain_long_corridor_goal(env, env_ids=None) -> None:
    """Pin the goal and finish delayed obstacle installation if necessary."""
    if not hasattr(env, "_long_corridor_active"):
        return
    active = env._long_corridor_active
    if env_ids is None:
        selected = active.nonzero(as_tuple=False).flatten()
    else:
        ids = _as_env_ids(env, env_ids)
        selected = ids[active[ids]]
    if selected.numel() == 0:
        return

    pending = selected[env._long_corridor_pending_obstacles[selected]]
    if pending.numel() > 0:
        speed_min, speed_max = env._long_corridor_speed_range
        _install_obstacles(
            env,
            pending,
            env._long_corridor_spec,
            speed_min,
            speed_max,
            *env._long_corridor_obstacle_counts,
        )
    _write_goal(env, selected)
