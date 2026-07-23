"""Deterministic, solvable blocker-with-corridor scene for promotion Gate 3.

Two full-length outer walls (slot 1 = +y, slot 7 = −y) create a 4.0 m-wide symmetric
corridor that spans the whole start→goal path, so the robot cannot escape around the
wall ends and must pass the blocker inside the corridor. The blocker resamples a safe
2D position every episode; dynamic blockers patrol in a random 2D direction and reflect
at bounds that keep the robot start and goal clear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ControlledBlockerSpec:
    # arena_size=12 places the east boundary wall at x=+6 with a 1 m body.
    # Keep the goal centre inside its inner face (x=5.5) with robot clearance.
    goal_x: float = 5.0
    sampler_bootstrap_distance_min: float = 1.5
    sampler_bootstrap_distance_max: float = 3.0
    blocker_x_min: float = 1.2
    blocker_x_max: float = 3.8
    blocker_y_max: float = 1.0   # uniform sample in [-blocker_y_max, +blocker_y_max]
    dynamic_ratio: float = 0.5   # fraction of envs where blocker moves (0 = all static)
    blocker_speed: float = 0.3
    corridor_clear_width: float = 4.0
    wall_width: float = 1.0
    corridor_length: float = 12.0   # wall x-length = deployment room size, so the robot
                                    # cannot escape around the wall ends (must pass in-corridor)
    robot_radius: float = 0.35
    blocker_radius: float = 0.35
    arena_half_extent: float = 6.0
    boundary_wall_width: float = 1.0
    safety_buffer: float = 0.10

    @property
    def corridor_center_x(self) -> float:
        # Walls span the whole start→goal path, centred at its midpoint.
        return 0.5 * self.goal_x

    @property
    def outer_wall_y(self) -> float:
        return 0.5 * (self.corridor_clear_width + self.wall_width)

    @property
    def side_clearance(self) -> float:
        # worst-case: blocker pushed to blocker_y_max
        return 0.5 * self.corridor_clear_width - self.blocker_radius - self.blocker_y_max

    @property
    def endpoint_clearance(self) -> float:
        """Worst edge-to-edge gap from the blocker to robot start or goal."""
        center_gap = min(self.blocker_x_min, self.goal_x - self.blocker_x_max)
        return center_gap - self.blocker_radius - self.robot_radius

    @property
    def goal_wall_clearance(self) -> float:
        """Edge clearance from a robot centred on the goal to the east wall."""
        east_inner_face = self.arena_half_extent - 0.5 * self.boundary_wall_width
        return east_inner_face - self.goal_x - self.robot_radius


def corridor_geometry(n: int, spec: ControlledBlockerSpec, device):
    """Return local wall geometry for the symmetric two-wall corridor."""
    centers = torch.zeros(n, 2, 2, device=device)
    sizes = torch.zeros_like(centers)

    # slot 1: outer wall at +y, spanning the whole corridor (start→goal)
    centers[:, 0, 0] = spec.corridor_center_x
    centers[:, 0, 1] = spec.outer_wall_y
    # slot 7: outer wall at −y
    centers[:, 1, 0] = spec.corridor_center_x
    centers[:, 1, 1] = -spec.outer_wall_y

    sizes[:, 0:2, 0] = spec.corridor_length
    sizes[:, 0:2, 1] = spec.wall_width
    return centers, sizes


def dynamic_env_mask(env_ids: torch.Tensor, num_envs: int, ratio: float) -> torch.Tensor:
    """Allocate an exact, stable fraction of global env IDs to dynamic blockers."""
    cutoff = int(round(ratio * num_envs))
    return env_ids < cutoff


def spec_from_cli(cli_args) -> ControlledBlockerSpec:
    return ControlledBlockerSpec(
        blocker_x_min=float(cli_args.controlled_blocker_x_min),
        blocker_x_max=float(cli_args.controlled_blocker_x_max),
        blocker_y_max=float(cli_args.controlled_blocker_y_max),
        dynamic_ratio=float(cli_args.controlled_blocker_dynamic_ratio),
        blocker_speed=float(cli_args.controlled_blocker_speed),
    )


def configure_controlled_blocker_env(env_cfg, scene_final: dict, cli_args, spec=None) -> None:
    """Remove random scene factors and reserve two corridor walls plus one blocker."""
    spec = spec or ControlledBlockerSpec()
    scene_final.update({
        "num_goals": 1,
        "num_static": 1,
        "num_dynamic": 0,
        "walls_min": 2,
        "walls_max": 2,
        "goal_dist_min": spec.goal_x,
        "goal_dist_max": spec.goal_x,
    })
    cli_args.no_goal_movement = True

    goal_cfg = env_cfg.commands.goal_command
    goal_cfg.num_goals = 1
    goal_cfg.num_obstacles = 1
    # CommandManager resets before ControlledBlockerController exists. Use a
    # temporary in-bounds goal for that bootstrap reset; the controller replaces
    # it with the real x=goal_x target immediately after env.reset().
    goal_cfg.ranges.distance = (
        spec.sampler_bootstrap_distance_min,
        spec.sampler_bootstrap_distance_max,
    )
    goal_cfg.ranges.angle = (-math.pi, math.pi)

    # The eval controller owns motion. Keep the environment obstacle static so a
    # BehaviorScheduler cannot apply a second, conflicting motion update.
    for event_name in ("randomize_obstacles", "randomize_obstacles_startup"):
        event = getattr(env_cfg.events, event_name, None)
        if event is not None:
            event.params.update({
                "empty_ratio": 0.0,
                "static_ratio": 1.0,
                "dynamic_ratio": 0.0,
                "num_obstacles_static": 1,
                "num_obstacles_dynamic": 0,
            })

    wall_event = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_event is not None:
        # Command reset runs before the controller. Keep that bootstrap scene
        # free of random walls; _place_walls installs the exact corridor later.
        wall_event.params.update({"min_walls": 0, "max_walls": 0})

    reset_event = getattr(env_cfg.events, "reset_base", None)
    if reset_event is not None:
        pose_range = reset_event.params.setdefault("pose_range", {})
        pose_range.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        velocity_range = reset_event.params.setdefault("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)

    for slot in (1, 7):
        wall_cfg = getattr(env_cfg.scene, f"wall_internal_{slot}", None)
        if wall_cfg is not None:
            wall_cfg.spawn.size = (spec.corridor_length, spec.wall_width, 3.0)


class ControlledBlockerController:
    """Own fixed robot, goal, blocker, and symmetric corridor wall placement across resets."""

    WALL_SLOTS = (1, 7)

    def __init__(self, raw_env, scheduler=None, spec=None) -> None:
        self.env = raw_env
        self.scheduler = scheduler
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.spec = spec or ControlledBlockerSpec()
        if not 0.0 <= self.spec.dynamic_ratio <= 1.0:
            raise ValueError("dynamic_ratio must be within [0, 1]")
        if self.spec.blocker_y_max < 0.0:
            raise ValueError("blocker_y_max must be non-negative")
        if not 0.0 < self.spec.blocker_x_min < self.spec.blocker_x_max < self.spec.goal_x:
            raise ValueError("blocker x bounds must satisfy 0 < min < max < goal_x")
        if self.spec.side_clearance <= self.spec.robot_radius:
            raise ValueError("blocker_y_max leaves no solvable side passage")
        if self.spec.endpoint_clearance <= 0.0:
            raise ValueError("blocker x bounds overlap the robot start or goal")
        if self.spec.goal_wall_clearance < self.spec.safety_buffer:
            raise ValueError("goal lies inside the east boundary-wall collision margin")
        if self.spec.blocker_speed < 0.0:
            raise ValueError("blocker_speed must be non-negative")
        self.blocker_x = torch.zeros(self.num_envs, device=self.device)
        self.blocker_y = torch.zeros(self.num_envs, device=self.device)
        self.blocker_vx = torch.zeros(self.num_envs, device=self.device)
        self.blocker_vy = torch.zeros(self.num_envs, device=self.device)
        self._motion_logged = False

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self._place_robot_goal(env_ids)
        self._place_walls(env_ids)
        self._place_blocker(env_ids)

    def advance(self) -> None:
        """Advance the controlled 2D patrol and reflect at guaranteed-safe bounds."""
        moving = (self.blocker_vx != 0.0) | (self.blocker_vy != 0.0)
        if not moving.any():
            return
        previous_x = self.blocker_x.clone()
        previous_y = self.blocker_y.clone()
        next_x = self.blocker_x + self.blocker_vx * self.env.step_dt
        next_y = self.blocker_y + self.blocker_vy * self.env.step_dt
        reflect_x = (next_x > self.spec.blocker_x_max) | (next_x < self.spec.blocker_x_min)
        reflect_y = (next_y > self.spec.blocker_y_max) | (next_y < -self.spec.blocker_y_max)
        self.blocker_vx[reflect_x] *= -1.0
        self.blocker_vy[reflect_y] *= -1.0
        self.blocker_x = next_x.clamp(self.spec.blocker_x_min, self.spec.blocker_x_max)
        self.blocker_y = next_y.clamp(-self.spec.blocker_y_max, self.spec.blocker_y_max)
        self._write_blocker(torch.arange(self.num_envs, device=self.device))
        if not self._motion_logged:
            self._motion_logged = True
            delta_x = (self.blocker_x - previous_x).abs()
            delta_y = (self.blocker_y - previous_y).abs()
            print(
                f"[GATE3] controller 2D patrol active: dynamic={int(moving.sum())}/{self.num_envs}, "
                f"speed={self.spec.blocker_speed:.2f}m/s, "
                f"first_step_max_dx={float(delta_x.max()):.3f}m, "
                f"first_step_max_dy={float(delta_y.max()):.3f}m",
                flush=True,
            )

    def _place_robot_goal(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        robot = self.env.scene["robot"]
        pose = robot.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 3:7] = 0.0
        pose[:, 3] = 1.0
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)

        goal = torch.zeros(len(env_ids), 3, device=self.device)
        goal[:, 0] = origins[:, 0] + self.spec.goal_x
        goal[:, 1] = origins[:, 1]
        goal_term = self.env.command_manager.get_term("goal_command")
        if hasattr(goal_term, "goal_pos_w"):
            goal_term.goal_pos_w[env_ids] = goal
        if hasattr(goal_term, "all_goals_pos_w"):
            goal_term.all_goals_pos_w[env_ids] = goal[:, None, :]
        if hasattr(self.env, "_local_goal_world") and self.env._local_goal_world is not None:
            self.env._local_goal_world[env_ids, :2] = goal[:, :2]
        if hasattr(goal_term, "_update_goal_markers"):
            goal_term._update_goal_markers()

    def _place_walls(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        centers, sizes = corridor_geometry(len(env_ids), self.spec, self.device)
        active = set(self.WALL_SLOTS)
        hidden_z = -10.0

        for slot in range(8):
            try:
                wall = self.env.scene[f"wall_internal_{slot}"]
            except KeyError:
                continue
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            pose[:, :2] = origins[:, :2]
            pose[:, 2] = hidden_z
            pose[:, 3] = 1.0
            if slot in active:
                geometry_idx = self.WALL_SLOTS.index(slot)
                pose[:, :2] += centers[:, geometry_idx]
                pose[:, 2] = 1.5
            wall.write_root_pose_to_sim(pose, env_ids=env_ids)

        if hasattr(self.env, "_maze_wall_mask"):
            self.env._maze_wall_mask[env_ids] = False
            for geometry_idx, slot in enumerate(self.WALL_SLOTS):
                self.env._maze_wall_mask[env_ids, slot] = True
                self.env._maze_wall_centers[env_ids, slot] = centers[:, geometry_idx]
                self.env._maze_wall_sizes[env_ids, slot] = sizes[:, geometry_idx]

    def _place_blocker(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        rand_x = torch.rand(n, device=self.device) * (
            self.spec.blocker_x_max - self.spec.blocker_x_min
        ) + self.spec.blocker_x_min
        rand_y = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spec.blocker_y_max

        self.blocker_x[env_ids] = rand_x
        self.blocker_y[env_ids] = rand_y
        # Exact deterministic allocation: for ratio=0.5, 32 of 64 envs are
        # dynamic. Membership is stable; 2D heading is resampled every episode.
        is_dynamic = dynamic_env_mask(env_ids, self.num_envs, self.spec.dynamic_ratio)
        heading = torch.rand(n, device=self.device) * (2.0 * torch.pi)
        speed = torch.where(
            is_dynamic,
            torch.full((n,), self.spec.blocker_speed, device=self.device),
            torch.zeros(n, device=self.device),
        )
        self.blocker_vx[env_ids] = speed * torch.cos(heading)
        self.blocker_vy[env_ids] = speed * torch.sin(heading)
        self._write_blocker(env_ids)

    def _write_blocker(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0] = origins[:, 0] + self.blocker_x[env_ids]
        pose[:, 1] = origins[:, 1] + self.blocker_y[env_ids]
        pose[:, 2] = 0.9
        pose[:, 3] = 1.0
        velocity = torch.zeros(n, 6, device=self.device)
        velocity[:, 0] = self.blocker_vx[env_ids]
        velocity[:, 1] = self.blocker_vy[env_ids]
        obstacle = self.env.scene["obstacle_0"]
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(velocity, env_ids=env_ids)
