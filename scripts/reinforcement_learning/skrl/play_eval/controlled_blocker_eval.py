"""Deterministic, solvable blocker-with-wall scene for promotion Gate 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ControlledBlockerSpec:
    goal_x: float = 6.0
    blocker_x: float = 1.8
    corridor_clear_width: float = 4.0
    wall_width: float = 1.0
    outer_wall_length: float = 3.0
    pressure_wall_length: float = 4.0
    robot_radius: float = 0.35
    blocker_radius: float = 0.35

    @property
    def outer_wall_y(self) -> float:
        return 0.5 * (self.corridor_clear_width + self.wall_width)

    @property
    def side_clearance(self) -> float:
        return 0.5 * self.corridor_clear_width - self.blocker_radius

    @property
    def pressure_wall_y(self) -> float:
        # Its inner face overlaps the blocker by 5 cm, closing exactly one side.
        return self.blocker_radius + 0.5 * self.wall_width - 0.05

    @property
    def pressure_wall_x(self) -> float:
        # The wall starts at the blocker centre and extends toward the goal.
        return self.blocker_x + 0.5 * self.pressure_wall_length


def mirrored_geometry(env_ids: torch.Tensor, spec: ControlledBlockerSpec):
    """Return local wall geometry; even envs close left, odd envs close right."""
    closed_sign = torch.where(env_ids % 2 == 0, 1.0, -1.0)
    n = env_ids.numel()
    centers = torch.zeros(n, 3, 2, device=env_ids.device)
    sizes = torch.zeros_like(centers)

    centers[:, 0, 0] = spec.blocker_x
    centers[:, 0, 1] = spec.outer_wall_y
    centers[:, 1, 0] = spec.blocker_x
    centers[:, 1, 1] = -spec.outer_wall_y
    centers[:, 2, 0] = spec.pressure_wall_x
    centers[:, 2, 1] = closed_sign * spec.pressure_wall_y
    sizes[:, 0:2, 0] = spec.outer_wall_length
    sizes[:, 0:2, 1] = spec.wall_width
    sizes[:, 2, 0] = spec.pressure_wall_length
    sizes[:, 2, 1] = spec.wall_width
    return closed_sign, centers, sizes


def configure_controlled_blocker_env(env_cfg, scene_final: dict, cli_args, spec=None) -> None:
    """Remove random scene factors and reserve three wall slots plus one blocker."""
    spec = spec or ControlledBlockerSpec()
    scene_final.update({
        "num_goals": 1,
        "num_static": 1,
        "num_dynamic": 0,
        "walls_min": 3,
        "walls_max": 3,
        "goal_dist_min": spec.goal_x,
        "goal_dist_max": spec.goal_x,
    })
    cli_args.no_goal_movement = True

    goal_cfg = env_cfg.commands.goal_command
    goal_cfg.num_goals = 1
    goal_cfg.num_obstacles = 1
    goal_cfg.ranges.distance = (spec.goal_x, spec.goal_x)
    goal_cfg.ranges.angle = (0.0, 0.0)

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
        wall_event.params.update({"min_walls": 3, "max_walls": 3})

    reset_event = getattr(env_cfg.events, "reset_base", None)
    if reset_event is not None:
        pose_range = reset_event.params.setdefault("pose_range", {})
        pose_range.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        velocity_range = reset_event.params.setdefault("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)

    # Slots 1 and 7 have identical 3 m meshes; slot 0 is the 4 m pressure wall.
    for slot, size in ((1, (spec.outer_wall_length, spec.wall_width, 3.0)),
                       (7, (spec.outer_wall_length, spec.wall_width, 3.0)),
                       (0, (spec.pressure_wall_length, spec.wall_width, 3.0))):
        wall_cfg = getattr(env_cfg.scene, f"wall_internal_{slot}", None)
        if wall_cfg is not None:
            wall_cfg.spawn.size = size


class ControlledBlockerController:
    """Own fixed robot, goal, blocker, and mirrored wall placement across resets."""

    WALL_SLOTS = (1, 7, 0)

    def __init__(self, raw_env, scheduler=None, spec=None) -> None:
        self.env = raw_env
        self.scheduler = scheduler
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.spec = spec or ControlledBlockerSpec()

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
        _, centers, sizes = mirrored_geometry(env_ids, self.spec)
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
        origins = self.env.scene.env_origins[env_ids]
        pose = torch.zeros(len(env_ids), 7, device=self.device)
        pose[:, 0] = origins[:, 0] + self.spec.blocker_x
        pose[:, 1] = origins[:, 1]
        pose[:, 2] = 0.9
        pose[:, 3] = 1.0
        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        obstacle = self.env.scene["obstacle_0"]
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(velocity, env_ids=env_ids)

        if self.scheduler is not None:
            self.scheduler.positions[env_ids, 0, 0] = self.spec.blocker_x
            self.scheduler.positions[env_ids, 0, 1] = 0.0
            self.scheduler.velocities[env_ids, 0] = 0.0

