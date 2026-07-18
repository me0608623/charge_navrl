"""Deterministic, solvable blocker-with-corridor scene for promotion Gate 3.

Two outer walls (slot 1 = +y, slot 7 = −y) create a 4.0 m symmetric corridor.
The blocker sits at x=1.8 m on the centre line; both sides offer 1.65 m clearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ControlledBlockerSpec:
    goal_x: float = 6.0
    blocker_x: float = 1.8
    blocker_y_max: float = 1.0   # uniform sample in [-blocker_y_max, +blocker_y_max]
    dynamic_ratio: float = 0.5   # fraction of envs where blocker moves (0 = all static)
    corridor_clear_width: float = 4.0
    wall_width: float = 1.0
    outer_wall_length: float = 3.0
    robot_radius: float = 0.35
    blocker_radius: float = 0.35

    @property
    def outer_wall_y(self) -> float:
        return 0.5 * (self.corridor_clear_width + self.wall_width)

    @property
    def side_clearance(self) -> float:
        # worst-case: blocker pushed to blocker_y_max
        return 0.5 * self.corridor_clear_width - self.blocker_radius - self.blocker_y_max


def corridor_geometry(n: int, spec: ControlledBlockerSpec, device):
    """Return local wall geometry for the symmetric two-wall corridor."""
    centers = torch.zeros(n, 2, 2, device=device)
    sizes = torch.zeros_like(centers)

    # slot 1: outer wall at +y
    centers[:, 0, 0] = spec.blocker_x
    centers[:, 0, 1] = spec.outer_wall_y
    # slot 7: outer wall at −y
    centers[:, 1, 0] = spec.blocker_x
    centers[:, 1, 1] = -spec.outer_wall_y

    sizes[:, 0:2, 0] = spec.outer_wall_length
    sizes[:, 0:2, 1] = spec.wall_width
    return centers, sizes


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
    goal_cfg.ranges.distance = (spec.goal_x, spec.goal_x)
    goal_cfg.ranges.angle = (0.0, 0.0)

    # static_ratio + dynamic_ratio must sum to 1.0; we control exact mixing per-env at reset
    for event_name in ("randomize_obstacles", "randomize_obstacles_startup"):
        event = getattr(env_cfg.events, event_name, None)
        if event is not None:
            n_dyn = 1 if spec.dynamic_ratio > 0.0 else 0
            event.params.update({
                "empty_ratio": 0.0,
                "static_ratio": 1.0 - spec.dynamic_ratio,
                "dynamic_ratio": spec.dynamic_ratio,
                "num_obstacles_static": 1,
                "num_obstacles_dynamic": n_dyn,
            })

    wall_event = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_event is not None:
        wall_event.params.update({"min_walls": 2, "max_walls": 2})

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
            wall_cfg.spawn.size = (spec.outer_wall_length, spec.wall_width, 3.0)


class ControlledBlockerController:
    """Own fixed robot, goal, blocker, and symmetric corridor wall placement across resets."""

    WALL_SLOTS = (1, 7)

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
        origins = self.env.scene.env_origins[env_ids]
        n = len(env_ids)

        # random lateral offset: Uniform[-blocker_y_max, +blocker_y_max]
        rand_y = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.spec.blocker_y_max

        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0] = origins[:, 0] + self.spec.blocker_x
        pose[:, 1] = origins[:, 1] + rand_y
        pose[:, 2] = 0.9
        pose[:, 3] = 1.0

        # static or slow patrol depending on dynamic_ratio
        velocity = torch.zeros(n, 6, device=self.device)
        if self.spec.dynamic_ratio > 0.0:
            is_dynamic = torch.rand(n, device=self.device) < self.spec.dynamic_ratio
            # slow lateral patrol speed ±0.3 m/s, direction random
            lateral_v = (torch.randint(0, 2, (n,), device=self.device).float() * 2.0 - 1.0) * 0.3
            velocity[:, 1] = torch.where(is_dynamic, lateral_v, torch.zeros(n, device=self.device))

        obstacle = self.env.scene["obstacle_0"]
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(velocity, env_ids=env_ids)

        if self.scheduler is not None:
            self.scheduler.positions[env_ids, 0, 0] = self.spec.blocker_x
            self.scheduler.positions[env_ids, 0, 1] = rand_y
            # static envs keep velocity 0; dynamic envs keep lateral_v set above
            if self.spec.dynamic_ratio > 0.0:
                self.scheduler.velocities[env_ids, 0] = velocity[:, 1]
            else:
                self.scheduler.velocities[env_ids, 0] = 0.0

