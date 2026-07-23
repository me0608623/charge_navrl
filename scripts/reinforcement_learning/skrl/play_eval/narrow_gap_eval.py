"""Deterministic configurable wall-gap policy evaluation scene.

Two wall segments span the arena from the north/south boundary to a centered
opening. The robot starts west of the barrier and the goal is east, so
reaching the goal requires traversing the opening rather than driving around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NarrowGapSpec:
    gap_width: float = 0.85
    arena_half_extent: float = 5.0
    boundary_wall_width: float = 1.0
    wall_width: float = 1.0
    barrier_x: float = 0.0
    start_x: float = -3.0
    goal_x: float = 3.0
    yaw_limit_deg: float = 2.52

    @property
    def boundary_inner_y(self) -> float:
        return self.arena_half_extent - 0.5 * self.boundary_wall_width

    @property
    def segment_length(self) -> float:
        return self.boundary_inner_y - 0.5 * self.gap_width

    @property
    def segment_center_y(self) -> float:
        return 0.5 * (self.boundary_inner_y + 0.5 * self.gap_width)


def configure_narrow_gap_env(env_cfg, scene_final: dict, cli_args, spec=None) -> None:
    """Remove random geometry and reserve two exact barrier-wall segments."""
    spec = spec or NarrowGapSpec()
    scene_final.update({
        "num_goals": 1,
        "num_static": 0,
        "num_dynamic": 0,
        "walls_min": 2,
        "walls_max": 2,
        "goal_dist_min": abs(spec.goal_x - spec.start_x),
        "goal_dist_max": abs(spec.goal_x - spec.start_x),
    })
    cli_args.no_goal_movement = True

    goal_cfg = env_cfg.commands.goal_command
    goal_cfg.num_goals = 1
    goal_cfg.num_obstacles = 0
    # Bootstrap only. The controller installs the exact target after env creation.
    goal_cfg.ranges.distance = (1.5, 3.0)
    goal_cfg.ranges.angle = (-math.pi, math.pi)

    for event_name in ("randomize_obstacles", "randomize_obstacles_startup"):
        event = getattr(env_cfg.events, event_name, None)
        if event is not None:
            event.params.update({
                "empty_ratio": 1.0,
                "static_ratio": 1.0,
                "dynamic_ratio": 0.0,
                "num_obstacles_static": 0,
                "num_obstacles_dynamic": 0,
            })

    wall_event = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_event is not None:
        wall_event.params.update({"min_walls": 0, "max_walls": 0})

    reset_event = getattr(env_cfg.events, "reset_base", None)
    if reset_event is not None:
        pose_range = reset_event.params.setdefault("pose_range", {})
        pose_range.update({"x": (spec.start_x, spec.start_x), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        velocity_range = reset_event.params.setdefault("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)

    for slot in NarrowGapController.WALL_SLOTS:
        wall_cfg = getattr(env_cfg.scene, f"wall_internal_{slot}", None)
        if wall_cfg is not None:
            # Asset local x is the long axis; the controller rotates it by 90 deg.
            wall_cfg.spawn.size = (spec.segment_length, spec.wall_width, 3.0)


class NarrowGapController:
    """Own exact scene placement and collect throat yaw/crossing statistics."""

    WALL_SLOTS = (1, 7)

    def __init__(self, raw_env, spec=None) -> None:
        self.env = raw_env
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.spec = spec or NarrowGapSpec()
        self._previous_x = torch.full((self.num_envs,), self.spec.start_x, device=self.device)
        self._crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completed_episodes = 0
        self.crossed_episodes = 0
        self._yaw_samples: list[np.ndarray] = []

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self._place_robot_goal(env_ids)
        self._place_walls(env_ids)
        self._previous_x[env_ids] = self.spec.start_x
        self._crossed[env_ids] = False

    def observe(self) -> None:
        robot = self.env.scene["robot"]
        pos = robot.data.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        quat = robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        # Vehicle overlaps the barrier slab longitudinally. These are the frames
        # where alignment determines whether the configured opening is traversable.
        throat = (pos[:, 0] - self.spec.barrier_x).abs() <= (0.5 * self.spec.wall_width + 0.35)
        if throat.any():
            yaw_abs_deg = torch.rad2deg(torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs())
            self._yaw_samples.append(yaw_abs_deg[throat].detach().cpu().numpy())

        crossed_now = (self._previous_x < self.spec.barrier_x) & (pos[:, 0] >= self.spec.barrier_x)
        self._crossed |= crossed_now
        self._previous_x.copy_(pos[:, 0])

    def finish_episodes(self, env_ids: torch.Tensor) -> None:
        self.completed_episodes += int(env_ids.numel())
        self.crossed_episodes += int(self._crossed[env_ids].sum().item())

    def summary_line(self) -> str:
        yaw = np.concatenate(self._yaw_samples) if self._yaw_samples else np.empty(0, dtype=np.float32)
        if yaw.size:
            p50, p90, p95 = np.percentile(yaw, [50, 90, 95])
            within = float((yaw <= self.spec.yaw_limit_deg).mean())
        else:
            p50 = p90 = p95 = float("nan")
            within = float("nan")
        crossing_rate = self.crossed_episodes / self.completed_episodes if self.completed_episodes else float("nan")
        return (
            f"[NARROW-GAP-METRICS] episodes={self.completed_episodes} "
            f"crossed={self.crossed_episodes} crossing_rate={crossing_rate:.6f} "
            f"yaw_frames={yaw.size} yaw_abs_p50_deg={p50:.4f} "
            f"yaw_abs_p90_deg={p90:.4f} yaw_abs_p95_deg={p95:.4f} "
            f"yaw_within_{self.spec.yaw_limit_deg:.2f}deg={within:.6f}"
        )

    def _place_robot_goal(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        robot = self.env.scene["robot"]
        pose = robot.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 0] += self.spec.start_x
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
        centers_y = (self.spec.segment_center_y, -self.spec.segment_center_y)
        half_yaw = math.pi / 4.0
        wall_quat = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            device=self.device,
        )

        for slot in range(8):
            try:
                wall = self.env.scene[f"wall_internal_{slot}"]
            except KeyError:
                continue
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            pose[:, :2] = origins[:, :2]
            pose[:, 2] = -10.0
            pose[:, 3] = 1.0
            if slot in self.WALL_SLOTS:
                idx = self.WALL_SLOTS.index(slot)
                pose[:, 0] += self.spec.barrier_x
                pose[:, 1] += centers_y[idx]
                pose[:, 2] = 1.5
                pose[:, 3:7] = wall_quat
            wall.write_root_pose_to_sim(pose, env_ids=env_ids)

        if hasattr(self.env, "_maze_wall_mask"):
            self.env._maze_wall_mask[env_ids] = False
            for idx, slot in enumerate(self.WALL_SLOTS):
                self.env._maze_wall_mask[env_ids, slot] = True
                self.env._maze_wall_centers[env_ids, slot, 0] = self.spec.barrier_x
                self.env._maze_wall_centers[env_ids, slot, 1] = centers_y[idx]
                self.env._maze_wall_sizes[env_ids, slot, 0] = self.spec.wall_width
                self.env._maze_wall_sizes[env_ids, slot, 1] = self.spec.segment_length
