"""Deterministic near-wall pedestrian crossing evaluation.

This scene isolates the real-car failure chain:
pedestrian crosses the robot path, a wall blocks the pedestrian's destination
side, and the goal stays straight ahead.  The left/right mirror is selected
deterministically so policy checkpoints see identical geometry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class NearWallCrossingSpec:
    goal_x: float = 6.0
    crossing_x: float = 2.4
    pedestrian_start_side: float = 2.1
    pedestrian_speed: float = 0.65
    wall_center_x: float = 3.7
    # Four deterministic cases = left/right mirror x two lateral clearances.
    # Near faces are at |y|=1.05/1.40m, leaving 0.70/1.05m from robot body edge.
    wall_side_y_values: tuple[float, float] = (1.55, 1.90)
    pedestrian_stop_side_values: tuple[float, float] = (0.65, 1.10)
    wall_length: float = 4.0
    wall_width: float = 1.0
    robot_radius: float = 0.35
    obstacle_radius: float = 0.35
    first_turn_threshold: float = 0.25
    spin_speed_threshold: float = 0.10
    spin_omega_threshold: float = 0.80


def configure_near_wall_crossing_env(env_cfg, scene_final: dict, cli_args) -> None:
    """Force one obstacle, one wall, fixed robot/goal ranges for eval-only D."""
    scene_final.update({
        "num_goals": 1,
        "num_static": 0,
        "num_dynamic": 1,
        "walls_min": 1,
        "walls_max": 1,
        "wall_length": 4.0,
        "goal_dist_min": 6.0,
        "goal_dist_max": 6.0,
        "episode_length_s": 15.0,
        "obstacle_behavior": "horizontal_crossing",
    })
    cli_args.obstacle_behavior = "horizontal_crossing"
    cli_args.no_goal_movement = True

    goal_cfg = env_cfg.commands.goal_command
    goal_cfg.num_goals = 1
    goal_cfg.num_obstacles = 1
    goal_cfg.ranges.distance = (6.0, 6.0)
    goal_cfg.ranges.angle = (0.0, 0.0)
    env_cfg.episode_length_s = 15.0

    for event_name in ("randomize_obstacles", "randomize_obstacles_startup"):
        event = getattr(env_cfg.events, event_name, None)
        if event is not None:
            event.params.update({
                "empty_ratio": 0.0,
                "static_ratio": 0.0,
                "dynamic_ratio": 1.0,
                "num_obstacles_static": 0,
                "num_obstacles_dynamic": 1,
            })

    wall_event = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_event is not None:
        wall_event.params.update({"min_walls": 1, "max_walls": 1, "target_wall_length": 4.0})

    reset_event = getattr(env_cfg.events, "reset_base", None)
    if reset_event is not None:
        pose_range = reset_event.params.setdefault("pose_range", {})
        pose_range.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        velocity_range = reset_event.params.setdefault("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)

    # Keep contact observable instead of auto-resetting at first touch. D records
    # would-collide flags itself, so a wall corner can expose the full 360 tail.
    for term_name in list(vars(env_cfg.terminations)):
        if "collision" in term_name:
            setattr(env_cfg.terminations, term_name, None)


class NearWallCrossingController:
    """Own deterministic scene placement and anti-spin acceptance metrics."""

    def __init__(
        self,
        raw_env,
        scheduler,
        step_dt: float,
        direction: str = "mirrored",
        output_path: str = "",
        probe_output_path: str = "",
        spec: NearWallCrossingSpec | None = None,
    ) -> None:
        self.env = raw_env
        self.scheduler = scheduler
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.dt = float(step_dt)
        self.direction = direction
        self.output_path = output_path
        self.probe_output_path = probe_output_path
        self.probe_factorial = bool(probe_output_path)
        self.spec = spec or NearWallCrossingSpec()
        self.episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.records: list[dict] = []

        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.cross_sign = torch.ones(self.num_envs, device=self.device)
        self.wall_sign = torch.ones(self.num_envs, device=self.device)
        self.wall_variant = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wall_side_y = torch.full((self.num_envs,), self.spec.wall_side_y_values[0], device=self.device)
        self.pedestrian_stop_side = torch.full(
            (self.num_envs,), self.spec.pedestrian_stop_side_values[0], device=self.device
        )
        self.first_turn_sign = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.first_turn_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.same_sign = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.same_sign_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_same_sign_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.same_sign_yaw = torch.zeros(self.num_envs, device=self.device)
        self.max_same_sign_yaw = torch.zeros(self.num_envs, device=self.device)
        self.spin_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_spin_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.signed_yaw = torch.zeros(self.num_envs, device=self.device)
        self.abs_yaw = torch.zeros(self.num_envs, device=self.device)
        self.net_lateral_drift = torch.zeros(self.num_envs, device=self.device)
        self.min_wall_clearance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.min_obstacle_clearance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._probe_lidar_stack: list[np.ndarray] = []
        self._probe_cross_sign: list[np.ndarray] = []
        self._probe_wall_sign: list[np.ndarray] = []
        self._probe_distance: list[np.ndarray] = []
        self._probe_ped_body_xy: list[np.ndarray] = []
        self._probe_moving: list[np.ndarray] = []
        self._probe_env_id: list[np.ndarray] = []
        self._probe_episode_id: list[np.ndarray] = []
        self._probe_step: list[np.ndarray] = []

    def _case_index(self, env_ids: torch.Tensor) -> torch.Tensor:
        return (env_ids + self.episode_index[env_ids] * self.num_envs) % 4

    def _direction_for(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.direction == "right_to_left":
            return torch.ones(len(env_ids), device=self.device)
        if self.direction == "left_to_right":
            return -torch.ones(len(env_ids), device=self.device)
        case_index = (
            (env_ids + self.episode_index[env_ids] * self.num_envs) % 8
            if self.probe_factorial
            else self._case_index(env_ids)
        )
        parity = case_index % 2
        return torch.where(parity == 0, 1.0, -1.0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.cross_sign[env_ids] = self._direction_for(env_ids)
        if self.probe_factorial and self.direction == "mirrored":
            case_index = (env_ids + self.episode_index[env_ids] * self.num_envs) % 8
            variant = (case_index // 2) % 2
            wall_relative_sign = torch.where(
                (case_index // 4) % 2 == 0,
                torch.ones_like(self.cross_sign[env_ids]),
                -torch.ones_like(self.cross_sign[env_ids]),
            )
            self.wall_sign[env_ids] = self.cross_sign[env_ids] * wall_relative_sign
        elif self.direction == "mirrored":
            variant = self._case_index(env_ids) // 2
            self.wall_sign[env_ids] = self.cross_sign[env_ids]
        else:
            variant = (env_ids + self.episode_index[env_ids]) % 2
            self.wall_sign[env_ids] = self.cross_sign[env_ids]
        self.wall_variant[env_ids] = variant
        wall_values = torch.tensor(self.spec.wall_side_y_values, device=self.device)
        stop_values = torch.tensor(self.spec.pedestrian_stop_side_values, device=self.device)
        self.wall_side_y[env_ids] = wall_values[variant]
        self.pedestrian_stop_side[env_ids] = stop_values[variant]
        self.elapsed[env_ids] = 0.0
        self.first_turn_sign[env_ids] = 0
        self.first_turn_step[env_ids] = -1
        self.steps[env_ids] = 0
        self.same_sign[env_ids] = 0
        self.same_sign_run[env_ids] = 0
        self.max_same_sign_run[env_ids] = 0
        self.same_sign_yaw[env_ids] = 0.0
        self.max_same_sign_yaw[env_ids] = 0.0
        self.spin_run[env_ids] = 0
        self.max_spin_run[env_ids] = 0
        self.signed_yaw[env_ids] = 0.0
        self.abs_yaw[env_ids] = 0.0
        self.net_lateral_drift[env_ids] = 0.0
        self.min_wall_clearance[env_ids] = float("inf")
        self.min_obstacle_clearance[env_ids] = float("inf")

        self._place_robot_and_goal(env_ids)
        self._place_walls(env_ids)
        self._place_pedestrian(env_ids)
        self.episode_index[env_ids] += 1

    def advance(self) -> None:
        self.elapsed += self.dt
        self._place_pedestrian(torch.arange(self.num_envs, device=self.device))

    def _place_robot_and_goal(self, env_ids: torch.Tensor) -> None:
        robot = self.env.scene["robot"]
        origins = self.env.scene.env_origins[env_ids]
        pose = robot.data.default_root_state[env_ids, :7].clone()
        pose[:, 0] = origins[:, 0]
        pose[:, 1] = origins[:, 1]
        pose[:, 3:7] = 0.0
        pose[:, 3] = 1.0
        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)

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
        hidden_z = -10.0
        for slot in range(8):
            name = f"wall_internal_{slot}"
            try:
                wall = self.env.scene[name]
            except KeyError:
                continue
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            pose[:, 0] = origins[:, 0]
            pose[:, 1] = origins[:, 1]
            pose[:, 2] = hidden_z
            pose[:, 3] = 1.0
            if slot == 0:
                pose[:, 0] += self.spec.wall_center_x
                pose[:, 1] += self.wall_sign[env_ids] * self.wall_side_y[env_ids]
                pose[:, 2] = 1.5
            wall.write_root_pose_to_sim(pose, env_ids=env_ids)

        if hasattr(self.env, "_maze_wall_mask"):
            self.env._maze_wall_mask[env_ids] = False
            self.env._maze_wall_mask[env_ids, 0] = True
            self.env._maze_wall_centers[env_ids, 0, 0] = self.spec.wall_center_x
            self.env._maze_wall_centers[env_ids, 0, 1] = self.wall_sign[env_ids] * self.wall_side_y[env_ids]
            self.env._maze_wall_sizes[env_ids, 0, 0] = self.spec.wall_length
            self.env._maze_wall_sizes[env_ids, 0, 1] = self.spec.wall_width

    def _place_pedestrian(self, env_ids: torch.Tensor) -> None:
        sign = self.cross_sign[env_ids]
        y = -sign * self.spec.pedestrian_start_side + sign * self.spec.pedestrian_speed * self.elapsed[env_ids]
        stop_side = (
            torch.full_like(sign, self.spec.pedestrian_start_side)
            if self.probe_factorial
            else self.pedestrian_stop_side[env_ids]
        )
        y = torch.where(sign > 0, torch.minimum(y, stop_side), torch.maximum(y, -stop_side))
        moving = (sign * y) < stop_side
        vy = torch.where(moving, sign * self.spec.pedestrian_speed, torch.zeros_like(sign))

        origins = self.env.scene.env_origins[env_ids]
        pose = torch.zeros(len(env_ids), 7, device=self.device)
        pose[:, 0] = origins[:, 0] + self.spec.crossing_x
        pose[:, 1] = origins[:, 1] + y
        pose[:, 2] = 0.85
        pose[:, 3] = 1.0
        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        velocity[:, 1] = vy
        obstacle = self.env.scene["obstacle_0"]
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(velocity, env_ids=env_ids)

        if self.scheduler is not None:
            self.scheduler.positions[env_ids, 0, 0] = self.spec.crossing_x
            self.scheduler.positions[env_ids, 0, 1] = y
            self.scheduler.velocities[env_ids, 0, 0] = 0.0
            self.scheduler.velocities[env_ids, 0, 1] = vy

    def record_step(
        self,
        linear_speed: torch.Tensor,
        omega: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> None:
        linear_speed = linear_speed.detach().to(self.device)
        omega = omega.detach().to(self.device)
        self.steps += 1

        first = (self.first_turn_step < 0) & (omega.abs() >= self.spec.first_turn_threshold)
        self.first_turn_step[first] = self.steps[first]
        self.first_turn_sign[first] = torch.sign(omega[first]).long()

        omega_sign = torch.where(omega.abs() >= self.spec.first_turn_threshold, torch.sign(omega).long(), 0)
        same = (omega_sign != 0) & (omega_sign == self.same_sign)
        self.same_sign_run = torch.where(same, self.same_sign_run + 1, torch.where(omega_sign != 0, 1, 0))
        self.same_sign_yaw = torch.where(
            same,
            self.same_sign_yaw + omega.abs() * self.dt,
            torch.where(omega_sign != 0, omega.abs() * self.dt, torch.zeros_like(omega)),
        )
        self.same_sign = torch.where(omega_sign != 0, omega_sign, torch.zeros_like(self.same_sign))
        self.max_same_sign_run = torch.maximum(self.max_same_sign_run, self.same_sign_run)
        self.max_same_sign_yaw = torch.maximum(self.max_same_sign_yaw, self.same_sign_yaw)

        spinning = (linear_speed.abs() < self.spec.spin_speed_threshold) & (omega.abs() > self.spec.spin_omega_threshold)
        self.spin_run = torch.where(spinning, self.spin_run + 1, torch.zeros_like(self.spin_run))
        self.max_spin_run = torch.maximum(self.max_spin_run, self.spin_run)
        self.signed_yaw += omega * self.dt
        self.abs_yaw += omega.abs() * self.dt

        robot_xy = self.env.scene["robot"].data.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        # Isaac Lab auto-resets terminal envs inside env.step(). Preserve the
        # previous valid position for those envs instead of recording reset y=0.
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if done is not None:
            active &= ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        signed_lateral = self.cross_sign * robot_xy[:, 1]
        self.net_lateral_drift = torch.where(active, signed_lateral, self.net_lateral_drift)
        wall_center = torch.stack([
            torch.full_like(self.cross_sign, self.spec.wall_center_x),
            self.wall_sign * self.wall_side_y,
        ], dim=1)
        wall_half = torch.tensor([self.spec.wall_length / 2, self.spec.wall_width / 2], device=self.device)
        wall_delta = (robot_xy - wall_center).abs() - wall_half
        wall_clearance = wall_delta.clamp_min(0.0).norm(dim=1) - self.spec.robot_radius
        self.min_wall_clearance = torch.minimum(self.min_wall_clearance, wall_clearance)

        ped_y = -self.cross_sign * self.spec.pedestrian_start_side + self.cross_sign * self.spec.pedestrian_speed * self.elapsed
        stop_side = (
            torch.full_like(self.cross_sign, self.spec.pedestrian_start_side)
            if self.probe_factorial
            else self.pedestrian_stop_side
        )
        ped_y = torch.where(
            self.cross_sign > 0,
            torch.minimum(ped_y, stop_side),
            torch.maximum(ped_y, -stop_side),
        )
        ped_xy = torch.stack([torch.full_like(ped_y, self.spec.crossing_x), ped_y], dim=1)
        obs_clearance = (robot_xy - ped_xy).norm(dim=1) - self.spec.robot_radius - self.spec.obstacle_radius
        self.min_obstacle_clearance = torch.minimum(self.min_obstacle_clearance, obs_clearance)

    def record_probe(self, lidar_stack: torch.Tensor) -> None:
        """Record the exact normalized Kx72 LiDAR tensor consumed by the policy."""
        if not self.probe_output_path:
            return
        if lidar_stack.ndim != 2 or lidar_stack.shape[0] != self.num_envs or lidar_stack.shape[1] % 72:
            raise ValueError(f"expected [E, K*72] LiDAR stack, got {tuple(lidar_stack.shape)}")

        robot = self.env.scene["robot"].data
        robot_xy = robot.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        quat = robot.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
        )
        ped_y = -self.cross_sign * self.spec.pedestrian_start_side + self.cross_sign * self.spec.pedestrian_speed * self.elapsed
        stop_side = (
            torch.full_like(self.cross_sign, self.spec.pedestrian_start_side)
            if self.probe_factorial
            else self.pedestrian_stop_side
        )
        ped_y = torch.where(
            self.cross_sign > 0,
            torch.minimum(ped_y, stop_side),
            torch.maximum(ped_y, -stop_side),
        )
        moving = (self.cross_sign * ped_y) < stop_side
        ped_xy = torch.stack([torch.full_like(ped_y, self.spec.crossing_x), ped_y], dim=1)
        delta = ped_xy - robot_xy
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        ped_body = torch.stack(
            [cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1], -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]],
            dim=1,
        )

        self._probe_lidar_stack.append(lidar_stack.detach().cpu().numpy().astype(np.float32, copy=True))
        self._probe_cross_sign.append(self.cross_sign.detach().cpu().numpy().astype(np.int8, copy=True))
        self._probe_wall_sign.append(self.wall_sign.detach().cpu().numpy().astype(np.int8, copy=True))
        self._probe_distance.append(delta.norm(dim=1).detach().cpu().numpy().astype(np.float32, copy=True))
        self._probe_ped_body_xy.append(ped_body.detach().cpu().numpy().astype(np.float32, copy=True))
        self._probe_moving.append(moving.detach().cpu().numpy().astype(np.bool_, copy=True))
        self._probe_env_id.append(np.arange(self.num_envs, dtype=np.int32))
        self._probe_episode_id.append((self.episode_index - 1).detach().cpu().numpy().astype(np.int32, copy=True))
        self._probe_step.append(self.steps.detach().cpu().numpy().astype(np.int32, copy=True))

    def finish_episodes(self, env_ids: torch.Tensor, causes: torch.Tensor) -> list[dict]:
        summaries = []
        for env_id in env_ids.detach().cpu().tolist():
            desired = -int(self.cross_sign[env_id].item())
            first = int(self.first_turn_sign[env_id].item())
            row = {
                "env_id": env_id,
                "episode_index": int(self.episode_index[env_id].item()) - 1,
                "crossing": "right_to_left" if self.cross_sign[env_id] > 0 else "left_to_right",
                "wall_variant": "near" if self.wall_variant[env_id] == 0 else "moderate",
                "wall_side_y_m": float(self.wall_side_y[env_id].item()),
                "side_body_clearance_m": float(
                    self.wall_side_y[env_id].item() - self.spec.wall_width / 2 - self.spec.robot_radius
                ),
                "desired_turn_sign": desired,
                "first_turn_sign": first,
                "first_turn_correct": first == desired,
                "first_turn_step": int(self.first_turn_step[env_id].item()),
                "max_same_sign_omega_s": float(self.max_same_sign_run[env_id].item() * self.dt),
                "max_same_sign_yaw_deg": float(torch.rad2deg(self.max_same_sign_yaw[env_id]).item()),
                "spin_360": bool(self.max_same_sign_yaw[env_id].item() >= 2.0 * torch.pi),
                "max_low_v_high_omega_s": float(self.max_spin_run[env_id].item() * self.dt),
                "signed_yaw_deg": float(torch.rad2deg(self.signed_yaw[env_id]).item()),
                "absolute_yaw_deg": float(torch.rad2deg(self.abs_yaw[env_id]).item()),
                "net_lateral_drift_m": float(self.net_lateral_drift[env_id].item()),
                "min_wall_clearance_m": float(self.min_wall_clearance[env_id].item()),
                "min_obstacle_clearance_m": float(self.min_obstacle_clearance[env_id].item()),
                "wall_contact": bool(self.min_wall_clearance[env_id].item() <= 0.0),
                "obstacle_contact": bool(self.min_obstacle_clearance[env_id].item() <= 0.0),
                "termination_cause": int(causes[env_id].item()),
                "steps": int(self.steps[env_id].item()),
            }
            summaries.append(row)
            self.records.append(row)
        return summaries

    def write_report(self) -> None:
        if not self.output_path and not self.probe_output_path:
            return
        if self.output_path:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"spec": asdict(self.spec), "direction": self.direction, "episodes": self.records}
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        if self.probe_output_path and self._probe_lidar_stack:
            probe_path = Path(self.probe_output_path)
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            lidar_te = np.stack(self._probe_lidar_stack)
            np.savez_compressed(
                probe_path,
                lidar_stack=lidar_te.reshape(-1, lidar_te.shape[-1]),
                crossing_sign=np.stack(self._probe_cross_sign).reshape(-1),
                wall_sign=np.stack(self._probe_wall_sign).reshape(-1),
                obstacle_distance_m=np.stack(self._probe_distance).reshape(-1),
                pedestrian_body_xy=np.stack(self._probe_ped_body_xy).reshape(-1, 2),
                moving=np.stack(self._probe_moving).reshape(-1),
                env_id=np.stack(self._probe_env_id).reshape(-1),
                episode_id=np.stack(self._probe_episode_id).reshape(-1),
                episode_step=np.stack(self._probe_step).reshape(-1),
                T=np.asarray(lidar_te.shape[0]),
                E=np.asarray(lidar_te.shape[1]),
                frame_stack=np.asarray(lidar_te.shape[2] // 72),
                lidar_semantics=np.asarray("policy_normalized_current_then_history"),
                factorial_wall_side=np.asarray(True),
            )
