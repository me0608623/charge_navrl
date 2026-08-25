"""Stateful interaction metrics for fixed-delay corridor teacher screens."""

from __future__ import annotations

import math

import torch


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "maximum": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "samples": len(values),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "maximum": float(tensor.max().item()),
    }


class TeacherInteractionMetrics:
    """Measure waiting, relaunch, side commitment, reverse, and turning.

    A safe gap is operationally defined as at least one forward candidate whose
    delayed 2 s path is jointly feasible against dynamic obstacles and walls.
    The selected-side label is delayed by one step before it is compared with
    the applied command, matching the fixed d1 actuator queue.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        dt_s: float,
        stop_speed_mps: float = 0.10,
        launch_speed_mps: float = 0.15,
        reverse_speed_mps: float = 0.05,
        side_deadband_m: float = 0.15,
        passage_progress_m: float = 0.20,
        interaction_distance_m: float = 3.0,
        recent_vacated_window_s: float = 3.0,
        u_turn_threshold_deg: float = 135.0,
        loop_rotation_threshold_deg: float = 360.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        positive = {
            "dt_s": dt_s,
            "stop_speed_mps": stop_speed_mps,
            "launch_speed_mps": launch_speed_mps,
            "reverse_speed_mps": reverse_speed_mps,
            "side_deadband_m": side_deadband_m,
            "passage_progress_m": passage_progress_m,
            "interaction_distance_m": interaction_distance_m,
            "recent_vacated_window_s": recent_vacated_window_s,
            "u_turn_threshold_deg": u_turn_threshold_deg,
            "loop_rotation_threshold_deg": loop_rotation_threshold_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if launch_speed_mps <= stop_speed_mps:
            raise ValueError("launch speed must exceed stop speed")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt_s = float(dt_s)
        self.stop_speed_mps = float(stop_speed_mps)
        self.launch_speed_mps = float(launch_speed_mps)
        self.reverse_speed_mps = float(reverse_speed_mps)
        self.side_deadband_m = float(side_deadband_m)
        self.passage_progress_m = float(passage_progress_m)
        self.interaction_distance_m = float(interaction_distance_m)
        self.recent_vacated_steps = int(
            math.ceil(recent_vacated_window_s / dt_s)
        )
        self.u_turn_threshold_rad = math.radians(u_turn_threshold_deg)
        self.loop_rotation_threshold_rad = math.radians(
            loop_rotation_threshold_deg
        )

        e = self.num_envs
        self._step = torch.zeros(e, dtype=torch.long, device=self.device)
        self._episode_active = torch.zeros(
            e, dtype=torch.bool, device=self.device
        )
        self._initial_xy = torch.zeros(e, 2, device=self.device)
        self._forward = torch.zeros(e, 2, device=self.device)
        self._lateral = torch.zeros(e, 2, device=self.device)
        self._initial_yaw = torch.zeros(e, device=self.device)
        self._previous_yaw = torch.zeros(e, device=self.device)
        self._cumulative_abs_yaw = torch.zeros(e, device=self.device)
        self._max_heading_excursion = torch.zeros(e, device=self.device)
        self._wait_frames = torch.zeros(e, dtype=torch.long, device=self.device)
        self._interaction_frames = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._reverse_frames = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._reverse_distance = torch.zeros(e, device=self.device)
        self._stop_go = torch.zeros(e, dtype=torch.long, device=self.device)
        self._go_stop = torch.zeros(e, dtype=torch.long, device=self.device)
        self._side_switches = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._previous_motion = torch.full(
            (e,), -2, dtype=torch.int8, device=self.device
        )
        self._last_moving_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._pending_selected_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._pending_vacated_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._pending_vacated_eligible = torch.zeros(
            e, dtype=torch.bool, device=self.device
        )
        self._seen_blocked = torch.zeros(
            e, dtype=torch.bool, device=self.device
        )
        self._gap_open = torch.zeros(e, dtype=torch.bool, device=self.device)
        self._gap_open_step = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._recent_vacated_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._recent_vacated_step = torch.full(
            (e,), -10_000, dtype=torch.long, device=self.device
        )
        self._previous_dynamic_lateral: torch.Tensor | None = None
        self._previous_dynamic_valid: torch.Tensor | None = None

        self._frames = 0
        self._safe_gap_opportunities = 0
        self._safe_gap_launches = 0
        self._safe_gap_missed = 0
        self._vacated_side_eligible = 0
        self._vacated_side_matches = 0
        self._vacated_side_center = 0
        self._launch_delays_s: list[float] = []
        self._episode_wait_s: list[float] = []
        self._episode_reverse_distance_m: list[float] = []
        self._episode_stop_go: list[float] = []
        self._episode_go_stop: list[float] = []
        self._episode_side_switches: list[float] = []
        self._episode_max_heading_deg: list[float] = []
        self._episode_abs_rotation_deg: list[float] = []
        self._episodes = 0
        self._u_turn_episodes = 0
        self._full_rotation_episodes = 0

    def _reset_rows(
        self,
        env_ids: torch.Tensor,
        robot_xy_m: torch.Tensor,
        robot_yaw_rad: torch.Tensor,
        goal_xy_m: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        direction = goal_xy_m[env_ids] - robot_xy_m[env_ids]
        norm = direction.norm(dim=-1, keepdim=True)
        yaw_forward = torch.stack(
            [torch.cos(robot_yaw_rad[env_ids]), torch.sin(robot_yaw_rad[env_ids])],
            dim=1,
        )
        forward = torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), yaw_forward)
        self._initial_xy[env_ids] = robot_xy_m[env_ids]
        self._forward[env_ids] = forward
        self._lateral[env_ids, 0] = -forward[:, 1]
        self._lateral[env_ids, 1] = forward[:, 0]
        self._initial_yaw[env_ids] = robot_yaw_rad[env_ids]
        self._previous_yaw[env_ids] = robot_yaw_rad[env_ids]
        for value in (
            self._step,
            self._cumulative_abs_yaw,
            self._max_heading_excursion,
            self._wait_frames,
            self._interaction_frames,
            self._reverse_frames,
            self._reverse_distance,
            self._stop_go,
            self._go_stop,
            self._side_switches,
            self._last_moving_side,
            self._pending_selected_side,
            self._pending_vacated_side,
            self._gap_open_step,
            self._recent_vacated_side,
        ):
            value[env_ids] = 0
        self._previous_motion[env_ids] = -2
        self._pending_vacated_eligible[env_ids] = False
        self._seen_blocked[env_ids] = False
        self._gap_open[env_ids] = False
        self._recent_vacated_step[env_ids] = -10_000
        self._episode_active[env_ids] = True

    def record_step(
        self,
        *,
        just_reset: torch.Tensor,
        robot_xy_m: torch.Tensor,
        robot_yaw_rad: torch.Tensor,
        goal_xy_m: torch.Tensor,
        applied_command: torch.Tensor,
        selected_endpoint_m: torch.Tensor,
        selected_feasible: torch.Tensor,
        feasible_grid: torch.Tensor,
        linear_velocity_grid: torch.Tensor,
        endpoint_grid_m: torch.Tensor,
        dynamic_positions_m: torch.Tensor,
        dynamic_velocities_mps: torch.Tensor,
        dynamic_valid: torch.Tensor,
        selected_side_hint: torch.Tensor | None = None,
    ) -> None:
        """Record the current state and action selected for the d1 queue."""

        e = self.num_envs
        expected_vector = (e,)
        if just_reset.shape != expected_vector:
            raise ValueError("just_reset must have shape [E]")
        if robot_xy_m.shape != (e, 2) or goal_xy_m.shape != (e, 2):
            raise ValueError("robot and goal positions must have shape [E,2]")
        if robot_yaw_rad.shape != expected_vector:
            raise ValueError("robot yaw must have shape [E]")
        if applied_command.shape != (e, 2):
            raise ValueError("applied command must have shape [E,2]")
        if selected_endpoint_m.shape != (e, 2):
            raise ValueError("selected endpoint must have shape [E,2]")
        if selected_feasible.shape != expected_vector:
            raise ValueError("selected feasible must have shape [E]")
        if feasible_grid.shape != linear_velocity_grid.shape:
            raise ValueError("feasible and linear grids must match")
        if endpoint_grid_m.shape != (*feasible_grid.shape, 2):
            raise ValueError("endpoint grid must have shape [E,L,A,2]")
        if (
            dynamic_positions_m.shape != dynamic_velocities_mps.shape
            or dynamic_positions_m.ndim != 3
            or dynamic_positions_m.shape[0] != e
            or dynamic_positions_m.shape[-1] != 2
        ):
            raise ValueError("dynamic states must have shape [E,N,2]")
        if dynamic_valid.shape != dynamic_positions_m.shape[:2]:
            raise ValueError("dynamic valid must have shape [E,N]")
        if selected_side_hint is not None:
            if selected_side_hint.shape != expected_vector:
                raise ValueError("selected side hint must have shape [E]")
            if bool((selected_side_hint.abs() > 1).any()):
                raise ValueError("selected side hint must be -1, 0, or 1")

        reset = just_reset.to(device=self.device, dtype=torch.bool)
        reset_ids = reset.nonzero(as_tuple=False).reshape(-1)
        self._reset_rows(
            reset_ids, robot_xy_m, robot_yaw_rad, goal_xy_m
        )
        self._frames += e

        yaw_increment = _wrap_angle(robot_yaw_rad - self._previous_yaw).abs()
        self._cumulative_abs_yaw += yaw_increment
        heading_excursion = _wrap_angle(
            robot_yaw_rad - self._initial_yaw
        ).abs()
        self._max_heading_excursion = torch.maximum(
            self._max_heading_excursion, heading_excursion
        )
        self._previous_yaw[:] = robot_yaw_rad

        relative_dynamic = dynamic_positions_m - robot_xy_m[:, None, :]
        dynamic_distance = relative_dynamic.norm(dim=-1)
        dynamic_ahead = (
            relative_dynamic * self._forward[:, None, :]
        ).sum(dim=-1) > -0.5
        interaction = (
            dynamic_valid
            & dynamic_ahead
            & (dynamic_distance <= self.interaction_distance_m)
        ).any(dim=1)
        self._interaction_frames += interaction.long()

        lateral_position = (
            (dynamic_positions_m - self._initial_xy[:, None, :])
            * self._lateral[:, None, :]
        ).sum(dim=-1)
        lateral_velocity = (
            dynamic_velocities_mps * self._lateral[:, None, :]
        ).sum(dim=-1)
        if self._previous_dynamic_lateral is None:
            self._previous_dynamic_lateral = lateral_position.clone()
            self._previous_dynamic_valid = dynamic_valid.clone()
        if self._previous_dynamic_lateral.shape != lateral_position.shape:
            raise ValueError("dynamic slot count changed during teacher audit")
        self._previous_dynamic_lateral[reset] = lateral_position[reset]
        self._previous_dynamic_valid[reset] = dynamic_valid[reset]
        crossed = (
            dynamic_valid
            & self._previous_dynamic_valid
            & (lateral_velocity.abs() >= 0.05)
            & (self._previous_dynamic_lateral.abs() >= 0.02)
            & (self._previous_dynamic_lateral * lateral_position <= 0.0)
        )
        crossing_side = torch.sign(
            self._previous_dynamic_lateral
        ).to(torch.int8)
        positive = (crossed & (crossing_side > 0)).any(dim=1)
        negative = (crossed & (crossing_side < 0)).any(dim=1)
        unambiguous = positive ^ negative
        new_vacated_side = torch.where(
            positive,
            torch.ones(e, dtype=torch.int8, device=self.device),
            -torch.ones(e, dtype=torch.int8, device=self.device),
        )
        self._recent_vacated_side[unambiguous] = new_vacated_side[
            unambiguous
        ]
        self._recent_vacated_step[unambiguous] = self._step[unambiguous]
        ambiguous = positive & negative
        self._recent_vacated_side[ambiguous] = 0
        self._previous_dynamic_lateral[:] = lateral_position
        self._previous_dynamic_valid[:] = dynamic_valid

        endpoint_offset = (
            (endpoint_grid_m - robot_xy_m[:, None, None, :])
            * self._lateral[:, None, None, :]
        ).sum(dim=-1)
        endpoint_progress = (
            (endpoint_grid_m - robot_xy_m[:, None, None, :])
            * self._forward[:, None, None, :]
        ).sum(dim=-1)
        selected_offset = (
            (selected_endpoint_m - robot_xy_m) * self._lateral
        ).sum(dim=-1)
        selected_side = torch.where(
            selected_offset > self.side_deadband_m,
            torch.ones(e, dtype=torch.int8, device=self.device),
            torch.where(
                selected_offset < -self.side_deadband_m,
                -torch.ones(e, dtype=torch.int8, device=self.device),
                torch.zeros(e, dtype=torch.int8, device=self.device),
            ),
        )
        selected_side = torch.where(
            selected_feasible, selected_side, torch.zeros_like(selected_side)
        )
        if selected_side_hint is not None:
            hint = selected_side_hint.to(
                device=self.device, dtype=torch.int8
            )
            selected_side = torch.where(hint != 0, hint, selected_side)
        forward_feasible = feasible_grid & (
            linear_velocity_grid >= self.launch_speed_mps
        ) & (endpoint_progress >= self.passage_progress_m)
        any_forward_feasible = forward_feasible.flatten(1).any(dim=1)
        left_feasible = (
            forward_feasible & (endpoint_offset > self.side_deadband_m)
        ).flatten(1).any(dim=1)
        right_feasible = (
            forward_feasible & (endpoint_offset < -self.side_deadband_m)
        ).flatten(1).any(dim=1)

        recent = (
            (self._step - self._recent_vacated_step)
            <= self.recent_vacated_steps
        ) & (self._recent_vacated_side != 0)
        vacated_side_feasible = torch.where(
            self._recent_vacated_side > 0,
            left_feasible,
            right_feasible,
        )
        current_vacated_eligible = recent & vacated_side_feasible

        applied_v = applied_command[:, 0]
        motion = torch.where(
            applied_v >= self.launch_speed_mps,
            torch.ones(e, dtype=torch.int8, device=self.device),
            torch.where(
                applied_v <= -self.reverse_speed_mps,
                -torch.ones(e, dtype=torch.int8, device=self.device),
                torch.zeros(e, dtype=torch.int8, device=self.device),
            ),
        )
        stop = applied_v.abs() < self.stop_speed_mps
        self._wait_frames += (interaction & stop).long()
        reverse = motion < 0
        self._reverse_frames += reverse.long()
        self._reverse_distance += (-applied_v).clamp_min(0.0) * self.dt_s

        known_previous = self._previous_motion != -2
        self._stop_go += (
            known_previous & (self._previous_motion == 0) & (motion == 1)
        ).long()
        self._go_stop += (
            known_previous & (self._previous_motion == 1) & (motion == 0)
        ).long()

        blocked = interaction & ~any_forward_feasible
        self._seen_blocked |= blocked
        newly_open = (
            interaction
            & any_forward_feasible
            & self._seen_blocked
            & ~self._gap_open
        )
        if bool(newly_open.any()):
            self._safe_gap_opportunities += int(newly_open.sum().item())
            self._gap_open[newly_open] = True
            self._gap_open_step[newly_open] = self._step[newly_open]

        launch = (
            self._gap_open
            & (motion == 1)
            & (self._previous_motion != 1)
        )
        if bool(launch.any()):
            delays = (
                (self._step[launch] - self._gap_open_step[launch]).float()
                * self.dt_s
            )
            self._launch_delays_s.extend(delays.detach().cpu().tolist())
            launch_count = int(launch.sum().item())
            self._safe_gap_launches += launch_count
            eligible = launch & self._pending_vacated_eligible
            self._vacated_side_eligible += int(eligible.sum().item())
            matches = eligible & (
                self._pending_selected_side == self._pending_vacated_side
            )
            centers = eligible & (self._pending_selected_side == 0)
            self._vacated_side_matches += int(matches.sum().item())
            self._vacated_side_center += int(centers.sum().item())
            self._gap_open[launch] = False
            self._seen_blocked[launch] = False

        closed_without_launch = (
            self._gap_open & interaction & ~any_forward_feasible
        )
        if bool(closed_without_launch.any()):
            self._safe_gap_missed += int(
                closed_without_launch.sum().item()
            )
            self._gap_open[closed_without_launch] = False

        applied_side = self._pending_selected_side
        moving_with_side = (motion == 1) & (applied_side != 0)
        changed_side = (
            moving_with_side
            & (self._last_moving_side != 0)
            & (applied_side != self._last_moving_side)
        )
        self._side_switches += changed_side.long()
        self._last_moving_side = torch.where(
            moving_with_side, applied_side, self._last_moving_side
        )
        self._last_moving_side = torch.where(
            motion == 0,
            torch.zeros_like(self._last_moving_side),
            self._last_moving_side,
        )

        self._pending_selected_side[:] = selected_side
        self._pending_vacated_side[:] = self._recent_vacated_side
        self._pending_vacated_eligible[:] = current_vacated_eligible
        self._previous_motion[:] = motion
        self._step += 1

    def finish_episodes(self, done_ids: torch.Tensor) -> None:
        if done_ids.ndim != 1:
            raise ValueError("done_ids must have shape [D]")
        if done_ids.numel() == 0:
            return
        ids = done_ids.to(device=self.device, dtype=torch.long)
        ids = ids[self._episode_active[ids]]
        if ids.numel() == 0:
            return
        self._episodes += int(ids.numel())
        self._u_turn_episodes += int(
            (self._max_heading_excursion[ids] >= self.u_turn_threshold_rad)
            .sum()
            .item()
        )
        self._full_rotation_episodes += int(
            (self._cumulative_abs_yaw[ids] >= self.loop_rotation_threshold_rad)
            .sum()
            .item()
        )
        self._episode_wait_s.extend(
            (self._wait_frames[ids].float() * self.dt_s).cpu().tolist()
        )
        self._episode_reverse_distance_m.extend(
            self._reverse_distance[ids].cpu().tolist()
        )
        self._episode_stop_go.extend(self._stop_go[ids].float().cpu().tolist())
        self._episode_go_stop.extend(self._go_stop[ids].float().cpu().tolist())
        self._episode_side_switches.extend(
            self._side_switches[ids].float().cpu().tolist()
        )
        self._episode_max_heading_deg.extend(
            torch.rad2deg(self._max_heading_excursion[ids]).cpu().tolist()
        )
        self._episode_abs_rotation_deg.extend(
            torch.rad2deg(self._cumulative_abs_yaw[ids]).cpu().tolist()
        )
        self._episode_active[ids] = False

    def to_report(self) -> dict[str, object]:
        interaction_frames = int(self._interaction_frames.sum().item())
        wait_frames = int(self._wait_frames.sum().item())
        reverse_frames = int(self._reverse_frames.sum().item())
        eligible = self._vacated_side_eligible
        return {
            "schema": "teacher_interaction_metrics/v1",
            "semantics": {
                "safe_gap": (
                    "at least one v>=launch threshold candidate makes the "
                    "minimum corridor progress and has a jointly feasible "
                    "delayed 2 s path"
                ),
                "vacated_side": (
                    "source side of the most recent unambiguous pedestrian "
                    "centerline crossing; counted only when that side has a "
                    "feasible forward candidate"
                ),
                "u_turn": "episode max heading excursion from reset >= threshold",
                "full_rotation": (
                    "episode cumulative absolute yaw change >= threshold; this "
                    "is a rotation-budget diagnostic, not proof of a geometric loop"
                ),
                "d1_alignment": (
                    "selected-side labels, including an optional FSM committed-"
                    "side hint, are shifted by one control step before comparison "
                    "with applied motion"
                ),
            },
            "thresholds": {
                "dt_s": self.dt_s,
                "stop_speed_mps": self.stop_speed_mps,
                "launch_speed_mps": self.launch_speed_mps,
                "reverse_speed_mps": self.reverse_speed_mps,
                "side_deadband_m": self.side_deadband_m,
                "passage_progress_m": self.passage_progress_m,
                "interaction_distance_m": self.interaction_distance_m,
                "recent_vacated_window_s": self.recent_vacated_steps * self.dt_s,
                "u_turn_threshold_deg": math.degrees(self.u_turn_threshold_rad),
                "full_rotation_threshold_deg": math.degrees(
                    self.loop_rotation_threshold_rad
                ),
            },
            "frames": self._frames,
            "interaction_frames": interaction_frames,
            "wait_frame_fraction_during_interaction": (
                wait_frames / max(interaction_frames, 1)
            ),
            "reverse_frame_fraction": reverse_frames / max(self._frames, 1),
            "safe_gap": {
                "opportunities": self._safe_gap_opportunities,
                "launches": self._safe_gap_launches,
                "missed_before_reclosure": self._safe_gap_missed,
                "launch_fraction": (
                    self._safe_gap_launches
                    / max(self._safe_gap_opportunities, 1)
                ),
                "launch_delay_s": _distribution(self._launch_delays_s),
            },
            "vacated_side_choice": {
                "eligible_launches": eligible,
                "matches": self._vacated_side_matches,
                "center_or_uncommitted": self._vacated_side_center,
                "match_fraction": self._vacated_side_matches / max(eligible, 1),
            },
            "completed_episodes": self._episodes,
            "episode_wait_s": _distribution(self._episode_wait_s),
            "episode_reverse_distance_m": _distribution(
                self._episode_reverse_distance_m
            ),
            "episode_stop_to_go_transitions": _distribution(
                self._episode_stop_go
            ),
            "episode_go_to_stop_transitions": _distribution(
                self._episode_go_stop
            ),
            "episode_side_switches_after_launch": _distribution(
                self._episode_side_switches
            ),
            "heading": {
                "u_turn_episodes": self._u_turn_episodes,
                "u_turn_episode_fraction": (
                    self._u_turn_episodes / max(self._episodes, 1)
                ),
                "full_rotation_episodes": self._full_rotation_episodes,
                "full_rotation_episode_fraction": (
                    self._full_rotation_episodes / max(self._episodes, 1)
                ),
                "episode_max_heading_excursion_deg": _distribution(
                    self._episode_max_heading_deg
                ),
                "episode_cumulative_abs_rotation_deg": _distribution(
                    self._episode_abs_rotation_deg
                ),
            },
        }


__all__ = ["TeacherInteractionMetrics"]
