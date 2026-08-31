"""Stateful WAIT -> COMMIT_SIDE -> PASS controller for teacher diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch


WAIT = 0
COMMIT_SIDE = 1
PASS = 2
STATE_NAMES = {WAIT: "WAIT", COMMIT_SIDE: "COMMIT_SIDE", PASS: "PASS"}

# The seven independent gates behind a passage candidate on one side:
# static/dynamic obstacle clearance, wall clearance, and the four kinematic
# sub-filters (launch speed, forward progress, heading, side signal). Order
# fixes the bit assignment used by the pairwise-relaxation bitmask below.
FUNNEL_MASK_NAMES = (
    "static_obstacle",
    "dynamic_obstacle",
    "wall",
    "launch",
    "progress",
    "heading",
    "side_signal",
)
FUNNEL_MASK_BITS = {name: 1 << i for i, name in enumerate(FUNNEL_MASK_NAMES)}


@dataclass(frozen=True)
class StatefulTeacherSpec:
    interaction_distance_m: float = 3.0
    release_distance_m: float = 3.5
    launch_speed_mps: float = 0.15
    passage_progress_m: float = 0.20
    side_deadband_m: float = 0.15
    side_confirm_steps: int = 2
    pass_progress_m: float = 0.40
    release_confirm_steps: int = 3
    recent_vacated_window_s: float = 3.0
    max_reverse_speed_mps: float = 0.10
    max_reverse_distance_m: float = 0.30
    max_passage_heading_deviation_rad: float = math.pi / 2.0
    low_speed_reachable_fraction: float = 0.75
    low_speed_side_fraction: float = 0.50
    minimum_side_signal_m: float = 0.01
    crossing_lateral_speed_mps: float = 0.05
    crossing_lateral_span_m: float = 0.10
    # How a side is registered as vacated.
    #   "crossing" — the pedestrian must cross the robot's lateral reference
    #                line. Historical behaviour; keep as the default so frozen
    #                screens stay reproducible.
    #   "onset"    — the pedestrian's lateral direction has held for
    #                ``vacated_onset_steps``. Fires as soon as it walks away
    #                rather than after it has walked all the way across.
    vacated_trigger: str = "crossing"
    vacated_onset_steps: int = 2


def _validate_spec(spec: StatefulTeacherSpec, dt_s: float) -> None:
    values = {
        "dt_s": dt_s,
        "interaction_distance_m": spec.interaction_distance_m,
        "release_distance_m": spec.release_distance_m,
        "launch_speed_mps": spec.launch_speed_mps,
        "passage_progress_m": spec.passage_progress_m,
        "side_deadband_m": spec.side_deadband_m,
        "pass_progress_m": spec.pass_progress_m,
        "recent_vacated_window_s": spec.recent_vacated_window_s,
        "max_reverse_speed_mps": spec.max_reverse_speed_mps,
        "max_reverse_distance_m": spec.max_reverse_distance_m,
        "max_passage_heading_deviation_rad": (
            spec.max_passage_heading_deviation_rad
        ),
        "minimum_side_signal_m": spec.minimum_side_signal_m,
        "crossing_lateral_speed_mps": spec.crossing_lateral_speed_mps,
        "crossing_lateral_span_m": spec.crossing_lateral_span_m,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if spec.release_distance_m <= spec.interaction_distance_m:
        raise ValueError("release distance must exceed interaction distance")
    if spec.max_passage_heading_deviation_rad > math.pi:
        raise ValueError("passage heading deviation cannot exceed pi")
    if not 0.0 < spec.low_speed_reachable_fraction <= 1.0:
        raise ValueError("low-speed reachable fraction must be in (0,1]")
    if not 0.0 < spec.low_speed_side_fraction <= 1.0:
        raise ValueError("low-speed side fraction must be in (0,1]")
    if spec.minimum_side_signal_m >= spec.side_deadband_m:
        raise ValueError("minimum side signal must be below side deadband")
    if spec.side_confirm_steps < 1 or spec.release_confirm_steps < 1:
        raise ValueError("FSM confirmation steps must be positive")
    if spec.vacated_trigger not in ("crossing", "onset"):
        raise ValueError(
            "vacated_trigger must be 'crossing' or 'onset', got "
            f"{spec.vacated_trigger!r}"
        )
    if spec.vacated_onset_steps < 1:
        raise ValueError("vacated onset steps must be positive")


def _masked_argmin(
    cost: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ranked = torch.where(mask, cost, torch.full_like(cost, float("inf")))
    flat = ranked.flatten(1)
    index = flat.argmin(dim=1)
    any_valid = mask.flatten(1).any(dim=1)
    bins = int(cost.shape[2])
    actions = torch.stack(
        [
            torch.div(index, bins, rounding_mode="floor"),
            torch.remainder(index, bins),
        ],
        dim=1,
    )
    return actions, any_valid


class StatefulCorridorTeacher:
    """Add temporal commitment and bounded waiting to the geometric teacher."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        dt_s: float,
        spec: StatefulTeacherSpec = StatefulTeacherSpec(),
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        _validate_spec(spec, dt_s)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt_s = float(dt_s)
        self.spec = spec
        e = self.num_envs
        self.state = torch.full(
            (e,), WAIT, dtype=torch.int8, device=self.device
        )
        self.committed_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._proposed_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._proposal_run = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._release_run = torch.zeros(
            e, dtype=torch.long, device=self.device
        )
        self._initial_xy = torch.zeros(e, 2, device=self.device)
        self._forward = torch.zeros(e, 2, device=self.device)
        self._lateral = torch.zeros(e, 2, device=self.device)
        self._commit_xy = torch.zeros(e, 2, device=self.device)
        self._reverse_distance = torch.zeros(e, device=self.device)
        self._step = torch.zeros(e, dtype=torch.long, device=self.device)
        self._recent_vacated_side = torch.zeros(
            e, dtype=torch.int8, device=self.device
        )
        self._recent_vacated_step = torch.full(
            (e,), -10_000, dtype=torch.long, device=self.device
        )
        self._previous_dynamic_lateral: torch.Tensor | None = None
        self._previous_dynamic_valid: torch.Tensor | None = None
        self._previous_dynamic_lateral_velocity: torch.Tensor | None = None
        self._lateral_run: torch.Tensor | None = None

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
        norm = direction.norm(dim=1, keepdim=True)
        yaw_forward = torch.stack(
            [torch.cos(robot_yaw_rad[env_ids]), torch.sin(robot_yaw_rad[env_ids])],
            dim=1,
        )
        forward = torch.where(
            norm > 1e-6, direction / norm.clamp_min(1e-6), yaw_forward
        )
        self._initial_xy[env_ids] = robot_xy_m[env_ids]
        self._forward[env_ids] = forward
        self._lateral[env_ids, 0] = -forward[:, 1]
        self._lateral[env_ids, 1] = forward[:, 0]
        self.state[env_ids] = WAIT
        self.committed_side[env_ids] = 0
        self._proposed_side[env_ids] = 0
        self._proposal_run[env_ids] = 0
        self._release_run[env_ids] = 0
        self._commit_xy[env_ids] = robot_xy_m[env_ids]
        self._reverse_distance[env_ids] = 0.0
        self._step[env_ids] = 0
        self._recent_vacated_side[env_ids] = 0
        self._recent_vacated_step[env_ids] = -10_000

    def _update_vacated_side(
        self,
        *,
        reset: torch.Tensor,
        dynamic_positions_m: torch.Tensor,
        dynamic_velocities_mps: torch.Tensor,
        dynamic_valid: torch.Tensor,
    ) -> None:
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
            self._previous_dynamic_lateral_velocity = lateral_velocity.clone()
            self._lateral_run = torch.zeros_like(
                lateral_velocity, dtype=torch.long
            )
        if self._previous_dynamic_lateral.shape != lateral_position.shape:
            raise ValueError("dynamic slot count changed during FSM rollout")
        self._previous_dynamic_lateral[reset] = lateral_position[reset]
        self._previous_dynamic_valid[reset] = dynamic_valid[reset]
        self._previous_dynamic_lateral_velocity[reset] = (
            lateral_velocity[reset]
        )
        self._lateral_run[reset] = 0

        moving = (
            lateral_velocity.abs() >= self.spec.crossing_lateral_speed_mps
        )
        self._lateral_run = torch.where(
            moving
            & (
                torch.sign(lateral_velocity)
                == torch.sign(self._previous_dynamic_lateral_velocity)
            ),
            self._lateral_run + 1,
            torch.where(
                moving,
                torch.ones_like(self._lateral_run),
                torch.zeros_like(self._lateral_run),
            ),
        )

        if self.spec.vacated_trigger == "onset":
            # The side a pedestrian is walking AWAY from is free as soon as it
            # commits to a direction; waiting for it to reach the robot's
            # lateral line throws away the first half of the gap.
            crossed = (
                dynamic_valid
                & self._previous_dynamic_valid
                & (self._lateral_run >= self.spec.vacated_onset_steps)
            )
            source_side = (-torch.sign(lateral_velocity)).to(torch.int8)
        else:
            crossed = (
                dynamic_valid
                & self._previous_dynamic_valid
                & (lateral_velocity.abs() >= 0.05)
                & (self._previous_dynamic_lateral.abs() >= 0.02)
                & (self._previous_dynamic_lateral * lateral_position <= 0.0)
            )
            source_side = torch.sign(
                self._previous_dynamic_lateral
            ).to(torch.int8)
        positive = (crossed & (source_side > 0)).any(dim=1)
        negative = (crossed & (source_side < 0)).any(dim=1)
        unambiguous = positive ^ negative
        side = torch.where(
            positive,
            torch.ones(self.num_envs, dtype=torch.int8, device=self.device),
            -torch.ones(self.num_envs, dtype=torch.int8, device=self.device),
        )
        self._recent_vacated_side[unambiguous] = side[unambiguous]
        self._recent_vacated_step[unambiguous] = self._step[unambiguous]
        self._recent_vacated_side[positive & negative] = 0
        self._previous_dynamic_lateral[:] = lateral_position
        self._previous_dynamic_valid[:] = dynamic_valid
        self._previous_dynamic_lateral_velocity[:] = lateral_velocity

    def select(
        self,
        *,
        teacher_result: dict[str, torch.Tensor],
        just_reset: torch.Tensor,
        robot_xy_m: torch.Tensor,
        robot_yaw_rad: torch.Tensor,
        goal_xy_m: torch.Tensor,
        applied_command: torch.Tensor,
        dynamic_positions_m: torch.Tensor,
        dynamic_velocities_mps: torch.Tensor,
        dynamic_valid: torch.Tensor,
        dynamic_future_paths_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Choose stateful actions from the already evaluated 19x19 grid."""

        e = self.num_envs
        if just_reset.shape != (e,):
            raise ValueError("just_reset must have shape [E]")
        if robot_xy_m.shape != (e, 2) or goal_xy_m.shape != (e, 2):
            raise ValueError("robot and goal positions must have shape [E,2]")
        if applied_command.shape != (e, 2):
            raise ValueError("applied command must have shape [E,2]")
        if (
            dynamic_positions_m.shape != dynamic_velocities_mps.shape
            or dynamic_positions_m.ndim != 3
            or dynamic_positions_m.shape[0] != e
            or dynamic_positions_m.shape[-1] != 2
        ):
            raise ValueError("dynamic states must have shape [E,N,2]")
        if dynamic_valid.shape != dynamic_positions_m.shape[:2]:
            raise ValueError("dynamic valid must have shape [E,N]")
        if dynamic_future_paths_m is not None and (
            dynamic_future_paths_m.ndim != 4
            or dynamic_future_paths_m.shape[:2]
            != dynamic_positions_m.shape[:2]
            or dynamic_future_paths_m.shape[-1] != 2
        ):
            raise ValueError("dynamic future paths must have shape [E,N,T,2]")

        reset = just_reset.to(device=self.device, dtype=torch.bool)
        self._reset_rows(
            reset.nonzero(as_tuple=False).reshape(-1),
            robot_xy_m,
            robot_yaw_rad,
            goal_xy_m,
        )
        self._update_vacated_side(
            reset=reset,
            dynamic_positions_m=dynamic_positions_m,
            dynamic_velocities_mps=dynamic_velocities_mps,
            dynamic_valid=dynamic_valid,
        )
        self._reverse_distance += (
            -applied_command[:, 0]
        ).clamp_min(0.0) * self.dt_s

        joint = teacher_result["joint_feasible_grid"]
        linear = teacher_result["linear_velocity_grid"]
        angular = teacher_result["angular_velocity_grid"]
        endpoint = teacher_result["endpoint_grid"]
        endpoint_yaw = teacher_result["endpoint_yaw_grid"]
        raw_cost = teacher_result["raw_cost_grid"]
        obstacle_collision = teacher_result["obstacle_collision_grid"]
        wall_collision = teacher_result["wall_collision_grid"]
        if "static_obstacle_collision_grid" not in teacher_result or (
            "dynamic_obstacle_collision_grid" not in teacher_result
        ):
            raise ValueError(
                "teacher_result must include static_obstacle_collision_grid "
                "and dynamic_obstacle_collision_grid (diagnostic decomposition "
                "of obstacle_collision_grid)"
            )
        static_obstacle_collision = teacher_result[
            "static_obstacle_collision_grid"
        ]
        dynamic_obstacle_collision = teacher_result[
            "dynamic_obstacle_collision_grid"
        ]
        if joint.shape != linear.shape or angular.shape != linear.shape:
            raise ValueError("teacher action grids must share [E,L,A]")
        if (
            obstacle_collision.shape != joint.shape
            or wall_collision.shape != joint.shape
            or static_obstacle_collision.shape != joint.shape
            or dynamic_obstacle_collision.shape != joint.shape
        ):
            raise ValueError("collision grids must share [E,L,A] with joint")
        if endpoint.shape != (*joint.shape, 2):
            raise ValueError("teacher endpoint grid must have shape [E,L,A,2]")
        if endpoint_yaw.shape != joint.shape:
            raise ValueError("teacher endpoint yaw grid must have shape [E,L,A]")

        relative_dynamic = dynamic_positions_m - robot_xy_m[:, None, :]
        distance = relative_dynamic.norm(dim=-1)
        ahead = (
            relative_dynamic * self._forward[:, None, :]
        ).sum(dim=-1) > -0.5
        interaction = (
            dynamic_valid
            & ahead
            & (distance <= self.spec.interaction_distance_m)
        ).any(dim=1)
        release_clear = ~(
            dynamic_valid
            & ahead
            & (distance <= self.spec.release_distance_m)
        ).any(dim=1)

        endpoint_delta = endpoint - robot_xy_m[:, None, None, :]
        lateral_offset = (
            endpoint_delta * self._lateral[:, None, None, :]
        ).sum(dim=-1)
        forward_progress = (
            endpoint_delta * self._forward[:, None, None, :]
        ).sum(dim=-1)
        corridor_heading = torch.atan2(self._forward[:, 1], self._forward[:, 0])
        heading_deviation = torch.atan2(
            torch.sin(endpoint_yaw - corridor_heading[:, None, None]),
            torch.cos(endpoint_yaw - corridor_heading[:, None, None]),
        ).abs()
        heading_allowed = (
            heading_deviation
            <= self.spec.max_passage_heading_deviation_rad
        )
        reachable_linear = torch.where(
            heading_allowed,
            linear.clamp_min(0.0),
            torch.zeros_like(linear),
        ).amax(dim=(1, 2))
        reachable_progress = torch.where(
            heading_allowed,
            forward_progress.clamp_min(0.0),
            torch.zeros_like(forward_progress),
        ).amax(dim=(1, 2))
        reachable_lateral = torch.where(
            heading_allowed,
            lateral_offset.abs(),
            torch.zeros_like(lateral_offset),
        ).amax(dim=(1, 2))
        launch_threshold = torch.minimum(
            torch.full_like(reachable_linear, self.spec.launch_speed_mps),
            self.spec.low_speed_reachable_fraction * reachable_linear,
        )
        progress_threshold = torch.minimum(
            torch.full_like(
                reachable_progress, self.spec.passage_progress_m
            ),
            self.spec.low_speed_reachable_fraction * reachable_progress,
        )
        side_threshold = torch.minimum(
            torch.full_like(reachable_lateral, self.spec.side_deadband_m),
            self.spec.low_speed_side_fraction * reachable_lateral,
        )
        reachable_side_signal = (
            reachable_lateral >= self.spec.minimum_side_signal_m
        )
        # Split out the non-geometric half of the passage test so a diagnostic
        # can attribute a lost side to the kinematic filter, the obstacle mask
        # or the wall mask. ``passage`` is unchanged: & is associative.
        kinematic = (
            (linear >= launch_threshold[:, None, None])
            & (forward_progress >= progress_threshold[:, None, None])
            & heading_allowed
            & reachable_side_signal[:, None, None]
        )
        passage = joint & kinematic
        side_left = lateral_offset >= side_threshold[:, None, None]
        side_right = lateral_offset <= -side_threshold[:, None, None]
        left = passage & side_left
        right = passage & side_right
        left_actions, left_valid = _masked_argmin(raw_cost, left)
        right_actions, right_valid = _masked_argmin(raw_cost, right)
        left_cost = torch.where(
            left_valid,
            torch.where(left, raw_cost, torch.full_like(raw_cost, float("inf")))
            .flatten(1)
            .amin(dim=1),
            torch.full((e,), float("inf"), device=self.device),
        )
        right_cost = torch.where(
            right_valid,
            torch.where(right, raw_cost, torch.full_like(raw_cost, float("inf")))
            .flatten(1)
            .amin(dim=1),
            torch.full((e,), float("inf"), device=self.device),
        )

        recent = (
            (self._step - self._recent_vacated_step)
            <= int(math.ceil(self.spec.recent_vacated_window_s / self.dt_s))
        ) & (self._recent_vacated_side != 0)
        recent_valid = recent & torch.where(
            self._recent_vacated_side > 0, left_valid, right_valid
        )
        lower_cost_side = torch.where(
            left_cost <= right_cost,
            torch.ones(e, dtype=torch.int8, device=self.device),
            -torch.ones(e, dtype=torch.int8, device=self.device),
        )
        lower_cost_side = torch.where(
            left_valid | right_valid,
            lower_cost_side,
            torch.zeros_like(lower_cost_side),
        )
        lower_cost_side = torch.where(
            left_valid & ~right_valid,
            torch.ones_like(lower_cost_side),
            lower_cost_side,
        )
        lower_cost_side = torch.where(
            right_valid & ~left_valid,
            -torch.ones_like(lower_cost_side),
            lower_cost_side,
        )
        lateral_velocity = (
            dynamic_velocities_mps * self._lateral[:, None, :]
        ).sum(dim=-1)
        predicted_lateral_motion = torch.zeros_like(dynamic_valid)
        if dynamic_future_paths_m is not None:
            future_lateral = (
                (
                    dynamic_future_paths_m
                    - self._initial_xy[:, None, None, :]
                )
                * self._lateral[:, None, None, :]
            ).sum(dim=-1)
            predicted_lateral_motion = (
                future_lateral.amax(dim=-1)
                - future_lateral.amin(dim=-1)
            ) >= self.spec.crossing_lateral_span_m
        crossing_threat = (
            dynamic_valid
            & ahead
            & (distance <= self.spec.release_distance_m)
            & (
                (lateral_velocity.abs() >= self.spec.crossing_lateral_speed_mps)
                | predicted_lateral_motion
            )
        ).any(dim=1)
        proposal = torch.where(
            crossing_threat,
            torch.where(
                recent_valid,
                self._recent_vacated_side,
                torch.zeros_like(lower_cost_side),
            ),
            torch.where(
                recent_valid, self._recent_vacated_side, lower_cost_side
            ),
        )
        proposed_still_valid = torch.where(
            self._proposed_side > 0,
            left_valid,
            right_valid,
        )
        recent_requests_other_side = (
            recent_valid
            & (self._recent_vacated_side != self._proposed_side)
        )
        keep_proposal = (
            (self.state == WAIT)
            & interaction
            & (self._proposed_side != 0)
            & proposed_still_valid
            & ~recent_requests_other_side
        )
        proposal = torch.where(
            keep_proposal, self._proposed_side, proposal
        )
        proposal_active = (self.state == WAIT) & interaction & (proposal != 0)
        same_proposal = proposal_active & (proposal == self._proposed_side)
        self._proposal_run = torch.where(
            same_proposal,
            self._proposal_run + 1,
            torch.where(
                proposal_active,
                torch.ones_like(self._proposal_run),
                torch.zeros_like(self._proposal_run),
            ),
        )
        self._proposed_side[:] = torch.where(
            proposal_active, proposal, torch.zeros_like(proposal)
        )

        entering_commit = (
            (self.state == WAIT)
            & interaction
            & (proposal != 0)
            & (self._proposal_run >= self.spec.side_confirm_steps)
        )
        self.state[entering_commit] = COMMIT_SIDE
        self.committed_side[entering_commit] = proposal[entering_commit]
        self._commit_xy[entering_commit] = robot_xy_m[entering_commit]

        commit_progress = (
            (robot_xy_m - self._commit_xy) * self._forward
        ).sum(dim=1)
        entering_pass = (
            (self.state == COMMIT_SIDE)
            & (commit_progress >= self.spec.pass_progress_m)
        )
        self.state[entering_pass] = PASS

        self._release_run = torch.where(
            (self.state == PASS) & release_clear,
            self._release_run + 1,
            torch.zeros_like(self._release_run),
        )
        released = (
            (self.state == PASS)
            & (self._release_run >= self.spec.release_confirm_steps)
        )
        self.state[released] = WAIT
        self.committed_side[released] = 0
        self._proposed_side[released] = 0
        self._proposal_run[released] = 0
        self._reverse_distance[released] = 0.0

        base_actions = teacher_result["actions"].clone()
        actions = base_actions.clone()
        selected_geometric_feasible = teacher_result["any_feasible"].clone()
        override_mask = teacher_result["any_feasible"].clone()

        # WAIT minimizes speed and turn. It prefers non-negative braking; a
        # bounded reverse is considered only when every non-negative candidate
        # is geometrically infeasible.
        wait_cost = 10.0 * linear.abs() + 2.0 * angular.abs()
        wait_nonnegative = joint & (linear >= -1e-4)
        wait_actions, wait_valid = _masked_argmin(wait_cost, wait_nonnegative)
        reverse_allowed = (
            self._reverse_distance < self.spec.max_reverse_distance_m
        )[:, None, None]
        wait_reverse = (
            joint
            & (linear < -1e-4)
            & (linear >= -self.spec.max_reverse_speed_mps)
            & reverse_allowed
        )
        reverse_actions, reverse_valid = _masked_argmin(wait_cost, wait_reverse)
        wait_choice = torch.where(
            wait_valid[:, None], wait_actions, reverse_actions
        )
        wait_choice_valid = wait_valid | reverse_valid
        # If no geometrically feasible candidate exists, command the decoded
        # velocity closest to zero without crossing into reverse. If the robot
        # is already reversing and cannot reach zero in one step, choose the
        # largest next velocity. This is an emergency stop command, not a claim
        # that the predicted path is collision-free.
        emergency_cost = linear.abs() + 0.25 * angular.abs()
        emergency_nonnegative = linear >= -1e-4
        emergency_actions, emergency_nonnegative_valid = _masked_argmin(
            emergency_cost, emergency_nonnegative
        )
        reverse_recovery_cost = -linear + 0.25 * angular.abs()
        reverse_recovery_actions, _ = _masked_argmin(
            reverse_recovery_cost,
            torch.ones_like(joint, dtype=torch.bool),
        )
        emergency_actions = torch.where(
            emergency_nonnegative_valid[:, None],
            emergency_actions,
            reverse_recovery_actions,
        )
        wait_choice = torch.where(
            wait_choice_valid[:, None], wait_choice, emergency_actions
        )

        active_wait = interaction & (self.state == WAIT)
        actions[active_wait] = wait_choice[active_wait]
        selected_geometric_feasible[active_wait] = wait_choice_valid[active_wait]
        override_mask[active_wait] = True

        committed = (self.state == COMMIT_SIDE) | (self.state == PASS)
        committed_actions = torch.where(
            (self.committed_side > 0)[:, None], left_actions, right_actions
        )
        committed_valid = torch.where(
            self.committed_side > 0, left_valid, right_valid
        )
        actions[committed] = torch.where(
            committed_valid[committed][:, None],
            committed_actions[committed],
            wait_choice[committed],
        )
        selected_geometric_feasible[committed] = torch.where(
            committed_valid[committed],
            torch.ones_like(committed_valid[committed]),
            wait_choice_valid[committed],
        )
        override_mask[committed] = True

        used_wait = active_wait | (committed & ~committed_valid)
        used_reverse = used_wait & ~wait_valid & reverse_valid
        emergency_brake = used_wait & ~wait_choice_valid

        # Candidate funnel, second layer: decompose obstacle-collision into
        # static/dynamic and kinematic into launch/progress/heading/side-
        # signal, so a lost side can be attributed to exactly one of the
        # seven independent gates. ``passage``/``left``/``right`` above are
        # untouched -- this block only reads their inputs to build
        # diagnostic-only counts; it changes no action, state, or threshold.
        def _count(mask: torch.Tensor) -> torch.Tensor:
            return mask.flatten(1).sum(dim=1)

        ok_by_name = {
            "static_obstacle": ~static_obstacle_collision,
            "dynamic_obstacle": ~dynamic_obstacle_collision,
            "wall": ~wall_collision,
            "launch": linear >= launch_threshold[:, None, None],
            "progress": forward_progress >= progress_threshold[:, None, None],
            "heading": heading_allowed,
            "side_signal": reachable_side_signal[:, None, None].expand_as(joint),
        }

        def _all_except(exclude: frozenset[str]) -> torch.Tensor:
            combined = None
            for name in FUNNEL_MASK_NAMES:
                if name in exclude:
                    continue
                mask = ok_by_name[name]
                combined = mask if combined is None else (combined & mask)
            return combined

        without = {}
        for name in FUNNEL_MASK_NAMES:
            relaxed = _all_except(frozenset((name,)))
            without[f"n_without_{name}_left"] = _count(relaxed & side_left)
            without[f"n_without_{name}_right"] = _count(relaxed & side_right)

        pair_left_counts = []
        pair_right_counts = []
        pair_bitmasks = []
        for a, b in itertools.combinations(FUNNEL_MASK_NAMES, 2):
            relaxed = _all_except(frozenset((a, b)))
            pair_left_counts.append(_count(relaxed & side_left))
            pair_right_counts.append(_count(relaxed & side_right))
            pair_bitmasks.append(FUNNEL_MASK_BITS[a] | FUNNEL_MASK_BITS[b])
        pair_left = torch.stack(pair_left_counts, dim=0)
        pair_right = torch.stack(pair_right_counts, dim=0)
        pair_bitmask_tensor = torch.tensor(
            pair_bitmasks, device=self.device, dtype=torch.int32
        )
        best_left_count, best_left_idx = pair_left.max(dim=0)
        best_right_count, best_right_idx = pair_right.max(dim=0)
        pairwise_recovering_count_left = (pair_left > 0).sum(dim=0)
        pairwise_recovering_count_right = (pair_right > 0).sum(dim=0)
        pairwise_best_bitmask_left = torch.where(
            best_left_count > 0,
            pair_bitmask_tensor[best_left_idx],
            torch.zeros_like(pair_bitmask_tensor[best_left_idx]),
        )
        pairwise_best_bitmask_right = torch.where(
            best_right_count > 0,
            pair_bitmask_tensor[best_right_idx],
            torch.zeros_like(pair_bitmask_tensor[best_right_idx]),
        )

        funnel = {
            "n_raw_left": _count(side_left),
            "n_raw_right": _count(side_right),
            "n_final_left": _count(left),
            "n_final_right": _count(right),
            "reachable_linear_mps": reachable_linear,
            "reachable_progress_m": reachable_progress,
            "reachable_lateral_m": reachable_lateral,
            "launch_threshold_mps": launch_threshold,
            "progress_threshold_m": progress_threshold,
            "side_threshold_m": side_threshold,
            **without,
            "pairwise_recovering_count_left": pairwise_recovering_count_left,
            "pairwise_recovering_count_right": pairwise_recovering_count_right,
            "pairwise_best_count_left": best_left_count,
            "pairwise_best_count_right": best_right_count,
            "pairwise_best_bitmask_left": pairwise_best_bitmask_left,
            "pairwise_best_bitmask_right": pairwise_best_bitmask_right,
        }
        self._step += 1
        return {
            "actions": actions,
            "override_mask": override_mask,
            "selected_geometric_feasible": selected_geometric_feasible,
            "state": self.state.clone(),
            "committed_side": self.committed_side.clone(),
            # Side feasibility as the controller itself saw it this step.
            # ``committed_valid`` mirrors the value used to pick between the
            # committed action and the wait fallback, so it is only meaningful
            # where ``committed_side != 0``; ``left_valid``/``right_valid`` are
            # unconditional and let a diagnostic separate "the committed side
            # lost its path while the other side was open" (a commitment-rule
            # failure) from "both sides were blocked" (a geometry failure).
            "committed_valid": committed_valid,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "interaction_active": interaction,
            "used_wait": used_wait,
            "used_bounded_reverse": used_reverse,
            "emergency_brake": emergency_brake,
            "entered_commit": entering_commit,
            "entered_pass": entering_pass,
            "released": released,
            **funnel,
        }


__all__ = [
    "COMMIT_SIDE",
    "FUNNEL_MASK_BITS",
    "FUNNEL_MASK_NAMES",
    "PASS",
    "STATE_NAMES",
    "WAIT",
    "StatefulCorridorTeacher",
    "StatefulTeacherSpec",
]
