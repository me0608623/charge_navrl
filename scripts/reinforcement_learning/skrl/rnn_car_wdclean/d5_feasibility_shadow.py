"""Baseline-only feasibility-frontier audit for SA4 lateral diagnosis.

The D5 shadow evaluates the frozen D4 geometry model without changing policy
actions. It records when dynamic, static, and joint candidate sets disappear
before dynamic collisions and successful closest-approach controls. The audit
uses privileged simulator state and is diagnostic evidence only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch

from rnn_car_wdclean.d4_geometry_selector import (
    GeometrySelectorSpec,
    delayed_unicycle_paths,
    geometry_feasible_action_grid,
    geometry_selector_protocol,
)


MODE = "feasibility_shadow"
PROTOCOL_SCHEMA = "sa4_d5_feasibility_shadow/v1"
REPORT_SCHEMA = "sa4_d5_feasibility_frontier/v1"
ACTION_COUNT = 19 * 19


@dataclass(frozen=True)
class FeasibilityShadowSpec:
    """Frozen observation and event contract for D5."""

    control_dt_s: float = 0.2
    lead_steps: int = 25
    near_distance_m: float = 3.0
    near_exit_distance_m: float = 3.25
    contact_distance_m: float = 0.70
    linear_conflict_clearance_m: float = 0.10
    invalid_time_s: float = 999.0


def _validate_spec(
    spec: FeasibilityShadowSpec,
    geometry_spec: GeometrySelectorSpec,
) -> None:
    if abs(spec.control_dt_s - geometry_spec.control_dt_s) > 1.0e-9:
        raise ValueError("D5 and D4 control dt must match")
    if spec.lead_steps < 1:
        raise ValueError("lead_steps must be positive")
    if spec.near_distance_m <= spec.contact_distance_m:
        raise ValueError("near distance must exceed contact distance")
    if spec.near_exit_distance_m <= spec.near_distance_m:
        raise ValueError("near exit distance must exceed near distance")
    if spec.linear_conflict_clearance_m < 0.0:
        raise ValueError("linear conflict clearance must be non-negative")
    if spec.invalid_time_s <= geometry_spec.horizon_s:
        raise ValueError("invalid time sentinel must exceed D4 horizon")


def feasibility_shadow_protocol(
    spec: FeasibilityShadowSpec = FeasibilityShadowSpec(),
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return the content-addressed D5 preregistration contract."""

    _validate_spec(spec, geometry_spec)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "mode": MODE,
        "scope": (
            "single-checkpoint privileged baseline-only diagnostic; no action "
            "override, no training, and no deployment safety claim"
        ),
        "policy_action_contract": (
            "shadow observes the original policy tensor and must leave the "
            "same values unchanged before env.step"
        ),
        "evaluation_scope": (
            "every environment frame with at least one valid corridor dynamic "
            "obstacle; no radial-TTC trigger gates geometry evaluation"
        ),
        "geometry_protocol_sha256": geometry_selector_protocol(
            geometry_spec
        )["sha256"],
        "frontier_metrics": [
            "dynamic/static/joint feasible action counts over the frozen 19x19 grid",
            "policy action dynamic/static/joint feasibility",
            "radial TTC using the frozen 0.70m contact distance",
            "constant-relative-velocity closest-approach time and distance",
            "linear conflict time when closest approach is within 0.80m",
            "delay-aware policy-path first predicted dynamic conflict within 2.4s",
            "persistent jointly-feasible-to-zero suffix before each event",
        ],
        "event_alignment": {
            "lead_steps": spec.lead_steps,
            "lead_seconds": spec.lead_steps * spec.control_dt_s,
            "collision_event": "dynamic collision transition",
            "control_event": (
                "per-obstacle radial closing changes positive to non-positive "
                "inside 3.0m; successful controls terminate at the goal"
            ),
        },
        "interpretation_limits": [
            "no-feasible is relative to the frozen D4 model, not physical impossibility",
            "active-frame and event denominators must not be mixed",
            "linear closest approach is a circular constant-velocity proxy",
            "policy-path conflict uses 0.2s samples and the frozen 2.4s horizon",
            "trigger timing remains a hypothesis until this shadow distribution is analysed",
        ],
        "spec": asdict(spec),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return protocol


def _tensor(context: dict, key: str) -> torch.Tensor:
    value = context.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"D5 context {key!r} must be a tensor")
    return value


def _finite_or_sentinel(
    value: torch.Tensor,
    valid: torch.Tensor,
    sentinel: float,
) -> torch.Tensor:
    return torch.where(valid, value, torch.full_like(value, sentinel))


def _linear_cpa_metrics(
    positions_body_m: torch.Tensor,
    velocities_body_mps: torch.Tensor,
    robot_velocity_body_mps: torch.Tensor,
    valid: torch.Tensor,
    *,
    spec: FeasibilityShadowSpec,
) -> dict[str, torch.Tensor]:
    """Constant-relative-velocity closest-approach proxy."""

    relative_velocity = (
        velocities_body_mps - robot_velocity_body_mps[:, None, :]
    )
    speed_sq = relative_velocity.square().sum(dim=-1)
    current_distance = positions_body_m.norm(dim=-1)
    raw_time = -(
        positions_body_m * relative_velocity
    ).sum(dim=-1) / speed_sq.clamp(min=1.0e-9)
    moving_future = valid & (speed_sq > 1.0e-9) & (raw_time >= 0.0)
    stationary_conflict = (
        valid
        & (speed_sq <= 1.0e-9)
        & (
            current_distance
            <= spec.contact_distance_m + spec.linear_conflict_clearance_m
        )
    )
    candidate_time = torch.where(
        moving_future,
        raw_time,
        torch.where(
            stationary_conflict,
            torch.zeros_like(raw_time),
            torch.full_like(raw_time, float("inf")),
        ),
    )
    closest_position = (
        positions_body_m
        + relative_velocity * candidate_time.clamp(max=spec.invalid_time_s)[
            :, :, None
        ]
    )
    closest_distance = torch.where(
        torch.isfinite(candidate_time),
        closest_position.norm(dim=-1),
        torch.full_like(current_distance, float("inf")),
    )
    closest_slot = closest_distance.argmin(dim=1)
    env_index = torch.arange(
        positions_body_m.shape[0], device=positions_body_m.device
    )
    cpa_valid = torch.isfinite(closest_distance).any(dim=1)
    cpa_time = candidate_time[env_index, closest_slot]
    cpa_distance = closest_distance[env_index, closest_slot]
    conflict = (
        torch.isfinite(candidate_time)
        & (
            closest_distance
            <= spec.contact_distance_m + spec.linear_conflict_clearance_m
        )
    )
    conflict_time = torch.where(
        conflict,
        candidate_time,
        torch.full_like(candidate_time, float("inf")),
    ).amin(dim=1)
    conflict_valid = torch.isfinite(conflict_time)
    return {
        "linear_cpa_time_s": _finite_or_sentinel(
            cpa_time, cpa_valid, spec.invalid_time_s
        ),
        "linear_cpa_distance_m": _finite_or_sentinel(
            cpa_distance, cpa_valid, spec.invalid_time_s
        ),
        "linear_cpa_valid": cpa_valid,
        "linear_conflict_time_s": _finite_or_sentinel(
            conflict_time, conflict_valid, spec.invalid_time_s
        ),
        "linear_conflict_valid": conflict_valid,
    }


def _policy_path_dynamic_timeline(
    *,
    policy_actions: torch.Tensor,
    linear_velocity_grid: torch.Tensor,
    angular_velocity_grid: torch.Tensor,
    pending_command: torch.Tensor,
    dynamic_positions_body_m: torch.Tensor,
    dynamic_velocities_body_mps: torch.Tensor,
    dynamic_radii_m: torch.Tensor,
    dynamic_valid: torch.Tensor,
    geometry_spec: GeometrySelectorSpec,
    invalid_time_s: float,
) -> dict[str, torch.Tensor]:
    """First D4-clearance violation along the current policy candidate."""

    policy_index = policy_actions.round().long().clamp(
        0, geometry_spec.num_action_bins - 1
    )
    env_index = torch.arange(
        policy_actions.shape[0], device=policy_actions.device
    )
    policy_v = linear_velocity_grid[
        env_index, policy_index[:, 0], policy_index[:, 1]
    ][:, None, None]
    policy_w = angular_velocity_grid[
        env_index, policy_index[:, 0], policy_index[:, 1]
    ][:, None, None]
    path, yaw, times = delayed_unicycle_paths(
        policy_v, policy_w, pending_command, spec=geometry_spec
    )
    path = path[:, 0, 0]
    yaw = yaw[:, 0, 0]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    center = path + float(geometry_spec.robot_obb_offset_x_m) * torch.stack(
        [cos_yaw, sin_yaw], dim=-1
    )
    obstacle_path = (
        dynamic_positions_body_m[:, :, None, :]
        + dynamic_velocities_body_mps[:, :, None, :]
        * times[None, None, :, None]
    )
    delta = obstacle_path - center[:, None, :, :]
    local_x = (
        delta[..., 0] * cos_yaw[:, None, :]
        + delta[..., 1] * sin_yaw[:, None, :]
    )
    local_y = (
        -delta[..., 0] * sin_yaw[:, None, :]
        + delta[..., 1] * cos_yaw[:, None, :]
    )
    closest_x = local_x.clamp(
        -float(geometry_spec.robot_half_length_m),
        float(geometry_spec.robot_half_length_m),
    )
    closest_y = local_y.clamp(
        -float(geometry_spec.robot_half_width_m),
        float(geometry_spec.robot_half_width_m),
    )
    clearance = torch.sqrt(
        (local_x - closest_x).square() + (local_y - closest_y).square()
    ) - dynamic_radii_m[:, :, None]
    clearance = torch.where(
        dynamic_valid[:, :, None],
        clearance,
        torch.full_like(clearance, float("inf")),
    )
    timeline = clearance.amin(dim=1)
    conflict = timeline < float(geometry_spec.dynamic_surface_clearance_m)
    conflict_time = torch.where(
        conflict,
        times[None, :],
        torch.full_like(timeline, float("inf")),
    ).amin(dim=1)
    conflict_valid = torch.isfinite(conflict_time)
    return {
        "policy_path_first_conflict_s": _finite_or_sentinel(
            conflict_time, conflict_valid, invalid_time_s
        ),
        "policy_path_conflict_valid": conflict_valid,
        "policy_path_min_dynamic_clearance_m": timeline.amin(dim=1),
    }


class FeasibilityShadow:
    """Compute D4 feasibility on all valid-dynamic frames without acting."""

    mode = MODE

    def __init__(
        self,
        *,
        spec: FeasibilityShadowSpec = FeasibilityShadowSpec(),
        geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
    ) -> None:
        _validate_spec(spec, geometry_spec)
        self.spec = spec
        self.geometry_spec = geometry_spec
        self._calls = 0
        self._environment_frames = 0
        self._evaluated_frames = 0
        self._skipped_no_dynamic_frames = 0
        self._dynamic_feasible_frames = 0
        self._static_feasible_frames = 0
        self._jointly_feasible_frames = 0
        self._no_jointly_feasible_frames = 0
        self._policy_jointly_feasible_frames = 0
        self._dynamic_lidar_returns_removed = 0
        self._action_identity_checks = 0
        self._action_identity_errors = 0

    @torch.no_grad()
    def observe(self, policy_actions: torch.Tensor, context: dict) -> dict:
        if policy_actions.ndim != 2 or policy_actions.shape[1] != 2:
            raise ValueError("D5 policy actions must have shape [E,2]")
        before = policy_actions.detach().clone()
        envs = int(policy_actions.shape[0])
        device = policy_actions.device
        valid = _tensor(context, "dynamic_valid").bool()
        if valid.ndim != 2 or valid.shape[0] != envs:
            raise ValueError("D5 dynamic_valid must have shape [E,N]")
        evaluated = valid.any(dim=1)

        integer_default = torch.full(
            (envs,), -1, dtype=torch.long, device=device
        )
        bool_default = torch.zeros(envs, dtype=torch.bool, device=device)
        float_default = torch.full(
            (envs,), self.spec.invalid_time_s, dtype=torch.float32, device=device
        )
        snapshot = {
            "evaluated": evaluated,
            "valid_dynamic_count": valid.sum(dim=1).long(),
            "dynamic_feasible_count": integer_default.clone(),
            "static_feasible_count": integer_default.clone(),
            "jointly_feasible_count": integer_default.clone(),
            "policy_dynamic_feasible": bool_default.clone(),
            "policy_static_feasible": bool_default.clone(),
            "policy_jointly_feasible": bool_default.clone(),
            "nearest_distance_m": float_default.clone(),
            "nearest_closing_mps": torch.zeros_like(float_default),
            "radial_ttc_s": float_default.clone(),
            "radial_ttc_valid": bool_default.clone(),
            "linear_cpa_time_s": float_default.clone(),
            "linear_cpa_distance_m": float_default.clone(),
            "linear_cpa_valid": bool_default.clone(),
            "linear_conflict_time_s": float_default.clone(),
            "linear_conflict_valid": bool_default.clone(),
            "policy_path_first_conflict_s": float_default.clone(),
            "policy_path_conflict_valid": bool_default.clone(),
            "policy_path_min_dynamic_clearance_m": float_default.clone(),
            "dynamic_lidar_returns_removed": torch.zeros(
                envs, dtype=torch.long, device=device
            ),
        }

        distances = _tensor(context, "obstacle_distances_m").float()
        closings = _tensor(
            context, "relative_closing_speeds_mps"
        ).float()
        if distances.shape != valid.shape or closings.shape != valid.shape:
            raise ValueError("D5 distance/closing tensors must match dynamic_valid")
        masked_distance = torch.where(
            valid, distances, torch.full_like(distances, float("inf"))
        )
        nearest_distance, nearest_slot = masked_distance.min(dim=1)
        env_index = torch.arange(envs, device=device)
        nearest_closing = closings[env_index, nearest_slot]
        snapshot["nearest_distance_m"] = _finite_or_sentinel(
            nearest_distance, evaluated, self.spec.invalid_time_s
        )
        snapshot["nearest_closing_mps"] = torch.where(
            evaluated, nearest_closing, torch.zeros_like(nearest_closing)
        )
        radial_valid_grid = valid & (closings > 0.0)
        radial_ttc_grid = torch.where(
            radial_valid_grid,
            (distances - self.spec.contact_distance_m).clamp(min=0.0)
            / closings.clamp(min=1.0e-6),
            torch.full_like(distances, float("inf")),
        )
        radial_ttc = radial_ttc_grid.amin(dim=1)
        radial_valid = torch.isfinite(radial_ttc)
        snapshot["radial_ttc_s"] = _finite_or_sentinel(
            radial_ttc, radial_valid, self.spec.invalid_time_s
        )
        snapshot["radial_ttc_valid"] = radial_valid

        if bool(evaluated.any()):
            keys = (
                "current_velocity_mps",
                "current_omega_rad_s",
                "pending_command",
                "policy_lidar_clearance",
                "dynamic_positions_body_m",
                "dynamic_velocities_body_mps",
                "dynamic_radii_m",
                "robot_velocity_body_mps",
            )
            values = {key: _tensor(context, key) for key in keys}
            selected = geometry_feasible_action_grid(
                policy_actions=policy_actions[evaluated],
                current_velocity_mps=values["current_velocity_mps"][evaluated],
                current_omega_rad_s=values["current_omega_rad_s"][evaluated],
                pending_command=values["pending_command"][evaluated],
                policy_lidar_clearance=values["policy_lidar_clearance"][evaluated],
                dynamic_positions_body_m=values["dynamic_positions_body_m"][evaluated],
                dynamic_velocities_body_mps=values[
                    "dynamic_velocities_body_mps"
                ][evaluated],
                dynamic_radii_m=values["dynamic_radii_m"][evaluated],
                dynamic_valid=valid[evaluated],
                num_bins=int(context["num_bins"]),
                max_linear_velocity=float(context["max_linear_velocity"]),
                reverse_velocity_scale=float(context["reverse_velocity_scale"]),
                max_linear_accel=float(context["max_linear_accel"]),
                max_angular_velocity=float(context["max_angular_velocity"]),
                max_angular_accel=float(context["max_angular_accel"]),
                spec=self.geometry_spec,
            )
            dynamic_grid = (
                selected["dynamic_clearance_grid_m"]
                >= float(self.geometry_spec.dynamic_surface_clearance_m)
            )
            static_grid = (
                selected["static_clearance_grid_m"]
                >= float(self.geometry_spec.static_surface_clearance_m)
            )
            snapshot["dynamic_feasible_count"][evaluated] = (
                dynamic_grid.flatten(1).sum(dim=1)
            )
            snapshot["static_feasible_count"][evaluated] = (
                static_grid.flatten(1).sum(dim=1)
            )
            snapshot["jointly_feasible_count"][evaluated] = (
                selected["feasible_grid"].flatten(1).sum(dim=1)
            )
            snapshot["policy_dynamic_feasible"][evaluated] = (
                selected["policy_dynamic_clearance_m"]
                >= float(self.geometry_spec.dynamic_surface_clearance_m)
            )
            snapshot["policy_static_feasible"][evaluated] = (
                selected["policy_static_clearance_m"]
                >= float(self.geometry_spec.static_surface_clearance_m)
            )
            snapshot["policy_jointly_feasible"][evaluated] = selected[
                "policy_feasible"
            ]
            snapshot["dynamic_lidar_returns_removed"][evaluated] = selected[
                "dynamic_lidar_returns_removed"
            ].long()

            cpa = _linear_cpa_metrics(
                values["dynamic_positions_body_m"][evaluated],
                values["dynamic_velocities_body_mps"][evaluated],
                values["robot_velocity_body_mps"][evaluated],
                valid[evaluated],
                spec=self.spec,
            )
            timeline = _policy_path_dynamic_timeline(
                policy_actions=policy_actions[evaluated],
                linear_velocity_grid=selected["linear_velocity_grid"],
                angular_velocity_grid=selected["angular_velocity_grid"],
                pending_command=values["pending_command"][evaluated],
                dynamic_positions_body_m=values["dynamic_positions_body_m"][evaluated],
                dynamic_velocities_body_mps=values[
                    "dynamic_velocities_body_mps"
                ][evaluated],
                dynamic_radii_m=values["dynamic_radii_m"][evaluated],
                dynamic_valid=valid[evaluated],
                geometry_spec=self.geometry_spec,
                invalid_time_s=self.spec.invalid_time_s,
            )
            for key, value in {**cpa, **timeline}.items():
                snapshot[key][evaluated] = value

        identity_ok = torch.equal(policy_actions, before)
        self._action_identity_checks += envs
        if not identity_ok:
            self._action_identity_errors += envs
            raise RuntimeError("D5 shadow modified the policy action tensor")

        self._calls += 1
        self._environment_frames += envs
        evaluated_count = int(evaluated.sum().item())
        self._evaluated_frames += evaluated_count
        self._skipped_no_dynamic_frames += envs - evaluated_count
        if evaluated_count:
            dynamic_count = snapshot["dynamic_feasible_count"][evaluated]
            static_count = snapshot["static_feasible_count"][evaluated]
            joint_count = snapshot["jointly_feasible_count"][evaluated]
            self._dynamic_feasible_frames += int((dynamic_count > 0).sum().item())
            self._static_feasible_frames += int((static_count > 0).sum().item())
            self._jointly_feasible_frames += int((joint_count > 0).sum().item())
            self._no_jointly_feasible_frames += int((joint_count == 0).sum().item())
            self._policy_jointly_feasible_frames += int(
                snapshot["policy_jointly_feasible"][evaluated].sum().item()
            )
            self._dynamic_lidar_returns_removed += int(
                snapshot["dynamic_lidar_returns_removed"][evaluated]
                .sum()
                .item()
            )
        return snapshot

    def report(self) -> dict:
        return {
            "mode": MODE,
            "calls": self._calls,
            "environment_frames": self._environment_frames,
            "evaluated_environment_frames": self._evaluated_frames,
            "skipped_no_dynamic_frames": self._skipped_no_dynamic_frames,
            "dynamic_feasible_frames": self._dynamic_feasible_frames,
            "static_feasible_frames": self._static_feasible_frames,
            "jointly_feasible_frames": self._jointly_feasible_frames,
            "no_jointly_feasible_frames": self._no_jointly_feasible_frames,
            "policy_jointly_feasible_frames": self._policy_jointly_feasible_frames,
            "dynamic_lidar_returns_removed": self._dynamic_lidar_returns_removed,
            "action_identity_checks": self._action_identity_checks,
            "action_identity_errors": self._action_identity_errors,
            "protocol_sha256": feasibility_shadow_protocol(
                self.spec, self.geometry_spec
            )["sha256"],
        }


def build_feasibility_shadow(
    *,
    spec: FeasibilityShadowSpec = FeasibilityShadowSpec(),
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> FeasibilityShadow:
    return FeasibilityShadow(spec=spec, geometry_spec=geometry_spec)


@dataclass(frozen=True)
class StepRecord:
    step: int
    env_id: int
    episode_id: int
    policy_action_indices: tuple[int, int]
    effective_action_indices: tuple[int, int]
    evaluated: bool
    valid_dynamic_count: int
    dynamic_feasible_count: int
    static_feasible_count: int
    jointly_feasible_count: int
    policy_dynamic_feasible: bool
    policy_static_feasible: bool
    policy_jointly_feasible: bool
    nearest_distance_m: float
    nearest_closing_mps: float
    radial_ttc_s: float
    radial_ttc_valid: bool
    linear_cpa_time_s: float
    linear_cpa_distance_m: float
    linear_cpa_valid: bool
    linear_conflict_time_s: float
    linear_conflict_valid: bool
    policy_path_first_conflict_s: float
    policy_path_conflict_valid: bool
    policy_path_min_dynamic_clearance_m: float
    dynamic_obstacle_center_distances_m: tuple[float, ...]
    dynamic_obstacle_relative_closing_speeds_mps: tuple[float, ...]
    dynamic_collision: bool
    done: bool
    termination_cause: int = 0

    def as_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "policy_action_indices",
            "effective_action_indices",
            "dynamic_obstacle_center_distances_m",
            "dynamic_obstacle_relative_closing_speeds_mps",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass
class WindowBuffer:
    capacity: int
    records: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]

    def clear(self) -> None:
        self.records.clear()

    def window(self) -> list[StepRecord]:
        return list(self.records)


def feasibility_frontier(
    records: Iterable[StepRecord], *, control_dt_s: float
) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("frontier requires at least one record")
    last = len(rows) - 1
    feasible_indices = [
        index
        for index, row in enumerate(rows)
        if row.evaluated and row.jointly_feasible_count > 0
    ]
    dynamic_feasible_indices = [
        index
        for index, row in enumerate(rows)
        if row.evaluated and row.dynamic_feasible_count > 0
    ]
    evaluated_indices = [
        index for index, row in enumerate(rows) if row.evaluated
    ]
    closest_feasible = max(feasible_indices) if feasible_indices else None
    suffix_start = None
    if rows[-1].evaluated and rows[-1].jointly_feasible_count == 0:
        index = last
        while (
            index >= 0
            and rows[index].evaluated
            and rows[index].jointly_feasible_count == 0
        ):
            index -= 1
        suffix_start = index + 1
    dynamic_suffix_start = None
    if rows[-1].evaluated and rows[-1].dynamic_feasible_count == 0:
        index = last
        while (
            index >= 0
            and rows[index].evaluated
            and rows[index].dynamic_feasible_count == 0
        ):
            index -= 1
        dynamic_suffix_start = index + 1

    def lead(index: int | None) -> int | None:
        return None if index is None else last - index

    closest_steps = lead(closest_feasible)
    collapse_steps = lead(suffix_start)
    closest_dynamic_steps = lead(
        max(dynamic_feasible_indices) if dynamic_feasible_indices else None
    )
    dynamic_collapse_steps = lead(dynamic_suffix_start)
    return {
        "window_records": len(rows),
        "evaluated_records": len(evaluated_indices),
        "event_evaluated": rows[-1].evaluated,
        "event_jointly_feasible_count": rows[-1].jointly_feasible_count,
        "jointly_feasible_seen_in_window": bool(feasible_indices),
        "no_jointly_feasible_entire_evaluated_window": bool(
            evaluated_indices and not feasible_indices
        ),
        "closest_jointly_feasible_steps_before_event": closest_steps,
        "closest_jointly_feasible_s_before_event": (
            None
            if closest_steps is None
            else round(closest_steps * control_dt_s, 3)
        ),
        "persistent_no_feasible_onset_steps_before_event": collapse_steps,
        "persistent_no_feasible_onset_s_before_event": (
            None
            if collapse_steps is None
            else round(collapse_steps * control_dt_s, 3)
        ),
        "persistent_no_feasible_left_censored": bool(
            suffix_start == 0
            and len(rows) > 1
            and rows[0].evaluated
            and rows[0].jointly_feasible_count == 0
        ),
        "event_dynamic_feasible_count": rows[-1].dynamic_feasible_count,
        "dynamic_feasible_seen_in_window": bool(dynamic_feasible_indices),
        "no_dynamic_feasible_entire_evaluated_window": bool(
            evaluated_indices and not dynamic_feasible_indices
        ),
        "closest_dynamic_feasible_steps_before_event": closest_dynamic_steps,
        "closest_dynamic_feasible_s_before_event": (
            None
            if closest_dynamic_steps is None
            else round(closest_dynamic_steps * control_dt_s, 3)
        ),
        "persistent_no_dynamic_feasible_onset_steps_before_event": (
            dynamic_collapse_steps
        ),
        "persistent_no_dynamic_feasible_onset_s_before_event": (
            None
            if dynamic_collapse_steps is None
            else round(dynamic_collapse_steps * control_dt_s, 3)
        ),
        "persistent_no_dynamic_feasible_left_censored": bool(
            dynamic_suffix_start == 0
            and len(rows) > 1
            and rows[0].evaluated
            and rows[0].dynamic_feasible_count == 0
        ),
        "event_radial_ttc_s": (
            rows[-1].radial_ttc_s if rows[-1].radial_ttc_valid else None
        ),
        "event_linear_conflict_time_s": (
            rows[-1].linear_conflict_time_s
            if rows[-1].linear_conflict_valid
            else None
        ),
        "event_policy_path_first_conflict_s": (
            rows[-1].policy_path_first_conflict_s
            if rows[-1].policy_path_conflict_valid
            else None
        ),
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p10", "p25", "p50", "p75", "p90")}
    ordered = sorted(values)

    def nearest(fraction: float) -> float:
        index = int(round(fraction * (len(ordered) - 1)))
        return round(float(ordered[index]), 4)

    return {
        "p10": nearest(0.10),
        "p25": nearest(0.25),
        "p50": nearest(0.50),
        "p75": nearest(0.75),
        "p90": nearest(0.90),
    }


def _summarize_events(events: list[dict]) -> dict:
    collapse = [
        float(event["frontier"]["persistent_no_feasible_onset_s_before_event"])
        for event in events
        if event["frontier"]["persistent_no_feasible_onset_s_before_event"]
        is not None
    ]
    closest = [
        float(event["frontier"]["closest_jointly_feasible_s_before_event"])
        for event in events
        if event["frontier"]["closest_jointly_feasible_s_before_event"]
        is not None
    ]
    dynamic_collapse = [
        float(
            event["frontier"][
                "persistent_no_dynamic_feasible_onset_s_before_event"
            ]
        )
        for event in events
        if event["frontier"][
            "persistent_no_dynamic_feasible_onset_s_before_event"
        ]
        is not None
    ]
    dynamic_closest = [
        float(event["frontier"]["closest_dynamic_feasible_s_before_event"])
        for event in events
        if event["frontier"]["closest_dynamic_feasible_s_before_event"]
        is not None
    ]
    count = len(events)
    denominator = max(count, 1)
    return {
        "events": count,
        "event_no_feasible_fraction": sum(
            event["frontier"]["event_evaluated"]
            and event["frontier"]["event_jointly_feasible_count"] == 0
            for event in events
        )
        / denominator,
        "feasible_seen_in_window_fraction": sum(
            event["frontier"]["jointly_feasible_seen_in_window"]
            for event in events
        )
        / denominator,
        "no_feasible_entire_window_fraction": sum(
            event["frontier"]["no_jointly_feasible_entire_evaluated_window"]
            for event in events
        )
        / denominator,
        "persistent_collapse_observed_fraction": len(collapse) / denominator,
        "persistent_collapse_lead_s": _percentiles(collapse),
        "closest_feasible_lead_s": _percentiles(closest),
        "event_no_dynamic_feasible_fraction": sum(
            event["frontier"]["event_evaluated"]
            and event["frontier"]["event_dynamic_feasible_count"] == 0
            for event in events
        )
        / denominator,
        "dynamic_feasible_seen_in_window_fraction": sum(
            event["frontier"]["dynamic_feasible_seen_in_window"]
            for event in events
        )
        / denominator,
        "no_dynamic_feasible_entire_window_fraction": sum(
            event["frontier"][
                "no_dynamic_feasible_entire_evaluated_window"
            ]
            for event in events
        )
        / denominator,
        "persistent_dynamic_collapse_observed_fraction": (
            len(dynamic_collapse) / denominator
        ),
        "persistent_dynamic_collapse_lead_s": _percentiles(
            dynamic_collapse
        ),
        "closest_dynamic_feasible_lead_s": _percentiles(dynamic_closest),
    }


class FeasibilityFrontierRecorder:
    """Align D5 shadow metrics to collision and successful control events."""

    def __init__(
        self,
        num_envs: int,
        *,
        num_dynamic_obstacles: int,
        spec: FeasibilityShadowSpec = FeasibilityShadowSpec(),
        geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
    ) -> None:
        if num_envs < 1 or num_dynamic_obstacles < 1:
            raise ValueError("D5 recorder dimensions must be positive")
        _validate_spec(spec, geometry_spec)
        self.num_envs = int(num_envs)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.spec = spec
        self.geometry_spec = geometry_spec
        capacity = spec.lead_steps + 1
        self._buffers = [WindowBuffer(capacity) for _ in range(num_envs)]
        self._episode_ids = [0 for _ in range(num_envs)]
        self._encounter_phase = [
            ["idle" for _ in range(num_dynamic_obstacles)]
            for _ in range(num_envs)
        ]
        self._encounter_min_distance = [
            [float("inf") for _ in range(num_dynamic_obstacles)]
            for _ in range(num_envs)
        ]
        self._previous: list[StepRecord | None] = [None for _ in range(num_envs)]
        self._episode_event_indices = [[] for _ in range(num_envs)]
        self.events: list[dict] = []
        self.total_records = 0
        self.evaluated_records = 0
        self.no_jointly_feasible_records = 0
        self.dynamic_collision_records = 0
        self.done_records = 0
        self.termination_cause_counts: dict[int, int] = {}

    def episode_id(self, env_id: int) -> int:
        return self._episode_ids[int(env_id)]

    def _check_record(self, record: StepRecord) -> None:
        if not 0 <= record.env_id < self.num_envs:
            raise ValueError("invalid D5 env id")
        if record.episode_id != self._episode_ids[record.env_id]:
            raise ValueError("D5 episode id mismatch")
        if record.policy_action_indices != record.effective_action_indices:
            raise RuntimeError("D5 baseline action identity violated")
        counts = (
            record.dynamic_feasible_count,
            record.static_feasible_count,
            record.jointly_feasible_count,
        )
        if record.evaluated:
            if record.valid_dynamic_count < 1:
                raise RuntimeError("evaluated D5 record has no valid dynamic")
            if any(value < 0 or value > ACTION_COUNT for value in counts):
                raise RuntimeError("D5 feasible count outside 0..361")
        elif record.valid_dynamic_count != 0 or counts != (-1, -1, -1):
            raise RuntimeError("skipped D5 record has inconsistent counts")
        numeric = (
            record.nearest_distance_m,
            record.nearest_closing_mps,
            record.radial_ttc_s,
            record.linear_cpa_time_s,
            record.linear_cpa_distance_m,
            record.linear_conflict_time_s,
            record.policy_path_first_conflict_s,
            record.policy_path_min_dynamic_clearance_m,
            *record.dynamic_obstacle_center_distances_m,
            *record.dynamic_obstacle_relative_closing_speeds_mps,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("D5 record contains non-finite data")
        if len(record.dynamic_obstacle_center_distances_m) != self.num_dynamic_obstacles:
            raise ValueError("D5 obstacle distance slot count mismatch")
        if len(record.dynamic_obstacle_relative_closing_speeds_mps) != self.num_dynamic_obstacles:
            raise ValueError("D5 obstacle closing slot count mismatch")

    def _emit(
        self,
        event_type: str,
        env_id: int,
        *,
        obstacle_slot: int,
        closest_distance_m: float,
    ) -> None:
        window = self._buffers[env_id].window()
        if not window:
            raise RuntimeError("cannot emit empty D5 event window")
        if len({row.episode_id for row in window}) != 1:
            raise RuntimeError("D5 event window crossed episode boundary")
        self.events.append(
            {
                "event_type": event_type,
                "env_id": env_id,
                "episode_id": window[-1].episode_id,
                "terminal_cause": window[-1].termination_cause,
                "episode_outcome_cause": None,
                "tracked_obstacle_slot": int(obstacle_slot),
                "closest_distance_m": float(closest_distance_m),
                "frontier": feasibility_frontier(
                    window, control_dt_s=self.spec.control_dt_s
                ),
                "records": [row.as_dict() for row in window],
            }
        )
        self._episode_event_indices[env_id].append(len(self.events) - 1)

    def _record_noncollision(self, record: StepRecord) -> None:
        previous = self._previous[record.env_id]
        for slot in range(self.num_dynamic_obstacles):
            distance = record.dynamic_obstacle_center_distances_m[slot]
            closing = record.dynamic_obstacle_relative_closing_speeds_mps[slot]
            phase = self._encounter_phase[record.env_id][slot]
            if phase == "idle":
                if distance <= self.spec.near_distance_m and closing > 0.0:
                    self._encounter_phase[record.env_id][slot] = "approaching"
                    self._encounter_min_distance[record.env_id][slot] = distance
                continue
            if phase == "approaching":
                self._encounter_min_distance[record.env_id][slot] = min(
                    self._encounter_min_distance[record.env_id][slot], distance
                )
                previous_closing = (
                    previous.dynamic_obstacle_relative_closing_speeds_mps[slot]
                    if previous is not None
                    else closing
                )
                if previous_closing > 0.0 and closing <= 0.0:
                    self._emit(
                        "noncollision_closest_approach",
                        record.env_id,
                        obstacle_slot=slot,
                        closest_distance_m=self._encounter_min_distance[
                            record.env_id
                        ][slot],
                    )
                    self._encounter_phase[record.env_id][slot] = "resolved"
                continue
            if phase == "resolved" and distance >= self.spec.near_exit_distance_m:
                self._encounter_phase[record.env_id][slot] = "idle"
                self._encounter_min_distance[record.env_id][slot] = float("inf")

    def _reset_encounters(self, env_id: int) -> None:
        for slot in range(self.num_dynamic_obstacles):
            self._encounter_phase[env_id][slot] = "idle"
            self._encounter_min_distance[env_id][slot] = float("inf")

    def record_batch(self, records: Iterable[StepRecord]) -> None:
        rows = list(records)
        if len(rows) != self.num_envs:
            raise ValueError("D5 record batch must contain every environment")
        if {row.env_id for row in rows} != set(range(self.num_envs)):
            raise ValueError("D5 record batch env ids are not complete")
        for record in sorted(rows, key=lambda item: item.env_id):
            self._check_record(record)
            env_id = record.env_id
            self.total_records += 1
            self.evaluated_records += int(record.evaluated)
            self.no_jointly_feasible_records += int(
                record.evaluated and record.jointly_feasible_count == 0
            )
            self.dynamic_collision_records += int(record.dynamic_collision)
            self.done_records += int(record.done)
            if record.done:
                cause = int(record.termination_cause)
                self.termination_cause_counts[cause] = (
                    self.termination_cause_counts.get(cause, 0) + 1
                )
            self._buffers[env_id].append(record)
            if record.dynamic_collision:
                slot = min(
                    range(self.num_dynamic_obstacles),
                    key=lambda index: record.dynamic_obstacle_center_distances_m[
                        index
                    ],
                )
                self._emit(
                    "dynamic_collision",
                    env_id,
                    obstacle_slot=slot,
                    closest_distance_m=record.dynamic_obstacle_center_distances_m[
                        slot
                    ],
                )
            elif not record.done:
                self._record_noncollision(record)
            if record.done:
                for event_index in self._episode_event_indices[env_id]:
                    self.events[event_index]["episode_outcome_cause"] = int(
                        record.termination_cause
                    )
                self._episode_event_indices[env_id].clear()
                self._buffers[env_id].clear()
                self._reset_encounters(env_id)
                self._previous[env_id] = None
                self._episode_ids[env_id] += 1
            else:
                self._previous[env_id] = record

    def record_tensor_batch(
        self,
        *,
        step: int,
        policy_actions: torch.Tensor,
        effective_actions: torch.Tensor,
        snapshot: dict[str, torch.Tensor],
        obstacle_distances_m: torch.Tensor,
        obstacle_closings_mps: torch.Tensor,
        dynamic_collision: torch.Tensor,
        done: torch.Tensor,
        termination_cause: torch.Tensor,
    ) -> None:
        columns = torch.cat(
            [
                policy_actions.float(),
                effective_actions.float(),
                snapshot["evaluated"].float()[:, None],
                snapshot["valid_dynamic_count"].float()[:, None],
                snapshot["dynamic_feasible_count"].float()[:, None],
                snapshot["static_feasible_count"].float()[:, None],
                snapshot["jointly_feasible_count"].float()[:, None],
                snapshot["policy_dynamic_feasible"].float()[:, None],
                snapshot["policy_static_feasible"].float()[:, None],
                snapshot["policy_jointly_feasible"].float()[:, None],
                snapshot["nearest_distance_m"].float()[:, None],
                snapshot["nearest_closing_mps"].float()[:, None],
                snapshot["radial_ttc_s"].float()[:, None],
                snapshot["radial_ttc_valid"].float()[:, None],
                snapshot["linear_cpa_time_s"].float()[:, None],
                snapshot["linear_cpa_distance_m"].float()[:, None],
                snapshot["linear_cpa_valid"].float()[:, None],
                snapshot["linear_conflict_time_s"].float()[:, None],
                snapshot["linear_conflict_valid"].float()[:, None],
                snapshot["policy_path_first_conflict_s"].float()[:, None],
                snapshot["policy_path_conflict_valid"].float()[:, None],
                snapshot["policy_path_min_dynamic_clearance_m"].float()[:, None],
                dynamic_collision.reshape(-1).float()[:, None],
                done.reshape(-1).float()[:, None],
                termination_cause.reshape(-1).float()[:, None],
                obstacle_distances_m.float(),
                obstacle_closings_mps.float(),
            ],
            dim=1,
        ).detach().cpu().tolist()
        records = []
        slot_start = 27
        for env_id, row in enumerate(columns):
            slot_distances = tuple(
                float(value)
                for value in row[
                    slot_start : slot_start + self.num_dynamic_obstacles
                ]
            )
            closing_start = slot_start + self.num_dynamic_obstacles
            slot_closings = tuple(
                float(value)
                for value in row[
                    closing_start : closing_start + self.num_dynamic_obstacles
                ]
            )
            records.append(
                StepRecord(
                    step=int(step),
                    env_id=env_id,
                    episode_id=self.episode_id(env_id),
                    policy_action_indices=(int(round(row[0])), int(round(row[1]))),
                    effective_action_indices=(int(round(row[2])), int(round(row[3]))),
                    evaluated=bool(row[4]),
                    valid_dynamic_count=int(round(row[5])),
                    dynamic_feasible_count=int(round(row[6])),
                    static_feasible_count=int(round(row[7])),
                    jointly_feasible_count=int(round(row[8])),
                    policy_dynamic_feasible=bool(row[9]),
                    policy_static_feasible=bool(row[10]),
                    policy_jointly_feasible=bool(row[11]),
                    nearest_distance_m=float(row[12]),
                    nearest_closing_mps=float(row[13]),
                    radial_ttc_s=float(row[14]),
                    radial_ttc_valid=bool(row[15]),
                    linear_cpa_time_s=float(row[16]),
                    linear_cpa_distance_m=float(row[17]),
                    linear_cpa_valid=bool(row[18]),
                    linear_conflict_time_s=float(row[19]),
                    linear_conflict_valid=bool(row[20]),
                    policy_path_first_conflict_s=float(row[21]),
                    policy_path_conflict_valid=bool(row[22]),
                    policy_path_min_dynamic_clearance_m=float(row[23]),
                    dynamic_obstacle_center_distances_m=slot_distances,
                    dynamic_obstacle_relative_closing_speeds_mps=slot_closings,
                    dynamic_collision=bool(row[24]),
                    done=bool(row[25]),
                    termination_cause=int(round(row[26])),
                )
            )
        self.record_batch(records)

    def report(self, *, metadata: dict) -> dict:
        collision_events = [
            event for event in self.events if event["event_type"] == "dynamic_collision"
        ]
        controls = [
            event
            for event in self.events
            if event["event_type"] == "noncollision_closest_approach"
        ]
        successful_controls = [
            event for event in controls if event["episode_outcome_cause"] == 1
        ]
        unresolved_controls = [
            event for event in controls if event["episode_outcome_cause"] is None
        ]
        runtime = metadata.get("shadow_runtime") or {}
        expected_records = int(metadata.get("expected_records", self.total_records))
        expected_episodes = int(metadata.get("completed_episodes", self.done_records))
        identity_ok = all(
            event_record["policy_action_indices"]
            == event_record["effective_action_indices"]
            for event in self.events
            for event_record in event["records"]
        )
        coverage_ok = expected_records == self.total_records
        episode_ok = expected_episodes == self.done_records
        collision_ok = len(collision_events) == self.dynamic_collision_records
        runtime_ok = (
            int(runtime.get("environment_frames", -1)) == self.total_records
            and int(runtime.get("evaluated_environment_frames", -1))
            == self.evaluated_records
            and int(runtime.get("no_jointly_feasible_frames", -1))
            == self.no_jointly_feasible_records
            and int(runtime.get("action_identity_errors", -1)) == 0
        )
        d3_ok = metadata.get("d3_reconciliation_ok") is True
        return {
            "schema": REPORT_SCHEMA,
            "mode": MODE,
            "dt_s": self.spec.control_dt_s,
            "max_lead_steps": self.spec.lead_steps,
            "max_lead_s": self.spec.lead_steps * self.spec.control_dt_s,
            "protocol": feasibility_shadow_protocol(
                self.spec, self.geometry_spec
            ),
            "metadata": metadata,
            "counts": {
                "records": self.total_records,
                "evaluated_records": self.evaluated_records,
                "no_jointly_feasible_records": self.no_jointly_feasible_records,
                "completed_episodes": self.done_records,
                "dynamic_collision": self.dynamic_collision_records,
                "events": len(self.events),
                "noncollision_closest_approach": len(controls),
                "successful_noncollision_closest_approach": len(successful_controls),
                "unresolved_noncollision_closest_approach": len(unresolved_controls),
                "termination_causes": {
                    str(key): value
                    for key, value in sorted(self.termination_cause_counts.items())
                },
            },
            "frontier_summary": {
                "dynamic_collision": _summarize_events(collision_events),
                "successful_noncollision_closest_approach": _summarize_events(
                    successful_controls
                ),
            },
            "self_check": {
                "baseline_action_identity_ok": identity_ok,
                "record_coverage_ok": coverage_ok,
                "episode_reconciliation_ok": episode_ok,
                "dynamic_collision_reconciliation_ok": collision_ok,
                "shadow_runtime_reconciliation_ok": runtime_ok,
                "d3_reconciliation_ok": d3_ok,
                "reconciliation_ok": (
                    identity_ok
                    and coverage_ok
                    and episode_ok
                    and collision_ok
                    and runtime_ok
                    and d3_ok
                ),
            },
            "events": self.events,
        }

    def write(self, path: str | Path, *, metadata: dict) -> dict:
        report = self.report(metadata=metadata)
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report


__all__ = [
    "FeasibilityFrontierRecorder",
    "FeasibilityShadow",
    "FeasibilityShadowSpec",
    "MODE",
    "StepRecord",
    "build_feasibility_shadow",
    "feasibility_frontier",
    "feasibility_shadow_protocol",
]
