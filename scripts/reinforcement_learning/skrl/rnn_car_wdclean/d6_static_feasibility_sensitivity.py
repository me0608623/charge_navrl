"""Baseline-only sensitivity audit for the SA4 static-feasibility model.

The D6 shadow keeps the policy action unchanged and re-evaluates the frozen
D4 candidate trajectories under alternate LiDAR source subsets, prediction
horizons, and surface clearances. Privileged corridor geometry is used only to
attribute LiDAR returns to walls or static circular obstacles; feasibility is
still computed from the same policy-visible LiDAR surface points as D4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_wdclean.d4_geometry_selector import (
    GeometrySelectorSpec,
    _dynamic_obb_circle_clearance_grid,
    delayed_unicycle_paths,
    geometry_selector_protocol,
    lidar_static_points,
    validate_action_contract,
)
from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)
from rnn_car_wdclean.swept_arc import policy_lidar_to_sensor_range


MODE = "static_feasibility_sensitivity"
PROTOCOL_SCHEMA = "sa4_d6_static_feasibility_sensitivity/v2"
REPORT_SCHEMA = "sa4_d6_static_feasibility_report/v2"
SOURCE_NAMES = ("wall", "static_obstacle", "ambiguous", "residual")
SCOPE_NAMES = (
    "all_evaluated",
    "near_dynamic_3m",
    "frozen_static_blocked",
    "frozen_joint_blocked",
    "dynamic_collision_transition",
)


@dataclass(frozen=True)
class StaticSensitivitySpec:
    """Frozen factor grid and attribution contract for D6."""

    control_dt_s: float = 0.2
    horizons_s: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
    clearances_m: tuple[float, ...] = (
        0.0,
        0.01,
        0.02,
        0.05,
        0.08,
        0.10,
        0.15,
    )
    source_horizon_s: float = 2.4
    source_clearance_m: float = 0.10
    source_match_tolerance_m: float = 0.20
    source_ambiguity_tolerance_m: float = 0.05
    near_dynamic_distance_m: float = 3.0


def _validate_spec(
    spec: StaticSensitivitySpec,
    geometry_spec: GeometrySelectorSpec,
) -> None:
    if abs(spec.control_dt_s - geometry_spec.control_dt_s) > 1.0e-9:
        raise ValueError("D6 and D4 control dt must match")
    if not spec.horizons_s or not spec.clearances_m:
        raise ValueError("D6 factor grids must not be empty")
    if tuple(sorted(set(spec.horizons_s))) != spec.horizons_s:
        raise ValueError("D6 horizons must be unique and ascending")
    if tuple(sorted(set(spec.clearances_m))) != spec.clearances_m:
        raise ValueError("D6 clearances must be unique and ascending")
    if spec.horizons_s[-1] != geometry_spec.horizon_s:
        raise ValueError("D6 maximum horizon must equal frozen D4 horizon")
    for horizon in spec.horizons_s:
        samples = horizon / spec.control_dt_s
        if horizon <= 0.0 or abs(samples - round(samples)) > 1.0e-9:
            raise ValueError("D6 horizons must be positive control-step multiples")
    if any(clearance < 0.0 for clearance in spec.clearances_m):
        raise ValueError("D6 clearances must be non-negative")
    if spec.source_horizon_s not in spec.horizons_s:
        raise ValueError("source horizon must be in the D6 horizon grid")
    if spec.source_clearance_m not in spec.clearances_m:
        raise ValueError("source clearance must be in the D6 clearance grid")
    if spec.source_match_tolerance_m <= 0.0:
        raise ValueError("source match tolerance must be positive")
    if not 0.0 <= spec.source_ambiguity_tolerance_m <= spec.source_match_tolerance_m:
        raise ValueError("source ambiguity tolerance is invalid")
    if spec.near_dynamic_distance_m <= 0.0:
        raise ValueError("near-dynamic distance must be positive")


def _factor_key(horizon_s: float, clearance_m: float) -> str:
    return f"lidar_all_h{horizon_s:.1f}_c{clearance_m:.2f}"


def frozen_variant_key(
    spec: StaticSensitivitySpec = StaticSensitivitySpec(),
) -> str:
    return _factor_key(spec.source_horizon_s, spec.source_clearance_m)


def variant_keys(
    spec: StaticSensitivitySpec = StaticSensitivitySpec(),
) -> tuple[str, ...]:
    factor_keys = tuple(
        _factor_key(horizon, clearance)
        for horizon in spec.horizons_s
        for clearance in spec.clearances_m
    )
    source_keys = (
        "lidar_without_wall_assigned_h2.4_c0.10",
        "lidar_without_static_assigned_h2.4_c0.10",
        "lidar_without_ambiguous_h2.4_c0.10",
        "lidar_without_residual_h2.4_c0.10",
        "lidar_wall_only_h2.4_c0.10",
        "lidar_static_only_h2.4_c0.10",
        "lidar_ambiguous_only_h2.4_c0.10",
        "lidar_residual_only_h2.4_c0.10",
        "no_static_lidar_dynamic_ceiling",
    )
    return factor_keys + source_keys


def static_sensitivity_protocol(
    spec: StaticSensitivitySpec = StaticSensitivitySpec(),
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return the content-addressed D6 preregistration contract."""

    _validate_spec(spec, geometry_spec)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "mode": MODE,
        "scope": (
            "single-checkpoint privileged baseline-only shadow; policy action "
            "is never modified; diagnostic evidence only"
        ),
        "geometry_protocol_sha256": geometry_selector_protocol(
            geometry_spec
        )["sha256"],
        "factor_isolation": {
            "horizon_clearance": (
                "same D4 LiDAR points and fixed 2.4s dynamic grid; vary only "
                "the prefix of the static trajectory and static clearance"
            ),
            "source": (
                "same frozen 2.4s/0.10m model; remove one mutually exclusive "
                "privileged-attributed LiDAR point partition at a time"
            ),
            "dynamic_ceiling": (
                "ignore every static LiDAR point while preserving the frozen "
                "dynamic feasibility grid"
            ),
        },
        "source_attribution": {
            "wall": (
                "nearest active wall AABB surface from get_combined_wall_data "
                "(long-corridor side walls plus arena boundary walls; inactive "
                "maze/narrow slots remain masked)"
            ),
            "static_obstacle": "nearest active static obstacle circle surface",
            "ambiguous": "both sources match and surface distances differ by at most 0.05m",
            "residual": "valid non-dynamic LiDAR point unmatched within 0.20m",
            "partition": "mutually exclusive and exhaustive over D4 static-valid points",
            "geometry_use_limit": (
                "privileged geometry labels LiDAR returns only; it does not "
                "replace D4 LiDAR-point feasibility with an oracle planner"
            ),
        },
        "scopes": list(SCOPE_NAMES),
        "variant_keys": list(variant_keys(spec)),
        "interpretation_limits": [
            "a rescued frame diagnoses this frozen model, not physical solvability",
            "wall/static labels inherit the frozen matching tolerances and noisy LiDAR",
            "residual does not automatically mean sensor error or phantom geometry",
            "clearance=0 is a degenerate lower bound because point-to-OBB distances are non-negative",
            "source removals are diagnostic ablations and are not deployable shields",
            "single checkpoint and evaluator seed do not establish population effects",
        ],
        "spec": {
            **asdict(spec),
            "horizons_s": list(spec.horizons_s),
            "clearances_m": list(spec.clearances_m),
        },
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return protocol


def _require_tensor(context: dict, key: str) -> torch.Tensor:
    value = context.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"D6 context {key!r} must be a tensor")
    return value


def _wall_surface_distance(
    points_local_m: torch.Tensor,
    wall_centers_local_m: torch.Tensor,
    wall_sizes_m: torch.Tensor,
    wall_valid: torch.Tensor,
) -> torch.Tensor:
    """Distance from points to the nearest axis-aligned wall-box surface."""

    delta = (
        points_local_m[:, :, None, :] - wall_centers_local_m[:, None, :, :]
    ).abs()
    half = 0.5 * wall_sizes_m[:, None, :, :]
    outside_axis = (delta - half).clamp(min=0.0)
    outside = outside_axis.norm(dim=-1)
    inside = (half - delta).amin(dim=-1).clamp(min=0.0)
    surface = torch.where(
        (outside_axis > 0.0).any(dim=-1), outside, inside
    )
    surface = torch.where(
        wall_valid[:, None, :].bool(),
        surface,
        torch.full_like(surface, float("inf")),
    )
    return surface.amin(dim=-1)


def attribute_static_lidar_sources(
    *,
    points_body_m: torch.Tensor,
    static_valid: torch.Tensor,
    robot_position_local_m: torch.Tensor,
    robot_yaw_rad: torch.Tensor,
    static_positions_local_m: torch.Tensor,
    static_radii_m: torch.Tensor,
    static_obstacle_valid: torch.Tensor,
    wall_centers_local_m: torch.Tensor,
    wall_sizes_m: torch.Tensor,
    wall_valid: torch.Tensor,
    spec: StaticSensitivitySpec = StaticSensitivitySpec(),
) -> dict[str, torch.Tensor]:
    """Partition D4 static-valid LiDAR points by privileged scene source."""

    envs, beams, xy = points_body_m.shape
    if xy != 2 or static_valid.shape != (envs, beams):
        raise ValueError("D6 LiDAR points/mask have invalid shapes")
    if robot_position_local_m.shape != (envs, 2):
        raise ValueError("D6 robot local position must have shape [E,2]")
    if robot_yaw_rad.shape != (envs,):
        raise ValueError("D6 robot yaw must have shape [E]")
    if static_positions_local_m.ndim != 3 or static_positions_local_m.shape[0] != envs:
        raise ValueError("D6 static positions must have shape [E,S,2]")
    if static_positions_local_m.shape[-1] != 2:
        raise ValueError("D6 static positions must have shape [E,S,2]")
    if static_radii_m.shape != static_positions_local_m.shape[:2]:
        raise ValueError("D6 static radii shape mismatch")
    if static_obstacle_valid.shape != static_positions_local_m.shape[:2]:
        raise ValueError("D6 static validity shape mismatch")
    if wall_centers_local_m.ndim != 3 or wall_centers_local_m.shape[0] != envs:
        raise ValueError("D6 wall centers must have shape [E,W,2]")
    if wall_centers_local_m.shape[-1] != 2:
        raise ValueError("D6 wall centers must have shape [E,W,2]")
    if wall_sizes_m.shape != wall_centers_local_m.shape:
        raise ValueError("D6 wall sizes shape mismatch")
    if wall_valid.shape != wall_centers_local_m.shape[:2]:
        raise ValueError("D6 wall validity shape mismatch")

    cos_yaw = torch.cos(robot_yaw_rad)[:, None]
    sin_yaw = torch.sin(robot_yaw_rad)[:, None]
    point_local_x = (
        robot_position_local_m[:, 0, None]
        + cos_yaw * points_body_m[:, :, 0]
        - sin_yaw * points_body_m[:, :, 1]
    )
    point_local_y = (
        robot_position_local_m[:, 1, None]
        + sin_yaw * points_body_m[:, :, 0]
        + cos_yaw * points_body_m[:, :, 1]
    )
    points_local = torch.stack([point_local_x, point_local_y], dim=-1)

    wall_distance = _wall_surface_distance(
        points_local, wall_centers_local_m, wall_sizes_m, wall_valid
    )
    static_surface = (
        (
            points_local[:, :, None, :]
            - static_positions_local_m[:, None, :, :]
        ).norm(dim=-1)
        - static_radii_m[:, None, :]
    ).abs()
    static_surface = torch.where(
        static_obstacle_valid[:, None, :].bool(),
        static_surface,
        torch.full_like(static_surface, float("inf")),
    )
    static_distance = static_surface.amin(dim=-1)

    wall_match = wall_distance <= float(spec.source_match_tolerance_m)
    obstacle_match = static_distance <= float(spec.source_match_tolerance_m)
    both = wall_match & obstacle_match
    ambiguous = (
        static_valid
        & both
        & (
            (wall_distance - static_distance).abs()
            <= float(spec.source_ambiguity_tolerance_m)
        )
    )
    wall = (
        static_valid
        & wall_match
        & ~ambiguous
        & (~obstacle_match | (wall_distance < static_distance))
    )
    static_obstacle = (
        static_valid
        & obstacle_match
        & ~ambiguous
        & (~wall_match | (static_distance <= wall_distance))
    )
    residual = static_valid & ~(wall | static_obstacle | ambiguous)
    partition_count = (
        wall.long()
        + static_obstacle.long()
        + ambiguous.long()
        + residual.long()
    )
    if not torch.equal(partition_count, static_valid.long()):
        raise RuntimeError("D6 LiDAR source partition is not exhaustive/exclusive")
    return {
        "wall": wall,
        "static_obstacle": static_obstacle,
        "ambiguous": ambiguous,
        "residual": residual,
        "wall_surface_distance_m": wall_distance,
        "static_surface_distance_m": static_distance,
    }


def point_obb_clearance_timelines(
    *,
    path_m: torch.Tensor,
    yaw_rad: torch.Tensor,
    points_m: torch.Tensor,
    source_masks: dict[str, torch.Tensor],
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict[str, torch.Tensor]:
    """Minimum source-point clearance at every action and future sample."""

    envs, linear_bins, angular_bins, samples, xy = path_m.shape
    if xy != 2 or yaw_rad.shape != path_m.shape[:-1]:
        raise ValueError("D6 path/yaw shapes are invalid")
    if points_m.ndim != 3 or points_m.shape[0] != envs or points_m.shape[-1] != 2:
        raise ValueError("D6 LiDAR points must have shape [E,K,2]")
    for name, mask in source_masks.items():
        if mask.shape != points_m.shape[:2]:
            raise ValueError(f"D6 source mask {name!r} has invalid shape")

    flat_path = path_m.reshape(envs, linear_bins * angular_bins, samples, 2)
    flat_yaw = yaw_rad.reshape(envs, linear_bins * angular_bins, samples)
    outputs: dict[str, list[torch.Tensor]] = {
        name: [] for name in source_masks
    }
    for start in range(0, flat_path.shape[1], geometry_spec.action_chunk_size):
        stop = min(
            start + geometry_spec.action_chunk_size, flat_path.shape[1]
        )
        path = flat_path[:, start:stop]
        yaw = flat_yaw[:, start:stop]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        center = path + float(geometry_spec.robot_obb_offset_x_m) * torch.stack(
            [cos_yaw, sin_yaw], dim=-1
        )
        delta = points_m[:, None, None, :, :] - center[:, :, :, None, :]
        local_x = (
            delta[..., 0] * cos_yaw[:, :, :, None]
            + delta[..., 1] * sin_yaw[:, :, :, None]
        )
        local_y = (
            -delta[..., 0] * sin_yaw[:, :, :, None]
            + delta[..., 1] * cos_yaw[:, :, :, None]
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
            (local_x - closest_x).square()
            + (local_y - closest_y).square()
        )
        for name, mask in source_masks.items():
            masked = torch.where(
                mask[:, None, None, :].bool(),
                clearance,
                torch.full_like(clearance, float("inf")),
            )
            outputs[name].append(masked.amin(dim=-1))
    return {
        name: torch.cat(chunks, dim=1).reshape(
            envs, linear_bins, angular_bins, samples
        )
        for name, chunks in outputs.items()
    }


def _minimum_timelines(
    timelines: dict[str, torch.Tensor], names: tuple[str, ...]
) -> torch.Tensor:
    if not names:
        reference = next(iter(timelines.values()))
        return torch.full_like(reference, float("inf"))
    result = timelines[names[0]]
    for name in names[1:]:
        result = torch.minimum(result, timelines[name])
    return result


class StaticFeasibilitySensitivity:
    """Accumulate D6 factor sensitivities without modifying actions."""

    mode = MODE

    def __init__(
        self,
        *,
        spec: StaticSensitivitySpec = StaticSensitivitySpec(),
        geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
    ) -> None:
        _validate_spec(spec, geometry_spec)
        self.spec = spec
        self.geometry_spec = geometry_spec
        self.keys = variant_keys(spec)
        self._key_index = {name: index for index, name in enumerate(self.keys)}
        self._calls = 0
        self._transitions = 0
        self._environment_frames = 0
        self._action_identity_checks = 0
        self._action_identity_errors = 0
        self._partition_errors = 0
        self._scope_frames: torch.Tensor | None = None
        self._static_any: torch.Tensor | None = None
        self._joint_any: torch.Tensor | None = None
        self._static_action_sum: torch.Tensor | None = None
        self._joint_action_sum: torch.Tensor | None = None
        self._source_points: torch.Tensor | None = None
        self._static_valid_points: torch.Tensor | None = None
        self._dynamic_returns_removed: torch.Tensor | None = None
        self._snapshot_serial = 0
        self._last_recorded_serial = 0

    def _ensure_counters(self, device: torch.device) -> None:
        if self._scope_frames is not None:
            if self._scope_frames.device != device:
                raise RuntimeError("D6 device changed during the audit")
            return
        scopes = len(SCOPE_NAMES)
        variants = len(self.keys)
        self._scope_frames = torch.zeros(scopes, dtype=torch.long, device=device)
        shape = (scopes, variants)
        self._static_any = torch.zeros(shape, dtype=torch.long, device=device)
        self._joint_any = torch.zeros(shape, dtype=torch.long, device=device)
        self._static_action_sum = torch.zeros(shape, dtype=torch.long, device=device)
        self._joint_action_sum = torch.zeros(shape, dtype=torch.long, device=device)
        self._source_points = torch.zeros(
            len(SOURCE_NAMES), dtype=torch.long, device=device
        )
        self._static_valid_points = torch.zeros((), dtype=torch.long, device=device)
        self._dynamic_returns_removed = torch.zeros((), dtype=torch.long, device=device)

    def _accumulate_scope(
        self,
        scope_name: str,
        mask: torch.Tensor,
        snapshot: dict[str, torch.Tensor | int],
    ) -> None:
        assert self._scope_frames is not None
        assert self._static_any is not None
        assert self._joint_any is not None
        assert self._static_action_sum is not None
        assert self._joint_action_sum is not None
        index = SCOPE_NAMES.index(scope_name)
        self._scope_frames[index] += mask.long().sum()
        self._static_any[index] += snapshot["static_any"][mask].long().sum(dim=0)
        self._joint_any[index] += snapshot["joint_any"][mask].long().sum(dim=0)
        self._static_action_sum[index] += snapshot["static_count"][mask].long().sum(dim=0)
        self._joint_action_sum[index] += snapshot["joint_count"][mask].long().sum(dim=0)

    @torch.no_grad()
    def observe(self, policy_actions: torch.Tensor, context: dict) -> dict:
        if policy_actions.ndim != 2 or policy_actions.shape[1] != 2:
            raise ValueError("D6 policy actions must have shape [E,2]")
        before = policy_actions.detach().clone()
        envs = int(policy_actions.shape[0])
        device = policy_actions.device
        self._ensure_counters(device)

        dynamic_valid = _require_tensor(context, "dynamic_valid").bool()
        if dynamic_valid.ndim != 2 or dynamic_valid.shape[0] != envs:
            raise ValueError("D6 dynamic_valid must have shape [E,N]")
        evaluated = dynamic_valid.any(dim=1)
        validate_action_contract(
            num_bins=int(context["num_bins"]),
            max_linear_velocity=float(context["max_linear_velocity"]),
            reverse_velocity_scale=float(context["reverse_velocity_scale"]),
            max_linear_accel=float(context["max_linear_accel"]),
            max_angular_velocity=float(context["max_angular_velocity"]),
            max_angular_accel=float(context["max_angular_accel"]),
            spec=self.geometry_spec,
        )
        current_velocity = _require_tensor(context, "current_velocity_mps")
        current_omega = _require_tensor(context, "current_omega_rad_s")
        pending_command = _require_tensor(context, "pending_command")
        policy_lidar = _require_tensor(context, "policy_lidar_clearance")
        dynamic_positions = _require_tensor(
            context, "dynamic_positions_body_m"
        )
        dynamic_velocities = _require_tensor(
            context, "dynamic_velocities_body_mps"
        )
        dynamic_radii = _require_tensor(context, "dynamic_radii_m")
        linear, angular = decode_discrete_drive_action_grid(
            current_velocity,
            current_omega,
            num_bins=int(context["num_bins"]),
            dt=float(self.geometry_spec.control_dt_s),
            max_linear_velocity=float(context["max_linear_velocity"]),
            reverse_velocity_scale=float(context["reverse_velocity_scale"]),
            max_linear_accel=float(context["max_linear_accel"]),
            max_angular_velocity=float(context["max_angular_velocity"]),
            max_angular_accel=float(context["max_angular_accel"]),
        )
        path, yaw, times = delayed_unicycle_paths(
            linear, angular, pending_command, spec=self.geometry_spec
        )
        sensor_ranges = policy_lidar_to_sensor_range(
            policy_lidar,
            max_range=float(self.geometry_spec.lidar_max_range_m),
            body_radius=float(self.geometry_spec.lidar_body_radius_m),
        )
        points, static_valid, dynamic_returns = lidar_static_points(
            sensor_ranges,
            dynamic_positions,
            dynamic_radii,
            dynamic_valid,
            spec=self.geometry_spec,
        )
        source = attribute_static_lidar_sources(
            points_body_m=points,
            static_valid=static_valid,
            robot_position_local_m=_require_tensor(
                context, "robot_position_local_m"
            ),
            robot_yaw_rad=_require_tensor(context, "robot_yaw_rad"),
            static_positions_local_m=_require_tensor(
                context, "static_positions_local_m"
            ),
            static_radii_m=_require_tensor(context, "static_radii_m"),
            static_obstacle_valid=_require_tensor(
                context, "static_obstacle_valid"
            ),
            wall_centers_local_m=_require_tensor(
                context, "wall_centers_local_m"
            ),
            wall_sizes_m=_require_tensor(context, "wall_sizes_m"),
            wall_valid=_require_tensor(context, "wall_valid"),
            spec=self.spec,
        )
        source_masks = {name: source[name] for name in SOURCE_NAMES}
        timelines = point_obb_clearance_timelines(
            path_m=path,
            yaw_rad=yaw,
            points_m=points,
            source_masks=source_masks,
            geometry_spec=self.geometry_spec,
        )
        all_timeline = _minimum_timelines(timelines, SOURCE_NAMES)
        dynamic_clearance = _dynamic_obb_circle_clearance_grid(
            path,
            yaw,
            times,
            dynamic_positions,
            dynamic_velocities,
            dynamic_radii,
            dynamic_valid,
            self.geometry_spec,
        )
        dynamic_feasible = dynamic_clearance >= float(
            self.geometry_spec.dynamic_surface_clearance_m
        )

        static_grids: list[torch.Tensor] = []
        for horizon in self.spec.horizons_s:
            sample_count = int(round(horizon / self.spec.control_dt_s))
            clearance_grid = all_timeline[..., :sample_count].amin(dim=-1)
            for clearance in self.spec.clearances_m:
                static_grids.append(clearance_grid >= float(clearance))

        source_timelines = {
            "lidar_without_wall_assigned_h2.4_c0.10": _minimum_timelines(
                timelines, ("static_obstacle", "ambiguous", "residual")
            ),
            "lidar_without_static_assigned_h2.4_c0.10": _minimum_timelines(
                timelines, ("wall", "ambiguous", "residual")
            ),
            "lidar_without_ambiguous_h2.4_c0.10": _minimum_timelines(
                timelines, ("wall", "static_obstacle", "residual")
            ),
            "lidar_without_residual_h2.4_c0.10": _minimum_timelines(
                timelines, ("wall", "static_obstacle", "ambiguous")
            ),
            "lidar_wall_only_h2.4_c0.10": timelines["wall"],
            "lidar_static_only_h2.4_c0.10": timelines["static_obstacle"],
            "lidar_ambiguous_only_h2.4_c0.10": timelines["ambiguous"],
            "lidar_residual_only_h2.4_c0.10": timelines["residual"],
            "no_static_lidar_dynamic_ceiling": _minimum_timelines(timelines, ()),
        }
        for key in self.keys[len(self.spec.horizons_s) * len(self.spec.clearances_m):]:
            source_clearance = source_timelines[key].amin(dim=-1)
            static_grids.append(
                source_clearance >= float(self.spec.source_clearance_m)
            )
        static_stack = torch.stack(static_grids, dim=1)
        if static_stack.shape[1] != len(self.keys):
            raise RuntimeError("D6 variant construction count mismatch")
        joint_stack = static_stack & dynamic_feasible[:, None, :, :]
        static_any = static_stack.flatten(2).any(dim=2)
        joint_any = joint_stack.flatten(2).any(dim=2)
        static_count = static_stack.flatten(2).sum(dim=2)
        joint_count = joint_stack.flatten(2).sum(dim=2)

        frozen_index = self._key_index[frozen_variant_key(self.spec)]
        distances = _require_tensor(context, "obstacle_distances_m")
        if distances.shape != dynamic_valid.shape:
            raise ValueError("D6 obstacle distance shape mismatch")
        nearest_distance = torch.where(
            dynamic_valid,
            distances,
            torch.full_like(distances, float("inf")),
        ).amin(dim=1)
        snapshot: dict[str, torch.Tensor | int] = {
            "serial": self._snapshot_serial + 1,
            "evaluated": evaluated,
            "near_dynamic": evaluated
            & (nearest_distance <= float(self.spec.near_dynamic_distance_m)),
            "frozen_static_blocked": evaluated & ~static_any[:, frozen_index],
            "frozen_joint_blocked": evaluated & ~joint_any[:, frozen_index],
            "static_any": static_any,
            "joint_any": joint_any,
            "static_count": static_count,
            "joint_count": joint_count,
        }
        self._snapshot_serial += 1
        self._accumulate_scope("all_evaluated", evaluated, snapshot)
        self._accumulate_scope("near_dynamic_3m", snapshot["near_dynamic"], snapshot)
        self._accumulate_scope(
            "frozen_static_blocked", snapshot["frozen_static_blocked"], snapshot
        )
        self._accumulate_scope(
            "frozen_joint_blocked", snapshot["frozen_joint_blocked"], snapshot
        )

        assert self._source_points is not None
        assert self._static_valid_points is not None
        assert self._dynamic_returns_removed is not None
        evaluated_points = evaluated[:, None]
        self._source_points += torch.stack(
            [
                (source[name] & evaluated_points).long().sum()
                for name in SOURCE_NAMES
            ]
        )
        self._static_valid_points += (
            static_valid & evaluated_points
        ).long().sum()
        self._dynamic_returns_removed += (
            dynamic_returns & evaluated_points
        ).long().sum()

        self._calls += 1
        self._environment_frames += envs
        self._action_identity_checks += envs
        errors = int((policy_actions != before).any(dim=1).sum().item())
        self._action_identity_errors += errors
        if errors:
            raise RuntimeError("D6 shadow modified policy actions")
        return snapshot

    @torch.no_grad()
    def record_transition(
        self,
        snapshot: dict[str, torch.Tensor | int],
        dynamic_collision: torch.Tensor,
    ) -> None:
        serial = int(snapshot.get("serial", -1))
        if serial != self._last_recorded_serial + 1:
            raise RuntimeError("D6 transition snapshots are missing or out of order")
        mask = dynamic_collision.reshape(-1).bool()
        if mask.shape != snapshot["evaluated"].shape:
            raise ValueError("D6 dynamic collision shape mismatch")
        self._accumulate_scope(
            "dynamic_collision_transition", mask, snapshot
        )
        self._last_recorded_serial = serial
        self._transitions += 1

    def _matrix_report(self) -> dict:
        if self._scope_frames is None:
            return {}
        assert self._static_any is not None
        assert self._joint_any is not None
        assert self._static_action_sum is not None
        assert self._joint_action_sum is not None
        frames = self._scope_frames.detach().cpu().tolist()
        static_any = self._static_any.detach().cpu().tolist()
        joint_any = self._joint_any.detach().cpu().tolist()
        static_sum = self._static_action_sum.detach().cpu().tolist()
        joint_sum = self._joint_action_sum.detach().cpu().tolist()
        report = {}
        for scope_index, scope_name in enumerate(SCOPE_NAMES):
            count = int(frames[scope_index])
            variants = {}
            for variant_index, key in enumerate(self.keys):
                variants[key] = {
                    "static_any_feasible_frames": int(
                        static_any[scope_index][variant_index]
                    ),
                    "static_any_feasible_rate": (
                        float(static_any[scope_index][variant_index]) / count
                        if count
                        else None
                    ),
                    "joint_any_feasible_frames": int(
                        joint_any[scope_index][variant_index]
                    ),
                    "joint_any_feasible_rate": (
                        float(joint_any[scope_index][variant_index]) / count
                        if count
                        else None
                    ),
                    "mean_static_feasible_actions": (
                        float(static_sum[scope_index][variant_index]) / count
                        if count
                        else None
                    ),
                    "mean_joint_feasible_actions": (
                        float(joint_sum[scope_index][variant_index]) / count
                        if count
                        else None
                    ),
                }
            report[scope_name] = {"frames": count, "variants": variants}
        return report

    def report(self, *, metadata: dict | None = None) -> dict:
        metadata = dict(metadata or {})
        matrix = self._matrix_report()
        source_counts = (
            self._source_points.detach().cpu().tolist()
            if self._source_points is not None
            else [0] * len(SOURCE_NAMES)
        )
        static_points = int(
            self._static_valid_points.detach().cpu().item()
            if self._static_valid_points is not None
            else 0
        )
        dynamic_removed = int(
            self._dynamic_returns_removed.detach().cpu().item()
            if self._dynamic_returns_removed is not None
            else 0
        )
        partition_sum = sum(int(value) for value in source_counts)
        expected_records = int(
            metadata.get("expected_records", self._environment_frames)
        )
        action_identity_ok = self._action_identity_errors == 0
        transition_ok = self._transitions == self._calls
        coverage_ok = expected_records == self._environment_frames
        partition_ok = partition_sum == static_points and self._partition_errors == 0
        return {
            "schema": REPORT_SCHEMA,
            "mode": MODE,
            "protocol": static_sensitivity_protocol(
                self.spec, self.geometry_spec
            ),
            "metadata": metadata,
            "runtime": {
                "calls": self._calls,
                "transitions": self._transitions,
                "environment_frames": self._environment_frames,
                "action_identity_checks": self._action_identity_checks,
                "action_identity_errors": self._action_identity_errors,
                "source_partition_errors": self._partition_errors,
            },
            "lidar_point_attribution": {
                "static_valid_points": static_points,
                "dynamic_returns_removed": dynamic_removed,
                "counts": {
                    name: int(source_counts[index])
                    for index, name in enumerate(SOURCE_NAMES)
                },
                "fractions": {
                    name: (
                        float(source_counts[index]) / static_points
                        if static_points
                        else None
                    )
                    for index, name in enumerate(SOURCE_NAMES)
                },
            },
            "matrix": matrix,
            "self_check": {
                "baseline_action_identity_ok": action_identity_ok,
                "transition_coverage_ok": transition_ok,
                "record_coverage_ok": coverage_ok,
                "source_partition_ok": partition_ok,
                "reconciliation_ok": (
                    action_identity_ok
                    and transition_ok
                    and coverage_ok
                    and partition_ok
                ),
            },
        }

    def write(self, path: str | Path, *, metadata: dict) -> dict:
        report = self.report(metadata=metadata)
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report


def build_static_sensitivity() -> StaticFeasibilitySensitivity:
    return StaticFeasibilitySensitivity()


__all__ = [
    "MODE",
    "REPORT_SCHEMA",
    "SOURCE_NAMES",
    "SCOPE_NAMES",
    "StaticFeasibilitySensitivity",
    "StaticSensitivitySpec",
    "attribute_static_lidar_sources",
    "build_static_sensitivity",
    "frozen_variant_key",
    "point_obb_clearance_timelines",
    "static_sensitivity_protocol",
    "variant_keys",
]
