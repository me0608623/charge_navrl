"""Trace the realized origin of SA4-D6 residual 72-bin LiDAR points.

D7 is a baseline-only shadow audit. The observation function records the
already-realized raw, bias, sigma, dropout, and distractor stages without
drawing any additional random numbers. This module matches that trace to the
exact policy observation and attributes final D6 residual points to the
realized corruption stage and privileged scene geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from rnn_car_wdclean.d4_geometry_selector import (
    GeometrySelectorSpec,
    lidar_static_points,
)
from rnn_car_wdclean.d6_static_feasibility_sensitivity import (
    SOURCE_NAMES,
    StaticSensitivitySpec,
    attribute_static_lidar_sources,
)
from rnn_car_wdclean.swept_arc import policy_lidar_to_sensor_range


MODE = "lidar_residual_origin"
PROTOCOL_SCHEMA = "sa4_d7_lidar_residual_origin/v2"
REPORT_SCHEMA = "sa4_d7_lidar_residual_origin_report/v2"
TRACE_SCHEMA = "vlp16_d7_realized_trace/v1"
TRACE_STAGE_NAMES = ("raw", "bias", "sigma", "dropout", "distractor")
STAGE_NAMES = ("raw", "bias", "sigma", "dropout", "d6_model_final")
SCOPE_NAMES = (
    "all_evaluated",
    "frozen_static_blocked",
    "frozen_static_feasible",
    "dynamic_collision_transition",
)
MECHANISM_NAMES = (
    "distractor_winner",
    "dynamic_attribution_miss",
    "bin_center_reconstruction",
    "range_corruption",
    "raw_dynamic_unmatched",
    "raw_geometry_unmatched",
    "unresolved",
)
CREATION_NAMES = (
    "raw",
    "bias",
    "sigma",
    "dropout",
    "distractor",
    "policy_roundtrip",
)
ACTUAL_SOURCE_NAMES = (
    "wall",
    "static_obstacle",
    "ambiguous",
    "dynamic",
    "unmatched",
    "invalid",
)
RANGE_EDGES_M = (0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
GEOMETRY_DISTANCE_EDGES_M = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class ResidualOriginSpec:
    """Frozen D7 trace and classification contract."""

    expected_num_bins: int = 72
    expected_num_rays: int = 5760
    expected_r_robot_m: float = 0.35
    expected_r_max_m: float = 20.0
    expected_sigma_m: float = 0.008672
    expected_hole_rate: float = 0.194859
    expected_distractor_rate: float = 0.002515
    require_per_ring_bias: bool = True
    require_human_dynamic_dropout: bool = False
    require_block_dropout_prob: float = 0.0
    exact_policy_trace_match: bool = True


def residual_origin_protocol(
    spec: ResidualOriginSpec = ResidualOriginSpec(),
    source_spec: StaticSensitivitySpec = StaticSensitivitySpec(),
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return the content-addressed D7 preregistration contract."""

    source_contract = asdict(source_spec)
    source_contract["horizons_s"] = list(source_spec.horizons_s)
    source_contract["clearances_m"] = list(source_spec.clearances_m)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "mode": MODE,
        "scope": (
            "single-checkpoint baseline-only shadow; no action modification, "
            "training, reward change, scene change, or SA5 launch"
        ),
        "trace_order": list(TRACE_STAGE_NAMES),
        "source_evaluation_order": list(STAGE_NAMES),
        "trace_rule": (
            "record tensors already realized by wd_like_sweep_72; never draw "
            "additional random numbers; select the trace whose normalized "
            "72-bin sweep is bitwise identical to policy obs columns [6:78]"
        ),
        "final_source_range": (
            "classify the final source with D6's exact float32 recovery, "
            "policy_lidar * 20.0 + 0.35; retain the pre-normalization realized "
            "range for winner-origin attribution"
        ),
        "source_question": [
            "did a mixed-pixel distractor ray win the final 5-degree bin",
            "at which realized stage did the bin most recently become residual",
            "does the actual winning 1-degree ray match known geometry while the 5-degree center does not",
            "does actual-angle dynamic attribution disagree with center-angle attribution",
            "is the original physical ray hit itself absent from the privileged geometry model",
        ],
        "mechanism_priority": list(MECHANISM_NAMES),
        "interpretation_limits": [
            "raw_geometry_unmatched does not identify a USD prim by itself",
            "a residual point in a blocked frame is not necessarily the unique binding point",
            "privileged source labels inherit D6's 0.20m geometry tolerance",
            "single checkpoint and evaluator seed are diagnostic evidence only",
        ],
        "trace_spec": asdict(spec),
        "source_spec": source_contract,
        "geometry_spec": asdict(geometry_spec),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def select_matching_lidar_trace(
    raw_env,
    policy_lidar: torch.Tensor,
) -> tuple[dict, int]:
    """Return the realized trace matching the exact current policy sweep."""

    candidates = list(getattr(raw_env, "_d7_lidar_trace_candidates", []))
    if not candidates:
        raise RuntimeError("D7 found no realized LiDAR trace candidates")
    matches = []
    max_errors = []
    for candidate in candidates:
        if candidate.get("schema") != TRACE_SCHEMA:
            continue
        sweep = candidate.get("sweep_normalized")
        if not isinstance(sweep, torch.Tensor) or sweep.shape != policy_lidar.shape:
            continue
        if torch.equal(sweep, policy_lidar):
            matches.append(candidate)
        else:
            max_errors.append(float((sweep - policy_lidar).abs().max().item()))
    if not matches:
        detail = min(max_errors) if max_errors else None
        raise RuntimeError(
            "D7 found no bitwise policy LiDAR trace match"
            + (f"; smallest max error={detail:.9g}" if detail is not None else "")
        )
    reference = matches[0]
    for duplicate in matches[1:]:
        for key in (
            "sweep_sensor_ranges_m",
            "winner_ray_index",
            "winner_distractor",
            "winner_hit_body_xyz_m",
            "winner_actual_angle_rad",
        ):
            if not torch.equal(reference[key], duplicate[key]):
                raise RuntimeError(
                    "D7 policy trace match is ambiguous with different realized tensors"
                )
    return reference, len(matches)


def _require_tensor(mapping: dict, key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"D7 field {key!r} must be a tensor")
    return value


def _validate_trace_contract(trace: dict, spec: ResidualOriginSpec) -> None:
    contract = trace.get("noise_contract") or {}
    expected = {
        "num_bins": spec.expected_num_bins,
        "num_rays": spec.expected_num_rays,
        "per_ring_bias": spec.require_per_ring_bias,
        "human_dynamic_dropout": spec.require_human_dynamic_dropout,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"D7 realized trace contract drift: {key}={contract.get(key)!r}, "
                f"expected {value!r}"
            )
    float_expected = {
        "r_robot": spec.expected_r_robot_m,
        "r_max": spec.expected_r_max_m,
        "displacement_std_soft": spec.expected_sigma_m,
        "hole_rate": spec.expected_hole_rate,
        "distractor_rate": spec.expected_distractor_rate,
        "block_dropout_prob": spec.require_block_dropout_prob,
    }
    for key, value in float_expected.items():
        actual = float(contract.get(key, float("nan")))
        if not math.isfinite(actual) or abs(actual - value) > 1.0e-9:
            raise RuntimeError(
                f"D7 realized trace contract drift: {key}={actual!r}, expected {value!r}"
            )


def _actual_point_sources(
    *,
    points_body_m: torch.Tensor,
    valid: torch.Tensor,
    context: dict,
    source_spec: StaticSensitivitySpec,
    geometry_spec: GeometrySelectorSpec,
) -> dict[str, torch.Tensor]:
    dynamic_positions = _require_tensor(context, "dynamic_positions_body_m")
    dynamic_radii = _require_tensor(context, "dynamic_radii_m")
    dynamic_valid = _require_tensor(context, "dynamic_valid").bool()
    distance = (
        points_body_m[:, :, None, :] - dynamic_positions[:, None, :, :]
    ).norm(dim=-1)
    dynamic = (
        distance
        <= (
            dynamic_radii[:, None, :]
            + float(geometry_spec.dynamic_return_margin_m)
        )
    ) & dynamic_valid[:, None, :]
    dynamic = valid & dynamic.any(dim=-1)
    static_valid = valid & ~dynamic
    attributed = attribute_static_lidar_sources(
        points_body_m=points_body_m,
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
        spec=source_spec,
    )
    return {
        "wall": attributed["wall"],
        "static_obstacle": attributed["static_obstacle"],
        "ambiguous": attributed["ambiguous"],
        "dynamic": dynamic,
        "unmatched": static_valid & attributed["residual"],
        "invalid": ~valid,
        "wall_surface_distance_m": attributed["wall_surface_distance_m"],
        "static_surface_distance_m": attributed["static_surface_distance_m"],
    }


def _histogram(values: torch.Tensor, valid: torch.Tensor, edges: tuple[float, ...]) -> torch.Tensor:
    selected = values[valid]
    if selected.numel() == 0:
        return torch.zeros(len(edges) - 1, dtype=torch.long, device=values.device)
    boundaries = torch.tensor(edges[1:-1], device=values.device, dtype=values.dtype)
    buckets = torch.bucketize(selected, boundaries)
    return torch.bincount(buckets, minlength=len(edges) - 1)


class LidarResidualOriginAudit:
    """Accumulate exact realized origins for D6 residual policy bins."""

    mode = MODE

    def __init__(
        self,
        *,
        spec: ResidualOriginSpec = ResidualOriginSpec(),
        source_spec: StaticSensitivitySpec = StaticSensitivitySpec(),
        geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
    ) -> None:
        self.spec = spec
        self.source_spec = source_spec
        self.geometry_spec = geometry_spec
        self._calls = 0
        self._transitions = 0
        self._environment_frames = 0
        self._action_identity_checks = 0
        self._action_identity_errors = 0
        self._trace_match_candidates = 0
        self._trace_match_ambiguities = 0
        self._stage_residual: torch.Tensor | None = None
        self._stage_entered: torch.Tensor | None = None
        self._stage_exited: torch.Tensor | None = None
        self._mechanisms: torch.Tensor | None = None
        self._creation: torch.Tensor | None = None
        self._actual_final_sources: torch.Tensor | None = None
        self._raw_hit_sources: torch.Tensor | None = None
        self._final_source_counts: torch.Tensor | None = None
        self._scope_frames: torch.Tensor | None = None
        self._residual_points: torch.Tensor | None = None
        self._residual_frames: torch.Tensor | None = None
        self._distractor_frames: torch.Tensor | None = None
        self._nearest_distractor_frames: torch.Tensor | None = None
        self._range_hist: torch.Tensor | None = None
        self._distance_hist: torch.Tensor | None = None
        self._bin_hist: torch.Tensor | None = None
        self._ring_hist: torch.Tensor | None = None

    def _ensure_counters(self, device: torch.device) -> None:
        if self._scope_frames is not None:
            if self._scope_frames.device != device:
                raise RuntimeError("D7 device changed during audit")
            return
        scopes = len(SCOPE_NAMES)
        self._scope_frames = torch.zeros(scopes, dtype=torch.long, device=device)
        self._stage_residual = torch.zeros(
            scopes, len(STAGE_NAMES), dtype=torch.long, device=device
        )
        self._stage_entered = torch.zeros_like(self._stage_residual)
        self._stage_exited = torch.zeros_like(self._stage_residual)
        self._mechanisms = torch.zeros(
            scopes, len(MECHANISM_NAMES), dtype=torch.long, device=device
        )
        self._creation = torch.zeros(
            scopes, len(CREATION_NAMES), dtype=torch.long, device=device
        )
        self._actual_final_sources = torch.zeros(
            scopes, len(ACTUAL_SOURCE_NAMES), dtype=torch.long, device=device
        )
        self._raw_hit_sources = torch.zeros_like(self._actual_final_sources)
        self._final_source_counts = torch.zeros(
            scopes, len(SOURCE_NAMES), dtype=torch.long, device=device
        )
        self._residual_points = torch.zeros(scopes, dtype=torch.long, device=device)
        self._residual_frames = torch.zeros_like(self._residual_points)
        self._distractor_frames = torch.zeros_like(self._residual_points)
        self._nearest_distractor_frames = torch.zeros_like(self._residual_points)
        self._range_hist = torch.zeros(
            scopes, len(RANGE_EDGES_M) - 1, dtype=torch.long, device=device
        )
        self._distance_hist = torch.zeros(
            scopes,
            len(GEOMETRY_DISTANCE_EDGES_M) - 1,
            dtype=torch.long,
            device=device,
        )
        self._bin_hist = torch.zeros(
            scopes, self.spec.expected_num_bins, dtype=torch.long, device=device
        )
        self._ring_hist = torch.zeros(scopes, 16, dtype=torch.long, device=device)

    def _center_sources(self, ranges: torch.Tensor, context: dict) -> dict:
        points, static_valid, dynamic_return = lidar_static_points(
            ranges,
            _require_tensor(context, "dynamic_positions_body_m"),
            _require_tensor(context, "dynamic_radii_m"),
            _require_tensor(context, "dynamic_valid"),
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
            spec=self.source_spec,
        )
        return {
            **source,
            "points": points,
            "static_valid": static_valid,
            "dynamic_return": dynamic_return,
        }

    def _accumulate_scope(
        self,
        scope_name: str,
        env_mask: torch.Tensor,
        derived: dict,
    ) -> None:
        scope = SCOPE_NAMES.index(scope_name)
        assert self._scope_frames is not None
        assert self._stage_residual is not None
        assert self._stage_entered is not None
        assert self._stage_exited is not None
        assert self._mechanisms is not None
        assert self._creation is not None
        assert self._actual_final_sources is not None
        assert self._raw_hit_sources is not None
        assert self._final_source_counts is not None
        assert self._residual_points is not None
        assert self._residual_frames is not None
        assert self._distractor_frames is not None
        assert self._nearest_distractor_frames is not None
        assert self._range_hist is not None
        assert self._distance_hist is not None
        assert self._bin_hist is not None
        assert self._ring_hist is not None

        self._scope_frames[scope] += env_mask.long().sum()
        point_scope = env_mask[:, None]
        stage_residual = derived["stage_residual"] & point_scope[:, None, :]
        self._stage_residual[scope] += stage_residual.long().sum(dim=(0, 2))
        previous = stage_residual[:, :-1]
        current = stage_residual[:, 1:]
        self._stage_entered[scope, 1:] += (
            current & ~previous
        ).long().sum(dim=(0, 2))
        self._stage_exited[scope, 1:] += (
            previous & ~current
        ).long().sum(dim=(0, 2))

        final_residual = derived["final_residual"] & point_scope
        self._residual_points[scope] += final_residual.long().sum()
        self._residual_frames[scope] += (
            final_residual.any(dim=1) & env_mask
        ).long().sum()
        for index, name in enumerate(MECHANISM_NAMES):
            self._mechanisms[scope, index] += (
                derived["mechanisms"][name] & point_scope
            ).long().sum()
        for index, name in enumerate(CREATION_NAMES):
            self._creation[scope, index] += (
                derived["creation"][name] & point_scope
            ).long().sum()
        for index, name in enumerate(ACTUAL_SOURCE_NAMES):
            self._actual_final_sources[scope, index] += (
                derived["actual_final_sources"][name]
                & final_residual
            ).long().sum()
            self._raw_hit_sources[scope, index] += (
                derived["raw_hit_sources"][name]
                & final_residual
            ).long().sum()
        for index, name in enumerate(SOURCE_NAMES):
            self._final_source_counts[scope, index] += (
                derived["final_sources"][name] & point_scope
            ).long().sum()

        distractor_residual = (
            derived["winner_distractor"] & final_residual
        )
        self._distractor_frames[scope] += (
            distractor_residual.any(dim=1) & env_mask
        ).long().sum()
        residual_ranges = torch.where(
            final_residual,
            derived["final_ranges_m"],
            torch.full_like(derived["final_ranges_m"], float("inf")),
        )
        nearest_range, nearest_bin = residual_ranges.min(dim=1)
        nearest_valid = env_mask & torch.isfinite(nearest_range)
        nearest_is_distractor = torch.gather(
            derived["winner_distractor"], 1, nearest_bin[:, None]
        ).squeeze(1)
        self._nearest_distractor_frames[scope] += (
            nearest_valid & nearest_is_distractor
        ).long().sum()
        self._range_hist[scope] += _histogram(
            nearest_range, nearest_valid, RANGE_EDGES_M
        )

        known_distance = torch.minimum(
            derived["final_wall_distance_m"],
            derived["final_static_distance_m"],
        )
        self._distance_hist[scope] += _histogram(
            known_distance, final_residual, GEOMETRY_DISTANCE_EDGES_M
        )
        bin_ids = torch.arange(
            self.spec.expected_num_bins,
            device=env_mask.device,
            dtype=torch.long,
        )[None, :].expand_as(final_residual)
        selected_bins = bin_ids[final_residual]
        if selected_bins.numel():
            self._bin_hist[scope] += torch.bincount(
                selected_bins, minlength=self.spec.expected_num_bins
            )
        rings = derived["winner_ring_index"].clamp(0, 15)
        selected_rings = rings[final_residual]
        if selected_rings.numel():
            self._ring_hist[scope] += torch.bincount(
                selected_rings, minlength=16
            )

    @torch.no_grad()
    def observe(
        self,
        policy_actions: torch.Tensor,
        *,
        policy_lidar: torch.Tensor,
        trace: dict,
        trace_match_count: int,
        context: dict,
        d6_snapshot: dict,
    ) -> dict:
        if policy_actions.ndim != 2 or policy_actions.shape[1] != 2:
            raise ValueError("D7 policy actions must have shape [E,2]")
        before = policy_actions.detach().clone()
        _validate_trace_contract(trace, self.spec)
        if not torch.equal(trace["sweep_normalized"], policy_lidar):
            raise RuntimeError("D7 selected trace is not bitwise policy-identical")
        ranges = _require_tensor(trace, "sweep_sensor_ranges_m")
        envs = policy_actions.shape[0]
        expected_shape = (envs, len(STAGE_NAMES), self.spec.expected_num_bins)
        if ranges.shape != expected_shape:
            raise ValueError(
                f"D7 stage sweep shape {tuple(ranges.shape)} != {expected_shape}"
            )
        self._ensure_counters(policy_actions.device)

        d6_final_ranges = policy_lidar_to_sensor_range(
            policy_lidar,
            max_range=float(self.geometry_spec.lidar_max_range_m),
            body_radius=float(self.geometry_spec.lidar_body_radius_m),
        )
        center = [
            self._center_sources(ranges[:, index], context)
            for index in range(len(STAGE_NAMES) - 1)
        ]
        direct_final_center = self._center_sources(ranges[:, -1], context)
        center.append(self._center_sources(d6_final_ranges, context))
        stage_residual = torch.stack(
            [stage["residual"] for stage in center], dim=1
        )
        final = center[-1]
        evaluated = _require_tensor(context, "dynamic_valid").bool().any(dim=1)
        frozen_blocked = _require_tensor(
            d6_snapshot, "frozen_static_blocked"
        ).bool()
        if frozen_blocked.shape != evaluated.shape:
            raise ValueError("D7/D6 frozen blocked shape mismatch")

        winner_valid = _require_tensor(trace, "winner_valid").bool()
        winner_raw_valid = _require_tensor(trace, "winner_raw_valid").bool()
        final_ranges = _require_tensor(trace, "winner_final_range_m")
        raw_ranges = _require_tensor(trace, "winner_raw_range_m")
        angles = _require_tensor(trace, "winner_actual_angle_rad")
        final_actual_points = torch.stack(
            [final_ranges * torch.cos(angles), final_ranges * torch.sin(angles)],
            dim=-1,
        )
        actual_valid = (
            winner_valid
            & (final_ranges > float(self.geometry_spec.lidar_hole_threshold_m))
            & (final_ranges < float(self.geometry_spec.lidar_max_range_m))
        )
        actual_final_sources = _actual_point_sources(
            points_body_m=final_actual_points,
            valid=actual_valid,
            context=context,
            source_spec=self.source_spec,
            geometry_spec=self.geometry_spec,
        )
        raw_hit_points = _require_tensor(
            trace, "winner_hit_body_xyz_m"
        )[..., :2]
        raw_valid = (
            winner_raw_valid
            & (raw_ranges > float(self.geometry_spec.lidar_hole_threshold_m))
            & (raw_ranges < float(self.geometry_spec.lidar_max_range_m))
        )
        raw_hit_sources = _actual_point_sources(
            points_body_m=raw_hit_points,
            valid=raw_valid,
            context=context,
            source_spec=self.source_spec,
            geometry_spec=self.geometry_spec,
        )

        final_residual = final["residual"]
        winner_distractor = _require_tensor(
            trace, "winner_distractor"
        ).bool()
        known_actual = (
            actual_final_sources["wall"]
            | actual_final_sources["static_obstacle"]
            | actual_final_sources["ambiguous"]
        )
        known_raw = (
            raw_hit_sources["wall"]
            | raw_hit_sources["static_obstacle"]
            | raw_hit_sources["ambiguous"]
        )
        remaining = final_residual.clone()
        mechanisms: dict[str, torch.Tensor] = {}
        mechanisms["distractor_winner"] = remaining & winner_distractor
        remaining &= ~mechanisms["distractor_winner"]
        mechanisms["dynamic_attribution_miss"] = (
            remaining & actual_final_sources["dynamic"]
        )
        remaining &= ~mechanisms["dynamic_attribution_miss"]
        mechanisms["bin_center_reconstruction"] = remaining & known_actual
        remaining &= ~mechanisms["bin_center_reconstruction"]
        mechanisms["range_corruption"] = remaining & known_raw
        remaining &= ~mechanisms["range_corruption"]
        mechanisms["raw_dynamic_unmatched"] = (
            remaining & raw_hit_sources["dynamic"]
        )
        remaining &= ~mechanisms["raw_dynamic_unmatched"]
        mechanisms["raw_geometry_unmatched"] = (
            remaining & raw_hit_sources["unmatched"]
        )
        remaining &= ~mechanisms["raw_geometry_unmatched"]
        mechanisms["unresolved"] = remaining
        mechanism_sum = sum(mask.long() for mask in mechanisms.values())
        if not torch.equal(mechanism_sum, final_residual.long()):
            raise RuntimeError("D7 residual mechanism partition failed")

        raw_r, bias_r, sigma_r, dropout_r, final_r = stage_residual.unbind(dim=1)
        direct_final_r = direct_final_center["residual"]
        final_direct = final_r & direct_final_r
        creation = {
            "policy_roundtrip": final_r & ~direct_final_r,
            "distractor": final_direct & ~dropout_r,
            "dropout": final_direct & dropout_r & ~sigma_r,
            "sigma": final_direct & dropout_r & sigma_r & ~bias_r,
            "bias": final_direct & dropout_r & sigma_r & bias_r & ~raw_r,
            "raw": final_direct & dropout_r & sigma_r & bias_r & raw_r,
        }
        creation_sum = sum(mask.long() for mask in creation.values())
        if not torch.equal(creation_sum, final_residual.long()):
            raise RuntimeError("D7 residual creation-stage partition failed")

        derived = {
            "stage_residual": stage_residual,
            "final_residual": final_residual,
            "final_sources": {name: final[name] for name in SOURCE_NAMES},
            "mechanisms": mechanisms,
            "creation": creation,
            "actual_final_sources": actual_final_sources,
            "raw_hit_sources": raw_hit_sources,
            "winner_distractor": winner_distractor,
            "winner_ring_index": _require_tensor(trace, "winner_ring_index"),
            "final_ranges_m": d6_final_ranges,
            "final_wall_distance_m": final["wall_surface_distance_m"],
            "final_static_distance_m": final["static_surface_distance_m"],
            "evaluated": evaluated,
        }
        self._accumulate_scope("all_evaluated", evaluated, derived)
        self._accumulate_scope(
            "frozen_static_blocked", evaluated & frozen_blocked, derived
        )
        self._accumulate_scope(
            "frozen_static_feasible", evaluated & ~frozen_blocked, derived
        )

        self._calls += 1
        self._environment_frames += envs
        self._action_identity_checks += envs
        self._trace_match_candidates += int(trace_match_count)
        if trace_match_count > 1:
            self._trace_match_ambiguities += 1
        errors = int((policy_actions != before).any(dim=1).sum().item())
        self._action_identity_errors += errors
        if errors:
            raise RuntimeError("D7 shadow modified policy actions")
        return derived

    @torch.no_grad()
    def record_transition(
        self,
        derived: dict,
        dynamic_collision: torch.Tensor,
    ) -> None:
        mask = dynamic_collision.reshape(-1).bool()
        if mask.shape != derived["evaluated"].shape:
            raise ValueError("D7 dynamic collision shape mismatch")
        self._accumulate_scope("dynamic_collision_transition", mask, derived)
        self._transitions += 1

    def _named_counts(self, tensor: torch.Tensor, names: tuple[str, ...]) -> dict:
        values = tensor.detach().cpu().tolist()
        return {name: int(values[index]) for index, name in enumerate(names)}

    def _scope_report(self, scope_index: int) -> dict:
        assert self._scope_frames is not None
        assert self._stage_residual is not None
        assert self._stage_entered is not None
        assert self._stage_exited is not None
        assert self._mechanisms is not None
        assert self._creation is not None
        assert self._actual_final_sources is not None
        assert self._raw_hit_sources is not None
        assert self._final_source_counts is not None
        assert self._residual_points is not None
        assert self._residual_frames is not None
        assert self._distractor_frames is not None
        assert self._nearest_distractor_frames is not None
        assert self._range_hist is not None
        assert self._distance_hist is not None
        assert self._bin_hist is not None
        assert self._ring_hist is not None
        frames = int(self._scope_frames[scope_index].item())
        residual_points = int(self._residual_points[scope_index].item())
        residual_frames = int(self._residual_frames[scope_index].item())
        return {
            "frames": frames,
            "final_residual_points": residual_points,
            "residual_points_per_frame": residual_points / frames if frames else None,
            "frames_with_residual": residual_frames,
            "frames_with_residual_rate": residual_frames / frames if frames else None,
            "frames_with_distractor_residual": int(
                self._distractor_frames[scope_index].item()
            ),
            "nearest_residual_is_distractor_frames": int(
                self._nearest_distractor_frames[scope_index].item()
            ),
            "stage_residual_counts": self._named_counts(
                self._stage_residual[scope_index], STAGE_NAMES
            ),
            "stage_entered_residual_counts": self._named_counts(
                self._stage_entered[scope_index], STAGE_NAMES
            ),
            "stage_exited_residual_counts": self._named_counts(
                self._stage_exited[scope_index], STAGE_NAMES
            ),
            "final_creation_stage": self._named_counts(
                self._creation[scope_index], CREATION_NAMES
            ),
            "final_mechanism": self._named_counts(
                self._mechanisms[scope_index], MECHANISM_NAMES
            ),
            "final_actual_angle_source": self._named_counts(
                self._actual_final_sources[scope_index], ACTUAL_SOURCE_NAMES
            ),
            "final_winner_raw_hit_source": self._named_counts(
                self._raw_hit_sources[scope_index], ACTUAL_SOURCE_NAMES
            ),
            "final_center_source_counts": self._named_counts(
                self._final_source_counts[scope_index], SOURCE_NAMES
            ),
            "nearest_residual_range_histogram": {
                "edges_m": list(RANGE_EDGES_M),
                "counts": self._range_hist[scope_index].detach().cpu().tolist(),
            },
            "residual_nearest_known_geometry_distance_histogram": {
                "edges_m": list(GEOMETRY_DISTANCE_EDGES_M),
                "counts": self._distance_hist[scope_index].detach().cpu().tolist(),
            },
            "residual_bin_counts": self._bin_hist[scope_index].detach().cpu().tolist(),
            "residual_ring_counts": self._ring_hist[scope_index].detach().cpu().tolist(),
        }

    def report(self, *, metadata: dict | None = None) -> dict:
        metadata = dict(metadata or {})
        expected_records = int(metadata.get("expected_records", self._environment_frames))
        scopes = {}
        if self._scope_frames is not None:
            scopes = {
                name: self._scope_report(index)
                for index, name in enumerate(SCOPE_NAMES)
            }
        action_ok = self._action_identity_errors == 0
        transition_ok = self._transitions == self._calls
        coverage_ok = self._environment_frames == expected_records
        return {
            "schema": REPORT_SCHEMA,
            "mode": MODE,
            "protocol": residual_origin_protocol(
                self.spec, self.source_spec, self.geometry_spec
            ),
            "metadata": metadata,
            "runtime": {
                "calls": self._calls,
                "transitions": self._transitions,
                "environment_frames": self._environment_frames,
                "action_identity_checks": self._action_identity_checks,
                "action_identity_errors": self._action_identity_errors,
                "trace_match_candidates": self._trace_match_candidates,
                "trace_match_ambiguous_calls": self._trace_match_ambiguities,
            },
            "scopes": scopes,
            "self_check": {
                "baseline_action_identity_ok": action_ok,
                "transition_coverage_ok": transition_ok,
                "record_coverage_ok": coverage_ok,
                "exact_policy_trace_match_ok": self._calls > 0,
                "reconciliation_ok": action_ok and transition_ok and coverage_ok and self._calls > 0,
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


def build_residual_origin_audit() -> LidarResidualOriginAudit:
    return LidarResidualOriginAudit()


__all__ = [
    "ACTUAL_SOURCE_NAMES",
    "CREATION_NAMES",
    "LidarResidualOriginAudit",
    "MECHANISM_NAMES",
    "MODE",
    "REPORT_SCHEMA",
    "ResidualOriginSpec",
    "SCOPE_NAMES",
    "STAGE_NAMES",
    "build_residual_origin_audit",
    "residual_origin_protocol",
    "select_matching_lidar_trace",
]
