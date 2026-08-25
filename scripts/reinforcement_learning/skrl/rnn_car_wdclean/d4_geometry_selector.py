"""Delay-aware geometry-feasible 19x19 selector for SA4 lateral diagnosis.

This is a diagnostic upper-bound controller, not a deployment shield. Dynamic
obstacle state and dynamic-return attribution are privileged in the current
integration. Static and wall geometry comes from the same 72-bin LiDAR scan
seen by the policy and is checked against the robot OBB along each candidate
trajectory.

The selector is inserted before ``process_actions``. For fixed d1, the first
0.2 s of every candidate path uses the command already pending in the actuator
queue; the newly selected command can affect only subsequent samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Iterable

import torch

from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)
from rnn_car_wdclean.swept_arc import (
    beam_angles,
    policy_lidar_to_sensor_range,
)


MODE = "geometry_feasible"
PROTOCOL_SCHEMA = "sa4_d4_geometry_selector/v1"
ARGMIN_MODE = "geometry_feasible_argmin"
ARGMIN_PROTOCOL_SCHEMA = "sa4_d4_geometry_selector_argmin/v1"


@dataclass(frozen=True)
class GeometrySelectorSpec:
    """Frozen geometry and selection contract for the bounded D4 probe."""

    control_dt_s: float = 0.2
    delay_steps: int = 1
    num_action_bins: int = 19
    max_linear_velocity_mps: float = 1.0
    reverse_velocity_scale: float = 0.2
    max_linear_accel_mps2: float = 0.5
    max_angular_velocity_rad_s: float = 1.2
    max_angular_accel_rad_s2: float = 3.0
    horizon_s: float = 2.4
    samples: int = 12
    trigger_distance_m: float = 3.0
    trigger_ttc_s: float = 2.0
    trigger_confirm_steps: int = 2
    release_distance_m: float = 3.25
    release_confirm_steps: int = 2
    contact_distance_m: float = 0.70
    dynamic_surface_clearance_m: float = 0.10
    static_surface_clearance_m: float = 0.10
    robot_half_length_m: float = 0.35
    robot_half_width_m: float = 0.30
    robot_obb_offset_x_m: float = -0.128
    lidar_max_range_m: float = 20.0
    lidar_body_radius_m: float = 0.35
    lidar_hole_threshold_m: float = 0.40
    dynamic_return_margin_m: float = 0.15
    action_chunk_size: int = 64


def _validate_spec(spec: GeometrySelectorSpec) -> None:
    if spec.control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive")
    if spec.delay_steps != 1:
        raise ValueError("SA4-D4 is frozen for fixed d1 only")
    if spec.num_action_bins != 19:
        raise ValueError("SA4-D4 is frozen for a 19x19 action grid")
    action_limits = (
        spec.max_linear_velocity_mps,
        spec.reverse_velocity_scale,
        spec.max_linear_accel_mps2,
        spec.max_angular_velocity_rad_s,
        spec.max_angular_accel_rad_s2,
    )
    if any(value <= 0.0 for value in action_limits):
        raise ValueError("frozen action limits must be positive")
    expected_samples = int(round(spec.horizon_s / spec.control_dt_s))
    if spec.samples != expected_samples:
        raise ValueError(
            "samples must cover horizon at one sample per control step: "
            f"expected {expected_samples}, got {spec.samples}"
        )
    if spec.trigger_confirm_steps < 1 or spec.release_confirm_steps < 1:
        raise ValueError("trigger/release confirmations must be positive")
    if spec.trigger_distance_m <= spec.contact_distance_m:
        raise ValueError("trigger distance must exceed contact distance")
    if spec.release_distance_m <= spec.trigger_distance_m:
        raise ValueError("release distance must exceed trigger distance")
    if spec.dynamic_surface_clearance_m < 0.0:
        raise ValueError("dynamic clearance must be non-negative")
    if spec.static_surface_clearance_m < 0.0:
        raise ValueError("static clearance must be non-negative")
    if spec.robot_half_length_m <= 0.0 or spec.robot_half_width_m <= 0.0:
        raise ValueError("robot OBB half extents must be positive")
    if spec.action_chunk_size < 1:
        raise ValueError("action_chunk_size must be positive")


def speed_scaled_geometry_spec(
    speed_rate: float,
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> GeometrySelectorSpec:
    """Derive the D4/D5 action contract under vehicle time dilation."""

    _validate_spec(spec)
    rate = float(speed_rate)
    if not 0.0 < rate <= 1.0:
        raise ValueError("speed_rate must be within (0, 1]")
    scaled = replace(
        spec,
        max_linear_velocity_mps=spec.max_linear_velocity_mps * rate,
        max_linear_accel_mps2=spec.max_linear_accel_mps2 * rate,
        max_angular_velocity_rad_s=spec.max_angular_velocity_rad_s * rate,
        max_angular_accel_rad_s2=spec.max_angular_accel_rad_s2 * rate,
    )
    _validate_spec(scaled)
    return scaled


def geometry_selector_protocol(
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return a content-addressed protocol suitable for preregistration."""

    _validate_spec(spec)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "mode": MODE,
        "scope": (
            "fixed-checkpoint diagnostic upper bound; privileged dynamic "
            "state/attribution; not a deployment shield"
        ),
        "actuator_order": (
            "policy indices -> selector indices -> decode -> d1 queue -> applied"
        ),
        "delay_model": (
            "first 0.2s uses the already pending decoded command; selected "
            "candidate is held only after the d1 prefix"
        ),
        "dynamic_geometry": (
            "constant-velocity obstacle prediction; robot OBB versus obstacle "
            "circle surface clearance"
        ),
        "static_wall_geometry": (
            "policy-visible 72-bin LiDAR points versus swept robot OBB; LiDAR "
            "returns attributed to known dynamic circles are removed"
        ),
        "selection": [
            "activate only after the frozen two-frame radial-TTC trigger",
            "keep the policy action whenever it satisfies both hard clearances",
            "otherwise choose the feasible action with minimum L1 index change",
            "break ties by maximum bottleneck clearance then forward speed",
            "break any remaining tie by the smallest flattened action index",
            (
                "if no jointly feasible action exists, preserve policy action, "
                "mark unresolved, and make no protection claim"
            ),
        ],
        "limitations": [
            "dynamic state and dynamic-return attribution are privileged",
            "static geometry contains only currently visible 72-bin LiDAR surface samples",
            "occluded static surfaces cannot be reconstructed after removing a dynamic return",
            "dynamic obstacle motion is predicted at constant velocity",
            "an unresolved no-feasible frame has no safety guarantee",
        ],
        "spec": asdict(spec),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def geometry_argmin_selector_protocol(
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return the D4-r2 contract using each bin's realized winning-ray angle."""

    _validate_spec(spec)
    protocol = {
        "schema": ARGMIN_PROTOCOL_SCHEMA,
        "mode": ARGMIN_MODE,
        "scope": (
            "fixed-checkpoint diagnostic upper bound; privileged dynamic "
            "state/attribution and realized LiDAR winner trace; not a "
            "deployment shield"
        ),
        "actuator_order": (
            "policy indices -> selector indices -> decode -> d1 queue -> applied"
        ),
        "delay_model": (
            "first 0.2s uses the already pending decoded command; selected "
            "candidate is held only after the d1 prefix"
        ),
        "dynamic_geometry": (
            "constant-velocity obstacle prediction; robot OBB versus obstacle "
            "circle surface clearance"
        ),
        "static_wall_geometry": (
            "policy-visible 72-bin LiDAR ranges placed at the realized raw-ray "
            "angle that won each amin bin; LiDAR returns attributed to known "
            "dynamic circles are removed"
        ),
        "lidar_angle_contract": {
            "source": "winner_actual_angle_rad from the realized 5760-to-72 amin trace",
            "match": "trace sweep must be bitwise identical to current policy LiDAR",
            "fallback": None,
        },
        "paired_static_shadow": (
            "on every active frame, recompute the legacy 5-degree-center grid "
            "on the identical state; only the actual-angle grid may select actions"
        ),
        "selection": [
            "activate only after the frozen two-frame radial-TTC trigger",
            "keep the policy action whenever it satisfies both hard clearances",
            "otherwise choose the feasible action with minimum L1 index change",
            "break ties by maximum bottleneck clearance then forward speed",
            "break any remaining tie by the smallest flattened action index",
            (
                "if no jointly feasible action exists, preserve policy action, "
                "mark unresolved, and make no protection claim"
            ),
        ],
        "limitations": [
            "dynamic state and dynamic-return attribution are privileged",
            "static geometry contains only currently visible 72-bin LiDAR surface samples",
            "occluded static surfaces cannot be reconstructed after removing a dynamic return",
            "dynamic obstacle motion is predicted at constant velocity",
            "an unresolved no-feasible frame has no safety guarantee",
            "the realized winner trace is diagnostic instrumentation unavailable to the policy",
        ],
        "spec": asdict(spec),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def validate_action_contract(
    *,
    num_bins: int,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> None:
    """Reject runtime action dynamics that differ from the frozen D4 cell."""

    _validate_spec(spec)
    runtime = {
        "num_bins": int(num_bins),
        "max_linear_velocity_mps": float(max_linear_velocity),
        "reverse_velocity_scale": float(reverse_velocity_scale),
        "max_linear_accel_mps2": float(max_linear_accel),
        "max_angular_velocity_rad_s": float(max_angular_velocity),
        "max_angular_accel_rad_s2": float(max_angular_accel),
    }
    expected = {
        "num_bins": int(spec.num_action_bins),
        "max_linear_velocity_mps": float(spec.max_linear_velocity_mps),
        "reverse_velocity_scale": float(spec.reverse_velocity_scale),
        "max_linear_accel_mps2": float(spec.max_linear_accel_mps2),
        "max_angular_velocity_rad_s": float(
            spec.max_angular_velocity_rad_s
        ),
        "max_angular_accel_rad_s2": float(
            spec.max_angular_accel_rad_s2
        ),
    }
    drifted = [
        key
        for key, expected_value in expected.items()
        if abs(runtime[key] - expected_value) > 1.0e-9
    ]
    if drifted:
        raise ValueError(
            "SA4-D4 action contract drift: " + ", ".join(drifted)
        )


def _arc_displacement(
    velocity: torch.Tensor,
    omega: torch.Tensor,
    duration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcasted unicycle displacement in the segment's initial frame."""

    straight = omega.abs() < 1.0e-4
    omega_safe = torch.where(straight, torch.ones_like(omega), omega)
    radius = velocity / omega_safe
    angle = omega * duration
    x = radius * torch.sin(angle)
    y = radius * (1.0 - torch.cos(angle))
    x = torch.where(straight, velocity * duration, x)
    y = torch.where(straight, torch.zeros_like(y), y)
    return x, y


def delayed_unicycle_paths(
    candidate_velocity_mps: torch.Tensor,
    candidate_omega_rad_s: torch.Tensor,
    pending_command: torch.Tensor,
    *,
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict candidate paths while respecting the fixed one-step delay.

    Returns ``(path, yaw, times)`` with path/yaw shapes ``[E,L,A,T,...]``.
    """

    _validate_spec(spec)
    if candidate_velocity_mps.shape != candidate_omega_rad_s.shape:
        raise ValueError("candidate velocity and omega grids must match")
    if candidate_velocity_mps.ndim != 3:
        raise ValueError("candidate grids must have shape [E,L,A]")
    envs = int(candidate_velocity_mps.shape[0])
    if pending_command.shape != (envs, 2):
        raise ValueError("pending command must have shape [E,2]")
    tensors = (
        candidate_velocity_mps,
        candidate_omega_rad_s,
        pending_command,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("path inputs contain non-finite data")

    dtype = candidate_velocity_mps.dtype
    device = candidate_velocity_mps.device
    times = (
        torch.arange(1, spec.samples + 1, device=device, dtype=dtype)
        * float(spec.control_dt_s)
    )
    delay_s = float(spec.delay_steps) * float(spec.control_dt_s)
    prefix_t = times.clamp(max=delay_s)[None, None, None, :]
    candidate_t = (times - delay_s).clamp(min=0.0)[None, None, None, :]

    pending_v = pending_command[:, 0, None, None, None]
    pending_w = pending_command[:, 1, None, None, None]
    prefix_x, prefix_y = _arc_displacement(
        pending_v, pending_w, prefix_t
    )
    prefix_yaw = pending_w * prefix_t

    candidate_v = candidate_velocity_mps[..., None]
    candidate_w = candidate_omega_rad_s[..., None]
    local_x, local_y = _arc_displacement(
        candidate_v, candidate_w, candidate_t
    )
    cos_prefix = torch.cos(prefix_yaw)
    sin_prefix = torch.sin(prefix_yaw)
    x = prefix_x + cos_prefix * local_x - sin_prefix * local_y
    y = prefix_y + sin_prefix * local_x + cos_prefix * local_y
    yaw = prefix_yaw + candidate_w * candidate_t
    return torch.stack([x, y], dim=-1), yaw, times


def pending_d1_command(
    action_delay_buffer: Iterable[torch.Tensor] | None,
    just_reset: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return the decoded command that fixed d1 will apply next.

    ``apply_action_delay`` clears every buffered command for reset
    environments before appending the newly decoded command. Consequently,
    the next applied command is zero for those rows even if the pre-step deque
    still contains an older episode's command.
    """

    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("pending-command reference must have shape [E,2]")
    reset = just_reset.reshape(-1).to(
        device=reference.device, dtype=torch.bool
    )
    if reset.shape != reference.shape[:1]:
        raise ValueError("just_reset must have shape [E]")
    buffered = list(action_delay_buffer or ())
    if not buffered:
        pending = torch.zeros_like(reference)
    else:
        latest = buffered[-1]
        if not isinstance(latest, torch.Tensor):
            raise ValueError("action delay buffer must contain tensors")
        if latest.shape != reference.shape:
            raise ValueError("action delay buffer shape does not match reference")
        pending = latest.to(
            device=reference.device, dtype=reference.dtype
        ).clone()
    pending[reset] = 0.0
    if not bool(torch.isfinite(pending).all()):
        raise ValueError("pending command contains non-finite data")
    return pending


def lidar_static_points(
    sensor_ranges_m: torch.Tensor,
    dynamic_positions_body_m: torch.Tensor,
    dynamic_radii_m: torch.Tensor,
    dynamic_valid: torch.Tensor,
    *,
    beam_angles_rad: torch.Tensor | None = None,
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build static/wall LiDAR points after privileged dynamic attribution.

    Historical D4 calls omit ``beam_angles_rad`` and retain the 5-degree bin
    centers. D4-r2 supplies the realized raw-ray angle that won each amin bin.
    """

    _validate_spec(spec)
    if sensor_ranges_m.ndim != 2:
        raise ValueError("LiDAR ranges must have shape [E,K]")
    envs, beams = sensor_ranges_m.shape
    if (
        dynamic_positions_body_m.ndim != 3
        or dynamic_positions_body_m.shape[0] != envs
        or dynamic_positions_body_m.shape[-1] != 2
    ):
        raise ValueError("dynamic positions must have shape [E,N,2]")
    if dynamic_radii_m.shape != dynamic_positions_body_m.shape[:2]:
        raise ValueError("dynamic radii must have shape [E,N]")
    if dynamic_valid.shape != dynamic_positions_body_m.shape[:2]:
        raise ValueError("dynamic valid mask must have shape [E,N]")
    if not bool(torch.isfinite(sensor_ranges_m).all()):
        raise ValueError("LiDAR ranges contain non-finite data")
    if not bool(torch.isfinite(dynamic_positions_body_m).all()):
        raise ValueError("dynamic positions contain non-finite data")
    if not bool(torch.isfinite(dynamic_radii_m).all()):
        raise ValueError("dynamic radii contain non-finite data")

    if beam_angles_rad is None:
        angles = beam_angles(
            beams, device=sensor_ranges_m.device, dtype=sensor_ranges_m.dtype
        )[None, :].expand(envs, -1)
    else:
        if beam_angles_rad.shape != sensor_ranges_m.shape:
            raise ValueError("LiDAR beam angles must have shape [E,K]")
        if beam_angles_rad.device != sensor_ranges_m.device:
            raise ValueError("LiDAR beam angles must share the ranges device")
        if not bool(torch.isfinite(beam_angles_rad).all()):
            raise ValueError("LiDAR beam angles contain non-finite data")
        if bool((beam_angles_rad.abs() > torch.pi + 1.0e-6).any()):
            raise ValueError("LiDAR beam angles must lie in [-pi, pi]")
        angles = beam_angles_rad.to(dtype=sensor_ranges_m.dtype)
    points = torch.stack(
        [
            sensor_ranges_m * torch.cos(angles),
            sensor_ranges_m * torch.sin(angles),
        ],
        dim=-1,
    )
    valid = (
        (sensor_ranges_m > float(spec.lidar_hole_threshold_m))
        & (sensor_ranges_m < float(spec.lidar_max_range_m))
    )
    point_to_dynamic = (
        points[:, :, None, :] - dynamic_positions_body_m[:, None, :, :]
    ).norm(dim=-1)
    attributed_dynamic = (
        point_to_dynamic
        <= (
            dynamic_radii_m[:, None, :]
            + float(spec.dynamic_return_margin_m)
        )
    ) & dynamic_valid[:, None, :].bool()
    dynamic_return = attributed_dynamic.any(dim=-1)
    static_valid = valid & ~dynamic_return
    return points, static_valid, dynamic_return


def _point_obb_clearance_grid(
    path_m: torch.Tensor,
    yaw_rad: torch.Tensor,
    points_m: torch.Tensor,
    point_valid: torch.Tensor,
    spec: GeometrySelectorSpec,
) -> torch.Tensor:
    """Minimum point-to-robot-OBB surface clearance for every action."""

    envs, linear_bins, angular_bins, samples, _ = path_m.shape
    flat_path = path_m.reshape(envs, linear_bins * angular_bins, samples, 2)
    flat_yaw = yaw_rad.reshape(envs, linear_bins * angular_bins, samples)
    outputs: list[torch.Tensor] = []
    for start in range(0, flat_path.shape[1], spec.action_chunk_size):
        stop = min(start + spec.action_chunk_size, flat_path.shape[1])
        path = flat_path[:, start:stop]
        yaw = flat_yaw[:, start:stop]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        center = path + float(spec.robot_obb_offset_x_m) * torch.stack(
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
            -float(spec.robot_half_length_m),
            float(spec.robot_half_length_m),
        )
        closest_y = local_y.clamp(
            -float(spec.robot_half_width_m),
            float(spec.robot_half_width_m),
        )
        clearance = torch.sqrt(
            (local_x - closest_x).square()
            + (local_y - closest_y).square()
        )
        clearance = torch.where(
            point_valid[:, None, None, :],
            clearance,
            torch.full_like(clearance, float("inf")),
        )
        outputs.append(clearance.amin(dim=(-1, -2)))
    return torch.cat(outputs, dim=1).reshape(
        envs, linear_bins, angular_bins
    )


def _dynamic_obb_circle_clearance_grid(
    path_m: torch.Tensor,
    yaw_rad: torch.Tensor,
    times_s: torch.Tensor,
    dynamic_positions_body_m: torch.Tensor,
    dynamic_velocities_body_mps: torch.Tensor,
    dynamic_radii_m: torch.Tensor,
    dynamic_valid: torch.Tensor,
    spec: GeometrySelectorSpec,
) -> torch.Tensor:
    """Minimum robot-OBB to predicted dynamic-circle surface clearance."""

    envs, linear_bins, angular_bins, samples, _ = path_m.shape
    obstacle_path = (
        dynamic_positions_body_m[:, :, None, :]
        + dynamic_velocities_body_mps[:, :, None, :]
        * times_s[None, None, :, None]
    )
    flat_path = path_m.reshape(envs, linear_bins * angular_bins, samples, 2)
    flat_yaw = yaw_rad.reshape(envs, linear_bins * angular_bins, samples)
    outputs: list[torch.Tensor] = []
    for start in range(0, flat_path.shape[1], spec.action_chunk_size):
        stop = min(start + spec.action_chunk_size, flat_path.shape[1])
        path = flat_path[:, start:stop]
        yaw = flat_yaw[:, start:stop]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        center = path + float(spec.robot_obb_offset_x_m) * torch.stack(
            [cos_yaw, sin_yaw], dim=-1
        )
        delta = obstacle_path[:, None, :, :, :] - center[:, :, None, :, :]
        local_x = (
            delta[..., 0] * cos_yaw[:, :, None, :]
            + delta[..., 1] * sin_yaw[:, :, None, :]
        )
        local_y = (
            -delta[..., 0] * sin_yaw[:, :, None, :]
            + delta[..., 1] * cos_yaw[:, :, None, :]
        )
        closest_x = local_x.clamp(
            -float(spec.robot_half_length_m),
            float(spec.robot_half_length_m),
        )
        closest_y = local_y.clamp(
            -float(spec.robot_half_width_m),
            float(spec.robot_half_width_m),
        )
        clearance = torch.sqrt(
            (local_x - closest_x).square()
            + (local_y - closest_y).square()
        ) - dynamic_radii_m[:, None, :, None]
        clearance = torch.where(
            dynamic_valid[:, None, :, None].bool(),
            clearance,
            torch.full_like(clearance, float("inf")),
        )
        outputs.append(clearance.amin(dim=(-1, -2)))
    return torch.cat(outputs, dim=1).reshape(
        envs, linear_bins, angular_bins
    )


@torch.no_grad()
def geometry_feasible_action_grid(
    *,
    policy_actions: torch.Tensor,
    current_velocity_mps: torch.Tensor,
    current_omega_rad_s: torch.Tensor,
    pending_command: torch.Tensor,
    policy_lidar_clearance: torch.Tensor,
    lidar_beam_angles_rad: torch.Tensor | None = None,
    dynamic_positions_body_m: torch.Tensor,
    dynamic_velocities_body_mps: torch.Tensor,
    dynamic_radii_m: torch.Tensor,
    dynamic_valid: torch.Tensor,
    num_bins: int,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
    spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict[str, torch.Tensor]:
    """Select the nearest policy action satisfying both hard geometries."""

    _validate_spec(spec)
    validate_action_contract(
        num_bins=num_bins,
        max_linear_velocity=max_linear_velocity,
        reverse_velocity_scale=reverse_velocity_scale,
        max_linear_accel=max_linear_accel,
        max_angular_velocity=max_angular_velocity,
        max_angular_accel=max_angular_accel,
        spec=spec,
    )
    if policy_actions.ndim != 2 or policy_actions.shape[1] != 2:
        raise ValueError("policy actions must have shape [E,2]")
    envs = int(policy_actions.shape[0])
    if current_velocity_mps.shape != (envs,):
        raise ValueError("current velocity must have shape [E]")
    if current_omega_rad_s.shape != (envs,):
        raise ValueError("current omega must have shape [E]")
    if policy_lidar_clearance.shape != (envs, 72):
        raise ValueError("policy LiDAR clearance must have shape [E,72]")
    if dynamic_velocities_body_mps.shape != dynamic_positions_body_m.shape:
        raise ValueError("dynamic position/velocity shapes must match")
    if (
        dynamic_positions_body_m.ndim != 3
        or dynamic_positions_body_m.shape[0] != envs
        or dynamic_positions_body_m.shape[-1] != 2
    ):
        raise ValueError("dynamic positions must have shape [E,N,2]")
    if dynamic_radii_m.shape != dynamic_positions_body_m.shape[:2]:
        raise ValueError("dynamic radii must have shape [E,N]")
    if dynamic_valid.shape != dynamic_positions_body_m.shape[:2]:
        raise ValueError("dynamic valid must have shape [E,N]")

    linear, angular = decode_discrete_drive_action_grid(
        current_velocity_mps,
        current_omega_rad_s,
        num_bins=num_bins,
        dt=float(spec.control_dt_s),
        max_linear_velocity=float(max_linear_velocity),
        reverse_velocity_scale=float(reverse_velocity_scale),
        max_linear_accel=float(max_linear_accel),
        max_angular_velocity=float(max_angular_velocity),
        max_angular_accel=float(max_angular_accel),
    )
    path, yaw, times = delayed_unicycle_paths(
        linear, angular, pending_command, spec=spec
    )
    sensor_ranges = policy_lidar_to_sensor_range(
        policy_lidar_clearance,
        max_range=float(spec.lidar_max_range_m),
        body_radius=float(spec.lidar_body_radius_m),
    )
    points, static_valid, dynamic_returns = lidar_static_points(
        sensor_ranges,
        dynamic_positions_body_m,
        dynamic_radii_m,
        dynamic_valid,
        beam_angles_rad=lidar_beam_angles_rad,
        spec=spec,
    )
    static_clearance = _point_obb_clearance_grid(
        path, yaw, points, static_valid, spec
    )
    dynamic_clearance = _dynamic_obb_circle_clearance_grid(
        path,
        yaw,
        times,
        dynamic_positions_body_m,
        dynamic_velocities_body_mps,
        dynamic_radii_m,
        dynamic_valid,
        spec,
    )
    feasible = (
        dynamic_clearance >= float(spec.dynamic_surface_clearance_m)
    ) & (static_clearance >= float(spec.static_surface_clearance_m))

    policy_index = policy_actions.round().long().clamp(0, num_bins - 1)
    indices = torch.arange(num_bins, device=policy_actions.device)
    linear_indices = indices[:, None].expand(num_bins, num_bins)
    angular_indices = indices[None, :].expand(num_bins, num_bins)
    deviation = (
        (linear_indices[None] - policy_index[:, 0, None, None]).abs()
        + (angular_indices[None] - policy_index[:, 1, None, None]).abs()
    )
    infeasible_rank = torch.full_like(deviation, 10_000)
    feasible_deviation = torch.where(feasible, deviation, infeasible_rank)
    minimum_deviation = feasible_deviation.flatten(1).amin(dim=1)
    nearest = feasible & (
        deviation == minimum_deviation[:, None, None]
    )

    dynamic_margin = dynamic_clearance - float(
        spec.dynamic_surface_clearance_m
    )
    static_margin = static_clearance - float(spec.static_surface_clearance_m)
    bottleneck = torch.minimum(dynamic_margin, static_margin)
    capped_bottleneck = torch.nan_to_num(
        bottleneck, posinf=100.0, neginf=-100.0
    ).clamp(-100.0, 100.0)
    nearest_margin = torch.where(
        nearest,
        capped_bottleneck,
        torch.full_like(capped_bottleneck, -float("inf")),
    )
    maximum_margin = nearest_margin.flatten(1).amax(dim=1)
    safest_nearest = nearest & (
        nearest_margin >= maximum_margin[:, None, None] - 1.0e-6
    )
    forward_rank = torch.where(
        safest_nearest,
        linear,
        torch.full_like(linear, -float("inf")),
    )
    maximum_forward = forward_rank.flatten(1).amax(dim=1)
    finalists = safest_nearest & (
        forward_rank >= maximum_forward[:, None, None] - 1.0e-6
    )
    flat_selected = finalists.flatten(1).float().argmax(dim=1)
    any_feasible = feasible.flatten(1).any(dim=1)
    selected = torch.stack(
        [
            torch.div(flat_selected, num_bins, rounding_mode="floor"),
            torch.remainder(flat_selected, num_bins),
        ],
        dim=1,
    )
    selected = torch.where(any_feasible[:, None], selected, policy_index)
    env_index = torch.arange(envs, device=policy_actions.device)
    selected_dynamic = dynamic_clearance[
        env_index, selected[:, 0], selected[:, 1]
    ]
    selected_static = static_clearance[
        env_index, selected[:, 0], selected[:, 1]
    ]
    policy_dynamic = dynamic_clearance[
        env_index, policy_index[:, 0], policy_index[:, 1]
    ]
    policy_static = static_clearance[
        env_index, policy_index[:, 0], policy_index[:, 1]
    ]
    policy_feasible = feasible[
        env_index, policy_index[:, 0], policy_index[:, 1]
    ]
    return {
        "actions": selected,
        "any_feasible": any_feasible,
        "policy_feasible": policy_feasible,
        "feasible_fraction": feasible.float().mean(dim=(1, 2)),
        "selected_dynamic_clearance_m": selected_dynamic,
        "selected_static_clearance_m": selected_static,
        "policy_dynamic_clearance_m": policy_dynamic,
        "policy_static_clearance_m": policy_static,
        "dynamic_clearance_grid_m": dynamic_clearance,
        "static_clearance_grid_m": static_clearance,
        "feasible_grid": feasible,
        "linear_velocity_grid": linear,
        "angular_velocity_grid": angular,
        "dynamic_lidar_returns_removed": dynamic_returns.sum(dim=1),
    }


class GeometryFeasibleSelector:
    """Stateful two-frame-triggered wrapper around the geometry action grid."""

    mode = MODE

    def __init__(
        self,
        *,
        spec: GeometrySelectorSpec = GeometrySelectorSpec(),
        mode: str = MODE,
        require_actual_lidar_angles: bool = False,
    ) -> None:
        _validate_spec(spec)
        if mode not in (MODE, ARGMIN_MODE):
            raise ValueError(f"unsupported geometry selector mode: {mode!r}")
        if require_actual_lidar_angles != (mode == ARGMIN_MODE):
            raise ValueError(
                "actual-angle requirement must be enabled exactly for the argmin mode"
            )
        self.spec = spec
        self.mode = mode
        self.require_actual_lidar_angles = require_actual_lidar_angles
        self._active: torch.Tensor | None = None
        self._confirm: torch.Tensor | None = None
        self._release: torch.Tensor | None = None
        self._tracked_slot: torch.Tensor | None = None
        self._calls = 0
        self._environment_frames = 0
        self._trigger_activations = 0
        self._release_events = 0
        self._active_environment_frames = 0
        self._override_environment_frames = 0
        self._policy_feasible_frames = 0
        self._jointly_feasible_frames = 0
        self._no_feasible_frames = 0
        self._dynamic_lidar_returns_removed = 0
        self._center_jointly_feasible_frames = 0
        self._both_jointly_feasible_frames = 0
        self._argmin_only_feasible_frames = 0
        self._center_only_feasible_frames = 0
        self._both_no_feasible_frames = 0

    def _ensure_state(self, actions: torch.Tensor) -> None:
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("D4 actions must have shape [E,2]")
        envs = int(actions.shape[0])
        if self._active is not None:
            if self._active.shape != (envs,) or self._active.device != actions.device:
                raise RuntimeError("D4 environment shape/device changed")
            return
        self._active = torch.zeros(envs, dtype=torch.bool, device=actions.device)
        self._confirm = torch.zeros(envs, dtype=torch.long, device=actions.device)
        self._release = torch.zeros(envs, dtype=torch.long, device=actions.device)
        self._tracked_slot = torch.zeros(envs, dtype=torch.long, device=actions.device)

    @staticmethod
    def _tensor(context: dict, key: str) -> torch.Tensor:
        value = context.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"D4 context {key!r} must be a tensor")
        return value

    def __call__(self, action_indices: torch.Tensor, context: dict):
        self._ensure_state(action_indices)
        assert self._active is not None
        assert self._confirm is not None
        assert self._release is not None
        assert self._tracked_slot is not None
        envs = int(action_indices.shape[0])
        device = action_indices.device

        distances = self._tensor(context, "obstacle_distances_m").float()
        closings = self._tensor(
            context, "relative_closing_speeds_mps"
        ).float()
        if distances.ndim != 2 or closings.shape != distances.shape:
            raise ValueError("D4 distance/closing context must have shape [E,N]")
        if distances.shape[0] != envs or distances.device != device:
            raise ValueError("D4 distance context has wrong env count/device")
        slots = int(distances.shape[1])
        if not bool(torch.isfinite(distances).all()):
            raise ValueError("D4 distances contain non-finite data")
        if not bool(torch.isfinite(closings).all()):
            raise ValueError("D4 closings contain non-finite data")

        env_index = torch.arange(envs, device=device)
        tracked_distance = distances[env_index, self._tracked_slot]
        tracked_closing = closings[env_index, self._tracked_slot]
        resolved = (
            (tracked_distance >= float(self.spec.release_distance_m))
            | (tracked_closing <= 0.0)
        )
        self._release = torch.where(
            self._active & resolved,
            self._release + 1,
            torch.zeros_like(self._release),
        )
        release_now = self._active & (
            self._release >= int(self.spec.release_confirm_steps)
        )
        self._release_events += int(release_now.sum().item())
        self._active[release_now] = False
        self._release[release_now] = 0
        self._tracked_slot[release_now] = 0

        ttc = torch.where(
            closings > 0.0,
            (distances - float(self.spec.contact_distance_m)).clamp(min=0.0)
            / closings.clamp(min=1.0e-6),
            torch.full_like(distances, float("inf")),
        )
        qualifies = (
            (distances <= float(self.spec.trigger_distance_m))
            & (closings > 0.0)
            & (ttc <= float(self.spec.trigger_ttc_s))
        )
        inactive = ~self._active
        any_qualifies = qualifies.any(dim=1)
        self._confirm = torch.where(
            inactive & any_qualifies,
            (self._confirm + 1).clamp(max=self.spec.trigger_confirm_steps),
            torch.zeros_like(self._confirm),
        )
        trigger_now = inactive & (
            self._confirm >= int(self.spec.trigger_confirm_steps)
        )
        candidate_ttc = torch.where(
            qualifies, ttc, torch.full_like(ttc, float("inf"))
        )
        selected_slot = candidate_ttc.argmin(dim=1).clamp(0, slots - 1)
        self._tracked_slot[trigger_now] = selected_slot[trigger_now]
        self._active[trigger_now] = True
        self._confirm[trigger_now] = 0
        self._trigger_activations += int(trigger_now.sum().item())

        active = self._active.clone()
        effective = action_indices
        if bool(active.any()):
            keys = {
                "current_velocity_mps": "current_velocity_mps",
                "current_omega_rad_s": "current_omega_rad_s",
                "pending_command": "pending_command",
                "policy_lidar_clearance": "policy_lidar_clearance",
                "dynamic_positions_body_m": "dynamic_positions_body_m",
                "dynamic_velocities_body_mps": "dynamic_velocities_body_mps",
                "dynamic_radii_m": "dynamic_radii_m",
                "dynamic_valid": "dynamic_valid",
            }
            if self.require_actual_lidar_angles:
                keys["lidar_beam_angles_rad"] = "lidar_beam_angles_rad"
            values = {name: self._tensor(context, key) for name, key in keys.items()}
            selected = geometry_feasible_action_grid(
                policy_actions=action_indices[active],
                current_velocity_mps=values["current_velocity_mps"][active],
                current_omega_rad_s=values["current_omega_rad_s"][active],
                pending_command=values["pending_command"][active],
                policy_lidar_clearance=values["policy_lidar_clearance"][active],
                lidar_beam_angles_rad=(
                    values["lidar_beam_angles_rad"][active]
                    if self.require_actual_lidar_angles
                    else None
                ),
                dynamic_positions_body_m=values["dynamic_positions_body_m"][active],
                dynamic_velocities_body_mps=values[
                    "dynamic_velocities_body_mps"
                ][active],
                dynamic_radii_m=values["dynamic_radii_m"][active],
                dynamic_valid=values["dynamic_valid"][active].bool(),
                num_bins=int(context["num_bins"]),
                max_linear_velocity=float(context["max_linear_velocity"]),
                reverse_velocity_scale=float(context["reverse_velocity_scale"]),
                max_linear_accel=float(context["max_linear_accel"]),
                max_angular_velocity=float(context["max_angular_velocity"]),
                max_angular_accel=float(context["max_angular_accel"]),
                spec=self.spec,
            )
            if self.require_actual_lidar_angles:
                center_shadow = geometry_feasible_action_grid(
                    policy_actions=action_indices[active],
                    current_velocity_mps=values["current_velocity_mps"][active],
                    current_omega_rad_s=values["current_omega_rad_s"][active],
                    pending_command=values["pending_command"][active],
                    policy_lidar_clearance=values["policy_lidar_clearance"][active],
                    lidar_beam_angles_rad=None,
                    dynamic_positions_body_m=values["dynamic_positions_body_m"][active],
                    dynamic_velocities_body_mps=values[
                        "dynamic_velocities_body_mps"
                    ][active],
                    dynamic_radii_m=values["dynamic_radii_m"][active],
                    dynamic_valid=values["dynamic_valid"][active].bool(),
                    num_bins=int(context["num_bins"]),
                    max_linear_velocity=float(context["max_linear_velocity"]),
                    reverse_velocity_scale=float(context["reverse_velocity_scale"]),
                    max_linear_accel=float(context["max_linear_accel"]),
                    max_angular_velocity=float(context["max_angular_velocity"]),
                    max_angular_accel=float(context["max_angular_accel"]),
                    spec=self.spec,
                )
                actual_any = selected["any_feasible"]
                center_any = center_shadow["any_feasible"]
                self._center_jointly_feasible_frames += int(
                    center_any.sum().item()
                )
                self._both_jointly_feasible_frames += int(
                    (actual_any & center_any).sum().item()
                )
                self._argmin_only_feasible_frames += int(
                    (actual_any & ~center_any).sum().item()
                )
                self._center_only_feasible_frames += int(
                    (~actual_any & center_any).sum().item()
                )
                self._both_no_feasible_frames += int(
                    (~actual_any & ~center_any).sum().item()
                )
            chosen = selected["actions"].to(dtype=action_indices.dtype)
            changed_active = (
                chosen.round().long()
                != action_indices[active].round().long()
            ).any(dim=1)
            if bool(changed_active.any()):
                effective = action_indices.clone()
                effective[active] = chosen
            self._override_environment_frames += int(changed_active.sum().item())
            self._policy_feasible_frames += int(
                selected["policy_feasible"].sum().item()
            )
            self._jointly_feasible_frames += int(
                selected["any_feasible"].sum().item()
            )
            self._no_feasible_frames += int(
                (~selected["any_feasible"]).sum().item()
            )
            self._dynamic_lidar_returns_removed += int(
                selected["dynamic_lidar_returns_removed"].sum().item()
            )

        self._calls += 1
        self._environment_frames += envs
        self._active_environment_frames += int(active.sum().item())
        return effective

    def reset(self, done_mask: torch.Tensor) -> None:
        if self._active is None:
            return
        done = done_mask.reshape(-1).to(
            device=self._active.device, dtype=torch.bool
        )
        if done.shape != self._active.shape:
            raise ValueError("D4 reset mask has wrong shape")
        self._active[done] = False
        self._confirm[done] = 0
        self._release[done] = 0
        self._tracked_slot[done] = 0

    def report(self) -> dict:
        protocol = (
            geometry_argmin_selector_protocol(self.spec)
            if self.require_actual_lidar_angles
            else geometry_selector_protocol(self.spec)
        )
        return {
            "mode": self.mode,
            "lidar_angle_source": (
                "winning_raw_ray" if self.require_actual_lidar_angles else "bin_center"
            ),
            "calls": self._calls,
            "environment_frames": self._environment_frames,
            "trigger_activations": self._trigger_activations,
            "release_events": self._release_events,
            "active_environment_frames": self._active_environment_frames,
            "override_environment_frames": self._override_environment_frames,
            "policy_feasible_active_frames": self._policy_feasible_frames,
            "jointly_feasible_active_frames": self._jointly_feasible_frames,
            "no_feasible_active_frames": self._no_feasible_frames,
            "dynamic_lidar_returns_removed": self._dynamic_lidar_returns_removed,
            "paired_center_shadow_active_frames": (
                self._active_environment_frames
                if self.require_actual_lidar_angles
                else 0
            ),
            "center_jointly_feasible_active_frames": (
                self._center_jointly_feasible_frames
            ),
            "both_jointly_feasible_active_frames": (
                self._both_jointly_feasible_frames
            ),
            "argmin_only_feasible_active_frames": (
                self._argmin_only_feasible_frames
            ),
            "center_only_feasible_active_frames": (
                self._center_only_feasible_frames
            ),
            "both_no_feasible_active_frames": self._both_no_feasible_frames,
            "active_at_end": (
                0 if self._active is None else int(self._active.sum().item())
            ),
            "protocol_sha256": protocol["sha256"],
        }


def build_geometry_selector(
    *, spec: GeometrySelectorSpec = GeometrySelectorSpec()
) -> GeometryFeasibleSelector:
    return GeometryFeasibleSelector(spec=spec)


def build_argmin_geometry_selector(
    *, spec: GeometrySelectorSpec = GeometrySelectorSpec()
) -> GeometryFeasibleSelector:
    return GeometryFeasibleSelector(
        spec=spec,
        mode=ARGMIN_MODE,
        require_actual_lidar_angles=True,
    )


__all__ = [
    "GeometryFeasibleSelector",
    "GeometrySelectorSpec",
    "ARGMIN_MODE",
    "ARGMIN_PROTOCOL_SCHEMA",
    "MODE",
    "PROTOCOL_SCHEMA",
    "build_argmin_geometry_selector",
    "build_geometry_selector",
    "delayed_unicycle_paths",
    "geometry_argmin_selector_protocol",
    "geometry_feasible_action_grid",
    "geometry_selector_protocol",
    "lidar_static_points",
    "pending_d1_command",
    "validate_action_contract",
]
