"""Run one frozen SA4-v3 checkpoint-screen cell."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import sa4_v3_checkpoint_screen as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


@contextmanager
def _stage4_enabled():
    """Temporarily extend the frozen SA3 helper without leaking state."""
    original = base.SUPPORTED_GEOMETRY_STAGES
    base.SUPPORTED_GEOMETRY_STAGES = tuple(sorted(set(original) | {protocol.STAGE}))
    try:
        yield
    finally:
        base.SUPPORTED_GEOMETRY_STAGES = original


def scene_values() -> dict:
    with _stage4_enabled():
        return base.scene_values(protocol.STAGE)


def build_scene_args(spec: dict, corridor_path: Path) -> list[str]:
    values = scene_values()
    actuator = fixed_actuator_cli_args(
        protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    speed_lo, speed_hi = spec["pedestrian_speed_range_m_s"]
    return [
        "--stage",
        str(protocol.STAGE),
        "--vlp16_noise_mode",
        protocol.LIDAR_NOISE_MODE,
        "--lidar-distractor-eligibility",
        protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "--arena_size",
        f"{values['arena_size_m']:g}",
        "--obs_near_goal_count",
        "0",
        "--seed",
        str(protocol.SEED),
        "--long_corridor_eval",
        "--long_corridor_free_width",
        f"{values['corridor_free_width_m']:g}",
        "--long_corridor_dynamic_speed_range",
        f"{speed_lo:g}",
        f"{speed_hi:g}",
        "--long_corridor_static_obstacles",
        str(spec["static_obstacles"]),
        "--long_corridor_dynamic_obstacles",
        str(spec["dynamic_obstacles"]),
        "--long_corridor_motion_mode",
        protocol.MOTION_MODE,
        "--long_corridor_random_2d_kinematics",
        "patrol",
        "--long_corridor_pause_mode",
        "default",
        "--long_corridor_output",
        str(corridor_path),
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
        *actuator,
    ]


def verify_speed_rate_runtime(report: dict) -> dict:
    observed = float(report.get("speed_rate", float("nan")))
    error = float(report.get("deployment_scale_max_abs_error", float("nan")))
    samples = int(report.get("deployment_scale_samples", 0))
    if not math.isfinite(observed) or abs(observed - protocol.SPEED_RATE) > 1e-9:
        raise RuntimeError("vehicle speed-rate runtime mismatch")
    if report.get("speed_rate_obs_mode") != protocol.SPEED_RATE_OBS:
        raise RuntimeError("vehicle speed-rate observation mode mismatch")
    if (
        float(report.get("deployment_speed_scale", float("nan")))
        != protocol.DEPLOYMENT_SPEED_SCALE
    ):
        raise RuntimeError("downstream deployment scaling was stacked")
    if not math.isfinite(error) or error > 1e-6 or samples <= 0:
        raise RuntimeError("deployment output failed reconciliation")
    limits = report.get("speed_rate_action_limits") or {}
    for key in (
        "max_linear_velocity",
        "max_linear_accel",
        "max_angular_vel",
        "max_angular_accel",
    ):
        if key not in limits or not math.isfinite(float(limits[key])):
            raise RuntimeError(f"speed-rate runtime lacks {key}")
    return {
        "speed_rate": observed,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "action_limits": limits,
        "samples": samples,
        "identity_deployment_max_abs_error": error,
    }


def verify_corridor_runtime(report: dict, spec: dict, values: dict) -> dict:
    problems: list[str] = []
    expected_scalars = {
        "requested_free_width_m": values["corridor_free_width_m"],
        "requested_length_m": values["corridor_length_m"],
        "requested_wall_span_m": values["arena_size_m"],
        "configured_static_obstacles": spec["static_obstacles"],
        "configured_dynamic_obstacles": spec["dynamic_obstacles"],
        "static_obstacles_per_env": spec["static_obstacles"],
        "dynamic_obstacles_per_env": spec["dynamic_obstacles"],
    }
    for key, expected in expected_scalars.items():
        observed = float(report.get(key, float("nan")))
        if not math.isfinite(observed) or abs(observed - float(expected)) > 1e-6:
            problems.append(f"{key} mismatch")
    if [
        float(value)
        for value in report.get("requested_dynamic_speed_range_m_s", [])
    ] != [float(value) for value in spec["pedestrian_speed_range_m_s"]]:
        problems.append("dynamic speed range mismatch")
    if report.get("dynamic_motion_mode") != protocol.MOTION_MODE:
        problems.append("motion mode mismatch")
    fractions = report.get("dynamic_motion_type_fractions") or {}
    if abs(float(fractions.get(protocol.MOTION_MODE, 0.0)) - 1.0) > 1e-6:
        problems.append("lateral motion fraction != 1")
    for key in (
        "geometry_pass",
        "movement_pass",
        "motion_mode_pass",
        "obstacle_mix_pass",
        "goal_alignment_pass",
        "penetration_pass",
        "speed_upper_bound_pass",
        "sealed_to_boundary_pass",
    ):
        if not bool(report.get(key)):
            problems.append(f"{key}=false")
    if int(report.get("constructive_unsolvable_count", -1)) != 0:
        problems.append("constructive_unsolvable_count != 0")
    if problems:
        raise RuntimeError(
            f"SA4-v3 runtime contract failed for {spec['label']}: {problems}"
        )
    return {
        "fixed_cell": spec["label"],
        "density": spec["density"],
        "motion_mode": protocol.MOTION_MODE,
        "free_width_m": float(report["free_width_m_mean"]),
        "length_m": float(report["length_m_mean"]),
        "wall_span_m": float(report["wall_span_m_mean"]),
        "wall_boundary_overlap_m": float(report["wall_boundary_overlap_m_min"]),
        "sealed_to_boundary": bool(report["sealed_to_boundary_pass"]),
        "observed_max_dynamic_speed_m_s": float(
            report["observed_max_dynamic_speed_m_s"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--fixed-cell", required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args(argv)

    frozen = protocol.screen_protocol()
    candidate = protocol.candidate_by_name(args.checkpoint_name)
    spec = protocol.fixed_cell_by_label(args.fixed_cell)
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != candidate["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 128:
        raise ValueError("smoke steps must be within [1, 128]")
    steps = args.smoke_steps or protocol.STEPS
    is_smoke = args.smoke_steps is not None

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint != protocol.checkpoint_path(candidate).resolve():
        raise ValueError("checkpoint path differs from frozen lineage")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.sha256_of(checkpoint)
    if digest != candidate["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{steps}" if is_smoke else ""
    stem = (
        f"{candidate['name']}_{spec['density'].lower()}_"
        f"{spec['pedestrian_speed_label'].lower()}_lateral_s070_"
        f"g4_d{protocol.DELAY_STEPS}_s{protocol.SEED}{suffix}"
    )
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    d9_runner._require_new_targets([log_path, corridor_path, cell_path])

    print(
        f"[SA4-V3-SCREEN-CELL] checkpoint={candidate['name']} "
        f"fixed_cell={spec['label']} steps={steps} smoke={is_smoke} "
        f"sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=protocol.NUM_ENVS,
        steps=steps,
        extra=build_scene_args(spec, corridor_path),
    )

    d9_runner._validate_runtime_log(
        log_path, protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    values = scene_values()
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")
    report = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = verify_corridor_runtime(report, spec, values)
    speed_rate_runtime = verify_speed_rate_runtime(report)
    metrics = base.reconcile_corridor_metrics(dict(summary), report)
    if not is_smoke and int(metrics["n"]) < protocol.MIN_EPISODES:
        raise RuntimeError(
            f"cell completed only {metrics['n']} episodes; "
            f"minimum is {protocol.MIN_EPISODES}"
        )

    gate = protocol.evaluate_metrics(metrics)
    cell = {
        "schema": "sa4_v3_checkpoint_screen_cell/v1",
        "cell_valid": not is_smoke,
        "formal_evidence": not is_smoke,
        "cell": protocol.cell_label(candidate["name"], spec["label"]),
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "fixed_cell": spec["label"],
        "density": spec["density"],
        "pedestrian_speed_label": spec["pedestrian_speed_label"],
        "scenario": spec["scenario"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": steps,
        "delay_steps": protocol.DELAY_STEPS,
        "actuator_eval": fixed_actuator_metadata(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "pedestrian_speed_range_m_s": list(
            spec["pedestrian_speed_range_m_s"]
        ),
        "motion_mode": protocol.MOTION_MODE,
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "speed_rate_runtime": speed_rate_runtime,
        "metrics": metrics,
        "gate": gate,
        "corridor_report": report,
        "log": str(log_path),
    }
    if is_smoke:
        cell["schema"] = "sa4_v3_checkpoint_screen_smoke/v1"
    else:
        protocol.validate_cell(cell)
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-V3-SCREEN-CELL] {cell['cell']} "
        f"{'PASS' if gate['pass'] else 'FAIL'} n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
