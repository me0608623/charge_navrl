"""Run one frozen SA6-v3 checkpoint-screen cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import sa6_v3_checkpoint_screen as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


def stage_values() -> dict:
    return dict(protocol.screen_protocol()["fixed_evaluation"]["scene"])


def _common_args(values: dict) -> list[str]:
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
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
    ]


def _build_corridor_args(
    spec: dict,
    corridor_path: Path,
    *,
    use_count_mix: bool,
) -> list[str]:
    values = stage_values()
    speed_lo, speed_hi = spec["speed_range_m_s"]
    motion_mode = str(spec["motion_mode"])
    if motion_mode not in (*protocol.CORRIDOR_FAMILIES, "env_stratified"):
        raise ValueError(f"unsupported corridor motion mode {motion_mode!r}")
    actual_dynamic = int(spec["dynamic_obstacles"])
    cli_dynamic = min(actual_dynamic, 2) if use_count_mix else actual_dynamic
    args = [
        *_common_args(values),
        "--long_corridor_eval",
        "--long_corridor_free_width",
        f"{values['corridor_free_width_m']:g}",
        "--long_corridor_dynamic_speed_range",
        f"{speed_lo:g}",
        f"{speed_hi:g}",
        "--long_corridor_static_obstacles",
        str(spec["static_obstacles"]),
        "--long_corridor_dynamic_obstacles",
        str(cli_dynamic),
        "--long_corridor_motion_mode",
        motion_mode,
        "--long_corridor_random_2d_kinematics",
        values["random_2d_kinematics"],
        "--long_corridor_pause_mode",
        "default",
        "--long_corridor_output",
        str(corridor_path),
        *fixed_actuator_cli_args(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
    ]
    if use_count_mix:
        count_mix = f"{spec['static_obstacles']},{actual_dynamic}:1.0"
        insertion = args.index("--long_corridor_motion_mode")
        args[insertion:insertion] = ["--long_corridor_count_mix", count_mix]
    return args


def build_phase_a_args(spec: dict, corridor_path: Path) -> list[str]:
    installation = str(spec["installation"])
    if installation == "single-entry production count-mix":
        return _build_corridor_args(spec, corridor_path, use_count_mix=True)
    if installation == "legacy fixed-count 4S2D gate path":
        return _build_corridor_args(spec, corridor_path, use_count_mix=False)
    raise ValueError(f"unknown corridor installation {installation!r}")


def build_phase_b_args(scenario: str, corridor_path: Path) -> list[str]:
    values = stage_values()
    actuator = fixed_actuator_cli_args(
        protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    common = _common_args(values)
    if scenario == "nav_native":
        return [*common, *actuator]
    if scenario == "narrow_range":
        width_lo, width_hi = values["narrow_width_range_m"]
        return [
            *common,
            "--num_static_obs",
            "0",
            "--num_dynamic_obs",
            "0",
            "--num_walls",
            "0",
            "--narrow_replay_eval",
            "--narrow_replay_width_range",
            f"{width_lo:g}",
            f"{width_hi:g}",
            "--narrow_replay_yaw_limit_deg",
            f"{values['narrow_yaw_limit_deg']:g}",
            "--narrow_replay_segment_length",
            f"{values['narrow_segment_length_m']:g}",
            *actuator,
        ]
    profile = protocol.retention_profile_by_scenario(scenario)
    spec = {
        **profile,
        "motion_mode": "lateral",
        "installation": "legacy fixed-count 4S2D gate path",
    }
    return _build_corridor_args(spec, corridor_path, use_count_mix=False)


def verify_speed_rate_runtime(log_path: Path, report: dict | None) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"\[VEHICLE-SPEED-RATE\] rate=([\d.]+) obs=(\S+) "
        r"lidar_scaled=(\S+) deployment_scale=([\d.]+)",
        text,
    )
    if match is None:
        raise RuntimeError("vehicle speed-rate runtime banner is missing")
    observed_rate, observed_obs, lidar_scaled, deployment_scale = match.groups()
    if abs(float(observed_rate) - protocol.SPEED_RATE) > 1e-9:
        raise RuntimeError("vehicle speed-rate runtime mismatch")
    if observed_obs != protocol.SPEED_RATE_OBS:
        raise RuntimeError("vehicle speed-rate observation mode mismatch")
    if lidar_scaled != "False":
        raise RuntimeError("LiDAR was unexpectedly scaled by vehicle speed rate")
    if abs(float(deployment_scale) - protocol.DEPLOYMENT_SPEED_SCALE) > 1e-9:
        raise RuntimeError("downstream deployment scaling was stacked")

    result = {
        "speed_rate": float(observed_rate),
        "speed_rate_obs": observed_obs,
        "lidar_scaled": False,
        "deployment_speed_scale": float(deployment_scale),
    }
    if report is not None:
        error = float(report.get("deployment_scale_max_abs_error", float("nan")))
        samples = int(report.get("deployment_scale_samples", 0))
        if not math.isfinite(error) or error > 1e-6 or samples <= 0:
            raise RuntimeError("deployment output failed speed-rate reconciliation")
        if abs(float(report.get("speed_rate", float("nan"))) - protocol.SPEED_RATE) > 1e-9:
            raise RuntimeError("corridor report speed-rate mismatch")
        if report.get("speed_rate_obs_mode") != protocol.SPEED_RATE_OBS:
            raise RuntimeError("corridor report speed-rate observation mismatch")
        result["deployment_scale_samples"] = samples
        result["deployment_scale_max_abs_error"] = error
        result["action_limits"] = report.get("speed_rate_action_limits")
    return result


def verify_corridor_runtime(
    report: dict,
    spec: dict,
    values: dict,
    log_path: Path,
    steps: int,
) -> dict:
    problems: list[str] = []
    actual_dynamic = int(spec["dynamic_obstacles"])
    use_count_mix = spec["installation"] == "single-entry production count-mix"
    expected_scalars = {
        "requested_free_width_m": values["corridor_free_width_m"],
        "requested_length_m": values["corridor_length_m"],
        "requested_wall_span_m": values["arena_size_m"],
        "configured_static_obstacles": spec["static_obstacles"],
        "configured_dynamic_obstacles": (
            min(actual_dynamic, 2) if use_count_mix else actual_dynamic
        ),
        "static_obstacles_per_env": spec["static_obstacles"],
        "dynamic_obstacles_per_env": actual_dynamic,
    }
    for key, expected in expected_scalars.items():
        observed = float(report.get(key, float("nan")))
        if not math.isfinite(observed) or abs(observed - float(expected)) > 1e-6:
            problems.append(f"{key} mismatch")
    observed_speed = [
        float(value)
        for value in report.get("requested_dynamic_speed_range_m_s", [])
    ]
    if observed_speed != [float(value) for value in spec["speed_range_m_s"]]:
        problems.append("dynamic speed range mismatch")
    if report.get("dynamic_motion_mode") != spec["motion_mode"]:
        problems.append("motion mode mismatch")
    if report.get("random_2d_kinematics") != values["random_2d_kinematics"]:
        problems.append("random-2D kinematics mismatch")
    for key in (
        "geometry_pass",
        "goal_alignment_pass",
        "speed_upper_bound_pass",
        "sealed_to_boundary_pass",
    ):
        if not bool(report.get(key)):
            problems.append(f"{key}=false")
    if int(report.get("constructive_unsolvable_count", -1)) != 0:
        problems.append("constructive_unsolvable_count != 0")
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    fractions = report.get("dynamic_motion_type_fractions") or {}
    audited = int(report.get("penetration_audited_slot_frames", -1))
    expected_audited = int(steps) * protocol.NUM_ENVS * actual_dynamic
    if audited != expected_audited:
        problems.append(
            f"active-slot accounting mismatch ({audited} != {expected_audited})"
        )
    if int(report.get("corridor_active_not_ready_frames", -1)) != 0:
        problems.append("corridor_active_not_ready_frames != 0")

    if use_count_mix:
        count_mix_marker = (
            f"((({spec['static_obstacles']}, {actual_dynamic}), 1.0),)"
        )
        if "count-mix:" not in log_text or count_mix_marker not in log_text:
            problems.append("exact production count-mix marker missing")
        capacity = int(
            protocol.screen_protocol()["fixed_evaluation"]
            ["count_mix_runtime_audit"]["production_dynamic_slot_capacity"]
        )
        expected_moved = actual_dynamic / capacity
        observed_moved = float(
            report.get("dynamic_slots_moved_fraction", float("nan"))
        )
        if not math.isfinite(observed_moved) or abs(
            observed_moved - expected_moved
        ) > 1e-5:
            problems.append("production active-slot movement accounting mismatch")
        if not all(
            float(fractions.get(name, 0.0)) > 0.0
            for name in ("lateral", "longitudinal", "random_2d")
        ):
            problems.append("env-stratified profile missed a motion family")
    else:
        for key in (
            "movement_pass",
            "motion_mode_pass",
            "penetration_pass",
            "obstacle_mix_pass",
        ):
            if not bool(report.get(key)):
                problems.append(f"{key}=false")
        family = str(spec["motion_mode"])
        if family in ("lateral", "longitudinal", "random_2d"):
            if abs(float(fractions.get(family, 0.0)) - 1.0) > 1e-6:
                problems.append(f"{family} motion fraction != 1")
        elif not all(
            float(fractions.get(name, 0.0)) > 0.0
            for name in ("lateral", "longitudinal", "random_2d")
        ):
            problems.append("mixed cell did not observe every motion family")
    if problems:
        raise RuntimeError(
            f"SA6 corridor runtime contract failed for "
            f"{spec['label']}: {problems}"
        )
    return {
        "profile": spec["source_profile"],
        "density": spec["density"],
        "static_obstacles": spec["static_obstacles"],
        "dynamic_obstacles": actual_dynamic,
        "pedestrian_speed_range_m_s": list(spec["speed_range_m_s"]),
        "motion_mode": spec["motion_mode"],
        "installation": spec["installation"],
        "active_slot_accounting_pass": audited == expected_audited,
        "legacy_audit_dynamic_slots": min(actual_dynamic, 2),
        "obstacle_mix_pass": bool(report.get("obstacle_mix_pass")),
        "movement_pass": bool(report.get("movement_pass")),
        "motion_mode_pass": bool(report.get("motion_mode_pass")),
        "penetration_pass": bool(report.get("penetration_pass")),
        "dynamic_static_penetration_count": int(
            report.get("dynamic_static_penetration_count", -1)
        ),
        "production_profile_audit_limitation": (
            "legacy fixed-slot gates are descriptive only"
            if use_count_mix
            else None
        ),
        "motion_type_fractions": fractions,
        "free_width_m": float(report["free_width_m_mean"]),
        "length_m": float(report["length_m_mean"]),
        "wall_span_m": float(report["wall_span_m_mean"]),
        "wall_boundary_overlap_m": float(report["wall_boundary_overlap_m_min"]),
        "sealed_to_boundary": bool(report["sealed_to_boundary_pass"]),
    }


def _paths(output: Path) -> tuple[Path, Path, Path]:
    return output / "run.log", output / "corridor.json", output / "cell.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--fixed-cell")
    parser.add_argument("--scenario", choices=protocol.PHASE_B_SCENARIOS)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args(argv)

    frozen = protocol.screen_protocol()
    candidate = protocol.candidate_by_name(args.checkpoint_name)
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != candidate["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 128:
        raise ValueError("smoke steps must be within [1, 128]")
    steps = args.smoke_steps or args.steps
    is_smoke = args.smoke_steps is not None

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint != protocol.checkpoint_path(candidate).resolve():
        raise ValueError("checkpoint path differs from frozen lineage")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.sha256_of(checkpoint)
    if digest != candidate["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")

    if args.phase == "a":
        if not args.fixed_cell or args.scenario:
            raise ValueError("Phase A requires --fixed-cell only")
        spec = protocol.phase_a_spec_by_label(args.fixed_cell)
        profile = spec
        family = spec["motion_mode"]
        scenario = None
        cell_label = protocol.phase_a_cell_label(candidate["name"], spec["label"])
        expected_steps = protocol.phase_a_steps(spec["label"])
    else:
        if not args.scenario or args.fixed_cell:
            raise ValueError("Phase B requires --scenario only")
        spec = None
        scenario = args.scenario
        profile = (
            protocol.retention_profile_by_scenario(scenario)
            if scenario.startswith("retention_")
            else None
        )
        family = "lateral" if profile is not None else None
        cell_label = protocol.phase_b_cell_label(candidate["name"], scenario)
        expected_steps = protocol.phase_b_steps(scenario)
    if args.steps != expected_steps:
        raise ValueError(
            f"formal cell {cell_label} requires {expected_steps} steps, "
            f"got {args.steps}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path, corridor_path, cell_path = _paths(output)
    targets = [log_path, cell_path]
    if profile is not None:
        targets.append(corridor_path)
    d9_runner._require_new_targets(targets)
    print(
        f"[SA6-V3-SCREEN-CELL] phase={args.phase} cell={cell_label} "
        f"steps={steps} smoke={is_smoke} sha={digest[:12]}",
        flush=True,
    )
    extra = (
        build_phase_a_args(spec, corridor_path)
        if args.phase == "a"
        else build_phase_b_args(str(scenario), corridor_path)
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=protocol.NUM_ENVS,
        steps=steps,
        extra=extra,
    )

    d9_runner._validate_runtime_log(
        log_path, protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    values = stage_values()
    stage_banner = base.verify_common_runtime(log_path, values)
    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")

    report = None
    if profile is not None:
        report = json.loads(corridor_path.read_text(encoding="utf-8"))
        runtime_spec = spec or {
            **profile,
            "label": scenario,
            "source_profile": profile["label"],
            "motion_mode": family,
            "installation": "legacy fixed-count 4S2D gate path",
        }
        runtime = verify_corridor_runtime(
            report, runtime_spec, values, log_path, steps
        )
        metrics = base.reconcile_corridor_metrics(dict(summary), report)
        metric_scenario = "corridor" if args.phase == "a" else str(scenario)
    elif scenario == "narrow_range":
        metrics = base.reconcile_console_metrics(
            dict(summary), base.parse_exact_outcome_counts(log_path)
        )
        narrow = base.parse_narrow_replay_metrics(log_path)
        if int(narrow["episodes"]) != int(metrics["n"]):
            raise RuntimeError("narrow episode ledgers disagree")
        metrics.update(
            {key: value for key, value in narrow.items() if key != "episodes"}
        )
        runtime = base.verify_narrow_runtime(log_path, values)
        metric_scenario = str(scenario)
    else:
        metrics = base.reconcile_console_metrics(
            dict(summary), base.parse_exact_outcome_counts(log_path)
        )
        runtime = {
            "scenario": scenario,
            "mechanism": "stage scene via STAGE_PARAMETER",
        }
        metric_scenario = str(scenario)

    for key in ("sr", "cr", "to"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite {key}")
    if not is_smoke and int(metrics["n"]) < protocol.MIN_EPISODES:
        raise RuntimeError(
            f"cell completed only {metrics['n']} episodes; "
            f"minimum is {protocol.MIN_EPISODES}"
        )
    speed_rate_runtime = verify_speed_rate_runtime(log_path, report)
    gate = protocol.evaluate_metrics(metric_scenario, metrics)

    cell = {
        "schema": (
            protocol.PHASE_A_CELL_SCHEMA
            if args.phase == "a"
            else protocol.PHASE_B_CELL_SCHEMA
        ),
        "cell_valid": not is_smoke,
        "formal_evidence": not is_smoke,
        "cell": cell_label,
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": steps,
        "delay_steps": protocol.DELAY_STEPS,
        "delay_ms": protocol.DELAY_MS,
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "actuator_eval": fixed_actuator_metadata(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
        "stage_banner": stage_banner,
        "speed_rate_runtime": speed_rate_runtime,
        "metrics": metrics,
        "gate": gate,
        "runtime": runtime,
        "corridor_report": report,
        "log": str(log_path),
    }
    if args.phase == "a":
        cell.update(
            {
                "fixed_cell": spec["label"],
                "cell_kind": spec["cell_kind"],
            }
        )
    else:
        cell.update({"scenario": scenario})
    if is_smoke:
        cell["schema"] = "sa6_v3_checkpoint_screen_smoke/v1"
    elif args.phase == "a":
        protocol.validate_phase_a_cell(cell)
    else:
        protocol.validate_phase_b_cell(cell, candidate["name"])
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA6-V3-SCREEN-CELL] {cell_label} "
        f"{'PASS' if gate['pass'] else 'FAIL'} n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
