"""Run one frozen SA5-R2 checkpoint-screen cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import sa5_r2_checkpoint_screen as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


MOTION_MODE = {
    "corridor_lateral": "lateral",
    "corridor_longitudinal": "longitudinal",
    "corridor_random2d": "random_2d",
    "corridor_mixed": "mixed",
    "corridor_low_0s1d": "mixed",
    "corridor_low_1s1d": "mixed",
}

DENSITY = {
    "corridor_low_0s1d": (0, 1),
    "corridor_low_1s1d": (1, 1),
}


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
    ]


def build_scene_args(
    scenario: str,
    corridor_json: Path,
    *,
    dynamic_speed_range: tuple[float, float] | None = None,
    corridor_motion_mode: str | None = None,
) -> list[str]:
    values = stage_values()
    actuator = fixed_actuator_cli_args(
        protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    common = _common_args(values)
    if scenario == "nav_native":
        return [*common, *actuator]
    if scenario == "narrow_range":
        width = values["narrow_width_range_m"]
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
            f"{width[0]:g}",
            f"{width[1]:g}",
            "--narrow_replay_yaw_limit_deg",
            f"{values['narrow_yaw_limit_deg']:g}",
            "--narrow_replay_segment_length",
            f"{values['narrow_segment_length_m']:g}",
            *actuator,
        ]
    if scenario in protocol.CORRIDOR_SCENARIOS + protocol.LOW_DENSITY_SCENARIOS:
        speed = (
            values["corridor_speed_range_m_s"]
            if dynamic_speed_range is None
            else dynamic_speed_range
        )
        if len(speed) != 2 or not (0.0 < float(speed[0]) <= float(speed[1])):
            raise ValueError("dynamic speed range must be positive and ordered")
        motion_mode = MOTION_MODE[scenario]
        if corridor_motion_mode is not None:
            motion_mode = str(corridor_motion_mode)
            if motion_mode not in ("lateral", "longitudinal", "random_2d", "mixed"):
                raise ValueError(f"unsupported corridor motion mode {motion_mode!r}")
        static_count, dynamic_count = DENSITY.get(
            scenario,
            (
                int(values["corridor_static_obstacles"]),
                int(values["corridor_dynamic_obstacles"]),
            ),
        )
        return [
            *common,
            "--long_corridor_eval",
            "--long_corridor_free_width",
            f"{values['corridor_free_width_m']:g}",
            "--long_corridor_dynamic_speed_range",
            f"{speed[0]:g}",
            f"{speed[1]:g}",
            "--long_corridor_static_obstacles",
            str(static_count),
            "--long_corridor_dynamic_obstacles",
            str(dynamic_count),
            "--long_corridor_motion_mode",
            motion_mode,
            "--long_corridor_random_2d_kinematics",
            values["random_2d_kinematics"],
            "--long_corridor_pause_mode",
            "default",
            "--long_corridor_output",
            str(corridor_json),
            *actuator,
        ]
    raise ValueError(f"unknown scenario {scenario!r}")


def verify_corridor_runtime(
    report: dict,
    scenario: str,
    values: dict,
    *,
    corridor_motion_mode: str | None = None,
) -> dict:
    mode = MOTION_MODE[scenario]
    if corridor_motion_mode is not None:
        mode = str(corridor_motion_mode)
    static_count, dynamic_count = DENSITY.get(
        scenario,
        (
            int(values["corridor_static_obstacles"]),
            int(values["corridor_dynamic_obstacles"]),
        ),
    )
    problems = []
    expected_scalars = {
        "requested_free_width_m": values["corridor_free_width_m"],
        "requested_length_m": values["corridor_length_m"],
        "requested_wall_span_m": values["arena_size_m"],
        "static_obstacles_per_env": static_count,
        "dynamic_obstacles_per_env": dynamic_count,
    }
    for key, expected in expected_scalars.items():
        if abs(float(report.get(key, float("nan"))) - float(expected)) > 1e-6:
            problems.append(f"{key} mismatch")
    if [float(value) for value in report.get("requested_dynamic_speed_range_m_s", [])] != [
        float(value) for value in values["corridor_speed_range_m_s"]
    ]:
        problems.append("dynamic speed range mismatch")
    if report.get("dynamic_motion_mode") != mode:
        problems.append("dynamic motion mode mismatch")
    if report.get("random_2d_kinematics") != values["random_2d_kinematics"]:
        problems.append("random-2D kinematics mismatch")
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

    fractions = report.get("dynamic_motion_type_fractions") or {}
    if mode in ("lateral", "longitudinal", "random_2d"):
        if abs(float(fractions.get(mode, 0.0)) - 1.0) > 1e-6:
            problems.append(f"{mode} motion fraction != 1")
    elif not all(float(fractions.get(name, 0.0)) > 0.0 for name in (
        "lateral", "longitudinal", "random_2d"
    )):
        problems.append("mixed cell did not observe every motion family")
    if problems:
        raise RuntimeError(
            f"SA5 corridor runtime contract failed for {scenario}: {problems}"
        )
    return {
        "scenario": scenario,
        "fixed_density": f"{static_count}S{dynamic_count}D",
        "motion_mode": mode,
        "motion_type_fractions": fractions,
        "free_width_m": float(report["free_width_m_mean"]),
        "length_m": float(report["length_m_mean"]),
        "wall_span_m": float(report["wall_span_m_mean"]),
        "wall_boundary_overlap_m": float(
            report["wall_boundary_overlap_m_min"]
        ),
        "sealed_to_boundary": bool(report["sealed_to_boundary_pass"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--scenario", choices=protocol.ALL_SCENARIOS, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)

    candidate = protocol.candidate_by_name(args.checkpoint_name)
    frozen = protocol.screen_protocol()
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != candidate["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    expected_steps = protocol.STEPS_BY_SCENARIO[args.scenario]
    if args.steps != expected_steps:
        raise ValueError(
            f"{args.scenario} requires {expected_steps} steps, got {args.steps}"
        )

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
    stem = f"{args.scenario}_g5_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    targets = [log_path, cell_path]
    if args.scenario in protocol.CORRIDOR_SCENARIOS + protocol.LOW_DENSITY_SCENARIOS:
        targets.append(corridor_path)
    d9_runner._require_new_targets(targets)

    values = stage_values()
    print(
        f"[SA5-R2-SCREEN-CELL] checkpoint={candidate['name']} "
        f"scenario={args.scenario} steps={args.steps} sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=protocol.NUM_ENVS,
        steps=args.steps,
        extra=build_scene_args(args.scenario, corridor_path),
    )
    d9_runner._validate_runtime_log(
        log_path, protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    metrics = dict(summary)
    corridor_report = None
    if args.scenario in protocol.CORRIDOR_SCENARIOS + protocol.LOW_DENSITY_SCENARIOS:
        corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
        scene_contract = verify_corridor_runtime(
            corridor_report, args.scenario, values
        )
        metrics = base.reconcile_corridor_metrics(metrics, corridor_report)
    elif args.scenario == "narrow_range":
        narrow = base.parse_narrow_replay_metrics(log_path)
        exact = base.parse_exact_outcome_counts(log_path)
        metrics = base.reconcile_console_metrics(metrics, exact)
        if int(narrow["episodes"]) != int(metrics["n"]):
            raise RuntimeError("narrow episode ledgers disagree")
        metrics.update({key: value for key, value in narrow.items() if key != "episodes"})
        scene_contract = base.verify_narrow_runtime(log_path, values)
    else:
        metrics = base.reconcile_console_metrics(
            metrics, base.parse_exact_outcome_counts(log_path)
        )
        scene_contract = {
            "scenario": args.scenario,
            "mechanism": "stage scene via STAGE_PARAMETER",
        }
    for key in ("sr", "cr", "to"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite {args.scenario} {key}")

    verdict = protocol.evaluate_metrics(args.scenario, metrics)
    cell = {
        "schema": "sa5_r2_checkpoint_screen_cell/v1",
        "cell_valid": True,
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "geometry_stage": protocol.STAGE,
        "scenario": args.scenario,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "delay_ms": protocol.DELAY_MS,
        "num_envs": protocol.NUM_ENVS,
        "steps": args.steps,
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "actuator_eval": fixed_actuator_metadata(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
        "metrics": metrics,
        "corridor_report": corridor_report,
        "log": str(log_path),
        **verdict,
    }
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA5-R2-SCREEN-CELL] {candidate['name']} {args.scenario} "
        f"{'PASS' if verdict['threshold_pass'] else 'FAIL'} "
        f"n={metrics['n']} SR={metrics['sr']:.4f} "
        f"CR={metrics['cr']:.4f} TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0 if verdict["threshold_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
