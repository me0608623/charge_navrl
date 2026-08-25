"""Run one frozen SA5 low-density P100 x robot speed-rate cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa5_pedestrian_speed_screen_cell as pedestrian_runner  # noqa: E402
import run_sa5_r2_checkpoint_screen_cell as parent_runner  # noqa: E402
import run_sa5_r2_speed_scale_screen_cell as speed_runner  # noqa: E402
import sa5_low_density_speed_ratio_screen as protocol  # noqa: E402
import sa5_pedestrian_speed_screen as pedestrian_protocol  # noqa: E402
import sa5_r2_checkpoint_screen as parent_protocol  # noqa: E402
import sa5_r2_speed_scale_screen as speed_protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


def validate_d5_runtime(
    report: dict,
    *,
    checkpoint: Path,
    corridor_report: dict,
    steps: int,
) -> None:
    problems = []
    metadata = report.get("metadata") or {}
    counts = report.get("counts") or {}
    if report.get("schema") != "sa4_d5_feasibility_frontier/v1":
        problems.append("unexpected schema")
    if report.get("mode") != "feasibility_shadow":
        problems.append("unexpected mode")
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("checkpoint mismatch")
    if int(metadata.get("stage", -1)) != parent_protocol.STAGE:
        problems.append("stage mismatch")
    if int(metadata.get("seed", -1)) != parent_protocol.SEED:
        problems.append("seed mismatch")
    if int(metadata.get("num_envs", -1)) != parent_protocol.NUM_ENVS:
        problems.append("num_envs mismatch")
    if int(metadata.get("rollout_steps_requested", -1)) != steps:
        problems.append("step budget mismatch")
    if metadata.get("corridor_motion_mode") != protocol.MOTION_MODE:
        problems.append("corridor motion mismatch")
    if metadata.get("baseline_policy_action_unchanged") is not True:
        problems.append("D5 action identity declaration missing")
    if int(counts.get("records", -1)) != parent_protocol.NUM_ENVS * steps:
        problems.append("record coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(
        corridor_report["episodes"]
    ):
        problems.append("episode reconciliation mismatch")
    frontier = (report.get("frontier_summary") or {}).get(
        "dynamic_collision"
    ) or {}
    for key in (
        "event_no_dynamic_feasible_fraction",
        "persistent_dynamic_collapse_observed_fraction",
        "persistent_dynamic_collapse_lead_s",
        "closest_dynamic_feasible_lead_s",
    ):
        if key not in frontier:
            problems.append(f"dynamic-only frontier lacks {key}")
    if (report.get("self_check") or {}).get("reconciliation_ok") is not True:
        problems.append("D5 reconciliation failed")
    if problems:
        raise RuntimeError("invalid low-density D5 evidence: " + "; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--speed-rate", type=float, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args(argv)

    frozen = protocol.screen_protocol()
    density = protocol.density_by_label(args.density)
    rate = float(args.speed_rate)
    label = protocol.cell_label(density["label"], rate)
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != protocol.CHECKPOINT["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 128:
        raise ValueError("smoke steps must be within [1, 128]")
    steps = args.smoke_steps or protocol.STEPS
    is_smoke = args.smoke_steps is not None

    checkpoint = protocol.checkpoint_path().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.sha256_of(checkpoint)
    if digest != protocol.CHECKPOINT["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{steps}" if is_smoke else ""
    stem = f"c250_{label}_p100_g5_d1_s818{suffix}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    timing_path = output / f"{stem}_timing.json"
    feasibility_path = output / f"{stem}_d5.json"
    cell_path = output / f"{stem}_cell.json"
    d9_runner._require_new_targets(
        [log_path, corridor_path, timing_path, feasibility_path, cell_path]
    )

    extra = [
        *parent_runner.build_scene_args(
            density["scenario"],
            corridor_path,
            dynamic_speed_range=protocol.PEDESTRIAN_SPEED_RANGE_M_S,
            corridor_motion_mode=protocol.MOTION_MODE,
        ),
        "--speed_rate",
        f"{rate:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
        "--d3_yield_audit",
        "--d3_yield_output",
        str(timing_path),
        "--d3_shield_mode",
        "baseline",
        "--d5_feasibility_shadow",
        "--d5_feasibility_output",
        str(feasibility_path),
    ]
    print(
        f"[LOW-DENSITY-SPEED-CELL] {label} P100 steps={steps} "
        f"smoke={is_smoke} sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=parent_protocol.NUM_ENVS,
        steps=steps,
        extra=extra,
    )

    d9_runner._validate_runtime_log(
        log_path, parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path,
        parent_protocol.DELAY_STEPS,
        parent_protocol.ACTUATOR_PROFILE,
    )
    values = parent_runner.stage_values()
    values["corridor_speed_range_m_s"] = list(
        protocol.PEDESTRIAN_SPEED_RANGE_M_S
    )
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")
    corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = parent_runner.verify_corridor_runtime(
        corridor_report,
        density["scenario"],
        values,
        corridor_motion_mode=protocol.MOTION_MODE,
    )
    metrics = base.reconcile_corridor_metrics(dict(summary), corridor_report)
    if not is_smoke and int(metrics["n"]) < protocol.MIN_EPISODES:
        raise RuntimeError(
            f"cell completed only {metrics['n']} episodes; "
            f"minimum is {protocol.MIN_EPISODES}"
        )
    speed_rate_runtime = speed_runner.verify_speed_rate_runtime(
        corridor_report, rate
    )

    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    reaction_timing = speed_protocol.reaction_timing_summary(timing_payload)
    impact_severity = pedestrian_protocol.collision_radial_closing_summary(
        timing_payload
    )
    feasibility_payload = json.loads(
        feasibility_path.read_text(encoding="utf-8")
    )
    validate_d5_runtime(
        feasibility_payload,
        checkpoint=checkpoint,
        corridor_report=corridor_report,
        steps=steps,
    )
    dynamic_feasibility = pedestrian_protocol.dynamic_feasibility_summary(
        feasibility_payload
    )

    cell = {
        "schema": "sa5_low_density_speed_ratio_cell/v1",
        "cell_valid": True,
        "cell": label,
        "density": density["label"],
        "scenario": density["scenario"],
        "pedestrian_speed_range_m_s": list(
            protocol.PEDESTRIAN_SPEED_RANGE_M_S
        ),
        "speed_rate": rate,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "checkpoint_name": protocol.CHECKPOINT_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "stage": parent_protocol.STAGE,
        "seed": parent_protocol.SEED,
        "num_envs": parent_protocol.NUM_ENVS,
        "steps": steps,
        "actuator_eval": fixed_actuator_metadata(
            parent_protocol.DELAY_STEPS,
            parent_protocol.ACTUATOR_PROFILE,
        ),
        "speed_rate_runtime": speed_rate_runtime,
        "lidar_noise_mode": parent_protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": (
            parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "metrics": metrics,
        "corridor_report": corridor_report,
        "reaction_timing": reaction_timing,
        "impact_severity": impact_severity,
        "dynamic_feasibility": dynamic_feasibility,
        "log": str(log_path),
        "timing_log": str(timing_path),
        "feasibility_log": str(feasibility_path),
    }
    if is_smoke:
        cell["schema"] = "sa5_low_density_speed_ratio_smoke/v1"
        cell["formal_evidence"] = False
        cell["cell_valid"] = False
    else:
        protocol.validate_cell(cell)
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[LOW-DENSITY-SPEED-CELL] {label} n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f} dynamic_feasible_1s="
        f"{dynamic_feasibility['dynamic_feasible_within_1s_fraction']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
