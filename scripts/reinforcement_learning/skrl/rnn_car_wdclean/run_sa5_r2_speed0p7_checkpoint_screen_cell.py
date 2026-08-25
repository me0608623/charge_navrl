"""Run one cell of the frozen SA5-R2 speed-0.7 checkpoint screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa5_r2_checkpoint_screen_cell as parent_runner  # noqa: E402
import sa5_r2_checkpoint_screen as parent_protocol  # noqa: E402
import sa5_r2_speed0p7_checkpoint_screen as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


EXPECTED_LIMITS = {
    "max_linear_velocity": 0.7,
    "max_linear_accel": 0.35,
    "max_angular_vel": 0.84,
    "max_angular_accel": 2.1,
}


def verify_speed_rate_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    marker = (
        "[VEHICLE-SPEED-RATE] rate=0.7 obs=ego lidar_scaled=False "
        "deployment_scale=1"
    )
    if text.splitlines().count(marker) != 1:
        raise RuntimeError("expected exactly one speed-rate runtime marker")
    limits = {}
    for key, expected in EXPECTED_LIMITS.items():
        prefix = f"[SPEED_RATE] {key}:"
        matches = [line for line in text.splitlines() if line.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one runtime limit line for {key}")
        observed = float(matches[0].rsplit(maxsplit=1)[-1])
        if not math.isfinite(observed) or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RuntimeError(f"speed-rate action limit mismatch for {key}")
        limits[key] = observed
    return {
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        "lidar_scaled": False,
        "action_limits": limits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--scenario", choices=protocol.SCENARIOS, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--step-tier-index", type=int, required=True)
    args = parser.parse_args(argv)

    candidate = protocol.candidate_by_name(args.checkpoint_name)
    frozen = protocol.screen_protocol()
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != candidate["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    tiers = protocol.STEP_TIERS_BY_SCENARIO[args.scenario]
    if args.step_tier_index not in range(len(tiers)):
        raise ValueError("cell step tier index is outside the frozen protocol")
    if args.steps != tiers[args.step_tier_index]:
        raise ValueError("cell step budget differs from frozen protocol")

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
    stem = f"{args.scenario}_speed0p7_g5_d1_s{parent_protocol.SEED}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    targets = [log_path, cell_path]
    if args.scenario in protocol.CORRIDOR_SCENARIOS:
        targets.append(corridor_path)
    d9_runner._require_new_targets(targets)

    extra = [
        *parent_runner.build_scene_args(args.scenario, corridor_path),
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
    ]
    print(
        f"[SA5-R2-SPEED0P7-CELL] {candidate['name']} {args.scenario} "
        f"steps={args.steps} sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=parent_protocol.NUM_ENVS,
        steps=args.steps,
        extra=extra,
    )

    d9_runner._validate_runtime_log(
        log_path, parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, parent_protocol.DELAY_STEPS, parent_protocol.ACTUATOR_PROFILE
    )
    values = parent_runner.stage_values()
    stage_banner = base.verify_common_runtime(log_path, values)
    speed_runtime = verify_speed_rate_log(log_path)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")
    metrics = dict(summary)
    corridor_report = None
    if args.scenario in protocol.CORRIDOR_SCENARIOS:
        corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
        scene_contract = parent_runner.verify_corridor_runtime(
            corridor_report, args.scenario, values
        )
        metrics = base.reconcile_corridor_metrics(metrics, corridor_report)
        report_runtime = {
            "speed_rate": float(corridor_report["speed_rate"]),
            "speed_rate_obs": corridor_report["speed_rate_obs_mode"],
            "deployment_speed_scale": float(
                corridor_report["deployment_speed_scale"]
            ),
        }
        if report_runtime != {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        }:
            raise RuntimeError("corridor report speed-rate contract mismatch")
        if int(corridor_report.get("deployment_scale_samples", 0)) <= 0:
            raise RuntimeError("corridor report has no deployment-scale samples")
        if float(corridor_report["deployment_scale_max_abs_error"]) > 1.0e-6:
            raise RuntimeError("deployment output scaling did not reconcile")
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

    outcome_gate = parent_protocol.evaluate_metrics(args.scenario, metrics)
    cell = {
        "schema": "sa5_r2_speed0p7_checkpoint_screen_cell/v3",
        "cell_valid": int(metrics["n"]) >= parent_protocol.MIN_EPISODES,
        "checkpoint_name": candidate["name"],
        "checkpoint_role": candidate["role"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "scenario": args.scenario,
        "stage": parent_protocol.STAGE,
        "seed": parent_protocol.SEED,
        "num_envs": parent_protocol.NUM_ENVS,
        "steps": args.steps,
        "step_tier_index": args.step_tier_index,
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        "speed_rate_runtime": speed_runtime,
        "actuator_eval": fixed_actuator_metadata(
            parent_protocol.DELAY_STEPS, parent_protocol.ACTUATOR_PROFILE
        ),
        "lidar_noise_mode": parent_protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": (
            parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "metrics": metrics,
        "corridor_report": corridor_report,
        "outcome_gate": outcome_gate,
        "log": str(log_path),
        "measurement_provenance": {
            "mode": "fresh_v3_step_tier",
        },
    }
    protocol.validate_cell(cell, require_min_episodes=False)
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not cell["cell_valid"]:
        print(
            f"[SA5-R2-SPEED0P7-CELL] INSUFFICIENT {candidate['name']} "
            f"{args.scenario} n={metrics['n']} steps={args.steps}",
            flush=True,
        )
        return 3
    print(
        f"[SA5-R2-SPEED0P7-CELL] DONE {candidate['name']} "
        f"{args.scenario} n={metrics['n']} SR={metrics['sr']:.4f} "
        f"CR={metrics['cr']:.4f} TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
