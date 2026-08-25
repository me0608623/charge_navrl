"""Run one frozen SA4-R3 it125 three-scenario screen cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa4_checkpoint_screen as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import sa4_r3_it125_checkpoint_screen as protocol  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--scenario", choices=protocol.SCENARIOS, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)

    if args.checkpoint_name != protocol.CHECKPOINT_NAME:
        raise ValueError(f"unexpected checkpoint name {args.checkpoint_name!r}")
    expected_steps = protocol.STEPS_BY_SCENARIO[args.scenario]
    if args.steps != expected_steps:
        raise ValueError(
            f"{args.scenario} requires {expected_steps} steps, got {args.steps}"
        )
    if args.expect_checkpoint_sha256 != protocol.CHECKPOINT_SHA256:
        raise ValueError("CLI checkpoint hash differs from the frozen protocol")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.base.sha256_of(checkpoint)
    if digest != protocol.CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {protocol.CHECKPOINT_SHA256}, "
            f"got {digest}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_g4_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    d9_runner._require_new_targets([log_path, corridor_path, cell_path])

    values = base.scene_values()
    actuator_args = base.base.fixed_actuator_cli_args(
        protocol.DELAY_STEPS, base.base.ACTUATOR_PROFILE
    )
    scene_args = base.build_scene_args(
        args.scenario,
        seed=protocol.SEED,
        actuator_args=actuator_args,
        corridor_json=corridor_path,
    )
    print(
        f"[SA4-R3-IT125-SCREEN] scenario={args.scenario} steps={args.steps} "
        f"checkpoint_sha={digest[:12]} eligibility={protocol.ELIGIBILITY}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=64,
        steps=args.steps,
        extra=[
            *scene_args,
            "--lidar-distractor-eligibility",
            protocol.ELIGIBILITY,
        ],
    )
    d9_runner._validate_runtime_log(log_path, protocol.ELIGIBILITY)
    base.base.verify_fixed_actuator_runtime(
        log_path, protocol.DELAY_STEPS, base.base.ACTUATOR_PROFILE
    )
    stage_banner = base.base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    metrics = dict(summary)
    corridor_report = None
    if args.scenario in base.base.CORRIDOR_SCENARIOS:
        corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
        family = args.scenario.split("_", 1)[1]
        scene_contract = base.base.verify_corridor_runtime(
            log_path, corridor_report, family, values
        )
        metrics = base.base.reconcile_corridor_metrics(metrics, corridor_report)
    else:
        metrics = base.base.reconcile_console_metrics(
            metrics, base.base.parse_exact_outcome_counts(log_path)
        )
        scene_contract = {
            "scenario": args.scenario,
            "mechanism": "stage scene via STAGE_PARAMETER",
        }
    for key in ("sr", "cr", "to"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"{args.scenario}: non-finite {key}")

    verdict = base.base.evaluate(args.scenario, 4, metrics)
    cell = {
        "schema": "sa4_r3_it125_checkpoint_screen_cell/v1",
        "checkpoint_name": protocol.CHECKPOINT_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "conceptual_iteration": protocol.CONCEPTUAL_ITERATION,
        "geometry_stage": 4,
        "scenario": args.scenario,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "num_envs": 64,
        "steps": args.steps,
        "lidar_distractor_eligibility": protocol.ELIGIBILITY,
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "metrics": metrics,
        "corridor_report": corridor_report,
        "log": str(log_path),
        **verdict,
    }
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-R3-IT125-SCREEN] {args.scenario} "
        f"{'PASS' if verdict['threshold_pass'] else 'FAIL'} "
        f"n={metrics['n']} SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0 if verdict["threshold_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

