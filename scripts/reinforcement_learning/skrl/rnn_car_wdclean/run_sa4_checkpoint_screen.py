"""Run one SA4 it50/it100 checkpoint-screen cell.

This is a thin, stage-4-specific front end over the frozen SA3 B/C evaluator
helpers. The earlier runner is imported but not edited, so its recorded source
fingerprint remains reproducible.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sa3_gate_bc as base  # noqa: E402


GEOMETRY_STAGE = 4
SCENARIOS = (
    "nav_native",
    "native_crossing",
    "corridor_lateral",
    "corridor_longitudinal",
)


@contextmanager
def _stage4_enabled():
    """Temporarily extend the imported runner without leaking global state."""
    original = base.SUPPORTED_GEOMETRY_STAGES
    base.SUPPORTED_GEOMETRY_STAGES = tuple(
        sorted(set(original) | {GEOMETRY_STAGE})
    )
    try:
        yield
    finally:
        base.SUPPORTED_GEOMETRY_STAGES = original


def scene_values() -> dict:
    with _stage4_enabled():
        return base.scene_values(GEOMETRY_STAGE)


def build_scene_args(
    scenario: str,
    *,
    seed: int,
    actuator_args: list[str],
    corridor_json: Path,
) -> list[str]:
    with _stage4_enabled():
        return base.build_scene_args(
            scenario,
            geometry_stage=GEOMETRY_STAGE,
            seed=seed,
            actuator_args=actuator_args,
            corridor_json=corridor_json,
        )


def make_cell_id(
    checkpoint: Path, scenario: str, delay_steps: int, seed: int
) -> str:
    with _stage4_enabled():
        return base.make_cell_id(
            checkpoint, scenario, GEOMETRY_STAGE, delay_steps, seed
        )


def main(argv: list[str] | None = None) -> int:
    from run_sa5_joint_retention_gates import _run_play
    from validate_gates import parse_play_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--actuator-delay-steps", type=int, choices=(0, 1, 2), required=True
    )
    parser.add_argument(
        "--actuator-profile",
        choices=(base.ACTUATOR_PROFILE,),
        default=base.ACTUATOR_PROFILE,
    )
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.sha256_of(checkpoint)
    if digest != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch for {checkpoint}: "
            f"expected {args.expect_checkpoint_sha256}, got {digest}"
        )

    values = scene_values()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actuator_args = base.fixed_actuator_cli_args(
        args.actuator_delay_steps, args.actuator_profile
    )
    stem = (
        f"{args.scenario}_g{GEOMETRY_STAGE}"
        f"_d{args.actuator_delay_steps}_s{args.seed}"
    )
    log_path = output / f"{stem}.log"
    corridor_json = output / f"{stem}_corridor.json"

    print(
        f"[SA4-SCREEN] scenario={args.scenario} stage={GEOMETRY_STAGE} "
        f"seed={args.seed} delay={args.actuator_delay_steps} "
        f"({base.DELAY_MS[args.actuator_delay_steps]} ms) "
        f"arena={values['arena_size_m']:g}m ckpt={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=args.num_envs,
        steps=args.steps,
        extra=build_scene_args(
            args.scenario,
            seed=args.seed,
            actuator_args=actuator_args,
            corridor_json=corridor_json,
        ),
    )
    base.verify_fixed_actuator_runtime(
        log_path, args.actuator_delay_steps, args.actuator_profile
    )
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    metrics = dict(summary)
    corridor_report = None
    if args.scenario in base.CORRIDOR_SCENARIOS:
        corridor_report = json.loads(corridor_json.read_text(encoding="utf-8"))
        family = args.scenario.split("_", 1)[1]
        scene_contract = base.verify_corridor_runtime(
            log_path, corridor_report, family, values
        )
        metrics = base.reconcile_corridor_metrics(metrics, corridor_report)
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
            raise RuntimeError(f"{args.scenario}: non-finite {key}")
    verdict = base.evaluate(args.scenario, GEOMETRY_STAGE, metrics)
    cell = {
        "schema": "sa4_checkpoint_screen/v1",
        "cell_id": make_cell_id(
            checkpoint,
            args.scenario,
            args.actuator_delay_steps,
            args.seed,
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "geometry_stage": GEOMETRY_STAGE,
        "scene_values": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in values.items()
        },
        "scenario": args.scenario,
        "seed": args.seed,
        "delay_steps": args.actuator_delay_steps,
        "delay_ms": base.DELAY_MS[args.actuator_delay_steps],
        "num_envs": args.num_envs,
        "steps": args.steps,
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "actuator_eval": base.fixed_actuator_metadata(
            args.actuator_delay_steps, args.actuator_profile
        ),
        "metrics": metrics,
        "corridor_report": corridor_report,
        "log": str(log_path),
        **verdict,
    }
    cell_path = output / f"{stem}_cell.json"
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-SCREEN] {stem}="
        f"{'PASS' if verdict['threshold_pass'] else 'FAIL'} "
        f"n={metrics['n']} SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f} "
        f"worst={verdict['worst_check']}({verdict['worst_margin']:+.4f}) "
        f"report={cell_path}",
        flush=True,
    )
    return 0 if verdict["threshold_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
