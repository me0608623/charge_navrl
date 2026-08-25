"""Run the frozen SA4-D5 baseline-only feasibility-frontier audit.

This runner executes one fresh identity baseline at the fixed SA4-R2 c6400
lateral cell. The D5 shadow evaluates the D4 19x19 geometry on every frame with
valid dynamic obstacles but never changes a policy action. It does not train or
start another curriculum stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_d3_baseline as baseline  # noqa: E402
from d4_geometry_selector import geometry_selector_protocol  # noqa: E402
from d5_feasibility_shadow import (  # noqa: E402
    feasibility_shadow_protocol,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


D4_SELECTOR = HERE / "d4_geometry_selector.py"
D5_SHADOW = HERE / "d5_feasibility_shadow.py"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "baseline_runner": Path(baseline.__file__).resolve(),
        "play": baseline.PLAY,
        "action_term": baseline.ACTION_TERM,
        "d3_recorder": baseline.RECORDER,
        "d4_selector": D4_SELECTOR,
        "d5_shadow": D5_SHADOW,
    }
    return {name: sha256_of(path) for name, path in paths.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D5 shadow audit refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def validate_d5_report(
    report: dict,
    *,
    checkpoint: Path,
    corridor_report: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    if report.get("schema") != "sa4_d5_feasibility_frontier/v1":
        problems.append(f"unexpected D5 schema {report.get('schema')!r}")
    if report.get("mode") != "feasibility_shadow":
        problems.append(f"unexpected D5 mode {report.get('mode')!r}")
    if report.get("protocol") != feasibility_shadow_protocol():
        problems.append("D5 protocol drift")
    metadata = report.get("metadata") or {}
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("checkpoint path mismatch")
    if int(metadata.get("stage", -1)) != baseline.GEOMETRY_STAGE:
        problems.append("stage mismatch")
    if int(metadata.get("seed", -1)) != baseline.SEED:
        problems.append("seed mismatch")
    if int(metadata.get("num_envs", -1)) != baseline.NUM_ENVS:
        problems.append("num_envs mismatch")
    if int(metadata.get("rollout_steps_requested", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("rollout budget mismatch")
    if metadata.get("corridor_motion_mode") != "lateral":
        problems.append("corridor family mismatch")
    if metadata.get("baseline_policy_action_unchanged") is not True:
        problems.append("baseline action identity declaration missing")
    if metadata.get("shadow_protocol") != feasibility_shadow_protocol():
        problems.append("metadata D5 protocol drift")
    if metadata.get("geometry_protocol") != geometry_selector_protocol():
        problems.append("metadata D4 geometry protocol drift")
    actuator = metadata.get("actuator") or {}
    if actuator.get("delay_range") != [1, 1]:
        problems.append("actuator delay is not fixed d1")
    if actuator.get("velocity_scale_range") != [1.0, 1.0]:
        problems.append("actuator scale is not fixed at 1")
    if float(actuator.get("motor_lag_alpha", -1.0)) != 1.0:
        problems.append("actuator motor lag is not disabled")

    counts = report.get("counts") or {}
    if int(counts.get("records", -1)) != expected_records:
        problems.append("D5 record coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(
        corridor_report["episodes"]
    ):
        problems.append("D5 episode ledger mismatch")
    if int(counts.get("dynamic_collision", -1)) <= 0:
        problems.append("D5 captured no dynamic collisions")
    if int(counts.get("successful_noncollision_closest_approach", -1)) <= 0:
        problems.append("D5 captured no successful control events")

    runtime = metadata.get("shadow_runtime") or {}
    if runtime.get("mode") != "feasibility_shadow":
        problems.append("shadow runtime mode mismatch")
    if runtime.get("protocol_sha256") != feasibility_shadow_protocol()["sha256"]:
        problems.append("shadow runtime protocol hash mismatch")
    if int(runtime.get("calls", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("shadow call count mismatch")
    if int(runtime.get("environment_frames", -1)) != expected_records:
        problems.append("shadow environment-frame count mismatch")
    evaluated = int(runtime.get("evaluated_environment_frames", -1))
    skipped = int(runtime.get("skipped_no_dynamic_frames", -1))
    feasible = int(runtime.get("jointly_feasible_frames", -1))
    no_feasible = int(runtime.get("no_jointly_feasible_frames", -1))
    if min(evaluated, skipped, feasible, no_feasible) < 0:
        problems.append("shadow accounting field missing")
    else:
        if evaluated + skipped != expected_records:
            problems.append("evaluated/skipped frames do not reconcile")
        if feasible + no_feasible != evaluated:
            problems.append("feasible/no-feasible frames do not reconcile")
    if int(runtime.get("action_identity_checks", -1)) != expected_records:
        problems.append("shadow action identity coverage mismatch")
    if int(runtime.get("action_identity_errors", -1)) != 0:
        problems.append("shadow modified policy actions")

    checks = report.get("self_check") or {}
    for key in (
        "baseline_action_identity_ok",
        "record_coverage_ok",
        "episode_reconciliation_ok",
        "dynamic_collision_reconciliation_ok",
        "shadow_runtime_reconciliation_ok",
        "d3_reconciliation_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"{key} is not true")
    frontier = report.get("frontier_summary") or {}
    collision_summary = frontier.get("dynamic_collision") or {}
    control_summary = frontier.get(
        "successful_noncollision_closest_approach"
    ) or {}
    if int(collision_summary.get("events", -1)) != int(
        counts.get("dynamic_collision", -2)
    ):
        problems.append("collision frontier count mismatch")
    if int(control_summary.get("events", -1)) != int(
        counts.get("successful_noncollision_closest_approach", -2)
    ):
        problems.append("control frontier count mismatch")
    if problems:
        raise RuntimeError("invalid SA4-D5 evidence: " + "; ".join(problems))


def _render_summary(report: dict, corridor: dict) -> str:
    runtime = report["metadata"]["shadow_runtime"]
    collision = report["frontier_summary"]["dynamic_collision"]
    control = report["frontier_summary"][
        "successful_noncollision_closest_approach"
    ]
    evaluated = int(runtime["evaluated_environment_frames"])
    no_feasible = int(runtime["no_jointly_feasible_frames"])
    return "\n".join(
        [
            "# SA4-D5 baseline-only feasibility frontier",
            "",
            "> Fixed c6400, stage4 lateral, seed818, d1=200 ms. Shadow only; policy actions unchanged.",
            "",
            f"- episodes: {int(corridor['episodes']):,}",
            f"- SR / CR / TO: {float(corridor['success_rate']):.2%} / {float(corridor['collision_rate']):.2%} / {float(corridor['timeout_rate']):.2%}",
            f"- evaluated frames: {evaluated:,}",
            f"- no jointly feasible frames: {no_feasible:,} ({no_feasible / max(evaluated, 1):.2%})",
            f"- dynamic collision events: {int(collision['events']):,}",
            f"- successful control events: {int(control['events']):,}",
            f"- collision event no-feasible fraction: {float(collision['event_no_feasible_fraction']):.2%}",
            f"- control event no-feasible fraction: {float(control['event_no_feasible_fraction']):.2%}",
            "",
            "Interpretation must use event-aligned frontier distributions. The all-frame no-feasible share is not a collision-unavoidability rate.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_of(checkpoint)
    if checkpoint_sha256 != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {args.expect_checkpoint_sha256}, "
            f"got {checkpoint_sha256}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "PROTOCOL.json"
    log_path = output / "baseline_lateral_g4_d1_s818.log"
    corridor_path = output / "baseline_lateral_g4_d1_s818_corridor.json"
    d3_path = output / "baseline_lateral_g4_d1_s818_d3.json"
    d5_path = output / "baseline_lateral_g4_d1_s818_d5.json"
    summary_path = output / "SUMMARY.md"
    manifest_path = output / "suite_manifest.json"
    before_path = output / "source_fingerprint_before.json"
    after_path = output / "source_fingerprint_after.json"
    _require_new_targets(
        [
            protocol_path,
            log_path,
            corridor_path,
            d3_path,
            d5_path,
            summary_path,
            manifest_path,
            before_path,
            after_path,
        ]
    )

    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d5_shadow_preregistration/v1",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "locate when the frozen D4 jointly-feasible action set disappears "
            "before collisions and successful passes, without modifying actions"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_cell": {
            "geometry_stage": baseline.GEOMETRY_STAGE,
            "scenario": baseline.SCENARIO,
            "seed": baseline.SEED,
            "delay_steps": baseline.DELAY_STEPS,
            "delay_ms": 200,
            "actuator_profile": baseline.ACTUATOR_PROFILE,
            "num_envs": baseline.NUM_ENVS,
            "steps": baseline.ROLLOUT_STEPS,
        },
        "action_arm": "fresh identity baseline only",
        "d5_protocol": feasibility_shadow_protocol(),
        "d4_geometry_protocol": geometry_selector_protocol(),
        "source_fingerprint": planned_fingerprint,
        "forbidden_during_run": [
            "action override",
            "training",
            "reward or trigger changes",
            "SA5 launch",
        ],
    }
    protocol_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    before_path.write_text(
        json.dumps(planned_fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[SA4-D5] protocol frozen before rollout: {protocol_path}", flush=True)

    values = baseline.sa4.scene_values()
    actuator_args = baseline.sa4.base.fixed_actuator_cli_args(
        baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
    )
    scene_args = baseline.sa4.build_scene_args(
        baseline.SCENARIO,
        seed=baseline.SEED,
        actuator_args=actuator_args,
        corridor_json=corridor_path,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=baseline.NUM_ENVS,
        steps=baseline.ROLLOUT_STEPS,
        extra=[
            *scene_args,
            "--d3_yield_audit",
            "--d3_yield_output",
            str(d3_path),
            "--d3_shield_mode", "baseline",
            "--d5_feasibility_shadow",
            "--d5_feasibility_output",
            str(d5_path),
        ],
    )

    final_fingerprint = source_fingerprint()
    after_path.write_text(
        json.dumps(final_fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D5 source changed during rollout")

    baseline.sa4.base.verify_fixed_actuator_runtime(
        log_path, baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
    )
    stage_banner = baseline.sa4.base.verify_common_runtime(log_path, values)
    console = parse_play_summary(str(log_path))
    if not console or console.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    corridor = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = baseline.sa4.base.verify_corridor_runtime(
        log_path, corridor, "lateral", values
    )
    metrics = baseline.sa4.base.reconcile_corridor_metrics(console, corridor)
    d3_report = json.loads(d3_path.read_text(encoding="utf-8"))
    baseline.validate_d3_report(
        d3_report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        corridor_report=corridor,
    )
    d5_report = json.loads(d5_path.read_text(encoding="utf-8"))
    validate_d5_report(
        d5_report,
        checkpoint=checkpoint,
        corridor_report=corridor,
    )
    summary_path.write_text(
        _render_summary(d5_report, corridor), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d5_shadow_bundle/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "not_a_graduation_gate": True,
        "policy_actions_modified": False,
        "training_started": False,
        "next_stage_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": feasibility_shadow_protocol()["sha256"],
        "fixed_cell": preregistration["fixed_cell"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics": metrics,
        "d3_counts": d3_report["counts"],
        "d3_self_check": d3_report["self_check"],
        "d5_counts": d5_report["counts"],
        "d5_frontier_summary": d5_report["frontier_summary"],
        "d5_self_check": d5_report["self_check"],
        "shadow_runtime": d5_report["metadata"]["shadow_runtime"],
        "source_fingerprint": final_fingerprint,
        "files": {
            "log": str(log_path),
            "corridor": str(corridor_path),
            "d3": str(d3_path),
            "d5": str(d5_path),
            "summary": str(summary_path),
            "source_before": str(before_path),
            "source_after": str(after_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D5] COMPLETE valid shadow evidence: {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
