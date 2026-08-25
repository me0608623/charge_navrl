"""Run the pre-registered SA4-D3 lateral intervention suite.

This is a fixed-checkpoint diagnostic 2x2 mechanism probe. It evaluates the
three non-baseline arms against the already completed baseline without
starting training or advancing to another curriculum stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_d3_baseline as baseline  # noqa: E402
from d3_yield_recorder import (  # noqa: E402
    ShieldMode,
    shield_protocol,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


MODES = (
    ShieldMode.SUSTAINED_BRAKE,
    ShieldMode.BEST_TURN,
    ShieldMode.COMBINED,
)
PASS_SR = 0.90
PASS_CR = 0.10
PASS_TO = 0.05


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "play": baseline.PLAY,
        "action_term": baseline.ACTION_TERM,
        "recorder_shield": baseline.RECORDER,
    }
    return {name: sha256_of(path) for name, path in files.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D3 intervention suite refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def _metrics(corridor: dict) -> dict:
    result = {
        "n": int(corridor["episodes"]),
        "sr": float(corridor["success_rate"]),
        "cr": float(corridor["collision_rate"]),
        "to": float(corridor["timeout_rate"]),
        "obstacle_cr": float(corridor["obstacle_collision_rate"]),
        "wall_cr": float(corridor["wall_collision_rate"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite corridor metric")
    return result


def _validate_intervention_report(
    report: dict,
    *,
    mode: ShieldMode,
    checkpoint: Path,
    corridor: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    if report.get("schema") != "sa4_d3_yield_timing/v2":
        problems.append(f"unexpected schema {report.get('schema')!r}")
    if report.get("mode") != mode.value:
        problems.append(f"mode mismatch {report.get('mode')!r}")
    metadata = report.get("metadata") or {}
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("checkpoint mismatch")
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
    if metadata.get("shield_protocol") != shield_protocol():
        problems.append("shield protocol drift")

    runtime = metadata.get("shield_runtime") or {}
    if runtime.get("mode") != mode.value:
        problems.append("shield runtime mode mismatch")
    if runtime.get("protocol_sha256") != shield_protocol()["sha256"]:
        problems.append("shield runtime protocol hash mismatch")
    if int(runtime.get("calls", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("shield call count mismatch")
    if int(runtime.get("environment_frames", -1)) != expected_records:
        problems.append("shield environment-frame count mismatch")
    if int(runtime.get("trigger_activations", 0)) <= 0:
        problems.append("intervention never triggered")
    if int(runtime.get("override_environment_frames", 0)) <= 0:
        problems.append("intervention never changed an action")
    if int(runtime.get("active_environment_frames", -1)) < int(
        runtime.get("override_environment_frames", 0)
    ):
        problems.append("override frames exceed active frames")

    counts = report.get("counts") or {}
    if int(counts.get("records", -1)) != expected_records:
        problems.append("D3 record coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(corridor["episodes"]):
        problems.append("episode ledger mismatch")
    checks = report.get("self_check") or {}
    for key in (
        "intervention_action_observed",
        "action_contract_ok",
        "record_coverage_ok",
        "episode_reconciliation_ok",
        "delay_alignment_ok",
        "dynamic_collision_reconciliation_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"{key} is not true")
    if checks.get("baseline_action_identity_ok") is not None:
        problems.append("baseline identity field must be null for intervention")
    if int(checks.get("delay_alignment_errors", -1)) != 0:
        problems.append("d1 delay alignment error")
    if int(checks.get("shield_engaged_records", -1)) != int(
        runtime.get("override_environment_frames", -2)
    ):
        problems.append("shield override ledger mismatch")
    if problems:
        raise RuntimeError(
            f"invalid SA4-D3 {mode.value} evidence: " + "; ".join(problems)
        )


def _run_arm(
    mode: ShieldMode,
    checkpoint: Path,
    output: Path,
    values: dict,
) -> dict:
    stem = f"{mode.value}_lateral_g4_d1_s818"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    d3_path = output / f"{stem}_d3.json"
    before_path = output / f"{stem}_source_before.json"
    after_path = output / f"{stem}_source_after.json"
    _require_new_targets(
        [log_path, corridor_path, d3_path, before_path, after_path]
    )

    before = source_fingerprint()
    before_path.write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )
    actuator_args = baseline.sa4.base.fixed_actuator_cli_args(
        baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
    )
    scene_args = baseline.sa4.build_scene_args(
        baseline.SCENARIO,
        seed=baseline.SEED,
        actuator_args=actuator_args,
        corridor_json=corridor_path,
    )
    print(
        f"[SA4-D3-SUITE] START {mode.value}: c6400, lateral, "
        "seed818, d1=200ms, 64 env x 2500 steps",
        flush=True,
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
            "--d3_shield_mode",
            mode.value,
        ],
    )
    after = source_fingerprint()
    after_path.write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError(f"source drift during {mode.value}")

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
    baseline.sa4.base.reconcile_corridor_metrics(console, corridor)
    d3_report = json.loads(d3_path.read_text(encoding="utf-8"))
    _validate_intervention_report(
        d3_report,
        mode=mode,
        checkpoint=checkpoint,
        corridor=corridor,
    )
    metrics = _metrics(corridor)
    result = {
        "mode": mode.value,
        "metrics": metrics,
        "threshold_pass": (
            metrics["sr"] >= PASS_SR
            and metrics["cr"] <= PASS_CR
            and metrics["to"] <= PASS_TO
        ),
        "shield_runtime": d3_report["metadata"]["shield_runtime"],
        "d3_counts": d3_report["counts"],
        "d3_self_check": d3_report["self_check"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "files": {
            "log": str(log_path),
            "corridor": str(corridor_path),
            "d3": str(d3_path),
            "source_before": str(before_path),
            "source_after": str(after_path),
        },
    }
    print(
        f"[SA4-D3-SUITE] VALID {mode.value}: n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f} "
        f"overrides={result['shield_runtime']['override_environment_frames']}",
        flush=True,
    )
    return result


def _render_comparison(baseline_metrics: dict, arms: list[dict]) -> str:
    rows = [("baseline", baseline_metrics, None)] + [
        (
            arm["mode"],
            arm["metrics"],
            arm["shield_runtime"]["override_environment_frames"],
        )
        for arm in arms
    ]
    lines = [
        "# SA4-D3 lateral mechanism-probe comparison",
        "",
        "> Fixed checkpoint c6400, stage 4 lateral, seed 818, d1=200 ms. "
        "This is diagnostic evidence, not a deployment shield.",
        "",
        "| Arm | episodes | SR | CR | obstacle CR | wall CR | TO | CR change vs baseline | overrides | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, metrics, overrides in rows:
        change = 100.0 * (metrics["cr"] - baseline_metrics["cr"])
        passed = (
            metrics["sr"] >= PASS_SR
            and metrics["cr"] <= PASS_CR
            and metrics["to"] <= PASS_TO
        )
        lines.append(
            f"| {name} | {metrics['n']:,} | {metrics['sr']:.2%} | "
            f"{metrics['cr']:.2%} | {metrics['obstacle_cr']:.2%} | "
            f"{metrics['wall_cr']:.2%} | {metrics['to']:.2%} | "
            f"{change:+.2f} pp | "
            f"{'n/a' if overrides is None else f'{overrides:,}'} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    lines += [
        "",
        "Gate: SR >= 90%, CR <= 10%, TO <= 5%.",
        "A lower collision rate is not sufficient if success is replaced by timeout or wall collision.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
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
    baseline_manifest_path = args.baseline_manifest.expanduser().resolve()
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    if baseline_manifest.get("status") != "VALID_DIAGNOSTIC_EVIDENCE":
        raise RuntimeError("baseline manifest is not valid diagnostic evidence")
    if baseline_manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("baseline and intervention checkpoint hashes differ")
    baseline_corridor = json.loads(
        Path(baseline_manifest["files"]["corridor"]).read_text(encoding="utf-8")
    )
    baseline_metrics = _metrics(baseline_corridor)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "PROTOCOL.json"
    manifest_path = output / "suite_manifest.json"
    comparison_json_path = output / "comparison.json"
    comparison_md_path = output / "COMPARISON.md"
    _require_new_targets(
        [protocol_path, manifest_path, comparison_json_path, comparison_md_path]
    )
    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d3_suite_preregistration/v1",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "baseline_manifest": str(baseline_manifest_path),
        "fixed_cell": baseline_manifest["fixed_cell"],
        "arms_in_fixed_order": [mode.value for mode in MODES],
        "thresholds": {"sr_min": PASS_SR, "cr_max": PASS_CR, "to_max": PASS_TO},
        "shield_protocol": shield_protocol(),
        "source_fingerprint": planned_fingerprint,
    }
    protocol_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D3-SUITE] protocol frozen before rollout: {protocol_path}",
        flush=True,
    )

    values = baseline.sa4.scene_values()
    arms = [
        _run_arm(mode, checkpoint, output, values)
        for mode in MODES
    ]
    final_fingerprint = source_fingerprint()
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D3 source changed during intervention suite")

    comparison = {
        "schema": "sa4_d3_intervention_comparison/v1",
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": shield_protocol()["sha256"],
        "baseline": baseline_metrics,
        "arms": arms,
        "thresholds": {"sr_min": PASS_SR, "cr_max": PASS_CR, "to_max": PASS_TO},
        "limitations": [
            "single checkpoint and single evaluation seed",
            "fixed diagnostic trigger has a high baseline activation rate on safe close approaches",
            "results identify closed-loop mechanism effects but do not establish deployment safety",
        ],
    }
    comparison_json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    comparison_md_path.write_text(
        _render_comparison(baseline_metrics, arms), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d3_intervention_suite/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": shield_protocol()["sha256"],
        "source_fingerprint": final_fingerprint,
        "baseline": baseline_metrics,
        "arms": arms,
        "comparison_json": str(comparison_json_path),
        "comparison_markdown": str(comparison_md_path),
        "training_started": False,
        "next_stage_started": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D3-SUITE] COMPLETE comparison={comparison_md_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
