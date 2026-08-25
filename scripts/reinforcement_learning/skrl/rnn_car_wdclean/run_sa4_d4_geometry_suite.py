"""Run the preregistered SA4-D4 geometry-selector diagnostic comparison.

The suite evaluates a fresh identity baseline and the delay-aware
``geometry_feasible`` 19x19 selector on the same fixed SA4 lateral cell. It
does not train a policy, change curriculum stages, or claim deployment safety.
The protocol is written before the first rollout and all evidence targets are
fail-closed against overwrite and source drift.
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
from d3_yield_recorder import shield_protocol as d3_shield_protocol  # noqa: E402
from d4_geometry_selector import (  # noqa: E402
    ARGMIN_MODE as ARGMIN_GEOMETRY_MODE,
    MODE as GEOMETRY_MODE,
    geometry_argmin_selector_protocol,
    geometry_selector_protocol,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


ARMS = ("baseline", GEOMETRY_MODE)
PASS_SR = 0.90
PASS_CR = 0.10
PASS_TO = 0.05
SELECTOR = HERE / "d4_geometry_selector.py"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "baseline_runner": Path(baseline.__file__).resolve(),
        "play": baseline.PLAY,
        "action_term": baseline.ACTION_TERM,
        "recorder": baseline.RECORDER,
        "selector": SELECTOR,
    }
    return {name: sha256_of(path) for name, path in files.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D4 geometry suite refuses to overwrite evidence: "
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
    if result["n"] < 1 or not all(
        math.isfinite(value)
        for key, value in result.items()
        if key != "n"
    ):
        raise RuntimeError("invalid or non-finite corridor metrics")
    return result


def _expected_protocol(mode: str) -> dict:
    if mode == "baseline":
        return d3_shield_protocol()
    if mode == GEOMETRY_MODE:
        return geometry_selector_protocol()
    if mode == ARGMIN_GEOMETRY_MODE:
        return geometry_argmin_selector_protocol()
    raise ValueError(f"unsupported D4 arm {mode!r}")


def _validate_report(
    report: dict,
    *,
    mode: str,
    checkpoint: Path,
    corridor: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    protocol = _expected_protocol(mode)
    if report.get("schema") != "sa4_d3_yield_timing/v2":
        problems.append(f"unexpected recorder schema {report.get('schema')!r}")
    if report.get("mode") != mode:
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
    if metadata.get("shield_protocol") != protocol:
        problems.append("selector protocol drift")
    actuator = metadata.get("actuator") or {}
    if actuator.get("delay_range") != [1, 1]:
        problems.append("actuator delay is not fixed d1")
    if actuator.get("velocity_scale_range") != [1.0, 1.0]:
        problems.append("actuator scale is not fixed at 1")
    if float(actuator.get("motor_lag_alpha", -1.0)) != 1.0:
        problems.append("actuator motor lag is not disabled")

    runtime = metadata.get("shield_runtime") or {}
    if runtime.get("mode") != mode:
        problems.append("selector runtime mode mismatch")
    if runtime.get("protocol_sha256") != protocol["sha256"]:
        problems.append("selector runtime protocol hash mismatch")
    if int(runtime.get("calls", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("selector call count mismatch")
    if int(runtime.get("environment_frames", -1)) != expected_records:
        problems.append("selector environment-frame count mismatch")

    counts = report.get("counts") or {}
    if int(counts.get("records", -1)) != expected_records:
        problems.append("recorder coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(
        corridor["episodes"]
    ):
        problems.append("episode ledger mismatch")
    checks = report.get("self_check") or {}
    for key in (
        "action_contract_ok",
        "record_coverage_ok",
        "episode_reconciliation_ok",
        "delay_alignment_ok",
        "dynamic_collision_reconciliation_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"{key} is not true")
    if int(checks.get("delay_alignment_errors", -1)) != 0:
        problems.append("d1 delay alignment error")

    overrides = int(runtime.get("override_environment_frames", -1))
    if mode == "baseline":
        if metadata.get("baseline_tensor_identity_checked_each_step") is not True:
            problems.append("baseline tensor-identity guard missing")
        if checks.get("baseline_action_identity_ok") is not True:
            problems.append("baseline action identity failed")
        if overrides != 0:
            problems.append("baseline changed policy actions")
    else:
        active = int(runtime.get("active_environment_frames", -1))
        feasible = int(runtime.get("jointly_feasible_active_frames", -1))
        no_feasible = int(runtime.get("no_feasible_active_frames", -1))
        policy_feasible = int(
            runtime.get("policy_feasible_active_frames", -1)
        )
        if checks.get("baseline_action_identity_ok") is not None:
            problems.append("intervention baseline identity field is not null")
        if checks.get("intervention_action_observed") is not True:
            problems.append("geometry selector never changed an action")
        if int(runtime.get("trigger_activations", 0)) <= 0:
            problems.append("geometry selector never triggered")
        if overrides <= 0:
            problems.append("geometry selector never overrode the policy")
        if min(active, feasible, no_feasible, policy_feasible) < 0:
            problems.append("geometry selector accounting field missing")
        elif feasible + no_feasible != active:
            problems.append("feasible/no-feasible frames do not reconcile")
        if overrides > feasible or policy_feasible > feasible:
            problems.append("geometry selector feasibility ledger is impossible")
        if int(checks.get("shield_engaged_records", -1)) != overrides:
            problems.append("selector and recorder override ledgers disagree")
    if problems:
        raise RuntimeError(
            f"invalid SA4-D4 {mode} evidence: " + "; ".join(problems)
        )


def _run_arm(
    mode: str,
    checkpoint: Path,
    output: Path,
    values: dict,
) -> dict:
    stem = f"{mode}_lateral_g4_d1_s818"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    recorder_path = output / f"{stem}_timing.json"
    before_path = output / f"{stem}_source_before.json"
    after_path = output / f"{stem}_source_after.json"
    _require_new_targets(
        [log_path, corridor_path, recorder_path, before_path, after_path]
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
        f"[SA4-D4] START {mode}: c6400, lateral, seed818, "
        "d1=200ms, 64 env x 2500 steps",
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
            str(recorder_path),
            "--d3_shield_mode",
            mode,
        ],
    )
    after = source_fingerprint()
    after_path.write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError(f"D4 source drift during {mode}")

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
    report = json.loads(recorder_path.read_text(encoding="utf-8"))
    _validate_report(
        report,
        mode=mode,
        checkpoint=checkpoint,
        corridor=corridor,
    )
    metrics = _metrics(corridor)
    result = {
        "mode": mode,
        "metrics": metrics,
        "graduation_threshold_pass": (
            metrics["sr"] >= PASS_SR
            and metrics["cr"] <= PASS_CR
            and metrics["to"] <= PASS_TO
        ),
        "selector_runtime": report["metadata"]["shield_runtime"],
        "recorder_counts": report["counts"],
        "recorder_self_check": report["self_check"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "files": {
            "log": str(log_path),
            "corridor": str(corridor_path),
            "timing": str(recorder_path),
            "source_before": str(before_path),
            "source_after": str(after_path),
        },
    }
    print(
        f"[SA4-D4] VALID {mode}: n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"obstacle_CR={metrics['obstacle_cr']:.4f} "
        f"wall_CR={metrics['wall_cr']:.4f} TO={metrics['to']:.4f} "
        f"overrides={result['selector_runtime']['override_environment_frames']}",
        flush=True,
    )
    return result


def _render_comparison(arms: list[dict]) -> str:
    baseline_metrics = arms[0]["metrics"]
    lines = [
        "# SA4-D4 geometry-feasible selector comparison",
        "",
        "> Fixed checkpoint c6400, SA4 lateral, seed 818, d1=200 ms. "
        "This is privileged diagnostic evidence, not a deployment shield.",
        "",
        (
            "| Arm | episodes | SR | obstacle CR | wall CR | total CR | TO | "
            "obstacle CR change | overrides | no feasible | Gate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        metrics = arm["metrics"]
        runtime = arm["selector_runtime"]
        obstacle_change = 100.0 * (
            metrics["obstacle_cr"] - baseline_metrics["obstacle_cr"]
        )
        lines.append(
            f"| {arm['mode']} | {metrics['n']:,} | {metrics['sr']:.2%} | "
            f"{metrics['obstacle_cr']:.2%} | {metrics['wall_cr']:.2%} | "
            f"{metrics['cr']:.2%} | {metrics['to']:.2%} | "
            f"{obstacle_change:+.2f} pp | "
            f"{int(runtime['override_environment_frames']):,} | "
            f"{int(runtime.get('no_feasible_active_frames', 0)):,} | "
            f"{'PASS' if arm['graduation_threshold_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Gate shown only for context: SR >= 90%, CR <= 10%, TO <= 5%.",
        (
            "Interpret obstacle CR together with wall CR and TO; moving "
            "failures between categories is not an improvement."
        ),
        "The selector uses privileged dynamic state and is an upper-bound mechanism probe.",
        "",
    ]
    return "\n".join(lines)


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
    manifest_path = output / "suite_manifest.json"
    comparison_json_path = output / "comparison.json"
    comparison_md_path = output / "COMPARISON.md"
    _require_new_targets(
        [protocol_path, manifest_path, comparison_json_path, comparison_md_path]
    )

    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d4_geometry_suite_preregistration/v1",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "test whether a delay-aware selector that jointly constrains "
            "pedestrian and LiDAR static/wall geometry reduces lateral "
            "obstacle collisions without converting them to walls/timeouts"
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
        "arms_in_fixed_order": list(ARMS),
        "geometry_selector_protocol": geometry_selector_protocol(),
        "baseline_identity_protocol": d3_shield_protocol(),
        "reported_outcomes": [
            "SR",
            "obstacle_CR",
            "wall_CR",
            "total_CR",
            "TO",
            "selector trigger/override/no-feasible accounting",
        ],
        "graduation_thresholds_for_context_only": {
            "sr_min": PASS_SR,
            "cr_max": PASS_CR,
            "to_max": PASS_TO,
        },
        "source_fingerprint": planned_fingerprint,
    }
    protocol_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[SA4-D4] protocol frozen before rollout: {protocol_path}",
        flush=True,
    )

    values = baseline.sa4.scene_values()
    arms = [
        _run_arm(mode, checkpoint, output, values)
        for mode in ARMS
    ]
    final_fingerprint = source_fingerprint()
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D4 source changed during the comparison")

    comparison = {
        "schema": "sa4_d4_geometry_comparison/v1",
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": geometry_selector_protocol()["sha256"],
        "arms": arms,
        "delta_geometry_minus_baseline": {
            key: arms[1]["metrics"][key] - arms[0]["metrics"][key]
            for key in ("sr", "cr", "obstacle_cr", "wall_cr", "to")
        },
        "limitations": [
            "single checkpoint and single evaluation seed",
            "privileged dynamic state and dynamic-return attribution",
            "static geometry is limited to the current 72-bin LiDAR surface samples",
            "constant-velocity dynamic prediction and receding-horizon action selection",
            "diagnostic upper bound, not a deployment safety claim",
        ],
    }
    comparison_json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    comparison_md_path.write_text(
        _render_comparison(arms), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d4_geometry_suite/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": geometry_selector_protocol()["sha256"],
        "source_fingerprint": final_fingerprint,
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
        f"[SA4-D4] COMPLETE comparison={comparison_md_path}", flush=True
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
