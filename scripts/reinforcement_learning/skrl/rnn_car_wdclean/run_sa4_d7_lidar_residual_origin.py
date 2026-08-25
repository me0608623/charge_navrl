"""Run the frozen SA4-D7 baseline-only LiDAR residual-origin audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_d3_baseline as baseline  # noqa: E402
from d6_static_feasibility_sensitivity import (  # noqa: E402
    SOURCE_NAMES,
    frozen_variant_key,
    static_sensitivity_protocol,
)
from d7_lidar_residual_origin import (  # noqa: E402
    ACTUAL_SOURCE_NAMES,
    CREATION_NAMES,
    MECHANISM_NAMES,
    REPORT_SCHEMA,
    STAGE_NAMES,
    residual_origin_protocol,
)
from run_sa4_d6_static_sensitivity import validate_d6_report  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


D4_SELECTOR = HERE / "d4_geometry_selector.py"
D6_SENSITIVITY = HERE / "d6_static_feasibility_sensitivity.py"
D7_ORIGIN = HERE / "d7_lidar_residual_origin.py"
OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)


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
        "d6_sensitivity": D6_SENSITIVITY,
        "d7_origin": D7_ORIGIN,
        "lidar_observation": OBS_FUNCTIONS,
    }
    return {name: sha256_of(path) for name, path in paths.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D7 audit refuses to overwrite evidence: " + ", ".join(existing)
        )


def gpu_is_busy() -> str | None:
    """Return non-desktop compute rows using more than 2 GiB."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()!r}")
    heavy = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        if int(fields[2]) > 2000:
            heavy.append(line.strip())
    return "; ".join(heavy) if heavy else None


def validate_d7_report(
    report: dict,
    *,
    checkpoint: Path,
    corridor_report: dict,
    d3_report: dict,
    d6_report: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    if report.get("schema") != REPORT_SCHEMA:
        problems.append(f"unexpected D7 schema {report.get('schema')!r}")
    if report.get("mode") != "lidar_residual_origin":
        problems.append(f"unexpected D7 mode {report.get('mode')!r}")
    protocol = report.get("protocol") or {}
    if protocol.get("sha256") != residual_origin_protocol()["sha256"]:
        problems.append("D7 protocol hash drift")
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
    if metadata.get("d6_reconciliation_ok") is not True:
        problems.append("D6 reconciliation is not true")
    if metadata.get("d3_reconciliation_ok") is not True:
        problems.append("D3 reconciliation is not true")
    if int(metadata.get("d3_delay_alignment_errors", -1)) != 0:
        problems.append("D3 delay alignment errors are nonzero")

    runtime = report.get("runtime") or {}
    if int(runtime.get("calls", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("D7 call coverage mismatch")
    if int(runtime.get("transitions", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("D7 transition coverage mismatch")
    if int(runtime.get("environment_frames", -1)) != expected_records:
        problems.append("D7 environment-frame coverage mismatch")
    if int(runtime.get("action_identity_checks", -1)) != expected_records:
        problems.append("D7 action identity coverage mismatch")
    if int(runtime.get("action_identity_errors", -1)) != 0:
        problems.append("D7 modified policy actions")
    if int(runtime.get("trace_match_candidates", 0)) < baseline.ROLLOUT_STEPS:
        problems.append("D7 exact trace matches are incomplete")

    checks = report.get("self_check") or {}
    for key in (
        "baseline_action_identity_ok",
        "transition_coverage_ok",
        "record_coverage_ok",
        "exact_policy_trace_match_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"{key} is not true")

    scopes = report.get("scopes") or {}
    d6_scopes = d6_report.get("matrix") or {}
    expected_scope_frames = {
        "all_evaluated": expected_records,
        "frozen_static_blocked": int(
            d6_scopes["frozen_static_blocked"]["frames"]
        ),
        "frozen_static_feasible": expected_records
        - int(d6_scopes["frozen_static_blocked"]["frames"]),
        "dynamic_collision_transition": int(
            d3_report["counts"]["dynamic_collision"]
        ),
    }
    for scope_name, expected_frames in expected_scope_frames.items():
        scope = scopes.get(scope_name) or {}
        if int(scope.get("frames", -1)) != expected_frames:
            problems.append(f"{scope_name} frame count mismatch")
        residual = int(scope.get("final_residual_points", -1))
        for field, names in (
            ("final_creation_stage", CREATION_NAMES),
            ("final_mechanism", MECHANISM_NAMES),
            ("final_actual_angle_source", ACTUAL_SOURCE_NAMES),
            ("final_winner_raw_hit_source", ACTUAL_SOURCE_NAMES),
        ):
            values = scope.get(field) or {}
            if sum(int(values.get(name, -1)) for name in names) != residual:
                problems.append(f"{scope_name} {field} does not partition residuals")
        stage_counts = scope.get("stage_residual_counts") or {}
        if int(stage_counts.get("d6_model_final", -2)) != residual:
            problems.append(f"{scope_name} final stage residual mismatch")

    all_scope = scopes.get("all_evaluated") or {}
    d6_points = d6_report["lidar_point_attribution"]
    d7_sources = all_scope.get("final_center_source_counts") or {}
    for name in SOURCE_NAMES:
        if int(d7_sources.get(name, -1)) != int(d6_points["counts"][name]):
            problems.append(f"D7/D6 final source count mismatch for {name}")
    if int(all_scope.get("final_residual_points", -1)) != int(
        d6_points["counts"]["residual"]
    ):
        problems.append("D7/D6 residual point total mismatch")
    if int(corridor_report.get("episodes", -1)) != int(
        d3_report["counts"]["completed_episodes"]
    ):
        problems.append("D3/corridor episode ledger mismatch")
    if problems:
        raise RuntimeError("invalid SA4-D7 evidence: " + "; ".join(problems))


def _fraction(counts: dict, key: str, total: int) -> float:
    return int(counts[key]) / total if total else float("nan")


def _render_summary(report: dict, corridor: dict) -> str:
    all_scope = report["scopes"]["all_evaluated"]
    blocked = report["scopes"]["frozen_static_blocked"]
    total = int(all_scope["final_residual_points"])
    blocked_total = int(blocked["final_residual_points"])
    mechanisms = all_scope["final_mechanism"]
    creation = all_scope["final_creation_stage"]
    blocked_mechanisms = blocked["final_mechanism"]
    rows = [
        f"| {name} | {int(mechanisms[name]):,} | {_fraction(mechanisms, name, total):.2%} | "
        f"{int(blocked_mechanisms[name]):,} | {_fraction(blocked_mechanisms, name, blocked_total):.2%} |"
        for name in MECHANISM_NAMES
    ]
    creation_rows = [
        f"| {name} | {int(creation[name]):,} | {_fraction(creation, name, total):.2%} |"
        for name in CREATION_NAMES
    ]
    return "\n".join(
        [
            "# SA4-D7 LiDAR residual-origin audit",
            "",
            "> Fixed c6400, stage4 lateral, seed818, d1=200 ms. Baseline-only realized trace; no action modification.",
            "",
            f"- episodes: {int(corridor['episodes']):,}",
            f"- SR / CR / TO: {float(corridor['success_rate']):.2%} / {float(corridor['collision_rate']):.2%} / {float(corridor['timeout_rate']):.2%}",
            f"- final residual points: {total:,}",
            f"- frozen static-blocked frames: {int(blocked['frames']):,}",
            "",
            "## Final residual mechanism",
            "",
            "| mechanism | all count | all share | blocked-frame count | blocked-frame share |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Most recent stage that created the final residual",
            "",
            "| stage | count | share |",
            "|---|---:|---:|",
            *creation_rows,
            "",
            "Mechanism labels describe the realized 72-bin winner. They do not identify an unmatched USD prim or establish a population effect.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="wait for every >2 GiB third-party compute process to exit",
    )
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
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

    busy = gpu_is_busy()
    while busy and args.wait_for_gpu:
        if args.gpu_poll_seconds <= 0.0:
            raise ValueError("--gpu-poll-seconds must be positive")
        print(f"[SA4-D7] waiting for exclusive GPU: {busy}", flush=True)
        time.sleep(args.gpu_poll_seconds)
        busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"D7 refuses to share a busy GPU: {busy}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol": output / "PROTOCOL.json",
        "log": output / "baseline_lateral_g4_d1_s818.log",
        "corridor": output / "baseline_lateral_g4_d1_s818_corridor.json",
        "d3": output / "baseline_lateral_g4_d1_s818_d3.json",
        "d6": output / "baseline_lateral_g4_d1_s818_d6.json",
        "d7": output / "baseline_lateral_g4_d1_s818_d7.json",
        "summary": output / "SUMMARY.md",
        "manifest": output / "suite_manifest.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    _require_new_targets(list(paths.values()))

    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d7_lidar_residual_origin_preregistration/v2",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "identify whether D6 residual 72-bin points originate from measured "
            "distractors, dropout, sigma/bias, bin-center reconstruction, dynamic "
            "return mismatch, or privileged geometry incompleteness"
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
            "vlp16_noise_mode": "full",
        },
        "action_arm": "fresh identity baseline only",
        "d7_protocol": residual_origin_protocol(),
        "d6_protocol": static_sensitivity_protocol(),
        "source_fingerprint": planned_fingerprint,
        "forbidden_during_run": [
            "action override",
            "training",
            "reward/scene/checkpoint/noise changes",
            "SA5 launch",
        ],
    }
    paths["protocol"].write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["source_before"].write_text(
        json.dumps(planned_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[SA4-D7] protocol frozen before rollout: {paths['protocol']}", flush=True)

    values = baseline.sa4.scene_values()
    actuator_args = baseline.sa4.base.fixed_actuator_cli_args(
        baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
    )
    scene_args = baseline.sa4.build_scene_args(
        baseline.SCENARIO,
        seed=baseline.SEED,
        actuator_args=actuator_args,
        corridor_json=paths["corridor"],
    )
    _run_play(
        checkpoint,
        paths["log"],
        num_envs=baseline.NUM_ENVS,
        steps=baseline.ROLLOUT_STEPS,
        extra=[
            *scene_args,
            "--d3_yield_audit",
            "--d3_yield_output",
            str(paths["d3"]),
            "--d3_shield_mode",
            "baseline",
            "--d6_static_sensitivity",
            "--d6_static_sensitivity_output",
            str(paths["d6"]),
            "--d7_lidar_residual_origin",
            "--d7_lidar_residual_origin_output",
            str(paths["d7"]),
        ],
    )

    final_fingerprint = source_fingerprint()
    paths["source_after"].write_text(
        json.dumps(final_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D7 source changed during rollout")

    baseline.sa4.base.verify_fixed_actuator_runtime(
        paths["log"], baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
    )
    stage_banner = baseline.sa4.base.verify_common_runtime(paths["log"], values)
    console = parse_play_summary(str(paths["log"]))
    if not console or console.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {paths['log']}")
    corridor = json.loads(paths["corridor"].read_text(encoding="utf-8"))
    scene_contract = baseline.sa4.base.verify_corridor_runtime(
        paths["log"], corridor, "lateral", values
    )
    metrics = baseline.sa4.base.reconcile_corridor_metrics(console, corridor)
    d3_report = json.loads(paths["d3"].read_text(encoding="utf-8"))
    baseline.validate_d3_report(
        d3_report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        corridor_report=corridor,
    )
    d6_report = json.loads(paths["d6"].read_text(encoding="utf-8"))
    validate_d6_report(
        d6_report,
        checkpoint=checkpoint,
        corridor_report=corridor,
        d3_report=d3_report,
    )
    d7_report = json.loads(paths["d7"].read_text(encoding="utf-8"))
    validate_d7_report(
        d7_report,
        checkpoint=checkpoint,
        corridor_report=corridor,
        d3_report=d3_report,
        d6_report=d6_report,
    )
    paths["summary"].write_text(
        _render_summary(d7_report, corridor), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d7_lidar_residual_origin_bundle/v2",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "not_a_graduation_gate": True,
        "policy_actions_modified": False,
        "training_started": False,
        "next_stage_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(paths["protocol"]),
        "protocol_sha256": residual_origin_protocol()["sha256"],
        "fixed_cell": preregistration["fixed_cell"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics": metrics,
        "d3_counts": d3_report["counts"],
        "d3_self_check": d3_report["self_check"],
        "d6_runtime": d6_report["runtime"],
        "d6_self_check": d6_report["self_check"],
        "d7_runtime": d7_report["runtime"],
        "d7_self_check": d7_report["self_check"],
        "d7_scopes": d7_report["scopes"],
        "source_fingerprint": final_fingerprint,
        "files": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D7] COMPLETE valid diagnostic evidence: {paths['manifest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
