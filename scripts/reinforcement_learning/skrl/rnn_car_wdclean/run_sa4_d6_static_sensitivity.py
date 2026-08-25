"""Run the frozen SA4-D6 baseline-only static-feasibility sensitivity audit."""

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
from d6_static_feasibility_sensitivity import (  # noqa: E402
    SOURCE_NAMES,
    frozen_variant_key,
    static_sensitivity_protocol,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


D4_SELECTOR = HERE / "d4_geometry_selector.py"
D6_SENSITIVITY = HERE / "d6_static_feasibility_sensitivity.py"


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
    }
    return {name: sha256_of(path) for name, path in paths.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D6 audit refuses to overwrite evidence: " + ", ".join(existing)
        )


def _variant(report: dict, scope: str, key: str) -> dict:
    return report["matrix"][scope]["variants"][key]


def _assert_monotonic(report: dict, problems: list[str]) -> None:
    scope = "all_evaluated"
    horizons = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
    clearances = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15)
    for clearance in clearances:
        rates = [
            float(
                _variant(
                    report,
                    scope,
                    f"lidar_all_h{horizon:.1f}_c{clearance:.2f}",
                )["static_any_feasible_rate"]
            )
            for horizon in horizons
        ]
        if any(left + 1.0e-12 < right for left, right in zip(rates, rates[1:])):
            problems.append(
                f"static feasibility increased with horizon at c={clearance:.2f}"
            )
    for horizon in horizons:
        rates = [
            float(
                _variant(
                    report,
                    scope,
                    f"lidar_all_h{horizon:.1f}_c{clearance:.2f}",
                )["static_any_feasible_rate"]
            )
            for clearance in clearances
        ]
        if any(left + 1.0e-12 < right for left, right in zip(rates, rates[1:])):
            problems.append(
                f"static feasibility increased with clearance at h={horizon:.1f}"
            )


def validate_d6_report(
    report: dict,
    *,
    checkpoint: Path,
    corridor_report: dict,
    d3_report: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    protocol = report.get("protocol") or {}
    if report.get("schema") != "sa4_d6_static_feasibility_report/v2":
        problems.append(f"unexpected D6 schema {report.get('schema')!r}")
    if report.get("mode") != "static_feasibility_sensitivity":
        problems.append(f"unexpected D6 mode {report.get('mode')!r}")
    if protocol.get("sha256") != static_sensitivity_protocol()["sha256"]:
        problems.append("D6 protocol hash drift")
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
    if metadata.get("d3_reconciliation_ok") is not True:
        problems.append("D3 reconciliation is not true")
    if int(metadata.get("d3_delay_alignment_errors", -1)) != 0:
        problems.append("D3 delay alignment errors are nonzero")
    actuator = metadata.get("actuator") or {}
    if actuator.get("delay_range") != [1, 1]:
        problems.append("actuator delay is not fixed d1")
    if actuator.get("velocity_scale_range") != [1.0, 1.0]:
        problems.append("actuator scale is not fixed at 1")
    if float(actuator.get("motor_lag_alpha", -1.0)) != 1.0:
        problems.append("actuator motor lag is not disabled")

    runtime = report.get("runtime") or {}
    if int(runtime.get("calls", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("D6 call coverage mismatch")
    if int(runtime.get("transitions", -1)) != baseline.ROLLOUT_STEPS:
        problems.append("D6 transition coverage mismatch")
    if int(runtime.get("environment_frames", -1)) != expected_records:
        problems.append("D6 environment-frame coverage mismatch")
    if int(runtime.get("action_identity_checks", -1)) != expected_records:
        problems.append("D6 action identity coverage mismatch")
    if int(runtime.get("action_identity_errors", -1)) != 0:
        problems.append("D6 modified policy actions")

    checks = report.get("self_check") or {}
    for key in (
        "baseline_action_identity_ok",
        "transition_coverage_ok",
        "record_coverage_ok",
        "source_partition_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"{key} is not true")

    attribution = report.get("lidar_point_attribution") or {}
    source_counts = attribution.get("counts") or {}
    if int(attribution.get("static_valid_points", 0)) <= 0:
        problems.append("D6 captured no static-valid LiDAR points")
    if sum(int(source_counts.get(name, -1)) for name in SOURCE_NAMES) != int(
        attribution.get("static_valid_points", -2)
    ):
        problems.append("D6 source attribution does not reconcile")
    if int(source_counts.get("wall", 0)) <= 0:
        problems.append("D6 attributed no LiDAR points to corridor walls")
    if int(source_counts.get("static_obstacle", 0)) <= 0:
        problems.append("D6 attributed no LiDAR points to static obstacles")

    matrix = report.get("matrix") or {}
    all_scope = matrix.get("all_evaluated") or {}
    if int(all_scope.get("frames", -1)) != expected_records:
        problems.append("D6 all-evaluated scope coverage mismatch")
    collision_frames = int(
        (matrix.get("dynamic_collision_transition") or {}).get("frames", -1)
    )
    expected_collision = int(d3_report["counts"]["dynamic_collision"])
    if collision_frames != expected_collision:
        problems.append("D6/D3 dynamic collision count mismatch")
    if int(corridor_report.get("episodes", -1)) != int(
        d3_report["counts"]["completed_episodes"]
    ):
        problems.append("D3/corridor episode ledger mismatch")

    frozen = _variant(report, "all_evaluated", frozen_variant_key())
    frozen_rate = float(frozen["static_any_feasible_rate"])
    for key in (
        "lidar_without_wall_assigned_h2.4_c0.10",
        "lidar_without_static_assigned_h2.4_c0.10",
        "lidar_without_ambiguous_h2.4_c0.10",
        "lidar_without_residual_h2.4_c0.10",
        "no_static_lidar_dynamic_ceiling",
    ):
        if float(
            _variant(report, "all_evaluated", key)[
                "static_any_feasible_rate"
            ]
        ) + 1.0e-12 < frozen_rate:
            problems.append(f"source removal unexpectedly reduced feasibility: {key}")
    blocked = matrix.get("frozen_static_blocked") or {}
    if int(blocked.get("frames", 0)) <= 0:
        problems.append("D6 found no frozen static-blocked frames")
    elif float(
        blocked["variants"][frozen_variant_key()][
            "static_any_feasible_rate"
        ]
    ) != 0.0:
        problems.append("frozen static-blocked scope contains feasible frames")
    _assert_monotonic(report, problems)
    if problems:
        raise RuntimeError("invalid SA4-D6 evidence: " + "; ".join(problems))


def _rate(report: dict, scope: str, key: str, metric: str) -> float:
    value = _variant(report, scope, key)[metric]
    return float(value) if value is not None else float("nan")


def _render_summary(report: dict, corridor: dict) -> str:
    all_scope = "all_evaluated"
    blocked_scope = "frozen_static_blocked"
    frozen = frozen_variant_key()
    points = report["lidar_point_attribution"]
    horizon_lines = [
        f"| {horizon:.1f} | "
        f"{_rate(report, all_scope, f'lidar_all_h{horizon:.1f}_c0.10', 'static_any_feasible_rate'):.2%} | "
        f"{_rate(report, blocked_scope, f'lidar_all_h{horizon:.1f}_c0.10', 'static_any_feasible_rate'):.2%} |"
        for horizon in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
    ]
    clearance_lines = [
        f"| {clearance:.2f} | "
        f"{_rate(report, all_scope, f'lidar_all_h2.4_c{clearance:.2f}', 'static_any_feasible_rate'):.2%} | "
        f"{_rate(report, blocked_scope, f'lidar_all_h2.4_c{clearance:.2f}', 'static_any_feasible_rate'):.2%} |"
        for clearance in (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15)
    ]
    source_rows = (
        ("remove wall-assigned", "lidar_without_wall_assigned_h2.4_c0.10"),
        ("remove static-assigned", "lidar_without_static_assigned_h2.4_c0.10"),
        ("remove ambiguous", "lidar_without_ambiguous_h2.4_c0.10"),
        ("remove residual", "lidar_without_residual_h2.4_c0.10"),
        ("remove all static LiDAR", "no_static_lidar_dynamic_ceiling"),
    )
    source_lines = [
        f"| {label} | "
        f"{_rate(report, all_scope, key, 'static_any_feasible_rate'):.2%} | "
        f"{_rate(report, blocked_scope, key, 'static_any_feasible_rate'):.2%} |"
        for label, key in source_rows
    ]
    return "\n".join(
        [
            "# SA4-D6 static-feasibility sensitivity",
            "",
            "> Fixed c6400, stage4 lateral, seed818, d1=200 ms. Baseline shadow only; policy actions unchanged.",
            "",
            f"- episodes: {int(corridor['episodes']):,}",
            f"- SR / CR / TO: {float(corridor['success_rate']):.2%} / {float(corridor['collision_rate']):.2%} / {float(corridor['timeout_rate']):.2%}",
            f"- frozen static-any rate: {_rate(report, all_scope, frozen, 'static_any_feasible_rate'):.2%}",
            f"- frozen static-blocked frames: {report['matrix'][blocked_scope]['frames']:,}",
            "",
            "## LiDAR attribution",
            "",
            *[
                f"- {name}: {int(points['counts'][name]):,} ({float(points['fractions'][name]):.2%})"
                for name in SOURCE_NAMES
            ],
            "",
            "## Horizon sensitivity at clearance=0.10 m",
            "",
            "| horizon (s) | all-frame static-any | rescue within frozen-blocked |",
            "|---:|---:|---:|",
            *horizon_lines,
            "",
            "## Clearance sensitivity at horizon=2.4 s",
            "",
            "| clearance (m) | all-frame static-any | rescue within frozen-blocked |",
            "|---:|---:|---:|",
            *clearance_lines,
            "",
            "## Source ablation at h=2.4 s, c=0.10 m",
            "",
            "| ablation | all-frame static-any | rescue within frozen-blocked |",
            "|---|---:|---:|",
            *source_lines,
            "",
            "Rescue rates diagnose the frozen D4 LiDAR-point model. They are not physical-solvability or deployment-safety rates.",
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
    paths = {
        "protocol": output / "PROTOCOL.json",
        "log": output / "baseline_lateral_g4_d1_s818.log",
        "corridor": output / "baseline_lateral_g4_d1_s818_corridor.json",
        "d3": output / "baseline_lateral_g4_d1_s818_d3.json",
        "d6": output / "baseline_lateral_g4_d1_s818_d6.json",
        "summary": output / "SUMMARY.md",
        "manifest": output / "suite_manifest.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    _require_new_targets(list(paths.values()))

    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d6_static_sensitivity_preregistration/v2",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "identify whether the frozen D4 static-feasibility bottleneck is "
            "associated with LiDAR source attribution, 2.4s horizon, or 0.10m clearance"
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
        "d6_protocol": static_sensitivity_protocol(),
        "d4_geometry_protocol": geometry_selector_protocol(),
        "source_fingerprint": planned_fingerprint,
        "forbidden_during_run": [
            "action override",
            "training",
            "reward/scene/checkpoint changes",
            "SA5 launch",
        ],
    }
    paths["protocol"].write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["source_before"].write_text(
        json.dumps(planned_fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[SA4-D6] protocol frozen before rollout: {paths['protocol']}", flush=True)

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
        ],
    )

    final_fingerprint = source_fingerprint()
    paths["source_after"].write_text(
        json.dumps(final_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D6 source changed during rollout")

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
    paths["summary"].write_text(
        _render_summary(d6_report, corridor), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d6_static_sensitivity_bundle/v2",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "not_a_graduation_gate": True,
        "policy_actions_modified": False,
        "training_started": False,
        "next_stage_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(paths["protocol"]),
        "protocol_sha256": static_sensitivity_protocol()["sha256"],
        "fixed_cell": preregistration["fixed_cell"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics": metrics,
        "d3_counts": d3_report["counts"],
        "d3_self_check": d3_report["self_check"],
        "d6_runtime": d6_report["runtime"],
        "d6_self_check": d6_report["self_check"],
        "lidar_point_attribution": d6_report["lidar_point_attribution"],
        "matrix": d6_report["matrix"],
        "source_fingerprint": final_fingerprint,
        "files": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D6] COMPLETE valid diagnostic evidence: {paths['manifest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
