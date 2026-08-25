"""Run the frozen SA4-D3 lateral timing baseline.

This is diagnostic evidence, not a graduation gate. The scene, actuator bundle,
seed and rollout budget are intentionally constants so a later intervention arm
can reuse the same contract. Intervention modes remain disabled in
``d3_yield_recorder`` until this baseline has been analysed and trigger values
have been frozen.
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

import run_sa4_checkpoint_screen as sa4  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


GEOMETRY_STAGE = 4
SCENARIO = "corridor_lateral"
SEED = 818
DELAY_STEPS = 1
ACTUATOR_PROFILE = "sa1_delay_only"
NUM_ENVS = 64
ROLLOUT_STEPS = 2500

PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
ACTION_TERM = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
RECORDER = HERE / "d3_yield_recorder.py"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "play": PLAY,
        "action_term": ACTION_TERM,
        "recorder": RECORDER,
    }
    return {name: sha256_of(path) for name, path in paths.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D3 baseline refuses to overwrite existing evidence: "
            + ", ".join(existing)
        )


def validate_d3_report(
    report: dict,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    corridor_report: dict,
) -> None:
    problems = []
    expected_records = NUM_ENVS * ROLLOUT_STEPS
    if report.get("schema") != "sa4_d3_yield_timing/v2":
        problems.append(f"unexpected schema {report.get('schema')!r}")
    if report.get("mode") != "baseline":
        problems.append(f"unexpected mode {report.get('mode')!r}")
    metadata = report.get("metadata") or {}
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("checkpoint path mismatch")
    if int(metadata.get("stage", -1)) != GEOMETRY_STAGE:
        problems.append("stage mismatch")
    if int(metadata.get("seed", -1)) != SEED:
        problems.append("seed mismatch")
    if int(metadata.get("num_envs", -1)) != NUM_ENVS:
        problems.append("num_envs mismatch")
    if int(metadata.get("num_dynamic_obstacles", -1)) != int(
        sa4.scene_values()["corridor_dynamic"]
    ):
        problems.append("dynamic obstacle count mismatch")
    if int(metadata.get("rollout_steps_requested", -1)) != ROLLOUT_STEPS:
        problems.append("rollout step mismatch")
    if metadata.get("corridor_motion_mode") != "lateral":
        problems.append("corridor family mismatch")
    if not bool(metadata.get("baseline_tensor_identity_checked_each_step")):
        problems.append("runtime baseline identity guard missing")
    actuator = metadata.get("actuator") or {}
    if actuator.get("delay_range") != [1, 1]:
        problems.append("actuator delay is not fixed d1")
    if actuator.get("velocity_scale_range") != [1.0, 1.0]:
        problems.append("actuator scale is not fixed at 1")
    if float(actuator.get("motor_lag_alpha", -1.0)) != 1.0:
        problems.append("actuator motor lag is not disabled")

    counts = report.get("counts") or {}
    if int(counts.get("records", -1)) != expected_records:
        problems.append(
            f"record coverage {counts.get('records')} != {expected_records}"
        )
    if int(counts.get("completed_episodes", -1)) != int(
        corridor_report["episodes"]
    ):
        problems.append("episode ledger disagrees with corridor report")
    if int(counts.get("noncollision_closest_approach", 0)) <= 0:
        problems.append("no per-obstacle non-collision controls were captured")
    if int(counts.get("successful_noncollision_closest_approach", 0)) <= 0:
        problems.append("no successful closest-approach controls were captured")
    self_check = report.get("self_check") or {}
    required_checks = (
        "baseline_action_identity_ok",
        "record_coverage_ok",
        "episode_reconciliation_ok",
        "delay_alignment_ok",
        "dynamic_collision_reconciliation_ok",
        "reconciliation_ok",
    )
    for name in required_checks:
        if self_check.get(name) is not True:
            problems.append(f"{name} is not true")
    if int(self_check.get("shield_engaged_records", -1)) != 0:
        problems.append("baseline shield engaged")
    if int(self_check.get("delay_alignment_errors", -1)) != 0:
        problems.append("d1 delay alignment error")
    if int(self_check.get("delay_alignment_samples", 0)) <= 0:
        problems.append("no d1 delay alignment samples")

    # The hash is supplied by the runner rather than duplicated in the large
    # event file. Keeping it in this validation signature ensures the manifest
    # and report are checked as one evidence bundle.
    if len(checkpoint_sha256) != 64:
        problems.append("invalid checkpoint SHA-256")
    if problems:
        raise RuntimeError("invalid SA4-D3 baseline report: " + "; ".join(problems))


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
            f"checkpoint hash mismatch for {checkpoint}: expected "
            f"{args.expect_checkpoint_sha256}, got {checkpoint_sha256}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "baseline_lateral_g4_d1_s818.log"
    corridor_path = output / "baseline_lateral_g4_d1_s818_corridor.json"
    d3_path = output / "baseline_lateral_g4_d1_s818_d3.json"
    manifest_path = output / "baseline_manifest.json"
    before_path = output / "source_fingerprint_before.json"
    after_path = output / "source_fingerprint_after.json"
    _require_new_targets(
        [
            log_path,
            corridor_path,
            d3_path,
            manifest_path,
            before_path,
            after_path,
        ]
    )

    values = sa4.scene_values()
    actuator_args = sa4.base.fixed_actuator_cli_args(
        DELAY_STEPS, ACTUATOR_PROFILE
    )
    scene_args = sa4.build_scene_args(
        SCENARIO,
        seed=SEED,
        actuator_args=actuator_args,
        corridor_json=corridor_path,
    )
    d3_args = [
        "--d3_yield_audit",
        "--d3_yield_output",
        str(d3_path),
        "--d3_shield_mode",
        "baseline",
    ]
    fingerprint_before = source_fingerprint()
    before_path.write_text(
        json.dumps(fingerprint_before, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[SA4-D3] diagnostic baseline: c6400, stage4 lateral, "
        "seed818, d1=200ms, 64 env x 2500 steps; no intervention",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=NUM_ENVS,
        steps=ROLLOUT_STEPS,
        extra=[*scene_args, *d3_args],
    )

    fingerprint_after = source_fingerprint()
    after_path.write_text(
        json.dumps(fingerprint_after, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if fingerprint_before != fingerprint_after:
        raise RuntimeError(
            "D3 source fingerprint changed during the rollout; evidence is invalid"
        )

    sa4.base.verify_fixed_actuator_runtime(
        log_path, DELAY_STEPS, ACTUATOR_PROFILE
    )
    stage_banner = sa4.base.verify_common_runtime(log_path, values)
    console = parse_play_summary(str(log_path))
    if not console or console.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = sa4.base.verify_corridor_runtime(
        log_path, corridor_report, "lateral", values
    )
    metrics = sa4.base.reconcile_corridor_metrics(console, corridor_report)
    d3_report = json.loads(d3_path.read_text(encoding="utf-8"))
    validate_d3_report(
        d3_report,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        corridor_report=corridor_report,
    )

    manifest = {
        "schema": "sa4_d3_baseline_bundle/v1",
        "status": "VALID_DIAGNOSTIC_EVIDENCE",
        "not_a_graduation_gate": True,
        "interventions_enabled": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_cell": {
            "geometry_stage": GEOMETRY_STAGE,
            "scenario": SCENARIO,
            "seed": SEED,
            "delay_steps": DELAY_STEPS,
            "delay_ms": 200,
            "actuator_profile": ACTUATOR_PROFILE,
            "num_envs": NUM_ENVS,
            "steps": ROLLOUT_STEPS,
        },
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics": metrics,
        "d3_counts": d3_report["counts"],
        "d3_self_check": d3_report["self_check"],
        "source_fingerprint": fingerprint_after,
        "files": {
            "log": str(log_path),
            "corridor": str(corridor_path),
            "d3": str(d3_path),
            "source_before": str(before_path),
            "source_after": str(after_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[SA4-D3] VALID diagnostic evidence: "
        f"n={metrics['n']} SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f} "
        f"dynamic_collision_events={d3_report['counts']['dynamic_collision']} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
