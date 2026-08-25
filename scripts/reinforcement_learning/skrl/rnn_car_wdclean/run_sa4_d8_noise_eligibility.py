"""Run the fixed SA4-D8 current-vs-valid-return-only noise shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_d3_baseline as baseline  # noqa: E402
import run_sa4_d4_geometry_suite as d4_suite  # noqa: E402
from d8_noise_eligibility_sensitivity import (  # noqa: E402
    REPORT_SCHEMA,
    noise_eligibility_protocol,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


D4_SELECTOR = HERE / "d4_geometry_selector.py"
D7_MATCHER = HERE / "d7_lidar_residual_origin.py"
D8_AUDIT = HERE / "d8_noise_eligibility_sensitivity.py"
OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "baseline_runner": Path(baseline.__file__).resolve(),
        "play": baseline.PLAY,
        "action_term": baseline.ACTION_TERM,
        "d3_recorder": baseline.RECORDER,
        "d4_selector": D4_SELECTOR,
        "d7_trace_matcher": D7_MATCHER,
        "d8_audit": D8_AUDIT,
        "lidar_observation": OBS_FUNCTIONS,
    }
    return {name: sha256_of(path) for name, path in files.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D8 audit refuses to overwrite evidence: " + ", ".join(existing)
        )


def gpu_is_busy() -> str | None:
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
        if len(fields) == 3 and int(fields[2]) > 2000:
            heavy.append(line.strip())
    return "; ".join(heavy) if heavy else None


def validate_d8_report(
    report: dict,
    *,
    checkpoint: Path,
    corridor: dict,
    d3_report: dict,
) -> None:
    problems: list[str] = []
    expected_records = baseline.NUM_ENVS * baseline.ROLLOUT_STEPS
    if report.get("schema") != REPORT_SCHEMA:
        problems.append(f"unexpected D8 schema {report.get('schema')!r}")
    if report.get("mode") != "mixed_pixel_valid_return_shadow":
        problems.append("D8 mode mismatch")
    if (report.get("protocol") or {}).get("sha256") != noise_eligibility_protocol()[
        "sha256"
    ]:
        problems.append("D8 protocol hash drift")
    metadata = report.get("metadata") or {}
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("checkpoint path mismatch")
    fixed = {
        "stage": baseline.GEOMETRY_STAGE,
        "seed": baseline.SEED,
        "num_envs": baseline.NUM_ENVS,
        "rollout_steps_requested": baseline.ROLLOUT_STEPS,
    }
    for key, expected in fixed.items():
        if int(metadata.get(key, -1)) != expected:
            problems.append(f"{key} mismatch")
    if metadata.get("corridor_motion_mode") != "lateral":
        problems.append("corridor family mismatch")
    if metadata.get("baseline_policy_action_unchanged") is not True:
        problems.append("baseline action identity declaration missing")
    if metadata.get("counterfactual_fed_to_policy") is not False:
        problems.append("counterfactual policy isolation declaration missing")
    if metadata.get("d3_reconciliation_ok") is not True:
        problems.append("D3 reconciliation is not true")
    if int(metadata.get("d3_delay_alignment_errors", -1)) != 0:
        problems.append("D3 delay alignment errors are nonzero")

    runtime = report.get("runtime") or {}
    for key, expected in (
        ("calls", baseline.ROLLOUT_STEPS),
        ("transitions", baseline.ROLLOUT_STEPS),
        ("environment_frames", expected_records),
        ("action_identity_checks", expected_records),
        ("action_identity_errors", 0),
        ("monotonicity_errors", 0),
        ("dynamic_grid_mismatch_frames", 0),
    ):
        if int(runtime.get(key, -1)) != expected:
            problems.append(f"runtime {key} mismatch")
    if int(runtime.get("active_environment_frames", 0)) <= 0:
        problems.append("D8 observed no frozen D4 active frames")
    if int(runtime.get("trace_match_candidates", 0)) < baseline.ROLLOUT_STEPS:
        problems.append("exact trace matches are incomplete")

    noise = report.get("noise_ledger") or {}
    if int(noise.get("realized_distractor_rays", -1)) != int(
        noise.get("eligible_distractor_rays", -2)
    ) + int(noise.get("removed_no_return_or_dropped_distractor_rays", -3)):
        problems.append("noise eligibility ledger mismatch")
    if int(noise.get("removed_no_return_or_dropped_distractor_rays", 0)) <= 0:
        problems.append("counterfactual removed no distractor rays")

    paired = report.get("paired_feasibility") or {}
    active = int(paired.get("active_frames", -1))
    quadrants = sum(
        int(paired.get(key, -1))
        for key in (
            "both_feasible_frames",
            "current_only_feasible_frames",
            "corrected_only_feasible_frames",
            "both_no_feasible_frames",
        )
    )
    if quadrants != active:
        problems.append("paired feasibility quadrants do not reconcile")
    if int(paired.get("current_no_feasible_frames", -1)) != active - int(
        paired.get("current_feasible_frames", -2)
    ):
        problems.append("current no-feasible ledger mismatch")
    checks = report.get("self_check") or {}
    if checks.get("reconciliation_ok") is not True:
        problems.append("D8 reconciliation is not true")
    if int(corridor.get("episodes", -1)) != int(
        d3_report["counts"]["completed_episodes"]
    ):
        problems.append("D3/corridor episode ledger mismatch")
    if problems:
        raise RuntimeError("invalid SA4-D8 evidence: " + "; ".join(problems))


def _render(report: dict, metrics: dict) -> str:
    noise = report["noise_ledger"]
    paired = report["paired_feasibility"]
    removed_share = (
        noise["removed_no_return_or_dropped_distractor_rays"]
        / noise["realized_distractor_rays"]
        if noise["realized_distractor_rays"]
        else 0.0
    )
    return "\n".join(
        [
            "# SA4-D8 current vs valid-return-only mixed-pixel shadow",
            "",
            "> Fixed c6400, stage4 lateral, seed818, d1=200 ms, 64 env x 2500 steps. The policy used current noise throughout; the correction existed only in the paired geometry shadow.",
            "",
            "## Realized policy outcome (current noise only)",
            "",
            f"- episodes: {metrics['n']:,}",
            f"- SR / CR / TO: {metrics['sr']:.2%} / {metrics['cr']:.2%} / {metrics['to']:.2%}",
            "",
            "## Noise eligibility ledger",
            "",
            f"- realized distractor rays: {noise['realized_distractor_rays']:,}",
            f"- removed because no valid surviving return: {noise['removed_no_return_or_dropped_distractor_rays']:,} ({removed_share:.2%})",
            f"- current winner bins removed: {noise['current_winner_bins_removed']:,}",
            f"- changed 72-bin values: {noise['changed_72_bins']:,}",
            "",
            "## Paired frozen-D4 feasibility",
            "",
            f"- active frames: {paired['active_frames']:,}",
            f"- current no-feasible: {paired['current_no_feasible_fraction']:.2%}",
            f"- corrected no-feasible: {paired['corrected_no_feasible_fraction']:.2%}",
            f"- current-no-feasible frames rescued: {paired['rescued_current_no_feasible_fraction']:.2%}",
            f"- net recovery: {paired['net_feasible_recovery_fraction_of_active']:+.2%} ({paired['net_feasible_recovery_frames']:+,} frames)",
            f"- major fake-obstacle contributor by preregistered descriptive rule: {paired['major_fake_obstacle_contributor']}",
            "",
            "This audit does not estimate corrected-policy SR/CR and does not validate the replacement distribution as a real VLP-16 model.",
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
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"D8 refuses to share a busy GPU: {busy}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol": output / "PROTOCOL.json",
        "log": output / "baseline_lateral_g4_d1_s818.log",
        "corridor": output / "baseline_lateral_g4_d1_s818_corridor.json",
        "d3": output / "baseline_lateral_g4_d1_s818_d3.json",
        "d8": output / "baseline_lateral_g4_d1_s818_d8.json",
        "summary": output / "SUMMARY.md",
        "manifest": output / "suite_manifest.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    _require_new_targets(list(paths.values()))
    planned_fingerprint = source_fingerprint()
    preregistration = {
        "schema": "sa4_d8_noise_eligibility_preregistration/v1",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "test whether distractors created on no-return or dropped raw rays "
            "are a major source of frozen-D4 no-feasible active frames"
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
        "d8_protocol": noise_eligibility_protocol(),
        "source_fingerprint": planned_fingerprint,
        "forbidden_during_run": [
            "counterfactual observation fed to policy",
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
        json.dumps(planned_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[SA4-D8] protocol frozen before rollout: {paths['protocol']}", flush=True)

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
            "--d8_noise_eligibility_sensitivity",
            "--d8_noise_eligibility_output",
            str(paths["d8"]),
        ],
    )

    final_fingerprint = source_fingerprint()
    paths["source_after"].write_text(
        json.dumps(final_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    if planned_fingerprint != final_fingerprint:
        raise RuntimeError("D8 source changed during rollout")
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
    d8_report = json.loads(paths["d8"].read_text(encoding="utf-8"))
    validate_d8_report(
        d8_report,
        checkpoint=checkpoint,
        corridor=corridor,
        d3_report=d3_report,
    )
    compact_metrics = d4_suite._metrics(corridor)
    paths["summary"].write_text(
        _render(d8_report, compact_metrics), encoding="utf-8"
    )
    manifest = {
        "schema": "sa4_d8_noise_eligibility_bundle/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "not_a_graduation_gate": True,
        "policy_actions_modified": False,
        "counterfactual_fed_to_policy": False,
        "training_started": False,
        "next_stage_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(paths["protocol"]),
        "protocol_sha256": noise_eligibility_protocol()["sha256"],
        "fixed_cell": preregistration["fixed_cell"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics_current_policy_only": metrics,
        "d3_self_check": d3_report["self_check"],
        "d8_runtime": d8_report["runtime"],
        "d8_noise_ledger": d8_report["noise_ledger"],
        "d8_paired_feasibility": d8_report["paired_feasibility"],
        "d8_self_check": d8_report["self_check"],
        "source_fingerprint": final_fingerprint,
        "files": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D8] COMPLETE valid diagnostic evidence: {paths['manifest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
