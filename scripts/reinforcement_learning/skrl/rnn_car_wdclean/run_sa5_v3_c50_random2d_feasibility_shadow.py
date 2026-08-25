"""Run the frozen c50 4S2D random-2D D5 feasibility-shadow audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

from d4_geometry_selector import speed_scaled_geometry_spec  # noqa: E402
from d5_feasibility_shadow import feasibility_shadow_protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    verify_fixed_actuator_runtime,
)
import run_sa3_gate_bc as base  # noqa: E402
import run_sa3_v3_retention_screen_cell as retention_helpers  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa5_r2_speed_scale_screen_cell as speed_helpers  # noqa: E402
import run_sa5_v3_c50_difficulty_frontier_cell as frontier_cell  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
import sa5_v3_c50_random2d_feasibility_shadow as protocol  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


FREEZE = (
    REPO
    / "docs/freeze/sa5_v3_c50_random2d_feasibility_shadow_v1.json"
)
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
ACTION_TERM = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        FREEZE.resolve(),
        protocol.AUTHORIZATION.resolve(),
        PLAY.resolve(),
        ACTION_TERM.resolve(),
        HERE / "d3_yield_recorder.py",
        HERE / "d4_geometry_selector.py",
        HERE / "d5_feasibility_shadow.py",
        Path(frontier_cell.__file__).resolve(),
        Path(d9_runner.__file__).resolve(),
        REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
        REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
        "config/charge_skrl/mdp/events/long_corridor_replay.py",
        REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
        "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
        REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
        "config/charge_skrl/mdp/events/corridor_density.py",
    )


def source_fingerprint() -> dict[str, str]:
    result = {}
    for path in source_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.relative_to(REPO))] = sha256_of(path)
    return result


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "random2d feasibility audit refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def build_scene_args(corridor_path: Path) -> list[str]:
    values = protocol.stage_scene()
    actuator = fixed_actuator_cli_args(
        protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    return [
        "--stage",
        str(protocol.STAGE),
        "--vlp16_noise_mode",
        protocol.LIDAR_NOISE_MODE,
        "--lidar-distractor-eligibility",
        protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "--arena_size",
        f"{values['arena_size_m']:g}",
        "--obs_near_goal_count",
        "0",
        "--seed",
        str(protocol.SEED),
        "--long_corridor_eval",
        "--long_corridor_free_width",
        f"{values['corridor_free_width_m']:g}",
        "--long_corridor_dynamic_speed_range",
        f"{protocol.DYNAMIC_SPEED_RANGE[0]:g}",
        f"{protocol.DYNAMIC_SPEED_RANGE[1]:g}",
        "--long_corridor_static_obstacles",
        str(protocol.STATIC_OBSTACLES),
        "--long_corridor_dynamic_obstacles",
        str(protocol.DYNAMIC_OBSTACLES),
        "--long_corridor_motion_mode",
        protocol.MOTION_MODE,
        "--long_corridor_random_2d_kinematics",
        protocol.RANDOM_2D_KINEMATICS,
        "--long_corridor_pause_mode",
        "default",
        "--long_corridor_output",
        str(corridor_path),
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
        *actuator,
    ]


def validate_d3_report(
    report: dict,
    *,
    checkpoint: Path,
    corridor_report: dict,
    steps: int,
    require_dynamic_collision: bool = True,
) -> None:
    problems = []
    metadata = report.get("metadata") or {}
    counts = report.get("counts") or {}
    checks = report.get("self_check") or {}
    if report.get("schema") != "sa4_d3_yield_timing/v2":
        problems.append("unexpected D3 schema")
    if report.get("mode") != "baseline":
        problems.append("D3 was not identity baseline")
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("D3 checkpoint mismatch")
    expected_metadata = {
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "num_dynamic_obstacles": protocol.DYNAMIC_OBSTACLES,
        "rollout_steps_requested": steps,
    }
    for key, expected in expected_metadata.items():
        if int(metadata.get(key, -1)) != int(expected):
            problems.append(f"D3 {key} mismatch")
    if metadata.get("corridor_motion_mode") != protocol.MOTION_MODE:
        problems.append("D3 motion mode mismatch")
    if int(counts.get("records", -1)) != protocol.NUM_ENVS * steps:
        problems.append("D3 record coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(
        corridor_report["episodes"]
    ):
        problems.append("D3 episode ledger mismatch")
    if (
        require_dynamic_collision
        and int(counts.get("dynamic_collision", 0)) <= 0
    ):
        problems.append("D3 captured no dynamic collision")
    for key in (
        "baseline_action_identity_ok",
        "record_coverage_ok",
        "episode_reconciliation_ok",
        "delay_alignment_ok",
        "dynamic_collision_reconciliation_ok",
        "reconciliation_ok",
    ):
        if checks.get(key) is not True:
            problems.append(f"D3 {key} is not true")
    if int(checks.get("delay_alignment_errors", -1)) != 0:
        problems.append("D3 delay alignment errors")
    if int(checks.get("delay_alignment_samples", 0)) <= 0:
        problems.append("D3 has no delay alignment samples")
    if problems:
        raise RuntimeError("invalid random2d D3 evidence: " + "; ".join(problems))


def validate_d5_report(
    report: dict, *, checkpoint: Path, corridor_report: dict, steps: int
) -> None:
    problems = []
    metadata = report.get("metadata") or {}
    counts = report.get("counts") or {}
    runtime = metadata.get("shadow_runtime") or {}
    expected_geometry = speed_scaled_geometry_spec(protocol.SPEED_RATE)
    expected_protocol = feasibility_shadow_protocol(
        geometry_spec=expected_geometry
    )
    if report.get("schema") != "sa4_d5_feasibility_frontier/v1":
        problems.append("unexpected D5 schema")
    if report.get("mode") != "feasibility_shadow":
        problems.append("D5 mode mismatch")
    if report.get("protocol") != expected_protocol:
        problems.append("D5 protocol drift")
    if Path(metadata.get("checkpoint", "")).resolve() != checkpoint:
        problems.append("D5 checkpoint mismatch")
    expected_metadata = {
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "num_dynamic_obstacles": protocol.DYNAMIC_OBSTACLES,
        "rollout_steps_requested": steps,
    }
    for key, expected in expected_metadata.items():
        if int(metadata.get(key, -1)) != int(expected):
            problems.append(f"D5 {key} mismatch")
    if metadata.get("corridor_motion_mode") != protocol.MOTION_MODE:
        problems.append("D5 motion mode mismatch")
    if metadata.get("baseline_policy_action_unchanged") is not True:
        problems.append("D5 action identity declaration missing")
    if metadata.get("shadow_protocol") != expected_protocol:
        problems.append("D5 metadata protocol drift")
    if int(counts.get("records", -1)) != protocol.NUM_ENVS * steps:
        problems.append("D5 record coverage mismatch")
    if int(counts.get("completed_episodes", -1)) != int(
        corridor_report["episodes"]
    ):
        problems.append("D5 episode ledger mismatch")
    if runtime.get("protocol_sha256") != expected_protocol["sha256"]:
        problems.append("D5 runtime protocol hash mismatch")
    if int(runtime.get("action_identity_errors", -1)) != 0:
        problems.append("D5 modified policy action")
    if int(runtime.get("environment_frames", -1)) != protocol.NUM_ENVS * steps:
        problems.append("D5 runtime coverage mismatch")
    if (report.get("self_check") or {}).get("reconciliation_ok") is not True:
        problems.append("D5 reconciliation failed")
    if problems:
        raise RuntimeError("invalid random2d D5 evidence: " + "; ".join(problems))


def render_summary(analysis: dict, metrics: dict) -> str:
    collision = analysis["collision_frontier"]
    control = analysis["successful_control_frontier"]
    return "\n".join(
        [
            "# SA5-v3 c50 4S2D random-2D feasibility shadow",
            "",
            f"Status: `{analysis['status']}`",
            f"Decision: `{analysis['decision']}`",
            "",
            f"- episodes: {metrics['n']:,}",
            f"- SR / CR / TO: {metrics['sr']:.2%} / {metrics['cr']:.2%} / {metrics['to']:.2%}",
            f"- dynamic-collision events: {analysis['collision_events']:,}",
            f"- successful closest-approach controls: {analysis['successful_control_events']:,}",
            f"- collisions with a feasible candidate somewhere in prior 5 s: {collision['feasible_seen_in_window_fraction']:.2%}",
            f"- collisions with no feasible candidate in the entire observed window: {collision['no_feasible_entire_window_fraction']:.2%}",
            f"- collision-time no-feasible fraction: {collision['event_no_feasible_fraction']:.2%}",
            f"- successful-control event no-feasible fraction: {control['event_no_feasible_fraction']:.2%}",
            f"- next step: `{analysis['next_step']}`",
            "",
            "> This is model-bounded diagnostic evidence. It does not prove physical inevitability or episode-level solvability.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=protocol.OUTPUT_ROOT
    )
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args(argv)

    frozen = protocol.protocol_payload()
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != protocol.CHECKPOINT["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")
    checked_in = json.loads(FREEZE.read_text(encoding="utf-8"))
    if checked_in != frozen:
        raise RuntimeError("checked-in feasibility protocol drift")
    if sha256_of(protocol.AUTHORIZATION) != protocol.AUTHORIZATION_SHA256:
        raise RuntimeError("authorization hash mismatch")

    checkpoint = protocol.checkpoint_path().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = sha256_of(checkpoint)
    if digest != protocol.CHECKPOINT["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        int(payload.get("iteration", -1))
        != protocol.CHECKPOINT["embedded_iteration"]
        or int(payload.get("total_steps", -1))
        != protocol.CHECKPOINT["embedded_total_steps"]
    ):
        raise RuntimeError("checkpoint embedded ledger mismatch")

    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 16:
        raise ValueError("smoke steps must be in [1, 16]")
    steps = args.smoke_steps or protocol.STEPS
    is_smoke = args.smoke_steps is not None
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{steps}" if is_smoke else ""
    stem = f"c50_s4_d2_random2d_g5_d1_s818{suffix}"
    paths = {
        "protocol": output / "PROTOCOL.json",
        "checkpoint": output / "CHECKPOINT_MANIFEST.json",
        "source_before": output / "SOURCE_FINGERPRINT_BEFORE.json",
        "source_after": output / "SOURCE_FINGERPRINT_AFTER.json",
        "log": output / f"{stem}.log",
        "corridor": output / f"{stem}_corridor.json",
        "d3": output / f"{stem}_d3.json",
        "d5": output / f"{stem}_d5.json",
        "analysis": output / ("SMOKE.json" if is_smoke else "ANALYSIS.json"),
        "summary": output / "SUMMARY.md",
        "manifest": output / "MANIFEST.json",
    }
    _require_new_targets(list(paths.values()))

    before = source_fingerprint()
    paths["protocol"].write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["checkpoint"].write_text(
        json.dumps(
            {
                "schema": "sa5_v3_c50_random2d_checkpoint_manifest/v1",
                "checkpoint": str(checkpoint),
                "sha256": digest,
                "embedded_iteration": int(payload["iteration"]),
                "embedded_total_steps": int(payload["total_steps"]),
                "status": protocol.CHECKPOINT["status"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["source_before"].write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    extra = [
        *build_scene_args(paths["corridor"]),
        "--d3_yield_audit",
        "--d3_yield_output",
        str(paths["d3"]),
        "--d3_shield_mode",
        "baseline",
        "--d5_feasibility_shadow",
        "--d5_feasibility_output",
        str(paths["d5"]),
    ]
    print(
        "[SA5-V3-R2D-D5] START c50 4S2D random_2d "
        f"steps={steps} smoke={is_smoke} protocol={frozen['sha256'][:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        paths["log"],
        num_envs=protocol.NUM_ENVS,
        steps=steps,
        extra=extra,
    )

    after = source_fingerprint()
    paths["source_after"].write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("source fingerprint changed during rollout")

    d9_runner._validate_runtime_log(
        paths["log"], protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        paths["log"], protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    values = protocol.stage_scene()
    stage_banner = base.verify_common_runtime(paths["log"], values)
    console = parse_play_summary(str(paths["log"]))
    if (not console or console.get("sr") is None) and not is_smoke:
        raise RuntimeError("play outcome summary is missing")
    corridor = json.loads(paths["corridor"].read_text(encoding="utf-8"))
    scene_contract = frontier_cell.verify_corridor_runtime(
        corridor, "s4_d2_random_2d", values
    )
    if is_smoke and (not console or console.get("sr") is None):
        metrics = {
            "n": int(corridor["episodes"]),
            "sr": float(corridor["success_rate"]),
            "cr": float(corridor["collision_rate"]),
            "to": float(corridor["timeout_rate"]),
        }
    else:
        metrics = base.reconcile_corridor_metrics(dict(console), corridor)
    speed_runtime = retention_helpers.verify_speed_rate_log(paths["log"])
    speed_helpers.verify_speed_rate_runtime(corridor, protocol.SPEED_RATE)

    d3_report = json.loads(paths["d3"].read_text(encoding="utf-8"))
    validate_d3_report(
        d3_report,
        checkpoint=checkpoint,
        corridor_report=corridor,
        steps=steps,
        require_dynamic_collision=not is_smoke,
    )
    d5_report = json.loads(paths["d5"].read_text(encoding="utf-8"))
    validate_d5_report(
        d5_report,
        checkpoint=checkpoint,
        corridor_report=corridor,
        steps=steps,
    )

    if is_smoke:
        analysis = {
            "schema": "sa5_v3_c50_random2d_feasibility_smoke/v1",
            "status": "GPU_SMOKE_VALID_NON_EVIDENCE",
            "formal_evidence": False,
            "training_started": False,
            "sa6_started": False,
        }
        summary = "# GPU smoke\n\nValid non-evidence smoke; no training started.\n"
    else:
        analysis = protocol.analyze_report(d5_report)
        summary = render_summary(analysis, metrics)
    paths["analysis"].write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["summary"].write_text(summary, encoding="utf-8")
    manifest = {
        "schema": "sa5_v3_c50_random2d_feasibility_bundle/v1",
        "status": analysis["status"],
        "formal_evidence": not is_smoke,
        "protocol_sha256": frozen["sha256"],
        "checkpoint_sha256": digest,
        "fixed_cell": frozen["fixed_cell"],
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "speed_rate_runtime": speed_runtime,
        "metrics": metrics,
        "decision": analysis.get("decision"),
        "source_fingerprint_stable": True,
        "training_started": False,
        "sa6_started": False,
        "files": {name: str(path) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA5-V3-R2D-D5] {analysis['status']} "
        f"decision={analysis.get('decision')} SR={metrics['sr']:.4f} "
        f"CR={metrics['cr']:.4f} TO={metrics['to']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
