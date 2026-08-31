"""Run the frozen policy-only SA5 c600 2S2D walking-state screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_c600_2s2d_motion_state_screen as protocol  # noqa: E402


PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
TEACHER = HERE / "privileged_corridor_teacher.py"
INTERACTION = HERE / "teacher_interaction_metrics.py"
CLOSED_LOOP = HERE / "teacher_closed_loop_metrics.py"
CORRIDOR_METRICS = HERE / "corridor_eval_metrics.py"
PHASE_AUDIT = HERE / "corridor_motion_phase_audit.py"
ACTION = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
EVENTS = ACTION.parent.parent / "events"
ACTUATOR = ACTION.parent.parent.parent / "domain_randomization/actuator_dr.py"
LONG_CORRIDOR = EVENTS / "long_corridor_replay.py"
LONG_CORRIDOR_GEOMETRY = EVENTS / "long_corridor_replay_geometry.py"
BEHAVIOR_SCHEDULER = EVENTS / "behavior_scheduler.py"
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "protocol": Path(protocol.__file__).resolve(),
        "play": PLAY,
        "teacher_shadow_geometry": TEACHER,
        "interaction_metrics": INTERACTION,
        "closed_loop_metrics": CLOSED_LOOP,
        "corridor_metrics": CORRIDOR_METRICS,
        "phase_audit": PHASE_AUDIT,
        "action": ACTION,
        "actuator": ACTUATOR,
        "long_corridor": LONG_CORRIDOR,
        "long_corridor_geometry": LONG_CORRIDOR_GEOMETRY,
        "behavior_scheduler": BEHAVIOR_SCHEDULER,
    }
    return {name: sha256_of(path) for name, path in files.items()}


def _assert_finite_json(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value!r}")


def gpu_snapshot() -> dict[str, object]:
    row = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    used, total, utilization = [int(value.strip()) for value in row.split(",")]
    process_scan = subprocess.run(
        ["pgrep", "-af", "train_rnn_car|play_rnn_car|isaaclab.sh"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    ).stdout.splitlines()
    active = [
        line.strip()
        for line in process_scan
        if line.strip()
        and "pgrep -af" not in line
        and str(os.getpid()) not in line.split(maxsplit=1)[0]
    ]
    if active:
        raise RuntimeError("another training/evaluation process is active: " + "; ".join(active))
    if total - used < 12_000:
        raise RuntimeError(f"less than 12 GB GPU memory free: {total - used} MiB")
    return {
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
        "active_training_or_evaluation": active,
    }


def build_command(
    checkpoint: Path,
    scenario: dict[str, str],
    *,
    corridor_path: Path,
    shadow_path: Path,
    phase_path: Path,
) -> list[str]:
    return [
        str(REPO / "isaaclab.sh"),
        "-p",
        str(PLAY),
        "--checkpoint",
        str(checkpoint),
        "--task",
        "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
        "--curriculum_version",
        "warp_drive_e2e_final20_v1",
        "--stage",
        "5",
        "--seed",
        str(protocol.SEED),
        "--deterministic",
        "--num_envs",
        str(protocol.NUM_ENVS),
        "--steps",
        str(protocol.ROLLOUT_STEPS),
        "--headless",
        "--no_lidar_vis",
        "--arena_size",
        "15.0",
        "--num_goals_override",
        "1",
        "--no_goal_movement",
        "--long_corridor_eval",
        "--long_corridor_static_obstacles",
        "2",
        "--long_corridor_dynamic_obstacles",
        "2",
        "--long_corridor_free_width",
        "4.2",
        "--long_corridor_dynamic_speed_range",
        "0.25",
        "0.45",
        "--long_corridor_motion_mode",
        scenario["motion_mode"],
        "--long_corridor_pause_mode",
        "default",
        "--long_corridor_random_2d_kinematics",
        "patrol",
        "--long_corridor_phase_audit",
        "--long_corridor_output",
        str(corridor_path),
        "--long_corridor_phase_audit_output",
        str(phase_path),
        "--vlp16_noise_mode",
        "full",
        "--lidar-distractor-eligibility",
        "valid_return_only",
        "--speed_rate",
        "0.7",
        "--speed_rate_obs",
        "ego",
        "--enable_actuator_dr",
        "--actuator_delay_range",
        "1",
        "1",
        "--actuator_velocity_scale",
        "1.0",
        "1.0",
        "--actuator_motor_lag",
        "1.0",
        "--privileged_corridor_teacher_shadow",
        "--privileged_corridor_teacher_output",
        str(shadow_path),
        "--corridor_teacher_goal_denominator_floor_m",
        "10.0",
        "--corridor_teacher_controller",
        "memoryless",
    ]


def _run_cell(
    checkpoint: Path,
    scenario: dict[str, str],
    *,
    log_path: Path,
    corridor_path: Path,
    shadow_path: Path,
    phase_path: Path,
) -> None:
    command = build_command(
        checkpoint,
        scenario,
        corridor_path=corridor_path,
        shadow_path=shadow_path,
        phase_path=phase_path,
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CONDA_PREFIX"] = str(PYTHON.parent.parent)
    env["CONDA_DEFAULT_ENV"] = "env_isaaclab"
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"play exited {result.returncode}: {log_path}")


def _validate_log(path: Path, motion_mode: str) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    required = (
        "[CORRIDOR-TEACHER] enabled (mode=shadow)",
        "actuator=fixed_d1_queue",
        "Actuator DR: delay=(1, 1) steps, vel_scale=(1.0, 1.0), motor_lag alpha=1.0",
        "mixed-pixel-eligibility] mode=valid_return_only",
        f"obstacles=2S+2D speed=[0.25,0.45]m/s motion={motion_mode}",
        "pause_mode=default",
    )
    missing = [marker for marker in required if marker not in text]
    errors = []
    if missing:
        errors.append("missing runtime markers: " + repr(missing))
    for label, pattern in (
        ("traceback", r"Traceback \(most recent call last\)"),
        ("CUDA OOM", r"CUDA out of memory"),
        ("CUDA error", r"CUDA error:"),
        ("NaN", r"(?<![A-Za-z])nan(?![A-Za-z])"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            errors.append(label)
    if errors:
        raise RuntimeError(f"invalid runtime log {path}: " + "; ".join(errors))


def _render_summary(decision: dict[str, object]) -> str:
    lines = [
        "# SA5 c600 policy-only 2S2D walking-state screen",
        "",
        "> c600 was accepted by explicit human waiver. This screen does not rewrite the prior machine Gate failure.",
        "> The privileged teacher ran in shadow mode only; policy actions were never replaced.",
        "",
        "| state | n | SR | CR | TO | wait | launch p95 | vacated side | U-turn | 360 deg | outcome |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for scenario in protocol.SCENARIOS:
        cell = decision["cells"][scenario["name"]]
        launch = cell["launch_delay_p95_s"]
        launch_text = "n/a" if launch is None else f"{float(launch):.2f}s"
        lines.append(
            f"| {scenario['name']} | {cell['episodes']} | {cell['sr']:.2%} | "
            f"{cell['cr']:.2%} | {cell['to']:.2%} | "
            f"{cell['wait_frame_fraction_during_interaction']:.1%} | "
            f"{launch_text} | {cell['vacated_side_match_fraction']:.1%} "
            f"({cell['vacated_side_eligible']}) | {cell['u_turn_fraction']:.2%} | "
            f"{cell['full_rotation_fraction']:.2%} | "
            f"{'PASS' if cell['outcome_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Motion phases", ""])
    for scenario in protocol.SCENARIOS:
        cell = decision["cells"][scenario["name"]]
        lines.extend(
            [
                f"### {scenario['name']}",
                "",
                "| phase | frame share | collisions | collision share | enrichment |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name in (
            "steady",
            "pre_waypoint_1s",
            "paused",
            "post_switch_or_resume_1s",
        ):
            row = cell["motion_phases"][name]
            lines.append(
                f"| {name} | {row['frame_exposure_fraction']:.2%} | "
                f"{row['collision_count']} | {row['collision_fraction']:.2%} | "
                f"{row['collision_enrichment']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- all motion outcome pass: **{decision['all_motion_outcome_pass']}**",
            f"- lateral behavior pass: **{decision['lateral_behavior_pass']}**",
            f"- next step: **{decision['next_step']}**",
            "- training / distillation / SA6 authorized: **False / False / False**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_of(checkpoint)
    if checkpoint_sha != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: {checkpoint_sha} != {args.expect_checkpoint_sha256}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": output / "PROTOCOL.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
        "manifest": output / "suite_manifest.json",
        "summary_json": output / "SUMMARY.json",
        "summary_md": output / "SUMMARY.md",
        "incomplete": output / "INCOMPLETE_NO_VERDICT.json",
    }
    cells = {
        row["name"]: {
            "log": output / f"{row['name']}.log",
            "corridor": output / f"{row['name']}_corridor.json",
            "shadow": output / f"{row['name']}_policy_shadow.json",
            "phase": output / f"{row['name']}_motion_phase.json",
            "cell": output / f"{row['name']}_cell.json",
        }
        for row in protocol.SCENARIOS
    }
    existing = [
        str(path)
        for path in [*common.values(), *(p for bundle in cells.values() for p in bundle.values())]
        if path.exists()
    ]
    if existing:
        raise FileExistsError("screen refuses to overwrite evidence: " + ", ".join(existing))

    frozen = protocol.frozen_protocol()
    fingerprint = source_fingerprint()
    before = gpu_snapshot()
    common["protocol"].write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_BEFORE_ROLLOUT",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "protocol": frozen,
                "source_fingerprint": fingerprint,
                "gpu_before": before,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common["source_before"].write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[C600-2S2D] protocol frozen: {frozen['sha256']}", flush=True)

    completed: dict[str, dict[str, object]] = {}
    try:
        for scenario in protocol.SCENARIOS:
            name = scenario["name"]
            paths = cells[name]
            if gpu_snapshot()["active_training_or_evaluation"]:
                raise RuntimeError("GPU became busy before cell start")
            print(f"[C600-2S2D] {name} starting", flush=True)
            _run_cell(
                checkpoint,
                scenario,
                log_path=paths["log"],
                corridor_path=paths["corridor"],
                shadow_path=paths["shadow"],
                phase_path=paths["phase"],
            )
            if source_fingerprint() != fingerprint:
                raise RuntimeError(f"source drift during {name}")
            if sha256_of(checkpoint) != checkpoint_sha:
                raise RuntimeError(f"checkpoint drift during {name}")
            _validate_log(paths["log"], scenario["motion_mode"])
            corridor = json.loads(paths["corridor"].read_text(encoding="utf-8"))
            shadow = json.loads(paths["shadow"].read_text(encoding="utf-8"))
            phase = json.loads(paths["phase"].read_text(encoding="utf-8"))
            _assert_finite_json(corridor)
            _assert_finite_json(shadow)
            _assert_finite_json(phase)
            if corridor.get("motion_phase_audit") != {
                "scope": "patrol_slots_only",
                "applicable": phase["applicable"],
                "applicable_patrol_slot_frames": phase["applicable_patrol_slot_frames"],
                "excluded_wander_slot_frames": phase["excluded_wander_slot_frames"],
                **{
                    key: phase[key]
                    for key in (
                        "step_dt_s",
                        "phase_window_s",
                        "total_dynamic_slot_frames",
                        "total_dynamic_slot_collisions",
                        "context_keys",
                        "phases",
                    )
                },
            }:
                raise ValueError(f"standalone/main phase audit mismatch for {name}")
            metrics = protocol.evaluate_cell(scenario, corridor, shadow)
            cell = {
                "schema": "sa5_c600_2s2d_motion_state_screen_cell/v1",
                "checkpoint_sha256": checkpoint_sha,
                "protocol_sha256": frozen["sha256"],
                "metrics": metrics,
            }
            paths["cell"].write_text(
                json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
            )
            completed[name] = metrics
            print(
                f"[C600-2S2D] {name} n={metrics['episodes']} "
                f"SR={metrics['sr']:.2%} CR={metrics['cr']:.2%} "
                f"TO={metrics['to']:.2%} U-turn={metrics['u_turn_fraction']:.2%}",
                flush=True,
            )

        after = source_fingerprint()
        common["source_after"].write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if after != fingerprint:
            raise RuntimeError("source fingerprint drifted before decision")
        decision = protocol.decide(completed)
        after_gpu = gpu_snapshot()
        manifest = {
            "schema": "sa5_c600_2s2d_motion_state_screen_manifest/v1",
            "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "protocol_sha256": frozen["sha256"],
            "source_fingerprint_stable": True,
            "teacher_shadow_only": True,
            "policy_actions_replaced": False,
            "gpu_before": before,
            "gpu_after": after_gpu,
            "decision": decision,
        }
        common["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        common["summary_json"].write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        common["summary_md"].write_text(_render_summary(decision), encoding="utf-8")
        print(
            f"[C600-2S2D] complete all_motion_pass={decision['all_motion_outcome_pass']} "
            f"next={decision['next_step']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        common["incomplete"].write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error": repr(exc),
                    "completed_cells": list(completed),
                    "training_authorized": False,
                    "distillation_authorized": False,
                    "sa6_authorized": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
