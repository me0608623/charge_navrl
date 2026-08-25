"""Run the frozen delay-aware 2S2D privileged-teacher diagnostic."""

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

import teacher_2s2d_d1_screen as protocol  # noqa: E402


PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
TEACHER = HERE / "privileged_corridor_teacher.py"
STATEFUL_TEACHER = HERE / "stateful_corridor_teacher.py"
INTERACTION = HERE / "teacher_interaction_metrics.py"
CLOSED_LOOP = HERE / "teacher_closed_loop_metrics.py"
CORRIDOR_METRICS = HERE / "corridor_eval_metrics.py"
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
        "runner": Path(
            getattr(protocol, "RUNNER_PATH", Path(__file__).resolve())
        ).resolve(),
        "protocol": Path(protocol.__file__).resolve(),
        "play": PLAY,
        "teacher": TEACHER,
        "interaction_metrics": INTERACTION,
        "closed_loop_metrics": CLOSED_LOOP,
        "corridor_metrics": CORRIDOR_METRICS,
        "action": ACTION,
        "actuator": ACTUATOR,
        "long_corridor": LONG_CORRIDOR,
        "long_corridor_geometry": LONG_CORRIDOR_GEOMETRY,
        "behavior_scheduler": BEHAVIOR_SCHEDULER,
    }
    if getattr(protocol, "CONTROLLER", "memoryless") == "stateful":
        files["shared_runner"] = Path(__file__).resolve()
        files["stateful_teacher"] = STATEFUL_TEACHER
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
    gpu = subprocess.run(
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
    used, total, utilization = [int(value.strip()) for value in gpu.split(",")]
    apps_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    apps = [line.strip() for line in apps_result.stdout.splitlines() if line.strip()]
    active_isaac = [
        line
        for line in apps
        if re.search(r"train_rnn_car|play_rnn_car|isaac", line, re.IGNORECASE)
    ]
    if active_isaac:
        raise RuntimeError("another Isaac process is active: " + "; ".join(active_isaac))
    if total - used < 12_000:
        raise RuntimeError(
            f"less than 12 GB GPU memory free before screen: {total - used} MiB"
        )
    return {
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
        "compute_apps": apps,
        "shared_gpu": len(apps) > 1,
        "timing_metrics_valid": False,
    }


def _require_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "teacher 2S2D screen refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def _run_cell(
    checkpoint: Path,
    *,
    motion_mode: str,
    log_path: Path,
    teacher_path: Path,
    corridor_path: Path,
    diag_path: Path | None = None,
) -> None:
    command = [
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
        motion_mode,
        "--long_corridor_pause_mode",
        "default",
        "--long_corridor_output",
        str(corridor_path),
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
        "--privileged_corridor_teacher",
        "--privileged_corridor_teacher_output",
        str(teacher_path),
        "--corridor_teacher_goal_denominator_floor_m",
        "10.0",
        "--corridor_teacher_controller",
        getattr(protocol, "CONTROLLER", "memoryless"),
    ]
    if diag_path is not None:
        command += [
            "--stateful_teacher_diagnostic_output",
            str(diag_path),
        ]
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
        "[CORRIDOR-TEACHER] enabled (mode=override)",
        "actuator=fixed_d1_queue",
        "goal_denominator_floor=10m",
        f"controller={getattr(protocol, 'CONTROLLER', 'memoryless')}",
        "Actuator DR: delay=(1, 1) steps, vel_scale=(1.0, 1.0), motor_lag alpha=1.0",
        "mixed-pixel-eligibility] mode=valid_return_only",
        f"obstacles=2S+2D speed=[0.25,0.45]m/s motion={motion_mode}",
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
        getattr(
            protocol,
            "REPORT_TITLE",
            "# Delay-aware 2S2D privileged-teacher screen",
        ),
        "",
        "> Single checkpoint/seed diagnostic. GPU was shared; fps/timing are not evidence.",
        "",
        "| scenario | n | SR | CR | TO | launch p95 | vacated-side | U-turn | 360 deg | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for scenario in protocol.SCENARIOS:
        name = scenario["name"]
        cell = decision["cells"][name]
        launch = cell["launch_delay_p95_s"]
        launch_text = "n/a" if launch is None else f"{float(launch):.2f}s"
        lines.append(
            f"| {name} | {cell['episodes']} | {cell['sr']:.2%} | "
            f"{cell['cr']:.2%} | {cell['to']:.2%} | {launch_text} | "
            f"{cell['vacated_side_match_fraction']:.1%} "
            f"({cell['vacated_side_eligible']}) | {cell['u_turn_fraction']:.2%} | "
            f"{cell['full_rotation_fraction']:.2%} | "
            f"{'PASS' if cell['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- teacher pass: **{decision['teacher_pass']}**",
            f"- next step: **{decision['next_step']}**",
            "- training/distillation/SA6 authorized: **False / False / False**",
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
            f"checkpoint hash mismatch: {checkpoint_sha} != "
            f"{args.expect_checkpoint_sha256}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": output / "PROTOCOL.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
        "manifest": output / "suite_manifest.json",
        "summary": output / "SUMMARY.md",
        "incomplete": output / "INCOMPLETE_NO_VERDICT.json",
    }
    _stateful = getattr(protocol, "CONTROLLER", "memoryless") == "stateful"
    cells = {
        scenario["name"]: {
            "log": output / f"{scenario['name']}.log",
            "teacher": output / f"{scenario['name']}_teacher.json",
            "corridor": output / f"{scenario['name']}_corridor.json",
            "cell": output / f"{scenario['name']}_cell.json",
            # Record-only per-step diagnostic; the stateful controller is the
            # only one that produces FSM commitment state to record.
            **(
                {"diag": output / f"{scenario['name']}_stateful_diag.npz"}
                if _stateful
                else {}
            ),
        }
        for scenario in protocol.SCENARIOS
    }
    _require_new(
        list(common.values())
        + [path for bundle in cells.values() for path in bundle.values()]
    )

    frozen = protocol.frozen_protocol()
    fingerprint = source_fingerprint()
    gpu_before = gpu_snapshot()
    common["protocol"].write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_BEFORE_ROLLOUT",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "protocol": frozen,
                "source_fingerprint": fingerprint,
                "gpu_before": gpu_before,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common["source_before"].write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[TEACHER-2S2D] protocol frozen: {frozen['sha256']}", flush=True)

    completed: dict[str, dict[str, object]] = {}
    try:
        for scenario in protocol.SCENARIOS:
            name = scenario["name"]
            paths = cells[name]
            print(f"[TEACHER-2S2D] {name} starting", flush=True)
            _run_cell(
                checkpoint,
                motion_mode=scenario["motion_mode"],
                log_path=paths["log"],
                teacher_path=paths["teacher"],
                corridor_path=paths["corridor"],
                diag_path=paths.get("diag"),
            )
            if source_fingerprint() != fingerprint:
                raise RuntimeError(f"source drift during {name}")
            if sha256_of(checkpoint) != checkpoint_sha:
                raise RuntimeError(f"checkpoint drift during {name}")
            _validate_log(paths["log"], scenario["motion_mode"])
            teacher = json.loads(paths["teacher"].read_text(encoding="utf-8"))
            corridor = json.loads(paths["corridor"].read_text(encoding="utf-8"))
            _assert_finite_json(teacher)
            _assert_finite_json(corridor)
            nested = teacher["closed_loop_outcome"]
            for key in ("episodes", "success_rate", "collision_rate", "timeout_rate"):
                if nested[key] != corridor[key]:
                    raise ValueError(f"teacher/corridor mismatch for {name}.{key}")
            metrics = protocol.evaluate_cell(teacher)
            cell = {
                "schema": (
                    f"{getattr(protocol, 'SCHEMA_PREFIX', 'teacher_2s2d_d1_screen')}"
                    "_cell/v1"
                ),
                "scenario": name,
                "motion_mode": scenario["motion_mode"],
                "checkpoint_sha256": checkpoint_sha,
                "protocol_sha256": frozen["sha256"],
                "metrics": metrics,
            }
            paths["cell"].write_text(
                json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
            )
            completed[name] = metrics
            print(
                f"[TEACHER-2S2D] {name} n={metrics['episodes']} "
                f"SR={metrics['sr']:.2%} CR={metrics['cr']:.2%} "
                f"U-turn={metrics['u_turn_fraction']:.2%}",
                flush=True,
            )

        after = source_fingerprint()
        common["source_after"].write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if after != fingerprint:
            raise RuntimeError("source fingerprint drifted before decision")
        decision = protocol.decide(completed)
        manifest = {
            "schema": (
                f"{getattr(protocol, 'SCHEMA_PREFIX', 'teacher_2s2d_d1_screen')}"
                "_manifest/v1"
            ),
            "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "protocol_sha256": frozen["sha256"],
            "source_fingerprint_stable": True,
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
            "decision": decision,
        }
        common["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        common["summary"].write_text(
            _render_summary(decision), encoding="utf-8"
        )
        print(
            f"[TEACHER-2S2D] complete teacher_pass={decision['teacher_pass']} "
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
                    "completed_cells": sorted(completed),
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
