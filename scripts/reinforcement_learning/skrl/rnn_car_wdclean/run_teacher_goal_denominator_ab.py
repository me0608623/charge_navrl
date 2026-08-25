"""Run the frozen teacher-only R=1 m versus R=10 m closed-loop A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import teacher_goal_denominator_ab as ab  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402


PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
TEACHER = HERE / "privileged_corridor_teacher.py"
TEACHER_METRICS = HERE / "teacher_closed_loop_metrics.py"
CORRIDOR_METRICS = HERE / "corridor_eval_metrics.py"
PHASE_AUDIT = HERE / "corridor_motion_phase_audit.py"
LONG_CORRIDOR = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay.py"
)
BEHAVIOR_SCHEDULER = LONG_CORRIDOR.parent / "behavior_scheduler.py"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "protocol": Path(ab.__file__).resolve(),
        "play": PLAY,
        "teacher": TEACHER,
        "teacher_metrics": TEACHER_METRICS,
        "corridor_metrics": CORRIDOR_METRICS,
        "phase_audit": PHASE_AUDIT,
        "long_corridor": LONG_CORRIDOR,
        "behavior_scheduler": BEHAVIOR_SCHEDULER,
    }
    return {name: sha256_of(path) for name, path in files.items()}


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


def _require_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "teacher A/B refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def _assert_finite_json(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value!r}")


def _validate_runtime_log(path: Path, floor_m: float) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    required = (
        "[CORRIDOR-TEACHER] enabled (mode=override)",
        f"goal_denominator_floor={floor_m:g}m",
        "pause_mode=default 實際 pause_steps_range=(0, 5)",
        "obstacles=2S+1D motion=lateral",
    )
    missing = [marker for marker in required if marker not in text]
    problems = []
    if missing:
        problems.append("missing markers: " + repr(missing))
    for label, pattern in (
        ("traceback", r"Traceback \(most recent call last\)"),
        ("CUDA OOM", r"CUDA out of memory"),
        ("CUDA error", r"CUDA error:"),
        ("NaN", r"(?<![A-Za-z])nan(?![A-Za-z])"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            problems.append(label)
    if problems:
        raise RuntimeError(f"invalid runtime log {path}: " + "; ".join(problems))


def _validate_cell_outputs(
    *,
    arm: str,
    floor_m: float,
    teacher_path: Path,
    corridor_path: Path,
    phase_path: Path,
) -> tuple[dict, dict[str, object]]:
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    corridor = json.loads(corridor_path.read_text(encoding="utf-8"))
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    _assert_finite_json(teacher)
    _assert_finite_json(corridor)
    _assert_finite_json(phase)

    if teacher["teacher_mode"] != "override":
        raise ValueError(f"arm {arm} was not teacher override")
    observed_floor = float(
        teacher["teacher_spec"]["goal_denominator_floor_m"]
    )
    if observed_floor != floor_m:
        raise ValueError(
            f"arm {arm} floor mismatch: {observed_floor} != {floor_m}"
        )
    nested = teacher["closed_loop_outcome"]
    for key in (
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "wall_collision_rate",
        "obstacle_collision_rate",
    ):
        if nested[key] != corridor[key]:
            raise ValueError(f"teacher/corridor mismatch for {key}")
    if corridor["dynamic_pause_steps_range"] != [0, 5]:
        raise ValueError("runtime pause range drifted")
    nested_phase = corridor["motion_phase_audit"]
    for key in (
        "total_dynamic_slot_frames",
        "total_dynamic_slot_collisions",
    ):
        if nested_phase[key] != phase[key]:
            raise ValueError(f"standalone/nested phase mismatch for {key}")
    return teacher, ab.extract_cell_metrics(teacher)


def _render_summary(comparison: dict[str, object]) -> str:
    a = comparison["pooled"]["historical_r1"]
    b = comparison["pooled"]["candidate_r10"]
    delta = comparison["delta_candidate_minus_historical"]
    lines = [
        "# Teacher goal denominator closed-loop A/B",
        "",
        "> Teacher override; 4 m x 10 m, 2S+1D lateral patrol, default dynamic pause; 3 seeds.",
        "",
        "| metric | R=1 historical | R=10 candidate | R10 - R1 |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percent in (
        ("SR", "sr", True),
        ("CR", "cr", True),
        ("TO", "to", True),
        ("wall CR", "wall_cr", True),
        ("obstacle CR", "obstacle_cr", True),
        ("near-goal completion", "near_goal_completion_rate", True),
        ("near-goal collision", "near_goal_collision_rate", True),
        ("near-hard clearance fraction", "near_goal_near_hard_fraction", True),
        ("mean |speed| m/s", "linear_speed_abs_mean_mps", False),
        ("stop fraction", "stop_command_fraction", True),
        ("near-goal mean clearance m", "near_goal_clearance_mean_m", False),
    ):
        if percent:
            lines.append(
                f"| {label} | {a[key]:.3%} | {b[key]:.3%} | "
                f"{delta[key] * 100:+.2f} pp |"
            )
        else:
            lines.append(
                f"| {label} | {a[key]:.4f} | {b[key]:.4f} | "
                f"{delta[key]:+.4f} |"
            )
    lines.extend(["", "## Frozen checks", ""])
    for name, passed in comparison["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- candidate teacher pass: **{comparison['candidate_teacher_pass']}**",
            "- SA5 training started: **False**",
            "- SA6 modified or started: **False**",
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
    checkpoint_sha256 = sha256_of(checkpoint)
    if checkpoint_sha256 != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: {checkpoint_sha256} != "
            f"{args.expect_checkpoint_sha256}"
        )
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"teacher A/B refuses to share a busy GPU: {busy}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": output / "PROTOCOL.json",
        "manifest": output / "suite_manifest.json",
        "summary": output / "SUMMARY.md",
        "incomplete": output / "INCOMPLETE_NO_VERDICT.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    cells = {
        (seed, arm): {
            "log": output / f"{arm}_s{seed}.log",
            "teacher": output / f"{arm}_s{seed}_teacher.json",
            "corridor": output / f"{arm}_s{seed}_corridor.json",
            "phase": output / f"{arm}_s{seed}_phase.json",
            "cell": output / f"{arm}_s{seed}_cell.json",
        }
        for seed in ab.SEEDS
        for arm in ab.ARMS
    }
    _require_new(
        list(common.values())
        + [path for bundle in cells.values() for path in bundle.values()]
    )

    protocol = ab.frozen_protocol()
    fingerprint = source_fingerprint()
    common["protocol"].write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_BEFORE_ROLLOUT",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "protocol": protocol,
                "source_fingerprint": fingerprint,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common["source_before"].write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[TEACHER-DENOM-AB] protocol frozen: {protocol['sha256']}",
        flush=True,
    )

    completed: dict[str, dict[int, dict[str, object]]] = {
        arm: {} for arm in ab.ARMS
    }
    try:
        for seed, arm in protocol["run_order"]:
            floor_m = ab.FLOOR_BY_ARM[arm]
            paths = cells[(seed, arm)]
            print(
                f"[TEACHER-DENOM-AB] seed={seed} arm={arm} R={floor_m:g} starting",
                flush=True,
            )
            _run_play(
                checkpoint,
                paths["log"],
                num_envs=ab.NUM_ENVS,
                steps=ab.ROLLOUT_STEPS,
                extra=[
                    "--stage", "1",
                    "--seed", str(seed),
                    "--no_lidar_vis",
                    "--long_corridor_eval",
                    "--long_corridor_static_obstacles", "2",
                    "--long_corridor_dynamic_obstacles", "1",
                    "--long_corridor_free_width", "4.0",
                    "--long_corridor_dynamic_speed_range", "0.30", "0.60",
                    "--long_corridor_motion_mode", "lateral",
                    "--long_corridor_pause_mode", "default",
                    "--long_corridor_phase_audit",
                    "--long_corridor_output", str(paths["corridor"]),
                    "--long_corridor_phase_audit_output", str(paths["phase"]),
                    "--privileged_corridor_teacher",
                    "--privileged_corridor_teacher_output", str(paths["teacher"]),
                    "--corridor_teacher_goal_denominator_floor_m", str(floor_m),
                ],
            )
            if source_fingerprint() != fingerprint:
                raise RuntimeError(f"source drift during seed={seed} arm={arm}")
            _validate_runtime_log(paths["log"], floor_m)
            teacher, metrics = _validate_cell_outputs(
                arm=arm,
                floor_m=floor_m,
                teacher_path=paths["teacher"],
                corridor_path=paths["corridor"],
                phase_path=paths["phase"],
            )
            cell = {
                "schema": "teacher_goal_denominator_ab_cell/v1",
                "seed": seed,
                "arm": arm,
                "goal_denominator_floor_m": floor_m,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "protocol_sha256": protocol["sha256"],
                "metrics": metrics,
                "teacher_spec": teacher["teacher_spec"],
                "source_fingerprint": fingerprint,
                "files": {name: str(path) for name, path in paths.items()},
            }
            paths["cell"].write_text(
                json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
            )
            completed[arm][seed] = metrics
            print(
                f"[TEACHER-DENOM-AB] seed={seed} arm={arm} "
                f"n={metrics['episodes']} SR={metrics['sr']:.4f} "
                f"CR={metrics['cr']:.4f} TO={metrics['to']:.4f} "
                f"near_p05={metrics['near_goal_clearance_p05_m']:.3f}m",
                flush=True,
            )

        ordered = {
            arm: [completed[arm][seed] for seed in ab.SEEDS]
            for arm in ab.ARMS
        }
        comparison = ab.compare(ordered)
        final_fingerprint = source_fingerprint()
        if final_fingerprint != fingerprint:
            raise RuntimeError("source drift after A/B")
        common["source_after"].write_text(
            json.dumps(final_fingerprint, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        common["summary"].write_text(
            _render_summary(comparison), encoding="utf-8"
        )
        manifest = {
            "schema": "teacher_goal_denominator_ab_bundle/v1",
            "status": "COMPLETE_VALID_TEACHER_AB",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "protocol": protocol,
            "comparison": comparison,
            "source_fingerprint": final_fingerprint,
            "sa5_branch_creation_authorized": bool(
                comparison["candidate_teacher_pass"]
            ),
            "sa5_training_started": False,
            "sa6_modified_or_started": False,
            "files": {
                **{
                    name: str(path)
                    for name, path in common.items()
                    if name != "incomplete"
                },
                "cells": {
                    f"{arm}_s{seed}": {
                        name: str(path) for name, path in bundle.items()
                    }
                    for (seed, arm), bundle in cells.items()
                },
            },
        }
        common["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "[TEACHER-DENOM-AB] COMPLETE "
            f"candidate_pass={comparison['candidate_teacher_pass']} "
            f"manifest={common['manifest']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        common["incomplete"].write_text(
            json.dumps(
                {
                    "schema": "teacher_goal_denominator_ab_incomplete/v1",
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "protocol_sha256": protocol["sha256"],
                    "completed": completed,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
