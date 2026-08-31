"""Run the two frozen c600 policy-only step-trace cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from rnn_car_wdclean.analyze_sa5_c600_step_trace import (
    analyze_lateral_trace,
    analyze_random2d_trace,
    load_trace,
)
from rnn_car_wdclean import sa5_c600_step_trace_protocol as protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
ACTION = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
SCHEDULER = ACTION.parent.parent / "events/behavior_scheduler.py"
D3 = HERE / "d3_yield_recorder.py"
D5 = HERE / "d5_feasibility_shadow.py"
TRACE = HERE / "policy_step_trace.py"
ANALYZER = HERE / "analyze_sa5_c600_step_trace.py"
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
SCENARIOS = ("lateral", "random_2d")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    return {
        name: sha256(path)
        for name, path in {
            "runner": Path(__file__).resolve(),
            "protocol": Path(protocol.__file__).resolve(),
            "play": PLAY,
            "action": ACTION,
            "scheduler": SCHEDULER,
            "d3": D3,
            "d5": D5,
            "trace": TRACE,
            "analyzer": ANALYZER,
        }.items()
    }


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
    used, total, utilization = [int(part.strip()) for part in row.split(",")]
    scan = subprocess.run(
        ["pgrep", "-af", "train_rnn_car|play_rnn_car|isaaclab.sh"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    ).stdout.splitlines()
    active = [
        line.strip()
        for line in scan
        if line.strip()
        and "pgrep -af" not in line
        and str(os.getpid()) not in line.split(maxsplit=1)[0]
    ]
    if active:
        raise RuntimeError("training/evaluation already active: " + "; ".join(active))
    if total - used < 12_000:
        raise RuntimeError(f"less than 12 GB GPU memory free: {total - used} MiB")
    return {
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
        "active_training_or_evaluation": active,
    }


def build_command(
    scenario: str,
    paths: dict[str, Path],
    *,
    steps: int,
    envs: int,
    include_k8_input: bool = False,
) -> list[str]:
    command = [
        str(REPO / "isaaclab.sh"), "-p", str(PLAY),
        "--checkpoint", str(protocol.CHECKPOINT),
        "--task", "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
        "--curriculum_version", "warp_drive_e2e_final20_v1",
        "--stage", "5", "--seed", str(protocol.SEED), "--deterministic",
        "--num_envs", str(envs), "--steps", str(steps), "--headless", "--no_lidar_vis",
        "--arena_size", "15.0", "--num_goals_override", "1", "--no_goal_movement",
        "--long_corridor_eval", "--long_corridor_static_obstacles", "2",
        "--long_corridor_dynamic_obstacles", "2",
        "--long_corridor_free_width", "4.2",
        "--long_corridor_dynamic_speed_range", "0.25", "0.45",
        "--long_corridor_motion_mode", scenario,
        "--long_corridor_pause_mode", "default",
        "--long_corridor_random_2d_kinematics", "patrol",
        "--long_corridor_phase_audit",
        "--long_corridor_output", str(paths["corridor"]),
        "--long_corridor_phase_audit_output", str(paths["phase"]),
        "--vlp16_noise_mode", "full",
        "--lidar-distractor-eligibility", "valid_return_only",
        "--speed_rate", "0.7", "--speed_rate_obs", "ego",
        "--enable_actuator_dr", "--actuator_delay_range", "1", "1",
        "--actuator_velocity_scale", "1.0", "1.0",
        "--actuator_motor_lag", "1.0",
        "--d3_yield_audit", "--d3_yield_output", str(paths["d3"]),
        "--d3_shield_mode", "baseline",
        "--d5_feasibility_shadow", "--d5_feasibility_output", str(paths["d5"]),
        "--policy_step_trace_output", str(paths["trace"]),
    ]
    if include_k8_input:
        command.append("--policy_step_trace_include_k8_input")
    return command


def _finite(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")


def _paths(output: Path, scenario: str) -> dict[str, Path]:
    return {
        key: output / f"{scenario}_{suffix}"
        for key, suffix in {
            "log": "console.log",
            "corridor": "corridor.json",
            "phase": "motion_phase.json",
            "d3": "d3.json",
            "d5": "d5.json",
            "trace": "step_trace.npz",
            "analysis": "analysis.json",
        }.items()
    }


def run_cell(
    output: Path,
    scenario: str,
    *,
    steps: int,
    envs: int,
    include_k8_input: bool = False,
) -> dict:
    paths = _paths(output, scenario)
    command = build_command(
        scenario,
        paths,
        steps=steps,
        envs=envs,
        include_k8_input=include_k8_input,
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["CONDA_PREFIX"] = str(PYTHON.parent.parent)
    environment["CONDA_DEFAULT_ENV"] = "env_isaaclab"
    environment["PATH"] = f"{PYTHON.parent}:{environment['PATH']}"
    environment["PYTHONUNBUFFERED"] = "1"
    with paths["log"].open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{scenario} exited {result.returncode}: {paths['log']}")
    log = paths["log"].read_text(encoding="utf-8", errors="ignore")
    required = (
        "[POLICY-STEP-TRACE] record-only baseline enabled",
        "[POLICY-STEP-TRACE] npz written:",
        "reconciliation=True",
        "Actuator DR: delay=(1, 1) steps",
        f"motion={scenario}",
    )
    missing = [marker for marker in required if marker not in log]
    if missing:
        raise RuntimeError(f"{scenario} missing runtime markers: {missing}")
    for pattern in (r"Traceback \(most recent call last\)", r"CUDA out of memory", r"CUDA error:"):
        if re.search(pattern, log, flags=re.IGNORECASE):
            raise RuntimeError(f"{scenario} runtime error matched {pattern}")
    d3 = json.loads(paths["d3"].read_text())
    d5 = json.loads(paths["d5"].read_text())
    corridor = json.loads(paths["corridor"].read_text())
    if not bool(d3["self_check"]["reconciliation_ok"]):
        raise RuntimeError(f"{scenario} D3 reconciliation failed")
    if not bool(d5["self_check"]["reconciliation_ok"]):
        raise RuntimeError(f"{scenario} D5 reconciliation failed")
    arrays, metadata = load_trace(paths["trace"])
    if not bool(metadata["self_check"]["reconciliation_ok"]):
        raise RuntimeError(f"{scenario} trace reconciliation failed")
    analysis = (
        analyze_lateral_trace(arrays)
        if scenario == "lateral"
        else analyze_random2d_trace(arrays)
    )
    analysis["trace_metadata"] = metadata
    _finite(analysis)
    paths["analysis"].write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return {
        "scenario": scenario,
        "episodes": int(corridor["episodes"]),
        "sr": float(corridor["success_rate"]),
        "cr": float(corridor["collision_rate"]),
        "to": float(corridor["timeout_rate"]),
        "trace_self_check": metadata["self_check"],
        "analysis": analysis,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def render_summary(cells: dict[str, dict]) -> str:
    lat = cells["lateral"]["analysis"]
    rnd = cells["random_2d"]["analysis"]
    return "\n".join(
        [
            "# SA5 c600 policy-only step trace",
            "",
            "> Diagnostic evidence only. c600 is retained; no training or SA6 was started.",
            "",
            "| scenario | episodes | SR | CR | TO |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {name} | {cell['episodes']} | {cell['sr']:.2%} | {cell['cr']:.2%} | {cell['to']:.2%} |"
                for name, cell in cells.items()
            ],
            "",
            "## Lateral vacated-side events",
            "",
            f"- crossings: {lat['crossing_events']}",
            f"- departures observed within 3 s: {lat['departure_events']}",
            f"- vacated-side matches: {lat['vacated_side_matches']} / {lat['departure_events']}",
            f"- match fraction: {lat['vacated_side_match_fraction']}",
            f"- departure latency: {lat['departure_latency_s']}",
            "",
            "## Random-2D waypoint reversals",
            "",
            f"- waypoint switches: {rnd['waypoint_switch_events']}",
            f"- direction reversals: {rnd['direction_reversal_events']}",
            f"- stop latency: {rnd['stop_latency_s']}",
            f"- turn latency: {rnd['turn_latency_s']}",
            f"- obstacle collision within 3 s: {rnd['collision_within_3s_fraction']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="logs/gates/sa5_c600_step_trace/screen_20260826_r1",
    )
    parser.add_argument("--steps", type=int, default=protocol.ROLLOUT_STEPS)
    parser.add_argument("--num-envs", type=int, default=protocol.NUM_ENVS)
    args = parser.parse_args()
    output = (REPO / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    before = source_fingerprint()
    document = protocol.frozen_protocol()
    if sha256(protocol.CHECKPOINT) != protocol.CHECKPOINT_SHA256:
        raise RuntimeError("c600 checkpoint SHA-256 drifted")
    preflight = {
        "protocol": document,
        "source_fingerprint": before,
        "gpu_before": gpu_snapshot(),
        "status": "FROZEN_BEFORE_GPU",
    }
    (output / "PROTOCOL.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    try:
        cells = {
            scenario: run_cell(output, scenario, steps=args.steps, envs=args.num_envs)
            for scenario in SCENARIOS
        }
        after = source_fingerprint()
        if before != after:
            raise RuntimeError("source fingerprint drifted during trace suite")
        if sha256(protocol.CHECKPOINT) != protocol.CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint drifted during trace suite")
        manifest = {
            "schema": "sa5_c600_policy_step_trace_suite/v1",
            "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
            "protocol_sha256": document["sha256"],
            "checkpoint_sha256": protocol.CHECKPOINT_SHA256,
            "source_fingerprint_stable": True,
            "policy_actions_replaced": False,
            "training_authorized": False,
            "sa6_authorized": False,
            "cells": cells,
            "gpu_after": gpu_snapshot(),
        }
        (output / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "SUMMARY.md").write_text(render_summary(cells), encoding="utf-8")
    except Exception as error:
        (output / "INCOMPLETE_NO_VERDICT.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error": repr(error),
                    "completed_outputs_are_preserved": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    print(f"[SA5-C600-STEP-TRACE] complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
