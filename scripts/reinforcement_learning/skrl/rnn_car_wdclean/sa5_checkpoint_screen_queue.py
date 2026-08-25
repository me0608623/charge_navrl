"""Run the frozen SA5 corridor ranking, then top-two retention screen.

The queue may wait for the shared GPU, but it never launches training or SA6.
Any missing cell, checkpoint mismatch, source drift, or accounting error stops
the queue before a downstream verdict is produced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = HERE / "run_sa5_checkpoint_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_checkpoint_screen_v2.json"
sys.path.insert(0, str(HERE))

import sa5_checkpoint_screen as protocol  # noqa: E402


FINGERPRINTED_SOURCES = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "sim2real_stage_curriculum_v2.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/corridor_density.py",
)

MIN_FREE_GPU_MIB = 26000
MAX_GPU_UTIL_PERCENT = 10
MAX_COMPUTE_APP_MIB = 1000
MIN_AVAILABLE_RAM_GIB = 12.0
READY_POLLS_REQUIRED = 2
SHARED_MIN_FREE_GPU_MIB = 12000


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> dict[str, str]:
    missing = [str(path) for path in FINGERPRINTED_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    return {
        str(path.relative_to(REPO)): sha256_of(path)
        for path in FINGERPRINTED_SOURCES
    }


def verify_frozen_protocol() -> dict:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = protocol.screen_protocol()
    if frozen != current:
        raise RuntimeError("SA5 screen protocol differs from frozen JSON")
    return frozen


def verify_checkpoints() -> dict[str, dict]:
    observed = {}
    for candidate in protocol.CANDIDATES:
        path = protocol.checkpoint_path(candidate)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_of(path)
        if digest != candidate["sha256"]:
            raise RuntimeError(
                f"{candidate['name']} hash mismatch: {digest} != "
                f"{candidate['sha256']}"
            )
        observed[candidate["name"]] = {
            "path": str(path),
            "sha256": digest,
            "conceptual_iteration": candidate["conceptual_iteration"],
        }
    return observed


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def resource_snapshot(*, allow_shared_gpu: bool = False) -> dict:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    apps = subprocess.run(
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
    if gpu.returncode != 0 or apps.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: gpu={gpu.stderr!r} apps={apps.stderr!r}"
        )
    fields = [field.strip() for field in gpu.stdout.strip().split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi GPU output: {gpu.stdout!r}")
    free_mib, util_percent, temperature_c = (int(value) for value in fields)
    compute_apps = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            compute_apps.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": int(parts[2]),
                }
            )
    heavy_apps = [
        app for app in compute_apps if app["used_memory_mib"] > MAX_COMPUTE_APP_MIB
    ]
    blockers = [] if allow_shared_gpu else heavy_apps
    available_ram_gib = _mem_available_gib()
    if allow_shared_gpu:
        ready = bool(
            free_mib >= SHARED_MIN_FREE_GPU_MIB
            and available_ram_gib >= MIN_AVAILABLE_RAM_GIB
        )
        mode = "shared_memory_guarded"
    else:
        ready = bool(
            free_mib >= MIN_FREE_GPU_MIB
            and util_percent <= MAX_GPU_UTIL_PERCENT
            and not blockers
            and available_ram_gib >= MIN_AVAILABLE_RAM_GIB
        )
        mode = "exclusive"
    return {
        "timestamp": now_iso(),
        "policy_mode": mode,
        "ready": ready,
        "gpu_free_mib": free_mib,
        "gpu_util_percent": util_percent,
        "gpu_temperature_c": temperature_c,
        "available_ram_gib": available_ram_gib,
        "compute_apps": compute_apps,
        "coexisting_heavy_apps": heavy_apps,
        "blocking_apps": blockers,
        "requirements": {
            "min_gpu_free_mib": (
                SHARED_MIN_FREE_GPU_MIB if allow_shared_gpu else MIN_FREE_GPU_MIB
            ),
            "max_gpu_util_percent": (
                None if allow_shared_gpu else MAX_GPU_UTIL_PERCENT
            ),
            "max_compute_app_mib": (
                None if allow_shared_gpu else MAX_COMPUTE_APP_MIB
            ),
            "min_available_ram_gib": MIN_AVAILABLE_RAM_GIB,
            "consecutive_ready_polls": READY_POLLS_REQUIRED,
        },
    }


def wait_for_resources(
    log_path: Path,
    *,
    poll_seconds: int,
    max_wait_minutes: int,
    expected_fingerprint: dict[str, str],
    allow_shared_gpu: bool,
) -> None:
    started = time.monotonic()
    ready_polls = 0
    while True:
        snapshot = resource_snapshot(allow_shared_gpu=allow_shared_gpu)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        blockers = ", ".join(
            f"pid={app['pid']}:{app['used_memory_mib']}MiB"
            for app in snapshot["blocking_apps"]
        ) or "none"
        sharing = ", ".join(
            f"pid={app['pid']}:{app['used_memory_mib']}MiB"
            for app in snapshot["coexisting_heavy_apps"]
        ) or "none"
        print(
            f"[SA5-SCREEN-WAIT] mode={snapshot['policy_mode']} "
            f"ready={snapshot['ready']} "
            f"free={snapshot['gpu_free_mib']}MiB util={snapshot['gpu_util_percent']}% "
            f"RAM={snapshot['available_ram_gib']:.1f}GiB blockers={blockers} "
            f"coexisting={sharing}",
            flush=True,
        )
        if source_fingerprint() != expected_fingerprint:
            raise RuntimeError("measurement source drifted while waiting for GPU")
        verify_checkpoints()
        if snapshot["ready"]:
            ready_polls += 1
            if ready_polls >= READY_POLLS_REQUIRED:
                return
        else:
            ready_polls = 0
        if max_wait_minutes > 0 and (
            time.monotonic() - started
        ) >= max_wait_minutes * 60:
            raise TimeoutError("timed out waiting for an idle GPU")
        time.sleep(poll_seconds)


def planned_phase_a_cells() -> list[dict]:
    return [
        {
            "checkpoint_name": candidate["name"],
            "checkpoint": protocol.checkpoint_path(candidate),
            "checkpoint_sha256": candidate["sha256"],
            "scenario": scenario,
            "steps": protocol.STEPS_BY_SCENARIO[scenario],
        }
        for candidate in protocol.CANDIDATES
        for scenario in protocol.CORRIDOR_SCENARIOS
    ]


def planned_phase_b_cells(top_two: list[str]) -> list[dict]:
    if len(top_two) != 2 or len(set(top_two)) != 2:
        raise ValueError(f"Phase B requires two distinct candidates, got {top_two}")
    return [
        {
            "checkpoint_name": name,
            "checkpoint": protocol.checkpoint_path(name),
            "checkpoint_sha256": protocol.candidate_by_name(name)["sha256"],
            "scenario": scenario,
            "steps": protocol.STEPS_BY_SCENARIO[scenario],
        }
        for name in top_two
        for scenario in protocol.RETENTION_SCENARIOS
    ]


def run_cell(cell: dict, phase_dir: Path, protocol_sha256: str) -> dict:
    cell_dir = phase_dir / cell["checkpoint_name"] / cell["scenario"]
    command = [
        str(PYTHON),
        str(RUNNER),
        str(cell["checkpoint"]),
        "--output-dir",
        str(cell_dir),
        "--checkpoint-name",
        cell["checkpoint_name"],
        "--scenario",
        cell["scenario"],
        "--expect-checkpoint-sha256",
        cell["checkpoint_sha256"],
        "--expect-protocol-sha256",
        protocol_sha256,
        "--steps",
        str(cell["steps"]),
    ]
    result = subprocess.run(command, check=False, cwd=REPO)
    stem = (
        f"{cell['scenario']}_g5_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    )
    cell_path = cell_dir / f"{stem}_cell.json"
    if not cell_path.is_file():
        raise RuntimeError(
            f"cell runner exited {result.returncode} without JSON: "
            f"{cell['checkpoint_name']} {cell['scenario']}"
        )
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    payload["runner_returncode"] = result.returncode
    return payload


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _render_markdown(verdict: dict) -> str:
    phase_a = verdict["phase_a"]
    by_name = {row["checkpoint_name"]: row for row in phase_a["summaries"]}
    lines = [
        "# SA5 fixed checkpoint screen",
        "",
        "## Phase A corridor-family ranking",
        "",
        "| rank | checkpoint | lateral CR | longitudinal CR | random2d CR | mixed CR | worst CR | mean CR | corridor gates |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, name in enumerate(phase_a["ranked"], start=1):
        row = by_name[name]
        cr = row["family_cr"]
        lines.append(
            f"| {rank} | {name} | {cr['corridor_lateral']:.4%} | "
            f"{cr['corridor_longitudinal']:.4%} | "
            f"{cr['corridor_random2d']:.4%} | {cr['corridor_mixed']:.4%} | "
            f"{row['worst_family_cr']:.4%} | {row['mean_family_cr']:.4%} | "
            f"{'PASS' if row['all_corridor_hard_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Top two: {', '.join(phase_a['top_two'])}",
            f"Leader tie group (< {phase_a['tie_margin']:.3f}): "
            f"{', '.join(phase_a['leader_tie_group'])}",
            "",
            "## Phase B native and narrow retention",
            "",
            "| checkpoint | native SR | native CR | narrow SR | narrow CR | crossing | direct | all six gates |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in verdict["phase_b_candidates"]:
        native = row["native"]
        narrow = row["narrow"]
        lines.append(
            f"| {row['checkpoint_name']} | {native['sr']:.4%} | "
            f"{native['cr']:.4%} | {narrow['sr']:.4%} | "
            f"{narrow['cr']:.4%} | {narrow['crossing_rate']:.4%} | "
            f"{narrow['direct_crossing_rate']:.4%} | "
            f"{'PASS' if row['all_six_hard_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Recommended for human consideration: "
            f"{verdict['recommended_for_human_consideration'] or 'none'}",
            "",
            "This single-seed screen accepts no parent and starts neither training nor SA6.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help=(
            "allow coexisting GPU work when at least 12,000 MiB remains free; "
            "GPU utilization and other app size become descriptive only"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=0,
        help="0 waits indefinitely",
    )
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")

    frozen = verify_frozen_protocol()
    checkpoints = verify_checkpoints()
    cells = planned_phase_a_cells()
    print(
        f"[SA5-SCREEN] Phase A {len(cells)} cells = "
        f"{len(protocol.CANDIDATES)} checkpoints x "
        f"{len(protocol.CORRIDOR_SCENARIOS)} families",
        flush=True,
    )
    print(
        f"[SA5-SCREEN] fixed seed={protocol.SEED} d{protocol.DELAY_STEPS} "
        f"eligibility={protocol.LIDAR_DISTRACTOR_ELIGIBILITY} "
        f"min_episodes={protocol.MIN_EPISODES}",
        flush=True,
    )
    for cell in cells:
        print(
            f"  {cell['checkpoint_name']:4s} {cell['scenario']:24s} "
            f"steps={cell['steps']}",
            flush=True,
        )
    if not args.execute:
        print("[SA5-SCREEN] dry run; pass --execute to run")
        return 0

    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    paths = {
        "preregistration": output / "PREREGISTRATION.json",
        "wait_log": output / "resource_wait.jsonl",
        "phase_a": output / "PHASE_A_SUMMARY.json",
        "selection": output / "TOP2_SELECTION.json",
        "summary": output / "SUMMARY.json",
        "markdown": output / "SUMMARY.md",
        "incomplete": output / "INCOMPLETE_NO_VERDICT.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    before = source_fingerprint()
    _write_json(paths["source_before"], before)
    _write_json(
        paths["preregistration"],
        {
            "status": "PREREGISTERED_BEFORE_ROLLOUT",
            "created_at": now_iso(),
            "protocol": frozen,
            "checkpoints": checkpoints,
            "source_fingerprint": before,
            "phase_a_cells": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in cell.items()
                }
                for cell in cells
            ],
            "resource_policy": {
                "mode": (
                    "shared_memory_guarded"
                    if args.allow_shared_gpu
                    else "exclusive"
                ),
                "min_gpu_free_mib": (
                    SHARED_MIN_FREE_GPU_MIB
                    if args.allow_shared_gpu
                    else MIN_FREE_GPU_MIB
                ),
            },
        },
    )

    phase_a_payloads: list[dict] = []
    phase_b_payloads: list[dict] = []
    try:
        if args.wait_for_gpu:
            wait_for_resources(
                paths["wait_log"],
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
                expected_fingerprint=before,
                allow_shared_gpu=args.allow_shared_gpu,
            )
        else:
            snapshot = resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                raise RuntimeError(f"GPU resources not ready: {snapshot}")

        phase_a_dir = output / "phase_a_corridor"
        for index, cell in enumerate(cells, start=1):
            snapshot = resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                wait_for_resources(
                    paths["wait_log"],
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    expected_fingerprint=before,
                    allow_shared_gpu=args.allow_shared_gpu,
                )
            print(
                f"[SA5-SCREEN] Phase A cell {index}/{len(cells)}: "
                f"{cell['checkpoint_name']} {cell['scenario']}",
                flush=True,
            )
            phase_a_payloads.append(
                run_cell(cell, phase_a_dir, frozen["sha256"])
            )
            if source_fingerprint() != before:
                raise RuntimeError(f"measurement source drift after Phase-A cell {index}")
            verify_checkpoints()

        phase_a_result = protocol.rank_phase_a(phase_a_payloads)
        phase_a_summary = {
            "schema": "sa5_checkpoint_screen_phase_a/v1",
            "status": "COMPLETE_VALID_PHASE_A",
            "cells_planned": len(cells),
            "cells_completed": len(phase_a_payloads),
            **phase_a_result,
        }
        _write_json(paths["phase_a"], phase_a_summary)
        _write_json(
            paths["selection"],
            {
                "status": "TOP_TWO_SELECTED_AFTER_COMPLETE_PHASE_A",
                "top_two": phase_a_result["top_two"],
                "ranking": phase_a_result["ranked"],
                "protocol_sha256": frozen["sha256"],
            },
        )
        print(
            f"[SA5-SCREEN] Phase A ranked={phase_a_result['ranked']} "
            f"top_two={phase_a_result['top_two']}",
            flush=True,
        )

        phase_b_cells = planned_phase_b_cells(phase_a_result["top_two"])
        phase_b_dir = output / "phase_b_retention"
        for index, cell in enumerate(phase_b_cells, start=1):
            snapshot = resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                wait_for_resources(
                    paths["wait_log"],
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    expected_fingerprint=before,
                    allow_shared_gpu=args.allow_shared_gpu,
                )
            print(
                f"[SA5-SCREEN] Phase B cell {index}/{len(phase_b_cells)}: "
                f"{cell['checkpoint_name']} {cell['scenario']}",
                flush=True,
            )
            phase_b_payloads.append(
                run_cell(cell, phase_b_dir, frozen["sha256"])
            )
            if source_fingerprint() != before:
                raise RuntimeError(f"measurement source drift after Phase-B cell {index}")
            verify_checkpoints()

        verdict = protocol.final_verdict(phase_a_payloads, phase_b_payloads)
        after = source_fingerprint()
        if after != before:
            raise RuntimeError("measurement source fingerprint changed")
        _write_json(paths["source_after"], after)
        summary = {
            "schema": "sa5_checkpoint_screen_bundle/v1",
            "status": "COMPLETE_VALID_SINGLE_SEED_SCREEN",
            "completed_at": now_iso(),
            "protocol": frozen,
            "source_fingerprint_stable": True,
            "resource_policy_mode": (
                "shared_memory_guarded"
                if args.allow_shared_gpu
                else "exclusive"
            ),
            "phase_a_cells": phase_a_payloads,
            "phase_b_cells": phase_b_payloads,
            "verdict": verdict,
            "accepted_parent": False,
            "training_started": False,
            "sa6_started": False,
        }
        _write_json(paths["summary"], summary)
        paths["markdown"].write_text(
            _render_markdown(verdict), encoding="utf-8"
        )
        print(
            "[SA5-SCREEN] COMPLETE "
            f"recommended={verdict['recommended_for_human_consideration']} "
            "accepted_parent=False training_started=False sa6_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json(
            paths["incomplete"],
            {
                "schema": "sa5_checkpoint_screen_incomplete/v1",
                "status": "INCOMPLETE_NO_VERDICT",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "phase_a_cells_completed": len(phase_a_payloads),
                "phase_b_cells_completed": len(phase_b_payloads),
                "protocol_sha256": frozen["sha256"],
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
