"""Run the frozen SA6-v3 target ranking and top-two retention screen.

The queue may wait for shared GPU resources. It never launches training or
SA7. Missing cells, source drift, checkpoint drift, or accounting failures
produce INCOMPLETE_NO_VERDICT and stop downstream selection.
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
RUNNER = HERE / "run_sa6_v3_checkpoint_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa6_v3_checkpoint_screen_v3.json"
sys.path.insert(0, str(HERE))

import sa6_v3_checkpoint_screen as protocol  # noqa: E402


FINGERPRINTED_SOURCES = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "e2e_sa6_v3_from_sa5_c600_p50.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "sim2real_stage_curriculum_v2.py",
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
SHARED_MIN_FREE_GPU_MIB = 12000
READY_POLLS_REQUIRED = 2


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
        raise RuntimeError("SA6-v3 screen protocol differs from frozen JSON")
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
        raise RuntimeError(f"unexpected nvidia-smi output: {gpu.stdout!r}")
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
    available_ram_gib = _mem_available_gib()
    if allow_shared_gpu:
        ready = bool(
            free_mib >= SHARED_MIN_FREE_GPU_MIB
            and available_ram_gib >= MIN_AVAILABLE_RAM_GIB
        )
        blockers = []
        policy_mode = "shared_memory_guarded"
    else:
        blockers = heavy_apps
        ready = bool(
            free_mib >= MIN_FREE_GPU_MIB
            and util_percent <= MAX_GPU_UTIL_PERCENT
            and not blockers
            and available_ram_gib >= MIN_AVAILABLE_RAM_GIB
        )
        policy_mode = "exclusive"
    return {
        "timestamp": now_iso(),
        "policy_mode": policy_mode,
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


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def wait_for_resources(
    wait_log: Path,
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
        with wait_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        print(
            f"[SA6-V3-SCREEN-WAIT] mode={snapshot['policy_mode']} "
            f"ready={snapshot['ready']} free={snapshot['gpu_free_mib']}MiB "
            f"util={snapshot['gpu_util_percent']}% "
            f"RAM={snapshot['available_ram_gib']:.1f}GiB",
            flush=True,
        )
        if source_fingerprint() != expected_fingerprint:
            raise RuntimeError("measurement source fingerprint drifted while waiting")
        verify_checkpoints()
        ready_polls = ready_polls + 1 if snapshot["ready"] else 0
        if ready_polls >= READY_POLLS_REQUIRED:
            return
        if max_wait_minutes > 0 and (
            time.monotonic() - started
        ) >= max_wait_minutes * 60:
            raise TimeoutError("timed out waiting for GPU resources")
        time.sleep(poll_seconds)


def planned_phase_a_cells() -> list[dict]:
    return [
        {
            **cell,
            "checkpoint": protocol.checkpoint_path(cell["checkpoint_name"]),
            "checkpoint_sha256": protocol.candidate_by_name(
                cell["checkpoint_name"]
            )["sha256"],
            "steps": protocol.phase_a_steps(cell["fixed_cell"]),
        }
        for cell in protocol.phase_a_cells()
    ]


def planned_phase_b_cells(top_two: list[str]) -> list[dict]:
    if len(top_two) != 2 or len(set(top_two)) != 2:
        raise ValueError(f"Phase B needs two distinct candidates, got {top_two}")
    return [
        {
            "cell": protocol.phase_b_cell_label(name, scenario),
            "checkpoint_name": name,
            "checkpoint": protocol.checkpoint_path(name),
            "checkpoint_sha256": protocol.candidate_by_name(name)["sha256"],
            "scenario": scenario,
            "steps": protocol.phase_b_steps(scenario),
        }
        for name in top_two
        for scenario in protocol.PHASE_B_SCENARIOS
    ]


def _cell_command(cell: dict, output_dir: Path, protocol_sha256: str, phase: str) -> list[str]:
    command = [
        str(PYTHON),
        str(RUNNER),
        str(cell["checkpoint"]),
        "--output-dir",
        str(output_dir),
        "--checkpoint-name",
        cell["checkpoint_name"],
        "--phase",
        phase,
        "--expect-checkpoint-sha256",
        cell["checkpoint_sha256"],
        "--expect-protocol-sha256",
        protocol_sha256,
        "--steps",
        str(cell["steps"]),
    ]
    if phase == "a":
        command.extend(["--fixed-cell", cell["fixed_cell"]])
    else:
        command.extend(["--scenario", cell["scenario"]])
    return command


def run_cell(cell: dict, phase_dir: Path, protocol_sha256: str, phase: str) -> dict:
    if phase == "a":
        cell_dir = phase_dir / cell["checkpoint_name"] / cell["fixed_cell"]
    else:
        cell_dir = phase_dir / cell["checkpoint_name"] / cell["scenario"]
    result = subprocess.run(
        _cell_command(cell, cell_dir, protocol_sha256, phase),
        check=False,
        cwd=REPO,
    )
    cell_path = cell_dir / "cell.json"
    if not cell_path.is_file():
        raise RuntimeError(
            f"cell runner exited {result.returncode} without cell.json: "
            f"{cell['cell']}"
        )
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    if result.returncode != 0:
        raise RuntimeError(
            f"cell runner returned {result.returncode} after writing {cell_path}"
        )
    return payload


def _render_markdown(verdict: dict) -> str:
    phase_a = verdict["phase_a"]
    lines = [
        "# SA6-v3 fixed checkpoint screen",
        "",
        "## Phase A: three SA6 profiles + four fixed 4S2D corridor families",
        "",
        "| rank | checkpoint | worst failure | worst CR | mean failure | mean CR | target gates | worst cell |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, name in enumerate(phase_a["ranked"], start=1):
        row = phase_a["summaries_by_checkpoint"][name]
        lines.append(
            f"| {rank} | {name} | {row['worst_failure_rate']:.4%} | "
            f"{row['worst_cr']:.4%} | {row['mean_failure_rate']:.4%} | "
            f"{row['mean_cr']:.4%} | "
            f"{'PASS' if row['all_target_hard_pass'] else 'FAIL'} | "
            f"{row['worst_cell']} |"
        )
    lines.extend(
        [
            "",
            f"Top two: {', '.join(phase_a['top_two'])}",
            "",
            "## Phase B: native, narrow, and P060 low-density retention",
            "",
            "| checkpoint | scenario | n | SR | CR | TO | gate |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in verdict["phase_b_candidates"]:
        for scenario in protocol.PHASE_B_SCENARIOS:
            metrics = candidate["cells"][scenario]
            gate = protocol.evaluate_metrics(scenario, metrics)
            lines.append(
                f"| {candidate['checkpoint_name']} | {scenario} | "
                f"{int(metrics['n'])} | {float(metrics['sr']):.4%} | "
                f"{float(metrics['cr']):.4%} | {float(metrics['to']):.4%} | "
                f"{'PASS' if gate['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "Recommended for human consideration: "
            f"{verdict['recommended_for_human_consideration'] or 'none'}",
            "",
            "This single-seed screen accepts no parent and starts neither training nor SA7.",
            "",
        ]
    )
    return "\n".join(lines)


def _resource_ready_or_wait(
    *,
    wait_log: Path,
    poll_seconds: int,
    max_wait_minutes: int,
    before: dict[str, str],
    allow_shared_gpu: bool,
) -> None:
    snapshot = resource_snapshot(allow_shared_gpu=allow_shared_gpu)
    if snapshot["ready"]:
        return
    wait_for_resources(
        wait_log,
        poll_seconds=poll_seconds,
        max_wait_minutes=max_wait_minutes,
        expected_fingerprint=before,
        allow_shared_gpu=allow_shared_gpu,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-minutes", type=int, default=0)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")

    frozen = verify_frozen_protocol()
    checkpoints = verify_checkpoints()
    phase_a_cells = planned_phase_a_cells()
    print(
        f"[SA6-V3-SCREEN] Phase A {len(phase_a_cells)} cells = "
        f"{len(protocol.CANDIDATES)} checkpoints x "
        f"{len(protocol.PHASE_A_SPECS)} fixed cells "
        "(3 profiles + 4 family regressions)",
        flush=True,
    )
    print(
        f"[SA6-V3-SCREEN] fixed seed={protocol.SEED} d{protocol.DELAY_STEPS} "
        f"speed_rate={protocol.SPEED_RATE:g} full-noise valid-only "
        f"min_episodes={protocol.MIN_EPISODES}",
        flush=True,
    )
    if not args.execute:
        for cell in phase_a_cells:
            print(
                f"  {cell['checkpoint_name']:4s} "
                f"{cell['fixed_cell']:30s} steps={cell['steps']}",
                flush=True,
            )
        print("[SA6-V3-SCREEN] dry run; pass --execute to run")
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
                for cell in phase_a_cells
            ],
            "resource_policy": (
                "shared_memory_guarded" if args.allow_shared_gpu else "exclusive"
            ),
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
            snapshot = resource_snapshot(allow_shared_gpu=args.allow_shared_gpu)
            if not snapshot["ready"]:
                raise RuntimeError(f"GPU resources not ready: {snapshot}")

        phase_a_dir = output / "phase_a_target_profiles"
        for index, cell in enumerate(phase_a_cells, start=1):
            _resource_ready_or_wait(
                wait_log=paths["wait_log"],
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
                before=before,
                allow_shared_gpu=args.allow_shared_gpu,
            )
            print(
                f"[SA6-V3-SCREEN] Phase A {index}/{len(phase_a_cells)} "
                f"{cell['cell']}",
                flush=True,
            )
            payload = run_cell(cell, phase_a_dir, frozen["sha256"], "a")
            protocol.validate_phase_a_cell(payload)
            phase_a_payloads.append(payload)
            if source_fingerprint() != before:
                raise RuntimeError(
                    f"measurement source fingerprint drift after Phase-A cell {index}"
                )
            verify_checkpoints()

        phase_a = protocol.rank_phase_a(phase_a_payloads)
        _write_json(
            paths["phase_a"],
            {
                "status": "COMPLETE_VALID_PHASE_A",
                "cells_planned": len(phase_a_cells),
                "cells_completed": len(phase_a_payloads),
                **phase_a,
            },
        )
        _write_json(
            paths["selection"],
            {
                "status": "TOP_TWO_SELECTED_AFTER_COMPLETE_PHASE_A",
                "top_two": phase_a["top_two"],
                "ranking": phase_a["ranked"],
                "protocol_sha256": frozen["sha256"],
            },
        )
        print(
            f"[SA6-V3-SCREEN] Phase A ranked={phase_a['ranked']} "
            f"top_two={phase_a['top_two']}",
            flush=True,
        )

        phase_b_cells = planned_phase_b_cells(phase_a["top_two"])
        phase_b_dir = output / "phase_b_retention"
        for index, cell in enumerate(phase_b_cells, start=1):
            _resource_ready_or_wait(
                wait_log=paths["wait_log"],
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
                before=before,
                allow_shared_gpu=args.allow_shared_gpu,
            )
            print(
                f"[SA6-V3-SCREEN] Phase B {index}/{len(phase_b_cells)} "
                f"{cell['cell']}",
                flush=True,
            )
            payload = run_cell(cell, phase_b_dir, frozen["sha256"], "b")
            protocol.validate_phase_b_cell(payload, cell["checkpoint_name"])
            phase_b_payloads.append(payload)
            if source_fingerprint() != before:
                raise RuntimeError(
                    f"measurement source fingerprint drift after Phase-B cell {index}"
                )
            verify_checkpoints()

        verdict = protocol.final_verdict(phase_a_payloads, phase_b_payloads)
        after = source_fingerprint()
        if after != before:
            raise RuntimeError("measurement source fingerprint changed")
        _write_json(paths["source_after"], after)
        summary = {
            "schema": "sa6_v3_checkpoint_screen_bundle/v1",
            "status": "COMPLETE_VALID_SINGLE_SEED_SCREEN",
            "completed_at": now_iso(),
            "protocol": frozen,
            "source_fingerprint_stable": True,
            "phase_a_cells": phase_a_payloads,
            "phase_b_cells": phase_b_payloads,
            "verdict": verdict,
            "accepted_parent": False,
            "training_started": False,
            "sa7_started": False,
        }
        _write_json(paths["summary"], summary)
        paths["markdown"].write_text(
            _render_markdown(verdict), encoding="utf-8"
        )
        print(
            "[SA6-V3-SCREEN] COMPLETE recommended="
            f"{verdict['recommended_for_human_consideration']} "
            "accepted_parent=False training_started=False sa7_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json(
            paths["incomplete"],
            {
                "schema": "sa6_v3_checkpoint_screen_incomplete/v1",
                "status": "INCOMPLETE_NO_VERDICT",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "phase_a_cells_completed": len(phase_a_payloads),
                "phase_b_cells_completed": len(phase_b_payloads),
                "protocol_sha256": frozen["sha256"],
                "accepted_parent": False,
                "training_started": False,
                "sa7_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
