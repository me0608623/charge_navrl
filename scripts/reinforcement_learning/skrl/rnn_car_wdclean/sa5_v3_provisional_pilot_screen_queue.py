"""Run the frozen 15-cell provisional SA5-v3 pilot screen."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = HERE / "run_sa5_v3_provisional_pilot_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_provisional_pilot_screen_v1.json"
sys.path.insert(0, str(HERE))

import sa5_r2_checkpoint_screen_queue as resource_guard  # noqa: E402
import sa5_v3_provisional_pilot_screen as protocol  # noqa: E402


SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO
    / "docs/freeze/sa5_v3_provisional_c550_parent_authorization_20260822.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_provisional_from_sa4v3_c550_p50.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "run_sa3_v3_retention_screen_cell.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/observations/obs_functions.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_protocol() -> dict:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = protocol.screen_protocol()
    if frozen != current:
        raise RuntimeError("SA5-v3 pilot screen differs from frozen protocol")
    return frozen


def build_checkpoint_manifest() -> dict:
    checkpoints = {}
    expected_runtime = {
        "parent_c550": (249, 32_000),
        "pilot_it25": (24, 3_200),
        "pilot_it50": (49, 6_400),
    }
    for spec in protocol.CANDIDATE_SPECS:
        name = spec["name"]
        path = protocol.checkpoint_path(spec).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_of(path)
        expected_hash = spec.get("expected_sha256")
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_iteration, expected_steps = expected_runtime[name]
        if (
            int(payload.get("iteration", -1)) != expected_iteration
            or int(payload.get("total_steps", -1)) != expected_steps
        ):
            raise RuntimeError(f"{name} checkpoint iteration ledger mismatch")
        optimizer = payload.get("charge_opt_rl") or {}
        if not optimizer.get("state") or not optimizer.get("param_groups"):
            raise RuntimeError(f"{name} lacks resumable RL optimizer state")
        checkpoints[name] = {
            "path": str(path),
            "sha256": digest,
            "iteration": expected_iteration,
            "total_steps": expected_steps,
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_provisional_pilot_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "run_name": protocol.RUN_NAME,
        "checkpoints": checkpoints,
    }


def source_fingerprint() -> dict:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    files = {
        str(path.relative_to(REPO)): sha256_of(path) for path in SOURCE_PATHS
    }
    for spec in protocol.CANDIDATE_SPECS:
        path = protocol.checkpoint_path(spec).resolve()
        files[str(path.relative_to(REPO))] = sha256_of(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sa5_v3_provisional_pilot_screen_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def wait_for_resources(
    log_path: Path,
    *,
    allow_shared_gpu: bool,
    max_wait_minutes: int,
) -> None:
    started = time.monotonic()
    ready_polls = 0
    while True:
        snapshot = resource_guard.resource_snapshot(
            allow_shared_gpu=allow_shared_gpu
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        print(
            "[SA5-V3-SCREEN-WAIT] "
            f"mode={snapshot['policy_mode']} ready={snapshot['ready']} "
            f"free={snapshot['gpu_free_mib']}MiB "
            f"util={snapshot['gpu_util_percent']}%",
            flush=True,
        )
        if snapshot["ready"]:
            ready_polls += 1
            if ready_polls >= 2:
                return
        else:
            ready_polls = 0
        if max_wait_minutes > 0 and time.monotonic() - started > (
            max_wait_minutes * 60
        ):
            raise TimeoutError("timed out waiting for screen GPU resources")
        time.sleep(10)


def _run_cell(spec: dict, scenario: str, output: Path, frozen: dict) -> dict:
    candidate = protocol.candidate_by_name(spec["name"])
    cell_dir = output / spec["name"] / scenario
    command = [
        str(PYTHON),
        str(RUNNER),
        str(protocol.checkpoint_path(spec)),
        "--output-dir",
        str(cell_dir),
        "--checkpoint-name",
        spec["name"],
        "--scenario",
        scenario,
        "--expect-checkpoint-sha256",
        candidate["sha256"],
        "--expect-protocol-sha256",
        frozen["sha256"],
        "--steps",
        str(protocol.STEPS_BY_SCENARIO[scenario]),
    ]
    print(f"[SA5-V3-SCREEN] START {spec['name']} {scenario}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=REPO, env=env, check=False)
    matches = list(cell_dir.glob("*_cell.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{spec['name']}/{scenario} produced {len(matches)} cell files"
        )
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    protocol.validate_cell(payload)
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"{spec['name']}/{scenario} runner exited {result.returncode}"
        )
    print(
        f"[SA5-V3-SCREEN] DONE {spec['name']} {scenario} "
        f"n={payload['metrics']['n']} CR={payload['metrics']['cr']:.4f} "
        f"gate={'PASS' if payload['threshold_pass'] else 'FAIL'}",
        flush=True,
    )
    return payload


def render_markdown(summary: dict) -> str:
    lines = [
        "# Provisional SA5-v3 c550 pilot screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| checkpoint | target SR | target CR | target z | retention | extension |",
        "|---|---:|---:|---:|---|---|",
    ]
    parent = summary["parent"]
    lines.append(
        f"| parent_c550 | {parent['target_metrics']['sr']:.2%} | "
        f"{parent['target_metrics']['cr']:.2%} | - | baseline | - |"
    )
    for row in summary["candidate_results"]:
        lines.append(
            f"| {row['checkpoint_name']} | {row['target_metrics']['sr']:.2%} | "
            f"{row['target_metrics']['cr']:.2%} | "
            f"{row['target_improvement']['z']:.2f} | "
            f"{'PASS' if row['retention_pass'] else 'FAIL'} | "
            f"{'PASS' if row['extension_eligible'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Selected extension checkpoint: `{summary['selected_extension_checkpoint']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "SA4 remains not graduated. This screen starts neither training nor SA6.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    args = parser.parse_args(argv)

    output = args.output_root.expanduser().resolve()
    if output != protocol.SCREEN_ROOT.resolve():
        raise ValueError("output root differs from frozen screen root")
    if not args.execute:
        raise ValueError("formal screen requires --execute")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"screen output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen = verify_frozen_protocol()
    manifest = build_checkpoint_manifest()
    protocol.CHECKPOINT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    (output / "PREREGISTRATION.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    before = source_fingerprint()
    (output / "SOURCE_FINGERPRINT_BEFORE.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )
    resource_log = output / "RESOURCE_WAIT.jsonl"

    completed: list[dict] = []
    try:
        for spec in protocol.CANDIDATE_SPECS:
            for scenario in protocol.ALL_SCENARIOS:
                wait_for_resources(
                    resource_log,
                    allow_shared_gpu=args.allow_shared_gpu,
                    max_wait_minutes=args.max_wait_minutes,
                )
                completed.append(_run_cell(spec, scenario, output, frozen))
                if source_fingerprint() != before:
                    raise RuntimeError(
                        f"source drift after {spec['name']}/{scenario}"
                    )
        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("source fingerprint drifted during screen")
        summary = protocol.final_verdict(completed)
        summary.update(
            {
                "completed_at": datetime.now().astimezone().isoformat(),
                "source_fingerprint_stable": True,
                "source_fingerprint_sha256": before["sha256"],
                "checkpoint_manifest_sha256": sha256_of(
                    protocol.CHECKPOINT_MANIFEST
                ),
            }
        )
        (output / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output / "SUMMARY.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(
            "[SA5-V3-SCREEN] COMPLETE selected="
            f"{summary['selected_extension_checkpoint']} "
            f"next={summary['next_action']} sa4_graduated=false sa6_started=false",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_v3_provisional_pilot_screen_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [
                f"{row.get('checkpoint_name')}/{row.get('scenario')}"
                for row in completed
            ],
            "expected_cells": len(protocol.CANDIDATE_SPECS)
            * len(protocol.ALL_SCENARIOS),
            "extension_authorized": False,
            "sa6_started": False,
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
