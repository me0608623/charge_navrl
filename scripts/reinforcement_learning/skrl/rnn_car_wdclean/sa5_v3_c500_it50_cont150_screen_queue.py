"""Run the frozen two-phase c500-it50 continuation screen."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNNER = HERE / "run_sa5_v3_c500_it50_cont150_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c500_it50_cont150_screen_v1.json"
sys.path.insert(0, str(HERE))

import sa5_v3_c500_it50_cont150_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as implementation  # noqa: E402


implementation.RUNNER = RUNNER
implementation.FREEZE = FREEZE
implementation.protocol = protocol

SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO
    / "docs/freeze/sa5_v3_c500_it50_cont150_authorization_20260823.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_it50_cont150.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py",
    protocol.HISTORICAL_C50_SUMMARY,
    Path(implementation.__file__).resolve(),
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
    if frozen != protocol.screen_protocol():
        raise RuntimeError("continuation screen differs from frozen protocol")
    return frozen


def build_checkpoint_manifest() -> dict:
    expected_runtime = {
        "c50": (49, 6_400),
        "c75": (24, 3_200),
        "c100": (49, 6_400),
        "c125": (74, 9_600),
        "c150": (99, 12_800),
    }
    checkpoints = {}
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
            "conceptual_iteration": spec["conceptual_iteration"],
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_c500_it50_cont150_checkpoint_manifest/v1",
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
        "schema": "sa5_v3_c500_it50_cont150_screen_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c500-it50 bounded continuation screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Phase A",
        "",
        "| rank | checkpoint | target CR | native CR | worst normalized |",
        "|---:|---|---:|---:|---:|",
    ]
    rows = {
        row["checkpoint_name"]: row for row in summary["phase_a"]["rows"]
    }
    for rank, name in enumerate(summary["phase_a"]["ranking"], 1):
        row = rows[name]
        lines.append(
            f"| {rank} | {name} | {row['target_metrics']['cr']:.2%} | "
            f"{row['native_metrics']['cr']:.2%} | "
            f"{row['normalized']['worst']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Phase B",
            "",
            "| checkpoint | target z | native | narrow | P060 0S1D | P060 1S1D | accept |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    for row in summary["phase_b_results"]:
        retention = row["retention"]
        lines.append(
            f"| {row['checkpoint_name']} | "
            f"{row['target_improvement']['z']:.2f} | "
            f"{'PASS' if retention['nav_native']['pass'] else 'FAIL'} | "
            f"{'PASS' if retention['narrow_range']['pass'] else 'FAIL'} | "
            f"{'PASS' if retention['corridor_low_0s1d']['pass'] else 'FAIL'} | "
            f"{'PASS' if retention['corridor_low_1s1d']['pass'] else 'FAIL'} | "
            f"{'PASS' if row['acceptance_eligible'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Selected provisional SA5 parent: "
            f"`{summary['selected_provisional_sa5_parent']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "SA4 remains not graduated. SA6 was not started.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_cell(spec: dict, scenario: str, output: Path, frozen: dict) -> dict:
    implementation.protocol = protocol
    implementation.RUNNER = RUNNER
    return implementation._run_cell(spec, scenario, output, frozen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    args = parser.parse_args(argv)

    output = args.output_root.expanduser().resolve()
    if output != protocol.SCREEN_ROOT.resolve():
        raise ValueError("output root differs from frozen continuation root")
    if not args.execute:
        raise ValueError("formal continuation screen requires --execute")
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

    phase_a_payloads: list[dict] = []
    phase_b_payloads: list[dict] = []
    selected_names: list[str] = []
    try:
        for spec in protocol.CANDIDATE_SPECS:
            for scenario in protocol.PHASE_A_SCENARIOS:
                implementation.wait_for_resources(
                    resource_log,
                    allow_shared_gpu=args.allow_shared_gpu,
                    max_wait_minutes=args.max_wait_minutes,
                )
                phase_a_payloads.append(
                    _run_cell(spec, scenario, output / "phase_a", frozen)
                )
                if source_fingerprint() != before:
                    raise RuntimeError(
                        f"source drift after phase A {spec['name']}/{scenario}"
                    )

        phase_a = protocol.rank_phase_a(phase_a_payloads)
        selected_names = list(phase_a["selected_top_two"])
        (output / "PHASE_A_SUMMARY.json").write_text(
            json.dumps(phase_a, indent=2, sort_keys=True), encoding="utf-8"
        )
        selected_specs = [
            next(row for row in protocol.CANDIDATE_SPECS if row["name"] == name)
            for name in selected_names
        ]
        for spec in selected_specs:
            for scenario in protocol.PHASE_B_SCENARIOS:
                implementation.wait_for_resources(
                    resource_log,
                    allow_shared_gpu=args.allow_shared_gpu,
                    max_wait_minutes=args.max_wait_minutes,
                )
                phase_b_payloads.append(
                    _run_cell(spec, scenario, output / "phase_b", frozen)
                )
                if source_fingerprint() != before:
                    raise RuntimeError(
                        f"source drift after phase B {spec['name']}/{scenario}"
                    )

        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("continuation source fingerprint drifted")
        summary = protocol.final_verdict(phase_a_payloads, phase_b_payloads)
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
            "[SA5-V3-CONT150] COMPLETE selected="
            f"{summary['selected_provisional_sa5_parent']} "
            f"next={summary['next_action']} sa6_started=false",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_v3_c500_it50_cont150_screen_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "phase_a_completed_cells": len(phase_a_payloads),
            "phase_b_completed_cells": len(phase_b_payloads),
            "selected_top_two": selected_names,
            "acceptance_authorized": False,
            "sa6_started": False,
            "recorded_at": datetime.now().astimezone().isoformat(),
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[SA5-V3-CONT150] STOPPED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
