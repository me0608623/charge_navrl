"""Run the frozen 24-cell SA5-v3 c50 difficulty-frontier screen."""

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
RUNNER = HERE / "run_sa5_v3_c50_difficulty_frontier_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c50_difficulty_frontier_v1.json"
sys.path.insert(0, str(HERE))

import sa5_v3_c50_difficulty_frontier as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as implementation  # noqa: E402


SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    protocol.AUTHORIZATION,
    Path(implementation.__file__).resolve(),
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "run_sa3_v3_retention_screen_cell.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py",
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
        raise RuntimeError("difficulty-frontier screen differs from frozen protocol")
    if sha256_of(protocol.AUTHORIZATION) != protocol.AUTHORIZATION_SHA256:
        raise RuntimeError("difficulty-frontier authorization hash mismatch")
    return frozen


def build_checkpoint_manifest() -> dict:
    path = protocol.checkpoint_path().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_of(path)
    if digest != protocol.ANCHOR["expected_sha256"]:
        raise RuntimeError("c50 diagnostic anchor checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != 49 or int(payload.get("total_steps", -1)) != 6400:
        raise RuntimeError("c50 diagnostic anchor iteration ledger mismatch")
    optimizer = payload.get("charge_opt_rl") or {}
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise RuntimeError("c50 diagnostic anchor lacks resumable RL optimizer")
    return {
        "schema": "sa5_v3_c50_difficulty_frontier_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "anchor_status": protocol.ANCHOR["status"],
        "checkpoint": {
            "name": protocol.ANCHOR["name"],
            "path": str(path),
            "sha256": digest,
            "iteration": 49,
            "total_steps": 6400,
            "conceptual_iteration": 50,
            "rl_optimizer_state_entries": len(optimizer["state"]),
        },
    }


def source_fingerprint() -> dict:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    files = {
        str(path.relative_to(REPO)): sha256_of(path) for path in SOURCE_PATHS
    }
    checkpoint = protocol.checkpoint_path().resolve()
    files[str(checkpoint.relative_to(REPO))] = sha256_of(checkpoint)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sa5_v3_c50_difficulty_frontier_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c50 difficulty frontier",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The checkpoint is an ungraduated diagnostic anchor, not a formal parent.",
        "",
        "| density | mode | n | SR | CR | TO | gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        density = f"{row['static_obstacles']}S{row['dynamic_obstacles']}D"
        lines.append(
            f"| {density} | {row['motion_mode']} | {row['n']} | "
            f"{row['sr']:.2%} | {row['cr']:.2%} | {row['to']:.2%} | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Boundary edges",
            "",
            "| kind | mode | source | destination | delta CR | gate crossing | severe jump |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for edge in summary["edges"]:
        lines.append(
            f"| {edge['kind']} | {edge['motion_mode']} | {edge['source']} | "
            f"{edge['destination']} | {edge['delta_cr']:+.2%} | "
            f"{edge['gate_crossing']} | {edge['severe_jump']} |"
        )
    lines.extend(
        [
            "",
            "## Layout 405/410",
            "",
            "| scenario | layout 405 CR | layout 410 CR | delta | isolated 410 failure |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["layout_405_410"]:
        lines.append(
            f"| {row['scenario']} | {row['layout_405_cr']:.2%} | "
            f"{row['layout_410_cr']:.2%} | "
            f"{row['layout_410_minus_405_cr']:+.2%} | "
            f"{row['layout_410_isolated_failure']} |"
        )
    lines.extend(
        [
            "",
            f"Recommended next action: `{summary['recommended_next_action']}`",
            "",
            "The screen did not start a pilot or SA6.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_cell(scenario: str, output: Path, frozen: dict) -> dict:
    saved = {
        "protocol": implementation.protocol,
        "RUNNER": implementation.RUNNER,
        "FREEZE": implementation.FREEZE,
    }
    implementation.protocol = protocol
    implementation.RUNNER = RUNNER
    implementation.FREEZE = FREEZE
    try:
        return implementation._run_cell(
            protocol.ANCHOR, scenario, output, frozen
        )
    finally:
        for name, value in saved.items():
            setattr(implementation, name, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    args = parser.parse_args(argv)

    output = args.output_root.expanduser().resolve()
    if output != protocol.SCREEN_ROOT.resolve():
        raise ValueError("output root differs from frozen difficulty-frontier root")
    if not args.execute:
        raise ValueError("formal difficulty-frontier screen requires --execute")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"screen output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    completed: list[dict] = []
    try:
        frozen = verify_frozen_protocol()
        manifest = build_checkpoint_manifest()
        protocol.CHECKPOINT_MANIFEST.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output / "PREREGISTRATION.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
        )
        before = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_BEFORE.json").write_text(
            json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
        )
        resource_log = output / "RESOURCE_WAIT.jsonl"

        for scenario in protocol.ALL_SCENARIOS:
            implementation.wait_for_resources(
                resource_log,
                allow_shared_gpu=args.allow_shared_gpu,
                max_wait_minutes=args.max_wait_minutes,
            )
            completed.append(_run_cell(scenario, output / "cells", frozen))
            if source_fingerprint() != before:
                raise RuntimeError(f"source drift after c50/{scenario}")

        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("difficulty-frontier source fingerprint drifted")
        summary = protocol.analyze_cells(completed)
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
            "[SA5-C50-FRONTIER] COMPLETE next="
            f"{summary['recommended_next_action']} pilot_started=false "
            "sa6_started=false",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_v3_c50_difficulty_frontier_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [row.get("scenario") for row in completed],
            "expected_cells": len(protocol.SCENARIO_SPECS),
            "pilot_authorized": False,
            "pilot_started": False,
            "sa6_started": False,
            "recorded_at": datetime.now().astimezone().isoformat(),
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[SA5-C50-FRONTIER] STOPPED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
