"""Run the frozen eight-cell SA4-v3 c550/c600 retention screen."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = HERE / "run_sa4_v3_c550_c600_retention_cell.py"
FREEZE = REPO / "docs/freeze/sa4_v3_c550_c600_retention_v1.json"
sys.path.insert(0, str(HERE))

import sa4_v3_c550_c600_retention as protocol  # noqa: E402
import sa5_r2_checkpoint_screen_queue as resource_guard  # noqa: E402


SOURCE_PATHS = (
    HERE / "sa4_v3_c550_c600_retention.py",
    RUNNER,
    Path(__file__).resolve(),
    FREEZE,
    protocol.GATE_SUMMARY,
    protocol.GATE_FREEZE,
    HERE / "sa4_v3_c300_c600_gate.py",
    HERE / "sa4_v3_checkpoint_screen.py",
    HERE / "run_sa4_v3_checkpoint_screen_cell.py",
    HERE / "run_sa3_v3_retention_screen_cell.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "run_sa5_joint_retention_gates.py",
    REPO / "docs/freeze/sim2real_speed_density_curriculum_v3_20260821.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_speed_density_curriculum_v3.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/curriculum/phases/e2e_final20_v1.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/corridor_density.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/observations/obs_functions.py",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> dict:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    files = {
        str(path.relative_to(REPO)): sha256_of(path) for path in SOURCE_PATHS
    }
    for candidate in protocol.CANDIDATES:
        checkpoint = protocol.checkpoint_path(candidate).resolve()
        files[str(checkpoint.relative_to(REPO))] = sha256_of(checkpoint)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sa4_v3_c550_c600_retention_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA4-v3 c550/c600 retention screen",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['protocol_sha256']}`",
        f"Source Gate summary: `{summary['gate_summary_sha256']}`",
        "",
        "| checkpoint | scenario | n | SR | CR | TO | extra | gate |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for candidate in summary["candidate_results"]:
        for scenario in protocol.SCENARIOS:
            row = candidate["cells"][scenario["name"]]
            extra = ""
            if scenario["kind"] == "narrow":
                extra = (
                    f"cross={float(row['crossing_rate']):.2%}, "
                    f"direct={float(row['direct_crossing_rate']):.2%}"
                )
            lines.append(
                f"| {candidate['checkpoint_name']} | {scenario['display']} | "
                f"{row['n']} | {row['sr']:.2%} | {row['cr']:.2%} | "
                f"{row['to']:.2%} | {extra} | "
                f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
            )
    lines.extend(["", "## Candidate result", ""])
    for candidate in summary["candidate_results"]:
        lines.append(
            f"- `{candidate['checkpoint_name']}`: retention "
            f"{'PASS' if candidate['retention_pass'] else 'FAIL'}"
        )
    lines.extend(
        [
            "",
            "Recommended exposure parent candidate: "
            f"`{summary['recommended_exposure_parent_candidate']}`",
            "",
            "c550 decision: "
            f"`{summary['c550_exposure_parent_decision']}`",
            "",
            "SA4 graduation remains `False` because the source Gate failed.",
            "No parent was accepted, and neither exposure training nor SA5 was started.",
            "",
            "## Evidence boundary",
            "",
            summary["interpretation_limit"],
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen = protocol.retention_protocol()
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    if json.loads(FREEZE.read_text(encoding="utf-8")) != frozen:
        raise RuntimeError("frozen retention protocol differs from runtime protocol")

    if not protocol.GATE_SUMMARY.is_file():
        raise FileNotFoundError(protocol.GATE_SUMMARY)
    if sha256_of(protocol.GATE_SUMMARY) != protocol.GATE_SUMMARY_SHA256:
        raise RuntimeError("source Gate summary content hash mismatch")
    if sha256_of(protocol.GATE_FREEZE) != protocol.GATE_FREEZE_SHA256:
        raise RuntimeError("source Gate freeze content hash mismatch")
    gate = json.loads(protocol.GATE_SUMMARY.read_text(encoding="utf-8"))
    protocol.validate_gate_summary(gate)

    for candidate in protocol.CANDIDATES:
        checkpoint = protocol.checkpoint_path(candidate)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_of(checkpoint) != candidate["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {candidate['name']}")
    snapshot = resource_guard.resource_snapshot(allow_shared_gpu=False)
    if not snapshot["ready"]:
        raise RuntimeError(f"GPU resources are not ready: {snapshot}")

    (output / "PREREGISTRATION.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "SOURCE_GATE_SUMMARY.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    before = source_fingerprint()
    (output / "SOURCE_FINGERPRINT_BEFORE.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    completed: list[dict] = []
    try:
        for spec in protocol.cells():
            candidate = protocol.candidate_by_name(spec["checkpoint_name"])
            label = spec["cell"]
            cell_dir = output / label
            command = [
                str(PYTHON),
                str(RUNNER),
                str(protocol.checkpoint_path(candidate)),
                "--output-dir",
                str(cell_dir),
                "--checkpoint-name",
                candidate["name"],
                "--scenario",
                spec["scenario"],
                "--expect-checkpoint-sha256",
                candidate["sha256"],
                "--expect-protocol-sha256",
                frozen["sha256"],
            ]
            print(f"[SA4-V3-RETENTION-QUEUE] START {label}", flush=True)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(command, cwd=REPO, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{label} runner exited {result.returncode}")
            matches = list(cell_dir.glob("*_cell.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"{label} produced {len(matches)} cell JSON files"
                )
            cell = json.loads(matches[0].read_text(encoding="utf-8"))
            protocol.validate_cell(cell)
            completed.append(cell)
            if source_fingerprint() != before:
                raise RuntimeError(f"source fingerprint drift after {label}")
            print(
                f"[SA4-V3-RETENTION-QUEUE] DONE {label} "
                f"n={cell['metrics']['n']} CR={cell['metrics']['cr']:.4f} "
                f"gate={'PASS' if cell['gate']['pass'] else 'FAIL'}",
                flush=True,
            )

        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("source fingerprint drifted during retention")
        summary = protocol.summarize_cells(completed)
        summary.update(
            {
                "completed_at": datetime.now().astimezone().isoformat(),
                "source_fingerprint_stable": True,
                "source_fingerprint_sha256": before["sha256"],
            }
        )
        (output / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output / "SUMMARY.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(
            "[SA4-V3-RETENTION-QUEUE] COMPLETE recommendation="
            f"{summary['recommended_exposure_parent_candidate']} "
            f"c550={summary['c550_exposure_parent_decision']} "
            "accepted_parent=false exposure_started=false sa5_started=false",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa4_v3_c550_c600_retention_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [cell.get("cell") for cell in completed],
            "expected_cells": len(protocol.cells()),
            "timestamp": datetime.now().astimezone().isoformat(),
            "recommended_exposure_parent_candidate": None,
            "accepted_exposure_parent": False,
            "exposure_curriculum_started": False,
            "sa4_graduated": False,
            "sa5_started": False,
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
