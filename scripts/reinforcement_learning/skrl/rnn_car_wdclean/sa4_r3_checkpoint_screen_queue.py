"""Run and compare the six frozen SA4-R3 checkpoint-screen cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_checkpoint_screen as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import sa4_r3_checkpoint_screen as protocol  # noqa: E402


RUN_NAME = "sa4_r3_validreturn_from_sa3r1_c100_ne1024_s42_p50_r1"
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME
RUNNER = HERE / "run_sa4_r3_checkpoint_screen.py"
FINGERPRINTED_SOURCES = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    Path(base.__file__).resolve(),
    Path(base.base.__file__).resolve(),
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
    / "sa3_sa8_acceptance_contract.py",
)


def source_fingerprint() -> dict[str, str]:
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FINGERPRINTED_SOURCES
    }


def checkpoints() -> list[dict]:
    result = []
    for name, filename, iteration in protocol.CHECKPOINTS:
        path = RUN_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(
            {
                "name": name,
                "path": path,
                "iteration": iteration,
                "sha256": base.base.sha256_of(path),
            }
        )
    return result


def cells(checkpoint_rows: list[dict]) -> list[dict]:
    return [
        {
            "checkpoint_name": checkpoint["name"],
            "checkpoint": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "scenario": scenario,
            "steps": protocol.STEPS_BY_SCENARIO[scenario],
        }
        for checkpoint in checkpoint_rows
        for scenario in protocol.SCENARIOS
    ]


def run_cell(cell: dict, output_root: Path) -> Path:
    cell_dir = output_root / f"{cell['checkpoint_name']}__{cell['scenario']}"
    command = [
        str(sys.executable),
        str(RUNNER),
        str(cell["checkpoint"]),
        "--output-dir",
        str(cell_dir),
        "--checkpoint-name",
        cell["checkpoint_name"],
        "--scenario",
        cell["scenario"],
        "--expect-checkpoint-sha256",
        cell["sha256"],
        "--steps",
        str(cell["steps"]),
    ]
    subprocess.run(command, check=False, cwd=REPO)
    stem = (
        f"{cell['scenario']}_g4_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    )
    return cell_dir / f"{stem}_cell.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    checkpoint_rows = checkpoints()
    planned = cells(checkpoint_rows)
    print(
        f"[SA4-R3-SCREEN] {len(planned)} cells, seed={protocol.SEED}, "
        f"d{protocol.DELAY_STEPS}, eligibility={protocol.ELIGIBILITY}"
    )
    for cell in planned:
        print(
            f"  {cell['checkpoint_name']:12s} {cell['scenario']:22s} "
            f"steps={cell['steps']} sha={cell['sha256'][:12]}"
        )
    if not args.execute:
        return 0
    busy = d9_runner.gpu_is_busy()
    if busy:
        raise RuntimeError(f"SA4-R3 screen refuses to share a busy GPU: {busy}")

    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common_paths = {
        "protocol": output / "PROTOCOL.json",
        "summary": output / "sa4_r3_checkpoint_screen_summary.json",
        "incomplete": output / "SA4_R3_SCREEN_INCOMPLETE.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    d9_runner._require_new_targets(list(common_paths.values()))
    before = source_fingerprint()
    preregistration = {
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "protocol": protocol.screen_protocol(),
        "checkpoints": [
            {
                "name": row["name"],
                "path": str(row["path"]),
                "iteration": row["iteration"],
                "sha256": row["sha256"],
            }
            for row in checkpoint_rows
        ],
        "source_fingerprint": before,
    }
    common_paths["protocol"].write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    common_paths["source_before"].write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    payloads = []
    try:
        for index, cell in enumerate(planned, start=1):
            print(
                f"[SA4-R3-SCREEN] cell {index}/{len(planned)} "
                f"{cell['checkpoint_name']} {cell['scenario']}",
                flush=True,
            )
            cell_path = run_cell(cell, output)
            if not cell_path.is_file():
                raise RuntimeError(f"cell did not produce JSON: {cell_path}")
            if source_fingerprint() != before:
                raise RuntimeError(f"source drift after cell {index}")
            payloads.append(json.loads(cell_path.read_text(encoding="utf-8")))

        after = source_fingerprint()
        common_paths["source_after"].write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        comparison = protocol.compare_checkpoints(payloads)
        summary = {
            "schema": "sa4_r3_checkpoint_screen_bundle/v1",
            "status": "COMPLETE_VALID_SCREEN",
            "cells_planned": len(planned),
            "cells_completed": len(payloads),
            "source_fingerprint_stable": after == before,
            "protocol": protocol.screen_protocol(),
            "checkpoints": preregistration["checkpoints"],
            "cells": payloads,
            "comparison": comparison,
            "training_extension_started": False,
            "sa5_started": False,
        }
        common_paths["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "[SA4-R3-SCREEN] COMPLETE "
            f"selected={comparison['selected_checkpoint'] or '<none>'} "
            "extension_or_sa5_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        common_paths["incomplete"].write_text(
            json.dumps(
                {
                    "schema": "sa4_r3_checkpoint_screen_incomplete/v1",
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "cells_completed": len(payloads),
                    "protocol_sha256": protocol.screen_protocol()["sha256"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

