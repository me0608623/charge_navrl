"""Run the frozen three-cell SA4-R3 it125 checkpoint screen."""

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
import sa4_r3_it125_checkpoint_screen as protocol  # noqa: E402


RUN_DIR = REPO / "logs/rnn_car" / protocol.RUN_NAME
CHECKPOINT = RUN_DIR / protocol.CHECKPOINT_FILENAME
RUNNER = HERE / "run_sa4_r3_it125_checkpoint_screen.py"
TRAINING_SOURCE_FILES = {
    "config": REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "e2e_sa4_r3_cont25_from_it100.py",
    "trainer": REPO
    / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py",
    "env_overrides": REPO
    / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    "lidar_observation": REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py",
    "actuator_action": REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
}
EVALUATION_SOURCE_FILES = {
    "stage4_runner": Path(base.__file__).resolve(),
    "stage_runner_base": Path(base.base.__file__).resolve(),
    "play_launcher": REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
    / "run_sa5_joint_retention_gates.py",
    "gate_parser": REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/validate_gates.py",
    "fixed_actuator": REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/fixed_actuator_eval.py",
    "play": REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    "env_overrides": TRAINING_SOURCE_FILES["env_overrides"],
    "lidar_observation": TRAINING_SOURCE_FILES["lidar_observation"],
    "actuator_action": TRAINING_SOURCE_FILES["actuator_action"],
    "acceptance_contract": REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
    / "sa3_sa8_acceptance_contract.py",
}
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
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
    / "sa3_sa8_acceptance_contract.py",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> dict[str, str]:
    return {
        str(path.relative_to(REPO)): sha256_of(path)
        for path in FINGERPRINTED_SOURCES
    }


def verify_training_source_contract() -> dict[str, str]:
    current = {name: sha256_of(path) for name, path in TRAINING_SOURCE_FILES.items()}
    if current != protocol.TRAINING_SOURCE_HASHES:
        drifted = sorted(
            name
            for name in current
            if current[name] != protocol.TRAINING_SOURCE_HASHES[name]
        )
        raise RuntimeError(f"training-source contract drifted: {drifted}")
    return current


def verify_evaluation_source_contract() -> dict[str, str]:
    current = {
        name: sha256_of(path) for name, path in EVALUATION_SOURCE_FILES.items()
    }
    if current != protocol.EVALUATION_SOURCE_HASHES:
        drifted = sorted(
            name
            for name in current
            if current[name] != protocol.EVALUATION_SOURCE_HASHES[name]
        )
        raise RuntimeError(f"evaluation-source contract drifted: {drifted}")
    return current


def planned_cells() -> list[dict]:
    return [
        {
            "checkpoint_name": protocol.CHECKPOINT_NAME,
            "checkpoint": CHECKPOINT,
            "sha256": protocol.CHECKPOINT_SHA256,
            "scenario": scenario,
            "steps": protocol.STEPS_BY_SCENARIO[scenario],
        }
        for scenario in protocol.SCENARIOS
    ]


def run_cell(cell: dict, output_root: Path) -> Path:
    cell_dir = output_root / cell["scenario"]
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
    stem = f"{cell['scenario']}_g4_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    return cell_dir / f"{stem}_cell.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    digest = sha256_of(CHECKPOINT)
    if digest != protocol.CHECKPOINT_SHA256:
        raise RuntimeError(
            f"it125 checkpoint hash mismatch: {digest} != "
            f"{protocol.CHECKPOINT_SHA256}"
        )
    training_sources = verify_training_source_contract()
    evaluation_sources = verify_evaluation_source_contract()
    cells = planned_cells()
    print(
        f"[SA4-R3-IT125-SCREEN] {len(cells)} cells, seed={protocol.SEED}, "
        f"d{protocol.DELAY_STEPS}, eligibility={protocol.ELIGIBILITY}"
    )
    for cell in cells:
        print(
            f"  {cell['scenario']:22s} steps={cell['steps']} "
            f"sha={cell['sha256'][:12]}"
        )
    if not args.execute:
        return 0

    busy = d9_runner.gpu_is_busy()
    if busy:
        raise RuntimeError(f"SA4-R3 it125 screen refuses busy GPU: {busy}")

    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol": output / "PROTOCOL.json",
        "summary": output / "sa4_r3_it125_checkpoint_screen_summary.json",
        "incomplete": output / "SA4_R3_IT125_SCREEN_INCOMPLETE.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    d9_runner._require_new_targets(list(paths.values()))
    before = source_fingerprint()
    preregistration = {
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "protocol": protocol.screen_protocol(),
        "checkpoint": {
            "name": protocol.CHECKPOINT_NAME,
            "path": str(CHECKPOINT),
            "conceptual_iteration": protocol.CONCEPTUAL_ITERATION,
            "sha256": digest,
        },
        "training_source_fingerprint": training_sources,
        "evaluation_contract_fingerprint": evaluation_sources,
        "bundle_source_fingerprint": before,
    }
    paths["protocol"].write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["source_before"].write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    payloads = []
    try:
        for index, cell in enumerate(cells, start=1):
            print(
                f"[SA4-R3-IT125-SCREEN] cell {index}/{len(cells)} "
                f"{cell['scenario']}",
                flush=True,
            )
            cell_path = run_cell(cell, output)
            if not cell_path.is_file():
                raise RuntimeError(f"cell did not produce JSON: {cell_path}")
            if source_fingerprint() != before:
                raise RuntimeError(f"evaluation source drift after cell {index}")
            if verify_training_source_contract() != training_sources:
                raise RuntimeError(f"training source drift after cell {index}")
            if verify_evaluation_source_contract() != evaluation_sources:
                raise RuntimeError(f"evaluation contract drift after cell {index}")
            payloads.append(json.loads(cell_path.read_text(encoding="utf-8")))

        after = source_fingerprint()
        paths["source_after"].write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        verdict = protocol.evaluate_candidate(payloads)
        summary = {
            "schema": "sa4_r3_it125_checkpoint_screen_bundle/v1",
            "status": "COMPLETE_VALID_SCREEN",
            "cells_planned": len(cells),
            "cells_completed": len(payloads),
            "source_fingerprint_stable": after == before,
            "protocol": protocol.screen_protocol(),
            "checkpoint": preregistration["checkpoint"],
            "cells": payloads,
            "verdict": verdict,
            "accepted_parent": False,
            "additional_training_started": False,
            "sa5_started": False,
        }
        paths["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "[SA4-R3-IT125-SCREEN] COMPLETE "
            f"screen_pass={verdict['screen_pass']} accepted_parent=False "
            "additional_training_or_sa5_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        paths["incomplete"].write_text(
            json.dumps(
                {
                    "schema": "sa4_r3_it125_checkpoint_screen_incomplete/v1",
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
