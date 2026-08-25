"""Run the exact SA4-R3 it125 parent on four sealed Stage-5 corridor families."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa5_checkpoint_screen_cell as screen_cell  # noqa: E402
import sa5_checkpoint_screen as screen  # noqa: E402
import sa5_checkpoint_screen_queue as screen_queue  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


PARENT_NAME = "sa4r3_it125"
PARENT_CONCEPTUAL_ITERATION = 125
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/sa4_r3_cont25_from_it100_ne1024_s42_p25_r1"
    / "checkpoint_3200.pt"
)
PARENT_SHA256 = (
    "57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d"
)
FREEZE = REPO / "docs/freeze/sa5_r2_parent_sealed_baseline_v1.json"
SCENARIOS = tuple(screen.CORRIDOR_SCENARIOS)
STEPS = 2500

FINGERPRINTED_SOURCES = tuple(
    dict.fromkeys(
        (
            *screen_queue.FINGERPRINTED_SOURCES,
            Path(__file__).resolve(),
            FREEZE,
            REPO
            / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
            / "e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12.py",
        )
    )
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def source_fingerprint() -> dict[str, str]:
    missing = [str(path) for path in FINGERPRINTED_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"baseline sources missing: {missing}")
    return {
        str(path.relative_to(REPO)): sha256_of(path)
        for path in FINGERPRINTED_SOURCES
    }


def baseline_protocol() -> dict:
    screen_frozen = screen.screen_protocol()
    payload = {
        "schema": "sa5_r2_parent_sealed_baseline/v1",
        "status": "AUTHORIZED_NOT_STARTED",
        "purpose": (
            "Measure the exact SA4-R3 conceptual-it125 parent before any "
            "SA5-R2 sealed corrective update."
        ),
        "parent": {
            "name": PARENT_NAME,
            "conceptual_iteration": PARENT_CONCEPTUAL_ITERATION,
            "checkpoint": str(PARENT_CHECKPOINT),
            "sha256": PARENT_SHA256,
        },
        "fixed_evaluation": {
            "screen_protocol_sha256": screen_frozen["sha256"],
            "stage": screen.STAGE,
            "seed": screen.SEED,
            "num_envs": screen.NUM_ENVS,
            "steps": STEPS,
            "scenarios": list(SCENARIOS),
            "actuator_delay_steps": screen.DELAY_STEPS,
            "actuator_delay_ms": screen.DELAY_MS,
            "actuator_profile": screen.ACTUATOR_PROFILE,
            "lidar_noise_mode": screen.LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": (
                screen.LIDAR_DISTRACTOR_ELIGIBILITY
            ),
            "scene": screen_frozen["fixed_evaluation"]["scene"],
            "corridor_geometry_contract": screen_frozen["fixed_evaluation"][
                "corridor_geometry_contract"
            ],
        },
        "required_metrics": [
            "sr",
            "cr",
            "to",
            "wall_cr",
            "obstacle_cr",
        ],
        "evidence_boundary": (
            "single evaluator seed and diagnostic baseline only; this run "
            "cannot accept or reject a parent and cannot launch training or SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_frozen_protocol() -> dict:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = baseline_protocol()
    if frozen != current:
        raise RuntimeError("SA5-R2 parent baseline differs from frozen JSON")
    return frozen


def verify_parent() -> dict:
    checkpoint = PARENT_CHECKPOINT.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = sha256_of(checkpoint)
    if digest != PARENT_SHA256:
        raise RuntimeError(
            f"parent hash mismatch: expected {PARENT_SHA256}, got {digest}"
        )
    return {
        "name": PARENT_NAME,
        "conceptual_iteration": PARENT_CONCEPTUAL_ITERATION,
        "checkpoint": str(checkpoint),
        "sha256": digest,
    }


def run_cell(scenario: str, output_root: Path, protocol_sha256: str) -> dict:
    output = output_root / scenario
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario}_g5_d{screen.DELAY_STEPS}_s{screen.SEED}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    d9_runner._require_new_targets([log_path, corridor_path, cell_path])

    print(
        f"[SA5-R2-PARENT] scenario={scenario} steps={STEPS} "
        f"parent_sha={PARENT_SHA256[:12]}",
        flush=True,
    )
    _run_play(
        PARENT_CHECKPOINT,
        log_path,
        num_envs=screen.NUM_ENVS,
        steps=STEPS,
        extra=screen_cell.build_scene_args(scenario, corridor_path),
    )
    d9_runner._validate_runtime_log(
        log_path, screen.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, screen.DELAY_STEPS, screen.ACTUATOR_PROFILE
    )
    values = screen_cell.stage_values()
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError(f"no play outcome summary in {log_path}")
    report = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = screen_cell.verify_corridor_runtime(
        report, scenario, values
    )
    metrics = base.reconcile_corridor_metrics(dict(summary), report)
    for key in ("sr", "cr", "to"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite {scenario} {key}")
    if int(metrics["n"]) < screen.MIN_EPISODES:
        raise RuntimeError(
            f"{scenario} has {metrics['n']} episodes, need {screen.MIN_EPISODES}"
        )
    verdict = screen.evaluate_metrics(scenario, metrics)
    cell = {
        "schema": "sa5_r2_parent_sealed_baseline_cell/v1",
        "cell_valid": True,
        "checkpoint_name": PARENT_NAME,
        "conceptual_iteration": PARENT_CONCEPTUAL_ITERATION,
        "checkpoint": str(PARENT_CHECKPOINT.resolve()),
        "checkpoint_sha256": PARENT_SHA256,
        "baseline_protocol_sha256": protocol_sha256,
        "source_screen_protocol_sha256": screen.screen_protocol()["sha256"],
        "geometry_stage": screen.STAGE,
        "scenario": scenario,
        "seed": screen.SEED,
        "delay_steps": screen.DELAY_STEPS,
        "delay_ms": screen.DELAY_MS,
        "num_envs": screen.NUM_ENVS,
        "steps": STEPS,
        "lidar_noise_mode": screen.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": (
            screen.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "actuator_eval": fixed_actuator_metadata(
            screen.DELAY_STEPS, screen.ACTUATOR_PROFILE
        ),
        "metrics": metrics,
        "corridor_report": report,
        "log": str(log_path),
        **verdict,
    }
    _write_json(cell_path, cell)
    print(
        f"[SA5-R2-PARENT] complete {scenario}: n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"wall={report['wall_collision_rate']:.4f} "
        f"obstacle={report['obstacle_collision_rate']:.4f}",
        flush=True,
    )
    return cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-wait-minutes", type=int, default=0)
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args(argv)

    if args.print_protocol:
        print(json.dumps(baseline_protocol(), indent=2, sort_keys=True))
        return 0
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")

    frozen = verify_frozen_protocol()
    parent = verify_parent()
    print(
        f"[SA5-R2-PARENT] 4 sealed cells; parent={PARENT_NAME} "
        f"seed={screen.SEED} d{screen.DELAY_STEPS} steps={STEPS}",
        flush=True,
    )
    if not args.execute:
        print("[SA5-R2-PARENT] dry run; pass --execute to run", flush=True)
        return 0

    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    before = source_fingerprint()
    screen_before = screen_queue.source_fingerprint()
    _write_json(output / "source_fingerprint_before.json", before)
    _write_json(
        output / "PREREGISTRATION.json",
        {
            "status": "PREREGISTERED_BEFORE_ROLLOUT",
            "created_at": now_iso(),
            "protocol": frozen,
            "parent": parent,
            "source_fingerprint": before,
            "cells": [
                {"scenario": scenario, "steps": STEPS}
                for scenario in SCENARIOS
            ],
            "accepted_parent": False,
            "training_started": False,
            "sa6_started": False,
        },
    )

    cells: list[dict] = []
    try:
        if args.wait_for_gpu:
            screen_queue.wait_for_resources(
                output / "resource_wait.jsonl",
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
                expected_fingerprint=screen_before,
                allow_shared_gpu=args.allow_shared_gpu,
            )
        else:
            snapshot = screen_queue.resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                raise RuntimeError(f"GPU resources not ready: {snapshot}")

        for index, scenario in enumerate(SCENARIOS, start=1):
            snapshot = screen_queue.resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                screen_queue.wait_for_resources(
                    output / "resource_wait.jsonl",
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    expected_fingerprint=screen_before,
                    allow_shared_gpu=args.allow_shared_gpu,
                )
            print(
                f"[SA5-R2-PARENT] cell {index}/{len(SCENARIOS)} "
                f"{scenario}",
                flush=True,
            )
            cells.append(run_cell(scenario, output / "cells", frozen["sha256"]))
            if source_fingerprint() != before:
                raise RuntimeError(
                    f"measurement source drift after parent cell {index}"
                )
            verify_parent()

        after = source_fingerprint()
        if after != before:
            raise RuntimeError("measurement source fingerprint changed")
        _write_json(output / "source_fingerprint_after.json", after)
        result_rows = [
            {
                "scenario": cell["scenario"],
                "n": cell["metrics"]["n"],
                "sr": cell["metrics"]["sr"],
                "cr": cell["metrics"]["cr"],
                "to": cell["metrics"]["to"],
                "wall_cr": cell["corridor_report"]["wall_collision_rate"],
                "obstacle_cr": cell["corridor_report"][
                    "obstacle_collision_rate"
                ],
                "threshold_pass": cell["threshold_pass"],
            }
            for cell in cells
        ]
        _write_json(
            output / "SUMMARY.json",
            {
                "schema": "sa5_r2_parent_sealed_baseline_bundle/v1",
                "status": "COMPLETE_VALID_DIAGNOSTIC_BASELINE",
                "completed_at": now_iso(),
                "protocol": frozen,
                "parent": parent,
                "cells": result_rows,
                "source_fingerprint_stable": True,
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        print(
            "[SA5-R2-PARENT] COMPLETE_VALID_DIAGNOSTIC_BASELINE",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json(
            output / "INCOMPLETE_NO_BASELINE.json",
            {
                "schema": "sa5_r2_parent_sealed_baseline_incomplete/v1",
                "status": "INCOMPLETE_NO_BASELINE",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cells_completed": len(cells),
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
