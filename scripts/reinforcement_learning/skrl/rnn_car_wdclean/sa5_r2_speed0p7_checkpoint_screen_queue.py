"""Run the frozen 24-cell SA5-R2 fixed speed-0.7 checkpoint screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = HERE / "run_sa5_r2_speed0p7_checkpoint_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_r2_speed0p7_checkpoint_screen_v3.json"
sys.path.insert(0, str(HERE))

import sa5_r2_checkpoint_screen_queue as resource_guard  # noqa: E402
import sa5_r2_speed0p7_checkpoint_screen as protocol  # noqa: E402


FINGERPRINTED_SOURCES = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    HERE / "sa5_r2_checkpoint_screen.py",
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "sa5_r2_checkpoint_screen_queue.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/vehicle_speed_rate.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
)
MEASUREMENT_SOURCES = FINGERPRINTED_SOURCES[4:]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_protocol() -> dict:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("protocol_sha256") != protocol.screen_protocol()["sha256"]:
        raise RuntimeError("runtime protocol differs from frozen SHA-256")
    expected = {
        candidate["name"]: candidate["sha256"]
        for candidate in protocol.CANDIDATES
    }
    if frozen.get("checkpoint_sha256") != expected:
        raise RuntimeError("frozen checkpoint identities differ from protocol")
    return frozen


def verify_checkpoints() -> dict:
    observed = {}
    for candidate in protocol.CANDIDATES:
        path = protocol.checkpoint_path(candidate).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_of(path)
        if digest != candidate["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {candidate['name']}")
        observed[candidate["name"]] = {
            "path": str(path),
            "sha256": digest,
            "role": candidate["role"],
        }
    return observed


def source_fingerprint() -> dict:
    missing = [str(path) for path in FINGERPRINTED_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    files = {
        str(path.relative_to(REPO)): sha256_of(path)
        for path in FINGERPRINTED_SOURCES
    }
    checkpoints = verify_checkpoints()
    canonical = json.dumps(
        {"files": files, "checkpoints": checkpoints},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": "sa5_r2_speed0p7_checkpoint_screen_fingerprint/v3",
        "files": files,
        "checkpoints": checkpoints,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def planned_cells() -> list[dict]:
    return [
        {
            "checkpoint_name": candidate["name"],
            "checkpoint": protocol.checkpoint_path(candidate).resolve(),
            "checkpoint_sha256": candidate["sha256"],
            "scenario": scenario,
            "steps": protocol.STEPS_BY_SCENARIO[scenario],
            "step_tiers": protocol.STEP_TIERS_BY_SCENARIO[scenario],
        }
        for candidate in protocol.CANDIDATES
        for scenario in protocol.SCENARIOS
    ]


def wait_for_gpu(
    wait_log: Path,
    expected_fingerprint: dict,
    *,
    poll_seconds: int,
    max_wait_minutes: int,
    allow_shared_gpu: bool,
) -> None:
    started = time.monotonic()
    ready_polls = 0
    while True:
        snapshot = resource_guard.resource_snapshot(
            allow_shared_gpu=allow_shared_gpu
        )
        with wait_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        print(
            f"[SA5-R2-SPEED0P7-WAIT] ready={snapshot['ready']} "
            f"free={snapshot['gpu_free_mib']}MiB "
            f"util={snapshot['gpu_util_percent']}%",
            flush=True,
        )
        if source_fingerprint() != expected_fingerprint:
            raise RuntimeError("measurement source drifted while waiting")
        if snapshot["ready"]:
            ready_polls += 1
            if ready_polls >= resource_guard.READY_POLLS_REQUIRED:
                return
        else:
            ready_polls = 0
        if max_wait_minutes > 0 and (
            time.monotonic() - started >= max_wait_minutes * 60
        ):
            raise TimeoutError("timed out waiting for GPU")
        time.sleep(poll_seconds)


def run_cell(cell: dict, output: Path, protocol_sha256: str) -> dict:
    base_dir = output / "cells" / cell["checkpoint_name"] / cell["scenario"]
    for tier_index, steps in enumerate(cell["step_tiers"]):
        cell_dir = base_dir / f"attempt_{tier_index}_steps_{steps}"
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
            str(steps),
            "--step-tier-index",
            str(tier_index),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(command, cwd=REPO, env=env, check=False)
        matches = list(cell_dir.glob("*_cell.json"))
        if result.returncode not in (0, 3) or len(matches) != 1:
            raise RuntimeError(
                f"cell failed rc={result.returncode} json={len(matches)}: "
                f"{cell['checkpoint_name']}/{cell['scenario']} tier={tier_index}"
            )
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        protocol.validate_cell(payload, require_min_episodes=False)
        if payload["cell_valid"]:
            protocol.validate_cell(payload)
            return payload
        if result.returncode != 3:
            raise RuntimeError("insufficient cell did not return retry status 3")
        print(
            f"[SA5-R2-SPEED0P7] RETRY {cell['checkpoint_name']} "
            f"{cell['scenario']} n={payload['metrics']['n']} next_tier={tier_index + 1}",
            flush=True,
        )
    raise RuntimeError(
        f"all frozen step tiers exhausted below n=1000: "
        f"{cell['checkpoint_name']}/{cell['scenario']}"
    )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_resume_cells(resume_root: Path) -> tuple[list[dict], dict]:
    root = resume_root.expanduser().resolve()
    incomplete_path = root / "INCOMPLETE_NO_VERDICT.json"
    prereg_path = root / "PREREGISTRATION.json"
    fingerprint_path = root / "SOURCE_FINGERPRINT_BEFORE.json"
    for required in (incomplete_path, prereg_path, fingerprint_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    old_fingerprint = json.loads(
        fingerprint_path.read_text(encoding="utf-8")
    )
    if incomplete.get("status") != "INCOMPLETE_NO_VERDICT":
        raise RuntimeError("resume root is not a fail-closed incomplete screen")
    origin_protocol_sha256 = prereg.get("protocol", {}).get("sha256")
    accepted_protocols = {
        protocol.V2_PROTOCOL_SHA256,
        protocol.screen_protocol()["sha256"],
    }
    if origin_protocol_sha256 not in accepted_protocols:
        raise RuntimeError("resume root has an unsupported protocol")

    drifted = []
    for path in MEASUREMENT_SOURCES:
        relative = str(path.relative_to(REPO))
        if old_fingerprint.get("files", {}).get(relative) != sha256_of(path):
            drifted.append(relative)
    if drifted:
        raise RuntimeError(
            f"measurement sources drifted since v2 cells were produced: {drifted}"
        )

    migrated = []
    origins = []
    seen = set()
    if origin_protocol_sha256 == protocol.V2_PROTOCOL_SHA256:
        cell_paths = sorted((root / "cells").glob("*/*/*_cell.json"))
    else:
        cell_paths = sorted((root / "imported_cells").glob("*/*/cell.json"))
        cell_paths.extend(
            sorted((root / "cells").glob("*/*/attempt_*/*_cell.json"))
        )
    for cell_path in cell_paths:
        original = json.loads(cell_path.read_text(encoding="utf-8"))
        if not original.get("cell_valid"):
            continue
        key = (original.get("checkpoint_name"), original.get("scenario"))
        if key in seen:
            raise RuntimeError(f"duplicate v2 resume cell {key}")
        seen.add(key)
        cell_sha = sha256_of(cell_path)
        if origin_protocol_sha256 == protocol.V2_PROTOCOL_SHA256:
            wrapped = protocol.migrate_v2_cell(
                original,
                origin_path=str(cell_path),
                origin_cell_sha256=cell_sha,
                origin_source_fingerprint_sha256=old_fingerprint["sha256"],
            )
        else:
            protocol.validate_cell(original)
            wrapped = dict(original)
            wrapped["resume_provenance"] = {
                "mode": "resumed_v3_valid_cell",
                "origin_path": str(cell_path),
                "origin_cell_sha256": cell_sha,
                "origin_source_fingerprint_sha256": old_fingerprint["sha256"],
            }
            protocol.validate_cell(wrapped)
        migrated.append(wrapped)
        origins.append(
            {
                "checkpoint_name": key[0],
                "scenario": key[1],
                "origin_path": str(cell_path),
                "origin_cell_sha256": cell_sha,
                "steps": original["steps"],
                "episodes": original["metrics"]["n"],
            }
        )
    if not migrated:
        raise RuntimeError("resume root contains no valid v2 cells")
    return migrated, {
        "schema": "sa5_r2_speed0p7_checkpoint_screen_resume/v2",
        "resume_root": str(root),
        "origin_protocol_sha256": origin_protocol_sha256,
        "origin_source_fingerprint_sha256": old_fingerprint["sha256"],
        "measurement_sources_stable": True,
        "imported_count": len(migrated),
        "origins": origins,
    }


def render_markdown(summary: dict) -> str:
    by_name = {row["checkpoint_name"]: row for row in summary["rows"]}
    lines = [
        "# SA5-R2 fixed speed-0.7 checkpoint screen",
        "",
        f"Status: `{summary['status']}`",
        f"Decision: `{summary['decision']}`",
        f"Reason: {summary['decision_reason']}",
        (
            "Preregistered rule outcome (audit only): `"
            f"{summary['preregistered_rule_outcome']['decision']}`"
        ),
        "",
        "| rank | checkpoint | lateral CR | longitudinal CR | random2d CR | mixed CR | worst CR | mean CR | retention |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, name in enumerate(summary["ranked"], start=1):
        row = by_name[name]
        cr = row["family_cr"]
        lines.append(
            f"| {rank} | {name} | {cr['corridor_lateral']:.2%} | "
            f"{cr['corridor_longitudinal']:.2%} | "
            f"{cr['corridor_random2d']:.2%} | "
            f"{cr['corridor_mixed']:.2%} | {row['worst_family_cr']:.2%} | "
            f"{row['mean_family_cr']:.2%} | "
            f"{'PASS' if row['retention_acceptable'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "| checkpoint | native SR | native CR | narrow SR | narrow CR | crossing | direct |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in summary["ranked"]:
        row = by_name[name]
        native = row["native"]
        narrow = row["narrow"]
        lines.append(
            f"| {name} | {native['sr']:.2%} | {native['cr']:.2%} | "
            f"{narrow['sr']:.2%} | {narrow['cr']:.2%} | "
            f"{narrow['crossing_rate']:.2%} | "
            f"{narrow['direct_crossing_rate']:.2%} |"
        )
    comparisons = summary["posthoc_standardized_comparisons"]["comparisons"]
    lines.extend(
        [
            "",
            "## Standardized evidence interpretation",
            "",
            (
                "Uses a conservative independent-binomial SE with n=1,200 "
                "per arm. These are descriptive single-seed comparisons, not "
                "multi-seed hypothesis tests."
            ),
            "",
            "| comparison | delta CR | SE | standardized difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison in comparisons.values():
        lines.append(
            f"| {comparison['left']} vs {comparison['right']} | "
            f"{comparison['delta_cr']:+.2%} | "
            f"{comparison['standard_error']:.2%} | "
            f"{comparison['standardized_difference_se']:+.2f} SE |"
        )
    lines.extend(
        [
            "",
            f"Engineering action: `{summary['engineering_action']}`",
            "",
            "No parent was accepted. No training or SA6 was started.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-minutes", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")

    frozen = verify_frozen_protocol()
    checkpoints = verify_checkpoints()
    cells = planned_cells()
    resumed_payloads = []
    resume_manifest = None
    if args.resume_from is not None:
        resumed_payloads, resume_manifest = load_resume_cells(args.resume_from)
    print(
        f"[SA5-R2-SPEED0P7] {len(cells)} cells = "
        f"{len(protocol.CANDIDATES)} checkpoints x {len(protocol.SCENARIOS)} scenarios",
        flush=True,
    )
    for cell in cells:
        print(
            f"  {cell['checkpoint_name']:14s} {cell['scenario']:24s} "
            f"step_tiers={list(cell['step_tiers'])}",
            flush=True,
        )
    if resume_manifest is not None:
        print(
            f"[SA5-R2-SPEED0P7] verified/importing "
            f"{resume_manifest['imported_count']} valid prior cells",
            flush=True,
        )
    if not args.execute:
        print("[SA5-R2-SPEED0P7] dry run; pass --execute to run")
        return 0

    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    before = source_fingerprint()
    write_json(output / "SOURCE_FINGERPRINT_BEFORE.json", before)
    write_json(
        output / "PREREGISTRATION.json",
        {
            "status": "PREREGISTERED_BEFORE_ROLLOUT",
            "created_at": now_iso(),
            "protocol": protocol.screen_protocol(),
            "freeze": frozen,
            "checkpoints": checkpoints,
            "source_fingerprint": before,
            "resume": resume_manifest,
            "resource_policy": {
                "mode": (
                    "shared_memory_guarded"
                    if args.allow_shared_gpu
                    else "exclusive"
                ),
                "shared_min_free_gpu_mib": (
                    resource_guard.SHARED_MIN_FREE_GPU_MIB
                    if args.allow_shared_gpu
                    else None
                ),
            },
            "cells": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in cell.items()
                }
                for cell in cells
            ],
        },
    )

    payloads = list(resumed_payloads)
    imported_keys = {
        (payload["checkpoint_name"], payload["scenario"])
        for payload in resumed_payloads
    }
    for payload in resumed_payloads:
        write_json(
            output
            / "imported_cells"
            / payload["checkpoint_name"]
            / payload["scenario"]
            / "cell.json",
            payload,
        )
    if resume_manifest is not None:
        write_json(output / "IMPORTED_CELLS.json", resume_manifest)
    try:
        if args.wait_for_gpu:
            wait_for_gpu(
                output / "resource_wait.jsonl",
                before,
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
                allow_shared_gpu=args.allow_shared_gpu,
            )
        elif not resource_guard.resource_snapshot(
            allow_shared_gpu=args.allow_shared_gpu
        )["ready"]:
            raise RuntimeError("GPU resources are not ready")

        for index, cell in enumerate(cells, start=1):
            key = (cell["checkpoint_name"], cell["scenario"])
            if key in imported_keys:
                print(
                    f"[SA5-R2-SPEED0P7] cell {index}/{len(cells)} "
                    f"{cell['checkpoint_name']} {cell['scenario']} IMPORTED",
                    flush=True,
                )
                continue
            snapshot = resource_guard.resource_snapshot(
                allow_shared_gpu=args.allow_shared_gpu
            )
            if not snapshot["ready"]:
                wait_for_gpu(
                    output / "resource_wait.jsonl",
                    before,
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    allow_shared_gpu=args.allow_shared_gpu,
                )
            print(
                f"[SA5-R2-SPEED0P7] cell {index}/{len(cells)} "
                f"{cell['checkpoint_name']} {cell['scenario']}",
                flush=True,
            )
            payloads.append(run_cell(cell, output, protocol.screen_protocol()["sha256"]))
            if source_fingerprint() != before:
                raise RuntimeError(f"source fingerprint drift after cell {index}")

        summary = protocol.summarize(payloads)
        after = source_fingerprint()
        if after != before:
            raise RuntimeError("source fingerprint changed before verdict")
        summary.update(
            {
                "completed_at": now_iso(),
                "source_fingerprint_stable": True,
                "source_fingerprint_sha256": before["sha256"],
                "imported_cell_count": len(resumed_payloads),
                "fresh_cell_count": len(payloads) - len(resumed_payloads),
            }
        )
        write_json(output / "SOURCE_FINGERPRINT_AFTER.json", after)
        write_json(output / "SUMMARY.json", summary)
        (output / "SUMMARY.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(
            f"[SA5-R2-SPEED0P7] COMPLETE decision={summary['decision']} "
            "accepted_parent=False training_started=False sa6_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        write_json(
            output / "INCOMPLETE_NO_VERDICT.json",
            {
                "schema": "sa5_r2_speed0p7_checkpoint_screen_incomplete/v3",
                "status": "INCOMPLETE_NO_VERDICT",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cells_completed": len(payloads),
                "cells_planned": len(cells),
                "imported_cells": len(resumed_payloads),
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
