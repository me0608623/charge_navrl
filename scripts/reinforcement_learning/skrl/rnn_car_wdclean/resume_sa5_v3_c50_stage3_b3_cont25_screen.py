"""Resume the frozen B3 continuation screen at a completed-cell boundary.

This recovery only changes GPU resource scheduling. It reuses the frozen
protocol, runner, validators, checkpoints, and original source fingerprint.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_stage3_b3_cont25_screen_queue as queue  # noqa: E402


protocol = queue.protocol
base = queue.base_queue
ROOT = protocol.SCREEN_ROOT.resolve()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_cells() -> list[tuple[dict, str]]:
    return [
        (spec, scenario)
        for spec in protocol.CANDIDATE_SPECS
        for scenario in protocol.ALL_SCENARIOS
    ]


def _load_existing_prefix() -> list[dict]:
    expected = _expected_cells()
    completed: list[dict] = []
    seen_missing = False
    for spec, scenario in expected:
        cell_dir = ROOT / spec["name"] / scenario
        matches = list(cell_dir.glob("*_cell.json")) if cell_dir.is_dir() else []
        if not matches:
            seen_missing = True
            continue
        if seen_missing:
            raise RuntimeError("existing cells are not a contiguous matrix prefix")
        if len(matches) != 1:
            raise RuntimeError(
                f"{spec['name']}/{scenario} has {len(matches)} cell JSON files"
            )
        payload = _read_json(matches[0])
        protocol.validate_cell(payload)
        if (
            payload.get("checkpoint_name") != spec["name"]
            or payload.get("scenario") != scenario
        ):
            raise RuntimeError("existing cell identity differs from matrix order")
        completed.append(payload)
    return completed


def _verify_original_bundle() -> dict:
    before_path = ROOT / "SOURCE_FINGERPRINT_BEFORE.json"
    if not before_path.is_file():
        raise FileNotFoundError(before_path)
    before = _read_json(before_path)
    current = base.source_fingerprint()
    if current != before:
        raise RuntimeError("source bundle differs from the original r2 fingerprint")
    prereg = _read_json(ROOT / "PREREGISTRATION.json")
    if prereg != queue.verify_frozen_protocol():
        raise RuntimeError("r2 preregistration differs from frozen protocol")
    manifest = _read_json(ROOT / "CHECKPOINT_MANIFEST.json")
    if manifest != queue.build_checkpoint_manifest():
        raise RuntimeError("r2 checkpoint manifest differs from current files")
    return before


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    args = parser.parse_args(argv)

    before = _verify_original_bundle()
    completed = _load_existing_prefix()
    expected = _expected_cells()
    if len(completed) >= len(expected):
        raise RuntimeError("screen is already complete")

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    resume_record = {
        "schema": "sa5_v3_c50_stage3_b3_cont25_shared_resume/v1",
        "status": "READY" if not args.execute else "RUNNING",
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "source_fingerprint_sha256": before["sha256"],
        "resume_orchestrator_sha256": script_hash,
        "validated_existing_cells": len(completed),
        "remaining_cells": len(expected) - len(completed),
        "resume_boundary": (
            f"{expected[len(completed)][0]['name']}/"
            f"{expected[len(completed)][1]}"
        ),
        "resource_policy_before": "exclusive",
        "resource_policy_after": "allow_shared_gpu",
        "simulation_contract_changed": False,
        "reuses_existing_cell": True,
        "starts_training": False,
        "starts_phase_b": False,
        "selects_parent": False,
        "starts_sa6": False,
    }
    if not args.execute:
        print(json.dumps(resume_record, indent=2, sort_keys=True))
        return 0
    _write_json(ROOT / "SHARED_GPU_RESUME_PROTOCOL.json", resume_record)

    resource_log = ROOT / "RESOURCE_WAIT.jsonl"
    try:
        for spec, scenario in expected[len(completed) :]:
            base.wait_for_resources(
                resource_log,
                allow_shared_gpu=True,
                max_wait_minutes=args.max_wait_minutes,
            )
            completed.append(base._run_cell(spec, scenario, ROOT, prereg := protocol.screen_protocol()))
            if base.source_fingerprint() != before:
                raise RuntimeError(
                    f"source drift after {spec['name']}/{scenario}"
                )

        after = base.source_fingerprint()
        if after != before:
            raise RuntimeError("source fingerprint drifted during resumed screen")
        _write_json(ROOT / "SOURCE_FINGERPRINT_AFTER.json", after)
        summary = protocol.final_verdict(completed)
        summary.update(
            {
                "completed_at": datetime.now().astimezone().isoformat(),
                "source_fingerprint_stable": True,
                "source_fingerprint_sha256": before["sha256"],
                "checkpoint_manifest_sha256": protocol.sha256_of(
                    protocol.CHECKPOINT_MANIFEST
                ),
                "operational_resume": {
                    "validated_exclusive_prefix_cells": resume_record[
                        "validated_existing_cells"
                    ],
                    "shared_gpu_suffix_cells": resume_record["remaining_cells"],
                    "simulation_contract_changed": False,
                    "resume_orchestrator_sha256": script_hash,
                },
            }
        )
        _write_json(ROOT / "SUMMARY.json", summary)
        (ROOT / "SUMMARY.md").write_text(
            queue.render_markdown(summary), encoding="utf-8"
        )
        resume_record.update(
            {
                "status": "COMPLETE",
                "completed_at": summary["completed_at"],
                "completed_cells": len(completed),
                "source_fingerprint_stable": True,
            }
        )
        _write_json(ROOT / "SHARED_GPU_RESUME_PROTOCOL.json", resume_record)
        print(
            "[SA5-V3-SCREEN-RESUME] COMPLETE "
            f"selected={summary['sa5_candidate']} "
            f"next={summary['next_action']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        _write_json(
            ROOT / "RESUME_INCOMPLETE.json",
            {
                "schema": "sa5_v3_c50_stage3_b3_cont25_resume_incomplete/v1",
                "status": "INCOMPLETE_FAIL_CLOSED",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_cells": len(completed),
                "expected_cells": len(expected),
                "phase_b_started": False,
                "parent_selected": False,
                "sa6_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
