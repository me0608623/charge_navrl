"""Continue the SA5-R2 screen after deterministic sample-size repairs.

This continuation consumes the preserved 24-cell Phase-A matrix. It reruns
only cells whose completed episode count is below the frozen minimum, ranks
the effective replacement matrix, and then runs top-two retention. It never
starts training, accepts a parent, or launches SA6.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
REPAIR_RUNNER = HERE / "run_sa5_r2_checkpoint_screen_sample_repair_cell.py"
ADDENDUM_FREEZE = REPO / "docs/freeze/sa5_r2_checkpoint_screen_sample_repair_v1.json"
sys.path.insert(0, str(HERE))

import sa5_r2_checkpoint_screen as base  # noqa: E402
import sa5_r2_checkpoint_screen_queue as base_queue  # noqa: E402
import sa5_r2_checkpoint_screen_sample_repair as repair  # noqa: E402


CONTINUATION_SOURCES = (
    Path(__file__).resolve(),
    Path(repair.__file__).resolve(),
    REPAIR_RUNNER,
    ADDENDUM_FREEZE,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def continuation_fingerprint() -> dict[str, str]:
    result = dict(base_queue.source_fingerprint())
    for path in CONTINUATION_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.relative_to(REPO))] = base_queue.sha256_of(path)
    return result


def verify_addendum_freeze() -> dict:
    if not ADDENDUM_FREEZE.is_file():
        raise FileNotFoundError(ADDENDUM_FREEZE)
    frozen = json.loads(ADDENDUM_FREEZE.read_text(encoding="utf-8"))
    current = repair.repair_addendum()
    if frozen != current:
        raise RuntimeError("sample-size addendum differs from frozen JSON")
    return frozen


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _cell_path(phase_dir: Path, name: str, scenario: str) -> Path:
    stem = f"{scenario}_g5_d{base.DELAY_STEPS}_s{base.SEED}"
    return phase_dir / name / scenario / f"{stem}_cell.json"


def _load_cell(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_resources(
    wait_log: Path,
    *,
    wait_for_gpu: bool,
    poll_seconds: int,
    max_wait_minutes: int,
    expected_fingerprint: dict[str, str],
) -> None:
    snapshot = base_queue.resource_snapshot()
    if snapshot["ready"]:
        return
    if not wait_for_gpu:
        raise RuntimeError(f"GPU resources not ready: {snapshot}")
    started = time.monotonic()
    ready_polls = 0
    while True:
        snapshot = base_queue.resource_snapshot()
        with wait_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        print(
            f"[SA5-R2-REPAIR-WAIT] ready={snapshot['ready']} "
            f"free={snapshot['gpu_free_mib']}MiB "
            f"util={snapshot['gpu_util_percent']}%",
            flush=True,
        )
        if continuation_fingerprint() != expected_fingerprint:
            raise RuntimeError("continuation source drift while waiting")
        base_queue.verify_checkpoints()
        ready_polls = ready_polls + 1 if snapshot["ready"] else 0
        if ready_polls >= base_queue.READY_POLLS_REQUIRED:
            return
        if max_wait_minutes > 0 and (
            time.monotonic() - started
        ) >= max_wait_minutes * 60:
            raise TimeoutError("timed out waiting for an idle GPU")
        time.sleep(poll_seconds)


def _run_repair_cell(
    base_payload: dict,
    phase_dir: Path,
    addendum_sha256: str,
) -> dict:
    name, scenario = repair.cell_key(base_payload)
    candidate = base.candidate_by_name(name)
    cell_dir = phase_dir / name / scenario
    command = [
        str(PYTHON),
        str(REPAIR_RUNNER),
        str(base.checkpoint_path(candidate)),
        "--output-dir",
        str(cell_dir),
        "--checkpoint-name",
        name,
        "--scenario",
        scenario,
        "--expect-checkpoint-sha256",
        candidate["sha256"],
        "--expect-addendum-sha256",
        addendum_sha256,
        "--base-n",
        str(int(base_payload["metrics"]["n"])),
    ]
    result = subprocess.run(command, check=False, cwd=REPO)
    cell_path = _cell_path(phase_dir, name, scenario)
    if not cell_path.is_file():
        raise RuntimeError(
            f"repair runner exited {result.returncode} without JSON: "
            f"{name} {scenario}"
        )
    payload = _load_cell(cell_path)
    payload["runner_returncode"] = result.returncode
    return payload


def _load_phase_a(output: Path) -> list[dict]:
    phase_dir = output / "phase_a_corridor"
    payloads = []
    for cell in base_queue.planned_phase_a_cells():
        payload = _load_cell(
            _cell_path(
                phase_dir, cell["checkpoint_name"], cell["scenario"]
            )
        )
        candidate = base.candidate_by_name(cell["checkpoint_name"])
        repair.validate_base_cell(payload, candidate, cell["scenario"])
        payloads.append(payload)
    return payloads


def _repair_underfilled(
    base_payloads: list[dict],
    *,
    phase_dir: Path,
    addendum_sha256: str,
    wait_log: Path,
    wait_for_gpu: bool,
    poll_seconds: int,
    max_wait_minutes: int,
    expected_fingerprint: dict[str, str],
) -> list[dict]:
    repairs = []
    for payload in base_payloads:
        n = int(payload["metrics"]["n"])
        steps = repair.repair_steps(int(payload["steps"]), n)
        if steps is None:
            continue
        name, scenario = repair.cell_key(payload)
        cell_path = _cell_path(phase_dir, name, scenario)
        if cell_path.is_file():
            repaired = _load_cell(cell_path)
            print(
                f"[SA5-R2-REPAIR] reuse {name} {scenario} "
                f"n={repaired['metrics']['n']}",
                flush=True,
            )
        else:
            _ensure_resources(
                wait_log,
                wait_for_gpu=wait_for_gpu,
                poll_seconds=poll_seconds,
                max_wait_minutes=max_wait_minutes,
                expected_fingerprint=expected_fingerprint,
            )
            print(
                f"[SA5-R2-REPAIR] run {name} {scenario}: "
                f"base_n={n} base_steps={payload['steps']} repair_steps={steps}",
                flush=True,
            )
            repaired = _run_repair_cell(payload, phase_dir, addendum_sha256)
        candidate = base.candidate_by_name(name)
        repair.validate_repair_cell(
            repaired, candidate, scenario, base_n=n
        )
        repairs.append(repaired)
        if continuation_fingerprint() != expected_fingerprint:
            raise RuntimeError("continuation source drift after repair cell")
        base_queue.verify_checkpoints()
    return repairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-minutes", type=int, default=0)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")

    frozen = base_queue.verify_frozen_protocol()
    addendum = verify_addendum_freeze()
    base_queue.verify_checkpoints()
    output = args.output_root.expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    prereg = _load_cell(output / "PREREGISTRATION.json")
    if prereg.get("protocol", {}).get("sha256") != frozen["sha256"]:
        raise RuntimeError("base output protocol mismatch")
    base_before = _load_cell(output / "source_fingerprint_before.json")
    if base_before != base_queue.source_fingerprint():
        raise RuntimeError("base measurement sources drifted after Phase A")

    phase_a_base = _load_phase_a(output)
    repair_plan = []
    for payload in phase_a_base:
        n = int(payload["metrics"]["n"])
        steps = repair.repair_steps(int(payload["steps"]), n)
        if steps is not None:
            repair_plan.append(
                {
                    "checkpoint_name": payload["checkpoint_name"],
                    "scenario": payload["scenario"],
                    "base_n": n,
                    "base_steps": int(payload["steps"]),
                    "repair_steps": steps,
                }
            )
    print(
        f"[SA5-R2-REPAIR] Phase A base cells={len(phase_a_base)} "
        f"underfilled={len(repair_plan)}",
        flush=True,
    )
    for item in repair_plan:
        print(
            f"  {item['checkpoint_name']:4s} {item['scenario']:24s} "
            f"n={item['base_n']} steps={item['base_steps']}->{item['repair_steps']}",
            flush=True,
        )
    if not args.execute:
        print("[SA5-R2-REPAIR] dry run; pass --execute to continue")
        return 0

    paths = {
        "prereg": output / "SAMPLE_SIZE_REPAIR_PREREGISTRATION.json",
        "fingerprint_before": output / "continuation_fingerprint_before.json",
        "fingerprint_after": output / "continuation_fingerprint_after.json",
        "wait": output / "sample_repair_resource_wait.jsonl",
        "phase_a": output / "PHASE_A_SUMMARY.json",
        "selection": output / "TOP2_SELECTION.json",
        "summary": output / "SUMMARY.json",
        "markdown": output / "SUMMARY.md",
        "incomplete": output / "CONTINUATION_INCOMPLETE_NO_VERDICT.json",
    }
    before = continuation_fingerprint()
    _write_json(paths["fingerprint_before"], before)
    _write_json(
        paths["prereg"],
        {
            "status": "PREREGISTERED_BEFORE_SAMPLE_REPAIR",
            "created_at": now_iso(),
            "base_protocol_sha256": frozen["sha256"],
            "addendum": addendum,
            "phase_a_repair_plan": repair_plan,
            "rule_uses_outcome_metrics": False,
            "base_results_preserved": True,
            "pool_base_and_repair": False,
        },
    )

    phase_a_repairs: list[dict] = []
    phase_b_base: list[dict] = []
    phase_b_repairs: list[dict] = []
    try:
        phase_a_repairs = _repair_underfilled(
            phase_a_base,
            phase_dir=output / "phase_a_sample_size_repair",
            addendum_sha256=addendum["sha256"],
            wait_log=paths["wait"],
            wait_for_gpu=args.wait_for_gpu,
            poll_seconds=args.poll_seconds,
            max_wait_minutes=args.max_wait_minutes,
            expected_fingerprint=before,
        )
        phase_a_result = repair.rank_phase_a(
            phase_a_base, phase_a_repairs
        )
        _write_json(
            paths["phase_a"],
            {
                "schema": "sa5_r2_checkpoint_screen_phase_a/v1.1",
                "status": "COMPLETE_VALID_PHASE_A_WITH_SAMPLE_REPAIR",
                "cells_planned": len(phase_a_base),
                "base_cells_preserved": len(phase_a_base),
                "repair_cells": len(phase_a_repairs),
                **phase_a_result,
            },
        )
        _write_json(
            paths["selection"],
            {
                "status": "TOP_TWO_SELECTED_AFTER_VALID_SAMPLE_REPAIR",
                "top_two": phase_a_result["top_two"],
                "ranking": phase_a_result["ranked"],
                "base_protocol_sha256": frozen["sha256"],
                "sample_size_addendum_sha256": addendum["sha256"],
            },
        )
        print(
            f"[SA5-R2-REPAIR] Phase A ranked={phase_a_result['ranked']} "
            f"top_two={phase_a_result['top_two']}",
            flush=True,
        )

        phase_b_dir = output / "phase_b_retention"
        phase_b_repair_dir = output / "phase_b_sample_size_repair"
        phase_b_cells = base_queue.planned_phase_b_cells(
            phase_a_result["top_two"]
        )
        for index, cell in enumerate(phase_b_cells, start=1):
            cell_path = _cell_path(
                phase_b_dir, cell["checkpoint_name"], cell["scenario"]
            )
            if cell_path.is_file():
                payload = _load_cell(cell_path)
                print(
                    f"[SA5-R2-REPAIR] reuse Phase B {index}/{len(phase_b_cells)} "
                    f"{cell['checkpoint_name']} {cell['scenario']}",
                    flush=True,
                )
            else:
                _ensure_resources(
                    paths["wait"],
                    wait_for_gpu=args.wait_for_gpu,
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    expected_fingerprint=before,
                )
                print(
                    f"[SA5-R2-REPAIR] Phase B {index}/{len(phase_b_cells)}: "
                    f"{cell['checkpoint_name']} {cell['scenario']}",
                    flush=True,
                )
                payload = base_queue.run_cell(
                    cell, phase_b_dir, frozen["sha256"]
                )
            candidate = base.candidate_by_name(cell["checkpoint_name"])
            repair.validate_base_cell(payload, candidate, cell["scenario"])
            phase_b_base.append(payload)
            if continuation_fingerprint() != before:
                raise RuntimeError("continuation source drift after Phase-B cell")
            base_queue.verify_checkpoints()

            if repair.repair_steps(
                int(payload["steps"]), int(payload["metrics"]["n"])
            ) is not None:
                repaired = _repair_underfilled(
                    [payload],
                    phase_dir=phase_b_repair_dir,
                    addendum_sha256=addendum["sha256"],
                    wait_log=paths["wait"],
                    wait_for_gpu=args.wait_for_gpu,
                    poll_seconds=args.poll_seconds,
                    max_wait_minutes=args.max_wait_minutes,
                    expected_fingerprint=before,
                )
                phase_b_repairs.extend(repaired)

        verdict = repair.final_verdict(
            phase_a_base,
            phase_a_repairs,
            phase_b_base,
            phase_b_repairs,
        )
        after = continuation_fingerprint()
        if after != before:
            raise RuntimeError("continuation source fingerprint changed")
        _write_json(paths["fingerprint_after"], after)
        _write_json(
            paths["summary"],
            {
                "schema": "sa5_r2_checkpoint_screen_bundle/v1.1",
                "status": "COMPLETE_VALID_SINGLE_SEED_SCREEN_WITH_SAMPLE_REPAIR",
                "completed_at": now_iso(),
                "base_protocol": frozen,
                "sample_size_addendum": addendum,
                "source_fingerprint_stable": True,
                "phase_a_base_cells": phase_a_base,
                "phase_a_repair_cells": phase_a_repairs,
                "phase_b_base_cells": phase_b_base,
                "phase_b_repair_cells": phase_b_repairs,
                "verdict": verdict,
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        paths["markdown"].write_text(
            base_queue._render_markdown(verdict), encoding="utf-8"
        )
        print(
            "[SA5-R2-REPAIR] COMPLETE "
            f"recommended={verdict['recommended_for_human_consideration']} "
            "accepted_parent=False training_started=False sa6_started=False",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json(
            paths["incomplete"],
            {
                "schema": "sa5_r2_checkpoint_screen_continuation_incomplete/v1",
                "status": "INCOMPLETE_NO_VERDICT",
                "timestamp": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "phase_a_repair_cells": len(phase_a_repairs),
                "phase_b_base_cells": len(phase_b_base),
                "phase_b_repair_cells": len(phase_b_repairs),
                "accepted_parent": False,
                "training_started": False,
                "sa6_started": False,
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

