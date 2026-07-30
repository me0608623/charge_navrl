"""Checkpoint-level SA1 hard-gate acceptance for c1700, run while R1 trains.

Scope is deliberately narrower than the post-training matrix in
``sa1_r1_acceptance_queue.py``: only the SA1 blocking scene (Nav20-clean) is
evaluated, across ``d0/d1/d2`` x seeds ``515/616/717`` -- nine independent
cells. Nav20-native already has a d1 sentinel and does not block SA1, so it is
not re-run here.

Every cell is produced by the frozen runner, canonicalised, and checked by the
same ``sa1_r1_acceptance/v3`` validator used by the full matrix. No score is
pooled across seeds or delay conditions.

The run-completion precondition of the full matrix is intentionally *not*
applied: this entry point is authorised to evaluate an immutable intermediate
checkpoint while training continues. Every other fail-closed rule is kept, and
one extra rule is added -- only one evaluation may occupy the GPU at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sa1_r1_acceptance_queue as q  # noqa: E402

SCENARIO = "nav20_clean"
TARGET_CHECKPOINT = q.RUN_DIR / "checkpoint_217600.pt"
DEFAULT_OUTPUT_ROOT = (
    Path("/home/aa/IsaacLab/logs/gates/sa1_r1_nav20_acceptance/early_c1700_clean")
)
# R1 was stopped externally at it1808, so the frozen comparison points it1900 and
# it2109 can never exist: the trainer restarts its iteration counter at zero on
# resume and names checkpoints from that counter. The surviving late checkpoints
# are therefore the only comparison material, and each is immutable.
AUTHORISED_CHECKPOINTS = {
    (q.RUN_DIR / "checkpoint_217600.pt").resolve(),  # it1700, already 9/9 PASS
    (q.RUN_DIR / "checkpoint_224000.pt").resolve(),  # it1750
    (q.RUN_DIR / "checkpoint_230400.pt").resolve(),  # it1800, last one written
}
GPU_EXCLUSIVITY_POLL_S = 30
GPU_EXCLUSIVITY_TIMEOUT_S = 3600


def clean_matrix(checkpoint: Path) -> list[q.Cell]:
    """The nine blocking cells: three delay conditions x three seeds."""
    return [
        q.Cell(checkpoint, SCENARIO, delay, seed, stage="B")
        for delay in q.DELAY_CONDITIONS
        for seed in q.SEEDS
    ]


def _gpu_eval_processes(train_pids: Sequence[str]) -> list[str]:
    """Compute processes that are neither the trainer nor the desktop daemon."""
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if smi.returncode != 0:
        raise q.PreconditionError(f"nvidia-smi failed: {smi.stderr.strip()!r}")
    foreign: list[str] = []
    for line in smi.stdout.splitlines():
        if not line.strip():
            continue
        if "gnome-remote-desktop" in line:
            continue
        pid = line.split(",")[0].strip()
        if pid in train_pids:
            continue
        foreign.append(line.strip())
    return foreign


def _training_pids() -> list[str]:
    procs = subprocess.run(
        ["pgrep", "-f", q.TRAIN_PROCESS_PATTERN],
        capture_output=True,
        text=True,
        check=False,
    )
    if procs.returncode not in (0, 1):
        raise q.PreconditionError(f"pgrep failed: {procs.stderr.strip()!r}")
    return procs.stdout.split()


def early_preconditions(checkpoint: Path) -> list[str]:
    """Fail-closed checks that still apply while training runs."""
    unmet: list[str] = []

    if checkpoint.resolve() not in AUTHORISED_CHECKPOINTS:
        unmet.append(
            f"checkpoint {checkpoint} is outside the authorised set "
            f"{sorted(str(p.name) for p in AUTHORISED_CHECKPOINTS)}"
        )
    if not checkpoint.is_file():
        unmet.append(f"checkpoint missing: {checkpoint}")
    else:
        try:
            metadata = q._read_checkpoint_metadata(checkpoint)
        except Exception as exc:  # noqa: BLE001 - report the exact probe failure
            unmet.append(f"checkpoint unreadable: {exc}")
        else:
            # The filename encodes total env-steps; the trainer writes
            # total_steps = (iteration + 1) * 128. Cross-check both against the
            # name so a mislabelled or truncated file cannot pass.
            expected_steps = int(checkpoint.stem.split("_")[-1])
            iteration = metadata.get("iteration")
            total_steps = metadata.get("total_steps")
            if total_steps != expected_steps:
                unmet.append(
                    f"checkpoint total_steps={total_steps!r} disagrees with "
                    f"filename {expected_steps}"
                )
            if iteration is None or (int(iteration) + 1) * 128 != expected_steps:
                unmet.append(
                    f"checkpoint iteration={iteration!r} is inconsistent with "
                    f"total_steps {expected_steps}"
                )

    runner = q.RUNNER_DIR / q.runner_name(SCENARIO)
    if not runner.is_file():
        unmet.append(f"runner missing: {runner}")
    if not Path(q.ISAAC_PYTHON).is_file():
        unmet.append(f"isaac python missing: {q.ISAAC_PYTHON}")

    try:
        foreign = _gpu_eval_processes(_training_pids())
    except q.PreconditionError as exc:
        unmet.append(str(exc))
    else:
        if foreign:
            unmet.append(f"another GPU evaluation is already running: {foreign}")

    return unmet


def wait_for_exclusive_gpu(*, poll_s: int = GPU_EXCLUSIVITY_POLL_S,
                           timeout_s: int = GPU_EXCLUSIVITY_TIMEOUT_S) -> None:
    """Block until the trainer is the only compute process left on the GPU."""
    waited = 0
    while True:
        foreign = _gpu_eval_processes(_training_pids())
        if not foreign:
            return
        if waited >= timeout_s:
            raise q.PreconditionError(
                f"GPU still occupied by another evaluation after {waited}s: {foreign}"
            )
        print(f"    [gpu-wait] another eval present {foreign}; retry in {poll_s}s")
        time.sleep(poll_s)
        waited += poll_s


def verdict(summary: dict, expected_cells: int, checkpoint: Path) -> dict:
    """SA1 pass requires every blocking cell to pass on its own."""
    candidates = summary.get("candidates") or []
    candidate = candidates[0] if len(candidates) == 1 else None
    blocking_complete = bool(candidate and candidate["blocking_cells_complete"])
    recorded = int(candidate["cells"]) if candidate else 0
    return {
        "checkpoint": str(checkpoint),
        "scope": "sa1_hard_gate_nav20_clean_only",
        "blocking_scenario": SCENARIO,
        "expected_blocking_cells": expected_cells,
        "recorded_blocking_cells": recorded,
        "blocking_cells_complete": blocking_complete,
        "sa1_pass": bool(
            candidate
            and blocking_complete
            and candidate["all_blocking_gates_pass"]
            and recorded == expected_cells
        ),
        "advisory_note": (
            "Nav20-native is not re-run here; its d1 sentinel is advisory and "
            "cannot block SA1. advisory_cells_complete=false is expected."
        ),
        "sa2_note": "SA2 is never launched by this entry point.",
        "deployable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TARGET_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    cells = clean_matrix(checkpoint)
    root = args.output_root.expanduser().resolve()

    print(f"SA1 early acceptance: {len(cells)} blocking Nav20-clean cells")
    for cell in cells:
        print(f"  {cell.cell_id}")

    if not args.execute:
        print("\n--- plan only (no --execute); commands: ---")
        for cell in cells:
            print(" ".join(q.build_command(cell, root / cell.cell_id)))
        return 0

    unmet = early_preconditions(checkpoint)
    if unmet:
        print("PRECONDITIONS UNMET:")
        for item in unmet:
            print(f"  - {item}")
        return 1

    root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.cell_id}", flush=True)
        wait_for_exclusive_gpu()
        started = time.monotonic()
        payloads.append(q._execute_cell(cell, root / cell.cell_id))
        print(f"    took {time.monotonic() - started:.0f}s", flush=True)

    q.assert_no_cross_delay_average(payloads)
    summary = q.summarize_results(payloads, "B")
    summary["scope"] = "sa1_hard_gate_nav20_clean_only"
    q._write_json(root / "stage_summary.json", summary)

    result = verdict(summary, len(cells), checkpoint)
    q._write_json(root / "sa1_verdict.json", result)

    print("\n=== per-cell results (no pooling) ===")
    for payload in payloads:
        print(
            f"  {payload['delay_condition']} seed{payload['seed']}: "
            f"SR={payload['sr']:.4f} CR={payload['cr']:.4f} "
            f"TO={payload['timeout']:.4f} n={payload['episodes_completed']} "
            f"gate={'PASS' if payload['gate_pass'] else 'FAIL'}"
        )
    print("\n" + json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["sa1_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
