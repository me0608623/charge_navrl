"""Drive the SA2 capability acceptance: Stage A screen, Stage B verdict.

Protocol
--------
Stage A -- screen the six saved SA2 checkpoints on one independent seed (818)
at a single fixed 200 ms delay, across all three SA2 scenarios. Rank by the
worst margin any scenario produced and carry forward the top ``CARRY_FORWARD``
-- **plus every candidate tied with the last carried slot**, where tied means
within ``TIE_MARGIN``. A screen is not a verdict: one seed and one delay can
only order candidates, and when it cannot even do that it must say so rather
than pick.

Ties expand, they are never broken. Two consequences are deliberate:

* An unsafe cut exits non-zero, so "the screen finished" cannot be mistaken for
  "the screen chose". Stage B is never started by this command under any exit
  code -- it must be invoked explicitly with the carried checkpoints.
* The ranking rule is fixed before the data is seen. Switching to a criterion
  that happens to separate a tied set is post-hoc method selection, not a
  tiebreak, and it is not available here.

Saturated checks stay blocking. A check every candidate satisfies with the same
slack still decides PASS/FAIL; it just carries no ranking information, and the
``saturation`` report names which checks are in that state so ``worst_margin``
is not read as a quality score.

Stage B -- the verdict. Each carried checkpoint runs every
(scenario x seed x delay) combination as an independent cell. A checkpoint
passes only if every one of its cells passes on its own. Nothing is averaged
across seeds or delay conditions, and there is no code path that could: the
runner takes a single seed and a single delay, and the verdict is a conjunction
over cells.

Reference -- the same Stage B grid may be run on an SA1 checkpoint to separate
"SA2 taught this" from "SA2 kept this". Because that checkpoint trained on a
different (and in places harder) distribution, a pass there means the capability
predates SA2; it can never be read as SA2 having taught it.

Failure is closed. A cell whose runner raises is recorded as an invalid cell and
counted as a failure, never skipped, so a crashed cell cannot silently shrink
the denominator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from run_sa2_capability_suite import (  # noqa: E402
    ACTUATOR_PROFILE,
    DELAY_MS,
    MIN_EPISODES,
    SCENARIOS,
    THRESHOLDS,
    ARENA_SIZE_M,
    CORRIDOR_FREE_WIDTH_M,
    CORRIDOR_LENGTH_M,
    CORRIDOR_SPEED_RANGE,
    NARROW_WIDTH_RANGE,
    make_cell_id,
)


REPO = Path("/home/aa/IsaacLab")
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = _HERE / "run_sa2_capability_suite.py"

SA2_RUN_DIR = (
    REPO / "logs/rnn_car/sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1"
)
SA1_RUN_DIR = REPO / "logs/rnn_car/sa1_sim2real_v1_ne1024_s42_r1"

ROLLOUT_LENGTH = 128
#: The six SA2 checkpoints, keyed by training iteration. The trainer names files
#: by total env-steps = (iteration) * rollout_length at the save boundary.
SA2_CHECKPOINTS = {
    iteration: SA2_RUN_DIR / f"checkpoint_{iteration * ROLLOUT_LENGTH}.pt"
    for iteration in (50, 100, 150, 200, 250, 300)
}
#: SA1 c1700, the SA2 parent. Reference arm only; never a Stage B candidate.
SA1_REFERENCE = SA1_RUN_DIR / "checkpoint_217600.pt"

SCREEN_SEED = 818
SCREEN_DELAY_STEPS = 1              # 200 ms
VERDICT_SEEDS = (515, 616, 717)
VERDICT_DELAY_STEPS = (0, 1, 2)
CARRY_FORWARD = 2
#: Below this the Stage A ordering is inside measurement noise and the cut
#: between 2nd and 3rd place is reported as unsafe rather than presented as a
#: ranking.
TIE_MARGIN = 0.005

DEFAULT_OUTPUT_ROOT = REPO / "logs/gates/sa2_capability"


#: Files every cell re-imports. Each cell is a fresh Isaac process, so an edit
#: landing mid-stage silently changes the scene or the verdict for later cells
#: only -- and afterwards there is no way to tell which cells ran which version.
#: Fingerprinting turns that into a loud failure. Relevant here because other
#: sessions edit this tree concurrently.
FINGERPRINTED_SOURCES = (
    Path(__file__).resolve(),
    RUNNER,
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "sim2real_stage_curriculum_v2.py",
)


def source_fingerprint() -> dict[str, str]:
    import hashlib

    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FINGERPRINTED_SOURCES
    }


class PreconditionError(RuntimeError):
    """A fail-closed precondition was not met."""


@dataclass(frozen=True)
class Cell:
    checkpoint: Path
    scenario: str
    seed: int
    delay_steps: int

    @property
    def cell_id(self) -> str:
        return make_cell_id(
            self.checkpoint,
            self.scenario,
            self.delay_steps,
            self.seed,
        )


def screen_cells(checkpoints: Sequence[Path]) -> list[Cell]:
    return [
        Cell(checkpoint, scenario, SCREEN_SEED, SCREEN_DELAY_STEPS)
        for checkpoint in checkpoints
        for scenario in SCENARIOS
    ]


def verdict_cells(checkpoints: Sequence[Path]) -> list[Cell]:
    return [
        Cell(checkpoint, scenario, seed, delay)
        for checkpoint in checkpoints
        for scenario in SCENARIOS
        for delay in VERDICT_DELAY_STEPS
        for seed in VERDICT_SEEDS
    ]


def build_command(cell: Cell, output_dir: Path, *, num_envs: int, steps: int):
    return [
        str(PYTHON),
        str(RUNNER),
        str(cell.checkpoint),
        "--output-dir",
        str(output_dir),
        "--scenario",
        cell.scenario,
        "--seed",
        str(cell.seed),
        "--actuator-delay-steps",
        str(cell.delay_steps),
        "--actuator-profile",
        ACTUATOR_PROFILE,
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
    ]


def _gpu_foreign_processes() -> list[str]:
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
        raise PreconditionError(f"nvidia-smi failed: {smi.stderr.strip()!r}")
    return [
        line.strip()
        for line in smi.stdout.splitlines()
        if line.strip() and "gnome-remote-desktop" not in line
    ]


def wait_for_exclusive_gpu(*, poll_s: int = 30, timeout_s: int = 3600) -> None:
    """Block until nothing else is computing on the GPU.

    One evaluation at a time. Two Isaac processes sharing the card change each
    other's step timing, and timing is part of what a delay condition measures.
    """
    waited = 0
    while True:
        foreign = _gpu_foreign_processes()
        if not foreign:
            return
        if waited >= timeout_s:
            raise PreconditionError(
                f"GPU still busy after {waited}s: {foreign}"
            )
        print(f"    [gpu-wait] {foreign}; retry in {poll_s}s", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def preconditions(checkpoints: Sequence[Path]) -> list[str]:
    unmet: list[str] = []
    if not RUNNER.is_file():
        unmet.append(f"runner missing: {RUNNER}")
    if not PYTHON.is_file():
        unmet.append(f"python missing: {PYTHON}")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            unmet.append(f"checkpoint missing: {checkpoint}")
    if not checkpoints:
        unmet.append("no checkpoints requested")
    try:
        foreign = _gpu_foreign_processes()
    except PreconditionError as error:
        unmet.append(str(error))
    else:
        if foreign:
            unmet.append(f"GPU already busy: {foreign}")
    training = subprocess.run(
        ["pgrep", "-f", "train_rnn_car_wdclip.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if training.returncode == 0 and training.stdout.split():
        unmet.append(
            "a training process is running; evaluation would contend for the "
            f"GPU (pids {training.stdout.split()})"
        )
    return unmet


def execute_cell(
    cell: Cell, output_dir: Path, *, num_envs: int, steps: int
) -> dict:
    """Run one cell and return its payload, or an invalid-cell record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cell.scenario}_d{cell.delay_steps}_s{cell.seed}"
    cell_path = output_dir / f"{stem}_cell.json"
    invalid_path = output_dir / "cell.invalid.json"
    log = output_dir / "runner.log"
    stale = [path for path in (cell_path, invalid_path, log) if path.exists()]
    if stale:
        raise PreconditionError(
            "refusing to reuse a cell directory with stale outputs: "
            f"{[str(path) for path in stale]}"
        )

    command = build_command(cell, output_dir, num_envs=num_envs, steps=steps)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )

    if not cell_path.is_file():
        invalid = {
            "schema": "sa2_capability/v1",
            "cell_id": cell.cell_id,
            "checkpoint": str(cell.checkpoint),
            "scenario": cell.scenario,
            "seed": cell.seed,
            "delay_steps": cell.delay_steps,
            "valid": False,
            "threshold_pass": False,
            "worst_margin": None,
            "runner_returncode": completed.returncode,
            "reason": "runner produced no cell report",
            "runner_log": str(log),
        }
        invalid_path.write_text(
            json.dumps(invalid, indent=2, sort_keys=True), encoding="utf-8"
        )
        return invalid

    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    payload["valid"] = True
    payload["runner_returncode"] = completed.returncode
    # The runner exits non-zero exactly when it fails its thresholds. A
    # disagreement means one of the two is lying about the same cell.
    reported_pass = bool(payload["threshold_pass"])
    expected_rc = 0 if reported_pass else 1
    if completed.returncode != expected_rc:
        payload["valid"] = False
        payload["threshold_pass"] = False
        payload["reason"] = (
            f"runner exit {completed.returncode} contradicts "
            f"threshold_pass={reported_pass}"
        )
    if payload.get("cell_id") != cell.cell_id:
        payload["valid"] = False
        payload["threshold_pass"] = False
        payload["reason"] = (
            f"cell_id {payload.get('cell_id')!r} != requested {cell.cell_id!r}"
        )
    return payload


def assert_cells_are_distinct(payloads: Sequence[dict]) -> None:
    """No two recorded cells may describe the same (ckpt, scene, seed, delay).

    A duplicate would let one condition contribute twice to a conjunction, which
    is pooling wearing a different hat.
    """
    keys = [
        (
            payload["checkpoint"],
            payload["scenario"],
            payload["seed"],
            payload["delay_steps"],
        )
        for payload in payloads
    ]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise RuntimeError(f"duplicate cells recorded: {sorted(duplicates)}")


def summarize_checkpoint(
    checkpoint: Path,
    payloads: Sequence[dict],
    *,
    expected_cells: int,
) -> dict:
    """Fold one checkpoint's cells into a per-scenario and overall record.

    Paths are matched after ``resolve()``, never as strings. The queue is given
    ``IsaacLab/logs/rnn_car/<run>`` while the runner resolves it and records
    ``/home/aa/logs/rnn_car/<run>``, because that run directory is a symlink.
    A string comparison silently matched nothing, so every checkpoint folded to
    ``cells=0, worst_margin=None`` even though all 18 cells were recorded and
    passing -- an aggregation that reports a clean run as no data at all.
    """
    target = Path(checkpoint).resolve()
    mine = [
        p for p in payloads if Path(p["checkpoint"]).resolve() == target
    ]
    if not mine:
        raise RuntimeError(
            f"no cells matched {target}; recorded checkpoints are "
            f"{sorted({str(Path(p['checkpoint']).resolve()) for p in payloads})}"
        )
    per_scenario: dict[str, dict] = {}
    for scenario in SCENARIOS:
        cells = [p for p in mine if p["scenario"] == scenario]
        margins = [
            p["worst_margin"] for p in cells if p.get("worst_margin") is not None
        ]
        per_scenario[scenario] = {
            "cells": len(cells),
            "valid_cells": sum(1 for p in cells if p.get("valid")),
            "passing_cells": sum(1 for p in cells if p.get("threshold_pass")),
            "all_cells_pass": bool(cells)
            and all(p.get("threshold_pass") and p.get("valid") for p in cells),
            "worst_margin": min(margins) if margins else None,
        }
    overall = [
        info["worst_margin"]
        for info in per_scenario.values()
        if info["worst_margin"] is not None
    ]
    collision_rates = [
        float(p["metrics"]["cr"])
        for p in mine
        if p.get("valid") and p.get("metrics", {}).get("cr") is not None
    ]
    return {
        "checkpoint": str(checkpoint),
        "expected_cells": int(expected_cells),
        "cells": len(mine),
        "valid_cells": sum(1 for p in mine if p.get("valid")),
        "invalid_cells": sum(1 for p in mine if not p.get("valid")),
        "per_scenario": per_scenario,
        "worst_margin": min(overall) if overall else None,
        "worst_collision_rate": (
            max(collision_rates) if collision_rates else None
        ),
        "all_cells_pass": len(mine) == int(expected_cells)
        and all(p.get("threshold_pass") and p.get("valid") for p in mine),
    }


def saturation_report(payloads: Sequence[dict]) -> dict:
    """Split the acceptance checks into discriminating and saturated.

    A check is *saturated* when its margin is the same for every candidate: it
    still decides PASS/FAIL, but it cannot order anyone. Because the screen
    ranks on ``min`` over the checks, a saturated check whose margin is smaller
    than every discriminating one becomes the reported ``worst_margin`` for all
    candidates -- a tie manufactured by a constraint nobody violated. Naming
    them keeps the margin from being read as a quality score.

    Note also that with ``TO == 0`` the corridor's ``sr >= 0.90`` and
    ``cr <= 0.10`` are the same constraint, since ``sr = 1 - cr`` makes
    ``sr - 0.90`` and ``0.10 - cr`` identical. They are not two independent
    checks in that regime.
    """
    per_check: dict[str, dict[str, set]] = {}
    for payload in payloads:
        for scenario_check, info in (payload.get("checks") or {}).items():
            key = f"{payload['scenario']}/{scenario_check}"
            entry = per_check.setdefault(key, {"margins": set(), "values": set()})
            entry["margins"].add(round(float(info["margin"]), 6))
            entry["values"].add(round(float(info["value"]), 6))

    discriminating, saturated = {}, {}
    for key, entry in sorted(per_check.items()):
        record = {
            "distinct_margins": len(entry["margins"]),
            "margin_span": (
                round(max(entry["margins"]) - min(entry["margins"]), 6)
                if entry["margins"]
                else None
            ),
        }
        if len(entry["margins"]) <= 1:
            saturated[key] = record
        else:
            discriminating[key] = record
    return {
        "discriminating_checks": discriminating,
        "saturated_checks": saturated,
        "saturated_still_blocking": True,
    }


def rank_candidates(summaries: Sequence[dict], payloads: Sequence[dict]) -> dict:
    """Order Stage A candidates by worst margin; report an unsafe cut.

    Sorting key is the worst margin, then the mean of the per-scenario worst
    margins as a tiebreak. Checkpoints with any invalid cell are excluded from
    the ranking entirely -- an unmeasured scenario is not a good score.
    """

    def sortable(summary: dict) -> tuple[float, float]:
        margins = [
            info["worst_margin"]
            for info in summary["per_scenario"].values()
            if info["worst_margin"] is not None
        ]
        mean = sum(margins) / len(margins) if margins else float("-inf")
        return (
            summary["worst_margin"]
            if summary["worst_margin"] is not None
            else float("-inf"),
            mean,
        )

    eligible = [s for s in summaries if s["invalid_cells"] == 0]
    excluded = [
        {"checkpoint": s["checkpoint"], "invalid_cells": s["invalid_cells"]}
        for s in summaries
        if s["invalid_cells"] != 0
    ]
    ordered = sorted(eligible, key=sortable, reverse=True)

    # A tie is carried, never broken. Every candidate whose margin is within
    # TIE_MARGIN of the last carried slot is indistinguishable from it, so it
    # goes to Stage B too. Truncating to CARRY_FORWARD would pick by sort order
    # -- which here means "whichever checkpoint happens to be earliest" -- and
    # present that as a ranking.
    #
    # The alternative is worse than arbitrary: after seeing that the specified
    # criterion tied, one could switch to a criterion that does separate the
    # candidates. Choosing a ranking rule once its outcome is known is post-hoc
    # method selection, not a tiebreak. The rule stays fixed and ties expand.
    cut_is_safe = True
    cut_detail = "fewer candidates than carry-forward slots"
    carried = ordered[:CARRY_FORWARD]
    if len(ordered) > CARRY_FORWARD:
        boundary = sortable(ordered[CARRY_FORWARD - 1])[0]
        gap = boundary - sortable(ordered[CARRY_FORWARD])[0]
        cut_is_safe = gap >= TIE_MARGIN
        if not cut_is_safe:
            carried = [
                summary
                for summary in ordered
                if boundary - sortable(summary)[0] < TIE_MARGIN
            ]
        cut_detail = (
            f"gap between rank {CARRY_FORWARD} and {CARRY_FORWARD + 1} is "
            f"{gap:+.4f} (needs >= {TIE_MARGIN} to be outside noise); "
            + (
                f"cut is safe, carrying {CARRY_FORWARD}"
                if cut_is_safe
                else f"cut is NOT safe, carrying all {len(carried)} candidates "
                f"within {TIE_MARGIN} of the rank-{CARRY_FORWARD} margin"
            )
        )
    return {
        "tie_margin": TIE_MARGIN,
        "carry_forward_slots": CARRY_FORWARD,
        "tie_expanded": not cut_is_safe and len(carried) > CARRY_FORWARD,
        "saturation": saturation_report(payloads),
        "ranking_note": (
            "worst_margin is a distance-to-failure, not a quality score. A check "
            "that every candidate satisfies with the same slack contributes the "
            "same margin to all of them, so it can decide PASS/FAIL and still "
            "carry zero ranking information. See 'saturation' for which checks "
            "are in that state, and treat the margin as a ranking signal only "
            "for the checks listed as discriminating."
        ),
        "ranking": [
            {
                "rank": index + 1,
                "checkpoint": summary["checkpoint"],
                "worst_margin": summary["worst_margin"],
                "all_cells_pass": summary["all_cells_pass"],
                "per_scenario_worst_margin": {
                    name: info["worst_margin"]
                    for name, info in summary["per_scenario"].items()
                },
            }
            for index, summary in enumerate(ordered)
        ],
        "excluded_for_invalid_cells": excluded,
        "carried_forward": [s["checkpoint"] for s in carried],
        "cut_is_safe": cut_is_safe,
        "cut_detail": cut_detail,
    }


def scene_contract_record() -> dict:
    """The scene this queue grades on, restated for the report."""
    return {
        "stage": 2,
        "arena_size_m": ARENA_SIZE_M,
        "narrow_width_range_m": list(NARROW_WIDTH_RANGE),
        "corridor_free_width_m": CORRIDOR_FREE_WIDTH_M,
        "corridor_length_m": CORRIDOR_LENGTH_M,
        "corridor_static_obstacles": 3,
        "corridor_dynamic_obstacles": 1,
        "corridor_speed_range_m_s": list(CORRIDOR_SPEED_RANGE),
        "scenarios": list(SCENARIOS),
        "thresholds": {
            scenario: {
                "episodes_min": MIN_EPISODES,
                **THRESHOLDS[scenario],
            }
            for scenario in SCENARIOS
        },
        "actuator_profile": ACTUATOR_PROFILE,
        "source": "STAGE_SCENE_CURRICULUM[2] (the SA2 training spec)",
        "note": (
            "4.8 m is the corridor free width; the corridor length is 10.0 m "
            "and is fixed across SA1-SA8."
        ),
    }


def run_stage(
    stage: str,
    cells: Sequence[Cell],
    checkpoints: Sequence[Path],
    root: Path,
    *,
    num_envs: int,
    steps: int,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.cell_id}", flush=True)
        wait_for_exclusive_gpu()
        started = time.monotonic()
        # Taken per cell, not once per stage: the point is to catch an edit that
        # lands between cells.
        fingerprint_before = source_fingerprint()
        payload = execute_cell(
            cell, root / cell.cell_id, num_envs=num_envs, steps=steps
        )
        fingerprint_after = source_fingerprint()
        elapsed = time.monotonic() - started
        payload["elapsed_s"] = round(elapsed, 1)
        payload["source_fingerprint_before"] = fingerprint_before
        payload["source_fingerprint_after"] = fingerprint_after
        if fingerprint_before != fingerprint_after:
            prior_reason = payload.get("reason")
            payload["valid"] = False
            payload["threshold_pass"] = False
            payload["reason"] = (
                "source changed while this cell was running"
                + (f"; prior reason: {prior_reason}" if prior_reason else "")
            )
        payloads.append(payload)
        state = (
            "PASS"
            if payload.get("threshold_pass")
            else ("INVALID" if not payload.get("valid") else "FAIL")
        )
        margin = payload.get("worst_margin")
        print(
            f"    {state} worst_margin="
            f"{'n/a' if margin is None else f'{margin:+.4f}'} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    assert_cells_are_distinct(payloads)

    fingerprints = {
        json.dumps(fingerprint, sort_keys=True)
        for payload in payloads
        for fingerprint in (
            payload["source_fingerprint_before"],
            payload["source_fingerprint_after"],
        )
    }
    source_stable = len(fingerprints) == 1
    if not source_stable:
        # Do not raise: the cells already ran and their data is worth keeping.
        # Record it so no reader can mistake a mixed-version stage for a clean
        # one, and so the boundary is locatable from the per-cell fingerprints.
        print(
            f"\n*** SOURCE CHANGED MID-STAGE: {len(fingerprints)} distinct "
            "versions across cells. This stage is not a valid single-version "
            "result; see per-cell source_fingerprint. ***",
            flush=True,
        )

    expected_per_checkpoint = (
        len(cells) // len(checkpoints) if checkpoints else 0
    )
    summaries = [
        summarize_checkpoint(
            checkpoint,
            payloads,
            expected_cells=expected_per_checkpoint,
        )
        for checkpoint in checkpoints
    ]
    summary = {
        "schema": "sa2_capability_queue/v1",
        "stage": stage,
        "source_fingerprint_stable": source_stable,
        "source_fingerprints_observed": sorted(fingerprints),
        "scene_contract": scene_contract_record(),
        "cells_expected": len(cells),
        "cells_recorded": len(payloads),
        "invalid_cells": sum(1 for p in payloads if not p.get("valid")),
        "per_checkpoint": summaries,
        "cells": payloads,
        "pooling": (
            "none: one seed and one delay per cell, verdict is a conjunction "
            "over cells"
        ),
    }
    if stage == "A":
        summary["screen"] = rank_candidates(summaries, payloads)
        summary["screen_note"] = (
            f"one seed ({SCREEN_SEED}) at "
            f"{DELAY_MS[SCREEN_DELAY_STEPS]} ms; ordering only, not a verdict"
        )
    else:
        passing = [s for s in summaries if s["all_cells_pass"]]
        ordered_passing = sorted(
            passing,
            key=lambda summary: (
                summary["worst_margin"],
                -summary["worst_collision_rate"],
            ),
            reverse=True,
        )
        summary["verdict"] = {
            "cells_complete": len(payloads) == len(cells),
            "source_fingerprint_stable": source_stable,
            "passing_checkpoints": [s["checkpoint"] for s in ordered_passing],
            "selected_checkpoint": (
                ordered_passing[0]["checkpoint"] if ordered_passing else None
            ),
            "selection_rule": (
                "largest checkpoint worst_margin, then lowest worst cell CR"
            ),
            # A stage whose source changed partway through is not one experiment,
            # so it cannot certify anything regardless of how the cells scored.
            "sa2_pass": bool(
                payloads
                and source_stable
                and len(payloads) == len(cells)
                and ordered_passing
            ),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Stage B only: a checkpoint to grade (repeatable)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Stage B on the SA1 parent c1700 (retain-vs-teach reference arm)",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--check-preconditions", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.stage == "A":
        checkpoints = [SA2_CHECKPOINTS[i] for i in sorted(SA2_CHECKPOINTS)]
        cells = screen_cells(checkpoints)
        label = "stage_a_screen"
    else:
        if args.reference:
            if args.checkpoint:
                parser.error("--reference and --checkpoint are mutually exclusive")
            checkpoints = [SA1_REFERENCE]
            label = "stage_b_reference_c1700"
        elif args.checkpoint:
            checkpoints = [p.expanduser().resolve() for p in args.checkpoint]
            label = "stage_b_verdict"
        else:
            parser.error("Stage B needs --checkpoint or --reference")
        cells = verdict_cells(checkpoints)

    root = (args.output_root or (DEFAULT_OUTPUT_ROOT / label)).expanduser().resolve()

    print(f"SA2 capability stage {args.stage}: {len(cells)} cells -> {root}")
    for checkpoint in checkpoints:
        print(f"  checkpoint {checkpoint}")

    if args.check_preconditions:
        unmet = preconditions(checkpoints)
        if unmet:
            print("PRECONDITIONS UNMET:")
            for item in unmet:
                print(f"  - {item}")
            return 1
        print("PRECONDITIONS OK")
        return 0

    if not args.execute:
        print("\n--- plan only (no --execute) ---")
        for cell in cells:
            print(
                " ".join(
                    build_command(
                        cell,
                        root / cell.cell_id,
                        num_envs=args.num_envs,
                        steps=args.steps,
                    )
                )
            )
        return 0

    unmet = preconditions(checkpoints)
    if unmet:
        print("PRECONDITIONS UNMET:")
        for item in unmet:
            print(f"  - {item}")
        return 1

    summary = run_stage(
        args.stage,
        cells,
        checkpoints,
        root,
        num_envs=args.num_envs,
        steps=args.steps,
    )
    summary_path = root / f"{label}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n=== stage {args.stage} per-cell (no pooling) ===")
    for payload in summary["cells"]:
        margin = payload.get("worst_margin")
        print(
            f"  {payload['cell_id']}: "
            f"{'PASS' if payload.get('threshold_pass') else 'FAIL'} "
            f"worst={'n/a' if margin is None else f'{margin:+.4f}'} "
            f"{'' if payload.get('valid') else '[INVALID] ' + str(payload.get('reason'))}"
        )
    if args.stage == "A":
        screen = summary["screen"]
        print("\n=== stage A ranking (screen, not a verdict) ===")
        for row in screen["ranking"]:
            # Reporting must survive a missing margin: crashing here loses the
            # ranking even though the summary JSON is already on disk.
            margin = row["worst_margin"]
            print(
                f"  #{row['rank']} {Path(row['checkpoint']).name} "
                f"worst={'n/a' if margin is None else f'{margin:+.4f}'} "
                f"per_scenario={row['per_scenario_worst_margin']}"
            )
        print(f"  carried forward ({len(screen['carried_forward'])}): "
              f"{[Path(p).name for p in screen['carried_forward']]}")
        print(f"  cut_is_safe={screen['cut_is_safe']} — {screen['cut_detail']}")
        if screen["tie_expanded"]:
            print(
                f"  TIE EXPANDED: {len(screen['carried_forward'])} candidates "
                f"x {len(VERDICT_SEEDS) * len(VERDICT_DELAY_STEPS) * len(SCENARIOS)}"
                f" cells = "
                f"{len(screen['carried_forward']) * len(VERDICT_SEEDS) * len(VERDICT_DELAY_STEPS) * len(SCENARIOS)}"
                " Stage B cells"
            )
        # An unsafe cut must not exit 0. Exiting 0 is what lets a wrapper, a
        # supervisor, or an operator reading "screen finished" move straight to
        # Stage B on a candidate set the screen could not actually order.
        ok = bool(
            summary["source_fingerprint_stable"]
            and screen["cut_is_safe"]
            and screen["carried_forward"]
        )
        print(
            "\n  Stage B is NEVER started by this command. Run it explicitly "
            "with --stage B --checkpoint <each carried checkpoint>."
        )
        if not screen["cut_is_safe"]:
            print(
                "  Exit code is non-zero because the cut is unsafe: the screen "
                "ordered nothing, it only produced a tied set."
            )
    else:
        verdict = summary["verdict"]
        print(f"\n=== stage B verdict ===\n{json.dumps(verdict, indent=2)}")
        ok = verdict["sa2_pass"]
    print(f"\nsummary: {summary_path}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
