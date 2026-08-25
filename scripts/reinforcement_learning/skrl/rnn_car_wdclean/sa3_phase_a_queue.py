"""Phase A: SA3 corridor direction screen — 6 checkpoints x 2 families = 12 cells.

Why this exists
---------------
The training-time ``scene/corridor`` metric merges lateral and longitudinal
pedestrian motion into one rate. SA2 showed those two can move in opposite
directions, so a healthy merged number can hide a degrading lateral. This queue
measures them separately, on the scene SA3 actually trained on.

Pre-registered ranking rule
---------------------------
Fixed here, in source, before any cell runs. Choosing a ranking rule after
seeing which candidate it favours is not a tiebreak.

1. **Hard gate first.** A checkpoint is eligible only if *both* of its cells are
   valid (>= 1000 completed episodes) and pass every bar
   (SR >= 0.90, CR <= 0.10, TO <= 0.05).
2. **Primary key: worst direction.** Rank by ``max(CR_lateral, CR_longitudinal)``
   ascending. "Most balanced" means *the worse direction is as good as
   possible* -- not that the two directions are closest together. A checkpoint
   at (0.090, 0.090) is perfectly symmetric and worse in both directions than
   one at (0.020, 0.050); symmetry is not the goal, avoiding a hidden weak
   direction is.
3. **Tie-break: mean CR** across the two directions.
4. **Tie band.** Candidates within ``TIE_MARGIN`` of the leader on the primary
   key are reported as tied. Inside the band the final checkpoint wins, so a
   mid-run checkpoint is never selected on noise alone.
5. ``|CR_lateral - CR_longitudinal|`` is reported as a descriptive asymmetry
   column. It never enters the ranking.

This command reports and stops. It never starts training for the next stage and
never chains into the follow-up navigation/narrow/retention cells; both are
separate, manually launched decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sa3_capability_suite as suite  # noqa: E402


REPO = Path(__file__).resolve().parents[4]
PYTHON = Path(sys.executable)
RUNNER = Path(__file__).resolve().parent / "run_sa3_capability_suite.py"

SA3_RUN_DIR = (
    REPO / "logs" / "rnn_car"
    / "sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1"
)

#: The six checkpoints saved every 50 training iterations.
CANDIDATES: tuple[str, ...] = tuple(
    str(SA3_RUN_DIR / f"checkpoint_{steps}.pt")
    for steps in (6400, 12800, 19200, 25600, 32000, 38400)
)
FINAL_CHECKPOINT_NAME = "checkpoint_38400.pt"

SCREEN_SEED = 818
SCREEN_DELAY_STEPS = 1
FAMILIES = suite.CORRIDOR_FAMILIES
TIE_MARGIN = 0.005

#: Source files whose content defines what a cell measured. Recorded per run so
#: a mid-run edit is loud rather than silently mixing two versions of the scene.
FINGERPRINTED_SOURCES = (
    RUNNER,
    Path(__file__).resolve(),
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
         / "sim2real_stage_curriculum_v2.py",
    REPO / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
         / "sa3_sa8_acceptance_contract.py",
)


def source_fingerprint() -> dict[str, str]:
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FINGERPRINTED_SOURCES
    }


def build_cells(candidates) -> list[dict]:
    """The full Phase A grid: every candidate x every corridor family."""
    return [
        {
            "checkpoint": checkpoint,
            "name": Path(checkpoint).name,
            "scenario": family,
            "seed": SCREEN_SEED,
            "delay_steps": SCREEN_DELAY_STEPS,
            "cell_id": suite.make_cell_id(
                Path(checkpoint), family, SCREEN_DELAY_STEPS, SCREEN_SEED
            ),
        }
        for checkpoint in candidates
        for family in FAMILIES
    ]


def _primary(summary: dict) -> float:
    return max(summary["cr_lateral"], summary["cr_longitudinal"])


def _mean(summary: dict) -> float:
    return (summary["cr_lateral"] + summary["cr_longitudinal"]) / 2.0


def asymmetry(summary: dict) -> float:
    """Descriptive only. Deliberately not part of the ranking key."""
    return abs(summary["cr_lateral"] - summary["cr_longitudinal"])


def rank(summaries) -> list[dict]:
    """Eligible candidates, best first. See the pre-registered rule above."""
    eligible = [
        s for s in summaries if s.get("all_pass") and s.get("all_valid")
    ]
    return sorted(eligible, key=lambda s: (_primary(s), _mean(s), s["name"]))


def tie_group(ranked) -> list[dict]:
    """Candidates statistically indistinguishable from the leader."""
    if not ranked:
        return []
    best = _primary(ranked[0])
    return [s for s in ranked if _primary(s) - best < TIE_MARGIN]


def select(ranked):
    """Leader, preferring the final checkpoint when it sits inside the band."""
    tied = tie_group(ranked)
    if not tied:
        return None
    for candidate in tied:
        if candidate["name"] == FINAL_CHECKPOINT_NAME:
            return candidate
    return tied[0]


def summarize(candidate: str, payloads) -> dict:
    """Fold one checkpoint's two cells into a single comparable row."""
    target = Path(candidate).resolve()
    mine = {
        p["scenario"]: p
        for p in payloads
        if Path(p["checkpoint"]).resolve() == target
    }
    missing = [f for f in FAMILIES if f not in mine]
    return {
        "checkpoint": candidate,
        "name": Path(candidate).name,
        "cells": len(mine),
        "missing_families": missing,
        "all_valid": not missing,
        "all_pass": bool(mine) and not missing
        and all(p["threshold_pass"] for p in mine.values()),
        "cr_lateral": mine["lateral"]["metrics"]["cr"] if "lateral" in mine else None,
        "cr_longitudinal": (
            mine["longitudinal"]["metrics"]["cr"] if "longitudinal" in mine else None
        ),
        "sr_lateral": mine["lateral"]["metrics"]["sr"] if "lateral" in mine else None,
        "sr_longitudinal": (
            mine["longitudinal"]["metrics"]["sr"] if "longitudinal" in mine else None
        ),
        "episodes": {f: mine[f]["metrics"]["n"] for f in mine},
    }


def gpu_is_busy() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    heavy = [
        line for line in out.splitlines()
        if line.strip() and int(line.split(",")[1].strip().split()[0]) > 2000
    ]
    return "; ".join(heavy) if heavy else None


def run_cell(cell: dict, output_root: Path) -> Path:
    cell_dir = output_root / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(PYTHON), str(RUNNER), cell["checkpoint"],
        "--output-dir", str(cell_dir),
        "--scenario", cell["scenario"],
        "--seed", str(cell["seed"]),
        "--actuator-delay-steps", str(cell["delay_steps"]),
    ]
    subprocess.run(argv, check=False, cwd=str(REPO))
    stem = f"{cell['scenario']}_d{cell['delay_steps']}_s{cell['seed']}"
    return cell_dir / f"{stem}_cell.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true",
        help="without this the grid is printed and nothing runs",
    )
    args = parser.parse_args(argv)

    cells = build_cells(CANDIDATES)
    fingerprint_before = source_fingerprint()

    print(f"[SA3-PHASE-A] {len(cells)} cells "
          f"= {len(CANDIDATES)} checkpoints x {len(FAMILIES)} families")
    print(f"[SA3-PHASE-A] fixed: seed={SCREEN_SEED} "
          f"delay={SCREEN_DELAY_STEPS} ({suite.DELAY_MS[SCREEN_DELAY_STEPS]} ms) "
          f"min_episodes={suite.MIN_EPISODES}")
    print(f"[SA3-PHASE-A] scene: arena={suite.ARENA_SIZE_M:g}m "
          f"corridor={suite.CORRIDOR_FREE_WIDTH_M:g}m x {suite.CORRIDOR_LENGTH_M:g}m "
          f"{suite.CORRIDOR_STATIC}S+{suite.CORRIDOR_DYNAMIC}D "
          f"speed={suite.CORRIDOR_SPEED_RANGE}")
    for cell in cells:
        print(f"    {cell['cell_id']}")

    if not args.execute:
        print("[SA3-PHASE-A] dry run; pass --execute to run")
        return 0

    for candidate in CANDIDATES:
        if not Path(candidate).is_file():
            raise FileNotFoundError(candidate)
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"GPU already busy: {busy}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        print(f"\n[SA3-PHASE-A] cell {index}/{len(cells)}: {cell['cell_id']}",
              flush=True)
        cell_path = run_cell(cell, output_root)
        if not cell_path.is_file():
            (output_root / cell["cell_id"] / "cell.invalid.json").write_text(
                json.dumps({"cell": cell, "reason": "runner produced no cell.json"},
                           indent=2),
                encoding="utf-8",
            )
            continue
        payloads.append(json.loads(cell_path.read_text(encoding="utf-8")))

    fingerprint_after = source_fingerprint()
    stable = fingerprint_before == fingerprint_after

    summaries = [summarize(c, payloads) for c in CANDIDATES]
    ranked = rank(summaries)
    tied = tie_group(ranked)
    chosen = select(ranked)

    result = {
        "schema": "sa3_phase_a/v1",
        "cells_planned": len(cells),
        "cells_completed": len(payloads),
        "source_fingerprint": fingerprint_after,
        "source_fingerprint_stable": stable,
        "ranking_rule": (
            "hard gate, then min max(CR_lateral, CR_longitudinal), then min mean CR; "
            f"tie band {TIE_MARGIN}; final checkpoint preferred inside the band; "
            "asymmetry is descriptive and never ranked"
        ),
        "summaries": summaries,
        "ranked": [s["name"] for s in ranked],
        "tie_group": [s["name"] for s in tied],
        "selected": chosen["name"] if chosen else None,
        "phase_a_pass": bool(chosen) and stable and len(payloads) == len(cells),
    }
    (output_root / "phase_a_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n[SA3-PHASE-A] " + "=" * 60)
    header = (f"{'checkpoint':22s} {'CR_lat':>8s} {'CR_lon':>8s} "
              f"{'worst':>8s} {'mean':>8s} {'|Δ|':>8s}  verdict")
    print(header)
    for s in summaries:
        if s["cr_lateral"] is None or s["cr_longitudinal"] is None:
            print(f"{s['name']:22s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s} "
                  f"{'—':>8s}  VOID {s['missing_families']}")
            continue
        print(f"{s['name']:22s} {s['cr_lateral']:8.4f} {s['cr_longitudinal']:8.4f} "
              f"{_primary(s):8.4f} {_mean(s):8.4f} {asymmetry(s):8.4f}  "
              f"{'PASS' if s['all_pass'] else 'FAIL'}")
    print(f"\n[SA3-PHASE-A] ranked   : {result['ranked']}")
    print(f"[SA3-PHASE-A] tie group: {result['tie_group']}")
    print(f"[SA3-PHASE-A] selected : {result['selected']}")
    print(f"[SA3-PHASE-A] fingerprint stable: {stable}")
    print("[SA3-PHASE-A] Reporting only. Nothing further is started by this "
          "command; the follow-up cells and the next training stage are "
          "separate manual decisions.")
    return 0 if result["phase_a_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
