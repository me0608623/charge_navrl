"""SA3 minimal graduation gate: Phase B (4 cells) and Phase C (6 cells).

Fail-closed
-----------
Cells run in order and the queue stops at the **first** failure, invalid cell,
runtime-contract violation or source-fingerprint drift. Phase C never starts
unless Phase B is 4/4. Nothing downstream is started by this command.

Phase B -- does the selected SA3 checkpoint still do the core jobs?
    nav_clean, nav_native, native_crossing, narrow_range   on SA3 geometry.

Phase C -- did adaptation cost the parent's ability?
    1. SA2 parent on SA3 corridor geometry            (2 cells)
    2. SA2 parent and SA3 child on SA2 geometry       (4 cells)

    Every cell must clear the absolute corridor gate. The retention comparison
    is *matched*: same geometry, same seed, same delay, so the only difference
    is which policy ran. Retention bars: SR drop <= 0.02, CR increase <= 0.02.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sa3_gate_bc as runner  # noqa: E402


REPO = Path(__file__).resolve().parents[4]
PYTHON = Path(sys.executable)
RUNNER = Path(__file__).resolve().parent / "run_sa3_gate_bc.py"

SA3_CHECKPOINT = (
    REPO / "logs/rnn_car/sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1"
    / "checkpoint_12800.pt"
)
SA3_SHA256 = "e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88"
SA2_CHECKPOINT = (
    REPO / "logs/rnn_car/sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1"
    / "checkpoint_12800.pt"
)
SA2_SHA256 = "a5ea893446c69257caf7e1d5e55dd3dfc6bfaeeb384429552f727f8ef524188c"

GATE_SEED = 818
GATE_DELAY_STEPS = 1
RETENTION_SR_DROP_MAX = 0.02
RETENTION_CR_INCREASE_MAX = 0.02

PHASE_B_SCENARIOS = ("nav_clean", "nav_native", "native_crossing", "narrow_range")
CORRIDOR_SCENARIOS = runner.CORRIDOR_SCENARIOS

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
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in FINGERPRINTED_SOURCES
    }


def phase_b_cells() -> list[dict]:
    return [
        {
            "checkpoint": str(SA3_CHECKPOINT), "sha256": SA3_SHA256,
            "policy": "sa3_c100", "scenario": scenario, "geometry_stage": 3,
        }
        for scenario in PHASE_B_SCENARIOS
    ]


def phase_c_cells() -> list[dict]:
    cells = [
        # 1. parent on the child's geometry
        {"checkpoint": str(SA2_CHECKPOINT), "sha256": SA2_SHA256,
         "policy": "sa2_c100", "scenario": s, "geometry_stage": 3}
        for s in CORRIDOR_SCENARIOS
    ]
    # 2. matched pair on the parent's geometry
    for policy, checkpoint, digest in (
        ("sa2_c100", SA2_CHECKPOINT, SA2_SHA256),
        ("sa3_c100", SA3_CHECKPOINT, SA3_SHA256),
    ):
        cells += [
            {"checkpoint": str(checkpoint), "sha256": digest, "policy": policy,
             "scenario": s, "geometry_stage": 2}
            for s in CORRIDOR_SCENARIOS
        ]
    return cells


def cells_for(phase: str) -> list[dict]:
    if phase == "B":
        return phase_b_cells()
    if phase == "C":
        return phase_c_cells()
    raise ValueError(f"phase must be B or C, got {phase!r}")


def retention_verdict(pairs) -> dict:
    """Matched SA2-geometry comparison: what did the child lose vs the parent?"""
    rows = []
    for scenario, parent, child in pairs:
        sr_drop = parent["sr"] - child["sr"]
        cr_increase = child["cr"] - parent["cr"]
        rows.append({
            "scenario": scenario,
            "parent_sr": parent["sr"], "child_sr": child["sr"],
            "parent_cr": parent["cr"], "child_cr": child["cr"],
            "sr_drop": sr_drop, "cr_increase": cr_increase,
            "sr_drop_max": RETENTION_SR_DROP_MAX,
            "cr_increase_max": RETENTION_CR_INCREASE_MAX,
            "pass": (sr_drop <= RETENTION_SR_DROP_MAX
                     and cr_increase <= RETENTION_CR_INCREASE_MAX),
        })
    return {"rows": rows, "pass": all(r["pass"] for r in rows)}


def gpu_is_busy() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    heavy = [ln for ln in out.splitlines()
             if ln.strip() and int(ln.split(",")[1].strip().split()[0]) > 2000]
    return "; ".join(heavy) if heavy else None


def run_cell(cell: dict, output_root: Path) -> Path:
    cell_id = runner.make_cell_id(
        Path(cell["checkpoint"]), cell["scenario"], cell["geometry_stage"],
        GATE_DELAY_STEPS, GATE_SEED,
    )
    cell_dir = output_root / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(PYTHON), str(RUNNER), cell["checkpoint"],
        "--output-dir", str(cell_dir),
        "--scenario", cell["scenario"],
        "--geometry-stage", str(cell["geometry_stage"]),
        "--seed", str(GATE_SEED),
        "--actuator-delay-steps", str(GATE_DELAY_STEPS),
        "--expect-checkpoint-sha256", cell["sha256"],
    ]
    subprocess.run(argv, check=False, cwd=str(REPO))
    stem = (f"{cell['scenario']}_g{cell['geometry_stage']}"
            f"_d{GATE_DELAY_STEPS}_s{GATE_SEED}")
    return cell_dir / f"{stem}_cell.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("B", "C"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cells = cells_for(args.phase)
    before = source_fingerprint()
    print(f"[SA3-GATE-{args.phase}] {len(cells)} cells, "
          f"seed={GATE_SEED} delay={GATE_DELAY_STEPS} "
          f"min_episodes={runner.MIN_EPISODES}")
    for cell in cells:
        print(f"    {cell['policy']:9s} {cell['scenario']:20s} "
              f"geometry_stage={cell['geometry_stage']}")
    if not args.execute:
        print(f"[SA3-GATE-{args.phase}] dry run; pass --execute to run")
        return 0

    for cell in cells:
        actual = runner.sha256_of(cell["checkpoint"])
        if actual != cell["sha256"]:
            raise RuntimeError(
                f"checkpoint hash mismatch before start: {cell['checkpoint']} "
                f"expected {cell['sha256']} got {actual}")
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"GPU already busy: {busy}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict] = []
    stopped_at = None

    for index, cell in enumerate(cells, start=1):
        print(f"\n[SA3-GATE-{args.phase}] cell {index}/{len(cells)}: "
              f"{cell['policy']} {cell['scenario']} g{cell['geometry_stage']}",
              flush=True)
        cell_path = run_cell(cell, output_root)
        if not cell_path.is_file():
            stopped_at = {"index": index, "cell": cell, "reason": "no cell.json (invalid)"}
            break
        payload = json.loads(cell_path.read_text(encoding="utf-8"))
        payload["policy"] = cell["policy"]
        payloads.append(payload)
        if not payload["threshold_pass"]:
            stopped_at = {"index": index, "cell": cell,
                          "reason": f"FAIL on {payload['worst_check']} "
                                    f"({payload['worst_margin']:+.4f})"}
            break

    after = source_fingerprint()
    stable = before == after

    result = {
        "schema": f"sa3_gate_{args.phase.lower()}/v1",
        "phase": args.phase,
        "cells_planned": len(cells),
        "cells_completed": len(payloads),
        "stopped_at": stopped_at,
        "source_fingerprint": after,
        "source_fingerprint_stable": stable,
        "seed": GATE_SEED,
        "delay_steps": GATE_DELAY_STEPS,
        "checkpoints": {"sa3_c100": SA3_SHA256, "sa2_c100": SA2_SHA256},
        "cells": [
            {"cell_id": p["cell_id"], "policy": p["policy"],
             "scenario": p["scenario"], "geometry_stage": p["geometry_stage"],
             "n": p["metrics"]["n"], "sr": p["metrics"]["sr"],
             "cr": p["metrics"]["cr"], "to": p["metrics"]["to"],
             "worst_check": p["worst_check"], "worst_margin": p["worst_margin"],
             "pass": p["threshold_pass"]}
            for p in payloads
        ],
    }

    if args.phase == "C" and len(payloads) == len(cells):
        by_key = {(p["policy"], p["scenario"], p["geometry_stage"]): p["metrics"]
                  for p in payloads}
        pairs = [
            (s, by_key[("sa2_c100", s, 2)], by_key[("sa3_c100", s, 2)])
            for s in CORRIDOR_SCENARIOS
        ]
        result["retention"] = retention_verdict(pairs)

    all_pass = (len(payloads) == len(cells)
                and all(p["threshold_pass"] for p in payloads)
                and stable
                and (args.phase != "C" or result.get("retention", {}).get("pass")))
    result["phase_pass"] = bool(all_pass)

    (output_root / f"phase_{args.phase.lower()}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n[SA3-GATE-{args.phase}] " + "=" * 58)
    print(f"{'policy':10s}{'scenario':22s}{'geom':>5s}{'n':>7s}{'SR':>9s}"
          f"{'CR':>9s}{'TO':>9s}  verdict")
    for c in result["cells"]:
        print(f"{c['policy']:10s}{c['scenario']:22s}{c['geometry_stage']:5d}"
              f"{c['n']:7d}{c['sr']:9.4f}{c['cr']:9.4f}{c['to']:9.4f}  "
              f"{'PASS' if c['pass'] else 'FAIL'}")
    if "retention" in result:
        print(f"\n[SA3-GATE-C] retention on SA2 geometry (parent -> child)")
        for r in result["retention"]["rows"]:
            print(f"  {r['scenario']:22s} SR {r['parent_sr']:.4f}->{r['child_sr']:.4f} "
                  f"(drop {r['sr_drop']:+.4f} <= {r['sr_drop_max']})  "
                  f"CR {r['parent_cr']:.4f}->{r['child_cr']:.4f} "
                  f"(rise {r['cr_increase']:+.4f} <= {r['cr_increase_max']})  "
                  f"{'PASS' if r['pass'] else 'FAIL'}")
    if stopped_at:
        print(f"\n[SA3-GATE-{args.phase}] STOPPED at cell {stopped_at['index']}: "
              f"{stopped_at['reason']}")
    print(f"[SA3-GATE-{args.phase}] fingerprint stable: {stable}")
    print(f"[SA3-GATE-{args.phase}] phase_pass: {result['phase_pass']}")
    print(f"[SA3-GATE-{args.phase}] Reporting only; nothing downstream is started.")
    return 0 if result["phase_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
