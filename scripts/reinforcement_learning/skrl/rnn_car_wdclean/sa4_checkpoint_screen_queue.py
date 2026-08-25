"""Compare SA4 it50 and it100 on the two native and two corridor probes.

Hard-gate failures are expected diagnostic outcomes and do not truncate this
eight-cell screen. Missing/invalid cells, hash drift, source drift, or a busy
GPU remain fail-closed. This queue never starts or extends training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sa4_checkpoint_screen as runner  # noqa: E402


REPO = Path(__file__).resolve().parents[4]
PYTHON = Path(sys.executable)
RUNNER = Path(__file__).resolve().parent / "run_sa4_checkpoint_screen.py"
BASE_RUNNER = Path(__file__).resolve().parent / "run_sa3_gate_bc.py"
RUN_DIR = (
    REPO
    / "logs/rnn_car/sa4_sim2real_v2_from_sa3r1_c100_ne1024_s42_p100_r1"
)
CHECKPOINTS = (
    {
        "name": "sa4_c50",
        "path": RUN_DIR / "checkpoint_6400.pt",
        "sha256": "09a5f2364f6de48be0f23bca6978788624d83fb0895339985495dfda16b8ad09",
    },
    {
        "name": "sa4_c100",
        "path": RUN_DIR / "checkpoint_12800.pt",
        "sha256": "ad7b73561561ec1abd2a44605afd5a5cfc16385fbb292af0d9ea696fb11a49ad",
    },
)
SCENARIOS = runner.SCENARIOS
SEED = 818
DELAY_STEPS = 1
RETENTION_SR_DROP_MAX = 0.02
RETENTION_CR_INCREASE_MAX = 0.02
CORRIDOR_IMPROVEMENT_MIN = 0.005
STEPS_BY_SCENARIO = {
    "nav_native": 1200,
    "native_crossing": 1200,
    "corridor_lateral": 1200,
    # SA4 c50 produced only 390 completed longitudinal episodes at 1200
    # steps. Keep both checkpoints matched and raise the predeclared budget
    # rather than grading a sub-minimum cell.
    "corridor_longitudinal": 4000,
}

FINGERPRINTED_SOURCES = (
    RUNNER,
    BASE_RUNNER,
    Path(__file__).resolve(),
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
    / "sim2real_stage_curriculum_v2.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean"
    / "sa3_sa8_acceptance_contract.py",
)


def source_fingerprint() -> dict[str, str]:
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FINGERPRINTED_SOURCES
    }


def cells() -> list[dict]:
    return [
        {
            "checkpoint_name": checkpoint["name"],
            "checkpoint": str(checkpoint["path"]),
            "sha256": checkpoint["sha256"],
            "scenario": scenario,
            "steps": STEPS_BY_SCENARIO[scenario],
        }
        for checkpoint in CHECKPOINTS
        for scenario in SCENARIOS
    ]


def gpu_is_busy() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    heavy = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        used_mib = int(line.split(",")[1].strip().split()[0])
        if used_mib > 2000:
            heavy.append(line.strip())
    return "; ".join(heavy) if heavy else None


def run_cell(cell: dict, output_root: Path) -> Path:
    cell_dir = output_root / f"{cell['checkpoint_name']}__{cell['scenario']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(RUNNER),
        cell["checkpoint"],
        "--output-dir",
        str(cell_dir),
        "--scenario",
        cell["scenario"],
        "--seed",
        str(SEED),
        "--actuator-delay-steps",
        str(DELAY_STEPS),
        "--expect-checkpoint-sha256",
        cell["sha256"],
        "--steps",
        str(cell["steps"]),
    ]
    subprocess.run(command, check=False, cwd=str(REPO))
    stem = f"{cell['scenario']}_g4_d{DELAY_STEPS}_s{SEED}"
    return cell_dir / f"{stem}_cell.json"


def compare_checkpoints(payloads: list[dict]) -> dict:
    by_key = {
        (payload["checkpoint_name"], payload["scenario"]): payload["metrics"]
        for payload in payloads
    }
    native_rows = []
    for scenario in ("nav_native", "native_crossing"):
        early = by_key[("sa4_c50", scenario)]
        late = by_key[("sa4_c100", scenario)]
        sr_drop = early["sr"] - late["sr"]
        cr_increase = late["cr"] - early["cr"]
        native_rows.append(
            {
                "scenario": scenario,
                "c50_sr": early["sr"],
                "c100_sr": late["sr"],
                "sr_drop": sr_drop,
                "sr_drop_max": RETENTION_SR_DROP_MAX,
                "c50_cr": early["cr"],
                "c100_cr": late["cr"],
                "cr_increase": cr_increase,
                "cr_increase_max": RETENTION_CR_INCREASE_MAX,
                "pass": (
                    sr_drop <= RETENTION_SR_DROP_MAX
                    and cr_increase <= RETENTION_CR_INCREASE_MAX
                ),
            }
        )

    corridor_rows = []
    for scenario in ("corridor_lateral", "corridor_longitudinal"):
        early = by_key[("sa4_c50", scenario)]
        late = by_key[("sa4_c100", scenario)]
        corridor_rows.append(
            {
                "scenario": scenario,
                "c50_cr": early["cr"],
                "c100_cr": late["cr"],
                "cr_reduction": early["cr"] - late["cr"],
            }
        )
    c50_worst = max(row["c50_cr"] for row in corridor_rows)
    c100_worst = max(row["c100_cr"] for row in corridor_rows)
    corridor_improvement = c50_worst - c100_worst
    retention_pass = all(row["pass"] for row in native_rows)
    resume_candidate = (
        "sa4_c100"
        if retention_pass and corridor_improvement >= CORRIDOR_IMPROVEMENT_MIN
        else None
    )
    return {
        "native_retention": {
            "rows": native_rows,
            "pass": retention_pass,
        },
        "corridor": {
            "rows": corridor_rows,
            "c50_worst_cr": c50_worst,
            "c100_worst_cr": c100_worst,
            "worst_cr_reduction": corridor_improvement,
            "improvement_min": CORRIDOR_IMPROVEMENT_MIN,
        },
        "resume_candidate": resume_candidate,
        "training_extension_authorized": False,
        "reason": (
            "screen selects a possible resume point only; extension requires "
            "a separate human decision after reviewing every hard-gate result"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    planned = cells()
    print(
        f"[SA4-SCREEN] {len(planned)} cells, seed={SEED}, "
        f"delay={DELAY_STEPS}, stage=4"
    )
    for cell in planned:
        print(
            f"    {cell['checkpoint_name']:9s} {cell['scenario']:24s} "
            f"steps={cell['steps']}"
        )
    if not args.execute:
        print("[SA4-SCREEN] dry run; pass --execute to run")
        return 0

    for checkpoint in CHECKPOINTS:
        actual = runner.base.sha256_of(checkpoint["path"])
        if actual != checkpoint["sha256"]:
            raise RuntimeError(
                f"checkpoint hash mismatch: {checkpoint['path']} "
                f"expected {checkpoint['sha256']} got {actual}"
            )
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"GPU already busy: {busy}")

    before = source_fingerprint()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = []
    invalid = None
    for index, cell in enumerate(planned, start=1):
        print(
            f"\n[SA4-SCREEN] cell {index}/{len(planned)}: "
            f"{cell['checkpoint_name']} {cell['scenario']}",
            flush=True,
        )
        cell_path = run_cell(cell, output_root)
        if not cell_path.is_file():
            invalid = {
                "index": index,
                "cell": cell,
                "reason": "cell JSON is missing",
            }
            break
        payload = json.loads(cell_path.read_text(encoding="utf-8"))
        payload["checkpoint_name"] = cell["checkpoint_name"]
        payloads.append(payload)

    after = source_fingerprint()
    stable = before == after
    complete = len(payloads) == len(planned) and invalid is None
    comparison = compare_checkpoints(payloads) if complete else None
    summary = {
        "schema": "sa4_checkpoint_screen_summary/v1",
        "cells_planned": len(planned),
        "cells_completed": len(payloads),
        "invalid": invalid,
        "source_fingerprint": after,
        "source_fingerprint_stable": stable,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "geometry_stage": 4,
        "checkpoints": {
            checkpoint["name"]: {
                "path": str(checkpoint["path"]),
                "sha256": checkpoint["sha256"],
            }
            for checkpoint in CHECKPOINTS
        },
        "cells": [
            {
                "checkpoint_name": payload["checkpoint_name"],
                "scenario": payload["scenario"],
                "steps": payload["steps"],
                "n": payload["metrics"]["n"],
                "sr": payload["metrics"]["sr"],
                "cr": payload["metrics"]["cr"],
                "to": payload["metrics"]["to"],
                "hard_pass": payload["threshold_pass"],
                "worst_check": payload["worst_check"],
                "worst_margin": payload["worst_margin"],
            }
            for payload in payloads
        ],
        "comparison": comparison,
        "screen_valid": complete and stable,
    }
    summary_path = output_root / "sa4_checkpoint_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n[SA4-SCREEN] " + "=" * 70)
    print(
        f"{'checkpoint':11s}{'scenario':24s}{'n':>7s}"
        f"{'SR':>9s}{'CR':>9s}{'TO':>9s}  hard"
    )
    for cell in summary["cells"]:
        print(
            f"{cell['checkpoint_name']:11s}{cell['scenario']:24s}"
            f"{cell['n']:7d}{cell['sr']:9.4f}{cell['cr']:9.4f}"
            f"{cell['to']:9.4f}  {'PASS' if cell['hard_pass'] else 'FAIL'}"
        )
    print(f"[SA4-SCREEN] fingerprint stable: {stable}")
    print(f"[SA4-SCREEN] screen valid: {summary['screen_valid']}")
    if comparison:
        print(
            "[SA4-SCREEN] resume candidate: "
            f"{comparison['resume_candidate'] or '<none>'}"
        )
    print("[SA4-SCREEN] Reporting only; training is not started or extended.")
    return 0 if summary["screen_valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
