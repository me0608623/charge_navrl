"""SA4-D1: add eval seeds 515 and 616 to the SA4 c50 / c100 corridor screen.

Why a separate queue
--------------------
``sa4_checkpoint_screen_queue.py`` pins ``SEED = 818`` and its hash backs the
already-recorded seed-818 evidence. Editing it to take a seed would change the
fingerprint of a file that the existing summary claims it ran under, so this
queue reuses the *runner* unchanged and only supplies the new seeds.

Grid
----
``2 checkpoints x 2 corridor families x 2 new seeds = 8 cells``. Pooled with the
four corridor seed-818 cells already on disk, that is three eval seeds per
(checkpoint, family).

Pooling rule
------------
Rates are pooled **by counts**, never by averaging per-seed rates. Cells have
different episode totals (1022-2043 in the seed-818 batch), so a plain mean
would silently weight a short cell the same as a long one.

The reported statistic is the **worst direction**:
``max(pooled CR_lateral, pooled CR_longitudinal)``. The merged corridor average
is deliberately not used -- it is what hid SA3's lateral degradation.

This command reports and stops. It does not extend SA4, change any reward, or
start another training stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sa4_checkpoint_screen as runner  # noqa: E402


REPO = Path(__file__).resolve().parents[4]
PYTHON = Path(sys.executable)
RUNNER = Path(__file__).resolve().parent / "run_sa4_checkpoint_screen.py"

SA4_RUN_DIR = (
    REPO / "logs" / "rnn_car"
    / "sa4_sim2real_v2_from_sa3r1_c100_ne1024_s42_p100_r1"
)
CHECKPOINTS = (
    ("sa4_c50", str(SA4_RUN_DIR / "checkpoint_6400.pt"),
     "09a5f2364f6de48be0f23bca6978788624d83fb0895339985495dfda16b8ad09"),
    ("sa4_c100", str(SA4_RUN_DIR / "checkpoint_12800.pt"),
     "ad7b73561561ec1abd2a44605afd5a5cfc16385fbb292af0d9ea696fb11a49ad"),
)

#: Seeds added by D1. 818 already exists on disk and is pooled in afterwards.
NEW_SEEDS = (515, 616)
EXISTING_SEED = 818
CORRIDOR_SCENARIOS = ("corridor_lateral", "corridor_longitudinal")
DELAY_STEPS = 1
GEOMETRY_STAGE = 4
MIN_EPISODES = 1000
STEPS_BY_SCENARIO = {
    "corridor_lateral": 1200,
    "corridor_longitudinal": 4000,
}

#: Where the seed-818 batch already lives.
EXISTING_SCREEN_ROOT = (
    REPO / "logs" / "gates" / "sa4_checkpoint_screen" / "screen_r2_20260731"
)

FINGERPRINTED_SOURCES = (
    RUNNER,
    Path(__file__).resolve(),
    REPO / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa3_gate_bc.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
         / "sim2real_stage_curriculum_v2.py",
)


def source_fingerprint() -> dict[str, str]:
    return {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in FINGERPRINTED_SOURCES
    }


def build_cells() -> list[dict]:
    return [
        {"policy": policy, "checkpoint": checkpoint, "sha256": digest,
         "scenario": scenario, "seed": seed,
         "steps": STEPS_BY_SCENARIO[scenario]}
        for policy, checkpoint, digest in CHECKPOINTS
        for scenario in CORRIDOR_SCENARIOS
        for seed in NEW_SEEDS
    ]


def collisions_from(metrics: dict) -> int:
    """Collision count for pooling.

    Corridor cells carry exact rates reconciled against the corridor JSON, so
    ``cr * n`` lands on an integer to within float error. Anything further off
    means the rate was not exact and must not be pooled silently.
    """
    raw = metrics["cr"] * metrics["n"]
    nearest = round(raw)
    if abs(raw - nearest) > 1e-6:
        raise RuntimeError(
            f"cr*n = {raw!r} is not an integer collision count; the rate is not "
            "exact and pooling would fabricate precision"
        )
    return nearest


def load_existing_seed_cells(root: Path = EXISTING_SCREEN_ROOT) -> list[dict]:
    """The seed-818 corridor cells already on disk."""
    out = []
    for path in sorted(Path(root).glob("*/*_cell.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["scenario"] not in CORRIDOR_SCENARIOS:
            continue
        out.append(payload)
    return out


def validate_payload(payload: dict, expected: dict) -> list[str]:
    """Return every identity or evidence error that forbids pooling a cell."""
    errors = []
    expected_checkpoint = Path(expected["checkpoint"]).resolve()
    try:
        actual_checkpoint = Path(payload["checkpoint"]).resolve()
    except (KeyError, TypeError):
        actual_checkpoint = None
    if actual_checkpoint != expected_checkpoint:
        errors.append(
            f"checkpoint={actual_checkpoint}, expected={expected_checkpoint}"
        )
    checks = {
        "checkpoint_sha256": expected["sha256"],
        "scenario": expected["scenario"],
        "seed": expected["seed"],
        "delay_steps": DELAY_STEPS,
        "geometry_stage": GEOMETRY_STAGE,
        "steps": expected["steps"],
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            errors.append(f"{key}={payload.get(key)!r}, expected={value!r}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics is missing or not an object")
        return errors
    n = metrics.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < MIN_EPISODES:
        errors.append(f"episodes={n!r}, minimum={MIN_EPISODES}")
    for key in ("sr", "cr", "to"):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            errors.append(f"metrics.{key}={value!r} is not a finite rate")
    return errors


def expected_existing_cells() -> list[dict]:
    """The four frozen seed-818 cells that are allowed into the pool."""
    return [
        {
            "policy": policy,
            "checkpoint": checkpoint,
            "sha256": digest,
            "scenario": scenario,
            "seed": EXISTING_SEED,
            "steps": STEPS_BY_SCENARIO[scenario],
        }
        for policy, checkpoint, digest in CHECKPOINTS
        for scenario in CORRIDOR_SCENARIOS
    ]


def validate_payload_set(
    payloads: list[dict],
    expected_cells: list[dict],
    *,
    source: str,
) -> list[dict]:
    """Validate a complete, duplicate-free grid before any pooling."""
    expected = {
        (cell["policy"], cell["scenario"], cell["seed"]): cell
        for cell in expected_cells
    }
    seen = set()
    invalid = []
    for payload in payloads:
        try:
            key = (
                policy_of(payload),
                payload["scenario"],
                payload["seed"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(
                {"source": source, "reason": f"unreadable identity: {exc}"}
            )
            continue
        if key in seen:
            invalid.append(
                {"source": source, "key": list(key), "reason": "duplicate cell"}
            )
            continue
        seen.add(key)
        cell = expected.get(key)
        if cell is None:
            invalid.append(
                {"source": source, "key": list(key), "reason": "unexpected cell"}
            )
            continue
        errors = validate_payload(payload, cell)
        if errors:
            invalid.append(
                {"source": source, "key": list(key), "reason": "; ".join(errors)}
            )
    for key in sorted(set(expected) - seen):
        invalid.append(
            {"source": source, "key": list(key), "reason": "missing cell"}
        )
    return invalid


def policy_of(payload: dict) -> str:
    """c50 / c100 from the checkpoint file, not from the directory name."""
    stem = Path(payload["checkpoint"]).stem
    return {"checkpoint_6400": "sa4_c50", "checkpoint_12800": "sa4_c100"}[stem]


def pool(payloads) -> dict:
    """Pool by counts into {(policy, scenario): {...}}."""
    buckets: dict[tuple[str, str], dict] = {}
    for payload in payloads:
        key = (policy_of(payload), payload["scenario"])
        bucket = buckets.setdefault(
            key, {"episodes": 0, "collisions": 0, "seeds": [], "per_seed": []}
        )
        metrics = payload["metrics"]
        collisions = collisions_from(metrics)
        bucket["episodes"] += int(metrics["n"])
        bucket["collisions"] += collisions
        bucket["seeds"].append(payload["seed"])
        bucket["per_seed"].append(
            {"seed": payload["seed"], "n": int(metrics["n"]),
             "collisions": collisions, "cr": metrics["cr"]}
        )
    for bucket in buckets.values():
        bucket["cr"] = bucket["collisions"] / bucket["episodes"]
        bucket["seeds"] = sorted(bucket["seeds"])
    return buckets


def worst_direction(buckets: dict, policy: str) -> dict:
    """max(pooled CR) over the two families -- never the merged average."""
    rows = {
        scenario: buckets[(policy, scenario)]
        for scenario in CORRIDOR_SCENARIOS
        if (policy, scenario) in buckets
    }
    missing = [s for s in CORRIDOR_SCENARIOS if s not in rows]
    if missing:
        raise RuntimeError(f"{policy} is missing pooled families: {missing}")
    worst = max(rows, key=lambda s: rows[s]["cr"])
    return {
        "policy": policy,
        "per_family": {s: rows[s]["cr"] for s in rows},
        "episodes": {s: rows[s]["episodes"] for s in rows},
        "worst_family": worst,
        "worst_cr": rows[worst]["cr"],
        "seeds": rows[worst]["seeds"],
    }


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
    cell_dir = output_root / f"{cell['policy']}__{cell['scenario']}__s{cell['seed']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cell['scenario']}_g{GEOMETRY_STAGE}_d{DELAY_STEPS}_s{cell['seed']}"
    cell_path = cell_dir / f"{stem}_cell.json"
    cell_path.unlink(missing_ok=True)
    subprocess.run(
        [str(PYTHON), str(RUNNER), cell["checkpoint"],
         "--output-dir", str(cell_dir),
         "--scenario", cell["scenario"],
         "--seed", str(cell["seed"]),
         "--actuator-delay-steps", str(DELAY_STEPS),
         "--expect-checkpoint-sha256", cell["sha256"],
         "--steps", str(cell["steps"])],
        check=False, cwd=str(REPO),
    )
    return cell_path


def write_incomplete(
    output_root: Path,
    *,
    cells_planned: int,
    cells_completed: int,
    invalid: list[dict],
    reason: str,
    source_fingerprint_stable: bool | None = None,
) -> None:
    payload = {
        "schema": "sa4_d1_incomplete/v1",
        "new_cells_planned": cells_planned,
        "new_cells_completed": cells_completed,
        "invalid": invalid,
        "reason": reason,
        "pooled_verdict_reported": False,
    }
    if source_fingerprint_stable is not None:
        payload["source_fingerprint_stable"] = source_fingerprint_stable
    (output_root / "D1_INCOMPLETE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cells = build_cells()
    before = source_fingerprint()
    print(f"[SA4-D1] {len(cells)} new cells "
          f"= {len(CHECKPOINTS)} checkpoints x {len(CORRIDOR_SCENARIOS)} families "
          f"x {len(NEW_SEEDS)} seeds")
    print(f"[SA4-D1] fixed: stage={GEOMETRY_STAGE} delay={DELAY_STEPS} (200 ms); "
          f"pooled afterwards with the existing seed-{EXISTING_SEED} batch")
    for cell in cells:
        print(
            f"    {cell['policy']:9s} {cell['scenario']:22s} "
            f"seed={cell['seed']} steps={cell['steps']}"
        )
    if not args.execute:
        print("[SA4-D1] dry run; pass --execute to run")
        return 0

    for _, checkpoint, digest in CHECKPOINTS:
        actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"GPU already busy: {busy}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    new_payloads: list[dict] = []
    invalid: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        print(f"\n[SA4-D1] cell {index}/{len(cells)}: {cell['policy']} "
              f"{cell['scenario']} seed={cell['seed']}", flush=True)
        cell_path = run_cell(cell, output_root)
        if not cell_path.is_file():
            invalid.append(
                {"source": "new", "cell": cell, "reason": "cell JSON is missing"}
            )
            continue
        try:
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(
                {"source": "new", "cell": cell, "reason": f"invalid JSON: {exc}"}
            )
            continue
        errors = validate_payload(payload, cell)
        if errors:
            invalid.append(
                {"source": "new", "cell": cell, "reason": "; ".join(errors)}
            )
            continue
        new_payloads.append(payload)

    # Fail closed. Pooling the pre-existing seed-818 cells on their own would
    # print a plausible table whose "three seed" claim is false, and the command
    # would still exit 0. A D1 verdict requires every new cell.
    if invalid or len(new_payloads) != len(cells):
        write_incomplete(
            output_root,
            cells_planned=len(cells),
            cells_completed=len(new_payloads),
            invalid=invalid,
            reason="no pooled verdict is reported from a partial or invalid run",
        )
        print(f"\n[SA4-D1] STOPPED: {len(invalid)}/{len(cells)} new cells produced "
              "no valid cell.json. No pooled verdict is reported.")
        for item in invalid:
            print(f"    invalid: {item}")
        return 1

    existing = load_existing_seed_cells()
    after = source_fingerprint()
    stable = before == after
    evidence_invalid = validate_payload_set(
        existing,
        expected_existing_cells(),
        source="existing_seed818",
    )
    if evidence_invalid or not stable:
        if not stable:
            evidence_invalid.append(
                {"source": "source_fingerprint", "reason": "source drift"}
            )
        write_incomplete(
            output_root,
            cells_planned=len(cells),
            cells_completed=len(new_payloads),
            invalid=evidence_invalid,
            reason="existing evidence or source fingerprint is invalid",
            source_fingerprint_stable=stable,
        )
        print(
            "\n[SA4-D1] STOPPED: existing evidence or source fingerprint "
            "failed validation. No pooled verdict is reported."
        )
        for item in evidence_invalid:
            print(f"    invalid: {item}")
        return 1

    buckets = pool(new_payloads + existing)

    verdicts = {p: worst_direction(buckets, p) for p, _, _ in CHECKPOINTS}
    c50, c100 = verdicts["sa4_c50"], verdicts["sa4_c100"]
    result = {
        "schema": "sa4_d1_seed_pool/v1",
        "new_cells_planned": len(cells),
        "new_cells_completed": len(new_payloads),
        "invalid": invalid,
        "existing_cells_pooled": len(existing),
        "seeds_new": list(NEW_SEEDS),
        "seed_existing": EXISTING_SEED,
        "delay_steps": DELAY_STEPS,
        "geometry_stage": GEOMETRY_STAGE,
        "source_fingerprint_before": before,
        "source_fingerprint": after,
        "source_fingerprint_stable": stable,
        "pooled": {f"{p}|{s}": v for (p, s), v in buckets.items()},
        "worst_direction": verdicts,
        "worst_direction_delta_c100_minus_c50": c100["worst_cr"] - c50["worst_cr"],
        "pooling_rule": (
            "counts, not averaged rates; statistic is max over families, "
            "never the merged corridor average"
        ),
    }
    (output_root / "sa4_d1_pool_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("\n[SA4-D1] " + "=" * 62)
    print(f"{'policy':10s}{'family':24s}{'seeds':>18s}{'episodes':>10s}"
          f"{'collisions':>12s}{'pooled CR':>11s}")
    for (policy, scenario), b in sorted(buckets.items()):
        print(f"{policy:10s}{scenario:24s}{str(b['seeds']):>18s}"
              f"{b['episodes']:10d}{b['collisions']:12d}{b['cr']:11.4f}")
    print()
    for policy in ("sa4_c50", "sa4_c100"):
        v = verdicts[policy]
        print(f"  {policy:9s} worst direction = {v['worst_family']} "
              f"CR={v['worst_cr']:.4f}")
    print(f"\n  worst-direction CR  c50 {c50['worst_cr']:.4f} -> "
          f"c100 {c100['worst_cr']:.4f}  "
          f"delta {result['worst_direction_delta_c100_minus_c50']:+.4f}")
    print(f"[SA4-D1] fingerprint stable: {result['source_fingerprint_stable']}")
    print("[SA4-D1] Reporting only. SA4 is not extended, no reward is changed, "
          "and no further training stage is started.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
