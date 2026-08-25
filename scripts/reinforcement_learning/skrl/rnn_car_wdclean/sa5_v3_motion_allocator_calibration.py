"""Calibrate SA5 motion-family slot weights against observed reset shares.

This is an offline CPU diagnostic. It starts neither simulation nor training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FAMILIES = ("lateral", "longitudinal", "random_2d", "mixed")
CONTROL_METRICS = (
    REPO
    / "logs/rnn_car/sa5_v3_c50_equalweight_control_ne1024_s42_p25_r1/"
    "supervisor_metrics.jsonl"
)
WEIGHTED_METRICS = (
    REPO
    / "logs/rnn_car/sa5_v3_c50_random2d_weighted_ne1024_s42_p25_r1/"
    "supervisor_metrics.jsonl"
)
CONTROL_METRICS_SHA256 = (
    "857dff6fd6a61d5e3eb1872a374946cf2a0fc344dde8718bdd5f0052efd67987"
)
WEIGHTED_METRICS_SHA256 = (
    "31f55849cf291364ba92792c5545ebb580dfded6a44fa23459aeba8f33e7b97b"
)
ALLOCATOR = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/corridor_density.py"
)
ALLOCATOR_SHA256 = (
    "f6877a92699406809675f4013a86d4be6ba6d9db7cf93564106d19ad8cf91028"
)
SPEED_DENSITY_MIX = (
    ((0, 1), (0.70, 0.90), 0.10),
    ((1, 1), (0.70, 0.90), 0.10),
    ((2, 1), (0.50, 0.70), 0.15),
    ((3, 2), (0.25, 0.45), 0.30),
    ((4, 2), (0.25, 0.45), 0.35),
)
REJECTED_WEIGHTS = (0.15, 0.15, 0.70)
CANDIDATE_WEIGHTS = (0.30, 0.30, 0.40)
CALIBRATION_LAMBDA = 2.0
CALIBRATION_BATCHES = 12_000
CALIBRATION_SEED = 20_260_824
MAX_MODEL_ERROR = 0.02


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sources() -> None:
    for path, expected in (
        (CONTROL_METRICS, CONTROL_METRICS_SHA256),
        (WEIGHTED_METRICS, WEIGHTED_METRICS_SHA256),
        (ALLOCATOR, ALLOCATOR_SHA256),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"calibration source drift: {path}")


def _load_allocator():
    spec = importlib.util.spec_from_file_location("sa5_corridor_density", ALLOCATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load corridor allocator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observed_reset_share(path: Path) -> list[float]:
    totals = []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for family in FAMILIES:
        key = f"corridor_family/{family}/reset_count"
        totals.append(sum(float(row.get(key, 0.0)) for row in rows))
    total = sum(totals)
    if total <= 0:
        raise RuntimeError(f"no reset accounting in {path}")
    return [value / total for value in totals]


def simulate(weights: tuple[float, float, float] | None) -> dict:
    allocator = _load_allocator()
    torch.manual_seed(CALIBRATION_SEED)
    density_carry: dict = {}
    family_debt: dict = {}
    categories = torch.zeros(4, dtype=torch.long)
    slot_families = torch.zeros(3, dtype=torch.long)
    for _ in range(CALIBRATION_BATCHES):
        count = max(
            1,
            int(torch.poisson(torch.tensor(CALIBRATION_LAMBDA)).item()),
        )
        counts, _, _ = allocator.sample_speed_density_profiles_systematic(
            count,
            mix=SPEED_DENSITY_MIX,
            device="cpu",
            carry=density_carry,
        )
        interactions = allocator.sample_interaction_types(
            dynamic_counts=counts[:, 1], device="cpu"
        )
        families, _ = allocator.assign_families_and_pairs(
            counts[:, 1],
            interactions,
            device="cpu",
            family_debt=family_debt,
            family_weights=weights,
        )
        active = (
            torch.arange(allocator.MAX_CORRIDOR_DYNAMIC)[None, :]
            < counts[:, 1, None]
        )
        slot_families += torch.bincount(families[active], minlength=3)
        lo = torch.where(active, families, torch.full_like(families, 99)).min(1).values
        hi = torch.where(active, families, torch.full_like(families, -1)).max(1).values
        category = torch.where(lo == hi, lo, torch.full_like(lo, 3))
        categories += torch.bincount(category, minlength=4)
    return {
        "weights": None if weights is None else list(weights),
        "predicted_reset_share": (categories.double() / categories.sum()).tolist(),
        "realized_slot_share": (
            slot_families.double() / slot_families.sum()
        ).tolist(),
        "sampled_corridor_resets": int(categories.sum()),
    }


def build_report() -> dict:
    _verify_sources()
    observed_control = _observed_reset_share(CONTROL_METRICS)
    observed_rejected = _observed_reset_share(WEIGHTED_METRICS)
    simulated_control = simulate(None)
    simulated_rejected = simulate(REJECTED_WEIGHTS)
    simulated_candidate = simulate(CANDIDATE_WEIGHTS)

    errors = [
        abs(a - b)
        for observed, simulated in (
            (observed_control, simulated_control["predicted_reset_share"]),
            (observed_rejected, simulated_rejected["predicted_reset_share"]),
        )
        for a, b in zip(observed, simulated)
    ]
    candidate = simulated_candidate["predicted_reset_share"]
    checks = {
        "model_reproduces_known_arms": max(errors) <= MAX_MODEL_ERROR,
        "lateral_reset_share_at_least_15pct": candidate[0] >= 0.15,
        "longitudinal_reset_share_at_least_15pct": candidate[1] >= 0.15,
        "random_2d_reset_share_between_32_and_40pct": 0.32 <= candidate[2] <= 0.40,
        "mixed_reset_share_at_least_25pct": candidate[3] >= 0.25,
    }
    return {
        "schema": "sa5_v3_motion_allocator_calibration/v1",
        "status": (
            "COMPLETE_VALID_CPU_CALIBRATION"
            if all(checks.values())
            else "CALIBRATION_FAILED_CLOSED"
        ),
        "source_hashes": {
            str(CONTROL_METRICS.relative_to(REPO)): CONTROL_METRICS_SHA256,
            str(WEIGHTED_METRICS.relative_to(REPO)): WEIGHTED_METRICS_SHA256,
            str(ALLOCATOR.relative_to(REPO)): ALLOCATOR_SHA256,
        },
        "simulation": {
            "seed": CALIBRATION_SEED,
            "batches": CALIBRATION_BATCHES,
            "corridor_reset_batch_size_model": f"max(1, Poisson({CALIBRATION_LAMBDA:g}))",
            "speed_density_mix": SPEED_DENSITY_MIX,
        },
        "observed": {
            "equalweight_control": observed_control,
            "rejected_015_015_070": observed_rejected,
        },
        "simulated": {
            "equalweight_control": simulated_control,
            "rejected_015_015_070": simulated_rejected,
            "candidate_030_030_040": simulated_candidate,
        },
        "maximum_known_arm_absolute_error": max(errors),
        "maximum_allowed_error": MAX_MODEL_ERROR,
        "candidate_checks": checks,
        "selected_candidate_weights": (
            list(CANDIDATE_WEIGHTS) if all(checks.values()) else None
        ),
        "training_started": False,
        "sa5_graduation_authorized": False,
        "sa6_started": False,
        "interpretation_limit": (
            "Allocator exposure calibration only; it predicts family exposure, "
            "not policy SR/CR or training efficacy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "COMPLETE_VALID_CPU_CALIBRATION" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
