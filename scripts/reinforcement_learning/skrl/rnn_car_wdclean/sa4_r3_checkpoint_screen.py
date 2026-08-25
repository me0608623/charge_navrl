"""Frozen protocol and selection rule for the SA4-R3 checkpoint screen."""

from __future__ import annotations

import hashlib
import json


PROTOCOL_SCHEMA = "sa4_r3_checkpoint_screen/v1"
CHECKPOINTS = (
    ("sa4_r3_it25", "checkpoint_3200.pt", 25),
    ("sa4_r3_it50", "checkpoint_6400.pt", 50),
)
SCENARIOS = ("corridor_lateral", "corridor_longitudinal", "nav_native")
STEPS_BY_SCENARIO = {
    "corridor_lateral": 2500,
    "corridor_longitudinal": 4000,
    "nav_native": 1200,
}
SEED = 818
DELAY_STEPS = 1
ELIGIBILITY = "valid_return_only"
TIE_MARGIN = 0.005


def screen_protocol() -> dict:
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "purpose": (
            "screen SA4-R3 it25 and it50 on lateral, longitudinal, and native "
            "stage-4 capabilities before considering extension or SA5"
        ),
        "checkpoints": [
            {"name": name, "filename": filename, "iteration": iteration}
            for name, filename, iteration in CHECKPOINTS
        ],
        "fixed_evaluation": {
            "geometry_stage": 4,
            "evaluator_seed": SEED,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": 200,
            "actuator_profile": "sa1_delay_only",
            "lidar_distractor_eligibility": ELIGIBILITY,
            "num_envs": 64,
            "steps_by_scenario": STEPS_BY_SCENARIO,
        },
        "threshold_source": "sa3_sa8_acceptance_contract.py",
        "candidate_rule": (
            "lateral, longitudinal, and nav_native must each pass their stage-4 "
            "absolute SR/CR/TO thresholds with at least 1000 episodes"
        ),
        "selection_rule": {
            "primary": "minimum max(lateral CR, longitudinal CR)",
            "secondary": "minimum mean(lateral CR, longitudinal CR)",
            "tie_margin": TIE_MARGIN,
            "within_tie": "prefer it50",
        },
        "validity": [
            "all six cells produce cell JSON",
            "checkpoint and source fingerprints remain stable",
            "runtime eligibility marker is valid_return_only for policy and critic",
            "fixed d1 actuator and stage-4 geometry markers match",
            "console and exact episode ledgers reconcile",
            "no NaN, OOM, traceback, action override, training, or SA5 launch",
        ],
        "authorization_limit": (
            "a passing selection permits human consideration of extension or SA5; "
            "the screen itself starts neither"
        ),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def compare_checkpoints(payloads: list[dict]) -> dict:
    expected = {
        (checkpoint_name, scenario)
        for checkpoint_name, _, _ in CHECKPOINTS
        for scenario in SCENARIOS
    }
    by_key = {
        (payload["checkpoint_name"], payload["scenario"]): payload
        for payload in payloads
    }
    if set(by_key) != expected:
        raise ValueError(
            f"expected six fixed cells {sorted(expected)}, got {sorted(by_key)}"
        )

    rows = []
    candidates = []
    for checkpoint_name, _, iteration in CHECKPOINTS:
        cells = {
            scenario: by_key[(checkpoint_name, scenario)] for scenario in SCENARIOS
        }
        all_hard_pass = all(bool(cell["threshold_pass"]) for cell in cells.values())
        lateral_cr = float(cells["corridor_lateral"]["metrics"]["cr"])
        longitudinal_cr = float(cells["corridor_longitudinal"]["metrics"]["cr"])
        row = {
            "checkpoint_name": checkpoint_name,
            "iteration": iteration,
            "all_three_hard_pass": all_hard_pass,
            "worst_corridor_cr": max(lateral_cr, longitudinal_cr),
            "mean_corridor_cr": 0.5 * (lateral_cr + longitudinal_cr),
            "cells": {
                scenario: {
                    "n": int(cell["metrics"]["n"]),
                    "sr": float(cell["metrics"]["sr"]),
                    "cr": float(cell["metrics"]["cr"]),
                    "to": float(cell["metrics"]["to"]),
                    "hard_pass": bool(cell["threshold_pass"]),
                }
                for scenario, cell in cells.items()
            },
        }
        rows.append(row)
        if all_hard_pass:
            candidates.append(row)

    selected = None
    if candidates:
        candidates.sort(
            key=lambda row: (
                row["worst_corridor_cr"],
                row["mean_corridor_cr"],
                -row["iteration"],
            )
        )
        best = candidates[0]
        tied = [
            row
            for row in candidates
            if row["worst_corridor_cr"] - best["worst_corridor_cr"] < TIE_MARGIN
        ]
        selected = max(tied, key=lambda row: row["iteration"])["checkpoint_name"]

    return {
        "schema": "sa4_r3_checkpoint_comparison/v1",
        "rows": rows,
        "qualifying_candidates": [row["checkpoint_name"] for row in candidates],
        "selected_checkpoint": selected,
        "extension_or_sa5_consideration_eligible": selected is not None,
        "training_extension_started": False,
        "sa5_started": False,
    }

