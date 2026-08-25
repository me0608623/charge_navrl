"""Frozen two-phase checkpoint screen for the completed SA5 run.

This module is intentionally pure: it defines checkpoint identity, fixed exam
conditions, ranking, and final verdict folding. It never launches Isaac Sim or
training. The executable queue lives in ``sa5_checkpoint_screen_queue.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

RUN_NAME = (
    "sa5_sim2real_v2_from_sa4r3_it125_hwaiver_"
    "actd12_ne1024_s42_p300_r1"
)
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME

STAGE = 5
SEED = 818
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
NUM_ENVS = 64
MIN_EPISODES = 1000
TIE_MARGIN = 0.005

CORRIDOR_SCENARIOS = (
    "corridor_lateral",
    "corridor_longitudinal",
    "corridor_random2d",
    "corridor_mixed",
)
RETENTION_SCENARIOS = ("nav_native", "narrow_range")
ALL_SCENARIOS = CORRIDOR_SCENARIOS + RETENTION_SCENARIOS

STEPS_BY_SCENARIO = {
    "corridor_lateral": 2500,
    "corridor_longitudinal": 2500,
    "corridor_random2d": 2500,
    "corridor_mixed": 2500,
    "nav_native": 1200,
    "narrow_range": 1200,
}

CANDIDATES = (
    {
        "name": "c150",
        "conceptual_iteration": 150,
        "filename": "checkpoint_19200.pt",
        "sha256": "cf66f3e064bccf8b71e1109ce70950669b620737af050ec544edb1d4d000d171",
    },
    {
        "name": "c200",
        "conceptual_iteration": 200,
        "filename": "checkpoint_25600.pt",
        "sha256": "b007a5d3851711d932664e2e27b805ffa11b2ec4c07589ec06be545937b2f551",
    },
    {
        "name": "c250",
        "conceptual_iteration": 250,
        "filename": "checkpoint_32000.pt",
        "sha256": "b00118781a554cec0a922ae9ace81610f4cd279ac24aed58509a2c0ab6be29e3",
    },
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "filename": "checkpoint_38400.pt",
        "sha256": "5cc978df7ce38cbf9412844734377fc089cb389a791165cc747bb81575b53f79",
    },
)

CORRIDOR_THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}
NATIVE_THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.09,
    "to_max": 0.04,
}
NARROW_THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.05,
    "to_max": 0.05,
    "crossing_min": 0.95,
    "direct_crossing_min": 0.95,
}


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return RUN_DIR / str(data["filename"])


def thresholds_for(scenario: str) -> dict[str, float | int]:
    if scenario in CORRIDOR_SCENARIOS:
        return dict(CORRIDOR_THRESHOLDS)
    if scenario == "nav_native":
        return dict(NATIVE_THRESHOLDS)
    if scenario == "narrow_range":
        return dict(NARROW_THRESHOLDS)
    raise ValueError(f"unknown scenario {scenario!r}")


def _stage_scene() -> dict:
    # Imported lazily so pure ranking tests do not need the Isaac Python stack.
    import sys

    skrl_root = HERE.parent
    if str(skrl_root) not in sys.path:
        sys.path.insert(0, str(skrl_root))
    from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
        make_sim2real_curriculum_config,
    )
    from sa3_sa8_acceptance_contract import stage_scene_contract

    values = stage_scene_contract(STAGE)
    values["room_half_extent_m"] = float(values["arena_size_m"]) / 2.0
    values["corridor_wall_span_m"] = float(values["arena_size_m"])
    values["corridor_boundary_wall_width_m"] = 1.0
    values["corridor_expected_boundary_overlap_m"] = 0.5
    values["corridor_sealed_to_boundary"] = True
    cfg = make_sim2real_curriculum_config(
        STAGE, checkpoint="__SA5_SCREEN_SCENE_PROBE_NOT_LOADED__.pt"
    )
    values["narrow_segment_length_m"] = float(
        cfg.narrow_passage_segment_length
    )
    return json.loads(json.dumps(values))


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_checkpoint_screen_protocol/v2",
        "run_name": RUN_NAME,
        "stage": STAGE,
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "phase_a": {
            "purpose": "rank four checkpoints on four fixed corridor families",
            "scenarios": list(CORRIDOR_SCENARIOS),
            "cells": len(CANDIDATES) * len(CORRIDOR_SCENARIOS),
            "ranking": {
                "primary": "min max(CR across four corridor families)",
                "secondary": "min episode-unweighted mean family CR",
                "tertiary": "prefer later checkpoint only after exact score ties",
                "tie_band": TIE_MARGIN,
                "tie_band_use": "descriptive; Phase B still receives exactly two candidates",
                "hard_gate_not_ranking_key": True,
            },
            "promotion": "best two structurally valid checkpoints",
        },
        "phase_b": {
            "purpose": "native and narrow retention for Phase-A top two only",
            "scenarios": list(RETENTION_SCENARIOS),
            "cells": 2 * len(RETENTION_SCENARIOS),
        },
        "fixed_evaluation": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "steps_by_scenario": dict(STEPS_BY_SCENARIO),
            "scene": _stage_scene(),
            "corridor_fixed_density": "4S2D",
            "corridor_geometry_contract": (
                "10 m interaction zone; physical side walls span the 15 m "
                "arena and overlap each north/south boundary wall by 0.5 m"
            ),
            "corridor_family_modes": {
                "corridor_lateral": "lateral",
                "corridor_longitudinal": "longitudinal",
                "corridor_random2d": "random_2d patrol",
                "corridor_mixed": "legacy balanced mixed patrol",
            },
        },
        "thresholds": {
            "corridor": dict(CORRIDOR_THRESHOLDS),
            "native_stage5": dict(NATIVE_THRESHOLDS),
            "narrow": dict(NARROW_THRESHOLDS),
        },
        "evidence_boundary": (
            "single training seed and single fixed evaluator seed; this selects "
            "checkpoints for further consideration and is not the formal SA5 "
            "multi-seed/multi-delay graduation matrix; unsealed r2 results are "
            "invalid and cannot be pooled with this protocol"
        ),
        "authorization_limit": (
            "the screen starts neither training nor SA6 and never accepts a parent"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def evaluate_metrics(scenario: str, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario)
    checks: dict[str, dict] = {}

    def add(name: str, value: float | int, bound: float | int, direction: str):
        numeric = float(value)
        limit = float(bound)
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite {scenario} metric {name}={value!r}")
        if direction == ">=":
            margin = numeric - limit
        elif direction == "<=":
            margin = limit - numeric
        else:  # pragma: no cover - internal contract
            raise ValueError(direction)
        checks[name] = {
            "value": numeric,
            "bound": limit,
            "direction": direction,
            "margin": margin,
            "pass": margin >= 0.0,
        }

    add("episodes", int(metrics["n"]), thresholds["episodes_min"], ">=")
    add("sr", metrics["sr"], thresholds["sr_min"], ">=")
    add("cr", metrics["cr"], thresholds["cr_max"], "<=")
    add("to", metrics["to"], thresholds["to_max"], "<=")
    if scenario == "narrow_range":
        add(
            "crossing_rate",
            metrics["crossing_rate"],
            thresholds["crossing_min"],
            ">=",
        )
        add(
            "direct_crossing_rate",
            metrics["direct_crossing_rate"],
            thresholds["direct_crossing_min"],
            ">=",
        )
    worst = min(checks, key=lambda key: checks[key]["margin"])
    return {
        "thresholds": thresholds,
        "checks": checks,
        "worst_check": worst,
        "worst_margin": checks[worst]["margin"],
        "threshold_pass": all(check["pass"] for check in checks.values()),
    }


def _validate_cell(payload: dict, candidate: dict, scenario: str) -> None:
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    if payload.get("checkpoint_name") != candidate["name"]:
        raise ValueError("cell checkpoint name mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    if payload.get("scenario") != scenario:
        raise ValueError("cell scenario mismatch")
    if not payload.get("cell_valid"):
        raise ValueError("cell is not structurally valid")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")


def summarize_phase_a(payloads: Iterable[dict]) -> list[dict]:
    items = list(payloads)
    expected = len(CANDIDATES) * len(CORRIDOR_SCENARIOS)
    if len(items) != expected:
        raise ValueError(f"Phase A needs exactly {expected} cells, got {len(items)}")
    summaries = []
    for candidate in CANDIDATES:
        by_scenario = {
            item["scenario"]: item
            for item in items
            if item.get("checkpoint_name") == candidate["name"]
        }
        if set(by_scenario) != set(CORRIDOR_SCENARIOS):
            raise ValueError(
                f"{candidate['name']} Phase-A scenarios mismatch: "
                f"{sorted(by_scenario)}"
            )
        for scenario in CORRIDOR_SCENARIOS:
            _validate_cell(by_scenario[scenario], candidate, scenario)
        cr = {
            scenario: float(by_scenario[scenario]["metrics"]["cr"])
            for scenario in CORRIDOR_SCENARIOS
        }
        worst = max(cr.values())
        mean = sum(cr.values()) / len(cr)
        summaries.append(
            {
                "checkpoint_name": candidate["name"],
                "conceptual_iteration": candidate["conceptual_iteration"],
                "checkpoint_sha256": candidate["sha256"],
                "family_cr": cr,
                "family_sr": {
                    scenario: float(by_scenario[scenario]["metrics"]["sr"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "family_to": {
                    scenario: float(by_scenario[scenario]["metrics"]["to"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "family_episodes": {
                    scenario: int(by_scenario[scenario]["metrics"]["n"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "worst_family": max(cr, key=cr.get),
                "worst_family_cr": worst,
                "mean_family_cr": mean,
                "all_corridor_hard_pass": all(
                    bool(by_scenario[scenario]["threshold_pass"])
                    for scenario in CORRIDOR_SCENARIOS
                ),
                "all_cells_valid": True,
            }
        )
    return summaries


def rank_phase_a(payloads: Iterable[dict]) -> dict:
    summaries = summarize_phase_a(payloads)
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["worst_family_cr"],
            row["mean_family_cr"],
            -row["conceptual_iteration"],
        ),
    )
    best = ranked[0]["worst_family_cr"]
    tie_group = [
        row["checkpoint_name"]
        for row in ranked
        if row["worst_family_cr"] - best < TIE_MARGIN
    ]
    return {
        "summaries": summaries,
        "ranked": [row["checkpoint_name"] for row in ranked],
        "top_two": [row["checkpoint_name"] for row in ranked[:2]],
        "leader_tie_group": tie_group,
        "tie_margin": TIE_MARGIN,
    }


def final_verdict(phase_a_payloads: Iterable[dict], phase_b_payloads: Iterable[dict]) -> dict:
    phase_a = rank_phase_a(phase_a_payloads)
    top_two = phase_a["top_two"]
    items = list(phase_b_payloads)
    expected = len(top_two) * len(RETENTION_SCENARIOS)
    if len(items) != expected:
        raise ValueError(f"Phase B needs exactly {expected} cells, got {len(items)}")

    candidate_results = []
    phase_a_by_name = {
        row["checkpoint_name"]: row for row in phase_a["summaries"]
    }
    for name in top_two:
        candidate = candidate_by_name(name)
        by_scenario = {
            item["scenario"]: item
            for item in items
            if item.get("checkpoint_name") == name
        }
        if set(by_scenario) != set(RETENTION_SCENARIOS):
            raise ValueError(f"{name} Phase-B scenarios mismatch")
        for scenario in RETENTION_SCENARIOS:
            _validate_cell(by_scenario[scenario], candidate, scenario)
        corridor_pass = phase_a_by_name[name]["all_corridor_hard_pass"]
        retention_pass = all(
            bool(by_scenario[scenario]["threshold_pass"])
            for scenario in RETENTION_SCENARIOS
        )
        candidate_results.append(
            {
                "checkpoint_name": name,
                "corridor_hard_pass": corridor_pass,
                "retention_hard_pass": retention_pass,
                "all_six_hard_pass": corridor_pass and retention_pass,
                "native": by_scenario["nav_native"]["metrics"],
                "narrow": by_scenario["narrow_range"]["metrics"],
            }
        )
    eligible = [
        result["checkpoint_name"]
        for result in candidate_results
        if result["all_six_hard_pass"]
    ]
    recommendation = next(
        (name for name in phase_a["ranked"] if name in eligible), None
    )
    return {
        "schema": "sa5_checkpoint_screen_verdict/v1",
        "phase_a": phase_a,
        "phase_b_candidates": candidate_results,
        "screen_eligible_checkpoints": eligible,
        "recommended_for_human_consideration": recommendation,
        "accepted_parent": False,
        "training_started": False,
        "sa6_started": False,
    }
