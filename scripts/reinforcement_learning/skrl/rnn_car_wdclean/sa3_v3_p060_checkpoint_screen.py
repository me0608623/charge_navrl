"""Frozen SA3-v3 P060 low-density checkpoint-screen protocol.

This module is pure: it defines candidate identity, fixed cells, validation,
and ranking. The queue and Isaac Sim cell runner live in separate modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

RUN_NAME = "sa3_speed_density_v3_from_sa2r1_c100_ne1024_s42_p300_r1"
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME

STAGE = 3
SEED = 818
NUM_ENVS = 64
STEPS = 2500
MIN_EPISODES = 1000
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
PEDESTRIAN_SPEED_LABEL = "P060"
PEDESTRIAN_SPEED_RANGE_M_S = (0.50, 0.70)
MOTION_MODE = "lateral"
TIE_MARGIN = 0.005

THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}

DENSITY_ARMS = (
    {
        "label": "0S1D",
        "scenario": "corridor_low_0s1d",
        "static_obstacles": 0,
        "dynamic_obstacles": 1,
    },
    {
        "label": "1S1D",
        "scenario": "corridor_low_1s1d",
        "static_obstacles": 1,
        "dynamic_obstacles": 1,
    },
)

CANDIDATES = (
    {
        "name": "c100",
        "conceptual_iteration": 100,
        "filename": "checkpoint_12800.pt",
        "sha256": "87eb7fcc88179875b9a545f904c078fc546e3e0d162e72b3c5f67835053be6ae",
    },
    {
        "name": "c150",
        "conceptual_iteration": 150,
        "filename": "checkpoint_19200.pt",
        "sha256": "055ad9f6be976868e478ed9c44deb62fa9b073903bffa221d809e764752c7d9a",
    },
    {
        "name": "c200",
        "conceptual_iteration": 200,
        "filename": "checkpoint_25600.pt",
        "sha256": "32d368e2c9e5bbca2abd11b2768e8b865148e6647ae22a8e7a281e843e9e2ded",
    },
    {
        "name": "c250",
        "conceptual_iteration": 250,
        "filename": "checkpoint_32000.pt",
        "sha256": "1465abe11436a823e190bb8a75bf1f4773460b43844f1553eefbca75fb7dd779",
    },
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "filename": "checkpoint_38400.pt",
        "sha256": "6888d4c4759413ddc81f1e1eb13f6fb02244823ea1977a97705f4ee2e5c79122",
    },
)

EXCLUDED_CANDIDATES = (
    {
        "name": "c50",
        "conceptual_iteration": 50,
        "filename": "checkpoint_6400.pt",
        "sha256": "71df125ad753c38f17fd07fd7e6202eed9e09a808c890d0bc2574bb197d6f9ea",
        "reason": (
            "pre-screen training-window worst-direction CR was about 36 "
            "percentage points behind the leading cluster; retained on disk "
            "but omitted from the fixed screen to reduce GPU cost"
        ),
    },
)


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown SA3-v3 candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return RUN_DIR / str(data["filename"])


def density_by_label(label: str) -> dict:
    for density in DENSITY_ARMS:
        if density["label"] == label:
            return dict(density)
    raise ValueError(f"unknown density arm {label!r}")


def cell_label(checkpoint_name: str, density_label: str) -> str:
    candidate_by_name(checkpoint_name)
    density_by_label(density_label)
    return f"{checkpoint_name}_{density_label.lower()}_p060_lateral_s070"


def cells() -> tuple[dict, ...]:
    return tuple(
        {
            "cell": cell_label(candidate["name"], density["label"]),
            "checkpoint_name": candidate["name"],
            "density": density["label"],
            "scenario": density["scenario"],
        }
        for candidate in CANDIDATES
        for density in DENSITY_ARMS
    )


def _protocol_body() -> dict:
    return {
        "schema": "sa3_v3_p060_checkpoint_screen_protocol/v1",
        "purpose": (
            "rank the retained SA3-v3 checkpoints on the frozen speed_rate=0.7 "
            "P060 0S1D/1S1D lateral graduation cells"
        ),
        "run_name": RUN_NAME,
        "stage": STAGE,
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "excluded_candidates": [dict(candidate) for candidate in EXCLUDED_CANDIDATES],
        "cells": [dict(cell) for cell in cells()],
        "fixed_evaluation": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": STEPS,
            "minimum_completed_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "pedestrian_speed_label": PEDESTRIAN_SPEED_LABEL,
            "pedestrian_speed_range_m_s": list(PEDESTRIAN_SPEED_RANGE_M_S),
            "motion_mode": MOTION_MODE,
            "density_arms": [dict(density) for density in DENSITY_ARMS],
        },
        "thresholds": dict(THRESHOLDS),
        "ranking": {
            "hard_gate": "both fixed cells must meet SR/CR/TO and n",
            "primary": "minimize max(CR_0S1D, CR_1S1D)",
            "secondary": "minimize mean(CR_0S1D, CR_1S1D)",
            "tertiary": "prefer later checkpoint only after exact score ties",
            "descriptive_tie_margin": TIE_MARGIN,
            "promotion": "top two valid checkpoints proceed to retention",
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "checkpoint screen; it ranks candidates for retention but does not "
            "accept an SA3 parent, replace the SA4 sentinel, or launch SA4"
        ),
        "forbidden": [
            "evaluate the omitted c50 as if it were part of this protocol",
            "change density, P060 speed, lateral motion, speed_rate, delay, or seed",
            "continue with partial or invalid cells",
            "auto-run retention",
            "replace the SA4 parent sentinel",
            "launch SA4",
        ],
    }


def screen_protocol() -> dict:
    payload = json.loads(json.dumps(_protocol_body(), sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(metrics: dict) -> dict:
    n = int(metrics.get("n", 0))
    checks = {
        "episodes": n >= MIN_EPISODES,
        "sr": _finite(metrics.get("sr"), "sr") >= THRESHOLDS["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= THRESHOLDS["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= THRESHOLDS["to_max"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != "sa3_v3_p060_checkpoint_screen_cell/v1":
        raise ValueError("unexpected SA3-v3 checkpoint-screen cell schema")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    density = density_by_label(str(payload.get("density")))
    if payload.get("cell") != cell_label(candidate["name"], density["label"]):
        raise ValueError("cell label mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    if payload.get("scenario") != density["scenario"]:
        raise ValueError("cell scenario mismatch")
    if int(payload.get("stage", -1)) != STAGE:
        raise ValueError("cell stage mismatch")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("cell seed mismatch")
    if int(payload.get("delay_steps", -1)) != DELAY_STEPS:
        raise ValueError("cell delay mismatch")
    if _finite(payload.get("speed_rate"), "speed_rate") != SPEED_RATE:
        raise ValueError("cell speed_rate mismatch")
    if list(payload.get("pedestrian_speed_range_m_s") or []) != list(
        PEDESTRIAN_SPEED_RANGE_M_S
    ):
        raise ValueError("cell pedestrian-speed range mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    verdict = evaluate_metrics(metrics)
    if payload.get("gate") != verdict:
        raise ValueError("cell gate payload mismatch")
    report = payload.get("corridor_report") or {}
    if int(report.get("configured_static_obstacles", -1)) != density[
        "static_obstacles"
    ]:
        raise ValueError("runtime static density mismatch")
    if int(report.get("configured_dynamic_obstacles", -1)) != density[
        "dynamic_obstacles"
    ]:
        raise ValueError("runtime dynamic density mismatch")
    if list(report.get("requested_dynamic_speed_range_m_s") or []) != list(
        PEDESTRIAN_SPEED_RANGE_M_S
    ):
        raise ValueError("runtime pedestrian-speed range mismatch")
    if report.get("dynamic_motion_mode") != MOTION_MODE:
        raise ValueError("runtime motion mode mismatch")
    if _finite(report.get("speed_rate"), "runtime speed_rate") != SPEED_RATE:
        raise ValueError("runtime speed_rate mismatch")


def _candidate_summary(candidate: dict, by_cell: dict[str, dict]) -> dict:
    rows = [
        by_cell[cell_label(candidate["name"], density["label"])]
        for density in DENSITY_ARMS
    ]
    cr_values = [float(row["metrics"]["cr"]) for row in rows]
    return {
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "cells": {
            row["density"]: {
                "n": int(row["metrics"]["n"]),
                "sr": float(row["metrics"]["sr"]),
                "cr": float(row["metrics"]["cr"]),
                "to": float(row["metrics"]["to"]),
                "gate_pass": bool(row["gate"]["pass"]),
            }
            for row in rows
        },
        "hard_gate_pass": all(bool(row["gate"]["pass"]) for row in rows),
        "worst_fixed_cell_cr": max(cr_values),
        "mean_fixed_cell_cr": sum(cr_values) / len(cr_values),
    }


def rank_summaries(summaries: list[dict]) -> list[dict]:
    return sorted(
        summaries,
        key=lambda row: (
            float(row["worst_fixed_cell_cr"]),
            float(row["mean_fixed_cell_cr"]),
            -int(row["conceptual_iteration"]),
        ),
    )


def summarize_cells(payloads: list[dict]) -> dict:
    expected = {cell["cell"] for cell in cells()}
    if len(payloads) != len(expected):
        raise ValueError(f"screen requires exactly {len(expected)} valid cells")
    by_cell: dict[str, dict] = {}
    for payload in payloads:
        validate_cell(payload)
        label = str(payload["cell"])
        if label in by_cell:
            raise ValueError(f"duplicate cell {label}")
        by_cell[label] = payload
    if set(by_cell) != expected:
        raise ValueError("checkpoint screen is missing a frozen cell")

    ranked = rank_summaries(
        [_candidate_summary(candidate, by_cell) for candidate in CANDIDATES]
    )
    best = float(ranked[0]["worst_fixed_cell_cr"])
    tie_group = [
        row["checkpoint_name"]
        for row in ranked
        if float(row["worst_fixed_cell_cr"]) - best < TIE_MARGIN
    ]
    return {
        "schema": "sa3_v3_p060_checkpoint_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "ranked_candidates": ranked,
        "descriptive_tie_group": tie_group,
        "promoted_for_retention": [
            row["checkpoint_name"] for row in ranked[:2]
        ],
        "all_candidates_hard_gate_pass": all(
            bool(row["hard_gate_pass"]) for row in ranked
        ),
        "accepted_parent": None,
        "retention_started": False,
        "sa4_parent_replaced": False,
        "sa4_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
