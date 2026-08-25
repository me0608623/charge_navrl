"""Frozen SA4-v3 checkpoint-screen protocol.

This module is pure: it fixes candidate identity, the three SA4 graduation
cells, validation, and ranking. GPU execution lives in the companion runner
and queue modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

RUN_NAME = "sa4_speed_density_v3_from_sa3r1_c300_ne1024_s42_p300_r1"
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME

STAGE = 4
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
MOTION_MODE = "lateral"
TIE_MARGIN = 0.005

THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}

CANDIDATES = (
    {
        "name": "c50",
        "conceptual_iteration": 50,
        "filename": "checkpoint_6400.pt",
        "sha256": "e1b5fb7f3cac946dc74454a648c617edb213a387f258067211e982157a6c18ab",
    },
    {
        "name": "c100",
        "conceptual_iteration": 100,
        "filename": "checkpoint_12800.pt",
        "sha256": "10b3bc4a8ad9527be6e1b5b960803543495c15846d0fd19a32174abd65c65d29",
    },
    {
        "name": "c150",
        "conceptual_iteration": 150,
        "filename": "checkpoint_19200.pt",
        "sha256": "7979676cc8df6392831bf06d24e31ceae60dcf8d7e5e0eda8852cf10f2101ee4",
    },
    {
        "name": "c200",
        "conceptual_iteration": 200,
        "filename": "checkpoint_25600.pt",
        "sha256": "b59c69be15171d53ad44fc7fdbe027b97f1e1958d026e9fa62978672737f4838",
    },
    {
        "name": "c250",
        "conceptual_iteration": 250,
        "filename": "checkpoint_32000.pt",
        "sha256": "0afe1924cfa2b7d82e719432dcd6cde3ea4c662bece29f7a3d519609ca0d0974",
    },
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "filename": "checkpoint_38400.pt",
        "sha256": "ccb95b1667e0f3f9330ffaed62ea8ad728221d270bf9b8a30b6135f421998e73",
    },
)

# These are copied verbatim from the frozen SA4 graduation contract in
# sim2real_speed_density_curriculum_v3_20260821.json.
FIXED_CELLS = (
    {
        "label": "0S1D_P080",
        "density": "0S1D",
        "pedestrian_speed_label": "P080",
        "pedestrian_speed_range_m_s": (0.70, 0.90),
        "scenario": "corridor_low_0s1d",
        "static_obstacles": 0,
        "dynamic_obstacles": 1,
    },
    {
        "label": "1S1D_P080",
        "density": "1S1D",
        "pedestrian_speed_label": "P080",
        "pedestrian_speed_range_m_s": (0.70, 0.90),
        "scenario": "corridor_low_1s1d",
        "static_obstacles": 1,
        "dynamic_obstacles": 1,
    },
    {
        "label": "2S1D_P060",
        "density": "2S1D",
        "pedestrian_speed_label": "P060",
        "pedestrian_speed_range_m_s": (0.50, 0.70),
        "scenario": "corridor_stage4_2s1d",
        "static_obstacles": 2,
        "dynamic_obstacles": 1,
    },
)


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown SA4-v3 candidate {name!r}")


def fixed_cell_by_label(label: str) -> dict:
    for spec in FIXED_CELLS:
        if spec["label"] == label:
            return dict(spec)
    raise ValueError(f"unknown SA4-v3 fixed cell {label!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return RUN_DIR / str(data["filename"])


def cell_label(checkpoint_name: str, fixed_cell_label: str) -> str:
    candidate_by_name(checkpoint_name)
    spec = fixed_cell_by_label(fixed_cell_label)
    return (
        f"{checkpoint_name}_{spec['density'].lower()}_"
        f"{spec['pedestrian_speed_label'].lower()}_lateral_s070"
    )


def cells() -> tuple[dict, ...]:
    return tuple(
        {
            "cell": cell_label(candidate["name"], spec["label"]),
            "checkpoint_name": candidate["name"],
            "fixed_cell": spec["label"],
            "density": spec["density"],
            "pedestrian_speed_label": spec["pedestrian_speed_label"],
            "scenario": spec["scenario"],
        }
        for candidate in CANDIDATES
        for spec in FIXED_CELLS
    )


def _protocol_body() -> dict:
    return {
        "schema": "sa4_v3_checkpoint_screen_protocol/v1",
        "purpose": (
            "rank all six completed SA4-v3 checkpoints on the three frozen "
            "speed_rate=0.7 lateral graduation cells before any continuation"
        ),
        "run_name": RUN_NAME,
        "stage": STAGE,
        "candidates": [dict(candidate) for candidate in CANDIDATES],
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
            "motion_mode": MOTION_MODE,
            "fixed_cells": [
                {
                    **dict(spec),
                    "pedestrian_speed_range_m_s": list(
                        spec["pedestrian_speed_range_m_s"]
                    ),
                }
                for spec in FIXED_CELLS
            ],
        },
        "thresholds": dict(THRESHOLDS),
        "ranking": {
            "hard_gate": "all three fixed cells must meet SR/CR/TO and n",
            "primary": "minimize worst CR across the three fixed cells",
            "secondary": "minimize mean CR across the three fixed cells",
            "tertiary": "prefer later checkpoint only after exact score ties",
            "descriptive_tie_margin": TIE_MARGIN,
            "promotion": "top two valid checkpoints proceed to retention",
        },
        "retention_after_promotion": [
            "native",
            "narrow",
            "P060 low-density retention",
        ],
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "checkpoint screen; it ranks candidates for retention but does not "
            "authorize continuation, accept an SA5 parent, or launch SA5"
        ),
        "forbidden": [
            "omit any of the six checkpoints or three frozen cells",
            "change density, speed range, lateral motion, speed_rate, delay, or seed",
            "continue with partial, duplicate, or invalid cells",
            "auto-run retention",
            "auto-start a 50-iteration continuation",
            "accept an SA5 parent or launch SA5",
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
    if payload.get("schema") != "sa4_v3_checkpoint_screen_cell/v1":
        raise ValueError("unexpected SA4-v3 checkpoint-screen cell schema")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    spec = fixed_cell_by_label(str(payload.get("fixed_cell")))
    if payload.get("cell") != cell_label(candidate["name"], spec["label"]):
        raise ValueError("cell label mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    if payload.get("scenario") != spec["scenario"]:
        raise ValueError("cell scenario mismatch")
    if payload.get("density") != spec["density"]:
        raise ValueError("cell density mismatch")
    if int(payload.get("stage", -1)) != STAGE:
        raise ValueError("cell stage mismatch")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("cell seed mismatch")
    if int(payload.get("delay_steps", -1)) != DELAY_STEPS:
        raise ValueError("cell delay mismatch")
    if _finite(payload.get("speed_rate"), "speed_rate") != SPEED_RATE:
        raise ValueError("cell speed_rate mismatch")
    if payload.get("motion_mode") != MOTION_MODE:
        raise ValueError("cell motion-mode mismatch")
    if list(payload.get("pedestrian_speed_range_m_s") or []) != list(
        spec["pedestrian_speed_range_m_s"]
    ):
        raise ValueError("cell pedestrian-speed range mismatch")

    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    verdict = evaluate_metrics(metrics)
    if payload.get("gate") != verdict:
        raise ValueError("cell gate payload mismatch")

    report = payload.get("corridor_report") or {}
    if int(report.get("configured_static_obstacles", -1)) != spec[
        "static_obstacles"
    ]:
        raise ValueError("runtime static density mismatch")
    if int(report.get("configured_dynamic_obstacles", -1)) != spec[
        "dynamic_obstacles"
    ]:
        raise ValueError("runtime dynamic density mismatch")
    if list(report.get("requested_dynamic_speed_range_m_s") or []) != list(
        spec["pedestrian_speed_range_m_s"]
    ):
        raise ValueError("runtime pedestrian-speed range mismatch")
    if report.get("dynamic_motion_mode") != MOTION_MODE:
        raise ValueError("runtime motion mode mismatch")
    if _finite(report.get("speed_rate"), "runtime speed_rate") != SPEED_RATE:
        raise ValueError("runtime speed_rate mismatch")


def _candidate_summary(candidate: dict, by_cell: dict[str, dict]) -> dict:
    rows = [
        by_cell[cell_label(candidate["name"], spec["label"])]
        for spec in FIXED_CELLS
    ]
    cr_values = [float(row["metrics"]["cr"]) for row in rows]
    return {
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "cells": {
            row["fixed_cell"]: {
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
        "schema": "sa4_v3_checkpoint_screen_summary/v1",
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
        "retention_started": False,
        "continuation_authorized": False,
        "continuation_started": False,
        "accepted_sa5_parent": None,
        "sa5_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
