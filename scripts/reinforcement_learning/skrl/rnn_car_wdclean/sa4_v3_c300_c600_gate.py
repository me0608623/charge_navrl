"""Frozen c300-c600 SA4-v3 Gate comparison protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa4_v3_checkpoint_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

STAGE = base.STAGE
SEED = base.SEED
NUM_ENVS = base.NUM_ENVS
STEPS = base.STEPS
MIN_EPISODES = base.MIN_EPISODES
DELAY_STEPS = base.DELAY_STEPS
DELAY_MS = base.DELAY_MS
ACTUATOR_PROFILE = base.ACTUATOR_PROFILE
LIDAR_NOISE_MODE = base.LIDAR_NOISE_MODE
LIDAR_DISTRACTOR_ELIGIBILITY = base.LIDAR_DISTRACTOR_ELIGIBILITY
SPEED_RATE = base.SPEED_RATE
SPEED_RATE_OBS = base.SPEED_RATE_OBS
DEPLOYMENT_SPEED_SCALE = base.DEPLOYMENT_SPEED_SCALE
MOTION_MODE = base.MOTION_MODE
TIE_MARGIN = base.TIE_MARGIN
THRESHOLDS = dict(base.THRESHOLDS)
FIXED_CELLS = tuple(dict(spec) for spec in base.FIXED_CELLS)

SOURCE_GATE_FREEZE = REPO / "docs/freeze/sa4_v3_checkpoint_screen_v1.json"
SOURCE_GATE_FREEZE_SHA256 = (
    "9031081d2d84d7ebb855a87f4d225d1742a7a85c44928137f9ab2987fafa41ef"
)
SOURCE_GATE_PROTOCOL_SHA256 = (
    "d2be765424df30a84246186c7f3ad2777dc93e1d04906adbc2c2fa76ef41a8ee"
)
CONTINUATION_AUTHORIZATION = (
    REPO / "docs/freeze/sa4_v3_cont300_from_c300_v1.json"
)
CONTINUATION_AUTHORIZATION_SHA256 = (
    "acdc9b1c4f775b28d80c7a376e3cfdfbe190ab60675c2ab4b4aadfea224bfe99"
)

CANDIDATES = (
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "path": (
            "logs/rnn_car/sa4_speed_density_v3_from_sa3r1_c300_"
            "ne1024_s42_p300_r1/checkpoint_38400.pt"
        ),
        "sha256": "ccb95b1667e0f3f9330ffaed62ea8ad728221d270bf9b8a30b6135f421998e73",
    },
    {
        "name": "c350",
        "conceptual_iteration": 350,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_6400.pt"
        ),
        "sha256": "359afe7b5b313f578e7ea80c92ee33615181d7738598c9086b8cc0fce859c65e",
    },
    {
        "name": "c400",
        "conceptual_iteration": 400,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_12800.pt"
        ),
        "sha256": "cdd6386e9a71940168c57bd2862e98d117b41a5d7724759b88e2d84f6a574ffb",
    },
    {
        "name": "c450",
        "conceptual_iteration": 450,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_19200.pt"
        ),
        "sha256": "08ac91df310cafc0828a6d683e39ad4d97f3dfe665b202e64048f4a0685bcbca",
    },
    {
        "name": "c500",
        "conceptual_iteration": 500,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_25600.pt"
        ),
        "sha256": "e27677349eba8f37e2e6937f7d92400a1550e58d89d4430e556932697e9cb9f4",
    },
    {
        "name": "c550",
        "conceptual_iteration": 550,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_32000.pt"
        ),
        "sha256": "235a79a41b1d83510bbbb314c965a7ef5ba73e898aec121310a92965092c4ba8",
    },
    {
        "name": "c600",
        "conceptual_iteration": 600,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_38400.pt"
        ),
        "sha256": "7c849d0b0e33846d228ae7f2914dd8a868b6beb590d65b8e4b528c0e32e820cc",
    },
)


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown c300-c600 candidate {name!r}")


def fixed_cell_by_label(label: str) -> dict:
    return base.fixed_cell_by_label(label)


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return REPO / str(data["path"])


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
        "schema": "sa4_v3_c300_c600_gate_protocol/v1",
        "purpose": (
            "compare conceptual c300-c600 on the unchanged three-cell "
            "SA4-v3 lateral Gate after exact optimizer continuation"
        ),
        "source_gate": {
            "path": str(SOURCE_GATE_FREEZE.relative_to(REPO)),
            "file_sha256": SOURCE_GATE_FREEZE_SHA256,
            "protocol_sha256": SOURCE_GATE_PROTOCOL_SHA256,
        },
        "continuation_authorization": {
            "path": str(CONTINUATION_AUTHORIZATION.relative_to(REPO)),
            "sha256": CONTINUATION_AUTHORIZATION_SHA256,
        },
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "cells": [dict(cell) for cell in cells()],
        "fixed_evaluation": {
            "stage": STAGE,
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
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "Gate comparison; ranking does not itself accept an SA5 parent"
        ),
        "forbidden": [
            "change any source Gate condition or threshold",
            "continue after a missing, duplicate, invalid, or drifted cell",
            "auto-run retention or training",
            "auto-accept a parent or launch SA5",
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
    return base.evaluate_metrics(metrics)


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != "sa4_v3_checkpoint_screen_cell/v1":
        raise ValueError("unexpected c300-c600 Gate cell schema")
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
        raise ValueError("cell Gate payload mismatch")

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
        raise ValueError(f"Gate comparison requires exactly {len(expected)} cells")
    by_cell: dict[str, dict] = {}
    for payload in payloads:
        validate_cell(payload)
        label = str(payload["cell"])
        if label in by_cell:
            raise ValueError(f"duplicate cell {label}")
        by_cell[label] = payload
    if set(by_cell) != expected:
        raise ValueError("Gate comparison is missing a frozen cell")

    ranked = rank_summaries(
        [_candidate_summary(candidate, by_cell) for candidate in CANDIDATES]
    )
    best = float(ranked[0]["worst_fixed_cell_cr"])
    return {
        "schema": "sa4_v3_c300_c600_gate_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_GATE_COMPARISON",
        "protocol_sha256": screen_protocol()["sha256"],
        "ranked_candidates": ranked,
        "descriptive_tie_group": [
            row["checkpoint_name"]
            for row in ranked
            if float(row["worst_fixed_cell_cr"]) - best < TIE_MARGIN
        ],
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

