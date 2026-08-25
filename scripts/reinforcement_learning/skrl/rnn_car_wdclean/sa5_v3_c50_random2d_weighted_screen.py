"""Frozen 8-cell screen for the bounded c50 random-2D exposure pilot.

This evaluation compares the unchanged c50 diagnostic anchor with the
25-iteration weighted candidate on four fixed 4S2D motion families. It starts
no training and cannot graduate SA5 or authorize SA6.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import sa5_v3_c50_it50_direction_ab as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

SCREEN_ROOT = REPO / "logs/gates/sa5_v3_c50_random2d_weighted_screen/screen_20260824_r1"
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c50_random2d_weighted_screen_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "bca66d805048218c2fed6dbee3adf88ec8393a1eb013e6636f8e807604b9e073"
)

# Exact reuse of the established 4S2D evaluator contract.
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
DYNAMIC_SPEED_RANGE = base.DYNAMIC_SPEED_RANGE
STATIC_COUNT = base.STATIC_COUNT
DYNAMIC_COUNT = base.DYNAMIC_COUNT
MOTION_MODES = base.MOTION_MODES
SCENARIO_SPECS = base.SCENARIO_SPECS
ALL_SCENARIOS = base.ALL_SCENARIOS
CORRIDOR_SCENARIOS = base.CORRIDOR_SCENARIOS
LOW_DENSITY_SCENARIOS = base.LOW_DENSITY_SCENARIOS
STEPS_BY_SCENARIO = base.STEPS_BY_SCENARIO
THRESHOLDS = base.THRESHOLDS
BEHAVIOR_KEYS = base.BEHAVIOR_KEYS
ANCHOR_REFERENCE = base.ANCHOR_REFERENCE

TARGET_MOTION_MODE = "random_2d"
RETENTION_MOTION_MODES = ("lateral", "longitudinal", "mixed")
MIN_ABS_Z = 2.0
MIN_TARGET_DELTA = 0.05
MAX_TARGET_TO = 0.05
MAX_DEGRADATION = 0.02

ANCHOR = dict(base.ANCHOR)
WEIGHTED_IT25 = {
    "name": "weighted_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_random2d_weighted_ne1024_s42_p25_r1/"
        "checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "cbc8a91a43eb0674dcd33118b3436f73f5d611b33860ebab5f7c83e6048f670f"
    ),
    "status": "UNSCREENED_DEVELOPMENT_CANDIDATE_NOT_PARENT",
}
CANDIDATE_SPECS = (ANCHOR, WEIGHTED_IT25)
_BY_NAME = {spec["name"]: spec for spec in CANDIDATE_SPECS}
CELL_SCHEMA = "sa5_v3_c50_random2d_weighted_screen_cell/v1"


sha256_of = base.sha256_of
scenario_by_name = base.scenario_by_name
_stage_scene = base._stage_scene
thresholds_for = base.thresholds_for
evaluate_metrics = base.evaluate_metrics
_finite = base._finite


def checkpoint_path(spec: dict | str = ANCHOR) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError("unknown weighted-screen checkpoint")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown weighted-screen checkpoint {name!r}")
    result = dict(spec)
    result["sha256"] = spec["expected_sha256"]
    return result


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_random2d_weighted_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "question": (
            "does bounded random-2D exposure improve fixed 4S2D random-2D "
            "without degrading lateral, longitudinal, or mixed retention"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "anchor": dict(ANCHOR),
            "candidate": dict(WEIGHTED_IT25),
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "matrix": {
            "static_obstacles": STATIC_COUNT,
            "dynamic_obstacles": DYNAMIC_COUNT,
            "motion_modes": list(MOTION_MODES),
            "checkpoints": [spec["name"] for spec in CANDIDATE_SPECS],
            "cells": [dict(row) for row in SCENARIO_SPECS],
            "cell_count": len(CANDIDATE_SPECS) * len(SCENARIO_SPECS),
        },
        "fixed_evaluation": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": STEPS,
            "minimum_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "dynamic_speed_range_m_s": list(DYNAMIC_SPEED_RANGE),
            "scene": _stage_scene(),
        },
        "thresholds": dict(THRESHOLDS),
        "acceptance": {
            "target_motion_mode": TARGET_MOTION_MODE,
            "minimum_target_sr_improvement": MIN_TARGET_DELTA,
            "minimum_target_cr_reduction": MIN_TARGET_DELTA,
            "minimum_absolute_z": MIN_ABS_Z,
            "target_to_max": MAX_TARGET_TO,
            "target_to_degradation_max": MAX_DEGRADATION,
            "retention_motion_modes": list(RETENTION_MOTION_MODES),
            "retention_sr_degradation_max": MAX_DEGRADATION,
            "retention_cr_degradation_max": MAX_DEGRADATION,
            "retention_to_degradation_max": MAX_DEGRADATION,
            "required_behavior_metrics": list(BEHAVIOR_KEYS),
        },
        "decision_rules": {
            "on_pass": (
                "RUN_NATIVE_NARROW_AND_P060_RETENTION_BEFORE_ANY_EXTENSION"
            ),
            "on_target_failure": "REJECT_WEIGHT_ONLY_HYPOTHESIS_DO_NOT_EXTEND",
            "on_retention_failure": (
                "REJECT_WEIGHTED_CANDIDATE_DUE_TO_DIRECTION_RETENTION"
            ),
        },
        "screen_starts_training": False,
        "screen_starts_sa6": False,
        "screen_authorizes_graduation": False,
        "screen_authorizes_extension": False,
        "interpretation_limit": (
            "single evaluator seed evaluation-only screen; a pass only "
            "authorizes native, narrow, and P060 retention evaluation"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


@contextmanager
def _base_validation_context():
    saved = {
        "_BY_NAME": base._BY_NAME,
        "CELL_SCHEMA": base.CELL_SCHEMA,
        "screen_protocol": base.screen_protocol,
    }
    base._BY_NAME = _BY_NAME
    base.CELL_SCHEMA = CELL_SCHEMA
    base.screen_protocol = screen_protocol
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def validate_cell(payload: dict) -> None:
    with _base_validation_context():
        base.validate_cell(payload)


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _compare(label: str, before: dict, after: dict) -> dict:
    p0, p1 = float(before[label]), float(after[label])
    delta = p1 - p0
    se = math.hypot(_se(p0, int(before["n"])), _se(p1, int(after["n"])))
    z = delta / se if se > 0 else 0.0
    return {
        "metric": label,
        "anchor": p0,
        "weighted_it25": p1,
        "delta": delta,
        "se_diff": se,
        "z": z,
    }


def _behavior(payload: dict) -> dict:
    return base._behavior(payload)


def _row(payload: dict) -> dict:
    validate_cell(payload)
    scenario = scenario_by_name(str(payload["scenario"]))
    metrics = payload["metrics"]
    report = payload["corridor_report"]
    return {
        "checkpoint_name": str(payload["checkpoint_name"]),
        "motion_mode": scenario["motion_mode"],
        "scenario": scenario["name"],
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "gate_pass": bool(payload["threshold_pass"]),
        "wall_cr": _finite(report["wall_collision_rate"], "wall_cr"),
        "obstacle_cr": _finite(report["obstacle_collision_rate"], "obstacle_cr"),
        "behavior": _behavior(payload),
        "layouts": report.get("static_layout_outcomes") or {},
    }


def _anchor_replication(by_key: dict) -> tuple[list[dict], bool]:
    rows = []
    for mode, reference in ANCHOR_REFERENCE["cells"].items():
        observed = by_key[(ANCHOR["name"], mode)]
        identical = (
            observed["n"] == reference["n"]
            and abs(observed["cr"] - reference["cr"]) < 1e-12
        )
        rows.append(
            {
                "motion_mode": mode,
                "reference_n": reference["n"],
                "observed_n": observed["n"],
                "reference_cr": reference["cr"],
                "observed_cr": observed["cr"],
                "identical": identical,
            }
        )
    return rows, all(row["identical"] for row in rows)


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(SCENARIO_SPECS)
    if len(payloads) != expected:
        raise ValueError(f"expected exactly {expected} weighted-screen cells")
    rows = [_row(payload) for payload in payloads]
    by_key = {(row["checkpoint_name"], row["motion_mode"]): row for row in rows}
    if len(by_key) != expected:
        raise ValueError("weighted-screen cells are duplicated or missing")

    comparisons = []
    for mode in MOTION_MODES:
        anchor = by_key[(ANCHOR["name"], mode)]
        candidate = by_key[(WEIGHTED_IT25["name"], mode)]
        comparisons.append(
            {
                "motion_mode": mode,
                "anchor": {key: anchor[key] for key in ("n", "sr", "cr", "to", "behavior")},
                "weighted_it25": {
                    key: candidate[key]
                    for key in ("n", "sr", "cr", "to", "behavior")
                },
                "sr": _compare("sr", anchor, candidate),
                "cr": _compare("cr", anchor, candidate),
                "to": _compare("to", anchor, candidate),
            }
        )

    target = next(row for row in comparisons if row["motion_mode"] == TARGET_MOTION_MODE)
    target_checks = {
        "sr_improves": (
            target["sr"]["delta"] >= MIN_TARGET_DELTA
            and target["sr"]["z"] >= MIN_ABS_Z
        ),
        "cr_reduces": (
            target["cr"]["delta"] <= -MIN_TARGET_DELTA
            and target["cr"]["z"] <= -MIN_ABS_Z
        ),
        "to_absolute": target["weighted_it25"]["to"] <= MAX_TARGET_TO,
        "to_not_degraded": target["to"]["delta"] <= MAX_DEGRADATION,
    }
    target_pass = all(target_checks.values())

    retention = []
    for row in comparisons:
        if row["motion_mode"] not in RETENTION_MOTION_MODES:
            continue
        checks = {
            "sr": row["sr"]["delta"] >= -MAX_DEGRADATION,
            "cr": row["cr"]["delta"] <= MAX_DEGRADATION,
            "to": row["to"]["delta"] <= MAX_DEGRADATION,
        }
        retention.append(
            {
                "motion_mode": row["motion_mode"],
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    retention_pass = all(row["pass"] for row in retention)
    acceptance_observed = target_pass and retention_pass

    if not target_pass:
        next_action = "REJECT_WEIGHT_ONLY_HYPOTHESIS_DO_NOT_EXTEND"
    elif not retention_pass:
        next_action = "REJECT_WEIGHTED_CANDIDATE_DUE_TO_DIRECTION_RETENTION"
    else:
        next_action = "RUN_NATIVE_NARROW_AND_P060_RETENTION_BEFORE_ANY_EXTENSION"

    behavior_delta = {
        key: (
            target["weighted_it25"]["behavior"][key]
            - target["anchor"]["behavior"][key]
        )
        for key in BEHAVIOR_KEYS
    }
    conservative_shift_descriptive = (
        behavior_delta["stop_command_fraction"] > 0.05
        or behavior_delta["reverse_command_fraction"] > 0.05
        or behavior_delta["linear_speed_abs_mean_mps"] < -0.05
    )
    replication, replication_exact = _anchor_replication(by_key)

    return {
        "schema": "sa5_v3_c50_random2d_weighted_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_EVALUATION_ONLY_SCREEN",
        "anchor_status": ANCHOR["status"],
        "candidate_status": WEIGHTED_IT25["status"],
        "rows": sorted(
            rows,
            key=lambda row: (
                row["checkpoint_name"],
                MOTION_MODES.index(row["motion_mode"]),
            ),
        ),
        "comparisons": comparisons,
        "target_motion_mode": TARGET_MOTION_MODE,
        "target_checks": target_checks,
        "target_pass": target_pass,
        "retention": retention,
        "retention_pass": retention_pass,
        "pilot_acceptance_observed": acceptance_observed,
        "target_behavior_delta": behavior_delta,
        "conservative_shift_descriptive": conservative_shift_descriptive,
        "anchor_replication": replication,
        "anchor_replication_exact": replication_exact,
        "next_action": next_action,
        "selected_development_checkpoint": (
            WEIGHTED_IT25["name"] if acceptance_observed else None
        ),
        "selected_extension_checkpoint": None,
        "training_started": False,
        "extension_authorized": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
