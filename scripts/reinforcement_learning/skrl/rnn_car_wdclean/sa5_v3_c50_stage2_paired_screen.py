"""Frozen paired A2/B2 fixed 4S2D screen for mild exposure weighting."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import sa5_v3_c50_random2d_weighted_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCREEN_ROOT = (
    REPO / "logs/gates/sa5_v3_c50_stage2_paired_screen/screen_20260824_r2"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
AUTHORIZATION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_paired_p25_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "ae5bc7b00311a61a9c461f2441a6bd32002de23b03e48396cd3a86180d180950"
)
A2_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_a2_equal_p25_completion_20260824.json"
)
A2_COMPLETION_SHA256 = (
    "72d5d1e7f9a4cc6f47dc6d2284a28de8d18478e85b5aca5ee36f30acd0d6c5f6"
)
B2_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_b2_mild040_p25_completion_20260824.json"
)
B2_COMPLETION_SHA256 = (
    "452926026b680709b91df22cd14938c2dc2be9cc5541b6dae27820d4d4fc45d7"
)

# Exact reuse of the established fixed 4S2D evaluator contract.
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

TARGET_MOTION_MODE = "random_2d"
RETENTION_MOTION_MODES = ("lateral", "longitudinal", "mixed")
MIN_ABS_Z = 2.0
MIN_TARGET_DELTA = 0.02
MAX_TARGET_TO = 0.05
MAX_DEGRADATION = 0.02

A2_EQUAL = {
    "name": "stage2_a2_equal_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_stage2_equal_ne1024_s42_p25_r1/checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "0f3c9d47a267082c7ddc0dc309d39c390dd0d32eba09e7d0371261951ee17b52"
    ),
    "motion_weights": None,
    "status": "EQUALWEIGHT_CONTINUATION_CONTROL_NOT_PARENT",
}
B2_MILD = {
    "name": "stage2_b2_mild040_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_stage2_mild040_ne1024_s42_p25_r1/checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "45c972f804f1e9b24ade9b8164568cd8ac93f33e4f8b8685ee7347723b1aee30"
    ),
    "motion_weights": [0.30, 0.30, 0.40],
    "status": "MILD_WEIGHT_DEVELOPMENT_CANDIDATE_NOT_PARENT",
}
CANDIDATE_SPECS = (A2_EQUAL, B2_MILD)
_BY_NAME = {spec["name"]: spec for spec in CANDIDATE_SPECS}
CELL_SCHEMA = "sa5_v3_c50_stage2_paired_screen_cell/v1"

sha256_of = base.sha256_of
scenario_by_name = base.scenario_by_name
_stage_scene = base._stage_scene
thresholds_for = base.thresholds_for
evaluate_metrics = base.evaluate_metrics
_finite = base._finite


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


for _path, _expected, _label in (
    (AUTHORIZATION, AUTHORIZATION_SHA256, "paired authorization"),
    (A2_COMPLETION, A2_COMPLETION_SHA256, "A2 completion"),
    (B2_COMPLETION, B2_COMPLETION_SHA256, "B2 completion"),
    (REPO / A2_EQUAL["path"], A2_EQUAL["expected_sha256"], "A2 checkpoint"),
    (REPO / B2_MILD["path"], B2_MILD["expected_sha256"], "B2 checkpoint"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
_a2_completion = json.loads(A2_COMPLETION.read_text(encoding="utf-8"))
_b2_completion = json.loads(B2_COMPLETION.read_text(encoding="utf-8"))
if (
    _authorization.get("decision")
    != "RUN_PAIRED_STAGE2_EQUAL_VS_MILD_030_030_040_P25"
    or (_authorization.get("post_training_screen") or {}).get("sa6_auto_start")
    is not False
    or _a2_completion.get("status")
    != "COMPLETE_VALID_PAIRED_A2_AWAITING_B2_AND_FIXED_SCREEN"
    or _b2_completion.get("status")
    != "COMPLETE_VALID_PAIRED_B2_AWAITING_FIXED_SCREEN"
    or (_a2_completion.get("checkpoint") or {}).get("sha256")
    != A2_EQUAL["expected_sha256"]
    or (_b2_completion.get("checkpoint") or {}).get("sha256")
    != B2_MILD["expected_sha256"]
):
    raise RuntimeError("frozen evidence does not authorize paired fixed screen")


def checkpoint_path(spec: dict | str = A2_EQUAL) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError("unknown paired-screen checkpoint")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown paired-screen checkpoint {name!r}")
    result = dict(spec)
    result["sha256"] = spec["expected_sha256"]
    return result


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_stage2_paired_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "completion_evidence": {
            "a2": str(A2_COMPLETION.relative_to(REPO)),
            "a2_sha256": A2_COMPLETION_SHA256,
            "b2": str(B2_COMPLETION.relative_to(REPO)),
            "b2_sha256": B2_COMPLETION_SHA256,
        },
        "question": (
            "does 30/30/40 mild motion weighting improve fixed random-2D "
            "relative to an equal-allocation continuation from the same parent"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "a2": dict(A2_EQUAL),
            "b2": dict(B2_MILD),
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
        "acceptance": {
            "target_motion_mode": TARGET_MOTION_MODE,
            "minimum_target_sr_improvement": MIN_TARGET_DELTA,
            "minimum_target_cr_reduction": MIN_TARGET_DELTA,
            "minimum_absolute_z": MIN_ABS_Z,
            "maximum_target_to": MAX_TARGET_TO,
            "maximum_target_to_degradation": MAX_DEGRADATION,
            "retention_motion_modes": list(RETENTION_MOTION_MODES),
            "maximum_retention_sr_degradation": MAX_DEGRADATION,
            "maximum_retention_cr_degradation": MAX_DEGRADATION,
            "maximum_retention_to_degradation": MAX_DEGRADATION,
        },
        "decision_rules": {
            "on_target_and_retention_pass": (
                "RUN_NATIVE_NARROW_P060_RETENTION_BEFORE_ANY_EXTENSION"
            ),
            "on_target_failure": (
                "MILD_030_030_040_TARGET_NOT_SUPPORTED_STOP_WEIGHT_TUNING"
            ),
            "on_retention_failure": (
                "MILD_030_030_040_CAPABILITY_REDISTRIBUTION_REJECT"
            ),
        },
        "screen_starts_training": False,
        "screen_starts_sa6": False,
        "screen_authorizes_extension": False,
        "screen_authorizes_graduation": False,
        "interpretation_limit": _authorization["interpretation_limit"],
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


def _contrast(metric: str, before: dict, after: dict) -> dict:
    p0, p1 = float(before[metric]), float(after[metric])
    delta = p1 - p0
    se = math.hypot(_se(p0, int(before["n"])), _se(p1, int(after["n"])))
    return {
        "metric": metric,
        "a2_equal": p0,
        "b2_mild": p1,
        "delta": delta,
        "se_diff": se,
        "z": delta / se if se > 0 else 0.0,
    }


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
        "obstacle_cr": _finite(
            report["obstacle_collision_rate"], "obstacle_cr"
        ),
        "behavior": base._behavior(payload),
    }


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(SCENARIO_SPECS)
    if len(payloads) != expected:
        raise ValueError(f"expected exactly {expected} paired-screen cells")
    rows = [_row(payload) for payload in payloads]
    by_key = {(row["checkpoint_name"], row["motion_mode"]): row for row in rows}
    if len(by_key) != expected:
        raise ValueError("paired-screen cells are duplicated or missing")

    comparisons = []
    for mode in MOTION_MODES:
        a2 = by_key[(A2_EQUAL["name"], mode)]
        b2 = by_key[(B2_MILD["name"], mode)]
        comparisons.append(
            {
                "motion_mode": mode,
                "a2_equal": {
                    key: a2[key] for key in ("n", "sr", "cr", "to", "behavior")
                },
                "b2_mild": {
                    key: b2[key] for key in ("n", "sr", "cr", "to", "behavior")
                },
                "sr": _contrast("sr", a2, b2),
                "cr": _contrast("cr", a2, b2),
                "to": _contrast("to", a2, b2),
            }
        )

    target = next(
        row for row in comparisons if row["motion_mode"] == TARGET_MOTION_MODE
    )
    target_checks = {
        "sr_improves": (
            target["sr"]["delta"] >= MIN_TARGET_DELTA
            and target["sr"]["z"] >= MIN_ABS_Z
        ),
        "cr_reduces": (
            target["cr"]["delta"] <= -MIN_TARGET_DELTA
            and target["cr"]["z"] <= -MIN_ABS_Z
        ),
        "to_absolute": target["b2_mild"]["to"] <= MAX_TARGET_TO,
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
    accepted = target_pass and retention_pass
    if not target_pass:
        next_action = (
            "MILD_030_030_040_TARGET_NOT_SUPPORTED_STOP_WEIGHT_TUNING"
        )
    elif not retention_pass:
        next_action = "MILD_030_030_040_CAPABILITY_REDISTRIBUTION_REJECT"
    else:
        next_action = "RUN_NATIVE_NARROW_P060_RETENTION_BEFORE_ANY_EXTENSION"

    return {
        "schema": "sa5_v3_c50_stage2_paired_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_TRAINING_SEED_PAIRED_SCREEN",
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
        "mild_weight_acceptance_observed": accepted,
        "next_action": next_action,
        "selected_development_checkpoint": B2_MILD["name"] if accepted else None,
        "selected_extension_checkpoint": None,
        "training_started": False,
        "extension_authorized": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
