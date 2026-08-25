"""Frozen 4-cell matched-control screen with a three-arm analysis.

Only the equal-weight control is newly evaluated. The anchor and weighted
cells are reused by exact hash from the completed 8-cell screen. This module
starts no training and cannot graduate SA5 or authorize SA6.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import sa5_v3_c50_random2d_weighted_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

SCREEN_ROOT = REPO / "logs/gates/sa5_v3_c50_equalweight_control_screen/screen_20260824_r1"
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
AUTHORIZATION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_equalweight_control_screen_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "65aaa49ed77767020ebdc1f1277043f5cc32168284185205aced65ac5ba0094d"
)
CONTROL_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_equalweight_control_p25_completion_20260824.json"
)
CONTROL_COMPLETION_SHA256 = (
    "81771f0d7c5d7b10ec90494f8590b4c5a5bf77bcea3ab40a1d1efa1c56807e50"
)
FROZEN_PRIOR_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_random2d_weighted_screen/"
    "screen_20260824_r1/SUMMARY.json"
)
FROZEN_PRIOR_SUMMARY_SHA256 = (
    "2f428c4c2ce25a9c27d0fc15b5f705be824e9789e72cabcd04496dcefd9ad3b4"
)
FROZEN_PRIOR_PROTOCOL = (
    REPO / "docs/freeze/sa5_v3_c50_random2d_weighted_screen_v1.json"
)
FROZEN_PRIOR_PROTOCOL_SHA256 = (
    "d360a760d34f78d549ffd8bed4785ce8d4243752a54f25d563ec9a9f7efd97ae"
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

TARGET_MOTION_MODE = "random_2d"
RETENTION_MOTION_MODES = ("lateral", "longitudinal", "mixed")
MIN_ABS_Z = 2.0
MIN_TARGET_DELTA = 0.05
MAX_TARGET_TO = 0.05
MAX_DEGRADATION = 0.02
MEANINGFUL_DELTA = 0.02

ANCHOR = dict(base.ANCHOR)
WEIGHTED_IT25 = dict(base.WEIGHTED_IT25)
CONTROL_IT25 = {
    "name": "equalweight_control_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_equalweight_control_ne1024_s42_p25_r1/"
        "checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "d9095e7df21633056ef3f491d460520b99dbd72d9924355821c1e48783b3419a"
    ),
    "status": "MATCHED_DEVELOPMENT_CONTROL_NOT_PARENT",
}
CANDIDATE_SPECS = (CONTROL_IT25,)
_BY_NAME = {spec["name"]: spec for spec in CANDIDATE_SPECS}
CELL_SCHEMA = "sa5_v3_c50_equalweight_control_screen_cell/v1"

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
    (AUTHORIZATION, AUTHORIZATION_SHA256, "screen authorization"),
    (CONTROL_COMPLETION, CONTROL_COMPLETION_SHA256, "control completion"),
    (FROZEN_PRIOR_SUMMARY, FROZEN_PRIOR_SUMMARY_SHA256, "prior summary"),
    (FROZEN_PRIOR_PROTOCOL, FROZEN_PRIOR_PROTOCOL_SHA256, "prior protocol"),
    (
        REPO / CONTROL_IT25["path"],
        CONTROL_IT25["expected_sha256"],
        "equal-weight control checkpoint",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
_completion = json.loads(CONTROL_COMPLETION.read_text(encoding="utf-8"))
_prior_summary = json.loads(FROZEN_PRIOR_SUMMARY.read_text(encoding="utf-8"))
if (
    _authorization.get("decision")
    != "RUN_4_CELL_MATCHED_CONTROL_SCREEN_REUSE_FROZEN_8_CELLS"
    or (_authorization.get("automatic_actions") or {}).get("sa6_launch")
    is not False
    or _completion.get("status")
    != "COMPLETE_VALID_MATCHED_CONTROL_AWAITING_FIXED_SCREEN"
    or (_completion.get("interpretation") or {}).get("sa6_authorized")
    is not False
    or _prior_summary.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_EVALUATION_ONLY_SCREEN"
    or _prior_summary.get("source_fingerprint_stable") is not True
    or _prior_summary.get("anchor_replication_exact") is not True
    or _prior_summary.get("sa6_started") is not False
):
    raise RuntimeError("frozen evidence does not authorize the control screen")


def checkpoint_path(spec: dict | str = CONTROL_IT25) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError("unknown matched-control checkpoint")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown matched-control checkpoint {name!r}")
    result = dict(spec)
    result["sha256"] = spec["expected_sha256"]
    return result


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_equalweight_control_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "question": (
            "separate generic 25-iteration continuation from the random-2D "
            "motion-family weight intervention"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "anchor": dict(ANCHOR),
            "control": dict(CONTROL_IT25),
            "weighted": dict(WEIGHTED_IT25),
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "evidence_reuse": {
            "prior_summary": str(FROZEN_PRIOR_SUMMARY.relative_to(REPO)),
            "prior_summary_sha256": FROZEN_PRIOR_SUMMARY_SHA256,
            "prior_protocol": str(FROZEN_PRIOR_PROTOCOL.relative_to(REPO)),
            "prior_protocol_sha256": FROZEN_PRIOR_PROTOCOL_SHA256,
            "control_completion": str(CONTROL_COMPLETION.relative_to(REPO)),
            "control_completion_sha256": CONTROL_COMPLETION_SHA256,
        },
        "new_matrix": {
            "static_obstacles": STATIC_COUNT,
            "dynamic_obstacles": DYNAMIC_COUNT,
            "motion_modes": list(MOTION_MODES),
            "checkpoints": [CONTROL_IT25["name"]],
            "cells": [dict(row) for row in SCENARIO_SPECS],
            "new_cell_count": len(SCENARIO_SPECS),
            "reused_cell_count": 8,
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
        "analysis": {
            "generic_continuation": "control - anchor",
            "weight_specific": "weighted - control",
            "meaningful_absolute_delta": MEANINGFUL_DELTA,
            "minimum_absolute_z": MIN_ABS_Z,
            "target_motion_mode": TARGET_MOTION_MODE,
            "minimum_target_sr_improvement": MIN_TARGET_DELTA,
            "minimum_target_cr_reduction": MIN_TARGET_DELTA,
            "maximum_target_to": MAX_TARGET_TO,
            "maximum_degradation": MAX_DEGRADATION,
            "retention_motion_modes": list(RETENTION_MOTION_MODES),
        },
        "decision_rules": dict(_authorization["decision_rules"]),
        "screen_starts_training": False,
        "screen_starts_sa6": False,
        "screen_authorizes_graduation": False,
        "screen_authorizes_extension": False,
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


def _contrast(
    metric: str,
    before: dict,
    after: dict,
    *,
    before_name: str,
    after_name: str,
) -> dict:
    p0, p1 = float(before[metric]), float(after[metric])
    delta = p1 - p0
    se = math.hypot(_se(p0, int(before["n"])), _se(p1, int(after["n"])))
    z = delta / se if se > 0 else 0.0
    return {
        "metric": metric,
        "before_name": before_name,
        "after_name": after_name,
        "before": p0,
        "after": p1,
        "delta": delta,
        "se_diff": se,
        "z": z,
        "meaningful_descriptive": (
            abs(delta) >= MEANINGFUL_DELTA and abs(z) >= MIN_ABS_Z
        ),
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
        "obstacle_cr": _finite(report["obstacle_collision_rate"], "obstacle_cr"),
        "behavior": base._behavior(payload),
        "layouts": report.get("static_layout_outcomes") or {},
    }


def _frozen_prior_rows() -> list[dict]:
    rows = json.loads(json.dumps(_prior_summary.get("rows") or []))
    expected = {
        (name, mode)
        for name in (ANCHOR["name"], WEIGHTED_IT25["name"])
        for mode in MOTION_MODES
    }
    observed = {
        (str(row.get("checkpoint_name")), str(row.get("motion_mode")))
        for row in rows
    }
    if len(rows) != 8 or observed != expected:
        raise RuntimeError("frozen prior summary lacks the exact 8-cell matrix")
    return rows


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(SCENARIO_SPECS)
    if len(payloads) != expected:
        raise ValueError(f"expected exactly {expected} control-screen cells")
    control_rows = [_row(payload) for payload in payloads]
    rows = [*_frozen_prior_rows(), *control_rows]
    by_key = {
        (str(row["checkpoint_name"]), str(row["motion_mode"])): row
        for row in rows
    }
    expected_keys = {
        (name, mode)
        for name in (
            ANCHOR["name"],
            CONTROL_IT25["name"],
            WEIGHTED_IT25["name"],
        )
        for mode in MOTION_MODES
    }
    if len(by_key) != 12 or set(by_key) != expected_keys:
        raise ValueError("three-arm cells are duplicated or missing")

    comparisons = []
    generic_effects = []
    weight_specific_effects = []
    for mode in MOTION_MODES:
        anchor = by_key[(ANCHOR["name"], mode)]
        control = by_key[(CONTROL_IT25["name"], mode)]
        weighted = by_key[(WEIGHTED_IT25["name"], mode)]
        generic = {
            metric: _contrast(
                metric,
                anchor,
                control,
                before_name=ANCHOR["name"],
                after_name=CONTROL_IT25["name"],
            )
            for metric in ("sr", "cr", "to")
        }
        specific = {
            metric: _contrast(
                metric,
                control,
                weighted,
                before_name=CONTROL_IT25["name"],
                after_name=WEIGHTED_IT25["name"],
            )
            for metric in ("sr", "cr", "to")
        }
        generic_effects.extend(
            {"motion_mode": mode, **value}
            for value in generic.values()
            if value["meaningful_descriptive"]
        )
        weight_specific_effects.extend(
            {"motion_mode": mode, **value}
            for value in specific.values()
            if value["meaningful_descriptive"]
        )
        comparisons.append(
            {
                "motion_mode": mode,
                "anchor": anchor,
                "equalweight_control_it25": control,
                "weighted_it25": weighted,
                "control_minus_anchor": generic,
                "weighted_minus_control": specific,
            }
        )

    target = next(
        row for row in comparisons if row["motion_mode"] == TARGET_MOTION_MODE
    )
    specific = target["weighted_minus_control"]
    target_checks = {
        "sr_improves": (
            specific["sr"]["delta"] >= MIN_TARGET_DELTA
            and specific["sr"]["z"] >= MIN_ABS_Z
        ),
        "cr_reduces": (
            specific["cr"]["delta"] <= -MIN_TARGET_DELTA
            and specific["cr"]["z"] <= -MIN_ABS_Z
        ),
        "to_absolute": target["weighted_it25"]["to"] <= MAX_TARGET_TO,
        "to_not_degraded": specific["to"]["delta"] <= MAX_DEGRADATION,
    }
    target_pass = all(target_checks.values())

    retention = []
    for row in comparisons:
        if row["motion_mode"] not in RETENTION_MOTION_MODES:
            continue
        specific = row["weighted_minus_control"]
        checks = {
            "sr": specific["sr"]["delta"] >= -MAX_DEGRADATION,
            "cr": specific["cr"]["delta"] <= MAX_DEGRADATION,
            "to": specific["to"]["delta"] <= MAX_DEGRADATION,
        }
        retention.append(
            {
                "motion_mode": row["motion_mode"],
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    retention_pass = all(row["pass"] for row in retention)

    if target_pass and retention_pass:
        next_action = (
            "WEIGHT_SPECIFIC_SUPPORT_OBSERVED_"
            "CALIBRATE_ALLOCATOR_BEFORE_ANY_NEW_PILOT"
        )
    elif weight_specific_effects:
        next_action = (
            "WEIGHT_SPECIFIC_CAPABILITY_REDISTRIBUTION_OBSERVED_"
            "REJECT_CURRENT_WEIGHTS"
        )
    elif generic_effects:
        next_action = (
            "GENERIC_CONTINUATION_EFFECT_OBSERVED_DO_NOT_ATTRIBUTE_TO_WEIGHTS"
        )
    else:
        next_action = "NO_CLEAN_ATTRIBUTION_HOLD_AND_CALIBRATE_ALLOCATOR"

    return {
        "schema": "sa5_v3_c50_equalweight_control_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_MATCHED_THREE_ARM_DIAGNOSTIC",
        "lineage_status": {
            "sa5": "HOLD_NOT_GRADUATED",
            "anchor": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
            "control": "MATCHED_DEVELOPMENT_CONTROL_NOT_PARENT",
            "weighted": "REJECTED_DEVELOPMENT_CANDIDATE_NOT_PARENT",
        },
        "rows": rows,
        "comparisons": comparisons,
        "generic_continuation_effects": generic_effects,
        "weight_specific_effects": weight_specific_effects,
        "weight_specific_target_checks": target_checks,
        "weight_specific_target_pass": target_pass,
        "weight_specific_retention": retention,
        "weight_specific_retention_pass": retention_pass,
        "weight_intervention_accepted": target_pass and retention_pass,
        "next_action": next_action,
        "selected_development_checkpoint": None,
        "selected_extension_checkpoint": None,
        "extension_authorized": False,
        "graduation_authorized": False,
        "training_started": False,
        "sa6_started": False,
        "prior_summary_sha256": FROZEN_PRIOR_SUMMARY_SHA256,
        "interpretation_limit": _authorization["interpretation_limit"],
    }
