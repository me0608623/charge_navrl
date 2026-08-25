"""Frozen paired retention screen for the SA5-v3 stage-2 mild weighting."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_v3_provisional_pilot_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCREEN_ROOT = (
    REPO / "logs/gates/sa5_v3_c50_stage2_retention_screen/screen_20260824_r1"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
PAIRED_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_stage2_paired_screen/"
    "screen_20260824_r2/SUMMARY.json"
)
PAIRED_SUMMARY_SHA256 = (
    "d4957498b3e413f7035c5131c625a902c180f1d237b2f2072df94e7137c9ca6f"
)

STAGE = base.STAGE
SEED = base.SEED
NUM_ENVS = base.NUM_ENVS
MIN_EPISODES = base.MIN_EPISODES
DELAY_STEPS = base.DELAY_STEPS
DELAY_MS = base.DELAY_MS
ACTUATOR_PROFILE = base.ACTUATOR_PROFILE
LIDAR_NOISE_MODE = base.LIDAR_NOISE_MODE
LIDAR_DISTRACTOR_ELIGIBILITY = base.LIDAR_DISTRACTOR_ELIGIBILITY
SPEED_RATE = base.SPEED_RATE
SPEED_RATE_OBS = base.SPEED_RATE_OBS
DEPLOYMENT_SPEED_SCALE = base.DEPLOYMENT_SPEED_SCALE
P035 = base.P035
P060 = base.P060

# The inherited cell runner uses TARGET_SCENARIO only to configure a 4S2D target.
# This screen intentionally contains retention scenarios only.
TARGET_SCENARIO = "__no_target_in_retention_screen__"
LOW_DENSITY_SCENARIOS = base.LOW_DENSITY_SCENARIOS
CORRIDOR_SCENARIOS = LOW_DENSITY_SCENARIOS
RETENTION_SCENARIOS = (
    "nav_native",
    "narrow_range",
    *LOW_DENSITY_SCENARIOS,
)
ALL_SCENARIOS = RETENTION_SCENARIOS
STEPS_BY_SCENARIO = {
    "nav_native": base.STEPS_BY_SCENARIO["nav_native"],
    "narrow_range": base.STEPS_BY_SCENARIO["narrow_range"],
    "corridor_low_0s1d": base.STEPS_BY_SCENARIO["corridor_low_0s1d"],
    "corridor_low_1s1d": base.STEPS_BY_SCENARIO["corridor_low_1s1d"],
}
NATIVE_THRESHOLDS = dict(base.NATIVE_THRESHOLDS)
NARROW_THRESHOLDS = dict(base.NARROW_THRESHOLDS)
P060_THRESHOLDS = dict(base.P060_THRESHOLDS)
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
_BY_NAME = {row["name"]: row for row in CANDIDATE_SPECS}
CELL_SCHEMA = "sa5_v3_c50_stage2_retention_screen_cell/v1"

sha256_of = base.sha256_of
_stage_scene = base._stage_scene
_finite = base._finite


def _verify_input(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


_verify_input(PAIRED_SUMMARY, PAIRED_SUMMARY_SHA256, "paired-screen summary")
for _candidate in CANDIDATE_SPECS:
    _verify_input(
        REPO / str(_candidate["path"]),
        str(_candidate["expected_sha256"]),
        str(_candidate["name"]),
    )
_paired_summary = json.loads(PAIRED_SUMMARY.read_text(encoding="utf-8"))
if (
    _paired_summary.get("status")
    != "COMPLETE_VALID_SINGLE_TRAINING_SEED_PAIRED_SCREEN"
    or not bool(_paired_summary.get("mild_weight_acceptance_observed"))
    or _paired_summary.get("next_action")
    != "RUN_NATIVE_NARROW_P060_RETENTION_BEFORE_ANY_EXTENSION"
    or bool(_paired_summary.get("extension_authorized"))
    or bool(_paired_summary.get("graduation_authorized"))
):
    raise RuntimeError("paired screen does not authorize this retention screen")


def checkpoint_path(spec: dict | str = A2_EQUAL) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError(f"unknown retention checkpoint {spec!r}")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown retention checkpoint {name!r}")
    result = dict(spec)
    result["sha256"] = str(spec["expected_sha256"])
    return result


def thresholds_for(scenario: str) -> dict:
    if scenario == "nav_native":
        return dict(NATIVE_THRESHOLDS)
    if scenario == "narrow_range":
        return dict(NARROW_THRESHOLDS)
    if scenario in LOW_DENSITY_SCENARIOS:
        return dict(P060_THRESHOLDS)
    raise ValueError(f"unknown retention scenario {scenario!r}")


def evaluate_metrics(scenario: str, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario)
    checks = {
        "episodes": int(metrics.get("n", 0)) >= thresholds["episodes_min"],
        "sr": _finite(metrics.get("sr"), "sr") >= thresholds["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= thresholds["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= thresholds["to_max"],
    }
    if scenario == "narrow_range":
        checks["crossing"] = (
            _finite(metrics.get("crossing_rate"), "crossing_rate")
            >= thresholds["crossing_min"]
        )
        checks["direct_crossing"] = (
            _finite(metrics.get("direct_crossing_rate"), "direct_crossing_rate")
            >= thresholds["direct_crossing_min"]
        )
    return {
        "thresholds": thresholds,
        "checks": checks,
        "threshold_pass": all(checks.values()),
    }


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_stage2_retention_screen_protocol/v1",
        "trigger": {
            "summary": str(PAIRED_SUMMARY.relative_to(REPO)),
            "summary_sha256": PAIRED_SUMMARY_SHA256,
            "required_next_action": (
                "RUN_NATIVE_NARROW_P060_RETENTION_BEFORE_ANY_EXTENSION"
            ),
        },
        "question": (
            "does the 30/30/40 development candidate preserve native, narrow, "
            "and historical P060 low-density capabilities relative to the "
            "matched equal-allocation continuation"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "a2": dict(A2_EQUAL),
            "b2": dict(B2_MILD),
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "matrix": {
            "candidates": [row["name"] for row in CANDIDATE_SPECS],
            "scenarios": list(ALL_SCENARIOS),
            "cell_count": len(CANDIDATE_SPECS) * len(ALL_SCENARIOS),
        },
        "fixed_evaluation": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps_by_scenario": dict(STEPS_BY_SCENARIO),
            "minimum_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "p060_speed_range_m_s": list(P060),
            "p060_motion_mode": "lateral",
            "scene": _stage_scene(),
        },
        "acceptance": {
            "absolute_thresholds": {
                "native": dict(NATIVE_THRESHOLDS),
                "narrow": dict(NARROW_THRESHOLDS),
                "p060": dict(P060_THRESHOLDS),
            },
            "maximum_sr_degradation": MAX_DEGRADATION,
            "maximum_cr_degradation": MAX_DEGRADATION,
            "maximum_to_degradation": MAX_DEGRADATION,
            "rule": (
                "B2 must pass all four absolute gates and must not degrade "
                "SR, CR, or TO by more than 2 percentage points versus A2"
            ),
        },
        "decision": {
            "on_pass": (
                "RETENTION_SUPPORTED_REQUEST_SEPARATE_BOUNDED_EXTENSION_AUTHORIZATION"
            ),
            "on_fail": "REJECT_MILD_WEIGHT_FOR_RETENTION_FAILURE",
            "starts_training": False,
            "authorizes_extension": False,
            "authorizes_graduation": False,
            "starts_sa6": False,
        },
        "interpretation_limit": (
            "single training seed and single evaluator seed paired developmental "
            "retention evidence; passing does not graduate SA5, choose a parent, "
            "or authorize extension or SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid retention cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("retention cell protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    if scenario not in ALL_SCENARIOS:
        raise ValueError("unknown retention cell scenario")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("retention checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"retention {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("retention LiDAR mode mismatch")
    if (
        payload.get("lidar_distractor_eligibility")
        != LIDAR_DISTRACTOR_ELIGIBILITY
    ):
        raise ValueError("retention LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(
            runtime.get("deployment_speed_scale"), "deployment_speed_scale"
        )
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("retention speed-rate runtime mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("retention cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("retention threshold verdict mismatch")


def _relative(a2: dict, b2: dict) -> dict:
    deltas = {
        "sr_degradation": float(a2["sr"]) - float(b2["sr"]),
        "cr_degradation": float(b2["cr"]) - float(a2["cr"]),
        "to_degradation": float(b2["to"]) - float(a2["to"]),
    }
    checks = {
        key: value <= MAX_DEGRADATION + 1e-12 for key, value in deltas.items()
    }
    return {"deltas": deltas, "checks": checks, "pass": all(checks.values())}


def _row(payload: dict) -> dict:
    metrics = payload["metrics"]
    row = {
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "absolute_pass": bool(payload["threshold_pass"]),
    }
    for key in ("crossing_rate", "direct_crossing_rate"):
        if key in metrics:
            row[key] = _finite(metrics[key], key)
    return row


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"retention screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate retention cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in CANDIDATE_SPECS]
    if list(by_name) != expected_names:
        raise ValueError("retention candidate order mismatch")
    if any(set(cells) != set(ALL_SCENARIOS) for cells in by_name.values()):
        raise ValueError("retention scenario matrix mismatch")

    comparisons = []
    for scenario in ALL_SCENARIOS:
        a2 = _row(by_name[A2_EQUAL["name"]][scenario])
        b2 = _row(by_name[B2_MILD["name"]][scenario])
        relative = _relative(a2, b2)
        comparisons.append(
            {
                "scenario": scenario,
                "a2_equal": a2,
                "b2_mild": b2,
                "relative": relative,
                "pass": b2["absolute_pass"] and relative["pass"],
            }
        )
    supported = all(row["pass"] for row in comparisons)
    return {
        "schema": "sa5_v3_c50_stage2_retention_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_TRAINING_SEED_PAIRED_RETENTION",
        "protocol_sha256": screen_protocol()["sha256"],
        "comparisons": comparisons,
        "retention_supported": supported,
        "selected_development_checkpoint": (
            B2_MILD["name"] if supported else None
        ),
        "selected_extension_checkpoint": None,
        "extension_authorized": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "next_action": (
            "REQUEST_SEPARATE_BOUNDED_EXTENSION_AUTHORIZATION"
            if supported
            else "REJECT_MILD_WEIGHT_FOR_RETENTION_FAILURE"
        ),
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
