"""Frozen Phase-A screen for the paired SA5-v3 Stage-3 P060 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sa5_v3_c50_stage2_retention_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCREEN_ROOT = REPO / "logs/gates/sa5_v3_c50_stage3_phase_a_screen/screen_20260824_r1"
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c50_stage3_p060_alignment_p25_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "f72e9afcc5e840de52ab8571edb0ddfde943c6bbad60334f41fc174a52ccb1dc"
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
P060 = (0.50, 0.70)
P080 = (0.70, 0.90)

TARGET_SCENARIO = "__phase_a_has_no_4s2d_target__"
P060_SCENARIOS = ("p060_0s1d", "p060_1s1d")
P080_SCENARIOS = ("p080_0s1d", "p080_1s1d")
ALL_SCENARIOS = (*P060_SCENARIOS, *P080_SCENARIOS)
LOW_DENSITY_SCENARIOS = ALL_SCENARIOS
CORRIDOR_SCENARIOS: tuple[str, ...] = ()
STEPS_BY_SCENARIO = {scenario: 2_500 for scenario in ALL_SCENARIOS}
THRESHOLDS = dict(base.P060_THRESHOLDS)
MAX_DEGRADATION = 0.02
MIN_WORST_P060_CR_IMPROVEMENT = 0.02

A3_CONTROL = {
    "name": "stage3_a3_control_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_stage3_a3_control_ne1024_s42_p25_r1/checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "ae6c2495040b8582a5d82ccff3fefea02ff10cc209e7e37cb8230be7d61dde13"
    ),
    "status": "DIAGNOSTIC_CONTROL_NOT_PARENT",
}
B3_P060 = {
    "name": "stage3_b3_p060aligned_it25",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_stage3_b3_p060aligned_ne1024_s42_p25_r1/checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "5645f84625649a6050e862eb17b8e81e2e02225973fd371f29e663785104f2b8"
    ),
    "status": "DIAGNOSTIC_INTERVENTION_NOT_PARENT",
}
CANDIDATE_SPECS = (A3_CONTROL, B3_P060)
_BY_NAME = {row["name"]: row for row in CANDIDATE_SPECS}
CELL_SCHEMA = "sa5_v3_c50_stage3_phase_a_screen_cell/v1"

sha256_of = base.sha256_of
_stage_scene = base._stage_scene
_finite = base._finite


def speed_range_for(scenario: str) -> tuple[float, float]:
    if scenario in P060_SCENARIOS:
        return P060
    if scenario in P080_SCENARIOS:
        return P080
    raise ValueError(f"unknown Stage-3 Phase-A scenario {scenario!r}")


def density_for(scenario: str) -> tuple[int, int]:
    if scenario.endswith("_0s1d"):
        return (0, 1)
    if scenario.endswith("_1s1d"):
        return (1, 1)
    raise ValueError(f"unknown Stage-3 Phase-A density {scenario!r}")


def checkpoint_path(spec: dict | str = A3_CONTROL) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError(f"unknown Stage-3 candidate {spec!r}")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown Stage-3 candidate {name!r}")
    result = dict(spec)
    result["sha256"] = str(spec["expected_sha256"])
    return result


def thresholds_for(scenario: str) -> dict:
    speed_range_for(scenario)
    return dict(THRESHOLDS)


def evaluate_metrics(scenario: str, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario)
    checks = {
        "episodes": int(metrics.get("n", 0)) >= thresholds["episodes_min"],
        "sr": _finite(metrics.get("sr"), "sr") >= thresholds["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= thresholds["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= thresholds["to_max"],
    }
    return {
        "thresholds": thresholds,
        "checks": checks,
        "threshold_pass": all(checks.values()),
    }


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_stage3_phase_a_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "question": (
            "does changing only the 0S1D and 1S1D training speed ranges from "
            "P080 to P060 restore fixed P060 capability without erasing P080"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "a3": dict(A3_CONTROL),
            "b3": dict(B3_P060),
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
            "motion_mode": "lateral",
            "p060_speed_range_m_s": list(P060),
            "p080_speed_range_m_s": list(P080),
            "densities": ["0S1D", "1S1D"],
            "scene": _stage_scene(),
        },
        "acceptance": {
            "absolute_p060_thresholds": dict(THRESHOLDS),
            "minimum_worst_p060_cr_improvement": MIN_WORST_P060_CR_IMPROVEMENT,
            "maximum_p080_sr_degradation": MAX_DEGRADATION,
            "maximum_p080_cr_degradation": MAX_DEGRADATION,
            "maximum_p080_to_degradation": MAX_DEGRADATION,
            "rule": (
                "B3 passes both P060 absolute gates, improves worst P060 CR "
                "by at least 2 percentage points versus A3, and degrades no "
                "P080 SR/CR/TO metric by more than 2 percentage points"
            ),
        },
        "decision": {
            "on_pass": "RUN_FROZEN_PHASE_B_TARGET_AND_RETENTION_SCREEN",
            "on_fail": "REJECT_P060_ALIGNMENT_NO_PHASE_B",
            "starts_training": False,
            "authorizes_extension": False,
            "authorizes_graduation": False,
            "selects_parent": False,
            "starts_sa6": False,
        },
        "interpretation_limit": (
            "single training seed and single evaluator seed paired development "
            "screen; no outcome selects a parent, authorizes extension, "
            "graduates SA5, or starts SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid Stage-3 Phase-A cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("Stage-3 Phase-A protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    expected_speed = speed_range_for(scenario)
    expected_density = density_for(scenario)
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("Stage-3 Phase-A checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"Stage-3 Phase-A {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("Stage-3 Phase-A LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("Stage-3 Phase-A LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("Stage-3 Phase-A speed-rate runtime mismatch")
    report = payload.get("corridor_report") or {}
    report_speed = tuple(float(value) for value in report.get("requested_dynamic_speed_range_m_s", ()))
    report_density = (
        int(report.get("static_obstacles_per_env", -1)),
        int(report.get("dynamic_obstacles_per_env", -1)),
    )
    if report_speed != expected_speed or report_density != expected_density:
        raise ValueError("Stage-3 Phase-A speed/density runtime mismatch")
    if report.get("dynamic_motion_mode") != "lateral":
        raise ValueError("Stage-3 Phase-A motion-mode mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("Stage-3 Phase-A cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("Stage-3 Phase-A threshold verdict mismatch")


def _row(payload: dict) -> dict:
    metrics = payload["metrics"]
    return {
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "absolute_pass": bool(payload["threshold_pass"]),
    }


def _relative(control: dict, intervention: dict) -> dict:
    deltas = {
        "sr_degradation": control["sr"] - intervention["sr"],
        "cr_degradation": intervention["cr"] - control["cr"],
        "to_degradation": intervention["to"] - control["to"],
    }
    checks = {key: value <= MAX_DEGRADATION + 1e-12 for key, value in deltas.items()}
    return {"deltas": deltas, "checks": checks, "pass": all(checks.values())}


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"Stage-3 Phase-A screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate Stage-3 Phase-A cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in CANDIDATE_SPECS]
    if list(by_name) != expected_names:
        raise ValueError("Stage-3 Phase-A candidate order mismatch")
    if any(set(cells) != set(ALL_SCENARIOS) for cells in by_name.values()):
        raise ValueError("Stage-3 Phase-A scenario matrix mismatch")

    comparisons = []
    control_rows = {}
    intervention_rows = {}
    for scenario in ALL_SCENARIOS:
        control = _row(by_name[A3_CONTROL["name"]][scenario])
        intervention = _row(by_name[B3_P060["name"]][scenario])
        control_rows[scenario] = control
        intervention_rows[scenario] = intervention
        comparisons.append(
            {
                "scenario": scenario,
                "a3_control": control,
                "b3_p060_aligned": intervention,
                "relative": _relative(control, intervention),
            }
        )

    worst_control = max(control_rows[name]["cr"] for name in P060_SCENARIOS)
    worst_intervention = max(intervention_rows[name]["cr"] for name in P060_SCENARIOS)
    improvement = worst_control - worst_intervention
    p060_absolute_pass = all(intervention_rows[name]["absolute_pass"] for name in P060_SCENARIOS)
    p080_retention_pass = all(
        next(row for row in comparisons if row["scenario"] == name)["relative"]["pass"]
        for name in P080_SCENARIOS
    )
    phase_a_pass = (
        p060_absolute_pass
        and improvement + 1e-12 >= MIN_WORST_P060_CR_IMPROVEMENT
        and p080_retention_pass
    )
    return {
        "schema": "sa5_v3_c50_stage3_phase_a_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_TRAINING_SEED_PAIRED_PHASE_A",
        "protocol_sha256": screen_protocol()["sha256"],
        "comparisons": comparisons,
        "worst_p060_cr": {
            "a3_control": worst_control,
            "b3_p060_aligned": worst_intervention,
            "improvement": improvement,
            "minimum_required": MIN_WORST_P060_CR_IMPROVEMENT,
        },
        "p060_absolute_pass": p060_absolute_pass,
        "p080_retention_pass": p080_retention_pass,
        "phase_a_pass": phase_a_pass,
        "selected_parent": None,
        "extension_authorized": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "next_action": (
            "RUN_FROZEN_PHASE_B_TARGET_AND_RETENTION_SCREEN"
            if phase_a_pass
            else "REJECT_P060_ALIGNMENT_NO_PHASE_B"
        ),
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
