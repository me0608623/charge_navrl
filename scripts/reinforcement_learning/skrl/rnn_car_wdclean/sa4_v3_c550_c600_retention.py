"""Frozen SA4-v3 retention screen for the promoted c550 and c600."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

GATE_SUMMARY = (
    REPO
    / "logs/gates/sa4_v3_c300_c600_gate/screen_20260822_r1/SUMMARY.json"
)
GATE_SUMMARY_SHA256 = (
    "ab0e41e148fd82582b15e9e7c473297d5ec5b8a3a050411801b639e1214e3df5"
)
GATE_PROTOCOL_SHA256 = (
    "14cd30ca27a8e986bba183db361c9bc75ff66d81b4ef89023b07dc2a08e2e285"
)
GATE_FREEZE = REPO / "docs/freeze/sa4_v3_c300_c600_gate_v1.json"
GATE_FREEZE_SHA256 = (
    "924b05c15486d844f4610ab0ac2d0dc0f4421c79ca536f2ffddd26785e9157c1"
)

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
P060_SPEED_RANGE_M_S = (0.50, 0.70)
MOTION_MODE = "lateral"

NATIVE_SCENE = {
    "stage_name": "SA4_spatial_plan",
    "static_obstacles": 8,
    "dynamic_obstacles": 2,
    "source": (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
        "velocity/config/charge_skrl/curriculum/phases/e2e_final20_v1.py"
    ),
}

THRESHOLDS = {
    "nav_native": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.92,
        "cr_max": 0.07,
        "to_max": 0.035,
    },
    "narrow_range": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.05,
        "to_max": 0.05,
        "crossing_min": 0.95,
        "direct_crossing_min": 0.95,
    },
    "p060_0s1d_lateral": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.10,
        "to_max": 0.05,
    },
    "p060_1s1d_lateral": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.10,
        "to_max": 0.05,
    },
}

SCENARIOS = (
    {"name": "nav_native", "kind": "native", "display": "native"},
    {"name": "narrow_range", "kind": "narrow", "display": "narrow"},
    {
        "name": "p060_0s1d_lateral",
        "kind": "corridor",
        "display": "P060 0S1D lateral",
        "label": "0S1D_P060",
        "density": "0S1D",
        "pedestrian_speed_label": "P060",
        "pedestrian_speed_range_m_s": P060_SPEED_RANGE_M_S,
        "scenario": "corridor_low_0s1d",
        "static_obstacles": 0,
        "dynamic_obstacles": 1,
    },
    {
        "name": "p060_1s1d_lateral",
        "kind": "corridor",
        "display": "P060 1S1D lateral",
        "label": "1S1D_P060",
        "density": "1S1D",
        "pedestrian_speed_label": "P060",
        "pedestrian_speed_range_m_s": P060_SPEED_RANGE_M_S,
        "scenario": "corridor_low_1s1d",
        "static_obstacles": 1,
        "dynamic_obstacles": 1,
    },
)

# This order is inherited from the completed c300-c600 Gate comparison.
CANDIDATES = (
    {
        "name": "c550",
        "conceptual_iteration": 550,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_32000.pt"
        ),
        "sha256": (
            "235a79a41b1d83510bbbb314c965a7ef5ba73e898aec121310a92965092c4ba8"
        ),
        "gate_rank": 1,
        "gate_hard_pass": False,
    },
    {
        "name": "c600",
        "conceptual_iteration": 600,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_38400.pt"
        ),
        "sha256": (
            "7c849d0b0e33846d228ae7f2914dd8a868b6beb590d65b8e4b528c0e32e820cc"
        ),
        "gate_rank": 2,
        "gate_hard_pass": False,
    },
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown SA4-v3 retention candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return REPO / str(data["path"])


def scenario_by_name(name: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["name"] == name:
            return dict(scenario)
    raise ValueError(f"unknown SA4-v3 retention scenario {name!r}")


def cell_label(checkpoint_name: str, scenario_name: str) -> str:
    candidate_by_name(checkpoint_name)
    scenario_by_name(scenario_name)
    return f"{checkpoint_name}_{scenario_name}_s070_g4_d1_s818"


def cells() -> tuple[dict, ...]:
    return tuple(
        {
            "cell": cell_label(candidate["name"], scenario["name"]),
            "checkpoint_name": candidate["name"],
            "scenario": scenario["name"],
            "kind": scenario["kind"],
        }
        for candidate in CANDIDATES
        for scenario in SCENARIOS
    )


def _jsonable_scenarios() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        row = dict(scenario)
        if "pedestrian_speed_range_m_s" in row:
            row["pedestrian_speed_range_m_s"] = list(
                row["pedestrian_speed_range_m_s"]
            )
        rows.append(row)
    return rows


def _protocol_body() -> dict:
    return {
        "schema": "sa4_v3_c550_c600_retention_protocol/v1",
        "purpose": (
            "evaluate native, narrow, and P060 low-density retention for the "
            "c550 and c600 checkpoints promoted by the frozen SA4-v3 Gate"
        ),
        "gate_source": {
            "summary": str(GATE_SUMMARY.relative_to(REPO)),
            "summary_sha256": GATE_SUMMARY_SHA256,
            "protocol_sha256": GATE_PROTOCOL_SHA256,
            "freeze": str(GATE_FREEZE.relative_to(REPO)),
            "freeze_sha256": GATE_FREEZE_SHA256,
            "required_promoted_order": [row["name"] for row in CANDIDATES],
            "gate_hard_pass": False,
        },
        "stage": STAGE,
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "scenarios": _jsonable_scenarios(),
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
            "native_scene": dict(NATIVE_SCENE),
            "p060_speed_range_m_s": list(P060_SPEED_RANGE_M_S),
            "p060_motion_mode": MOTION_MODE,
        },
        "thresholds": json.loads(json.dumps(THRESHOLDS, sort_keys=True)),
        "decision": {
            "candidate_retention_pass": "all four retention cells pass",
            "recommended_exposure_parent": (
                "first passing candidate in the preregistered Gate order"
            ),
            "c550_eligibility": (
                "c550 is eligible only for a bounded exposure-curriculum pilot "
                "when all four retention cells pass"
            ),
            "sa4_graduation": (
                "remains failed because no candidate passed the source Gate"
            ),
            "accepted_exposure_parent": None,
            "auto_launch_exposure_curriculum": False,
            "auto_launch_sa5": False,
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "retention screen; a retention pass can support a bounded exposure "
            "curriculum parent choice, but cannot convert the failed SA4 Gate "
            "into graduation or authorize SA5"
        ),
        "forbidden": [
            "change the promoted candidate order after seeing retention results",
            "change stage, scene, speed, delay, noise, seed, or thresholds",
            "summarize a missing, duplicate, invalid, or drifted cell",
            "claim SA4 graduation from retention",
            "auto-start exposure training or SA5",
        ],
    }


def retention_protocol() -> dict:
    payload = json.loads(json.dumps(_protocol_body(), sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_gate_summary(payload: dict) -> None:
    if payload.get("schema") != "sa4_v3_c300_c600_gate_summary/v1":
        raise ValueError("unexpected source Gate summary schema")
    if payload.get("status") != (
        "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_GATE_COMPARISON"
    ):
        raise ValueError("source Gate summary is not complete and valid")
    if payload.get("protocol_sha256") != GATE_PROTOCOL_SHA256:
        raise ValueError("source Gate protocol hash mismatch")
    expected = [candidate["name"] for candidate in CANDIDATES]
    if payload.get("promoted_for_retention") != expected:
        raise ValueError("source Gate promoted order mismatch")
    ranked = payload.get("ranked_candidates") or []
    if [row.get("checkpoint_name") for row in ranked[:2]] != expected:
        raise ValueError("source Gate top-two ranking mismatch")
    if any(bool(row.get("hard_gate_pass")) for row in ranked):
        raise ValueError("source Gate unexpectedly records a graduating candidate")
    if bool(payload.get("all_candidates_hard_gate_pass")):
        raise ValueError("source Gate unexpectedly records all candidates passing")
    if not bool(payload.get("source_fingerprint_stable")):
        raise ValueError("source Gate fingerprint was not stable")
    if payload.get("accepted_sa5_parent") is not None:
        raise ValueError("source Gate unexpectedly accepted an SA5 parent")
    if bool(payload.get("retention_started")) or bool(payload.get("sa5_started")):
        raise ValueError("source Gate records an unexpected downstream start")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(scenario_name: str, metrics: dict) -> dict:
    scenario_by_name(scenario_name)
    thresholds = THRESHOLDS[scenario_name]
    checks = {
        "episodes": int(metrics.get("n", 0))
        >= int(thresholds["episodes_min"]),
        "sr": _finite(metrics.get("sr"), "sr") >= thresholds["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= thresholds["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= thresholds["to_max"],
    }
    if "crossing_min" in thresholds:
        checks["crossing"] = (
            _finite(metrics.get("crossing_rate"), "crossing_rate")
            >= thresholds["crossing_min"]
        )
    if "direct_crossing_min" in thresholds:
        checks["direct_crossing"] = (
            _finite(metrics.get("direct_crossing_rate"), "direct_crossing_rate")
            >= thresholds["direct_crossing_min"]
        )
    return {
        "thresholds": dict(thresholds),
        "checks": checks,
        "pass": all(checks.values()),
    }


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != "sa4_v3_c550_c600_retention_cell/v1":
        raise ValueError("unexpected SA4-v3 retention cell schema")
    if not bool(payload.get("cell_valid")) or not bool(
        payload.get("formal_evidence")
    ):
        raise ValueError("retention cell is not formal valid evidence")
    if payload.get("protocol_sha256") != retention_protocol()["sha256"]:
        raise ValueError("retention cell protocol hash mismatch")

    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = scenario_by_name(str(payload.get("scenario")))
    if payload.get("cell") != cell_label(candidate["name"], scenario["name"]):
        raise ValueError("retention cell label mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("retention checkpoint hash mismatch")
    expected_integers = {
        "stage": STAGE,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        "delay_steps": DELAY_STEPS,
    }
    for key, expected in expected_integers.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"retention cell {key} mismatch")
    if _finite(payload.get("speed_rate"), "speed_rate") != SPEED_RATE:
        raise ValueError("retention cell speed_rate mismatch")
    if payload.get("speed_rate_obs") != SPEED_RATE_OBS:
        raise ValueError("retention cell speed-rate observation mismatch")
    if _finite(
        payload.get("deployment_speed_scale"), "deployment_speed_scale"
    ) != DEPLOYMENT_SPEED_SCALE:
        raise ValueError("retention cell deployment scaling mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("retention cell LiDAR-noise mode mismatch")
    if payload.get("lidar_distractor_eligibility") != (
        LIDAR_DISTRACTOR_ELIGIBILITY
    ):
        raise ValueError("retention cell LiDAR eligibility mismatch")

    actuator = payload.get("actuator_eval") or {}
    if (
        actuator.get("profile") != ACTUATOR_PROFILE
        or int(actuator.get("delay_steps", -1)) != DELAY_STEPS
        or list(actuator.get("velocity_scale_range") or []) != [1.0, 1.0]
        or _finite(actuator.get("motor_lag_alpha"), "motor_lag_alpha") != 1.0
        or actuator.get("pipeline") != "decode->delay->scale->lag"
    ):
        raise ValueError("retention cell actuator metadata mismatch")

    speed_runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(speed_runtime.get("speed_rate"), "runtime speed_rate")
        != SPEED_RATE
        or speed_runtime.get("speed_rate_obs") != SPEED_RATE_OBS
    ):
        raise ValueError("retention cell speed-rate runtime mismatch")
    required_limits = {
        "max_linear_velocity",
        "max_linear_accel",
        "max_angular_vel",
        "max_angular_accel",
    }
    if set((speed_runtime.get("action_limits") or {}).keys()) != required_limits:
        raise ValueError("retention cell speed-rate action limits missing")

    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("retention cell has too few completed episodes")
    if payload.get("gate") != evaluate_metrics(scenario["name"], metrics):
        raise ValueError("retention cell gate payload mismatch")

    stage_banner = payload.get("stage_banner") or {}
    if int(stage_banner.get("stage_built", -1)) != STAGE:
        raise ValueError("runtime stage mismatch")
    if scenario["kind"] == "native":
        if stage_banner.get("stage_name") != NATIVE_SCENE["stage_name"]:
            raise ValueError("native runtime stage name mismatch")
        if int(stage_banner.get("native_static_obstacles", -1)) != int(
            NATIVE_SCENE["static_obstacles"]
        ):
            raise ValueError("native runtime static count mismatch")
        if int(stage_banner.get("native_dynamic_obstacles", -1)) != int(
            NATIVE_SCENE["dynamic_obstacles"]
        ):
            raise ValueError("native runtime dynamic count mismatch")
    elif scenario["kind"] == "narrow":
        if (payload.get("scene_contract") or {}).get("scenario") != (
            "narrow_range"
        ):
            raise ValueError("narrow runtime contract missing")
    else:
        report = payload.get("corridor_report") or {}
        if int(report.get("configured_static_obstacles", -1)) != int(
            scenario["static_obstacles"]
        ):
            raise ValueError("P060 runtime static density mismatch")
        if int(report.get("configured_dynamic_obstacles", -1)) != int(
            scenario["dynamic_obstacles"]
        ):
            raise ValueError("P060 runtime dynamic density mismatch")
        if [
            float(value)
            for value in report.get("requested_dynamic_speed_range_m_s", [])
        ] != [float(value) for value in P060_SPEED_RANGE_M_S]:
            raise ValueError("P060 runtime speed range mismatch")
        if report.get("dynamic_motion_mode") != MOTION_MODE:
            raise ValueError("P060 runtime motion mode mismatch")


def _candidate_summary(candidate: dict, by_cell: dict[str, dict]) -> dict:
    rows = [
        by_cell[cell_label(candidate["name"], scenario["name"])]
        for scenario in SCENARIOS
    ]
    return {
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "gate_rank": candidate["gate_rank"],
        "source_gate_hard_pass": candidate["gate_hard_pass"],
        "cells": {
            row["scenario"]: {
                "n": int(row["metrics"]["n"]),
                "sr": float(row["metrics"]["sr"]),
                "cr": float(row["metrics"]["cr"]),
                "to": float(row["metrics"]["to"]),
                "crossing_rate": row["metrics"].get("crossing_rate"),
                "direct_crossing_rate": row["metrics"].get(
                    "direct_crossing_rate"
                ),
                "gate_pass": bool(row["gate"]["pass"]),
            }
            for row in rows
        },
        "retention_pass": all(bool(row["gate"]["pass"]) for row in rows),
    }


def summarize_cells(payloads: list[dict]) -> dict:
    expected = {cell["cell"] for cell in cells()}
    if len(payloads) != len(expected):
        raise ValueError(f"retention screen requires exactly {len(expected)} cells")
    by_cell: dict[str, dict] = {}
    for payload in payloads:
        validate_cell(payload)
        label = str(payload["cell"])
        if label in by_cell:
            raise ValueError(f"duplicate retention cell {label}")
        by_cell[label] = payload
    if set(by_cell) != expected:
        raise ValueError("retention screen is missing a frozen cell")

    candidates = [_candidate_summary(candidate, by_cell) for candidate in CANDIDATES]
    recommendation = next(
        (
            row["checkpoint_name"]
            for row in candidates
            if bool(row["retention_pass"])
        ),
        None,
    )
    c550_pass = bool(candidates[0]["retention_pass"])
    return {
        "schema": "sa4_v3_c550_c600_retention_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN",
        "protocol_sha256": retention_protocol()["sha256"],
        "gate_summary_sha256": GATE_SUMMARY_SHA256,
        "candidate_results": candidates,
        "recommended_exposure_parent_candidate": recommendation,
        "c550_exposure_parent_decision": (
            "ELIGIBLE_FOR_BOUNDED_EXPOSURE_CURRICULUM_PILOT"
            if c550_pass
            else "NOT_ELIGIBLE_DUE_TO_RETENTION_FAILURE"
        ),
        "sa4_graduation_gate_pass": False,
        "sa4_graduated": False,
        "accepted_exposure_parent": None,
        "exposure_curriculum_started": False,
        "sa5_parent_accepted": None,
        "sa5_started": False,
        "interpretation_limit": retention_protocol()["evidence_boundary"],
    }
