"""Frozen SA3-v3 retention screen for the promoted c300 and c200.

This module is intentionally pure. It defines the exact checkpoint identities,
the eight fixed cells, blocking thresholds, validation, and summary semantics.
Isaac Sim execution lives in the sibling runner and queue modules.
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
PHASE_A_SUMMARY = (
    REPO
    / "logs/gates/sa3_v3_p060_checkpoint_screen/screen_20260821_r1/"
    "SUMMARY.json"
)
PHASE_A_SUMMARY_SHA256 = (
    "2b95ab43c93aa141c796999966865e1eedf924fb98fd6b6c14797dc581f80e23"
)
PHASE_A_PROTOCOL_SHA256 = (
    "b127a7cd704001c01a5a50f9e0ce3e6ef7e6f8c049c8c3b2aacb905ab5095067"
)

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
P035_SPEED_RANGE_M_S = (0.25, 0.45)
P035_MOTION_MODE = "lateral"

NATIVE_SCENE = {
    "stage_name": "SA3_walls_crossing",
    "static_obstacles": 6,
    "dynamic_obstacles": 1,
    "source": (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
        "velocity/config/charge_skrl/curriculum/phases/e2e_final20_v1.py"
    ),
}

THRESHOLDS = {
    "nav_native": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.94,
        "cr_max": 0.05,
        "to_max": 0.03,
    },
    "narrow_range": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.05,
        "to_max": 0.05,
        "crossing_min": 0.95,
        "direct_crossing_min": 0.95,
    },
    "p035_0s1d_lateral": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.10,
        "to_max": 0.05,
    },
    "p035_1s1d_lateral": {
        "episodes_min": MIN_EPISODES,
        "sr_min": 0.90,
        "cr_max": 0.10,
        "to_max": 0.05,
    },
}

SCENARIOS = (
    {
        "name": "nav_native",
        "kind": "native",
        "display": "native",
    },
    {
        "name": "narrow_range",
        "kind": "narrow",
        "display": "narrow",
    },
    {
        "name": "p035_0s1d_lateral",
        "kind": "corridor",
        "display": "P035 0S1D lateral",
        "static_obstacles": 0,
        "dynamic_obstacles": 1,
    },
    {
        "name": "p035_1s1d_lateral",
        "kind": "corridor",
        "display": "P035 1S1D lateral",
        "static_obstacles": 1,
        "dynamic_obstacles": 1,
    },
)

# Order is inherited from the completed P060 screen, not re-ranked here.
CANDIDATES = (
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "filename": "checkpoint_38400.pt",
        "sha256": (
            "6888d4c4759413ddc81f1e1eb13f6fb02244823ea1977a97705f4ee2e5c79122"
        ),
        "phase_a_rank": 1,
    },
    {
        "name": "c200",
        "conceptual_iteration": 200,
        "filename": "checkpoint_25600.pt",
        "sha256": (
            "32d368e2c9e5bbca2abd11b2768e8b865148e6647ae22a8e7a281e843e9e2ded"
        ),
        "phase_a_rank": 2,
    },
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown SA3-v3 retention candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return RUN_DIR / str(data["filename"])


def scenario_by_name(name: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["name"] == name:
            return dict(scenario)
    raise ValueError(f"unknown SA3-v3 retention scenario {name!r}")


def cell_label(checkpoint_name: str, scenario_name: str) -> str:
    candidate_by_name(checkpoint_name)
    scenario_by_name(scenario_name)
    return f"{checkpoint_name}_{scenario_name}_s070_g3_d1_s818"


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


def _protocol_body() -> dict:
    return {
        "schema": "sa3_v3_retention_screen_protocol/v1",
        "purpose": (
            "apply frozen native, narrow, and historical P035 retention gates "
            "to the c300 and c200 checkpoints promoted by the P060 screen"
        ),
        "run_name": RUN_NAME,
        "stage": STAGE,
        "phase_a_source": {
            "summary": str(PHASE_A_SUMMARY.relative_to(REPO)),
            "summary_sha256": PHASE_A_SUMMARY_SHA256,
            "protocol_sha256": PHASE_A_PROTOCOL_SHA256,
            "required_promoted_order": [row["name"] for row in CANDIDATES],
        },
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "scenarios": [dict(scenario) for scenario in SCENARIOS],
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
            "p035_speed_range_m_s": list(P035_SPEED_RANGE_M_S),
            "p035_motion_mode": P035_MOTION_MODE,
        },
        "thresholds": json.loads(json.dumps(THRESHOLDS, sort_keys=True)),
        "decision": {
            "candidate_retention_pass": "all four retention cells pass",
            "recommended_candidate": (
                "first passing candidate in the preregistered Phase A order"
            ),
            "accepted_parent": None,
            "auto_launch_sa4": False,
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "retention screen; it may recommend a parent candidate but does not "
            "formally accept the parent or launch SA4"
        ),
        "forbidden": [
            "change candidate order after seeing retention results",
            "change scene, speed, delay, noise, seed, or thresholds",
            "continue to a summary with a missing or invalid cell",
            "replace the SA4 parent sentinel",
            "launch SA4",
        ],
    }


def retention_protocol() -> dict:
    payload = json.loads(json.dumps(_protocol_body(), sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_phase_a_summary(payload: dict) -> None:
    if payload.get("schema") != "sa3_v3_p060_checkpoint_screen_summary/v1":
        raise ValueError("unexpected Phase A summary schema")
    if payload.get("status") != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN":
        raise ValueError("Phase A summary is not complete and valid")
    if payload.get("protocol_sha256") != PHASE_A_PROTOCOL_SHA256:
        raise ValueError("Phase A protocol hash mismatch")
    expected = [candidate["name"] for candidate in CANDIDATES]
    if payload.get("promoted_for_retention") != expected:
        raise ValueError("Phase A promoted candidate order mismatch")
    if not bool(payload.get("source_fingerprint_stable")):
        raise ValueError("Phase A source fingerprint was not stable")
    ranked = {
        row.get("checkpoint_name"): row
        for row in payload.get("ranked_candidates", [])
    }
    for name in expected:
        if name not in ranked or not bool(ranked[name].get("hard_gate_pass")):
            raise ValueError(f"Phase A candidate {name} did not pass P060")
    if payload.get("accepted_parent") is not None:
        raise ValueError("Phase A unexpectedly accepted a parent")
    if bool(payload.get("retention_started")) or bool(payload.get("sa4_started")):
        raise ValueError("Phase A summary records an unexpected downstream start")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(scenario_name: str, metrics: dict) -> dict:
    scenario_by_name(scenario_name)
    thresholds = THRESHOLDS[scenario_name]
    n = int(metrics.get("n", 0))
    checks = {
        "episodes": n >= int(thresholds["episodes_min"]),
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
    if payload.get("schema") != "sa3_v3_retention_screen_cell/v1":
        raise ValueError("unexpected SA3-v3 retention cell schema")
    if not bool(payload.get("cell_valid")) or not bool(payload.get("formal_evidence")):
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
    if (
        _finite(payload.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("retention cell deployment scaling mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("retention cell LiDAR-noise mode mismatch")
    if (
        payload.get("lidar_distractor_eligibility")
        != LIDAR_DISTRACTOR_ELIGIBILITY
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
        or bool(speed_runtime.get("lidar_scaled"))
        or _finite(
            speed_runtime.get("deployment_speed_scale"),
            "runtime deployment_speed_scale",
        )
        != DEPLOYMENT_SPEED_SCALE
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
    verdict = evaluate_metrics(scenario["name"], metrics)
    if payload.get("gate") != verdict:
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
        contract = payload.get("scene_contract") or {}
        if contract.get("scenario") != "narrow_range":
            raise ValueError("narrow runtime contract missing")
    else:
        report = payload.get("corridor_report") or {}
        if int(report.get("configured_static_obstacles", -1)) != int(
            scenario["static_obstacles"]
        ):
            raise ValueError("P035 runtime static density mismatch")
        if int(report.get("configured_dynamic_obstacles", -1)) != int(
            scenario["dynamic_obstacles"]
        ):
            raise ValueError("P035 runtime dynamic density mismatch")
        if [float(value) for value in report.get(
            "requested_dynamic_speed_range_m_s", []
        )] != [float(value) for value in P035_SPEED_RANGE_M_S]:
            raise ValueError("P035 runtime speed range mismatch")
        if report.get("dynamic_motion_mode") != P035_MOTION_MODE:
            raise ValueError("P035 runtime motion mode mismatch")


def _candidate_summary(candidate: dict, by_cell: dict[str, dict]) -> dict:
    rows = [
        by_cell[cell_label(candidate["name"], scenario["name"])]
        for scenario in SCENARIOS
    ]
    return {
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "phase_a_rank": candidate["phase_a_rank"],
        "phase_a_p060_pass": True,
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
    return {
        "schema": "sa3_v3_retention_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN",
        "protocol_sha256": retention_protocol()["sha256"],
        "phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
        "candidate_results": candidates,
        "all_candidates_retention_pass": all(
            bool(row["retention_pass"]) for row in candidates
        ),
        "recommended_parent_candidate": recommendation,
        "accepted_parent": None,
        "sa4_parent_replaced": False,
        "sa4_started": False,
        "interpretation_limit": retention_protocol()["evidence_boundary"],
    }
