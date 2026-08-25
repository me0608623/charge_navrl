"""Frozen staged SA4-v3 retention screen for conceptual checkpoint c500."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa4_v3_c550_c600_retention as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

GATE_SUMMARY = base.GATE_SUMMARY
GATE_SUMMARY_SHA256 = base.GATE_SUMMARY_SHA256
GATE_PROTOCOL_SHA256 = base.GATE_PROTOCOL_SHA256
GATE_FREEZE = base.GATE_FREEZE
GATE_FREEZE_SHA256 = base.GATE_FREEZE_SHA256

PREVIOUS_RETENTION_SUMMARY = (
    REPO
    / "logs/gates/sa4_v3_c550_c600_retention/screen_20260822_r1/"
    "SUMMARY.json"
)
PREVIOUS_RETENTION_SUMMARY_SHA256 = (
    "620b777dc1fa611b3e24cef8cc0a6f88f14783cd2a9235418615b58a6d15f25a"
)
PREVIOUS_RETENTION_PROTOCOL_SHA256 = (
    "35ed50df1083c3c99cf809b8a3044abc9c29749e93ffad723c103ecc9e9458a1"
)
PREVIOUS_RETENTION_FREEZE = (
    REPO / "docs/freeze/sa4_v3_c550_c600_retention_v1.json"
)
PREVIOUS_RETENTION_FREEZE_SHA256 = (
    "d1ed8f6463884184cedc030aa07208af31b97cbbb6b7425dcd0d74a26460d7e2"
)

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
P060_SPEED_RANGE_M_S = base.P060_SPEED_RANGE_M_S
MOTION_MODE = base.MOTION_MODE
NATIVE_SCENE = dict(base.NATIVE_SCENE)
THRESHOLDS = json.loads(json.dumps(base.THRESHOLDS, sort_keys=True))
SCENARIOS = tuple(dict(scenario) for scenario in base.SCENARIOS)

CELL_SCHEMA = "sa4_v3_c500_staged_retention_cell/v1"
SMOKE_SCHEMA = "sa4_v3_c500_staged_retention_smoke/v1"

CANDIDATES = (
    {
        "name": "c500",
        "conceptual_iteration": 500,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_25600.pt"
        ),
        "sha256": (
            "e27677349eba8f37e2e6937f7d92400a1550e58d89d4430e556932697e9cb9f4"
        ),
        "gate_rank": 3,
        "gate_hard_pass": False,
    },
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_by_name(name: str) -> dict:
    if name == "c500":
        return dict(CANDIDATES[0])
    raise ValueError(f"unknown staged-retention candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return REPO / str(data["path"])


def scenario_by_name(name: str) -> dict:
    for scenario in SCENARIOS:
        if scenario["name"] == name:
            return dict(scenario)
    raise ValueError(f"unknown staged-retention scenario {name!r}")


def cell_label(checkpoint_name: str, scenario_name: str) -> str:
    candidate_by_name(checkpoint_name)
    scenario_by_name(scenario_name)
    return f"{checkpoint_name}_{scenario_name}_s070_g4_d1_s818"


def cells() -> tuple[dict, ...]:
    candidate = CANDIDATES[0]
    return tuple(
        {
            "cell": cell_label(candidate["name"], scenario["name"]),
            "checkpoint_name": candidate["name"],
            "scenario": scenario["name"],
            "kind": scenario["kind"],
        }
        for scenario in SCENARIOS
    )


def native_cell() -> dict:
    return dict(cells()[0])


def followup_cells() -> tuple[dict, ...]:
    return tuple(dict(cell) for cell in cells()[1:])


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
        "schema": "sa4_v3_c500_staged_retention_protocol/v1",
        "purpose": (
            "test c500 native retention first and run narrow plus two P060 "
            "low-density cells only when native passes"
        ),
        "authorization": (
            "user-authorized staged fallback after c550 and c600 both failed "
            "blocking native retention"
        ),
        "gate_source": {
            "summary": str(GATE_SUMMARY.relative_to(REPO)),
            "summary_sha256": GATE_SUMMARY_SHA256,
            "protocol_sha256": GATE_PROTOCOL_SHA256,
            "freeze": str(GATE_FREEZE.relative_to(REPO)),
            "freeze_sha256": GATE_FREEZE_SHA256,
            "required_c500_rank": 3,
        },
        "previous_retention_source": {
            "summary": str(PREVIOUS_RETENTION_SUMMARY.relative_to(REPO)),
            "summary_sha256": PREVIOUS_RETENTION_SUMMARY_SHA256,
            "protocol_sha256": PREVIOUS_RETENTION_PROTOCOL_SHA256,
            "freeze": str(PREVIOUS_RETENTION_FREEZE.relative_to(REPO)),
            "freeze_sha256": PREVIOUS_RETENTION_FREEZE_SHA256,
            "required_recommendation": None,
        },
        "stage": STAGE,
        "candidate": dict(CANDIDATES[0]),
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
        "staging": {
            "phase_1": "run nav_native only",
            "native_pass": "run all three follow-up cells",
            "native_fail": (
                "stop with a complete valid blocking verdict; do not run "
                "narrow or P060 cells"
            ),
            "candidate_pass": "native and all three follow-up cells pass",
        },
        "decision": {
            "eligible_parent": (
                "c500 is eligible only for a bounded exposure-curriculum "
                "pilot when all four retention cells pass"
            ),
            "sa4_graduation": "remains failed under the source Gate",
            "accepted_exposure_parent": None,
            "auto_launch_exposure_curriculum": False,
            "auto_launch_sa5": False,
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "staged retention screen; it may reject c500 after native alone, "
            "but cannot graduate SA4, accept a parent, or authorize SA5"
        ),
        "forbidden": [
            "run follow-up cells after native fails",
            "skip any follow-up cell after native passes",
            "change checkpoint, stage, scene, speed, delay, noise, seed, or threshold",
            "summarize invalid, duplicate, partial, or drifted evidence",
            "auto-start exposure training or SA5",
        ],
    }


def retention_protocol() -> dict:
    payload = json.loads(json.dumps(_protocol_body(), sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_sources(gate: dict, previous: dict) -> None:
    base.validate_gate_summary(gate)
    ranked = gate.get("ranked_candidates") or []
    if len(ranked) < 3 or ranked[2].get("checkpoint_name") != "c500":
        raise ValueError("source Gate does not rank c500 third")
    if int(ranked[2].get("conceptual_iteration", -1)) != 500:
        raise ValueError("source Gate c500 iteration mismatch")
    if bool(ranked[2].get("hard_gate_pass")):
        raise ValueError("source Gate unexpectedly records c500 passing")

    if previous.get("schema") != "sa4_v3_c550_c600_retention_summary/v1":
        raise ValueError("unexpected previous retention schema")
    if previous.get("status") != (
        "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN"
    ):
        raise ValueError("previous retention is not complete and valid")
    if previous.get("protocol_sha256") != PREVIOUS_RETENTION_PROTOCOL_SHA256:
        raise ValueError("previous retention protocol hash mismatch")
    if previous.get("recommended_exposure_parent_candidate") is not None:
        raise ValueError("previous retention unexpectedly recommended a parent")
    if not bool(previous.get("source_fingerprint_stable")):
        raise ValueError("previous retention fingerprint was not stable")
    results = previous.get("candidate_results") or []
    if [row.get("checkpoint_name") for row in results] != ["c550", "c600"]:
        raise ValueError("previous retention candidate set mismatch")
    for row in results:
        if bool(row.get("retention_pass")):
            raise ValueError("previous retention unexpectedly records a pass")
        native = (row.get("cells") or {}).get("nav_native") or {}
        if bool(native.get("gate_pass")):
            raise ValueError("previous retention lacks the required native failure")
    if (
        bool(previous.get("exposure_curriculum_started"))
        or bool(previous.get("sa5_started"))
        or previous.get("accepted_exposure_parent") is not None
    ):
        raise ValueError("previous retention records an unexpected downstream action")


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
    if payload.get("schema") != CELL_SCHEMA:
        raise ValueError("unexpected c500 staged-retention cell schema")
    if not bool(payload.get("cell_valid")) or not bool(
        payload.get("formal_evidence")
    ):
        raise ValueError("staged-retention cell is not formal valid evidence")
    if payload.get("protocol_sha256") != retention_protocol()["sha256"]:
        raise ValueError("staged-retention cell protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = scenario_by_name(str(payload.get("scenario")))
    if payload.get("cell") != cell_label(candidate["name"], scenario["name"]):
        raise ValueError("staged-retention cell label mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("staged-retention checkpoint hash mismatch")
    for key, expected in {
        "stage": STAGE,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        "delay_steps": DELAY_STEPS,
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"staged-retention cell {key} mismatch")
    if _finite(payload.get("speed_rate"), "speed_rate") != SPEED_RATE:
        raise ValueError("staged-retention speed_rate mismatch")
    if payload.get("speed_rate_obs") != SPEED_RATE_OBS:
        raise ValueError("staged-retention speed-rate observation mismatch")
    if _finite(
        payload.get("deployment_speed_scale"), "deployment_speed_scale"
    ) != DEPLOYMENT_SPEED_SCALE:
        raise ValueError("staged-retention deployment scaling mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("staged-retention LiDAR-noise mode mismatch")
    if payload.get("lidar_distractor_eligibility") != (
        LIDAR_DISTRACTOR_ELIGIBILITY
    ):
        raise ValueError("staged-retention LiDAR eligibility mismatch")
    actuator = payload.get("actuator_eval") or {}
    if (
        actuator.get("profile") != ACTUATOR_PROFILE
        or int(actuator.get("delay_steps", -1)) != DELAY_STEPS
        or list(actuator.get("velocity_scale_range") or []) != [1.0, 1.0]
        or _finite(actuator.get("motor_lag_alpha"), "motor_lag_alpha") != 1.0
        or actuator.get("pipeline") != "decode->delay->scale->lag"
    ):
        raise ValueError("staged-retention actuator metadata mismatch")
    speed_runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(speed_runtime.get("speed_rate"), "runtime speed_rate")
        != SPEED_RATE
        or speed_runtime.get("speed_rate_obs") != SPEED_RATE_OBS
    ):
        raise ValueError("staged-retention speed-rate runtime mismatch")
    required_limits = {
        "max_linear_velocity",
        "max_linear_accel",
        "max_angular_vel",
        "max_angular_accel",
    }
    if set((speed_runtime.get("action_limits") or {}).keys()) != required_limits:
        raise ValueError("staged-retention speed-rate action limits missing")

    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("staged-retention cell has too few episodes")
    if payload.get("gate") != evaluate_metrics(scenario["name"], metrics):
        raise ValueError("staged-retention gate payload mismatch")
    stage_banner = payload.get("stage_banner") or {}
    if int(stage_banner.get("stage_built", -1)) != STAGE:
        raise ValueError("staged-retention runtime stage mismatch")
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
        if (payload.get("scene_contract") or {}).get("scenario") != "narrow_range":
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


def _cell_result(payload: dict) -> dict:
    return {
        "n": int(payload["metrics"]["n"]),
        "sr": float(payload["metrics"]["sr"]),
        "cr": float(payload["metrics"]["cr"]),
        "to": float(payload["metrics"]["to"]),
        "crossing_rate": payload["metrics"].get("crossing_rate"),
        "direct_crossing_rate": payload["metrics"].get("direct_crossing_rate"),
        "gate_pass": bool(payload["gate"]["pass"]),
    }


def summarize_staged(payloads: list[dict]) -> dict:
    if not payloads:
        raise ValueError("staged retention cannot summarize zero cells")
    by_cell: dict[str, dict] = {}
    for payload in payloads:
        validate_cell(payload)
        label = str(payload["cell"])
        if label in by_cell:
            raise ValueError(f"duplicate staged-retention cell {label}")
        by_cell[label] = payload

    native_label = native_cell()["cell"]
    if native_label not in by_cell:
        raise ValueError("staged retention lacks the blocking native cell")
    native_pass = bool(by_cell[native_label]["gate"]["pass"])
    expected = {cell["cell"] for cell in cells()}
    if not native_pass:
        if set(by_cell) != {native_label}:
            raise ValueError("follow-up cells were run after native failed")
        status = "COMPLETE_VALID_NATIVE_BLOCKED_STAGED_RETENTION"
        remaining = [cell["cell"] for cell in followup_cells()]
    else:
        if set(by_cell) != expected or len(payloads) != len(expected):
            raise ValueError("native passed but the three follow-up cells are incomplete")
        status = "COMPLETE_VALID_FULL_STAGED_RETENTION"
        remaining = []
    all_pass = native_pass and set(by_cell) == expected and all(
        bool(payload["gate"]["pass"]) for payload in by_cell.values()
    )
    return {
        "schema": "sa4_v3_c500_staged_retention_summary/v1",
        "status": status,
        "protocol_sha256": retention_protocol()["sha256"],
        "gate_summary_sha256": GATE_SUMMARY_SHA256,
        "previous_retention_summary_sha256": PREVIOUS_RETENTION_SUMMARY_SHA256,
        "checkpoint_name": "c500",
        "conceptual_iteration": 500,
        "cells": {
            payload["scenario"]: _cell_result(payload)
            for payload in by_cell.values()
        },
        "native_pass": native_pass,
        "followup_authorized": native_pass,
        "remaining_cells_not_run": remaining,
        "retention_pass": all_pass,
        "c500_exposure_parent_decision": (
            "ELIGIBLE_FOR_BOUNDED_EXPOSURE_CURRICULUM_PILOT"
            if all_pass
            else "NOT_ELIGIBLE_DUE_TO_RETENTION_FAILURE"
        ),
        "sa4_graduation_gate_pass": False,
        "sa4_graduated": False,
        "accepted_exposure_parent": None,
        "exposure_curriculum_started": False,
        "sa5_started": False,
        "interpretation_limit": retention_protocol()["evidence_boundary"],
    }
