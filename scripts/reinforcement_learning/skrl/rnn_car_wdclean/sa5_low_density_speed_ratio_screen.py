"""Frozen low-density pedestrian/robot speed-ratio screen for SA5 c250."""

from __future__ import annotations

import hashlib
import json
import math

import sa5_pedestrian_speed_screen as pedestrian_screen
import sa5_r2_checkpoint_screen as parent_screen


CHECKPOINT_NAME = pedestrian_screen.CHECKPOINT_NAME
CHECKPOINT = pedestrian_screen.CHECKPOINT
PEDESTRIAN_SPEED_RANGE_M_S = (0.90, 1.10)
PEDESTRIAN_SPEED_LABEL = "P100"
SPEED_RATES = (1.0, 0.7)
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
MOTION_MODE = "lateral"
STEPS = 3000
MIN_EPISODES = 1000
CR_REFERENCE_EPISODES = 1000
CR_CHANGE_SE_THRESHOLD = 2.0

DENSITY_ARMS = (
    {
        "label": "0S1D",
        "scenario": "corridor_low_0s1d",
        "static_obstacles": 0,
        "dynamic_obstacles": 1,
    },
    {
        "label": "1S1D",
        "scenario": "corridor_low_1s1d",
        "static_obstacles": 1,
        "dynamic_obstacles": 1,
    },
)


def density_by_label(label: str) -> dict:
    for arm in DENSITY_ARMS:
        if arm["label"] == label:
            return dict(arm)
    raise ValueError(f"unknown density arm {label!r}")


def cell_label(density_label: str, speed_rate: float) -> str:
    density_by_label(density_label)
    rate = float(speed_rate)
    if rate not in SPEED_RATES:
        raise ValueError(f"unsupported speed rate {rate}")
    return f"{density_label.lower()}_s{round(rate * 100):03d}"


def cells() -> tuple[dict, ...]:
    return tuple(
        {
            **dict(density),
            "speed_rate": speed_rate,
            "cell": cell_label(density["label"], speed_rate),
        }
        for density in DENSITY_ARMS
        for speed_rate in SPEED_RATES
    )


def checkpoint_path():
    return pedestrian_screen.checkpoint_path()


def screen_protocol() -> dict:
    parent = parent_screen.screen_protocol()
    payload = {
        "schema": "sa5_low_density_speed_ratio_screen_protocol/v1",
        "purpose": (
            "measure the P100 pedestrian-speed boundary at low corridor density "
            "under simulation and deployment-representative robot speed rates"
        ),
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path().resolve()),
        },
        "cells": [dict(cell) for cell in cells()],
        "fixed_evaluation": {
            "stage": parent_screen.STAGE,
            "seed": parent_screen.SEED,
            "num_envs": parent_screen.NUM_ENVS,
            "steps": STEPS,
            "minimum_completed_episodes": MIN_EPISODES,
            "pedestrian_speed_label": PEDESTRIAN_SPEED_LABEL,
            "pedestrian_speed_range_m_s": list(PEDESTRIAN_SPEED_RANGE_M_S),
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "motion_mode": MOTION_MODE,
            "actuator_delay_steps": parent_screen.DELAY_STEPS,
            "actuator_delay_ms": parent_screen.DELAY_MS,
            "actuator_profile": parent_screen.ACTUATOR_PROFILE,
            "lidar_noise_mode": parent_screen.LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": (
                parent_screen.LIDAR_DISTRACTOR_ELIGIBILITY
            ),
            "scene": parent["fixed_evaluation"]["scene"],
        },
        "independent_variables": [
            "corridor obstacle density: 0S1D versus 1S1D",
            "vehicle policy speed_rate: 1.0 versus 0.7",
        ],
        "fixed_variables": [
            "checkpoint",
            "P100 pedestrian speed range",
            "corridor geometry and lateral motion family",
            "LiDAR noise and valid-return eligibility",
            "actuator delay and profile",
            "evaluator seed and rollout budget",
        ],
        "outcomes": {
            "primary": ["SR", "CR", "TO"],
            "collision_breakdown": ["obstacle_CR", "wall_CR"],
            "behavior": [
                "commanded_stop_fraction",
                "actual_body_stop_fraction",
                "first_actual_deceleration",
                "first_actual_stop",
            ],
            "impact_severity": (
                "positive radial closing component at dynamic collision p50/p90; "
                "not full 2D relative speed"
            ),
            "dynamic_feasibility": (
                "D5 dynamic-only action feasibility within final 1.0 s; model-"
                "bounded and not physical inevitability"
            ),
        },
        "interpretation_rule": {
            "historical_gate": {
                "sr_min": 0.90,
                "cr_max": 0.10,
                "to_max": 0.05,
            },
            "speed_rate_effect": (
                "within each density, call a CR change established only when its "
                "magnitude is at least 2 conservative independent-binomial SE "
                "using n=1000 per arm"
            ),
            "curriculum_boundary": {
                "P100_low_density_ready": (
                    "speed_rate=0.7 passes SR/CR/TO in both 0S1D and 1S1D"
                ),
                "P100_0S1D_only": (
                    "speed_rate=0.7 passes 0S1D but not 1S1D"
                ),
                "P100_not_entry_ready": (
                    "speed_rate=0.7 fails 0S1D; begin with P060/P080"
                ),
            },
        },
        "forbidden": [
            "training",
            "checkpoint changes",
            "reward changes",
            "geometry changes",
            "SA6 launch",
        ],
        "evidence_boundary": (
            "single checkpoint and evaluator seed diagnostic; may freeze a "
            "candidate speed-density curriculum but does not prove learnability, "
            "accept a parent, launch training, or authorize SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _gate(metrics: dict) -> dict:
    checks = {
        "sr": float(metrics["sr"]) >= 0.90,
        "cr": float(metrics["cr"]) <= 0.10,
        "to": float(metrics["to"]) <= 0.05,
    }
    return {"checks": checks, "pass": all(checks.values())}


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != "sa5_low_density_speed_ratio_cell/v1":
        raise ValueError("unexpected low-density speed-ratio cell schema")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    if payload.get("checkpoint_sha256") != CHECKPOINT["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    density = density_by_label(str(payload.get("density")))
    rate = _finite(payload.get("speed_rate"), "speed rate")
    expected_label = cell_label(density["label"], rate)
    if payload.get("cell") != expected_label:
        raise ValueError("cell label mismatch")
    if payload.get("scenario") != density["scenario"]:
        raise ValueError("cell density scenario mismatch")
    if list(payload.get("pedestrian_speed_range_m_s") or []) != list(
        PEDESTRIAN_SPEED_RANGE_M_S
    ):
        raise ValueError("cell pedestrian-speed range mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    for key in ("sr", "cr", "to"):
        _finite(metrics.get(key), key)
    report = payload.get("corridor_report") or {}
    if int(report.get("configured_static_obstacles", -1)) != density[
        "static_obstacles"
    ]:
        raise ValueError("runtime static density mismatch")
    if int(report.get("configured_dynamic_obstacles", -1)) != density[
        "dynamic_obstacles"
    ]:
        raise ValueError("runtime dynamic density mismatch")
    if list(report.get("requested_dynamic_speed_range_m_s") or []) != list(
        PEDESTRIAN_SPEED_RANGE_M_S
    ):
        raise ValueError("runtime pedestrian-speed range mismatch")
    if _finite(report.get("speed_rate"), "runtime speed rate") != rate:
        raise ValueError("runtime speed rate mismatch")
    severity = payload.get("impact_severity") or {}
    if int(severity.get("events", -1)) != int(severity.get("observed", -2)):
        raise ValueError("impact severity lacks full collision coverage")
    feasibility = payload.get("dynamic_feasibility") or {}
    if int(feasibility.get("events", -1)) != int(severity.get("events", -2)):
        raise ValueError("D3/D5 collision event counts do not reconcile")


def _comparison(baseline: dict, deployment: dict) -> dict:
    base_cr = float(baseline["metrics"]["cr"])
    deployment_cr = float(deployment["metrics"]["cr"])
    delta = deployment_cr - base_cr
    se = math.sqrt(
        base_cr * (1.0 - base_cr) / CR_REFERENCE_EPISODES
        + deployment_cr
        * (1.0 - deployment_cr)
        / CR_REFERENCE_EPISODES
    )
    standardized = 0.0 if se == 0.0 else delta / se
    if standardized >= CR_CHANGE_SE_THRESHOLD:
        direction = "SPEED_RATE_0P7_CR_WORSE"
    elif standardized <= -CR_CHANGE_SE_THRESHOLD:
        direction = "SPEED_RATE_0P7_CR_BETTER"
    else:
        direction = "NO_ESTABLISHED_CR_DIFFERENCE"
    return {
        "delta_cr_s070_minus_s100": delta,
        "conservative_se": se,
        "standardized_delta": standardized,
        "classification": direction,
    }


def summarize_cells(payloads: list[dict]) -> dict:
    expected = {cell["cell"] for cell in cells()}
    if len(payloads) != len(expected):
        raise ValueError("low-density speed-ratio screen requires exactly four cells")
    by_cell = {}
    for payload in payloads:
        validate_cell(payload)
        label = payload["cell"]
        if label in by_cell:
            raise ValueError(f"duplicate cell {label}")
        by_cell[label] = payload
    if set(by_cell) != expected:
        raise ValueError("low-density speed-ratio screen is missing a frozen cell")

    rows = []
    for spec in cells():
        payload = by_cell[spec["cell"]]
        metrics = payload["metrics"]
        report = payload["corridor_report"]
        reaction = payload["reaction_timing"]["dynamic_collision"]
        severity = payload["impact_severity"]
        feasibility = payload["dynamic_feasibility"]
        rows.append(
            {
                "cell": spec["cell"],
                "density": spec["label"],
                "speed_rate": spec["speed_rate"],
                "n": int(metrics["n"]),
                "sr": float(metrics["sr"]),
                "cr": float(metrics["cr"]),
                "to": float(metrics["to"]),
                "gate": _gate(metrics),
                "obstacle_cr": float(report["obstacle_collision_rate"]),
                "wall_cr": float(report["wall_collision_rate"]),
                "commanded_stop_fraction": float(
                    report["stop_command_fraction"]
                ),
                "actual_body_stop_fraction": float(
                    report["actual_body_stop_fraction"]
                ),
                "actual_decel_observed_fraction": float(
                    reaction["first_actual_decel"]["observed_fraction"]
                ),
                "actual_stop_observed_fraction": float(
                    reaction["first_actual_stop"]["observed_fraction"]
                ),
                "actual_stop_lead_p50_s": reaction["first_actual_stop"][
                    "lead_s"
                ]["p50"],
                "impact_radial_closing_p50_mps": severity["p50"],
                "impact_radial_closing_p90_mps": severity["p90"],
                "dynamic_feasible_within_1s_fraction": feasibility[
                    "dynamic_feasible_within_1s_fraction"
                ],
            }
        )

    comparisons = {}
    for density in DENSITY_ARMS:
        base = by_cell[cell_label(density["label"], 1.0)]
        deployment = by_cell[cell_label(density["label"], 0.7)]
        comparisons[density["label"]] = _comparison(base, deployment)

    deployment_gate = {
        row["density"]: row["gate"]["pass"]
        for row in rows
        if row["speed_rate"] == 0.7
    }
    if deployment_gate["0S1D"] and deployment_gate["1S1D"]:
        boundary = "P100_LOW_DENSITY_READY"
    elif deployment_gate["0S1D"]:
        boundary = "P100_0S1D_ONLY"
    else:
        boundary = "P100_NOT_ENTRY_READY_BEGIN_WITH_P060_P080"

    return {
        "schema": "sa5_low_density_speed_ratio_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC",
        "protocol_sha256": screen_protocol()["sha256"],
        "checkpoint": CHECKPOINT,
        "pedestrian_speed_range_m_s": list(PEDESTRIAN_SPEED_RANGE_M_S),
        "rows": rows,
        "within_density_comparisons": comparisons,
        "curriculum_boundary": boundary,
        "accepted_parent": False,
        "training_started": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
