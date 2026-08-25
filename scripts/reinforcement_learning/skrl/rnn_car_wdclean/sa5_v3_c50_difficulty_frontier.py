"""Frozen SA5-v3 difficulty-frontier protocol for the c50 diagnostic anchor."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_v3_provisional_pilot_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

SCREEN_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c50_difficulty_frontier/screen_20260823_r1"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"

AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c50_difficulty_frontier_authorization_20260823.json"
)
AUTHORIZATION_SHA256 = (
    "11b02e2863cddea056e198cf3e28d55f6559ba46b49c91e8f62bf7c56383af87"
)

STAGE = 5
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
DYNAMIC_SPEED_RANGE = (0.25, 0.45)
STATIC_COUNTS = (1, 2, 4)
DYNAMIC_COUNTS = (1, 2)
MOTION_MODES = ("lateral", "longitudinal", "random_2d", "mixed")

ANCHOR = {
    "name": "c50",
    "conceptual_iteration": 50,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
        "checkpoint_6400.pt"
    ),
    "expected_sha256": (
        "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
    ),
    "status": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
}
CANDIDATE_SPECS = (ANCHOR,)


def _scenario_name(static_count: int, dynamic_count: int, mode: str) -> str:
    return f"s{static_count}_d{dynamic_count}_{mode}"


SCENARIO_SPECS = tuple(
    {
        "name": _scenario_name(static_count, dynamic_count, mode),
        "static_obstacles": static_count,
        "dynamic_obstacles": dynamic_count,
        "motion_mode": mode,
    }
    for static_count in STATIC_COUNTS
    for dynamic_count in DYNAMIC_COUNTS
    for mode in MOTION_MODES
)
ALL_SCENARIOS = tuple(row["name"] for row in SCENARIO_SPECS)
CORRIDOR_SCENARIOS = ALL_SCENARIOS
LOW_DENSITY_SCENARIOS: tuple[str, ...] = ()
STEPS_BY_SCENARIO = {name: STEPS for name in ALL_SCENARIOS}

THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}
CELL_SCHEMA = "sa5_v3_c50_difficulty_frontier_cell/v1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(spec: dict | str = ANCHOR) -> Path:
    if isinstance(spec, str):
        if spec != ANCHOR["name"]:
            raise ValueError(f"unknown difficulty-frontier checkpoint {spec!r}")
        spec = ANCHOR
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    if name != ANCHOR["name"]:
        raise ValueError(f"unknown difficulty-frontier checkpoint {name!r}")
    result = dict(ANCHOR)
    result["sha256"] = ANCHOR["expected_sha256"]
    return result


def scenario_by_name(name: str) -> dict:
    result = next((row for row in SCENARIO_SPECS if row["name"] == name), None)
    if result is None:
        raise ValueError(f"unknown difficulty-frontier scenario {name!r}")
    return dict(result)


def _stage_scene() -> dict:
    values = base._stage_scene()
    values["corridor_speed_range_m_s"] = list(DYNAMIC_SPEED_RANGE)
    return values


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_difficulty_frontier_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "lineage": {
            "sa4": "SA4_NOT_GRADUATED",
            "anchor": dict(ANCHOR),
            "c100": "DO_NOT_CONTINUE",
            "c150": "DO_NOT_CONTINUE",
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "matrix": {
            "static_obstacle_counts": list(STATIC_COUNTS),
            "dynamic_obstacle_counts": list(DYNAMIC_COUNTS),
            "motion_modes": list(MOTION_MODES),
            "cells": [dict(row) for row in SCENARIO_SPECS],
            "cell_count": len(SCENARIO_SPECS),
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
        "boundary_analysis": {
            "gate_crossing": "source CR <=0.10 and destination CR >0.10",
            "severe_jump": "source CR <0.10 and destination CR >=0.50",
            "dynamic_edges": "1D to 2D at fixed static count and motion mode",
            "static_edges": "1S to 2S and 2S to 4S at fixed dynamic count and motion mode",
            "random_2d_only": (
                "random_2d fails while lateral, longitudinal and mixed all pass "
                "at every tested density"
            ),
            "layout_410_isolated": (
                "within a 4S cell layout 405 passes and layout 410 has CR >=0.50"
            ),
        },
        "post_screen": {
            "pilot_budget_iterations": 50,
            "resume_c50_rl_optimizer": True,
            "allowed_behavioral_changes": [
                "scene difficulty order",
                "scene sampling proportions",
            ],
            "locked_unchanged": [
                "action",
                "observation",
                "reward",
                "network",
                "lidar noise",
                "actuator delay",
            ],
            "screen_auto_launches_pilot": False,
            "screen_auto_launches_sa6": False,
        },
        "interpretation_limit": (
            "single checkpoint and evaluator seed developmental diagnostic; "
            "layout and motion subgroups are descriptive unless separately replicated"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def thresholds_for(_: str) -> dict:
    return dict(THRESHOLDS)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(_: str, metrics: dict) -> dict:
    checks = {
        "episodes": int(metrics.get("n", 0)) >= MIN_EPISODES,
        "sr": _finite(metrics.get("sr"), "sr") >= THRESHOLDS["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= THRESHOLDS["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= THRESHOLDS["to_max"],
    }
    return {
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "threshold_pass": all(checks.values()),
    }


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid difficulty-frontier cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("difficulty-frontier protocol hash mismatch")
    if payload.get("checkpoint_name") != ANCHOR["name"]:
        raise ValueError("difficulty-frontier checkpoint name mismatch")
    if payload.get("checkpoint_sha256") != ANCHOR["expected_sha256"]:
        raise ValueError("difficulty-frontier checkpoint hash mismatch")
    scenario = scenario_by_name(str(payload.get("scenario")))
    expected_scalars = {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
    }
    for key, expected in expected_scalars.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"difficulty-frontier {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("difficulty-frontier LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("difficulty-frontier LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("difficulty-frontier speed runtime mismatch")
    contract = payload.get("scene_contract") or {}
    if contract.get("fixed_density") != (
        f"{scenario['static_obstacles']}S{scenario['dynamic_obstacles']}D"
    ):
        raise ValueError("difficulty-frontier density mismatch")
    if contract.get("motion_mode") != scenario["motion_mode"]:
        raise ValueError("difficulty-frontier motion mode mismatch")
    report = payload.get("corridor_report") or {}
    if (
        int(report.get("configured_static_obstacles", -1))
        != scenario["static_obstacles"]
        or int(report.get("configured_dynamic_obstacles", -1))
        != scenario["dynamic_obstacles"]
    ):
        raise ValueError("difficulty-frontier corridor count ledger mismatch")
    observed_speed = tuple(
        float(value)
        for value in report.get("requested_dynamic_speed_range_m_s", ())
    )
    if observed_speed != DYNAMIC_SPEED_RANGE:
        raise ValueError("difficulty-frontier dynamic speed mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("difficulty-frontier cell has too few episodes")
    if int(report.get("static_layout_episodes_total", -1)) != int(metrics["n"]):
        raise ValueError("difficulty-frontier static layout ledger mismatch")
    layouts = report.get("static_layout_outcomes") or {}
    if scenario["static_obstacles"] == 4:
        if set(layouts) != {"405", "410"}:
            raise ValueError("4S cell must contain layouts 405 and 410")
        if any(int(row.get("episodes", 0)) < 100 for row in layouts.values()):
            raise ValueError("4S layout subgroup has too few episodes")
    verdict = evaluate_metrics(str(payload["scenario"]), metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("difficulty-frontier threshold verdict mismatch")


def _row(payload: dict) -> dict:
    validate_cell(payload)
    scenario = scenario_by_name(payload["scenario"])
    metrics = payload["metrics"]
    report = payload["corridor_report"]
    return {
        **scenario,
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "gate_pass": bool(payload["threshold_pass"]),
        "wall_cr": _finite(report["wall_collision_rate"], "wall_cr"),
        "obstacle_cr": _finite(report["obstacle_collision_rate"], "obstacle_cr"),
        "layouts": report.get("static_layout_outcomes") or {},
    }


def _edge(kind: str, source: dict, destination: dict) -> dict:
    delta = destination["cr"] - source["cr"]
    return {
        "kind": kind,
        "motion_mode": source["motion_mode"],
        "source": source["name"],
        "destination": destination["name"],
        "source_cr": source["cr"],
        "destination_cr": destination["cr"],
        "delta_cr": delta,
        "gate_crossing": source["cr"] <= 0.10 and destination["cr"] > 0.10,
        "severe_jump": source["cr"] < 0.10 and destination["cr"] >= 0.50,
    }


def analyze_cells(payloads: list[dict]) -> dict:
    if len(payloads) != len(SCENARIO_SPECS):
        raise ValueError(f"expected exactly {len(SCENARIO_SPECS)} cells")
    rows = [_row(payload) for payload in payloads]
    by_key = {
        (row["static_obstacles"], row["dynamic_obstacles"], row["motion_mode"]): row
        for row in rows
    }
    if len(by_key) != len(SCENARIO_SPECS):
        raise ValueError("difficulty-frontier cells are duplicated or missing")

    edges = []
    for mode in MOTION_MODES:
        for static_count in STATIC_COUNTS:
            edges.append(
                _edge(
                    "dynamic_1D_to_2D",
                    by_key[(static_count, 1, mode)],
                    by_key[(static_count, 2, mode)],
                )
            )
        for dynamic_count in DYNAMIC_COUNTS:
            edges.append(
                _edge(
                    "static_1S_to_2S",
                    by_key[(1, dynamic_count, mode)],
                    by_key[(2, dynamic_count, mode)],
                )
            )
            edges.append(
                _edge(
                    "static_2S_to_4S",
                    by_key[(2, dynamic_count, mode)],
                    by_key[(4, dynamic_count, mode)],
                )
            )

    layout_rows = []
    for row in rows:
        if row["static_obstacles"] != 4:
            continue
        layout_405 = row["layouts"]["405"]
        layout_410 = row["layouts"]["410"]
        cr_405 = _finite(layout_405["collision_rate"], "layout405_cr")
        cr_410 = _finite(layout_410["collision_rate"], "layout410_cr")
        layout_rows.append(
            {
                "scenario": row["name"],
                "dynamic_obstacles": row["dynamic_obstacles"],
                "motion_mode": row["motion_mode"],
                "layout_405_n": int(layout_405["episodes"]),
                "layout_405_cr": cr_405,
                "layout_410_n": int(layout_410["episodes"]),
                "layout_410_cr": cr_410,
                "layout_410_minus_405_cr": cr_410 - cr_405,
                "layout_410_isolated_failure": cr_405 <= 0.10 and cr_410 >= 0.50,
            }
        )

    density_status = {}
    for static_count in STATIC_COUNTS:
        for dynamic_count in DYNAMIC_COUNTS:
            key = f"{static_count}S{dynamic_count}D"
            mode_rows = [
                by_key[(static_count, dynamic_count, mode)] for mode in MOTION_MODES
            ]
            density_status[key] = {
                "passing_modes": [row["motion_mode"] for row in mode_rows if row["gate_pass"]],
                "failing_modes": [row["motion_mode"] for row in mode_rows if not row["gate_pass"]],
                "mean_cr": sum(row["cr"] for row in mode_rows) / len(mode_rows),
                "worst_cr": max(row["cr"] for row in mode_rows),
            }

    random_only = all(
        not by_key[(s, d, "random_2d")]["gate_pass"]
        and all(by_key[(s, d, mode)]["gate_pass"] for mode in ("lateral", "longitudinal", "mixed"))
        for s in STATIC_COUNTS
        for d in DYNAMIC_COUNTS
    )
    dynamic_crossings = [
        edge for edge in edges
        if edge["kind"] == "dynamic_1D_to_2D" and edge["gate_crossing"]
    ]
    static_crossings = [
        edge for edge in edges
        if edge["kind"].startswith("static_") and edge["gate_crossing"]
    ]
    layout_isolated = [
        row for row in layout_rows if row["layout_410_isolated_failure"]
    ]

    if layout_isolated and not dynamic_crossings and not static_crossings:
        next_action = "AUDIT_LAYOUT_410_BEFORE_CURRICULUM_PILOT"
    elif random_only:
        next_action = "BUILD_RANDOM2D_EXPOSURE_PILOT_FROM_C50"
    elif dynamic_crossings and static_crossings:
        next_action = "BUILD_JOINT_1D_2D_AND_1S_2S_4S_CURRICULUM_PILOT_FROM_C50"
    elif dynamic_crossings:
        next_action = "BUILD_1D_TO_2D_CURRICULUM_PILOT_FROM_C50"
    elif static_crossings:
        next_action = "BUILD_1S_TO_2S_TO_4S_CURRICULUM_PILOT_FROM_C50"
    else:
        next_action = "HOLD_FOR_MANUAL_FRONTIER_INTERPRETATION"

    return {
        "schema": "sa5_v3_c50_difficulty_frontier_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_CHECKPOINT_SINGLE_SEED_DIAGNOSTIC",
        "anchor_status": ANCHOR["status"],
        "rows": sorted(
            rows,
            key=lambda row: (
                row["static_obstacles"],
                row["dynamic_obstacles"],
                MOTION_MODES.index(row["motion_mode"]),
            ),
        ),
        "edges": edges,
        "gate_crossings": [edge for edge in edges if edge["gate_crossing"]],
        "severe_jumps": [edge for edge in edges if edge["severe_jump"]],
        "density_status": density_status,
        "layout_405_410": layout_rows,
        "random_2d_only_failure": random_only,
        "recommended_next_action": next_action,
        "pilot_auto_started": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
