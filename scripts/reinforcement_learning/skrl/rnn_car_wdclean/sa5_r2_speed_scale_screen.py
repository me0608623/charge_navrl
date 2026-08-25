"""Frozen c250 vehicle speed-rate screen contract and pure reducers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_r2_checkpoint_screen as parent_screen  # noqa: E402
from analyze_sa4_d3_baseline import (  # noqa: E402
    first_event_summary,
    group_summary,
)


CHECKPOINT_NAME = "c250"
CHECKPOINT = parent_screen.candidate_by_name(CHECKPOINT_NAME)
SPEED_SCALES = (1.0, 0.8, 0.7, 0.6)
SCENARIO = "corridor_lateral"
STEPS = 3000
MIN_EPISODES = 1000
SPEED_RATE_OBS_MODE = "ego"
VEHICLE_POLICY_SOURCE = (
    REPO
    / "rover_rl/src/rover_rl_inference/rover_rl_inference/policy_node.py"
)
VEHICLE_POLICY_INSTALL = (
    REPO
    / "rover_rl/install/rover_rl_inference/lib/python3.12/site-packages/"
    "rover_rl_inference/policy_node.py"
)
VEHICLE_POLICY_SOURCE_SHA256 = (
    "7ba84c14c4041eb80c9585ad5af4b7ee9c656510cee37d6fd89b87e3acbea17f"
)
VEHICLE_POLICY_INSTALL_SHA256 = (
    "2f95ac341e4444c18b67a83f6c91c74864d9ee91dcc1a1c854dbbf99b57db783"
)


def scale_label(scale: float) -> str:
    if scale not in SPEED_SCALES:
        raise ValueError(f"unsupported speed scale {scale}")
    return f"s{round(scale * 100):03d}"


def screen_protocol() -> dict:
    parent = parent_screen.screen_protocol()
    payload = {
        "schema": "sa5_r2_c250_speed_scale_screen_protocol/v2",
        "decision": {
            "checkpoint_status": "relative_best_diagnostic_candidate",
            "formal_graduation": False,
            "source_screen": str(
                REPO
                / "logs/gates/sa5_r2_checkpoint_screen/screen_20260819_r1/SUMMARY.json"
            ),
        },
        "checkpoint": {
            **CHECKPOINT,
            "path": str(parent_screen.checkpoint_path(CHECKPOINT).resolve()),
        },
        "arms": [
            {
                "label": scale_label(scale),
                "speed_rate": scale,
                "speed_rate_obs": SPEED_RATE_OBS_MODE,
            }
            for scale in SPEED_SCALES
        ],
        "fixed_evaluation": {
            "stage": parent_screen.STAGE,
            "scenario": SCENARIO,
            "corridor_density": "4S2D",
            "motion_family": "lateral",
            "seed": parent_screen.SEED,
            "num_envs": parent_screen.NUM_ENVS,
            "steps": STEPS,
            "minimum_completed_episodes": MIN_EPISODES,
            "actuator_delay_steps": parent_screen.DELAY_STEPS,
            "actuator_delay_ms": parent_screen.DELAY_MS,
            "actuator_profile": parent_screen.ACTUATOR_PROFILE,
            "lidar_noise_mode": parent_screen.LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": (
                parent_screen.LIDAR_DISTRACTOR_ELIGIBILITY
            ),
            "scene": parent["fixed_evaluation"]["scene"],
        },
        "only_independent_variable": "vehicle_policy_node.speed_rate",
        "vehicle_speed_rate_contract": {
            "semantic": "current rover policy-node time dilation",
            "speed_rate_obs": SPEED_RATE_OBS_MODE,
            "action_limits_scaled": [
                "max_linear_velocity",
                "max_linear_accel",
                "max_angular_velocity",
                "max_angular_accel",
            ],
            "observation_policy_frame": [
                "ego_acceleration",
                "ego_linear_velocity",
                "ego_angular_velocity",
                "goal_body_xy",
                "action_history",
            ],
            "observation_scale": "multiply listed policy-frame inputs by 1/rate",
            "lidar_scaled": False,
            "deployment_speed_scale": 1.0,
            "pipeline": (
                "raw observation -> policy-frame 1/rate transform -> policy -> "
                "decode with four action limits multiplied by rate -> fixed d1 "
                "actuator queue -> simulator"
            ),
            "source": {
                "path": str(VEHICLE_POLICY_SOURCE),
                "sha256": VEHICLE_POLICY_SOURCE_SHA256,
            },
            "installed_copy": {
                "path": str(VEHICLE_POLICY_INSTALL),
                "sha256": VEHICLE_POLICY_INSTALL_SHA256,
            },
        },
        "outcomes": {
            "primary": ["SR", "CR", "TO"],
            "collision_breakdown": ["obstacle_CR", "wall_CR"],
            "behavior": [
                "mean_abs_commanded_linear_speed_mps",
                "commanded_stop_fraction",
                "mean_abs_actual_body_speed_mps",
                "actual_body_stop_fraction",
            ],
            "reaction_timing": [
                "first_near_3m",
                "first_risk_warning",
                "first_issued_deceleration",
                "first_actual_deceleration",
                "first_actual_stop",
            ],
        },
        "evidence_boundary": (
            "single checkpoint, single evaluator seed, one fixed high-density "
            "lateral corridor; diagnostic vehicle speed-rate screen, not SA5 "
            "graduation and not authorization to train or start SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def reaction_timing_summary(d3_payload: dict) -> dict:
    if d3_payload.get("schema") != "sa4_d3_yield_timing/v2":
        raise ValueError("unexpected D3 timing schema")
    if d3_payload.get("mode") != "baseline":
        raise ValueError("speed screen requires identity baseline recorder")
    checks = d3_payload.get("self_check") or {}
    if checks.get("reconciliation_ok") is not True:
        raise ValueError("D3 timing recorder failed reconciliation")
    collisions = [
        event
        for event in d3_payload.get("events", [])
        if event.get("event_type") == "dynamic_collision"
    ]
    successful = [
        event
        for event in d3_payload.get("events", [])
        if event.get("event_type") == "noncollision_closest_approach"
        and event.get("episode_outcome_cause") == 1
    ]

    def summarize(events: list[dict]) -> dict:
        grouped = group_summary(events)
        return {
            "events": len(events),
            "first_near_3m": first_event_summary(events, "near_3m"),
            "first_risk_ge_0p05": first_event_summary(
                events, "risk_ge_0p05"
            ),
            "first_issued_decel": first_event_summary(
                events, "issued_decel"
            ),
            "first_actual_decel": first_event_summary(
                events, "actual_decel"
            ),
            "first_actual_stop": first_event_summary(events, "actual_stop"),
            "event_body_speed_mps": grouped["event_body_speed_mps"],
            "risk_conditioned_response": grouped[
                "risk_conditioned_response"
            ],
        }

    return {
        "dynamic_collision": summarize(collisions),
        "successful_noncollision": summarize(successful),
        "self_check": checks,
    }


def _finite(value, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite {label}: {value!r}")
    return numeric


def validate_cell(cell: dict) -> None:
    frozen = screen_protocol()
    if cell.get("protocol_sha256") != frozen["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    if cell.get("checkpoint_sha256") != CHECKPOINT["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    scale = _finite(cell.get("speed_rate"), "speed rate")
    if scale not in SPEED_SCALES:
        raise ValueError("cell speed scale is outside the frozen arms")
    if cell.get("scale_label") != scale_label(scale):
        raise ValueError("cell speed label mismatch")
    if cell.get("scenario") != SCENARIO or not cell.get("cell_valid"):
        raise ValueError("invalid scenario or cell validity")
    metrics = cell.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    for key in ("sr", "cr", "to"):
        _finite(metrics.get(key), key)
    report = cell.get("corridor_report") or {}
    if _finite(report.get("speed_rate"), "runtime speed rate") != scale:
        raise ValueError("runtime speed rate does not match the frozen arm")
    if report.get("speed_rate_obs_mode") != SPEED_RATE_OBS_MODE:
        raise ValueError("runtime speed-rate observation mode mismatch")
    if _finite(
        report.get("deployment_speed_scale"), "deployment speed scale"
    ) != 1.0:
        raise ValueError("speed-rate screen must not stack downstream scaling")
    if int(report.get("deployment_scale_samples", 0)) <= 0:
        raise ValueError("identity deployment output was not sampled")
    if _finite(
        report.get("deployment_scale_max_abs_error"), "scale error"
    ) > 1.0e-6:
        raise ValueError("deployment output scale did not reconcile")
    timing = cell.get("reaction_timing") or {}
    if (timing.get("self_check") or {}).get("reconciliation_ok") is not True:
        raise ValueError("reaction timing did not reconcile")


def summarize_cells(cells: list[dict]) -> dict:
    if len(cells) != len(SPEED_SCALES):
        raise ValueError("speed screen requires exactly four cells")
    by_scale = {}
    for cell in cells:
        validate_cell(cell)
        scale = float(cell["speed_rate"])
        if scale in by_scale:
            raise ValueError(f"duplicate speed scale {scale}")
        by_scale[scale] = cell
    if set(by_scale) != set(SPEED_SCALES):
        raise ValueError("speed screen is missing one or more frozen arms")

    baseline = by_scale[1.0]
    base_metrics = baseline["metrics"]
    rows = []
    for scale in SPEED_SCALES:
        cell = by_scale[scale]
        metrics = cell["metrics"]
        report = cell["corridor_report"]
        reaction = cell["reaction_timing"]["dynamic_collision"]
        rows.append(
            {
                "scale": scale,
                "label": scale_label(scale),
                "n": int(metrics["n"]),
                "sr": float(metrics["sr"]),
                "cr": float(metrics["cr"]),
                "to": float(metrics["to"]),
                "delta_sr_vs_1p0": float(metrics["sr"])
                - float(base_metrics["sr"]),
                "delta_cr_vs_1p0": float(metrics["cr"])
                - float(base_metrics["cr"]),
                "delta_to_vs_1p0": float(metrics["to"])
                - float(base_metrics["to"]),
                "obstacle_cr": float(report["obstacle_collision_rate"]),
                "wall_cr": float(report["wall_collision_rate"]),
                "commanded_mean_abs_speed_mps": float(
                    report["linear_speed_abs_mean_mps"]
                ),
                "commanded_stop_fraction": float(
                    report["stop_command_fraction"]
                ),
                "actual_body_mean_abs_speed_mps": float(
                    report["actual_body_linear_speed_abs_mean_mps"]
                ),
                "actual_body_stop_fraction": float(
                    report["actual_body_stop_fraction"]
                ),
                "dynamic_collision_events": int(reaction["events"]),
                "actual_decel_observed_fraction": float(
                    reaction["first_actual_decel"]["observed_fraction"]
                ),
                "actual_decel_lead_p50_s": reaction[
                    "first_actual_decel"
                ]["lead_s"]["p50"],
                "actual_stop_observed_fraction": float(
                    reaction["first_actual_stop"]["observed_fraction"]
                ),
                "actual_stop_lead_p50_s": reaction["first_actual_stop"][
                    "lead_s"
                ]["p50"],
                "cell": cell["cell"],
            }
        )
    return {
        "schema": "sa5_r2_c250_speed_scale_screen_summary/v2",
        "status": "COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC",
        "protocol_sha256": screen_protocol()["sha256"],
        "checkpoint": CHECKPOINT,
        "rows": rows,
        "scale_0p7_vs_1p0": next(row for row in rows if row["scale"] == 0.7),
        "lowest_observed_cr_scale": min(rows, key=lambda row: row["cr"])[
            "scale"
        ],
        "accepted_parent": False,
        "training_started": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
