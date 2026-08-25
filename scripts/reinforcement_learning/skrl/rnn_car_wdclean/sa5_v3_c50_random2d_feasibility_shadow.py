"""Frozen SA5-v3 c50 4S2D random-2D feasibility-shadow protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from d4_geometry_selector import speed_scaled_geometry_spec
from d5_feasibility_shadow import feasibility_shadow_protocol
import sa5_v3_c50_difficulty_frontier as frontier


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

OUTPUT_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c50_random2d_feasibility_shadow/"
    "audit_20260824_r1"
)
AUTHORIZATION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_random2d_feasibility_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "0bb6f6e05922adc406286bc99fffef8c548549c85e91fd45be387672b36398ec"
)

STAGE = 5
SEED = 818
NUM_ENVS = 64
STEPS = 2500
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
DYNAMIC_SPEED_RANGE = (0.25, 0.45)
STATIC_OBSTACLES = 4
DYNAMIC_OBSTACLES = 2
MOTION_MODE = "random_2d"
RANDOM_2D_KINEMATICS = "patrol"

CHECKPOINT = {
    "name": "c50",
    "status": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
        "checkpoint_6400.pt"
    ),
    "sha256": (
        "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
    ),
    "embedded_iteration": 49,
    "embedded_total_steps": 6400,
}

# These thresholds classify whether the frozen short-horizon model saw an
# actionable opportunity. They do not classify physical or episode solvability.
MIN_DYNAMIC_COLLISION_EVENTS = 100
MIN_SUCCESSFUL_CONTROL_EVENTS = 50
OPPORTUNITY_SEEN_FRACTION_MIN = 0.80
MOSTLY_NO_CANDIDATE_FRACTION_MIN = 0.50


def checkpoint_path() -> Path:
    return REPO / CHECKPOINT["path"]


def stage_scene() -> dict:
    values = dict(frontier._stage_scene())
    values["corridor_speed_range_m_s"] = list(DYNAMIC_SPEED_RANGE)
    return values


def protocol_payload() -> dict:
    geometry_spec = speed_scaled_geometry_spec(SPEED_RATE)
    payload = {
        "schema": "sa5_v3_c50_random2d_feasibility_shadow_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
            "maximum_wall_clock_hours": 8,
        },
        "question": (
            "within the frozen D5 short-horizon geometry model, did jointly "
            "feasible action candidates exist during the five seconds before "
            "c50 collisions in the 4S2D random_2d cell"
        ),
        "checkpoint": dict(CHECKPOINT),
        "fixed_cell": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": STEPS,
            "static_obstacles": STATIC_OBSTACLES,
            "dynamic_obstacles": DYNAMIC_OBSTACLES,
            "motion_mode": MOTION_MODE,
            "random_2d_kinematics": RANDOM_2D_KINEMATICS,
            "dynamic_speed_range_m_s": list(DYNAMIC_SPEED_RANGE),
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "scene": stage_scene(),
        },
        "instrument": {
            "mode": "baseline_identity_feasibility_shadow",
            "d5": feasibility_shadow_protocol(geometry_spec=geometry_spec),
            "candidate_actions_per_frame": 361,
            "event_trace_seconds": 5.0,
            "policy_actions_modified": False,
        },
        "minimum_evidence": {
            "dynamic_collision_events": MIN_DYNAMIC_COLLISION_EVENTS,
            "successful_noncollision_controls": MIN_SUCCESSFUL_CONTROL_EVENTS,
        },
        "decision_rules": {
            "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED": (
                "collision feasible_seen_in_window_fraction >= 0.80 and "
                "no_feasible_entire_window_fraction <= 0.20"
            ),
            "MOSTLY_NO_CANDIDATE_UNDER_FROZEN_MODEL": (
                "collision no_feasible_entire_window_fraction >= 0.50"
            ),
            "INDETERMINATE_REQUIRES_MORE_DIAGNOSTIC": (
                "valid evidence that falls between the two frozen branches"
            ),
        },
        "conditional_next_step": {
            "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED": (
                "from c50, test a bounded random_2d-only exposure pilot at one "
                "difficulty-axis increment at a time"
            ),
            "MOSTLY_NO_CANDIDATE_UNDER_FROZEN_MODEL": (
                "do not increase 4S2D exposure; map the lower-density random_2d "
                "feasibility boundary"
            ),
            "INDETERMINATE_REQUIRES_MORE_DIAGNOSTIC": (
                "run a lower-density feasibility comparison before training"
            ),
        },
        "forbidden": [
            "training in this audit",
            "action override",
            "checkpoint selection or graduation",
            "SA6 launch",
            "claiming physical inevitability or episode-level solvability",
        ],
        "interpretation_limit": (
            "D5 enumerates constant-action candidates under a frozen 2.4 s "
            "geometry model and aligns them to 5 s event windows. Results are "
            "model-bounded diagnostic evidence, not physical inevitability."
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def analyze_report(report: dict) -> dict:
    if report.get("schema") != "sa4_d5_feasibility_frontier/v1":
        raise ValueError("unexpected D5 report schema")
    if report.get("mode") != "feasibility_shadow":
        raise ValueError("unexpected D5 report mode")
    if (report.get("self_check") or {}).get("reconciliation_ok") is not True:
        raise ValueError("D5 reconciliation failed")

    counts = report.get("counts") or {}
    collision_events = int(counts.get("dynamic_collision", -1))
    control_events = int(
        counts.get("successful_noncollision_closest_approach", -1)
    )
    if collision_events < MIN_DYNAMIC_COLLISION_EVENTS:
        raise ValueError("too few dynamic-collision events")
    if control_events < MIN_SUCCESSFUL_CONTROL_EVENTS:
        raise ValueError("too few successful noncollision controls")

    summaries = report.get("frontier_summary") or {}
    collision = summaries.get("dynamic_collision") or {}
    control = summaries.get("successful_noncollision_closest_approach") or {}
    feasible_seen = _finite(
        collision.get("feasible_seen_in_window_fraction"),
        "collision feasible-seen fraction",
    )
    no_feasible_entire = _finite(
        collision.get("no_feasible_entire_window_fraction"),
        "collision no-feasible-entire-window fraction",
    )
    if (
        feasible_seen >= OPPORTUNITY_SEEN_FRACTION_MIN
        and no_feasible_entire <= 1.0 - OPPORTUNITY_SEEN_FRACTION_MIN
    ):
        decision = "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED"
    elif no_feasible_entire >= MOSTLY_NO_CANDIDATE_FRACTION_MIN:
        decision = "MOSTLY_NO_CANDIDATE_UNDER_FROZEN_MODEL"
    else:
        decision = "INDETERMINATE_REQUIRES_MORE_DIAGNOSTIC"

    return {
        "schema": "sa5_v3_c50_random2d_feasibility_shadow_analysis/v1",
        "status": "COMPLETE_VALID_MODEL_BOUNDED_DIAGNOSTIC_EVIDENCE",
        "decision": decision,
        "collision_events": collision_events,
        "successful_control_events": control_events,
        "collision_frontier": collision,
        "successful_control_frontier": control,
        "all_frame_runtime": (report.get("metadata") or {}).get(
            "shadow_runtime"
        ),
        "next_step": protocol_payload()["conditional_next_step"][decision],
        "training_started": False,
        "sa6_started": False,
        "interpretation_limit": protocol_payload()["interpretation_limit"],
    }
