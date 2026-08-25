"""Frozen decision rules for the delay-aware 2S2D teacher diagnostic."""

from __future__ import annotations

import hashlib
import json
import math


SCENARIOS = (
    {"name": "lateral_lateral", "motion_mode": "lateral"},
    {"name": "mixed", "motion_mode": "mixed"},
)
SEED = 818
NUM_ENVS = 64
ROLLOUT_STEPS = 2500
MIN_EPISODES = 1000
MIN_LAUNCH_EVENTS = 30
MIN_VACATED_SIDE_EVENTS = 30

SR_MIN = 0.90
CR_MAX = 0.10
TO_MAX = 0.05
U_TURN_RATE_MAX = 0.02
FULL_ROTATION_RATE_MAX = 0.005
LAUNCH_DELAY_P95_MAX_S = 1.0
VACATED_SIDE_MATCH_MIN = 0.70
SIDE_SWITCH_P95_MAX = 1.0
STOP_GO_P95_MAX = 4.0


def protocol_payload() -> dict[str, object]:
    return {
        "schema": "teacher_2s2d_d1_screen_protocol/v1",
        "purpose": (
            "test whether the accepted R=10 privileged teacher can wait and "
            "commit under fixed d1 in 2S2D lateral+lateral and mixed corridors"
        ),
        "evidence_scope": (
            "single checkpoint and single evaluator seed diagnostic; no "
            "generalization or distillation claim"
        ),
        "run_order": [scenario["name"] for scenario in SCENARIOS],
        "fixed_cell": {
            "task": "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
            "curriculum_version": "warp_drive_e2e_final20_v1",
            "stage": 5,
            "seed": SEED,
            "deterministic": True,
            "teacher_override": True,
            "teacher_goal_denominator_floor_m": 10.0,
            "num_envs": NUM_ENVS,
            "steps": ROLLOUT_STEPS,
            "arena_size_m": 15.0,
            "corridor_free_width_m": 4.2,
            "corridor_interaction_length_m": 10.0,
            "static_obstacles": 2,
            "dynamic_obstacles": 2,
            "dynamic_speed_range_m_s": [0.25, 0.45],
            "dynamic_pause_mode": "default",
            "speed_rate": 0.7,
            "speed_rate_observation": "ego",
            "vlp16_noise_mode": "full",
            "lidar_distractor_eligibility": "valid_return_only",
            "actuator_delay_steps": [1, 1],
            "actuator_velocity_scale": [1.0, 1.0],
            "actuator_motor_lag_alpha": 1.0,
        },
        "behavior_definitions": {
            "safe_gap": (
                "a forward candidate with >=0.20 m corridor progress has a "
                "jointly feasible delayed 2 s path"
            ),
            "u_turn": "max heading excursion from episode start >=135 deg",
            "full_rotation": "cumulative absolute yaw >=360 deg",
            "vacated_side": (
                "source side of the latest unambiguous pedestrian centerline "
                "crossing, only when that side remains feasible"
            ),
        },
        "validity": {
            "minimum_episodes_per_cell": MIN_EPISODES,
            "source_fingerprint_stable": True,
            "checkpoint_sha256_stable": True,
            "teacher_actuator_model": "fixed_d1_queue",
            "all_outputs_finite": True,
        },
        "teacher_pass_rules": {
            "sr_min": SR_MIN,
            "cr_max": CR_MAX,
            "to_max": TO_MAX,
            "u_turn_episode_fraction_max": U_TURN_RATE_MAX,
            "full_rotation_episode_fraction_max": FULL_ROTATION_RATE_MAX,
            "safe_gap_min_launch_events": MIN_LAUNCH_EVENTS,
            "safe_gap_launch_delay_p95_max_s": LAUNCH_DELAY_P95_MAX_S,
            "vacated_side_min_eligible_events": MIN_VACATED_SIDE_EVENTS,
            "vacated_side_match_fraction_min": VACATED_SIDE_MATCH_MIN,
            "side_switches_after_launch_p95_max": SIDE_SWITCH_P95_MAX,
            "stop_to_go_transitions_p95_max": STOP_GO_P95_MAX,
        },
        "decision_boundary": (
            "pass permits a K8 teacher-label inferability probe only. Failure "
            "selects a WAIT->COMMIT_SIDE->PASS teacher redesign; neither path "
            "authorizes training, distillation, or SA6"
        ),
    }


def frozen_protocol() -> dict[str, object]:
    payload = protocol_payload()
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _finite_rate(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid {name}={value!r}")
    return number


def evaluate_cell(teacher_report: dict[str, object]) -> dict[str, object]:
    if teacher_report["teacher_mode"] != "override":
        raise ValueError("teacher cell was not closed-loop override")
    spec = teacher_report["teacher_spec"]
    if spec["actuator_model"] != "fixed_d1_queue":
        raise ValueError("teacher did not model the fixed d1 queue")
    if float(spec["goal_denominator_floor_m"]) != 10.0:
        raise ValueError("teacher goal denominator drifted")

    outcome = teacher_report["closed_loop_outcome"]
    episodes = int(outcome["episodes"])
    if episodes < MIN_EPISODES:
        raise ValueError(f"too few episodes: {episodes} < {MIN_EPISODES}")
    sr = _finite_rate(outcome["success_rate"], "SR")
    cr = _finite_rate(outcome["collision_rate"], "CR")
    to = _finite_rate(outcome["timeout_rate"], "TO")
    if abs(sr + cr + to - 1.0) > 1e-6:
        raise ValueError("SR/CR/TO do not reconcile")

    behavior = teacher_report["interaction_behavior"]
    if behavior["schema"] != "teacher_interaction_metrics/v1":
        raise ValueError("teacher interaction schema mismatch")
    heading = behavior["heading"]
    safe_gap = behavior["safe_gap"]
    vacated = behavior["vacated_side_choice"]
    side_switch_p95 = behavior[
        "episode_side_switches_after_launch"
    ]["p95"]
    stop_go_p95 = behavior["episode_stop_to_go_transitions"]["p95"]
    launch_p95 = safe_gap["launch_delay_s"]["p95"]
    launch_events = int(safe_gap["launch_delay_s"]["samples"])
    vacated_events = int(vacated["eligible_launches"])

    checks = {
        "sr": sr >= SR_MIN,
        "cr": cr <= CR_MAX,
        "to": to <= TO_MAX,
        "u_turn": _finite_rate(
            heading["u_turn_episode_fraction"], "u_turn"
        )
        <= U_TURN_RATE_MAX,
        "full_rotation": _finite_rate(
            heading["full_rotation_episode_fraction"], "full_rotation"
        )
        <= FULL_ROTATION_RATE_MAX,
        "safe_gap_event_count": launch_events >= MIN_LAUNCH_EVENTS,
        "safe_gap_launch_delay": (
            launch_p95 is not None
            and float(launch_p95) <= LAUNCH_DELAY_P95_MAX_S
        ),
        "vacated_side_event_count": (
            vacated_events >= MIN_VACATED_SIDE_EVENTS
        ),
        "vacated_side_choice": (
            float(vacated["match_fraction"]) >= VACATED_SIDE_MATCH_MIN
        ),
        "side_switches": (
            side_switch_p95 is not None
            and float(side_switch_p95) <= SIDE_SWITCH_P95_MAX
        ),
        "stop_go_transitions": (
            stop_go_p95 is not None
            and float(stop_go_p95) <= STOP_GO_P95_MAX
        ),
    }
    return {
        "episodes": episodes,
        "sr": sr,
        "cr": cr,
        "to": to,
        "wall_cr": _finite_rate(outcome["wall_collision_rate"], "wall CR"),
        "obstacle_cr": _finite_rate(
            outcome["obstacle_collision_rate"], "obstacle CR"
        ),
        "safe_gap_opportunities": int(safe_gap["opportunities"]),
        "safe_gap_launches": int(safe_gap["launches"]),
        "launch_delay_p95_s": launch_p95,
        "vacated_side_eligible": vacated_events,
        "vacated_side_match_fraction": float(vacated["match_fraction"]),
        "side_switch_p95": side_switch_p95,
        "stop_go_p95": stop_go_p95,
        "u_turn_fraction": float(heading["u_turn_episode_fraction"]),
        "full_rotation_fraction": float(
            heading["full_rotation_episode_fraction"]
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def decide(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    expected = {scenario["name"] for scenario in SCENARIOS}
    if set(cells) != expected:
        raise ValueError("decision requires both frozen scenarios")
    evidence_sufficient = all(
        bool(cell["checks"]["safe_gap_event_count"])
        and bool(cell["checks"]["vacated_side_event_count"])
        for cell in cells.values()
    )
    passed = evidence_sufficient and all(
        bool(cell["pass"]) for cell in cells.values()
    )
    if passed:
        next_step = "RUN_K8_LABEL_INFERABILITY_PROBE"
    elif not evidence_sufficient:
        next_step = "EXTEND_DIAGNOSTIC_FOR_BEHAVIOR_EVENTS"
    else:
        next_step = "REDESIGN_STATEFUL_WAIT_COMMIT_SIDE_PASS"
    return {
        "schema": "teacher_2s2d_d1_screen_decision/v1",
        "cells": cells,
        "behavior_evidence_sufficient": evidence_sufficient,
        "teacher_pass": passed,
        "next_step": next_step,
        "training_authorized": False,
        "distillation_authorized": False,
        "sa6_authorized": False,
    }


__all__ = [
    "NUM_ENVS",
    "ROLLOUT_STEPS",
    "SCENARIOS",
    "SEED",
    "decide",
    "evaluate_cell",
    "frozen_protocol",
]
