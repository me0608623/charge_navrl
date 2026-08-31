"""Frozen policy-only 2S2D walking-state screen for accepted SA5 c600."""

from __future__ import annotations

import hashlib
import json
import math


SCENARIOS = (
    {
        "name": "lateral_lateral",
        "motion_mode": "lateral",
        "description": "two pedestrians repeatedly cross the robot path",
    },
    {
        "name": "longitudinal_longitudinal",
        "motion_mode": "longitudinal",
        "description": "two pedestrians move along the corridor",
    },
    {
        "name": "random2d_random2d",
        "motion_mode": "random_2d",
        "description": "two pedestrians follow bounded two-dimensional patrols",
    },
    {
        "name": "mixed_iid",
        "motion_mode": "mixed_iid",
        "description": "each pedestrian independently draws a motion family",
    },
)

SEED = 818
NUM_ENVS = 64
ROLLOUT_STEPS = 4000
MIN_EPISODES = 1000

SR_MIN = 0.90
CR_MAX = 0.10
TO_MAX = 0.05
U_TURN_RATE_MAX = 0.02
FULL_ROTATION_RATE_MAX = 0.005
LATERAL_MIN_LAUNCH_EVENTS = 30
LATERAL_MIN_VACATED_EVENTS = 30
LATERAL_LAUNCH_DELAY_P95_MAX_S = 1.0
LATERAL_VACATED_SIDE_MATCH_MIN = 0.70
LATERAL_SIDE_SWITCH_P95_MAX = 1.0
LATERAL_STOP_GO_P95_MAX = 4.0


def protocol_payload() -> dict[str, object]:
    return {
        "schema": "sa5_c600_2s2d_motion_state_screen_protocol/v1",
        "purpose": (
            "measure the accepted c600 policy itself under four fixed 2S2D "
            "pedestrian motion families and four patrol phases"
        ),
        "checkpoint_acceptance": {
            "mode": "HUMAN_ACCEPTED_WITH_DISCLOSED_MACHINE_GATE_WAIVER",
            "machine_gate_pass": False,
            "waived_cell": "P060_0S1D",
            "observed_cr": 0.1201171875,
            "required_cr_max": 0.10,
        },
        "run_order": [row["name"] for row in SCENARIOS],
        "fixed_cell": {
            "task": "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
            "curriculum_version": "warp_drive_e2e_final20_v1",
            "stage": 5,
            "seed": SEED,
            "deterministic": True,
            "teacher_mode": "shadow_record_only",
            "teacher_replaced_policy_actions": False,
            "num_envs": NUM_ENVS,
            "steps": ROLLOUT_STEPS,
            "arena_size_m": 15.0,
            "corridor_free_width_m": 4.2,
            "corridor_interaction_length_m": 10.0,
            "static_obstacles": 2,
            "dynamic_obstacles": 2,
            "dynamic_speed_range_m_s": [0.25, 0.45],
            "dynamic_pause_mode": "default",
            "dynamic_pause_steps_range": [0, 5],
            "random_2d_kinematics": "patrol",
            "speed_rate": 0.7,
            "speed_rate_observation": "ego",
            "vlp16_noise_mode": "full",
            "lidar_distractor_eligibility": "valid_return_only",
            "actuator_delay_steps": [1, 1],
            "actuator_velocity_scale": [1.0, 1.0],
            "actuator_motor_lag_alpha": 1.0,
        },
        "walking_state_observation": {
            "motion_families": [row["motion_mode"] for row in SCENARIOS],
            "patrol_phases": [
                "steady",
                "pre_waypoint_1s",
                "paused",
                "post_switch_or_resume_1s",
            ],
            "policy_behavior": [
                "safe_gap_launch_delay",
                "vacated_side_choice",
                "side_switches_after_launch",
                "stop_go_transitions",
                "reverse_distance",
                "u_turn",
                "full_rotation",
            ],
        },
        "validity": {
            "minimum_episodes_per_cell": MIN_EPISODES,
            "source_fingerprint_stable": True,
            "checkpoint_sha256_stable": True,
            "teacher_shadow_does_not_replace_policy": True,
            "motion_phase_reconciliation": True,
            "all_outputs_finite": True,
        },
        "descriptive_gates": {
            "all_cells": {
                "sr_min": SR_MIN,
                "cr_max": CR_MAX,
                "to_max": TO_MAX,
                "u_turn_episode_fraction_max": U_TURN_RATE_MAX,
                "full_rotation_episode_fraction_max": FULL_ROTATION_RATE_MAX,
            },
            "lateral_lateral_only": {
                "minimum_safe_gap_launch_events": LATERAL_MIN_LAUNCH_EVENTS,
                "launch_delay_p95_max_s": LATERAL_LAUNCH_DELAY_P95_MAX_S,
                "minimum_vacated_side_events": LATERAL_MIN_VACATED_EVENTS,
                "vacated_side_match_fraction_min": LATERAL_VACATED_SIDE_MATCH_MIN,
                "side_switches_after_launch_p95_max": LATERAL_SIDE_SWITCH_P95_MAX,
                "stop_to_go_transitions_p95_max": LATERAL_STOP_GO_P95_MAX,
            },
        },
        "decision_boundary": (
            "diagnostic only: results describe c600 policy behavior and do not "
            "start training, distillation, or SA6"
        ),
    }


def frozen_protocol() -> dict[str, object]:
    payload = protocol_payload()
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _rate(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid {label}: {value!r}")
    return number


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid {label}: {value!r}")
    return number


def evaluate_cell(
    scenario: dict[str, str],
    corridor_report: dict[str, object],
    shadow_report: dict[str, object],
) -> dict[str, object]:
    if shadow_report.get("teacher_mode") != "shadow":
        raise ValueError("2S2D policy screen requires teacher shadow mode")
    if bool(shadow_report.get("teacher_replaced_policy_actions")):
        raise ValueError("teacher replaced policy actions in a policy-only screen")
    teacher_spec = shadow_report.get("teacher_spec") or {}
    if teacher_spec.get("actuator_model") != "fixed_d1_queue":
        raise ValueError("shadow geometry did not model the fixed d1 queue")
    if corridor_report.get("dynamic_motion_mode") != scenario["motion_mode"]:
        raise ValueError("corridor motion mode drifted")
    if corridor_report.get("dynamic_pause_steps_range") != [0, 5]:
        raise ValueError("corridor pause range drifted")

    nested = shadow_report.get("closed_loop_outcome") or {}
    for key in ("episodes", "success_rate", "collision_rate", "timeout_rate"):
        if nested.get(key) != corridor_report.get(key):
            raise ValueError(f"shadow/corridor outcome mismatch for {key}")

    episodes = int(corridor_report["episodes"])
    sr = _rate(corridor_report["success_rate"], "SR")
    cr = _rate(corridor_report["collision_rate"], "CR")
    to = _rate(corridor_report["timeout_rate"], "TO")
    if abs(sr + cr + to - 1.0) > 1e-6:
        raise ValueError("SR/CR/TO do not reconcile")

    behavior = shadow_report.get("interaction_behavior") or {}
    if behavior.get("schema") != "teacher_interaction_metrics/v1":
        raise ValueError("policy interaction metrics are missing")
    if int(behavior.get("completed_episodes", -1)) != episodes:
        raise ValueError("interaction episode count does not reconcile")
    heading = behavior["heading"]
    safe_gap = behavior["safe_gap"]
    vacated = behavior["vacated_side_choice"]
    launch_p95 = _optional_number(
        safe_gap["launch_delay_s"]["p95"], "launch delay p95"
    )
    side_switch_p95 = _optional_number(
        behavior["episode_side_switches_after_launch"]["p95"],
        "side switch p95",
    )
    stop_go_p95 = _optional_number(
        behavior["episode_stop_to_go_transitions"]["p95"],
        "stop-go p95",
    )
    u_turn = _rate(heading["u_turn_episode_fraction"], "U-turn rate")
    full_rotation = _rate(
        heading["full_rotation_episode_fraction"], "full-rotation rate"
    )

    phase = corridor_report.get("motion_phase_audit")
    if not isinstance(phase, dict) or not bool(phase.get("applicable")):
        raise ValueError("motion-phase audit is missing or inapplicable")
    phases = phase.get("phases") or {}
    expected_phases = {
        "steady",
        "pre_waypoint_1s",
        "paused",
        "post_switch_or_resume_1s",
    }
    if set(phases) != expected_phases:
        raise ValueError("motion-phase buckets drifted")
    frame_sum = sum(int(row["frame_exposure"]) for row in phases.values())
    collision_sum = sum(int(row["collision_count"]) for row in phases.values())
    if frame_sum != int(phase["total_dynamic_slot_frames"]):
        raise ValueError("motion-phase frame counts do not reconcile")
    if collision_sum != int(phase["total_dynamic_slot_collisions"]):
        raise ValueError("motion-phase collision counts do not reconcile")

    outcome_checks = {
        "episodes": episodes >= MIN_EPISODES,
        "sr": sr >= SR_MIN,
        "cr": cr <= CR_MAX,
        "to": to <= TO_MAX,
        "u_turn": u_turn <= U_TURN_RATE_MAX,
        "full_rotation": full_rotation <= FULL_ROTATION_RATE_MAX,
    }
    lateral_checks: dict[str, bool] = {}
    if scenario["name"] == "lateral_lateral":
        launch_events = int(safe_gap["launch_delay_s"]["samples"])
        vacated_events = int(vacated["eligible_launches"])
        lateral_checks = {
            "safe_gap_event_count": launch_events >= LATERAL_MIN_LAUNCH_EVENTS,
            "safe_gap_launch_delay": (
                launch_p95 is not None
                and launch_p95 <= LATERAL_LAUNCH_DELAY_P95_MAX_S
            ),
            "vacated_side_event_count": vacated_events >= LATERAL_MIN_VACATED_EVENTS,
            "vacated_side_choice": (
                _rate(vacated["match_fraction"], "vacated-side match")
                >= LATERAL_VACATED_SIDE_MATCH_MIN
            ),
            "side_switches": (
                side_switch_p95 is not None
                and side_switch_p95 <= LATERAL_SIDE_SWITCH_P95_MAX
            ),
            "stop_go_transitions": (
                stop_go_p95 is not None
                and stop_go_p95 <= LATERAL_STOP_GO_P95_MAX
            ),
        }

    return {
        "scenario": scenario["name"],
        "motion_mode": scenario["motion_mode"],
        "episodes": episodes,
        "sr": sr,
        "cr": cr,
        "to": to,
        "wall_cr": _rate(corridor_report["wall_collision_rate"], "wall CR"),
        "obstacle_cr": _rate(
            corridor_report["obstacle_collision_rate"], "obstacle CR"
        ),
        "wait_frame_fraction_during_interaction": _rate(
            behavior["wait_frame_fraction_during_interaction"], "wait fraction"
        ),
        "reverse_frame_fraction": _rate(
            behavior["reverse_frame_fraction"], "reverse fraction"
        ),
        "launch_events": int(safe_gap["launch_delay_s"]["samples"]),
        "launch_delay_p95_s": launch_p95,
        "vacated_side_eligible": int(vacated["eligible_launches"]),
        "vacated_side_match_fraction": _rate(
            vacated["match_fraction"], "vacated-side match"
        ),
        "side_switch_p95": side_switch_p95,
        "stop_go_p95": stop_go_p95,
        "u_turn_fraction": u_turn,
        "full_rotation_fraction": full_rotation,
        "motion_phases": phases,
        "outcome_checks": outcome_checks,
        "lateral_behavior_checks": lateral_checks,
        "outcome_pass": all(outcome_checks.values()),
        "lateral_behavior_pass": (
            all(lateral_checks.values()) if lateral_checks else None
        ),
    }


def decide(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    expected = [row["name"] for row in SCENARIOS]
    if list(cells) != expected:
        raise ValueError("decision requires all four cells in frozen order")
    outcome_pass = all(bool(cell["outcome_pass"]) for cell in cells.values())
    lateral_behavior_pass = bool(
        cells["lateral_lateral"]["lateral_behavior_pass"]
    )
    return {
        "schema": "sa5_c600_2s2d_motion_state_screen_decision/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "cells": cells,
        "all_motion_outcome_pass": outcome_pass,
        "lateral_behavior_pass": lateral_behavior_pass,
        "c600_human_acceptance_rewritten": False,
        "training_authorized": False,
        "distillation_authorized": False,
        "sa6_authorized": False,
        "next_step": (
            "REVIEW_2S2D_BEHAVIOR_BEFORE_ANY_NEW_TRAINING"
            if outcome_pass and lateral_behavior_pass
            else "HOLD_AND_LOCALIZE_FAILED_2S2D_MOTION_STATES"
        ),
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
