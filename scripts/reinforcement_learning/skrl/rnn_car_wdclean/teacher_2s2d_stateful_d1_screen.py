"""Frozen decision rules for the stateful 2S2D teacher diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import teacher_2s2d_d1_screen as memoryless
from stateful_corridor_teacher import StatefulTeacherSpec


SCENARIOS = memoryless.SCENARIOS
SEED = memoryless.SEED
NUM_ENVS = memoryless.NUM_ENVS
ROLLOUT_STEPS = 4000
CONTROLLER = "stateful"
SCHEMA_PREFIX = "teacher_2s2d_stateful_d1_screen"
REPORT_TITLE = "# Stateful delay-aware 2S2D privileged-teacher screen"
RUNNER_PATH = Path(__file__).with_name(
    "run_teacher_2s2d_stateful_d1_screen.py"
)


def _stateful_spec_payload() -> dict[str, object]:
    spec = StatefulTeacherSpec()
    return {
        "states": ["WAIT", "COMMIT_SIDE", "PASS"],
        "interaction_distance_m": spec.interaction_distance_m,
        "release_distance_m": spec.release_distance_m,
        "launch_speed_mps": spec.launch_speed_mps,
        "passage_progress_m": spec.passage_progress_m,
        "side_deadband_m": spec.side_deadband_m,
        "side_confirm_steps": spec.side_confirm_steps,
        "pass_progress_m": spec.pass_progress_m,
        "release_confirm_steps": spec.release_confirm_steps,
        "recent_vacated_window_s": spec.recent_vacated_window_s,
        "max_reverse_speed_mps": spec.max_reverse_speed_mps,
        "max_reverse_distance_m": spec.max_reverse_distance_m,
        "max_passage_heading_deviation_rad": (
            spec.max_passage_heading_deviation_rad
        ),
        "low_speed_reachable_fraction": spec.low_speed_reachable_fraction,
        "low_speed_side_fraction": spec.low_speed_side_fraction,
        "minimum_side_signal_m": spec.minimum_side_signal_m,
        "crossing_lateral_speed_mps": spec.crossing_lateral_speed_mps,
        "crossing_lateral_span_m": spec.crossing_lateral_span_m,
        "no_side_switch_after_commit": True,
        "crossing_requires_recent_vacated_side": True,
        "crossing_detection": "current_velocity_or_predicted_2s_lateral_span",
        "emergency_behavior": "decelerate_toward_zero_without_turning",
    }


def protocol_payload() -> dict[str, object]:
    payload = copy.deepcopy(memoryless.protocol_payload())
    payload["schema"] = "teacher_2s2d_stateful_d1_screen_protocol/v1"
    payload["purpose"] = (
        "test whether a frozen WAIT->COMMIT_SIDE->PASS teacher can wait, "
        "choose a recently vacated feasible side, preserve that commitment, "
        "and pass without looping under fixed d1"
    )
    payload["fixed_cell"]["teacher_controller"] = CONTROLLER
    payload["fixed_cell"]["steps"] = ROLLOUT_STEPS
    payload["fixed_cell"]["stateful_teacher"] = _stateful_spec_payload()
    payload["sampling_rationale"] = (
        "stateful waiting lengthens episodes; 4000 fixed steps preserve the "
        "pre-existing minimum of 1000 completed episodes without changing "
        "any behavior or acceptance threshold"
    )
    payload["validity"]["stateful_runtime_reconciliation"] = True
    payload["decision_boundary"] = (
        "pass permits a K8 teacher-label inferability probe only. Failure "
        "keeps the teacher and SA5/SA6 on hold for mechanism analysis; no "
        "outcome authorizes training or distillation"
    )
    return payload


def frozen_protocol() -> dict[str, object]:
    payload = protocol_payload()
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _finite_fraction(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid {name}={value!r}")
    return number


def evaluate_cell(teacher_report: dict[str, object]) -> dict[str, object]:
    spec = teacher_report["teacher_spec"]
    if spec.get("controller") != CONTROLLER:
        raise ValueError("teacher cell did not use the stateful controller")
    runtime = teacher_report.get("stateful_controller")
    if not isinstance(runtime, dict):
        raise ValueError("stateful teacher runtime report is missing")
    if runtime.get("schema") != "stateful_corridor_teacher_runtime/v1":
        raise ValueError("stateful teacher runtime schema mismatch")
    states = runtime["states"]
    state_sum = sum(
        _finite_fraction(states[name], f"state {name}")
        for name in ("WAIT", "COMMIT_SIDE", "PASS")
    )
    if abs(state_sum - 1.0) > 1e-6:
        raise ValueError("stateful teacher state fractions do not reconcile")
    for name in (
        "interaction_frame_fraction",
        "override_frame_fraction",
        "geometric_feasible_frame_fraction",
        "used_wait_frame_fraction",
        "used_bounded_reverse_frame_fraction",
        "emergency_brake_frame_fraction",
    ):
        _finite_fraction(runtime[name], name)
    transitions = runtime["transitions"]
    if int(transitions["entered_commit"]) <= 0:
        raise ValueError("stateful teacher never entered COMMIT_SIDE")
    if int(transitions["entered_pass"]) <= 0:
        raise ValueError("stateful teacher never entered PASS")

    result = memoryless.evaluate_cell(teacher_report)
    result["stateful_runtime"] = runtime
    return result


def decide(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    base = memoryless.decide(cells)
    passed = bool(base["teacher_pass"])
    return {
        **base,
        "schema": "teacher_2s2d_stateful_d1_screen_decision/v1",
        "next_step": (
            "RUN_K8_LABEL_INFERABILITY_PROBE"
            if passed
            else "HOLD_AND_ANALYZE_STATEFUL_TEACHER_FAILURE"
        ),
        "training_authorized": False,
        "distillation_authorized": False,
        "sa6_authorized": False,
    }


__all__ = [
    "CONTROLLER",
    "NUM_ENVS",
    "REPORT_TITLE",
    "ROLLOUT_STEPS",
    "RUNNER_PATH",
    "SCENARIOS",
    "SCHEMA_PREFIX",
    "SEED",
    "decide",
    "evaluate_cell",
    "frozen_protocol",
]
