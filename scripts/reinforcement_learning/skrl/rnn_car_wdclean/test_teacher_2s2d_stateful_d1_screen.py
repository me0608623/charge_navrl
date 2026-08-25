import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

import teacher_2s2d_stateful_d1_screen as screen


def _teacher_report():
    runtime = {
        "schema": "stateful_corridor_teacher_runtime/v1",
        "states": {"WAIT": 0.4, "COMMIT_SIDE": 0.3, "PASS": 0.3},
        "interaction_frame_fraction": 0.5,
        "override_frame_fraction": 1.0,
        "geometric_feasible_frame_fraction": 0.9,
        "used_wait_frame_fraction": 0.2,
        "used_bounded_reverse_frame_fraction": 0.01,
        "emergency_brake_frame_fraction": 0.02,
        "transitions": {
            "entered_commit": 100,
            "entered_pass": 90,
            "released": 80,
        },
        "spec": {},
    }
    return {
        "teacher_mode": "override",
        "teacher_spec": {
            "actuator_model": "fixed_d1_queue",
            "goal_denominator_floor_m": 10.0,
            "controller": "stateful",
        },
        "stateful_controller": runtime,
        "closed_loop_outcome": {
            "episodes": 1200,
            "success_rate": 0.92,
            "collision_rate": 0.08,
            "timeout_rate": 0.0,
            "wall_collision_rate": 0.01,
            "obstacle_collision_rate": 0.07,
        },
        "interaction_behavior": {
            "schema": "teacher_interaction_metrics/v1",
            "safe_gap": {
                "opportunities": 100,
                "launches": 80,
                "launch_delay_s": {"samples": 80, "p95": 0.6},
            },
            "vacated_side_choice": {
                "eligible_launches": 60,
                "match_fraction": 0.8,
            },
            "episode_side_switches_after_launch": {"p95": 1.0},
            "episode_stop_to_go_transitions": {"p95": 3.0},
            "heading": {
                "u_turn_episode_fraction": 0.01,
                "full_rotation_episode_fraction": 0.0,
            },
        },
    }


def test_protocol_freezes_stateful_controller_without_changing_scene():
    fixed = screen.frozen_protocol()["fixed_cell"]

    assert fixed["teacher_controller"] == "stateful"
    assert fixed["stateful_teacher"]["no_side_switch_after_commit"] is True
    assert fixed["stateful_teacher"]["max_reverse_distance_m"] == 0.30
    assert fixed["actuator_delay_steps"] == [1, 1]
    assert fixed["static_obstacles"] == 2
    assert fixed["dynamic_obstacles"] == 2
    assert fixed["steps"] == 4000


def test_valid_stateful_cell_can_authorize_only_k8_probe():
    cell = screen.evaluate_cell(_teacher_report())
    decision = screen.decide(
        {"lateral_lateral": cell, "mixed": copy.deepcopy(cell)}
    )

    assert decision["teacher_pass"] is True
    assert decision["next_step"] == "RUN_K8_LABEL_INFERABILITY_PROBE"
    assert decision["training_authorized"] is False
    assert decision["distillation_authorized"] is False
    assert decision["sa6_authorized"] is False


def test_missing_stateful_transitions_is_invalid_not_a_pass():
    report = _teacher_report()
    report["stateful_controller"]["transitions"]["entered_pass"] = 0

    with pytest.raises(ValueError, match="never entered PASS"):
        screen.evaluate_cell(report)


def test_state_fractions_must_reconcile():
    report = _teacher_report()
    report["stateful_controller"]["states"]["WAIT"] = 0.5

    with pytest.raises(ValueError, match="do not reconcile"):
        screen.evaluate_cell(report)


def test_shared_runner_injects_stateful_controller_before_rollout():
    source = Path(__file__).with_name(
        "run_teacher_2s2d_d1_screen.py"
    ).read_text(encoding="utf-8")

    assert '"--corridor_teacher_controller"' in source
    assert 'getattr(protocol, "CONTROLLER", "memoryless")' in source


def test_stateful_runner_imports_with_clean_pythonpath():
    runner = Path(__file__).with_name(
        "run_teacher_2s2d_stateful_d1_screen.py"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=runner.parents[4],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--expect-checkpoint-sha256" in result.stdout
