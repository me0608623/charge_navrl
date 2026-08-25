from pathlib import Path

import teacher_2s2d_d1_screen as screen


def _teacher_report(*, launch_samples=40, vacated_events=40):
    return {
        "teacher_mode": "override",
        "teacher_spec": {
            "actuator_model": "fixed_d1_queue",
            "goal_denominator_floor_m": 10.0,
        },
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
                "opportunities": 45,
                "launches": launch_samples,
                "launch_delay_s": {"samples": launch_samples, "p95": 0.6},
            },
            "vacated_side_choice": {
                "eligible_launches": vacated_events,
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


def test_protocol_pins_exact_2s2d_d1_deployment_contract():
    fixed = screen.frozen_protocol()["fixed_cell"]

    assert fixed["static_obstacles"] == 2
    assert fixed["dynamic_obstacles"] == 2
    assert fixed["speed_rate"] == 0.7
    assert fixed["vlp16_noise_mode"] == "full"
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["actuator_delay_steps"] == [1, 1]
    assert fixed["actuator_velocity_scale"] == [1.0, 1.0]
    assert fixed["actuator_motor_lag_alpha"] == 1.0
    assert [row["motion_mode"] for row in screen.SCENARIOS] == [
        "lateral",
        "mixed",
    ]


def test_passing_cells_authorize_only_k8_probe():
    cell = screen.evaluate_cell(_teacher_report())
    decision = screen.decide(
        {"lateral_lateral": cell, "mixed": dict(cell)}
    )

    assert decision["teacher_pass"] is True
    assert decision["next_step"] == "RUN_K8_LABEL_INFERABILITY_PROBE"
    assert decision["training_authorized"] is False
    assert decision["distillation_authorized"] is False
    assert decision["sa6_authorized"] is False


def test_too_few_behavior_events_is_inconclusive_not_teacher_failure():
    cell = screen.evaluate_cell(
        _teacher_report(launch_samples=5, vacated_events=4)
    )
    decision = screen.decide(
        {"lateral_lateral": cell, "mixed": dict(cell)}
    )

    assert decision["teacher_pass"] is False
    assert decision["behavior_evidence_sufficient"] is False
    assert decision["next_step"] == "EXTEND_DIAGNOSTIC_FOR_BEHAVIOR_EVENTS"


def test_runner_passes_teacher_override_before_the_fixed_d1_pipeline():
    source = (
        Path(__file__).with_name("run_teacher_2s2d_d1_screen.py")
    ).read_text(encoding="utf-8")

    for marker in (
        '"--privileged_corridor_teacher"',
        '"--corridor_teacher_goal_denominator_floor_m"',
        '"--enable_actuator_dr"',
        '"--actuator_delay_range"',
        '"--actuator_velocity_scale"',
        '"--actuator_motor_lag"',
        '"--speed_rate"',
        '"--lidar-distractor-eligibility"',
    ):
        assert marker in source
    assert source.index('"--actuator_motor_lag"') < source.index(
        '"--privileged_corridor_teacher"'
    )
