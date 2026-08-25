from pathlib import Path

from rnn_car_wdclean import teacher_goal_denominator_ab as ab


def _cell(
    *,
    p05: float,
    near_hard: float,
    sr: float = 0.95,
    cr: float = 0.04,
    to: float = 0.01,
) -> dict[str, object]:
    return {
        "episodes": 1000,
        "successes": round(sr * 1000),
        "collisions": round(cr * 1000),
        "timeouts": round(to * 1000),
        "sr": sr,
        "cr": cr,
        "to": to,
        "wall_cr": 0.005,
        "obstacle_cr": cr - 0.005,
        "gate_pass": True,
        "action_frames": 60000,
        "linear_speed_abs_mean_mps": 0.8,
        "stop_command_fraction": 0.1,
        "near_goal_entered_episodes": 900,
        "near_goal_successes": 855,
        "near_goal_collisions": 36,
        "near_goal_timeouts": 9,
        "near_goal_completion_rate": 0.95,
        "near_goal_clearance_samples": 10000,
        "near_goal_clearance_p05_m": p05,
        "near_goal_clearance_mean_m": p05 + 0.2,
        "near_goal_near_hard_fraction": near_hard,
        "phase": {
            name: {
                "exposure": 10000,
                "collisions": 10,
                "collisions_per_frame": 0.001,
            }
            for name in (
                "paused",
                "post_switch_or_resume_1s",
                "pre_waypoint_1s",
                "steady",
            )
        },
    }


def test_protocol_keeps_everything_fixed_except_goal_denominator():
    protocol = ab.frozen_protocol()
    assert protocol["arms"] == {
        "historical_r1": 1.0,
        "candidate_r10": 10.0,
    }
    assert protocol["fixed_cell"]["dynamic_pause_steps_range"] == [0, 5]
    assert len(protocol["sha256"]) == 64


def test_candidate_pass_requires_multi_seed_clearance_and_outcome_checks():
    comparison = ab.compare(
        {
            "historical_r1": [
                _cell(p05=0.10, near_hard=0.25) for _ in ab.SEEDS
            ],
            "candidate_r10": [
                _cell(p05=0.30, near_hard=0.02) for _ in ab.SEEDS
            ],
        }
    )

    assert comparison["candidate_teacher_pass"] is True
    assert all(comparison["checks"].values())


def test_one_seed_clearance_improvement_is_not_enough():
    comparison = ab.compare(
        {
            "historical_r1": [
                _cell(p05=0.10, near_hard=0.25) for _ in ab.SEEDS
            ],
            "candidate_r10": [
                _cell(p05=0.30, near_hard=0.02),
                _cell(p05=0.12, near_hard=0.30),
                _cell(p05=0.12, near_hard=0.30),
            ],
        }
    )

    assert comparison["checks"]["near_goal_clearance"] is False
    assert comparison["candidate_teacher_pass"] is False


def test_runner_passes_every_frozen_scene_and_output_argument():
    runner = (
        Path(__file__).with_name("run_teacher_goal_denominator_ab.py")
        .read_text(encoding="utf-8")
    )
    for flag in (
        "--stage",
        "--seed",
        "--long_corridor_static_obstacles",
        "--long_corridor_dynamic_obstacles",
        "--long_corridor_free_width",
        "--long_corridor_dynamic_speed_range",
        "--long_corridor_motion_mode",
        "--long_corridor_pause_mode",
        "--long_corridor_phase_audit",
        "--long_corridor_output",
        "--long_corridor_phase_audit_output",
        "--privileged_corridor_teacher",
        "--privileged_corridor_teacher_output",
        "--corridor_teacher_goal_denominator_floor_m",
    ):
        assert f'"{flag}"' in runner
    assert "INCOMPLETE_NO_VERDICT" in runner
