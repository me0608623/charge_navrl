import json
from pathlib import Path

import pytest

import rnn_car_wdclean.sa5_c600_2s2d_motion_state_screen as protocol
import rnn_car_wdclean.run_sa5_c600_2s2d_motion_state_screen as runner


def _phase_report():
    phases = {
        name: {
            "frame_exposure": 25,
            "frame_exposure_fraction": 0.25,
            "collision_count": 1,
            "collision_fraction": 0.25,
            "collision_enrichment": 1.0,
        }
        for name in (
            "steady",
            "pre_waypoint_1s",
            "paused",
            "post_switch_or_resume_1s",
        )
    }
    return {
        "applicable": True,
        "total_dynamic_slot_frames": 100,
        "total_dynamic_slot_collisions": 4,
        "phases": phases,
    }


def _reports(mode="lateral", teacher_mode="shadow", replaced=False):
    corridor = {
        "episodes": 1200,
        "success_rate": 0.91,
        "collision_rate": 0.08,
        "timeout_rate": 0.01,
        "wall_collision_rate": 0.01,
        "obstacle_collision_rate": 0.07,
        "dynamic_motion_mode": mode,
        "dynamic_pause_steps_range": [0, 5],
        "motion_phase_audit": _phase_report(),
    }
    behavior = {
        "schema": "teacher_interaction_metrics/v1",
        "completed_episodes": 1200,
        "wait_frame_fraction_during_interaction": 0.5,
        "reverse_frame_fraction": 0.01,
        "safe_gap": {"launch_delay_s": {"samples": 50, "p95": 0.8}},
        "vacated_side_choice": {"eligible_launches": 40, "match_fraction": 0.75},
        "episode_side_switches_after_launch": {"p95": 1.0},
        "episode_stop_to_go_transitions": {"p95": 3.0},
        "heading": {
            "u_turn_episode_fraction": 0.01,
            "full_rotation_episode_fraction": 0.001,
        },
    }
    shadow = {
        "teacher_mode": teacher_mode,
        "teacher_replaced_policy_actions": replaced,
        "teacher_spec": {"actuator_model": "fixed_d1_queue"},
        "closed_loop_outcome": corridor,
        "interaction_behavior": behavior,
    }
    return corridor, shadow


def test_protocol_covers_four_motion_families_and_patrol_phases():
    payload = protocol.protocol_payload()
    assert [row["motion_mode"] for row in protocol.SCENARIOS] == [
        "lateral",
        "longitudinal",
        "random_2d",
        "mixed_iid",
    ]
    assert payload["fixed_cell"]["static_obstacles"] == 2
    assert payload["fixed_cell"]["dynamic_obstacles"] == 2
    assert payload["fixed_cell"]["teacher_mode"] == "shadow_record_only"
    assert payload["fixed_cell"]["teacher_replaced_policy_actions"] is False
    assert payload["fixed_cell"]["dynamic_pause_steps_range"] == [0, 5]


def test_command_is_policy_only_and_keeps_deployment_contract(tmp_path):
    command = runner.build_command(
        Path("checkpoint.pt"),
        protocol.SCENARIOS[0],
        corridor_path=tmp_path / "corridor.json",
        shadow_path=tmp_path / "shadow.json",
        phase_path=tmp_path / "phase.json",
    )
    assert "--privileged_corridor_teacher_shadow" in command
    assert "--privileged_corridor_teacher" not in command
    assert command[command.index("--actuator_delay_range") + 1 : command.index("--actuator_delay_range") + 3] == ["1", "1"]
    assert command[command.index("--speed_rate") + 1] == "0.7"
    assert command[command.index("--vlp16_noise_mode") + 1] == "full"
    assert command[command.index("--lidar-distractor-eligibility") + 1] == "valid_return_only"
    assert "--long_corridor_phase_audit" in command


def test_evaluate_cell_accepts_reconciled_shadow_policy_report():
    corridor, shadow = _reports()
    result = protocol.evaluate_cell(protocol.SCENARIOS[0], corridor, shadow)
    assert result["outcome_pass"] is True
    assert result["lateral_behavior_pass"] is True


@pytest.mark.parametrize(
    ("teacher_mode", "replaced"),
    [("override", True), ("shadow", True)],
)
def test_evaluate_cell_rejects_any_policy_action_replacement(teacher_mode, replaced):
    corridor, shadow = _reports(teacher_mode=teacher_mode, replaced=replaced)
    with pytest.raises(ValueError):
        protocol.evaluate_cell(protocol.SCENARIOS[0], corridor, shadow)


def test_decision_never_authorizes_training_or_sa6():
    cells = {}
    for scenario in protocol.SCENARIOS:
        corridor, shadow = _reports(mode=scenario["motion_mode"])
        cells[scenario["name"]] = protocol.evaluate_cell(scenario, corridor, shadow)
    decision = protocol.decide(cells)
    assert decision["training_authorized"] is False
    assert decision["distillation_authorized"] is False
    assert decision["sa6_authorized"] is False
    assert decision["c600_human_acceptance_rewritten"] is False


def test_human_acceptance_record_discloses_machine_gate_waiver():
    path = Path(__file__).resolve().parents[4] / "docs/freeze/sa5_c600_human_acceptance_20260826.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "HUMAN_ACCEPTED_SA5_PARENT_C600_WITH_WAIVER"
    assert payload["machine_screen"]["candidate_pass"] is False
    assert payload["waiver"]["p060_0s1d_cr"] == pytest.approx(0.1201171875)
    assert payload["checkpoint"]["sha256"] == "c1a24684eb787a0865b007a2a712b744f0f74c39f812c62b9b18d4ed893c271a"
    assert payload["sa6_authorized"] is False
