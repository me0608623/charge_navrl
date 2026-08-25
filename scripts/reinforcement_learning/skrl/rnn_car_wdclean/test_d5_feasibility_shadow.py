"""CPU contracts for the SA4-D5 baseline-only shadow audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from rnn_car_wdclean import d4_geometry_selector as d4
from rnn_car_wdclean import d5_feasibility_shadow as d5


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d5_shadow_audit.py"
FREEZE = REPO / "docs/freeze/sa4_d5_feasibility_shadow_v1.json"


def _policy_lidar(sensor_range_m: torch.Tensor) -> torch.Tensor:
    return (sensor_range_m - 0.35).clamp(min=0.0) / 20.0


def _context(*, valid: bool = True) -> dict:
    positions = torch.tensor([[[2.0, -1.0]]])
    velocities = torch.tensor([[[0.0, 0.4]]])
    distance = positions.norm(dim=-1)
    robot_velocity = torch.tensor([[0.8, 0.0]])
    relative = velocities - robot_velocity[:, None, :]
    radial = positions / distance[:, :, None]
    closing = -(relative * radial).sum(dim=-1)
    return {
        "obstacle_distances_m": distance,
        "relative_closing_speeds_mps": closing,
        "current_velocity_mps": torch.tensor([0.8]),
        "current_omega_rad_s": torch.tensor([0.0]),
        "pending_command": torch.tensor([[0.8, 0.0]]),
        "policy_lidar_clearance": _policy_lidar(
            torch.full((1, 72), 20.0)
        ),
        "dynamic_positions_body_m": positions,
        "dynamic_velocities_body_mps": velocities,
        "dynamic_radii_m": torch.tensor([[0.35]]),
        "dynamic_valid": torch.tensor([[valid]]),
        "robot_velocity_body_mps": robot_velocity,
        "num_bins": 19,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }


def _record(
    step: int,
    *,
    episode_id: int = 0,
    joint: int = 1,
    dynamic: int | None = None,
    distance: float = 2.5,
    closing: float = 1.0,
    collision: bool = False,
    done: bool = False,
    cause: int = 0,
) -> d5.StepRecord:
    return d5.StepRecord(
        step=step,
        env_id=0,
        episode_id=episode_id,
        policy_action_indices=(18, 9),
        effective_action_indices=(18, 9),
        evaluated=True,
        valid_dynamic_count=1,
        dynamic_feasible_count=(max(joint, 1) if dynamic is None else dynamic),
        static_feasible_count=max(joint, 1),
        jointly_feasible_count=joint,
        policy_dynamic_feasible=joint > 0,
        policy_static_feasible=True,
        policy_jointly_feasible=joint > 0,
        nearest_distance_m=distance,
        nearest_closing_mps=closing,
        radial_ttc_s=1.0,
        radial_ttc_valid=True,
        linear_cpa_time_s=1.2,
        linear_cpa_distance_m=0.5,
        linear_cpa_valid=True,
        linear_conflict_time_s=1.2,
        linear_conflict_valid=True,
        policy_path_first_conflict_s=1.0,
        policy_path_conflict_valid=True,
        policy_path_min_dynamic_clearance_m=0.0,
        dynamic_obstacle_center_distances_m=(distance,),
        dynamic_obstacle_relative_closing_speeds_mps=(closing,),
        dynamic_collision=collision,
        done=done,
        termination_cause=cause,
    )


def test_protocol_is_baseline_only_and_content_addressed():
    protocol = d5.feasibility_shadow_protocol()
    assert protocol["schema"] == "sa4_d5_feasibility_shadow/v1"
    assert "no action override" in protocol["scope"]
    assert "no radial-TTC trigger" in protocol["evaluation_scope"]
    assert protocol["event_alignment"]["lead_seconds"] == pytest.approx(5.0)
    assert len(protocol["sha256"]) == 64
    assert protocol == d5.feasibility_shadow_protocol()


def test_shadow_never_modifies_policy_and_evaluates_all_valid_frames():
    shadow = d5.build_feasibility_shadow()
    policy = torch.tensor([[18.0, 9.0]])
    before = policy.clone()

    snapshot = shadow.observe(policy, _context())

    assert torch.equal(policy, before)
    assert bool(snapshot["evaluated"][0])
    for key in (
        "dynamic_feasible_count",
        "static_feasible_count",
        "jointly_feasible_count",
    ):
        assert 0 <= int(snapshot[key][0]) <= 361
    report = shadow.report()
    assert report["calls"] == 1
    assert report["evaluated_environment_frames"] == 1
    assert report["action_identity_errors"] == 0


def test_shadow_skips_only_frames_without_a_valid_dynamic_obstacle():
    shadow = d5.build_feasibility_shadow()
    snapshot = shadow.observe(torch.tensor([[18.0, 9.0]]), _context(valid=False))

    assert not bool(snapshot["evaluated"][0])
    assert int(snapshot["jointly_feasible_count"][0]) == -1
    assert shadow.report()["skipped_no_dynamic_frames"] == 1


def test_linear_cpa_and_policy_path_conflict_are_reported_separately():
    snapshot = d5.build_feasibility_shadow().observe(
        torch.tensor([[18.0, 9.0]]), _context()
    )

    assert bool(snapshot["linear_cpa_valid"][0])
    assert snapshot["linear_cpa_time_s"][0] < 999.0
    assert snapshot["linear_cpa_distance_m"][0] < 999.0
    assert snapshot["policy_path_min_dynamic_clearance_m"][0] < 999.0


def test_frontier_reports_closest_feasible_and_persistent_zero_suffix():
    rows = [
        _record(0, joint=4),
        _record(1, joint=0),
        _record(2, joint=0, collision=True, done=True, cause=3),
    ]

    frontier = d5.feasibility_frontier(rows, control_dt_s=0.2)

    assert frontier["closest_jointly_feasible_s_before_event"] == pytest.approx(0.4)
    assert frontier["persistent_no_feasible_onset_s_before_event"] == pytest.approx(0.2)
    assert not frontier["persistent_no_feasible_left_censored"]
    assert frontier["event_dynamic_feasible_count"] == 1
    assert frontier["dynamic_feasible_seen_in_window"]
    assert frontier["persistent_no_dynamic_feasible_onset_s_before_event"] is None


def test_frontier_reports_dynamic_only_persistent_zero_suffix():
    rows = [
        _record(0, joint=4, dynamic=4),
        _record(1, joint=0, dynamic=0),
        _record(
            2,
            joint=0,
            dynamic=0,
            collision=True,
            done=True,
            cause=3,
        ),
    ]

    frontier = d5.feasibility_frontier(rows, control_dt_s=0.2)

    assert frontier["closest_dynamic_feasible_s_before_event"] == pytest.approx(0.4)
    assert frontier[
        "persistent_no_dynamic_feasible_onset_s_before_event"
    ] == pytest.approx(0.2)
    assert not frontier["persistent_no_dynamic_feasible_left_censored"]


def test_recorder_reconciles_collision_and_successful_control_events():
    recorder = d5.FeasibilityFrontierRecorder(
        1, num_dynamic_obstacles=1
    )
    recorder.record_batch([_record(0, joint=2, distance=2.5, closing=1.0)])
    recorder.record_batch([_record(1, joint=1, distance=2.0, closing=0.0)])
    recorder.record_batch([
        _record(2, episode_id=0, joint=1, distance=3.5, closing=0.0, done=True, cause=1)
    ])
    recorder.record_batch([
        _record(3, episode_id=1, joint=0, collision=True, done=True, cause=3)
    ])
    report = recorder.report(
        metadata={
            "expected_records": 4,
            "completed_episodes": 2,
            "d3_reconciliation_ok": True,
            "shadow_runtime": {
                "environment_frames": 4,
                "evaluated_environment_frames": 4,
                "no_jointly_feasible_frames": 1,
                "action_identity_errors": 0,
            },
        }
    )

    assert report["counts"]["dynamic_collision"] == 1
    assert report["counts"]["successful_noncollision_closest_approach"] == 1
    assert report["self_check"]["reconciliation_ok"]


def test_recorder_report_preserves_speed_scaled_geometry_protocol():
    geometry = d4.speed_scaled_geometry_spec(0.7)
    recorder = d5.FeasibilityFrontierRecorder(
        1, num_dynamic_obstacles=1, geometry_spec=geometry
    )
    recorder.record_batch([
        _record(0, joint=1, collision=True, done=True, cause=3)
    ])
    report = recorder.report(
        metadata={
            "expected_records": 1,
            "completed_episodes": 1,
            "d3_reconciliation_ok": True,
            "shadow_runtime": {
                "environment_frames": 1,
                "evaluated_environment_frames": 1,
                "no_jointly_feasible_frames": 0,
                "action_identity_errors": 0,
            },
        }
    )

    expected = d5.feasibility_shadow_protocol(geometry_spec=geometry)
    assert report["protocol"] == expected
    assert report["protocol"]["geometry_protocol_sha256"] == (
        d4.geometry_selector_protocol(geometry)["sha256"]
    )


def test_tensor_batch_column_contract_preserves_cause_and_obstacle_slots():
    recorder = d5.FeasibilityFrontierRecorder(1, num_dynamic_obstacles=1)
    shadow = d5.build_feasibility_shadow()
    policy = torch.tensor([[18.0, 9.0]])
    snapshot = shadow.observe(policy, _context())
    recorder.record_tensor_batch(
        step=0,
        policy_actions=policy,
        effective_actions=policy,
        snapshot=snapshot,
        obstacle_distances_m=torch.tensor([[0.7]]),
        obstacle_closings_mps=torch.tensor([[0.4]]),
        dynamic_collision=torch.tensor([True]),
        done=torch.tensor([True]),
        termination_cause=torch.tensor([3]),
    )

    event = recorder.events[0]
    assert event["terminal_cause"] == 3
    assert event["records"][0]["dynamic_obstacle_center_distances_m"] == pytest.approx([0.7])


def test_checked_in_protocol_freeze_matches_generated_contract():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen == d5.feasibility_shadow_protocol()


def test_play_wires_shadow_before_identity_baseline_and_env_step():
    source = PLAY.read_text(encoding="utf-8")
    capture = source.index("_d3_capture_pre_step(actions, obs_tensor)")
    shadow = source.index("_d5_shadow.observe(", capture)
    baseline = source.index("_d3_effective_actions = _d3_shield(", shadow)
    step = source.index("env.step(actions.float())", baseline)
    assert capture < shadow < baseline < step
    assert "D5 shadow modified policy actions" in source
    assert '"d3_reconciliation_ok"' in source


def test_runner_preregisters_protocol_and_cannot_start_training():
    source = RUNNER.read_text(encoding="utf-8")
    protocol_write = source.index("protocol_path.write_text(")
    rollout = source.index("_run_play(", protocol_write)
    assert protocol_write < rollout
    assert '"--d5_feasibility_shadow"' in source
    assert '"--d3_shield_mode", "baseline"' in source
    for forbidden in (
        "train_rnn_car_wdclip.py",
        "systemctl start",
        "sa5-sim",
    ):
        assert forbidden not in source


def test_shadow_module_starts_no_processes():
    tree = ast.parse(Path(d5.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "subprocess" not in imports
