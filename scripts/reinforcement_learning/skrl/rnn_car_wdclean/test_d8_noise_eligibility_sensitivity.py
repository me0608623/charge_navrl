from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import d4_geometry_selector as d4
from d8_noise_eligibility_sensitivity import (
    COUNTERFACTUAL_SCHEMA,
    NoiseEligibilitySensitivity,
    noise_eligibility_protocol,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d8_noise_eligibility.py"
FREEZE = REPO / "docs/freeze/sa4_d8_noise_eligibility_sensitivity_v1.json"


def _noise_contract() -> dict:
    return {
        "displacement_std_soft": 0.008672,
        "hole_rate": 0.194859,
        "distractor_rate": 0.002515,
        "per_ring_bias": True,
        "human_dynamic_dropout": False,
        "block_dropout_prob": 0.0,
        "r_robot": 0.35,
        "r_max": 20.0,
        "num_bins": 72,
        "num_rays": 5760,
    }


def _trace(*, current_range: float = 0.60, corrected_range: float = 20.0) -> dict:
    stages = torch.full((1, 5, 72), current_range)
    current = torch.clamp(stages[:, -1] - 0.35, min=0.0, max=20.0) / 20.0
    corrected_sensor = torch.full((1, 72), corrected_range)
    corrected = torch.clamp(
        corrected_sensor - 0.35, min=0.0, max=20.0
    ) / 20.0
    angles = (torch.arange(72).float() * 5.0 - 177.5) * torch.pi / 180.0
    winner_distractor = torch.ones((1, 72), dtype=torch.bool)
    return {
        "schema": "vlp16_d7_realized_trace/v1",
        "sweep_sensor_ranges_m": stages,
        "sweep_normalized": current,
        "winner_ray_index": torch.zeros((1, 72), dtype=torch.long),
        "winner_valid": torch.ones((1, 72), dtype=torch.bool),
        "winner_raw_valid": torch.zeros((1, 72), dtype=torch.bool),
        "winner_raw_range_m": torch.full((1, 72), 20.0),
        "winner_bias_range_m": torch.full((1, 72), 20.0),
        "winner_sigma_range_m": torch.full((1, 72), 20.0),
        "winner_dropout_range_m": torch.full((1, 72), 20.0),
        "winner_final_range_m": stages[:, -1].clone(),
        "winner_hole": torch.zeros((1, 72), dtype=torch.bool),
        "winner_distractor": winner_distractor,
        "winner_hit_body_xyz_m": torch.zeros((1, 72, 3)),
        "winner_actual_angle_rad": angles[None, :],
        "winner_ring_index": torch.zeros((1, 72), dtype=torch.long),
        "winner_horizontal_index": torch.zeros((1, 72), dtype=torch.long),
        "valid_return_only_counterfactual_schema": COUNTERFACTUAL_SCHEMA,
        "valid_return_only_sweep_sensor_ranges_m": corrected_sensor,
        "valid_return_only_sweep_normalized": corrected,
        "valid_return_only_winner_ray_index": torch.zeros(
            (1, 72), dtype=torch.long
        ),
        "valid_return_only_winner_valid": torch.ones(
            (1, 72), dtype=torch.bool
        ),
        "valid_return_only_winner_actual_angle_rad": angles[None, :],
        "realized_distractor_rays": torch.tensor([72]),
        "valid_return_only_eligible_distractor_rays": torch.tensor([0]),
        "valid_return_only_removed_distractor_rays": torch.tensor([72]),
        "current_winner_removed_by_valid_return_only": torch.ones(
            (1, 72), dtype=torch.bool
        ),
        "noise_contract": _noise_contract(),
    }


def _context() -> dict:
    return {
        "obstacle_distances_m": torch.tensor([[1.50]]),
        "relative_closing_speeds_mps": torch.tensor([[1.0]]),
        "current_velocity_mps": torch.tensor([0.8]),
        "current_omega_rad_s": torch.tensor([0.0]),
        "pending_command": torch.tensor([[0.8, 0.0]]),
        "dynamic_positions_body_m": torch.tensor([[[100.0, 100.0]]]),
        "dynamic_velocities_body_mps": torch.zeros((1, 1, 2)),
        "dynamic_radii_m": torch.tensor([[0.35]]),
        "dynamic_valid": torch.tensor([[True]]),
        "num_bins": 19,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }


def test_protocol_is_paired_baseline_only_and_content_addressed():
    protocol = noise_eligibility_protocol()
    assert protocol["schema"] == "sa4_d8_noise_eligibility_sensitivity/v1"
    assert len(protocol["sha256"]) == 64
    assert "no random number is redrawn" in protocol["counterfactual_noise"]
    assert "not fed to the policy" in " ".join(protocol["interpretation_limits"])


def test_frozen_protocol_matches_runtime_protocol():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == noise_eligibility_protocol()


def test_two_frame_trigger_and_paired_recovery_do_not_modify_actions():
    audit = NoiseEligibilitySensitivity()
    trace = _trace()
    actions = torch.tensor([[18.0, 9.0]])
    for _ in range(2):
        audit.observe(
            actions,
            policy_lidar=trace["sweep_normalized"],
            trace=trace,
            trace_match_count=1,
            context=_context(),
        )
        audit.record_transition(torch.tensor([False]))
    report = audit.report(metadata={"expected_records": 2})
    paired = report["paired_feasibility"]
    assert report["runtime"]["active_environment_frames"] == 1
    assert paired["current_feasible_frames"] == 0
    assert paired["corrected_feasible_frames"] == 1
    assert paired["corrected_only_feasible_frames"] == 1
    assert report["self_check"]["reconciliation_ok"] is True
    assert torch.equal(actions, torch.tensor([[18.0, 9.0]]))


def test_active_state_matches_frozen_d4_trigger_and_release_sequence():
    audit = NoiseEligibilitySensitivity()
    selector = d4.GeometryFeasibleSelector()
    actions = torch.tensor([[18.0, 9.0]])
    trace = _trace()
    base = _context()
    sequence = (
        (1.50, 1.0),
        (1.50, 1.0),
        (4.00, 0.0),
        (4.00, 0.0),
        (1.50, 1.0),
        (1.50, 1.0),
    )
    for distance, closing in sequence:
        distances = torch.tensor([[distance]])
        closings = torch.tensor([[closing]])
        d8_active = audit._update_active(distances, closings)
        selector(
            actions,
            context={
                **base,
                "obstacle_distances_m": distances,
                "relative_closing_speeds_mps": closings,
                "policy_lidar_clearance": trace["sweep_normalized"],
            },
        )
        assert selector._active is not None
        assert torch.equal(d8_active, selector._active)


def test_counterfactual_must_not_create_nearer_ranges():
    audit = NoiseEligibilitySensitivity()
    trace = _trace(current_range=2.0, corrected_range=1.0)
    with pytest.raises(RuntimeError, match="nearer LiDAR return"):
        audit.observe(
            torch.tensor([[18.0, 9.0]]),
            policy_lidar=trace["sweep_normalized"],
            trace=trace,
            trace_match_count=1,
            context=_context(),
        )


def test_observation_counterfactual_reuses_masks_and_never_changes_policy_sweep():
    source = OBS_FUNCTIONS.read_text(encoding="utf-8")
    section = source[source.index("# D8 valid-return-only sensitivity") :]
    assert "valid_return_eligible = d7_raw_valid & ~d7_hole_mask" in section
    assert "removed_distractor = d7_distractor_mask & ~valid_return_eligible" in section
    assert "valid_return_only_per_ray = torch.where(" in section
    assert "torch.rand" not in section[: section.index('trace = {')]
    assert "trace-only and never changes ``sweep``" in section


def test_play_wires_d8_before_environment_and_keeps_baseline_actions():
    source = PLAY.read_text(encoding="utf-8")
    enable = source.index("args_cli.d8_noise_eligibility_sensitivity")
    make = source.index("env = gym.make")
    assert enable < make
    assert "D8 permits only the identity D3 baseline arm" in source
    assert "D8 shadow modified policy actions" in source
    assert "_d8_audit.record_transition(done)" in source


def test_runner_is_fixed_identity_shadow_and_fail_closed():
    source = RUNNER.read_text(encoding="utf-8")
    for literal in (
        '"geometry_stage": baseline.GEOMETRY_STAGE',
        '"seed": baseline.SEED',
        '"delay_steps": baseline.DELAY_STEPS',
        '"vlp16_noise_mode": "full"',
        '"--d3_shield_mode",\n            "baseline"',
        '"--d8_noise_eligibility_sensitivity"',
        '"policy_actions_modified": False',
        '"counterfactual_fed_to_policy": False',
        '"training_started": False',
        '"next_stage_started": False',
    ):
        assert literal in source
    assert "D8 refuses to share a busy GPU" in source
    assert "D8 audit refuses to overwrite evidence" in source
