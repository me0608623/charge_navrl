import torch

from rnn_car_wdclean.reward_diagnostics import (
    FutureOccupancyLeadTimeAudit,
    add_long_corridor_reward_diagnostics,
    decode_discrete_drive_action_grid,
    future_occupancy_action_counterfactuals,
)


def test_long_corridor_diagnostics_use_only_captured_scene_frames():
    breakdown = {
        "future_occupancy": torch.tensor([-0.1, -0.2, 0.0, -0.4]),
        "future_occupancy_active": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "future_occupancy_risk": torch.tensor([0.1, 0.2, 0.0, 0.4]),
        "future_occupancy_min_distance_m": torch.tensor([0.9, 0.7, 0.0, 0.3]),
        "progress_reward": torch.tensor([0.5, 0.4, 0.3, -0.1]),
        "wall_collision": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "static_obs_collision": torch.zeros(4),
        "dynamic_obs_collision": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    captured_mask = torch.tensor([False, True, False, True])

    assert add_long_corridor_reward_diagnostics(breakdown, captured_mask)
    assert torch.isclose(
        breakdown["future_occupancy_risk_long_corridor"],
        torch.tensor(0.3),
    )
    assert torch.isclose(
        breakdown["progress_reward_long_corridor"],
        torch.tensor(0.15),
    )
    assert torch.isclose(
        breakdown["future_occupancy_min_distance_m_active_long_corridor"],
        torch.tensor(0.5),
    )
    assert torch.isclose(
        breakdown[
            "future_occupancy_risk_on_dynamic_collision_long_corridor"
        ],
        torch.tensor(0.4),
    )
    assert torch.isclose(
        breakdown[
            "future_occupancy_active_on_dynamic_collision_long_corridor"
        ],
        torch.tensor(1.0),
    )


def test_long_corridor_diagnostics_skip_empty_mask():
    breakdown = {"future_occupancy_risk": torch.tensor([0.1, 0.2])}

    assert not add_long_corridor_reward_diagnostics(
        breakdown, torch.zeros(2, dtype=torch.bool)
    )
    assert set(breakdown) == {"future_occupancy_risk"}


def test_long_corridor_diagnostics_reject_shape_mismatch():
    breakdown = {"future_occupancy_risk": torch.tensor([0.1])}

    try:
        add_long_corridor_reward_diagnostics(
            breakdown, torch.tensor([True, False])
        )
    except ValueError as exc:
        assert "expected (2,)" in str(exc)
    else:
        raise AssertionError("shape mismatch must raise ValueError")


def test_discrete_drive_grid_contains_coast_and_straight_action():
    linear, angular = decode_discrete_drive_action_grid(
        torch.tensor([0.4]),
        torch.tensor([0.2]),
        num_bins=19,
        dt=0.2,
        max_linear_velocity=1.0,
        reverse_velocity_scale=0.2,
        max_linear_accel=0.5,
        max_angular_velocity=1.2,
        max_angular_accel=3.0,
    )

    assert linear.shape == (1, 19, 19)
    assert angular.shape == (1, 19, 19)
    assert torch.allclose(linear[0, 9], torch.full((19,), 0.4))
    assert torch.isclose(angular[0, 9, 9], torch.tensor(0.0))


def test_future_counterfactual_finds_safer_turn_for_crossing_obstacle():
    result = future_occupancy_action_counterfactuals(
        current_velocity=torch.tensor([0.8]),
        current_omega=torch.tensor([0.0]),
        selected_actions=torch.tensor([[9.0, 9.0]]),
        obstacle_positions_body_m=torch.tensor([[[1.0, 0.0]]]),
        obstacle_velocities_body_mps=torch.tensor([[[0.0, 0.3]]]),
        num_bins=19,
        dt=0.2,
        max_linear_velocity=1.0,
        reverse_velocity_scale=0.2,
        max_linear_accel=0.5,
        max_angular_velocity=1.2,
        max_angular_accel=3.0,
    )

    assert bool(result["active"][0])
    assert result["selected_risk"][0] > 0.0
    assert result["best_turn_risk"][0] < result["selected_risk"][0]
    assert result["best_any_risk"][0] <= result["best_turn_risk"][0]


def test_lead_time_audit_attributes_prior_safe_turn_to_collision():
    audit = FutureOccupancyLeadTimeAudit(
        1, max_lead_steps=3, device="cpu"
    )
    audit.record_step(
        selected_risk=torch.tensor([0.5]),
        best_any_risk=torch.tensor([0.0]),
        best_turn_risk=torch.tensor([0.0]),
        best_brake_risk=torch.tensor([0.4]),
        audited_mask=torch.tensor([True]),
        dynamic_collision=torch.tensor([False]),
        done=torch.tensor([False]),
    )
    audit.record_step(
        selected_risk=torch.tensor([0.2]),
        best_any_risk=torch.tensor([0.1]),
        best_turn_risk=torch.tensor([0.1]),
        best_brake_risk=torch.tensor([0.2]),
        audited_mask=torch.tensor([True]),
        dynamic_collision=torch.tensor([True]),
        done=torch.tensor([True]),
    )

    rows = audit.report(step_dt=0.2)
    lead_one = rows[0]
    assert lead_one["eligible_collision_events"] == 1
    assert lead_one["audited_risky_frames"] == 1
    assert lead_one["safe_turn_fraction_of_collisions"] == 1.0
    assert lead_one["safe_brake_fraction_of_collisions"] == 0.0
    assert lead_one["meaningful_turn_fraction_of_audited"] == 1.0
    assert rows[1]["eligible_collision_events"] == 0


def test_lead_time_audit_clears_history_at_episode_boundary():
    audit = FutureOccupancyLeadTimeAudit(
        1, max_lead_steps=2, device="cpu"
    )
    values = {
        "selected_risk": torch.tensor([0.5]),
        "best_any_risk": torch.tensor([0.0]),
        "best_turn_risk": torch.tensor([0.0]),
        "best_brake_risk": torch.tensor([0.4]),
        "audited_mask": torch.tensor([True]),
    }
    audit.record_step(
        **values,
        dynamic_collision=torch.tensor([False]),
        done=torch.tensor([False]),
    )
    audit.record_step(
        **values,
        dynamic_collision=torch.tensor([False]),
        done=torch.tensor([True]),
    )
    audit.record_step(
        **values,
        dynamic_collision=torch.tensor([True]),
        done=torch.tensor([True]),
    )

    rows = audit.report(step_dt=0.2)
    assert rows[0]["eligible_collision_events"] == 0
    assert rows[1]["eligible_collision_events"] == 0
