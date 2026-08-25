import math

import pytest
import torch

from rnn_car_wdclean.teacher_interaction_metrics import (
    TeacherInteractionMetrics,
)


def _record(
    metrics,
    *,
    reset=False,
    yaw=0.0,
    applied_v=0.0,
    obstacle_y=2.0,
    obstacle_vy=0.0,
    feasible=True,
    selected_y=0.0,
    selected_side_hint=None,
):
    feasible_grid = torch.tensor([[[feasible]]])
    endpoint = torch.tensor([[[[1.0, selected_y]]]])
    metrics.record_step(
        just_reset=torch.tensor([reset]),
        robot_xy_m=torch.zeros(1, 2),
        robot_yaw_rad=torch.tensor([yaw]),
        goal_xy_m=torch.tensor([[5.0, 0.0]]),
        applied_command=torch.tensor([[applied_v, 0.0]]),
        selected_endpoint_m=torch.tensor([[1.0, selected_y]]),
        selected_feasible=torch.tensor([feasible]),
        feasible_grid=feasible_grid,
        linear_velocity_grid=torch.tensor([[[0.2]]]),
        endpoint_grid_m=endpoint,
        dynamic_positions_m=torch.tensor([[[1.0, obstacle_y]]]),
        dynamic_velocities_mps=torch.tensor([[[0.0, obstacle_vy]]]),
        dynamic_valid=torch.tensor([[True]]),
        selected_side_hint=(
            None
            if selected_side_hint is None
            else torch.tensor([selected_side_hint])
        ),
    )


def test_safe_gap_launch_and_vacated_side_are_d1_aligned():
    metrics = TeacherInteractionMetrics(1, "cpu", dt_s=0.2)
    _record(
        metrics,
        reset=True,
        obstacle_y=0.10,
        obstacle_vy=-1.0,
        feasible=False,
    )
    _record(
        metrics,
        obstacle_y=-0.10,
        obstacle_vy=-1.0,
        feasible=True,
        selected_y=0.30,
    )
    _record(
        metrics,
        applied_v=0.20,
        obstacle_y=-0.20,
        obstacle_vy=-1.0,
        feasible=True,
        selected_y=0.30,
    )
    metrics.finish_episodes(torch.tensor([0]))

    report = metrics.to_report()
    assert report["safe_gap"]["opportunities"] == 1
    assert report["safe_gap"]["launches"] == 1
    assert report["safe_gap"]["launch_delay_s"]["p50"] == pytest.approx(0.2)
    assert report["vacated_side_choice"]["eligible_launches"] == 1
    assert report["vacated_side_choice"]["matches"] == 1


def test_u_turn_and_full_rotation_are_reported_separately():
    metrics = TeacherInteractionMetrics(1, "cpu", dt_s=0.2)
    for index, yaw in enumerate(
        (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0)
    ):
        _record(
            metrics,
            reset=index == 0,
            yaw=yaw,
            applied_v=-0.2 if index == 2 else 0.0,
        )
    metrics.finish_episodes(torch.tensor([0]))

    report = metrics.to_report()
    heading = report["heading"]
    assert heading["u_turn_episodes"] == 1
    assert heading["full_rotation_episodes"] == 1
    assert heading["episode_max_heading_excursion_deg"]["maximum"] == pytest.approx(180.0)
    assert report["episode_reverse_distance_m"]["maximum"] == pytest.approx(0.04)


def test_stateful_committed_side_hint_is_d1_aligned_with_launch():
    metrics = TeacherInteractionMetrics(1, "cpu", dt_s=0.2)
    _record(
        metrics,
        reset=True,
        feasible=False,
        obstacle_y=0.1,
        obstacle_vy=-1.0,
    )
    _record(
        metrics,
        feasible=True,
        obstacle_y=-0.1,
        obstacle_vy=-1.0,
        selected_y=0.3,
        selected_side_hint=1,
    )
    _record(
        metrics,
        feasible=True,
        applied_v=0.2,
        obstacle_y=-0.2,
        obstacle_vy=-1.0,
        selected_y=0.3,
        selected_side_hint=1,
    )
    metrics.finish_episodes(torch.tensor([0]))

    vacated = metrics.to_report()["vacated_side_choice"]
    assert vacated["eligible_launches"] == 1
    assert vacated["matches"] == 1


def test_launch_threshold_must_exceed_stop_threshold():
    with pytest.raises(ValueError, match="launch speed"):
        TeacherInteractionMetrics(
            1,
            "cpu",
            dt_s=0.2,
            stop_speed_mps=0.2,
            launch_speed_mps=0.1,
        )
