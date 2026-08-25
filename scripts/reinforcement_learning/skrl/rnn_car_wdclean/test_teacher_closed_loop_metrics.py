import pytest
import torch

from rnn_car_wdclean.teacher_closed_loop_metrics import (
    TeacherClosedLoopMetrics,
)


def test_clearance_and_near_goal_outcomes_are_accounted_separately():
    metrics = TeacherClosedLoopMetrics(3, "cpu", near_goal_threshold_m=3.0)
    metrics.record_step(
        goal_distance_m=torch.tensor([2.5, 4.0, 1.0]),
        selected_clearance_m=torch.tensor([0.2, 0.4, float("inf")]),
        feasible=torch.tensor([True, True, False]),
    )
    metrics.finish_episodes(
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 1, 3]),
    )

    report = metrics.to_report()
    assert report["selected_predicted_clearance"]["samples"] == 2
    assert report["selected_predicted_clearance"]["minimum_m"] == pytest.approx(0.2)
    assert report["near_goal_selected_predicted_clearance"]["samples"] == 1
    near = report["near_goal_episode_outcomes"]
    assert near["entered_episodes"] == 2
    assert near["success"] == 1
    assert near["obstacle_collision"] == 1
    assert near["completion_rate"] == pytest.approx(0.5)
    assert near["collision_rate"] == pytest.approx(0.5)


def test_done_resets_near_goal_entry_for_the_next_episode():
    metrics = TeacherClosedLoopMetrics(1, "cpu")
    metrics.record_step(
        goal_distance_m=torch.tensor([2.0]),
        selected_clearance_m=torch.tensor([0.3]),
        feasible=torch.tensor([True]),
    )
    metrics.finish_episodes(torch.tensor([0]), torch.tensor([1]))
    metrics.record_step(
        goal_distance_m=torch.tensor([5.0]),
        selected_clearance_m=torch.tensor([0.3]),
        feasible=torch.tensor([True]),
    )
    metrics.finish_episodes(torch.tensor([0]), torch.tensor([4]))

    near = metrics.to_report()["near_goal_episode_outcomes"]
    assert near["entered_episodes"] == 1
    assert near["success"] == 1
    assert near["timeout"] == 0


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_near_goal_threshold_must_be_finite_and_positive(threshold):
    with pytest.raises(ValueError, match="near_goal_threshold_m"):
        TeacherClosedLoopMetrics(1, "cpu", near_goal_threshold_m=threshold)
