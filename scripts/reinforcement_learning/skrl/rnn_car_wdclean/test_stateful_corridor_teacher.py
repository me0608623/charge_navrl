import torch

from rnn_car_wdclean.stateful_corridor_teacher import (
    COMMIT_SIDE,
    PASS,
    WAIT,
    StatefulCorridorTeacher,
)


def _teacher_result(*, joint: torch.Tensor | None = None):
    linear_values = torch.tensor([-0.1, 0.0, 0.2])
    angular_values = torch.tensor([-0.5, 0.0, 0.5])
    linear = linear_values[:, None].expand(3, 3).unsqueeze(0)
    angular = angular_values[None, :].expand(3, 3).unsqueeze(0)
    endpoint = torch.zeros(1, 3, 3, 2)
    endpoint[..., 0] = (linear >= 0.15).float() * 0.5
    endpoint[..., 1] = angular
    raw_cost = torch.full((1, 3, 3), 10.0)
    raw_cost[0, 2, 2] = 0.0
    raw_cost[0, 2, 0] = 1.0
    raw_cost[0, 2, 1] = 2.0
    if joint is None:
        joint = torch.ones(1, 3, 3, dtype=torch.bool)
    feasible = joint & (linear >= -1e-4)
    any_feasible = feasible.flatten(1).any(dim=1)
    return {
        "actions": torch.tensor([[2, 1]]),
        "any_feasible": any_feasible,
        "joint_feasible_grid": joint,
        "feasible_grid": feasible,
        "linear_velocity_grid": linear,
        "angular_velocity_grid": angular,
        "endpoint_grid": endpoint,
        "endpoint_yaw_grid": angular.clone(),
        "raw_cost_grid": raw_cost,
    }


def _select(
    controller,
    result,
    *,
    reset=False,
    robot_x=0.0,
    applied_v=0.0,
    dynamic_y=0.0,
    dynamic_vy=0.0,
    dynamic_x=1.0,
    dynamic_future_paths=None,
):
    return controller.select(
        teacher_result=result,
        just_reset=torch.tensor([reset]),
        robot_xy_m=torch.tensor([[robot_x, 0.0]]),
        robot_yaw_rad=torch.zeros(1),
        goal_xy_m=torch.tensor([[5.0, 0.0]]),
        applied_command=torch.tensor([[applied_v, 0.0]]),
        dynamic_positions_m=torch.tensor([[[dynamic_x, dynamic_y]]]),
        dynamic_velocities_mps=torch.tensor([[[0.0, dynamic_vy]]]),
        dynamic_valid=torch.tensor([[True]]),
        dynamic_future_paths_m=dynamic_future_paths,
    )


def test_outside_interaction_preserves_memoryless_teacher_action():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)

    selected = _select(
        controller,
        _teacher_result(),
        reset=True,
        dynamic_x=10.0,
    )

    assert selected["actions"].tolist() == [[2, 1]]
    assert selected["override_mask"].tolist() == [True]
    assert selected["state"].tolist() == [WAIT]


def test_wait_then_commits_to_recently_vacated_side():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    result["raw_cost_grid"][0, 2, 0] = -1.0

    first = _select(
        controller,
        result,
        reset=True,
        dynamic_y=0.2,
        dynamic_vy=-1.0,
    )
    crossed = _select(
        controller,
        result,
        dynamic_y=-0.2,
        dynamic_vy=-1.0,
    )
    committed = _select(
        controller,
        result,
        dynamic_y=-0.3,
        dynamic_vy=-1.0,
    )

    assert first["state"].tolist() == [WAIT]
    assert crossed["state"].tolist() == [WAIT]
    assert committed["state"].tolist() == [COMMIT_SIDE]
    assert committed["committed_side"].tolist() == [1]
    assert committed["actions"].tolist() == [[2, 2]]


def test_crossing_pedestrian_blocks_commit_until_a_side_is_vacated():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()

    first = _select(
        controller,
        result,
        reset=True,
        dynamic_y=0.5,
        dynamic_vy=-0.3,
    )
    second = _select(
        controller,
        result,
        dynamic_y=0.4,
        dynamic_vy=-0.3,
    )
    crossed = _select(
        controller,
        result,
        dynamic_y=-0.1,
        dynamic_vy=-0.3,
    )
    committed = _select(
        controller,
        result,
        dynamic_y=-0.2,
        dynamic_vy=-0.3,
    )

    assert first["state"].tolist() == [WAIT]
    assert second["state"].tolist() == [WAIT]
    assert crossed["state"].tolist() == [WAIT]
    assert committed["state"].tolist() == [COMMIT_SIDE]
    assert committed["committed_side"].tolist() == [1]


def test_paused_crossing_uses_predicted_path_and_keeps_waiting():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    predicted = torch.tensor([[[[1.0, 0.5], [1.0, -0.5]]]])

    first = _select(
        controller,
        result,
        reset=True,
        dynamic_y=0.5,
        dynamic_vy=0.0,
        dynamic_future_paths=predicted,
    )
    second = _select(
        controller,
        result,
        dynamic_y=0.5,
        dynamic_vy=0.0,
        dynamic_future_paths=predicted,
    )

    assert first["state"].tolist() == [WAIT]
    assert second["state"].tolist() == [WAIT]


def test_committed_side_does_not_switch_when_only_other_side_is_open():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    open_grid = _teacher_result()
    _select(controller, open_grid, reset=True)
    _select(controller, open_grid)

    right_only = torch.zeros(1, 3, 3, dtype=torch.bool)
    right_only[0, 2, 0] = True
    right_only[0, 1, 1] = True
    selected = _select(controller, _teacher_result(joint=right_only))

    assert selected["state"].tolist() == [COMMIT_SIDE]
    assert selected["committed_side"].tolist() == [1]
    assert selected["actions"].tolist() == [[1, 1]]
    assert selected["used_wait"].tolist() == [True]


def test_wait_proposal_is_stable_when_cost_ranking_flips():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    left_first = _teacher_result()
    first = _select(controller, left_first, reset=True)
    assert first["state"].tolist() == [WAIT]

    right_cheaper = _teacher_result()
    right_cheaper["raw_cost_grid"][0, 2, 0] = -1.0
    selected = _select(controller, right_cheaper)

    assert selected["state"].tolist() == [COMMIT_SIDE]
    assert selected["committed_side"].tolist() == [1]


def test_stopped_robot_can_commit_with_one_step_reachable_motion():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    result["linear_velocity_grid"] = torch.tensor(
        [-0.07, 0.0, 0.07]
    )[None, :, None].expand(1, 3, 3).clone()
    result["endpoint_grid"][..., 0] = (
        result["linear_velocity_grid"] > 0.0
    ).float() * 0.126
    result["endpoint_grid"][..., 1] = (
        result["angular_velocity_grid"] * 0.08
    )

    first = _select(controller, result, reset=True)
    second = _select(controller, result)

    assert first["state"].tolist() == [WAIT]
    assert second["state"].tolist() == [COMMIT_SIDE]
    assert second["actions"][0, 0].item() == 2


def test_wait_uses_bounded_reverse_then_emergency_stop():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    reverse_only = torch.zeros(1, 3, 3, dtype=torch.bool)
    reverse_only[0, 0, 1] = True
    result = _teacher_result(joint=reverse_only)

    selected = _select(controller, result, reset=True)
    assert selected["actions"].tolist() == [[0, 1]]
    assert selected["used_bounded_reverse"].tolist() == [True]

    for _ in range(15):
        selected = _select(controller, result, applied_v=-0.1)

    assert selected["actions"].tolist() == [[1, 1]]
    assert selected["used_bounded_reverse"].tolist() == [False]
    assert selected["emergency_brake"].tolist() == [True]


def test_no_geometric_candidate_uses_non_turning_emergency_stop():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result(
        joint=torch.zeros(1, 3, 3, dtype=torch.bool)
    )

    selected = _select(controller, result, reset=True)

    assert selected["actions"].tolist() == [[1, 1]]
    assert selected["override_mask"].tolist() == [True]
    assert selected["selected_geometric_feasible"].tolist() == [False]
    assert selected["emergency_brake"].tolist() == [True]


def test_backward_facing_passage_candidate_is_rejected():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    result["endpoint_yaw_grid"][0, 2, 2] = torch.pi
    result["raw_cost_grid"][0, 2, 0] = 1.0

    _select(controller, result, reset=True)
    selected = _select(controller, result)

    assert selected["state"].tolist() == [COMMIT_SIDE]
    assert selected["committed_side"].tolist() == [-1]
    assert selected["actions"].tolist() == [[2, 0]]


def test_pass_releases_only_after_clearance_is_stable():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    _select(controller, result, reset=True)
    _select(controller, result)
    passing = _select(controller, result, robot_x=0.5)
    assert passing["state"].tolist() == [PASS]

    first = _select(controller, result, robot_x=0.6, dynamic_x=10.0)
    second = _select(controller, result, robot_x=0.7, dynamic_x=10.0)
    released = _select(controller, result, robot_x=0.8, dynamic_x=10.0)
    assert first["state"].tolist() == [PASS]
    assert second["state"].tolist() == [PASS]
    assert released["state"].tolist() == [WAIT]
    assert released["released"].tolist() == [True]


def test_reset_clears_commitment_and_reverse_budget():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()
    _select(controller, result, reset=True)
    _select(controller, result)

    reset = _select(
        controller,
        result,
        reset=True,
        dynamic_x=10.0,
    )

    assert reset["state"].tolist() == [WAIT]
    assert reset["committed_side"].tolist() == [0]
