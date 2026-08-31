import torch

from rnn_car_wdclean.stateful_corridor_teacher import (
    COMMIT_SIDE,
    FUNNEL_MASK_BITS,
    PASS,
    WAIT,
    StatefulCorridorTeacher,
    StatefulTeacherSpec,
)


def _teacher_result(
    *,
    joint: torch.Tensor | None = None,
    obstacle_collision: torch.Tensor | None = None,
    wall_collision: torch.Tensor | None = None,
    endpoint_yaw: torch.Tensor | None = None,
    static_collision: torch.Tensor | None = None,
    dynamic_collision: torch.Tensor | None = None,
):
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
    zeros = torch.zeros(1, 3, 3, dtype=torch.bool)
    if obstacle_collision is None and wall_collision is None:
        # Legacy call style: attribute all infeasibility to the obstacle mask.
        if joint is None:
            joint = torch.ones(1, 3, 3, dtype=torch.bool)
        obstacle_collision, wall_collision = ~joint, zeros
    else:
        obstacle_collision = (
            zeros if obstacle_collision is None else obstacle_collision
        )
        wall_collision = zeros if wall_collision is None else wall_collision
        joint = ~obstacle_collision & ~wall_collision
    if static_collision is None and dynamic_collision is None:
        # Legacy call style: attribute all obstacle failures to the static
        # slot group so pre-existing tests keep their exact meaning.
        static_collision = obstacle_collision
        dynamic_collision = torch.zeros_like(obstacle_collision)
    else:
        static_collision = zeros if static_collision is None else static_collision
        dynamic_collision = (
            zeros if dynamic_collision is None else dynamic_collision
        )
        # Keep the aggregate mask reconciled with the explicit split, exactly
        # as the real geometric teacher guarantees by construction.
        obstacle_collision = static_collision | dynamic_collision
        joint = ~obstacle_collision & ~wall_collision
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
        "endpoint_yaw_grid": (
            angular.clone() if endpoint_yaw is None else endpoint_yaw
        ),
        "raw_cost_grid": raw_cost,
        "obstacle_collision_grid": obstacle_collision,
        "wall_collision_grid": wall_collision,
        "static_obstacle_collision_grid": static_collision,
        "dynamic_obstacle_collision_grid": dynamic_collision,
    }


def _funnel_teacher_result(
    *,
    static_collision: bool = False,
    dynamic_collision: bool = False,
    wall_collision: bool = False,
    linear_mps: float = 0.5,
    forward_progress_m: float = 0.5,
    heading_deviation_rad: float = 0.0,
    lateral_offset_m: float = 0.5,
    anchor_lateral_m: float = -0.5,
):
    """A two-cell grid isolating exactly one candidate to inspect.

    Column 0 is a fixed, fully-feasible "right" anchor: it keeps the
    env-level reachable/threshold statistics at their ordinary (non-
    degenerate) values regardless of what column 1 does, so column 1's
    failure can be attributed to exactly one of the seven masks. Column 1 is
    the "left" cell under test; its seven inputs are given directly so a test
    can fail exactly one of them and leave the other six passing.
    """
    anchor_static = torch.zeros(1, 1, 1, dtype=torch.bool)
    anchor_dynamic = torch.zeros(1, 1, 1, dtype=torch.bool)
    anchor_wall = torch.zeros(1, 1, 1, dtype=torch.bool)
    test_static = torch.tensor([[[static_collision]]])
    test_dynamic = torch.tensor([[[dynamic_collision]]])
    test_wall = torch.tensor([[[wall_collision]]])

    static = torch.cat([anchor_static, test_static], dim=2)
    dynamic = torch.cat([anchor_dynamic, test_dynamic], dim=2)
    wall = torch.cat([anchor_wall, test_wall], dim=2)
    obstacle = static | dynamic
    joint = ~obstacle & ~wall

    linear = torch.tensor([[[0.5, linear_mps]]])
    angular = torch.zeros(1, 1, 2)
    endpoint = torch.zeros(1, 1, 2, 2)
    endpoint[..., 0, 0] = 0.5
    endpoint[..., 0, 1] = anchor_lateral_m
    endpoint[..., 1, 0] = forward_progress_m
    endpoint[..., 1, 1] = lateral_offset_m
    endpoint_yaw = torch.tensor([[[0.0, heading_deviation_rad]]])
    raw_cost = torch.zeros(1, 1, 2)

    feasible = joint & (linear >= -1e-4)
    any_feasible = feasible.flatten(1).any(dim=1)
    return {
        "actions": torch.tensor([[0, 0]]),
        "any_feasible": any_feasible,
        "joint_feasible_grid": joint,
        "feasible_grid": feasible,
        "linear_velocity_grid": linear,
        "angular_velocity_grid": angular,
        "endpoint_grid": endpoint,
        "endpoint_yaw_grid": endpoint_yaw,
        "raw_cost_grid": raw_cost,
        "obstacle_collision_grid": obstacle,
        "wall_collision_grid": wall,
        "static_obstacle_collision_grid": static,
        "dynamic_obstacle_collision_grid": dynamic,
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


def test_select_reports_left_and_right_side_feasibility():
    """The FSM must expose which sides still have a passage candidate.

    Without these flags a diagnostic cannot tell "the committed side lost its
    path" apart from "both sides are blocked" -- two failures with opposite
    fixes.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)

    both = _select(controller, _teacher_result(), reset=True)
    assert both["left_valid"].tolist() == [True]
    assert both["right_valid"].tolist() == [True]
    assert both["left_valid"].shape == (1,)
    assert both["left_valid"].dtype == torch.bool

    # Grid layout: only [2, 2] satisfies the left mask and only [2, 0]
    # satisfies the right mask, so killing [2, 0] must kill right_valid alone.
    joint = torch.ones(1, 3, 3, dtype=torch.bool)
    joint[0, 2, 0] = False
    left_only = _select(controller, _teacher_result(joint=joint))
    assert left_only["left_valid"].tolist() == [True]
    assert left_only["right_valid"].tolist() == [False]


def test_committed_valid_exposes_no_switch_deadlock():
    """Commit left, then remove the left candidate while right stays open.

    This is the exact signature the r3 screen could not measure: the committed
    side is infeasible, the opposite side is feasible, and the controller stops
    instead of switching.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    result = _teacher_result()

    _select(controller, result, reset=True, dynamic_y=0.2, dynamic_vy=-1.0)
    _select(controller, result, dynamic_y=-0.2, dynamic_vy=-1.0)
    committed = _select(controller, result, dynamic_y=-0.3, dynamic_vy=-1.0)
    assert committed["state"].tolist() == [COMMIT_SIDE]
    assert committed["committed_side"].tolist() == [1]
    assert committed["committed_valid"].tolist() == [True]

    joint = torch.ones(1, 3, 3, dtype=torch.bool)
    joint[0, 2, 2] = False
    stuck = _select(
        controller,
        _teacher_result(joint=joint),
        dynamic_y=-0.4,
        dynamic_vy=-1.0,
    )

    assert stuck["committed_side"].tolist() == [1]
    assert stuck["committed_valid"].tolist() == [False]
    assert stuck["right_valid"].tolist() == [True]
    assert stuck["used_wait"].tolist() == [True]


def test_candidate_funnel_counts_attribute_loss_to_the_static_obstacle_mask():
    """The static-obstacle mask alone removes the last left passage candidate.

    The leave-one-out counts must name the static-obstacle mask as the
    binding constraint, so a diagnostic can tell a static-clearance problem
    apart from a dynamic-pedestrian, wall, or kinematic-filter problem.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    obstacle = torch.zeros(1, 3, 3, dtype=torch.bool)
    obstacle[0, 2, 2] = True          # the only left passage candidate

    out = _select(
        controller,
        _teacher_result(obstacle_collision=obstacle),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [1]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_final_right"].tolist() == [1]        # right side untouched


def test_candidate_funnel_counts_attribute_loss_to_the_heading_filter():
    """Heading rejection, not geometry, removes the last left candidate.

    ``n_without_heading_left`` isolates the heading filter alone -- unlike the
    old kinematic-bundle leave-one-out, it keeps launch/progress/side-signal
    active, so it must not credit cells that heading was never blocking.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    angular_values = torch.tensor([-0.5, 0.0, 0.5])
    endpoint_yaw = angular_values[None, :].expand(3, 3).unsqueeze(0).clone()
    endpoint_yaw[0, 2, 2] = 2.0       # beyond max_passage_heading_deviation

    out = _select(
        controller,
        _teacher_result(endpoint_yaw=endpoint_yaw),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    # Only [2, 2] clears launch+progress+side-signal on the left; relaxing
    # heading alone recovers exactly that one cell, not the whole row.
    assert out["n_without_heading_left"].tolist() == [1]


def test_funnel_reachable_and_threshold_fields_match_the_scene():
    """The adaptive thresholds/reachable stats must reflect the actual scene.

    This is what lets an analyst tell "the action grid cannot physically get
    there" apart from "a fixed threshold cut it off".
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)

    out = _select(controller, _funnel_teacher_result(), reset=True)

    torch.testing.assert_close(out["reachable_linear_mps"], torch.tensor([0.5]))
    torch.testing.assert_close(out["reachable_progress_m"], torch.tensor([0.5]))
    torch.testing.assert_close(out["reachable_lateral_m"], torch.tensor([0.5]))
    torch.testing.assert_close(
        out["launch_threshold_mps"], torch.tensor([0.15])
    )
    torch.testing.assert_close(
        out["progress_threshold_m"], torch.tensor([0.20])
    )
    torch.testing.assert_close(out["side_threshold_m"], torch.tensor([0.15]))


def test_funnel_static_obstacle_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(static_collision=True),
        reset=True,
    )

    assert out["n_raw_left"].tolist() == [1]
    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [1]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [0]
    assert out["n_final_right"].tolist() == [1]


def test_funnel_dynamic_obstacle_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(dynamic_collision=True),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [1]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [0]


def test_funnel_wall_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(wall_collision=True),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [1]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [0]


def test_funnel_launch_speed_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(linear_mps=-0.05),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [1]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [0]


def test_funnel_forward_progress_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(forward_progress_m=-0.05),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [1]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [0]


def test_funnel_heading_is_the_unique_binder():
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(heading_deviation_rad=2.0),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [1]
    # The anchor cell keeps reachable stats healthy, so the heading failure
    # at the test cell must not zero out the env-level side-signal gate.
    assert out["n_without_side_signal_left"].tolist() == [0]


def test_funnel_side_signal_is_the_unique_binder():
    """Side-signal is env-level: the anchor must also have a tiny offset.

    Otherwise the anchor's own lateral offset would keep the scene's
    reachable-lateral statistic healthy and this test would never be able to
    make ``side_signal`` fail on its own.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(
            lateral_offset_m=0.005, anchor_lateral_m=-0.005
        ),
        reset=True,
    )

    assert out["n_raw_left"].tolist() == [1]
    assert out["n_final_left"].tolist() == [0]
    assert out["n_without_static_obstacle_left"].tolist() == [0]
    assert out["n_without_dynamic_obstacle_left"].tolist() == [0]
    assert out["n_without_wall_left"].tolist() == [0]
    assert out["n_without_launch_left"].tolist() == [0]
    assert out["n_without_progress_left"].tolist() == [0]
    assert out["n_without_heading_left"].tolist() == [0]
    assert out["n_without_side_signal_left"].tolist() == [1]


def test_funnel_two_simultaneous_binders_are_not_misattributed_to_one():
    """Static collision AND a wall both block the same cell.

    No single leave-one-out may claim to be "the" cause; only relaxing both
    together should recover the candidate. This is the ambiguous/multi-filter
    case the analyzer must not collapse into a unique binder.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(static_collision=True, wall_collision=True),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    for name in (
        "static_obstacle",
        "dynamic_obstacle",
        "wall",
        "launch",
        "progress",
        "heading",
        "side_signal",
    ):
        assert out[f"n_without_{name}_left"].tolist() == [0], name

    expected_bitmask = (
        FUNNEL_MASK_BITS["static_obstacle"] | FUNNEL_MASK_BITS["wall"]
    )
    assert out["pairwise_recovering_count_left"].tolist() == [1]
    assert out["pairwise_best_count_left"].tolist() == [1]
    assert out["pairwise_best_bitmask_left"].tolist() == [expected_bitmask]


def test_funnel_three_simultaneous_binders_exceed_pairwise_relaxation():
    """Three independent binders on one cell: even the best pair can't save it.

    This is the "multi-filter/grid-limited" bucket beyond pairwise relaxation.
    """
    controller = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = _select(
        controller,
        _funnel_teacher_result(
            static_collision=True,
            wall_collision=True,
            heading_deviation_rad=2.0,
        ),
        reset=True,
    )

    assert out["n_final_left"].tolist() == [0]
    assert out["pairwise_recovering_count_left"].tolist() == [0]
    assert out["pairwise_best_count_left"].tolist() == [0]
    assert out["pairwise_best_bitmask_left"].tolist() == [0]


def test_onset_trigger_commits_before_the_pedestrian_crosses():
    """A pedestrian walking away never reaches the robot's lateral line here.

    The crossing trigger therefore never registers a vacated side, and because
    a crossing threat is active the FSM proposes nothing and stays in WAIT.
    The onset trigger reads the sustained lateral direction instead, so it
    commits to the side the pedestrian is leaving without waiting for it to
    walk all the way across.
    """
    walk = [(0.5, -1.0), (0.4, -1.0), (0.3, -1.0)]

    crossing = StatefulCorridorTeacher(1, "cpu", dt_s=0.2)
    out = None
    for i, (y, vy) in enumerate(walk):
        out = _select(crossing, _teacher_result(), reset=(i == 0),
                      dynamic_y=y, dynamic_vy=vy)
    assert out["state"].tolist() == [WAIT]
    assert out["committed_side"].tolist() == [0]

    onset = StatefulCorridorTeacher(
        1, "cpu", dt_s=0.2,
        spec=StatefulTeacherSpec(vacated_trigger="onset", vacated_onset_steps=2),
    )
    out = None
    for i, (y, vy) in enumerate(walk):
        out = _select(onset, _teacher_result(), reset=(i == 0),
                      dynamic_y=y, dynamic_vy=vy)
    assert out["state"].tolist() == [COMMIT_SIDE]
    assert out["committed_side"].tolist() == [1]


def test_onset_trigger_picks_the_side_being_vacated_not_the_destination():
    """Mirror case: a pedestrian on the right walking left vacates the right."""
    onset = StatefulCorridorTeacher(
        1, "cpu", dt_s=0.2,
        spec=StatefulTeacherSpec(vacated_trigger="onset", vacated_onset_steps=2),
    )
    out = None
    for i, y in enumerate((-0.5, -0.4, -0.3)):
        out = _select(onset, _teacher_result(), reset=(i == 0),
                      dynamic_y=y, dynamic_vy=1.0)
    assert out["state"].tolist() == [COMMIT_SIDE]
    assert out["committed_side"].tolist() == [-1]


def test_unknown_vacated_trigger_is_rejected():
    try:
        StatefulCorridorTeacher(
            1, "cpu", dt_s=0.2,
            spec=StatefulTeacherSpec(vacated_trigger="whenever"),
        )
    except ValueError:
        return
    raise AssertionError("unknown vacated trigger must raise")
