import numpy as np

from scene_solvability import SolvabilitySpec, analyze_scene, summarize_records


SPEC = SolvabilitySpec(half_extent=4.0, resolution=0.1)
NO_OBS = np.empty((0, 2), dtype=np.float32)
NO_RADII = np.empty((0,), dtype=np.float32)
NO_MASK = np.empty((0,), dtype=bool)


def _scene(wall_centers, wall_sizes, start=(-3.0, 0.0), goal=(3.0, 0.0)):
    return analyze_scene(
        SPEC,
        np.asarray(start, dtype=np.float32),
        np.asarray(goal, dtype=np.float32),
        np.asarray(wall_centers, dtype=np.float32),
        np.asarray(wall_sizes, dtype=np.float32),
        np.ones(len(wall_centers), dtype=bool),
        NO_OBS,
        NO_RADII,
        NO_MASK,
        NO_MASK,
    )


def test_wall_with_gap_is_reachable():
    result = _scene([[0.0, 2.5], [0.0, -2.5]], [[0.5, 3.0], [0.5, 3.0]])
    assert result["walls"]["reachable"]


def test_full_cross_wall_is_unreachable():
    result = _scene([[0.0, 0.0]], [[0.5, 8.0]])
    assert not result["walls"]["reachable"]
    assert result["walls"]["start_free"]
    assert result["walls"]["goal_free"]


def test_goal_inside_inflated_obstacle_is_not_free():
    result = analyze_scene(
        SPEC,
        np.asarray([-3.0, 0.0]),
        np.asarray([2.0, 0.0]),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=bool),
        np.asarray([[2.6, 0.0]], dtype=np.float32),
        np.asarray([0.3], dtype=np.float32),
        np.asarray([True]),
        np.asarray([True]),
    )
    assert result["walls"]["goal_free"]
    assert not result["static"]["goal_free"]
    assert not result["static"]["reachable"]
    assert result["static"]["goal_clearance"] < SPEC.footprint_radius


def test_exact_endpoint_clearance_avoids_grid_quantization_false_positive():
    result = _scene(
        [[0.0, 0.0]],
        [[8.0, 0.5]],
        start=(-3.0, 0.71),
        goal=(3.0, 0.71),
    )
    assert result["walls"]["start_clearance"] > SPEC.footprint_radius
    assert result["walls"]["start_free"]


def test_dynamic_obstacle_only_affects_instantaneous_tier():
    result = analyze_scene(
        SPEC,
        np.asarray([-3.0, 0.0]),
        np.asarray([3.0, 0.0]),
        np.asarray([[0.0, 2.5], [0.0, -2.5]], dtype=np.float32),
        np.asarray([[0.5, 3.0], [0.5, 3.0]], dtype=np.float32),
        np.asarray([True, True]),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([1.1], dtype=np.float32),
        np.asarray([False]),
        np.asarray([True]),
    )
    assert result["static"]["reachable"]
    assert not result["all_initial"]["reachable"]


def test_summary_conditions_causes_on_reachability():
    records = [
        {"cause": 1, **{t: {"reachable": True, "free_fraction": 0.8} for t in ("walls", "static", "all_initial")}},
        {"cause": 3, **{t: {"reachable": False, "free_fraction": 0.4} for t in ("walls", "static", "all_initial")}},
    ]
    summary = summarize_records(records)
    static = summary["static"]
    assert static["unreachable"]["episodes"] == 1
    assert static["unreachable"]["cause_fraction"]["obstacle_collision"] == 1.0
    assert static["reachable"]["cause_fraction"]["goal"] == 1.0
