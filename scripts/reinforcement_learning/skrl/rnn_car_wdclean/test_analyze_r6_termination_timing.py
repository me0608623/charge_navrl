import numpy as np

from rnn_car_wdclean.analyze_r6_termination_timing import (
    build_report,
    dynamic_obstacle_recovers_a_blocked_side,
    episode_windows,
    launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side,
    static_and_progress_both_recover_a_blocked_side,
)


def test_episode_windows_selects_only_the_requested_cause():
    episode_end = np.array(
        [
            [10, 0, 3, 11],  # obstacle, length 11
            [20, 1, 4, 8],  # timeout, length 8
        ]
    )
    windows = episode_windows(episode_end, "obstacle", window_steps=25)
    assert windows == [(10, 0, 11)]


def test_episode_windows_clips_at_episode_start_not_the_fixed_window():
    episode_end = np.array([[10, 0, 3, 3]])  # only 3 steps long
    windows = episode_windows(episode_end, "obstacle", window_steps=25)
    assert windows == [(10, 0, 3)]


def _arrays(n_final_left, n_without_dynamic_left, n_without_static_left,
            n_without_progress_left):
    shape = n_final_left.shape
    zeros = np.zeros(shape, dtype=np.int64)
    return {
        "n_final_left": n_final_left,
        "n_final_right": np.ones(shape, dtype=np.int64),  # right never blocked
        "n_without_dynamic_obstacle_left": n_without_dynamic_left,
        "n_without_dynamic_obstacle_right": zeros,
        "n_without_static_obstacle_left": n_without_static_left,
        "n_without_static_obstacle_right": zeros,
        "n_without_progress_left": n_without_progress_left,
        "n_without_progress_right": zeros,
    }


def test_dynamic_obstacle_metric_requires_both_blocked_and_recovering():
    arrays = _arrays(
        n_final_left=np.array([[0, 0, 3]]),
        n_without_dynamic_left=np.array([[1, 0, 1]]),
        n_without_static_left=np.zeros((1, 3), dtype=np.int64),
        n_without_progress_left=np.zeros((1, 3), dtype=np.int64),
    )
    # col 0: blocked and dynamic recovers -> True
    assert dynamic_obstacle_recovers_a_blocked_side(arrays, 0, 0) is True
    # col 1: blocked but dynamic does NOT recover -> False
    assert dynamic_obstacle_recovers_a_blocked_side(arrays, 0, 1) is False
    # col 2: not blocked (has candidates) -> False regardless of recovery
    assert dynamic_obstacle_recovers_a_blocked_side(arrays, 0, 2) is False


def test_static_progress_metric_requires_both_gates_to_recover_together():
    arrays = _arrays(
        n_final_left=np.array([[0, 0, 0]]),
        n_without_dynamic_left=np.zeros((1, 3), dtype=np.int64),
        n_without_static_left=np.array([[1, 1, 0]]),
        n_without_progress_left=np.array([[1, 0, 1]]),
    )
    # col 0: both static and progress recover -> True
    assert static_and_progress_both_recover_a_blocked_side(arrays, 0, 0) is True
    # col 1: only static recovers -> False (this is the unique-binder case,
    # not the ambiguous pair)
    assert static_and_progress_both_recover_a_blocked_side(arrays, 0, 1) is False
    # col 2: only progress recovers -> False
    assert static_and_progress_both_recover_a_blocked_side(arrays, 0, 2) is False


def _full_gate_arrays(*, n_without_overrides, bitmask, best_count, n_final=0):
    """One env/frame with all 7 singles at 0 (true multi_filter) unless
    overridden, plus the pairwise best-pair fields."""
    names = (
        "static_obstacle",
        "dynamic_obstacle",
        "wall",
        "launch",
        "progress",
        "heading",
        "side_signal",
    )
    arrays = {
        "n_final_left": np.array([[n_final]]),
        "n_final_right": np.array([[1]]),  # right side never blocked
        "pairwise_best_bitmask_right": np.array([[0]]),
        "pairwise_best_count_right": np.array([[0]]),
    }
    for name in names:
        value = n_without_overrides.get(name, 0)
        arrays[f"n_without_{name}_left"] = np.array([[value]])
        arrays[f"n_without_{name}_right"] = np.array([[0]])
    arrays["pairwise_best_bitmask_left"] = np.array([[bitmask]])
    arrays["pairwise_best_count_left"] = np.array([[best_count]])
    return arrays


def test_launch_progress_multi_filter_metric_requires_true_multi_filter():
    """Must not fire when a single gate already recovers the cell.

    Otherwise a unique-binder frame with a coincidentally matching best-pair
    bitmask would be double-counted as both unique and multi_filter.
    """
    launch_progress_bitmask = (1 << 3) | (1 << 4)  # launch, progress
    fires = launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side(
        _full_gate_arrays(
            n_without_overrides={"static_obstacle": 1},  # a single already recovers
            bitmask=launch_progress_bitmask,
            best_count=5,
        ),
        0,
        0,
    )
    assert fires is False


def test_launch_progress_multi_filter_metric_requires_matching_bitmask():
    launch_progress_bitmask = (1 << 3) | (1 << 4)
    wall_progress_bitmask = (1 << 2) | (1 << 4)
    fires = launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side(
        _full_gate_arrays(
            n_without_overrides={},
            bitmask=wall_progress_bitmask,
            best_count=3,
        ),
        0,
        0,
    )
    assert fires is False

    fires = launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side(
        _full_gate_arrays(
            n_without_overrides={}, bitmask=launch_progress_bitmask, best_count=3
        ),
        0,
        0,
    )
    assert fires is True


def test_build_report_aggregates_fraction_correctly_across_episodes():
    # Two obstacle-terminal episodes, env 0 and env 1, each 2 steps long.
    # env0: dynamic recovers at both offsets. env1: never.
    n_final_left = np.array([[0, 0], [0, 0]])  # [row, env]
    n_without_dynamic_left = np.array([[1, 0], [1, 0]])
    arrays = {
        "episode_end": np.array([[1, 0, 3, 2], [1, 1, 3, 2]]),
        "n_final_left": n_final_left,
        "n_final_right": np.ones_like(n_final_left),
        "n_without_dynamic_obstacle_left": n_without_dynamic_left,
        "n_without_dynamic_obstacle_right": np.zeros_like(n_final_left),
        "n_without_static_obstacle_left": np.zeros_like(n_final_left),
        "n_without_static_obstacle_right": np.zeros_like(n_final_left),
        "n_without_progress_left": np.zeros_like(n_final_left),
        "n_without_progress_right": np.zeros_like(n_final_left),
        "n_without_wall_left": np.zeros_like(n_final_left),
        "n_without_wall_right": np.zeros_like(n_final_left),
        "n_without_launch_left": np.zeros_like(n_final_left),
        "n_without_launch_right": np.zeros_like(n_final_left),
        "n_without_heading_left": np.zeros_like(n_final_left),
        "n_without_heading_right": np.zeros_like(n_final_left),
        "n_without_side_signal_left": np.zeros_like(n_final_left),
        "n_without_side_signal_right": np.zeros_like(n_final_left),
        "pairwise_best_bitmask_left": np.zeros_like(n_final_left),
        "pairwise_best_bitmask_right": np.zeros_like(n_final_left),
        "pairwise_best_count_left": np.zeros_like(n_final_left),
        "pairwise_best_count_right": np.zeros_like(n_final_left),
    }

    report = build_report({"cell": arrays}, window_steps=2, dt_s=0.2)

    obstacle = report["causes"]["obstacle"]
    assert obstacle["n_episodes"] == 2
    curve = obstacle["curves"]["dynamic_obstacle_recovers"]
    terminal_point = next(p for p in curve if p["seconds_before_end"] == 0.0)
    # terminal row=1: env0 recovers, env1 doesn't -> 1/2
    assert terminal_point["n_total"] == 2
    assert terminal_point["fraction"] == 0.5
    earlier_point = next(p for p in curve if p["seconds_before_end"] == 0.2)
    # row=0: env0 recovers, env1 doesn't -> 1/2
    assert earlier_point["n_total"] == 2
    assert earlier_point["fraction"] == 0.5
