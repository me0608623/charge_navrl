"""Geometry-aware direct-crossing metrics for randomised narrow replay.

The Gate5 evaluator (`narrow_gap_eval.NarrowGapController`) hard-codes the
fixed scene: barrier at x=0, gap on the axis, west->east only. It measures
pre-cross lateral excursion as |y|, which is meaningless once the production
replay puts the gap at y in [-1,+1], and its crossing test assumes
increasing x.

These tests pin the generalised accounting used to verify the scripted
teacher on the real randomised replay (N1 step 2): lateral excursion is
measured against each env's own gap center, and crossing/backtrack are
measured along each env's own travel direction.
"""

import math
import unittest

import torch

from narrow_replay_metrics import (
    NarrowReplayMetrics,
    NarrowReplayThresholds,
)


def _metrics(num_envs=2, **kwargs):
    return NarrowReplayMetrics(
        num_envs=num_envs,
        device="cpu",
        control_dt_s=0.2,
        thresholds=NarrowReplayThresholds(**kwargs),
    )


def _begin(m, env_ids, *, barrier, gap, goal, active):
    m.begin_episodes(
        torch.tensor(env_ids),
        barrier_x_m=torch.tensor(barrier, dtype=torch.float32),
        gap_center_y_m=torch.tensor(gap, dtype=torch.float32),
        goal_xy_m=torch.tensor(goal, dtype=torch.float32),
        active=torch.tensor(active),
    )


def _straight_run(m, env_id, *, barrier_x, gap_y, direction, start_x, steps=8):
    """Drive straight down the gap axis at 1 m/s (0.2 m per control step)."""
    active = [False] * m.num_envs
    active[env_id] = True
    barrier = [0.0] * m.num_envs
    barrier[env_id] = barrier_x
    gap = [0.0] * m.num_envs
    gap[env_id] = gap_y
    goal = [[0.0, 0.0] for _ in range(m.num_envs)]
    goal[env_id] = [barrier_x + direction * 3.0, gap_y]
    _begin(m, [env_id], barrier=barrier, gap=gap, goal=goal, active=active)
    xy = torch.zeros(m.num_envs, 2)
    for step in range(steps):
        xy[env_id, 0] = start_x + direction * 0.2 * (step + 1)
        xy[env_id, 1] = gap_y
        m.observe(xy)


class DirectionAgnosticCrossingTest(unittest.TestCase):
    def test_straight_west_to_east_counts_as_direct(self):
        m = _metrics()
        _straight_run(m, 0, barrier_x=0.0, gap_y=0.0, direction=1.0, start_x=-1.0)
        m.finish_episodes(torch.tensor([0]))
        s = m.summary()
        self.assertEqual(s["episodes"], 1)
        self.assertEqual(s["crossed"], 1)
        self.assertEqual(s["direct_crossed"], 1)

    def test_straight_east_to_west_counts_as_direct(self):
        """Mirrored direction must not be scored as backtracking."""
        m = _metrics()
        _straight_run(m, 0, barrier_x=0.0, gap_y=0.0, direction=-1.0, start_x=1.0)
        m.finish_episodes(torch.tensor([0]))
        s = m.summary()
        self.assertEqual(s["crossed"], 1)
        self.assertEqual(s["direct_crossed"], 1)
        self.assertEqual(s["backtrack_distance_m_p50"], 0.0)

    def test_backtrack_is_measured_along_travel_direction(self):
        for direction, retreat in ((1.0, -0.6), (-1.0, 0.6)):
            with self.subTest(direction=direction):
                m = _metrics()
                goal = [[direction * 3.0, 0.0], [0.0, 0.0]]
                _begin(
                    m, [0], barrier=[0.0, 0.0], gap=[0.0, 0.0], goal=goal,
                    active=[True, False],
                )
                xy = torch.zeros(2, 2)
                start = -direction * 1.0
                xy[0, 0] = start
                m.observe(xy)
                xy[0, 0] = start + retreat        # away from the barrier
                m.observe(xy)
                for step in range(12):
                    xy[0, 0] = start + retreat + direction * 0.3 * (step + 1)
                    m.observe(xy)
                m.finish_episodes(torch.tensor([0]))
                s = m.summary()
                self.assertEqual(s["crossed"], 1)
                self.assertGreaterEqual(s["backtrack_distance_m_p50"], 0.55)
                self.assertEqual(s["direct_crossed"], 0)


class OffsetGapLateralTest(unittest.TestCase):
    def test_offset_gap_straight_run_is_direct_despite_large_abs_y(self):
        """Gap at y=+0.9: driving along it is not a 0.9 m excursion."""
        m = _metrics()
        _straight_run(m, 0, barrier_x=0.3, gap_y=0.9, direction=1.0, start_x=-0.7)
        m.finish_episodes(torch.tensor([0]))
        s = m.summary()
        self.assertEqual(s["direct_crossed"], 1)
        self.assertLess(s["max_pre_cross_lateral_m_p50"], 1e-6)

    def test_detour_relative_to_offset_gap_is_not_direct(self):
        m = _metrics()
        _begin(
            m, [0], barrier=[0.0, 0.0], gap=[0.9, 0.0],
            goal=[[3.0, 0.9], [0.0, 0.0]], active=[True, False],
        )
        xy = torch.zeros(2, 2)
        for y in (1.5, 2.5, 3.9, 2.0, 0.9):   # detour up, then back to the gap
            xy[0] = torch.tensor([-1.0, y])
            m.observe(xy)
        for x in (-0.5, 0.5, 1.5):            # then cross
            xy[0] = torch.tensor([x, 0.9])
            m.observe(xy)
        m.finish_episodes(torch.tensor([0]))
        s = m.summary()
        self.assertEqual(s["crossed"], 1)
        self.assertEqual(s["direct_crossed"], 0)
        self.assertGreater(s["max_pre_cross_lateral_m_p50"], 2.9)


class BatchAccountingTest(unittest.TestCase):
    def test_inactive_envs_are_ignored(self):
        m = _metrics(num_envs=3)
        _begin(
            m, [0, 1, 2], barrier=[0.0] * 3, gap=[0.0] * 3,
            goal=[[3.0, 0.0]] * 3, active=[True, False, False],
        )
        xy = torch.zeros(3, 2)
        for step in range(10):
            xy[:, 0] = -1.0 + 0.3 * (step + 1)
            m.observe(xy)
        m.finish_episodes(torch.tensor([0, 1, 2]))
        self.assertEqual(m.summary()["episodes"], 1)

    def test_mixed_batch_directions_scored_independently(self):
        m = _metrics(num_envs=2)
        _begin(
            m, [0, 1], barrier=[0.2, -0.4], gap=[0.5, -0.7],
            goal=[[3.2, 0.5], [-3.4, -0.7]], active=[True, True],
        )
        xy = torch.zeros(2, 2)
        xy[0] = torch.tensor([-0.8, 0.5])
        xy[1] = torch.tensor([0.6, -0.7])
        m.observe(xy)
        for step in range(10):
            xy[0] = torch.tensor([-0.8 + 0.3 * (step + 1), 0.5])
            xy[1] = torch.tensor([0.6 - 0.3 * (step + 1), -0.7])
            m.observe(xy)
        m.finish_episodes(torch.tensor([0, 1]))
        s = m.summary()
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["crossed"], 2)
        self.assertEqual(s["direct_crossed"], 2)


class DoneMaskTest(unittest.TestCase):
    def test_done_envs_are_skipped_for_that_step(self):
        """The env auto-resets inside step(), so the post-step pose of a done
        env is already the next episode's spawn — folding it in would charge a
        teleport to the finished episode's path length."""
        m = _metrics()
        _begin(
            m, [0], barrier=[0.0, 0.0], gap=[0.0, 0.0],
            goal=[[3.0, 0.0], [0.0, 0.0]], active=[True, False],
        )
        xy = torch.zeros(2, 2)
        for step in range(6):
            xy[0, 0] = -1.0 + 0.2 * (step + 1)
            m.observe(xy)
        before = float(m._path_length_m[0])
        teleported = xy.clone()
        teleported[0] = torch.tensor([-40.0, 25.0])   # next episode's spawn
        m.observe(teleported, done=torch.tensor([True, False]))
        self.assertAlmostEqual(float(m._path_length_m[0]), before, places=6)

    def test_done_envs_still_accumulate_when_not_masked(self):
        m = _metrics()
        _begin(
            m, [0], barrier=[0.0, 0.0], gap=[0.0, 0.0],
            goal=[[3.0, 0.0], [0.0, 0.0]], active=[True, False],
        )
        xy = torch.zeros(2, 2)
        xy[0, 0] = -1.0
        m.observe(xy)
        xy[0, 0] = -0.5
        m.observe(xy)
        self.assertGreater(float(m._path_length_m[0]), 0.4)


class ThresholdTest(unittest.TestCase):
    def test_slow_crossing_fails_the_time_threshold(self):
        """This run crosses at exactly 1.0 s; 0.5 s must reject it."""
        m = _metrics(first_cross_time_s_max=0.5)
        _straight_run(m, 0, barrier_x=0.0, gap_y=0.0, direction=1.0, start_x=-1.0)
        m.finish_episodes(torch.tensor([0]))
        s = m.summary()
        self.assertEqual(s["crossed"], 1)
        self.assertAlmostEqual(s["first_cross_time_s_p50"], 1.0, places=6)
        self.assertEqual(s["direct_crossed"], 0)

    def test_time_threshold_boundary_is_inclusive(self):
        m = _metrics(first_cross_time_s_max=1.0)
        _straight_run(m, 0, barrier_x=0.0, gap_y=0.0, direction=1.0, start_x=-1.0)
        m.finish_episodes(torch.tensor([0]))
        self.assertEqual(m.summary()["direct_crossed"], 1)

    def test_summary_of_empty_run_is_safe(self):
        s = _metrics().summary()
        self.assertEqual(s["episodes"], 0)
        self.assertTrue(math.isnan(s["crossing_rate"]))


if __name__ == "__main__":
    unittest.main()
