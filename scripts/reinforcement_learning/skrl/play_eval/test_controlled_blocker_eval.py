import unittest
from types import SimpleNamespace

import torch

from controlled_blocker_eval import (
    ControlledBlockerController,
    ControlledBlockerSpec,
    corridor_geometry,
    dynamic_env_mask,
)


class ControlledBlockerGeometryTest(unittest.TestCase):
    def test_exact_four_metre_corridor(self):
        spec = ControlledBlockerSpec()
        centers, sizes = corridor_geometry(2, spec, "cpu")
        upper_inner_face = centers[:, 0, 1] - sizes[:, 0, 1] / 2
        lower_inner_face = centers[:, 1, 1] + sizes[:, 1, 1] / 2
        torch.testing.assert_close(upper_inner_face - lower_inner_face, torch.full((2,), 4.0))

    def test_walls_cover_start_blocker_region_and_goal(self):
        spec = ControlledBlockerSpec()
        centers, sizes = corridor_geometry(1, spec, "cpu")
        wall_min_x = centers[0, 0, 0] - sizes[0, 0, 0] / 2
        wall_max_x = centers[0, 0, 0] + sizes[0, 0, 0] / 2
        self.assertAlmostEqual(float(wall_min_x), -3.0)
        self.assertAlmostEqual(float(wall_max_x), 9.0)

    def test_worst_case_clearance_is_solvable(self):
        spec = ControlledBlockerSpec(
            blocker_x_min=1.2, blocker_x_max=4.8, blocker_y_max=1.0
        )
        self.assertAlmostEqual(spec.side_clearance, 0.65)
        self.assertGreater(spec.side_clearance, spec.robot_radius + 0.10)
        self.assertAlmostEqual(spec.endpoint_clearance, 0.5)

    def test_corridor_is_symmetric(self):
        spec = ControlledBlockerSpec()
        centers, sizes = corridor_geometry(1, spec, "cpu")
        torch.testing.assert_close(centers[0, 0], centers[0, 1] * torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(sizes[0, 0], sizes[0, 1])

    def test_default_dynamic_ratio(self):
        spec = ControlledBlockerSpec()
        self.assertEqual(spec.dynamic_ratio, 0.5)
        self.assertEqual(round(spec.dynamic_ratio * 64), 32)

    def test_bootstrap_goal_is_nontrivial_and_inside_stage7_sampling_boundary(self):
        spec = ControlledBlockerSpec()
        self.assertGreater(spec.sampler_bootstrap_distance_min, 0.7)
        self.assertGreater(spec.sampler_bootstrap_distance_max, spec.sampler_bootstrap_distance_min)
        self.assertLess(spec.sampler_bootstrap_distance_max, spec.goal_x)

    def test_dynamic_allocation_is_exact_and_stable_across_reset_subsets(self):
        env_ids = torch.arange(64)
        mask = dynamic_env_mask(env_ids, 64, 0.5)
        self.assertEqual(int(mask.sum()), 32)
        subset = torch.tensor([2, 31, 32, 63])
        self.assertEqual(dynamic_env_mask(subset, 64, 0.5).tolist(), [True, True, False, False])

    def test_random_2d_spawn_and_velocity_are_bounded(self):
        torch.manual_seed(7)
        controller = ControlledBlockerController.__new__(ControlledBlockerController)
        controller.spec = ControlledBlockerSpec(dynamic_ratio=0.5, blocker_speed=0.3)
        controller.device = "cpu"
        controller.num_envs = 4
        controller.blocker_x = torch.zeros(4)
        controller.blocker_y = torch.zeros(4)
        controller.blocker_vx = torch.zeros(4)
        controller.blocker_vy = torch.zeros(4)
        writes = []
        controller._write_blocker = lambda env_ids: writes.append(env_ids.clone())

        controller._place_blocker(torch.arange(4))

        self.assertTrue(bool((controller.blocker_x >= 1.2).all()))
        self.assertTrue(bool((controller.blocker_x <= 4.8).all()))
        self.assertTrue(bool((controller.blocker_y.abs() <= 1.0).all()))
        speed = torch.hypot(controller.blocker_vx, controller.blocker_vy)
        torch.testing.assert_close(speed[:2], torch.full((2,), 0.3))
        torch.testing.assert_close(speed[2:], torch.zeros(2))
        self.assertTrue(bool((controller.blocker_vx[:2].abs() > 1e-4).all()))
        self.assertEqual(writes[0].tolist(), [0, 1, 2, 3])

    def test_2d_patrol_advances_and_reflects_at_both_bounds(self):
        controller = ControlledBlockerController.__new__(ControlledBlockerController)
        controller.spec = ControlledBlockerSpec(
            blocker_x_min=1.2, blocker_x_max=4.8, blocker_y_max=1.0
        )
        controller.env = SimpleNamespace(step_dt=0.2)
        controller.device = "cpu"
        controller.num_envs = 2
        controller.blocker_x = torch.tensor([4.78, 2.0])
        controller.blocker_y = torch.tensor([0.0, 0.98])
        controller.blocker_vx = torch.tensor([0.30, 0.0])
        controller.blocker_vy = torch.tensor([0.0, 0.30])
        controller._motion_logged = True
        writes = []
        controller._write_blocker = lambda env_ids: writes.append(env_ids.clone())

        controller.advance()

        torch.testing.assert_close(controller.blocker_x, torch.tensor([4.8, 2.0]))
        torch.testing.assert_close(controller.blocker_y, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(controller.blocker_vx, torch.tensor([-0.30, 0.0]))
        torch.testing.assert_close(controller.blocker_vy, torch.tensor([0.0, -0.30]))
        self.assertEqual(writes[0].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
