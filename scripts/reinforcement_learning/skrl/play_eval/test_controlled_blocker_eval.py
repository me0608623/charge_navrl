import unittest

import torch

from controlled_blocker_eval import ControlledBlockerSpec, mirrored_geometry


class ControlledBlockerGeometryTest(unittest.TestCase):
    def test_exact_four_metre_corridor_and_clearance(self):
        spec = ControlledBlockerSpec()
        env_ids = torch.arange(2)
        _, centers, sizes = mirrored_geometry(env_ids, spec)
        upper_inner_face = centers[:, 0, 1] - sizes[:, 0, 1] / 2
        lower_inner_face = centers[:, 1, 1] + sizes[:, 1, 1] / 2
        torch.testing.assert_close(upper_inner_face - lower_inner_face, torch.full((2,), 4.0))
        self.assertAlmostEqual(spec.side_clearance, 1.65)

    def test_mirror_balance_and_open_side_clearance(self):
        spec = ControlledBlockerSpec()
        env_ids = torch.arange(64)
        closed_sign, centers, _ = mirrored_geometry(env_ids, spec)
        self.assertEqual(int((closed_sign > 0).sum()), 32)
        self.assertEqual(int((closed_sign < 0).sum()), 32)
        torch.testing.assert_close(centers[0, 2, 1], -centers[1, 2, 1])
        self.assertGreater(spec.side_clearance, spec.robot_radius + 0.10)

    def test_pressure_wall_closes_only_mirrored_side(self):
        spec = ControlledBlockerSpec()
        closed_sign, centers, sizes = mirrored_geometry(torch.arange(2), spec)
        pressure_inner_abs = centers[:, 2, 1].abs() - sizes[:, 2, 1] / 2
        overlap = spec.blocker_radius - pressure_inner_abs
        torch.testing.assert_close(overlap, torch.full((2,), 0.05))
        self.assertEqual(closed_sign.tolist(), [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
