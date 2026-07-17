import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from near_wall_crossing_eval import NearWallCrossingController


class _RobotData:
    def __init__(self, n: int):
        self.root_pos_w = torch.zeros(n, 3)
        self.root_quat_w = torch.zeros(n, 4)
        self.root_quat_w[:, 0] = 1.0


class _Robot:
    def __init__(self, n: int):
        self.data = _RobotData(n)


class _Scene(dict):
    def __init__(self, n: int):
        super().__init__(robot=_Robot(n))
        self.env_origins = torch.zeros(n, 3)


class _Env:
    device = "cpu"

    def __init__(self, n: int):
        self.num_envs = n
        self.scene = _Scene(n)


class NearWallCrossingControllerTest(unittest.TestCase):
    def test_mirrored_direction_alternates_by_env_and_episode(self):
        controller = NearWallCrossingController(_Env(2), None, step_dt=0.2)
        env_ids = torch.arange(2)
        self.assertEqual(controller._direction_for(env_ids).tolist(), [1.0, -1.0])
        self.assertEqual((controller._case_index(env_ids) // 2).tolist(), [0, 0])
        controller.episode_index += 1
        self.assertEqual(controller._direction_for(env_ids).tolist(), [1.0, -1.0])
        self.assertEqual((controller._case_index(env_ids) // 2).tolist(), [1, 1])

    def test_same_sign_yaw_detects_full_rotation(self):
        controller = NearWallCrossingController(_Env(1), None, step_dt=0.2)
        controller.cross_sign[:] = 1.0
        for _ in range(27):
            controller.record_step(torch.tensor([0.0]), torch.tensor([1.2]))
        self.assertGreaterEqual(controller.max_same_sign_yaw.item(), 2.0 * torch.pi)
        self.assertEqual(controller.max_same_sign_run.item(), 27)
        self.assertEqual(controller.max_spin_run.item(), 27)

    def test_terminal_auto_reset_does_not_erase_lateral_drift(self):
        env = _Env(1)
        controller = NearWallCrossingController(env, None, step_dt=0.2)
        controller.cross_sign[:] = 1.0

        env.scene["robot"].data.root_pos_w[:, 1] = 1.25
        controller.record_step(torch.tensor([0.4]), torch.tensor([0.0]), done=torch.tensor([False]))
        self.assertAlmostEqual(controller.net_lateral_drift.item(), 1.25)

        # Isaac Lab has already reset the terminal robot to y=0 when metrics run.
        env.scene["robot"].data.root_pos_w[:, 1] = 0.0
        controller.record_step(torch.tensor([0.0]), torch.tensor([0.0]), done=torch.tensor([True]))
        self.assertAlmostEqual(controller.net_lateral_drift.item(), 1.25)

    def test_probe_factorial_direction_is_not_tied_to_wall_side(self):
        controller = NearWallCrossingController(
            _Env(8), None, step_dt=0.2, probe_output_path="/tmp/not_written.npz"
        )
        env_ids = torch.arange(8)
        crossing = controller._direction_for(env_ids)
        case_index = env_ids % 8
        relative_wall = torch.where((case_index // 4) % 2 == 0, 1.0, -1.0)
        wall = crossing * relative_wall
        combinations = set(zip(crossing.tolist(), wall.tolist()))
        self.assertEqual(combinations, {(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)})

    def test_probe_dump_preserves_exact_stack_and_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "probe.npz"
            controller = NearWallCrossingController(
                _Env(2), None, step_dt=0.2, probe_output_path=str(output)
            )
            controller.cross_sign[:] = torch.tensor([1.0, -1.0])
            controller.wall_sign[:] = torch.tensor([-1.0, 1.0])
            controller.episode_index[:] = 1
            stack = torch.arange(2 * 288, dtype=torch.float32).reshape(2, 288)
            controller.record_probe(stack)
            controller.write_report()
            dump = np.load(output, allow_pickle=False)
            np.testing.assert_array_equal(dump["lidar_stack"], stack.numpy())
            np.testing.assert_array_equal(dump["crossing_sign"], [1, -1])
            np.testing.assert_array_equal(dump["wall_sign"], [-1, 1])
            self.assertEqual(int(dump["frame_stack"]), 4)


if __name__ == "__main__":
    unittest.main()
