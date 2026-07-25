"""Sim-free checks for Gate5 path-quality accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "play_eval"
    / "narrow_gap_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("narrow_gap_eval", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

NarrowGapController = _MOD.NarrowGapController
NarrowGapSpec = _MOD.NarrowGapSpec


class _Scene(dict):
    def __init__(self, robot, num_envs: int) -> None:
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(num_envs, 3)


def _controller(num_envs: int = 2) -> tuple[NarrowGapController, object]:
    data = SimpleNamespace(
        root_pos_w=torch.zeros(num_envs, 3),
        root_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * num_envs
        ),
    )
    robot = SimpleNamespace(data=data)
    env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene=_Scene(robot, num_envs),
    )
    controller = NarrowGapController(
        env, NarrowGapSpec(yaw_limit_deg=10.0)
    )
    return controller, robot


def test_directness_metrics_distinguish_straight_and_detour_paths() -> None:
    controller, robot = _controller()
    direct = [
        (-2.0, 0.0),
        (-1.0, 0.0),
        (0.1, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (3.0, 0.0),
    ]
    detour = [
        (-2.5, 0.0),
        (-2.0, 1.4),
        (-2.8, 1.6),
        (-1.5, 1.6),
        (0.1, 1.6),
        (1.5, 1.0),
        (3.0, 0.0),
    ]

    for direct_xy, detour_xy in zip(direct, detour):
        robot.data.root_pos_w[:, :2] = torch.tensor(
            [direct_xy, detour_xy]
        )
        controller.observe()
    controller.finish_episodes(torch.tensor([0, 1]))

    assert controller.completed_episodes == 2
    assert controller.crossed_episodes == 2
    assert controller.direct_crossed_episodes == 1
    summary = controller.summary_line()
    assert "direct_crossing_rate=0.500000" in summary
    assert "path_length_ratio_p95=" in summary
    assert "first_cross_time_s_p95=" in summary
    assert "max_pre_cross_abs_y_m_p95=" in summary
    assert "backtrack_distance_m_p95=" in summary


def test_terminal_auto_reset_jump_is_not_counted_as_backtracking() -> None:
    controller, robot = _controller(num_envs=1)
    robot.data.root_pos_w[0, :2] = torch.tensor([-1.0, 0.0])
    controller.observe()
    path_before = controller._path_length_m.clone()
    backtrack_before = controller._backtrack_distance_m.clone()

    robot.data.root_pos_w[0, :2] = torch.tensor([-3.0, 0.0])
    controller.observe(torch.tensor([True]))

    assert torch.equal(controller._path_length_m, path_before)
    assert torch.equal(controller._backtrack_distance_m, backtrack_before)
