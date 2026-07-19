import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config"
    / "charge_skrl/goal_sampling.py"
)
_SPEC = importlib.util.spec_from_file_location("charge_goal_sampling", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
masked_obstacle_distances = _MODULE.masked_obstacle_distances
visible_obstacle_xy = _MODULE.visible_obstacle_xy


def test_hidden_and_nonfinite_obstacles_are_not_visible():
    roots = torch.tensor(
        [
            [2.0, 3.0, 0.9],
            [4.0, 5.0, -10.0],
            [float("nan"), 1.0, 0.9],
            [1.0, float("inf"), 0.9],
        ]
    )

    xy, visible = visible_obstacle_xy(roots, torch.zeros(4, 2))

    assert visible.tolist() == [True, False, False, False]
    assert torch.isfinite(xy).all()


def test_masked_obstacles_cannot_poison_clearance_minimum():
    candidate = torch.tensor([[0.0, 0.0]])
    obstacles = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    visible = torch.tensor([[True, False, False]])

    distances = masked_obstacle_distances(candidate, obstacles, visible)

    assert distances[0, 0].item() == 2.0
    assert torch.isinf(distances[0, 1:]).all()
    assert distances.min().item() == 2.0


def test_no_visible_obstacles_has_infinite_clearance():
    distances = masked_obstacle_distances(
        torch.tensor([[1.0, -1.0]]),
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, dtype=torch.bool),
    )

    assert torch.isinf(distances).all()
