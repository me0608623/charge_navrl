import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "charge_skrl"
    / "fixed_scene_goals.py"
)
_SPEC = importlib.util.spec_from_file_location("fixed_scene_goals", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _goal_term() -> SimpleNamespace:
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        _long_corridor_active=torch.tensor([False, True, False, False]),
        _long_corridor_goal_w=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 4.1, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        _narrow_bridge_active=torch.tensor([False, False, True, False]),
        _narrow_bridge_goal_w=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.25, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        _local_goal_world=torch.full((4, 2), -9.0),
    )
    term = SimpleNamespace(
        goal_pos_w=torch.full((4, 3), -5.0),
        all_goals_pos_w=torch.full((4, 3, 3), -7.0),
        env=env,
    )
    return term


def test_fixed_scene_goal_mask_is_scoped_to_injected_envs() -> None:
    term = _goal_term()

    mask = _MOD.fixed_scene_goal_mask(term.env, torch.arange(4))

    assert mask.tolist() == [False, True, True, False]


def test_restore_fixed_scene_goals_updates_every_goal_view() -> None:
    term = _goal_term()

    claimed = _MOD.restore_fixed_scene_goals(
        term.env, term, torch.arange(4)
    )

    assert claimed.tolist() == [False, True, True, False]
    torch.testing.assert_close(
        term.goal_pos_w[1], torch.tensor([0.0, 4.1, 0.0])
    )
    torch.testing.assert_close(
        term.goal_pos_w[2], torch.tensor([3.0, 0.25, 0.0])
    )
    torch.testing.assert_close(
        term.all_goals_pos_w[1],
        torch.tensor([[0.0, 4.1, 0.0]]).expand(3, -1),
    )
    torch.testing.assert_close(
        term.all_goals_pos_w[2],
        torch.tensor([[3.0, 0.25, 0.0]]).expand(3, -1),
    )
    torch.testing.assert_close(
        term.env._local_goal_world[1], torch.tensor([0.0, 4.1])
    )
    torch.testing.assert_close(
        term.env._local_goal_world[2], torch.tensor([3.0, 0.25])
    )
    assert torch.equal(term.goal_pos_w[0], torch.full((3,), -5.0))
    assert torch.equal(term.goal_pos_w[3], torch.full((3,), -5.0))


def test_overlapping_fixed_scene_injectors_fail_fast() -> None:
    term = _goal_term()
    term.env._narrow_bridge_active[1] = True

    with pytest.raises(RuntimeError, match="multiple fixed-scene injectors"):
        _MOD.restore_fixed_scene_goals(term.env, term, torch.arange(4))
