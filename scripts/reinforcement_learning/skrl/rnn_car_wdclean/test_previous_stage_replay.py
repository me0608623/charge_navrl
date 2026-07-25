"""Sim-free tests for the fixed SA5 general-scene replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


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
    / "mdp"
    / "events"
    / "previous_stage_replay.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "previous_stage_replay", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

configure_previous_stage_replay = _MOD.configure_previous_stage_replay
deployment_behavior_mix = _MOD.deployment_behavior_mix


def _allocate_counts(mix: dict[str, float], total: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    remaining = total
    for index, (name, ratio) in enumerate(mix.items()):
        if index == len(mix) - 1:
            counts[name] = remaining
        else:
            count = min(round(ratio * total), remaining)
            counts[name] = count
            remaining -= count
    return counts


def test_sa5_mix_allocates_exactly_ten_static_and_three_dynamic() -> None:
    mix = deployment_behavior_mix(10, 3)
    counts = _allocate_counts(mix, 13)
    assert sum(counts.values()) == 13
    assert counts == {
        "static": 10,
        "horizontal_crossing": 1,
        "path_crossing": 1,
        "patrol": 1,
        "random_walk": 0,
        "head_on": 0,
    }


def test_configure_previous_stage_replay_updates_reset_event() -> None:
    event = SimpleNamespace(params={})
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(previous_stage_replay=event)
    )
    configure_previous_stage_replay(
        env_cfg,
        fraction=0.10,
        static_obstacles=10,
        dynamic_obstacles=3,
        min_walls=2,
        max_walls=3,
        wall_length=4.0,
        obstacle_boundary=5.5,
    )
    assert event.params == {
        "fraction": 0.10,
        "static_obstacles": 10,
        "dynamic_obstacles": 3,
        "min_walls": 2,
        "max_walls": 3,
        "wall_length": 4.0,
        "obstacle_boundary": 5.5,
    }


def test_invalid_previous_stage_fraction_is_rejected() -> None:
    event = SimpleNamespace(params={})
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(previous_stage_replay=event)
    )
    try:
        configure_previous_stage_replay(env_cfg, fraction=0.61)
    except ValueError:
        return
    raise AssertionError("unsafe previous-stage replay fraction was accepted")


def test_recovery_allows_forty_percent_previous_stage_replay() -> None:
    event = SimpleNamespace(params={})
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(previous_stage_replay=event)
    )
    configure_previous_stage_replay(env_cfg, fraction=0.40)
    assert event.params["fraction"] == 0.40
