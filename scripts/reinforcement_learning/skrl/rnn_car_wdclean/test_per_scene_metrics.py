"""Tests for per-scene SR/CR/TO accounting in train_rnn_car_wdclip.

The trainer imports Isaac Lab at module scope, so it cannot be imported here.
Instead the scene-identity helper is lifted out of the real file by AST, written
to a temporary module, and imported. The behavioural tests therefore run the
code that actually ships; the remaining tests assert source-level contracts that
behaviour alone cannot check (ordering against ``env.step``, which denominator
is used, which key timeout is *not* read from).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

TRAINER = (
    Path(__file__).resolve().parents[1] / "train" / "train_rnn_car_wdclip.py"
)
SOURCE = TRAINER.read_text(encoding="utf-8")


def _lift(tmp_dir: Path, *names: str):
    """Copy the named top-level definitions into an importable module."""
    tree = ast.parse(SOURCE)
    picked: list[ast.stmt] = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in names:
                picked.append(node)
                found.add(node.name)
        elif isinstance(node, ast.Assign):
            hit = [t.id for t in node.targets
                   if isinstance(t, ast.Name) and t.id in names]
            if hit:
                picked.append(node)
                found.update(hit)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in names:
                picked.append(node)
                found.add(node.target.id)
    missing = set(names) - found
    assert not missing, f"trainer no longer defines: {sorted(missing)}"

    path = tmp_dir / "_lifted_scene.py"
    path.write_text(
        "import torch\n\n" + ast.unparse(ast.Module(body=picked, type_ignores=[])),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("_lifted_scene", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lifted_scene"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return _lift(
        tmp_path_factory.mktemp("lifted"),
        "SCENE_NAMES",
        "_SCENE_ID",
        "scene_ids_from_env",
    )


class FakeEnv:
    def __init__(self, n, narrow=None, corridor=None, previous=None):
        z = torch.zeros(n, dtype=torch.bool)
        self._narrow_bridge_active = z.clone() if narrow is None else narrow
        self._long_corridor_active = z.clone() if corridor is None else corridor
        self._previous_stage_replay_active = (
            z.clone() if previous is None else previous
        )


def _mask(n, *idx):
    m = torch.zeros(n, dtype=torch.bool)
    for i in idx:
        m[i] = True
    return m


# --- behaviour: scene_ids_from_env --------------------------------------


def test_unclaimed_envs_are_native(scene):
    ids = scene.scene_ids_from_env(FakeEnv(4), 4, "cpu")
    assert [scene.SCENE_NAMES[i] for i in ids.tolist()] == ["native"] * 4


def test_each_mask_maps_to_its_own_scene(scene):
    env = FakeEnv(
        6, narrow=_mask(6, 1), corridor=_mask(6, 2, 3), previous=_mask(6, 5)
    )
    ids = scene.scene_ids_from_env(env, 6, "cpu")
    assert [scene.SCENE_NAMES[i] for i in ids.tolist()] == [
        "native", "narrow", "corridor", "corridor", "native", "sa5_general",
    ]


def test_ids_partition_the_env_set(scene):
    env = FakeEnv(8, narrow=_mask(8, 0), corridor=_mask(8, 4))
    ids = scene.scene_ids_from_env(env, 8, "cpu")
    assert ids.numel() == 8
    assert set(ids.tolist()) <= set(range(len(scene.SCENE_NAMES)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"narrow": _mask(4, 1), "corridor": _mask(4, 1)},
        {"narrow": _mask(4, 2), "previous": _mask(4, 2)},
        {"corridor": _mask(4, 3), "previous": _mask(4, 3)},
    ],
)
def test_overlapping_masks_raise(scene, kwargs):
    with pytest.raises(RuntimeError, match="overlap"):
        scene.scene_ids_from_env(FakeEnv(4, **kwargs), 4, "cpu")


def test_overlap_message_reports_env_count(scene):
    env = FakeEnv(5, narrow=_mask(5, 0, 1), corridor=_mask(5, 0, 1))
    with pytest.raises(RuntimeError, match="2 envs"):
        scene.scene_ids_from_env(env, 5, "cpu")


def test_native_is_id_zero_so_full_is_a_valid_default(scene):
    assert scene.SCENE_NAMES[0] == "native"


def test_the_three_requested_scenes_are_present(scene):
    assert {"native", "narrow", "corridor"} <= set(scene.SCENE_NAMES)


# --- source contracts ---------------------------------------------------


def test_scene_snapshot_is_taken_before_env_step():
    snap = SOURCE.index("_scene_id_step = scene_ids_from_env(")
    step = SOURCE.index(
        "next_obs, reward, terminated, truncated, info = env.step("
    )
    assert snap < step, "scene identity must be sampled before env.step()"


def test_corridor_family_snapshot_is_taken_before_env_step():
    snap = SOURCE.index(
        "_corridor_family_snapshot_step = snapshot_corridor_families("
    )
    step = SOURCE.index(
        "next_obs, reward, terminated, truncated, info = env.step("
    )
    assert snap < step, "corridor family must be sampled before env.step()"


def test_step_receives_scene_and_termination_flags():
    call = SOURCE[SOURCE.index("            metrics.step(obs,"):]
    call = call[: call.index("\n\n")]
    for kw in (
        "scene_id=",
        "corridor_family_snapshot=",
        "terminated_flat=",
        "truncated_flat=",
    ):
        assert kw in call, f"metrics.step is missing {kw}"


def test_timeout_is_not_read_from_reward_breakdown():
    """clean_progress drops `timeout` when merging, so a lookup reads nothing."""
    block = SOURCE[SOURCE.index("Per-scene attribution"):]
    block = block[: block.index("self._ep_reward[ids] = 0.0")]
    assert 'reward_breakdown["timeout"]' not in block
    assert "truncated_flat[idx]" in block


def _emission_block() -> str:
    block = SOURCE[SOURCE.index("for _scene in SCENE_NAMES:"):]
    return block[: block.index("# --- Goal-directed")]


def test_each_scene_divides_by_its_own_episode_count():
    block = _emission_block()
    assert '_c["goal"] / _n' in block
    assert '_c["collision"] / _n' in block
    assert '_c["timeout"] / _n' in block
    assert "self._total_eps" not in block, "must not use the global denominator"


def test_empty_scene_emits_nothing_rather_than_zero():
    block = _emission_block()
    assert "if _n == 0:" in block
    assert "continue" in block


def test_scene_counters_are_reset_each_iteration():
    assert "for _sc in self._scene_counts.values():" in SOURCE


def test_corridor_family_iteration_counters_are_reset():
    assert "self._corridor_family_metrics.reset_iteration()" in SOURCE


def test_outcome_gate_requires_exactly_one_class():
    assert "if hits != 1:" in SOURCE
    assert "exactly one" in SOURCE


def test_global_metrics_are_untouched():
    """The rate-derived global path must keep its original denominator."""
    assert 'm["charge/goal_reach_rate"] = self._goal_reached / te' in SOURCE
    assert 'm["charge/hit_probability"] = self._collision / te' in SOURCE
    assert 'm["charge/timeout_rate"] = self._timeout / te' in SOURCE


def test_scene_fields_reach_the_supervisor_jsonl():
    """supervisor_metrics.jsonl is a curated dict; scene keys must be copied in."""
    block = SOURCE[SOURCE.index("_supervisor_metrics = {"):]
    block = block[: block.index("\n        }")]
    assert 'if _k.startswith("scene/")' in block
    assert '"total_episodes"' in block, (
        "the shared denominator must be logged so per-scene counts can be "
        "reconciled against the global path"
    )


def test_corridor_family_fields_reach_the_supervisor_jsonl():
    block = SOURCE[SOURCE.index("_supervisor_metrics = {"):]
    block = block[: block.index("\n        }")]
    assert 'if _k.startswith("corridor_family/")' in block


def test_existing_corridor_scene_metrics_are_not_replaced():
    emission = _emission_block()
    for key in ("episodes", "sr", "cr", "timeout"):
        assert f'm[f"scene/{{_scene}}/{key}"]' in emission
