"""Launch contracts for the paper LiDAR-noise x command-delay factorial."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from rnn_car_modular.configs import e2e_sa1_paper_factorial_a_ideal_d0_p600 as formal_a
from rnn_car_modular.configs import e2e_sa1_paper_factorial_a_ideal_d0_p600_smoke as smoke_a
from rnn_car_modular.configs import e2e_sa1_paper_factorial_b_full_d0_p600 as formal_b
from rnn_car_modular.configs import e2e_sa1_paper_factorial_b_full_d0_p600_smoke as smoke_b
from rnn_car_modular.configs import e2e_sa1_paper_factorial_c_ideal_u012_p600 as formal_c
from rnn_car_modular.configs import e2e_sa1_paper_factorial_c_ideal_u012_p600_smoke as smoke_c
from rnn_car_modular.configs import e2e_sa1_paper_factorial_d_full_u012_p600 as formal_d
from rnn_car_modular.configs import e2e_sa1_paper_factorial_d_full_u012_p600_smoke as smoke_d

_QUEUE_PATH = Path(__file__).with_name("run_paper_realism_factorial_v1.py")
_QUEUE_SPEC = importlib.util.spec_from_file_location(
    "run_paper_realism_factorial_v1", _QUEUE_PATH
)
assert _QUEUE_SPEC is not None and _QUEUE_SPEC.loader is not None
queue = importlib.util.module_from_spec(_QUEUE_SPEC)
sys.modules[_QUEUE_SPEC.name] = queue
_QUEUE_SPEC.loader.exec_module(queue)


FORMAL_SMOKE_PAIRS = (
    (formal_a.CONFIG, smoke_a.CONFIG),
    (formal_b.CONFIG, smoke_b.CONFIG),
    (formal_c.CONFIG, smoke_c.CONFIG),
    (formal_d.CONFIG, smoke_d.CONFIG),
)
METADATA = {"name", "description", "tags", "notes"}


def _behavioral_diffs(left, right):
    return {
        field.name
        for field in fields(left)
        if field.name not in METADATA
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_each_smoke_only_reduces_runtime_scale():
    for formal, smoke in FORMAL_SMOKE_PAIRS:
        assert _behavioral_diffs(formal, smoke) == {
            "num_envs",
            "timesteps",
            "save_interval",
        }
        assert smoke.num_envs == 64
        assert smoke.timesteps == smoke.rollout_length == 128
        assert smoke.save_interval == 1
        assert smoke.checkpoint is None
        assert smoke.no_resume_optimizer is True
        assert smoke.model_init_seed == -1


def test_queue_is_four_smokes_then_twelve_preregistered_formal_runs():
    assert [arm.key for arm in queue.SMOKE_ARMS] == [
        "smoke_A",
        "smoke_B",
        "smoke_C",
        "smoke_D",
    ]
    assert queue.FORMAL_ORDER == ((42, "CBDA"), (43, "BDCA"), (44, "ABDC"))
    assert len(queue.FORMAL_ARMS) == 12
    assert [arm.key for arm in queue.FORMAL_ARMS] == [
        "formal_s42_C",
        "formal_s42_B",
        "formal_s42_D",
        "formal_s42_A",
        "formal_s43_B",
        "formal_s43_D",
        "formal_s43_C",
        "formal_s43_A",
        "formal_s44_A",
        "formal_s44_B",
        "formal_s44_D",
        "formal_s44_C",
    ]
    assert all(arm.iterations == 600 for arm in queue.FORMAL_ARMS)
    assert all(arm.checkpoints[-1] == "checkpoint_76800.pt" for arm in queue.FORMAL_ARMS)


def test_every_training_seed_has_all_four_cells_once():
    for seed in (42, 43, 44):
        block = [arm for arm in queue.FORMAL_ARMS if arm.seed == seed]
        assert sorted(arm.cell for arm in block) == list("ABCD")
        assert all(f"_s{seed}_" in arm.run_name for arm in block)


def test_runtime_factor_markers_match_each_cell():
    expected = {
        "A": ("ideal", False),
        "B": ("full", False),
        "C": ("ideal", True),
        "D": ("full", True),
    }
    for arm in queue.ARMS:
        assert (arm.lidar_mode, arm.delay_enabled) == expected[arm.cell]
        markers = queue._required_runtime_markers(arm)
        assert any(f"mode={arm.lidar_mode}" in marker for marker in markers)
        assert any("MODEL-INIT-SEED" in marker and f"resolved={arm.seed}" in marker for marker in markers)
        assert any("Actuator DR" in marker for marker in markers) is arm.delay_enabled


def test_authorization_hash_order_and_scope_are_exact():
    with queue.AUTHORIZATION.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == queue.AUTHORIZATION_SHA256
    payload = json.loads(queue.AUTHORIZATION.read_text(encoding="utf-8"))
    assert payload["decision"] == "HUMAN_AUTHORIZED_SMOKE_THEN_12_FORMAL_RUNS"
    assert payload["human_instruction"] == "啟動"
    assert payload["execution"]["order"] == [arm.key for arm in queue.ARMS]
    assert payload["automatic_actions"] == {
        "gpu_smoke_authorized": True,
        "formal_training_authorized": True,
        "run_sequentially": True,
        "fixed_gate_launch_authorized": False,
        "auto_advance_authorized": False,
    }
    assert "evaluation_contract.blocking_instrumentation_gap" in payload["authorization_scope"]["does_not_supersede"]


def test_authorization_and_frozen_source_locks_resolve_now():
    queue._verify_authorization()


def test_launcher_waits_without_reserving_expected_run_and_fails_closed():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    wait_start = source.index("def _wait_for_exclusive_gpu")
    run_start = source.index("def _run_arm")
    expected_write = source.index("_atomic_text(EXPECTED_RUN, arm.run_name", run_start)
    assert wait_start < run_start < expected_write
    assert "INCOMPLETE_NO_VERDICT" in source
    assert '"fixed_gate_started": False' in source
    assert '"auto_advance_started": False' in source
    assert "--wait-for-gpu" in source


def test_launcher_uses_isaaclab_environment_and_explicit_seed_override():
    assert queue.CONDA_PREFIX == Path("/home/aa/miniconda3/envs/env_isaaclab")
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert 'environment["CONDA_PREFIX"] = str(CONDA_PREFIX)' in source
    assert '"--seed",' in source
    assert "str(arm.seed)" in source
