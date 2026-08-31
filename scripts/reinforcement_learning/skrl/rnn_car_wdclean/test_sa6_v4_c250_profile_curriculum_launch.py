"""Launch and smoke contracts for the SA6 c250 matched pilot."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_control_p50 as control,
)
from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_control_p50_smoke as control_smoke,
)
from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_rung1_p50 as rung1,
)
from rnn_car_modular.configs import (
    e2e_sa6_v4_c250_profile_curriculum_rung1_p50_smoke as rung1_smoke,
)

import run_sa6_v4_c250_profile_curriculum_pilot as queue


def _diffs(left, right, metadata):
    return {
        field.name
        for field in fields(left)
        if field.name not in metadata
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_smokes_only_change_runtime_scale():
    expected = {"num_envs", "timesteps", "save_interval"}
    assert _diffs(control_smoke.CONFIG, control.CONFIG, control_smoke.METADATA_FIELDS) == expected
    assert _diffs(rung1_smoke.CONFIG, rung1.CONFIG, rung1_smoke.METADATA_FIELDS) == expected
    assert control_smoke.CONFIG.num_envs == rung1_smoke.CONFIG.num_envs == 64
    assert control_smoke.CONFIG.timesteps == rung1_smoke.CONFIG.timesteps == 128


def test_queue_order_is_smoke_then_matched_control_then_intervention():
    assert [arm.label for arm in queue.ARMS] == [
        "control_smoke",
        "rung1_smoke",
        "control",
        "rung1",
    ]
    assert [arm.iterations for arm in queue.ARMS] == [1, 1, 50, 50]
    assert queue.ARMS[2].config_module == control.CONFIG.name
    assert queue.ARMS[3].config_module == rung1.CONFIG.name


def test_launch_authorization_is_exact_and_bounded():
    authorization = json.loads(queue.AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["decision"] == "HUMAN_AUTHORIZED_MATCHED_CONTROL_THEN_RUNG1_LAUNCH"
    assert authorization["execution"]["order"] == [arm.label for arm in queue.ARMS]
    assert authorization["automatic_actions"] == {
        "gpu_smoke_authorized": True,
        "formal_control_launch_authorized": True,
        "formal_rung1_launch_authorized": True,
        "run_arms_sequentially": True,
        "fixed_gate_auto_start": False,
        "extend_beyond_50_iterations": False,
        "start_sa7": False,
        "enable_teacher_distillation_or_override": False,
    }


def test_queue_fail_closes_before_second_arm_and_never_starts_gate_or_sa7():
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "subprocess.run" in source
    assert "if completed.returncode != 0" in source
    assert "raise RuntimeError" in source
    assert '"fixed_gate_auto_start": False' in source
    assert '"sa7_started": False' in source
    assert "auto_advance" not in source


def test_queue_pins_isaaclab_subprocess_to_the_isaaclab_conda_environment():
    assert queue.CONDA_PREFIX == Path("/home/aa/miniconda3/envs/env_isaaclab")
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert 'environment["CONDA_PREFIX"] = str(CONDA_PREFIX)' in source
    assert 'environment["PATH"] = f"{CONDA_PREFIX / \'bin\'}:' in source


def test_formal_arms_keep_exact_fifty_iteration_budget():
    assert control.CONFIG.timesteps == rung1.CONFIG.timesteps == 6_400
    assert control.CONFIG.save_interval == rung1.CONFIG.save_interval == 25
    assert queue.ARMS[2].checkpoints == ("checkpoint_3200.pt", "checkpoint_6400.pt")
    assert queue.ARMS[3].checkpoints == ("checkpoint_3200.pt", "checkpoint_6400.pt")
