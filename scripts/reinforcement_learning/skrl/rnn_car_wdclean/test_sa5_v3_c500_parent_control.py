"""Contracts for the identical c500 SA5-v3 parent control."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SKRL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SKRL))

import run_sa5_v3_c500_parent_control_screen_cell as runner  # noqa: E402
import sa5_v3_c500_parent_control_screen as protocol  # noqa: E402
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 as control,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_provisional_from_sa4v3_c550_p50 as c550,
)


AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c500_parent_control_authorization_20260823.json"
)
LAUNCH_FREEZE = (
    REPO / "docs/freeze/sa5_v3_c500_parent_control_pilot_launch_v1.json"
)
SCREEN_FREEZE = (
    REPO / "docs/freeze/sa5_v3_c500_parent_control_screen_v1.json"
)
BASE_QUEUE = HERE / "sa5_v3_provisional_pilot_screen_queue.py"
QUEUE = HERE / "sa5_v3_c500_parent_control_screen_queue.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_candidate_manifest_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        protocol,
        "candidate_by_name",
        lambda name: {"name": name, "sha256": f"sha-{name}"},
    )


def _metrics(
    scenario: str,
    cr: float,
    *,
    n: int = 4000,
    sr: float | None = None,
    to: float = 0.0,
) -> dict:
    result = {
        "n": n,
        "sr": 1.0 - cr - to if sr is None else sr,
        "cr": cr,
        "to": to,
    }
    if scenario == "narrow_range":
        result.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return result


def _cell(name: str, scenario: str, metrics: dict) -> dict:
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": name,
        "checkpoint_sha256": f"sha-{name}",
        "scenario": scenario,
        "geometry_stage": protocol.STAGE,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS_BY_SCENARIO[scenario],
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        },
        "metrics": metrics,
        "threshold_pass": protocol.evaluate_metrics(scenario, metrics)[
            "threshold_pass"
        ],
    }


def _matrix(*, native_cr: float = 0.09, target_cr: float = 0.08) -> list[dict]:
    parent = {
        protocol.TARGET_SCENARIO: _metrics(protocol.TARGET_SCENARIO, 0.12),
        "nav_native": _metrics("nav_native", 0.08),
        "narrow_range": _metrics("narrow_range", 0.01),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.04),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.05),
    }
    candidate = {
        protocol.TARGET_SCENARIO: _metrics(protocol.TARGET_SCENARIO, target_cr),
        "nav_native": _metrics("nav_native", native_cr),
        "narrow_range": _metrics("narrow_range", 0.015),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.045),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.055),
    }
    values = {
        "parent_c500": parent,
        "control_it25": candidate,
        "control_it50": {
            **candidate,
            protocol.TARGET_SCENARIO: _metrics(protocol.TARGET_SCENARIO, 0.10),
        },
    }
    return [
        _cell(name, scenario, values[name][scenario])
        for name in ("parent_c500", "control_it25", "control_it50")
        for scenario in protocol.ALL_SCENARIOS
    ]


def test_authorization_is_triggered_by_failed_c550_screen() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    summary_path = REPO / record["trigger"]["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert _sha256(summary_path) == record["trigger"]["summary_sha256"]
    assert summary["extension_authorized"] is False
    assert summary["next_action"] == "RUN_IDENTICAL_C500_PARENT_CONTROL_P50"
    assert record["lineage_status"]["sa4"] == "SA4_NOT_GRADUATED"


def test_c500_checkpoint_contains_resumable_optimizer() -> None:
    path = Path(control.PARENT_CHECKPOINT)
    assert _sha256(path) == control.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 199
    assert payload["total_steps"] == 25600
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_control_changes_only_parent_checkpoint() -> None:
    actual = {
        field.name
        for field in fields(control.CONFIG)
        if field.name not in control.METADATA_FIELDS
        and getattr(control.CONFIG, field.name) != getattr(c550.CONFIG, field.name)
    }
    assert actual == {"checkpoint"}
    assert control.CONFIG.no_resume_optimizer is False


def test_curriculum_and_budget_are_identical_to_c550() -> None:
    for field in fields(control.CONFIG):
        if field.name in control.METADATA_FIELDS or field.name == "checkpoint":
            continue
        assert getattr(control.CONFIG, field.name) == getattr(c550.CONFIG, field.name)
    assert 1.0 - control.CONFIG.narrow_passage_fraction - control.CONFIG.long_corridor_fraction == pytest.approx(0.78)
    assert control.CONFIG.long_corridor_speed_density_mix[:2] == (
        ((0, 1), (0.70, 0.90), 0.10),
        ((1, 1), (0.70, 0.90), 0.10),
    )
    assert control.CONFIG.timesteps == 6400
    assert control.CONFIG.save_interval == 25


def test_launch_and_screen_freezes_match_sources() -> None:
    launch = json.loads(LAUNCH_FREEZE.read_text(encoding="utf-8"))
    assert launch["run"]["config_sha256"] == _sha256(Path(control.__file__))
    assert launch["run"]["trainer_sha256"] == _sha256(
        REPO / launch["run"]["trainer_path"]
    )
    assert launch["identity_contract"]["behavioral_differences"] == [
        "checkpoint"
    ]
    assert json.loads(SCREEN_FREEZE.read_text(encoding="utf-8")) == (
        protocol.screen_protocol()
    )
    assert len(protocol.CANDIDATE_SPECS) * len(protocol.ALL_SCENARIOS) == 15


def test_runner_pins_same_fixed_conditions() -> None:
    for scenario, density, speed, mode in (
        (protocol.TARGET_SCENARIO, (4, 2), protocol.P035, "mixed"),
        ("corridor_low_0s1d", (0, 1), protocol.P060, "lateral"),
        ("corridor_low_1s1d", (1, 1), protocol.P060, "lateral"),
    ):
        joined = " ".join(runner.build_scene_args(scenario, Path("c.json")))
        assert f"--long_corridor_static_obstacles {density[0]}" in joined
        assert f"--long_corridor_dynamic_obstacles {density[1]}" in joined
        assert f"--long_corridor_dynamic_speed_range {speed[0]:g} {speed[1]:g}" in joined
        assert f"--long_corridor_motion_mode {mode}" in joined
        assert "--speed_rate 0.7" in joined
        assert "--actuator_delay_range 1 1" in joined
        assert "--vlp16_noise_mode full" in joined


def test_valid_improvement_and_retention_can_authorize_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_matrix())
    assert verdict["selected_extension_checkpoint"] == "control_it25"
    assert verdict["extension_authorized"] is True
    assert verdict["next_action"] == "EXTEND_SELECTED_C500_LINEAGE_TO_TOTAL_IT300"
    assert verdict["sa4_graduated"] is False
    assert verdict["sa6_started"] is False


def test_native_failure_holds_without_extra_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_matrix(native_cr=0.11))
    assert verdict["extension_authorized"] is False
    assert verdict["next_action"] == (
        "HOLD_SA5_REDESIGN_NO_REWARD_CHANGE_NO_EXTRA_ITERATIONS"
    )


def test_partial_matrix_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    with pytest.raises(ValueError, match="exactly 15 cells"):
        protocol.final_verdict(_matrix()[:-1])


def test_screen_queue_cannot_launch_training_or_sa6() -> None:
    wrapper = QUEUE.read_text(encoding="utf-8")
    implementation = BASE_QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_FAIL_CLOSED" in implementation
    assert "source fingerprint drifted during screen" in implementation
    assert "train_rnn_car_wdclip.py" not in wrapper
    assert "systemd-run" not in wrapper
