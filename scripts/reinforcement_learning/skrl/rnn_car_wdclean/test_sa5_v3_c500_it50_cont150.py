"""Contracts for the bounded c500-it50 to it150 continuation."""

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

import run_sa5_v3_c500_it50_cont150_screen_cell as runner  # noqa: E402
import sa5_v3_c500_it50_cont150_screen as protocol  # noqa: E402
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c500_it50_cont150 as continuation,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 as pilot,
)


AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c500_it50_cont150_authorization_20260823.json"
)
LAUNCH_FREEZE = REPO / "docs/freeze/sa5_v3_c500_it50_cont150_launch_v1.json"
SCREEN_FREEZE = (
    REPO / "docs/freeze/sa5_v3_c500_it50_cont150_screen_v1.json"
)
QUEUE = HERE / "sa5_v3_c500_it50_cont150_screen_queue.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_manifest_stub(monkeypatch: pytest.MonkeyPatch) -> None:
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
    values = {
        "n": n,
        "sr": 1.0 - cr - to if sr is None else sr,
        "cr": cr,
        "to": to,
    }
    if scenario == "narrow_range":
        values.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return values


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


def _phase_a() -> list[dict]:
    values = {
        "c50": (0.56, 0.133),
        "c75": (0.50, 0.12),
        "c100": (0.52, 0.09),
        "c125": (0.48, 0.11),
        "c150": (0.45, 0.095),
    }
    return [
        _cell(name, scenario, _metrics(scenario, cr))
        for name in values
        for scenario, cr in zip(protocol.PHASE_A_SCENARIOS, values[name])
    ]


def _phase_b(names: tuple[str, str] = ("c100", "c150")) -> list[dict]:
    values = {
        "narrow_range": 0.0,
        "corridor_low_0s1d": 0.04,
        "corridor_low_1s1d": 0.07,
    }
    return [
        _cell(name, scenario, _metrics(scenario, values[scenario]))
        for name in names
        for scenario in protocol.PHASE_B_SCENARIOS
    ]


def test_authorization_preserves_prior_fail_and_bounds_continuation() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    prior_path = REPO / record["protocol_amendment"]["prior_summary"]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert _sha256(prior_path) == record["protocol_amendment"][
        "prior_summary_sha256"
    ]
    assert prior["extension_authorized"] is False
    assert record["parent"]["optimizer_resume"] is True
    assert record["continuation"]["conceptual_end_iteration"] == 150
    assert record["lineage_status"]["sa6"] == "HOLD_NOT_AUTHORIZED"


def test_parent_is_exact_it50_with_resumable_optimizer() -> None:
    path = Path(continuation.PARENT_CHECKPOINT)
    assert _sha256(path) == continuation.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 49
    assert payload["total_steps"] == 6400
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_continuation_changes_only_checkpoint_and_budget() -> None:
    actual = {
        field.name
        for field in fields(continuation.CONFIG)
        if field.name not in continuation.METADATA_FIELDS
        and getattr(continuation.CONFIG, field.name)
        != getattr(pilot.CONFIG, field.name)
    }
    assert actual == {"checkpoint", "timesteps"}
    assert continuation.CONFIG.no_resume_optimizer is False
    assert continuation.CONFIG.save_interval == pilot.CONFIG.save_interval == 25


def test_behavioral_contract_is_identical_except_budget_and_parent() -> None:
    for field in fields(continuation.CONFIG):
        if field.name in continuation.METADATA_FIELDS | {"checkpoint", "timesteps"}:
            continue
        assert getattr(continuation.CONFIG, field.name) == getattr(
            pilot.CONFIG, field.name
        )
    assert 1.0 - continuation.CONFIG.narrow_passage_fraction - continuation.CONFIG.long_corridor_fraction == pytest.approx(0.78)
    assert continuation.CONFIG.long_corridor_speed_density_mix == pilot.CONFIG.long_corridor_speed_density_mix
    assert continuation.CONFIG.timesteps == 12800
    assert continuation.CHECKPOINTS == (
        (75, "checkpoint_3200.pt"),
        (100, "checkpoint_6400.pt"),
        (125, "checkpoint_9600.pt"),
        (150, "checkpoint_12800.pt"),
    )


def test_launch_and_screen_freezes_match_sources() -> None:
    launch = json.loads(LAUNCH_FREEZE.read_text(encoding="utf-8"))
    assert launch["run"]["config_sha256"] == _sha256(Path(continuation.__file__))
    assert launch["run"]["trainer_sha256"] == _sha256(
        REPO / launch["run"]["trainer_path"]
    )
    assert launch["identity_contract"]["behavioral_differences"] == [
        "checkpoint",
        "timesteps",
    ]
    assert json.loads(SCREEN_FREEZE.read_text(encoding="utf-8")) == (
        protocol.screen_protocol()
    )
    assert len(protocol.CANDIDATE_SPECS) * len(protocol.PHASE_A_SCENARIOS) == 10
    assert 2 * len(protocol.PHASE_B_SCENARIOS) == 6


def test_runner_pins_fixed_conditions() -> None:
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


def test_phase_a_ranking_is_balanced_minimax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    summary = protocol.rank_phase_a(_phase_a())
    assert summary["ranking"][:2] == ["c100", "c150"]
    assert summary["selected_top_two"] == ["c100", "c150"]


def test_valid_top_two_can_accept_one_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_phase_a(), _phase_b())
    assert verdict["acceptance_authorized"] is True
    assert verdict["selected_provisional_sa5_parent"] == "c100"
    assert verdict["sa4_graduated"] is False
    assert verdict["sa6_started"] is False


def test_native_failure_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_manifest_stub(monkeypatch)
    phase_a = _phase_a()
    for cell in phase_a:
        if cell["checkpoint_name"] in {"c100", "c150"} and cell["scenario"] == "nav_native":
            cell["metrics"] = _metrics("nav_native", 0.11)
            cell["threshold_pass"] = False
    selected = tuple(protocol.rank_phase_a(phase_a)["selected_top_two"])
    verdict = protocol.final_verdict(phase_a, _phase_b(selected))
    assert verdict["acceptance_authorized"] is False
    assert verdict["next_action"] == (
        "HOLD_SA5_AFTER_BOUNDED_CONTINUATION_NO_EXTRA_ITERATIONS"
    )


def test_partial_phase_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_manifest_stub(monkeypatch)
    with pytest.raises(ValueError, match="exactly 10 cells"):
        protocol.rank_phase_a(_phase_a()[:-1])
    with pytest.raises(ValueError, match="exactly 6 cells"):
        protocol.final_verdict(_phase_a(), _phase_b()[:-1])


def test_screen_queue_cannot_launch_training_or_sa6() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_FAIL_CLOSED" in source
    assert "continuation source fingerprint drifted" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source
