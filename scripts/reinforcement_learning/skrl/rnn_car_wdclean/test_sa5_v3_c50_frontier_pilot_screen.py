"""Contracts for the c50 difficulty-frontier curriculum acceptance screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_sa5_v3_c50_frontier_pilot_screen_cell as runner  # noqa: E402
import sa5_v3_c50_frontier_pilot_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_v3_c50_frontier_pilot_screen_v1.json"
AUTHORIZATION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json"
)
QUEUE = HERE / "sa5_v3_c50_frontier_pilot_screen_queue.py"


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
    sr: float | None = None,
    to: float = 0.0,
    n: int = 4000,
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
    verdict = protocol.evaluate_metrics(scenario, metrics)
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
        "lidar_distractor_eligibility": (
            protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        },
        "metrics": metrics,
        "threshold_pass": verdict["threshold_pass"],
    }


def _matrix(
    *,
    it25_target_cr: float = 0.20,
    it25_target_sr: float = 0.80,
    it25_target_to: float = 0.0,
    it25_native_cr: float = 0.09,
) -> list[dict]:
    anchor = {
        protocol.TARGET_SCENARIO: _metrics(
            protocol.TARGET_SCENARIO, 0.30, sr=0.70
        ),
        "nav_native": _metrics("nav_native", 0.085),
        "narrow_range": _metrics("narrow_range", 0.01),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.04),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.05),
    }
    it25 = {
        protocol.TARGET_SCENARIO: _metrics(
            protocol.TARGET_SCENARIO,
            it25_target_cr,
            sr=it25_target_sr,
            to=it25_target_to,
        ),
        "nav_native": _metrics("nav_native", it25_native_cr),
        "narrow_range": _metrics("narrow_range", 0.015),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.045),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.055),
    }
    it50 = {
        protocol.TARGET_SCENARIO: _metrics(
            protocol.TARGET_SCENARIO, 0.28, sr=0.72
        ),
        "nav_native": _metrics("nav_native", 0.09),
        "narrow_range": _metrics("narrow_range", 0.015),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.045),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.055),
    }
    values = {
        "anchor_c50": anchor,
        "pilot_it25": it25,
        "pilot_it50": it50,
    }
    return [
        _cell(name, scenario, values[name][scenario])
        for name in ("anchor_c50", "pilot_it25", "pilot_it50")
        for scenario in protocol.ALL_SCENARIOS
    ]


def test_authorization_and_frozen_protocol_match() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["acceptance"]["target_sr_must_increase"] is True
    assert authorization["acceptance"]["target_to_max"] == 0.05
    assert authorization["acceptance"]["native_degradation_max_pp"] == 2.0
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        protocol.screen_protocol()
    )
    assert len(protocol.CANDIDATE_SPECS) * len(protocol.ALL_SCENARIOS) == 15


def test_anchor_is_diagnostic_and_sa6_is_forbidden() -> None:
    frozen = protocol.screen_protocol()
    assert frozen["lineage"] == {
        "anchor": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
        "sa5": "HOLD_NOT_GRADUATED",
        "sa6": "HOLD_NOT_AUTHORIZED",
    }
    assert frozen["decision"]["screen_starts_sa6"] is False
    assert frozen["decision"]["screen_authorizes_graduation"] is False


def test_runner_pins_target_retention_noise_delay_and_speed() -> None:
    for scenario, density, speed, mode in (
        (protocol.TARGET_SCENARIO, (4, 2), protocol.P035, "mixed"),
        ("corridor_low_0s1d", (0, 1), protocol.P060, "lateral"),
        ("corridor_low_1s1d", (1, 1), protocol.P060, "lateral"),
    ):
        joined = " ".join(runner.build_scene_args(scenario, Path("out.json")))
        assert f"--long_corridor_static_obstacles {density[0]}" in joined
        assert f"--long_corridor_dynamic_obstacles {density[1]}" in joined
        assert f"--long_corridor_dynamic_speed_range {speed[0]:g} {speed[1]:g}" in joined
        assert f"--long_corridor_motion_mode {mode}" in joined
        assert "--speed_rate 0.7" in joined
        assert "--speed_rate_obs ego" in joined
        assert "--actuator_delay_range 1 1" in joined
        assert "--vlp16_noise_mode full" in joined


def test_target_sr_and_cr_improve_with_retention_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_matrix())
    assert verdict["selected_development_checkpoint"] == "pilot_it25"
    assert verdict["selected_extension_checkpoint"] is None
    assert verdict["pilot_acceptance_observed"] is True
    assert verdict["graduation_authorized"] is False
    assert verdict["extension_authorized"] is False
    assert verdict["sa6_started"] is False


def test_collision_to_timeout_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(
        _matrix(
            it25_target_cr=0.20,
            it25_target_sr=0.70,
            it25_target_to=0.10,
        )
    )
    row = verdict["candidate_results"][0]
    assert row["target_improvement"]["checks"]["minimum_cr_improvement"]
    assert not row["target_improvement"]["checks"]["minimum_sr_improvement"]
    assert not row["target_improvement"]["checks"]["timeout"]
    assert row["pilot_acceptance_pass"] is False


def test_sub_half_point_sr_change_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(
        _matrix(it25_target_cr=0.20, it25_target_sr=0.704)
    )
    assert not verdict["candidate_results"][0]["target_improvement"]["pass"]


def test_native_above_ten_percent_blocks_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_matrix(it25_native_cr=0.11))
    row = verdict["candidate_results"][0]
    assert row["target_improvement"]["pass"] is True
    assert row["retention"]["nav_native"]["absolute_pass"] is False
    assert row["pilot_acceptance_pass"] is False


def test_partial_matrix_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_manifest_stub(monkeypatch)
    with pytest.raises(ValueError, match="exactly 15 cells"):
        protocol.final_verdict(_matrix()[:-1])


def test_queue_cannot_launch_training_or_sa6() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    base_source = (
        HERE / "sa5_v3_provisional_pilot_screen_queue.py"
    ).read_text(encoding="utf-8")
    combined = source + base_source
    assert "INCOMPLETE_FAIL_CLOSED" in combined
    assert "source fingerprint drifted during screen" in combined
    assert "train_rnn_car_wdclip.py" not in combined
    assert "systemd-run" not in combined
