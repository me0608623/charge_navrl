"""Contract tests for the SA5-v3 Stage-3 A3/B3 Phase-A screen."""

from __future__ import annotations

from pathlib import Path

import pytest

import sa5_v3_c50_stage3_phase_a_screen as protocol


def _payload(name: str, scenario: str, sr: float, cr: float, to: float = 0.0) -> dict:
    metrics = {"n": 2_000, "sr": sr, "cr": cr, "to": to}
    return {
        "checkpoint_name": name,
        "scenario": scenario,
        "metrics": metrics,
        "threshold_pass": protocol.evaluate_metrics(scenario, metrics)["threshold_pass"],
    }


def _matrix(
    *,
    a3_p060: tuple[float, float] = (0.15, 0.14),
    b3_p060: tuple[float, float] = (0.09, 0.10),
    a3_p080: tuple[float, float] = (0.08, 0.09),
    b3_p080: tuple[float, float] = (0.08, 0.09),
) -> list[dict]:
    rows = []
    values = {
        protocol.A3_CONTROL["name"]: (*a3_p060, *a3_p080),
        protocol.B3_P060["name"]: (*b3_p060, *b3_p080),
    }
    for candidate in protocol.CANDIDATE_SPECS:
        for scenario, cr in zip(protocol.ALL_SCENARIOS, values[candidate["name"]]):
            rows.append(_payload(candidate["name"], scenario, 1.0 - cr, cr))
    return rows


def test_protocol_is_eight_cells_and_never_authorizes_progression() -> None:
    frozen = protocol.screen_protocol()
    assert frozen["matrix"]["cell_count"] == 8
    assert frozen["fixed_evaluation"]["seed"] == 818
    assert frozen["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert frozen["decision"]["starts_training"] is False
    assert frozen["decision"]["selects_parent"] is False
    assert frozen["decision"]["starts_sa6"] is False


@pytest.mark.parametrize(
    ("scenario", "speed", "density"),
    [
        ("p060_0s1d", (0.50, 0.70), (0, 1)),
        ("p060_1s1d", (0.50, 0.70), (1, 1)),
        ("p080_0s1d", (0.70, 0.90), (0, 1)),
        ("p080_1s1d", (0.70, 0.90), (1, 1)),
    ],
)
def test_speed_and_density_mapping(
    scenario: str,
    speed: tuple[float, float],
    density: tuple[int, int],
) -> None:
    assert protocol.speed_range_for(scenario) == speed
    assert protocol.density_for(scenario) == density


def test_phase_a_pass_requires_absolute_p060_and_two_point_gain(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "validate_cell", lambda _: None)
    verdict = protocol.final_verdict(_matrix())
    assert verdict["p060_absolute_pass"] is True
    assert verdict["worst_p060_cr"]["improvement"] >= 0.02
    assert verdict["p080_retention_pass"] is True
    assert verdict["phase_a_pass"] is True
    assert verdict["selected_parent"] is None


def test_phase_a_rejects_small_p060_gain(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "validate_cell", lambda _: None)
    verdict = protocol.final_verdict(
        _matrix(a3_p060=(0.11, 0.11), b3_p060=(0.095, 0.10))
    )
    assert verdict["p060_absolute_pass"] is True
    assert verdict["worst_p060_cr"]["improvement"] < 0.02
    assert verdict["phase_a_pass"] is False


def test_phase_a_rejects_p080_regression(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "validate_cell", lambda _: None)
    verdict = protocol.final_verdict(_matrix(b3_p080=(0.11, 0.09)))
    assert verdict["p060_absolute_pass"] is True
    assert verdict["p080_retention_pass"] is False
    assert verdict["phase_a_pass"] is False


def test_runner_preserves_distinct_p060_and_p080_ranges() -> None:
    source = Path(__file__).with_name(
        "run_sa5_v3_c50_stage3_phase_a_screen_cell.py"
    ).read_text(encoding="utf-8")
    assert "protocol.speed_range_for(scenario)" in source
    assert "protocol.density_for(_scenario)" in source
    assert "corridor_motion_mode=\"lateral\"" in source
