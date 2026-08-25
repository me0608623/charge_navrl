"""Decision and wiring tests for the bounded-continuation 24-cell screen."""

from __future__ import annotations

from pathlib import Path

import pytest

import sa5_v3_c50_stage3_b3_cont25_screen as protocol


SPECS = tuple(
    {"name": name, "conceptual_iteration": iteration}
    for name, iteration in (
        ("b3_it25_baseline", 25),
        ("it30", 30),
        ("it35", 35),
        ("it40", 40),
        ("it45", 45),
        ("it50", 50),
    )
)


def _payload(name: str, scenario: str, cr: float, to: float = 0.0) -> dict:
    sr = 1.0 - cr - to
    return {
        "checkpoint_name": name,
        "scenario": scenario,
        "metrics": {"n": 2000, "sr": sr, "cr": cr, "to": to},
        "threshold_pass": sr >= 0.90 and cr <= 0.10 and to <= 0.05,
    }


def _matrix(values: dict[str, tuple[float, float, float, float]]) -> list[dict]:
    return [
        _payload(spec["name"], scenario, cr)
        for spec in SPECS
        for scenario, cr in zip(protocol.ALL_SCENARIOS, values[spec["name"]])
    ]


@pytest.fixture(autouse=True)
def _pure_decision(monkeypatch):
    monkeypatch.setattr(protocol, "_candidate_specs", lambda: SPECS)
    monkeypatch.setattr(protocol, "validate_cell", lambda _: None)
    monkeypatch.setattr(
        protocol,
        "screen_protocol",
        lambda: {"sha256": "test", "interpretation_limit": "test only"},
    )


def test_matrix_is_six_checkpoints_by_four_scenarios():
    assert len(SPECS) * len(protocol.ALL_SCENARIOS) == 24
    assert protocol.SEED == 818
    assert protocol.DELAY_STEPS == 1
    assert protocol.SPEED_RATE == 0.7
    assert protocol.LIDAR_NOISE_MODE == "full"


def test_ranking_applies_p060_gate_p080_guardrail_and_tie_band():
    values = {
        "b3_it25_baseline": (0.15, 0.14, 0.08, 0.08),
        "it30": (0.11, 0.12, 0.08, 0.08),
        "it35": (0.09, 0.095, 0.08, 0.08),
        "it40": (0.08, 0.09, 0.08, 0.08),
        "it45": (0.085, 0.085, 0.08, 0.08),
        "it50": (0.07, 0.08, 0.11, 0.08),
    }
    verdict = protocol.final_verdict(_matrix(values))
    assert verdict["passing_candidates"] == ["it35", "it40", "it45"]
    assert verdict["tie_group"] == ["it40", "it45"]
    assert verdict["sa5_candidate"] == "it40"
    assert verdict["phase_b_authorized"] is False
    assert verdict["parent_selected"] is False
    assert verdict["sa6_started"] is False


def test_no_p060_pass_stops_iteration_stacking():
    values = {
        spec["name"]: (0.12, 0.14, 0.08, 0.08) for spec in SPECS
    }
    verdict = protocol.final_verdict(_matrix(values))
    assert verdict["sa5_candidate"] is None
    assert verdict["next_action"].startswith("STOP_ITERATION_STACKING")


def test_runner_preserves_fixed_density_speed_and_lateral_mode():
    source = Path(__file__).with_name(
        "run_sa5_v3_c50_stage3_b3_cont25_screen_cell.py"
    ).read_text(encoding="utf-8")
    assert "protocol.speed_range_for(scenario)" in source
    assert "protocol.density_for(" in source
    assert 'corridor_motion_mode="lateral"' in source
