"""Contract tests for the paired SA5-v3 stage-2 retention screen."""

from __future__ import annotations

import copy

import sa5_v3_c50_stage2_retention_screen as protocol


def _cell(candidate: dict, scenario: str, *, sr: float, cr: float, to: float) -> dict:
    metrics = {"n": 2_000, "sr": sr, "cr": cr, "to": to}
    if scenario == "narrow_range":
        metrics.update({"crossing_rate": 0.99, "direct_crossing_rate": 0.99})
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": candidate["name"],
        "checkpoint_sha256": candidate["expected_sha256"],
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


def _passing_cells() -> list[dict]:
    rows = []
    for candidate in protocol.CANDIDATE_SPECS:
        for scenario in protocol.ALL_SCENARIOS:
            rows.append(_cell(candidate, scenario, sr=0.96, cr=0.03, to=0.01))
    return rows


def test_matrix_and_runtime_contract_are_frozen() -> None:
    frozen = protocol.screen_protocol()
    assert frozen["matrix"]["cell_count"] == 8
    assert frozen["matrix"]["scenarios"] == list(protocol.RETENTION_SCENARIOS)
    assert frozen["fixed_evaluation"]["speed_rate"] == 0.7
    assert frozen["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert frozen["fixed_evaluation"]["lidar_distractor_eligibility"] == (
        "valid_return_only"
    )


def test_checkpoints_are_exact_completed_pair() -> None:
    assert protocol.A2_EQUAL["expected_sha256"].startswith("0f3c9d47")
    assert protocol.B2_MILD["expected_sha256"].startswith("45c972f8")
    assert protocol.B2_MILD["motion_weights"] == [0.30, 0.30, 0.40]


def test_screen_cannot_start_downstream_work() -> None:
    decision = protocol.screen_protocol()["decision"]
    assert decision["starts_training"] is False
    assert decision["authorizes_extension"] is False
    assert decision["authorizes_graduation"] is False
    assert decision["starts_sa6"] is False


def test_passing_pair_only_requests_separate_authorization() -> None:
    verdict = protocol.final_verdict(_passing_cells())
    assert verdict["retention_supported"] is True
    assert verdict["selected_development_checkpoint"] == protocol.B2_MILD["name"]
    assert verdict["selected_extension_checkpoint"] is None
    assert verdict["extension_authorized"] is False
    assert verdict["sa6_started"] is False


def test_relative_degradation_over_two_points_rejects() -> None:
    rows = _passing_cells()
    for row in rows:
        if (
            row["checkpoint_name"] == protocol.B2_MILD["name"]
            and row["scenario"] == "nav_native"
        ):
            row["metrics"]["sr"] = 0.93
            row["threshold_pass"] = protocol.evaluate_metrics(
                row["scenario"], row["metrics"]
            )["threshold_pass"]
    verdict = protocol.final_verdict(rows)
    assert verdict["retention_supported"] is False
    assert verdict["next_action"] == "REJECT_MILD_WEIGHT_FOR_RETENTION_FAILURE"


def test_absolute_gate_failure_rejects_even_without_relative_loss() -> None:
    rows = _passing_cells()
    for row in rows:
        if row["scenario"] == "nav_native":
            row["metrics"].update({"sr": 0.89, "cr": 0.11})
            row["threshold_pass"] = protocol.evaluate_metrics(
                row["scenario"], row["metrics"]
            )["threshold_pass"]
    verdict = protocol.final_verdict(rows)
    assert verdict["retention_supported"] is False


def test_missing_cell_fails_closed() -> None:
    rows = _passing_cells()
    try:
        protocol.final_verdict(rows[:-1])
    except ValueError as exc:
        assert "exactly 8" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing cell did not fail closed")


def test_cell_runtime_drift_is_rejected() -> None:
    row = _passing_cells()[0]
    broken = copy.deepcopy(row)
    broken["delay_steps"] = 0
    try:
        protocol.validate_cell(broken)
    except ValueError as exc:
        assert "delay_steps mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("runtime drift did not fail closed")
