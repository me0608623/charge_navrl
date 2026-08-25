"""CPU contracts for the SA5-R2 fixed speed-0.7 checkpoint screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa5_r2_speed0p7_checkpoint_screen_cell as runner  # noqa: E402
import sa5_r2_checkpoint_screen as parent  # noqa: E402
import sa5_r2_speed0p7_checkpoint_screen as screen  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_r2_speed0p7_checkpoint_screen_v3.json"


def _metrics(scenario: str, cr: float) -> dict:
    metrics = {"n": 1200, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario == "narrow_range":
        metrics.update(
            {
                "crossing_rate": 0.99,
                "direct_crossing_rate": 0.99,
            }
        )
    return metrics


def _cells(worst_by_name: dict[str, float]) -> list[dict]:
    payloads = []
    for candidate in screen.CANDIDATES:
        name = candidate["name"]
        for index, scenario in enumerate(screen.SCENARIOS):
            cr = worst_by_name[name] if index == 0 else 0.02
            if scenario in screen.RETENTION_SCENARIOS:
                cr = 0.02 if scenario == "nav_native" else 0.01
            metrics = _metrics(scenario, cr)
            payloads.append(
                {
                    "schema": "sa5_r2_speed0p7_checkpoint_screen_cell/v3",
                    "cell_valid": True,
                    "checkpoint_name": name,
                    "checkpoint_sha256": candidate["sha256"],
                    "protocol_sha256": screen.screen_protocol()["sha256"],
                    "scenario": scenario,
                    "steps": screen.STEP_TIERS_BY_SCENARIO[scenario][0],
                    "step_tier_index": 0,
                    "speed_rate": 0.7,
                    "speed_rate_obs": "ego",
                    "deployment_speed_scale": 1.0,
                    "metrics": metrics,
                    "outcome_gate": parent.evaluate_metrics(scenario, metrics),
                }
            )
    return payloads


def test_protocol_locks_four_checkpoints_six_scenarios_and_rate0p7():
    frozen = screen.screen_protocol()
    assert [candidate["name"] for candidate in screen.CANDIDATES] == [
        "baseline_c250",
        "control_c300",
        "adapt_it25",
        "adapt_it50",
    ]
    assert tuple(frozen["fixed_evaluation"]["scenarios"]) == screen.SCENARIOS
    assert frozen["cells"] == 24
    assert frozen["fixed_evaluation"]["speed_rate"] == 0.7
    assert frozen["fixed_evaluation"]["speed_rate_obs"] == "ego"
    assert frozen["fixed_evaluation"]["deployment_speed_scale"] == 1.0
    assert frozen["ranking"]["clear_improvement_margin"] == 0.005
    assert frozen["retention"]["maximum_regression"] == 0.02


def test_v3_freezes_sample_size_only_step_tiers():
    assert screen.STEPS_BY_SCENARIO == {
        "corridor_lateral": 2500,
        "corridor_longitudinal": 2500,
        "corridor_random2d": 5000,
        "corridor_mixed": 5000,
        "nav_native": 1200,
        "narrow_range": 1200,
    }
    assert screen.STEP_TIERS_BY_SCENARIO == {
        "corridor_lateral": (2500, 5000, 7500),
        "corridor_longitudinal": (2500, 5000, 7500),
        "corridor_random2d": (5000, 7500),
        "corridor_mixed": (5000, 7500),
        "nav_native": (1200, 2400, 3600),
        "narrow_range": (1200, 2400, 3600),
    }
    assert screen.screen_protocol()["schema"].endswith("/v3")


def test_insufficient_first_tier_is_valid_retry_evidence_but_not_a_result():
    cell = _cells(
        {
            "baseline_c250": 0.20,
            "control_c300": 0.19,
            "adapt_it25": 0.16,
            "adapt_it50": 0.14,
        }
    )[0]
    cell["metrics"]["n"] = 829
    cell["cell_valid"] = False
    cell["outcome_gate"] = parent.evaluate_metrics(
        cell["scenario"], cell["metrics"]
    )
    screen.validate_cell(cell, require_min_episodes=False)
    try:
        screen.validate_cell(cell)
    except ValueError as exc:
        assert "fewer than" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("insufficient tier became ranking evidence")


def test_v2_first_tier_migration_preserves_measurement_with_provenance():
    cell = _cells(
        {
            "baseline_c250": 0.20,
            "control_c300": 0.19,
            "adapt_it25": 0.16,
            "adapt_it50": 0.14,
        }
    )[0]
    original_metrics = dict(cell["metrics"])
    cell["schema"] = "sa5_r2_speed0p7_checkpoint_screen_cell/v2"
    cell["protocol_sha256"] = screen.V2_PROTOCOL_SHA256
    cell.pop("step_tier_index")
    migrated = screen.migrate_v2_cell(
        cell,
        origin_path="/evidence/cell.json",
        origin_cell_sha256="a" * 64,
        origin_source_fingerprint_sha256="b" * 64,
    )
    assert migrated["metrics"] == original_metrics
    assert migrated["measurement_provenance"]["mode"] == "imported_v2_first_tier"
    screen.validate_cell(migrated)


def test_checkpoint_hashes_match_disk():
    for candidate in screen.CANDIDATES:
        path = screen.checkpoint_path(candidate)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate["sha256"]


def test_freeze_matches_runtime_hash_and_checkpoint_set():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["protocol_sha256"] == screen.screen_protocol()["sha256"]
    assert frozen["checkpoint_sha256"] == {
        candidate["name"]: candidate["sha256"]
        for candidate in screen.CANDIDATES
    }


def test_it50_clear_leader_with_retention_authorizes_only_bounded_extension():
    result = screen.summarize(
        _cells(
            {
                "baseline_c250": 0.20,
                "control_c300": 0.19,
                "adapt_it25": 0.16,
                "adapt_it50": 0.14,
            }
        )
    )
    assert result["decision"] == "EXTEND_ADAPT_IT50_BY_25_TO_50"
    assert result["accepted_parent"] is False
    assert result["training_started"] is False
    assert result["sa6_started"] is False


def test_raw_margin_alone_does_not_claim_late_forgetting():
    result = screen.summarize(
        _cells(
            {
                "baseline_c250": 0.20,
                "control_c300": 0.19,
                "adapt_it25": 0.14,
                "adapt_it50": 0.16,
            }
        )
    )
    assert result["preregistered_rule_outcome"]["decision"] == (
        "STOP_IT50_LATE_FORGETTING"
    )
    assert result["decision"] == "INCONCLUSIVE_DO_NOT_EXTEND"


def test_late_regression_requires_it25_gain_and_two_se_return():
    result = screen.summarize(
        _cells(
            {
                "baseline_c250": 0.20,
                "control_c300": 0.20,
                "adapt_it25": 0.10,
                "adapt_it50": 0.18,
            }
        )
    )
    assert result["decision"] == "STOP_IT50_LATE_REGRESSION_SUPPORTED"


def test_observed_pattern_is_degradation_without_prior_improvement():
    result = screen.summarize(
        _cells(
            {
                "baseline_c250": 0.8136739293764087,
                "control_c300": 0.8143800440205429,
                "adapt_it25": 0.8362445414847162,
                "adapt_it50": 0.8603174603174604,
            }
        )
    )
    assert result["decision"] == (
        "ADAPTATION_DEGRADED_NO_IMPROVEMENT_AT_ANY_CHECKPOINT"
    )
    comparisons = result["posthoc_standardized_comparisons"]["comparisons"]
    assert round(
        comparisons["control_vs_baseline"]["standardized_difference_se"], 2
    ) == 0.04
    assert round(
        comparisons["adapt_it25_vs_baseline"]["standardized_difference_se"], 2
    ) == 1.46
    assert round(
        comparisons["adapt_it50_vs_baseline"]["standardized_difference_se"], 2
    ) == 3.10
    assert round(
        comparisons["adapt_it50_vs_adapt_it25"][
            "standardized_difference_se"
        ],
        2,
    ) == 1.64


def test_no_adaptation_gain_moves_speed_contract_earlier():
    result = screen.summarize(
        _cells(
            {
                "baseline_c250": 0.15,
                "control_c300": 0.14,
                "adapt_it25": 0.15,
                "adapt_it50": 0.145,
            }
        )
    )
    assert result["decision"] == "MOVE_SPEED_CONTRACT_EARLIER_TO_SA3_SA4"


def test_missing_cell_fails_closed_before_verdict():
    cells = _cells(
        {
            "baseline_c250": 0.20,
            "control_c300": 0.19,
            "adapt_it25": 0.16,
            "adapt_it50": 0.14,
        }
    )
    try:
        screen.summarize(cells[:-1])
    except ValueError as exc:
        assert "exactly 24 cells" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("partial screen produced a verdict")


def test_runner_parses_exact_speed_rate_marker():
    log = Path("/tmp/sa5_r2_speed0p7_marker_test.log")
    log.write_text(
        "[SPEED_RATE] max_linear_velocity: 1.0 -> 0.7000\n"
        "[SPEED_RATE] max_linear_accel: 0.5 -> 0.3500\n"
        "[SPEED_RATE] max_angular_vel: 1.2 -> 0.8400\n"
        "[SPEED_RATE] max_angular_accel: 3.0 -> 2.1000\n"
        "[VEHICLE-SPEED-RATE] rate=0.7 obs=ego lidar_scaled=False "
        "deployment_scale=1\n",
        encoding="utf-8",
    )
    observed = runner.verify_speed_rate_log(log)
    assert observed["speed_rate"] == 0.7
    log.unlink()


def test_runner_and_queue_never_start_training_or_sa6():
    runner_source = (
        HERE / "run_sa5_r2_speed0p7_checkpoint_screen_cell.py"
    ).read_text(encoding="utf-8")
    queue_source = (
        HERE / "sa5_r2_speed0p7_checkpoint_screen_queue.py"
    ).read_text(encoding="utf-8")
    assert '"--speed_rate"' in runner_source
    assert '"--deployment_speed_scale"' in runner_source
    assert "train_rnn_car_wdclip.py" not in runner_source
    assert "train_rnn_car_wdclip.py" not in queue_source
    assert "accepted_parent=False training_started=False sa6_started=False" in queue_source
