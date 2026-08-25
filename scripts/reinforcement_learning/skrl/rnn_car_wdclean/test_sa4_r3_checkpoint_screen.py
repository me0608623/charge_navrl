from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa4_r3_checkpoint_screen as screen  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_r3_checkpoint_screen_v1.json"
RUNNER = HERE / "run_sa4_r3_checkpoint_screen.py"
QUEUE = HERE / "sa4_r3_checkpoint_screen_queue.py"


def _cell(checkpoint: str, scenario: str, cr: float, hard_pass: bool) -> dict:
    return {
        "checkpoint_name": checkpoint,
        "scenario": scenario,
        "threshold_pass": hard_pass,
        "metrics": {"n": 2000, "sr": 1.0 - cr, "cr": cr, "to": 0.0},
    }


def _payloads(*, it25_lat=0.09, it25_lon=0.08, it25_native=0.06,
              it50_lat=0.08, it50_lon=0.07, it50_native=0.06,
              it25_pass=True, it50_pass=True):
    return [
        _cell("sa4_r3_it25", "corridor_lateral", it25_lat, it25_pass),
        _cell("sa4_r3_it25", "corridor_longitudinal", it25_lon, it25_pass),
        _cell("sa4_r3_it25", "nav_native", it25_native, it25_pass),
        _cell("sa4_r3_it50", "corridor_lateral", it50_lat, it50_pass),
        _cell("sa4_r3_it50", "corridor_longitudinal", it50_lon, it50_pass),
        _cell("sa4_r3_it50", "nav_native", it50_native, it50_pass),
    ]


def test_frozen_protocol_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == screen.screen_protocol()


def test_protocol_freezes_six_valid_return_only_cells():
    protocol = screen.screen_protocol()
    assert protocol["fixed_evaluation"]["evaluator_seed"] == 818
    assert protocol["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert protocol["fixed_evaluation"]["lidar_distractor_eligibility"] == "valid_return_only"
    assert protocol["fixed_evaluation"]["steps_by_scenario"] == {
        "corridor_lateral": 2500,
        "corridor_longitudinal": 4000,
        "nav_native": 1200,
    }


def test_all_three_hard_gates_are_required():
    payloads = _payloads()
    payloads[2]["threshold_pass"] = False
    result = screen.compare_checkpoints(payloads)
    assert result["rows"][0]["all_three_hard_pass"] is False
    assert "sa4_r3_it25" not in result["qualifying_candidates"]


def test_better_worst_direction_wins():
    result = screen.compare_checkpoints(
        _payloads(it25_lat=0.06, it25_lon=0.06, it50_lat=0.08, it50_lon=0.07)
    )
    assert result["selected_checkpoint"] == "sa4_r3_it25"


def test_it50_wins_only_inside_tie_margin():
    result = screen.compare_checkpoints(
        _payloads(it25_lat=0.080, it25_lon=0.070, it50_lat=0.084, it50_lon=0.060)
    )
    assert result["selected_checkpoint"] == "sa4_r3_it50"


def test_no_passing_checkpoint_authorizes_nothing():
    result = screen.compare_checkpoints(_payloads(it25_pass=False, it50_pass=False))
    assert result["selected_checkpoint"] is None
    assert result["extension_or_sa5_consideration_eligible"] is False
    assert result["training_extension_started"] is False
    assert result["sa5_started"] is False


def test_runner_and_queue_are_fail_closed_and_do_not_train():
    runner = RUNNER.read_text(encoding="utf-8")
    queue = QUEUE.read_text(encoding="utf-8")
    assert '"--lidar-distractor-eligibility"' in runner
    assert "_validate_runtime_log" in runner
    assert "d9_runner._require_new_targets" in runner
    assert '"status": "INCOMPLETE_NO_VERDICT"' in queue
    assert "source drift after cell" in queue
    for source in (runner, queue):
        assert "train_rnn_car_wdclip" not in source
        assert "systemctl" not in source
