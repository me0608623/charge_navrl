from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa4_r3_cont50_checkpoint_screen as screen  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_r3_cont50_checkpoint_screen_v1.json"
RUNNER = HERE / "run_sa4_r3_cont50_checkpoint_screen.py"
QUEUE = HERE / "sa4_r3_cont50_checkpoint_screen_queue.py"


def _cell(checkpoint: str, scenario: str, cr: float, hard_pass: bool) -> dict:
    return {
        "checkpoint_name": checkpoint,
        "scenario": scenario,
        "threshold_pass": hard_pass,
        "metrics": {"n": 2000, "sr": 1.0 - cr, "cr": cr, "to": 0.0},
    }


def _payloads(
    *,
    it75_lat=0.09,
    it75_lon=0.08,
    it75_native=0.06,
    it100_lat=0.08,
    it100_lon=0.07,
    it100_native=0.06,
    it75_pass=True,
    it100_pass=True,
):
    return [
        _cell("sa4_r3_it75", "corridor_lateral", it75_lat, it75_pass),
        _cell("sa4_r3_it75", "corridor_longitudinal", it75_lon, it75_pass),
        _cell("sa4_r3_it75", "nav_native", it75_native, it75_pass),
        _cell("sa4_r3_it100", "corridor_lateral", it100_lat, it100_pass),
        _cell("sa4_r3_it100", "corridor_longitudinal", it100_lon, it100_pass),
        _cell("sa4_r3_it100", "nav_native", it100_native, it100_pass),
    ]


def test_frozen_protocol_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == screen.screen_protocol()


def test_protocol_preserves_original_r3_fixed_exam():
    protocol = screen.screen_protocol()
    assert protocol["checkpoints"] == [
        {"name": "sa4_r3_it75", "filename": "checkpoint_3200.pt", "iteration": 75},
        {"name": "sa4_r3_it100", "filename": "checkpoint_6400.pt", "iteration": 100},
    ]
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
    assert "sa4_r3_it75" not in result["qualifying_candidates"]


def test_better_worst_direction_wins():
    result = screen.compare_checkpoints(
        _payloads(it75_lat=0.06, it75_lon=0.06, it100_lat=0.08, it100_lon=0.07)
    )
    assert result["selected_checkpoint"] == "sa4_r3_it75"


def test_it100_wins_only_inside_tie_margin():
    result = screen.compare_checkpoints(
        _payloads(it75_lat=0.080, it75_lon=0.070, it100_lat=0.084, it100_lon=0.060)
    )
    assert result["selected_checkpoint"] == "sa4_r3_it100"


def test_no_passing_checkpoint_authorizes_nothing():
    result = screen.compare_checkpoints(
        _payloads(it75_pass=False, it100_pass=False)
    )
    assert result["selected_checkpoint"] is None
    assert result["sa5_consideration_eligible"] is False
    assert result["additional_training_started"] is False
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
