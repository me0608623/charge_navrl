from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa4_r3_it125_checkpoint_screen as screen  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_r3_it125_checkpoint_screen_v1.json"
RUNNER = HERE / "run_sa4_r3_it125_checkpoint_screen.py"
QUEUE = HERE / "sa4_r3_it125_checkpoint_screen_queue.py"


def _cell(scenario: str, cr: float, hard_pass: bool = True) -> dict:
    return {
        "checkpoint_name": screen.CHECKPOINT_NAME,
        "scenario": scenario,
        "threshold_pass": hard_pass,
        "metrics": {"n": 2000, "sr": 1.0 - cr, "cr": cr, "to": 0.0},
    }


def _payloads() -> list[dict]:
    return [
        _cell("corridor_lateral", 0.08),
        _cell("corridor_longitudinal", 0.04),
        _cell("nav_native", 0.06),
    ]


def test_frozen_protocol_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == screen.screen_protocol()


def test_checkpoint_identity_and_fixed_exam_are_frozen():
    protocol = screen.screen_protocol()
    assert protocol["checkpoint"] == {
        "run_name": "sa4_r3_cont25_from_it100_ne1024_s42_p25_r1",
        "name": "sa4_r3_it125",
        "filename": "checkpoint_3200.pt",
        "conceptual_iteration": 125,
        "sha256": "57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d",
    }
    fixed = protocol["fixed_evaluation"]
    assert fixed["evaluator_seed"] == 818
    assert fixed["actuator_delay_steps"] == 1
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["steps_by_scenario"] == {
        "corridor_lateral": 2500,
        "corridor_longitudinal": 4000,
        "nav_native": 1200,
    }


def test_all_three_hard_gates_are_required():
    payloads = _payloads()
    payloads[0]["threshold_pass"] = False
    verdict = screen.evaluate_candidate(payloads)
    assert verdict["screen_pass"] is False
    assert verdict["selected_checkpoint"] is None
    assert verdict["accepted_parent"] is False


def test_three_pass_only_permits_formal_consideration():
    verdict = screen.evaluate_candidate(_payloads())
    assert verdict["screen_pass"] is True
    assert verdict["selected_checkpoint"] == screen.CHECKPOINT_NAME
    assert verdict["formal_sa4_verification_consideration_eligible"] is True
    assert verdict["accepted_parent"] is False
    assert verdict["sa5_started"] is False


def test_missing_or_wrong_checkpoint_cell_fails_closed():
    payloads = _payloads()[:-1]
    try:
        screen.evaluate_candidate(payloads)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing cell must fail closed")

    payloads = _payloads()
    payloads[0]["checkpoint_name"] = "wrong"
    try:
        screen.evaluate_candidate(payloads)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("wrong checkpoint must fail closed")


def test_runner_and_queue_are_fail_closed_and_do_not_train():
    runner = RUNNER.read_text(encoding="utf-8")
    queue = QUEUE.read_text(encoding="utf-8")
    assert "protocol.CHECKPOINT_SHA256" in runner
    assert '"--lidar-distractor-eligibility"' in runner
    assert "_validate_runtime_log" in runner
    assert "verify_training_source_contract" in queue
    assert "verify_evaluation_source_contract" in queue
    assert '"status": "INCOMPLETE_NO_VERDICT"' in queue
    assert "evaluation source drift after cell" in queue
    assert "train_rnn_car_wdclip" not in runner
    # The queue names the trainer only as a read-only training-source hash.
    # Its sole subprocess command is the frozen evaluation runner.
    assert "str(RUNNER)" in queue
    assert "subprocess.run(command" in queue
    assert "subprocess.Popen" not in queue
    for source in (runner, queue):
        assert "systemctl" not in source
