from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_r2_checkpoint_screen as base  # noqa: E402
import sa5_r2_checkpoint_screen_sample_repair as repair  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_r2_checkpoint_screen_sample_repair_v1.json"
RUNNER = HERE / "run_sa5_r2_checkpoint_screen_sample_repair_cell.py"
QUEUE = HERE / "sa5_r2_checkpoint_screen_continue.py"


def _base_cell(name: str, scenario: str, *, n: int = 2000, cr: float = 0.05) -> dict:
    candidate = base.candidate_by_name(name)
    metrics = {"n": n, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario == "narrow_range":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return {
        "cell_valid": True,
        "checkpoint_name": name,
        "checkpoint_sha256": candidate["sha256"],
        "scenario": scenario,
        "protocol_sha256": base.screen_protocol()["sha256"],
        "steps": base.STEPS_BY_SCENARIO[scenario],
        "threshold_pass": cr <= 0.10,
        "metrics": metrics,
    }


def _repair_cell(original: dict, *, n: int = 1200, cr: float | None = None) -> dict:
    payload = copy.deepcopy(original)
    base_n = int(original["metrics"]["n"])
    scenario = original["scenario"]
    steps = repair.repair_steps(int(original["steps"]), base_n)
    assert steps is not None
    payload["steps"] = steps
    payload["protocol_sha256"] = repair.execution_protocol(
        scenario, steps
    )["sha256"]
    payload["metrics"]["n"] = n
    if cr is not None:
        payload["metrics"]["cr"] = cr
        payload["metrics"]["sr"] = 1.0 - cr
    payload["sample_size_repair"] = {
        "addendum_sha256": repair.repair_addendum()["sha256"],
        "base_n": base_n,
        "pooled_with_base": False,
    }
    return payload


def test_frozen_addendum_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == repair.repair_addendum()


def test_repair_duration_uses_only_n_and_has_a_buffer():
    assert repair.repair_steps(2500, 1000) is None
    assert repair.repair_steps(2500, 994) == 3500
    assert repair.repair_steps(2500, 830) == 4000
    assert repair.repair_steps(2500, 663) == 5000
    assert repair.repair_steps(2500, 994) == repair.repair_steps(2500, 994)


def test_sr_cr_to_cannot_change_the_repair_duration():
    first = _base_cell("c100", "corridor_random2d", n=994, cr=0.01)
    second = _base_cell("c100", "corridor_random2d", n=994, cr=0.99)
    assert repair.repair_steps(first["steps"], first["metrics"]["n"]) == repair.repair_steps(
        second["steps"], second["metrics"]["n"]
    )


def test_execution_protocol_changes_only_the_selected_step_budget():
    original = base.screen_protocol()
    amended = repair.execution_protocol("corridor_random2d", 3500)
    assert amended["sha256"] != original["sha256"]
    original_no_hash = copy.deepcopy(original)
    amended_no_hash = copy.deepcopy(amended)
    original_no_hash.pop("sha256")
    amended_no_hash.pop("sha256")
    amended_no_hash["fixed_evaluation"]["steps_by_scenario"][
        "corridor_random2d"
    ] = 2500
    assert amended_no_hash == original_no_hash


def test_underfilled_cell_is_replaced_and_never_pooled():
    base_cells = [
        _base_cell(name, scenario)
        for name in (item["name"] for item in base.CANDIDATES)
        for scenario in base.CORRIDOR_SCENARIOS
    ]
    target = next(
        cell
        for cell in base_cells
        if repair.cell_key(cell) == ("c100", "corridor_random2d")
    )
    target["metrics"]["n"] = 994
    target["metrics"]["cr"] = 0.90
    target["metrics"]["sr"] = 0.10
    replacement = _repair_cell(target, n=1210, cr=0.20)

    effective, log = repair.choose_effective_cells(
        base_cells,
        [replacement],
        checkpoint_names=[item["name"] for item in base.CANDIDATES],
        scenarios=base.CORRIDOR_SCENARIOS,
    )
    selected = next(
        cell
        for cell in effective
        if repair.cell_key(cell) == ("c100", "corridor_random2d")
    )
    assert selected["metrics"]["n"] == 1210
    assert selected["metrics"]["n"] != 994 + 1210
    assert log == [
        {
            "checkpoint_name": "c100",
            "scenario": "corridor_random2d",
            "base_n": 994,
            "base_steps": 2500,
            "repair_n": 1210,
            "repair_steps": 3500,
            "pooled": False,
        }
    ]


def test_missing_or_unnecessary_repair_fails_closed():
    original = _base_cell("c100", "corridor_random2d", n=994)
    with pytest.raises(ValueError, match="missing required repair"):
        repair.choose_effective_cells(
            [original],
            [],
            checkpoint_names=["c100"],
            scenarios=["corridor_random2d"],
        )
    sufficient = _base_cell("c100", "corridor_random2d", n=1000)
    fake_repair = copy.deepcopy(sufficient)
    with pytest.raises(ValueError, match="unexpected repair"):
        repair.choose_effective_cells(
            [sufficient],
            [fake_repair],
            checkpoint_names=["c100"],
            scenarios=["corridor_random2d"],
        )


def test_repair_cell_below_minimum_still_fails_closed():
    original = _base_cell("c150", "corridor_random2d", n=663)
    replacement = _repair_cell(original, n=999)
    with pytest.raises(ValueError, match="still has fewer"):
        repair.validate_repair_cell(
            replacement,
            base.candidate_by_name("c150"),
            "corridor_random2d",
            base_n=663,
        )


def test_continuation_has_no_training_or_sa6_launch_path():
    source = QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_NO_VERDICT" in source
    assert "continuation source drift" in source
    assert "base_queue.verify_checkpoints()" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemctl" not in source
    assert "subprocess.Popen" not in source
    assert '"sa6_started": False' in source


def test_repair_wrapper_reuses_frozen_runner_and_preserves_delay_pipeline():
    source = RUNNER.read_text(encoding="utf-8")
    assert "base_runner.main" in source
    assert "base.STEPS_BY_SCENARIO[args.scenario] = repair_steps" in source
    assert '"pooled_with_base": False' in source
    base_source = (HERE / "run_sa5_r2_checkpoint_screen_cell.py").read_text(
        encoding="utf-8"
    )
    assert "fixed_actuator_cli_args" in base_source
    assert "verify_fixed_actuator_runtime" in base_source

