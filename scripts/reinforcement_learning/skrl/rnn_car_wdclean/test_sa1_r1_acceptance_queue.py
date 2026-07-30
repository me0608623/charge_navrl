"""CPU-only tests for the SA1 R1 acceptance queue."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import sa1_r1_acceptance_queue as q  # noqa: E402


CKPT = Path("/tmp/checkpoint_269952.pt")
OUT = Path("/tmp/out")


def _cell(
    scenario: str = "nav20_clean",
    delay: str = "d1",
    seed: int = 515,
    checkpoint: Path = CKPT,
    stage: str = "B",
) -> q.Cell:
    return q.Cell(checkpoint, scenario, delay, seed, stage)


def _cmd(scenario: str, delay: str = "d1", seed: int = 515) -> list[str]:
    return q.build_command(_cell(scenario, delay, seed), OUT)


@pytest.mark.parametrize("scenario", q.SCENARIOS)
@pytest.mark.parametrize("delay", q.DELAY_CONDITIONS)
def test_fixed_delay_commands_pin_the_neutral_profile(scenario, delay):
    cmd = _cmd(scenario, delay)
    assert cmd[0] == str(q.ISAAC_PYTHON)
    assert cmd[cmd.index("--actuator-profile") + 1] == "sa1_delay_only"
    assert cmd[cmd.index("--actuator-delay-steps") + 1] == delay[1:]
    assert "bridge" not in cmd


def test_off_diagnostic_omits_only_the_delay_flag():
    cmd = _cmd("nav20_clean", q.OFF_CONDITION)
    assert "--actuator-delay-steps" not in cmd
    assert cmd[cmd.index("--actuator-profile") + 1] == "sa1_delay_only"


def test_clean_and_native_modes_are_explicit():
    clean = _cmd("nav20_clean")
    native = _cmd("nav20_native")
    assert clean[clean.index("--mode") + 1] == "clean"
    assert native[native.index("--mode") + 1] == "native"
    assert clean[1].endswith("run_sa1_nav20_suite.py")
    assert native[1].endswith("run_sa1_nav20_suite.py")


def test_every_command_pins_seed_explicitly():
    for scenario in q.SCENARIOS:
        command = _cmd(scenario, seed=616)
        assert command[command.index("--seeds") + 1] == "616"


def test_matrix_shapes_keep_off_out_of_official_acceptance():
    assert len(q.screening_matrix([CKPT])) == 2
    acceptance = q.acceptance_matrix([CKPT])
    assert len(acceptance) == 2 * 3 * 3 == 18
    assert {cell.delay_condition for cell in acceptance} == {"d0", "d1", "d2"}
    off = q.off_regression_matrix([CKPT])
    assert len(off) == 2
    assert {cell.delay_condition for cell in off} == {"off"}
    assert {cell.seed for cell in off} == {515}


def _marker(delay: int) -> str:
    return (
        "[SIM2REAL] Actuator DR: "
        f"delay=({delay}, {delay}) steps, "
        "vel_scale=(1.0, 1.0), motor_lag alpha=1.0, "
        "pipeline=decode->delay->scale->lag, "
        "history=issued_command_queue"
    )


def _ok_payload(
    *,
    cell: q.Cell | None = None,
    gate_pass: bool = True,
) -> dict:
    cell = cell or _cell()
    delay_steps = q._DELAY_STEPS[cell.delay_condition]
    stability = {
        "aggregate": {
            "samples": 1024,
            "omega_rms_rad_s": 0.2,
            "full_steer_fraction": 0.01,
            "ratio_flip_rate_mean": 0.03,
            "worst_seed_ratio_flip_rate_p95": 0.10,
            "worst_seed_omega_abs_std_p95_rad_s": 0.15,
            "worst_seed_dominant_frequency_p95_hz": 0.5,
            "worst_seed_max_same_sign_turn_p95_s": 1.2,
        }
    }
    return {
        "schema": q.SCHEMA,
        "cell_id": cell.cell_id,
        "stage": cell.stage,
        "checkpoint": str(cell.checkpoint.resolve()),
        "scenario": cell.scenario,
        "blocking": cell.scenario in q.BLOCKING_SCENARIOS,
        "delay_condition": cell.delay_condition,
        "actuator_profile": q.ACTUATOR_PROFILE,
        "seed": cell.seed,
        "num_envs": q.NUM_ENVS,
        "steps": q.STEPS,
        "episodes_completed": 1024,
        "sr": 0.94,
        "cr": 0.05,
        "timeout": 0.01,
        "gate_pass": gate_pass,
        "structural_pass": True,
        "thresholds": {},
        "action_stability": stability,
        "actuator_eval": q._expected_actuator_metadata(cell.delay_condition),
        "runner": q.runner_name(cell.scenario),
        "runner_argv": [],
        "source_report": "/tmp/source.json",
        "runtime_markers": {
            "actuator_dr": None if delay_steps is None else _marker(delay_steps),
            "vlp16_noise": "[SIM2REAL][VLP16-ablation] mode=full sigma=ON",
        },
        "exit_code": 0 if gate_pass else 1,
        "started_at": "2026-07-29T00:00:00+00:00",
        "finished_at": "2026-07-29T00:01:00+00:00",
    }


def test_valid_pass_and_valid_policy_fail_both_validate():
    cell = _cell()
    assert q.validate_result(_ok_payload(cell=cell), cell) == []
    assert q.validate_result(_ok_payload(cell=cell, gate_pass=False), cell) == []


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("sr", float("nan"), "sr must be finite"),
        ("cr", float("inf"), "cr must be finite"),
        ("timeout", -0.1, "timeout must be finite"),
        ("episodes_completed", 0, "positive integer"),
        ("structural_pass", False, "structural_pass"),
    ],
)
def test_invalid_metrics_and_structure_fail(field, value, needle):
    payload = _ok_payload()
    payload[field] = value
    assert any(needle in problem for problem in q.validate_result(payload))


def test_runner_exit_code_must_match_gate_verdict():
    payload = _ok_payload(gate_pass=False)
    payload["exit_code"] = 0
    assert any("expected 1" in problem for problem in q.validate_result(payload))


def test_runtime_markers_are_strict_for_fixed_delay_and_vlp():
    payload = _ok_payload()
    payload["runtime_markers"]["actuator_dr"] = None
    payload["runtime_markers"]["vlp16_noise"] = None
    problems = q.validate_result(payload)
    assert any("actuator_dr" in problem for problem in problems)
    assert any("VLP16" in problem for problem in problems)


def test_nav20_action_stability_fields_are_required():
    payload = _ok_payload()
    del payload["action_stability"]["aggregate"]["omega_rms_rad_s"]
    assert any(
        "omega_rms_rad_s" in problem for problem in q.validate_result(payload)
    )


def test_off_requires_negative_actuator_evidence():
    cell = _cell(delay="off", stage="O")
    payload = _ok_payload(cell=cell)
    assert q.validate_result(payload, cell) == []
    payload["runtime_markers"]["actuator_dr"] = _marker(0)
    assert any("unexpectedly emitted" in p for p in q.validate_result(payload, cell))


def test_cell_identity_mismatch_is_rejected():
    cell = _cell()
    payload = _ok_payload(cell=cell)
    payload["seed"] = 616
    assert any("expected 515" in p for p in q.validate_result(payload, cell))


def test_duplicate_guard_includes_checkpoint_identity():
    first = _ok_payload(cell=_cell(checkpoint=Path("/tmp/a.pt")))
    second = _ok_payload(cell=_cell(checkpoint=Path("/tmp/b.pt")))
    q.assert_no_cross_delay_average([first, second])
    with pytest.raises(q.PreconditionError):
        q.assert_no_cross_delay_average([first, dict(first)])


def _write_runtime_log(path: Path, delay: int = 1) -> None:
    path.write_text(
        _marker(delay)
        + "\n[SIM2REAL][VLP16-ablation] mode=full sigma=ON bias=ON\n",
        encoding="utf-8",
    )


def _source_report(cell: q.Cell, output: Path, gate_pass: bool) -> dict:
    actuator = q._expected_actuator_metadata(cell.delay_condition)
    mode = cell.scenario.removeprefix("nav20_")
    blocking = cell.scenario in q.BLOCKING_SCENARIOS
    log = output / f"{mode}_s{cell.seed}.log"
    _write_runtime_log(log, q._DELAY_STEPS[cell.delay_condition] or 0)
    common = {
        "checkpoint": str(cell.checkpoint.resolve()),
        "mode": mode,
        "blocking": blocking,
        "seeds": [cell.seed],
        "actuator_eval": actuator,
        "thresholds": dict(q._THRESHOLDS[cell.scenario]),
        "scene_contract": dict(q._SCENE_CONTRACTS[cell.scenario]),
    }
    return common | {
        "aggregate": {"n": 100, "sr": 0.8, "cr": 0.2, "to": 0.0},
        "action_stability": _ok_payload(cell=cell)["action_stability"],
        "pass": gate_pass,
        "logs": [str(log)],
    }


@pytest.mark.parametrize("scenario", q.SCENARIOS)
def test_all_runner_report_shapes_canonicalize(tmp_path, scenario):
    checkpoint = tmp_path / "checkpoint_269952.pt"
    checkpoint.touch()
    cell = _cell(scenario=scenario, checkpoint=checkpoint)
    output = tmp_path / "cell"
    output.mkdir()
    report = _source_report(cell, output, gate_pass=False)
    (output / q.report_name(scenario)).write_text(
        json.dumps(report), encoding="utf-8"
    )
    payload = q.canonicalize_runner_result(
        cell,
        output,
        ["runner"],
        1,
        "start",
        "finish",
    )
    assert payload["gate_pass"] is False
    assert q.validate_result(payload, cell) == []


def test_source_scene_contract_mismatch_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint_269952.pt"
    checkpoint.touch()
    cell = _cell(scenario="nav20_clean", checkpoint=checkpoint)
    output = tmp_path / "cell"
    output.mkdir()
    report = _source_report(cell, output, gate_pass=False)
    report["scene_contract"]["mode"] = "native"
    (output / q.report_name(cell.scenario)).write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(q.PreconditionError, match="scene_contract"):
        q.canonicalize_runner_result(
            cell, output, ["runner"], 1, "start", "finish"
        )


def test_source_relaxed_thresholds_fail_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint_269952.pt"
    checkpoint.touch()
    cell = _cell(scenario="nav20_clean", checkpoint=checkpoint)
    output = tmp_path / "cell"
    output.mkdir()
    report = _source_report(cell, output, gate_pass=False)
    report["thresholds"] = report["thresholds"] | {"success_rate_min": 0.50}
    report["pass"] = True
    (output / q.report_name(cell.scenario)).write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(q.PreconditionError, match="thresholds"):
        q.canonicalize_runner_result(
            cell, output, ["runner"], 0, "start", "finish"
        )


def _completed_metadata() -> dict:
    return {
        "iteration": q.FINAL_ITERATIONS - 1,
        "total_steps": q.FINAL_TOTAL_STEPS,
        "args": dict(q.EXPECTED_CHECKPOINT_ARGS),
    }


def _fake_precondition_commands(monkeypatch):
    def fake_run(command):
        if command[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(command, 3, "inactive\n", "")
        if command[:2] == ["pgrep", "-f"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        if command and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(q, "_run", fake_run)


def test_preconditions_accept_zero_based_final_iteration(tmp_path, monkeypatch):
    _fake_precondition_commands(monkeypatch)
    train_log = tmp_path / "train.log"
    train_log.write_text(q.COMPLETION_MARKER, encoding="utf-8")
    final = tmp_path / f"checkpoint_{q.FINAL_TOTAL_STEPS}.pt"
    final.touch()
    monkeypatch.setattr(q, "TRAIN_LOG", train_log)
    monkeypatch.setattr(q, "EXPECTED_FINAL_CHECKPOINT", final)
    monkeypatch.setattr(q, "final_checkpoint", lambda: final)
    assert q.check_preconditions(checkpoint_reader=lambda _: _completed_metadata()) == []


def test_preconditions_reject_missing_completion_and_partial_checkpoint(
    tmp_path, monkeypatch
):
    _fake_precondition_commands(monkeypatch)
    train_log = tmp_path / "train.log"
    train_log.write_text("still training", encoding="utf-8")
    final = tmp_path / f"checkpoint_{q.FINAL_TOTAL_STEPS}.pt"
    final.touch()
    monkeypatch.setattr(q, "TRAIN_LOG", train_log)
    monkeypatch.setattr(q, "EXPECTED_FINAL_CHECKPOINT", final)
    monkeypatch.setattr(q, "final_checkpoint", lambda: final)
    metadata = _completed_metadata() | {
        "iteration": q.FINAL_ITERATIONS - 2,
        "total_steps": q.FINAL_TOTAL_STEPS - q.STEPS_PER_ITERATION,
    }
    problems = q.check_preconditions(checkpoint_reader=lambda _: metadata)
    assert any("completion marker" in p for p in problems)
    assert any("completed iterations" in p for p in problems)
    assert any("total_steps" in p for p in problems)


def test_exact_checkpoint_resolution_never_substitutes_nearest(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "RUN_DIR", tmp_path)
    (tmp_path / "checkpoint_192000.pt").touch()
    with pytest.raises(q.PreconditionError, match="missing"):
        q.checkpoints_for_iterations((1500, 1700))


def test_execute_queue_records_rc1_and_continues(tmp_path):
    checkpoint = tmp_path / "checkpoint_269952.pt"
    checkpoint.touch()
    cells = [
        _cell("nav20_clean", checkpoint=checkpoint),
        _cell("nav20_native", checkpoint=checkpoint),
    ]
    calls: list[str] = []

    def fake_runner(command, _console):
        output = Path(command[command.index("--output-dir") + 1])
        mode = command[command.index("--mode") + 1]
        scenario = f"nav20_{mode}"
        cell = next(item for item in cells if item.scenario == scenario)
        report = _source_report(cell, output, gate_pass=False)
        (output / q.report_name(scenario)).write_text(
            json.dumps(report), encoding="utf-8"
        )
        calls.append(scenario)
        return subprocess.CompletedProcess(command, 1)

    assert (
        q.execute_queue(
            cells,
            tmp_path / "results",
            execute=True,
            precondition_checker=lambda: [],
            runner=fake_runner,
        )
        == 0
    )
    assert calls == ["nav20_clean", "nav20_native"]
    assert len(list((tmp_path / "results").glob("*/cell.json"))) == 2


def test_incomplete_existing_cell_directory_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint_269952.pt"
    checkpoint.touch()
    cell = _cell(checkpoint=checkpoint)
    output = tmp_path / cell.cell_id
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(q.PreconditionError, match="incomplete"):
        q._execute_cell(cell, output, runner=lambda *_: None)


def test_summary_never_claims_deployable():
    payload = _ok_payload()
    summary = q.summarize_results([payload], "A")
    assert summary["deployable"] is False
    assert summary["python_torchscript_parity"] == "required_external_83d_k8_gate"
    assert summary["candidates"][0]["all_blocking_gates_pass"] is True


def test_native_failure_is_reported_but_does_not_fail_blocking_gate():
    clean = _ok_payload(cell=_cell("nav20_clean"), gate_pass=True)
    native = _ok_payload(cell=_cell("nav20_native"), gate_pass=False)
    summary = q.summarize_results([clean, native], "A")
    candidate = summary["candidates"][0]
    assert candidate["all_recorded_gates_pass"] is False
    assert candidate["all_blocking_gates_pass"] is True
    assert candidate["advisory_failures"] == 1


def test_incomplete_stage_b_cannot_pass_blocking_gate():
    clean = _ok_payload(
        cell=_cell("nav20_clean", delay="d0", seed=515, stage="B"),
        gate_pass=True,
    )
    native = _ok_payload(
        cell=_cell("nav20_native", delay="d0", seed=515, stage="B"),
        gate_pass=True,
    )
    summary = q.summarize_results([clean, native], "B")
    candidate = summary["candidates"][0]
    assert candidate["matrix_complete"] is False
    assert candidate["blocking_cells_complete"] is False
    assert candidate["all_blocking_gates_pass"] is False


def test_frozen_constants():
    assert q.NUM_ENVS == 64
    assert q.STEPS == 1200
    assert q.SEEDS == (515, 616, 717)
    assert q.DELAY_CONDITIONS == ("d0", "d1", "d2")
    assert q.FINAL_ITERATIONS == 2109
    assert q.FINAL_TOTAL_STEPS == 269952
    assert q.SCREEN_CHECKPOINT_ITERATIONS == (1500, 1700, 1900, 2109)


def test_unknown_scenario_and_delay_raise():
    with pytest.raises(ValueError):
        _cmd("does_not_exist")
    with pytest.raises(ValueError):
        _cmd("nav20_clean", "d9")
