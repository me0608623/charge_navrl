import json
import os
from pathlib import Path

import auto_advance_supervisor as supervisor


def _metric_rows(count=500, **overrides):
    rows = []
    for iteration in range(1, count + 1):
        row = {
            "iteration": iteration,
            "sr": 0.92,
            "vf": 2.5,
            "entropy": 4.2,
            "kl": 0.004,
            "clip_fraction": 0.04,
            "encoder_grad": 14.0,
        }
        for key, value in overrides.items():
            row[key] = value(iteration) if callable(value) else value
        rows.append(row)
    return rows


def test_stable_high_metrics_converge():
    result = supervisor.evaluate_convergence(
        _metric_rows(), stage=4, min_iterations=420, window_size=10, window_count=5
    )
    assert result["converged"]
    assert all(result["checks"].values())


def test_warmup_cannot_converge():
    result = supervisor.evaluate_convergence(
        _metric_rows(419), stage=4, min_iterations=420, window_size=10, window_count=5
    )
    assert not result["converged"]
    assert result["reason"].startswith("warmup")


def test_sr_drift_blocks_convergence():
    rows = _metric_rows(sr=lambda iteration: 0.92 + 0.0005 * (iteration - 450))
    result = supervisor.evaluate_convergence(
        rows, stage=4, min_iterations=420, window_size=10, window_count=5
    )
    assert not result["converged"]
    assert not result["checks"]["sr_plateau"]


def test_value_trend_blocks_convergence():
    rows = _metric_rows(vf=lambda iteration: 4.0 - 0.02 * (iteration - 450))
    result = supervisor.evaluate_convergence(
        rows, stage=4, min_iterations=420, window_size=10, window_count=5
    )
    assert not result["converged"]
    assert not result["checks"]["vf_plateau"]


def test_read_metrics_deduplicates_iterations(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"iteration": 2, "sr": 0.8}),
                "not-json",
                json.dumps({"iteration": 1, "sr": 0.7}),
                json.dumps({"iteration": 2, "sr": 0.9}),
            ]
        ),
        encoding="utf-8",
    )
    rows = supervisor.read_metrics(path)
    assert [row["iteration"] for row in rows] == [1, 2]
    assert rows[-1]["sr"] == 0.9


def test_all_obb_stage_configs_exist():
    for stage in range(1, 9):
        assert (supervisor.CONFIG_DIR / f"e2e_sa{stage}_k8_obb.py").is_file()


def test_isaaclab_subprocess_env_selects_frozen_conda_python(monkeypatch):
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", str(supervisor.CONDA_ENV / "bin")]))
    env = supervisor.isaaclab_subprocess_env()
    assert env["CONDA_PREFIX"] == str(supervisor.CONDA_ENV)
    assert env["CONDA_DEFAULT_ENV"] == "env_isaaclab"
    assert env["PATH"].split(os.pathsep)[0] == str(supervisor.CONDA_ENV / "bin")
    assert env["PATH"].split(os.pathsep).count(str(supervisor.CONDA_ENV / "bin")) == 1


def test_gate_failure_is_fail_closed(tmp_path: Path):
    exit_file = tmp_path / "gate.exitcode"
    exit_file.write_text("1\n", encoding="utf-8")
    state = {
        "enabled": True,
        "phase": "GATING",
        "stage": 5,
        "run_name": "unit_test_run",
        "gate_exit_file": str(exit_file),
        "gate_pid": 99999999,
    }
    supervisor.DRY_RUN = True
    result = supervisor.tick(state, dry_run=True)
    assert result["phase"] == "HALTED_ALERT"
    assert "no advancement" in result["detail"]


def test_next_stage_command_uses_frozen_config_and_accepted_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint_123.pt"
    checkpoint.touch()
    state = {
        "enabled": True,
        "phase": "ADVANCING",
        "stage": 5,
        "run_name": "unit_test_run",
        "gate_checkpoint": str(checkpoint),
    }
    supervisor.DRY_RUN = True
    supervisor.launch_next_stage(state, dry_run=True)
    assert state["phase"] == "ADVANCING"
    assert "e2e_sa6_k8_obb" in state["next_train_command"]
    assert str(checkpoint) in state["next_train_command"]


def test_converged_training_requests_cooperative_stop(tmp_path: Path, monkeypatch):
    run_name = "unit_test_converged"
    run_dir = tmp_path / "logs" / "rnn_car" / run_name
    run_dir.mkdir(parents=True)
    metrics = run_dir / "supervisor_metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps(row) for row in _metric_rows()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "REPO", tmp_path)
    monkeypatch.setattr(supervisor, "STATUS_FILE", tmp_path / "status.txt")
    monkeypatch.setattr(supervisor, "find_training_pid", lambda _: 12345)
    supervisor.DRY_RUN = False
    state = {
        "enabled": True,
        "phase": "TRAINING",
        "stage": 4,
        "run_name": run_name,
        "min_iterations": 420,
        "window_size": 10,
        "window_count": 5,
    }
    result = supervisor.tick(state)
    assert result["phase"] == "STOP_REQUESTED"
    assert (run_dir / "supervisor_stop.request").is_file()


def test_passing_gate_prepares_next_stage_without_launching(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint_456.pt"
    checkpoint.touch()
    exit_file = tmp_path / "gate.exitcode"
    exit_file.write_text("0\n", encoding="utf-8")
    state = {
        "enabled": True,
        "phase": "GATING",
        "stage": 5,
        "run_name": "unit_test_pass",
        "gate_checkpoint": str(checkpoint),
        "gate_exit_file": str(exit_file),
        "gate_pid": 99999999,
    }
    supervisor.DRY_RUN = True
    result = supervisor.tick(state, dry_run=True)
    assert result["phase"] == "ADVANCING"
    assert "e2e_sa6_k8_obb" in result["next_train_command"]
