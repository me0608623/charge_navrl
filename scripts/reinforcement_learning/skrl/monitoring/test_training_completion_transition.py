from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_completion_transition import (
    CompletionNotReady,
    transition_if_complete,
)


RUN = "sa4_example_ne1024_s42_p100_r1"


def _fixture(
    tmp_path: Path,
    *,
    iteration: int = 100,
    target: int = 100,
    total_steps: int = 12800,
    checkpoint: bool = True,
    marker_steps: int | None = 12800,
    anomaly: str = "",
) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    state = repo / "logs" / "training_supervisor"
    run_dir = repo / "logs" / "rnn_car" / RUN
    run_dir.mkdir(parents=True)
    state.mkdir(parents=True)

    metrics = {
        "iteration": iteration,
        "iterations_target": target,
        "total_steps": total_steps,
    }
    (run_dir / "supervisor_metrics.jsonl").write_text(
        json.dumps({"iteration": 1}) + "\n" + json.dumps(metrics) + "\n",
        encoding="utf-8",
    )
    if checkpoint:
        (run_dir / f"checkpoint_{total_steps}.pt").write_bytes(b"checkpoint")

    console = "normal startup\n"
    if marker_steps is not None:
        console += f"Training complete: {marker_steps:,} steps in 123s\n"
    console += anomaly
    (repo / "logs" / "rnn_car" / f"{RUN}.console.log").write_text(
        console,
        encoding="utf-8",
    )

    expected = state / "expected_run.txt"
    status = state / "status.txt"
    ledger = state / "completion_ledger.jsonl"
    expected.write_text(RUN + "\n", encoding="utf-8")
    status.write_text("phase=RUNNING\n", encoding="utf-8")
    return repo, expected, status, ledger


def _transition(paths: tuple[Path, Path, Path, Path]):
    repo, expected, status, ledger = paths
    return transition_if_complete(
        repo=repo,
        expected_run_file=expected,
        status_file=status,
        ledger_file=ledger,
    )


def test_complete_run_transitions_atomically(tmp_path: Path) -> None:
    repo, expected, status, ledger = _fixture(tmp_path)
    evidence = _transition((repo, expected, status, ledger))

    assert evidence is not None
    assert evidence.run_name == RUN
    assert evidence.iteration == 100
    assert expected.read_text(encoding="utf-8") == ""
    assert "transition=COMPLETE_TO_IDLE" in status.read_text(encoding="utf-8")
    row = json.loads(ledger.read_text(encoding="utf-8"))
    assert row["run_name"] == RUN
    assert row["checkpoint"].endswith("checkpoint_12800.pt")


def test_incomplete_run_stays_expected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, iteration=99)
    with pytest.raises(CompletionNotReady, match="training is incomplete"):
        _transition(paths)
    assert paths[1].read_text(encoding="utf-8").strip() == RUN


def test_missing_checkpoint_stays_expected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, checkpoint=False)
    with pytest.raises(CompletionNotReady, match="final checkpoint"):
        _transition(paths)
    assert paths[1].read_text(encoding="utf-8").strip() == RUN


@pytest.mark.parametrize("marker_steps", [None, 12799])
def test_missing_or_mismatched_marker_stays_expected(
    tmp_path: Path, marker_steps: int | None
) -> None:
    paths = _fixture(tmp_path, marker_steps=marker_steps)
    with pytest.raises(CompletionNotReady, match="completion marker"):
        _transition(paths)
    assert paths[1].read_text(encoding="utf-8").strip() == RUN


@pytest.mark.parametrize(
    "anomaly",
    ["Traceback (most recent call last):\n", "CUDA out of memory\n", "NaN\n"],
)
def test_strict_anomaly_stays_expected(tmp_path: Path, anomaly: str) -> None:
    paths = _fixture(tmp_path, anomaly=anomaly)
    with pytest.raises(CompletionNotReady, match="anomaly"):
        _transition(paths)
    assert paths[1].read_text(encoding="utf-8").strip() == RUN


def test_unsafe_run_name_is_rejected(tmp_path: Path) -> None:
    repo, expected, status, ledger = _fixture(tmp_path)
    expected.write_text("../escape\n", encoding="utf-8")
    with pytest.raises(CompletionNotReady, match="safe-name"):
        _transition((repo, expected, status, ledger))
    assert expected.read_text(encoding="utf-8").strip() == "../escape"


def test_empty_expected_run_is_a_noop(tmp_path: Path) -> None:
    repo, expected, status, ledger = _fixture(tmp_path)
    expected.write_text("", encoding="utf-8")
    assert _transition((repo, expected, status, ledger)) is None
    assert status.read_text(encoding="utf-8") == "phase=RUNNING\n"
    assert not ledger.exists()


def test_retry_does_not_duplicate_ledger_entry(tmp_path: Path) -> None:
    repo, expected, status, ledger = _fixture(tmp_path)
    first = _transition((repo, expected, status, ledger))
    assert first is not None
    expected.write_text(RUN + "\n", encoding="utf-8")
    second = _transition((repo, expected, status, ledger))
    assert second is not None
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
