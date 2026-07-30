"""CPU-only decision tests for the SA2 capability queue."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import sa2_capability_queue as queue


def _checkpoints(count: int = 2) -> list[Path]:
    return [Path(f"/tmp/checkpoint_{index}.pt") for index in range(count)]


def _payload(
    cell: queue.Cell,
    *,
    passes: bool = True,
    valid: bool = True,
    margin: float = 0.02,
    cr: float = 0.01,
) -> dict:
    return {
        "schema": "sa2_capability/v1",
        "cell_id": cell.cell_id,
        "checkpoint": str(cell.checkpoint),
        "scenario": cell.scenario,
        "seed": cell.seed,
        "delay_steps": cell.delay_steps,
        "valid": valid,
        "threshold_pass": passes,
        "worst_margin": margin,
        "metrics": {"cr": cr},
    }


def test_screen_is_exactly_six_checkpoints_by_three_scenarios():
    cells = queue.screen_cells(_checkpoints(6))
    assert len(cells) == 18
    assert len({cell.cell_id for cell in cells}) == 18
    assert {cell.seed for cell in cells} == {818}
    assert {cell.delay_steps for cell in cells} == {1}


def test_formal_grid_is_27_independent_cells_per_checkpoint():
    cells = queue.verdict_cells(_checkpoints(1))
    assert len(cells) == 27
    assert len({cell.cell_id for cell in cells}) == 27
    assert {cell.seed for cell in cells} == {515, 616, 717}
    assert {cell.delay_steps for cell in cells} == {0, 1, 2}
    assert {cell.scenario for cell in cells} == set(queue.SCENARIOS)


def test_queue_and_runner_share_one_canonical_cell_id():
    cell = queue.Cell(Path("/tmp/checkpoint_38400.pt"), "narrow", 515, 1)
    assert cell.cell_id == (
        "checkpoint_38400__narrow__d1__seed515"
    )
    assert cell.cell_id == queue.make_cell_id(
        cell.checkpoint, cell.scenario, cell.delay_steps, cell.seed
    )


def test_build_command_pins_one_seed_delay_and_profile(tmp_path):
    cell = queue.Cell(Path("/tmp/checkpoint.pt"), "lateral", 616, 2)
    command = queue.build_command(cell, tmp_path, num_envs=64, steps=1200)
    joined = " ".join(command)
    assert "--seed 616" in joined
    assert "--actuator-delay-steps 2" in joined
    assert "--actuator-profile sa1_delay_only" in joined
    assert "--scenario lateral" in joined
    assert "--seeds" not in command


def test_duplicate_cell_guard_fails_closed():
    cell = queue.Cell(Path("/tmp/checkpoint.pt"), "narrow", 515, 0)
    payload = _payload(cell)
    with pytest.raises(RuntimeError, match="duplicate cells"):
        queue.assert_cells_are_distinct([payload, dict(payload)])


def test_checkpoint_summary_requires_every_expected_cell():
    checkpoint = Path("/tmp/checkpoint.pt")
    cells = queue.verdict_cells([checkpoint])
    payloads = [_payload(cell) for cell in cells[:-1]]
    summary = queue.summarize_checkpoint(
        checkpoint, payloads, expected_cells=27
    )
    assert summary["cells"] == 26
    assert summary["all_cells_pass"] is False


def test_checkpoint_summary_fails_when_one_cell_fails():
    checkpoint = Path("/tmp/checkpoint.pt")
    cells = queue.verdict_cells([checkpoint])
    payloads = [_payload(cell) for cell in cells]
    payloads[0]["threshold_pass"] = False
    payloads[0]["worst_margin"] = -0.01
    summary = queue.summarize_checkpoint(
        checkpoint, payloads, expected_cells=27
    )
    assert summary["worst_margin"] == pytest.approx(-0.01)
    assert summary["all_cells_pass"] is False


def test_rank_excludes_invalid_checkpoint():
    summaries = [
        {
            "checkpoint": "/tmp/good.pt",
            "invalid_cells": 0,
            "worst_margin": 0.01,
            "all_cells_pass": True,
            "per_scenario": {
                name: {"worst_margin": 0.01} for name in queue.SCENARIOS
            },
        },
        {
            "checkpoint": "/tmp/invalid.pt",
            "invalid_cells": 1,
            "worst_margin": 1.00,
            "all_cells_pass": False,
            "per_scenario": {
                name: {"worst_margin": 1.00} for name in queue.SCENARIOS
            },
        },
    ]
    ranking = queue.rank_candidates(summaries, [])
    assert [row["checkpoint"] for row in ranking["ranking"]] == [
        "/tmp/good.pt"
    ]
    assert ranking["excluded_for_invalid_cells"] == [
        {"checkpoint": "/tmp/invalid.pt", "invalid_cells": 1}
    ]


def test_scene_record_includes_sample_size_and_actuator_contract():
    record = queue.scene_contract_record()
    assert record["actuator_profile"] == "sa1_delay_only"
    for scenario in queue.SCENARIOS:
        assert record["thresholds"][scenario]["episodes_min"] == 1000


def test_execute_refuses_stale_cell_outputs(tmp_path):
    cell = queue.Cell(Path("/tmp/checkpoint.pt"), "narrow", 515, 0)
    output = tmp_path / cell.cell_id
    output.mkdir()
    (output / "runner.log").write_text("old", encoding="utf-8")
    with pytest.raises(queue.PreconditionError, match="stale outputs"):
        queue.execute_cell(cell, output, num_envs=64, steps=1200)


def test_stage_b_passes_when_one_of_two_candidates_passes(
    monkeypatch, tmp_path
):
    good, bad = _checkpoints(2)
    cells = queue.verdict_cells([good, bad])

    monkeypatch.setattr(queue, "wait_for_exclusive_gpu", lambda **_: None)
    monkeypatch.setattr(
        queue,
        "source_fingerprint",
        lambda: {"runner.py": "a" * 64},
    )

    def fake_execute(cell, output_dir, *, num_envs, steps):
        passes = cell.checkpoint == good
        return _payload(
            cell,
            passes=passes,
            margin=0.03 if passes else -0.02,
            cr=0.02 if passes else 0.20,
        )

    monkeypatch.setattr(queue, "execute_cell", fake_execute)
    summary = queue.run_stage(
        "B",
        cells,
        [good, bad],
        tmp_path,
        num_envs=64,
        steps=1200,
    )
    verdict = summary["verdict"]
    assert verdict["sa2_pass"] is True
    assert verdict["passing_checkpoints"] == [str(good)]
    assert verdict["selected_checkpoint"] == str(good)


def test_stage_b_selects_larger_worst_margin_then_lower_cr(
    monkeypatch, tmp_path
):
    first, second = _checkpoints(2)
    cells = queue.verdict_cells([first, second])

    monkeypatch.setattr(queue, "wait_for_exclusive_gpu", lambda **_: None)
    monkeypatch.setattr(
        queue,
        "source_fingerprint",
        lambda: {"runner.py": "a" * 64},
    )

    def fake_execute(cell, output_dir, *, num_envs, steps):
        if cell.checkpoint == first:
            return _payload(cell, margin=0.02, cr=0.01)
        return _payload(cell, margin=0.03, cr=0.03)

    monkeypatch.setattr(queue, "execute_cell", fake_execute)
    summary = queue.run_stage(
        "B",
        cells,
        [first, second],
        tmp_path,
        num_envs=64,
        steps=1200,
    )
    assert summary["verdict"]["selected_checkpoint"] == str(second)


def test_source_change_voids_stage_b(monkeypatch, tmp_path):
    checkpoint = _checkpoints(1)[0]
    cells = queue.verdict_cells([checkpoint])
    calls = 0

    monkeypatch.setattr(queue, "wait_for_exclusive_gpu", lambda **_: None)

    def changing_fingerprint():
        nonlocal calls
        calls += 1
        return {"runner.py": ("a" if calls == 1 else "b") * 64}

    monkeypatch.setattr(queue, "source_fingerprint", changing_fingerprint)
    monkeypatch.setattr(
        queue,
        "execute_cell",
        lambda cell, output_dir, *, num_envs, steps: _payload(cell),
    )
    summary = queue.run_stage(
        "B",
        cells,
        [checkpoint],
        tmp_path,
        num_envs=64,
        steps=1200,
    )
    assert summary["source_fingerprint_stable"] is False
    assert summary["verdict"]["sa2_pass"] is False
    assert summary["invalid_cells"] >= 1


def test_source_fingerprint_is_sha256():
    fingerprints = queue.source_fingerprint()
    assert fingerprints
    assert all(len(digest) == 64 for digest in fingerprints.values())
