"""CPU-only locks for the c300-c600 SA4-v3 Gate comparison."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sa4_v3_c300_c600_gate as gate
import sa4_v3_checkpoint_screen as source
import sa4_v3_c300_c600_gate_queue as queue


def test_candidate_identity_and_hashes_are_complete():
    assert [row["name"] for row in gate.CANDIDATES] == [
        "c300", "c350", "c400", "c450", "c500", "c550", "c600"
    ]
    assert [row["conceptual_iteration"] for row in gate.CANDIDATES] == [
        300, 350, 400, 450, 500, 550, 600
    ]
    assert len({row["sha256"] for row in gate.CANDIDATES}) == 7
    for candidate in gate.CANDIDATES:
        path = gate.checkpoint_path(candidate)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate["sha256"]


def test_source_gate_is_reused_without_condition_drift():
    assert gate.FIXED_CELLS == source.FIXED_CELLS
    assert gate.THRESHOLDS == source.THRESHOLDS
    for name in (
        "STAGE", "SEED", "NUM_ENVS", "STEPS", "MIN_EPISODES",
        "DELAY_STEPS", "DELAY_MS", "ACTUATOR_PROFILE", "LIDAR_NOISE_MODE",
        "LIDAR_DISTRACTOR_ELIGIBILITY", "SPEED_RATE", "SPEED_RATE_OBS",
        "DEPLOYMENT_SPEED_SCALE", "MOTION_MODE", "TIE_MARGIN",
    ):
        assert getattr(gate, name) == getattr(source, name), name
    assert hashlib.sha256(gate.SOURCE_GATE_FREEZE.read_bytes()).hexdigest() == (
        gate.SOURCE_GATE_FREEZE_SHA256
    )
    frozen_source = json.loads(gate.SOURCE_GATE_FREEZE.read_text())
    assert frozen_source["sha256"] == gate.SOURCE_GATE_PROTOCOL_SHA256


def test_exactly_twenty_one_unique_cells_are_fixed():
    cells = gate.cells()
    assert len(cells) == 21
    assert len({cell["cell"] for cell in cells}) == 21
    assert {cell["fixed_cell"] for cell in cells} == {
        "0S1D_P080", "1S1D_P080", "2S1D_P060"
    }


def _payload(candidate: dict, spec: dict, cr: float) -> dict:
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    return {
        "schema": "sa4_v3_checkpoint_screen_cell/v1",
        "cell": gate.cell_label(candidate["name"], spec["label"]),
        "checkpoint_name": candidate["name"],
        "checkpoint_sha256": candidate["sha256"],
        "protocol_sha256": gate.screen_protocol()["sha256"],
        "fixed_cell": spec["label"],
        "density": spec["density"],
        "scenario": spec["scenario"],
        "stage": gate.STAGE,
        "seed": gate.SEED,
        "delay_steps": gate.DELAY_STEPS,
        "speed_rate": gate.SPEED_RATE,
        "motion_mode": gate.MOTION_MODE,
        "pedestrian_speed_range_m_s": list(spec["pedestrian_speed_range_m_s"]),
        "metrics": metrics,
        "gate": gate.evaluate_metrics(metrics),
        "corridor_report": {
            "configured_static_obstacles": spec["static_obstacles"],
            "configured_dynamic_obstacles": spec["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                spec["pedestrian_speed_range_m_s"]
            ),
            "dynamic_motion_mode": gate.MOTION_MODE,
            "speed_rate": gate.SPEED_RATE,
        },
    }


def test_ranking_is_worst_cell_then_mean_and_promotes_only_two():
    payloads = []
    for index, candidate in enumerate(gate.CANDIDATES):
        for offset, spec in enumerate(gate.FIXED_CELLS):
            payloads.append(_payload(candidate, spec, 0.01 * index + 0.001 * offset))
    summary = gate.summarize_cells(payloads)
    assert summary["ranked_candidates"][0]["checkpoint_name"] == "c300"
    assert summary["promoted_for_retention"] == ["c300", "c350"]
    assert summary["retention_started"] is False
    assert summary["accepted_sa5_parent"] is None
    assert summary["sa5_started"] is False


def test_fail_closed_on_partial_duplicate_or_tampered_cells():
    payloads = [
        _payload(candidate, spec, 0.05)
        for candidate in gate.CANDIDATES
        for spec in gate.FIXED_CELLS
    ]
    with pytest.raises(ValueError):
        gate.summarize_cells(payloads[:-1])
    with pytest.raises(ValueError):
        gate.summarize_cells(payloads[:-1] + [payloads[0]])
    bad = dict(payloads[0])
    bad["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        gate.validate_cell(bad)


def test_queue_fingerprints_protocol_wrappers_and_all_checkpoints():
    source_paths = {path.resolve() for path in queue.implementation.SOURCE_PATHS}
    assert Path(gate.__file__).resolve() in source_paths
    assert Path(queue.__file__).resolve() in source_paths
    assert queue.RUNNER.resolve() in source_paths
    fingerprint = queue.implementation.source_fingerprint()
    for candidate in gate.CANDIDATES:
        key = str(gate.checkpoint_path(candidate).relative_to(gate.REPO))
        assert fingerprint["files"][key] == candidate["sha256"]


def test_runtime_protocol_matches_frozen_json():
    assert queue.FREEZE.is_file()
    frozen = json.loads(queue.FREEZE.read_text(encoding="utf-8"))
    assert frozen == gate.screen_protocol()


def test_queue_does_not_launch_training_retention_or_sa5():
    source_text = Path(queue.__file__).read_text(encoding="utf-8")
    assert "train_rnn_car_wdclip" not in source_text
    assert "systemd-run" not in source_text
    assert "sa5_started" not in source_text
